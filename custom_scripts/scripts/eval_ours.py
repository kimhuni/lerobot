import logging
import time
from contextlib import nullcontext
from dataclasses import asdict
from pprint import pformat
from typing import Any

import torch
from peft import PeftModel
from termcolor import colored
from tqdm import tqdm
import matplotlib.pyplot as plt

from custom_scripts.common.constants import GRIPPER_EFFORT
from custom_scripts.common.robot_devices.robot_utils import read_end_pose_msg, ctrl_end_pose
from custom_scripts.common.utils.utils import (
    load_buffer,
    get_current_action,
    random_piper_action,
    random_piper_image,
    plot_trajectory,
    pretty_plot,
    log_time,
    init_devices
)
from custom_scripts.configs.eval_ours import EvalOursPipelineConfig

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle

from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters

from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
)

from lerobot.configs import parser


def evaluate_policy(
    eval_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.eval()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    eval_metrics.loss = loss.item()
    eval_metrics.update_s = time.perf_counter() - start_time
    return eval_metrics, output_dict


def create_batch(piper, exo_rs_cam, wrist_rs_cam, use_devices, task):
    if use_devices:
        return {
            'observation.state': read_end_pose_msg(piper),
            'observation.images.exo': exo_rs_cam.image_for_inference(),
            'observation.images.wrist': wrist_rs_cam.image_for_inference(),
            'task': [task],
        }
    else:
        return {
            'observation.state': random_piper_action(),
            'observation.images.table': random_piper_image(),
            'observation.images.wrist': random_piper_image(),
            'task': [task],
        }


@parser.wrap()
def eval_main(cfg: EvalOursPipelineConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info("Creating dataset")
    train_dataset_meta = LeRobotDatasetMetadata(
        cfg.train_dataset.repo_id, cfg.train_dataset.root, revision=cfg.train_dataset.revision
    )
    if not cfg.online_evaluation:
        dataset = make_dataset(cfg)

    if cfg.online_evaluation:
        if cfg.use_devices:
            piper, cam = init_devices(cfg)
            wrist_rs_cam = cam['wrist_rs_cam']
            exo_rs_cam = cam['exo_rs_cam']
            table_rs_cam = cam['table_rs_cam']
        else:
            piper = None
            wrist_rs_cam = None
            exo_rs_cam = None
            table_rs_cam = None

    logging.info("Making policy.")

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=train_dataset_meta,
    )

    if cfg.use_peft:
        policy = PeftModel.from_pretrained(policy, cfg.peft_path)
        policy.to(device=device)
        policy.eval()

    step = 0  # number of policy updates (forward + backward + optim)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    if not cfg.online_evaluation:
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")

        # create dataloader for offline evaluation
        if hasattr(cfg.policy, "drop_n_last_frames"):
            shuffle = False
            sampler = EpisodeAwareSampler(
                dataset.episode_data_index,
                drop_n_last_frames=cfg.policy.drop_n_last_frames,
                shuffle=False,
            )
        else:
            shuffle = False
            sampler = None

        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=device.type != "cpu",
            drop_last=False,
        )
        dl_iter = cycle(dataloader)

    policy.eval()

    if not cfg.online_evaluation:
        eval_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }

        eval_tracker = MetricsTracker(
            cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
        )

    if cfg.online_evaluation:
        logging.info("Start online evaluation on a fixed dataset")
        episodes = 1
    else:
        logging.info("Start offline evaluation on a fixed dataset")
        episodes = len(dataset.episodes)

    for episode_num in range(0, episodes):
        if cfg.online_evaluation:
            start_frame=0
            end_frame = cfg.max_steps
        else:
            start_frame = dataset.episode_data_index['from'][episode_num]
            end_frame = dataset.episode_data_index['to'][episode_num]

        buffer = [[] for _ in range(policy.config.n_action_steps)]
        fig_2d, ax_2d = plt.subplots(4,2,figsize=[25, 15])
        fig_3d, ax_3d = plt.subplots(subplot_kw={'projection': '3d'}, figsize=[25, 15])

        action_pred_list = []
        action_ans_list = []

        for _ in tqdm(range(end_frame - start_frame)):
            t_start = time.time()
            start_time = time.perf_counter()

            if cfg.online_evaluation:
                batch = create_batch(piper, exo_rs_cam, wrist_rs_cam, cfg.use_devices, cfg.task)
            else:
                batch = next(dl_iter)
                eval_tracker.dataloading_s = time.perf_counter() - start_time

            t_create_batch = log_time()

            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            t_batch_to_gpu = log_time()

            # Plot Trajectory
            action_pred = policy.select_action(batch).squeeze()
            if not cfg.online_evaluation:
                action_ans =  batch['action'].squeeze()[0]
            # if len(policy._action_queue) < 45:
            #     policy.reset()

            logged_time = policy.logged_time
            t_action_pred = log_time()

            # TODO: Implement ACT
            if cfg.temporal_ensemble:
                action_pred_queue = policy._action_queue.copy()
                action_pred_queue.extendleft(action_pred.unsqueeze(0))
                policy.reset()

                buffer = load_buffer(buffer, action_pred_queue)
                buffer, action_pred = get_current_action(buffer)
                buffer.append([])

            if cfg.online_evaluation:
                end_pose_data = action_pred[:6].cpu().to(dtype=int).tolist()
                gripper_data = [action_pred[6].cpu().to(dtype=int), GRIPPER_EFFORT]
                ctrl_end_pose(piper, end_pose_data, gripper_data) if piper is not None else None
                t_action_publish = log_time()
            else:
                t_action_publish = log_time()

            action_pred_list.append(action_pred.cpu() if isinstance(action_pred, torch.Tensor) else action_pred)
            if not cfg.online_evaluation:
                action_ans_list.append(action_ans.cpu() if isinstance(action_ans, torch.Tensor) else action_ans)

            # Step Logger
            step += 1
            if cfg.online_evaluation:
                time.sleep(0.2)

            t_total = log_time()
            logged_time = logged_time | {
                "action_pred": action_pred,
                "t_create_batch": t_create_batch - t_start,
                "t_batch_to_gpu": t_batch_to_gpu - t_create_batch,
                "t_action_pred": t_action_pred - t_batch_to_gpu,
                "t_action_publish": t_action_publish - t_action_pred,
                "t_total": t_total - t_start,
            }
            logging.info(colored(pformat(logged_time), "yellow", attrs=["bold"]))

        plot_trajectory(ax_2d, action_pred_list)
        if not cfg.online_evaluation:
            plot_trajectory(ax_2d, action_ans_list, mode='ans')
        pretty_plot(fig_2d, ax_2d, 'pi0 evaluation')

        plot_trajectory(ax_3d, action_pred_list, projection='3d')
        if not cfg.online_evaluation:
            plot_trajectory(ax_3d, action_ans_list, projection='3d', mode='ans')
        pretty_plot(ax_3d)

        fig_2d.show()
        fig_3d.show()

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
