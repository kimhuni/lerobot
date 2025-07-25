import time
import logging
from pprint import pformat, pp
from dataclasses import asdict

import matplotlib.pyplot as plt
from termcolor import colored
import torch
import numpy as np
from huggingface_hub import login
from piper_sdk import C_PiperInterface

from custom_scripts.common.constants import GRIPPER_EFFORT
from custom_scripts.common.robot_devices.cam_utils import RealSenseCamera
from custom_scripts.common.robot_devices.robot_utils import read_end_pose_msg, set_zero_configuration, ctrl_end_pose
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
from custom_scripts.configs.eval_real_time_ours import EvalRealTimeOursPipelineConfig

from lerobot.configs import parser

from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    format_big_number,
)

@parser.wrap()
def eval_main(cfg: EvalRealTimeOursPipelineConfig):
    logging.info(pformat(asdict(cfg)))
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

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")

    logging.info("Creating dataset")
    train_dataset_meta = LeRobotDatasetMetadata(
        cfg.train_dataset.repo_id, cfg.train_dataset.root, revision=cfg.train_dataset.revision
    )

    logging.info("Making policy.")

    policy_fp32 = make_policy(
        cfg=cfg.policy,
        ds_meta=train_dataset_meta,
    )
    # policy_bf16 = make_policy(
    #     cfg=cfg.policy,
    #     ds_meta=train_dataset_meta,
    #     bf16 = True
    # )

    model = policy_fp32.model


if __name__ == "__main__":
    init_logging()
    eval_main()