import json
import os
from tqdm import tqdm
from datasets import load_dataset
import cv2
import numpy as np

from custom_scripts.common.constants import META_INFO_TEMPLATE, META_STATS_TEMPLATE

def create_meta(root_dir, episodes):
    ep = []
    tasks = []
    info = META_INFO_TEMPLATE
    stats = META_STATS_TEMPLATE

    tasks_dict = {
        '2': "Pick the grape and put it in the basket"
    }

    total_frames = 0

    action_df = []
    observation_state_df = []
    timestamp_df = []
    frame_index_df = []
    episode_index_df = []
    task_index_df = []
    index_df = []
    observation_images_exo_df = []
    observation_images_wrist_df = []
    observation_images_table_df = []

    # fetch data of each episode
    for i in tqdm(range(episodes)):
        parquet_file = os.path.join(root_dir, f"data/chunk-{i//50:03d}/episode_{i:06d}.parquet")
        exo_file = os.path.join(root_dir, f"videos/chunk-{i//50:03d}/observation.images.exo/episode_{i:06d}.mp4")
        wrist_file = os.path.join(root_dir, f"videos/chunk-{i//50:03d}/observation.images.wrist/episode_{i:06d}.mp4")
        table_file = os.path.join(root_dir, f"videos/chunk-{i//50:03d}/observation.images.table/episode_{i:06d}.mp4")

        parquet_data = load_dataset("parquet", data_files=parquet_file)['train']
        ep.append({
            "episode_index": i,
            "tasks": tasks_dict[str(parquet_data[0]['task_index'])],
            "length": len(parquet_data),
        })

        total_frames += len(parquet_data)
        exo_cap = cv2.VideoCapture(exo_file)
        wrist_cap = cv2.VideoCapture(wrist_file)
        table_cap = cv2.VideoCapture(table_file)

        for frame_idx in tqdm(range(len(parquet_data))):
            frame_data = parquet_data[frame_idx]

            action_df.append(frame_data['action'])
            observation_state_df.append(frame_data['observation.state'])
            timestamp_df.append(frame_data['timestamp'])
            frame_index_df.append(frame_data['frame_index'])
            episode_index_df.append(frame_data['episode_index'])
            task_index_df.append(frame_data['task_index'])
            index_df.append(frame_data['index'])

            _, exo_frame = exo_cap.read()
            _, wrist_frame = wrist_cap.read()
            _, table_frame = table_cap.read()

            observation_images_exo_df.append(exo_frame)
            observation_images_wrist_df.append(wrist_frame)
            observation_images_table_df.append(table_frame)

    # save meta/episodes.jsonl
    with open(os.path.join(root_dir, 'meta/episodes.jsonl'), "w") as f:
        for entry in ep:
            json.dump(entry, f)
            f.write("\n")

    # save meta/info.json
    info['total_episodes'] = episodes
    info['total_frames'] = total_frames
    info['total_videos'] = episodes
    info['total_chunks'] = (episodes // info['chunks_size'])+1
    info['splits']['train'] = f"0:{episodes}"

    with open(os.path.join(root_dir, 'meta/info.json'), "w") as f:
        json.dump(info, f, indent=4)

    # save meta/tasks.jsonl
    for k,v in tasks_dict.items():
        tasks.append({
            "task_index": int(k),
            "task": v
        })

    with open(os.path.join(root_dir, 'meta/tasks.jsonl'), "w") as f:
        for entry in tasks:
            json.dump(entry, f)
            f.write("\n")

    # save meta/stats.json
    stacked_action = np.stack(action_df, axis=0)
    stacked_observation_state = np.stack(observation_state_df, axis=0)
    stacked_timestamp = np.stack(timestamp_df, axis=0)
    stacked_frame_index = np.stack(frame_index_df, axis=0)
    stacked_episode_index = np.stack(episode_index_df, axis=0)
    stacked_task_index = np.stack(task_index_df, axis=0)
    stacked_index = np.stack(index_df, axis=0)

    stacked_exo = np.stack(observation_images_exo_df, axis=0)
    stacked_wrist = np.stack(observation_images_wrist_df, axis=0)
    stacked_table = np.stack(observation_images_table_df, axis=0)

    stats["action"]={
        "mean":stacked_action.mean(axis=0).tolist(),
        "std":stacked_action.std(axis=0).tolist(),
        "max":stacked_action.max(axis=0).tolist(),
        "min":stacked_action.min(axis=0).tolist(),
    }
    stats["observation.state"]={
        "mean":stacked_observation_state.mean(axis=0).tolist(),
        "std":stacked_observation_state.std(axis=0).tolist(),
        "max":stacked_observation_state.max(axis=0).tolist(),
        "min":stacked_observation_state.min(axis=0).tolist(),
    }
    stats["timestamp"]={
        "mean":stacked_timestamp.mean(axis=0).reshape(1).tolist(),
        "std":stacked_timestamp.std(axis=0).reshape(1).tolist(),
        "max":stacked_timestamp.max(axis=0).reshape(1).tolist(),
        "min":stacked_timestamp.min(axis=0).reshape(1).tolist(),
    }
    stats["frame_index"]={
        "mean":stacked_frame_index.mean(axis=0).reshape(1).tolist(),
        "std":stacked_frame_index.std(axis=0).reshape(1).tolist(),
        "max":stacked_frame_index.max(axis=0).reshape(1).tolist(),
        "min":stacked_frame_index.min(axis=0).reshape(1).tolist(),
    }
    stats["episode_index"]={
        "mean":stacked_episode_index.mean(axis=0).reshape(1).tolist(),
        "std":stacked_episode_index.std(axis=0).reshape(1).tolist(),
        "max":stacked_episode_index.max(axis=0).reshape(1).tolist(),
        "min":stacked_episode_index.min(axis=0).reshape(1).tolist(),
    }
    stats["task_index"]={
        "mean":stacked_task_index.mean(axis=0).reshape(1).tolist(),
        "std":stacked_task_index.std(axis=0).reshape(1).tolist(),
        "max":stacked_task_index.max(axis=0).reshape(1).tolist(),
        "min":stacked_task_index.min(axis=0).reshape(1).tolist(),
    }
    stats["index"]={
        "mean":stacked_index.mean(axis=0).reshape(1).tolist(),
        "std":stacked_index.std(axis=0).reshape(1).tolist(),
        "max":stacked_index.max(axis=0).reshape(1).tolist(),
        "min":stacked_index.min(axis=0).reshape(1).tolist(),
    }
    stats["observation.images.exo"]={
        "mean":stacked_exo.mean(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "std":stacked_exo.std(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "max":stacked_exo.max(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "min":stacked_exo.min(axis=(0,1,2)).reshape(3,1,1).tolist(),
    }
    stats["observation.images.wrist"]={
        "mean":stacked_wrist.mean(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "std":stacked_wrist.std(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "max":stacked_wrist.max(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "min":stacked_wrist.min(axis=(0,1,2)).reshape(3,1,1).tolist(),
    }
    stats["observation.images.table"]={
        "mean":stacked_table.mean(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "std":stacked_table.std(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "max":stacked_table.max(axis=(0,1,2)).reshape(3,1,1).tolist(),
        "min":stacked_table.min(axis=(0,1,2)).reshape(3,1,1).tolist(),
    }

    with open(os.path.join(root_dir,"meta/stats.json"),"w") as f:
        json.dump(stats,f,indent=4)


if __name__ == "__main__":
    root_dir = "/data/piper_grape0626_50"
    episodes = 300
    create_meta(root_dir, episodes)

    root_dir = "/data/piper_grape0626_75"
    episodes = 450
    create_meta(root_dir, episodes)