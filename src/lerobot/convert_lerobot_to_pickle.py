# convert_dataset.py

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


def convert_dataset(repo_path: Path):
    """
    LeRobotDataset을 커스텀 포맷으로 변환합니다.

    Args:
        repo_path (Path): 변환할 LeRobotDataset의 루트 경로.
    """
    init_logging()

    # 1. 원본 데이터셋 및 추가 메타데이터 로드
    logging.info(f"Loading original dataset from: {repo_path}")
    try:
        dataset = LeRobotDataset(repo_path.name, root=repo_path.parent)

        creator_meta_path = repo_path / "creator_meta.json"
        with open(creator_meta_path) as f:
            creator_meta = json.load(f)
        task_name = creator_meta["task_name"]
        creator = creator_meta["creator"]
    except FileNotFoundError:
        logging.error(f"Dataset or creator_meta.json not found in '{repo_path}'. Make sure the path is correct.")
        return

    # 2. 변환된 데이터가 저장될 새로운 루트 디렉토리 생성
    task_name_simple = task_name.split()[0].lower()
    output_root = repo_path.parent / task_name_simple
    output_root.mkdir(exist_ok=True)
    logging.info(f"Converted data will be saved to: {output_root}")

    # 3. 모든 에피소드를 순회하며 변환 작업 수행
    for i in range(len(dataset)):
        try:
            # 에피소드 데이터 불러오기 (DataFrame으로 변환)
            episode_data = dataset.get_episode_data(i)
            episode_df = pd.DataFrame(episode_data)

            frame_num = len(episode_df)
            episode_len = frame_num / dataset.fps

            # 새로운 에피소드 경로 생성
            episode_path = output_root / str(i)
            episode_path.mkdir(exist_ok=True)

            # episode.pickle 저장
            pickle_path = episode_path / "episode.pickle"
            episode_df.to_pickle(pickle_path)

            # 동영상 파일 이동 및 이름 변경
            for camera_name in dataset.meta.observation_camera_names:
                source_video_path = repo_path / "videos" / f"observation.{camera_name}" / f"episode_{i:06d}.mp4"
                target_video_path = episode_path / f"{camera_name}.mp4"
                if source_video_path.exists():
                    shutil.copy(source_video_path, target_video_path)  # 원본 보존을 위해 copy 사용
                else:
                    logging.warning(f"Video not found for episode {i}, camera {camera_name}")

            # meta.json 생성
            meta_data = {
                "task_name": task_name,
                "episode_num": i,
                "episode_len": episode_len,
                "frame_num": frame_num,
                "date": datetime.now().strftime("%Y/%m/%d, %H:%M"),
                "creator": creator,
            }
            meta_path = episode_path / "meta.json"
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=4)

            logging.info(f"Successfully converted episode {i} -> {episode_path}")

        except Exception as e:
            logging.error(f"Failed to convert episode {i}. Error: {e}")

    logging.info("Dataset conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LeRobotDataset to a custom format.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the root of the LeRobotDataset to convert (e.g., './my-lerobot-dataset').",
    )
    args = parser.parse_args()
    convert_dataset(Path(args.path))