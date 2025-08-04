python -m lerobot.record_so101 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=slave \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, \
    table: {type: intelrealsense, serial_number_or_name: Intel RealSense L515, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=master \
    --display_data=true \
    --dataset.repo_id=islab/my_dataset \
    --dataset.root=/home/isl-so100/Desktop/disk/test_data_0729 \
    --dataset.start_episode_idx=1 \
    --resume=True \
    --dataset.num_episodes=2 \
    --dataset.fps=30 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Grab the black cube"