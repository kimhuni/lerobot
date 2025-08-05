sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

start_episode_index=0
num_dataset_record=20
task_name="grab the black cube"
dataset_creator="gh"

python -m lerobot.record_so101 \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=slave \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: /dev/video0, width: 640, height: 480, fps: 30}, \
    table: {type: intelrealsense, serial_number_or_name: Intel RealSense L515, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=master \
    --dataset.repo_id=isl/dataset \
    --dataset.root="/home/isl-so100/Desktop/disk/data/${task_name}" \
    --dataset.start_episode_idx=${start_episode_index} \
    --dataset.num_episodes=${num_dataset_record} \
    --dataset.fps=30 \
    --dataset.episode_time_s=10 \
    --dataset.reset_time_s=1 \
    --dataset.single_task="${task_name}"

python -m lerobot.convert_lerobot_to_pickle \
  --path="/home/isl-so100/Desktop/disk/data/${task_name}"