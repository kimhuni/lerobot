python -m lerobot.teleoperate \
    --robot.type=koch_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --teleop.type=koch_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true
