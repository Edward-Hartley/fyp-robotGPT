#!/bin/bash

# Launch ROS components in a new terminal
gnome-terminal -- bash -c " \
echo 'Activating ROS environment and launching ROS components...'; \
source /home/edward/miniconda3/etc/profile.d/conda.sh; \
conda activate ros_env_full_bak; \
source /home/edward/catkin_ws/devel/setup.bash; \
cd /home/edward/catkin_ws; \
export LD_LIBRARY_PATH="/home/edward/miniconda3/envs/ros_env_full_bak/lib:$LD_LIBRARY_PATH"; \
export LD_LIBRARY_PATH=/home/edward/catkin_ws/src/kinova-ros/kinova_driver/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH; \
echo 'Launching ROS components...'; \
roslaunch kinova_bringup kinova_robot.launch kinova_robotType:=\$ROBOT_TYPE use_urdf:=true; \
exec bash"

# Launch the Robot Server in a new terminal
gnome-terminal -- bash -c " \
echo 'Activating ROS environment and starting Robot Server...'; \
source /home/edward/miniconda3/etc/profile.d/conda.sh; \
conda activate ros_env_full_bak; \
source /home/edward/catkin_ws/devel/setup.bash; \
python /home/edward/Imperial/fyp-robotGPT/src/fyp_package/robot_server.py; \
exec bash"
