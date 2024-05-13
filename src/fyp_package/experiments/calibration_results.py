import numpy as np
from fyp_package import model_client, realsense_camera, robot_client, config, object_detection_utils
import matplotlib.pyplot as plt
import time


camera = realsense_camera.RealsenseCamera()
robot = robot_client.RobotClient()
models = model_client.ModelClient()

# get camera image
frame = camera.get_frame(save=True)
color_image = frame[0]
depth_image = frame[1]

# segment object
masks, _, matches = models.langsam_predict(config.latest_camera_image_path, "espresso_cup")

# get object pose
object_poses = object_detection_utils.get_object_cube_from_segmentation(
    masks, matches,
    color_image, depth_image,
    config.camera_position, config.camera_orientation_q, camera.get_intrinsics()
    )

object_pose = object_poses[0]['position']

input("Press Enter to continue...")
orientation = list(config.robot_ready_orientation_q[3:]) + list(config.robot_ready_orientation_q[:3])

# move robot to object
robot.move_robot(object_pose, orientation, relative=False)
time.sleep(0.2)

# close gripper
robot.close_gripper()
time.sleep(0.2)

# move robot to ready position
robot.move_robot([0, 0, 0.1], [0, 0, 0, 1], relative=True)

# pause
input("Press Enter to continue...")

# open gripper
robot.open_gripper()

# close connection
robot.close()
models.close()
