import numpy as np
from fyp_package import model_client, realsense_camera, robot_client, config, object_detection_utils
import matplotlib.pyplot as plt
import time


camera = realsense_camera.RealsenseCamera()
robot = robot_client.RobotClient()
models = model_client.ModelClient()

# Move robot out of the way
robot.move_robot(config.robot_tucked_position, config.robot_vertical_orientation_q, relative=False)

# get camera image
frame = camera.get_frame(save=True)
color_image = frame[0]
depth_image = frame[1]

# segment object
masks, _, matches = models.langsam_predict(config.latest_rgb_image_path, "espresso_cup")

# get object pose
object_poses = object_detection_utils.get_object_cube_from_segmentation(
    masks, matches,
    color_image, depth_image,
    config.camera_position, config.camera_orientation_q, camera.get_intrinsics()
    )

object_pos = object_poses[0]['position']

input("Press Enter to continue...")

# move robot to object
above_object_position = [object_pos[0], object_pos[1], config.robot_ready_position[2]]
object_pos[2] += 0.02 # reduce collision risk with table

robot.open_gripper()
robot.move_robot(above_object_position, config.robot_vertical_orientation_q, relative=False)
robot.move_robot(object_pos, config.robot_vertical_orientation_q, relative=False)
time.sleep(0.2)

# close gripper
robot.close_gripper()
time.sleep(0.2)

# move robot to ready position
robot.move_robot(config.robot_ready_position, config.robot_vertical_orientation_q, relative=False)

# pause
input("Press Enter to continue...")

# open gripper
robot.open_gripper()

# close connection
robot.close()
models.close()
