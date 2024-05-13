import numpy as np
from math import cos, sin, sqrt
from fyp_package.utils import euler2quat, rot2quat

#### Camera

#### Camera simulation setup

# camera_position = (0, -0.85, 0.4)
# camera_orientation = (np.pi / 4 + np.pi / 48, np.pi, np.pi)
# camera_orientation_q = euler2quat(*camera_orientation)
# zrange = (0.01, 10.)

# camera_image_size = (480, 480)
# focal_length = 0.5 * camera_image_size[0]

# # camera implicit definitions
# # fov should be changed to tuple if not square
# fov = camera_image_size[0] / (2 * focal_length)
# fov = np.arctan(fov) * 2 * (180 / np.pi)
# intrinsics = np.array([[focal_length, 0, camera_image_size[0] / 2],
#                        [0, focal_length, camera_image_size[1] / 2],
#                        [0, 0, 1]])

# Real camera setup
calibration_results = np.load("./data/calibration_results.npy", allow_pickle=True).item()
camera_position = calibration_results["t_cam2base"]
camera_orientation_q = rot2quat(calibration_results["R_cam2base"])
zrange = (0., 3.)

# fov and intrinsics are defined by physical camera

#### Object detection

segmentation_threshold = 0.5
bounding_cube_mask_image_path = "./assets/bounding_cube_mask_{object}_{mask}.png"
depth_offset = 0.03
invalid_depth_value = 0

#### Real robot

robot_type = "m1n6s200"
robot_ready_position = [0, -0.35, 0.15]
robot_ready_orientation_e = [np.pi, 0, 0]
robot_ready_orientation_q = euler2quat(*robot_ready_orientation_e)

#### Paths

latest_camera_image_path = "./data/latest_camera_image.png"


