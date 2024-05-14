import numpy as np
from math import cos, sin, sqrt
from fyp_package.utils import euler2quat, rot2quat
import os

#### Camera

#### Camera simulation setup

# camera_position = (0, -0.85, 0.4)
# camera_orientation = (np.pi / 4 + np.pi / 48, np.pi, np.pi)
# camera_orientation_q = euler2quat(*camera_orientation)
# zrange = (0.01, 10.)

# camera_image_size = (480, 480)
# focal_lengths = [0.5 * camera_image_size[0], 0.5 * camera_image_size[0]]

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

latest_camera_specs_path = "./data/latest_camera_specs.npy"
if not os.path.exists(latest_camera_specs_path):
    print("Camera specs not found. Please run realsense_camera.py to generate camera specs.")
else:
    latest_camera_specs = np.load(latest_camera_specs_path, allow_pickle=True).item()
    camera_image_size = latest_camera_specs["camera_image_size"]
    intrinsics = latest_camera_specs["intrinsics"]
    fov = latest_camera_specs["fov"]
    focal_lengths = [intrinsics[0, 0], intrinsics[1, 1]]

# fov and intrinsics are defined by physical camera

#### Object detection

segmentation_threshold = 0.5
bounding_cube_mask_image_path = "./assets/bounding_cube_mask_{object}_{mask}.png"
depth_offset = 0.03
invalid_depth_value = 0

#### Real robot

robot_type = "m1n6s200"
robot_ready_position = [0, -0.35, 0.15]
robot_vertical_orientation_e = [np.pi, 0, 0]
robot_vertical_orientation_q = euler2quat(*robot_vertical_orientation_e)
robot_tucked_position = [0, -0.2, 0.35]

#### Paths

latest_rgb_image_path = "./data/latest_rgb_image.png"
latest_depth_image_path = "./data/latest_depth_image.npy"
latest_segmentation_masks_path = "./data/latest_segmentation_masks.npy"
latest_grasp_detection_path = "./data/latest_grasp_detection.npz"

#### Grasp detection

contact_graspnet_checkpoint_dir = "/home/edward/Imperial/fyp-robotGPT/src/fyp_package/experiments/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001"

#### Server ports
model_server_ports = {
    "graspnet": 9997,
    "langsam": 9998,
}
robot_server_port = 9999
