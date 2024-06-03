import numpy as np
from fyp_package.utils import euler2quat, rot2quat, tf
import os

simulation = False

#### Camera

#### Camera simulation setup
if simulation:
    camera_position = (0, -0.85, 0.4)
    camera_orientation = (np.pi / 4 + np.pi / 48, np.pi, np.pi)
    camera_orientation_q = euler2quat(*camera_orientation)
    cam2base_tf = tf(camera_orientation_q, camera_position)
    zrange = (0.01, 10.)

    camera_image_size = (480, 480)
    focal_lengths = [0.5 * camera_image_size[0], 0.5 * camera_image_size[0]]

    # camera implicit definitions
    # fov should be changed to tuple if not square
    fov = camera_image_size[0] / (2 * focal_lengths[0])
    fov = np.arctan(fov) * 2 * (180 / np.pi)
    intrinsics = np.array([[focal_lengths[0], 0, camera_image_size[0] / 2],
                        [0, focal_lengths[1], camera_image_size[1] / 2],
                        [0, 0, 1]])

#### Real camera setup
else:
    calibration_results = np.load("./data/calibration_results.npy", allow_pickle=True).item()
    camera_position = calibration_results["t_cam2base"]
    camera_orientation_q = rot2quat(calibration_results["R_cam2base"])
    cam2base_tf = tf(camera_orientation_q, camera_position)
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


#### Object detection

segmentation_threshold = 0.5
bounding_cube_mask_image_path = "./assets/bounding_cube_mask_{object}_{mask}.png"
depth_offset = 0.03
invalid_depth_values = 0.01, 0.0

#### Boundaries and known positions

## Simulation
sim_bounds = np.float32([[-0.3, 0.3], [-0.8, -0.2], [0, 0.15]])  # X Y Z
sim_left_bound = sim_bounds[0, 0]
sim_right_bound = sim_bounds[0, 1]
sim_top_bound = sim_bounds[1, 1]
sim_bottom_bound = sim_bounds[1, 0]
sim_middle_x = (sim_left_bound + sim_right_bound) / 2
sim_middle_y = (sim_top_bound + sim_bottom_bound) / 2
sim_corner_pos = {
  'top left corner':     (sim_left_bound + 0.05,   sim_top_bound - 0.05,    0),
  'top side':            (sim_middle_x,            sim_top_bound - 0.05,    0),
  'top right corner':    (sim_right_bound - 0.05,  sim_top_bound - 0.05,    0),
  'left side':           (sim_left_bound + 0.05,   sim_middle_y,            0),
  'middle':              (sim_middle_x,            sim_middle_y,            0),
  'right side':          (sim_right_bound - 0.05,  sim_middle_y,            0),
  'bottom left corner':  (sim_left_bound + 0.05,   sim_bottom_bound + 0.05, 0),
  'bottom side':         (sim_middle_x,            sim_bottom_bound + 0.05, 0),
  'bottom right corner': (sim_right_bound - 0.05,  sim_bottom_bound + 0.05, 0),
}
sim_table_z = sim_bounds[2, 0]

## Real world
# bounds are restrictive as they are a square which can be totally reached
# see the sides as well as corners for an octagon which can be reached
real_bounds = np.float32([[-0.228, 0.228], [-0.366, -0.261], [0.02, 0.15]])  # X Y Z
real_left_bound = real_bounds[0, 0]
real_right_bound = real_bounds[0, 1]
real_top_bound = real_bounds[1, 1]
real_bottom_bound = real_bounds[1, 0]
real_middle_x = (real_left_bound + real_right_bound) / 2
real_middle_y = (real_top_bound + real_bottom_bound) / 2
real_corner_pos = {
  'top left corner':     (real_left_bound,   real_top_bound,    0),
  'top side':            (real_middle_x,     real_top_bound,    0),
  'top right corner':    (real_right_bound,  real_top_bound,    0),
  'left side':           (real_left_bound,   real_middle_y,     0),
  'middle':              (real_middle_x,     real_middle_y,     0),
  'right side':          (real_right_bound,  real_middle_y,     0),
  'bottom left corner':  (real_left_bound,   real_bottom_bound, 0),
  'bottom side':         (real_middle_x,     real_bottom_bound, 0),
  'bottom right corner': (real_right_bound,  real_bottom_bound, 0),
}
real_table_z = real_bounds[2, 0]

real_object_list = ["paper cup", "white bowl", "red block"]

#### Real robot

robot_type = "m1n6s200"
robot_ready_position = [0, -0.25, 0.25]
# If modifying the orientation, check grasp transform still works
robot_vertical_orientation_e = [np.pi, 0, 0]
robot_vertical_orientation_q = euler2quat(*robot_vertical_orientation_e)
robot_tucked_position = [0, -0.2, 0.35]

#### Sim robot

sim_robot_vertical_e = [np.pi, 0, - np.pi / 2]
sim_robot_vertical_q = euler2quat(*sim_robot_vertical_e)

#### Paths

latest_rgb_image_path = "./data/latest_rgb_image.png"
latest_depth_image_path = "./data/latest_depth_image.npy"
latest_segmentation_masks_path = "./data/latest_segmentation_masks.npy"
latest_grasp_detection_path = "./data/latest_grasp_detection.npz"
latest_generation_logs_path = "./data/latest_generation_logs.txt"

chosen_segmentation_mask_path = "./data/chosen_segmentation_mask.npy"
chosen_depth_image_path = "./data/chosen_depth_image.npy"

image_to_display_in_message_path = "./data/image_to_display_in_message.png"
viewed_image_logs_directory = "./data/viewed_image_logs/"

#### Grasp detection

contact_graspnet_checkpoint_dir = "/home/edward/Imperial/fyp-robotGPT/src/fyp_package/experiments/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001"

#### Server ports

model_server_ports = {
    "graspnet": 9997,
    "langsam": 9998,
}
robot_server_port = 9999

#### OpenAI

default_openai_model = "gpt-4o"
cheap_openai_model = "gpt-3.5-turbo"
model_temperature = 0.0
max_tokens = 512
stop=None
completions_compatible_openai_model = "gpt-3.5-turbo-instruct"

#### Logging

run_id_file_path = "./results/run_id.txt"
log_directory_path = "./results/log_{run_id}/"
log_file_path = "./results/log_{run_id}/logs.txt"
