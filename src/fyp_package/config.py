import numpy as np
import pybullet

# Camera
camera_position = (0, -0.85, 0.4)
camera_orientation = (np.pi / 4 + np.pi / 48, np.pi, np.pi)
camera_orientation_q = pybullet.getQuaternionFromEuler(camera_orientation)
zrange = (0.01, 10.)

# camera definition
camera_image_size = (480, 480)
focal_length = 0.5 * camera_image_size[0]
# camera implicit definitions
# fov should be changed to tuple if not square
fov = camera_image_size[0] / (2 * focal_length)
fov = np.arctan(fov) * 2 * (180 / np.pi)
intrinsics = np.array([[focal_length, 0, camera_image_size[0] / 2],
                       [0, focal_length, camera_image_size[1] / 2],
                       [0, 0, 1]])

# Depth offset
depth_offset = 0.03

# Segmentation
segmentation_threshold = 0.5

bounding_cube_mask_image_path = "./assets/bounding_cube_mask_{object}_{mask}.png"



