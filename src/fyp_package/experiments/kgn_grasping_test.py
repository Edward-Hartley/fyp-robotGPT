#%%

demo_image_path = './KGN_grasping/src/demo/color_img_0.png'
demo_depth_path = './KGN_grasping/src/demo/depth_raw_0.npy'
demo_poses_path = './KGN_grasping/src/demo/poses_0.npz'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pybullet

#%%

# to start with just look at image, depth and find properties of poses
image = cv2.imread(demo_image_path)
depth = np.load(demo_depth_path)
poses = np.load(demo_poses_path)

# image
plt.imshow(image)
plt.show()
print(image.shape)

# depth
plt.imshow(depth)
plt.show()

# poses
print(poses.files)
for file in poses.files:
    print(file, poses[file].shape)
    print(poses[file])

##%
# %%

# imports for Franka Panda environment
from fyp_package import pick_and_place_env as franka_env

#%%

# create environment
env = franka_env.PickPlaceEnv(render=True)
block_list = franka_env.ALL_BLOCKS[:3]
bowl_list = franka_env.ALL_BOWLS[:3]
obj_list = block_list + bowl_list
_ = env.reset(obj_list)

# get camera image and depth from pybullet env
if not env.high_res:
    image_size = (240, 240)
    intrinsics = (120., 0, 120., 0, 120., 120., 0, 0, 1)
else:
    image_size=(360, 360)
    intrinsics=(180., 0, 180., 0, 180., 180., 0, 0, 1)

# top down view
position = (0, -0.7, 0.4)
# around x, around y, around z
# starts looking up, positive x is right, positive y is down, positive z is forward
orientation = (np.pi + np.pi / 16, 0, 0)
orientation = pybullet.getQuaternionFromEuler(orientation)

rgb, depth, position, orientation, intrinsics = env.render_image_from_position(position, orientation, image_size=image_size, intrinsics=intrinsics)


# image
plt.imshow(rgb)
plt.show()

#depth
plt.imshow(depth)
plt.show()

#%

# %%

# save rgb image as png
rgb_image = Image.fromarray(rgb)
rgb_image.save('./AFFGA-Net/demo/input/pybullet_tabletop_2.png')
