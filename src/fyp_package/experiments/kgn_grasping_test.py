#%%

demo_image_path = './KGN_grasping/src/demo/color_img_0.png'
demo_depth_path = './KGN_grasping/src/demo/depth_raw_0.npy'
demo_poses_path = './KGN_grasping/src/demo/poses_0.npz'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

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

# create environment
env = franka_env.PickPlaceEnv(render=True)

# get camera image and depth from pybullet env
if not env.high_res:
    image_size = (480, 480)
    intrinsics = (120., 0, 120., 0, 120., 120., 0, 0, 1)
else:
    image_size=(360, 360)
    intrinsics=(180., 0, 180., 0, 180., 180., 0, 0, 1)

rgb, depth, position, orientation, intrinsics = env.render_image(image_size, intrinsics)


# image
plt.imshow(rgb)
plt.show()

#depth
plt.imshow(depth)
plt.show()

#%
