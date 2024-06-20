import numpy as np
import os
import cv2
from fyp_package import config, utils
from fyp_package.perception_models import model_client

models = model_client.ModelClient()

def detect_grasp(mask, depth):
        depth_path = config.chosen_depth_image_path
        mask_path = config.chosen_segmentation_mask_path
        np.save(depth_path, depth)
        np.save(mask_path, mask)

        grasp2cam_tf, _score, contact_point_cam = models.graspnet_predict(depth_path=depth_path, rgb_path=None, mask_path=mask_path, save=True)
        grasp2base_tf = config.cam2base_tf @ grasp2cam_tf

        contact_point = config.cam2base_tf @ np.concatenate([contact_point_cam, [1]])
        grasp_position = utils.tf_trans(grasp2base_tf)
        grasp_orientation = utils.tf_rot(grasp2base_tf)

        print("Detected grasp with contact point:", contact_point)
        # print("Detected grasp with position:", grasp_position)
        # print("Detected grasp with orientation:", grasp_orientation)

        return contact_point # , grasp_position, grasp_orientation

mask = np.load(config.latest_segmentation_masks_path)[0]
depth = np.load(config.latest_depth_image_path)

detect_grasp(mask, depth)