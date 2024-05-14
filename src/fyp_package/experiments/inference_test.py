from fyp_package.experiments.contact_graspnet.contact_graspnet import config_utils, contact_grasp_estimator, visualization_utils, data
from fyp_package import config

import os
import sys
import numpy as np
import glob
import cv2

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))


checkpoint_dir = config.contact_graspnet_checkpoint_dir

"""
Predict 6-DoF grasp distribution for given model and input data

:param depth: Depth image, 2D numpy array
:param rgb: RGB image, 3D numpy array
:param segmap: Segmentation map, 2D numpy array
:param cam_K: Camera Matrix with intrinsics to convert depth to point cloud, 3x3 matrix
:param local_regions: Crop 3D local regions around given segments. 
:param skip_border_objects: When extracting local_regions, ignore segments at depth map boundary.
:param filter_grasps: Filter and assign grasp contacts according to segmap.
:param z_range: crop point cloud at a minimum/maximum z distance from camera to filter out outlier points. Default: [0.2, 1.8] m
:param forward_passes: Number of forward passes to run on each point cloud. Default: 1
:param results_path: Path to save results. Default: 'contact_graspnet.npz'
"""
def grasp_inference(
        depth,
        rgb=None,
        segmap=None,
        cam_K=config.intrinsics,
        local_regions=True,
        skip_border_objects=False, 
        filter_grasps=True, 
        z_range=[0.2,1.8],
        forward_passes=1,
        results_path=config.latest_grasp_detection_path
    ):
    
    global_config = config_utils.load_config(
        checkpoint_dir, batch_size=forward_passes, arg_configs=[]
        )

    
    # Build the model
    grasp_estimator = contact_grasp_estimator.GraspEstimator(global_config)
    grasp_estimator.build_network()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(save_relative_paths=True)

    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    tf_session = tf.Session(config=config)

    # Load weights
    grasp_estimator.load_weights(tf_session, saver, checkpoint_dir, mode='test')

    pc_segments = {}
    
    if segmap is None and (local_regions or filter_grasps):
        raise ValueError('Need segmentation map to extract local regions or filter grasps')

    print('Converting depth to point cloud(s)...')
    pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
        depth, cam_K, segmap=segmap, rgb=rgb, skip_border_objects=skip_border_objects, z_range=z_range
        )

    print('Generating Grasps...')
    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
        tf_session, pc_full, pc_segments=pc_segments, local_regions=local_regions,
        filter_grasps=filter_grasps, forward_passes=forward_passes
        )  

    # Save results

    best_grasp_idx = np.argmax(scores[True])
    best_score = scores[True][best_grasp_idx]
    best_grasp_cam = pred_grasps_cam[True][best_grasp_idx]
    best_contact_pt = contact_pts[True][best_grasp_idx]
    np.savez(results_path, pred_grasp_cam=best_grasp_cam, score=best_score, contact_pt=best_contact_pt)

    # Visualize results          
    # # filter grasps to only show the best grasp
    # pred_grasps_cam = {True: [best_grasp_cam]}
    # scores = {True: [best_score]}
    # contact_pts = {True: [best_contact_pt]}

    # visualization_utils.show_image(rgb, segmap)
    # visualization_utils.visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)


    return best_grasp_cam, best_score, best_contact_pt


if __name__ == '__main__':

    # mask_path = '/home/edward/Imperial/fyp-robotGPT/assets/pybullet_tabletop_2_mask_0_mask.npy'
    # depth_path = '/home/edward/Imperial/fyp-robotGPT/assets/pybullet_tabletop_2.npy'
    # rgb_path = '/home/edward/Imperial/fyp-robotGPT/assets/pybullet_tabletop_2.png'

    masks_path = config.latest_segmentation_masks_path
    depth_path = config.latest_depth_image_path
    rgb_path = config.latest_rgb_image_path

    # each must be a numpy array
    mask = np.load(masks_path)[0]
    depth = np.load(depth_path)
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)

    grasp_inference(depth, rgb, mask)

    results = np.load(config.latest_grasp_detection_path)
    print(results['pred_grasp_cam'], results['score'], results['contact_pts'])



