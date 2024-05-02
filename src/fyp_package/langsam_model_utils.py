import pybullet as p
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import math
from fyp_package import config
from torchvision import transforms
from torchvision.utils import save_image, draw_bounding_boxes, draw_segmentation_masks
from shapely.geometry import MultiPoint, Polygon, polygon

def langsam_predict(model, image, prompt):
    # model: LangSAM instance
    # image: PIL Image
    # prompt: string
    masks, boxes, phrases, logits = model.predict(image, prompt)

    _, ax = plt.subplots(1, 1 + len(masks), figsize=(5 + (5 * len(masks)), 5))
    [a.axis("off") for a in ax.flatten()]
    ax[0].imshow(image)

    for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
        to_tensor = transforms.PILToTensor()
        image_tensor = to_tensor(image)
        box = box.unsqueeze(dim=0)
        image_tensor = draw_bounding_boxes(image_tensor, box, colors=["red"], width=3)
        image_tensor = draw_segmentation_masks(image_tensor, mask, alpha=0.5, colors=["cyan"])
        to_pil_image = transforms.ToPILImage()
        image_pil = to_pil_image(image_tensor)

        ax[1 + i].imshow(image_pil)
        ax[1 + i].text(box[0][0], box[0][1] - 15, phrase, color="red", bbox={"facecolor":"white", "edgecolor":"red", "boxstyle":"square"})

    plt.show()

    return masks.float(), boxes, phrases

def get_segmentation_mask(model_predictions, segmentation_threshold):

    masks = []

    for model_prediction in model_predictions:
        model_prediction_np = model_prediction.detach().cpu().numpy()
        segmentation_threshold = np.max(model_prediction_np) - segmentation_threshold * (np.max(model_prediction_np) - np.min(model_prediction_np))
        model_prediction[model_prediction < segmentation_threshold] = False
        model_prediction[model_prediction >= segmentation_threshold] = True
        masks.append(model_prediction)

    return masks



def get_max_contour(image, image_width, image_height):

    ret, thresh = cv.threshold(image, 127, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
    cnt = contours[0]

    contour_index = None
    max_length = 0
    for c, contour in enumerate(contours):
        contour_points = [(c, r) for r in range(image_height) for c in range(image_width) if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1]
        if len(contour_points) > max_length:
            contour_index = c
            max_length = len(contour_points)

    if contour_index is None:
        return None

    return contours[contour_index]



def get_extrinsics(camera_position, camera_orientation_q):

    R = np.array(p.getMatrixFromQuaternion(camera_orientation_q)).reshape(3, 3)
    Rt = np.hstack((R, np.array(camera_position).reshape(3, 1)))
    Rt = np.vstack((Rt, np.array([0, 0, 0, 1])))

    return Rt



def get_bounding_cube_from_point_cloud(image, masks, depth_array, camera_position, camera_orientation_q):

    image_width, image_height = image.size
    # plt.imshow(depth_array)
    # plt.show()

    bounding_cubes = []
    bounding_cubes_orientations = []

    for i, mask in enumerate(masks):

        save_image(mask, config.bounding_cube_mask_image_path.format(object=1, mask=i))
        mask_np = cv.imread(config.bounding_cube_mask_image_path.format(object=1, mask=i), cv.IMREAD_GRAYSCALE)


        contour = get_max_contour(mask_np, image_width, image_height)

        # overlay contour on image, image is a PIL image
        overlayed = cv.drawContours(cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR), [contour], 0, (0,255,0), 3)
        # cv.imshow("overlayed", overlayed)
        # cv.waitKey(0)


        if contour is not None:

            contour_pixel_points = [(c, r, depth_array[r][c]) for r in range(image_height) for c in range(image_width) if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1]
            contour_world_points = get_world_points_world_frame(camera_position, camera_orientation_q, "head", contour_pixel_points)

            # # use matplotlib to plot the height map
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(np.array(contour_world_points)[::20, 0], np.array(contour_world_points)[::20, 1], np.array(contour_world_points)[::20, 2])
            plt.show()


            max_z_coordinate = np.max(np.array(contour_world_points)[:, 2])
            min_z_coordinate = np.min(np.array(contour_world_points)[:, 2])
            # previously only got top surface, going to see if performance drop is too severe by getting all points
            # top_surface_world_points = [world_point for world_point in contour_world_points if world_point[2] > max_z_coordinate - config.depth_offset]
            # rect = MultiPoint([world_point[:2] for world_point in top_surface_world_points]).minimum_rotated_rectangle

            rect = MultiPoint([world_point[:2] for world_point in contour_world_points]).minimum_rotated_rectangle

            if isinstance(rect, Polygon):
                rect = polygon.orient(rect, sign=-1)
                box = rect.exterior.coords
                box = np.array(box[:-1])
                box_min_x = np.argmin(box[:, 0])
                box = np.roll(box, -box_min_x, axis=0)
                box_top = [list(point) + [max_z_coordinate] for point in box]
                box_btm = [list(point) + [min_z_coordinate] for point in box]
                box_top.append(list(np.mean(box_top, axis=0)))
                box_btm.append(list(np.mean(box_btm, axis=0)))
                bounding_cubes.append(box_top + box_btm)

                # Calculating rotation in world frame
                bounding_cubes_orientation_width = np.arctan2(box[1][1] - box[0][1], box[1][0] - box[0][0])
                bounding_cubes_orientation_length = np.arctan2(box[2][1] - box[1][1], box[2][0] - box[1][0])
                bounding_cubes_orientations.append([bounding_cubes_orientation_width, bounding_cubes_orientation_length])

    bounding_cubes = np.array(bounding_cubes)

    return bounding_cubes, bounding_cubes_orientations



def get_world_points_world_frame(camera_position, camera_orientation_q, camera, points):

    Rt = get_extrinsics(camera_position, camera_orientation_q)
    # intrinsics are 3x3 matrix
    intrinsics = config.intrinsics

    # points are x, y, depth
    points = np.array(points)

    # convert to camera frame
    # adjust for center of image
    points = points - np.array([intrinsics[0, 2], intrinsics[1, 2], 0])
    # adjust for depth and focal length
    scaling_factor = np.vstack((points[:, 2] / intrinsics[0, 0],
                               points[:, 2] / intrinsics[1, 1],
                               np.ones(points.shape[0])))
    points = points.T * scaling_factor

    # convert to world frame
    points = np.vstack((points, np.ones((1, points.shape[1]))))
    points = Rt @ points
    points = points.T

    return points[:, :3]

