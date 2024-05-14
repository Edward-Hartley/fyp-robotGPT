import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from fyp_package import config
from fyp_package.utils import tf
from shapely.geometry import MultiPoint, Polygon, polygon

def get_object_cube_from_segmentation(masks, segmentation_texts, image, depth_array, camera_position, camera_orientation_q, camera_intrinsics):

    bounding_cubes_world_coordinates, bounding_cubes_orientations = get_bounding_cube_from_point_cloud(image, masks, depth_array, camera_position, camera_orientation_q, camera_intrinsics)

    results = [{}] * len(segmentation_texts)

    for i, bounding_cube_world_coordinates in enumerate(bounding_cubes_world_coordinates):

        bounding_cube_world_coordinates[4][2] -= config.depth_offset

        object_width = np.around(np.linalg.norm(bounding_cube_world_coordinates[1] - bounding_cube_world_coordinates[0]), 3)
        object_length = np.around(np.linalg.norm(bounding_cube_world_coordinates[2] - bounding_cube_world_coordinates[1]), 3)
        object_height = np.around(np.linalg.norm(bounding_cube_world_coordinates[5] - bounding_cube_world_coordinates[0]), 3)

        print("Position of " + segmentation_texts[i] + ":", list(np.around(bounding_cube_world_coordinates[4], 3)))
        results[i]['position'] = list(np.around(bounding_cube_world_coordinates[4], 3))

        print("Dimensions:")
        print("Width:", object_width)
        print("Length:", object_length)
        print("Height:", object_height)
        results[i]['dimensions'] = {'width': object_width, 'length': object_length, 'height': object_height}

        if object_width < object_length:
            print("Orientation along shorter side (width):", np.around(bounding_cubes_orientations[i][0], 3))
            print("Orientation along longer side (length):", np.around(bounding_cubes_orientations[i][1], 3), "\n")
            results[i]['orientation'] = {'width': np.around(bounding_cubes_orientations[i][0], 3), 'length': np.around(bounding_cubes_orientations[i][1], 3)}
        else:
            print("Orientation along shorter side (length):", np.around(bounding_cubes_orientations[i][1], 3))
            print("Orientation along longer side (width):", np.around(bounding_cubes_orientations[i][0], 3), "\n")
            results[i]['orientation'] = {'length': np.around(bounding_cubes_orientations[i][1], 3), 'width': np.around(bounding_cubes_orientations[i][0], 3)}

    return results

def get_segmentation_mask(model_predictions, segmentation_threshold):

    masks = []

    for model_prediction in model_predictions:
        segmentation_threshold = np.max(model_prediction) - segmentation_threshold * (np.max(model_prediction) - np.min(model_prediction))
        model_prediction[model_prediction < segmentation_threshold] = False
        model_prediction[model_prediction >= segmentation_threshold] = True
        masks.append(model_prediction)

    return masks

def get_max_contour(mask, mask_width, mask_height):

    # convert mask to binary image
    thresh = np.zeros((mask_width, mask_height), dtype=np.uint8)
    thresh[mask] = 255

    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
    cnt = contours[0]

    contour_index = None
    max_length = 0
    for c, contour in enumerate(contours):
        contour_points = [(c, r) for r in range(mask_height) for c in range(mask_width) if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1]
        if len(contour_points) > max_length:
            contour_index = c
            max_length = len(contour_points)

    if contour_index is None:
        return None

    return contours[contour_index]

def get_bounding_cube_from_point_cloud(image, masks, depth_array, camera_position, camera_orientation_q, camera_K):

    image_width, image_height, _ = image.shape
    # plt.imshow(depth_array)
    # plt.show()

    bounding_cubes = []
    bounding_cubes_orientations = []

    for i, mask in enumerate(masks):

        contour = get_max_contour(mask, image_width, image_height)

        # overlay contour on image, image is a PIL image
        overlayed = cv.drawContours(cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR), [contour], 0, (0,255,0), 3)
        # cv.imshow("overlayed", overlayed)
        # cv.waitKey(0)

        if contour is not None:

            contour_pixel_points = [(c, r, depth_array[r][c]) for r in range(image_height) for c in range(image_width) if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1 and depth_array[r][c] != config.invalid_depth_value]
            contour_world_points = get_world_points_world_frame(camera_position, camera_orientation_q, camera_K, contour_pixel_points)

            # # use matplotlib to plot the height map
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(np.array(contour_world_points)[::20, 0], np.array(contour_world_points)[::20, 1], np.array(contour_world_points)[::20, 2])
            # plt.show()


            max_z_coordinate = np.max(np.array(contour_world_points)[:, 2])
            min_z_coordinate = np.min(np.array(contour_world_points)[:, 2])

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



def get_world_points_world_frame(camera_position, camera_orientation_q, camera_K, points):

    cam2world_m = tf(camera_orientation_q, camera_position)

    # points are x, y, depth
    points = np.array(points)

    # convert to camera frame
    # adjust for center of image
    points = points - np.array([camera_K[0, 2], camera_K[1, 2], 0])
    # adjust for depth and focal length
    scaling_factor = np.vstack((points[:, 2] / camera_K[0, 0],
                               points[:, 2] / camera_K[1, 1],
                               np.ones(points.shape[0])))
    points = points.T * scaling_factor

    # convert to world frame
    points = np.vstack((points, np.ones((1, points.shape[1]))))
    points = cam2world_m @ points
    points = points.T

    return points[:, :3]
