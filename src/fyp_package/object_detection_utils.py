import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from fyp_package import config, agent_logging
from fyp_package.utils import tf
from shapely.geometry import MultiPoint, Polygon, polygon

@agent_logging.log_object_cube_calculations
def get_object_cube_from_segmentation(masks, segmentation_texts, image, depth_array, camera_position, camera_orientation_q, camera_intrinsics):

    cubes_coords, cubes_orients = get_bounding_cube_from_point_cloud(image, masks, depth_array, camera_position, camera_orientation_q, camera_intrinsics)

    results = [{} for _ in range(len(segmentation_texts))]

    for i, cube_coords in enumerate(cubes_coords):

        print("Detection " + str(i + 1))

        side_1 = np.around(np.linalg.norm(cube_coords['bottom']['corners'][1] - cube_coords['bottom']['corners'][0]), 3)
        side_2 = np.around(np.linalg.norm(cube_coords['bottom']['corners'][2] - cube_coords['bottom']['corners'][1]), 3)
        height = np.around(np.linalg.norm(cube_coords['top']['corners'][0] - cube_coords['bottom']['corners'][0]), 3)

        width = min(side_1, side_2)
        length = max(side_1, side_2)

        print("Position of " + segmentation_texts[i] + ":", list(np.around(cube_coords['top']['center'], 3)))
        results[i]['position'] = list(np.around(cube_coords['top']['center'], 3))

        print("Dimensions:")
        print("Width:", width)
        print("Length:", length)
        print("Height:", height)
        results[i]['width'] = width
        results[i]['length'] = length
        results[i]['height'] = height

        if width < length:
            print("Orientation along shorter side (width):", np.around(cubes_orients[i][0], 3))
            print("Orientation along longer side (length):", np.around(cubes_orients[i][1], 3), "\n")
            results[i]['orientation'] = {'width': np.around(cubes_orients[i][0], 3), 'length': np.around(cubes_orients[i][1], 3)}
        else:
            print("Orientation along shorter side (length):", np.around(cubes_orients[i][1], 3))
            print("Orientation along longer side (width):", np.around(cubes_orients[i][0], 3), "\n")
            results[i]['orientation'] = {'length': np.around(cubes_orients[i][1], 3), 'width': np.around(cubes_orients[i][0], 3)}

    print("Total number of detections made:", len(segmentation_texts))

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

    step = 10
    if np.count_nonzero(mask) > (mask_width * mask_height) / 4:
        # object is very large, reduce number of points
        step = 20

    # convert mask to binary image
    thresh = np.zeros((mask_width, mask_height), dtype=np.uint8)
    thresh[mask] = 255

    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_TC89_L1)
    cnt = contours[0]

    contour_index = None

    max_length = 0
    for c, contour in enumerate(contours):
        contour_points = [(c, r) for r in range(0, mask_height, step) for c in range(0, mask_width, step) if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1]
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

    for mask in masks:

        contour = get_max_contour(mask, image_width, image_height)

        # overlay contour on image, image is a PIL image
        # overlayed = cv.drawContours(cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR), [contour], 0, (0,255,0), 3)
        # cv.imshow("overlayed", overlayed)
        # cv.waitKey(0)

        step = 10
        if np.count_nonzero(mask) > (image_width * image_height) / 4:
            # object is very large, reduce number of points
            step = 20

        if contour is None:
            if config.simulation == False:
                conservative_mask = erode_mask(mask, 20)
            else:
                conservative_mask = erode_mask(mask, 5)
            contour_pixel_points = [(c, r, depth_array[r][c]) for r in range(0, image_width-1, step) for c in range(0, image_height-1, step) if conservative_mask[r][c] and depth_array[r][c] not in config.invalid_depth_values]
        else:
            contour_pixel_points = [(c, r, depth_array[r][c]) for r in range(0, image_height, step) for c in range(0, image_width, step) if cv.pointPolygonTest(contour, (c, r), measureDist=False) == 1 and depth_array[r][c] not in config.invalid_depth_values]
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
            bounding_cube = {'top': {}, 'bottom': {}}

            rect = polygon.orient(rect, sign=-1)
            box = rect.exterior.coords
            box = np.array(box[:-1])
            box_min_x = np.argmin(box[:, 0])
            box = np.roll(box, -box_min_x, axis=0)
            box_top = [list(point) + [max_z_coordinate] for point in box]
            box_btm = [list(point) + [min_z_coordinate] for point in box]
            box_top_mean = list(np.mean(box_top, axis=0))
            box_btm_mean = list(np.mean(box_btm, axis=0))

            bounding_cube['top']['corners'] = np.array(box_top)
            bounding_cube['bottom']['corners'] = np.array(box_btm)
            bounding_cube['top']['center'] = np.array(box_top_mean)
            bounding_cube['bottom']['center'] = np.array(box_btm_mean)

            bounding_cubes.append(bounding_cube)

            # Calculating rotation in world frame
            bounding_cubes_orientation_width = np.arctan2(box[1][1] - box[0][1], box[1][0] - box[0][0])
            bounding_cubes_orientation_length = np.arctan2(box[2][1] - box[1][1], box[2][0] - box[1][0])
            bounding_cubes_orientations.append([bounding_cubes_orientation_width, bounding_cubes_orientation_length])

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

def erode_mask(mask, erosion_size):
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    return cv.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)


if __name__ == '__main__':
    masks = np.load(config.latest_segmentation_masks_path)
    phrases = ["paper cup"] * len(masks)

    plt.imshow(masks[0])
    plt.show()
    plt.imshow(erode_mask(masks[0], 20))
    plt.show()

    depth_array = np.load(config.latest_depth_image_path)

    plt.imshow(depth_array)
    plt.show()

    # plot mask over depth image
    plt.imshow(depth_array)
    plt.imshow(masks[0], alpha=0.5)
    plt.show()

    # plot eroded mask over depth image
    plt.imshow(depth_array)
    plt.imshow(erode_mask(masks[0], 20), alpha=0.5)
    plt.show()

    results = get_object_cube_from_segmentation(masks, phrases, cv.imread(config.latest_rgb_image_path), np.load(config.latest_depth_image_path), config.camera_position, config.camera_orientation_q, config.intrinsics)
    print(results)
