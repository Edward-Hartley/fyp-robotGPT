import cv2
import numpy as np
import time
from fyp_package.environment import realsense_camera
from fyp_package.environment.environment import robot_client
from fyp_package.scratch_pad.proto import plot_tf, plot_set_axes_equal, AprilGrid
from fyp_package.utils import tf_rot, tf_trans, quat2rot, tf
import matplotlib.pyplot as plt

def inv_rot_and_trans(R, t):
    tf_m = tf(R, t)
    inv_tf_m = np.linalg.inv(tf_m)
    return tf_rot(inv_tf_m), tf_trans(inv_tf_m)

def new_tf_figure(R, t, ax=None):
    if ax is None:
      plt.figure()
      ax = plt.axes(projection='3d')
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      plot_tf(ax, tf(np.eye(3), np.zeros(3)))
    plot_tf(ax, tf(R, t))
    plot_set_axes_equal(ax)

if __name__ == "__main__":
    # aim is to output the extrinsics of the camera with respect to the robot base
    # save rotation and translation to an npy file

    # list of gripper poses to move to
    # aim for: small translation, large rotation
    poses = [
        [[-0.05734804645180702, -0.24750040471553802, 0.23155121505260468], [0.999803241409927, -0.004087502121872842, -0.011318399529475212, -0.015769103484280737]],
        [[-0.05675990507006645, -0.24642615020275116, 0.2465648651123047], [0.9837216963054365, 0.0009614009400748766, -0.0014214338761546076, 0.17969051018883184]],
        [[-0.027455130591988564, -0.24439038336277008, 0.25073909759521484], [0.9730611142918132, 0.22605737345440705, -0.008394657374867956, 0.04449338701007525]],
        [[-0.03237677738070488, -0.24626517295837402, 0.23826178908348083], [0.9961442141757545, 0.030631675641998434, -0.038290036465076845, -0.07274804546539342]],
        [[-0.10537391692399979, -0.24117302894592285, 0.2471749985218048], [0.9875343408668058, -0.15229897149713006, -0.030255840989266373, 0.02579792580251324]],
        [[-0.05199339985847473, -0.2657204866409302, 0.27463568449020386], [0.9510951345128951, -0.0019670311056789866, -0.009123791299809092, 0.3087570765615667]],
        [[-0.03672763705253601, -0.2535059452056885, 0.27129441499710083], [0.9311197951021751, -0.051925647752208835, -0.3300495727758844, 0.14624272216029452]],
        [[-0.033785391598939896, -0.2539682388305664, 0.2692146301269531], [0.9397278072859604, 0.05117859450010156, 0.30467612332092264, 0.146508906065976]]
    ]
    robot = robot_client.RobotClient()
    camera = realsense_camera.RealsenseCamera()
    K = camera.get_intrinsics()

    t_base2grippers = []
    R_base2grippers = []
    t_target2cams = []
    R_target2cams = []

    pose_idx = 0
    while pose_idx < len(poses):
        pose = poses[pose_idx]
        grid = AprilGrid(tag_rows=11, tag_cols=7, tag_size=0.024, tag_spacing=0.104)
        # move robot to pose
        result = robot.move_robot(pose[0], pose[1])
        time.sleep(0.1)
        if result is None:
            result = robot.move_robot([0, 0, 0], [0, 0, 0, 1], relative=True)
            time.sleep(0.1)
        else:
            pose_idx += 1

        # capture image
        # returned as np array hxwxc
        image, _ = camera.get_frame()

        # detect apriltag using opencv
        # make gray from array
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        arucoParams = cv2.aruco.DetectorParameters()
        arucoParams.markerBorderBits = 2

        (all_corners, ids, rejected) = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=arucoParams)

        # reasonable number of matches for good pnp
        if ids is None or len(ids) < 3:
            continue

        t_gripper2base, gripper2base_q = result

        # invert to get base to gripper
        R_base2gripper, t_base2gripper = inv_rot_and_trans(quat2rot(gripper2base_q), t_gripper2base)
        t_base2grippers.append(t_base2gripper)
        R_base2grippers.append(R_base2gripper)

        corner_mappings = {
            0: 1,
            1: 0,
            2: 3,
            3: 2
        }

        # add keypoints to grid
        for tag_id, corners in zip(ids, all_corners):
            for i, corner in enumerate(corners[0]):
                grid.add_keypoint(ts=None, tag_id=tag_id[0], corner_idx=corner_mappings[i], kp=corner)

        # solve pnp
        target2cam_t, target2cam_R = grid.solvepnp(K)

        t_target2cams.append(target2cam_t)
        R_target2cams.append(target2cam_R)

    # cv2 expects sequences of matlike objects
    R_base2grippers = [np.array(x) for x in R_base2grippers]
    t_base2grippers = [np.array(x) for x in t_base2grippers]
    R_target2cams = [np.array(x) for x in R_target2cams]
    t_target2cams = [np.array(x) for x in t_target2cams]

    # calibrate hand-eye function dictates the first two inputs are transforms FROM gripper TO base,
    # however, this is for eye-in-hand calibration (camera is on the gripper). For eye-to-hand calibration
    # (camera is stationary and robot moves), we need the base to gripper transforms, which is what we already have
    # 
    # The next two inputs are the target to camera transforms, again what we already have

    print("calibrating with this many poses: ", len(R_base2grippers))
    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_gripper2base=R_base2grippers,
        t_gripper2base=t_base2grippers,
        R_target2cam=R_target2cams,
        t_target2cam=t_target2cams,
    )
    t_cam2base = t_cam2base.flatten()

    R_base2cam, t_base2cam = inv_rot_and_trans(R_cam2base, t_cam2base)

    # Save the calibration results
    calibration_results = {
        "R_base2cam": R_base2cam,
        "t_base2cam": t_base2cam,
        "R_cam2base": R_cam2base,
        "t_cam2base": t_cam2base
    }
    np.save("./data/calibration_results.npy", calibration_results)

    # Checks
    print("t_base2cam: ", t_base2cam)
    print("R_base2cam: ", R_base2cam)
    print("t_cam2base: ", t_cam2base)
    print("R_cam2base: ", R_cam2base)

    new_tf_figure(R_base2cam, t_base2cam)
    new_tf_figure(R_cam2base, t_cam2base)
    plt.show()

    # object_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    # # project into camera frame
    # image_points, _ = cv2.projectPoints(object_points, cv2.Rodrigues(R_base2cam)[0], t_base2cam, K, np.array([0.0, 0.0, 0.0, 0.0]), None, None)
    # cv2.line(image, tuple(image_points[0][0].astype(int)), tuple(image_points[1][0].astype(int)), (0, 0, 255)) # red
    # cv2.line(image, tuple(image_points[0][0].astype(int)), tuple(image_points[2][0].astype(int)), (0, 255, 0)) # green
    # cv2.line(image, tuple(image_points[0][0].astype(int)), tuple(image_points[3][0].astype(int)), (255, 0, 0)) # blue
    # cv2.imshow('base_check', image)
    # cv2.waitKey(0)

