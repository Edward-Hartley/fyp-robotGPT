# Represents the 'physical' environment in which the agent operates
# Primarily used for interacting with robot and camera

from abc import ABC, abstractmethod
import numpy as np
import cv2
import time

from fyp_package import config, robot_client, realsense_camera, pick_and_place_env, utils, agent_logging

class Environment(ABC):
    def __init__(self, obj_list):
        self.obj_list = obj_list

    @abstractmethod
    def get_ee_pose(self):
        pass

    @abstractmethod
    def open_gripper(self):
        pass

    @abstractmethod
    def close_gripper(self):
        pass

    @abstractmethod
    def move_robot(self, position=config.robot_ready_position, orientation=None, relative=False):
        pass

    @abstractmethod
    def get_images(self, save=False, save_path_rgb=None, save_path_depth=None):
        pass

    @abstractmethod
    def put_first_on_second(self, pick_pos, place_pos, pick_angle=None):
        """
        Perform a pick and place operation.

        Args:
            pick_pos (tuple): Target position which can be either 2D or 3D, but the Z will be overridden either way.
            place_pos (tuple): Target position which can be either 2D or 3D, but the Z will be overridden either way.
            pick_angle (float): Optional angle of rotation around the z-axis in radians. Defaults to None.

        Returns:
            bool: True if the operation was successful, False otherwise.
        """
        pass

# For the physical robot and camera
class PhysicalEnvironment(Environment):
    def __init__(self, object_list=config.real_object_list):
        super().__init__(object_list)
        # Initialize your robot client and camera here
        self.robot = self.initialize_robot()
        self.reset_robot()
        self.camera = self.initialize_camera()

    def initialize_robot(self):
        robot = robot_client.RobotClient()
        return robot
    
    def reset_robot(self):
        self.robot.move_robot(config.robot_ready_position, config.robot_vertical_orientation_q)
        self.robot.open_gripper()

    def initialize_camera(self):
        return realsense_camera.RealsenseCamera()

    def get_ee_pose(self):
        pos, quat = self.robot.get_robot_pose()
        euler = utils.quat2euler(quat)
        adjusted_euler = utils.rotate_euler_by_inverse_of_quat(euler, config.robot_vertical_orientation_q)
        return pos, adjusted_euler

    @agent_logging.log_gripper_action
    def open_gripper(self):
        return self.robot.open_gripper()

    @agent_logging.log_gripper_action
    def close_gripper(self):
        return self.robot.close_gripper()

    @agent_logging.log_move_robot
    def move_robot(self, position=config.robot_ready_position, orientation_e=[0, 0, 0], relative=False):
        # can be actively set to None despite default value
        if orientation_e is None:
            orientation_e = [0, 0, 0]

        if relative:
            orientation = utils.euler2quat(*orientation_e)
        else:
            orientation = utils.rotate_quat_by_euler(config.robot_vertical_orientation_q, orientation_e)

        self.robot.move_robot(position, orientation, relative)
        return self.get_ee_pose()

    # pick orientation is a scalar representing the angle of rotation around the z-axis
    # it should be in the range [-pi/2, pi/2]
    @agent_logging.log_put_first_on_second
    def put_first_on_second(self, pick_pos, place_pos, pick_angle=None):
        if pick_angle is None:
            pick_orientation = config.robot_vertical_orientation_q
        else:
            rotation_euler = [0, 0, pick_angle]
            pick_orientation = utils.rotate_quat_by_euler(config.robot_vertical_orientation_q, rotation_euler)

        pick_pos = np.array(pick_pos)
        place_pos = np.array(place_pos)
        # Set fixed primitive z-heights.
        hover_xyz = np.float32([pick_pos[0], pick_pos[1], 0.2])
        if pick_pos.shape[-1] == 2:
            pick_xyz = np.append(pick_pos, 0.025)
        else:
            pick_xyz = np.float32([pick_pos[0], pick_pos[1], pick_pos[2]])
        if place_pos.shape[-1] == 2:
            place_xyz = np.append(place_pos, 0.15)
        else:
            place_xyz = np.float32([place_pos[0], place_pos[1], place_pos[2] + 0.03])

        # Move to object.
        ee_xyz = self.get_ee_pose()[0]
        while np.linalg.norm(hover_xyz - ee_xyz) > 0.03:
            self.robot.move_robot(hover_xyz, pick_orientation)
            ee_xyz = self.get_ee_pose()[0]

        while np.linalg.norm(pick_xyz - ee_xyz) > 0.03:
            self.robot.move_robot(pick_xyz, pick_orientation)
            ee_xyz = self.get_ee_pose()[0]

        # Pick up object.
        self.close_gripper()
        time.sleep(1)

        while np.linalg.norm(hover_xyz - ee_xyz) > 0.03:
            self.robot.move_robot(hover_xyz)
            ee_xyz = self.get_ee_pose()[0]

        # Move to place location.
        while np.linalg.norm(place_xyz - ee_xyz) > 0.03:
            self.robot.move_robot(place_xyz)
            ee_xyz = self.get_ee_pose()[0]

        # Place down object.
        place_xyz[2] = 0.1
        while np.linalg.norm(place_xyz - ee_xyz) > 0.03:
            self.robot.move_robot(place_xyz)
            ee_xyz = self.get_ee_pose()[0]

        self.open_gripper()
        time.sleep(1)

        place_xyz[2] = 0.2
        ee_xyz = self.get_ee_pose()[0]
        while np.linalg.norm(place_xyz - ee_xyz) > 0.03:
            self.robot.move_robot(place_xyz)
            ee_xyz = self.get_ee_pose()[0]

        self.reset_robot()

        return True

    def get_images(self, save=False, save_path_rgb=config.latest_rgb_image_path, save_path_depth=config.latest_depth_image_path):
        bgr, depth_image = self.camera.get_frame(save, save_path_rgb, save_path_depth)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, depth_image

# For the pybullet simulation environment
class SimulatedEnvironment(Environment):
    def __init__(self, num_blocks, num_bowls):
        # Initialize your pybullet environment here
        self.sim = pick_and_place_env.PickPlaceEnv(render=True)
        block_list = pick_and_place_env.ALL_BLOCKS[:num_blocks]
        bowl_list = pick_and_place_env.ALL_BOWLS[:num_bowls]
        obj_list = block_list + bowl_list

        super().__init__(obj_list)

        self.initialize_simulation()

    def initialize_simulation(self):
        self.sim.reset(self.obj_list)

    def get_ee_pose(self):
        pos, quat = self.sim.get_ee_pose()
        euler = utils.quat2euler(quat)
        adjusted_euler = utils.rotate_euler_by_inverse_of_quat(euler, config.sim_robot_vertical_q)
        # Rotate the orientation back by 90 degrees to match the orientation of the robot in the physical environment
        adjusted_euler = np.array([-adjusted_euler[1], adjusted_euler[0], adjusted_euler[2]])

        return pos, adjusted_euler

    @agent_logging.log_gripper_action
    def open_gripper(self):
        self.sim.gripper.release()
        for _ in range(240):
            self.sim.step_sim_and_render()

    @agent_logging.log_gripper_action
    def close_gripper(self):
        self.sim.gripper.activate()
        for _ in range(240):
            self.sim.step_sim_and_render()

    @agent_logging.log_move_robot
    def move_robot(self, position, orientation_e=[0, 0, 0], relative=False):
        # can be actively set to None despite default value
        if orientation_e is None:
            orientation_e = [0, 0, 0]

        # Rotate the orientation by 90 degrees to match the orientation of the robot in the physical environment
        orientation_e = np.array([orientation_e[1], -orientation_e[0], orientation_e[2]])

        if relative:
            orientation = utils.rotate_quat_by_euler(self.get_ee_pose()[1], orientation_e)
            position = np.array(position) + np.array(self.get_ee_pose()[0])
        else:
            orientation = utils.rotate_quat_by_euler(config.sim_robot_vertical_q, orientation_e)
            
        self.sim.move_ee(position, orientation)
        return self.get_ee_pose()

    @agent_logging.log_put_first_on_second
    def put_first_on_second(self, pick_pos, place_pos, pick_angle=None):
        self.sim.step(action={'pick': np.array(pick_pos), 'place': np.array(place_pos), 'pick_angle': pick_angle})
        return True

    def get_images(self, save=False, save_path_rgb=config.latest_rgb_image_path, save_path_depth=config.latest_depth_image_path):
        color_image_rgb, depth_image, _, _ = self.sim.render_image()
        if save:
            if save_path_rgb:
                utils.save_numpy_image(save_path_rgb, color_image_rgb)
            if save_path_depth:
                np.save(save_path_depth, depth_image)
        return color_image_rgb, depth_image
    

if __name__ == "__main__":

    # # For physical environment
    # physical_env = PhysicalEnvironment()
    # physical_env.move_robot([0, 0, 0])
    # physical_env.open_gripper()

    # # For simulated environment
    # simulated_env = SimulatedEnvironment(3, 3)
    # simulated_env.move_robot([0, 0, 0])
    # simulated_env.open_gripper()
    agent_logging.setup_logging()



    physical_env = PhysicalEnvironment()
    physical_env.move_robot([-0.206, -0.293, 0.136])
    input("Press Enter to move to bowl")
    # sim_env = SimulatedEnvironment(3, 3)

    # rgb1, depth1 = sim_env.get_images(save=True)
    # rgb2, depth2 = physical_env.get_images(save=True)
    import matplotlib.pyplot as plt
    # plt.imshow(rgb1)
    # plt.show()
    # plt.imshow(rgb2)
    # plt.show()
    sim_env=4



    # rgb, depth = env.get_images(save=True)
    print("sim:", sim_env.get_ee_pose())
    # print("physical:", physical_env.get_ee_pose())

    input("Press Enter to move to bowl")
    euler = [0, 0, 0]
    red_block_pos = sim_env.sim.get_obj_pos('red block')
    # add noise
    red_block_pos = [red_block_pos[0] + 0.015, red_block_pos[1] + 0.01, red_block_pos[2]]

    hover_pos = [red_block_pos[0], red_block_pos[1], red_block_pos[2] + 0.1]
    print("Hover over red block: ", sim_env.move_robot(hover_pos, [0, 0, 0]))
    print("Move to red block: ", sim_env.move_robot(red_block_pos, [0, 0, 0]))
    print("Close gripper: ", sim_env.close_gripper())
    print("Lift red block: ", sim_env.move_robot(hover_pos, [0, 0, 0]))
    input("Press Enter to move to bowl")


    red_bowl_pos = sim_env.sim.get_obj_pos('red bowl')
    # add noise
    red_bowl_pos = [red_bowl_pos[0] + 0.015, red_bowl_pos[1] + 0.01, red_bowl_pos[2]]

    hover_bowl_pos = [red_bowl_pos[0], red_bowl_pos[1], red_bowl_pos[2] + 0.1]    
    print("Hover over red bowl: ", sim_env.move_robot(hover_bowl_pos, [0, 0, 0]))
    place_pos = [red_bowl_pos[0], red_bowl_pos[1], red_bowl_pos[2] + 0.05]
    print("Move to red bowl: ", sim_env.move_robot(place_pos, [0, 0, 0]))
    print("Open gripper: ", sim_env.open_gripper())
    print("Return to hover: ", sim_env.move_robot(hover_bowl_pos, [0, 0, 0]))

    input("Press Enter to reset robot")

    # relative = False test
    while euler != -1:

        sim_env.move_robot([0, -0.5, 0.2], euler)
        sim_env.move_robot([0, -0.5, 0.3], euler)

        # physical_env.move_robot([0, -0.3, 0.2], euler)
        # physical_env.move_robot([0, -0.3, 0.3], euler)

        print("sim:", sim_env.get_ee_pose()[1])
        # print("physical:", physical_env.get_ee_pose()[1])

        euler = eval(input("Enter euler angles: "))
    # relative = True test

    euler = [0, 0, 0]
    while euler != -1:
        sim_env.move_robot([0, 0, -0.1], euler, relative=True)
        sim_env.move_robot([0, 0, 0.1], euler, relative=True)

        # physical_env.move_robot([0, 0, -0.1], euler, relative=True)
        # physical_env.move_robot([0, 0, 0.1], euler, relative=True)

        print("sim:", sim_env.get_ee_pose()[1])
        # print("physical:", physical_env.get_ee_pose()[1])

        euler = eval(input("Enter euler angles: "))

    # colour = input("Press Enter to pick and place")
    # block = env.sim.get_obj_pos(f'{colour} block')
    # bowl = env.sim.get_obj_pos(f'{colour} bowl')
    # env.put_first_on_second(block, bowl, np.pi/2)
