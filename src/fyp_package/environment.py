# Represents the 'physical' environment in which the agent operates
# Primarily used for interacting with robot and camera

from abc import ABC, abstractmethod
import numpy as np
import cv2
import time

from fyp_package import config, robot_client, realsense_camera, pick_and_place_env, utils

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
    def __init__(self):
        super().__init__(config.real_object_list)
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
        return self.robot.get_robot_pose()

    def open_gripper(self):
        return self.robot.open_gripper()

    def close_gripper(self):
        return self.robot.close_gripper()

    def move_robot(self, position=config.robot_ready_position, orientation=None, relative=False):
        return self.robot.move_robot(position, orientation, relative)

    # pick orientation is a scalar representing the angle of rotation around the z-axis
    # it should be in the range [-pi/2, pi/2]
    def put_first_on_second(self, pick_pos, place_pos, pick_angle=None):
        if pick_angle is None:
            pick_orientation = config.robot_vertical_orientation_q
        else:
            rotation_euler = [0, 0, pick_angle]
            pick_orientation = utils.rot2quat(utils.quat2rot(config.robot_vertical_orientation_q) @ utils.quat2rot(utils.euler2quat(*rotation_euler)))


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
            place_xyz = place_pos
            place_xyz[2] = 0.15

        # Move to object.
        ee_xyz = self.get_ee_pose()[0]
        while np.linalg.norm(hover_xyz - ee_xyz) > 0.03:
            self.move_robot(hover_xyz, pick_orientation)
            ee_xyz = self.get_ee_pose()[0]

        while np.linalg.norm(pick_xyz - ee_xyz) > 0.03:
            self.move_robot(pick_xyz, pick_orientation)
            ee_xyz = self.get_ee_pose()[0]

        # Pick up object.
        self.close_gripper()
        time.sleep(1)

        while np.linalg.norm(hover_xyz - ee_xyz) > 0.03:
            self.move_robot(hover_xyz)
            ee_xyz = self.get_ee_pose()[0]

        # Move to place location.
        while np.linalg.norm(place_xyz - ee_xyz) > 0.03:
            self.move_robot(place_xyz)
            ee_xyz = self.get_ee_pose()[0]

        # Place down object.
        place_xyz[2] = 0.1
        while np.linalg.norm(place_xyz - ee_xyz) > 0.03:
            self.move_robot(place_xyz)
            ee_xyz = self.get_ee_pose()[0]

        self.open_gripper()
        time.sleep(1)

        place_xyz[2] = 0.2
        ee_xyz = self.get_ee_pose()[0]
        while np.linalg.norm(place_xyz - ee_xyz) > 0.03:
            self.move_robot(place_xyz)
            ee_xyz = self.get_ee_pose()[0]

        self.reset_robot()

        return True

    def get_images(self, save=False, save_path_rgb=config.latest_rgb_image_path, save_path_depth=config.latest_depth_image_path):
        color_image, depth_image = self.camera.get_frame(save, save_path_rgb, save_path_depth)
        return color_image, depth_image

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
        return self.sim.get_ee_pose()

    def open_gripper(self):
        self.sim.gripper.release()
        for _ in range(240):
            self.sim.step_sim_and_render()

    def close_gripper(self):
        self.sim.gripper.activate()
        for _ in range(240):
            self.sim.step_sim_and_render()

    def move_robot(self, position, orientation=None, relative=False):
        if relative:
            position = np.array(position) + np.array(self.get_ee_pose()[0])
        self.sim.move_ee(position, orientation)
        return self.get_ee_pose()

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




    env = SimulatedEnvironment(3, 3)
    rgb, depth = env.get_images(save=True)
    print(env.get_ee_pose())

    current_quat = env.get_ee_pose()[1]
    # while True:
    #     euler = eval(input("Enter euler angles: "))
    #     print(utils.quat2euler(env.get_ee_pose()[1]))
    #     new_quat = utils.rot2quat(utils.quat2rot(current_quat) @ utils.quat2rot(utils.euler2quat(*euler)))

    #     env.move_robot([0, -0.5, 0.2], new_quat)
    #     env.move_robot([0, -0.5, 0.3], new_quat)
    #     env.move_robot([0, -0.5, 0.2], new_quat)
    #     env.move_robot([0, -0.5, 0.3], new_quat)

    #     print(utils.quat2euler(new_quat))


    colour = input("Press Enter to pick and place")
    block = env.sim.get_obj_pos(f'{colour} block')
    bowl = env.sim.get_obj_pos(f'{colour} bowl')
    env.put_first_on_second(block, bowl, np.pi/2)
