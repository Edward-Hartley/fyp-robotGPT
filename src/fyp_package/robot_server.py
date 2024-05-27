import sys
import socket
import signal
from threading import Thread

import roslib; roslib.load_manifest('kinova_demo')
import rospy
import kinova_demo.pose_action_client as pose_client # type: ignore
import kinova_demo.fingers_action_client as finger_client # type: ignore
import numpy as np

from fyp_package import config, utils

poseUnitParserUnit = ['mq', 'mdeg', 'mrad'][0]
fingerUnitParserUnit = ['mm', 'turn', 'percent'][0]

# Robot class
class Robot:
    def __init__(self, robot_type: str = config.robot_type, host='', port=config.robot_server_port):
        self.robot_type = robot_type

        pose_client.kinova_robotTypeParser(self.robot_type)
        finger_client.kinova_robotTypeParser(self.robot_type)
        assert(pose_client.prefix == finger_client.prefix)

        rospy.init_node(pose_client.prefix + 'robot_client')
        print('Connected to ROS node')

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)

    def handle_client(self, client_socket):
        client_socket.settimeout(300)
        while True:
            data = utils.recv_data(client_socket)
            if not data:
                break
            if data['command'] == 'move_robot':
                response = self.move_robot(data['position'], data['orientation_q'], data['relative'])
            elif data['command'] == 'move_fingers':
                response = self.move_fingers(data['finger_positions'], data['relative'])
            elif data['command'] == 'close_gripper':
                response = self.close_gripper()
            elif data['command'] == 'open_gripper':
                response = self.open_gripper()
            else:
                response = {'error': 'Invalid command'}
            utils.send_data(client_socket, response)
        client_socket.close()

    # Move robot to a specific position and orientation
    # When relative is False, the default orientation is vertical
    # When relative is True, by default the orientation is not changed
    def move_robot(self, position=config.robot_ready_position, orientation_q=None, relative=False):
        if orientation_q is None:
            if relative:
                orientation_q = [0, 0, 0, 1]
            else:
                orientation_q = config.robot_vertical_orientation_q
        
        # Sets global vars for current position and orientation of the robot
        pose_client.getcurrentCartesianCommand(pose_client.prefix)
        
        pose_mq, _, _ = pose_client.unitParser(poseUnitParserUnit, (list(position) + list(orientation_q)), relative)
        if relative:
            print('New cartesion pose: ', pose_mq)

        try:
            poses = [float(n) for n in pose_mq]
            result = pose_client.cartesian_pose_client(poses[:3], poses[3:])
            print('Cartesian pose sent!')
            print("result: ", result)
            if result:
                return unpack_pose(result)
            else:
                return None
        except rospy.ROSInterruptException:
            print("program interrupted before completion")

    def move_fingers(self, finger_positions_mm, relative=False):

        # Sets global vars for current finger positions
        finger_client.getCurrentFingerPosition(finger_client.prefix)

        finger_turn, _, _ = finger_client.unitParser('mm', finger_positions_mm, relative)
        
        try:

            positions = [max(0.0, min(n, finger_client.finger_maxTurn)) for n in finger_turn]
            positions = [float(n) for n in positions]

            result = finger_client.gripper_client(positions)
            print('Finger position sent!')
            print("result: ", result)
            if result:
                return unpack_positions(result)
            else:
                return None
        except rospy.ROSInterruptException:
            print('program interrupted before completion')

    def close_gripper(self):
        return self.move_fingers([9.5, 9.5], relative=False)
    
    def open_gripper(self):
        return self.move_fingers([0, 0], relative=False)

    def run(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            print(f"Connection from {addr} has been established!")
            thread = Thread(target=self.handle_client, args=(client_socket,))
            thread.start()

    def __del__(self):
        self.server_socket.close()


def unpack_pose(result):
    return [
        result.pose.pose.position.x,
        result.pose.pose.position.y,
        result.pose.pose.position.z,
        ], [
        result.pose.pose.orientation.x,
        result.pose.pose.orientation.y,
        result.pose.pose.orientation.z,
        result.pose.pose.orientation.w]

def unpack_positions(result):
    return [
        result.fingers.finger1,
        result.fingers.finger2,
    ]
    
if __name__ == '__main__':
    robot = Robot()
    signal.signal(signal.SIGINT, lambda signal, frame: rs.shutdown())
    robot.run()