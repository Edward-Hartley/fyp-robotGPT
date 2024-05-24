import socket
from fyp_package import config, utils
import math

class RobotClient:
    def __init__(self, host='localhost', port=config.robot_server_port):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))

    def _send_command(self, command):
        utils.send_data(self.client_socket, command)
        return utils.recv_data(self.client_socket)

    def move_robot(self, position, orientation_q=None, relative=False):
        command = {
            'command': 'move_robot',
            'position': list(position),
            'orientation_q': list(orientation_q) if orientation_q is not None else orientation_q,
            'relative': relative
        }
        return self._send_command(command)
    
    def get_robot_pose(self):
        command = {
            'command': 'move_robot',
            'position': [0, 0, 0],
            'orientation_q': [0, 0, 0, 1],
            'relative': True
        }
        return self._send_command(command)
    
    def move_fingers(self, finger_positions, relative=False):
        command = {
            'command': 'move_fingers',
            'finger_positions': finger_positions,
            'relative': relative
        }
        return self._send_command(command)
    
    def open_gripper(self):
        command = {
            'command': 'open_gripper'
        }
        return self._send_command(command)
    
    def close_gripper(self):
        command = {
            'command': 'close_gripper'
        }
        return self._send_command(command)
    
    def _max_finger_positions(self):
        return [9.45, 9.45]

    def close(self):
        self.client_socket.close()

if __name__ == "__main__":
    rc = RobotClient()
    # print(rc.move_robot(config.robot_ready_position, config.robot_vertical_orientation_q, relative=False))
    # print(rc.move_robot([0, 0, 0.02], [0, 0, 0, 1], relative=True))
    # print(rc.move_fingers([4.5, 0], relative=False))
    pos = rc.get_robot_pose()[0]

    # vertical down
    print(rc.move_robot(pos, [1, 0, 0, 0], relative=False))

    rc.close()
