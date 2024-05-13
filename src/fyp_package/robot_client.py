import socket
import json

class RobotClient:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))

    def _send_command(self, command):
        self.client_socket.sendall(json.dumps(command).encode('utf-8'))
        response = self.client_socket.recv(1024)
        return json.loads(response.decode('utf-8'))

    def move_robot(self, position, orientation_q, relative=False):
        command = {
            'command': 'move_robot',
            'position': list(position),
            'orientation_q': list(orientation_q),
            'relative': relative
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
    print(rc.move_robot([0, 0, 0.02], [0, 0, 0, 1], relative=True))
    print(rc.move_fingers([4.5, 0], relative=False))
    print(rc.open_gripper())

    rc.close()
