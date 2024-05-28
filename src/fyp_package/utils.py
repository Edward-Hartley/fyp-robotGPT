import numpy as np
from math import cos, sin, sqrt
import pickle
import cv2

# as defined in proto.py but using math for trigonometric functions
def euler2quat(x, y, z):
  """
  Convert yaw, pitch, roll in radians to a quaternion.

  Source:
  Kuipers, Jack B. Quaternions and Rotation Sequences: A Primer with
  Applications to Orbits, Aerospace, and Virtual Reality. Princeton, N.J:
  Princeton University Press, 1999. Print.
  [Page 166-167, "Euler Angles to Quaternion"]
  """
  psi = z  # Yaw
  theta = y  # Pitch
  phi = x  # Roll

  c_phi = cos(phi / 2.0)
  c_theta = cos(theta / 2.0)
  c_psi = cos(psi / 2.0)
  s_phi = sin(phi / 2.0)
  s_theta = sin(theta / 2.0)
  s_psi = sin(psi / 2.0)

  qx = c_psi * c_theta * s_phi - s_psi * s_theta * c_phi
  qy = c_psi * s_theta * c_phi + s_psi * c_theta * s_phi
  qz = s_psi * c_theta * c_phi - c_psi * s_theta * s_phi
  qw = c_psi * c_theta * c_phi + s_psi * s_theta * s_phi

  mag = sqrt(qx**2 + qy**2 + qz**2 + qw**2)
  return np.array([qx / mag, qy / mag, qz / mag, qw / mag])

def rot2quat(C):
  """
  Convert 3x3 rotation matrix to quaternion.
  """
  assert C.shape == (3, 3)

  m00 = C[0, 0]
  m01 = C[0, 1]
  m02 = C[0, 2]

  m10 = C[1, 0]
  m11 = C[1, 1]
  m12 = C[1, 2]

  m20 = C[2, 0]
  m21 = C[2, 1]
  m22 = C[2, 2]

  tr = m00 + m11 + m22

  if tr > 0:
    S = sqrt(tr + 1.0) * 2.0
    # S=4*qw
    qx = (m21 - m12) / S
    qy = (m02 - m20) / S
    qz = (m10 - m01) / S
    qw = 0.25 * S
  elif ((m00 > m11) and (m00 > m22)):
    S = sqrt(1.0 + m00 - m11 - m22) * 2.0
    # S=4*qx
    qx = 0.25 * S
    qy = (m01 + m10) / S
    qz = (m02 + m20) / S
    qw = (m21 - m12) / S
  elif m11 > m22:
    S = sqrt(1.0 + m11 - m00 - m22) * 2.0
    # S=4*qy
    qx = (m01 + m10) / S
    qy = 0.25 * S
    qz = (m12 + m21) / S
    qw = (m02 - m20) / S
  else:
    S = sqrt(1.0 + m22 - m00 - m11) * 2.0
    # S=4*qz
    qx = (m02 + m20) / S
    qy = (m12 + m21) / S
    qz = 0.25 * S
    qw = (m10 - m01) / S

  return quat_normalize(np.array([qx, qy, qz, qw]))

def quat2rot(q):
  """
  Convert quaternion to 3x3 rotation matrix.

  Source:
  Blanco, Jose-Luis. "A tutorial on se (3) transformation parameterizations
  and on-manifold optimization." University of Malaga, Tech. Rep 3 (2010): 6.
  [Page 18, Equation (2.20)]
  """
  assert len(q) == 4
  qx, qy, qz, qw = q

  qx2 = qx**2
  qy2 = qy**2
  qz2 = qz**2
  qw2 = qw**2

  # Homogeneous form
  C11 = qw2 + qx2 - qy2 - qz2
  C12 = 2.0 * (qx * qy - qw * qz)
  C13 = 2.0 * (qx * qz + qw * qy)

  C21 = 2.0 * (qx * qy + qw * qz)
  C22 = qw2 - qx2 + qy2 - qz2
  C23 = 2.0 * (qy * qz - qw * qx)

  C31 = 2.0 * (qx * qz - qw * qy)
  C32 = 2.0 * (qy * qz + qw * qx)
  C33 = qw2 - qx2 - qy2 + qz2

  return np.array([[C11, C12, C13], [C21, C22, C23], [C31, C32, C33]])

def quat_norm(q):
  """ Returns norm of a quaternion """
  qx, qy, qz, qw = q
  return sqrt(qx**2 + qy**2 + qz**2 + qw**2)


def quat_normalize(q):
  """ Normalize quaternion """
  n = quat_norm(q)
  qx, qy, qz, qw = q
  return np.array([qx / n, qy / n, qz / n, qw / n])

def tf(rot, trans):
  """
  Form 4x4 homogeneous transformation matrix from rotation `rot` and
  translation `trans`. Where the rotation component `rot` can be a rotation
  matrix or a quaternion.
  """
  C = None
  if rot.shape == (4,) or rot.shape == (4, 1):
    C = quat2rot(rot)
  elif rot.shape == (3, 3):
    C = rot
  else:
    raise RuntimeError("Invalid rotation!")

  T = np.eye(4, 4)
  T[0:3, 0:3] = C
  T[0:3, 3] = trans
  return T

def tf_rot(T):
  """ Return rotation matrix from 4x4 homogeneous transform """
  assert T.shape == (4, 4)
  return T[0:3, 0:3]

def tf_trans(T):
  """ Return translation vector from 4x4 homogeneous transform """
  assert T.shape == (4, 4)
  return T[0:3, 3]

def get_api_key():
    # openai api key in .openai_key file
    with open('.openai_key', 'r') as f:
        return f.readline().strip()
    
#### Server utils

def send_data(client_socket, data):
    serialized_data = pickle.dumps(data)
    length = len(serialized_data)
    client_socket.sendall(length.to_bytes(4, 'big'))
    client_socket.sendall(serialized_data)

def recv_data(client_socket):
    data_length = int.from_bytes(client_socket.recv(4), 'big')
    data = bytearray()
    while len(data) < data_length:
        packet = client_socket.recv(data_length - len(data))
        if not packet:
            return None
        data.extend(packet)
    return pickle.loads(data)

def print_openai_messages(messages):
    for message in messages:
        if message['role'] == 'assistant':
            print(f"Assistant:\n{message['content']}")
        elif message['role'] == 'user':
            print(f"User:\n{message['content']}")
        else:
            print(f"System:\n{message['content']}")

def log_completion(name, message, path):
    with open(path, 'a') as f:
        f.write(f"{name}:\n{message}\n\n")

def save_numpy_image(path, image):
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def load_numpy_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)