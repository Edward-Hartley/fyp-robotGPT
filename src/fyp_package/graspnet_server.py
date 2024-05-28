import socket
import pickle
import tensorflow.compat.v1 as tf
from multiprocessing import Process, Queue, set_start_method
from threading import Thread
import numpy as np
import signal
import os
import sys
import cv2
import gc

from fyp_package.experiments.contact_graspnet.contact_graspnet import config_utils, contact_grasp_estimator
from fyp_package import config, utils

# Ensure the 'spawn' start method is used for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR))

checkpoint_dir = config.contact_graspnet_checkpoint_dir

def load_model():
    global_config = config_utils.load_config(checkpoint_dir, batch_size=1, arg_configs=[])
    grasp_estimator = contact_grasp_estimator.GraspEstimator(global_config)
    grasp_estimator.build_network()

    saver = tf.train.Saver(save_relative_paths=True)
    tf_session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    grasp_estimator.load_weights(tf_session, saver, checkpoint_dir, mode='test')

    return grasp_estimator, tf_session

def predict_worker(depth_path, rgb_path, mask_path, result_queue):
    grasp_estimator, tf_session = load_model()
    depth = np.load(depth_path)
    rgb = None
    if rgb_path is not None:
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    segmap = np.load(mask_path)

    pc_full, pc_segments, _pc_colors = grasp_estimator.extract_point_clouds(
        depth, config.intrinsics, segmap=segmap, rgb=rgb
    )

    pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
        tf_session, pc_full, pc_segments=pc_segments, local_regions=True, filter_grasps=True
    )

    if not scores[True].any():
        result_queue.put(None)
    else:
        best_grasp_idx = np.argmax(scores[True])
        best_score = scores[True][best_grasp_idx]
        best_grasp_cam = pred_grasps_cam[True][best_grasp_idx]
        best_contact_pt = contact_pts[True][best_grasp_idx]

        result_queue.put((best_grasp_cam, best_score, best_contact_pt))

    # Free up GPU memory
    tf_session.close()
    tf.reset_default_graph()
    del grasp_estimator
    del tf_session
    gc.collect()


class GraspnetServer:
    def __init__(self, host='', port=config.model_server_ports['graspnet']):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        print("Server listening on port", port)

    def handle_client(self, client_socket):
        while True:
            data = utils.recv_data(client_socket)
            if not data:
                break
            if data == "close":
                continue
            depth_path, rgb_path, mask_path = data
            result_queue = Queue()
            process = Process(target=predict_worker, args=(depth_path, rgb_path, mask_path, result_queue))
            process.start()
            process.join()

            response = result_queue.get()
            utils.send_data(client_socket, response)
        client_socket.close()

    def run(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            print('Connected by', addr)
            client_thread = Thread(target=self.handle_client, args=(client_socket,))
            client_thread.start()

if __name__ == '__main__':
    gs = GraspnetServer()
    signal.signal(signal.SIGINT, lambda signal, frame: gs.shutdown())
    gs.run()



# import socket
# import pickle
# import tensorflow.compat.v1 as tf
# from threading import Thread
# import numpy as np
# import signal
# import os
# import sys
# import cv2
# import gc

# from fyp_package.experiments.contact_graspnet.contact_graspnet import config_utils, contact_grasp_estimator
# from fyp_package import config

# tf.disable_eager_execution()
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.join(BASE_DIR))

# checkpoint_dir = config.contact_graspnet_checkpoint_dir

# def load_model():
#     global_config = config_utils.load_config(checkpoint_dir, batch_size=1, arg_configs=[])
#     grasp_estimator = contact_grasp_estimator.GraspEstimator(global_config)
#     grasp_estimator.build_network()

#     saver = tf.train.Saver(save_relative_paths=True)
#     tf_session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
#     grasp_estimator.load_weights(tf_session, saver, checkpoint_dir, mode='test')

#     return grasp_estimator, tf_session

# def send_data(client_socket, data):
#     serialized_data = pickle.dumps(data)
#     length = len(serialized_data)
#     client_socket.sendall(length.to_bytes(4, 'big'))
#     client_socket.sendall(serialized_data)

# class GraspnetServer:
#     def __init__(self, host='', port=config.model_server_ports['graspnet']):
#         self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.server_socket.bind((host, port))
#         self.server_socket.listen(5)
#         # do this if your GPU has enough to keep it loaded
#         # self.grasp_estimator, self.tf_session = load_model() 
#         print("Server listening on port", port)

#     def handle_client(self, client_socket):
#         while True:
#             data = client_socket.recv(1024)
#             if not data:
#                 break
#             depth_path, rgb_path, mask_path = pickle.loads(data)
#             response = self.predict(depth_path, rgb_path, mask_path)
#             send_data(client_socket, response)
#         client_socket.close()

#     def predict(self, depth_path, rgb_path, mask_path):
#         grasp_estimator, tf_session = load_model() 
#         depth = np.load(depth_path)
#         rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
#         segmap = np.load(mask_path)[0]

#         pc_full, pc_segments, pc_colors = grasp_estimator.extract_point_clouds(
#             depth, config.intrinsics, segmap=segmap, rgb=rgb
#         )

#         pred_grasps_cam, scores, contact_pts, _ = grasp_estimator.predict_scene_grasps(
#             tf_session, pc_full, pc_segments=pc_segments, local_regions=True, filter_grasps=True
#         )

#         best_grasp_idx = np.argmax(scores[True])
#         best_score = scores[True][best_grasp_idx]
#         best_grasp_cam = pred_grasps_cam[True][best_grasp_idx]
#         best_contact_pt = contact_pts[True][best_grasp_idx]

#         # free up GPU memory
#         tf_session.close()
#         tf.reset_default_graph()
#         del grasp_estimator
#         del tf_session
#         gc.collect()

#         return best_grasp_cam, best_score, best_contact_pt

#     def run(self):
#         while True:
#             client_socket, addr = self.server_socket.accept()
#             print('Connected by', addr)
#             client_thread = Thread(target=self.handle_client, args=(client_socket,))
#             client_thread.start()

# if __name__ == '__main__':
#     gs = GraspnetServer()
#     signal.signal(signal.SIGINT, lambda signal, frame: gs.shutdown())
#     gs.run()
