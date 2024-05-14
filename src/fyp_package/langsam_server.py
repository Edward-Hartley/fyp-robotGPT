import socket
import pickle
from PIL import Image
import torch
from multiprocessing import Process, Queue, set_start_method
from threading import Thread
import signal
import numpy as np
import os
from lang_sam import LangSAM

# Ensure the 'spawn' start method is used for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass


def send_data(client_socket, data):
    serialized_data = pickle.dumps(data)
    length = len(serialized_data)
    client_socket.sendall(length.to_bytes(4, 'big'))
    client_socket.sendall(serialized_data)

def predict_worker(file_path, prompt, client_socket):
    model = LangSAM(sam_type="vit_b")
    image = Image.open(file_path)
    masks, boxes, phrases, logits = model.predict(image, prompt)
    
    # Turn data into numpy arrays
    masks = [mask.detach().cpu().numpy() for mask in masks]
    boxes = [box.detach().cpu().numpy() for box in boxes]
    phrases = [phrase for phrase in phrases]

    send_data(client_socket, (masks, boxes, phrases))

class LangsamServer:
    def __init__(self, host='', port=9998):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        print("Server listening on port", port)

    def handle_client(self, client_socket):
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            file_path, prompt = pickle.loads(data)
            # result_queue = Queue()
            # # check capacity
            # print(result_queue.qsize())
            process = Process(target=predict_worker, args=(file_path, prompt, client_socket))
            process.start()
            process.join()

            torch.cuda.empty_cache()

        client_socket.close()

    def run(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            print('Connected by', addr)
            client_thread = Thread(target=self.handle_client, args=(client_socket,))
            client_thread.start()

if __name__ == '__main__':
    ls = LangsamServer()
    signal.signal(signal.SIGINT, lambda signal, frame: ls.shutdown())
    ls.run()
