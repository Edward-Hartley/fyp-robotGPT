import socket
import pickle
from PIL import Image
import torch
from threading import Thread
import numpy as np

from lang_sam import LangSAM

def load_model():
    # Uses 450 MB of GPU memory to load the model, consider loading on demand
    model = LangSAM(sam_type="vit_b")
    return model

def send_data(client_socket, data):
    serialized_data = pickle.dumps(data)
    length = len(serialized_data)
    client_socket.sendall(length.to_bytes(4, 'big'))
    client_socket.sendall(serialized_data)

class ModelServer:
    def __init__(self, host='', port=9998):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(5)
        self.model = load_model()
        print("Server listening on port", port)

    def handle_client(self, client_socket):
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            file_path, prompt = pickle.loads(data)
            response = self.predict(file_path, prompt)

            response_length = len(pickle.dumps(response))
            send_data(client_socket, response)
        client_socket.close()

    def predict(self, file_path, prompt):
        image = Image.open(file_path)
        masks, boxes, phrases, logits = self.model.predict(image, prompt)
        # turn data into numpy arrays
        masks = [mask.detach().cpu().numpy() for mask in masks]
        boxes = [box.detach().cpu().numpy() for box in boxes]
        phrases = [phrase for phrase in phrases]
        return masks, boxes, phrases

    def run(self):
        while True:
            client_socket, addr = self.server_socket.accept()
            print('Connected by', addr)
            client_thread = Thread(target=self.handle_client, args=(client_socket,))
            client_thread.start()

if __name__ == '__main__':
    ms = ModelServer()
    ms.run()
