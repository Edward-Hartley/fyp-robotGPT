import socket
from PIL import Image
import torch
from multiprocessing import Process, Queue, set_start_method
from threading import Thread
import signal
from lang_sam import LangSAM
from fyp_package import config, utils

# Ensure the 'spawn' start method is used for multiprocessing
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def predict_worker(client_socket, first_data):
    data = None
    model = LangSAM(sam_type="vit_b")
    while True:
        if data is None:
            data = first_data
        else:
            data = utils.recv_data(client_socket)
            
        if not data:
            break
        if data == "close":
            break

        flag, (file_or_array, prompt) = data

        if flag == 'path':
            image = Image.open(file_or_array)
        elif flag == 'array':
            image = Image.fromarray(file_or_array)
        else:
            raise ValueError("Unsupported flag type")
        masks, boxes, phrases, logits = model.predict(image, prompt)
        
        # Turn data into numpy arrays
        masks = [mask.detach().cpu().numpy() for mask in masks]
        boxes = [box.detach().cpu().numpy() for box in boxes]
        phrases = [phrase for phrase in phrases]

        utils.send_data(client_socket, (masks, boxes, phrases))

class LangsamServer:
    def __init__(self, host='', port=config.model_server_ports["langsam"]):
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
            process = Process(target=predict_worker, args=(client_socket, data))
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
