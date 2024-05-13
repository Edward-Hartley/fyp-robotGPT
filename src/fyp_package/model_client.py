import socket
import pickle
import matplotlib.pyplot as plt
from PIL import Image
# from torchvision.transforms import functional as transforms
# from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

def recv_data(client_socket):
    data_length = int.from_bytes(client_socket.recv(4), 'big')
    data = bytearray()
    while len(data) < data_length:
        packet = client_socket.recv(data_length - len(data))
        if not packet:
            return None
        data.extend(packet)
    return pickle.loads(data)

class ModelClient:
    def __init__(self, host='localhost', port=9998):
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.host, self.port))

    def langsam_predict(self, image_path, prompt):
        data = pickle.dumps((image_path, prompt))
        self.client_socket.sendall(data)
        return recv_data(self.client_socket)

    # def visualize_results(self, image, masks, boxes, phrases):
    #     _, ax = plt.subplots(1, 1 + len(masks), figsize=(5 + (5 * len(masks)), 5))
    #     [a.axis("off") for a in ax.flatten()]
    #     ax[0].imshow(image)
        
    #     for i, (mask, box, phrase) in enumerate(zip(masks, boxes, phrases)):
    #         to_tensor = transforms.PILToTensor()
    #         image_tensor = to_tensor(image)
    #         box = box.unsqueeze(dim=0)
    #         image_tensor = draw_bounding_boxes(image_tensor, box, colors=["red"], width=3)
    #         image_tensor = draw_segmentation_masks(image_tensor, mask, alpha=0.5, colors=["cyan"])
    #         to_pil_image = transforms.ToPILImage()
    #         image_pil = to_pil_image(image_tensor)

    #         ax[1 + i].imshow(image_pil)
    #         ax[1 + i].text(box[0][0], box[0][1] - 15, phrase, color="red", bbox={"facecolor":"white", "edgecolor":"red", "boxstyle":"square"})

    #     plt.show()

    def close(self):
        self.client_socket.close()

if __name__ == "__main__":
    client = ModelClient()
    masks, boxes, phrases = client.langsam_predict('./assets/pybullet_tabletop_2.png', 'green_bowl')
    print(masks, boxes, phrases)
    client.close()
