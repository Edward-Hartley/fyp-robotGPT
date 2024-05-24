import socket
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from fyp_package import config, utils
import numpy as np
# from torchvision.transforms import functional as transforms
# from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

class ModelClient:
    def __init__(self, host='localhost'):
        self.servers = {}
        for model_name, port in config.model_server_ports.items():
            self.servers[model_name] = self.connect_to_server(host, port)

        self.active_model = None

    def connect_to_server(self, host, port):
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            client_socket.connect((host, port))
            print(f"Connected to {port}")
            return client_socket
        except ConnectionRefusedError:
            print(f"Could not connect to {port}")
            return None

    def send_request(self, model_name, data):
        if self.active_model is not None and self.active_model != model_name:
            utils.send_data(self.servers[self.active_model], "close")
            
        if model_name not in self.servers or self.servers[model_name] is None:
            raise ConnectionError(f"No connection to {model_name} server.")

        self.active_model = model_name
        client_socket = self.servers[model_name]
        utils.send_data(client_socket, data)
        return utils.recv_data(client_socket)

    def langsam_predict(self, image_path_or_array, prompt, save=False, save_path=config.latest_segmentation_masks_path):
        data = (image_path_or_array, prompt)
        if isinstance(data[0], str):  # file path
            flag = 'path'
        elif isinstance(data[0], np.ndarray):  # numpy array
            flag = 'array'
        else:
            raise ValueError("Unsupported data type")
        
        # Prepend the flag to the data
        flagged_data = (flag, data)

        response = self.send_request('langsam', flagged_data)
        masks, boxes, phrases = response
        if save:
            np.save(save_path, masks)
        return masks, boxes, phrases
    
    def contact_graspnet_predict(self, depth_path, rgb_path, mask_path, save=False, save_path=config.latest_grasp_detection_path):
        data = (depth_path, rgb_path, mask_path)
        response = self.send_request('graspnet', data)
        best_grasp_cam, best_score, best_contact_pt = response
        if save:
            np.savez(save_path, pred_grasp_cam=best_grasp_cam, score=best_score, contact_pt=best_contact_pt)
        return best_grasp_cam, best_score, best_contact_pt

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
        for model_name, client_socket in self.servers.items():
            if client_socket:
                client_socket.close()

if __name__ == "__main__":
    client = ModelClient()
    # Example usage
    try:
        masks, boxes, phrases = client.langsam_predict(config.latest_rgb_image_path, "espresso cup", save=True)
        print(masks, boxes, phrases)
        # plt.imshow(np.load(config.latest_segmentation_masks_path)[0])
        # plt.show()

        best_grasp_cam, best_score, best_contact_pt = client.contact_graspnet_predict(
            config.latest_depth_image_path, config.latest_rgb_image_path, config.latest_segmentation_masks_path, save=True
        )
        results = np.load(config.latest_grasp_detection_path)
        print(results['pred_grasp_cam'], results['score'], results['contact_pt'])

    except ConnectionError as e:
        print(e)
    finally:
        client.close()
