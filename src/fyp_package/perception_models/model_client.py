import socket
import pickle
import matplotlib.pyplot as plt
from PIL import Image
from fyp_package import config, utils
import numpy as np

from fyp_package.experiments import agent_logging
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

    @agent_logging.log_model_use
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
    
    @agent_logging.log_model_use
    def graspnet_predict(self, depth_path, rgb_path, mask_path, save=False, save_path=config.latest_grasp_detection_path):
        '''
        Returns the details of the best grasp prediction.
        grasp_cam is the 4x4 transformation matrix of the gripper in camera frame.
        score is the confidence score of the grasp.
        contact_pt is the contact point of the grasp, also in the camera frame.        
        '''
        data = (depth_path, rgb_path, mask_path)
        response = self.send_request('graspnet', data)
        if response is None:
            return None
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
        from fyp_package.environment import environment
        import pybullet as pb
        import cv2
        agent_logging.setup_logging()
        env = environment.PhysicalEnvironment()
        rgb, depth = env.get_images(save=True)

        masks, boxes, phrases = client.langsam_predict(config.latest_rgb_image_path, "coke can", save=True)

        masks = np.load(config.latest_segmentation_masks_path)
        phrases = ["paper cup"] * len(masks)

        np.save(config.chosen_segmentation_mask_path, masks[0])
        plt.imshow(np.load(config.latest_segmentation_masks_path)[0])
        plt.show()


        from fyp_package import object_detection_utils
        results= object_detection_utils.get_object_cube_from_segmentation(masks, phrases, cv2.imread(config.latest_rgb_image_path), np.load(config.latest_depth_image_path), config.camera_position, config.camera_orientation_q, config.intrinsics)
        above_object = results[0]['position'] + np.array([0, 0, 0.1])
        input("Press Enter to move")
        env.move_robot(above_object)

        result = client.graspnet_predict(
            config.latest_depth_image_path, config.latest_rgb_image_path, config.chosen_segmentation_mask_path, save=True
        )

        masks, boxes, phrases = client.langsam_predict(rgb, "paper cup", save=True)


        grasp2cam_tf, _score, contact_point_cam = result

        grasp2base_tf = config.cam2base_tf @ grasp2cam_tf
        grasp_rot = utils.tf_rot(grasp2base_tf)
        # rotate grasp_orientation by 90 degrees around z-axis
        # grasp_orientation = utils.rot2quat(grasp_rot @ utils.quat2rot(utils.euler2quat(*[0, 0, np.pi/2])))
        grasp_orientation = utils.rot2quat(grasp_rot)
        grasp_z_rot = -utils.quat2euler(grasp_orientation)[2]
        # wrap rot to [-pi/2, pi/2]
        if grasp_z_rot > np.pi/2:
            grasp_z_rot -= np.pi
        elif grasp_z_rot < -np.pi/2:
            grasp_z_rot += np.pi

        contact_point = (config.cam2base_tf @ np.concatenate([contact_point_cam, [1]]))[:3]
        contact_point[2] += 0.03

        print(grasp2base_tf, contact_point)
        print(utils.quat2euler(grasp_orientation))


        print(env.get_ee_pose()[1])
        
        env.put_first_on_second(contact_point, config.robot_ready_position, grasp_z_rot)

        input("Press Enter to move")
        env.move_robot([0, 0, 0.1], relative=True)



    except ConnectionError as e:
        print(e)
    finally:
        client.close()
