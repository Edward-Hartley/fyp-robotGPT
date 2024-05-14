
import pyrealsense2 as rs
import numpy as np
import cv2
import fyp_package.config as config

class RealsenseCamera:
    def __init__(self):

        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()

        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        self.device_product_line = str(device.get_info(rs.camera_info.product_line))

        config.enable_stream(rs.stream.depth, rs.format.z16, 30)

        config.enable_stream(rs.stream.color, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

        profile = self.pipeline.get_active_profile()
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        self.intrinsics = color_profile.get_intrinsics()
        self.K = np.array([[self.intrinsics.fx, 0, self.intrinsics.ppx],
                           [0, self.intrinsics.fy, self.intrinsics.ppy],
                           [0, 0, 1]])
        self.width = self.intrinsics.width
        self.height = self.intrinsics.height

        self.fovh_rad = 2 * np.arctan(self.width / (2 * self.intrinsics.fx))
        self.fovh_deg = np.degrees(self.fovh_rad)
        self.fovv_rad = 2 * np.arctan(self.height / (2 * self.intrinsics.fy))
        self.fovv_deg = np.degrees(self.fovv_rad)

    '''
    Get a frame from the camera
    save: bool, save the frame to the path specified in save_paths
    save_path_rgb: str, path to save the color image, saved with cv2.imwrite
    save_path_depth: str, path to save the depth image, saved as np.ndarray
    return: tuple of np.ndarray, color image and depth image
    '''
    def get_frame(self, save=False, save_path_rgb=config.latest_rgb_image_path, save_path_depth=config.latest_depth_image_path):
        # Streaming loop
        while True:
            # Get frameset of color and depth
            frames = self.pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # # Render images:
            # #   depth align to color on left
            # #   depth on right
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # images = np.hstack((color_image, depth_colormap))

            # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            # cv2.imshow('Align Example', images)
            # cv2.waitKey(1)

            depth_image = depth_image * self.depth_scale
            depth_image = np.clip(depth_image, config.zrange[0], config.zrange[1]) 

            break

        if save:
            cv2.imwrite(save_path_rgb, color_image)
            np.save(save_path_depth, depth_image)
        
        return color_image, depth_image
    
    def get_intrinsics(self):
        return self.K
    
    def get_resolution(self):
        return self.width, self.height
    
    def get_fov(self):
        return self.fovh_deg, self.fovv_deg
    
    def __del__(self):
        self.pipeline.stop()


if __name__ == "__main__":
    camera = RealsenseCamera()
    color_image, depth_image = camera.get_frame(save=True)
    print(color_image.shape, depth_image.shape)
    print(camera.get_intrinsics())
    print(camera.get_resolution())
    print(camera.get_fov())
    np.save("data/latest_camera_specs.npy",
            {
                "camera_image_size": camera.get_resolution(),
                "intrinsics": camera.get_intrinsics(),
                "fov": camera.get_fov()
            })
    del camera