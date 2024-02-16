from transformers import pipeline
import numpy as np
from PIL import Image
from image_utils import normalize_depth_map, overlay_depth_map
import torch.nn.functional as F

image_path = "../assets/tabletop.jpg"

depth_estimator = pipeline(task="depth-estimation")
output = depth_estimator(image_path)
# This is a tensor with the values being the depth expressed in meters for each pixel
print(output["predicted_depth"].shape)
print(output["predicted_depth"])

original_image = Image.open(image_path)

# Perform depth estimation (this is a placeholder for your depth estimation model's actual call)
# depth_map = predict_depth(np.array(original_image))
# For demonstration, let's assume we have a depth map as a numpy array
# (In practice, replace the following line with your actual depth estimation model's output)
depth_map = output["predicted_depth"].unsqueeze(0)
depth_map = F.interpolate(depth_map, size=original_image.size, mode='bilinear', align_corners=False)
depth_map = depth_map.numpy(force=True)
depth_map_reshaped = depth_map.reshape(original_image.size)

# Normalize the depth map
normalized_depth_map = normalize_depth_map(depth_map_reshaped)

# Overlay the depth map on the original image
overlayed_image = overlay_depth_map(original_image, normalized_depth_map)

# Save the modified image
overlayed_image.convert("RGB").save("../assets/modified_image_with_depth_overlay.jpg")