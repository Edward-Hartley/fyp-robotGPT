from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
# Assume you have a function `predict_depth` that takes an image and returns a depth map
# from your_depth_model import predict_depth

def normalize_depth_map(depth_map):
    normalized_depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))
    return (normalized_depth_map * 255).astype(np.uint8)

def overlay_depth_map(original_image, depth_map):
    print(original_image.size, depth_map.size)
    depth_image = Image.fromarray(depth_map).convert("RGB")
    overlayed_image = Image.blend(original_image.convert("RGBA"), depth_image.convert("RGBA"), alpha=0.5)
    return overlayed_image
