import torch
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the model and feature extractor
model_name = "Intel/dpt-beit-large-512"
feature_extractor = DPTFeatureExtractor.from_pretrained(model_name)
model = DPTForDepthEstimation.from_pretrained(model_name)

# Step 2: Load and preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return image, inputs

# Step 3: Perform depth estimation
def estimate_depth(image_path):
    # Preprocess the image
    original_image, inputs = preprocess_image(image_path)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

    # Normalize depth map for visualization
    depth_min = predicted_depth.min()
    depth_max = predicted_depth.max()
    normalized_depth = (predicted_depth - depth_min) / (depth_max - depth_min)

    return original_image, normalized_depth

# Step 4: Visualize the results
def visualize_results(original_image, depth_map):
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.axis("off")

    # Depth Map
    plt.subplot(1, 2, 2)
    plt.title("Depth Map")
    plt.imshow(depth_map, cmap="inferno")
    plt.axis("off")

    plt.show()

# Main Function
if __name__ == "__main__":
    # Provide the path to your input image
    input_image_path = r"C:\Users\sharvin1001\Desktop\depthai7jan\aa.png"

    # Perform depth estimation
    original_image, depth_map = estimate_depth(input_image_path)

    # Visualize the results
    visualize_results(original_image, depth_map)
