import cv2
import numpy as np
from PIL import Image
import os

def load_image(file_path):
    """
    Load image using OpenCV
    """
    image = cv2.imread(file_path)
    if image is None:
        raise ValueError(f"Could not load image from {file_path}")
    return image

def save_image(pil_image, file_path):
    """
    Save PIL Image to file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert to RGB if necessary and save
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    pil_image.save(file_path)

def get_available_images(directory="images"):
    """
    Get list of available images in images directory
    """
    if not os.path.exists(directory):
        return []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            images.append(os.path.join(directory, file))
    
    return images