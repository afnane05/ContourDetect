import cv2
import numpy as np

def apply_sobel(image):
    """
    Apply Sobel edge detection filter
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel in X and Y directions
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate magnitude
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Normalize to 0-255
    sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))
    
    return sobel_magnitude