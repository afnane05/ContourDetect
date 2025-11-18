import cv2
import numpy as np

def apply_prewitt(image):
    """
    Apply Prewitt edge detection filter
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Prewitt kernels
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    # Apply kernels
    prewitt_x = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
    prewitt_y = cv2.filter2D(gray, cv2.CV_64F, kernel_y)
    
    # Calculate magnitude
    prewitt_magnitude = np.sqrt(prewitt_x**2 + prewitt_y**2)
    
    # Normalize to 0-255
    prewitt_magnitude = np.uint8(255 * prewitt_magnitude / np.max(prewitt_magnitude))
    
    return prewitt_magnitude