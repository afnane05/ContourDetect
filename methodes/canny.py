import cv2
import numpy as np

def apply_canny(image):
    """
    Canny edge detection using OpenCV, with automatic thresholds.
    No threshold UI needed.
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Use automatic thresholds based on median
    median_val = np.median(gray)
    lower = int(max(0, 0.66 * median_val))
    upper = int(min(255, 1.33 * median_val))
    
    # Apply OpenCV Canny
    edges = cv2.Canny(gray, lower, upper)
    
    return edges
