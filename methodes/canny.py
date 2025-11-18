import cv2
import numpy as np

def apply_canny(image, threshold1=100, threshold2=200):
    """
    Apply Canny edge detection filter
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1, threshold2)
    
    return edges