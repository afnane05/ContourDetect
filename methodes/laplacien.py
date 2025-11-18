import cv2
import numpy as np

def apply_laplacian(image):
    """
    Apply Laplacian edge detection filter
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur first to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)
    
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Convert to absolute values and normalize
    laplacian_abs = np.absolute(laplacian)
    laplacian_normalized = np.uint8(255 * laplacian_abs / np.max(laplacian_abs))
    
    return laplacian_normalized