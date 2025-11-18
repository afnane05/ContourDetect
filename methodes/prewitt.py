import cv2
import numpy as np

def apply_prewitt(image):
    """
    Manual implementation of Prewitt edge detection
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Normalize to 0-1
    gray = gray.astype(np.float32) / 255.0
    
    # Define Prewitt kernels
    prewitt_x = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    prewitt_y = np.array([
        [-1, -1, -1],
        [ 0,  0,  0],
        [ 1,  1,  1]
    ], dtype=np.float32)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Initialize gradient arrays
    gradient_x = np.zeros_like(gray)
    gradient_y = np.zeros_like(gray)
    
    # Apply convolution manually
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Extract 3x3 region
            region = gray[i-1:i+2, j-1:j+2]
            
            # Compute gradients
            gx = np.sum(region * prewitt_x)
            gy = np.sum(region * prewitt_y)
            
            gradient_x[i, j] = gx
            gradient_y[i, j] = gy
    
    # Calculate gradient magnitude
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize to 0-255 for display
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))
    
    return gradient_magnitude