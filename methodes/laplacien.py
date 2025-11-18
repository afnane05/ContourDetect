import cv2
import numpy as np

def apply_laplacian(image):
    """
    Manual implementation of Laplacian edge detection
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Normalize to 0-1
    gray = gray.astype(np.float32) / 255.0
    
    # Define Laplacian kernel (4-neighborhood)
    laplacian_kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)
    
    # Alternative: 8-neighborhood Laplacian
    # laplacian_kernel = np.array([
    #     [1,  1, 1],
    #     [1, -8, 1],
    #     [1,  1, 1]
    # ], dtype=np.float32)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Initialize output
    laplacian = np.zeros_like(gray)
    
    # Apply convolution manually
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Extract 3x3 region
            region = gray[i-1:i+2, j-1:j+2]
            
            # Compute Laplacian
            lap_value = np.sum(region * laplacian_kernel)
            laplacian[i, j] = lap_value
    
    # Take absolute value and normalize
    laplacian_abs = np.abs(laplacian)
    laplacian_normalized = np.uint8(255 * laplacian_abs / np.max(laplacian_abs))
    
    return laplacian_normalized