import cv2
import numpy as np
import math

def apply_canny(image, threshold1=100, threshold2=200):
    """
    Manual implementation of Canny edge detection
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Step 1: Apply Gaussian blur manually
    gray = gaussian_blur(gray)
    
    # Step 2: Compute gradients using Sobel
    gradient_x, gradient_y = compute_gradients(gray)
    
    # Step 3: Compute gradient magnitude and direction
    magnitude, direction = compute_gradient_info(gradient_x, gradient_y)
    
    # Step 4: Non-maximum suppression
    suppressed = non_maximum_suppression(magnitude, direction)
    
    # Step 5: Double threshold and hysteresis
    edges = double_threshold(suppressed, threshold1, threshold2)
    
    return edges

def gaussian_blur(image, kernel_size=5, sigma=1.4):
    """Manual Gaussian blur implementation"""
    # Create Gaussian kernel
    kernel = create_gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    
    # Add padding
    padded = np.pad(image, pad, mode='reflect')
    blurred = np.zeros_like(image, dtype=np.float32)
    
    # Apply convolution
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            blurred[i, j] = np.sum(region * kernel)
    
    return blurred

def create_gaussian_kernel(size=5, sigma=1.4):
    """Create Gaussian kernel"""
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            kernel[i, j] = (1 / (2 * math.pi * sigma**2)) * math.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    return kernel / np.sum(kernel)  # Normalize

def compute_gradients(image):
    """Compute gradients using Sobel operators"""
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = np.zeros_like(image, dtype=np.float32)
    grad_y = np.zeros_like(image, dtype=np.float32)
    
    for i in range(1, image.shape[0]-1):
        for j in range(1, image.shape[1]-1):
            region = image[i-1:i+2, j-1:j+2]
            grad_x[i, j] = np.sum(region * sobel_x)
            grad_y[i, j] = np.sum(region * sobel_y)
    
    return grad_x, grad_y

def compute_gradient_info(grad_x, grad_y):
    """Compute gradient magnitude and direction"""
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x) * (180 / math.pi)
    direction = np.mod(direction, 180)  # Convert to 0-180 range
    
    return magnitude, direction

def non_maximum_suppression(magnitude, direction):
    """Apply non-maximum suppression"""
    suppressed = np.zeros_like(magnitude)
    
    for i in range(1, magnitude.shape[0]-1):
        for j in range(1, magnitude.shape[1]-1):
            angle = direction[i, j]
            mag = magnitude[i, j]
            
            # Determine neighbors based on gradient direction
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
            else:  # 112.5 <= angle < 157.5
                neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
            
            # Suppress if not maximum
            if mag >= max(neighbors):
                suppressed[i, j] = mag
    
    return suppressed

def double_threshold(image, low_threshold, high_threshold):
    """Apply double threshold and hysteresis"""
    # Normalize thresholds to image range
    low_thresh_norm = low_threshold / 255.0 * np.max(image)
    high_thresh_norm = high_threshold / 255.0 * np.max(image)
    
    # Create output image
    strong_edges = (image >= high_thresh_norm)
    weak_edges = (image >= low_thresh_norm) & (image < high_thresh_norm)
    
    # Final edges (strong edges + connected weak edges)
    edges = strong_edges.astype(np.uint8) * 255
    
    # Simple hysteresis: include weak edges adjacent to strong edges
    for i in range(1, edges.shape[0]-1):
        for j in range(1, edges.shape[1]-1):
            if weak_edges[i, j]:
                # Check 8-connected neighborhood for strong edges
                neighborhood = edges[i-1:i+2, j-1:j+2]
                if np.any(neighborhood == 255):
                    edges[i, j] = 255
    
    return edges