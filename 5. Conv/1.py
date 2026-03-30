import numpy as np
import cv2
import matplotlib.pyplot as plt

# ==========================================
# TASK 2: MANUAL CONVOLUTION ON GRAYSCALE
# ==========================================
def manual_convolve_grayscale(image, kernel):
    # Get the height and width of the input image
    image_height, image_width = image.shape
    
    # Get the height and width of the kernel (e.g., 3x3)
    kernel_height, kernel_width = kernel.shape
    
    # Calculate how much padding we need so the output image stays the same size
    # We use integer division (//) to get the center of the kernel
    pad_h = kernel_height // 2
    pad_w = kernel_width // 2
    
    # Add zero-padding around the image edges using numpy
    # 'mode=constant' defaults to adding 0s
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # Create an empty output image filled with zeros, same size as original image
    output_image = np.zeros((image_height, image_width), dtype=np.float32)
    
    # Loop over every pixel row in the original image
    for y in range(image_height):
        # Loop over every pixel column in the original image
        for x in range(image_width):
            # Extract the region of the padded image that the kernel is currently covering
            region = padded_image[y : y + kernel_height, x : x + kernel_width]
            
            # Multiply the region by the kernel element-wise, then sum all values
            # This is the core mathematical operation of convolution
            output_image[y, x] = np.sum(region * kernel)
            
    # Clip values to stay within valid pixel range [0, 255]
    output_image = np.clip(output_image, 0, 255)
    
    return output_image


# ==========================================
# TASK 1: MANUAL CONVOLUTION ON RGB IMAGE
# ==========================================
def manual_convolve_rgb(image, kernel):
    # Split the RGB image into its 3 individual color channels (Red, Green, Blue)
    # cv2 loads images in BGR format by default, so we split into B, G, R
    b, g, r = cv2.split(image)
    
    # Apply our grayscale convolution function to each channel independently
    convolved_b = manual_convolve_grayscale(b, kernel)
    convolved_g = manual_convolve_grayscale(g, kernel)
    convolved_r = manual_convolve_grayscale(r, kernel)
    
    # Merge the 3 convolved channels back together into a single color image
    output_image = cv2.merge((convolved_b, convolved_g, convolved_r))
    
    # Convert the float values back to 8-bit integers (standard image format)
    return output_image.astype(np.uint8)

# --- EXAMPLE USAGE ---
# Load an image using OpenCV
img_color = cv2.imread('your_image.jpg') # REPLACE with your image path
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Define a 3x3 Edge Detection Kernel
edge_kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])

# Run the functions
result_gray = manual_convolve_grayscale(img_gray, edge_kernel)
result_rgb = manual_convolve_rgb(img_color, edge_kernel)
