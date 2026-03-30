import numpy as np
import cv2

def conditional_laplacian_convolution(image):
    # QUESTION: Apply Laplacian sharpening ONLY to bright pixels (>100).
    
    # Define a standard 3x3 Laplacian kernel for sharpening
    kernel = np.array([[ 0, -1,  0],
                       [-1,  5, -1],
                       [ 0, -1,  0]])
    
    image_h, image_w = image.shape
    kernel_h, kernel_w = kernel.shape
    pad_h, pad_w = kernel_h // 2, kernel_w // 2
    
    # Pad the image to handle borders
    padded_img = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output_img = np.zeros_like(image, dtype=np.float32)
    
    for y in range(image_h):
        for x in range(image_w):
            # Extract the 3x3 region
            region = padded_img[y:y+kernel_h, x:x+kernel_w]
            
            # --- THE MODIFICATION ---
            # Check the intensity of the center pixel of the ORIGINAL image
            if image[y, x] > 100:
                # If bright enough, apply the convolution math (sum of element-wise multiplication)
                output_img[y, x] = np.sum(region * kernel)
            else:
                # Otherwise, keep the original pixel value
                output_img[y, x] = image[y, x]
                
    # Clip values to valid image range
    return np.clip(output_img, 0, 255).astype(np.uint8)
