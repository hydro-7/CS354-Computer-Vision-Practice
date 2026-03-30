def manual_convolve(image, kernel):
    # (Standard convolution logic from your previous studies goes here)
    # Assuming this function exists and returns a convolved image
    pass 

def unsharp_masking_pipeline(image, k_factor=1.0):
    # QUESTION: Combine manual convolution (blurring) with image arithmetic for Unsharp Masking.
    
    # 1. Create a 3x3 Arithmetic Mean kernel (all 1s divided by 9)
    mean_kernel = np.ones((3, 3), np.float32) / 9.0
    
    # 2. Blur the original image using manual convolution
    blurred_img = manual_convolve(image, mean_kernel)
    
    # Convert images to float32 to safely handle negative numbers during subtraction
    img_float = image.astype(np.float32)
    blur_float = blurred_img.astype(np.float32)
    
    # 3. Create the "Mask" (Original - Blurred)
    # This isolates the edges and high-frequency details
    mask = img_float - blur_float
    
    # 4. Add the scaled mask back to the original image (Original + k * Mask)
    # k_factor determines the strength of the sharpening
    sharpened_img = img_float + (k_factor * mask)
    
    return np.clip(sharpened_img, 0, 255).astype(np.uint8)
