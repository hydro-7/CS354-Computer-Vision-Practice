def gamma_and_edge_pipeline(image, gamma_value):
    # QUESTION: Brighten a dark image using Gamma Correction, then find edges via Convolution.
    
    # --- STEP 1: GAMMA TRANSFORMATION ---
    # Normalize pixel values to the range [0.0, 1.0]
    normalized_img = image / 255.0
    
    # Apply the gamma power law. (Gamma < 1 will brighten the image)
    gamma_corrected = np.power(normalized_img, gamma_value)
    
    # Scale back up to [0, 255] and convert to 8-bit integer
    brightened_img = np.uint8(gamma_corrected * 255)
    
    # --- STEP 2: EDGE DETECTION CONVOLUTION ---
    # Define a horizontal Sobel kernel (detects horizontal lines/edges)
    sobel_horizontal = np.array([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]])
    
    # Apply your manual convolution to the newly brightened image
    # (Assuming manual_convolve is already defined)
    edge_detected_img = manual_convolve(brightened_img, sobel_horizontal)
    
    return brightened_img, edge_detected_img
