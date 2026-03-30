def manual_max_filter(image):
    # QUESTION: Change the convolution math to simply find the maximum value in the region.
    
    # A max filter usually uses a neighborhood size, let's assume 3x3
    window_size = 3
    pad = window_size // 2
    
    image_h, image_w = image.shape
    padded_img = np.pad(image, ((pad, pad), (pad, pad)), mode='constant')
    output_img = np.zeros_like(image, dtype=np.uint8)
    
    for y in range(image_h):
        for x in range(image_w):
            # Extract the 3x3 neighborhood
            region = padded_img[y:y+window_size, x:x+window_size]
            
            # --- THE MODIFICATION ---
            # Completely replace the sum(region * kernel) logic.
            # We simply use np.max() to find the brightest pixel in this 3x3 box.
            output_img[y, x] = np.max(region)
            
    return output_img
