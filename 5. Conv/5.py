def contraharmonic_mean_filter(image, window_size, Q):
    # QUESTION: Use the sliding window technique to apply a Contraharmonic Mean.
    # Formula: Sum(pixels^(Q+1)) / Sum(pixels^Q)
    
    image_h, image_w = image.shape
    pad = window_size // 2
    
    # Pad image, convert to float to prevent overflow during power calculations
    padded_img = np.pad(image.astype(np.float64), ((pad, pad), (pad, pad)), mode='constant')
    output_img = np.zeros_like(image, dtype=np.float64)
    
    for y in range(image_h):
        for x in range(image_w):
            region = padded_img[y:y+window_size, x:x+window_size]
            
            # --- THE MODIFICATION ---
            # Instead of multiplying by a kernel, we apply the Contraharmonic formula.
            # We add a tiny epsilon (1e-8) to the denominator to prevent division by zero.
            numerator = np.sum(np.power(region, Q + 1))
            denominator = np.sum(np.power(region, Q)) + 1e-8 
            
            output_img[y, x] = numerator / denominator
            
    return np.clip(output_img, 0, 255).astype(np.uint8)
