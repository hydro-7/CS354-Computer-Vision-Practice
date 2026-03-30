from scipy.ndimage import minimum_filter, maximum_filter
import scipy.ndimage as ndimage

# ==========================================
# 1. GAUSSIAN NOISE -> GAUSSIAN / MEAN FILTER
# ==========================================
# Context Rule: Gaussian noise is best handled by Mean filter or Gaussian filter[cite: 77].
def denoise_gaussian(noisy_image):
    """Applies a Gaussian Blur to smooth out normal noise."""
    # Gaussian is smoother than mean and preserves structures slightly better[cite: 34].
    # kernel size 5x5, sigma calculated automatically
    return cv2.GaussianBlur(noisy_image, (5, 5), 0)


# ==========================================
# 2. SALT & PEPPER NOISE -> MEDIAN FILTER
# ==========================================
# Context Rule: Salt-and-pepper noise is best removed with a Median filter[cite: 78].
def denoise_salt_and_pepper(noisy_image):
    """Applies a Median filter which is strongest against impulses."""
    # The median filter replaces isolated spikes very well[cite: 38].
    # Using a 3x3 window size.
    return cv2.medianBlur(noisy_image, 3)


# ==========================================
# 3. BRIGHT IMPULSE (SALT) -> MIN FILTER
# ==========================================
# Context Rule: Bright impulse noise is best handled by Min filter [cite: 41] 
# or Contraharmonic with Q > 0[cite: 79]. We will use Min Filter here.
def denoise_bright_impulse(noisy_image):
    """Applies a Minimum filter to shrink bright regions."""
    # Min filter replaces each pixel with the minimum value in the neighborhood[cite: 41].
    # Useful when noise appears as bright specks[cite: 42].
    return minimum_filter(noisy_image, size=3)


# ==========================================
# 4. DARK IMPULSE (PEPPER) -> MAX FILTER
# ==========================================
# Context Rule: Dark impulse noise is best handled by Max filter [cite: 44]
# or Contraharmonic with Q < 0. We will use Max Filter here.
def denoise_dark_impulse(noisy_image):
    """Applies a Maximum filter to expand bright regions over dark specks."""
    # Max filter replaces each pixel with the maximum value in the neighborhood[cite: 44].
    # Useful when noise appears as black specks[cite: 45].
    return maximum_filter(noisy_image, size=3)


# ==========================================
# 5. MIXED NOISE -> ALPHA-TRIMMED MEAN FILTER
# ==========================================
# Context Rule: Mixed noise is best handled by an Alpha-trimmed mean filter[cite: 80].
def denoise_mixed_noise(noisy_image, window_size=3, d=2):
    """
    Applies an Alpha-Trimmed Mean Filter.
    It combines mean filter and median-like robustness[cite: 52].
    """
    pad = window_size // 2
    padded_img = np.pad(noisy_image, ((pad, pad), (pad, pad)), mode='constant')
    output_img = np.zeros_like(noisy_image)
    
    # Calculate how many pixels to trim from the top and bottom of the sorted array
    trim_amount = d // 2 
    
    for y in range(noisy_image.shape[0]):
        for x in range(noisy_image.shape[1]):
            # Extract neighborhood and flatten it to a 1D array
            region = padded_img[y:y+window_size, x:x+window_size].flatten()
            
            # Sort values in the neighborhood [cite: 51]
            region.sort()
            
            # Remove the lowest d/2 and highest d/2 values [cite: 51]
            if trim_amount > 0:
                trimmed_region = region[trim_amount : -trim_amount]
            else:
                trimmed_region = region
                
            # Average the remaining values [cite: 51]
            output_img[y, x] = np.mean(trimmed_region)
            
    return output_img.astype(np.uint8)

# --- BONUS IMPLEMENTATION: CONTRAHARMONIC MEAN ---
# Since your notes emphasize this heavily for exams [cite: 50]
def contraharmonic_mean_filter(image, Q, window_size=3):
    """
    Formula: g = sum(f^(Q+1)) / sum(f^Q) [cite: 46]
    Use Q > 0 for salt noise, Q < 0 for pepper noise[cite: 47].
    """
    pad = window_size // 2
    padded_img = np.pad(image.astype(np.float64), pad, mode='constant')
    output = np.zeros_like(image, dtype=np.float64)
    
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            region = padded_img[y:y+window_size, x:x+window_size]
            
            # Avoid division by zero issues, especially when Q is negative [cite: 48, 49]
            numerator = np.sum(np.power(region + 1e-8, Q + 1))
            denominator = np.sum(np.power(region + 1e-8, Q))
            
            output[y, x] = numerator / denominator
            
    return np.clip(output, 0, 255).astype(np.uint8)
