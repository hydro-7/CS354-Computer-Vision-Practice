import random

def add_noises_and_convolve(image):
    # We will work with a grayscale image for simplicity
    row, col = image.shape
    
    # --- NOISE 1: GAUSSIAN NOISE (Statistical static) ---
    # Generate random values from a normal distribution (mean=0, standard deviation=25)
    gauss_noise = np.random.normal(0, 25, (row, col))
    # Add the noise to the original image
    img_gaussian = image + gauss_noise
    # Clip values to keep them between 0 and 255
    img_gaussian = np.clip(img_gaussian, 0, 255).astype(np.uint8)
    
    # --- NOISE 2: SALT & PEPPER NOISE (Random black and white pixels) ---
    # Copy the image so we don't modify the original
    img_sp = np.copy(image)
    # Calculate how many pixels should be altered (e.g., 2% of total pixels)
    num_salt = np.ceil(0.02 * image.size)
    # Randomly scatter white pixels (Salt)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    img_sp[tuple(coords)] = 255
    # Randomly scatter black pixels (Pepper)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    img_sp[tuple(coords)] = 0
    
    # --- NOISE 3: SPECKLE NOISE (Multiplicative noise) ---
    # Generate random noise 
    speckle_noise = np.random.randn(row, col)
    # Multiply the image by the noise, then add it back to the image
    img_speckle = image + image * speckle_noise * 0.1 # 0.1 scales the intensity
    img_speckle = np.clip(img_speckle, 0, 255).astype(np.uint8)

    # --- APPLY CONVOLUTION TO DENOISE ---
    # Define a 3x3 Average (Blurring) Kernel. It averages nearby pixels to smooth noise.
    # We divide by 9 so the overall brightness stays the same (1+1+1+1+1+1+1+1+1 = 9)
    blur_kernel = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]) / 9.0
    
    # Apply our manual convolution function to one of the noisy images
    denoised_gaussian = manual_convolve_grayscale(img_gaussian, blur_kernel)
    
    return img_gaussian, img_sp, img_speckle, denoised_gaussian
