import numpy as np
import cv2

# ==========================================
# 1. NOISE GENERATION FUNCTIONS
# ==========================================

def add_gaussian_noise(image, mean=0, sigma=25):
    """Adds normally distributed statistical noise."""
    noise = np.random.normal(mean, sigma, image.shape)
    noisy_img = image + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)

def add_salt_and_pepper_noise(image, prob=0.05):
    """Adds random black and white pixels."""
    noisy_img = np.copy(image)
    # Generate a matrix of random probabilities
    rnd = np.random.rand(*noisy_img.shape)
    # Set to black (pepper) where probability is very low
    noisy_img[rnd < prob/2] = 0
    # Set to white (salt) where probability is very high
    noisy_img[rnd > 1 - prob/2] = 255
    return noisy_img

def add_bright_impulse_noise(image, prob=0.03):
    """Adds only 'Salt' (bright specks)."""
    noisy_img = np.copy(image)
    rnd = np.random.rand(*noisy_img.shape)
    # Only add maximum intensity white pixels
    noisy_img[rnd > 1 - prob] = 255
    return noisy_img

def add_dark_impulse_noise(image, prob=0.03):
    """Adds only 'Pepper' (dark specks)."""
    noisy_img = np.copy(image)
    rnd = np.random.rand(*noisy_img.shape)
    # Only add minimum intensity black pixels
    noisy_img[rnd < prob] = 0
    return noisy_img

def add_mixed_noise(image):
    """Combines Gaussian background noise with Salt & Pepper."""
    # First add Gaussian, then add S&P on top of it
    img_gauss = add_gaussian_noise(image, mean=0, sigma=15)
    img_mixed = add_salt_and_pepper_noise(img_gauss, prob=0.04)
    return img_mixed
