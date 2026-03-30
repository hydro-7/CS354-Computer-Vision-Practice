import tensorflow as tf
from tensorflow.keras import layers, models

def build_denoising_autoencoder(image_shape=(256, 256, 1)):
    """
    Builds a Convolutional Autoencoder.
    To use this, you would compile it and train it with pairs of images:
    model.fit(x=noisy_images, y=clean_images, epochs=10)
    """
    # --- ENCODER (Compressing the image) ---
    inputs = layers.Input(shape=image_shape)
    
    # 32 filters, 3x3 kernel, ReLU activation. Padding 'same' keeps dimensions.
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    # MaxPooling reduces the spatial dimensions by half (e.g., 256 -> 128)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    # The bottleneck (e.g., 128 -> 64)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # --- DECODER (Reconstructing the image without noise) ---
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    # UpSampling doubles the spatial dimensions (e.g., 64 -> 128)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # UpSampling (e.g., 128 -> 256) back to original size
    x = layers.UpSampling2D((2, 2))(x)
    
    # Final layer uses Sigmoid to output pixel values between 0 and 1
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Compile the model
    autoencoder = models.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    return autoencoder
