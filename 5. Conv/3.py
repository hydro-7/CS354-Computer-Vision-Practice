import tensorflow as tf
from tensorflow.keras.layers import Conv2D

def keras_convolution(image_rgb):
    # Keras expects images to be in a "Batch" format: (Batch_Size, Height, Width, Channels)
    # Right now, our image is (Height, Width, Channels). 
    # We use np.expand_dims to add the Batch dimension at index 0.
    input_tensor = np.expand_dims(image_rgb, axis=0)
    
    # Convert the numpy array to a float32 tensor (which Keras requires)
    input_tensor = tf.cast(input_tensor, tf.float32)
    
    # Create a 2D Convolutional Layer
    # filters=1: We want 1 output channel (a single feature map)
    # kernel_size=3: Our filter will be 3x3
    # padding='same': Keras handles the zero-padding automatically
    # use_bias=False: We disable bias to keep it identical to our manual math
    conv_layer = Conv2D(filters=1, kernel_size=3, padding='same', use_bias=False)
    
    # We must pass the tensor through the layer once to initialize its shape/weights
    _ = conv_layer(input_tensor)
    
    # --- SETTING CUSTOM WEIGHTS ---
    # We define the same edge detection kernel from earlier
    custom_kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]], dtype=np.float32)
    
    # Keras expects weights in the shape: (kernel_height, kernel_width, input_channels, output_channels)
    # For a 3x3 kernel, 3 input channels (RGB), and 1 output channel, shape is (3, 3, 3, 1)
    # We stack our kernel 3 times (once for each RGB channel) and reshape it
    keras_weights = np.stack([custom_kernel]*3, axis=-1)
    keras_weights = np.expand_dims(keras_weights, axis=-1)
    
    # Apply our custom weights to the Keras layer
    conv_layer.set_weights([keras_weights])
    
    # Finally, apply the convolution operation
    keras_output = conv_layer(input_tensor)
    
    # Extract the output from the batch by taking the first item [0]
    # Convert it back to a numpy array for viewing
    result = keras_output[0].numpy()
    
    return result
