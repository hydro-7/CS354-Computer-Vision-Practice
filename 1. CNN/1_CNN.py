# ==============================================================================
# 0. PREREQUISITES AND IMPORTS
# ==============================================================================

# WHAT: Imports the main TensorFlow library. 
# WHY: TensorFlow is the backend engine that performs the mathematical computations.
# HOW: Use the standard abbreviation 'tf'.
import tensorflow as tf

# WHAT: Imports the 'layers' and 'models' modules from Keras.
# WHY: These provide the building blocks (convolutions, pooling) and containers (Sequential).
# HOW: Call specific layers like layers.Conv2D() or models.Sequential().
from tensorflow.keras import layers, models

# WHAT: Imports regularizers, optimizers, losses, and metrics modules.
# WHY: Gives us explicit access to configure how the network learns and is evaluated.
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

print("TensorFlow version:", tf.__version__)


# ==============================================================================
# MODEL 1: THE BASELINE CNN (Simple & Standard)
# Use Case: Standard multi-class classification where labels are integers (e.g., 0, 1, 2).
# ==============================================================================
print("\n--- Building Model 1: Baseline ---")

# WHAT/HOW: Initializes a linear, top-to-bottom stack of layers.
model_1 = models.Sequential(name="Baseline_CNN")

# WHAT: 2D Convolutional layer (32 filters, 3x3 size, ReLU activation).
# WHY: Scans the 32x32 RGB image to extract basic features like edges and colors.
model_1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# WHAT: MaxPooling layer (2x2 window).
# WHY: Downsamples the image, reducing computational load and preventing overfitting.
model_1.add(layers.MaxPooling2D((2, 2)))

# WHAT: Second Convolutional layer (64 filters).
# WHY: Extracts higher-level features from the pooled data.
model_1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_1.add(layers.MaxPooling2D((2, 2)))

# WHAT: Flattens the 3D data into a 1D array.
# WHY: Dense layers require 1D inputs.
model_1.add(layers.Flatten())

# WHAT: Fully connected (Dense) layer (64 neurons).
# WHY: Combines features to reason about the image content.
model_1.add(layers.Dense(64, activation='relu'))

# WHAT: Final classification layer (10 neurons for 10 classes, Softmax activation).
# WHY: Softmax turns raw outputs into a probability distribution summing to 1.0.
model_1.add(layers.Dense(10, activation='softmax'))

# WHAT: Compiles the model with Adam and Sparse Categorical Crossentropy.
# WHY Adam?: Automatically adjusts learning rates, fast and reliable.
# WHY SparseCategoricalCrossentropy?: Best when target labels are simple integers (0, 1, 2).
model_1.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])


# ==============================================================================
# MODEL 2: THE ROBUST CNN (Handles Overfitting)
# Use Case: Complex datasets where the model memorizes data. Uses one-hot encoded labels.
# ==============================================================================
print("--- Building Model 2: Robust with Dropout & BatchNorm ---")

model_2 = models.Sequential(name="Robust_CNN")

model_2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# WHAT: Batch Normalization layer.
# WHY: Normalizes layer outputs, stabilizing learning and allowing faster convergence.
model_2.add(layers.BatchNormalization())
model_2.add(layers.MaxPooling2D((2, 2)))

model_2.add(layers.Conv2D(64, (3, 3), activation='relu'))
model_2.add(layers.BatchNormalization())
model_2.add(layers.MaxPooling2D((2, 2)))

model_2.add(layers.Flatten())

# WHAT: Dropout layer (drops 50% of connections).
# WHY: Forces the network to not rely on any single neuron, combating overfitting.
model_2.add(layers.Dropout(0.5))

model_2.add(layers.Dense(128, activation='relu'))
model_2.add(layers.Dense(10, activation='softmax'))

# WHAT: Stochastic Gradient Descent (SGD) with momentum.
# WHY SGD+Momentum?: Slower than Adam, but often finds a better, more generalized global minimum.
opt_sgd = optimizers.SGD(learning_rate=0.01, momentum=0.9)

# WHAT: Compiles with Categorical Crossentropy.
# WHY CategoricalCrossentropy?: Required when targets are one-hot encoded (e.g., [0, 0, 1, 0...]).
model_2.compile(optimizer=opt_sgd,
                loss='categorical_crossentropy',
                metrics=[metrics.CategoricalAccuracy(name='acc')])


# ==============================================================================
# MODEL 3: THE VGG-STYLE DEEP CNN
# Use Case: Extracting highly complex, hierarchical features from detailed images.
# ==============================================================================
print("--- Building Model 3: VGG-Style Deep CNN ---")

model_3 = models.Sequential(name="VGG_Style_CNN")

# WHAT: Block 1 - Two consecutive convolutions before pooling.
# WHY: Stacking convolutions increases the "receptive field" (how much of the original image the layer sees) without aggressively losing spatial resolution right away.
model_3.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model_3.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model_3.add(layers.MaxPooling2D((2, 2)))

# WHAT: Block 2 - Doubling filters after pooling.
# WHY: As image dimensions shrink, we increase depth to maintain information capacity.
model_3.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model_3.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model_3.add(layers.MaxPooling2D((2, 2)))

model_3.add(layers.Flatten())
model_3.add(layers.Dense(256, activation='relu'))
model_3.add(layers.Dense(10, activation='softmax'))

# WHAT: RMSprop optimizer and Precision/Recall metrics.
# WHY RMSprop?: Excellent for deep networks; it adapts learning rates by dividing the gradient by a running average of its recent magnitude.
# WHY Precision/Recall?: Better metrics than accuracy if your classes are imbalanced.
model_3.compile(optimizer=optimizers.RMSprop(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=[metrics.Precision(name='precision'), 
                         metrics.Recall(name='recall')])


# ==============================================================================
# MODEL 4: BINARY CLASSIFICATION CNN (Cats vs. Dogs)
# Use Case: When you only have TWO classes (True/False, Cat/Dog, Defective/Perfect).
# ==============================================================================
print("--- Building Model 4: Binary Classification ---")

model_4 = models.Sequential(name="Binary_CNN")

model_4.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3))) # Note: Assumes 64x64 images
model_4.add(layers.MaxPooling2D((2, 2)))
model_4.add(layers.Flatten())
model_4.add(layers.Dense(32, activation='relu'))

# WHAT: Final layer with ONE neuron and a 'sigmoid' activation.
# WHY 1 Neuron/Sigmoid?: Sigmoid outputs a single value between 0 and 1. If > 0.5, class 1; if < 0.5, class 0.
model_4.add(layers.Dense(1, activation='sigmoid'))

# WHAT: Adamax optimizer and Binary Crossentropy loss.
# WHY Adamax?: A variant of Adam based on the infinity norm. Sometimes superior on models with embeddings or sparse updates.
# WHY BinaryCrossentropy?: The mathematically correct loss function when predicting a single probability (0 to 1).
model_4.compile(optimizer=optimizers.Adamax(learning_rate=0.002),
                loss='binary_crossentropy',
                metrics=['accuracy'])


# ==============================================================================
# MODEL 5: THE L2-REGULARIZED CNN
# Use Case: Heavy overfitting where Dropout isn't enough; forces network weights to stay small.
# ==============================================================================
print("--- Building Model 5: L2 Regularized CNN ---")

model_5 = models.Sequential(name="Regularized_CNN")

# WHAT: Adds kernel_regularizer=regularizers.l2(0.001) to the Conv2D layer.
# WHY: Adds a penalty to the loss function based on the squared value of the weights. Keeps weights small, preventing the model from fitting too closely to training noise.
model_5.add(layers.Conv2D(32, (3, 3), activation='relu', 
                          kernel_regularizer=regularizers.l2(0.001),
                          input_shape=(32, 32, 3)))
model_5.add(layers.MaxPooling2D((2, 2)))

model_5.add(layers.Flatten())

# WHAT: Applies L2 regularization to the Dense layer as well.
model_5.add(layers.Dense(64, activation='relu', 
                         kernel_regularizer=regularizers.l2(0.001)))
model_5.add(layers.Dense(10, activation='softmax'))

# WHAT: Nadam optimizer.
# WHY Nadam?: Stands for Nesterov-accelerated Adaptive Moment Estimation. It combines Adam with Nesterov momentum, often yielding a slightly faster convergence than standard Adam.
model_5.compile(optimizer=optimizers.Nadam(learning_rate=0.001),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

print("\nAll 5 models have been successfully built and compiled!")




# ==============================================================================
# 6. DATA PREPARATION FOR ALL 5 MODELS
# ==============================================================================
import matplotlib.pyplot as plt

print("\n--- Downloading and Preparing CIFAR-10 Dataset ---")
# WHAT: Load standard 32x32 color images.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# WHAT: Normalize pixel values to be between 0 and 1 for faster convergence.
x_train, x_test = x_train / 255.0, x_test / 255.0

# --- Adjustments for Model 2 & 3 (Needs One-Hot Encoding) ---
y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

# --- Adjustments for Model 4 (Needs 64x64 images & Binary Labels) ---
print("--- Resizing a subset of data for Model 4 (64x64 Binary) ---")
# We use a 5,000 image subset so we don't crash your computer's RAM!
# We convert labels to 1 (if class is 0) and 0 (for all other classes).
x_train_bin = tf.image.resize(x_train[:5000], (64, 64)).numpy()
x_test_bin = tf.image.resize(x_test[:1000], (64, 64)).numpy()
y_train_bin = (y_train[:5000] == 0).astype(int)
y_test_bin = (y_test[:1000] == 0).astype(int)


# ==============================================================================
# 7. THE TRAINING AND PLOTTING ENGINE
# ==============================================================================
def train_and_evaluate(model, x_tr, y_tr, x_te, y_te, metric_key='accuracy'):
    print(f"\n{'='*50}")
    print(f" TRAINING: {model.name}")
    print(f"{'='*50}")
    
    # WHAT: Train the model for 5 epochs.
    history = model.fit(x_tr, y_tr, epochs=5, validation_data=(x_te, y_te), batch_size=64, verbose=1)
    
    # WHAT: Evaluate the model on both train and test data to get final numbers.
    train_results = model.evaluate(x_tr, y_tr, verbose=0)
    test_results = model.evaluate(x_te, y_te, verbose=0)
    
    # Keras returns [Loss, Metric]. We grab index 1 to print the primary metric.
    print(f"\n--> {model.name} Final Training {metric_key.capitalize()}: {train_results[1]*100:.2f}%")
    print(f"--> {model.name} Final Testing {metric_key.capitalize()}:  {test_results[1]*100:.2f}%")
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 4))
    
    # Plot 1: The Metric (Accuracy or Precision)
    plt.subplot(1, 2, 1)
    val_metric_key = f'val_{metric_key}'
    if metric_key in history.history:
        plt.plot(history.history[metric_key], label=f'Train {metric_key.capitalize()}', color='blue')
        plt.plot(history.history[val_metric_key], label=f'Test {metric_key.capitalize()}', color='orange')
    plt.title(f'{model.name} - {metric_key.capitalize()}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_key.capitalize())
    plt.legend()
    plt.grid(True)
    
    # Plot 2: The Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Test Loss', color='orange')
    plt.title(f'{model.name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# ==============================================================================
# 8. EXECUTE ALL MODELS
# ==============================================================================

# Run Model 1: Needs standard integer labels
train_and_evaluate(model_1, x_train, y_train, x_test, y_test, metric_key='accuracy')

# Run Model 2: Needs one-hot labels. Metric was named 'acc' in compilation.
train_and_evaluate(model_2, x_train, y_train_one_hot, x_test, y_test_one_hot, metric_key='acc')

# Run Model 3: Needs one-hot labels. Uses 'precision' and 'recall'. We plot precision.
train_and_evaluate(model_3, x_train, y_train_one_hot, x_test, y_test_one_hot, metric_key='precision')

# Run Model 4: Needs the 64x64 resized data and binary labels
train_and_evaluate(model_4, x_train_bin, y_train_bin, x_test_bin, y_test_bin, metric_key='accuracy')

# Run Model 5: Needs standard integer labels
train_and_evaluate(model_5, x_train, y_train, x_test, y_test, metric_key='accuracy')