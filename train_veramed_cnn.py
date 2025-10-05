import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import matplotlib.pyplot as plt
import os

# --- 1. SETUP PARAMETERS ---
DATASET_PATH = 'VERAMED-AI-DATASET'
IMG_WIDTH = 180
IMG_HEIGHT = 180
BATCH_SIZE = 32
EPOCHS = 20 # Increased epochs slightly for the more complex task

# --- 2. LOAD AND PREPARE DATA ---
print("Loading and preparing training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

print("Loading and preparing validation data...")
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names
num_classes = len(class_names) # Get the number of medicine classes
print(f"Found {num_classes} classes: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. BUILD THE CNN MODEL ---
print("Building the CNN model...")
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2), # Added a dropout layer to prevent overfitting
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    # --- THIS LAYER IS CHANGED ---
    layers.Dense(num_classes, activation='softmax')
])

# --- 4. COMPILE THE MODEL ---
print("Compiling the model...")
model.compile(optimizer='adam',
              # --- THIS LOSS FUNCTION IS CHANGED ---
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 5. TRAIN THE MODEL ---
print("Starting model training...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("Model training finished.")

# --- 6. SAVE THE MODEL ---
print("Saving the trained model...")
model.save('veramed_multiclass_cnn_model.h5')
print("Model saved as veramed_multiclass_cnn_model.h5")

# --- 7. VISUALIZE TRAINING RESULTS ---
# (This part of the code remains the same)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()