import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- CONFIGURATION (HYPERPARAMETERS) ---
IMG_SIZE = 128  # Smaller size for faster custom training (was 224)
BATCH_SIZE = 32
EPOCHS = 30     # Experiment: Try 10, 20, or 30
DATA_DIR = "data" # Ensure your images are in folders: data/up, data/down, etc.

# --- UPDATE THIS IN YOUR train.py ---

def build_model(num_classes):
    model = Sequential([
        # Block 1 (The Outline Detector)
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),

        # Block 2 (The Shape Detector)
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Block 3 (The Feature Detector)
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # NEW: Block 4 (The Detail Detector)
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),

        # Bigger Brain Layer
        Dense(256, activation='relu'), 
        Dropout(0.5), # Keep this! It stops the model from memorizing.
        
        Dense(num_classes, activation='softmax')
    ])
    return model

def main():
    # 1. Data Augmentation (The "Gym" for AI)
    # This creates "new" images by zooming, rotating, and shifting existing ones
    train_datagen = ImageDataGenerator(
        rescale=1./255,         # Normalize pixel values (0-1)
        rotation_range=20,      # Rotate slightly
        width_shift_range=0.2,  # Move left/right
        height_shift_range=0.2, # Move up/down
        zoom_range=0.2,         # Zoom in/out
        horizontal_flip=False,  # Keep False if left/right hands mean different things
        validation_split=0.2    # Use 20% of data for testing
    )

    # 2. Load Data Generators
    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='training'
    )

    print("Loading Validation Data...")
    val_generator = train_datagen.flow_from_directory(
        DATA_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        subset='validation'
    )

    # 3. Build & Compile Model
    print("Building Custom CNN...")
    model = build_model(num_classes=train_generator.num_classes)
    
    model.compile(
        optimizer='adam', # The "Smart Coach"
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary() # Prints the architecture table

    # 4. Train
    print(f"Starting Training for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # 5. Save the Model (Standard Format)
    model.save("custom_drone_model.h5")
    print("✅ Model saved as 'custom_drone_model.h5'")

    # 6. Convert to TFLite (Optimization)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    with open("custom_model.tflite", "wb") as f:
        f.write(tflite_model)
    print("✅ Quantized model saved as 'custom_model.tflite'")

    # Save Class Names
    class_names = list(train_generator.class_indices.keys())
    with open("labels.txt", "w") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"✅ Labels saved: {class_names}")

if __name__ == "__main__":
    main()