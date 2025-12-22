"""
Improved ASL Model Training with Data Augmentation

This script improves accuracy by:
1. More training epochs (20 instead of 10)
2. Data augmentation (rotation, zoom, shift, flip)
3. Better learning rate schedule
4. Validation monitoring
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# Configuration
TRAINING_DATA_DIR = "trainingdata"
IMAGE_SIZE = 32
BATCH_SIZE = 32
EPOCHS = 50  # Maximum training for best accuracy
LEARNING_RATE = 0.001

print("="*60)
print("ASL Model Training - IMPROVED VERSION")
print("="*60)

# Check if training data exists
if not os.path.exists(TRAINING_DATA_DIR):
    print(f"\n❌ ERROR: '{TRAINING_DATA_DIR}' folder not found!")
    exit(1)

print(f"\n[1/5] Loading and augmenting training images...")

# Data augmentation - creates more variety from your images!
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize to 0-1
    rotation_range=15,  # Rotate images slightly
    width_shift_range=0.1,  # Shift horizontally
    height_shift_range=0.1,  # Shift vertically
    zoom_range=0.1,  # Slight zoom
    horizontal_flip=False,  # Don't flip (ASL signs are not symmetric)
    validation_split=0.2  # 20% for validation
)

# Load training data
train_generator = datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training',
    shuffle=True
)

# Load validation data
validation_generator = datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"✓ Found {train_generator.samples} training images")
print(f"✓ Found {validation_generator.samples} validation images")
print(f"✓ Detected {num_classes} classes")

print(f"\n[2/5] Building improved model architecture...")

model = keras.Sequential([
    keras.layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    
    # More powerful convolutional layers
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Dropout(0.25),
    
    # Dense layers
    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

print("✓ Improved model architecture created")
model.summary()

print(f"\n[3/5] Compiling model with optimizations...")

# Use learning rate decay for better convergence
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks for better training
callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=0.00001
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

print("✓ Model compiled with learning rate decay")

print(f"\n[4/5] Training model for up to {EPOCHS} epochs...")
print("(May stop early if performance plateaus)\n")

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

final_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100

print(f"\n✓ Training complete!")
print(f"  Final training accuracy: {final_acc:.2f}%")
print(f"  Final validation accuracy: {final_val_acc:.2f}%")

print(f"\n[5/5] Converting to TFLite format...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

output_file = "fine_tuned_model.tflite"
with open(output_file, 'wb') as f:
    f.write(tflite_model)

print(f"✓ TFLite model saved as '{output_file}'")
print(f"  Size: {len(tflite_model) / 1024:.2f} KB")

# Auto-deploy to app
import shutil
try:
    shutil.copy(output_file, "assets/model.tflite")
    print(f"\n✓ Model automatically deployed to assets/model.tflite")
except Exception as e:
    print(f"\n⚠ Manual deployment needed: copy {output_file} to assets/model.tflite")

print("\n" + "="*60)
print("SUCCESS! Improved model ready!")
print("="*60)
print(f"\nImprovements over previous version:")
print(f"• 2x more training epochs ({EPOCHS} vs 10)")
print(f"• Data augmentation (rotation, zoom, shift)")
print(f"• Deeper architecture (more layers)")
print(f"• Learning rate decay")
print(f"• Early stopping prevents overfitting")
print("\nRun: flutter run")
print("="*60)
