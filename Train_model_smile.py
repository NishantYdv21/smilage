import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
DATA_DIR = "kaggle-genki4k"     # Dataset folder structure:
# kaggle-genki4k/
#   ├── smile/
#   └── non-smile/
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "smile_cnn_model.h5"
PLOT_PATH = "smile_accuracy_chart.png"

# =========================================================
# LOAD DATASET
# =========================================================
print("📦 Loading dataset from folders...")

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print(f"✅ Found {train_gen.samples} training and {val_gen.samples} validation images.\n")

# =========================================================
# BUILD CNN MODEL
# =========================================================
print("🧠 Building CNN model...")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary: smile / non-smile
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================================
# TRAINING
# =========================================================
print("\n🚀 Training model...\n")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# =========================================================
# EVALUATION
# =========================================================
val_loss, val_acc = model.evaluate(val_gen)
print(f"\n✅ Validation accuracy: {val_acc * 100:.2f}%")

# =========================================================
# SAVE MODEL
# =========================================================
model.save(MODEL_PATH)
print(f"💾 Model saved as {MODEL_PATH}")

# =========================================================
# SAVE TRAINING ACCURACY CHART
# =========================================================
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title("Smile Detection Model Accuracy", fontsize=14)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Add annotation with final accuracy
plt.text(
    0.5, 0.92,
    f"Final Validation Accuracy: {val_acc*100:.2f}%",
    transform=plt.gca().transAxes,
    fontsize=12,
    color='blue',
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='blue')
)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=300)
plt.close()

print(f"📊 Accuracy chart saved as '{PLOT_PATH}'")
print("🎉 Training completed successfully!")
