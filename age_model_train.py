import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

 
DATASET_PATH = "crop_part1"       # Path to UTKFace cropped dataset
IMG_SIZE = 64
EPOCHS = 15
BATCH_SIZE = 64
MODEL_SAVE_PATH = "age_cnn_model.h5"
PLOT_SAVE_PATH = "age_mae_chart.png"

 
# LOAD DATASET
 
print("📂 Loading UTKFace images...")
images = []
ages = []

for file in os.listdir(DATASET_PATH):
    try:
        if file.endswith(".jpg"):
            age = int(file.split("_")[0])
            img_path = os.path.join(DATASET_PATH, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            images.append(img)
            ages.append(age)
    except Exception as e:
        continue

images = np.array(images, dtype="float32")
ages = np.array(ages, dtype="float32")

print(f"✅ Loaded {len(images)} images successfully.")

 
# TRAIN-TEST SPLIT
 
X_train, X_test, y_train, y_test = train_test_split(
    images, ages, test_size=0.2, random_state=42
)

 
# MODEL DEFINITION
 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='linear')  # Regression output for age
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='mae', metrics=['mae'])
model.summary()

 
# TRAINING
 
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1
)

print("\n🚀 Starting training...\n")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint]
)

 
# EVALUATION
 
print("\n📊 Evaluating best model...")
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Final Test MAE: {mae:.2f} years")

 
# PLOT & SAVE MAE CHART
 
plt.figure(figsize=(8, 5))
plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
plt.title("Age Prediction Model Performance (MAE)", fontsize=14)
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error (years)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)

# Annotate final MAE value
plt.text(
    0.5, 0.92,
    f"Final Test MAE: {mae:.2f} years",
    transform=plt.gca().transAxes,
    fontsize=12,
    color='blue',
    bbox=dict(facecolor='white', alpha=0.6, edgecolor='blue')
)

plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH, dpi=300)
plt.close()

print(f"📉 Training chart saved as '{PLOT_SAVE_PATH}'")
print(f"💾 Model saved as '{MODEL_SAVE_PATH}'")
print("🎉 Training completed successfully!")
