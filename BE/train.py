import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

# Paths
DATASET_DIR = "dataset"  # update if needed
MODEL_PATH = "model.h5"

# Re-map dataset into 3 categories (create subfolders manually or with script)
# For training, folder structure should look like:
# dataset/
#   train/
#     plastic/
#     organic/
#     metal/
#   test/
#     plastic/
#     organic/
#     metal/

# Image preprocessing
img_size = (64, 64)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(3, activation="softmax")  # 3 classes
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5  # increase if time allows
)

# Save model
model.save(MODEL_PATH)
print("Model saved as", MODEL_PATH)
