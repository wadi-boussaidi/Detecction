import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Dataset Path
dataset_path = r"C:\Users\badii\.cache\kagglehub\datasets\pratik2901\animal-dataset\versions\1\animal_dataset_intermediate\train"

# Image Parameters
img_size = (128, 128)  # Resize all images to 128x128
batch_size = 32

# Load Dataset
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation"
)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile Model
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(train_generator, validation_data=val_generator, epochs=5)

# Save Model
model.save("animal_classifier.h5")

# Print Final Accuracy
final_acc = history.history['val_accuracy'][-1] * 100
print(f"Final Validation Accuracy: {final_acc:.2f}%")

# Save Accuracy to File
with open("accuracy.txt", "w") as f:
    f.write(f"Model Accuracy: {final_acc:.2f}%")