import tensorflow as tf
import numpy as np
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Create a simple dummy model for testing
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Save the dummy model
model.save('model/brain_tumor_model.h5')
print("Dummy model created successfully!")