import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_efficient_model():
    """Create an efficient model using EfficientNet backbone"""
    # Load pre-trained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(4, activation='softmax', name='predictions')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model

def train_model():
    """Train the improved model"""
    # Create model
    model, base_model = create_efficient_model()
    
    # Compile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_2_accuracy']
    )
    
    # Data augmentation for better generalization
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Load training data
    train_generator = train_datagen.flow_from_directory(
        'data/Training',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        'data/Training',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Callbacks for better training
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-7),
        ModelCheckpoint('model/best_model.h5', save_best_only=True)
    ]
    
    # Initial training
    print("Phase 1: Training with frozen base model...")
    history1 = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Fine-tuning phase
    print("Phase 2: Fine-tuning with unfrozen layers...")
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001/10),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_2_accuracy']
    )
    
    history2 = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('model/brain_tumor_model.h5')
    print("Model training completed and saved!")
    
    return model

if __name__ == "__main__":
    # Ensure model directory exists
    os.makedirs('model', exist_ok=True)
    train_model()