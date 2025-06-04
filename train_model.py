import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BrainTumorClassifier:
    def __init__(self, img_size=(224, 224), num_classes=4):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def create_cnn_model(self):
        """Create a custom CNN model"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def create_transfer_learning_model(self):
        """Create a transfer learning model using VGG16"""
        # Load pre-trained VGG16 model
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        model = Sequential([
            base_model,
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def prepare_data_generators(self, train_dir, validation_dir, batch_size=32):
        """Prepare data generators for training"""
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def train_model(self, train_generator, validation_generator, 
                   model_type='transfer_learning', epochs=50):
        """Train the model"""
        # Create model
        if model_type == 'cnn':
            self.model = self.create_cnn_model()
        else:
            self.model = self.create_transfer_learning_model()
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        print(self.model.summary())
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def save_model(self, filepath='model/brain_tumor_model.h5'):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('model/training_history.png')
        plt.show()

def main():
    """Main training function"""
    # Initialize classifier
    classifier = BrainTumorClassifier()
    
    # Data directories (you need to download and organize the dataset)
    train_dir = 'data/Training'
    validation_dir = 'data/Testing'
    
    # Check if data directories exist
    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        print("\n⚠️  Dataset not found!")
        print("Please download the Brain MRI Dataset from Kaggle and organize it as:")
        print("data/")
        print("├── Training/")
        print("│   ├── glioma_tumor/")
        print("│   ├── meningioma_tumor/")
        print("│   ├── no_tumor/")
        print("│   └── pituitary_tumor/")
        print("└── Testing/")
        print("    ├── glioma_tumor/")
        print("    ├── meningioma_tumor/")
        print("    ├── no_tumor/")
        print("    └── pituitary_tumor/")
        print("\nDataset URL: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
        return
    
    # Prepare data generators
    print("Preparing data generators...")
    train_gen, val_gen = classifier.prepare_data_generators(train_dir, validation_dir)
    
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Classes: {list(train_gen.class_indices.keys())}")
    
    # Train model
    print("\nStarting training...")
    history = classifier.train_model(
        train_gen, 
        val_gen, 
        model_type='transfer_learning',  # or 'cnn'
        epochs=30
    )
    
    # Save model
    classifier.save_model()
    
    # Plot training history
    classifier.plot_training_history()
    
    print("\n✅ Training completed successfully!")
    print("Model saved as 'model/brain_tumor_model.h5'")
    print("You can now run the Flask app with: python app.py")

if __name__ == "__main__":
    main()
# Change these lines (around line 15)
TRAIN_DIR = 'archive/Training'
TEST_DIR = 'archive/Testing'
# Add before saving:
model.compile(optimizer='adam', 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

# Then save:
model.save('model/brain_tumor_model.h5', save_format='h5')
print("Model saved in HDF5 format with optimizer state")