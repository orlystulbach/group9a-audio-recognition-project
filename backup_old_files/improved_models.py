import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

def create_improved_dense_model(input_shape, num_classes, dropout_rate=0.3):
    """
    Improved dense neural network with regularization
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First layer with more units
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Second layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Third layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_cnn_model(input_shape, num_classes, dropout_rate=0.3):
    """
    CNN model for mel spectrogram input (2D)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_1d_cnn_model(input_shape, num_classes, dropout_rate=0.3):
    """
    1D CNN model for raw audio input
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First conv block
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(dropout_rate),
        
        # Second conv block
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(dropout_rate),
        
        # Third conv block
        layers.Conv1D(256, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(dropout_rate),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_ensemble_model(models, input_shape, num_classes):
    """
    Create an ensemble model that combines multiple models
    """
    inputs = layers.Input(shape=input_shape)
    
    # Get predictions from each model
    predictions = []
    for model in models:
        model.trainable = False  # Freeze the models
        pred = model(inputs)
        predictions.append(pred)
    
    # Average the predictions
    ensemble_output = layers.Average()(predictions)
    
    ensemble_model = keras.Model(inputs=inputs, outputs=ensemble_output)
    return ensemble_model

def get_callbacks(patience=10, min_lr=1e-7):
    """
    Get training callbacks for better training
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=min_lr,
            verbose=1
        )
    ]
    return callbacks

def compile_model(model, learning_rate=0.001):
    """
    Compile model with appropriate optimizer and loss
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Test dense model
    dense_model = create_improved_dense_model((100,), 10)
    dense_model = compile_model(dense_model)
    print(f"Dense model summary:")
    dense_model.summary()
    
    # Test CNN model
    cnn_model = create_cnn_model((128, 128, 1), 10)
    cnn_model = compile_model(cnn_model)
    print(f"\nCNN model summary:")
    cnn_model.summary()
    
    # Test 1D CNN model
    cnn1d_model = create_1d_cnn_model((16000,), 10)
    cnn1d_model = compile_model(cnn1d_model)
    print(f"\n1D CNN model summary:")
    cnn1d_model.summary() 