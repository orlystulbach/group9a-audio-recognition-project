#!/usr/bin/env python3
"""
Retrain a simple, working digit classifier model
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = 'large_dataset'
N_MFCC = 13
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32
MODEL_PATH = 'simple_working_model.h5'

def extract_simple_features(file_path, n_mfcc=N_MFCC):
    """
    Extract simple, robust features
    """
    try:
        y, sr = librosa.load(file_path, sr=22050)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Ensure 2-second duration
        target_length = int(22050 * 2.0)
        if len(y) > target_length:
            y = y[:target_length]
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), 'constant')
        
        features = []
        
        # MFCC features (simple version)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        
        # Statistical features from MFCC
        features.extend(np.mean(mfcc, axis=1))  # Mean of each coefficient
        features.extend(np.std(mfcc, axis=1))   # Std of each coefficient
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
        
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def load_and_preprocess_data():
    """
    Load and preprocess data with proper label extraction
    """
    print("Loading and preprocessing data...")
    
    X, y = [], []
    
    # Get all wav files
    wav_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} audio files")
    
    # Process each file
    for i, fname in enumerate(wav_files):
        if i % 1000 == 0:
            print(f"Processing file {i+1}/{len(wav_files)}")
        
        # Extract label from filename (FIRST part is the digit)
        label = fname.split('_')[0]
        
        # Validate label
        if not label.isdigit() or int(label) < 0 or int(label) > 9:
            print(f"Skipping invalid label in {fname}: {label}")
            continue
        
        file_path = os.path.join(DATA_DIR, fname)
        
        try:
            features = extract_simple_features(file_path)
            if features is not None:
                X.append(features)
                y.append(label)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Final dataset shape: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    print(f"Label distribution: {np.bincount([int(label) for label in y])}")
    
    return X, y

def create_simple_model(input_shape, num_classes):
    """
    Create a simple, robust model
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third layer
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    """
    Main training function
    """
    print("=== SIMPLE WORKING DIGIT CLASSIFIER TRAINING ===")
    
    # Load data
    X, y = load_and_preprocess_data()
    
    if len(X) == 0:
        print("❌ No valid data found!")
        return
    
    # Encode labels
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    num_classes = y_encoded.shape[1]
    
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {dict(zip(range(num_classes), lb.classes_))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create model
    model = create_simple_model((X_train.shape[1],), num_classes)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print(f"\nModel Architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print(f"\nTraining model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print(f"\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save scaler for inference
    import joblib
    scaler_path = 'simple_model_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simple_model_training_history.png')
    plt.show()
    
    print(f"\n=== Training completed! ===")
    print(f"Model saved as: {MODEL_PATH}")
    print(f"Scaler saved as: {scaler_path}")
    print(f"Test accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main() 