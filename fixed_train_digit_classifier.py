import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Config
DATA_DIR = 'large_dataset'
N_MFCC = 13
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 50
BATCH_SIZE = 32
MODEL_PATH = 'fixed_digit_classifier_model.h5'

def extract_improved_features(file_path, n_mfcc=N_MFCC):
    """
    Extract comprehensive audio features
    """
    y, sr = librosa.load(file_path, sr=None)
    
    # Normalize audio
    y = librosa.util.normalize(y)
    
    # Check if audio is too quiet, shhhhhhhh
    rms = np.sqrt(np.mean(y**2))
    if rms < 0.01:
        print(f"Warning: Very quiet audio in {file_path} (RMS: {rms:.4f})")
    
    features = []
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Statistical features from MFCC
    features.extend(np.mean(mfcc, axis=1))  # Mean of each coefficient
    features.extend(np.std(mfcc, axis=1))   # Std of each coefficient
    features.extend(np.mean(mfcc_delta, axis=1))  # Delta features
    features.extend(np.mean(mfcc_delta2, axis=1)) # Delta-delta features
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
    
    features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
    features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
    features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    features.extend(np.mean(chroma, axis=1))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
    features.extend([np.mean(zcr), np.std(zcr)])
    
    # RMS energy
    rms = librosa.feature.rms(y=y, hop_length=512)[0]
    features.extend([np.mean(rms), np.std(rms)])
    
    return np.array(features)

def load_and_preprocess_data():
    """
    Load and preprocess data with CORRECTED label extraction
    """
    print("Loading and preprocessing data with CORRECTED labels...")
    
    X, y = [], []
    
    # Get all wav files
    wav_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.wav')]
    print(f"Found {len(wav_files)} audio files")
    
    # Process each file
    for i, fname in enumerate(wav_files):
        if i % 1000 == 0:
            print(f"Processing file {i+1}/{len(wav_files)}")
        
        # FIXED THE ISSUE!!!: Extract label from FIRST part of filename (the actual digit, this was the problem)
        label = fname.split('_')[0]  # This gets the digit, not the instance number
        
        file_path = os.path.join(DATA_DIR, fname)
        
        try:
            features = extract_improved_features(file_path)
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

def create_improved_model(input_shape, num_classes):
    """
    Create improved model with regularization
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First layer
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer, same as previous model
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def main():
    """
    Main training function with corrected labels
    """
    print("=== FIXED Digit Classifier Training ===")
    print("This version fixes the label extraction bug!\n")
    
    # Load data with corrected labels
    X, y = load_and_preprocess_data()
    
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
    
    # Create and compile model
    model = create_improved_model((X_train.shape[1],), num_classes)
    
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
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
    
    print("\nTraining model...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    print("\n=== Training completed! ===")
    print(f"Expected improvement: From ~2-6% to {accuracy*100:.1f}% accuracy")
    print("The main fix was correcting the label extraction logic.")

if __name__ == "__main__":
    main() 