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
MODEL_PATH = 'deeper_digit_classifier_model.h5'

def extract_improved_features(file_path, n_mfcc=N_MFCC):
    """
    Extract comprehensive audio features
    """
    y, sr = librosa.load(file_path, sr=None)
    
    # Normalize audio
    y = librosa.util.normalize(y)
    
    # Check if audio is too quiet
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
        
        # FIXED: Extract label from FIRST part of filename (the actual digit)
        label = fname.split('_')[0]
        
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

def create_deeper_model(input_shape, num_classes):
    """
    Create a DEEPER model with additional layers and dimensions
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First layer - larger
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second layer - larger
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third layer - same as original
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Fourth layer - new dimension
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Fifth layer - new dimension
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_wide_model(input_shape, num_classes):
    """
    Create a WIDE model with more units per layer
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First layer - very wide
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Second layer - wide
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Third layer
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Fourth layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_residual_model(input_shape, num_classes):
    """
    Create a model with residual connections (skip connections)
    """
    inputs = layers.Input(shape=input_shape)
    
    # First block
    x = layers.Dense(256, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Second block with residual connection
    residual = x
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Add()([x, residual])  # Residual connection
    
    # Third block
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Fourth block with residual connection
    residual = x
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Add()([x, residual])  # Residual connection
    
    # Fifth block
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    """
    Main training function with deeper models
    """
    print("=== DEEPER Digit Classifier Training ===")
    print("This version uses deeper and more complex models!\n")
    
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
    
    # Train different model architectures
    models = {
        'deeper': create_deeper_model((X_train.shape[1],), num_classes),
        'wide': create_wide_model((X_train.shape[1],), num_classes),
        'residual': create_residual_model((X_train.shape[1],), num_classes)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name.upper()} model...")
        print(f"{'='*50}")
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        print(f"\n{model_name.upper()} Model Architecture:")
        model.summary()
        
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
        
        # Train model
        print(f"\nTraining {model_name} model...")
        history = model.fit(
            X_train_scaled, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate
        print(f"\nEvaluating {model_name} model...")
        loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
        print(f"{model_name.upper()} Test Accuracy: {accuracy*100:.2f}%")
        
        # Save model
        model_path = f'{model_name}_digit_classifier_model.h5'
        model.save(model_path)
        print(f"{model_name.upper()} model saved to {model_path}")
        
        # Store results
        results[model_name] = {
            'model': model,
            'history': history,
            'accuracy': accuracy,
            'model_path': model_path
        }
    
    # Compare results
    print(f"\n{'='*60}")
    print("FINAL RESULTS COMPARISON")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(f"{model_name.upper()}: {result['accuracy']*100:.2f}% accuracy")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 BEST MODEL: {best_model[0].upper()} with {best_model[1]['accuracy']*100:.2f}% accuracy")
    
    # Plot training histories
    plt.figure(figsize=(15, 5))
    
    for i, (model_name, result) in enumerate(results.items()):
        plt.subplot(1, 3, i+1)
        plt.plot(result['history'].history['accuracy'], label='Training')
        plt.plot(result['history'].history['val_accuracy'], label='Validation')
        plt.title(f'{model_name.upper()} Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('deeper_models_comparison.png')
    plt.show()
    
    print(f"\n=== Training completed! ===")
    print(f"All models saved. Best model: {best_model[0]}")

if __name__ == "__main__":
    main() 