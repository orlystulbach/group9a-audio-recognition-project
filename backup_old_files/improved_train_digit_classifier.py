import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from improved_feature_extraction import extract_improved_features, extract_mel_spectrogram_features, extract_raw_audio_features
from improved_models import create_improved_dense_model, create_cnn_model, create_1d_cnn_model, get_callbacks, compile_model, plot_training_history

# Configuration
DATA_DIR = 'large_dataset'
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Feature extraction parameters
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512
MEL_TARGET_LENGTH = 128
RAW_TARGET_LENGTH = 16000

def load_and_preprocess_data():
    """
    Load and preprocess the dataset with multiple feature extraction methods
    """
    print("Loading and preprocessing data...")
    
    # Lists to store different feature types
    X_improved = []
    X_mel = []
    X_raw = []
    y = []
    
    # Get all wav files
    wav_files = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.wav'):
            wav_files.append(fname)
    
    print(f"Found {len(wav_files)} audio files")
    
    # Process each file
    for i, fname in enumerate(wav_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(wav_files)}")
        
        # Extract label from filename
        label = fname.split('_')[-1].split('.')[0]
        file_path = os.path.join(DATA_DIR, fname)
        
        try:
            # Extract different types of features
            features_improved = extract_improved_features(file_path, N_MFCC, N_FFT, HOP_LENGTH)
            features_mel = extract_mel_spectrogram_features(file_path, MEL_TARGET_LENGTH)
            features_raw = extract_raw_audio_features(file_path, RAW_TARGET_LENGTH)
            
            X_improved.append(features_improved)
            X_mel.append(features_mel)
            X_raw.append(features_raw)
            y.append(label)
            
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue
    
    # Convert to numpy arrays
    X_improved = np.array(X_improved)
    X_mel = np.array(X_mel)
    X_raw = np.array(X_raw)
    y = np.array(y)
    
    print(f"Final dataset shapes:")
    print(f"Improved features: {X_improved.shape}")
    print(f"Mel spectrograms: {X_mel.shape}")
    print(f"Raw audio: {X_raw.shape}")
    print(f"Labels: {y.shape}")
    
    return X_improved, X_mel, X_raw, y

def train_improved_dense_model(X, y, model_name="improved_dense"):
    """
    Train the improved dense model
    """
    print(f"\nTraining {model_name} model...")
    
    # Encode labels
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    num_classes = y_encoded.shape[1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and compile model
    model = create_improved_dense_model((X_train.shape[1],), num_classes)
    model = compile_model(model, LEARNING_RATE)
    
    # Get callbacks
    callbacks = get_callbacks(patience=15)
    
    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"{model_name} - Test Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    model.save(f'{model_name}_model.h5')
    print(f"Model saved as {model_name}_model.h5")
    
    return model, history, scaler, lb

def train_cnn_model(X, y, model_name="cnn"):
    """
    Train CNN model for mel spectrograms
    """
    print(f"\nTraining {model_name} model...")
    
    # Encode labels
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    num_classes = y_encoded.shape[1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    # Reshape for CNN (add channel dimension)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    
    # Create and compile model
    model = create_cnn_model((X_train.shape[1], X_train.shape[2], 1), num_classes)
    model = compile_model(model, LEARNING_RATE)
    
    # Get callbacks
    callbacks = get_callbacks(patience=15)
    
    # Train model
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"{model_name} - Test Accuracy: {accuracy*100:.2f}%")
    
    
    # Save model
    model.save(f'{model_name}_model.h5')
    print(f"Model saved as {model_name}_model.h5")
    
    return model, history, lb

def train_1d_cnn_model(X, y, model_name="1d_cnn"):
    """
    Train 1D CNN model for raw audio
    """
    print(f"\nTraining {model_name} model...")
    
    # Encode labels
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    num_classes = y_encoded.shape[1]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    
    # Reshape for 1D CNN
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Create and compile model
    model = create_1d_cnn_model((X_train.shape[1],), num_classes)
    model = compile_model(model, LEARNING_RATE)
    
    # Get callbacks
    callbacks = get_callbacks(patience=15)
    
    # Train model
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
    print(f"{model_name} - Test Accuracy: {accuracy*100:.2f}%")
    
    
    # Save model
    model.save(f'{model_name}_model.h5')
    print(f"Model saved as {model_name}_model.h5")
    
    return model, history, lb

def main():
    """
    Main training function
    """
    print("Starting improved digit classifier training...")
    
    # Load and preprocess data
    X_improved, X_mel, X_raw, y = load_and_preprocess_data()
    
    # Train different models
    results = {}
    
    # Train improved dense model
    dense_model, dense_history, dense_scaler, dense_lb = train_improved_dense_model(X_improved, y, "improved_dense")
    results['dense'] = {
        'model': dense_model,
        'history': dense_history,
        'scaler': dense_scaler,
        'label_binarizer': dense_lb
    }
    
    # Train CNN model
    cnn_model, cnn_history, cnn_lb = train_cnn_model(X_mel, y, "cnn")
    results['cnn'] = {
        'model': cnn_model,
        'history': cnn_history,
        'label_binarizer': cnn_lb
    }
    
    # Train 1D CNN model
    cnn1d_model, cnn1d_history, cnn1d_lb = train_1d_cnn_model(X_raw, y, "1d_cnn")
    results['1d_cnn'] = {
        'model': cnn1d_model,
        'history': cnn1d_history,
        'label_binarizer': cnn1d_lb
    }
    
    # Plot training histories
    print("\nPlotting training histories...")
    for name, result in results.items():
        print(f"\n{name.upper()} Model Training History:")
        plot_training_history(result['history'])
    
    print("\nTraining completed! All models saved.")
    return results

if __name__ == "__main__":
    main() 