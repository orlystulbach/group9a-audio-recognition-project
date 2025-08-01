#!/usr/bin/env python3
"""
Train Voice-Adaptive Model with Data Augmentation
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import noisereduce as nr

# Configuration
ORIGINAL_DATA_DIR = 'large_dataset'
VOICE_SAMPLES_DIR = 'my_voice_samples'
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def extract_robust_features(audio_data, sample_rate=22050, n_mfcc=13):
    """
    Extract robust features with noise reduction
    """
    try:
        # Noise reduction
        audio_clean = nr.reduce_noise(y=audio_data, sr=sample_rate)
        
        # Normalize audio
        audio_clean = librosa.util.normalize(audio_clean)
        
        # Ensure 2-second duration
        target_length = int(22050 * 2.0)
        if len(audio_clean) > target_length:
            audio_clean = audio_clean[:target_length]
        elif len(audio_clean) < target_length:
            audio_clean = np.pad(audio_clean, (0, target_length - len(audio_clean)), 'constant')
        
        features = []
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio_clean, sr=sample_rate, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_clean, sr=sample_rate, hop_length=512)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_clean, sr=sample_rate, hop_length=512)[0]
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_clean, hop_length=512)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # RMS energy
        rms = librosa.feature.rms(y=audio_clean, hop_length=512)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def augment_audio(audio_data, sample_rate=22050):
    """
    Apply data augmentation to audio
    """
    augmented_samples = []
    
    # Original sample
    augmented_samples.append(audio_data)
    
    # 1. Pitch shift (±2 semitones)
    try:
        pitch_shift_up = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=2)
        augmented_samples.append(pitch_shift_up)
        
        pitch_shift_down = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=-2)
        augmented_samples.append(pitch_shift_down)
    except:
        pass
    
    # 2. Time stretch (±20%)
    try:
        time_stretch_fast = librosa.effects.time_stretch(audio_data, rate=1.2)
        # Ensure 2-second duration
        target_length = int(22050 * 2.0)
        if len(time_stretch_fast) > target_length:
            time_stretch_fast = time_stretch_fast[:target_length]
        elif len(time_stretch_fast) < target_length:
            time_stretch_fast = np.pad(time_stretch_fast, (0, target_length - len(time_stretch_fast)), 'constant')
        augmented_samples.append(time_stretch_fast)
        
        time_stretch_slow = librosa.effects.time_stretch(audio_data, rate=0.8)
        if len(time_stretch_slow) > target_length:
            time_stretch_slow = time_stretch_slow[:target_length]
        elif len(time_stretch_slow) < target_length:
            time_stretch_slow = np.pad(time_stretch_slow, (0, target_length - len(time_stretch_slow)), 'constant')
        augmented_samples.append(time_stretch_slow)
    except:
        pass
    
    # 3. Add noise
    try:
        noise_factor = 0.005
        noise = np.random.randn(len(audio_data))
        noisy_audio = audio_data + noise_factor * noise
        augmented_samples.append(noisy_audio)
    except:
        pass
    
    # 4. Volume variation
    try:
        volume_up = audio_data * 1.3
        augmented_samples.append(volume_up)
        
        volume_down = audio_data * 0.7
        augmented_samples.append(volume_down)
    except:
        pass
    
    return augmented_samples

def load_combined_dataset():
    """
    Load and combine original dataset with voice samples
    """
    print("📊 Loading combined dataset...")
    
    X, y = [], []
    
    # Load original dataset
    if os.path.exists(ORIGINAL_DATA_DIR):
        print(f"Loading original dataset from {ORIGINAL_DATA_DIR}...")
        wav_files = [f for f in os.listdir(ORIGINAL_DATA_DIR) if f.endswith('.wav')]
        
        # Use a subset of original data to balance with voice samples
        subset_size = min(len(wav_files), 5000)  # Use 5000 samples from original dataset
        wav_files = wav_files[:subset_size]
        
        for i, wav_file in enumerate(wav_files):
            if i % 1000 == 0:
                print(f"Processing original file {i+1}/{len(wav_files)}")
            
            try:
                # Extract label
                label = wav_file.split('_')[0]
                
                # Load and process audio
                file_path = os.path.join(ORIGINAL_DATA_DIR, wav_file)
                audio_data, sr = librosa.load(file_path, sr=22050)
                
                # Extract features
                features = extract_robust_features(audio_data, sr)
                if features is not None:
                    X.append(features)
                    y.append(label)
                    
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
                continue
    
    # Load voice samples
    if os.path.exists(VOICE_SAMPLES_DIR):
        print(f"Loading voice samples from {VOICE_SAMPLES_DIR}...")
        
        for digit in range(10):
            digit_dir = os.path.join(VOICE_SAMPLES_DIR, str(digit))
            if os.path.exists(digit_dir):
                wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
                
                for wav_file in wav_files:
                    try:
                        file_path = os.path.join(digit_dir, wav_file)
                        audio_data, sr = librosa.load(file_path, sr=22050)
                        
                        # Extract features
                        features = extract_robust_features(audio_data, sr)
                        if features is not None:
                            X.append(features)
                            y.append(str(digit))
                            
                            # Apply data augmentation to voice samples
                            augmented_samples = augment_audio(audio_data, sr)
                            for aug_audio in augmented_samples[1:]:  # Skip original
                                aug_features = extract_robust_features(aug_audio, sr)
                                if aug_features is not None:
                                    X.append(aug_features)
                                    y.append(str(digit))
                                    
                    except Exception as e:
                        print(f"Error processing voice sample {wav_file}: {e}")
                        continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Combined dataset shape: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    print(f"Label distribution: {np.bincount([int(label) for label in y])}")
    
    return X, y

def create_adaptive_model(input_shape, num_classes):
    """
    Create an adaptive model with better architecture
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First layer - larger for better feature learning
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Second layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Third layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Fourth layer
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def train_voice_model():
    """
    Train the voice-adaptive model
    """
    print("\n🤖 Training Voice-Adaptive Model")
    print("=" * 60)
    
    # Check if voice samples exist
    if not os.path.exists(VOICE_SAMPLES_DIR):
        print("❌ Voice samples not found!")
        print("Please run: python3 collect_voice_samples.py")
        return None, None, None
    
    # Load combined dataset
    X, y = load_combined_dataset()
    
    if len(X) == 0:
        print("❌ No data found! Please collect voice samples first.")
        return None, None, None
    
    # Encode labels
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    num_classes = y_encoded.shape[1]
    
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {dict(zip(range(num_classes), lb.classes_))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Create model
    model = create_adaptive_model((X_train.shape[1],), num_classes)
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
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
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
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
    
    # Save model and scaler
    model_path = 'voice_adaptive_model.h5'
    scaler_path = 'voice_adaptive_scaler.pkl'
    label_binarizer_path = 'voice_adaptive_label_binarizer.pkl'
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(lb, label_binarizer_path)
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"✅ Label binarizer saved: {label_binarizer_path}")
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['lr'] if 'lr' in history.history else [LEARNING_RATE] * len(history.history['loss']), label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('voice_adaptive_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎉 Training complete!")
    print(f"Next step: streamlit run app_voice_adapted.py")
    
    return model, scaler, lb

if __name__ == "__main__":
    train_voice_model() 