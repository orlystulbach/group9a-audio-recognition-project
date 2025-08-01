#!/usr/bin/env python3
"""
Retrain Robust Model - Create a more robust model with better features
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

def extract_robust_features(audio_data, sample_rate=22050, n_mfcc=13):
    """Extract robust features with noise reduction and enhancement"""
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
        
        # Apply pre-emphasis filter to enhance high frequencies
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio_clean[0], audio_clean[1:] - pre_emphasis * audio_clean[:-1])
        
        features = []
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=emphasized_audio, sr=sample_rate, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        
        # Delta and delta-delta MFCC (captures temporal changes)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))
        features.extend(np.mean(mfcc_delta2, axis=1))
        features.extend(np.std(mfcc_delta2, axis=1))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=emphasized_audio, sr=sample_rate, hop_length=512)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=emphasized_audio, sr=sample_rate, hop_length=512)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=emphasized_audio, sr=sample_rate, hop_length=512)[0]
        
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(emphasized_audio, hop_length=512)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # RMS energy
        rms = librosa.feature.rms(y=emphasized_audio, hop_length=512)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        # Chroma features (pitch-based)
        chroma = librosa.feature.chroma_stft(y=emphasized_audio, sr=sample_rate, hop_length=512)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting robust features: {e}")
        return None

def augment_audio(audio_data, sample_rate=22050):
    """Apply data augmentation to create more training samples"""
    augmented_samples = []
    
    # Original sample
    augmented_samples.append(audio_data)
    
    # Pitch shift (±2 semitones)
    for shift in [-2, -1, 1, 2]:
        try:
            shifted = librosa.effects.pitch_shift(audio_data, sr=sample_rate, n_steps=shift)
            augmented_samples.append(shifted)
        except:
            pass
    
    # Time stretch (±20%)
    for stretch in [0.8, 0.9, 1.1, 1.2]:
        try:
            stretched = librosa.effects.time_stretch(audio_data, rate=stretch)
            # Ensure 2-second duration
            target_length = int(22050 * 2.0)
            if len(stretched) > target_length:
                stretched = stretched[:target_length]
            elif len(stretched) < target_length:
                stretched = np.pad(stretched, (0, target_length - len(stretched)), 'constant')
            augmented_samples.append(stretched)
        except:
            pass
    
    # Add noise (small amount)
    try:
        noise = np.random.normal(0, 0.001, len(audio_data))
        noisy = audio_data + noise
        augmented_samples.append(noisy)
    except:
        pass
    
    # Volume variation
    for factor in [0.8, 0.9, 1.1, 1.2]:
        try:
            volume_varied = audio_data * factor
            augmented_samples.append(volume_varied)
        except:
            pass
    
    return augmented_samples

def load_augmented_dataset():
    """Load dataset with augmentation"""
    print("📊 Loading augmented dataset...")
    
    X, y = [], []
    
    for digit in range(10):
        digit_dir = os.path.join('my_voice_samples', str(digit))
        if os.path.exists(digit_dir):
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            
            for wav_file in wav_files:
                try:
                    file_path = os.path.join(digit_dir, wav_file)
                    audio_data, sr = librosa.load(file_path, sr=22050)
                    
                    # Create augmented samples
                    augmented_samples = augment_audio(audio_data, sr)
                    
                    for sample in augmented_samples:
                        features = extract_robust_features(sample, sr)
                        if features is not None:
                            X.append(features)
                            y.append(str(digit))
                        
                except Exception as e:
                    print(f"Error processing {wav_file}: {e}")
                    continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Augmented dataset shape: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    print(f"Label distribution: {np.bincount([int(label) for label in y])}")
    
    return X, y

def create_robust_model(input_shape, num_classes):
    """Create a more robust model architecture"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First layer - larger
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

def retrain_robust_model():
    """Retrain with robust features and augmentation"""
    print("\n🤖 Retraining Robust Model")
    print("=" * 60)
    
    # Load augmented dataset
    X, y = load_augmented_dataset()
    
    if len(X) == 0:
        print("❌ No data found!")
        return None, None, None
    
    # Encode labels
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    num_classes = y_encoded.shape[1]
    
    print(f"Number of classes: {num_classes}")
    print(f"Class mapping: {dict(zip(range(num_classes), lb.classes_))}")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features properly
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Check feature statistics after scaling
    print(f"\n📊 Feature Statistics After Scaling:")
    print(f"  Training mean: {np.mean(X_train_scaled, axis=0)}")
    print(f"  Training std: {np.std(X_train_scaled, axis=0)}")
    print(f"  Training range: {np.min(X_train_scaled):.6f} to {np.max(X_train_scaled):.6f}")
    
    # Create model
    model = create_robust_model((X_train.shape[1],), num_classes)
    
    # Compile model with better optimizer
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)  # Lower learning rate
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
            patience=15,  # More patience
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
        epochs=100,  # More epochs
        batch_size=32,  # Larger batch size
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print(f"\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Test on training data to verify
    train_loss, train_accuracy = model.evaluate(X_train_scaled, y_train, verbose=0)
    print(f"Training Accuracy: {train_accuracy*100:.2f}%")
    
    # Save model and scaler
    model_path = 'robust_voice_model.h5'
    scaler_path = 'robust_voice_scaler.pkl'
    label_binarizer_path = 'robust_voice_label_binarizer.pkl'
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(lb, label_binarizer_path)
    
    print(f"✅ Model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"✅ Label binarizer saved: {label_binarizer_path}")
    
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
    plt.savefig('robust_model_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, scaler, lb

def test_robust_model():
    """Test the robust model"""
    print("\n🧪 Testing Robust Model")
    print("=" * 50)
    
    try:
        # Load robust model
        model = keras.models.load_model('robust_voice_model.h5')
        scaler = joblib.load('robust_voice_scaler.pkl')
        
        # Test on original training data
        correct = 0
        total = 0
        
        for digit in range(10):
            digit_dir = os.path.join('my_voice_samples', str(digit))
            if os.path.exists(digit_dir):
                wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
                
                if wav_files:
                    wav_file = wav_files[0]
                    try:
                        file_path = os.path.join(digit_dir, wav_file)
                        audio_data, sr = librosa.load(file_path, sr=22050)
                        
                        features = extract_robust_features(audio_data, sr)
                        if features is not None:
                            features_reshaped = features.reshape(1, -1)
                            features_scaled = scaler.transform(features_reshaped)
                            prediction = model.predict(features_scaled, verbose=0)
                            predicted_digit = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            total += 1
                            if predicted_digit == digit:
                                correct += 1
                                print(f"✅ Digit {digit}: Predicted {predicted_digit} (correct), Confidence: {confidence:.3f}")
                            else:
                                print(f"❌ Digit {digit}: Predicted {predicted_digit} (should be {digit}), Confidence: {confidence:.3f}")
                                
                    except Exception as e:
                        print(f"❌ Error processing digit {digit}: {e}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 Robust model accuracy on original training data: {accuracy*100:.2f}% ({correct}/{total})")
        
        if accuracy > 0.8:
            print("✅ Robust model is working correctly!")
        else:
            print("⚠️ Robust model still has issues")
            
    except Exception as e:
        print(f"❌ Error testing robust model: {e}")

def main():
    """Main function"""
    print("🔧 Retraining Robust Model")
    print("=" * 60)
    
    # Check if voice samples exist
    if not os.path.exists('my_voice_samples'):
        print("❌ my_voice_samples directory not found!")
        return
    
    print("✅ Voice samples directory found")
    
    # Retrain model
    model, scaler, lb = retrain_robust_model()
    
    if model is not None:
        # Test the robust model
        test_robust_model()
        
        print(f"\n🎉 Robust model training complete!")
        print(f"Next step: streamlit run app_robust_voice.py")

if __name__ == "__main__":
    main() 