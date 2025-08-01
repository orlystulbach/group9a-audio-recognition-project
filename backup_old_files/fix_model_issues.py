#!/usr/bin/env python3
"""
Fix Model Training Issues - Retrain with proper data balance and scaling
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

def extract_simple_features(audio_data, sample_rate=22050, n_mfcc=13):
    """Extract features without noise reduction"""
    try:
        # Normalize audio
        audio_clean = librosa.util.normalize(audio_data)
        
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

def load_balanced_dataset():
    """Load a balanced dataset with equal samples per digit"""
    print("📊 Loading balanced dataset...")
    
    X, y = [], []
    
    # Load voice samples with equal representation
    samples_per_digit = 15  # Use all 15 samples per digit
    
    for digit in range(10):
        digit_dir = os.path.join('my_voice_samples', str(digit))
        if os.path.exists(digit_dir):
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            
            # Use all available samples (up to samples_per_digit)
            for wav_file in wav_files[:samples_per_digit]:
                try:
                    file_path = os.path.join(digit_dir, wav_file)
                    audio_data, sr = librosa.load(file_path, sr=22050)
                    
                    features = extract_simple_features(audio_data, sr)
                    if features is not None:
                        X.append(features)
                        y.append(str(digit))
                        
                except Exception as e:
                    print(f"Error processing {wav_file}: {e}")
                    continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Balanced dataset shape: {X.shape}")
    print(f"Labels: {np.unique(y)}")
    print(f"Label distribution: {np.bincount([int(label) for label in y])}")
    
    return X, y

def create_simple_model(input_shape, num_classes):
    """Create a simpler, more robust model"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        # First layer
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Second layer
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        # Third layer
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def retrain_model():
    """Retrain the model with proper data balance and scaling"""
    print("\n🤖 Retraining Model with Fixed Issues")
    print("=" * 60)
    
    # Load balanced dataset
    X, y = load_balanced_dataset()
    
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
    
    # Check feature statistics after scaling
    print(f"\n📊 Feature Statistics After Scaling:")
    print(f"  Training mean: {np.mean(X_train_scaled, axis=0)}")
    print(f"  Training std: {np.std(X_train_scaled, axis=0)}")
    print(f"  Training range: {np.min(X_train_scaled):.6f} to {np.max(X_train_scaled):.6f}")
    
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
        epochs=50,
        batch_size=16,
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
    model_path = 'fixed_voice_model.h5'
    scaler_path = 'fixed_voice_scaler.pkl'
    label_binarizer_path = 'fixed_voice_label_binarizer.pkl'
    
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
    plt.savefig('fixed_model_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, scaler, lb

def test_fixed_model():
    """Test the fixed model"""
    print("\n🧪 Testing Fixed Model")
    print("=" * 50)
    
    try:
        # Load fixed model
        model = keras.models.load_model('fixed_voice_model.h5')
        scaler = joblib.load('fixed_voice_scaler.pkl')
        
        # Test on training data
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
                        
                        features = extract_simple_features(audio_data, sr)
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
        print(f"\n📊 Fixed model accuracy on training data: {accuracy*100:.2f}% ({correct}/{total})")
        
        if accuracy > 0.8:
            print("✅ Fixed model is working correctly!")
        else:
            print("⚠️ Fixed model still has issues")
            
    except Exception as e:
        print(f"❌ Error testing fixed model: {e}")

def main():
    """Main function"""
    print("🔧 Fixing Model Training Issues")
    print("=" * 60)
    
    # Check if voice samples exist
    if not os.path.exists('my_voice_samples'):
        print("❌ my_voice_samples directory not found!")
        return
    
    print("✅ Voice samples directory found")
    
    # Retrain model
    model, scaler, lb = retrain_model()
    
    if model is not None:
        # Test the fixed model
        test_fixed_model()
        
        print(f"\n🎉 Model fixing complete!")
        print(f"Next step: streamlit run app_fixed_voice.py")

if __name__ == "__main__":
    main() 