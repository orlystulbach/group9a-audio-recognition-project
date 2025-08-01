#!/usr/bin/env python3
"""
Voice Adaptive Training - Train a model with data augmentation and voice samples
"""

import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from datetime import datetime
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
VOICE_SAMPLES_PER_DIGIT = 15  # Number of voice samples to collect per digit

def record_voice_samples():
    """
    Record voice samples for each digit
    """
    print("🎤 Voice Sample Collection")
    print("=" * 60)
    print("We'll collect voice samples to adapt the model to your voice.")
    print("This will significantly improve prediction accuracy.")
    
    # Create directory for voice samples
    os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
    
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    for digit in digits:
        print(f"\n📝 Recording digit: {digit}")
        print(f"Please record {VOICE_SAMPLES_PER_DIGIT} samples of '{digit}'")
        print("💡 Tips: Speak clearly, vary your tone slightly, use different speeds")
        
        # Create digit directory
        digit_dir = os.path.join(VOICE_SAMPLES_DIR, digit)
        os.makedirs(digit_dir, exist_ok=True)
        
        for i in range(VOICE_SAMPLES_PER_DIGIT):
            input(f"Press Enter to record sample {i+1}/{VOICE_SAMPLES_PER_DIGIT} of '{digit}'...")
            
            try:
                # Record audio
                print("🎤 Recording... Speak clearly!")
                audio = sd.rec(int(2 * 22050), samplerate=22050, channels=1)
                sd.wait()
                audio_data = audio.flatten()
                
                # Check audio quality
                rms = np.sqrt(np.mean(audio_data**2))
                print(f"Audio RMS: {rms:.6f}")
                
                if rms < 0.01:
                    print("⚠️ Too quiet! Please speak louder.")
                    continue
                
                # Save audio
                filename = f"{digit}_{i+1}.wav"
                filepath = os.path.join(digit_dir, filename)
                sf.write(filepath, audio_data, 22050)
                print(f"✅ Saved: {filename}")
                
            except Exception as e:
                print(f"❌ Error recording: {e}")
                continue
    
    print(f"\n🎉 Voice sample collection complete!")
    print(f"Saved {VOICE_SAMPLES_PER_DIGIT * 10} samples in '{VOICE_SAMPLES_DIR}' directory")

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

def train_adaptive_model():
    """
    Train the adaptive model with combined dataset
    """
    print("\n🤖 Training Adaptive Model")
    print("=" * 60)
    
    # Load combined dataset
    X, y = load_combined_dataset()
    
    if len(X) == 0:
        print("❌ No data found! Please collect voice samples first.")
        return None, None
    
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
    
    return model, scaler, lb

def test_adaptive_model():
    """
    Test the adaptive model
    """
    print("\n🧪 Testing Adaptive Model")
    print("=" * 60)
    
    # Load model and scaler
    model = keras.models.load_model('voice_adaptive_model.h5')
    scaler = joblib.load('voice_adaptive_scaler.pkl')
    
    print("Test the adaptive model with your voice:")
    print("1. Speak a digit clearly")
    print("2. Check the prediction accuracy")
    print("3. Press Ctrl+C to exit")
    
    while True:
        try:
            input("\nPress Enter to record a test digit...")
            
            # Record audio
            print("🎤 Recording...")
            audio = sd.rec(int(2 * 22050), samplerate=22050, channels=1)
            sd.wait()
            audio_data = audio.flatten()
            
            # Check audio quality
            rms = np.sqrt(np.mean(audio_data**2))
            print(f"Audio RMS: {rms:.6f}")
            
            if rms < 0.01:
                print("⚠️ Too quiet! Please speak louder.")
                continue
            
            # Extract features with noise reduction
            features = extract_robust_features(audio_data)
            if features is None:
                continue
            
            # Make prediction
            features_reshaped = features.reshape(1, -1)
            features_scaled = scaler.transform(features_reshaped)
            prediction = model.predict(features_scaled, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            print(f"🎯 Predicted: {predicted_digit}")
            print(f"📈 Confidence: {confidence:.3f}")
            
            # Save test audio
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_adaptive_{predicted_digit}_{timestamp}.wav"
            sf.write(filename, audio_data, 22050)
            print(f"💾 Saved as: {filename}")
            
        except KeyboardInterrupt:
            print("\n👋 Testing complete!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def create_adaptive_app():
    """
    Create an app for the adaptive model
    """
    app_content = '''import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import librosa
import os
import tempfile
import noisereduce as nr

# Load adaptive model
@st.cache_resource
def load_adaptive_model():
    try:
        model = tf.keras.models.load_model('voice_adaptive_model.h5')
        scaler = joblib.load('voice_adaptive_scaler.pkl')
        lb = joblib.load('voice_adaptive_label_binarizer.pkl')
        st.success("✅ Loaded voice-adaptive model")
        return model, scaler, lb
    except Exception as e:
        st.error(f"❌ Error loading adaptive model: {e}")
        return None, None, None

def extract_robust_features(audio_data, sample_rate=22050):
    """Extract features with noise reduction"""
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
        mfcc = librosa.feature.mfcc(y=audio_clean, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
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
        st.error(f"Error extracting features: {e}")
        return None

def process_audio_file(audio_bytes):
    """Process uploaded audio file"""
    try:
        audio_file = audio_bytes.read()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file)
            tmp_file_path = tmp_file.name
        
        audio_data, sample_rate = librosa.load(tmp_file_path, sr=22050)
        os.unlink(tmp_file_path)
        
        return audio_data, sample_rate
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None, None

def predict_digit(audio_features, model, scaler):
    """Make prediction"""
    if model is None or scaler is None:
        return None, None
    
    try:
        features_scaled = scaler.transform(audio_features)
        prediction = model.predict(features_scaled, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_digit, confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Streamlit UI
st.title("🎤 Voice-Adaptive Digit Classifier")
st.markdown("**Trained on your voice with data augmentation for maximum accuracy**")

# Load model
model, scaler, lb = load_adaptive_model()

if model is None:
    st.error("❌ Cannot load model. Please train the adaptive model first.")
    st.stop()

st.header("Audio Recording")
st.info("💡 **Tips:** Speak clearly, quiet environment, 1-2 seconds, say digit clearly")

# Audio input
audio_input = st.audio_input("Record a voice message")

if audio_input is not None:
    st.audio(audio_input, format="audio/wav")

    if st.button("🔮 Predict Digit", type="primary"):
        with st.spinner("Processing audio with noise reduction..."):
            audio_data, sample_rate = process_audio_file(audio_input)

            if audio_data is not None:
                # Check audio quality
                rms = np.sqrt(np.mean(audio_data**2))
                st.info(f"📊 Audio RMS: {rms:.6f}")

                if rms < 0.01:
                    st.warning("🔇 Audio too quiet. Speak louder!")
                elif rms > 0.5:
                    st.warning("🔊 Audio too loud. Speak quieter!")

                # Extract features with noise reduction
                features = extract_robust_features(audio_data, sample_rate)
                
                if features is not None:
                    features_reshaped = features.reshape(1, -1)
                    predicted_digit, confidence = predict_digit(features_reshaped, model, scaler)

                    if predicted_digit is not None:
                        st.success("✅ Prediction complete!")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🎯 Predicted Digit", predicted_digit)
                        with col2:
                            st.metric("📈 Confidence", f"{confidence*100:.2f}%")

                        st.header(f"🎤 You said: **{predicted_digit}**")
                        st.progress(float(confidence))

                        if confidence < 0.5:
                            st.error("❌ Very low confidence. Try again.")
                        elif confidence < 0.7:
                            st.warning("⚠️ Low confidence. Speak more clearly.")
                        elif confidence > 0.9:
                            st.success("🎉 High confidence!")
                        else:
                            st.info("👍 Good prediction!")
                    else:
                        st.error("❌ Failed to make prediction.")
                else:
                    st.error("❌ Failed to process audio features.")
            else:
                st.error("❌ Failed to process audio data.")

# Instructions
st.header("📋 Instructions")
st.markdown("""
1. **🎤 Click microphone** to start/stop recording
2. **🔢 Speak a digit** (0-9) clearly
3. **🔮 Click 'Predict Digit'** for prediction
4. **🎯 Check confidence** - higher is better

**🎯 Tips for Best Accuracy:**
- Speak clearly and at normal pace
- Record in quiet environment
- Speak for about 1-2 seconds
- Say digit clearly (e.g., "five" not "fifth")
- Avoid background noise
""")

# Model Info
with st.expander("🤖 Model Information"):
    st.markdown("""
    **Model:** Voice-Adaptive Model with Data Augmentation
    
    **Features:** 
    - MFCC (13 coefficients) - mean and std
    - Spectral features (centroid, rolloff)
    - Zero crossing rate
    - RMS energy
    - **Noise reduction** for better accuracy
    
    **Training Data:**
    - Original dataset samples
    - Your voice samples (15 per digit)
    - Data augmentation (pitch, time, noise, volume)
    
    **Classes:** Digits 0-9
    
    **Improvements:**
    - Voice adaptation on your samples
    - Data augmentation for robustness
    - Noise reduction preprocessing
    - Advanced neural network architecture
    
    **Performance:** Optimized for your voice
    """)
'''
    
    with open('app_voice_adaptive.py', 'w') as f:
        f.write(app_content)
    
    print("✅ Created app_voice_adaptive.py")

def main():
    """
    Main function to run the complete voice adaptive training
    """
    print("🎯 VOICE ADAPTIVE TRAINING")
    print("=" * 60)
    print("This will create a highly accurate model for your voice!")
    
    while True:
        print("\nChoose an option:")
        print("1. 📝 Record voice samples (required first)")
        print("2. 🤖 Train adaptive model")
        print("3. 🧪 Test adaptive model")
        print("4. 🌐 Create adaptive app")
        print("5. 🚀 Run complete solution")
        print("6. ❌ Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            record_voice_samples()
            
        elif choice == '2':
            train_adaptive_model()
            
        elif choice == '3':
            test_adaptive_model()
            
        elif choice == '4':
            create_adaptive_app()
            
        elif choice == '5':
            print("\n🚀 Running complete solution...")
            record_voice_samples()
            train_adaptive_model()
            create_adaptive_app()
            print("\n🎉 Complete solution ready!")
            print("Run: streamlit run app_voice_adaptive.py")
            
        elif choice == '6':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main() 