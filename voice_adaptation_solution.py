#!/usr/bin/env python3
"""
Voice Adaptation Solution - Fix real-world voice prediction issues
"""

import numpy as np
import librosa
import tensorflow as tf
import joblib
import os
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import noisereduce as nr

def record_voice_samples(num_samples_per_digit=10):
    """
    Record voice samples for each digit to adapt the model
    """
    print("🎤 Voice Sample Collection")
    print("=" * 50)
    print("We'll collect voice samples to adapt the model to your voice.")
    print("This will significantly improve prediction accuracy.")
    
    # Create directory for voice samples
    voice_dir = "my_voice_samples"
    os.makedirs(voice_dir, exist_ok=True)
    
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    for digit in digits:
        print(f"\n📝 Recording digit: {digit}")
        print(f"Please record {num_samples_per_digit} samples of '{digit}'")
        
        # Create digit directory
        digit_dir = os.path.join(voice_dir, digit)
        os.makedirs(digit_dir, exist_ok=True)
        
        for i in range(num_samples_per_digit):
            input(f"Press Enter to record sample {i+1}/{num_samples_per_digit} of '{digit}'...")
            
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
    print(f"Saved {num_samples_per_digit * 10} samples in '{voice_dir}' directory")

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
        
        # MFCC features (simple version)
        mfcc = librosa.feature.mfcc(y=audio_clean, sr=sample_rate, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        
        # Statistical features from MFCC
        features.extend(np.mean(mfcc, axis=1))  # Mean of each coefficient
        features.extend(np.std(mfcc, axis=1))   # Std of each coefficient
        
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

def create_voice_adapted_model():
    """
    Create a voice-adapted model using collected samples
    """
    print("\n🤖 Creating Voice-Adapted Model")
    print("=" * 50)
    
    # Load original model and scaler
    original_model = tf.keras.models.load_model('simple_working_model.h5')
    original_scaler = joblib.load('simple_model_scaler.pkl')
    
    # Load voice samples
    voice_dir = "my_voice_samples"
    if not os.path.exists(voice_dir):
        print("❌ No voice samples found! Please run record_voice_samples() first.")
        return None, None
    
    X_voice, y_voice = [], []
    
    for digit in range(10):
        digit_dir = os.path.join(voice_dir, str(digit))
        if os.path.exists(digit_dir):
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            
            for wav_file in wav_files:
                try:
                    file_path = os.path.join(digit_dir, wav_file)
                    audio_data, sr = librosa.load(file_path, sr=22050)
                    
                    features = extract_robust_features(audio_data, sr)
                    if features is not None:
                        X_voice.append(features)
                        y_voice.append(str(digit))
                        
                except Exception as e:
                    print(f"Error processing {wav_file}: {e}")
                    continue
    
    if len(X_voice) == 0:
        print("❌ No valid voice samples found!")
        return None, None
    
    X_voice = np.array(X_voice)
    y_voice = np.array(y_voice)
    
    print(f"Voice samples loaded: {X_voice.shape}")
    print(f"Voice labels: {np.unique(y_voice)}")
    
    # Create new scaler with voice data
    new_scaler = StandardScaler()
    X_voice_scaled = new_scaler.fit_transform(X_voice)
    
    # Fine-tune the model
    print("Fine-tuning model on voice samples...")
    
    # Convert labels to one-hot encoding
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y_voice_encoded = lb.fit_transform(y_voice)
    
    # Compile model for fine-tuning
    original_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune on voice data
    history = original_model.fit(
        X_voice_scaled, y_voice_encoded,
        epochs=20,
        batch_size=8,
        validation_split=0.2,
        verbose=1
    )
    
    # Save adapted model and scaler
    adapted_model_path = 'voice_adapted_working_model.h5'
    adapted_scaler_path = 'voice_adapted_scaler.pkl'
    
    original_model.save(adapted_model_path)
    joblib.dump(new_scaler, adapted_scaler_path)
    
    print(f"✅ Voice-adapted model saved: {adapted_model_path}")
    print(f"✅ Voice-adapted scaler saved: {adapted_scaler_path}")
    
    return original_model, new_scaler

def test_voice_adapted_model():
    """
    Test the voice-adapted model
    """
    print("\n🧪 Testing Voice-Adapted Model")
    print("=" * 50)
    
    # Load adapted model and scaler
    model = tf.keras.models.load_model('voice_adapted_working_model.h5')
    scaler = joblib.load('voice_adapted_scaler.pkl')
    
    print("Test the adapted model with your voice:")
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
            filename = f"test_{predicted_digit}_{timestamp}.wav"
            sf.write(filename, audio_data, 22050)
            print(f"💾 Saved as: {filename}")
            
        except KeyboardInterrupt:
            print("\n👋 Testing complete!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def create_improved_app():
    """
    Create an improved app with voice adaptation
    """
    app_content = '''import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import librosa
import os
import tempfile
import noisereduce as nr

# Load voice-adapted model
@st.cache_resource
def load_voice_adapted_model():
    try:
        model = tf.keras.models.load_model('voice_adapted_working_model.h5')
        scaler = joblib.load('voice_adapted_scaler.pkl')
        st.success("✅ Loaded voice-adapted model")
        return model, scaler
    except:
        # Fallback to original model
        model = tf.keras.models.load_model('simple_working_model.h5')
        scaler = joblib.load('simple_model_scaler.pkl')
        st.warning("⚠️ Using original model (voice adaptation not available)")
        return model, scaler

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
st.title("🎤 Voice-Adapted Digit Classifier")
st.markdown("**Improved accuracy with voice adaptation and noise reduction**")

# Load model
model, scaler = load_voice_adapted_model()

st.header("Audio Recording")
st.info("💡 **Tips:** Speak clearly, quiet environment, 1-2 seconds, say digit clearly")

# Audio input
audio_input = st.audio_input("Record a voice message")

if audio_input is not None:
    st.audio(audio_input, format="audio/wav")

    if st.button("🔮 Predict Digit", type="primary"):
        with st.spinner("Processing audio..."):
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
    **Model:** Voice-Adapted Model with Noise Reduction
    
    **Features:** 
    - MFCC (13 coefficients) - mean and std
    - Spectral features (centroid, rolloff)
    - Zero crossing rate
    - RMS energy
    - **Noise reduction** for better accuracy
    
    **Classes:** Digits 0-9
    
    **Improvements:**
    - Voice adaptation on your samples
    - Noise reduction preprocessing
    - Robust feature extraction
    
    **Performance:** Significantly improved for real voice input
    """)
'''
    
    with open('app_voice_adapted.py', 'w') as f:
        f.write(app_content)
    
    print("✅ Created app_voice_adapted.py")

def main():
    """
    Main function to run the complete voice adaptation solution
    """
    print("🎯 VOICE ADAPTATION SOLUTION")
    print("=" * 60)
    print("This will significantly improve prediction accuracy for your voice!")
    
    while True:
        print("\nChoose an option:")
        print("1. 📝 Record voice samples (required first)")
        print("2. 🤖 Create voice-adapted model")
        print("3. 🧪 Test voice-adapted model")
        print("4. 🌐 Create improved app")
        print("5. 🚀 Run complete solution")
        print("6. ❌ Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            num_samples = input("How many samples per digit? (default: 10): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 10
            record_voice_samples(num_samples)
            
        elif choice == '2':
            create_voice_adapted_model()
            
        elif choice == '3':
            test_voice_adapted_model()
            
        elif choice == '4':
            create_improved_app()
            
        elif choice == '5':
            print("\n🚀 Running complete solution...")
            record_voice_samples(10)
            create_voice_adapted_model()
            create_improved_app()
            print("\n🎉 Complete solution ready!")
            print("Run: streamlit run app_voice_adapted.py")
            
        elif choice == '6':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main() 