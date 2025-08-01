import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import librosa
import os
import tempfile

# Load fixed model
@st.cache_resource
def load_fixed_model():
    try:
        model = tf.keras.models.load_model('fixed_voice_model.h5')
        scaler = joblib.load('fixed_voice_scaler.pkl')
        st.success("✅ Loaded fixed voice model")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading fixed model: {e}")
        return None, None

def extract_simple_features(audio_data, sample_rate=22050, n_mfcc=13):
    """Extract features EXACTLY like training"""
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
st.title("🎤 Fixed Voice-Adapted Digit Classifier")
st.markdown("**Retrained with proper data balance and scaling**")

# Load model
model, scaler = load_fixed_model()

if model is None:
    st.error("❌ Cannot load fixed model.")
    st.stop()

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

                # Extract features
                features = extract_simple_features(audio_data, sample_rate)
                
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
                            
                        # Show all predictions
                        st.subheader("📊 All Predictions:")
                        prediction_full = model.predict(scaler.transform(features_reshaped), verbose=0)
                        for i, conf in enumerate(prediction_full[0]):
                            st.write(f"Digit {i}: {conf:.3f}")
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
    **Model:** Fixed Voice-Adapted Model
    
    **Fixes Applied:**
    - ✅ Proper data balance (equal samples per digit)
    - ✅ Correct feature scaling
    - ✅ Simpler, more robust architecture
    - ✅ No noise reduction (matches training)
    
    **Features:** 
    - MFCC (13 coefficients) - mean and std
    - Spectral features (centroid, rolloff)
    - Zero crossing rate
    - RMS energy
    - **Simple preprocessing** (no noise reduction)
    
    **Training Data:**
    - Your voice samples (15 per digit)
    - Balanced dataset
    - Proper feature scaling
    
    **Classes:** Digits 0-9
    
    **Performance:** Should be much better than previous model
    """)

# Troubleshooting
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    **If you still have issues:**
    
    - Check that the fixed model files exist
    - Ensure your microphone is working properly
    - Try recording in a quieter environment
    - Make sure you're speaking clearly and at normal pace
    - Check the "All Predictions" section to see confidence levels
    - If confidence is low, try speaking more clearly or closer to the mic
    
    **Model Files:**
    - fixed_voice_model.h5
    - fixed_voice_scaler.pkl
    - fixed_voice_label_binarizer.pkl
    """) 