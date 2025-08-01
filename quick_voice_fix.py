#!/usr/bin/env python3
"""
Quick Voice Fix - Improve prediction accuracy without voice samples
"""

import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import librosa
import os
import tempfile
import noisereduce as nr

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = tf.keras.models.load_model('simple_working_model.h5')
        scaler = joblib.load('simple_model_scaler.pkl')
        st.success("✅ Loaded working model (98.67% accuracy)")
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def extract_robust_features(audio_data, sample_rate=22050):
    """
    Extract robust features with noise reduction and better preprocessing
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
        mfcc = librosa.feature.mfcc(y=audio_clean, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
        
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
        st.error(f"Error extracting features: {e}")
        return None

def process_audio_file(audio_bytes):
    """
    Process uploaded audio file and extract features
    """
    try:
        # Read bytes from the UploadedFile object
        audio_file = audio_bytes.read()
        
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file)
            tmp_file_path = tmp_file.name
        
        # Load audio using librosa (matching your training parameters)
        audio_data, sample_rate = librosa.load(tmp_file_path, sr=22050)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return audio_data, sample_rate
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None, None

def predict_digit(audio_features, model, scaler):
    """
    Make prediction using the loaded model
    """
    if model is None or scaler is None:
        return None, None
    
    try:
        # Scale features using the fitted scaler
        features_scaled = scaler.transform(audio_features)
        
        # Make prediction
        prediction = model.predict(features_scaled, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_digit, confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

# Streamlit UI
st.title("🎤 Improved Digit Classifier")
st.markdown('**Enhanced with noise reduction and better preprocessing**')

# Load model status
model, scaler = load_model_and_scaler()

st.header("Audio Recording")
st.info("💡 **Tips for best accuracy:**\n- Speak clearly and at normal pace\n- Record in a quiet environment\n- Speak for about 1-2 seconds\n- Say the digit clearly (e.g., 'five' not 'fifth')")

# Audio input!
audio_input = st.audio_input("Record a voice message")

if audio_input is not None:
    st.audio(audio_input, format="audio/wav")

    if st.button("🔮 Predict Digit", type="primary"):
      with st.spinner("Processing audio with noise reduction..."):
          # Process the audio
          audio_data, sample_rate = process_audio_file(audio_input)

          if audio_data is not None:
              # Check audio quality
              rms = np.sqrt(np.mean(audio_data**2))

              # Display audio quality info
              st.info(f"📊 Audio RMS: {rms:.6f}")

              if rms < 0.01:
                  st.warning("🔇 Audio seems very quiet. Please try speaking louder or closer to the mic.")
              elif rms > 0.5:
                  st.warning("🔊 Audio seems very loud. Please try speaking more quietly.")

              # Extract features with noise reduction
              features = extract_robust_features(audio_data, sample_rate)

              if features is not None:
                  # Make prediction
                  features_reshaped = features.reshape(1, -1)
                  predicted_digit, confidence = predict_digit(features_reshaped, model, scaler)

                  if predicted_digit is not None:
                      st.success("✅ Prediction complete!")

                      # Display results
                      col1, col2 = st.columns(2)

                      with col1:
                          st.metric("🎯 Predicted Digit", predicted_digit)
                      
                      with col2:
                          st.metric("📈 Confidence", f"{confidence*100:.2f}%")
                      
                      # Visual representation
                      st.header(f"🎤 You said: **{predicted_digit}**")

                      st.progress(float(confidence))

                      if confidence < 0.5:
                          st.error("❌ Very low confidence. Please try again with clearer speech.")
                      elif confidence < 0.7:
                          st.warning("⚠️ Low confidence prediction. Try speaking more clearly.")
                      elif confidence > 0.9:
                          st.success("🎉 High confidence prediction!")
                      else:
                          st.info("👍 Good prediction!")
                  else:
                      st.error("❌ Failed to make prediction. Please try again.")
          
              else:
                  st.error("❌ Failed to process audio features. Please try again.")
          
          else:
              st.error("❌ Failed to process audio data. Please try again.")

# Instructions
st.header("📋 Instructions")
st.markdown("""
1. **🎤 Click the microphone button** above to start and stop recording
2. **🔢 Speak a single digit** (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) clearly
3. **🔮 Click 'Predict Digit'** to see the model's prediction
4. The model will display the predicted digit and confidence level.
            
**🎯 Tips for Best Accuracy:**
- Speak clearly and at a normal pace
- Record yourself in a quiet environment
- Speak for about 1-2 seconds
- Say the digit clearly (e.g., "five" not "fifth")
- Avoid background noise
- Use consistent microphone distance
""")

# Model Info
with st.expander("🤖 Model Information"):
    st.markdown("""
    **Model:** Simple Working Model (98.67% accuracy)
    
    **Improvements:** 
    - **Noise reduction** preprocessing
    - **Robust feature extraction**
    - **Better audio normalization**
    
    **Features:** 
    - MFCC (13 coefficients) - mean and std
    - Spectral features (centroid, rolloff)
    - Zero crossing rate
    - RMS energy
    - **Total: 34 features per audio sample**
    
    **Classes:** Digits 0-9
    
    **Training:** 30,000 samples (3,000 per digit)
    
    **Input:** 2-second audio recordings at 22050 Hz sample rate
    
    **Performance:** 98.67% test accuracy
    """)

# Troubleshooting
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    **If you encounter issues:**
    
    - Make sure the model files are in the same directory
    - Ensure you have the required libraries installed
    - Check that your microphone is working and permissions are granted
    - Try recording in a quieter environment
    - Make sure you're speaking a single digit clearly
    - Try different digits if one isn't working well
    - Check audio RMS levels (should be 0.01-0.5)
    """)

# Next Steps
with st.expander("🚀 Next Steps for Better Accuracy"):
    st.markdown("""
    **For even better accuracy:**
    
    1. **Voice Adaptation:** Collect 10 samples per digit and fine-tune the model
    2. **Better Environment:** Use a high-quality microphone in a sound-treated room
    3. **Consistent Recording:** Use the same microphone and distance every time
    4. **Practice:** Record multiple samples and use the best ones
    5. **Advanced Features:** Add pitch, formant, and spectral contrast features
    
    **Current improvements:**
    - ✅ Noise reduction
    - ✅ Better preprocessing
    - ✅ Robust feature extraction
    - ✅ Audio quality monitoring
    """) 