import streamlit as st
import numpy as np
import pandas as pd
# from tensorflow.keras.models import load_model
import tensorflow as tf
from keras.models import load_model
import librosa
import os
import tempfile

# Load model
@st.cache_resource
def load_model():
  try:
    # Try to load the improved voice-adapted model first
    model = tf.keras.models.load_model('improved_voice_adapted_model.h5')
    return model
  except Exception as e:
    try:
      # Fallback to wide model
      model = tf.keras.models.load_model('wide_digit_classifier_model.h5')
      return model
    except Exception as e2:
      st.error(f"Error loading models: {e2}")
      return None

# Preprocess audio for prediction - matching training code exactly
def extract_features(audio_data, sample_rate=22050):
    """
    Extract features from audio data - matches training code exactly
    """
    try:
        # Normalize audio (matching training code)
        audio_data = librosa.util.normalize(audio_data)
        
        features = []
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Statistical features from MFCC
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.mean(mfcc_delta2, axis=1))
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, hop_length=512)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate, hop_length=512)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate, hop_length=512)[0]
        
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, hop_length=512)
        features.extend(np.mean(chroma, axis=1))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=512)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # RMS energy
        rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        return np.array(features)
        
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def preprocess_audio(audio_data, sample_rate=22050, duration=2.0):
    """
    Preprocess audio data to match the format expected by the model
    """
    try:
        # Ensure audio is the right length (pad or trim to 2 seconds - matching training)
        target_length = int(sample_rate * duration)
        
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
        
        # Extract features using the same method as your training code
        features = extract_features(audio_data, sample_rate)
        
        if features is not None:
            # Reshape for model input (add batch dimension)
            features = features.reshape(1, -1)
        
        return features
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None

def predict_digit(audio_features):
    """
    Make prediction using the loaded model
    """
    model = load_model()
    if model is None:
        return None, None
    
    try:
        prediction = model.predict(audio_features, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        return predicted_digit, confidence
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def process_audio_file(audio_bytes):
    """
    Process uploaded audio file and extract features
    """
    try:
        # Read bytes from the UploadedFile object
        audio_file = audio_bytes.read()
        
        # Reset the file pointer for potential re-use
        # audio_file.seek(0)

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

# Streamlit UI
st.title("Classifying Spoken Digits")
st.markdown('Record yourself saying a digit (0-9) and the model will predict which digit you spoke!')

st.header("Audio Recording")
st. info("💡 Tip: Speak clearly for 1-2 seconds")

# Audio input!
audio_input = st.audio_input("Record a voice message")

if audio_input is not None:
    st.audio(audio_input, format="audio/wav")

    if st.button("Predict Digit", type="primary"):
      with st.spinner("Processing audio..."):
          # Process the audio
          audio_data, sample_rate= process_audio_file(audio_input)

          if audio_data is not None:
              # Check audio quality (matching training code)
              rms = np.sqrt(np.mean(audio_data**2))

              # Display audio quality info
              st.info(f"Audio RMS: {rms:.6f}")

              if rms < 0.001:
                  st.warning("Audio seems very quiet. Please try speaking louder or closer to the mic.")

              # Preprocess for model (using 2-second duration like training)
              features = preprocess_audio(audio_data, sample_rate, duration=2.0)

              if features is not None:
                  # Make prediction
                  predicted_digit, confidence = predict_digit(features)

                  if predicted_digit is not None:
                      st.success("Prediction complete!")

                      # Display results
                      col1, col2 = st.columns(2)

                      with col1:
                          st.metric("Predicted Digit", predicted_digit)
                      
                      with col2:
                          st.metric("Confidence", f"{confidence*100:.2f}%")
                      
                      # Visual representation
                      st.header(f"You said: **{predicted_digit}**")

                      st.progress(float(confidence))

                      if confidence < 0.7:
                          st.warning("⚠️ Low confidence prediction. Try speaking more clearly or closer to the microphone.")
                      elif confidence > 0.9:
                          st.success("High confidence prediction!")
                  else:
                      st.error("Failed to make prediction. Please try again.")
          
              else:
                  st.error("Failed to process audio features. Please try again.")
          
          else:
              st.error("Failed to process audio data. Please try again.")


# Instructions
st.header("Instructions")
st.markdown("""
1. **Click the microphone button** above to start and stop recording
2. **Speak a single digit** (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) clearly
3. **Click 'Predict Digit'** to see the model's prediction
4. The model will display the predicted digit and confidence level.
            
**Tips for Better Accuracy:**
- Speak clearly and at a normal pace.
- Record yourself in a quiet environment.
- Speak for about 1-2 seconds.
""")

# Model Info
with st.expander("Model Information"):
    st.markdown("""
    **Model:** improved_voice_adapted_model.h5 (or fallback models)
    
    **Features:** 
    - MFCC (13 coefficients) + deltas + delta-deltas
    - Spectral features (centroid, rolloff, bandwidth)
    - Chroma features (12 coefficients)
    - Zero crossing rate
    - RMS energy
    - **Total: ~54 features per audio sample**
    
    **Classes:** Digits 0-9
    
    **Training:** Fine-tuned on your voice samples (10 per digit)
    
    **Input:** 2-second audio recordings at 22050 Hz sample rate
    """)

# Troubleshooting
with st.expander("Troubleshooting"):
    st.markdown("""
    **If you encounter issues:**
    
    - Make sure the model file `improved_voice_adaptation_model.h5` is in the same directory
    - Ensure you have the required libraries installed: `librosa`, `tensorflow`, `numpy`
    - Check that your microphone is working and permissions are granted
    - Try recording in a quieter environment
    - Make sure you're speaking a single digit clearly
    """)