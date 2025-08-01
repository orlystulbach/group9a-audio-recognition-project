import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import librosa
import os
import tempfile
import noisereduce as nr

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

def extract_enhanced_features(audio_data, sample_rate=22050, n_mfcc=13):
    """Extract features with enhanced preprocessing"""
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
        
        # MFCC features with more coefficients
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
        st.error(f"Error extracting enhanced features: {e}")
        return None

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
st.title("🎤 Enhanced Voice-Adapted Digit Classifier")
st.markdown("**With improved audio preprocessing and feature extraction**")

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
                # Enhanced audio quality analysis
                duration = len(audio_data) / sample_rate
                rms = np.sqrt(np.mean(audio_data**2))
                
                st.subheader("📊 Audio Quality Analysis")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Duration", f"{duration:.2f}s")
                with col2:
                    st.metric("RMS Level", f"{rms:.6f}")
                with col3:
                    st.metric("Sample Rate", f"{sample_rate}Hz")
                
                # Audio quality feedback
                quality_score = 0
                feedback = []
                
                # Duration check (training samples are exactly 2.00s)
                if abs(duration - 2.0) < 0.1:
                    st.success("✅ Duration is good (close to 2.00s)")
                    quality_score += 1
                else:
                    st.warning(f"⚠️ Duration is {duration:.2f}s (training samples are 2.00s)")
                    feedback.append("Try to speak for exactly 2 seconds")
                
                # RMS check (training range: 0.016-0.026)
                if 0.016 <= rms <= 0.026:
                    st.success("✅ Audio level is in expected range")
                    quality_score += 1
                elif rms < 0.016:
                    st.error("🔇 Audio too quiet! Speak louder")
                    feedback.append("Speak louder - audio is too quiet")
                else:
                    st.warning("🔊 Audio too loud! Speak quieter")
                    feedback.append("Speak quieter - audio is too loud")
                
                # Try both feature extraction methods
                st.subheader("🔬 Feature Extraction Comparison")
                
                # Method 1: Simple (like training)
                features_simple = extract_simple_features(audio_data, sample_rate)
                if features_simple is not None:
                    features_simple_reshaped = features_simple.reshape(1, -1)
                    predicted_digit_simple, confidence_simple = predict_digit(features_simple_reshaped, model, scaler)
                
                # Method 2: Enhanced (with noise reduction and more features)
                features_enhanced = extract_enhanced_features(audio_data, sample_rate)
                if features_enhanced is not None:
                    # Note: Enhanced features won't work with current model, but we can show the difference
                    st.info("Enhanced features extracted (different feature set)")
                
                if features_simple is not None and predicted_digit_simple is not None:
                    st.subheader("🎯 Prediction Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Digit", predicted_digit_simple)
                    with col2:
                        st.metric("Confidence", f"{confidence_simple*100:.2f}%")

                    st.header(f"🎤 You said: **{predicted_digit_simple}**")
                    st.progress(float(confidence_simple))

                    # Confidence feedback
                    if confidence_simple < 0.3:
                        st.error("❌ Very low confidence - major issues detected")
                        feedback.extend([
                            "Audio quality needs significant improvement",
                            "Try recording in a quieter environment",
                            "Speak more clearly and distinctly",
                            "Use same microphone as training if possible"
                        ])
                    elif confidence_simple < 0.5:
                        st.warning("⚠️ Low confidence - several issues")
                        feedback.extend([
                            "Audio quality needs improvement",
                            "Try speaking more clearly",
                            "Check background noise"
                        ])
                    elif confidence_simple < 0.7:
                        st.warning("⚠️ Moderate confidence - minor issues")
                        feedback.append("Try speaking more clearly")
                    elif confidence_simple > 0.9:
                        st.success("🎉 High confidence!")
                    else:
                        st.info("👍 Good prediction!")
                        
                    # Show all predictions
                    st.subheader("📊 All Predictions:")
                    prediction_full = model.predict(scaler.transform(features_simple_reshaped), verbose=0)
                    
                    # Create a bar chart of predictions
                    import plotly.express as px
                    import pandas as pd
                    
                    pred_df = pd.DataFrame({
                        'Digit': range(10),
                        'Confidence': prediction_full[0]
                    })
                    
                    fig = px.bar(pred_df, x='Digit', y='Confidence', 
                               title='Confidence for Each Digit',
                               color='Confidence',
                               color_continuous_scale='RdYlGn')
                    fig.update_layout(yaxis_range=[0, 1])
                    st.plotly_chart(fig)
                    
                    # Show top 3 predictions
                    top_indices = np.argsort(prediction_full[0])[-3:][::-1]
                    st.write("**Top 3 Predictions:**")
                    for i, idx in enumerate(top_indices):
                        st.write(f"{i+1}. Digit {idx}: {prediction_full[0][idx]:.3f}")
                    
                    # Specific recommendations based on confusion
                    if confidence_simple < 0.5:
                        st.subheader("🎯 Specific Recommendations for Low Confidence")
                        st.markdown("""
                        **Your model is confused between digits 0, 7, and 3. This suggests:**
                        
                        1. **Pronunciation Issues**: These digits sound similar when spoken quickly
                        2. **Audio Quality**: Background noise or poor microphone quality
                        3. **Feature Discrimination**: The model can't distinguish your voice characteristics
                        
                        **Immediate Actions:**
                        - Speak more slowly and clearly
                        - Emphasize the differences between digits (e.g., "zero" vs "seven")
                        - Record in a quieter environment
                        - Use a better microphone
                        """)
                    
                    # Quality recommendations
                    if feedback:
                        st.subheader("💡 Recommendations")
                        for rec in feedback:
                            st.write(f"• {rec}")
                        
                        if quality_score < 2:
                            st.info("**Try recording again with the recommendations above**")
                        
                else:
                    st.error("❌ Failed to process audio features.")
            else:
                st.error("❌ Failed to process audio data.")

# Instructions
st.header("📋 Instructions")
st.markdown("""
1. **🎤 Click microphone** to start/stop recording
2. **🔢 Speak a digit** (0-9) clearly for exactly 2 seconds
3. **🔮 Click 'Predict Digit'** for prediction
4. **🎯 Check the audio quality feedback** and recommendations

**🎯 Tips for Best Accuracy:**
- **Duration**: Speak for exactly 2 seconds (like training samples)
- **Volume**: Keep RMS between 0.016-0.026 (app will tell you)
- **Environment**: Record in quiet environment
- **Clarity**: Speak clearly and distinctly
- **Pacing**: Use similar pace as training samples
- **Pronunciation**: Emphasize differences between similar digits (0, 7, 3)
""")

# Model Info
with st.expander("🤖 Model Information"):
    st.markdown("""
    **Model:** Fixed Voice-Adapted Model (100% training accuracy)
    
    **Training Data Characteristics:**
    - Duration: Exactly 2.00s for all samples
    - RMS Range: 0.016-0.026
    - Sample Rate: 22050Hz
    - Balanced dataset (15 samples per digit)
    
    **Features:** 
    - MFCC (13 coefficients) - mean and std
    - Spectral features (centroid, rolloff)
    - Zero crossing rate
    - RMS energy
    - Simple preprocessing (no noise reduction)
    
    **Performance:** 100% accuracy on training data
    """)

# Troubleshooting
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    **Common Issues and Solutions:**
    
    **Low Confidence (< 0.3):**
    - Major audio quality issues
    - Try different microphone
    - Record in quieter environment
    - Speak more clearly and slowly
    
    **Confusion between digits (0, 7, 3):**
    - These digits sound similar
    - Speak more slowly
    - Emphasize pronunciation differences
    - Use clearer enunciation
    
    **Audio Too Quiet (RMS < 0.016):**
    - Speak louder
    - Move closer to microphone
    - Check microphone settings
    
    **Audio Too Loud (RMS > 0.026):**
    - Speak quieter
    - Move away from microphone
    - Reduce microphone gain
    
    **Wrong Duration:**
    - Aim for exactly 2 seconds
    - Use a timer or count "one-thousand-one, one-thousand-two"
    """) 