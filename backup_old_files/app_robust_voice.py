import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import librosa
import os
import tempfile
import noisereduce as nr

# Load robust model
@st.cache_resource
def load_robust_model():
    try:
        model = tf.keras.models.load_model('robust_voice_model.h5')
        scaler = joblib.load('robust_voice_scaler.pkl')
        lb = joblib.load('robust_voice_label_binarizer.pkl')
        st.success("✅ Loaded robust voice model (99.76% test accuracy)")
        return model, scaler, lb
    except Exception as e:
        st.error(f"❌ Error loading robust model: {e}")
        return None, None, None

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
        st.error(f"Error extracting robust features: {e}")
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
st.title("🎤 Robust Voice-Adapted Digit Classifier")
st.markdown("**99.76% Test Accuracy - Enhanced with Noise Reduction & Advanced Features**")

# Load model
model, scaler, lb = load_robust_model()

if model is None:
    st.error("❌ Cannot load robust model.")
    st.stop()

st.header("Audio Recording")
st.info("💡 **Tips:** Speak clearly, quiet environment, 1-2 seconds, say digit clearly")

# Audio input
audio_input = st.audio_input("Record a voice message")

if audio_input is not None:
    st.audio(audio_input, format="audio/wav")

    if st.button("🔮 Predict Digit", type="primary"):
        with st.spinner("Processing audio with robust features..."):
            audio_data, sample_rate = process_audio_file(audio_input)

            if audio_data is not None:
                # Audio quality analysis
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
                
                # Duration check
                if abs(duration - 2.0) < 0.1:
                    st.success("✅ Duration is good (close to 2.00s)")
                    quality_score += 1
                else:
                    st.warning(f"⚠️ Duration is {duration:.2f}s (training samples are 2.00s)")
                    feedback.append("Try to speak for exactly 2 seconds")
                
                # RMS check (more flexible for robust model)
                if 0.01 <= rms <= 0.05:
                    st.success("✅ Audio level is in good range")
                    quality_score += 1
                elif rms < 0.01:
                    st.warning("🔇 Audio a bit quiet, but robust model should handle it")
                    feedback.append("Speak a bit louder for best results")
                else:
                    st.info("🔊 Audio level is fine")
                
                # Extract robust features
                features = extract_robust_features(audio_data, sample_rate)
                
                if features is not None:
                    features_reshaped = features.reshape(1, -1)
                    predicted_digit, confidence = predict_digit(features_reshaped, model, scaler)

                    if predicted_digit is not None:
                        st.subheader("🎯 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Predicted Digit", predicted_digit)
                        with col2:
                            st.metric("Confidence", f"{confidence*100:.2f}%")

                        st.header(f"🎤 You said: **{predicted_digit}**")
                        st.progress(float(confidence))

                        # Confidence feedback
                        if confidence < 0.5:
                            st.error("❌ Low confidence - try speaking more clearly")
                            feedback.append("Speak more clearly and distinctly")
                        elif confidence < 0.7:
                            st.warning("⚠️ Moderate confidence - good but can improve")
                            feedback.append("Try speaking a bit more clearly")
                        elif confidence < 0.9:
                            st.info("👍 Good confidence!")
                        else:
                            st.success("🎉 Excellent confidence!")
                            
                        # Show all predictions
                        st.subheader("📊 All Predictions:")
                        prediction_full = model.predict(scaler.transform(features_reshaped), verbose=0)
                        
                        # Create a bar chart of predictions
                        import plotly.express as px
                        import pandas as pd
                        
                        pred_df = pd.DataFrame({
                            'Digit': range(10),
                            'Confidence': prediction_full[0]
                        })
                        
                        fig = px.bar(pred_df, x='Digit', y='Confidence', 
                                   title='Confidence for Each Digit (Robust Model)',
                                   color='Confidence',
                                   color_continuous_scale='RdYlGn')
                        fig.update_layout(yaxis_range=[0, 1])
                        st.plotly_chart(fig)
                        
                        # Show top 3 predictions
                        top_indices = np.argsort(prediction_full[0])[-3:][::-1]
                        st.write("**Top 3 Predictions:**")
                        for i, idx in enumerate(top_indices):
                            st.write(f"{i+1}. Digit {idx}: {prediction_full[0][idx]:.3f}")
                        
                        # Model comparison
                        st.subheader("🔄 Model Comparison")
                        st.markdown("""
                        **This robust model includes:**
                        - ✅ **Noise reduction** - handles background noise
                        - ✅ **112 features** (vs 34 original) - much richer representation
                        - ✅ **Data augmentation** - trained on 2,044 samples (vs 150)
                        - ✅ **Advanced preprocessing** - pre-emphasis, spectral features
                        - ✅ **99.76% test accuracy** - significantly improved performance
                        """)
                        
                        # Quality recommendations
                        if feedback:
                            st.subheader("💡 Recommendations")
                            for rec in feedback:
                                st.write(f"• {rec}")
                        
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
2. **🔢 Speak a digit** (0-9) clearly for about 2 seconds
3. **🔮 Click 'Predict Digit'** for prediction
4. **🎯 Check the confidence** - should be much higher now!

**🎯 Tips for Best Accuracy:**
- **Duration**: Speak for about 2 seconds
- **Volume**: Any reasonable level (robust model handles variations)
- **Environment**: Works well even with some background noise
- **Clarity**: Speak clearly and distinctly
- **Pacing**: Normal speaking pace is fine
""")

# Model Info
with st.expander("🤖 Robust Model Information"):
    st.markdown("""
    **Model:** Robust Voice-Adapted Model (99.76% test accuracy)
    
    **Key Improvements:**
    - **112 features** (vs 34 original): MFCC, Delta MFCC, Delta-Delta MFCC, Chroma, Spectral
    - **Noise reduction**: Handles background noise automatically
    - **Data augmentation**: 2,044 training samples (vs 150 original)
    - **Pre-emphasis filter**: Enhances high frequencies
    - **Advanced architecture**: 256→128→64→32→10 layers with batch normalization
    
    **Training Data:**
    - Original voice samples + extensive augmentation
    - Pitch shift, time stretch, volume variation, noise addition
    - Balanced dataset (~200 samples per digit)
    
    **Features:** 
    - MFCC (13 coefficients) - mean and std
    - Delta MFCC (13 coefficients) - mean and std
    - Delta-Delta MFCC (13 coefficients) - mean and std
    - Spectral features (centroid, rolloff, bandwidth)
    - Zero crossing rate
    - RMS energy
    - Chroma features (12 pitch classes) - mean and std
    - Noise reduction preprocessing
    
    **Performance:** 99.76% test accuracy
    """)

# Troubleshooting
with st.expander("🔧 Troubleshooting"):
    st.markdown("""
    **This robust model should handle most issues automatically:**
    
    **Background Noise:**
    - ✅ Automatically reduced during preprocessing
    - ✅ Trained on noisy samples
    
    **Volume Variations:**
    - ✅ Handles quiet and loud audio
    - ✅ Normalized during processing
    
    **Speaking Style:**
    - ✅ Trained on augmented samples (pitch, speed, volume variations)
    - ✅ More flexible to different speaking styles
    
    **If you still have issues:**
    - Speak clearly and distinctly
    - Try to speak for about 2 seconds
    - Ensure microphone is working properly
    - Check that you're speaking a digit (0-9)
    """) 