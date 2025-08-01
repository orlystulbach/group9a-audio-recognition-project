import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

st.title("🔧 Simple Test App")
st.write("Testing basic functionality...")

# Test model loading
try:
    model = tf.keras.models.load_model('simple_working_model.h5')
    scaler = joblib.load('simple_model_scaler.pkl')
    st.success("✅ Model and scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# Test audio input
st.header("Audio Test")
audio_input = st.audio_input("Test recording")

if audio_input is not None:
    st.write("Audio recorded successfully!")
    st.audio(audio_input, format="audio/wav")
else:
    st.info("No audio recorded yet")

# Test basic functionality
st.header("Basic Tests")
if st.button("Test Button"):
    st.success("Button works!")

# Test imports
try:
    import librosa
    st.success("✅ Librosa imported successfully")
except Exception as e:
    st.error(f"❌ Librosa import error: {e}")

try:
    import noisereduce as nr
    st.success("✅ Noisereduce imported successfully")
except Exception as e:
    st.error(f"❌ Noisereduce import error: {e}")

st.write("Test complete!") 