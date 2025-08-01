#!/usr/bin/env python3
"""
Voice Test Script - Record and test your voice
"""

import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

def record_audio(duration=2, sample_rate=22050):
    """Record audio from microphone"""
    print(f"Recording {duration} seconds... Speak a digit (0-9) clearly!")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    return audio.flatten()

def extract_simple_features(audio_data, sample_rate=22050, n_mfcc=13):
    """Extract features matching the model"""
    # Normalize audio
    audio_data = librosa.util.normalize(audio_data)
    
    # Ensure 2-second duration
    target_length = int(22050 * 2.0)
    if len(audio_data) > target_length:
        audio_data = audio_data[:target_length]
    elif len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
    
    features = []
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate, hop_length=512)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate, hop_length=512)[0]
    features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
    features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data, hop_length=512)[0]
    features.extend([np.mean(zcr), np.std(zcr)])
    
    # RMS energy
    rms = librosa.feature.rms(y=audio_data, hop_length=512)[0]
    features.extend([np.mean(rms), np.std(rms)])
    
    return np.array(features)

def main():
    # Load model and scaler
    model = tf.keras.models.load_model('simple_working_model.h5')
    scaler = joblib.load('simple_model_scaler.pkl')
    
    print("🎤 Voice Test Script")
    print("=" * 50)
    
    while True:
        input("Press Enter to record a digit (or Ctrl+C to exit)...")
        
        try:
            # Record audio
            audio_data = record_audio(duration=2)
            
            # Analyze audio
            rms = np.sqrt(np.mean(audio_data**2))
            print(f"Audio RMS: {rms:.6f}")
            
            if rms < 0.01:
                print("⚠️ Audio too quiet! Speak louder.")
                continue
            
            # Extract features
            features = extract_simple_features(audio_data)
            features_reshaped = features.reshape(1, -1)
            
            # Scale features
            features_scaled = scaler.transform(features_reshaped)
            
            # Make prediction
            prediction = model.predict(features_scaled, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction)
            
            print(f"🎯 Predicted: {predicted_digit}")
            print(f"📈 Confidence: {confidence:.3f}")
            
            # Save audio for analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"voice_test_{predicted_digit}_{timestamp}.wav"
            import soundfile as sf
            sf.write(filename, audio_data, 22050)
            print(f"💾 Saved as: {filename}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
