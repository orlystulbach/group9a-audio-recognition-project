#!/usr/bin/env python3
"""
Quick test to compare voice-adapted model with original models
"""

import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import time
from tensorflow import keras

def extract_features(audio_data, sample_rate=22050):
    """Extract features from audio data"""
    try:
        # Normalize audio
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
        print(f"Error extracting features: {e}")
        return None

def record_audio(duration=2, sample_rate=22050):
    """Record audio from microphone"""
    print(f"🎤 Recording for {duration} seconds...")
    
    try:
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype=np.float32)
        sd.wait()
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        return audio_data
        
    except Exception as e:
        print(f"❌ Recording error: {e}")
        return None

def test_models(audio_data, expected_digit):
    """Test both original and voice-adapted models"""
    features = extract_features(audio_data)
    if features is None:
        return {}
    
    results = {}
    features_reshaped = features.reshape(1, -1)
    
    # Test original model (residual)
    try:
        original_model = keras.models.load_model('residual_digit_classifier_model.h5')
        prediction = original_model.predict(features_reshaped, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        results['original'] = {
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'correct': str(predicted_digit) == str(expected_digit)
        }
    except Exception as e:
        print(f"❌ Error with original model: {e}")
        results['original'] = None
    
    # Test voice-adapted model
    try:
        adapted_model = keras.models.load_model('voice_adapted_model.h5')
        prediction = adapted_model.predict(features_reshaped, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        results['adapted'] = {
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'correct': str(predicted_digit) == str(expected_digit)
        }
    except Exception as e:
        print(f"❌ Error with adapted model: {e}")
        results['adapted'] = None
    
    return results

def main():
    """Main test function"""
    print("🎤 Voice-Adapted Model Test")
    print("=" * 50)
    print("This will compare the voice-adapted model with the original model.")
    print("Speak a digit and we'll see which model performs better.")
    print()
    
    while True:
        try:
            # Get expected digit from user
            expected_digit = input("Enter the digit you'll speak (0-9, or 'quit'): ").strip()
            
            if expected_digit.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if not expected_digit.isdigit() or int(expected_digit) < 0 or int(expected_digit) > 9:
                print("❌ Please enter a digit 0-9")
                continue
            
            print(f"\n🎯 You'll speak: {expected_digit}")
            print("Press ENTER when ready to record...")
            input()
            
            # Record audio
            audio_data = record_audio()
            if audio_data is None:
                continue
            
            # Check audio quality
            rms = np.sqrt(np.mean(audio_data**2))
            print(f"📊 Audio RMS: {rms:.6f}")
            
            if rms < 0.001:
                print("🔇 Audio too quiet. Please speak louder.")
                continue
            
            # Test both models
            print("\n🤖 Testing models...")
            results = test_models(audio_data, expected_digit)
            
            # Display results
            print("\n📊 Results:")
            print("-" * 50)
            
            if results.get('original'):
                orig = results['original']
                status = "✅" if orig['correct'] else "❌"
                print(f"{status} Original Model:")
                print(f"   Predicted: {orig['predicted_digit']} (Expected: {expected_digit})")
                print(f"   Confidence: {orig['confidence']:.2%}")
            
            if results.get('adapted'):
                adapt = results['adapted']
                status = "✅" if adapt['correct'] else "❌"
                print(f"{status} Voice-Adapted Model:")
                print(f"   Predicted: {adapt['predicted_digit']} (Expected: {expected_digit})")
                print(f"   Confidence: {adapt['confidence']:.2%}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 