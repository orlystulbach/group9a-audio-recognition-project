#!/usr/bin/env python3
"""
Quick test for voice-adapted model using existing working functionality
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

def test_voice_adapted_model(audio_data, expected_digit):
    """Test the voice-adapted model"""
    features = extract_features(audio_data)
    if features is None:
        return None
    
    try:
        # Load voice-adapted model
        model = keras.models.load_model('voice_adapted_model.h5')
        
        # Make prediction
        features_reshaped = features.reshape(1, -1)
        prediction = model.predict(features_reshaped, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        return {
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'correct': str(predicted_digit) == str(expected_digit),
            'all_probabilities': prediction[0]
        }
        
    except Exception as e:
        print(f"❌ Error with voice-adapted model: {e}")
        return None

def main():
    """Main test function"""
    print("🎤 Quick Voice-Adapted Model Test")
    print("=" * 50)
    print("Testing the voice-adapted model with your voice.")
    print("This model was trained on your voice samples!")
    print()
    
    # Check if voice-adapted model exists
    if not os.path.exists('voice_adapted_model.h5'):
        print("❌ Voice-adapted model not found!")
        print("Please run voice_adaptation_training.py first.")
        return
    
    print("✅ Voice-adapted model found!")
    print("Ready to test. Speak digits clearly.")
    
    while True:
        try:
            # Get expected digit from user
            expected_digit = input("\nEnter the digit you'll speak (0-9, or 'quit'): ").strip()
            
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
                print("❌ Recording failed. Try again.")
                continue
            
            # Check audio quality
            rms = np.sqrt(np.mean(audio_data**2))
            print(f"📊 Audio RMS: {rms:.6f}")
            
            if rms < 0.001:
                print("🔇 Audio too quiet. Please speak louder.")
                continue
            
            # Test voice-adapted model
            print("\n🤖 Testing voice-adapted model...")
            result = test_voice_adapted_model(audio_data, expected_digit)
            
            if result is None:
                print("❌ Model test failed.")
                continue
            
            # Display results
            print("\n📊 Results:")
            print("-" * 50)
            
            status = "✅" if result['correct'] else "❌"
            print(f"{status} Voice-Adapted Model:")
            print(f"   Predicted: {result['predicted_digit']} (Expected: {expected_digit})")
            print(f"   Confidence: {result['confidence']:.2%}")
            
            if result['correct']:
                print("   🎉 Correct prediction!")
            else:
                print("   🤔 Incorrect prediction")
            
            # Show top 3 predictions
            probs = result['all_probabilities']
            top_indices = np.argsort(probs)[-3:][::-1]
            print("   Top 3 predictions:")
            for i, idx in enumerate(top_indices):
                marker = "🎯" if i == 0 else "  "
                print(f"     {marker} Digit {idx}: {probs[idx]:.2%}")
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 