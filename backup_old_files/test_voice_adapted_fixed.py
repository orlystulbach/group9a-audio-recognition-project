#!/usr/bin/env python3
"""
Fixed test for voice-adapted model with better audio handling
"""

import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import time
from tensorflow import keras

def list_audio_devices():
    """List available audio devices"""
    print("🎵 Available Audio Input Devices:")
    devices = sd.query_devices()
    
    input_devices = []
    for i, device in enumerate(devices):
        if device['max_inputs'] > 0:
            print(f"  {i}: {device['name']}")
            input_devices.append(i)
    
    return input_devices

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

def record_audio_with_retry(duration=2, sample_rate=22050, device_id=None):
    """Record audio with retry logic and device selection"""
    print(f"🎤 Recording for {duration} seconds...")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Try recording with specified device or default
            if device_id is not None:
                audio_data = sd.rec(int(duration * sample_rate), 
                                   samplerate=sample_rate, 
                                   channels=1, 
                                   dtype=np.float32,
                                   device=device_id)
            else:
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
            print(f"   Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                print("   Retrying in 1 second...")
                time.sleep(1)
            else:
                print("   All attempts failed")
                return None
    
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
    print("🎤 Voice-Adapted Model Test (Fixed)")
    print("=" * 50)
    print("This will compare the voice-adapted model with the original model.")
    print()
    
    # List audio devices
    input_devices = list_audio_devices()
    
    # Let user select device
    device_id = None
    if len(input_devices) > 1:
        try:
            choice = input(f"\nSelect audio device (0-{len(input_devices)-1}, or ENTER for default): ").strip()
            if choice.isdigit() and 0 <= int(choice) < len(input_devices):
                device_id = input_devices[int(choice)]
                print(f"Using device: {device_id}")
        except:
            print("Using default device")
    
    print("\nReady to test! Speak digits clearly.")
    
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
            audio_data = record_audio_with_retry(device_id=device_id)
            if audio_data is None:
                print("❌ Recording failed. Try again or check microphone permissions.")
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