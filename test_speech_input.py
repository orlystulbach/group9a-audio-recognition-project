#!/usr/bin/env python3
"""
Simple test script for speech input functionality
"""

import os
import numpy as np
import librosa
from tensorflow import keras

def test_model_loading():
    """Test if models can be loaded properly"""
    print("🧪 Testing model loading...")
    
    model_files = [
        'fixed_digit_classifier_model.h5',
        'deeper_digit_classifier_model.h5',
        'residual_digit_classifier_model.h5',
        'wide_digit_classifier_model.h5'
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = keras.models.load_model(model_file)
                print(f"✅ {model_file}: Loaded successfully")
                print(f"   Model input shape: {model.input_shape}")
                print(f"   Model output shape: {model.output_shape}")
            except Exception as e:
                print(f"❌ {model_file}: Error loading - {e}")
        else:
            print(f"⚠️  {model_file}: Not found")

def test_feature_extraction():
    """Test feature extraction on a sample audio file"""
    print("\n🧪 Testing feature extraction...")
    
    # Find a sample audio file
    data_dir = 'large_dataset'
    if os.path.exists(data_dir):
        wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
        if wav_files:
            sample_file = os.path.join(data_dir, wav_files[0])
            print(f"Using sample file: {wav_files[0]}")
            
            try:
                # Load audio
                y, sr = librosa.load(sample_file, sr=22050)
                print(f"   Audio loaded: {len(y)} samples, {sr} Hz")
                
                # Test feature extraction (same as in speech_input.py)
                y = librosa.util.normalize(y)
                
                # MFCC features
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                
                features = []
                features.extend(np.mean(mfcc, axis=1))
                features.extend(np.std(mfcc, axis=1))
                features.extend(np.mean(mfcc_delta, axis=1))
                features.extend(np.mean(mfcc_delta2, axis=1))
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
                spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
                
                features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
                features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
                features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
                
                # Chroma features
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
                features.extend(np.mean(chroma, axis=1))
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
                features.extend([np.mean(zcr), np.std(zcr)])
                
                # RMS energy
                rms = librosa.feature.rms(y=y, hop_length=512)[0]
                features.extend([np.mean(rms), np.std(rms)])
                
                features = np.array(features)
                print(f"   Features extracted: {features.shape}")
                print(f"   Feature range: [{features.min():.4f}, {features.max():.4f}]")
                
                # Test prediction
                model = keras.models.load_model('fixed_digit_classifier_model.h5')
                prediction = model.predict(features.reshape(1, -1), verbose=0)
                predicted_digit = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                print(f"   Prediction test: digit {predicted_digit} with {confidence:.2%} confidence")
                print("✅ Feature extraction and prediction working!")
                
            except Exception as e:
                print(f"❌ Error in feature extraction: {e}")
        else:
            print("❌ No audio files found in large_dataset")
    else:
        print("❌ large_dataset directory not found")

def test_audio_recording():
    """Test if audio recording works"""
    print("\n🧪 Testing audio recording capability...")
    
    try:
        import sounddevice as sd
        print("✅ sounddevice imported successfully")
        
        # Test if we can get device info
        devices = sd.query_devices()
        print(f"   Found {len(devices)} audio devices")
        
        # Find default input device
        default_input = sd.query_devices(kind='input')
        print(f"   Default input device: {default_input['name']}")
        
        print("✅ Audio recording capability available")
        
    except ImportError:
        print("❌ sounddevice not installed")
    except Exception as e:
        print(f"❌ Error testing audio recording: {e}")

def main():
    """Run all tests"""
    print("🧪 Speech Input Test Suite")
    print("=" * 40)
    
    test_model_loading()
    test_feature_extraction()
    test_audio_recording()
    
    print("\n" + "=" * 40)
    print("🎯 Test Summary:")
    print("If all tests passed, you can run: python3 speech_input.py")
    print("The script will guide you through the interactive process.")

if __name__ == "__main__":
    main() 