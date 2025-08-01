#!/usr/bin/env python3
"""
Test Script for Robust Voice Model
Verifies that all model files load correctly and the model is functional
"""

import os
import numpy as np
import tensorflow as tf
import joblib
import librosa

def test_model_loading():
    """Test if all model files can be loaded"""
    print("🧪 Testing Model Loading")
    print("=" * 50)
    
    # Check if files exist
    required_files = [
        'robust_voice_model.h5',
        'robust_voice_scaler.pkl',
        'robust_voice_label_binarizer.pkl'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing!")
            return False
    
    # Try loading the model
    try:
        model = tf.keras.models.load_model('robust_voice_model.h5')
        print(f"✅ Model loaded successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        print(f"   Parameters: {model.count_params():,}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Try loading the scaler
    try:
        scaler = joblib.load('robust_voice_scaler.pkl')
        print(f"✅ Scaler loaded successfully")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return False
    
    # Try loading the label binarizer
    try:
        lb = joblib.load('robust_voice_label_binarizer.pkl')
        print(f"✅ Label binarizer loaded successfully")
        print(f"   Classes: {lb.classes_}")
    except Exception as e:
        print(f"❌ Error loading label binarizer: {e}")
        return False
    
    return True

def test_feature_extraction():
    """Test feature extraction function"""
    print("\n🔧 Testing Feature Extraction")
    print("=" * 50)
    
    try:
        # Create a dummy audio signal (2 seconds of random noise)
        sample_rate = 22050
        duration = 2.0
        audio_data = np.random.normal(0, 0.1, int(sample_rate * duration))
        
        # Test feature extraction
        features = extract_robust_features(audio_data, sample_rate)
        
        if features is not None:
            print(f"✅ Feature extraction successful")
            print(f"   Feature shape: {features.shape}")
            print(f"   Expected: (112,)")
            print(f"   Actual: {features.shape}")
            
            if features.shape == (112,):
                print("✅ Feature dimensions correct")
                return True
            else:
                print("❌ Feature dimensions incorrect")
                return False
        else:
            print("❌ Feature extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Error in feature extraction: {e}")
        return False

def extract_robust_features(audio_data, sample_rate=22050, n_mfcc=13):
    """Extract robust features with noise reduction and enhancement"""
    try:
        import noisereduce as nr
        
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
        print(f"Error extracting robust features: {e}")
        return None

def test_prediction():
    """Test model prediction with dummy data"""
    print("\n🎯 Testing Model Prediction")
    print("=" * 50)
    
    try:
        # Load model components
        model = tf.keras.models.load_model('robust_voice_model.h5')
        scaler = joblib.load('robust_voice_scaler.pkl')
        lb = joblib.load('robust_voice_label_binarizer.pkl')
        
        # Create dummy features
        dummy_features = np.random.normal(0, 1, (1, 112))
        
        # Scale features
        scaled_features = scaler.transform(dummy_features)
        
        # Make prediction
        prediction = model.predict(scaled_features, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        print(f"✅ Prediction successful")
        print(f"   Predicted digit: {predicted_digit}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   All predictions sum to: {np.sum(prediction):.6f} (should be ~1.0)")
        
        if 0.99 <= np.sum(prediction) <= 1.01:
            print("✅ Prediction probabilities are valid")
            return True
        else:
            print("❌ Prediction probabilities are invalid")
            return False
            
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Robust Voice Model Test Suite")
    print("=" * 60)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("Feature Extraction", test_feature_extraction),
        ("Model Prediction", test_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} Test...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! Model is ready to use.")
        print("\n🚀 To run the Streamlit app:")
        print("   streamlit run app_robust_voice.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    main() 