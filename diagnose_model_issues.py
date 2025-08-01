#!/usr/bin/env python3
"""
Diagnostic script to test model performance and identify issues
"""

import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
import glob

def extract_improved_features(audio_data, sample_rate=22050, n_mfcc=13):
    """
    Extract comprehensive audio features - EXACTLY matching training code
    """
    try:
        # Normalize audio (matching training code)
        audio_data = librosa.util.normalize(audio_data)
        
        # Check if audio is too quiet
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 0.01:
            print(f"Warning: Very quiet audio (RMS: {rms:.4f})")
        
        features = []
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # Statistical features from MFCC
        features.extend(np.mean(mfcc, axis=1))  # Mean of each coefficient
        features.extend(np.std(mfcc, axis=1))   # Std of each coefficient
        features.extend(np.mean(mfcc_delta, axis=1))  # Delta features
        features.extend(np.mean(mfcc_delta2, axis=1)) # Delta-delta features
        
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

def test_model_on_training_data(model_path, test_samples=5):
    """
    Test model on actual training data to see if it works correctly
    """
    print(f"\n{'='*60}")
    print(f"Testing model: {model_path}")
    print(f"{'='*60}")
    
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully")
        
        # Test on some training data
        data_dir = 'large_dataset'
        if not os.path.exists(data_dir):
            print(f"❌ Training data directory '{data_dir}' not found")
            return False
            
        # Get some test files
        wav_files = glob.glob(os.path.join(data_dir, '*.wav'))[:test_samples]
        
        if not wav_files:
            print(f"❌ No WAV files found in {data_dir}")
            return False
            
        print(f"Testing on {len(wav_files)} training samples...")
        
        correct_predictions = 0
        total_predictions = 0
        
        for wav_file in wav_files:
            try:
                # Extract true label from filename
                filename = os.path.basename(wav_file)
                true_label = filename.split('_')[0]  # First part is the digit
                
                # Load and process audio
                audio_data, sr = librosa.load(wav_file, sr=22050)
                
                # Ensure 2-second duration
                target_length = int(22050 * 2.0)
                if len(audio_data) > target_length:
                    audio_data = audio_data[:target_length]
                elif len(audio_data) < target_length:
                    audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
                
                # Extract features
                features = extract_improved_features(audio_data, sr)
                
                if features is not None:
                    # Reshape for model
                    features = features.reshape(1, -1)
                    
                    # Make prediction
                    prediction = model.predict(features, verbose=0)
                    predicted_digit = np.argmax(prediction)
                    confidence = np.max(prediction)
                    
                    # Check if correct
                    is_correct = str(predicted_digit) == true_label
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    print(f"File: {filename}")
                    print(f"  True: {true_label}, Predicted: {predicted_digit}, Confidence: {confidence:.3f}")
                    print(f"  {'✅ CORRECT' if is_correct else '❌ WRONG'}")
                    print()
                    
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"📊 Model Accuracy on Training Data: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
        
        return accuracy > 0.5  # Consider working if >50% accuracy
        
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False

def test_feature_extraction():
    """
    Test if feature extraction is working correctly
    """
    print(f"\n{'='*60}")
    print("Testing Feature Extraction")
    print(f"{'='*60}")
    
    # Test on a sample file
    data_dir = 'large_dataset'
    wav_files = glob.glob(os.path.join(data_dir, '*.wav'))
    
    if wav_files:
        test_file = wav_files[0]
        print(f"Testing feature extraction on: {os.path.basename(test_file)}")
        
        try:
            audio_data, sr = librosa.load(test_file, sr=22050)
            features = extract_improved_features(audio_data, sr)
            
            if features is not None:
                print(f"✅ Feature extraction successful")
                print(f"   Feature shape: {features.shape}")
                print(f"   Feature range: [{features.min():.4f}, {features.max():.4f}]")
                print(f"   Any NaN values: {np.any(np.isnan(features))}")
                print(f"   Any Inf values: {np.any(np.isinf(features))}")
                return True
            else:
                print(f"❌ Feature extraction failed")
                return False
                
        except Exception as e:
            print(f"❌ Error in feature extraction: {e}")
            return False
    else:
        print(f"❌ No test files found")
        return False

def check_model_files():
    """
    Check which model files are available
    """
    print(f"\n{'='*60}")
    print("Available Model Files")
    print(f"{'='*60}")
    
    model_files = [
        'improved_voice_adapted_model.h5',
        'deeper_digit_classifier_model.h5', 
        'wide_digit_classifier_model.h5',
        'residual_digit_classifier_model.h5',
        'voice_adapted_model.h5',
        'fixed_digit_classifier_model.h5'
    ]
    
    available_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            size_mb = os.path.getsize(model_file) / (1024 * 1024)
            print(f"✅ {model_file} ({size_mb:.1f}MB)")
            available_models.append(model_file)
        else:
            print(f"❌ {model_file} (not found)")
    
    return available_models

def main():
    """
    Main diagnostic function
    """
    print("🔍 DIGITAL AUDIO CLASSIFIER DIAGNOSTIC")
    print("=" * 60)
    
    # Check available models
    available_models = check_model_files()
    
    if not available_models:
        print("❌ No model files found!")
        return
    
    # Test feature extraction
    feature_extraction_ok = test_feature_extraction()
    
    if not feature_extraction_ok:
        print("❌ Feature extraction is broken - this is the main issue!")
        return
    
    # Test each model
    working_models = []
    for model_path in available_models:
        if test_model_on_training_data(model_path):
            working_models.append(model_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*60}")
    
    if working_models:
        print(f"✅ Working models: {len(working_models)}")
        for model in working_models:
            print(f"   - {model}")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        print(f"1. Use the first working model: {working_models[0]}")
        print(f"2. If all models work on training data but fail on live audio:")
        print(f"   - Check audio recording quality")
        print(f"   - Verify microphone settings")
        print(f"   - Test with different speakers")
        print(f"   - Check for background noise")
    else:
        print(f"❌ No working models found!")
        print(f"\n🔧 NEXT STEPS:")
        print(f"1. Retrain the models")
        print(f"2. Check training data quality")
        print(f"3. Verify model architecture")

if __name__ == "__main__":
    main() 