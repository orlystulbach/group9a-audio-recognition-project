#!/usr/bin/env python3
"""
Quick Feature Extraction Test - Identify main issues
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import joblib

def extract_simple_features(audio_data, sample_rate=22050, n_mfcc=13):
    """Extract features without noise reduction (like training)"""
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
        print(f"Error extracting features: {e}")
        return None

def test_model_on_training_data():
    """Test if the model works on its own training data"""
    print("🧪 Testing Model on Training Data")
    print("=" * 50)
    
    try:
        # Load model
        model = tf.keras.models.load_model('voice_adaptive_model.h5')
        scaler = joblib.load('voice_adaptive_scaler.pkl')
        
        print("✅ Model loaded successfully")
        
        # Test on a few training samples
        correct = 0
        total = 0
        
        for digit in range(10):
            digit_dir = os.path.join('my_voice_samples', str(digit))
            if os.path.exists(digit_dir):
                wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
                
                # Test first sample of each digit
                if wav_files:
                    wav_file = wav_files[0]
                    try:
                        file_path = os.path.join(digit_dir, wav_file)
                        audio_data, sr = librosa.load(file_path, sr=22050)
                        
                        features = extract_simple_features(audio_data, sr)
                        if features is not None:
                            features_reshaped = features.reshape(1, -1)
                            features_scaled = scaler.transform(features_reshaped)
                            prediction = model.predict(features_scaled, verbose=0)
                            predicted_digit = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            total += 1
                            if predicted_digit == digit:
                                correct += 1
                                print(f"✅ Digit {digit}: Predicted {predicted_digit} (correct), Confidence: {confidence:.3f}")
                            else:
                                print(f"❌ Digit {digit}: Predicted {predicted_digit} (should be {digit}), Confidence: {confidence:.3f}")
                                
                    except Exception as e:
                        print(f"❌ Error processing digit {digit}: {e}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 Accuracy on training data: {accuracy*100:.2f}% ({correct}/{total})")
        
        if accuracy < 0.8:
            print("⚠️ Model is not performing well on training data!")
            print("This suggests a fundamental issue with the model or feature extraction.")
        else:
            print("✅ Model performs well on training data.")
            print("The issue is likely with real-world input preprocessing.")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def check_feature_consistency():
    """Check if features are consistent across training samples"""
    print("\n📊 Checking Feature Consistency")
    print("=" * 50)
    
    all_features = []
    
    for digit in range(10):
        digit_dir = os.path.join('my_voice_samples', str(digit))
        if os.path.exists(digit_dir):
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            
            for wav_file in wav_files[:2]:  # Test first 2 samples per digit
                try:
                    file_path = os.path.join(digit_dir, wav_file)
                    audio_data, sr = librosa.load(file_path, sr=22050)
                    
                    features = extract_simple_features(audio_data, sr)
                    if features is not None:
                        all_features.append(features)
                        
                except Exception as e:
                    print(f"❌ Error processing {wav_file}: {e}")
    
    if len(all_features) > 1:
        all_features = np.array(all_features)
        print(f"✅ Extracted features from {len(all_features)} samples")
        print(f"Feature shape: {all_features.shape}")
        print(f"Feature range: {np.min(all_features):.6f} to {np.max(all_features):.6f}")
        
        # Check for problematic features
        feature_means = np.mean(all_features, axis=0)
        feature_stds = np.std(all_features, axis=0)
        
        print(f"\n🔍 Feature Analysis:")
        problematic_features = []
        for i, (mean, std) in enumerate(zip(feature_means, feature_stds)):
            if std < 0.001:
                print(f"  Feature {i}: Very low variance (std={std:.6f}) - problematic")
                problematic_features.append(i)
            elif std > 10:
                print(f"  Feature {i}: Very high variance (std={std:.6f}) - noisy")
                problematic_features.append(i)
            elif abs(mean) > 100:
                print(f"  Feature {i}: Very large mean ({mean:.6f}) - needs scaling")
                problematic_features.append(i)
        
        if problematic_features:
            print(f"\n⚠️ Found {len(problematic_features)} potentially problematic features")
        else:
            print(f"\n✅ All features look reasonable")

def main():
    """Main test function"""
    print("🔍 Quick Feature Extraction Test")
    print("=" * 60)
    
    # Check if files exist
    if not os.path.exists('voice_adaptive_model.h5'):
        print("❌ voice_adaptive_model.h5 not found!")
        return
    
    if not os.path.exists('voice_adaptive_scaler.pkl'):
        print("❌ voice_adaptive_scaler.pkl not found!")
        return
    
    if not os.path.exists('my_voice_samples'):
        print("❌ my_voice_samples directory not found!")
        return
    
    print("✅ All required files found")
    
    # Run tests
    test_model_on_training_data()
    check_feature_consistency()
    
    print("\n💡 Next Steps:")
    print("1. If training accuracy is low: Model needs retraining")
    print("2. If training accuracy is high: Issue is with live preprocessing")
    print("3. If features are problematic: Consider feature engineering")

if __name__ == "__main__":
    main() 