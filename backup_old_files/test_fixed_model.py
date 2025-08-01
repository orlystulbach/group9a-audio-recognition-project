#!/usr/bin/env python3
"""
Test Fixed Voice Model - Check if the retrained model works properly
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import joblib

def extract_simple_features(audio_data, sample_rate=22050, n_mfcc=13):
    """Extract features without noise reduction"""
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

def test_fixed_model():
    """Test the fixed voice model"""
    print("🧪 Testing Fixed Voice Model")
    print("=" * 50)
    
    try:
        # Load fixed model
        model = tf.keras.models.load_model('fixed_voice_model.h5')
        scaler = joblib.load('fixed_voice_scaler.pkl')
        lb = joblib.load('fixed_voice_label_binarizer.pkl')
        
        print("✅ Fixed model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Label binarizer classes: {lb.classes_}")
        
        # Test on training data
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
                                
                                # Show all predictions for debugging
                                print(f"   All predictions: {prediction[0]}")
                                
                    except Exception as e:
                        print(f"❌ Error processing digit {digit}: {e}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 Fixed model accuracy on training data: {accuracy*100:.2f}% ({correct}/{total})")
        
        if accuracy > 0.8:
            print("✅ Fixed model is working correctly!")
        else:
            print("⚠️ Fixed model still has issues")
            
    except Exception as e:
        print(f"❌ Error testing fixed model: {e}")

def test_feature_scaling():
    """Test if feature scaling is working properly"""
    print("\n📊 Testing Feature Scaling")
    print("=" * 50)
    
    try:
        scaler = joblib.load('fixed_voice_scaler.pkl')
        
        # Load a few samples and check scaling
        all_features = []
        for digit in range(10):
            digit_dir = os.path.join('my_voice_samples', str(digit))
            if os.path.exists(digit_dir):
                wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
                
                if wav_files:
                    wav_file = wav_files[0]
                    try:
                        file_path = os.path.join(digit_dir, wav_file)
                        audio_data, sr = librosa.load(file_path, sr=22050)
                        
                        features = extract_simple_features(audio_data, sr)
                        if features is not None:
                            all_features.append(features)
                            
                    except Exception as e:
                        continue
        
        if len(all_features) > 0:
            all_features = np.array(all_features)
            print(f"Original features shape: {all_features.shape}")
            print(f"Original features range: {np.min(all_features):.6f} to {np.max(all_features):.6f}")
            
            # Scale features
            scaled_features = scaler.transform(all_features)
            print(f"Scaled features range: {np.min(scaled_features):.6f} to {np.max(scaled_features):.6f}")
            print(f"Scaled features mean: {np.mean(scaled_features, axis=0)}")
            print(f"Scaled features std: {np.std(scaled_features, axis=0)}")
            
            # Check if scaling looks reasonable
            if np.abs(np.mean(scaled_features)) < 1.0 and np.std(scaled_features) < 2.0:
                print("✅ Feature scaling looks reasonable")
            else:
                print("⚠️ Feature scaling might be problematic")
                
    except Exception as e:
        print(f"❌ Error testing feature scaling: {e}")

def main():
    """Main test function"""
    print("🔍 Testing Fixed Voice Model")
    print("=" * 60)
    
    # Check if fixed model files exist
    if not os.path.exists('fixed_voice_model.h5'):
        print("❌ fixed_voice_model.h5 not found!")
        return
    
    if not os.path.exists('fixed_voice_scaler.pkl'):
        print("❌ fixed_voice_scaler.pkl not found!")
        return
    
    if not os.path.exists('fixed_voice_label_binarizer.pkl'):
        print("❌ fixed_voice_label_binarizer.pkl not found!")
        return
    
    print("✅ All fixed model files found")
    
    # Run tests
    test_fixed_model()
    test_feature_scaling()
    
    print("\n💡 Analysis:")
    print("If the fixed model still has issues, we may need to:")
    print("1. Collect better quality voice samples")
    print("2. Use a different feature extraction method")
    print("3. Try a different model architecture")
    print("4. Check for audio preprocessing issues")

if __name__ == "__main__":
    main() 