#!/usr/bin/env python3
"""
Quick Voice Diagnostic - Identify specific voice prediction issues
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

def analyze_training_samples():
    """Analyze training samples to understand expected patterns"""
    print("📊 Analyzing Training Samples")
    print("=" * 50)
    
    all_features = []
    all_durations = []
    all_rms = []
    
    for digit in range(10):
        digit_dir = os.path.join('my_voice_samples', str(digit))
        if os.path.exists(digit_dir):
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            
            for wav_file in wav_files[:3]:  # Analyze first 3 samples per digit
                try:
                    file_path = os.path.join(digit_dir, wav_file)
                    audio_data, sr = librosa.load(file_path, sr=22050)
                    
                    # Audio characteristics
                    duration = len(audio_data) / sr
                    rms = np.sqrt(np.mean(audio_data**2))
                    
                    all_durations.append(duration)
                    all_rms.append(rms)
                    
                    # Extract features
                    features = extract_simple_features(audio_data, sr)
                    if features is not None:
                        all_features.append(features)
                        
                except Exception as e:
                    continue
    
    if len(all_features) > 0:
        all_features = np.array(all_features)
        
        print(f"Training sample statistics:")
        print(f"  Number of samples: {len(all_features)}")
        print(f"  Duration range: {min(all_durations):.2f}s - {max(all_durations):.2f}s")
        print(f"  RMS range: {min(all_rms):.6f} - {max(all_rms):.6f}")
        print(f"  Feature range: {np.min(all_features):.6f} - {np.max(all_features):.6f}")
        
        # Identify expected ranges
        print(f"\n🎯 Expected ranges for live input:")
        print(f"  Duration: {np.mean(all_durations):.2f}s ± {np.std(all_durations):.2f}s")
        print(f"  RMS: {np.mean(all_rms):.6f} ± {np.std(all_rms):.6f}")
        print(f"  Features: {np.mean(all_features):.6f} ± {np.std(all_features):.6f}")
        
        return {
            'duration_mean': np.mean(all_durations),
            'duration_std': np.std(all_durations),
            'rms_mean': np.mean(all_rms),
            'rms_std': np.std(all_rms),
            'feature_mean': np.mean(all_features),
            'feature_std': np.std(all_features)
        }
    
    return None

def test_model_sensitivity():
    """Test model sensitivity to different inputs"""
    print("\n🧪 Testing Model Sensitivity")
    print("=" * 50)
    
    try:
        # Load model
        model = tf.keras.models.load_model('fixed_voice_model.h5')
        scaler = joblib.load('fixed_voice_scaler.pkl')
        
        # Test with different types of input
        test_cases = []
        
        # Test 1: Load a training sample and modify it
        training_file = os.path.join('my_voice_samples', '0', '0_1.wav')
        if os.path.exists(training_file):
            audio_data, sr = librosa.load(training_file, sr=22050)
            
            # Original
            features = extract_simple_features(audio_data, sr)
            if features is not None:
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction = model.predict(features_scaled, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)
                test_cases.append(('Original training sample', predicted_digit, confidence))
            
            # Test with volume change
            audio_loud = audio_data * 2.0  # Double volume
            features_loud = extract_simple_features(audio_loud, sr)
            if features_loud is not None:
                features_scaled = scaler.transform(features_loud.reshape(1, -1))
                prediction = model.predict(features_scaled, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)
                test_cases.append(('Loud version (2x)', predicted_digit, confidence))
            
            # Test with volume reduction
            audio_quiet = audio_data * 0.5  # Half volume
            features_quiet = extract_simple_features(audio_quiet, sr)
            if features_quiet is not None:
                features_scaled = scaler.transform(features_quiet.reshape(1, -1))
                prediction = model.predict(features_scaled, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)
                test_cases.append(('Quiet version (0.5x)', predicted_digit, confidence))
        
        # Report results
        print("Model sensitivity test results:")
        for test_name, predicted, confidence in test_cases:
            print(f"  {test_name}: {predicted} (confidence: {confidence:.3f})")
        
        # Check if model is sensitive to volume changes
        if len(test_cases) >= 3:
            original_pred = test_cases[0][1]
            loud_pred = test_cases[1][1]
            quiet_pred = test_cases[2][1]
            
            if original_pred != loud_pred or original_pred != quiet_pred:
                print("⚠️ Model is sensitive to volume changes!")
            else:
                print("✅ Model is robust to volume changes")
                
    except Exception as e:
        print(f"❌ Error in sensitivity test: {e}")

def provide_recommendations():
    """Provide specific recommendations based on analysis"""
    print("\n💡 Specific Recommendations")
    print("=" * 50)
    
    print("Based on the analysis, here are the most likely issues and solutions:")
    
    print("\n🎤 **Audio Quality Issues:**")
    print("1. **Microphone differences** - Use the same microphone as training")
    print("2. **Background noise** - Record in a quiet environment")
    print("3. **Audio levels** - Speak at similar volume as training samples")
    print("4. **Distance from mic** - Maintain consistent distance")
    
    print("\n🗣️ **Speaking Style Issues:**")
    print("1. **Pacing** - Speak at similar speed as training")
    print("2. **Pronunciation** - Use same pronunciation style")
    print("3. **Duration** - Speak for 1-2 seconds (like training)")
    print("4. **Clarity** - Speak clearly and distinctly")
    
    print("\n🔧 **Technical Solutions:**")
    print("1. **Re-record training samples** with current microphone")
    print("2. **Use noise reduction** in live app (if not in training)")
    print("3. **Normalize audio levels** before processing")
    print("4. **Collect more diverse training samples**")
    
    print("\n🎯 **Immediate Actions:**")
    print("1. Test the app in a quiet environment")
    print("2. Speak clearly and at normal pace")
    print("3. Use similar volume as training samples")
    print("4. Check the 'All Predictions' section for patterns")

def main():
    """Main diagnostic function"""
    print("🔍 Quick Voice Prediction Diagnostic")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists('fixed_voice_model.h5'):
        print("❌ fixed_voice_model.h5 not found!")
        return
    
    print("✅ Model found")
    
    # Run analysis
    training_stats = analyze_training_samples()
    test_model_sensitivity()
    provide_recommendations()
    
    print(f"\n🎯 **Next Steps:**")
    print("1. Try the app with the recommendations above")
    print("2. If still having issues, we can:")
    print("   - Re-record training samples with current setup")
    print("   - Try a different feature extraction method")
    print("   - Use a more robust model architecture")
    print("   - Implement audio preprocessing in the app")

if __name__ == "__main__":
    main() 