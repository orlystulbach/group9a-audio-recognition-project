#!/usr/bin/env python3
"""
Debug script to identify voice prediction issues
"""

import numpy as np
import librosa
import tensorflow as tf
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

def extract_simple_features(audio_data, sample_rate=22050, n_mfcc=13):
    """
    Extract simple, robust features - EXACTLY matching the working model
    """
    try:
        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        
        # Ensure 2-second duration
        target_length = int(22050 * 2.0)
        if len(audio_data) > target_length:
            audio_data = audio_data[:target_length]
        elif len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), 'constant')
        
        features = []
        
        # MFCC features (simple version)
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
        
        # Statistical features from MFCC
        features.extend(np.mean(mfcc, axis=1))  # Mean of each coefficient
        features.extend(np.std(mfcc, axis=1))   # Std of each coefficient
        
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
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def analyze_audio_characteristics(audio_data, sample_rate, title="Audio Analysis"):
    """
    Analyze audio characteristics to understand differences
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    # Basic stats
    rms = np.sqrt(np.mean(audio_data**2))
    print(f"RMS Energy: {rms:.6f}")
    print(f"Max Amplitude: {np.max(np.abs(audio_data)):.6f}")
    print(f"Min Amplitude: {np.min(audio_data):.6f}")
    print(f"Mean Amplitude: {np.mean(audio_data):.6f}")
    print(f"Std Amplitude: {np.std(audio_data):.6f}")
    
    # Duration
    duration = len(audio_data) / sample_rate
    print(f"Duration: {duration:.3f} seconds")
    
    # Spectral analysis
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate)[0]
    
    print(f"Spectral Centroid - Mean: {np.mean(spectral_centroids):.2f}, Std: {np.std(spectral_centroids):.2f}")
    print(f"Spectral Rolloff - Mean: {np.mean(spectral_rolloff):.2f}, Std: {np.std(spectral_rolloff):.2f}")
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
    print(f"Zero Crossing Rate - Mean: {np.mean(zcr):.4f}, Std: {np.std(zcr):.4f}")
    
    return {
        'rms': rms,
        'duration': duration,
        'spectral_centroid_mean': np.mean(spectral_centroids),
        'spectral_rolloff_mean': np.mean(spectral_rolloff),
        'zcr_mean': np.mean(zcr)
    }

def compare_features(training_features, voice_features, scaler):
    """
    Compare feature distributions between training and voice data
    """
    print(f"\n{'='*60}")
    print("FEATURE COMPARISON")
    print(f"{'='*60}")
    
    # Scale both using the same scaler
    training_scaled = scaler.transform(training_features)
    voice_scaled = scaler.transform(voice_features)
    
    print(f"Training features shape: {training_features.shape}")
    print(f"Voice features shape: {voice_features.shape}")
    
    print(f"\nFeature Statistics:")
    print(f"{'Feature':<20} {'Training Mean':<15} {'Voice Mean':<15} {'Difference':<15}")
    print("-" * 70)
    
    for i in range(min(len(training_features[0]), len(voice_features[0]))):
        train_mean = np.mean(training_scaled[:, i])
        voice_mean = np.mean(voice_scaled[:, i])
        diff = abs(train_mean - voice_mean)
        print(f"{f'Feature_{i}':<20} {train_mean:<15.4f} {voice_mean:<15.4f} {diff:<15.4f}")
    
    # Check for outliers
    print(f"\nOutlier Analysis:")
    for i in range(min(len(training_features[0]), len(voice_features[0]))):
        train_std = np.std(training_scaled[:, i])
        voice_val = voice_scaled[0, i]
        z_score = abs(voice_val - np.mean(training_scaled[:, i])) / train_std
        if z_score > 3:
            print(f"Feature_{i}: Z-score = {z_score:.2f} (OUTLIER!)")

def test_model_on_training_data():
    """
    Test model on training data to establish baseline
    """
    print(f"\n{'='*60}")
    print("TESTING MODEL ON TRAINING DATA")
    print(f"{'='*60}")
    
    # Load model and scaler
    model = tf.keras.models.load_model('simple_working_model.h5')
    scaler = joblib.load('simple_model_scaler.pkl')
    
    # Test on some training files
    data_dir = 'large_dataset'
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')][:10]
    
    training_features = []
    correct_predictions = 0
    total_predictions = 0
    
    for wav_file in wav_files:
        try:
            # Extract true label
            true_label = wav_file.split('_')[0]
            
            # Load and process audio
            file_path = os.path.join(data_dir, wav_file)
            audio_data, sr = librosa.load(file_path, sr=22050)
            
            # Extract features
            features = extract_simple_features(audio_data, sr)
            
            if features is not None:
                training_features.append(features)
                
                # Make prediction
                features_reshaped = features.reshape(1, -1)
                features_scaled = scaler.transform(features_reshaped)
                prediction = model.predict(features_scaled, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction)
                
                # Check if correct
                is_correct = str(predicted_digit) == true_label
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                print(f"File: {wav_file}")
                print(f"  True: {true_label}, Predicted: {predicted_digit}, Confidence: {confidence:.3f}")
                print(f"  {'✅ CORRECT' if is_correct else '❌ WRONG'}")
                
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")
            continue
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nTraining Data Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
    
    return np.array(training_features), scaler

def create_voice_test_script():
    """
    Create a script to help users test their voice
    """
    script_content = '''#!/usr/bin/env python3
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
            print("\\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open('voice_test_script.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created voice_test_script.py")
    print("Run: python3 voice_test_script.py")

def main():
    """
    Main debugging function
    """
    print("🔍 VOICE PREDICTION DEBUGGING")
    print("=" * 60)
    
    # Test model on training data first
    training_features, scaler = test_model_on_training_data()
    
    # Create voice test script
    create_voice_test_script()
    
    print(f"\n{'='*60}")
    print("DEBUGGING RECOMMENDATIONS")
    print(f"{'='*60}")
    
    print("1. **Test with voice_test_script.py**")
    print("   - Run: python3 voice_test_script.py")
    print("   - Record multiple digits and analyze results")
    
    print("\n2. **Check Audio Quality**")
    print("   - Ensure RMS > 0.01 (not too quiet)")
    print("   - Record in quiet environment")
    print("   - Use consistent microphone distance")
    
    print("\n3. **Feature Analysis**")
    print("   - Compare your voice features with training data")
    print("   - Look for outliers in feature values")
    
    print("\n4. **Common Issues**")
    print("   - Background noise affecting features")
    print("   - Different microphone characteristics")
    print("   - Speaking style differences")
    print("   - Audio preprocessing differences")
    
    print("\n5. **Next Steps**")
    print("   - Collect voice samples for fine-tuning")
    print("   - Implement noise reduction")
    print("   - Add data augmentation for robustness")

if __name__ == "__main__":
    main() 