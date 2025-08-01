#!/usr/bin/env python3
"""
Diagnose Voice-Adapted Model Issues
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import joblib
import noisereduce as nr
import sounddevice as sd
import soundfile as sf
from datetime import datetime

def extract_robust_features(audio_data, sample_rate=22050, n_mfcc=13):
    """Extract features with noise reduction"""
    try:
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
    """Test the model on the training data to see if it's working"""
    print("🧪 Testing Model on Training Data")
    print("=" * 50)
    
    try:
        # Load model
        model = tf.keras.models.load_model('voice_adaptive_model.h5')
        scaler = joblib.load('voice_adaptive_scaler.pkl')
        lb = joblib.load('voice_adaptive_label_binarizer.pkl')
        
        print("✅ Model loaded successfully")
        
        # Test on voice samples
        correct = 0
        total = 0
        
        for digit in range(10):
            digit_dir = os.path.join('my_voice_samples', str(digit))
            if os.path.exists(digit_dir):
                wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
                
                for wav_file in wav_files[:3]:  # Test first 3 samples per digit
                    try:
                        file_path = os.path.join(digit_dir, wav_file)
                        audio_data, sr = librosa.load(file_path, sr=22050)
                        
                        features = extract_robust_features(audio_data, sr)
                        if features is not None:
                            features_reshaped = features.reshape(1, -1)
                            features_scaled = scaler.transform(features_reshaped)
                            prediction = model.predict(features_scaled, verbose=0)
                            predicted_digit = np.argmax(prediction)
                            confidence = np.max(prediction)
                            
                            total += 1
                            if predicted_digit == digit:
                                correct += 1
                                print(f"✅ {wav_file}: Predicted {predicted_digit} (correct), Confidence: {confidence:.3f}")
                            else:
                                print(f"❌ {wav_file}: Predicted {predicted_digit} (should be {digit}), Confidence: {confidence:.3f}")
                                
                    except Exception as e:
                        print(f"Error processing {wav_file}: {e}")
        
        accuracy = correct / total if total > 0 else 0
        print(f"\n📊 Accuracy on training data: {accuracy*100:.2f}% ({correct}/{total})")
        
        if accuracy < 0.8:
            print("⚠️ Model is not performing well on training data - this indicates a fundamental issue")
        else:
            print("✅ Model performs well on training data - issue is with real-world input")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def test_live_recording():
    """Test with live recording"""
    print("\n🎤 Testing with Live Recording")
    print("=" * 50)
    
    try:
        # Load model
        model = tf.keras.models.load_model('voice_adaptive_model.h5')
        scaler = joblib.load('voice_adaptive_scaler.pkl')
        
        print("Speak a digit clearly and let's see what happens...")
        input("Press Enter to record...")
        
        # Record audio
        print("🎤 Recording...")
        audio = sd.rec(int(2 * 22050), samplerate=22050, channels=1)
        sd.wait()
        audio_data = audio.flatten()
        
        # Check audio quality
        rms = np.sqrt(np.mean(audio_data**2))
        print(f"Audio RMS: {rms:.6f}")
        
        if rms < 0.01:
            print("⚠️ Audio too quiet!")
            return
        
        # Extract features
        features = extract_robust_features(audio_data)
        if features is None:
            print("❌ Failed to extract features")
            return
        
        # Make prediction
        features_reshaped = features.reshape(1, -1)
        features_scaled = scaler.transform(features_reshaped)
        prediction = model.predict(features_scaled, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction)
        
        print(f"🎯 Predicted: {predicted_digit}")
        print(f"📈 Confidence: {confidence:.3f}")
        
        # Show all predictions
        print("\n📊 All predictions:")
        for i, conf in enumerate(prediction[0]):
            print(f"  Digit {i}: {conf:.3f}")
        
        # Save audio for analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostic_test_{predicted_digit}_{timestamp}.wav"
        sf.write(filename, audio_data, 22050)
        print(f"💾 Saved as: {filename}")
        
    except Exception as e:
        print(f"❌ Error in live test: {e}")

def compare_feature_extraction():
    """Compare feature extraction between training and live"""
    print("\n🔍 Comparing Feature Extraction")
    print("=" * 50)
    
    try:
        # Load a training sample
        training_file = os.path.join('my_voice_samples', '0', '0_1.wav')
        if os.path.exists(training_file):
            training_audio, sr = librosa.load(training_file, sr=22050)
            training_features = extract_robust_features(training_audio, sr)
            print(f"Training sample features shape: {training_features.shape}")
            print(f"Training sample features range: {np.min(training_features):.6f} to {np.max(training_features):.6f}")
            
            # Record a live sample
            print("\nNow record a live sample for comparison...")
            input("Press Enter to record...")
            
            print("🎤 Recording...")
            audio = sd.rec(int(2 * 22050), samplerate=22050, channels=1)
            sd.wait()
            live_audio = audio.flatten()
            
            live_features = extract_robust_features(live_audio)
            if live_features is not None:
                print(f"Live sample features shape: {live_features.shape}")
                print(f"Live sample features range: {np.min(live_features):.6f} to {np.max(live_features):.6f}")
                
                # Compare
                diff = np.abs(training_features - live_features)
                print(f"Feature difference (mean): {np.mean(diff):.6f}")
                print(f"Feature difference (max): {np.max(diff):.6f}")
                
                if np.mean(diff) > 1.0:
                    print("⚠️ Large feature differences detected - this might be the issue!")
                else:
                    print("✅ Feature differences are reasonable")
        
    except Exception as e:
        print(f"❌ Error in feature comparison: {e}")

def main():
    """Main diagnostic function"""
    print("🔍 Voice-Adapted Model Diagnostic")
    print("=" * 60)
    
    # Check if model files exist
    if not os.path.exists('voice_adaptive_model.h5'):
        print("❌ voice_adaptive_model.h5 not found!")
        return
    
    if not os.path.exists('voice_adaptive_scaler.pkl'):
        print("❌ voice_adaptive_scaler.pkl not found!")
        return
    
    if not os.path.exists('voice_adaptive_label_binarizer.pkl'):
        print("❌ voice_adaptive_label_binarizer.pkl not found!")
        return
    
    print("✅ All model files found")
    
    # Run diagnostics
    test_model_on_training_data()
    test_live_recording()
    compare_feature_extraction()
    
    print("\n💡 Recommendations:")
    print("1. If training accuracy is low: Model needs retraining")
    print("2. If training accuracy is high but live accuracy is low: Feature extraction mismatch")
    print("3. If confidence is low: Audio quality or preprocessing issues")
    print("4. If specific digits fail: Voice sample quality issues")

if __name__ == "__main__":
    main() 