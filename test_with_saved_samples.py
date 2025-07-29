#!/usr/bin/env python3
"""
Test voice-adapted model using saved voice samples
"""

import os
import numpy as np
import librosa
from tensorflow import keras
import random

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

def test_original_model(audio_data, expected_digit):
    """Test the original model for comparison"""
    features = extract_features(audio_data)
    if features is None:
        return None
    
    try:
        # Load original model
        model = keras.models.load_model('residual_digit_classifier_model.h5')
        
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
        print(f"❌ Error with original model: {e}")
        return None

def main():
    """Main test function"""
    print("🎤 Testing Voice-Adapted Model with Saved Samples")
    print("=" * 60)
    print("This will test the voice-adapted model on your recorded voice samples.")
    print()
    
    # Check if voice samples exist
    if not os.path.exists('user_voice_samples'):
        print("❌ Voice samples not found!")
        print("Please run voice_adaptation_training.py first.")
        return
    
    # Load voice-adapted model
    try:
        model = keras.models.load_model('voice_adapted_model.h5')
        print("✅ Voice-adapted model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test each digit
    print("\n🧪 Testing each digit...")
    print("-" * 60)
    
    total_tests = 0
    correct_predictions = 0
    
    for digit in range(10):
        digit_dir = os.path.join('user_voice_samples', str(digit))
        if not os.path.exists(digit_dir):
            continue
        
        wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
        if not wav_files:
            continue
        
        print(f"\n📝 Testing digit: {digit}")
        print(f"   Samples available: {len(wav_files)}")
        
        digit_correct = 0
        digit_total = 0
        
        # Test each sample for this digit
        for wav_file in wav_files:
            filepath = os.path.join(digit_dir, wav_file)
            
            try:
                # Load audio
                audio_data, sr = librosa.load(filepath, sr=22050)
                
                # Test voice-adapted model
                result = test_voice_adapted_model(audio_data, digit)
                
                if result is not None:
                    digit_total += 1
                    total_tests += 1
                    
                    if result['correct']:
                        digit_correct += 1
                        correct_predictions += 1
                        status = "✅"
                    else:
                        status = "❌"
                    
                    print(f"   {status} Sample {wav_file}: {result['predicted_digit']} ({result['confidence']:.1%})")
                
            except Exception as e:
                print(f"   ❌ Error processing {wav_file}: {e}")
        
        # Show digit summary
        if digit_total > 0:
            accuracy = (digit_correct / digit_total) * 100
            print(f"   📊 Digit {digit} accuracy: {accuracy:.1f}% ({digit_correct}/{digit_total})")
    
    # Show overall results
    print("\n" + "=" * 60)
    print("📊 OVERALL RESULTS:")
    
    if total_tests > 0:
        overall_accuracy = (correct_predictions / total_tests) * 100
        print(f"🎯 Overall Accuracy: {overall_accuracy:.1f}% ({correct_predictions}/{total_tests})")
        
        if overall_accuracy > 80:
            print("🎉 Excellent! The voice adaptation worked well!")
        elif overall_accuracy > 60:
            print("👍 Good! The voice adaptation shows improvement.")
        elif overall_accuracy > 40:
            print("⚠️  Fair. The voice adaptation needs more work.")
        else:
            print("❌ Poor. The voice adaptation didn't work as expected.")
    else:
        print("❌ No tests completed")
    
    print("\n💡 Next Steps:")
    print("1. If accuracy is low, try collecting more voice samples")
    print("2. If accuracy is good, the model should work well with your voice")
    print("3. Fix microphone permissions to test live recording")

if __name__ == "__main__":
    main() 