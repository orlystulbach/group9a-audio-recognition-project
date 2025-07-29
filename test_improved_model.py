#!/usr/bin/env python3
"""
Test the improved voice-adapted model
"""

import os
import numpy as np
import librosa
from tensorflow import keras

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

def test_improved_model(audio_data, expected_digit):
    """Test the improved voice-adapted model"""
    features = extract_features(audio_data)
    if features is None:
        return None
    
    try:
        # Load improved voice-adapted model
        model = keras.models.load_model('improved_voice_adapted_model.h5')
        
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
        print(f"❌ Error with improved model: {e}")
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
    print("🎤 Testing Improved Voice-Adapted Model")
    print("=" * 60)
    print("This will test the improved model on your new voice samples.")
    print()
    
    # Check if improved model exists
    if not os.path.exists('improved_voice_adapted_model.h5'):
        print("❌ Improved voice-adapted model not found!")
        print("Please run improved_voice_adaptation.py first.")
        return
    
    # Check if improved voice samples exist
    if not os.path.exists('improved_voice_samples'):
        print("❌ Improved voice samples not found!")
        print("Please run improved_voice_adaptation.py first.")
        return
    
    # Load improved model
    try:
        model = keras.models.load_model('improved_voice_adapted_model.h5')
        print("✅ Improved voice-adapted model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test each digit
    print("\n🧪 Testing each digit...")
    print("-" * 60)
    
    total_tests = 0
    correct_predictions_improved = 0
    correct_predictions_original = 0
    
    for digit in range(10):
        digit_dir = os.path.join('improved_voice_samples', str(digit))
        if not os.path.exists(digit_dir):
            continue
        
        wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
        if not wav_files:
            continue
        
        print(f"\n📝 Testing digit: {digit}")
        print(f"   Samples available: {len(wav_files)}")
        
        digit_correct_improved = 0
        digit_correct_original = 0
        digit_total = 0
        
        # Test each sample for this digit
        for wav_file in wav_files:
            filepath = os.path.join(digit_dir, wav_file)
            
            try:
                # Load audio
                audio_data, sr = librosa.load(filepath, sr=22050)
                
                # Test improved model
                result_improved = test_improved_model(audio_data, digit)
                result_original = test_original_model(audio_data, digit)
                
                if result_improved is not None and result_original is not None:
                    digit_total += 1
                    total_tests += 1
                    
                    # Check improved model
                    if result_improved['correct']:
                        digit_correct_improved += 1
                        correct_predictions_improved += 1
                        status_improved = "✅"
                    else:
                        status_improved = "❌"
                    
                    # Check original model
                    if result_original['correct']:
                        digit_correct_original += 1
                        correct_predictions_original += 1
                        status_original = "✅"
                    else:
                        status_original = "❌"
                    
                    print(f"   {status_improved} Improved: {result_improved['predicted_digit']} ({result_improved['confidence']:.1%}) | {status_original} Original: {result_original['predicted_digit']} ({result_original['confidence']:.1%})")
                
            except Exception as e:
                print(f"   ❌ Error processing {wav_file}: {e}")
        
        # Show digit summary
        if digit_total > 0:
            accuracy_improved = (digit_correct_improved / digit_total) * 100
            accuracy_original = (digit_correct_original / digit_total) * 100
            print(f"   📊 Digit {digit}: Improved {accuracy_improved:.1f}% | Original {accuracy_original:.1f}%")
    
    # Show overall results
    print("\n" + "=" * 60)
    print("📊 OVERALL RESULTS:")
    
    if total_tests > 0:
        overall_accuracy_improved = (correct_predictions_improved / total_tests) * 100
        overall_accuracy_original = (correct_predictions_original / total_tests) * 100
        
        print(f"🎯 Improved Model Accuracy: {overall_accuracy_improved:.1f}% ({correct_predictions_improved}/{total_tests})")
        print(f"🎯 Original Model Accuracy: {overall_accuracy_original:.1f}% ({correct_predictions_original}/{total_tests})")
        
        improvement = overall_accuracy_improved - overall_accuracy_original
        print(f"🚀 Improvement: +{improvement:.1f} percentage points")
        
        if overall_accuracy_improved > 80:
            print("🎉 Excellent! The improved voice adaptation worked very well!")
        elif overall_accuracy_improved > 60:
            print("👍 Good! The improved voice adaptation shows significant improvement.")
        elif overall_accuracy_improved > 40:
            print("⚠️  Fair. The improved voice adaptation needs more work.")
        else:
            print("❌ Poor. The improved voice adaptation didn't work as expected.")
    else:
        print("❌ No tests completed")
    
    print("\n💡 Next Steps:")
    print("1. If accuracy is good (>70%), the model should work well with your voice")
    print("2. Try the improved model with live speech input")
    print("3. If needed, collect even more samples for further improvement")

if __name__ == "__main__":
    main() 