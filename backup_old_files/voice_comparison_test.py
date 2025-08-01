#!/usr/bin/env python3
"""
Voice Comparison Test - Compare live voice with training samples
"""

import os
import numpy as np
import librosa
import tensorflow as tf
import joblib
import sounddevice as sd
import soundfile as sf
from datetime import datetime

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

def record_live_sample():
    """Record a live voice sample"""
    print("🎤 Recording live sample...")
    print("Speak a digit clearly (0-9)")
    input("Press Enter to start recording...")
    
    # Record audio
    audio = sd.rec(int(2 * 22050), samplerate=22050, channels=1)
    sd.wait()
    audio_data = audio.flatten()
    
    # Check audio quality
    rms = np.sqrt(np.mean(audio_data**2))
    print(f"Live audio RMS: {rms:.6f}")
    
    if rms < 0.01:
        print("⚠️ Audio too quiet!")
        return None
    
    return audio_data

def compare_with_training_samples(live_audio, target_digit):
    """Compare live audio with training samples for the same digit"""
    print(f"\n🔍 Comparing with training samples for digit {target_digit}")
    print("=" * 60)
    
    try:
        # Load model
        model = tf.keras.models.load_model('fixed_voice_model.h5')
        scaler = joblib.load('fixed_voice_scaler.pkl')
        
        # Extract features from live audio
        live_features = extract_simple_features(live_audio)
        if live_features is None:
            print("❌ Failed to extract live features")
            return
        
        live_features_scaled = scaler.transform(live_features.reshape(1, -1))
        live_prediction = model.predict(live_features_scaled, verbose=0)
        live_predicted_digit = np.argmax(live_prediction)
        live_confidence = np.max(live_prediction)
        
        print(f"Live prediction: {live_predicted_digit} (confidence: {live_confidence:.3f})")
        
        # Load training samples for the same digit
        digit_dir = os.path.join('my_voice_samples', str(target_digit))
        if not os.path.exists(digit_dir):
            print(f"❌ No training samples for digit {target_digit}")
            return
        
        wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
        
        print(f"\n📊 Comparing with {len(wav_files)} training samples:")
        
        training_predictions = []
        feature_differences = []
        
        for i, wav_file in enumerate(wav_files[:5]):  # Compare with first 5 training samples
            try:
                file_path = os.path.join(digit_dir, wav_file)
                training_audio, sr = librosa.load(file_path, sr=22050)
                
                training_features = extract_simple_features(training_audio, sr)
                if training_features is not None:
                    training_features_scaled = scaler.transform(training_features.reshape(1, -1))
                    training_prediction = model.predict(training_features_scaled, verbose=0)
                    training_predicted_digit = np.argmax(training_prediction)
                    training_confidence = np.max(training_prediction)
                    
                    training_predictions.append(training_predicted_digit)
                    
                    # Calculate feature difference
                    feature_diff = np.mean(np.abs(live_features - training_features))
                    feature_differences.append(feature_diff)
                    
                    print(f"  Training {i+1}: Predicted {training_predicted_digit} (confidence: {training_confidence:.3f})")
                    print(f"    Feature difference: {feature_diff:.6f}")
                    
            except Exception as e:
                print(f"  Error processing {wav_file}: {e}")
        
        # Analysis
        print(f"\n📈 Analysis:")
        print(f"  Live prediction: {live_predicted_digit} (confidence: {live_confidence:.3f})")
        print(f"  Training predictions: {training_predictions}")
        print(f"  Average feature difference: {np.mean(feature_differences):.6f}")
        
        if live_predicted_digit == target_digit:
            print("✅ Live prediction is correct!")
        else:
            print(f"❌ Live prediction is wrong (should be {target_digit})")
            
        if np.mean(feature_differences) > 100:
            print("⚠️ Large feature differences detected - this might be the issue!")
        else:
            print("✅ Feature differences are reasonable")
            
        # Show all predictions for live audio
        print(f"\n📊 All predictions for live audio:")
        for i, conf in enumerate(live_prediction[0]):
            print(f"  Digit {i}: {conf:.3f}")
        
        return live_predicted_digit, live_confidence, np.mean(feature_differences)
        
    except Exception as e:
        print(f"❌ Error in comparison: {e}")
        return None, None, None

def test_multiple_digits():
    """Test multiple digits to see patterns"""
    print("🧪 Testing Multiple Digits")
    print("=" * 60)
    
    results = []
    
    for digit in range(5):  # Test first 5 digits
        print(f"\n🎤 Testing digit {digit}")
        print("-" * 30)
        
        live_audio = record_live_sample()
        if live_audio is not None:
            predicted_digit, confidence, feature_diff = compare_with_training_samples(live_audio, digit)
            
            if predicted_digit is not None:
                results.append({
                    'target': digit,
                    'predicted': predicted_digit,
                    'confidence': confidence,
                    'correct': predicted_digit == digit,
                    'feature_diff': feature_diff
                })
                
                # Save audio for analysis
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"test_digit_{digit}_{predicted_digit}_{timestamp}.wav"
                sf.write(filename, live_audio, 22050)
                print(f"💾 Saved as: {filename}")
        
        print("\n" + "="*50)
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"Total tests: {len(results)}")
    correct = sum(1 for r in results if r['correct'])
    print(f"Correct predictions: {correct}/{len(results)} ({correct/len(results)*100:.1f}%)")
    
    if len(results) > 0:
        avg_confidence = np.mean([r['confidence'] for r in results])
        avg_feature_diff = np.mean([r['feature_diff'] for r in results])
        print(f"Average confidence: {avg_confidence:.3f}")
        print(f"Average feature difference: {avg_feature_diff:.6f}")
        
        # Identify patterns
        print(f"\n🔍 Pattern Analysis:")
        for r in results:
            if not r['correct']:
                print(f"  Digit {r['target']} → Predicted {r['predicted']} (confidence: {r['confidence']:.3f})")
    
    return results

def main():
    """Main function"""
    print("🎤 Voice Comparison Test")
    print("=" * 60)
    print("This test will help identify why live voice predictions differ from training")
    
    # Check if model exists
    if not os.path.exists('fixed_voice_model.h5'):
        print("❌ fixed_voice_model.h5 not found!")
        return
    
    print("✅ Model found")
    
    # Run tests
    results = test_multiple_digits()
    
    print(f"\n💡 Recommendations:")
    if len(results) > 0:
        accuracy = sum(1 for r in results if r['correct']) / len(results)
        if accuracy < 0.5:
            print("❌ Low accuracy - Consider:")
            print("  1. Recording in quieter environment")
            print("  2. Speaking more clearly and consistently")
            print("  3. Using same microphone as training")
            print("  4. Collecting new training samples")
        else:
            print("✅ Reasonable accuracy - Minor improvements possible")
    else:
        print("⚠️ No tests completed - check audio recording")

if __name__ == "__main__":
    main() 