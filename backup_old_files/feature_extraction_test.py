#!/usr/bin/env python3
"""
Feature Extraction Diagnostic - Identify preprocessing mismatches
"""

import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from datetime import datetime

def extract_features_with_noise_reduction(audio_data, sample_rate=22050, n_mfcc=13):
    """Feature extraction WITH noise reduction (like the original app)"""
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
        print(f"Error in noise reduction extraction: {e}")
        return None

def extract_features_without_noise_reduction(audio_data, sample_rate=22050, n_mfcc=13):
    """Feature extraction WITHOUT noise reduction (like training)"""
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
        print(f"Error in simple extraction: {e}")
        return None

def compare_training_vs_live_features():
    """Compare features from training samples vs live recording"""
    print("🔍 Comparing Training vs Live Feature Extraction")
    print("=" * 60)
    
    # Load a training sample
    training_file = os.path.join('my_voice_samples', '0', '0_1.wav')
    if not os.path.exists(training_file):
        print(f"❌ Training file not found: {training_file}")
        return
    
    print(f"📁 Loading training sample: {training_file}")
    training_audio, sr = librosa.load(training_file, sr=22050)
    
    # Extract features from training sample
    training_features_noise = extract_features_with_noise_reduction(training_audio, sr)
    training_features_simple = extract_features_without_noise_reduction(training_audio, sr)
    
    print(f"Training sample features shape: {training_features_simple.shape}")
    print(f"Training sample features range: {np.min(training_features_simple):.6f} to {np.max(training_features_simple):.6f}")
    
    # Record a live sample
    print("\n🎤 Now record a live sample for comparison...")
    input("Press Enter to record...")
    
    print("🎤 Recording...")
    audio = sd.rec(int(2 * 22050), samplerate=22050, channels=1)
    sd.wait()
    live_audio = audio.flatten()
    
    # Check audio quality
    rms = np.sqrt(np.mean(live_audio**2))
    print(f"Live audio RMS: {rms:.6f}")
    
    if rms < 0.01:
        print("⚠️ Live audio too quiet!")
        return
    
    # Extract features from live sample
    live_features_noise = extract_features_with_noise_reduction(live_audio)
    live_features_simple = extract_features_without_noise_reduction(live_audio)
    
    if live_features_simple is None:
        print("❌ Failed to extract live features")
        return
    
    print(f"Live sample features shape: {live_features_simple.shape}")
    print(f"Live sample features range: {np.min(live_features_simple):.6f} to {np.max(live_features_simple):.6f}")
    
    # Compare simple features (training vs live)
    if training_features_simple is not None and live_features_simple is not None:
        diff_simple = np.abs(training_features_simple - live_features_simple)
        print(f"\n📊 Simple Feature Comparison (Training vs Live):")
        print(f"  Mean difference: {np.mean(diff_simple):.6f}")
        print(f"  Max difference: {np.max(diff_simple):.6f}")
        print(f"  Std difference: {np.std(diff_simple):.6f}")
        
        if np.mean(diff_simple) > 1.0:
            print("⚠️ LARGE differences detected in simple features!")
        else:
            print("✅ Simple feature differences are reasonable")
    
    # Compare noise reduction vs simple
    if live_features_noise is not None and live_features_simple is not None:
        diff_noise = np.abs(live_features_noise - live_features_simple)
        print(f"\n📊 Noise Reduction vs Simple (Live Sample):")
        print(f"  Mean difference: {np.mean(diff_noise):.6f}")
        print(f"  Max difference: {np.max(diff_noise):.6f}")
        print(f"  Std difference: {np.std(diff_noise):.6f}")
        
        if np.mean(diff_noise) > 0.5:
            print("⚠️ Noise reduction significantly changes features!")
        else:
            print("✅ Noise reduction has minimal effect")
    
    # Save live audio for analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"feature_test_live_{timestamp}.wav"
    sf.write(filename, live_audio, 22050)
    print(f"💾 Saved live sample as: {filename}")
    
    return training_features_simple, live_features_simple, live_features_noise

def test_multiple_training_samples():
    """Test multiple training samples to see consistency"""
    print("\n📊 Testing Multiple Training Samples")
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
                    
                    features = extract_features_without_noise_reduction(audio_data, sr)
                    if features is not None:
                        all_features.append(features)
                        print(f"✅ {wav_file}: Features extracted")
                    else:
                        print(f"❌ {wav_file}: Failed to extract features")
                        
                except Exception as e:
                    print(f"❌ Error processing {wav_file}: {e}")
    
    if len(all_features) > 1:
        all_features = np.array(all_features)
        print(f"\n📊 Training Sample Statistics:")
        print(f"  Number of samples: {len(all_features)}")
        print(f"  Feature shape: {all_features.shape}")
        print(f"  Mean across samples: {np.mean(all_features, axis=0)}")
        print(f"  Std across samples: {np.std(all_features, axis=0)}")
        print(f"  Feature range: {np.min(all_features):.6f} to {np.max(all_features):.6f}")

def analyze_feature_distributions():
    """Analyze feature distributions to identify issues"""
    print("\n📈 Analyzing Feature Distributions")
    print("=" * 50)
    
    # Collect features from training samples
    training_features = []
    for digit in range(10):
        digit_dir = os.path.join('my_voice_samples', str(digit))
        if os.path.exists(digit_dir):
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            
            for wav_file in wav_files[:3]:  # 3 samples per digit
                try:
                    file_path = os.path.join(digit_dir, wav_file)
                    audio_data, sr = librosa.load(file_path, sr=22050)
                    
                    features = extract_features_without_noise_reduction(audio_data, sr)
                    if features is not None:
                        training_features.append(features)
                        
                except Exception as e:
                    continue
    
    if len(training_features) > 0:
        training_features = np.array(training_features)
        
        print(f"Training features shape: {training_features.shape}")
        print(f"Feature statistics:")
        print(f"  Mean: {np.mean(training_features, axis=0)}")
        print(f"  Std: {np.std(training_features, axis=0)}")
        print(f"  Min: {np.min(training_features, axis=0)}")
        print(f"  Max: {np.max(training_features, axis=0)}")
        
        # Check for problematic features
        feature_means = np.mean(training_features, axis=0)
        feature_stds = np.std(training_features, axis=0)
        
        print(f"\n🔍 Feature Analysis:")
        for i, (mean, std) in enumerate(zip(feature_means, feature_stds)):
            if std < 0.001:
                print(f"  Feature {i}: Very low variance (std={std:.6f}) - might be problematic")
            elif std > 10:
                print(f"  Feature {i}: Very high variance (std={std:.6f}) - might be noisy")
            elif abs(mean) > 100:
                print(f"  Feature {i}: Very large mean ({mean:.6f}) - might need scaling")

def main():
    """Main diagnostic function"""
    print("🔍 Feature Extraction Diagnostic")
    print("=" * 60)
    
    # Check if voice samples exist
    if not os.path.exists('my_voice_samples'):
        print("❌ my_voice_samples directory not found!")
        return
    
    print("✅ Voice samples directory found")
    
    # Run diagnostics
    compare_training_vs_live_features()
    test_multiple_training_samples()
    analyze_feature_distributions()
    
    print("\n💡 Feature Extraction Recommendations:")
    print("1. If training vs live differences are large: Preprocessing mismatch")
    print("2. If noise reduction changes features significantly: Use simple extraction")
    print("3. If feature variances are very low/high: Scaling or feature selection issues")
    print("4. If specific features are problematic: Consider feature engineering")

if __name__ == "__main__":
    main() 