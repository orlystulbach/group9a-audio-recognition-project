#!/usr/bin/env python3
"""
Demonstration of speech input functionality using a sample audio file
"""

import os
import numpy as np
import librosa
from tensorflow import keras

def demo_with_sample_audio():
    """Demonstrate the speech input functionality using a sample audio file"""
    print("🎤 Speech Input Demo (using sample audio)")
    print("=" * 50)
    
    # Find a sample audio file
    data_dir = 'large_dataset'
    if not os.path.exists(data_dir):
        print("❌ large_dataset directory not found")
        return
    
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    if not wav_files:
        print("❌ No audio files found in large_dataset")
        return
    
    # Use the first audio file as a sample
    sample_file = os.path.join(data_dir, wav_files[0])
    print(f"📁 Using sample file: {wav_files[0]}")
    
    # Extract the expected digit from filename
    expected_digit = wav_files[0].split('_')[0]
    print(f"🎯 Expected digit: {expected_digit}")
    
    try:
        # Load the model
        print("\n🤖 Loading model...")
        model = keras.models.load_model('fixed_digit_classifier_model.h5')
        print("✅ Model loaded successfully")
        
        # Load and process audio (same as speech_input.py)
        print("\n🎵 Processing audio...")
        y, sr = librosa.load(sample_file, sr=22050)
        print(f"   Audio loaded: {len(y)} samples, {sr} Hz")
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Extract features (same as speech_input.py)
        print("   Extracting features...")
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        features = []
        features.extend(np.mean(mfcc, axis=1))  # Mean of each coefficient
        features.extend(np.std(mfcc, axis=1))   # Std of each coefficient
        features.extend(np.mean(mfcc_delta, axis=1))  # Delta features
        features.extend(np.mean(mfcc_delta2, axis=1)) # Delta-delta features
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=512)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=512)[0]
        
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
        features.extend(np.mean(chroma, axis=1))  # Mean of each chroma bin
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # RMS energy
        rms = librosa.feature.rms(y=y, hop_length=512)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        features = np.array(features)
        print(f"   Features extracted: {features.shape}")
        
        # Check audio quality
        audio_rms = np.sqrt(np.mean(y**2))
        print(f"   Audio RMS: {audio_rms:.4f}")
        
        if audio_rms < 0.01:
            print("   ⚠️  Audio is quiet (this would trigger 'speak louder' in live mode)")
        else:
            print("   ✅ Audio level is good")
        
        # Make prediction
        print("\n🔮 Making prediction...")
        features_reshaped = features.reshape(1, -1)
        prediction = model.predict(features_reshaped, verbose=0)
        predicted_digit = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        # Show results
        print("\n📊 Results:")
        print(f"   Expected digit: {expected_digit}")
        print(f"   Predicted digit: {predicted_digit}")
        print(f"   Confidence: {confidence:.2%}")
        
        if str(predicted_digit) == str(expected_digit):
            print("   ✅ Prediction is correct!")
        else:
            print("   ❌ Prediction is incorrect")
        
        if confidence > 0.7:
            print("   🎯 High confidence prediction!")
        elif confidence > 0.5:
            print("   ⚠️  Medium confidence prediction")
        else:
            print("   ❓ Low confidence prediction")
        
        # Show all class probabilities
        print("\n📈 All digit probabilities:")
        for i, prob in enumerate(prediction[0]):
            marker = "🎯" if i == predicted_digit else "  "
            print(f"   {marker} Digit {i}: {prob:.2%}")
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("This shows how the speech_input.py script works.")
        print("To use it with your microphone, run: python3 speech_input.py")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")

def main():
    """Run the demonstration"""
    demo_with_sample_audio()

if __name__ == "__main__":
    main() 