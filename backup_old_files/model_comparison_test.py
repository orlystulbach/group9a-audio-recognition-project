#!/usr/bin/env python3
"""
Test all available models with user's voice to find the best one
"""

import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import time
from tensorflow import keras

class ModelComparisonTester:
    def __init__(self, sample_rate=22050, duration=2):
        self.sample_rate = sample_rate
        self.duration = duration
        self.models = {}
        
    def load_models(self):
        """Load all available models"""
        model_files = [
            'fixed_digit_classifier_model.h5',
            'deeper_digit_classifier_model.h5',
            'residual_digit_classifier_model.h5',
            'wide_digit_classifier_model.h5'
        ]
        
        print("🤖 Loading models...")
        for model_file in model_files:
            if os.path.exists(model_file):
                try:
                    model = keras.models.load_model(model_file)
                    self.models[model_file] = model
                    print(f"✅ {model_file}: Loaded")
                except Exception as e:
                    print(f"❌ {model_file}: Error - {e}")
        
        print(f"📊 Loaded {len(self.models)} models")
        return len(self.models) > 0
    
    def extract_features(self, audio_data):
        """Extract features from audio data"""
        try:
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            features = []
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Statistical features from MFCC
            features.extend(np.mean(mfcc, axis=1))
            features.extend(np.std(mfcc, axis=1))
            features.extend(np.mean(mfcc_delta, axis=1))
            features.extend(np.mean(mfcc_delta2, axis=1))
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate, hop_length=512)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate, hop_length=512)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate, hop_length=512)[0]
            
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
            features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate, hop_length=512)
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
    
    def record_audio(self):
        """Record audio from microphone"""
        print(f"🎤 Recording for {self.duration} seconds...")
        
        try:
            audio_data = sd.rec(int(self.duration * self.sample_rate), 
                               samplerate=self.sample_rate, 
                               channels=1, 
                               dtype=np.float32)
            sd.wait()
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None
    
    def test_all_models(self, audio_data, expected_digit):
        """Test all models with the same audio"""
        features = self.extract_features(audio_data)
        if features is None:
            return {}
        
        results = {}
        features_reshaped = features.reshape(1, -1)
        
        for model_name, model in self.models.items():
            try:
                prediction = model.predict(features_reshaped, verbose=0)
                predicted_digit = np.argmax(prediction[0])
                confidence = np.max(prediction[0])
                
                # Get all probabilities
                all_probs = prediction[0]
                
                results[model_name] = {
                    'predicted_digit': predicted_digit,
                    'confidence': confidence,
                    'all_probabilities': all_probs,
                    'correct': str(predicted_digit) == str(expected_digit)
                }
                
            except Exception as e:
                print(f"❌ Error with {model_name}: {e}")
                results[model_name] = None
        
        return results
    
    def run_comparison_test(self):
        """Run the model comparison test"""
        print("🎤 Model Comparison Test")
        print("=" * 50)
        print("This will test all models with your voice.")
        print("Speak a digit clearly and we'll compare predictions.")
        print()
        
        if not self.load_models():
            print("❌ No models loaded. Cannot run test.")
            return
        
        while True:
            try:
                # Get expected digit from user
                expected_digit = input("Enter the digit you'll speak (0-9, or 'quit'): ").strip()
                
                if expected_digit.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if not expected_digit.isdigit() or int(expected_digit) < 0 or int(expected_digit) > 9:
                    print("❌ Please enter a digit 0-9")
                    continue
                
                print(f"\n🎯 You'll speak: {expected_digit}")
                print("Press ENTER when ready to record...")
                input()
                
                # Record audio
                audio_data = self.record_audio()
                if audio_data is None:
                    continue
                
                # Check audio quality
                rms = np.sqrt(np.mean(audio_data**2))
                print(f"📊 Audio RMS: {rms:.6f}")
                
                if rms < 0.001:
                    print("🔇 Audio too quiet. Please speak louder.")
                    continue
                
                # Test all models
                print("\n🤖 Testing all models...")
                results = self.test_all_models(audio_data, expected_digit)
                
                # Display results
                print("\n📊 Results:")
                print("-" * 50)
                
                for model_name, result in results.items():
                    if result is None:
                        print(f"❌ {model_name}: Error")
                        continue
                    
                    model_short = model_name.replace('_digit_classifier_model.h5', '')
                    status = "✅" if result['correct'] else "❌"
                    
                    print(f"{status} {model_short}:")
                    print(f"   Predicted: {result['predicted_digit']} (Expected: {expected_digit})")
                    print(f"   Confidence: {result['confidence']:.2%}")
                    
                    # Show top 3 predictions
                    probs = result['all_probabilities']
                    top_indices = np.argsort(probs)[-3:][::-1]
                    print("   Top 3 predictions:")
                    for i, idx in enumerate(top_indices):
                        marker = "🎯" if i == 0 else "  "
                        print(f"     {marker} Digit {idx}: {probs[idx]:.2%}")
                    print()
                
                # Save audio for later analysis
                save_choice = input("Save this recording? (y/n): ").strip().lower()
                if save_choice == 'y':
                    filename = f"test_digit_{expected_digit}_{int(time.time())}.wav"
                    sf.write(filename, audio_data, self.sample_rate)
                    print(f"💾 Saved as {filename}")
                
                print("-" * 50)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function"""
    tester = ModelComparisonTester()
    tester.run_comparison_test()

if __name__ == "__main__":
    main() 