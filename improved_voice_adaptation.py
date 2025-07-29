#!/usr/bin/env python3
"""
Improved voice adaptation training with more samples and better parameters
"""

import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import time
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt

class ImprovedVoiceAdaptationTrainer:
    def __init__(self, sample_rate=22050, duration=2):
        self.sample_rate = sample_rate
        self.duration = duration
        self.recordings_dir = "improved_voice_samples"
        self.base_model = None
        
    def setup_directories(self):
        """Create directories for voice samples"""
        if not os.path.exists(self.recordings_dir):
            os.makedirs(self.recordings_dir)
        
        # Create subdirectories for each digit
        for digit in range(10):
            digit_dir = os.path.join(self.recordings_dir, str(digit))
            if not os.path.exists(digit_dir):
                os.makedirs(digit_dir)
    
    def load_base_model(self):
        """Load a base model to fine-tune"""
        model_files = [
            'fixed_digit_classifier_model.h5',
            'deeper_digit_classifier_model.h5',
            'residual_digit_classifier_model.h5',
            'wide_digit_classifier_model.h5'
        ]
        
        print("🤖 Available base models:")
        for i, model_file in enumerate(model_files):
            if os.path.exists(model_file):
                print(f"  {i+1}. {model_file}")
        
        try:
            choice = int(input(f"\nSelect base model (1-{len(model_files)}): ")) - 1
            if 0 <= choice < len(model_files) and os.path.exists(model_files[choice]):
                selected_model = model_files[choice]
            else:
                selected_model = model_files[0]
        except:
            selected_model = model_files[0]
        
        print(f"Loading {selected_model}...")
        self.base_model = keras.models.load_model(selected_model)
        print("✅ Base model loaded")
        
        return selected_model
    
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
    
    def collect_voice_samples(self):
        """Collect more voice samples for each digit"""
        print("🎤 Improved Voice Sample Collection")
        print("=" * 50)
        print("We'll collect 10 samples for each digit (0-9)")
        print("This will provide more data for better adaptation.")
        print()
        
        self.setup_directories()
        
        samples_per_digit = 10  # Increased from 5 to 10
        
        for digit in range(10):
            print(f"\n📝 Recording digit: {digit}")
            print(f"   Need {samples_per_digit} samples")
            
            for sample_num in range(samples_per_digit):
                print(f"\n   Sample {sample_num + 1}/{samples_per_digit}")
                print(f"   Say '{digit}' clearly...")
                
                input("   Press ENTER when ready...")
                
                audio_data = self.record_audio()
                if audio_data is None:
                    print("   ❌ Recording failed, trying again...")
                    sample_num -= 1
                    continue
                
                # Check audio quality
                rms = np.sqrt(np.mean(audio_data**2))
                print(f"   📊 Audio RMS: {rms:.6f}")
                
                if rms < 0.001:
                    print("   🔇 Audio too quiet, trying again...")
                    sample_num -= 1
                    continue
                
                # Save the recording
                filename = f"{digit}_{sample_num + 1}.wav"
                filepath = os.path.join(self.recordings_dir, str(digit), filename)
                sf.write(filepath, audio_data, self.sample_rate)
                
                print(f"   ✅ Saved: {filename}")
                
                # Test prediction with base model
                features = self.extract_features(audio_data)
                if features is not None and self.base_model is not None:
                    prediction = self.base_model.predict(features.reshape(1, -1), verbose=0)
                    predicted_digit = np.argmax(prediction[0])
                    confidence = np.max(prediction[0])
                    print(f"   🤖 Base model predicts: {predicted_digit} ({confidence:.1%})")
        
        print(f"\n✅ Voice sample collection complete!")
        print(f"📁 Samples saved in: {self.recordings_dir}")
    
    def prepare_training_data(self):
        """Prepare training data from collected samples"""
        print("📊 Preparing training data...")
        
        X, y = [], []
        
        for digit in range(10):
            digit_dir = os.path.join(self.recordings_dir, str(digit))
            if not os.path.exists(digit_dir):
                continue
            
            wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
            
            for wav_file in wav_files:
                filepath = os.path.join(digit_dir, wav_file)
                try:
                    features = self.extract_features(librosa.load(filepath, sr=self.sample_rate)[0])
                    if features is not None:
                        X.append(features)
                        y.append(str(digit))
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")
        
        if len(X) == 0:
            print("❌ No valid samples found!")
            return None, None
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Training data: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"📊 Digit distribution: {np.bincount([int(label) for label in y])}")
        
        return X, y
    
    def fine_tune_model(self, X, y):
        """Fine-tune the base model with improved parameters"""
        print("🎯 Fine-tuning model with improved parameters...")
        
        # Encode labels
        lb = LabelBinarizer()
        y_encoded = lb.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create a copy of the base model for fine-tuning
        fine_tuned_model = keras.models.clone_model(self.base_model)
        fine_tuned_model.set_weights(self.base_model.get_weights())
        
        # Use better learning rate and optimizer
        fine_tuned_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Slightly higher learning rate
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Better callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # More patience
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
        
        # Fine-tune the model with more epochs
        history = fine_tuned_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,  # More epochs
            batch_size=16,  # Larger batch size
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = fine_tuned_model.evaluate(X_test, y_test, verbose=0)
        print(f"✅ Fine-tuned model accuracy: {test_acc:.2%}")
        
        # Save the fine-tuned model
        model_path = "improved_voice_adapted_model.h5"
        fine_tuned_model.save(model_path)
        print(f"💾 Improved fine-tuned model saved as: {model_path}")
        
        return fine_tuned_model, history
    
    def run_improved_adaptation_training(self):
        """Run the improved voice adaptation training process"""
        print("🎤 Improved Voice Adaptation Training")
        print("=" * 50)
        print("This will collect more samples and use better training parameters.")
        print()
        
        # Load base model
        self.load_base_model()
        
        # Collect voice samples
        self.collect_voice_samples()
        
        # Prepare training data
        X, y = self.prepare_training_data()
        if X is None:
            print("❌ No training data available!")
            return
        
        # Fine-tune model
        fine_tuned_model, history = self.fine_tune_model(X, y)
        
        print("\n🎉 Improved voice adaptation training complete!")
        print("You can now use improved_voice_adapted_model.h5 for better accuracy.")

def main():
    """Main function"""
    trainer = ImprovedVoiceAdaptationTrainer()
    trainer.run_improved_adaptation_training()

if __name__ == "__main__":
    main() 