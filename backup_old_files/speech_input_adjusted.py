import os
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import tempfile
import time
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import threading
import queue

class SpeechInputClassifier:
    def __init__(self, model_path='fixed_digit_classifier_model.h5', sample_rate=22050, duration=2):
        """
        Initialize the speech input classifier with adjusted settings
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.model = None
        self.scaler = None
        self.is_recording = False
        self.audio_queue = queue.Queue()
        
        # Load the trained model
        try:
            self.model = keras.models.load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please make sure you have a trained model file available.")
            return
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
    def extract_features(self, audio_data):
        """
        Extract features from audio data using the same method as training
        """
        try:
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            features = []
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            # Statistical features from MFCC
            features.extend(np.mean(mfcc, axis=1))  # Mean of each coefficient
            features.extend(np.std(mfcc, axis=1))   # Std of each coefficient
            features.extend(np.mean(mfcc_delta, axis=1))  # Delta features
            features.extend(np.mean(mfcc_delta2, axis=1)) # Delta-delta features
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate, hop_length=512)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate, hop_length=512)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate, hop_length=512)[0]
            
            features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
            features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
            features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate, hop_length=512)
            features.extend(np.mean(chroma, axis=1))  # Mean of each chroma bin
            
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
        """
        Record audio from microphone with better error handling
        """
        print(f"🎤 Recording for {self.duration} seconds...")
        
        try:
            # Record audio with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    audio_data = sd.rec(int(self.duration * self.sample_rate), 
                                       samplerate=self.sample_rate, 
                                       channels=1, 
                                       dtype=np.float32)
                    sd.wait()  # Wait until recording is finished
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"   Retry {attempt + 1}/{max_retries}...")
                        time.sleep(0.5)
                    else:
                        raise e
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            return audio_data
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None
    
    def predict_digit(self, audio_data):
        """
        Predict digit from audio data
        """
        try:
            # Extract features
            features = self.extract_features(audio_data)
            if features is None:
                return None, 0.0
            
            # Reshape for model input
            features = features.reshape(1, -1)
            
            # Make prediction
            prediction = self.model.predict(features, verbose=0)
            predicted_digit = np.argmax(prediction[0])
            confidence = np.max(prediction[0])
            
            return predicted_digit, confidence
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, 0.0
    
    def interactive_mode(self):
        """
        Run interactive mode with adjusted thresholds
        """
        if self.model is None:
            print("❌ Model not loaded. Cannot start interactive mode.")
            return
        
        print("\n🎤 Interactive Digit Recognition Mode (Adjusted)")
        print("=" * 50)
        print("Commands:")
        print("  - Press ENTER to record and predict")
        print("  - Type 'quit' to exit")
        print("  - Type 'save' to save the last recording")
        print("  - Type 'info' to see audio levels")
        print("=" * 50)
        
        last_audio = None
        
        while True:
            try:
                user_input = input("\nPress ENTER to record (or type 'quit'/'save'/'info'): ").strip().lower()
                
                if user_input == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input == 'save' and last_audio is not None:
                    filename = f"recorded_digit_{int(time.time())}.wav"
                    sf.write(filename, last_audio, self.sample_rate)
                    print(f"💾 Audio saved as {filename}")
                    continue
                elif user_input == 'info':
                    print("📊 Audio Thresholds:")
                    print("   Very quiet: RMS < 0.001")
                    print("   Low: RMS 0.001 - 0.005")
                    print("   Acceptable: RMS 0.005 - 0.01")
                    print("   Good: RMS > 0.01")
                    continue
                elif user_input == '':
                    # Record audio
                    audio_data = self.record_audio()
                    if audio_data is None:
                        continue
                    
                    last_audio = audio_data
                    
                    # Check if audio is too quiet (adjusted threshold)
                    rms = np.sqrt(np.mean(audio_data**2))
                    print(f"📊 Audio RMS: {rms:.6f}")
                    
                    if rms < 0.001:
                        print("🔇 Audio too quiet. Please speak louder or check microphone.")
                        continue
                    elif rms < 0.005:
                        print("⚠️  Audio is low but trying to process...")
                    else:
                        print("✅ Audio level is good")
                    
                    # Predict digit
                    predicted_digit, confidence = self.predict_digit(audio_data)
                    
                    if predicted_digit is not None:
                        print(f"🎯 Predicted digit: {predicted_digit}")
                        print(f"📊 Confidence: {confidence:.2%}")
                        
                        if confidence > 0.7:
                            print("✅ High confidence prediction!")
                        elif confidence > 0.5:
                            print("⚠️  Medium confidence prediction")
                        else:
                            print("❓ Low confidence prediction")
                    else:
                        print("❌ Could not process audio")
                        
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def single_recording_mode(self):
        """
        Record once and predict with adjusted thresholds
        """
        if self.model is None:
            print("❌ Model not loaded.")
            return
        
        print(f"🎤 Recording for {self.duration} seconds...")
        audio_data = self.record_audio()
        
        if audio_data is None:
            return
        
        # Check audio quality
        rms = np.sqrt(np.mean(audio_data**2))
        print(f"📊 Audio RMS: {rms:.6f}")
        
        if rms < 0.001:
            print("🔇 Audio too quiet. Please speak louder.")
            return
        elif rms < 0.005:
            print("⚠️  Audio is low but trying to process...")
        else:
            print("✅ Audio level is good")
        
        # Predict
        predicted_digit, confidence = self.predict_digit(audio_data)
        
        if predicted_digit is not None:
            print(f"🎯 Predicted digit: {predicted_digit}")
            print(f"📊 Confidence: {confidence:.2%}")
        else:
            print("❌ Could not process audio")

def main():
    """
    Main function to run the adjusted speech input classifier
    """
    print("🎤 Speech Input Digit Classifier (Adjusted)")
    print("=" * 50)
    
    # Check for model files
    model_files = [
        'fixed_digit_classifier_model.h5',
        'deeper_digit_classifier_model.h5',
        'residual_digit_classifier_model.h5',
        'wide_digit_classifier_model.h5'
    ]
    
    available_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            available_models.append(model_file)
    
    if not available_models:
        print("❌ No trained model found!")
        print("Please train a model first using one of the training scripts.")
        return
    
    # Let user choose model
    print("Available models:")
    for i, model in enumerate(available_models):
        print(f"  {i+1}. {model}")
    
    try:
        choice = int(input(f"\nSelect model (1-{len(available_models)}): ")) - 1
        if 0 <= choice < len(available_models):
            selected_model = available_models[choice]
        else:
            selected_model = available_models[0]
    except:
        selected_model = available_models[0]
    
    print(f"Using model: {selected_model}")
    
    # Initialize classifier
    classifier = SpeechInputClassifier(model_path=selected_model)
    
    if classifier.model is None:
        return
    
    # Choose mode
    print("\nChoose mode:")
    print("1. Single recording")
    print("2. Interactive mode (continuous)")
    
    try:
        mode = input("Select mode (1 or 2): ").strip()
        if mode == "2":
            classifier.interactive_mode()
        else:
            classifier.single_recording_mode()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 