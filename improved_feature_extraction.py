import os
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def extract_improved_features(file_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract more comprehensive audio features including:
    - MFCC coefficients (mean, std, delta, delta-delta)
    - Spectral features
    - Chroma features
    - Zero crossing rate
    - Spectral centroid
    """
    # Load audio
    y, sr = librosa.load(file_path, sr=None)
    
    # Normalize audio
    y = librosa.util.normalize(y)
    
    features = []
    
    # 1. MFCC features (more comprehensive)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Statistical features from MFCC
    features.extend(np.mean(mfcc, axis=1))  # Mean of each coefficient
    features.extend(np.std(mfcc, axis=1))   # Std of each coefficient
    features.extend(np.mean(mfcc_delta, axis=1))  # Delta features
    features.extend(np.mean(mfcc_delta2, axis=1)) # Delta-delta features
    
    # 2. Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
    
    features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
    features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
    features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
    
    # 3. Chroma features (useful for pitch-based classification)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    features.extend(np.mean(chroma, axis=1))  # Mean of each chroma bin
    
    # 4. Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    features.extend([np.mean(zcr), np.std(zcr)])
    
    # 5. Root Mean Square Energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    features.extend([np.mean(rms), np.std(rms)])
    
    # 6. Mel-scaled spectrogram statistics
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    features.extend([np.mean(mel_spec_db), np.std(mel_spec_db)])
    
    return np.array(features)

def extract_mel_spectrogram_features(file_path, target_length=128):
    """
    Extract mel spectrogram as 2D features for CNN
    """
    y, sr = librosa.load(file_path, sr=None)
    y = librosa.util.normalize(y)
    
    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Resize to target length (time dimension)
    if mel_spec_db.shape[1] > target_length:
        # Truncate if too long
        mel_spec_db = mel_spec_db[:, :target_length]
    else:
        # Pad if too short
        pad_width = target_length - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    
    return mel_spec_db

def extract_raw_audio_features(file_path, target_length=16000):
    """
    Extract raw audio features (normalized and padded/truncated)
    """
    y, sr = librosa.load(file_path, sr=None)
    y = librosa.util.normalize(y)
    
    # Resample to target length
    if len(y) > target_length:
        y = y[:target_length]
    else:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    
    return y

# Example usage and testing
if __name__ == "__main__":
    # Test with a sample file
    sample_file = "large_dataset/9_60_0.wav"
    if os.path.exists(sample_file):
        print("Testing improved feature extraction...")
        
        # Test different feature extraction methods
        features_1d = extract_improved_features(sample_file)
        print(f"1D features shape: {features_1d.shape}")
        
        mel_features = extract_mel_spectrogram_features(sample_file)
        print(f"Mel spectrogram shape: {mel_features.shape}")
        
        raw_features = extract_raw_audio_features(sample_file)
        print(f"Raw audio features shape: {raw_features.shape}")
        
        print("Feature extraction test completed!") 