#!/usr/bin/env python3
"""
Quick test to demonstrate the label extraction fix
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

def extract_simple_features(file_path):
    """Simple feature extraction for quick test"""
    y, sr = librosa.load(file_path, sr=None)
    y = librosa.util.normalize(y)
    
    # Basic MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

def test_label_extraction():
    """Test both old and new label extraction methods"""
    print("=== Label Extraction Test ===\n")
    
    data_dir = 'large_dataset'
    sample_files = ['9_60_8.wav', '5_32_32.wav', '0_11_35.wav', '7_24_29.wav']
    
    print("Sample files and their labels:")
    print("File\t\tOld Method\tNew Method\tActual Digit")
    print("-" * 50)
    
    for filename in sample_files:
        if os.path.exists(os.path.join(data_dir, filename)):
            # Old method (WRONG)
            old_label = filename.split('_')[-1].split('.')[0]
            
            # New method (CORRECT)
            new_label = filename.split('_')[0]
            
            print(f"{filename}\t{old_label}\t\t{new_label}\t\t{new_label}")
    
    print(f"\nThe old method was extracting instance numbers instead of digits!")
    print(f"This explains the 2-6% accuracy - the model was learning the wrong patterns.")

def quick_training_test():
    """Quick training test with corrected labels"""
    print("\n=== Quick Training Test with Fixed Labels ===\n")
    
    data_dir = 'large_dataset'
    
    # Use first 500 files for quick test
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')][:500]
    
    X, y = [], []
    
    print("Loading data with CORRECTED labels...")
    for fname in wav_files:
        # CORRECTED: Extract digit from first part
        label = fname.split('_')[0]
        file_path = os.path.join(data_dir, fname)
        
        try:
            features = extract_simple_features(file_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Loaded {len(X)} samples")
    print(f"Labels: {np.unique(y)}")
    print(f"Label distribution: {np.bincount([int(label) for label in y])}")
    
    # Prepare data
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Simple model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(y_encoded.shape[1], activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Quick training
    print("\nTraining for 10 epochs...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    # Compare with expected performance
    print(f"\n=== Performance Comparison ===")
    print(f"Original approach (wrong labels): ~2-6% accuracy")
    print(f"Fixed approach (correct labels): {accuracy*100:.1f}% accuracy")
    print(f"Improvement: +{(accuracy*100 - 6):.1f} percentage points")
    
    if accuracy > 0.5:
        print(f"✅ SUCCESS: The fix worked! Accuracy improved dramatically.")
    else:
        print(f"⚠️  Still low accuracy - may need additional improvements.")

def main():
    """Main test function"""
    print("Testing the label extraction fix...\n")
    
    # Test label extraction
    test_label_extraction()
    
    # Quick training test
    quick_training_test()
    
    print(f"\n=== Next Steps ===")
    print(f"1. Run 'python3 fixed_train_digit_classifier.py' for full training")
    print(f"2. This should achieve 70-90%+ accuracy with the corrected labels")
    print(f"3. The main issue was the label extraction bug, not the model architecture")

if __name__ == "__main__":
    main() 