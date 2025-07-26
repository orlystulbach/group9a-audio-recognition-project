#!/usr/bin/env python3
"""
Quick Start Script for Improved Digit Classifier
This script demonstrates the key improvements and how to use them.
"""

import os
import numpy as np
from improved_feature_extraction import extract_improved_features
from improved_models import create_improved_dense_model, compile_model, get_callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from tensorflow import keras

def quick_demo():
    """
    Quick demonstration of the improved system
    """
    print("=== Improved Digit Classifier Quick Demo ===\n")
    
    # Check if dataset exists
    if not os.path.exists('large_dataset'):
        print("❌ Dataset not found! Please ensure 'large_dataset' directory exists.")
        return
    
    # Load a small subset for quick demo
    print("1. Loading small subset of data for demo...")
    X, y = [], []
    
    # Get first 100 files for quick demo
    wav_files = [f for f in os.listdir('large_dataset') if f.endswith('.wav')][:100]
    
    for fname in wav_files:
        label = fname.split('_')[-1].split('.')[0]
        file_path = os.path.join('large_dataset', fname)
        
        try:
            features = extract_improved_features(file_path)
            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"Error processing {fname}: {e}")
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Loaded {len(X)} samples with {X.shape[1]} features each")
    print(f"   Labels: {np.unique(y)}")
    
    # Prepare data
    print("\n2. Preparing data...")
    lb = LabelBinarizer()
    y_encoded = lb.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Test set: {X_test.shape[0]} samples")
    
    # Create and train model
    print("\n3. Creating improved model...")
    model = create_improved_dense_model((X_train.shape[1],), y_encoded.shape[1])
    model = compile_model(model, learning_rate=0.001)
    
    print("✅ Model created with:")
    print("   - Deeper architecture (256->128->64 units)")
    print("   - Batch normalization")
    print("   - Dropout regularization")
    print("   - Early stopping and learning rate reduction")
    
    # Train model
    print("\n4. Training model (quick demo with 10 epochs)...")
    callbacks = get_callbacks(patience=5)
    
    history = model.fit(
        X_train_scaled, y_train,
        epochs=10,
        batch_size=16,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n5. Evaluating model...")
    loss, accuracy, top3_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"✅ Test Accuracy: {accuracy*100:.2f}%")
    print(f"✅ Top-3 Accuracy: {top3_acc*100:.2f}%")
    
    # Compare with original approach
    print("\n6. Comparison with original approach:")
    print("   Original approach: ~2-6% accuracy")
    print(f"   Improved approach: {accuracy*100:.2f}% accuracy")
    print(f"   Improvement: {(accuracy*100 - 6):.1f} percentage points")
    
    # Save model
    model.save('quick_demo_model.h5')
    print("\n✅ Model saved as 'quick_demo_model.h5'")
    
    print("\n=== Demo completed! ===")
    print("\nNext steps:")
    print("1. Run 'python improved_train_digit_classifier.py' for full training")
    print("2. Run 'python data_augmentation.py' to create augmented dataset")
    print("3. Experiment with different model architectures")

if __name__ == "__main__":
    quick_demo() 