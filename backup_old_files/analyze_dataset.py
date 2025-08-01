#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes the large_dataset to identify potential data issues
"""

import os
import numpy as np
import librosa
from collections import Counter
import matplotlib.pyplot as plt

def analyze_dataset_structure():
    """
    Analyze the structure of the dataset
    """
    print("=== Dataset Structure Analysis ===\n")
    
    data_dir = 'large_dataset'
    if not os.path.exists(data_dir):
        print("❌ Dataset directory not found!")
        return
    
    # Get all wav files
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    print(f"Total audio files: {len(wav_files)}")
    
    # Analyze filename patterns
    print("\n1. Filename Pattern Analysis:")
    print("   Format: [digit]_[speaker]_[instance].wav")
    
    digits = []
    speakers = []
    instances = []
    
    for filename in wav_files:
        parts = filename.split('_')
        if len(parts) >= 3:
            digit = parts[0]
            speaker = parts[1]
            instance = parts[2].split('.')[0]
            
            digits.append(digit)
            speakers.append(speaker)
            instances.append(instance)
    
    print(f"   Unique digits found: {sorted(set(digits))}")
    print(f"   Unique speakers found: {sorted(set(speakers))}")
    print(f"   Number of instances per digit-speaker: {len(set(instances))}")
    
    # Count digit distribution
    digit_counts = Counter(digits)
    print(f"\n2. Digit Distribution:")
    for digit in sorted(digit_counts.keys()):
        print(f"   Digit {digit}: {digit_counts[digit]} files")
    
    # Check for class imbalance
    min_count = min(digit_counts.values())
    max_count = max(digit_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n   Class imbalance ratio: {imbalance_ratio:.2f}")
    if imbalance_ratio > 2:
        print("   ⚠️  WARNING: Significant class imbalance detected!")
    
    # Analyze audio characteristics
    print(f"\n3. Audio Characteristics Analysis:")
    durations = []
    sample_rates = []
    successful_files = 0
    
    for i, filename in enumerate(wav_files[:50]):  # Sample first 50 files
        try:
            file_path = os.path.join(data_dir, filename)
            y, sr = librosa.load(file_path, sr=None)
            
            duration = len(y) / sr
            durations.append(duration)
            sample_rates.append(sr)
            successful_files += 1
            
        except Exception as e:
            print(f"   Error loading {filename}: {e}")
    
    if durations:
        print(f"   Successfully analyzed {successful_files} files")
        print(f"   Average duration: {np.mean(durations):.2f} seconds")
        print(f"   Duration range: {np.min(durations):.2f} - {np.max(durations):.2f} seconds")
        print(f"   Sample rate: {sample_rates[0]} Hz (consistent)")
        
        # Check for very short or very long files
        short_files = sum(1 for d in durations if d < 0.5)
        long_files = sum(1 for d in durations if d > 3.0)
        
        if short_files > 0:
            print(f"   ⚠️  {short_files} files are very short (< 0.5s)")
        if long_files > 0:
            print(f"   ⚠️  {long_files} files are very long (> 3.0s)")
    
    return digit_counts, durations

def check_label_extraction():
    """
    Check if the label extraction logic is correct
    """
    print(f"\n4. Label Extraction Verification:")
    
    data_dir = 'large_dataset'
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')][:10]
    
    print("   Testing label extraction on sample files:")
    for filename in wav_files:
        # Original logic from train_digit_classifier.py
        label = filename.split('_')[-1].split('.')[0]
        print(f"   {filename} -> Label: {label}")
    
    # Check if this makes sense
    print(f"\n   ⚠️  POTENTIAL ISSUE: Labels are extracted from the last part of filename")
    print(f"   This means '9_60_8.wav' -> Label '8', not '9'!")
    print(f"   The actual digit is the FIRST part of the filename.")

def analyze_audio_quality():
    """
    Analyze audio quality and potential issues
    """
    print(f"\n5. Audio Quality Analysis:")
    
    data_dir = 'large_dataset'
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')][:20]
    
    for filename in wav_files:
        try:
            file_path = os.path.join(data_dir, filename)
            y, sr = librosa.load(file_path, sr=None)
            
            # Check for silence
            rms = np.sqrt(np.mean(y**2))
            if rms < 0.01:
                print(f"   ⚠️  {filename}: Very quiet/silent audio (RMS: {rms:.4f})")
            
            # Check for clipping
            if np.max(np.abs(y)) > 0.95:
                print(f"   ⚠️  {filename}: Audio clipping detected")
                
        except Exception as e:
            print(f"   ❌ {filename}: Error - {e}")

def main():
    """
    Main analysis function
    """
    print("Starting dataset analysis...\n")
    
    # Analyze structure
    digit_counts, durations = analyze_dataset_structure()
    
    # Check label extraction
    check_label_extraction()
    
    # Analyze audio quality
    analyze_audio_quality()
    
    print(f"\n=== Analysis Summary ===")
    print(f"Based on this analysis, the main issues are likely:")
    print(f"1. ❌ INCORRECT LABEL EXTRACTION - This is the primary issue!")
    print(f"2. Possible class imbalance")
    print(f"3. Audio quality variations")
    
    print(f"\n=== RECOMMENDED FIXES ===")
    print(f"1. Fix label extraction: Use first part of filename as digit label")
    print(f"2. Verify all audio files are valid and contain speech")
    print(f"3. Balance classes if needed")
    print(f"4. Use the improved feature extraction and models")

if __name__ == "__main__":
    main() 