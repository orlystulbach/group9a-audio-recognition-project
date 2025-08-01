#!/usr/bin/env python3
"""
Script to check for overlap between training and test data.
This helps identify if there's data leakage that could inflate accuracy scores.
"""

import os
import numpy as np
import librosa
from pathlib import Path
import hashlib
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def get_file_hash(file_path):
    """Calculate MD5 hash of a file to detect exact duplicates."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def extract_audio_features(file_path, sr=22050):
    """Extract basic audio features for similarity comparison."""
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr)
        
        # Extract basic features for comparison
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        
        # Return feature statistics
        features = {
            'mfcc_mean': np.mean(mfcc, axis=1),
            'mfcc_std': np.std(mfcc, axis=1),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'zcr_mean': np.mean(zero_crossing_rate),
            'zcr_std': np.std(zero_crossing_rate),
            'duration': len(y) / sr,
            'rms': np.sqrt(np.mean(y**2))
        }
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def simulate_train_test_split(data_dir, test_size=0.2, random_state=42):
    """Simulate the same train/test split used in training scripts."""
    print(f"🔄 Simulating train/test split (test_size={test_size}, random_state={random_state})...")
    
    # Get all audio files
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    wav_files.sort()  # Ensure consistent ordering
    
    # Create the same split as in training
    train_files, test_files = train_test_split(
        wav_files, 
        test_size=test_size, 
        random_state=random_state,
        stratify=[f.split('_')[0] for f in wav_files]  # Stratify by digit
    )
    
    print(f"   Train files: {len(train_files)}")
    print(f"   Test files: {len(test_files)}")
    
    return train_files, test_files

def find_duplicate_files(data_dir, train_files, test_files):
    """Find exact file duplicates between train and test sets."""
    print("🔍 Checking for exact file duplicates...")
    
    train_hashes = {}
    test_hashes = {}
    
    # Calculate hashes for train files
    for filename in train_files:
        file_path = os.path.join(data_dir, filename)
        file_hash = get_file_hash(file_path)
        train_hashes[file_hash] = filename
    
    # Calculate hashes for test files
    for filename in test_files:
        file_path = os.path.join(data_dir, filename)
        file_hash = get_file_hash(file_path)
        test_hashes[file_hash] = filename
    
    # Find overlapping hashes
    train_hash_set = set(train_hashes.keys())
    test_hash_set = set(test_hashes.keys())
    overlapping_hashes = train_hash_set.intersection(test_hash_set)
    
    print(f"📊 File Analysis Results:")
    print(f"   Train files: {len(train_files)}")
    print(f"   Test files: {len(test_files)}")
    print(f"   Exact duplicates: {len(overlapping_hashes)}")
    
    if overlapping_hashes:
        print(f"\n⚠️  WARNING: Found {len(overlapping_hashes)} exact duplicate files!")
        print("   Duplicate files:")
        for hash_val in list(overlapping_hashes)[:10]:  # Show first 10
            train_filename = train_hashes[hash_val]
            test_filename = test_hashes[hash_val]
            print(f"     Train: {train_filename}")
            print(f"     Test:  {test_filename}")
            print()
        if len(overlapping_hashes) > 10:
            print(f"     ... and {len(overlapping_hashes) - 10} more duplicates")
    else:
        print("✅ No exact file duplicates found!")
    
    return overlapping_hashes

def find_similar_files(data_dir, train_files, test_files, similarity_threshold=0.95):
    """Find files with very similar audio content."""
    print(f"\n🔍 Checking for similar audio content (threshold: {similarity_threshold})...")
    
    # Limit to reasonable sample size for performance
    max_samples = 50
    if len(train_files) > max_samples:
        train_files_sample = np.random.choice(train_files, max_samples, replace=False)
    else:
        train_files_sample = train_files
        
    if len(test_files) > max_samples:
        test_files_sample = np.random.choice(test_files, max_samples, replace=False)
    else:
        test_files_sample = test_files
    
    print(f"   Comparing {len(train_files_sample)} train files vs {len(test_files_sample)} test files...")
    
    similar_pairs = []
    
    for i, train_filename in enumerate(train_files_sample):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(train_files_sample)}")
        
        train_path = os.path.join(data_dir, train_filename)
        train_features = extract_audio_features(train_path)
        if train_features is None:
            continue
            
        for test_filename in test_files_sample:
            test_path = os.path.join(data_dir, test_filename)
            test_features = extract_audio_features(test_path)
            if test_features is None:
                continue
            
            # Calculate similarity based on feature correlation
            similarity = calculate_feature_similarity(train_features, test_features)
            
            if similarity > similarity_threshold:
                similar_pairs.append({
                    'train_file': train_filename,
                    'test_file': test_filename,
                    'similarity': similarity
                })
    
    print(f"   Found {len(similar_pairs)} pairs with high similarity (> {similarity_threshold})")
    
    if similar_pairs:
        print("\n⚠️  WARNING: Found files with very similar content!")
        for pair in similar_pairs[:5]:  # Show first 5
            print(f"   Similarity: {pair['similarity']:.3f}")
            print(f"     Train: {pair['train_file']}")
            print(f"     Test:  {pair['test_file']}")
            print()
    
    return similar_pairs

def calculate_feature_similarity(features1, features2):
    """Calculate similarity between two feature sets."""
    # Convert features to vectors for comparison
    vec1 = []
    vec2 = []
    
    for key in features1.keys():
        if isinstance(features1[key], np.ndarray):
            vec1.extend(features1[key].flatten())
            vec2.extend(features2[key].flatten())
        else:
            vec1.append(features1[key])
            vec2.append(features2[key])
    
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(vec1, vec2)[0, 1]
    return correlation if not np.isnan(correlation) else 0.0

def analyze_filename_patterns(train_files, test_files):
    """Analyze filename patterns to detect potential data leakage."""
    print(f"\n🔍 Analyzing filename patterns...")
    
    # Find common filenames
    train_set = set(train_files)
    test_set = set(test_files)
    common_filenames = train_set.intersection(test_set)
    
    print(f"   Train filenames: {len(train_files)}")
    print(f"   Test filenames: {len(test_files)}")
    print(f"   Common filenames: {len(common_filenames)}")
    
    if common_filenames:
        print(f"\n⚠️  WARNING: Found {len(common_filenames)} files with identical names!")
        print("   Common filenames:")
        for filename in list(common_filenames)[:10]:
            print(f"     {filename}")
        if len(common_filenames) > 10:
            print(f"     ... and {len(common_filenames) - 10} more")
    else:
        print("✅ No files with identical names found!")
    
    # Analyze digit distribution
    train_digits = [f.split('_')[0] for f in train_files]
    test_digits = [f.split('_')[0] for f in test_files]
    
    train_digit_counts = defaultdict(int)
    test_digit_counts = defaultdict(int)
    
    for digit in train_digits:
        train_digit_counts[digit] += 1
    for digit in test_digits:
        test_digit_counts[digit] += 1
    
    print(f"\n📊 Digit Distribution:")
    print(f"   {'Digit':<6} {'Train':<8} {'Test':<8} {'Ratio':<8}")
    print(f"   {'-'*30}")
    for digit in sorted(set(train_digit_counts.keys()) | set(test_digit_counts.keys())):
        train_count = train_digit_counts.get(digit, 0)
        test_count = test_digit_counts.get(digit, 0)
        ratio = test_count / (train_count + test_count) if (train_count + test_count) > 0 else 0
        print(f"   {digit:<6} {train_count:<8} {test_count:<8} {ratio:.3f}")
    
    return common_filenames

def main():
    """Main function to check for data overlap."""
    print("🚀 Data Overlap Analysis Tool")
    print("=" * 50)
    
    # Define data directory
    data_dir = "large_dataset"
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        print("Please adjust the data_dir path in the script.")
        return
    
    print(f"📁 Analyzing directory: {data_dir}")
    
    # Simulate the train/test split used in training
    train_files, test_files = simulate_train_test_split(data_dir, test_size=0.2, random_state=42)
    
    # 1. Check for exact file duplicates
    overlapping_hashes = find_duplicate_files(data_dir, train_files, test_files)
    
    # 2. Check for similar audio content
    similar_pairs = find_similar_files(data_dir, train_files, test_files)
    
    # 3. Check filename patterns and digit distribution
    common_filenames = analyze_filename_patterns(train_files, test_files)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    total_issues = len(overlapping_hashes) + len(similar_pairs) + len(common_filenames)
    
    if total_issues == 0:
        print("✅ No data overlap issues detected!")
        print("   Your train/test split appears to be clean.")
        print("   The high accuracy (95%) is likely genuine!")
    else:
        print(f"⚠️  Found {total_issues} potential data overlap issues:")
        print(f"   - {len(overlapping_hashes)} exact file duplicates")
        print(f"   - {len(similar_pairs)} similar audio files")
        print(f"   - {len(common_filenames)} files with identical names")
        print()
        print("🔧 Recommendations:")
        print("   1. Remove duplicate files from either train or test set")
        print("   2. Ensure proper random splitting of data")
        print("   3. Consider using stratified sampling to maintain class balance")
        print("   4. Re-run your training after fixing data overlap issues")
        print("   5. The high accuracy might be inflated due to data leakage")

if __name__ == "__main__":
    main() 