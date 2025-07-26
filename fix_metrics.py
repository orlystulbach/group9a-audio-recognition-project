#!/usr/bin/env python3
"""
Script to fix the top_3_accuracy metric issue in training files
"""

import re

def fix_file(filename):
    """Fix the top_3_accuracy metric in a file"""
    print(f"Fixing {filename}...")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Fix model.compile calls
    content = re.sub(
        r"metrics=\['accuracy', 'top_3_accuracy'\]",
        "metrics=['accuracy']",
        content
    )
    
    # Fix model.evaluate calls
    content = re.sub(
        r"loss, accuracy, top3_acc = model\.evaluate",
        "loss, accuracy = model.evaluate",
        content
    )
    
    # Remove top3_acc print statements
    content = re.sub(
        r"print\(f\"\{.*?\} - Top-3 Accuracy: \{top3_acc\*100:\.2f\}%\"\)",
        "",
        content
    )
    
    content = re.sub(
        r"print\(f\"Top-3 Accuracy: \{top3_acc\*100:\.2f\}%\"\)",
        "",
        content
    )
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"✅ Fixed {filename}")

def main():
    """Fix all training files"""
    files_to_fix = [
        'fixed_train_digit_classifier.py',
        'improved_models.py',
        'improved_train_digit_classifier.py'
    ]
    
    for filename in files_to_fix:
        try:
            fix_file(filename)
        except FileNotFoundError:
            print(f"⚠️  File {filename} not found, skipping...")

if __name__ == "__main__":
    main() 