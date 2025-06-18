import os
import shutil
from pathlib import Path

def copy_first_two_of_each_digit():
    source_dir = Path('01')
    target_dir = Path('smaller_dataset') / '01'
    target_dir.mkdir(parents=True, exist_ok=True)

    # For each digit 0-9, copy 0_01_0.wav, 0_01_1.wav, 1_01_0.wav, 1_01_1.wav, ..., 9_01_0.wav, 9_01_1.wav
    for digit in range(10):
        for idx in range(2):
            filename = f"{digit}_01_{idx}.wav"
            source_path = source_dir / filename
            target_path = target_dir / filename
            if source_path.exists():
                shutil.copy2(source_path, target_path)
                print(f"Copied {source_path} to {target_path}")
            else:
                print(f"File {source_path} does not exist, skipping.")

if __name__ == "__main__":
    copy_first_two_of_each_digit() 