import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
import random

def add_noise(audio, noise_factor=0.005):
    """
    Add random noise to audio
    """
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def time_shift(audio, shift_factor=0.1):
    """
    Shift audio in time
    """
    shift = int(len(audio) * shift_factor)
    if shift > 0:
        augmented_audio = np.pad(audio, (shift, 0), mode='constant')
        augmented_audio = augmented_audio[:-shift]
    else:
        augmented_audio = np.pad(audio, (0, -shift), mode='constant')
        augmented_audio = augmented_audio[-shift:]
    return augmented_audio

def pitch_shift(audio, sr, pitch_factor=2):
    """
    Shift pitch of audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_factor)

def speed_change(audio, speed_factor=1.2):
    """
    Change speed of audio
    """
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def add_reverb(audio, room_scale=0.1):
    """
    Add reverb effect
    """
    # Simple reverb simulation
    reverb = np.zeros_like(audio)
    delay_samples = int(room_scale * len(audio))
    reverb[delay_samples:] = audio[:-delay_samples] * 0.3
    return audio + reverb

def augment_audio_file(input_path, output_dir, augmentations_per_file=3):
    """
    Apply multiple augmentations to a single audio file
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)
    
    # Get filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    augmented_files = []
    
    for i in range(augmentations_per_file):
        # Apply random augmentation
        aug_audio = audio.copy()
        
        # Random augmentation type
        aug_type = random.choice(['noise', 'time_shift', 'pitch_shift', 'speed_change', 'reverb'])
        
        if aug_type == 'noise':
            noise_factor = random.uniform(0.001, 0.01)
            aug_audio = add_noise(aug_audio, noise_factor)
        elif aug_type == 'time_shift':
            shift_factor = random.uniform(-0.1, 0.1)
            aug_audio = time_shift(aug_audio, shift_factor)
        elif aug_type == 'pitch_shift':
            pitch_factor = random.uniform(-2, 2)
            aug_audio = pitch_shift(aug_audio, sr, pitch_factor)
        elif aug_type == 'speed_change':
            speed_factor = random.uniform(0.8, 1.2)
            aug_audio = speed_change(aug_audio, speed_factor)
        elif aug_type == 'reverb':
            room_scale = random.uniform(0.05, 0.2)
            aug_audio = add_reverb(aug_audio, room_scale)
        
        # Save augmented file
        output_filename = f"{base_name}_aug_{i+1}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        sf.write(output_path, aug_audio, sr)
        augmented_files.append(output_path)
    
    return augmented_files

def augment_dataset(input_dir, output_dir, augmentations_per_file=3):
    """
    Augment entire dataset
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all wav files
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    print(f"Found {len(wav_files)} files to augment")
    print(f"Will create {len(wav_files) * augmentations_per_file} augmented files")
    
    all_augmented_files = []
    
    for i, filename in enumerate(wav_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(wav_files)}")
        
        input_path = os.path.join(input_dir, filename)
        augmented_files = augment_audio_file(input_path, output_dir, augmentations_per_file)
        all_augmented_files.extend(augmented_files)
    
    print(f"Augmentation completed! Created {len(all_augmented_files)} files")
    return all_augmented_files

def create_augmented_dataset():
    """
    Create augmented version of the large dataset
    """
    input_dir = 'large_dataset'
    output_dir = 'large_dataset_augmented'
    
    # Copy original files first
    print("Copying original files...")
    os.makedirs(output_dir, exist_ok=True)
    
    import shutil
    for filename in os.listdir(input_dir):
        if filename.endswith('.wav'):
            src = os.path.join(input_dir, filename)
            dst = os.path.join(output_dir, filename)
            shutil.copy2(src, dst)
    
    # Add augmented files
    print("Creating augmented files...")
    augment_dataset(input_dir, output_dir, augmentations_per_file=2)
    
    print(f"Augmented dataset created in {output_dir}")
    print("You can now use this dataset for training with more data!, dope")

if __name__ == "__main__":
    create_augmented_dataset() 