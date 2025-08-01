#!/usr/bin/env python3
"""
Simple Voice Sample Collection for Digit Recognition
"""

import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime

# Configuration
VOICE_SAMPLES_DIR = 'my_voice_samples'
SAMPLES_PER_DIGIT = 15

def record_voice_samples():
    """
    Record voice samples for each digit
    """
    print("🎤 Voice Sample Collection for Digit Recognition")
    print("=" * 60)
    print("This will collect voice samples to train a model specifically for your voice.")
    print("This should significantly improve prediction accuracy!")
    print()
    
    # Create directory for voice samples
    os.makedirs(VOICE_SAMPLES_DIR, exist_ok=True)
    
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    total_samples = 0
    
    for digit in digits:
        print(f"\n📝 Recording digit: {digit}")
        print(f"Please record {SAMPLES_PER_DIGIT} samples of '{digit}'")
        print("💡 Tips:")
        print("   - Speak clearly and naturally")
        print("   - Vary your tone slightly between samples")
        print("   - Use different speaking speeds")
        print("   - Say the digit clearly (e.g., 'five' not 'fifth')")
        print()
        
        # Create digit directory
        digit_dir = os.path.join(VOICE_SAMPLES_DIR, digit)
        os.makedirs(digit_dir, exist_ok=True)
        
        for i in range(SAMPLES_PER_DIGIT):
            input(f"Press Enter to record sample {i+1}/{SAMPLES_PER_DIGIT} of '{digit}'...")
            
            try:
                # Record audio (2 seconds at 22050 Hz)
                print("🎤 Recording... Speak clearly!")
                audio = sd.rec(int(2 * 22050), samplerate=22050, channels=1)
                sd.wait()
                audio_data = audio.flatten()
                
                # Check audio quality
                rms = np.sqrt(np.mean(audio_data**2))
                print(f"Audio RMS: {rms:.6f}")
                
                if rms < 0.01:
                    print("⚠️ Too quiet! Please speak louder.")
                    continue
                elif rms > 0.5:
                    print("⚠️ Too loud! Please speak more quietly.")
                    continue
                
                # Save audio
                filename = f"{digit}_{i+1}.wav"
                filepath = os.path.join(digit_dir, filename)
                sf.write(filepath, audio_data, 22050)
                print(f"✅ Saved: {filename}")
                total_samples += 1
                
            except Exception as e:
                print(f"❌ Error recording: {e}")
                continue
    
    print(f"\n🎉 Voice sample collection complete!")
    print(f"✅ Saved {total_samples} samples in '{VOICE_SAMPLES_DIR}' directory")
    print()
    print("Next steps:")
    print("1. Run: python3 train_voice_model.py")
    print("2. This will train a model using your voice samples + data augmentation")
    print("3. Then run: streamlit run app_voice_adapted.py")
    
    return total_samples

if __name__ == "__main__":
    record_voice_samples() 