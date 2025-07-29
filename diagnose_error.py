#!/usr/bin/env python3
"""
Diagnostic script to identify errors
"""

import os
import sys

def check_imports():
    """Check if all required imports work"""
    print("🔍 Checking imports...")
    
    try:
        import numpy as np
        print("✅ numpy")
    except ImportError as e:
        print(f"❌ numpy: {e}")
        return False
    
    try:
        import librosa
        print("✅ librosa")
    except ImportError as e:
        print(f"❌ librosa: {e}")
        return False
    
    try:
        import sounddevice as sd
        print("✅ sounddevice")
    except ImportError as e:
        print(f"❌ sounddevice: {e}")
        return False
    
    try:
        import tensorflow as tf
        print("✅ tensorflow")
    except ImportError as e:
        print(f"❌ tensorflow: {e}")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    print("\n📁 Checking files...")
    
    files_to_check = [
        'voice_adapted_model.h5',
        'residual_digit_classifier_model.h5',
        'user_voice_samples'
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            if os.path.isdir(file):
                print(f"✅ {file}/ (directory)")
            else:
                size = os.path.getsize(file) / (1024 * 1024)
                print(f"✅ {file} ({size:.1f} MB)")
        else:
            print(f"❌ {file} (not found)")
    
    # Check voice samples
    if os.path.exists('user_voice_samples'):
        for digit in range(10):
            digit_dir = os.path.join('user_voice_samples', str(digit))
            if os.path.exists(digit_dir):
                wav_files = [f for f in os.listdir(digit_dir) if f.endswith('.wav')]
                print(f"   Digit {digit}: {len(wav_files)} samples")
            else:
                print(f"   Digit {digit}: directory not found")

def check_model():
    """Check if model can be loaded"""
    print("\n🤖 Checking model...")
    
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('voice_adapted_model.h5')
        print("✅ Voice-adapted model loads successfully")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def check_audio():
    """Check audio system"""
    print("\n🎤 Checking audio system...")
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print(f"✅ Found {len(devices)} audio devices")
        
        # Check for input devices
        input_devices = []
        for i, device in enumerate(devices):
            if hasattr(device, 'max_inputs') and device['max_inputs'] > 0:
                input_devices.append(i)
            elif 'max_inputs' in device and device['max_inputs'] > 0:
                input_devices.append(i)
        
        print(f"   Input devices: {input_devices}")
        
        if input_devices:
            print("✅ Audio input devices available")
            return True
        else:
            print("❌ No audio input devices found")
            return False
            
    except Exception as e:
        print(f"❌ Audio system error: {e}")
        return False

def main():
    """Run all diagnostics"""
    print("🔧 Error Diagnosis")
    print("=" * 50)
    
    # Check imports
    imports_ok = check_imports()
    
    # Check files
    check_files()
    
    # Check model
    model_ok = check_model()
    
    # Check audio
    audio_ok = check_audio()
    
    print("\n" + "=" * 50)
    print("📊 Diagnosis Summary:")
    
    if imports_ok and model_ok and audio_ok:
        print("✅ All systems appear to be working")
        print("The error might be in the test script logic")
    else:
        print("❌ Issues found:")
        if not imports_ok:
            print("   - Import issues")
        if not model_ok:
            print("   - Model loading issues")
        if not audio_ok:
            print("   - Audio system issues")
    
    print("\n💡 Suggestions:")
    print("1. Try running: python3 speech_input_adjusted.py")
    print("2. Select the voice-adapted model when prompted")
    print("3. If audio fails, try restarting your terminal/computer")
    print("4. Check microphone permissions in System Preferences")

if __name__ == "__main__":
    main() 