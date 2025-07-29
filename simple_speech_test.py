#!/usr/bin/env python3
"""
Simple test for speech input audio recording capability
"""

import os
import numpy as np
import time

def test_audio_recording():
    """Test if audio recording works"""
    print("🧪 Testing audio recording capability...")
    
    try:
        import sounddevice as sd
        print("✅ sounddevice imported successfully")
        
        # Test if we can get device info
        devices = sd.query_devices()
        print(f"   Found {len(devices)} audio devices")
        
        # Find default input device
        default_input = sd.query_devices(kind='input')
        print(f"   Default input device: {default_input['name']}")
        
        # Test a short recording
        print("   Testing 1-second recording...")
        sample_rate = 22050
        duration = 1  # 1 second
        
        print("   🎤 Recording for 1 second (speak something)...")
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype=np.float32)
        sd.wait()  # Wait until recording is finished
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        print(f"   ✅ Recording successful!")
        print(f"   Audio shape: {audio_data.shape}")
        print(f"   Audio range: [{audio_data.min():.4f}, {audio_data.max():.4f}]")
        
        # Check audio quality
        rms = np.sqrt(np.mean(audio_data**2))
        print(f"   Audio RMS: {rms:.4f}")
        
        if rms > 0.01:
            print("   ✅ Audio is loud enough")
        else:
            print("   ⚠️  Audio might be too quiet")
        
        return True
        
    except ImportError:
        print("❌ sounddevice not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing audio recording: {e}")
        return False

def test_model_files():
    """Test if model files exist"""
    print("\n🧪 Testing model files...")
    
    model_files = [
        'fixed_digit_classifier_model.h5',
        'deeper_digit_classifier_model.h5',
        'residual_digit_classifier_model.h5',
        'wide_digit_classifier_model.h5'
    ]
    
    found_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"   ✅ {model_file} ({size:.1f} MB)")
            found_models.append(model_file)
        else:
            print(f"   ❌ {model_file}: Not found")
    
    return found_models

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import numpy as np
        print("   ✅ numpy imported")
    except ImportError:
        print("   ❌ numpy not available")
        return False
    
    try:
        import librosa
        print("   ✅ librosa imported")
    except ImportError:
        print("   ❌ librosa not available")
        return False
    
    try:
        import soundfile as sf
        print("   ✅ soundfile imported")
    except ImportError:
        print("   ❌ soundfile not available")
        return False
    
    return True

def main():
    """Run all tests"""
    print("🧪 Simple Speech Input Test")
    print("=" * 40)
    
    # Test basic imports
    if not test_basic_imports():
        print("❌ Basic imports failed")
        return
    
    # Test model files
    found_models = test_model_files()
    
    # Test audio recording
    audio_works = test_audio_recording()
    
    print("\n" + "=" * 40)
    print("🎯 Test Summary:")
    
    if audio_works and found_models:
        print("✅ All tests passed!")
        print("You can now run: python3 speech_input.py")
        print("The script will guide you through the interactive process.")
    elif audio_works:
        print("⚠️  Audio recording works, but no models found.")
        print("Train a model first using one of the training scripts.")
    elif found_models:
        print("⚠️  Models found, but audio recording failed.")
        print("Check your microphone permissions and sounddevice installation.")
    else:
        print("❌ Both audio recording and models failed.")
        print("Please check your setup.")

if __name__ == "__main__":
    main() 