#!/usr/bin/env python3
"""
Microphone troubleshooting script
"""

import numpy as np
import sounddevice as sd
import time

def list_audio_devices():
    """List all available audio devices"""
    print("🎵 Available Audio Devices:")
    print("=" * 50)
    
    devices = sd.query_devices()
    
    print("INPUT DEVICES:")
    for i, device in enumerate(devices):
        # Check if device has input capability
        if 'max_inputs' in device and device['max_inputs'] > 0:
            print(f"  {i}: {device['name']}")
            print(f"     Channels: {device['max_inputs']}")
            print(f"     Sample Rate: {device['default_samplerate']} Hz")
            print()
    
    print("OUTPUT DEVICES:")
    for i, device in enumerate(devices):
        # Check if device has output capability
        if 'max_outputs' in device and device['max_outputs'] > 0:
            print(f"  {i}: {device['name']}")
            print(f"     Channels: {device['max_outputs']}")
            print(f"     Sample Rate: {device['default_samplerate']} Hz")
            print()

def test_microphone_levels():
    """Test microphone levels with different settings"""
    print("🎤 Testing Microphone Levels")
    print("=" * 50)
    
    # Get default input device
    try:
        default_input = sd.query_devices(kind='input')
        print(f"Default input device: {default_input['name']}")
        print(f"Default sample rate: {default_input['default_samplerate']} Hz")
        print()
    except Exception as e:
        print(f"Error getting default input: {e}")
        return
    
    # Test different recording durations and settings
    test_configs = [
        {"duration": 1, "sample_rate": 22050, "name": "Short test (1s)"},
        {"duration": 3, "sample_rate": 22050, "name": "Medium test (3s)"},
        {"duration": 1, "sample_rate": 44100, "name": "High quality (1s)"},
    ]
    
    for config in test_configs:
        print(f"Testing: {config['name']}")
        print(f"  Duration: {config['duration']}s, Sample Rate: {config['sample_rate']} Hz")
        
        try:
            # Record audio
            print("  🎤 Recording... (speak something)")
            audio_data = sd.rec(
                int(config['duration'] * config['sample_rate']),
                samplerate=config['sample_rate'],
                channels=1,
                dtype=np.float32
            )
            sd.wait()
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Analyze audio
            rms = np.sqrt(np.mean(audio_data**2))
            max_amplitude = np.max(np.abs(audio_data))
            min_amplitude = np.min(np.abs(audio_data))
            
            print(f"  📊 Results:")
            print(f"    RMS: {rms:.6f}")
            print(f"    Max amplitude: {max_amplitude:.6f}")
            print(f"    Min amplitude: {min_amplitude:.6f}")
            print(f"    Audio range: [{audio_data.min():.6f}, {audio_data.max():.6f}]")
            
            if rms > 0.01:
                print(f"    ✅ Audio level is good")
            elif rms > 0.001:
                print(f"    ⚠️  Audio level is low but detectable")
            else:
                print(f"    ❌ Audio level is too quiet")
            
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()

def interactive_level_test():
    """Interactive microphone level test"""
    print("🎤 Interactive Microphone Level Test")
    print("=" * 50)
    print("This will help you find the right volume level.")
    print("Speak normally and we'll check the levels.")
    print()
    
    while True:
        try:
            user_input = input("Press ENTER to test microphone level (or 'quit' to exit): ").strip().lower()
            
            if user_input == 'quit':
                print("👋 Goodbye!")
                break
            
            print("🎤 Recording for 2 seconds... (speak normally)")
            
            # Record audio
            audio_data = sd.rec(int(2 * 22050), samplerate=22050, channels=1, dtype=np.float32)
            sd.wait()
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Analyze audio
            rms = np.sqrt(np.mean(audio_data**2))
            max_amplitude = np.max(np.abs(audio_data))
            
            print(f"📊 Audio Level: RMS={rms:.6f}, Max={max_amplitude:.6f}")
            
            if rms > 0.05:
                print("✅ Excellent! Audio level is very good.")
            elif rms > 0.01:
                print("✅ Good! Audio level is acceptable.")
            elif rms > 0.001:
                print("⚠️  Low but detectable. Try speaking louder.")
            else:
                print("❌ Too quiet. Check microphone permissions and volume.")
                print("   - Make sure microphone access is allowed")
                print("   - Check system volume settings")
                print("   - Try a different microphone if available")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main function"""
    print("🎤 Microphone Troubleshooting Tool")
    print("=" * 50)
    
    # List devices
    list_audio_devices()
    
    # Test levels
    test_microphone_levels()
    
    # Interactive test
    print("Would you like to do an interactive level test?")
    try:
        choice = input("Enter 'y' for interactive test, or any other key to exit: ").strip().lower()
        if choice == 'y':
            interactive_level_test()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main() 