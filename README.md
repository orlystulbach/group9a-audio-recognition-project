# 🎤 Audio Digit Recognition Project

## 📁 Project Structure

This repository contains a complete audio digit recognition system that can classify spoken digits (0-9) with high accuracy.

### 🎯 **Final Model** (Ready to Use)
The production-ready model is located in the `final_model/` folder:
- **99.76% test accuracy**
- **Robust voice-adapted model**
- **Complete Streamlit application**
- **Comprehensive documentation**

### 🚀 Quick Start
```bash
cd final_model
pip install -r requirements.txt
streamlit run app_robust_voice.py
```

## 📊 Dataset Structure

- `data/` - Original Audio MNIST dataset (organized by speaker)
- `large_dataset/` - Flattened dataset for training
- `medium_dataset/` - Medium-sized subset
- `smaller_dataset/` - Small subset for testing
- `smol_dataset/` - Minimal dataset
- `my_voice_samples/` - User voice samples for adaptation
- `user_voice_samples/` - Additional voice samples
- `improved_voice_samples/` - Enhanced voice samples

## 🔧 Development Files

- `backup_old_files/` - All development scripts, old models, and temporary files
- `final_model/` - Production-ready model and application

## 📋 Requirements

See `requirements.txt` for the complete list of dependencies.

## 🎯 Model Performance

The final robust voice model achieves:
- **99.76% test accuracy**
- **100% training accuracy**
- **Robust to noise and variations**
- **High confidence predictions**

## 📖 Documentation

For detailed usage instructions, model architecture, and troubleshooting, see the comprehensive documentation in `final_model/README.md`.

---

**Status**: Production Ready ✅  
**Last Updated**: August 1, 2024
