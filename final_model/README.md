# 🎤 Robust Voice-Adapted Digit Classifier

## 📊 Model Performance
- **Test Accuracy: 99.76%**
- **Training Accuracy: 100%**
- **Model Architecture: 256→128→64→32→10 layers**
- **Features: 112 (MFCC + Delta + Delta-Delta + Chroma + Spectral)**

## 📁 Files Included

### 🤖 Model Files
- `robust_voice_model.h5` - Trained neural network model
- `robust_voice_scaler.pkl` - Feature normalization scaler
- `robust_voice_label_binarizer.pkl` - Digit label encoder

### 🎤 Application
- `app_robust_voice.py` - Streamlit web application

### 📈 Training Documentation
- `robust_model_training_history.png` - Training progress visualization
- `README.md` - This documentation file

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit tensorflow librosa noisereduce plotly pandas numpy scikit-learn
```

### Run the Application
```bash
streamlit run app_robust_voice.py
```

The app will be available at: http://localhost:8501

## 🎯 How to Use

1. **Open the app** in your browser
2. **Click the microphone** to start recording
3. **Speak a digit** (0-9) clearly for about 2 seconds
4. **Click "Predict Digit"** to get the prediction
5. **Check the confidence** and audio quality feedback

## 🔧 Model Features

### 🎤 Audio Processing
- **Noise Reduction**: Automatically handles background noise
- **Pre-emphasis Filter**: Enhances high frequencies
- **Normalization**: Consistent audio levels
- **Duration Standardization**: Ensures 2-second samples

### 🧠 Feature Extraction (112 features)
- **MFCC (13)**: Mel-frequency cepstral coefficients (mean + std)
- **Delta MFCC (13)**: First-order temporal changes (mean + std)
- **Delta-Delta MFCC (13)**: Second-order temporal changes (mean + std)
- **Spectral Features (6)**: Centroid, rolloff, bandwidth (mean + std)
- **Zero Crossing Rate (2)**: Signal complexity (mean + std)
- **RMS Energy (2)**: Audio intensity (mean + std)
- **Chroma Features (24)**: Pitch-based features (mean + std)

### 🏗️ Neural Network Architecture
```
Input (112 features)
    ↓
Dense(256) + BatchNorm + Dropout(0.4)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + BatchNorm + Dropout(0.2)
    ↓
Dense(32) + BatchNorm + Dropout(0.1)
    ↓
Dense(10) + Softmax
```

## 📊 Training Data

### 🎯 Dataset Statistics
- **Total Samples**: 2,044
- **Original Samples**: 150 (15 per digit)
- **Augmented Samples**: 1,894
- **Augmentation Techniques**:
  - Pitch shift (±2 semitones)
  - Time stretch (±20%)
  - Volume variation (±20%)
  - Noise addition

### 📈 Training Performance
- **Epochs**: 98 (with early stopping)
- **Learning Rate**: 0.0005 (with reduction on plateau)
- **Batch Size**: 32
- **Validation Split**: 20%

## 🎯 Expected Results

### ✅ High Confidence Predictions
- **Correct predictions**: 90%+ confidence
- **Clear digit discrimination**: No confusion between similar digits
- **Robust to variations**: Handles different speaking styles, volumes, and noise

### 📊 Example Output
```
Predicted Digit: 5
Confidence: 95.2%

Top 3 Predictions:
1. Digit 5: 0.952
2. Digit 2: 0.023
3. Digit 8: 0.015
```

## 🔧 Troubleshooting

### Common Issues
1. **Low Confidence (< 50%)**
   - Speak more clearly and distinctly
   - Ensure you're speaking a digit (0-9)
   - Check microphone functionality

2. **Wrong Predictions**
   - Verify you're speaking the intended digit
   - Try speaking more slowly
   - Check audio quality metrics in the app

3. **App Not Starting**
   - Ensure all dependencies are installed
   - Check that all model files are in the same directory
   - Verify Python environment is activated

### Audio Quality Tips
- **Duration**: Aim for 2 seconds
- **Volume**: Any reasonable level (model handles variations)
- **Environment**: Works well even with background noise
- **Clarity**: Speak clearly and distinctly

## 📋 Dependencies

### Required Packages
```
streamlit>=1.28.0
tensorflow>=2.13.0
librosa>=0.10.0
noisereduce>=3.0.0
plotly>=5.15.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

### Installation
```bash
pip install -r requirements.txt
```

## 🎉 Model Advantages

### ✅ Key Improvements Over Previous Versions
1. **13x More Training Data**: 2,044 vs 150 samples
2. **3x More Features**: 112 vs 34 features
3. **Noise Reduction**: Built-in background noise handling
4. **Data Augmentation**: Robust to audio variations
5. **Advanced Architecture**: Better regularization and training

### 🎯 Use Cases
- **Voice-controlled applications**
- **Accessibility tools**
- **Educational software**
- **Research and development**
- **Prototype demonstrations**

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Verify all files are present in the final_model directory
3. Ensure dependencies are correctly installed
4. Test with the provided Streamlit app

---

**Model Version**: Robust Voice-Adapted v1.0  
**Training Date**: July 31, 2024  
**Performance**: 99.76% test accuracy  
**Status**: Production Ready ✅ 