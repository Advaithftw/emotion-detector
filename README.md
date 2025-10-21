# 🎭 Speech Emotion Recognition Web App

A real-time emotion recognition system that analyzes speech audio and predicts emotions using Deep Learning.

## 🚀 Quick Start

### Run Locally

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Launch the app:**
```bash
streamlit run app.py
```

3. **Open your browser:**
The app will automatically open at `http://localhost:8501`

## 📱 How to Use

1. **Upload Audio File**
   - Click "Browse files" or drag & drop
   - Supported formats: WAV, MP3, OGG, FLAC
   - Recommended: 3-second clear speech audio

2. **Analyze Emotion**
   - Click "🔮 Analyze Emotion" button
   - View predicted emotion with confidence score
   - Explore probability distribution across all emotions

3. **Interpret Results**
   - 🟢 Green (70%+): High confidence
   - 🟠 Orange (50-70%): Medium confidence
   - 🔴 Red (<50%): Low confidence

## 🎯 Supported Emotions

The model can detect 8 different emotions:

- 😠 **Angry** - High arousal, negative valence
- 😌 **Calm** - Low arousal, positive/neutral valence
- 🤢 **Disgust** - Negative valence, aversion
- 😨 **Fearful** - High arousal, negative valence
- 😊 **Happy** - High arousal, positive valence
- 😐 **Neutral** - Baseline emotional state
- 😢 **Sad** - Low arousal, negative valence
- 😮 **Surprised** - High arousal, neutral valence

## 📊 Model Performance

- **Test Accuracy:** 69.98%
- **Weighted F1 Score:** 0.70
- **Architecture:** 3-Layer MLP (512→256→128)
- **Parameters:** 6.9 Million
- **Training Samples:** 8,640 (with augmentation)
- **GPU:** NVIDIA RTX 4050

### Per-Emotion Accuracy
| Emotion    | Accuracy |
|------------|----------|
| Neutral    | 79.1%    |
| Happy      | 75.7%    |
| Calm       | 76.3%    |
| Sad        | 68.2%    |
| Surprised  | 75.7%    |
| Fearful    | 61.6%    |
| Disgust    | 63.6%    |
| Angry      | 56.7%    |

## 🏗️ Architecture

```
Input: Audio File (WAV/MP3)
    ↓
Audio Processing (Librosa)
    ↓
Feature Extraction:
  - MFCC (20 coefficients)
  - Delta & Delta-Delta MFCC
  - Spectral Centroid
  - Spectral Rolloff
  - Zero Crossing Rate
  - Chroma Features
  - Statistical Features
    ↓
Normalization
    ↓
Deep Neural Network:
  - Layer 1: 512 neurons + BatchNorm + ReLU + Dropout(0.5)
  - Layer 2: 256 neurons + BatchNorm + ReLU + Dropout(0.5)
  - Layer 3: 128 neurons + BatchNorm + ReLU + Dropout(0.4)
  - Output: 8 emotions (Softmax)
    ↓
Prediction + Confidence Scores
```

## 🔧 Technical Details

### Features Extracted (13,134 total)
- **MFCC:** 20 coefficients × 174 frames = 3,480
- **Delta MFCC:** 3,480
- **Delta-Delta MFCC:** 3,480
- **Spectral Centroid:** 174
- **Spectral Rolloff:** 174
- **Zero Crossing Rate:** 174
- **Chroma Features:** 12 × 174 = 2,088
- **Statistical Features:** 84 (mean/std of temporal features)

### Data Augmentation (Training)
- Noise injection
- Time stretching (±10%)
- Pitch shifting (±2 semitones)
- 6x data multiplication

### Regularization Techniques
- Dropout (0.4-0.5)
- Batch Normalization
- L2 Weight Decay (0.02)
- Early Stopping (patience=20)

## 📁 Project Structure

```
project/
├── app.py                      # Streamlit web application
├── run_emotion.py              # Training script
├── best_model.pt              # Trained model weights
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── IMPROVEMENTS.md            # Model improvement details
├── RESULTS.md                 # Training results
├── training_history.png       # Training curves
├── confusion_matrix.png       # Confusion matrix
├── per_class_accuracy.png     # Per-class performance
└── audio_speech_actors_01-24/ # RAVDESS dataset
```

## 🎓 Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 24 professional actors (12 male, 12 female)
- 1,440 audio files
- 8 emotions × 2 intensities
- 16kHz, 16-bit mono WAV files

## 🔮 Future Improvements

1. **Model Enhancements:**
   - CNN/LSTM architectures for temporal patterns
   - Ensemble models (combine multiple models)
   - Attention mechanisms
   - Transfer learning from pre-trained models

2. **Feature Engineering:**
   - Prosodic features (pitch, energy)
   - Voice quality features
   - Deep audio embeddings (VGGish, YAMNet)

3. **Data:**
   - Multi-language support
   - Cross-corpus evaluation
   - Real-world noisy audio handling
   - Continuous emotion prediction

4. **Deployment:**
   - Real-time microphone input
   - Mobile app version
   - REST API for integration
   - Cloud deployment (AWS/Azure/GCP)

## 💻 System Requirements

- **Python:** 3.8+
- **RAM:** 4GB minimum (8GB recommended)
- **GPU:** Optional but recommended (CUDA-capable)
- **Storage:** ~500MB for dependencies

## 🐛 Troubleshooting

### Issue: Model not loading
**Solution:** Ensure `best_model.pt` is in the same directory as `app.py`

### Issue: Audio file not supported
**Solution:** Convert to WAV format using online converters or FFmpeg

### Issue: Low confidence predictions
**Solution:** 
- Use clear speech with minimal background noise
- Ensure audio is 2-3 seconds long
- Speak with clear emotional expression

### Issue: Slow prediction
**Solution:**
- Ensure librosa is properly installed
- Close other applications
- Use GPU if available

## 📝 License

This project is for educational purposes. The RAVDESS dataset has its own license terms.

## 🙏 Acknowledgments

- **RAVDESS Dataset:** Livingstone SR, Russo FA (2018)
- **PyTorch:** Deep learning framework
- **Librosa:** Audio processing library
- **Streamlit:** Web app framework

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Made with ❤️ using PyTorch & Streamlit**

**Model Accuracy: 69.98%** | **8 Emotions** | **Real-time Prediction**
