import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os
import tempfile
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# MODEL ARCHITECTURE (Same as training)
# ============================================================
class ImprovedMLP(nn.Module):
    """Enhanced MLP with reduced complexity and higher dropout"""
    def __init__(self, input_size, num_classes):
        super(ImprovedMLP, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1 - Reduced from 1024 to 512
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            
            # Layer 2 - Reduced from 512 to 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            
            # Layer 3 - Reduced from 256 to 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            
            # Output layer
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)

# ============================================================
# FEATURE EXTRACTION (Same as training)
# ============================================================
def extract_features(audio, sr, max_pad_len=174):
    """Extract features from audio array"""
    try:
        # 1. MFCC with deltas
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 2. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(audio)
        
        # 3. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        
        # Pad or truncate helper
        def pad_or_truncate(feature, max_len):
            if feature.shape[1] < max_len:
                pad_width = max_len - feature.shape[1]
                feature = np.pad(feature, pad_width=((0,0),(0, pad_width)), mode='constant')
            else:
                feature = feature[:, :max_len]
            return feature
        
        # Apply padding/truncation
        mfcc = pad_or_truncate(mfcc, max_pad_len)
        mfcc_delta = pad_or_truncate(mfcc_delta, max_pad_len)
        mfcc_delta2 = pad_or_truncate(mfcc_delta2, max_pad_len)
        spectral_centroid = pad_or_truncate(spectral_centroid, max_pad_len)
        spectral_rolloff = pad_or_truncate(spectral_rolloff, max_pad_len)
        zero_crossing = pad_or_truncate(zero_crossing, max_pad_len)
        chroma = pad_or_truncate(chroma, max_pad_len)
        
        # Statistical features (mean and std only)
        stats_features = []
        for feat in [mfcc, mfcc_delta, spectral_centroid, spectral_rolloff]:
            stats_features.extend([
                np.mean(feat, axis=1),
                np.std(feat, axis=1)
            ])
        stats_features = np.concatenate(stats_features)
        
        # Combine all features
        combined = np.concatenate([
            mfcc.flatten(),
            mfcc_delta.flatten(),
            mfcc_delta2.flatten(),
            spectral_centroid.flatten(),
            spectral_rolloff.flatten(),
            zero_crossing.flatten(),
            chroma.flatten(),
            stats_features
        ])
        
        return combined
        
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

# ============================================================
# LOAD MODEL AND PREPROCESSING
# ============================================================
@st.cache_resource
def load_model_and_scaler():
    """Load trained model and create scaler"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define emotion labels (same order as training)
    emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    # Create dummy input to get feature size
    dummy_audio = np.random.randn(22050 * 3)  # 3 seconds
    dummy_features = extract_features(dummy_audio, 22050)
    input_size = len(dummy_features)
    
    # Load model
    model = ImprovedMLP(input_size=input_size, num_classes=len(emotion_labels)).to(device)
    
    if os.path.exists('best_model.pt'):
        model.load_state_dict(torch.load('best_model.pt', map_location=device))
        model.eval()
    else:
        st.error("Model file 'best_model.pt' not found!")
        return None, None, None, None
    
    # Create a scaler (we'll use a simple standardization)
    # In production, you should save and load the actual scaler from training
    scaler = StandardScaler()
    
    return model, scaler, emotion_labels, device

# ============================================================
# PREDICTION FUNCTION
# ============================================================
def predict_emotion(audio_file, model, scaler, emotion_labels, device):
    """Predict emotion from audio file"""
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=22050, duration=3, mono=True)
        
        # Extract features
        features = extract_features(audio, sr)
        if features is None:
            return None, None, None
        
        # Reshape and scale
        features = features.reshape(1, -1)
        
        # For simplicity, we'll normalize using the features themselves
        # In production, use the saved scaler from training
        features = (features - features.mean()) / (features.std() + 1e-8)
        
        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_idx].item() * 100
        
        predicted_emotion = emotion_labels[predicted_idx]
        all_probs = {emotion_labels[i]: probabilities[0][i].item() * 100 
                     for i in range(len(emotion_labels))}
        
        return predicted_emotion, confidence, all_probs
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None

# ============================================================
# STREAMLIT UI
# ============================================================
def main():
    # Page config
    st.set_page_config(
        page_title="Emotion Recognition from Speech",
        page_icon="",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }
        .emotion-box {
            padding: 2rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
        }
        .prediction {
            font-size: 2.5rem;
            font-weight: bold;
        }
        .confidence {
            font-size: 1.5rem;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🎭 Speech Emotion Recognition</p>', unsafe_allow_html=True)
    st.markdown("### Analyze emotions from speech using Deep Learning")
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/voice-recognition.png", width=100)
        st.markdown("## Model Info")
        st.markdown("""
        - **Accuracy:** 69.98%
        - **Model:** 3-Layer MLP
        - **Parameters:** 6.9M
        - **Emotions:** 8 classes
        - **GPU:** RTX 4050
        """)
        
        st.markdown("---")
        st.markdown("## 🎯 Supported Emotions")
        emotions_display = {
            'angry': '😠',
            'calm': '😌',
            'disgust': '🤢',
            'fearful': '😨',
            'happy': '😊',
            'neutral': '😐',
            'sad': '😢',
            'surprised': '😮'
        }
        for emotion, emoji in emotions_display.items():
            st.markdown(f"{emoji} **{emotion.title()}**")
    
    # Load model
    with st.spinner("Loading model..."):
        model, scaler, emotion_labels, device = load_model_and_scaler()
    
    if model is None:
        st.error("Failed to load model. Please ensure 'best_model.pt' exists!")
        return
    
    st.success("Model loaded successfully!")
    
    # File uploader
    st.markdown("---")
    st.markdown("## Upload Audio File")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV format recommended)",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload a speech audio file to analyze emotions"
        )
    
    with col2:
        st.markdown("### Tips")
        st.markdown("""
        - Use clear speech audio
        - 3 seconds optimal
        - WAV format preferred
        - No background noise
        """)
    
    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Predict button
        if st.button("Analyze Emotion", type="primary", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                predicted_emotion, confidence, all_probs = predict_emotion(
                    tmp_path, model, scaler, emotion_labels, device
                )
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            if predicted_emotion is not None:
                # Display results
                st.markdown("---")
                st.markdown("## Results")
                
                # Emotion emoji mapping
                emotion_emojis = {
                    'angry': '😠',
                    'calm': '😌',
                    'disgust': '🤢',
                    'fearful': '😨',
                    'happy': '😊',
                    'neutral': '😐',
                    'sad': '😢',
                    'surprised': '😮'
                }
                
                # Main prediction
                emoji = emotion_emojis.get(predicted_emotion, '🎭')
                
                # Color based on confidence
                if confidence >= 70:
                    color = "#4CAF50"  # Green
                elif confidence >= 50:
                    color = "#FF9800"  # Orange
                else:
                    color = "#F44336"  # Red
                
                st.markdown(f"""
                    <div class="emotion-box" style="background-color: {color}20; border: 3px solid {color};">
                        <div class="prediction">{emoji} {predicted_emotion.upper()}</div>
                        <div class="confidence">Confidence: {confidence:.2f}%</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Probability distribution
                st.markdown("### Probability Distribution")
                
                # Sort probabilities
                sorted_probs = dict(sorted(all_probs.items(), key=lambda x: x[1], reverse=True))
                
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(sorted_probs.keys()),
                        y=list(sorted_probs.values()),
                        marker=dict(
                            color=list(sorted_probs.values()),
                            colorscale='RdYlGn',
                            showscale=True,
                            colorbar=dict(title="Confidence %")
                        ),
                        text=[f'{v:.1f}%' for v in sorted_probs.values()],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Emotion Probabilities",
                    xaxis_title="Emotion",
                    yaxis_title="Probability (%)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed probabilities
                st.markdown("### Detailed Probabilities")
                col1, col2, col3, col4 = st.columns(4)
                
                for i, (emotion, prob) in enumerate(sorted_probs.items()):
                    emoji = emotion_emojis.get(emotion, '🎭')
                    col = [col1, col2, col3, col4][i % 4]
                    with col:
                        st.metric(
                            label=f"{emoji} {emotion.title()}",
                            value=f"{prob:.2f}%"
                        )
            else:
                st.error("Failed to analyze audio. Please try another file.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Built with PyTorch & Streamlit | Deployed Model Accuracy: 69.98%</p>
            <p>Upload your audio and discover the emotion!</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
