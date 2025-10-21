import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import wave
import struct
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATASET_PATH = "./audio_speech_actors_01-24"
BATCH_SIZE = 32  # Smaller batch for better generalization
NUM_WORKERS = 4  # For faster data loading
EPOCHS = 200  # More epochs with early stopping
PATIENCE = 20  # Longer patience
LEARNING_RATE = 0.0005  # Lower learning rate for stability

# ============================================================
# GPU DETECTION AND SETUP
# ============================================================
def setup_device():
    """Detect and configure GPU"""
    print("="*60)
    print("GPU DETECTION")
    print("="*60)
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA Available: YES")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch Version: {torch.__version__}")
        
        # Enable cudnn optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        
        # Clear cache
        torch.cuda.empty_cache()
        print(f"   Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"   Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    else:
        device = torch.device("cpu")
        print(f"⚠️  CUDA Available: NO")
        print(f"   Using CPU instead")
        print(f"\n💡 To use GPU, install: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    
    print("="*60)
    return device

# ============================================================
# FEATURE EXTRACTION
# ============================================================
def extract_features(file_path, max_pad_len=174):
    """Extract comprehensive audio features"""
    try:
        # Load audio with fixed parameters
        audio, sr = load_audio_manual(file_path, duration=3)
        
        # 1. Mel-spectrogram
        melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
        logspec = librosa.power_to_db(melspec, ref=np.max)
        
        # 2. MFCC with deltas
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # 3. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(audio)
        
        # 4. Chroma features
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
        logspec = pad_or_truncate(logspec, max_pad_len)
        mfcc = pad_or_truncate(mfcc, max_pad_len)
        mfcc_delta = pad_or_truncate(mfcc_delta, max_pad_len)
        mfcc_delta2 = pad_or_truncate(mfcc_delta2, max_pad_len)
        spectral_centroid = pad_or_truncate(spectral_centroid, max_pad_len)
        spectral_rolloff = pad_or_truncate(spectral_rolloff, max_pad_len)
        spectral_contrast = pad_or_truncate(spectral_contrast, max_pad_len)
        zero_crossing = pad_or_truncate(zero_crossing, max_pad_len)
        chroma = pad_or_truncate(chroma, max_pad_len)
        
        # Statistical features
        stats_features = []
        for feat in [mfcc, mfcc_delta, mfcc_delta2, spectral_centroid, spectral_rolloff]:
            stats_features.extend([
                np.mean(feat, axis=1),
                np.std(feat, axis=1),
                np.max(feat, axis=1),
                np.min(feat, axis=1)
            ])
        stats_features = np.concatenate(stats_features)
        
        # Combine all features
        combined = np.concatenate([
            logspec.flatten(),
            mfcc.flatten(),
            mfcc_delta.flatten(),
            mfcc_delta2.flatten(),
            spectral_centroid.flatten(),
            spectral_rolloff.flatten(),
            spectral_contrast.flatten(),
            zero_crossing.flatten(),
            chroma.flatten(),
            stats_features
        ])
        
        return combined
        
    except Exception as e:
        print(f"\n⚠️  Error: {file_path}: {str(e)}")
        return None

# ============================================================
# DATA AUGMENTATION
# ============================================================
def augment_audio(audio, sr):
    """Apply data augmentation to audio"""
    augmented = [audio]  # Original
    
    # 1. Add noise
    noise = np.random.randn(len(audio)) * 0.005
    augmented.append(audio + noise)
    
    # 2. Time stretch (speed up/slow down)
    augmented.append(librosa.effects.time_stretch(audio, rate=0.9))
    augmented.append(librosa.effects.time_stretch(audio, rate=1.1))
    
    # 3. Pitch shift
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=2))
    augmented.append(librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2))
    
    return augmented

def extract_features_with_augmentation(file_path, max_pad_len=174, augment=False):
    """Extract features with optional augmentation"""
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=22050, duration=3, mono=True)
        
        if augment:
            # Create augmented versions
            audio_versions = augment_audio(audio, sr)
        else:
            audio_versions = [audio]
        
        features_list = []
        for audio_version in audio_versions:
            # Extract features for this version
            features = _extract_single_features(audio_version, sr, max_pad_len)
            if features is not None:
                features_list.append(features)
        
        return features_list
        
    except Exception as e:
        print(f"\n⚠️  Error: {file_path}: {str(e)}")
        return None

def _extract_single_features(audio, sr, max_pad_len=174):
    """Extract features from a single audio array"""
    try:
        # 1. MFCC with deltas (reduced from 40 to 20 for simplicity)
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
        
        # Combine all features (simplified)
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
        return None

# ============================================================
# DATA LOADING
# ============================================================
def verify_dataset():
    """Verify dataset exists and count files"""
    print("\n" + "="*60)
    print("DATASET VERIFICATION")
    print("="*60)
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Path not found: {DATASET_PATH}")
        print(f"   Current dir: {os.getcwd()}")
        return False
    
    # Count files
    total_wav = 0
    folders = []
    for root, dirs, files in os.walk(DATASET_PATH):
        wav_files = [f for f in files if f.endswith('.wav')]
        if wav_files:
            folders.append(os.path.basename(root))
            total_wav += len(wav_files)
    
    print(f"✅ Dataset found")
    print(f"   Actor folders: {len(folders)}")
    print(f"   Total .wav files: {total_wav}")
    print("="*60)
    
    return total_wav > 0

def load_audio_manual(file_path, duration=3):
    """Load WAV file without librosa - YOU WRITE THIS"""
    with wave.open(file_path, 'rb') as wav_file:
        # Get audio parameters
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Calculate how many frames to read
        frames_to_read = min(int(duration * framerate), n_frames)
        
        # Read audio data
        raw_data = wav_file.readframes(frames_to_read)
        
        # Convert bytes to numbers
        if sample_width == 2:  # 16-bit audio
            audio_data = struct.unpack(f'{frames_to_read * n_channels}h', raw_data)
        
        # Convert to mono if stereo
        if n_channels == 2:
            audio_data = [audio_data[i] for i in range(0, len(audio_data), 2)]
        
        # Normalize to [-1, 1]
        audio_data = [x / 32768.0 for x in audio_data]
        
        return audio_data, framerate


def load_data():
    """Load dataset with progress bar"""
    X, y = [], []
    emotion_dict = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    # Get all wav files
    all_files = []
    for root, dirs, files in os.walk(DATASET_PATH):
        for file in files:
            if file.endswith('.wav'):
                all_files.append(os.path.join(root, file))
    
    print(f"\n📊 Extracting features from {len(all_files)} files (with augmentation)...")
    
    # Process with progress bar - Use augmentation for training diversity
    for file_path in tqdm(all_files, desc="Processing", unit="file"):
        features_list = extract_features_with_augmentation(file_path, augment=True)
        if features_list is not None:
            filename = os.path.basename(file_path)
            emotion_code = filename.split('-')[2]
            emotion = emotion_dict.get(emotion_code)
            if emotion:
                # Add all augmented versions
                for features in features_list:
                    X.append(features)
                    y.append(emotion)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n✅ Feature extraction complete")
    print(f"   Shape: {X.shape}")
    print(f"   Samples: {len(y)}")
    print(f"   Emotion distribution:")
    for emotion, count in sorted(zip(*np.unique(y, return_counts=True))):
        print(f"      {emotion:12s}: {count:3d}")
    
    return X, y, emotion_dict

# ============================================================
# MODEL ARCHITECTURE
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
# TRAINING
# ============================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_x, batch_y in pbar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader), 100 * correct / total

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device):
    """Full training loop"""
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Check improvement
        improved = ""
        if val_loss < best_val_loss - 1e-4 or val_acc > best_val_acc:
            best_val_loss = min(val_loss, best_val_loss)
            best_val_acc = max(val_acc, best_val_acc)
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
            improved = "✓ BEST"
        else:
            epochs_no_improve += 1
            improved = f"({epochs_no_improve}/{PATIENCE})"
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train: {train_loss:.4f} / {train_acc:6.2f}% | "
              f"Val: {val_loss:.4f} / {val_acc:6.2f}% | {improved}")
        
        # Early stopping
        if epochs_no_improve >= PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
        
        # GPU memory info every 10 epochs
        if (epoch + 1) % 10 == 0 and torch.cuda.is_available():
            print(f"   GPU Memory: {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB")
    
    return model, history

# ============================================================
# EVALUATION
# ============================================================
def plot_training_history(history):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("   Saved: training_history.png")
    plt.show()

def evaluate_model(model, test_loader, device, emotion_dict):
    """Complete evaluation with metrics and plots"""
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()
    
    all_preds, all_labels = [], []
    
    print("\n📊 Evaluating on test set...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Testing"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Weighted F1: {f1:.4f}")
    print("="*60)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=list(emotion_dict.values()),
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=list(emotion_dict.values()),
                yticklabels=list(emotion_dict.values()))
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title("Confusion Matrix (Counts)")
    
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax2,
                xticklabels=list(emotion_dict.values()),
                yticklabels=list(emotion_dict.values()),
                vmin=0, vmax=1)
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Normalized Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n   Saved: confusion_matrix.png")
    plt.show()
    
    # Per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    plt.figure(figsize=(12, 6))
    bars = plt.bar(list(emotion_dict.values()), per_class_acc, 
                   color='steelblue', alpha=0.8, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    plt.xlabel('Emotion')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print("   Saved: per_class_accuracy.png")
    plt.show()
    
    return accuracy, f1

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Setup
    device = setup_device()
    
    # Verify dataset
    if not verify_dataset():
        print("\n❌ Cannot proceed without dataset!")
        exit()
    
    # Load data
    X, y, emotion_dict = load_data()
    
    if len(X) == 0:
        print("\n❌ No data loaded!")
        exit()
    
    # Preprocessing
    print("\n📊 Preprocessing...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
    )
    
    # Create datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    # Data loaders with pinned memory for faster GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=NUM_WORKERS, 
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=NUM_WORKERS, 
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=NUM_WORKERS, 
                            pin_memory=True)
    
    print(f"   Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Initialize model
    model = ImprovedMLP(input_size=X.shape[1], num_classes=len(np.unique(y))).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.02)  # Increased weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10  # Longer patience
    )
    
    # Train
    model, history = train_model(model, train_loader, val_loader, 
                                 criterion, optimizer, scheduler, device)
    
    # Plot training history
    print("\n📈 Plotting training history...")
    plot_training_history(history)
    
    # Evaluate
    accuracy, f1 = evaluate_model(model, test_loader, device, emotion_dict)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Model saved: best_model.pt")
    print("="*60)
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()