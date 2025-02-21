import torch
import torchaudio
import numpy as np
import pandas as pd
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import glob

class AudioFeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=40, n_fft=400, hop_length=320):  # Reduced n_mels, increased hop_length
        self.sample_rate = sample_rate
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform):
        with torch.no_grad():  # Added to speed up feature extraction
            mel_spec = self.mel_spectrogram(waveform)
            mel_db = self.amplitude_to_db(mel_spec)
        return mel_db

class CREMADataset(Dataset):
    def __init__(self, data_df, feature_extractor, max_length=None):
        self.data = data_df
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        self.emotion_to_idx = {emotion: idx for idx, emotion in enumerate(sorted(data_df['emotion'].unique()))}
        
        # Pre-compute features to speed up training
        self.cached_features = {}
        print("Pre-computing features...")
        for idx in tqdm(range(len(data_df))):
            self._cache_features(idx)
    
    def _cache_features(self, idx):
        row = self.data.iloc[idx]
        # Convert path to Mac-style and ensure it's absolute
        file_path = os.path.abspath(row['file_path']).replace('\\', '/')
        
        # Debug print for the first few files
        if idx < 5:
            print(f"Loading file: {file_path}")
            print(f"File exists: {os.path.exists(file_path)}")
            print(f"Directory contents: {os.listdir(os.path.dirname(file_path))}")
        
        try:
            waveform, _ = torchaudio.load(file_path)
            
            if self.max_length:
                if waveform.shape[1] > self.max_length:
                    waveform = waveform[:, :self.max_length]
                elif waveform.shape[1] < self.max_length:
                    pad_amount = self.max_length - waveform.shape[1]
                    waveform = torch.nn.functional.pad(waveform, (0, pad_amount))
            
            features = self.feature_extractor(waveform)
            self.cached_features[idx] = features
            
        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"File path exists: {os.path.exists(file_path)}")
            raise

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = self.cached_features[idx]
        label = self.emotion_to_idx[row['emotion']]
        return features, torch.tensor(label)

class LightweightEmotionModel(nn.Module):
    def __init__(self, num_emotions, n_mels=40):  # Reduced n_mels
        super().__init__()
        
        # Simplified CNN layers
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Reduced filters
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        
        # Calculate the size after CNN layers
        self.cnn_output_size = self._get_conv_output_size(n_mels)
        
        # Single layer LSTM
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=128,  # Reduced hidden size
            num_layers=1,     # Reduced to single layer
            batch_first=True,
            bidirectional=True,
            dropout=0.3       # Reduced dropout
        )
        
        # Simplified fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        )
    
    def _get_conv_output_size(self, n_mels):
        dummy_input = torch.zeros(1, 1, n_mels, 100)
        dummy_output = self.conv(dummy_input)
        return dummy_output.size(1) * dummy_output.size(2)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, -1, self.cnn_output_size)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    # Early stopping parameters
    patience = 5
    counter = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return train_losses, val_losses, train_accs, val_accs

def load_data():
    """Load and preprocess the emotion datasets"""
    data = []
    
    # Get absolute path to the archive directory
    current_dir = os.getcwd()
    base_path = os.path.abspath(os.path.join(current_dir, "archive"))
    print(f"Loading data from: {base_path}")
    
    # Load CREMA
    crema_path = os.path.join(base_path, "archive", "Crema")
    print(f"Looking for CREMA data in: {crema_path}")
    print(f"CREMA path exists: {os.path.exists(crema_path)}")
    
    # Print directory contents for debugging
    if os.path.exists(base_path):
        print(f"Contents of archive directory: {os.listdir(base_path)}")
    else:
        print(f"Archive directory not found at: {base_path}")
    
    if os.path.exists(crema_path):
        for file in glob.glob(os.path.join(crema_path, "**", "*.wav"), recursive=True):
            file = file.replace('\\', '/')
            file = os.path.abspath(file)
            filename = os.path.basename(file)
            parts = filename.split('_')
            if len(parts) >= 3:
                emotion = parts[2]
                data.append({
                    'file_path': file,
                    'text': "speech sample",
                    'dataset': 'crema'
                })
    
    # Load RAVDESS
    ravdess_path = os.path.join(base_path, "Ravdess", "audio_speech_actors_01-24", "Actor_*")
    if os.path.exists(os.path.dirname(os.path.dirname(ravdess_path))):
        for actor_dir in glob.glob(ravdess_path):
            for file in glob.glob(os.path.join(actor_dir, "*.wav")):
                file = file.replace('\\', '/')
                file = os.path.abspath(file)
                filename = os.path.basename(file)
                parts = filename.split("-")
                if len(parts) >= 3:
                    statement = "kids are talking by the door" if parts[4] == "01" else "dogs are sitting by the door"
                    data.append({
                        'file_path': file,
                        'text': statement,
                        'dataset': 'ravdess'
                    })
    
    # Load TESS
    tess_path = os.path.join(base_path, "Tess")
    if os.path.exists(tess_path):
        for emotion_dir in glob.glob(os.path.join(tess_path, "*_*")):
            for file in glob.glob(os.path.join(emotion_dir, "*.wav")):
                file = os.path.normpath(file)
                filename = os.path.basename(file)
                word = filename.split('_')[0]
                data.append({
                    'file_path': file,  # Changed from 'path' to 'file_path'
                    'text': word,
                    'dataset': 'tess'
                })
    
    # Load SAVEE
    savee_path = os.path.join(base_path, "Savee")
    if os.path.exists(savee_path):
        for file in glob.glob(os.path.join(savee_path, "**", "*.wav"), recursive=True):
            file = os.path.normpath(file)
            filename = os.path.basename(file)
            data.append({
                'file_path': file,  # Changed from 'path' to 'file_path'
                'text': "speech sample",
                'dataset': 'savee'
            })
    
    # Print directory structure for debugging
    print("\nDirectory structure:")
    print_directory_structure(base_path)
    
    # Print information about the loaded data
    df = pd.DataFrame(data)
    print(f"\nLoaded {len(df)} audio files")
    print("\nFiles per dataset:")
    print(df['dataset'].value_counts())
    
    # Print some example paths
    print("\nExample file paths:")
    for dataset in df['dataset'].unique():
        example = df[df['dataset'] == dataset]['file_path'].iloc[0] if len(df[df['dataset'] == dataset]) > 0 else "No files found"
        print(f"{dataset}: {example}")
    
    return df

def print_directory_structure(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for f in files[:5]:  # Print first 5 files in each directory
            print(f'{subindent}{f}')
        if len(files) > 5:
            print(f'{subindent}...')

def main():
    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if archive directory exists
    archive_path = os.path.join(".", "archive")
    print(f"Archive path: {os.path.abspath(archive_path)}")
    print(f"Archive exists: {os.path.exists(archive_path)}")
    
    # Optimized hyperparameters
    BATCH_SIZE = 64        # Increased batch size
    NUM_EPOCHS = 30        # Reduced epochs
    LEARNING_RATE = 0.001
    MAX_LENGTH = 200       # Reduced sequence length
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load dataset
    df = pd.read_csv('crema_dataset.csv')
    
    # Optional: Use subset of data for faster training during development
    # df = df.sample(frac=0.5, random_state=42)  # Uncomment to use 50% of data
    
    # Split dataset
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
    
    # Create feature extractor and datasets
    feature_extractor = AudioFeatureExtractor()
    train_dataset = CREMADataset(train_df, feature_extractor, max_length=MAX_LENGTH)
    val_dataset = CREMADataset(val_df, feature_extractor, max_length=MAX_LENGTH)
    
    # Create data loaders with more workers for parallel processing
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=4, pin_memory=True)
    
    # Create model
    num_emotions = len(train_dataset.emotion_to_idx)
    model = LightweightEmotionModel(num_emotions=num_emotions).to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train model
    history = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device)
    
    print("Training completed!")

    # Add this to your main function
    print("\nDirectory structure:")
    print_directory_structure(os.path.join(".", "archive"))

if __name__ == '__main__':
    main()

print(f"Current directory: {os.getcwd()}")
print(f"Contents of current directory: {os.listdir('.')}")
if os.path.exists('archive'):
    print(f"Contents of archive directory: {os.listdir('archive')}")
else:
    print("Archive directory not found!")