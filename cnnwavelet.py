import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import pywt
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.model_selection import train_test_split
from scipy import signal
from scipy.stats import kurtosis, skew

# Configuration - optimized values
config = {
    "wavelet": "sym6",
    "decomp_level": 5,
    "target_freq_band": (0.5, 5.0),
    "batch_size": 16,
    "learning_rate": 0.001,
    "epochs": 30,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "fs": 100
}

# 1. Wavelet Processing Class - optimized
class WaveletProcessor:
    def __init__(self, config):
        self.config = config
        self.wavelet = pywt.Wavelet(config["wavelet"])
        self.max_len = 2 ** int(np.ceil(np.log2(2700)))  # Precompute for 2700 samples
        
    def transform(self, ppg_signal):
        """Apply Stationary Wavelet Transform (SWT) to PPG signal"""
        # Pad signal to power of 2
        padded = np.pad(ppg_signal, (0, self.max_len - len(ppg_signal)), 'reflect')
        
        # Apply SWT
        coeffs = pywt.swt(padded, self.wavelet, level=self.config["decomp_level"], trim_approx=True)
        
        # Filter relevant frequency bands
        filtered_coeffs = []
        for i, (cA, cD) in enumerate(coeffs):
            freq_high = self.config["fs"] / (2 ** (i + 1))
            freq_low = max(0.1, freq_high / 2)  # Avoid division by zero
            
            if (freq_low <= self.config["target_freq_band"][1] and 
                freq_high >= self.config["target_freq_band"][0]):
                filtered_coeffs.append(cA[:len(ppg_signal)])
                filtered_coeffs.append(cD[:len(ppg_signal)])
        
        return np.stack(filtered_coeffs)

# 2. PPG Dataset with Wavelet Processing - fixed and optimized
class PPGWaveletDataset(Dataset):
    def __init__(self, df, wavelet_processor, target_col='ЛПНП', training=True):
        self.data = df
        self.target_col = target_col
        self.training = training
        self.wavelet_processor = wavelet_processor
        self.fs = 100
        self.nyq = 0.5 * self.fs
        self.b, self.a = signal.butter(3, [0.5/self.nyq, 8/self.nyq], 'bandpass')  # Precompute filter
        
    def __len__(self):
        return len(self.data)
    
    def preprocess_signal(self, ppg_signal):
        """Preprocess PPG signal - FIXED"""
        # 1. Bandpass filtering (0.5-8 Hz)
        filtered = signal.filtfilt(self.b, self.a, ppg_signal)
        
        # 2. Normalization
        return (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    def augment_signal(self, ppg_signal):
        """Data augmentation for training"""
        # Add Gaussian noise
        if np.random.rand() > 0.7:
            ppg_signal += np.random.normal(0, 0.02, len(ppg_signal))
        
        # Random scaling
        if np.random.rand() > 0.7:
            ppg_signal *= np.random.uniform(0.9, 1.1)
            
        # Random shift
        if np.random.rand() > 0.7:
            shift = np.random.randint(-50, 50)
            ppg_signal = np.roll(ppg_signal, shift)
            # Apply zero padding to rolled edges
            if shift > 0:
                ppg_signal[:shift] = 0
            else:
                ppg_signal[shift:] = 0
                
        return ppg_signal
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_id = row['ID']
        ppg = np.load(f'data/ppgs/{file_id}.npy')
        
        # Preprocessing - FIXED
        ppg = self.preprocess_signal(ppg)
        
        # Augmentation (training only)
        if self.training:
            ppg = self.augment_signal(ppg)
        
        # Wavelet transform
        wavelet_coeffs = self.wavelet_processor.transform(ppg)
        
        # Convert to tensor
        wavelet_coeffs = torch.tensor(wavelet_coeffs, dtype=torch.float32)
        
        if self.training:
            target = torch.tensor(row[self.target_col], dtype=torch.float32)
            return wavelet_coeffs, target
        return wavelet_coeffs, file_id

# 3. Optimized Wavelet-CNN Model
class WaveletCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2)
        )
        
        # Residual blocks with dilation
        self.res_blocks = nn.Sequential(
            ResBlock(64, 64, dilation=1),
            ResBlock(64, 128, dilation=2),
            ResBlock(128, 128, dilation=4),
            ResBlock(128, 256, dilation=8)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        attn_weights = self.attention(x)
        x = x * attn_weights
        return self.classifier(x)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Identity()
        self.relu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# 4. Training and Validation Loop with improvements
def train_model(model, train_loader, val_loader, optimizer, criterion, config):
    best_val_auc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
    
    for epoch in range(config["epochs"]):
        # Training phase
        model.train()
        train_loss = 0
        for signals, targets in train_loader:
            signals, targets = signals.to(config["device"]), targets.to(config["device"])
            
            # Forward pass
            outputs = model(signals)
            loss = criterion(outputs.squeeze(), targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * signals.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for signals, targets in val_loader:
                signals, targets = signals.to(config["device"]), targets.to(config["device"])
                
                outputs = model(signals)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * signals.size(0)
                
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(targets.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_auc = roc_auc_score(all_targets, all_probs)
        
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'model/wavelet_cnn_best.pth')
        
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"Best AUC: {best_val_auc:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_auc'], 'r-', label='Val AUC')
    plt.title('Validation AUC')
    plt.legend()
    plt.savefig('outputs/training_history.png', bbox_inches='tight')
    
    return history

# 5. Optimized Main Execution
if __name__ == "__main__":
    # Initialize wavelet processor
    wavelet_processor = WaveletProcessor(config)
    
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # Split data
    train_sub, val_sub = train_test_split(
        train_df, test_size=0.2, stratify=train_df['ЛПНП'], random_state=42
    )
    train_sub = train_sub.reset_index(drop=True)
    val_sub = val_sub.reset_index(drop=True)
    
    # Create datasets
    train_dataset = PPGWaveletDataset(train_sub, wavelet_processor, training=True)
    val_dataset = PPGWaveletDataset(val_sub, wavelet_processor, training=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    # Determine input channels
    sample_input, _ = train_dataset[0]
    in_channels = sample_input.shape[0]
    
    # Initialize model
    model = WaveletCNN(in_channels).to(config["device"])
    
    # Calculate class weights
    pos_weight = torch.tensor([(len(train_df) - sum(train_df['ЛПНП'])) / sum(train_df['ЛПНП'])])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(config["device"]))
    
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], 
                           weight_decay=1e-4, amsgrad=True)
    
    # Train model
    history = train_model(
        model, train_loader, val_loader, optimizer, criterion, config
    )
    
    # Generate predictions
    model.load_state_dict(torch.load('model/wavelet_cnn_best.pth'))
    model.eval()
    
    test_dataset = PPGWaveletDataset(test_df, wavelet_processor, training=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    test_preds = []
    test_ids = []
    
    with torch.no_grad():
        for signals, ids in test_loader:
            signals = signals.to(config["device"])
            outputs = model(signals)
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
            test_preds.extend(probs)
            test_ids.extend(ids)
    
    # Create submission
    submission = pd.DataFrame({'ID': test_ids, 'ЛПНП': test_preds})
    submission.to_csv('outputs/wavelet_cnn_submission.csv', index=False)
    print("Submission saved to outputs/wavelet_cnn_submission.csv")