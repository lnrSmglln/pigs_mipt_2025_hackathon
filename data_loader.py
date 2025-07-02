import numpy as np
import pandas as pd
import os
from scipy.signal import butter, filtfilt

def load_data(data_dir):
    """Load PPG data and labels"""
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    # Load signals
    def load_signals(ids):
        return [np.load(os.path.join(data_dir, "ppgs", f"{id}.npy")) for id in ids]
    
    X_train = load_signals(train_df['ID'])
    y_train = train_df['ЛПНП'].values
    X_test = load_signals(test_df['ID'])
    
    return X_train, y_train, test_df['ID'].values

def preprocess_signal(signal, fs=100):
    """PPG-specific preprocessing pipeline"""
    # 1. Normalization
    signal = (signal - np.mean(signal)) / np.std(signal)
    
    # 2. Bandpass filtering (0.5-8 Hz)
    nyq = 0.5 * fs
    b, a = butter(3, [0.5/nyq, 8/nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    
    return filtered