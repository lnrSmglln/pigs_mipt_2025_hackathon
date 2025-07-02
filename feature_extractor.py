import numpy as np
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from scipy import stats
from data_loader import preprocess_signal

def extract_pulse_features(signal, fs=100):
    """Extract pulse waveform characteristics"""
    peaks, _ = find_peaks(signal, distance=50, prominence=0.5)
    if len(peaks) < 3:
        return {k: 0 for k in ['pulse_amp', 'crest_time', 'half_width', 'aug_index']}
    
    # Extract individual pulses
    pulse_waves = []
    for i in range(1, len(peaks)-1):
        pulse = signal[peaks[i-1]:peaks[i+1]]
        pulse_waves.append(pulse)
    
    # Compute average pulse
    max_len = max(len(p) for p in pulse_waves)
    padded = [np.pad(p, (0, max_len-len(p)), 'constant', constant_values=np.min(p)) for p in pulse_waves]
    avg_pulse = np.mean(padded, axis=0)
    
    # Calculate features
    min_val = np.min(avg_pulse)
    max_val = np.max(avg_pulse)
    max_idx = np.argmax(avg_pulse)
    
    return {
        'pulse_amp': max_val - min_val,
        'crest_time': max_idx / fs,
        'half_width': np.sum(avg_pulse > (min_val + 0.5*(max_val-min_val))) / fs,
        'aug_index': (avg_pulse[-1] - avg_pulse[0]) / (max_val - min_val)
    }

def extract_freq_features(signal, fs=100):
    """Frequency domain analysis"""
    n = len(signal)
    yf = rfft(signal)
    xf = rfftfreq(n, 1/fs)
    
    # Cardiac frequency band (0.8-4 Hz)
    mask = (xf >= 0.8) & (xf <= 4)
    if not np.any(mask):
        return {k: 0 for k in ['dominant_freq', 'spectral_entropy', 'hf_power']}
    
    psd = np.abs(yf[mask])**2
    total_power = np.sum(psd) + 1e-12
    norm_psd = psd / total_power
    
    return {
        'dominant_freq': xf[mask][np.argmax(psd)],
        'spectral_entropy': -np.sum(norm_psd * np.log2(norm_psd + 1e-12)),
        'hf_power': np.sum(psd[xf[mask] > 2]) / total_power
    }

def extract_features(signal, fs=100):
    """Master feature extraction function"""
    preprocessed = preprocess_signal(signal, fs)
    features = {}
    
    # Time-domain features
    features.update({
        'mean': np.mean(preprocessed),
        'std': np.std(preprocessed),
        'skew': stats.skew(preprocessed),
        'kurtosis': stats.kurtosis(preprocessed),
        'rms': np.sqrt(np.mean(preprocessed**2))
    })
    
    # Pulse waveform features
    features.update(extract_pulse_features(preprocessed, fs))
    
    # Frequency features
    features.update(extract_freq_features(preprocessed, fs))
    
    return features