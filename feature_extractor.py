import numpy as np
from scipy.signal import find_peaks, welch
from scipy.fft import rfft, rfftfreq
from scipy import stats, signal
import pandas as pd
from data_loader import preprocess_signal

def extract_pulse_features(signal, fs=100, peaks=None):
    """Extract pulse waveform characteristics with precomputed peaks"""
    if peaks is None or len(peaks) < 3:
        peaks, _ = find_peaks(signal, distance=fs*0.6, prominence=0.5)
    
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

def extract_beat_features(signal, fs=100, peaks=None):
    """Extract features from individual heart beats with robust NaN handling"""
    if peaks is None or len(peaks) < 3:
        peaks, _ = find_peaks(signal, distance=fs*0.6, prominence=0.5)
    
    # Default values for cases with insufficient beats
    default_beat_features = {
        'mean_amp': np.mean(signal),
        'max_amp': np.max(signal),
        'min_amp': np.min(signal),
        'duration': len(signal)/fs if len(signal)>0 else 0,
        'peak_to_peak': np.ptp(signal),
        'slope': 0,
        'area': np.trapz(signal),
        'std_amp': np.std(signal),
    }
    
    if len(peaks) < 3:
        return {f'beat_{k}': v for k, v in default_beat_features.items()}
    
    # Split signal into individual beats
    beats = []
    for i in range(len(peaks)-1):
        beat = signal[peaks[i]:peaks[i+1]]
        beats.append(beat)
    
    # Extract features for each beat with NaN protection
    beat_features = []
    for beat in beats:
        if len(beat) == 0:  # Handle empty beats
            beat_features.append(default_beat_features)
            continue
            
        try:
            beat_max = np.max(beat)
            beat_min = np.min(beat)
            slope = (beat_max - beat_min) / (len(beat) + 1e-6)
            
            beat_features.append({
                'mean_amp': np.nanmean(beat),
                'max_amp': beat_max,
                'min_amp': beat_min,
                'duration': len(beat)/fs,
                'peak_to_peak': beat_max - beat_min,
                'slope': slope if not np.isnan(slope) else 0,
                'area': np.trapz(beat) if len(beat)>1 else 0,
                'std_amp': np.nanstd(beat) if len(beat)>1 else 0,
            })
        except:
            beat_features.append(default_beat_features)
    
    # Convert to DataFrame with robust aggregation
    df_beat = pd.DataFrame(beat_features)
    
    # Define aggregation rules for each feature
    agg_rules = {
        'mean_amp': 'mean',
        'max_amp': 'max',
        'min_amp': 'min',
        'duration': 'median',  # More robust than mean for duration
        'peak_to_peak': 'median',
        'slope': 'median',
        'area': 'sum',  # Total area under all beats
        'std_amp': 'mean'
    }
    
    # Aggregate with NaN protection
    agg_features = {}
    for col, agg_method in agg_rules.items():
        if agg_method == 'mean':
            agg_features[col] = df_beat[col].mean(skipna=True)
        elif agg_method == 'median':
            agg_features[col] = df_beat[col].median(skipna=True)
        elif agg_method == 'max':
            agg_features[col] = df_beat[col].max(skipna=True)
        elif agg_method == 'min':
            agg_features[col] = df_beat[col].min(skipna=True)
        elif agg_method == 'sum':
            agg_features[col] = df_beat[col].sum(skipna=True)
        
        # Final NaN check - fallback to signal-wide stats
        if pd.isna(agg_features[col]):
            agg_features[col] = default_beat_features[col]
    
    # Add prefix to beat features
    return {f'beat_{k}': v for k, v in agg_features.items()}

# def extract_beat_features(signal, fs=100, peaks=None):
#     """Extract features from individual heart beats"""
#     if peaks is None or len(peaks) < 3:
#         peaks, _ = find_peaks(signal, distance=fs*0.6, prominence=0.5)
    
#     # if len(peaks) < 3:
#     #     return {}
    
#     # Calculate beat intervals and heart rate
#     intervals = np.diff(peaks)
#     heart_rate = 60 / (intervals / fs)
    
#     # Split signal into individual beats
#     beats = []
#     for i in range(len(peaks)-1):
#         beat = signal[peaks[i]:peaks[i+1]]
#         beats.append(beat)
    
#     # Extract features for each beat
#     beat_features = []
#     for beat in beats:
#         # Calculate beat features
#         beat_features.append({
#             'mean_amp': np.mean(beat),
#             'max_amp': np.max(beat),
#             'min_amp': np.min(beat),
#             'duration': len(beat) / fs,
#             'peak_to_peak': np.ptp(beat),
#             'slope': (np.max(beat) - np.min(beat)) / (len(beat) + 1e-6),
#             'area': np.trapz(beat),
#             'std_amp': np.std(beat),
#         })


    
#     # Aggregate features across all beats
#     df_beat = pd.DataFrame(beat_features)
#     agg_features = df_beat.mean().to_dict()
    
#     # Add prefix to beat features
#     return {f'beat_{k}': v for k, v in agg_features.items()}

def extract_hrv_features(peaks, fs=100):
    """Extract Heart Rate Variability features"""
    if len(peaks) < 4:
        return {
            'sdnn': 0,
            'rmssd': 0,
            'pnn50': 0
        }
    
    # Calculate NN intervals in milliseconds
    nn_intervals = np.diff(peaks) / fs * 1000
    sdnn = np.std(nn_intervals)
    
    # Calculate RMSSD and pNN50
    diff_nn = np.diff(nn_intervals)
    rmssd = np.sqrt(np.mean(diff_nn**2))
    pnn50 = (np.sum(np.abs(diff_nn) > 50) / len(diff_nn)) * 100
    
    return {
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50
    }

def extract_gradient_features(signal):
    """Extract signal gradient features"""
    gradient = np.gradient(signal)
    second_deriv = np.gradient(gradient)
    
    # Zero crossings in second derivative
    zero_crossings = np.where(np.diff(np.sign(second_deriv)))[0]
    
    return {
        'max_gradient': np.max(gradient),
        'min_gradient': np.min(gradient),
        'mean_gradient': np.mean(gradient),
        'zero_crossings': len(zero_crossings)
    }

def extract_freq_features(signal, fs=100):
    """Enhanced frequency domain analysis"""
    n = len(signal)
    
    # FFT-based features
    yf = rfft(signal)
    xf = rfftfreq(n, 1/fs)
    psd = np.abs(yf)**2
    
    # Cardiac frequency band (0.8-4 Hz)
    hr_mask = (xf >= 0.8) & (xf <= 4)
    if not np.any(hr_mask):
        hr_features = {
            'dominant_freq': 0,
            'dominant_amp': 0,
            'spectral_entropy': 0
        }
    else:
        hr_psd = psd[hr_mask]
        total_power = np.sum(hr_psd) + 1e-12
        norm_psd = hr_psd / total_power
        
        dominant_idx = np.argmax(hr_psd)
        hr_features = {
            'dominant_freq': xf[hr_mask][dominant_idx],
            'dominant_amp': np.abs(yf[hr_mask][dominant_idx]) / n * 2,
            'spectral_entropy': -np.sum(norm_psd * np.log2(norm_psd + 1e-12))
        }
    
    # LF/HF power ratio (0.04-0.15 Hz and 0.15-0.4 Hz)
    lf_mask = (xf >= 0.04) & (xf < 0.15)
    hf_mask = (xf >= 0.15) & (xf < 0.4)
    
    lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else 1e-12
    hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 1e-12
    
    # Respiration rate (0.1-0.4 Hz)
    resp_mask = (xf >= 0.1) & (xf <= 0.4)
    resp_rate = 0
    if np.any(resp_mask):
        resp_psd = psd[resp_mask]
        resp_rate = xf[resp_mask][np.argmax(resp_psd)] * 60
    
    return {
        **hr_features,
        'power_ratio': lf_power / hf_power,
        'respiration_rate': resp_rate,
        'hf_power': hf_power
    }

def extract_features(signal, fs=100):
    """Master feature extraction function"""
    preprocessed = preprocess_signal(signal, fs)
    features = {}
    
    # Find peaks once and reuse
    peaks, _ = find_peaks(preprocessed, distance=fs*0.6, prominence=0.5)
    
    # Time-domain features
    features.update({
        'mean': np.mean(preprocessed),
        'std': np.std(preprocessed),
        'skew': stats.skew(preprocessed),
        'kurtosis': stats.kurtosis(preprocessed),
        'rms': np.sqrt(np.mean(preprocessed**2))
    })
    
    # Pulse waveform features
    features.update(extract_pulse_features(preprocessed, fs, peaks))
    
    # Beat features
    features.update(extract_beat_features(preprocessed, fs, peaks))
    
    # HRV features
    features.update(extract_hrv_features(peaks, fs))
    
    # Gradient features
    features.update(extract_gradient_features(preprocessed))
    
    # Frequency features
    features.update(extract_freq_features(preprocessed, fs))
    
    return features