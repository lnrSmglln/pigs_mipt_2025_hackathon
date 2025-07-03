import numpy as np
from scipy.signal import find_peaks, welch, butter, filtfilt, argrelextrema
from scipy.fft import rfft, rfftfreq
from scipy import stats, signal
import pandas as pd
from scipy.stats import entropy
from data_loader import preprocess_signal

def extract_pulse_features(signal, fs=100, peaks=None):
    """Extract pulse waveform characteristics with precomputed peaks"""
    if peaks is None or len(peaks) < 3:
        peaks, _ = find_peaks(signal, distance=fs*0.6, prominence=0.5)
    
    if len(peaks) < 3:
        return {k: 0 for k in ['pulse_amp', 'crest_time', 'half_width', 'aug_index', 'pulse_rise_time', 'pulse_decay_time', 'pulse_asymmetry']}
    
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
    
    # Calculate rise time (10% to 90% of amplitude)
    rise_start = np.where(avg_pulse[:max_idx] > min_val + 0.1*(max_val-min_val))[0]
    rise_end = np.where(avg_pulse[:max_idx] > min_val + 0.9*(max_val-min_val))[0]
    rise_time = (rise_end[0] - rise_start[0]) / fs if len(rise_start) > 0 and len(rise_end) > 0 else 0
    
    # Calculate decay time (90% to 10% of amplitude)
    decay_start = np.where(avg_pulse[max_idx:] > min_val + 0.9*(max_val-min_val))[0]
    decay_end = np.where(avg_pulse[max_idx:] > min_val + 0.1*(max_val-min_val))[0]
    decay_time = (decay_end[-1] - decay_start[0]) / fs if len(decay_start) > 0 and len(decay_end) > 0 else 0
    
    # Handle potential NaN in pulse_asymmetry
    total_time = rise_time + decay_time
    pulse_asymmetry = rise_time / total_time if total_time > 0 else 0
    
    return {
        'pulse_amp': max_val - min_val,
        'crest_time': max_idx / fs,
        'half_width': np.sum(avg_pulse > (min_val + 0.5*(max_val-min_val))) / fs,
        'aug_index': (avg_pulse[-1] - avg_pulse[0]) / (max_val - min_val),
        'pulse_rise_time': rise_time,
        'pulse_decay_time': decay_time,
        'pulse_asymmetry': pulse_asymmetry
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
        'systolic_diastolic_ratio': 1.0,  # Default ratio
        'beat_entropy': 0.0  # Default entropy
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
            
            # Find systolic and diastolic peaks
            systolic_diastolic_ratio = 1.0  # Default value
            if len(beat) > 20:
                # Find local maxima (potential diastolic peak)
                maxima = argrelextrema(beat, np.greater)[0]
                if len(maxima) > 0:
                    diastolic_peak = maxima[-1] if maxima[-1] > len(beat)*0.5 else len(beat)-1
                    # Prevent division by zero and handle small values
                    if beat[diastolic_peak] > 1e-6:
                        systolic_diastolic_ratio = beat_max / beat[diastolic_peak]
            
            # Calculate beat entropy safely
            hist, _ = np.histogram(beat, bins=10)
            hist_sum = np.sum(hist)
            beat_entropy_val = entropy(hist) if hist_sum > 0 and len(hist[hist > 0]) > 1 else 0
            
            beat_features.append({
                'mean_amp': np.nanmean(beat),
                'max_amp': beat_max,
                'min_amp': beat_min,
                'duration': len(beat)/fs,
                'peak_to_peak': beat_max - beat_min,
                'slope': slope if not np.isnan(slope) else 0,
                'area': np.trapz(beat) if len(beat)>1 else 0,
                'std_amp': np.nanstd(beat) if len(beat)>1 else 0,
                'systolic_diastolic_ratio': systolic_diastolic_ratio,
                'beat_entropy': beat_entropy_val
            })
        except Exception as e:
            # Fallback to default values on error
            print(f"Error processing beat: {e}")
            beat_features.append(default_beat_features)
    
    # Convert to DataFrame with robust aggregation
    df_beat = pd.DataFrame(beat_features)
    
    # Define aggregation rules for each feature
    agg_rules = {
        'mean_amp': 'mean',
        'max_amp': 'max',
        'min_amp': 'min',
        'duration': 'median',
        'peak_to_peak': 'median',
        'slope': 'median',
        'area': 'sum',
        'std_amp': 'mean',
        'systolic_diastolic_ratio': 'median',
        'beat_entropy': 'mean'
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
        
        # Final NaN check - fallback to default
        if pd.isna(agg_features[col]):
            agg_features[col] = default_beat_features.get(col, 0)
    
    # Add prefix to beat features
    return {f'beat_{k}': v for k, v in agg_features.items()}

def extract_hrv_features(peaks, fs=100):
    """Extract Heart Rate Variability features"""
    if len(peaks) < 4:
        return {
            'sdnn': 0,
            'rmssd': 0,
            'pnn50': 0,
            'heart_rate': 0,
            'hrv_triangular_index': 0
        }
    
    # Calculate NN intervals in milliseconds
    nn_intervals = np.diff(peaks) / fs * 1000
    sdnn = np.std(nn_intervals)
    
    # Calculate RMSSD and pNN50
    diff_nn = np.diff(nn_intervals)
    rmssd = np.sqrt(np.mean(diff_nn**2))
    pnn50 = (np.sum(np.abs(diff_nn) > 50) / len(diff_nn)) * 100 if len(diff_nn) > 0 else 0
    
    # Heart rate (beats per minute)
    heart_rate = 60 * fs / np.mean(np.diff(peaks)) if len(peaks) > 1 else 0
    
    # HRV triangular index
    hist, bin_edges = np.histogram(nn_intervals, bins=20, density=True)
    hrv_triangular_index = np.max(hist) / len(nn_intervals)
    
    return {
        'sdnn': sdnn,
        'rmssd': rmssd,
        'pnn50': pnn50,
        'heart_rate': heart_rate,
        'hrv_triangular_index': hrv_triangular_index
    }

def extract_gradient_features(signal):
    """Extract signal gradient features"""
    gradient = np.gradient(signal)
    second_deriv = np.gradient(gradient)
    
    # Zero crossings in second derivative
    zero_crossings = np.where(np.diff(np.sign(second_deriv)))[0]
    
    # Slope features
    pos_slope = gradient[gradient > 0]
    neg_slope = gradient[gradient < 0]
    
    return {
        'max_gradient': np.max(gradient),
        'min_gradient': np.min(gradient),
        'mean_gradient': np.mean(gradient),
        'zero_crossings': len(zero_crossings),
        'pos_slope_area': np.trapz(pos_slope) if len(pos_slope) > 0 else 0,
        'neg_slope_area': np.trapz(neg_slope) if len(neg_slope) > 0 else 0,
        'slope_ratio': np.mean(pos_slope) / abs(np.mean(neg_slope)) if len(neg_slope) > 0 else 1
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
            'spectral_entropy': 0,
            'harmonic_ratio_1': 0,
            'harmonic_ratio_2': 0
        }
    else:
        hr_psd = psd[hr_mask]
        total_power = np.sum(hr_psd) + 1e-12
        norm_psd = hr_psd / total_power
        
        dominant_idx = np.argmax(hr_psd)
        dominant_freq = xf[hr_mask][dominant_idx]
        dominant_amp = np.abs(yf[hr_mask][dominant_idx]) / n * 2
        
        # Harmonic ratios
        harmonic_ratios = []
        for i in range(1, 3):
            harmonic_freq = dominant_freq * (i+1)
            harmonic_idx = np.argmin(np.abs(xf - harmonic_freq))
            harmonic_ratio = psd[harmonic_idx] / hr_psd[dominant_idx] if hr_psd[dominant_idx] > 0 else 0
            harmonic_ratios.append(harmonic_ratio)
        
        hr_features = {
            'dominant_freq': dominant_freq,
            'dominant_amp': dominant_amp,
            'spectral_entropy': -np.sum(norm_psd * np.log2(norm_psd + 1e-12)),
            'harmonic_ratio_1': harmonic_ratios[0],
            'harmonic_ratio_2': harmonic_ratios[1]
        }
    
    # LF/HF power ratio (0.04-0.15 Hz and 0.15-0.4 Hz)
    lf_mask = (xf >= 0.04) & (xf < 0.15)
    hf_mask = (xf >= 0.15) & (xf < 0.4)
    vlf_mask = (xf >= 0.003) & (xf < 0.04)
    
    lf_power = np.sum(psd[lf_mask]) if np.any(lf_mask) else 1e-12
    hf_power = np.sum(psd[hf_mask]) if np.any(hf_mask) else 1e-12
    vlf_power = np.sum(psd[vlf_mask]) if np.any(vlf_mask) else 1e-12
    total_power = lf_power + hf_power + vlf_power + 1e-12
    
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
        'hf_power': hf_power,
        'lf_power_nu': lf_power / (lf_power + hf_power) * 100,
        'hf_power_nu': hf_power / (lf_power + hf_power) * 100,
        'vlf_power': vlf_power
    }

def extract_wavelet_features(signal, fs=100):
    """Extract wavelet-based features using bandpass filters"""
    features = {}
    
    # Define frequency bands
    bands = {
        'vlf': (0.003, 0.04),
        'lf': (0.04, 0.15),
        'hf': (0.15, 0.4),
        'cardiac': (0.8, 4.0)
    }
    
    for band_name, (low, high) in bands.items():
        try:
            # Apply bandpass filter
            nyq = 0.5 * fs
            low_norm = low / nyq
            high_norm = high / nyq
            b, a = butter(3, [low_norm, high_norm], btype='band')
            filtered = filtfilt(b, a, signal)
            
            # Extract features
            features[f'{band_name}_energy'] = np.sum(filtered**2)
            features[f'{band_name}_mean'] = np.mean(filtered)
            features[f'{band_name}_std'] = np.std(filtered)
            
            # Zero-crossing rate
            zero_crossings = np.where(np.diff(np.sign(filtered)))[0]
            features[f'{band_name}_zcr'] = len(zero_crossings) / len(signal)
            
        except:
            features[f'{band_name}_energy'] = 0
            features[f'{band_name}_mean'] = 0
            features[f'{band_name}_std'] = 0
            features[f'{band_name}_zcr'] = 0
    
    return features

def extract_nonlinear_features(signal):
    """Extract nonlinear dynamical features"""
    # Sample Entropy approximation
    def sample_entropy(data, m=2, r=0.2):
        n = len(data)
        std = np.std(data)
        if std == 0 or n <= m:
            return 0
            
        r_val = r * std
        patterns = np.array([data[i:i+m] for i in range(n-m+1)])
        
        # Count matches
        matches = 0
        for i in range(len(patterns)-1):
            diff = np.abs(patterns[i] - patterns[i+1:])
            matches += np.sum(np.all(diff <= r_val, axis=1))
        
        return -np.log((matches + 1e-6) / (n - m))
    
    # Detrended Fluctuation Analysis (DFA)
    def dfa(data):
        n = len(data)
        if n < 10:
            return 0
            
        # Integrated series
        y = np.cumsum(data - np.mean(data))
        
        # Calculate RMS for different box sizes
        scales = np.logspace(np.log10(4), np.log10(n//4), 10).astype(int)
        scales = np.unique(scales)
        
        f = []
        for scale in scales:
            # Divide into boxes
            boxes = len(data) // scale
            rms = 0
            for i in range(boxes):
                idx = i * scale
                box = y[idx:idx+scale]
                x = np.arange(len(box))
                coeffs = np.polyfit(x, box, 1)
                fit = np.polyval(coeffs, x)
                rms += np.sum((box - fit)**2)
            
            f.append(np.sqrt(rms / (boxes * scale)))
        
        # Fit to log-log scale
        coeffs = np.polyfit(np.log(scales), np.log(f), 1)
        return coeffs[0]
    
    return {
        'sample_entropy': sample_entropy(signal),
        'dfa_alpha': dfa(signal),
        'signal_entropy': entropy(np.histogram(signal, bins=20)[0]),
        'lyapunov_exponent': 0  # Placeholder for complex calculation
    }

def extract_features(signal, fs=100):
    """Master feature extraction function with enhanced capabilities"""
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
        'rms': np.sqrt(np.mean(preprocessed**2)),
        'perfusion_index': np.ptp(preprocessed) / np.mean(preprocessed) if np.mean(preprocessed) > 0 else 0,
        'autocorr_peak': np.correlate(preprocessed, preprocessed, mode='full')[len(preprocessed)] / (np.max(preprocessed) + 1e-12)
    })
    
    # Pulse waveform features
    pulse_features = extract_pulse_features(preprocessed, fs, peaks)
    features.update(pulse_features)
    
    # Beat features
    beat_features = extract_beat_features(preprocessed, fs, peaks)
    features.update(beat_features)
    
    # HRV features
    hrv_features = extract_hrv_features(peaks, fs)
    features.update(hrv_features)
    
    # Gradient features
    grad_features = extract_gradient_features(preprocessed)
    features.update(grad_features)
    
    # Frequency features
    freq_features = extract_freq_features(preprocessed, fs)
    features.update(freq_features)
    
    # Wavelet features
    wavelet_features = extract_wavelet_features(preprocessed, fs)
    features.update(wavelet_features)
    
    # Nonlinear features
    nonlinear_features = extract_nonlinear_features(preprocessed)
    features.update(nonlinear_features)
    
    # Post-processing: Handle NaNs and infs in specific features
    critical_features = [
        'beat_systolic_diastolic_ratio',
        'beat_entropy',
        'pulse_asymmetry'
    ]
    
    for feature in critical_features:
        if feature in features and not np.isfinite(features[feature]):
            features[feature] = 0
    
    # General NaN handling for all features
    for key in list(features.keys()):
        if not np.isfinite(features[key]):
            features[key] = 0
    
    return features