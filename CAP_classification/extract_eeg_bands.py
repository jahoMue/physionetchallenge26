import numpy as np
from scipy.signal import butter, filtfilt

def extract_eeg_bands(signal, fs, band_type):
    """
    Bandpass-Filter für EEG-Bänder (delta, theta, alpha, sigma, beta).
    Entspricht extractEEGBands.m in MATLAB.
    """
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'sigma': (12, 16),
        'beta': (16, 25)
    }
    fc1, fc2 = bands.get(band_type, (0.5, 4))
    nyq = 0.5 * fs
    b, a = butter(4, [fc1 / nyq, fc2 / nyq], btype='band')
    filtered = filtfilt(b, a, signal)
    return b, filtered