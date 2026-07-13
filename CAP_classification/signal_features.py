import numpy as np
from scipy.stats import entropy
from scipy.ndimage import uniform_filter1d
from CAP_classification.extract_eeg_bands import extract_eeg_bands
import numba

@numba.njit
def _numba_hist_density_precise(data, bins_num):
    n = len(data)
    min_val = data[0]
    max_val = data[0]
    for i in range(1, n):
        v = data[i]
        if v < min_val: min_val = v
        if v > max_val: max_val = v
    if min_val == max_val:
        max_val += 1e-12
        
    bin_edges = np.linspace(min_val, max_val, bins_num + 1)
    counts = np.zeros(bins_num, dtype=np.int32)
    
    for i in range(n):
        val = data[i]
        idx = np.searchsorted(bin_edges, val) - 1
        if idx >= bins_num:
            idx = bins_num - 1
        elif idx < 0:
            idx = 0
        counts[idx] += 1
        
    bin_width = bin_edges[1] - bin_edges[0]
    norm = n * bin_width
    density = np.zeros(bins_num, dtype=np.float64)
    for i in range(bins_num):
        density[i] = counts[i] / norm
    return density

@numba.njit
def get_shannon_entropy_numba(signal, fs_val, window_sec):
    window_len = int(window_sec * fs_val)
    step = window_len // 2
    n_steps = int((len(signal) - window_len) // step + 1)
    entropy_arr = np.zeros(n_steps, dtype=np.float64)
    for i in range(n_steps):
        seg = signal[i*step : i*step + window_len]
        hist = _numba_hist_density_precise(seg, 32) + 1e-12
        hist_sum = np.sum(hist)
        ent_val = 0.0
        for j in range(32):
            p = hist[j] / hist_sum
            if p > 0.0:
                ent_val -= p * np.log2(p)
        entropy_arr[i] = ent_val
    return entropy_arr

@numba.njit
def get_hjorth_activity_numba(signal, fs_val, window_sec):
    window_len = int(window_sec * fs_val)
    step = int(fs_val)
    n_steps = int((len(signal) - window_len) // step + 1)
    activity = np.zeros(n_steps, dtype=np.float64)
    for i in range(n_steps):
        activity[i] = np.var(signal[i*step : i*step + window_len])
    return activity

@numba.njit
def get_teo_numba(signal, fs_val):
    fs = int(fs_val)
    n = len(signal)
    if n < 3:
        return np.zeros(0, dtype=np.float64)
    teo_full = np.zeros(n - 2, dtype=np.float32)
    for i in range(n - 2):
        teo_full[i] = signal[i+1]**2 - signal[i]*signal[i+2]
    n_sec = int(len(teo_full) // fs)
    teo = np.zeros(n_sec, dtype=np.float64)
    for i in range(n_sec):
        teo[i] = np.mean(teo_full[i*fs : (i+1)*fs])
    return teo

@numba.njit
def _get_power_band_numba_core(signal, fs_val):
    fs = int(fs_val)
    n_sec = int(len(signal) // fs)
    power = np.zeros(n_sec, dtype=np.float64)
    for i in range(n_sec):
        val_sum = 0.0
        for j in range(fs):
            val_sum += signal[i*fs + j]**2
        power[i] = val_sum / fs
    return power

@numba.njit
def _get_eeg_var_vals_numba(signal, fs_val):
    fs = int(fs_val)
    n_sec = int(len(signal) // fs)
    var_vals = np.zeros(n_sec, dtype=np.float64)
    for i in range(n_sec):
        var_vals[i] = np.var(signal[i*fs : (i+1)*fs])
    return var_vals

# Public wrappers matching original API names:
def get_hjorth_activity(signal, fs, window_sec):
    return get_hjorth_activity_numba(signal, fs, window_sec)

def get_shannon_entropy(signal, fs, window_sec):
    return get_shannon_entropy_numba(signal, fs, window_sec)

def get_teo(signal, fs):
    return get_teo_numba(signal, fs)

def get_power_band_feature(signal, fs, smoothing):
    power = _get_power_band_numba_core(signal, fs)
    if smoothing:
        power = uniform_filter1d(power, size=5)
    return power

def get_eeg_var_diff(signal, fs):
    var_vals = _get_eeg_var_vals_numba(signal, fs)
    if len(var_vals) == 0:
        return np.zeros(0, dtype=float)
    return np.diff(var_vals, prepend=var_vals[0])

def _strfind(arr: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """
    MATLAB strfind-Äquivalent für numerische 1D-Arrays.
    Gibt alle Startindizes zurück, an denen pattern in arr vorkommt.
    """
    n, m = len(arr), len(pattern)
    if m > n:
        return np.array([], dtype=int)
    hits = []
    for i in range(n - m + 1):
        if np.array_equal(arr[i:i + m], pattern):
            hits.append(i)
    return np.array(hits, dtype=int)

def get_distance(signal, fs, stages, duration, time):
    """
    1:1 Übersetzung von SignalFeatures.getDistance aus MATLAB.
    Berechnet für jede Sekunde den Abstand (in Sekunden) bis zum nächsten Wake/REM-Abschnitt.
    """
    # 1. Alle Eingaben als flache 1D-Arrays (MATLAB: iscolumn → transpose)
    signal   = np.asarray(signal).flatten()
    stages   = np.asarray(stages,   dtype=float).flatten()
    duration = np.asarray(duration, dtype=float).flatten()
    time     = np.asarray(time,     dtype=float).flatten()

    # 2. Filtere alle Epochen mit stages > 5 (Artefakte/Unbekannt)
    mask = stages <= 5
    stages_new = stages[mask].copy()
    duration   = duration[mask].copy()
    time       = time[mask].copy()

    # 3. Gap-Filling: Füge synthetische 30s-Epochen bei Lücken >30s ein
    time_diff = np.diff(time)
    ind = np.where(time_diff > 30)[0]
    offset = 0
    for i in range(len(ind)):
        idx = ind[i] + offset
        if idx == 0:
            stages_new = np.concatenate([[stages_new[0]], stages_new])
            duration   = np.concatenate([[30], duration])
            time       = np.concatenate([[0], time])
        elif idx == len(time) - 1:
            stages_new = np.concatenate([stages_new, [stages_new[-1]]])
            duration   = np.concatenate([duration, [30]])
            time       = np.concatenate([time, [time[-1] + 30]])
        else:
            stages_new = np.concatenate([stages_new[:idx], [stages_new[idx - 1]], stages_new[idx:]])
            duration   = np.concatenate([duration[:idx], [30], duration[idx:]])
            time       = np.concatenate([time[:idx], [time[idx] + 30], time[idx:]])
        offset += 1

    # 4. Wake/REM-Indikator (0=Wake, 5=REM)
    wake_rem = ((stages_new == 0) | (stages_new == 5)).astype(float)

    # 5. Smoothing: 4x strfind-Pass wie in MATLAB
    # a) [0,1,0] → 0
    for start in _strfind(wake_rem, np.array([0.0, 1.0, 0.0])):
        wake_rem[start:start+3] = 0.0
    # b) [0,1,1,0] → 0
    for start in _strfind(wake_rem, np.array([0.0, 1.0, 1.0, 0.0])):
        wake_rem[start:start+4] = 0.0
    # c) [1,0,1] → 1
    for start in _strfind(wake_rem, np.array([1.0, 0.0, 1.0])):
        wake_rem[start:start+3] = 1.0
    # d) [1,0,0,1] → 1
    for start in _strfind(wake_rem, np.array([1.0, 0.0, 0.0, 1.0])):
        wake_rem[start:start+4] = 1.0

    # 6. Upsampling: Jede Epoche auf 30 Sekunden ausdehnen
    wake_rem_long = np.repeat(wake_rem, 30)

    # 7. Länge an Signal (in Sekunden) anpassen
    signal_len = int(len(signal) / fs)
    if signal_len > len(wake_rem_long):
        extra = signal_len - len(wake_rem_long)
        wake_rem_long = np.concatenate([wake_rem_long, np.ones(extra)])
    elif signal_len < len(wake_rem_long):
        wake_rem_long = wake_rem_long[:signal_len]

    # 8. Distance-Berechnung (O(n)-Effizienz)
    n = len(wake_rem_long)
    distance = np.zeros(n, dtype=float)
    next_one = np.full(n, -1, dtype=int)
    if wake_rem_long[-1] == 1:
        next_one[-1] = n - 1
    for i in range(n - 2, -1, -1):
        if wake_rem_long[i] == 1:
            next_one[i] = i
        else:
            next_one[i] = next_one[i + 1]
    for i in range(n):
        if wake_rem_long[i] == 1:
            distance[i] = 0
        else:
            j = next_one[i]
            if j == -1:
                distance[i] = float(n)
            else:
                distance[i] = float(j - i + 1)
    return distance

def get_features_paper(signal, fs, event, duration, eventtime):
    # Entspricht SignalFeatures.getFeaturesPaper
    _, delta = extract_eeg_bands(signal, fs, 'delta')
    _, theta = extract_eeg_bands(signal, fs, 'theta')
    _, alpha = extract_eeg_bands(signal, fs, 'alpha')
    _, sigma = extract_eeg_bands(signal, fs, 'sigma')
    _, beta  = extract_eeg_bands(signal, fs, 'beta')

    f1 = get_hjorth_activity(delta, fs, 3)
    f2 = get_shannon_entropy(signal, fs, 2)
    f3 = get_teo(delta, fs)
    f4 = get_teo(theta, fs)
    f5 = get_teo(alpha, fs)
    f6 = get_teo(sigma, fs)
    f7 = get_teo(beta, fs)
    f8 = get_power_band_feature(delta, fs, False)
    f9 = get_power_band_feature(theta, fs, True)
    f10 = get_power_band_feature(alpha, fs, True)
    f11 = get_power_band_feature(sigma, fs, True)
    f12 = get_power_band_feature(beta, fs, True)
    f13 = get_eeg_var_diff(signal, fs)
    f14 = get_distance(signal, fs, event, duration, eventtime)
    all_feats = [np.atleast_1d(f) for f in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14]]
    min_len = min(len(f) for f in all_feats)
    features = np.column_stack([f[:min_len] for f in all_feats])
    return features