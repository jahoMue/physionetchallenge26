import numpy as np
from scipy.stats import entropy
from scipy.ndimage import uniform_filter1d
from CAP_classification.extract_eeg_bands import extract_eeg_bands

def get_hjorth_activity(signal, fs, window_sec):
    window_len = int(window_sec * fs)
    step = int(fs)  # 1s Schritt (66% Overlap bei 3s Fenster)
    n_steps = int((len(signal) - window_len) // step + 1)
    activity = np.array([np.var(signal[i*step:i*step+window_len]) for i in range(n_steps)])
    return activity

def get_shannon_entropy(signal, fs, window_sec):
    window_len = int(window_sec * fs)
    step = window_len // 2
    n_steps = int((len(signal) - window_len) // step + 1)
    entropy_arr = []
    for i in range(n_steps):
        seg = signal[i*step:i*step+window_len]
        hist, _ = np.histogram(seg, bins=32, density=True)
        hist = hist + 1e-12  # Vermeide log(0)
        entropy_arr.append(entropy(hist, base=2))
    return np.array(entropy_arr)

def get_teo(signal, fs):
    teo_full = signal[1:-1]**2 - signal[:-2]*signal[2:]
    n_sec = int(len(teo_full) // fs)
    fs = int(fs)
    teo = np.array([np.mean(teo_full[i*fs:(i+1)*fs]) for i in range(n_sec)])
    return teo

def get_power_band_feature(signal, fs, smoothing):
    n_sec = int(len(signal) // fs)
    fs = int(fs)
    power = np.array([np.sum(signal[i*fs:(i+1)*fs]**2) / fs for i in range(n_sec)])
    if smoothing:
        power = uniform_filter1d(power, size=5)
    return power

def get_eeg_var_diff(signal, fs):
    n_sec = int(len(signal) // fs)
    fs = int(fs)
    var_vals = np.array([np.var(signal[i*fs:(i+1)*fs]) for i in range(n_sec)])
    if len(var_vals) == 0:
        return np.zeros(0, dtype=float)  # Leeres 1D-Array zurückgeben
    var_diff = np.diff(var_vals, prepend=var_vals[0])
    return var_diff

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