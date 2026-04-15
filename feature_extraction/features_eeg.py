"""
03_feature_extraction/features_eeg.py
=======================================
EEG Feature-Extraktion pro Segment und Region:

Spektrale Features:
- Absolute und relative Bandpower (Delta, Theta, Alpha, Sigma, Beta)
- Spektrale Entropie
- Spektrale Kantfrequenz (SEF50, SEF95)
- Peak-Frequenz pro Band

Schlafspezifische Features:
- Slow-Wave Activity (SWA) – Delta-Power als Marker für Schlaftiefe
- Schlafspindel-Aktivität (Sigma-Band)
- Alpha/Theta Ratio (Wachheits-Marker)
- Delta/Beta Ratio (Schlaftiefe-Marker)

Zeitdomäne / Komplexität:
- Hjorth-Parameter (Activity, Mobility, Complexity)
- Permutation Entropy
- Sample Entropy
- Higuchi Fractal Dimension
- Zero-Crossing Rate
- Line Length

Asymmetrie-Features (falls beide Hemisphären verfügbar):
- Interhemisphärische Kohärenz
- Bandpower-Asymmetrie

Alle Features werden nur für Segmente mit ausreichender
Signalqualität (SQI >= Schwellenwert) berechnet.
"""
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
from scipy.ndimage import gaussian_filter1d
from scipy.fft import fft
from typing import Dict, List, Optional, Tuple

from config import (
    EEG_FREQUENCY_BANDS, EEG_SQI_THRESHOLD,
    SEGMENT_LENGTH_SEC, EEG_HOMOLOG_PAIRS,
    SIGNAL_DTYPE,
)

from feature_extraction.features_eeg_spindle_so import (
    extract_spindle_so_coupling_features,
)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==============================================================================
# HAUPT-FUNKTION: FEATURES PRO SEGMENT UND REGION
# ==============================================================================

def extract_eeg_features_segment(
    segment_data: np.ndarray,
    fs: float,
    region: str = "unknown",
    sqi: float = 0.0,
    segment_idx: int = 0,
    strategy: str = "unknown",
    logger=None
) -> Dict:
    """
    Extrahiert alle EEG-Features für ein einzelnes Segment einer Region.
    
    Parameters
    ----------
    segment_data : np.ndarray
        EEG-Signaldaten des Segments.
    fs : float
        Sampling-Rate in Hz.
    region : str
        Hirnregion ("frontal", "central", "occipital").
    sqi : float
        Signal Quality Index des Segments.
    segment_idx : int
        Index des Segments.
    strategy : str
        Kanalstrategie ("averaged", "single_left", "single_right", "none").
    logger : loguru.Logger, optional
    
    Returns
    -------
    Dict
        Dictionary mit allen EEG-Features (mit Region-Prefix).
    """
    prefix = f"eeg_{region}"
    features = _get_empty_eeg_features(segment_idx, prefix)
    features[f"{prefix}_sqi"] = sqi
    features[f"{prefix}_quality_ok"] = sqi >= EEG_SQI_THRESHOLD
    features[f"{prefix}_strategy"] = strategy
    
    # --- Prüfe ob genügend Daten vorhanden ---
    if segment_data is None or len(segment_data) == 0:
        if logger:
            logger.debug(f"Segment {segment_idx} [{region}]: Keine EEG-Daten.")
        return features
    
    if sqi < EEG_SQI_THRESHOLD:
        if logger:
            logger.debug(f"Segment {segment_idx} [{region}]: SQI zu niedrig "
                         f"({sqi:.3f}), überspringe Feature-Berechnung.")
        # Trotzdem Basis-Statistiken
        features.update(_compute_basic_eeg_stats(segment_data, fs, prefix))
        return features
    
    # --- Feature-Berechnung ---
    features.update(_compute_basic_eeg_stats(segment_data, fs, prefix))
    features.update(_compute_spectral_features(segment_data, fs, prefix))
    features.update(_compute_sleep_specific_features(segment_data, fs, prefix))
    features.update(_compute_hjorth_parameters(segment_data, fs, prefix))
    features.update(_compute_entropy_features(segment_data, prefix))
    features.update(_compute_complexity_features(segment_data, fs, prefix))
    features.update(_compute_temporal_features(segment_data, fs, prefix))

    features.update(
        extract_spindle_so_coupling_features(
            segment_data=segment_data,
            fs=fs,
            region=region,
            sqi=sqi,
            segment_idx=segment_idx,
            logger=logger,
        )
    )
    
    if logger:
        logger.debug(f"Segment {segment_idx} [{region}]: "
                     f"{len(features)} EEG-Features extrahiert.")
    
    return features


# ==============================================================================
# BASIS-STATISTIKEN
# ==============================================================================

def _compute_basic_eeg_stats(
    signal: np.ndarray,
    fs: float,
    prefix: str
) -> Dict:
    """Basis-Signalstatistiken des EEG."""
    return {
        f"{prefix}_signal_mean": float(np.mean(signal)),
        f"{prefix}_signal_std": float(np.std(signal)),
        f"{prefix}_signal_range": float(np.ptp(signal)),
        f"{prefix}_signal_skewness": float(skew(signal, nan_policy='omit'))
            if len(signal) > 2 else np.nan,
        f"{prefix}_signal_kurtosis": float(kurtosis(signal, nan_policy='omit'))
            if len(signal) > 3 else np.nan,
        f"{prefix}_signal_rms": float(np.sqrt(np.mean(signal ** 2))),
        f"{prefix}_signal_abs_mean": float(np.mean(np.abs(signal))),
        f"{prefix}_duration_sec": float(len(signal) / fs),
    }


# ==============================================================================
# SPEKTRALE FEATURES
# ==============================================================================

def _compute_spectral_features(
    signal: np.ndarray,
    fs: float,
    prefix: str
) -> Dict:
    """
    Spektrale Features: Bandpower, spektrale Entropie, Kantfrequenzen.
    """
    features = {}
    
    # --- PSD berechnen (Welch-Methode) ---
    try:
        nperseg = min(len(signal), int(fs * 4))  # 4s Fenster
        noverlap = nperseg // 2
        
        freqs, psd = scipy_signal.welch(
            signal, fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            window='hann'
        )
    except Exception:
        return _get_empty_spectral_features(prefix)
    
    if len(freqs) == 0 or len(psd) == 0:
        return _get_empty_spectral_features(prefix)
    
    # --- Gesamtleistung (0.5-35 Hz) ---
    total_mask = (freqs >= 0.5) & (freqs <= 35.0)
    total_power = np.trapz(psd[total_mask], freqs[total_mask]) if total_mask.any() else 0
    features[f"{prefix}_total_power"] = float(total_power)
    
    # --- Bandpower (absolut und relativ) ---
    for band_name, (fmin, fmax) in EEG_FREQUENCY_BANDS.items():
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        
        if band_mask.any():
            abs_power = float(np.trapz(psd[band_mask], freqs[band_mask]))
            rel_power = abs_power / total_power if total_power > 0 else 0.0
            
            # Peak-Frequenz im Band
            band_freqs = freqs[band_mask]
            band_psd = psd[band_mask]
            peak_freq = float(band_freqs[np.argmax(band_psd)])
            
            # Mittlere Frequenz im Band (Schwerpunkt)
            mean_freq = (
                float(np.sum(band_freqs * band_psd) / np.sum(band_psd))
                if np.sum(band_psd) > 0 else np.nan
            )
        else:
            abs_power = 0.0
            rel_power = 0.0
            peak_freq = np.nan
            mean_freq = np.nan
        
        features[f"{prefix}_{band_name}_power_abs"] = abs_power
        features[f"{prefix}_{band_name}_power_rel"] = rel_power
        features[f"{prefix}_{band_name}_peak_freq"] = peak_freq
        features[f"{prefix}_{band_name}_mean_freq"] = mean_freq
    
    # --- Log-transformierte Bandpower ---
    for band_name, (fmin, fmax) in EEG_FREQUENCY_BANDS.items():
        abs_power = features.get(f"{prefix}_{band_name}_power_abs", 0)
        features[f"{prefix}_{band_name}_power_log"] = (
            float(np.log10(abs_power + 1e-10))
        )
    
    # --- Spektrale Entropie ---
    features[f"{prefix}_spectral_entropy"] = _spectral_entropy(psd[total_mask])
    
    # --- Spektrale Kantfrequenzen (SEF) ---
    sef50, sef95 = _spectral_edge_frequency(freqs[total_mask], psd[total_mask])
    features[f"{prefix}_sef50"] = sef50
    features[f"{prefix}_sef95"] = sef95
    
    # --- Mittlere Frequenz (gesamtes Spektrum) ---
    if total_mask.any() and np.sum(psd[total_mask]) > 0:
        features[f"{prefix}_mean_frequency"] = float(
            np.sum(freqs[total_mask] * psd[total_mask]) / np.sum(psd[total_mask])
        )
    else:
        features[f"{prefix}_mean_frequency"] = np.nan
    
    # --- Median-Frequenz ---
    features[f"{prefix}_median_frequency"] = sef50  # SEF50 = Median-Frequenz
    
    return features


def _spectral_entropy(psd: np.ndarray) -> float:
    """
    Berechnet die spektrale Entropie (normalisiert).
    Hohe Entropie = gleichmäßige Verteilung (z.B. Rauschen).
    Niedrige Entropie = dominante Frequenz.
    """
    if len(psd) == 0 or np.sum(psd) == 0:
        return np.nan
    
    # Normalisiere PSD zu Wahrscheinlichkeitsverteilung
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]  # Entferne Nullen
    
    # Shannon-Entropie
    entropy = -np.sum(psd_norm * np.log2(psd_norm))
    
    # Normalisiere auf [0, 1]
    max_entropy = np.log2(len(psd))
    if max_entropy > 0:
        return float(entropy / max_entropy)
    return np.nan


def _spectral_edge_frequency(
    freqs: np.ndarray,
    psd: np.ndarray
) -> Tuple[float, float]:
    """
    Berechnet die spektralen Kantfrequenzen.
    SEF50: Frequenz unter der 50% der Gesamtleistung liegt.
    SEF95: Frequenz unter der 95% der Gesamtleistung liegt.
    """
    if len(freqs) == 0 or len(psd) == 0:
        return np.nan, np.nan
    
    total_power = np.trapz(psd, freqs)
    if total_power == 0:
        return np.nan, np.nan
    
    cumulative_power = np.cumsum(psd * np.diff(np.concatenate([[freqs[0]], freqs])))
    cumulative_power /= cumulative_power[-1]  # Normalisiere
    
    # SEF50
    idx_50 = np.searchsorted(cumulative_power, 0.50)
    sef50 = float(freqs[min(idx_50, len(freqs) - 1)])
    
    # SEF95
    idx_95 = np.searchsorted(cumulative_power, 0.95)
    sef95 = float(freqs[min(idx_95, len(freqs) - 1)])
    
    return sef50, sef95


def _get_empty_spectral_features(prefix: str) -> Dict:
    """Leere spektrale Features."""
    features = {
        f"{prefix}_total_power": np.nan,
        f"{prefix}_spectral_entropy": np.nan,
        f"{prefix}_sef50": np.nan,
        f"{prefix}_sef95": np.nan,
        f"{prefix}_mean_frequency": np.nan,
        f"{prefix}_median_frequency": np.nan,
    }
    for band_name in EEG_FREQUENCY_BANDS:
        features[f"{prefix}_{band_name}_power_abs"] = np.nan
        features[f"{prefix}_{band_name}_power_rel"] = np.nan
        features[f"{prefix}_{band_name}_power_log"] = np.nan
        features[f"{prefix}_{band_name}_peak_freq"] = np.nan
        features[f"{prefix}_{band_name}_mean_freq"] = np.nan
    return features


# ==============================================================================
# SCHLAFSPEZIFISCHE FEATURES
# ==============================================================================

def _compute_sleep_specific_features(
    signal: np.ndarray,
    fs: float,
    prefix: str
) -> Dict:
    """
    Schlafmedizinisch relevante Features.
    
    Besonders relevant für Cognitive Impairment:
    - SWA (Slow-Wave Activity): Reduziert bei kognitiver Beeinträchtigung
    - Schlafspindeln: Reduziert bei Alzheimer/MCI
    - Alpha/Theta Ratio: Verändert bei kognitiver Beeinträchtigung
    - EEG Slowing: Delta/Alpha und Theta/Alpha Ratios
    """
    features = {}
    
    try:
        nperseg = min(len(signal), int(fs * 4))
        freqs, psd = scipy_signal.welch(
            signal, fs=fs, nperseg=nperseg,
            noverlap=nperseg // 2, window='hann'
        )
    except Exception:
        return _get_empty_sleep_features(prefix)
    
    # --- Bandpower für Ratios ---
    band_powers = {}
    for band_name, (fmin, fmax) in EEG_FREQUENCY_BANDS.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band_name] = (
            float(np.trapz(psd[mask], freqs[mask])) if mask.any() else 0
        )
    
    # --- Slow-Wave Activity (SWA) ---
    # SWA = Delta-Power (0.5-4 Hz), besonders 0.5-2 Hz
    swa_mask = (freqs >= 0.5) & (freqs <= 2.0)
    features[f"{prefix}_swa_power"] = (
        float(np.trapz(psd[swa_mask], freqs[swa_mask])) if swa_mask.any() else np.nan
    )
    features[f"{prefix}_swa_power_log"] = (
        float(np.log10(features[f"{prefix}_swa_power"] + 1e-10))
    )
    
    # --- Schlafspindel-Aktivität ---
    # Sigma-Band (12-16 Hz), besonders 12-14 Hz (langsame Spindeln)
    # und 14-16 Hz (schnelle Spindeln)
    slow_spindle_mask = (freqs >= 12.0) & (freqs <= 14.0)
    fast_spindle_mask = (freqs >= 14.0) & (freqs <= 16.0)
    
    features[f"{prefix}_slow_spindle_power"] = (
        float(np.trapz(psd[slow_spindle_mask], freqs[slow_spindle_mask]))
        if slow_spindle_mask.any() else np.nan
    )
    features[f"{prefix}_fast_spindle_power"] = (
        float(np.trapz(psd[fast_spindle_mask], freqs[fast_spindle_mask]))
        if fast_spindle_mask.any() else np.nan
    )
    features[f"{prefix}_spindle_power_total"] = (
        float(band_powers.get("sigma", 0))
    )
    
    # Spindel-Prominenz: Sigma relativ zu benachbarten Bändern
    theta_power = band_powers.get("theta", 0)
    sigma_power = band_powers.get("sigma", 0)
    beta_power = band_powers.get("beta", 0)
    neighbor_mean = (theta_power + beta_power) / 2 if (theta_power + beta_power) > 0 else 1
    features[f"{prefix}_spindle_prominence"] = (
        sigma_power / neighbor_mean if neighbor_mean > 0 else np.nan
    )
    
    # --- Band-Ratios (wichtig für Cognitive Impairment) ---
    delta_power = band_powers.get("delta", 0)
    alpha_power = band_powers.get("alpha", 0)
    
    # Alpha/Theta Ratio (Wachheit/Aufmerksamkeit)
    features[f"{prefix}_alpha_theta_ratio"] = (
        alpha_power / theta_power if theta_power > 0 else np.nan
    )
    
    # Theta/Alpha Ratio (EEG Slowing – erhöht bei Demenz)
    features[f"{prefix}_theta_alpha_ratio"] = (
        theta_power / alpha_power if alpha_power > 0 else np.nan
    )
    
    # Delta/Alpha Ratio (Schlaftiefe / EEG Slowing)
    features[f"{prefix}_delta_alpha_ratio"] = (
        delta_power / alpha_power if alpha_power > 0 else np.nan
    )
    
    # Delta/Beta Ratio (Schlaftiefe)
    features[f"{prefix}_delta_beta_ratio"] = (
        delta_power / beta_power if beta_power > 0 else np.nan
    )
    
    # (Delta+Theta) / (Alpha+Beta) – Global Slowing Index
    slow_power = delta_power + theta_power
    fast_power = alpha_power + beta_power
    features[f"{prefix}_slowing_ratio"] = (
        slow_power / fast_power if fast_power > 0 else np.nan
    )
    
    # --- DAR (Delta-Alpha Ratio) – Demenz-Marker ---
    total_power = sum(band_powers.values())
    features[f"{prefix}_dar"] = (
        delta_power / (alpha_power + 1e-10) if alpha_power > 0 else np.nan
    )
    
    # --- DTABR (Delta+Theta / Alpha+Beta Ratio) ---
    features[f"{prefix}_dtabr"] = features[f"{prefix}_slowing_ratio"]
    
    return features


def _get_empty_sleep_features(prefix: str) -> Dict:
    """Leere schlafspezifische Features."""
    return {
        f"{prefix}_swa_power": np.nan,
        f"{prefix}_swa_power_log": np.nan,
        f"{prefix}_slow_spindle_power": np.nan,
        f"{prefix}_fast_spindle_power": np.nan,
        f"{prefix}_spindle_power_total": np.nan,
        f"{prefix}_spindle_prominence": np.nan,
        f"{prefix}_alpha_theta_ratio": np.nan,
        f"{prefix}_theta_alpha_ratio": np.nan,
        f"{prefix}_delta_alpha_ratio": np.nan,
        f"{prefix}_delta_beta_ratio": np.nan,
        f"{prefix}_slowing_ratio": np.nan,
        f"{prefix}_dar": np.nan,
        f"{prefix}_dtabr": np.nan,
    }


# ==============================================================================
# HJORTH-PARAMETER
# ==============================================================================

def _compute_hjorth_parameters(
    signal: np.ndarray,
    fs: float,
    prefix: str
) -> Dict:
    """
    Hjorth-Parameter:
    - Activity: Varianz des Signals (Leistung)
    - Mobility: Mittlere Frequenz (Std der 1. Ableitung / Std des Signals)
    - Complexity: Bandbreite (Mobility der 1. Ableitung / Mobility des Signals)
    """
    features = {
        f"{prefix}_hjorth_activity": np.nan,
        f"{prefix}_hjorth_mobility": np.nan,
        f"{prefix}_hjorth_complexity": np.nan,
    }
    
    if len(signal) < 3:
        return features
    
    try:
        # Activity = Varianz
        activity = float(np.var(signal))
        
        # Erste Ableitung
        d1 = np.diff(signal)
        d1_var = np.var(d1)
        
        # Zweite Ableitung
        d2 = np.diff(d1)
        d2_var = np.var(d2)
        
        # Mobility
        mobility = float(np.sqrt(d1_var / activity)) if activity > 0 else np.nan
        
        # Complexity
        mobility_d1 = float(np.sqrt(d2_var / d1_var)) if d1_var > 0 else np.nan
        complexity = (
            mobility_d1 / mobility
            if mobility is not None and mobility > 0 and mobility_d1 is not None
            else np.nan
        )
        
        features[f"{prefix}_hjorth_activity"] = activity
        features[f"{prefix}_hjorth_mobility"] = mobility
        features[f"{prefix}_hjorth_complexity"] = complexity
    
    except Exception:
        pass
    
    return features


# ==============================================================================
# ENTROPIE-FEATURES
# ==============================================================================

def _compute_entropy_features(
    signal: np.ndarray,
    prefix: str
) -> Dict:
    """
    Entropie-basierte Features:
    - Sample Entropy
    - Permutation Entropy
    - Approximate Entropy
    
    Niedrigere Entropie bei Cognitive Impairment (weniger Komplexität).
    """
    features = {
        f"{prefix}_sample_entropy": np.nan,
        f"{prefix}_permutation_entropy": np.nan,
        f"{prefix}_approximate_entropy": np.nan,
    }
    
    if len(signal) < 100:
        return features
    
    # Downsampling für Effizienz bei langen Signalen
    max_samples = 5000
    if len(signal) > max_samples:
        step = len(signal) // max_samples
        signal_ds = signal[::step].astype(SIGNAL_DTYPE)  # CHANGED: ensure float32 copy
    else:
        signal_ds = signal  # Already float32 from preprocessing
    
    # --- Sample Entropy ---
    try:
        import neurokit2 as nk
        sampen = nk.entropy_sample(signal_ds, dimension=2, tolerance="sd")
        if isinstance(sampen, (list, tuple, np.ndarray)):
            sampen = sampen[0] if len(sampen) > 0 else np.nan
        features[f"{prefix}_sample_entropy"] = (
            float(sampen) if not np.isnan(sampen) and not np.isinf(sampen)
            else np.nan
        )
    except Exception:
        features[f"{prefix}_sample_entropy"] = _manual_sample_entropy(signal_ds)
    
    # --- Permutation Entropy ---
    try:
        features[f"{prefix}_permutation_entropy"] = _permutation_entropy(
            signal_ds, order=3, delay=1, normalize=True
        )
    except Exception:
        pass
    
    # --- Approximate Entropy ---
    try:
        import neurokit2 as nk
        apen = nk.entropy_approximate(signal_ds, dimension=2, tolerance="sd")
        if isinstance(apen, (list, tuple, np.ndarray)):
            apen = apen[0] if len(apen) > 0 else np.nan
        features[f"{prefix}_approximate_entropy"] = (
            float(apen) if not np.isnan(apen) and not np.isinf(apen)
            else np.nan
        )
    except Exception:
        pass
    
    return features


def _permutation_entropy(
    signal: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True
) -> float:
    """
    Berechnet die Permutation Entropy.
    
    Misst die Komplexität einer Zeitreihe basierend auf der
    Ordnung aufeinanderfolgender Werte.
    """
    n = len(signal)
    if n < (order - 1) * delay + 1:
        return np.nan
    
    # Erstelle Embedding-Matrix
    n_patterns = n - (order - 1) * delay
    patterns = np.zeros((n_patterns, order))
    
    for i in range(order):
        patterns[:, i] = signal[i * delay: i * delay + n_patterns]
    
    # Bestimme Permutationen (Ordnungsmuster)
    from collections import Counter
    permutations = []
    for i in range(n_patterns):
        perm = tuple(np.argsort(patterns[i]))
        permutations.append(perm)
    
    # Zähle Häufigkeiten
    counts = Counter(permutations)
    total = sum(counts.values())
    
    # Shannon-Entropie
    probs = np.array([c / total for c in counts.values()])
    entropy = -np.sum(probs * np.log2(probs))
    
    if normalize:
        import math
        max_entropy = np.log2(math.factorial(order))
        entropy = entropy / max_entropy if max_entropy > 0 else np.nan
    
    return float(entropy)


def _manual_sample_entropy(
    data: np.ndarray,
    m: int = 2,
    r_factor: float = 0.2
) -> float:
    """Manuelle Sample Entropy als Fallback."""
    N = len(data)
    r = r_factor * np.std(data)
    
    if N < m + 2 or r == 0:
        return np.nan
    
    # Vereinfachte Berechnung für Effizienz
    max_n = min(N, 1000)
    data_short = data[:max_n]
    N = len(data_short)
    
    def _count_templates(length):
        count = 0
        for i in range(N - length):
            template_i = data_short[i:i + length]
            for j in range(i + 1, N - length):
                template_j = data_short[j:j + length]
                if np.max(np.abs(template_i - template_j)) < r:
                    count += 1
        return count
    
    B = _count_templates(m)
    A = _count_templates(m + 1)
    
    if B == 0:
        return np.nan
    
    return float(-np.log(A / B)) if A > 0 else np.nan


# ==============================================================================
# KOMPLEXITÄTS-FEATURES
# ==============================================================================

def _compute_complexity_features(
    signal: np.ndarray,
    fs: float,
    prefix: str
) -> Dict:
    """
    Komplexitäts-Features:
    - Higuchi Fractal Dimension
    - Detrended Fluctuation Analysis (DFA)
    - Lempel-Ziv Complexity
    """
    features = {
        f"{prefix}_higuchi_fd": np.nan,
        f"{prefix}_dfa_alpha": np.nan,
        f"{prefix}_lzc": np.nan,
    }
    
    if len(signal) < 100:
        return features
    
    # Downsampling für Effizienz
    max_samples = 5000
    if len(signal) > max_samples:
        step = len(signal) // max_samples
        signal_ds = signal[::step]
    else:
        signal_ds = signal
    
    # --- Higuchi Fractal Dimension ---
    try:
        features[f"{prefix}_higuchi_fd"] = _higuchi_fd(signal_ds, kmax=10)
    except Exception:
        pass
    
    # --- DFA ---
    try:
        features[f"{prefix}_dfa_alpha"] = _dfa(signal_ds)
    except Exception:
        pass
    
    # --- Lempel-Ziv Complexity ---
    try:
        features[f"{prefix}_lzc"] = _lempel_ziv_complexity(signal_ds)
    except Exception:
        pass
    
    return features


def _higuchi_fd(signal: np.ndarray, kmax: int = 10) -> float:
    """
    Higuchi Fractal Dimension.
    Misst die Komplexität/Fraktalität einer Zeitreihe.
    Typische Werte: 1.0 (glatt) bis 2.0 (maximal komplex).
    """
    N = len(signal)
    if N < kmax * 4:
        return np.nan
    
    L = []
    x = np.arange(1, kmax + 1)
    
    for k in range(1, kmax + 1):
        Lk = []
        for m in range(1, k + 1):
            # Erstelle Teilsequenz
            indices = np.arange(m - 1, N, k)
            if len(indices) < 2:
                continue
            
            Lmk = np.sum(np.abs(np.diff(signal[indices])))
            norm = (N - 1) / (((len(indices) - 1) * k) * k)
            Lmk *= norm
            Lk.append(Lmk)
        
        if Lk:
            L.append(np.mean(Lk))
        else:
            L.append(np.nan)
    
    L = np.array(L)
    valid = ~np.isnan(L) & (L > 0)
    
    if valid.sum() < 3:
        return np.nan
    
    # Log-Log Regression
    log_k = np.log(x[valid])
    log_L = np.log(L[valid])
    
    coeffs = np.polyfit(log_k, log_L, 1)
    return float(-coeffs[0])  # Negative Steigung = Fraktale Dimension


def _dfa(signal: np.ndarray, min_box: int = 4, max_box: int = None) -> float:
    """
    Detrended Fluctuation Analysis (DFA).
    Misst langreichweitige Korrelationen in der Zeitreihe.
    
    Alpha < 0.5: Anti-korreliert
    Alpha = 0.5: Unkorrelliert (weißes Rauschen)
    Alpha = 1.0: 1/f Rauschen (Pink Noise)
    Alpha = 1.5: Brownsche Bewegung
    """
    N = len(signal)
    if max_box is None:
        max_box = N // 4
    
    if max_box < min_box + 3:
        return np.nan
    
    # Integriertes Signal
    y = np.cumsum(signal - np.mean(signal))
    
    # Box-Größen (logarithmisch verteilt)
    box_sizes = np.unique(np.logspace(
        np.log10(min_box), np.log10(max_box), num=20
    ).astype(int))
    box_sizes = box_sizes[box_sizes >= min_box]
    
    if len(box_sizes) < 4:
        return np.nan
    
    fluctuations = []
    valid_boxes = []
    
    for n in box_sizes:
        n_boxes = N // n
        if n_boxes < 1:
            continue
        
        rms_values = []
        for i in range(n_boxes):
            segment = y[i * n:(i + 1) * n]
            x_axis = np.arange(len(segment))
            coeffs = np.polyfit(x_axis, segment, 1)
            trend = np.polyval(coeffs, x_axis)
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_values.append(rms)
        
        if rms_values:
            mean_rms = np.mean(rms_values)
            if mean_rms > 0:
                fluctuations.append(mean_rms)
                valid_boxes.append(n)
    
    if len(valid_boxes) < 4:
        return np.nan
    
    # Log-Log Regression
    log_n = np.log(np.array(valid_boxes))
    log_f = np.log(np.array(fluctuations))
    
    coeffs = np.polyfit(log_n, log_f, 1)
    return float(coeffs[0])


def _lempel_ziv_complexity(signal: np.ndarray) -> float:
    """
    Lempel-Ziv Complexity (normalisiert).
    Misst die Komplexität einer binarisierten Zeitreihe.
    Höhere Werte = mehr Komplexität.
    """
    N = len(signal)
    if N < 10:
        return np.nan
    
    # Binarisiere Signal (über/unter Median)
    median_val = np.median(signal)
    binary = (signal >= median_val).astype(int)
    
    # Lempel-Ziv Algorithmus
    s = ''.join(map(str, binary))
    n = len(s)
    
    i = 0
    c = 1  # Komplexitätszähler
    l = 1
    
    while i + l <= n:
        # Suche ob das aktuelle Teilwort bereits vorkam
        substring = s[i:i + l]
        prefix = s[:i]
        
        if substring in prefix:
            l += 1
        else:
            c += 1
            i += l
            l = 1
    
    # Normalisierung
    if n > 0:
        b = n / np.log2(n) if n > 1 else 1
        return float(c / b) if b > 0 else np.nan
    
    return np.nan


# ==============================================================================
# TEMPORALE FEATURES
# ==============================================================================

def _compute_temporal_features(
    signal: np.ndarray,
    fs: float,
    prefix: str
) -> Dict:
    """
    Zeitdomäne-Features:
    - Zero-Crossing Rate
    - Line Length
    - Anzahl Peaks
    - Signal-Energie in Teilsegmenten (Variabilität über Zeit)
    """
    features = {
        f"{prefix}_zero_crossing_rate": np.nan,
        f"{prefix}_line_length": np.nan,
        f"{prefix}_n_peaks": np.nan,
        f"{prefix}_peak_rate_hz": np.nan,
        f"{prefix}_energy_variability": np.nan,
        f"{prefix}_amplitude_range_iqr": np.nan,
    }
    
    if len(signal) < 10:
        return features
    
    try:
        # --- Zero-Crossing Rate ---
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal))) > 0)
        duration_sec = len(signal) / fs
        features[f"{prefix}_zero_crossing_rate"] = float(
            zero_crossings / duration_sec if duration_sec > 0 else 0
        )
        
        # --- Line Length ---
        # Summe der absoluten Differenzen (Maß für Signalkomplexität)
        line_length = np.sum(np.abs(np.diff(signal)))
        features[f"{prefix}_line_length"] = float(line_length)
        # Normalisiert pro Sample
        features[f"{prefix}_line_length_norm"] = float(
            line_length / len(signal) if len(signal) > 0 else 0
        )
        
        # --- Peaks ---
        # Finde Peaks mit Mindestabstand
        min_distance = int(fs * 0.1)  # 100ms Mindestabstand
        peaks, properties = scipy_signal.find_peaks(
            signal, distance=max(1, min_distance)
        )
        features[f"{prefix}_n_peaks"] = len(peaks)
        features[f"{prefix}_peak_rate_hz"] = float(
            len(peaks) / duration_sec if duration_sec > 0 else 0
        )
        
        # --- Energie-Variabilität über Teilsegmente ---
        # Teile Signal in 10 gleiche Teile und berechne Energie-Variabilität
        n_subsegments = 10
        subseg_len = len(signal) // n_subsegments
        if subseg_len > 0:
            energies = []
            for i in range(n_subsegments):
                start = i * subseg_len
                end = start + subseg_len
                subseg = signal[start:end]
                energies.append(np.mean(subseg ** 2))
            
            energies = np.array(energies)
            features[f"{prefix}_energy_variability"] = float(
                np.std(energies) / np.mean(energies)
                if np.mean(energies) > 0 else np.nan
            )
        
        # --- Amplitude IQR ---
        q75, q25 = np.percentile(signal, [75, 25])
        features[f"{prefix}_amplitude_range_iqr"] = float(q75 - q25)
    
    except Exception:
        pass
    
    return features

# ==============================================================================
# DELTA POWER ENTROPY
# ==============================================================================

def compute_delta_power_entropy(
    eeg_preprocessed: Dict[str, Dict],
    sfreq: float,
    stages_raw: pd.DataFrame,
    logger: Optional[object] = None,
    window_sec: float = 5.0,
    delta_band: tuple = (0.5, 4.5),
    multitaper_bandwidth: float = 2.0  # Recommended for 5s windows
) -> float:
    """
    Compute the average delta power entropy (spectral entropy of the smoothed delta density function)
    across all preprocessed EEG derivations using multitaper PSD estimation.

    Args:
        eeg_preprocessed (Dict[str, Dict]): Dict mapping channel_name to dict with key 'eeg_cleaned' (np.ndarray).
        sfreq (float): Sampling frequency in Hz.
        stages_raw (pd.DataFrame): DataFrame with columns ['start_sec', 'end_sec', 'stage_numeric'].
            Stage encoding: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unavailable.
        logger (object, optional): Logger for error messages.
        window_sec (float): Window length in seconds (default: 5).
        delta_band (tuple): Delta band (low, high) in Hz (default: (0.5, 4.5)).
        multitaper_bandwidth (float): Bandwidth parameter for multitaper (Hz).

    Returns:
        float: Mean delta power entropy across all EEG derivations, number of channels used.
    """
    import numpy as np
    import mne
    from scipy.ndimage import gaussian_filter1d
    from scipy.fft import fft

    channel_names = ["F3", "F4", "C4", "C3"]
    try:
        # --- 1. Sleep onset/offset detection ---
        WAKE_CODE, UNAVAILABLE_CODE = 5, 9
        sleep_epochs = stages_raw[~stages_raw["stage_numeric"].isin([WAKE_CODE, UNAVAILABLE_CODE])]

        if sleep_epochs.empty:
            if logger:
                logger.warning("No non-Wake/Unavailable epochs found – returning nan.")
            return np.nan, 0

        onset_sec  = float(sleep_epochs["start_sec"].iloc[0])
        offset_sec = float(sleep_epochs["end_sec"].iloc[-1])

        if offset_sec <= onset_sec:
            if logger:
                logger.warning("Sleep offset ≤ onset – returning nan.")
            return np.nan, 0

        # --- 2. Prepare sample indices for sleep period ---
        onset_sample  = int(np.round(onset_sec  * sfreq))
        offset_sample = int(np.round(offset_sec * sfreq))
        win_samples = int(window_sec * sfreq)
        n_sleep_samples = offset_sample - onset_sample
        n_windows = n_sleep_samples // win_samples

        if n_windows == 0:
            if logger:
                logger.warning("Sleep period too short for a single window – returning nan.")
            return np.nan, 0

        # --- 3. Window times and stage mapping ---
        window_centers_sec = np.array([
            onset_sec + (w * win_samples + win_samples / 2) / sfreq
            for w in range(n_windows)
        ])

        stage_starts = stages_raw["start_sec"].values
        stage_indices = np.searchsorted(stage_starts, window_centers_sec, side="right") - 1
        stage_indices = np.clip(stage_indices, 0, len(stages_raw) - 1)
        window_stages = stages_raw["stage_numeric"].values[stage_indices]

        # --- 4–8. Per-channel entropy calculation ---
        N1_CODE = 3
        fmin, fmax = delta_band
        entropies = []
        used_channels = []

        for channel_name, ch_data in eeg_preprocessed.items():
            if (
                channel_name not in channel_names or
                ch_data is None or
                "eeg_cleaned" not in ch_data or
                ch_data["eeg_cleaned"] is None
            ):
                continue

            full_signal = ch_data["eeg_cleaned"]
            if len(full_signal) < offset_sample:
                if logger:
                    logger.warning(f"Channel {channel_name}: Signal too short for sleep period – skipped.")
                continue

            ch_signal = full_signal[onset_sample:offset_sample]
            if len(ch_signal) < n_windows * win_samples:
                if logger:
                    logger.warning(f"Channel {channel_name}: Not enough data for all windows – skipped.")
                continue

            delta_powers = []
            for w in range(n_windows):
                segment = ch_signal[w * win_samples : (w + 1) * win_samples]
                # Multitaper PSD (MNE)
                psd, freqs = mne.time_frequency.psd_array_multitaper(
                    segment,
                    sfreq=sfreq,
                    fmin=fmin,
                    fmax=fmax,
                    bandwidth=multitaper_bandwidth,
                    adaptive=True,
                    low_bias=True,
                    normalization='full',
                    output='power',
                    verbose=False
                )
                delta_power = np.sum(psd)

                # Mask Wake/N1
                if window_stages[w] in [WAKE_CODE, N1_CODE]:
                    delta_power = 0.0
                delta_powers.append(delta_power)
            delta_powers = np.array(delta_powers)
            if np.sum(delta_powers) == 0.0:
                if logger:
                    logger.warning(f"Channel {channel_name}: all windows masked – skipped.")
                continue

            # Smoothing
            smoothed = gaussian_filter1d(delta_powers, sigma=10)
            total = np.sum(smoothed)
            if total == 0.0:
                continue
            delta = smoothed / total

            delta_fft = fft(delta)
            delta_len = len(delta)
            delta_fft = np.abs(delta_fft[0:delta_len//2])

            # Spectral entropy (Shannon entropy)
            delta_fft_nonzero = delta_fft[delta_fft > 0]
            entropy = -np.sum(delta_fft_nonzero * np.log2(delta_fft_nonzero))
            entropies.append(entropy)
            used_channels.append(channel_name)

        if len(entropies) == 0:
            if logger:
                logger.warning("No valid channel entropies computed – returning nan.")
            return np.nan, 0

        mean_entropy = float(np.mean(entropies))
        return mean_entropy, len(entropies)

    except Exception as exc:
        if logger:
            logger.error(f"compute_delta_power_entropy failed: {exc}", exc_info=True)
        return np.nan, 0
    
# ==============================================================================
# ASYMMETRIE-FEATURES (INTERHEMISPHÄRISCH)
# ==============================================================================
def compute_asymmetry_features(
    left_features: Dict,
    right_features: Dict,
    left_signal: Optional[np.ndarray],
    right_signal: Optional[np.ndarray],
    fs: float,
    region: str,
    segment_idx: int = 0,
    logger=None
) -> Dict:
    """
    Berechnet interhemisphärische Asymmetrie-Features.
    
    Nur möglich wenn beide Hemisphären (links und rechts) verfügbar sind.
    
    Parameters
    ----------
    left_features : Dict
        Features des linken Kanals.
    right_features : Dict
        Features des rechten Kanals.
    left_signal : np.ndarray or None
        Signal des linken Kanals.
    right_signal : np.ndarray or None
        Signal des rechten Kanals.
    fs : float
        Sampling-Rate.
    region : str
        Hirnregion ("frontal", "central", "occipital").
    segment_idx : int
        Segment-Index.
    
    Returns
    -------
    Dict
        Asymmetrie-Features.
    """
    prefix = f"eeg_{region}_asym"
    features = _get_empty_asymmetry_features(prefix)
    features["segment_idx"] = segment_idx
    
    if left_features is None or right_features is None:
        return features
    
    left_prefix = f"eeg_{region}"
    right_prefix = f"eeg_{region}"
    
    # --- Bandpower-Asymmetrie ---
    # Asymmetrie = (Rechts - Links) / (Rechts + Links)
    # Positive Werte = rechts dominant, negative = links dominant
    for band_name in EEG_FREQUENCY_BANDS:
        left_key = f"{left_prefix}_{band_name}_power_abs"
        right_key = f"{right_prefix}_{band_name}_power_abs"
        
        left_power = left_features.get(left_key, np.nan)
        right_power = right_features.get(right_key, np.nan)
        
        if (not np.isnan(left_power) and not np.isnan(right_power) and
                (left_power + right_power) > 0):
            asym = (right_power - left_power) / (right_power + left_power)
            features[f"{prefix}_{band_name}_asym"] = float(asym)
            
            # Log-transformierte Asymmetrie (robuster)
            if left_power > 0 and right_power > 0:
                features[f"{prefix}_{band_name}_log_asym"] = float(
                    np.log(right_power) - np.log(left_power)
                )
        else:
            features[f"{prefix}_{band_name}_asym"] = np.nan
            features[f"{prefix}_{band_name}_log_asym"] = np.nan
    
    # --- Kohärenz zwischen Hemisphären ---
    if (left_signal is not None and right_signal is not None and
            len(left_signal) > 0 and len(right_signal) > 0):
        
        min_len = min(len(left_signal), len(right_signal))
        left_sig = left_signal[:min_len]
        right_sig = right_signal[:min_len]
        
        try:
            nperseg = min(min_len, int(fs * 4))
            freqs_coh, coherence = scipy_signal.coherence(
                left_sig, right_sig,
                fs=fs, nperseg=nperseg
            )
            
            # Kohärenz pro Frequenzband
            for band_name, (fmin, fmax) in EEG_FREQUENCY_BANDS.items():
                band_mask = (freqs_coh >= fmin) & (freqs_coh <= fmax)
                if band_mask.any():
                    features[f"{prefix}_{band_name}_coherence"] = float(
                        np.mean(coherence[band_mask])
                    )
                else:
                    features[f"{prefix}_{band_name}_coherence"] = np.nan
            
            # Gesamt-Kohärenz (0.5-35 Hz)
            total_mask = (freqs_coh >= 0.5) & (freqs_coh <= 35.0)
            if total_mask.any():
                features[f"{prefix}_total_coherence"] = float(
                    np.mean(coherence[total_mask])
                )
        except Exception:
            pass
    
    return features


def _get_empty_asymmetry_features(prefix: str) -> Dict:
    """Leere Asymmetrie-Features."""
    features = {
        f"{prefix}_total_coherence": np.nan,
    }
    for band_name in EEG_FREQUENCY_BANDS:
        features[f"{prefix}_{band_name}_asym"] = np.nan
        features[f"{prefix}_{band_name}_log_asym"] = np.nan
        features[f"{prefix}_{band_name}_coherence"] = np.nan
    return features


# ==============================================================================
# LEERES FEATURE-TEMPLATE
# ==============================================================================

def _get_empty_eeg_features(segment_idx: int, prefix: str) -> Dict:
    """
    Erstellt ein leeres Feature-Dictionary mit allen EEG-Feature-Namen.
    
    === CHANGED: Now includes spindle, SO, and coupling feature keys ===
    """
    features = {
        "segment_idx": segment_idx,
        f"{prefix}_sqi": 0.0,
        f"{prefix}_quality_ok": False,
        f"{prefix}_strategy": "none",
    }
    
    # Basis-Statistiken
    for key in ["signal_mean", "signal_std", "signal_range",
                 "signal_skewness", "signal_kurtosis", "signal_rms",
                 "signal_abs_mean", "duration_sec"]:
        features[f"{prefix}_{key}"] = np.nan
    
    # Spektrale Features
    features[f"{prefix}_total_power"] = np.nan
    features[f"{prefix}_spectral_entropy"] = np.nan
    features[f"{prefix}_sef50"] = np.nan
    features[f"{prefix}_sef95"] = np.nan
    features[f"{prefix}_mean_frequency"] = np.nan
    features[f"{prefix}_median_frequency"] = np.nan
    
    for band_name in EEG_FREQUENCY_BANDS:
        features[f"{prefix}_{band_name}_power_abs"] = np.nan
        features[f"{prefix}_{band_name}_power_rel"] = np.nan
        features[f"{prefix}_{band_name}_power_log"] = np.nan
        features[f"{prefix}_{band_name}_peak_freq"] = np.nan
        features[f"{prefix}_{band_name}_mean_freq"] = np.nan
    
    # Schlafspezifische Features
    for key in ["swa_power", "swa_power_log", "slow_spindle_power",
                 "fast_spindle_power", "spindle_power_total",
                 "spindle_prominence", "alpha_theta_ratio",
                 "theta_alpha_ratio", "delta_alpha_ratio",
                 "delta_beta_ratio", "slowing_ratio", "dar", "dtabr"]:
        features[f"{prefix}_{key}"] = np.nan
    
    # Hjorth
    for key in ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]:
        features[f"{prefix}_{key}"] = np.nan
    
    # Entropie
    for key in ["sample_entropy", "permutation_entropy", "approximate_entropy"]:
        features[f"{prefix}_{key}"] = np.nan
    
    # Komplexität
    for key in ["higuchi_fd", "dfa_alpha", "lzc"]:
        features[f"{prefix}_{key}"] = np.nan
    
    # Temporal
    for key in ["zero_crossing_rate", "line_length", "line_length_norm",
                 "n_peaks", "peak_rate_hz", "energy_variability",
                 "amplitude_range_iqr"]:
        features[f"{prefix}_{key}"] = np.nan
    
    # === NEW: Spindle features ===
    sp_prefix = f"{prefix}_sp"
    for key in ["count", "density", "duration_mean", "duration_std",
                 "duration_median", "amplitude_mean", "amplitude_std",
                 "amplitude_median", "rms_mean", "frequency_mean",
                 "frequency_std", "slow_count", "fast_count",
                 "slow_density", "fast_density", "fast_slow_ratio",
                 "symmetry_mean", "rel_power_mean", "oscillations_mean"]:
        features[f"{sp_prefix}_{key}"] = np.nan

    so_prefix = f"{prefix}_so"
    for key in ["count", "density", "duration_mean", "duration_std",
                 "ptp_amplitude_mean", "ptp_amplitude_std",
                 "ptp_amplitude_median", "neg_peak_mean", "neg_peak_std",
                 "pos_peak_mean", "frequency_mean", "slope_mean",
                 "slope_std"]:
        features[f"{so_prefix}_{key}"] = np.nan

    coup_prefix = f"{prefix}_coup"
    for key in ["count", "rate", "mean_phase_rad", "mean_phase_deg",
                 "mrl", "phase_std_rad", "rayleigh_z", "rayleigh_p",
                 "preferred_phase_quadrant", "pac_mi"]:
        features[f"{coup_prefix}_{key}"] = np.nan
    # === END NEW ===
    
    return features


# ==============================================================================
# BATCH-VERARBEITUNG: ALLE SEGMENTE EINES PATIENTEN
# ==============================================================================

def extract_eeg_features_all_segments(
    eeg_segments: Dict[str, list],
    eeg_strategies: Optional[Dict] = None,
    logger=None
) -> pd.DataFrame:
    """
    Extrahiert EEG-Features für alle Segmente und Regionen eines Patienten.
    
    Parameters
    ----------
    eeg_segments : Dict[str, list]
        Dictionary mit Regionen als Keys und Listen von SignalSegment als Values.
    eeg_strategies : Dict, optional
        Kanalstrategien pro Region (aus preprocess_eeg.py).
    logger : loguru.Logger, optional
    
    Returns
    -------
    pd.DataFrame
        Feature-Tabelle mit einer Zeile pro Segment.
        Enthält Features aller verfügbaren Regionen.
    """
    if not eeg_segments:
        if logger:
            logger.warning("Keine EEG-Segmente für Feature-Extraktion vorhanden.")
        return pd.DataFrame()
    
    # Bestimme Anzahl Segmente
    n_segments = max(len(segs) for segs in eeg_segments.values())
    
    # Sammle Features pro Segment
    all_segment_features = [{} for _ in range(n_segments)]
    
    # --- Features pro Region ---
    for region, segments in eeg_segments.items():
        strategy = "unknown"
        if eeg_strategies and region in eeg_strategies:
            strategy = eeg_strategies[region].get("strategy", "unknown")
        
        if logger:
            logger.info(f"EEG Feature-Extraktion [{region}]: "
                        f"{len(segments)} Segmente, Strategie: {strategy}")
        
        for i, seg in enumerate(segments):
            if i >= n_segments:
                break
            
            if seg is None:
                prefix = f"eeg_{region}"
                features = _get_empty_eeg_features(i, prefix)
            else:
                features = extract_eeg_features_segment(
                    segment_data=seg.data,
                    fs=seg.fs,
                    region=region,
                    sqi=seg.sqi,
                    segment_idx=seg.segment_idx,
                    strategy=strategy,
                    logger=logger,
                )
            
            all_segment_features[i].update(features)
    
    # --- Asymmetrie-Features (falls beide Hemisphären verfügbar) ---
    for region_name, (left_std, right_std) in EEG_HOMOLOG_PAIRS.items():
        left_region_segs = eeg_segments.get(region_name, [])
        
        # Prüfe ob die Strategie "averaged" ist – dann keine Asymmetrie
        if eeg_strategies and region_name in eeg_strategies:
            if eeg_strategies[region_name].get("strategy") == "averaged":
                if logger:
                    logger.debug(f"Asymmetrie [{region_name}]: Übersprungen "
                                 f"(Strategie=averaged)")
                continue
        
        # Für Asymmetrie brauchen wir die Rohsignale beider Kanäle
        # Diese sind nur verfügbar wenn die Strategie NICHT averaged ist
        # und beide Kanäle einzeln vorverarbeitet wurden
        # In diesem Fall fügen wir leere Asymmetrie-Features hinzu
        prefix = f"eeg_{region_name}_asym"
        for i in range(n_segments):
            empty_asym = _get_empty_asymmetry_features(prefix)
            empty_asym["segment_idx"] = i
            all_segment_features[i].update(empty_asym)
    
    # --- Zusammenführen ---
    df = pd.DataFrame(all_segment_features)
    
    # Stelle sicher dass segment_idx vorhanden und korrekt ist
    if "segment_idx" not in df.columns:
        df["segment_idx"] = range(len(df))
    
    if logger:
        for region in eeg_segments:
            prefix = f"eeg_{region}"
            quality_col = f"{prefix}_quality_ok"
            if quality_col in df.columns:
                n_computed = df[quality_col].sum()
                logger.info(f"EEG [{region}]: {n_computed}/{len(df)} Segmente "
                            f"mit Feature-Berechnung")
        
        # Feature-Zusammenfassung
        n_features = len([c for c in df.columns if c.startswith("eeg_")])
        logger.info(f"EEG Features gesamt: {n_features} Features pro Segment")
    
    return df

