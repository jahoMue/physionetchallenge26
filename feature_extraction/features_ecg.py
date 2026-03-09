"""
03_feature_extraction/features_ecg.py
=======================================
ECG Feature-Extraktion pro Segment:
- Time-Domain HRV (SDNN, RMSSD, pNN50, etc.)
- Frequency-Domain HRV (VLF, LF, HF, LF/HF Ratio)
- Nonlinear HRV (SD1, SD2, Sample Entropy, DFA)
- Herzfrequenz-Statistiken
- Respiratorische Sinus-Arrhythmie (RSA)

Alle HRV-Parameter werden nur für Segmente mit ausreichender
Signalqualität (SQI >= Schwellenwert) berechnet.

Verwendet NeuroKit2 als primäre Toolbox.
"""

import numpy as np
import pandas as pd
import neurokit2 as nk
from scipy import signal as scipy_signal
from typing import Dict, List, Optional, Tuple

from config import (
    SEGMENT_LENGTH_SEC, HRV_MIN_RR_INTERVALS,
    HRV_RR_MIN_MS, HRV_RR_MAX_MS,
    ECG_SQI_THRESHOLD, RSA_ENABLED,
    EEG_FREQUENCY_BANDS
)

# Suppress NeuroKit warnings for short segments
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ==============================================================================
# HAUPT-FUNKTION: FEATURES PRO SEGMENT
# ==============================================================================

def extract_ecg_features_segment(
    segment_data: np.ndarray,
    fs: float,
    rpeaks_local: Optional[np.ndarray] = None,
    rr_intervals: Optional[np.ndarray] = None,
    rr_valid_mask: Optional[np.ndarray] = None,
    sqi: float = 0.0,
    segment_idx: int = 0,
    resp_signal: Optional[np.ndarray] = None,
    resp_fs: Optional[float] = None,
    logger=None
) -> Dict:
    """
    Extrahiert alle ECG-Features für ein einzelnes Segment.
    
    Parameters
    ----------
    segment_data : np.ndarray
        ECG-Signaldaten des Segments.
    fs : float
        Sampling-Rate in Hz.
    rpeaks_local : np.ndarray, optional
        R-Peak Indizes (lokal zum Segment).
    rr_intervals : np.ndarray, optional
        RR-Intervalle in ms.
    rr_valid_mask : np.ndarray, optional
        Boolean-Maske für gültige RR-Intervalle.
    sqi : float
        Signal Quality Index des Segments.
    segment_idx : int
        Index des Segments.
    resp_signal : np.ndarray, optional
        Respirationssignal (für RSA-Berechnung).
    resp_fs : float, optional
        Sampling-Rate des Respirationssignals.
    logger : loguru.Logger, optional
    
    Returns
    -------
    Dict
        Dictionary mit allen ECG-Features.
    """
    features = _get_empty_ecg_features(segment_idx)
    features["ecg_sqi"] = sqi
    features["ecg_quality_ok"] = sqi >= ECG_SQI_THRESHOLD
    
    # --- Prüfe ob genügend Daten vorhanden ---
    if segment_data is None or len(segment_data) == 0:
        if logger:
            logger.debug(f"Segment {segment_idx}: Keine ECG-Daten.")
        return features
    
    if sqi < ECG_SQI_THRESHOLD:
        if logger:
            logger.debug(f"Segment {segment_idx}: SQI zu niedrig ({sqi:.3f}), "
                         f"überspringe HRV-Berechnung.")
        # Trotzdem Basis-Statistiken berechnen
        features.update(_compute_basic_ecg_stats(segment_data, fs))
        return features
    
    # --- R-Peaks: Falls nicht vorhanden, neu detektieren ---
    if rpeaks_local is None or len(rpeaks_local) < 2:
        try:
            _, peaks_info = nk.ecg_peaks(
                segment_data, sampling_rate=int(fs), method="neurokit"
            )
            rpeaks_local = peaks_info["ECG_R_Peaks"]
        except Exception as e:
            if logger:
                logger.warning(f"Segment {segment_idx}: R-Peak-Detektion "
                               f"fehlgeschlagen: {e}")
            features.update(_compute_basic_ecg_stats(segment_data, fs))
            return features
    
    if len(rpeaks_local) < 2:
        if logger:
            logger.debug(f"Segment {segment_idx}: Zu wenige R-Peaks "
                         f"({len(rpeaks_local)}).")
        features.update(_compute_basic_ecg_stats(segment_data, fs))
        return features
    
    # --- RR-Intervalle: Falls nicht vorhanden, berechnen ---
    if rr_intervals is None or len(rr_intervals) == 0:
        rr_intervals = np.diff(rpeaks_local) / fs * 1000  # in ms
        rr_valid_mask = (
            (rr_intervals >= HRV_RR_MIN_MS) &
            (rr_intervals <= HRV_RR_MAX_MS)
        )
    
    if rr_valid_mask is None:
        rr_valid_mask = np.ones(len(rr_intervals), dtype=bool)
    
    # Nur gültige RR-Intervalle verwenden
    valid_rr = rr_intervals[rr_valid_mask]
    
    if len(valid_rr) < HRV_MIN_RR_INTERVALS:
        if logger:
            logger.debug(f"Segment {segment_idx}: Zu wenige gültige RR-Intervalle "
                         f"({len(valid_rr)}/{HRV_MIN_RR_INTERVALS}).")
        features.update(_compute_basic_ecg_stats(segment_data, fs))
        features.update(_compute_hr_stats(valid_rr))
        return features
    
    # --- Feature-Berechnung ---
    features.update(_compute_basic_ecg_stats(segment_data, fs))
    features.update(_compute_hr_stats(valid_rr))
    features.update(_compute_hrv_time_domain(valid_rr, fs))
    features.update(_compute_hrv_frequency_domain(valid_rr, fs))
    features.update(_compute_hrv_nonlinear(valid_rr))
    features.update(_compute_rr_artifact_stats(rr_intervals, rr_valid_mask))
    
    # --- RSA (Respiratorische Sinus-Arrhythmie) ---
    if RSA_ENABLED and resp_signal is not None and resp_fs is not None:
        rsa_features = _compute_rsa(
            rpeaks_local, fs, resp_signal, resp_fs, logger
        )
        features.update(rsa_features)
    
    if logger:
        logger.debug(f"Segment {segment_idx}: {len(features)} ECG-Features extrahiert.")
    
    return features


# ==============================================================================
# HERZFREQUENZ-STATISTIKEN
# ==============================================================================

def _compute_basic_ecg_stats(signal: np.ndarray, fs: float) -> Dict:
    """Basis-Signalstatistiken des ECG."""
    return {
        "ecg_signal_mean": float(np.mean(signal)),
        "ecg_signal_std": float(np.std(signal)),
        "ecg_signal_range": float(np.ptp(signal)),
        "ecg_signal_skewness": float(_safe_skewness(signal)),
        "ecg_signal_kurtosis": float(_safe_kurtosis(signal)),
        "ecg_duration_sec": float(len(signal) / fs),
    }


def _compute_hr_stats(valid_rr: np.ndarray) -> Dict:
    """Herzfrequenz-Statistiken aus gültigen RR-Intervallen."""
    if len(valid_rr) == 0:
        return {
            "hr_mean": np.nan,
            "hr_std": np.nan,
            "hr_min": np.nan,
            "hr_max": np.nan,
            "hr_range": np.nan,
            "hr_median": np.nan,
            "hr_cv": np.nan,
            "rr_mean_ms": np.nan,
            "rr_std_ms": np.nan,
            "rr_median_ms": np.nan,
            "n_valid_rr": 0,
        }
    
    # Herzfrequenz in bpm
    hr = 60000.0 / valid_rr
    
    return {
        "hr_mean": float(np.mean(hr)),
        "hr_std": float(np.std(hr)),
        "hr_min": float(np.min(hr)),
        "hr_max": float(np.max(hr)),
        "hr_range": float(np.ptp(hr)),
        "hr_median": float(np.median(hr)),
        "hr_cv": float(np.std(hr) / np.mean(hr)) if np.mean(hr) > 0 else np.nan,
        "rr_mean_ms": float(np.mean(valid_rr)),
        "rr_std_ms": float(np.std(valid_rr)),
        "rr_median_ms": float(np.median(valid_rr)),
        "n_valid_rr": int(len(valid_rr)),
    }


# ==============================================================================
# TIME-DOMAIN HRV
# ==============================================================================

def _compute_hrv_time_domain(valid_rr: np.ndarray, fs: float) -> Dict:
    """
    Time-Domain HRV Parameter.
    
    Berechnet:
    - SDNN: Standard deviation of NN intervals
    - RMSSD: Root mean square of successive differences
    - pNN50: Percentage of successive differences > 50ms
    - pNN20: Percentage of successive differences > 20ms
    - SDSD: Standard deviation of successive differences
    - MeanNN, MedianNN, CVNN
    - HTI: HRV Triangular Index
    - TINN: Triangular Interpolation of NN interval histogram
    """
    features = {}
    
    try:
        # NeuroKit2 HRV Time-Domain
        # Erstelle ein Peaks-Array für NeuroKit
        # NeuroKit erwartet kumulative Sample-Indizes
        cumulative_samples = np.cumsum(
            np.concatenate([[0], valid_rr / 1000 * fs])
        ).astype(int)
        
        # Verwende NeuroKit2 für robuste Berechnung
        hrv_time = nk.hrv_time(
            {"ECG_R_Peaks": cumulative_samples[1:]},
            sampling_rate=int(fs),
            show=False
        )
        
        # Extrahiere relevante Parameter
        time_params = {
            "hrv_sdnn": "HRV_SDNN",
            "hrv_rmssd": "HRV_RMSSD",
            "hrv_pnn50": "HRV_pNN50",
            "hrv_pnn20": "HRV_pNN20",
            "hrv_sdsd": "HRV_SDSD",
            "hrv_mean_nn": "HRV_MeanNN",
            "hrv_median_nn": "HRV_MedianNN",
            "hrv_cv_nn": "HRV_CVNN",
            "hrv_cvsd": "HRV_CVSD",
            "hrv_hti": "HRV_HTI",
            "hrv_tinn": "HRV_TINN",
        }
        
        for feat_name, nk_name in time_params.items():
            if nk_name in hrv_time.columns:
                val = hrv_time[nk_name].values[0]
                features[feat_name] = float(val) if pd.notna(val) else np.nan
            else:
                features[feat_name] = np.nan
    
    except Exception:
        # Fallback: Manuelle Berechnung
        features = _compute_hrv_time_domain_manual(valid_rr)
    
    return features


def _compute_hrv_time_domain_manual(valid_rr: np.ndarray) -> Dict:
    """Manuelle Berechnung der Time-Domain HRV als Fallback."""
    features = {}
    
    nn = valid_rr  # NN-Intervalle in ms
    
    # SDNN
    features["hrv_sdnn"] = float(np.std(nn, ddof=1)) if len(nn) > 1 else np.nan
    
    # Successive Differences
    sd = np.diff(nn)
    
    # RMSSD
    features["hrv_rmssd"] = float(np.sqrt(np.mean(sd ** 2))) if len(sd) > 0 else np.nan
    
    # SDSD
    features["hrv_sdsd"] = float(np.std(sd, ddof=1)) if len(sd) > 1 else np.nan
    
    # pNN50
    features["hrv_pnn50"] = (
        float(np.sum(np.abs(sd) > 50) / len(sd) * 100)
        if len(sd) > 0 else np.nan
    )
    
    # pNN20
    features["hrv_pnn20"] = (
        float(np.sum(np.abs(sd) > 20) / len(sd) * 100)
        if len(sd) > 0 else np.nan
    )
    
    # MeanNN, MedianNN
    features["hrv_mean_nn"] = float(np.mean(nn))
    features["hrv_median_nn"] = float(np.median(nn))
    
    # CVNN (Coefficient of Variation)
    features["hrv_cv_nn"] = (
        float(np.std(nn, ddof=1) / np.mean(nn))
        if np.mean(nn) > 0 else np.nan
    )
    
    # CVSD
    features["hrv_cvsd"] = (
        float(features["hrv_rmssd"] / features["hrv_mean_nn"])
        if features["hrv_mean_nn"] > 0 and not np.isnan(features["hrv_rmssd"])
        else np.nan
    )
    
    # HTI (HRV Triangular Index)
    try:
        bin_width = 1000 / 128  # ~7.8125 ms bins
        hist, _ = np.histogram(nn, bins=np.arange(nn.min(), nn.max() + bin_width, bin_width))
        features["hrv_hti"] = float(len(nn) / np.max(hist)) if np.max(hist) > 0 else np.nan
    except Exception:
        features["hrv_hti"] = np.nan
    
    # TINN (Placeholder - complex to compute manually)
    features["hrv_tinn"] = np.nan
    
    return features


# ==============================================================================
# FREQUENCY-DOMAIN HRV
# ==============================================================================

def _compute_hrv_frequency_domain(valid_rr: np.ndarray, fs: float) -> Dict:
    """
    Frequency-Domain HRV Parameter.
    
    Berechnet:
    - VLF Power (0.003-0.04 Hz): Thermoregulation, RAAS
    - LF Power (0.04-0.15 Hz): Sympathisch + Parasympathisch
    - HF Power (0.15-0.4 Hz): Parasympathisch (vagal)
    - LF/HF Ratio: Sympathovagale Balance
    - Total Power
    - Normalisierte LF/HF
    
    Hinweis: Für 300s Segmente ist VLF nur eingeschränkt interpretierbar.
    """
    features = {
        "hrv_vlf_power": np.nan,
        "hrv_lf_power": np.nan,
        "hrv_hf_power": np.nan,
        "hrv_total_power": np.nan,
        "hrv_lf_hf_ratio": np.nan,
        "hrv_lf_norm": np.nan,
        "hrv_hf_norm": np.nan,
        "hrv_lf_peak_hz": np.nan,
        "hrv_hf_peak_hz": np.nan,
    }
    
    if len(valid_rr) < HRV_MIN_RR_INTERVALS:
        return features
    
    try:
        # NeuroKit2 HRV Frequency-Domain
        cumulative_samples = np.cumsum(
            np.concatenate([[0], valid_rr / 1000 * fs])
        ).astype(int)
        
        hrv_freq = nk.hrv_frequency(
            {"ECG_R_Peaks": cumulative_samples[1:]},
            sampling_rate=int(fs),
            show=False,
            psd_method="welch"
        )
        
        freq_params = {
            "hrv_vlf_power": "HRV_VLF",
            "hrv_lf_power": "HRV_LF",
            "hrv_hf_power": "HRV_HF",
            "hrv_lf_hf_ratio": "HRV_LFHF",
            "hrv_lf_norm": "HRV_LFn",
            "hrv_hf_norm": "HRV_HFn",
        }
        
        for feat_name, nk_name in freq_params.items():
            if nk_name in hrv_freq.columns:
                val = hrv_freq[nk_name].values[0]
                features[feat_name] = float(val) if pd.notna(val) else np.nan
        
        # Total Power
        vlf = features.get("hrv_vlf_power", 0) or 0
        lf = features.get("hrv_lf_power", 0) or 0
        hf = features.get("hrv_hf_power", 0) or 0
        features["hrv_total_power"] = vlf + lf + hf
    
    except Exception:
        # Fallback: Manuelle Berechnung
        features = _compute_hrv_frequency_manual(valid_rr)
    
    return features


def _compute_hrv_frequency_manual(valid_rr: np.ndarray) -> Dict:
    """Manuelle Frequency-Domain HRV Berechnung als Fallback."""
    features = {
        "hrv_vlf_power": np.nan,
        "hrv_lf_power": np.nan,
        "hrv_hf_power": np.nan,
        "hrv_total_power": np.nan,
        "hrv_lf_hf_ratio": np.nan,
        "hrv_lf_norm": np.nan,
        "hrv_hf_norm": np.nan,
        "hrv_lf_peak_hz": np.nan,
        "hrv_hf_peak_hz": np.nan,
    }
    
    try:
        # Interpoliere RR-Intervalle auf gleichmäßiges Zeitraster (4 Hz)
        rr_sec = valid_rr / 1000.0
        rr_cumsum = np.cumsum(rr_sec)
        rr_cumsum = np.concatenate([[0], rr_cumsum])
        
        interp_fs = 4.0  # Hz
        t_interp = np.arange(0, rr_cumsum[-1], 1.0 / interp_fs)
        
        if len(t_interp) < 16:
            return features
        
        rr_interp = np.interp(t_interp, rr_cumsum[1:], rr_sec)
        
        # Entferne Mittelwert
        rr_interp = rr_interp - np.mean(rr_interp)
        
        # Welch PSD
        nperseg = min(len(rr_interp), 256)
        freqs, psd = scipy_signal.welch(
            rr_interp, fs=interp_fs, nperseg=nperseg,
            noverlap=nperseg // 2
        )
        
        # Frequenzbänder
        vlf_mask = (freqs >= 0.003) & (freqs < 0.04)
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.4)
        
        vlf_power = float(np.trapz(psd[vlf_mask], freqs[vlf_mask])) if vlf_mask.any() else 0
        lf_power = float(np.trapz(psd[lf_mask], freqs[lf_mask])) if lf_mask.any() else 0
        hf_power = float(np.trapz(psd[hf_mask], freqs[hf_mask])) if hf_mask.any() else 0
        total_power = vlf_power + lf_power + hf_power
        
        features["hrv_vlf_power"] = vlf_power
        features["hrv_lf_power"] = lf_power
        features["hrv_hf_power"] = hf_power
        features["hrv_total_power"] = total_power
        
        # LF/HF Ratio
        features["hrv_lf_hf_ratio"] = (
            lf_power / hf_power if hf_power > 0 else np.nan
        )
        
        # Normalisierte Werte
        lf_hf_sum = lf_power + hf_power
        features["hrv_lf_norm"] = (
            lf_power / lf_hf_sum * 100 if lf_hf_sum > 0 else np.nan
        )
        features["hrv_hf_norm"] = (
            hf_power / lf_hf_sum * 100 if lf_hf_sum > 0 else np.nan
        )
        
        # Peak-Frequenzen
        if lf_mask.any():
            lf_freqs = freqs[lf_mask]
            lf_psd = psd[lf_mask]
            features["hrv_lf_peak_hz"] = float(lf_freqs[np.argmax(lf_psd)])
        
        if hf_mask.any():
            hf_freqs = freqs[hf_mask]
            hf_psd = psd[hf_mask]
            features["hrv_hf_peak_hz"] = float(hf_freqs[np.argmax(hf_psd)])
    
    except Exception:
        pass
    
    return features


# ==============================================================================
# NONLINEAR HRV
# ==============================================================================

def _compute_hrv_nonlinear(valid_rr: np.ndarray) -> Dict:
    """
    Nonlinear HRV Parameter.
    
    Berechnet:
    - SD1, SD2: Poincaré Plot Parameter
    - SD1/SD2 Ratio
    - Sample Entropy (SampEn)
    - Approximate Entropy (ApEn)
    - Detrended Fluctuation Analysis (DFA) α1, α2
    - Correlation Dimension (CD)
    """
    features = {
        "hrv_sd1": np.nan,
        "hrv_sd2": np.nan,
        "hrv_sd1_sd2_ratio": np.nan,
        "hrv_sample_entropy": np.nan,
        "hrv_approximate_entropy": np.nan,
        "hrv_dfa_alpha1": np.nan,
        "hrv_dfa_alpha2": np.nan,
        "hrv_csi": np.nan,   # Cardiac Sympathetic Index (SD2/SD1)
        "hrv_cvi": np.nan,   # Cardiac Vagal Index (log10(SD1*SD2))
    }
    
    if len(valid_rr) < HRV_MIN_RR_INTERVALS:
        return features
    
    # --- Poincaré Plot (SD1, SD2) ---
    try:
        sd = np.diff(valid_rr)
        sd1 = float(np.std(sd, ddof=1) / np.sqrt(2))
        sd2 = float(np.sqrt(2 * np.std(valid_rr, ddof=1) ** 2 - sd1 ** 2))
        
        # Sicherheitscheck für negative Werte unter der Wurzel
        if np.isnan(sd2):
            sd2 = float(np.std(valid_rr, ddof=1))
        
        features["hrv_sd1"] = sd1
        features["hrv_sd2"] = sd2
        features["hrv_sd1_sd2_ratio"] = sd1 / sd2 if sd2 > 0 else np.nan
        
        # CSI und CVI
        features["hrv_csi"] = sd2 / sd1 if sd1 > 0 else np.nan
        features["hrv_cvi"] = (
            float(np.log10(sd1 * sd2)) if sd1 > 0 and sd2 > 0 else np.nan
        )
    except Exception:
        pass
    
    # --- Sample Entropy ---
    try:
        sampen = nk.entropy_sample(valid_rr, dimension=2, tolerance="sd")
        features["hrv_sample_entropy"] = (
            float(sampen) if not np.isnan(sampen) and not np.isinf(sampen) 
            else np.nan
        )
    except Exception:
        try:
            # Fallback: Manuelle Berechnung
            features["hrv_sample_entropy"] = _sample_entropy(valid_rr, m=2)
        except Exception:
            pass
    
    # --- Approximate Entropy ---
    try:
        apen = nk.entropy_approximate(valid_rr, dimension=2, tolerance="sd")
        features["hrv_approximate_entropy"] = (
            float(apen) if not np.isnan(apen) and not np.isinf(apen)
            else np.nan
        )
    except Exception:
        pass
    
    # --- DFA (Detrended Fluctuation Analysis) ---
    try:
        dfa_result = nk.fractal_dfa(valid_rr, show=False)
        
        if isinstance(dfa_result, tuple):
            # NeuroKit2 gibt manchmal (alpha, info_dict) zurück
            alpha = dfa_result[0]
            if isinstance(alpha, (int, float)):
                features["hrv_dfa_alpha1"] = float(alpha)
            elif hasattr(alpha, '__len__') and len(alpha) >= 1:
                features["hrv_dfa_alpha1"] = float(alpha[0])
        elif isinstance(dfa_result, (int, float)):
            features["hrv_dfa_alpha1"] = float(dfa_result)
    except Exception:
        try:
            features["hrv_dfa_alpha1"] = _dfa_alpha1(valid_rr)
        except Exception:
            pass
    
    return features


def _sample_entropy(data: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Manuelle Sample Entropy Berechnung."""
    N = len(data)
    r = r_factor * np.std(data)
    
    if N < m + 2 or r == 0:
        return np.nan
    
    def _count_matches(template_length):
        count = 0
        templates = np.array([
            data[i:i + template_length] 
            for i in range(N - template_length)
        ])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) < r:
                    count += 1
        return count
    
    A = _count_matches(m + 1)
    B = _count_matches(m)
    
    if B == 0:
        return np.nan
    
    return -np.log(A / B) if A > 0 else np.nan


def _dfa_alpha1(rr_intervals: np.ndarray) -> float:
    """Vereinfachte DFA Alpha1 Berechnung (Short-term: 4-16 beats)."""
    N = len(rr_intervals)
    if N < 16:
        return np.nan
    
    # Integriertes Signal
    y = np.cumsum(rr_intervals - np.mean(rr_intervals))
    
    # Box-Größen (4-16 für Alpha1)
    box_sizes = np.arange(4, min(17, N // 4))
    if len(box_sizes) < 3:
        return np.nan
    
    fluctuations = []
    for n in box_sizes:
        n_boxes = N // n
        if n_boxes < 1:
            continue
        
        rms_values = []
        for i in range(n_boxes):
            segment = y[i * n:(i + 1) * n]
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            rms = np.sqrt(np.mean((segment - trend) ** 2))
            rms_values.append(rms)
        
        if rms_values:
            fluctuations.append(np.mean(rms_values))
        else:
            fluctuations.append(np.nan)
    
    # Log-Log Regression
    valid = ~np.isnan(fluctuations)
    if valid.sum() < 3:
        return np.nan
    
    log_n = np.log(box_sizes[valid])
    log_f = np.log(np.array(fluctuations)[valid])
    
    coeffs = np.polyfit(log_n, log_f, 1)
    return float(coeffs[0])


# ==============================================================================
# RR-ARTEFAKT-STATISTIKEN
# ==============================================================================

def _compute_rr_artifact_stats(
    rr_intervals: np.ndarray,
    rr_valid_mask: np.ndarray
) -> Dict:
    """Statistiken über RR-Artefakte im Segment."""
    if len(rr_intervals) == 0:
        return {
            "rr_total_count": 0,
            "rr_valid_count": 0,
            "rr_artifact_count": 0,
            "rr_artifact_pct": 100.0,
            "rr_longest_valid_run": 0,
            "rr_longest_artifact_run": 0,
        }
    
    n_total = len(rr_intervals)
    n_valid = int(rr_valid_mask.sum())
    n_artifact = n_total - n_valid
    
    # Längste zusammenhängende Sequenz gültiger RR-Intervalle
    longest_valid = _longest_run(rr_valid_mask, True)
    longest_artifact = _longest_run(rr_valid_mask, False)
    
    return {
        "rr_total_count": n_total,
        "rr_valid_count": n_valid,
        "rr_artifact_count": n_artifact,
        "rr_artifact_pct": float(n_artifact / n_total * 100) if n_total > 0 else 100.0,
        "rr_longest_valid_run": longest_valid,
        "rr_longest_artifact_run": longest_artifact,
    }


def _longest_run(mask: np.ndarray, value: bool) -> int:
    """Berechnet die längste zusammenhängende Sequenz eines Wertes."""
    if len(mask) == 0:
        return 0
    
    max_run = 0
    current_run = 0
    
    for m in mask:
        if m == value:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    
    return max_run


# ==============================================================================
# RESPIRATORISCHE SINUS-ARRHYTHMIE (RSA)
# ==============================================================================

def _compute_rsa(
    rpeaks_local: np.ndarray,
    ecg_fs: float,
    resp_signal: np.ndarray,
    resp_fs: float,
    logger=None
) -> Dict:
    """
    Berechnet die Respiratorische Sinus-Arrhythmie (RSA).
    
    RSA ist die herzfrequenzvariabilität, die mit dem Atemzyklus
    synchronisiert ist. Sie ist ein Marker für parasympathische
    (vagale) Aktivität.
    
    Methoden:
    1. Peak-Valley RSA: Differenz zwischen max und min HR pro Atemzyklus
    2. Frequenz-basierte RSA: HF-Power der HRV im Atemfrequenzbereich
    3. Porges-Bohrer RSA: Bandpassfilterung der RR-Zeitreihe
    
    Parameters
    ----------
    rpeaks_local : np.ndarray
        R-Peak Indizes (lokal zum Segment).
    ecg_fs : float
        ECG Sampling-Rate.
    resp_signal : np.ndarray
        Respirationssignal des Segments.
    resp_fs : float
        Respirations-Sampling-Rate.
    logger : loguru.Logger, optional
    
    Returns
    -------
    Dict
        RSA-Features.
    """
    features = {
        "rsa_peak_valley": np.nan,
        "rsa_p2t_mean": np.nan,
        "rsa_p2t_std": np.nan,
        "rsa_gates_mean": np.nan,
        "rsa_resp_rate": np.nan,
        "rsa_coupling_strength": np.nan,
    }
    
    if len(rpeaks_local) < 10 or len(resp_signal) < int(resp_fs * 10):
        return features
    
    try:
        # --- Atemfrequenz bestimmen ---
        resp_rate_hz = _estimate_respiratory_rate(resp_signal, resp_fs)
        if resp_rate_hz is None or resp_rate_hz <= 0:
            return features
        
        features["rsa_resp_rate"] = float(resp_rate_hz * 60)  # Atemzüge pro Minute
        
        # --- Peak-Valley RSA ---
        pv_rsa = _peak_valley_rsa(rpeaks_local, ecg_fs, resp_signal, resp_fs)
        if pv_rsa is not None:
            features["rsa_peak_valley"] = pv_rsa["mean"]
            features["rsa_p2t_mean"] = pv_rsa["mean"]
            features["rsa_p2t_std"] = pv_rsa["std"]
        
        # --- Porges-Bohrer RSA (Bandpass um Atemfrequenz) ---
        gates_rsa = _porges_bohrer_rsa(rpeaks_local, ecg_fs, resp_rate_hz)
        if gates_rsa is not None:
            features["rsa_gates_mean"] = gates_rsa
        
        # --- Kopplungsstärke (Kohärenz zwischen HR und Atmung) ---
        coupling = _compute_cardiorespiratory_coupling(
            rpeaks_local, ecg_fs, resp_signal, resp_fs
        )
        if coupling is not None:
            features["rsa_coupling_strength"] = coupling
    
    except Exception as e:
        if logger:
            logger.debug(f"RSA-Berechnung fehlgeschlagen: {e}")
    
    return features


def _estimate_respiratory_rate(
    resp_signal: np.ndarray,
    fs: float
) -> Optional[float]:
    """
    Schätzt die Atemfrequenz aus dem Respirationssignal.
    
    Returns
    -------
    float or None
        Atemfrequenz in Hz.
    """
    try:
        # Bandpassfilter: 0.05-1.0 Hz (3-60 Atemzüge/min)
        nyq = 0.5 * fs
        low = 0.05 / nyq
        high = min(1.0 / nyq, 0.99)
        
        if low >= high:
            return None
        
        b, a = scipy_signal.butter(3, [low, high], btype='band')
        resp_filtered = scipy_signal.filtfilt(b, a, resp_signal)
        
        # PSD berechnen
        nperseg = min(len(resp_filtered), int(fs * 60))
        freqs, psd = scipy_signal.welch(
            resp_filtered, fs=fs, nperseg=nperseg
        )
        
        # Peak im physiologischen Bereich (0.1-0.5 Hz = 6-30 Atemzüge/min)
        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not resp_mask.any():
            return None
        
        resp_freqs = freqs[resp_mask]
        resp_psd = psd[resp_mask]
        
        peak_freq = resp_freqs[np.argmax(resp_psd)]
        return float(peak_freq)
    
    except Exception:
        return None


def _peak_valley_rsa(
    rpeaks_local: np.ndarray,
    ecg_fs: float,
    resp_signal: np.ndarray,
    resp_fs: float
) -> Optional[Dict]:
    """
    Peak-Valley RSA: Differenz zwischen maximaler und minimaler
    Herzfrequenz innerhalb jedes Atemzyklus.
    """
    try:
        # RR-Intervalle
        rr_sec = np.diff(rpeaks_local) / ecg_fs
        rr_times = rpeaks_local[1:] / ecg_fs
        hr_instantaneous = 60.0 / rr_sec  # bpm
        
        # Respirations-Peaks und -Täler finden
        # Bandpassfilter
        nyq = 0.5 * resp_fs
        low = max(0.05 / nyq, 0.001)
        high = min(0.5 / nyq, 0.999)
        
        b, a = scipy_signal.butter(3, [low, high], btype='band')
        resp_filtered = scipy_signal.filtfilt(b, a, resp_signal)
        
        # Peaks finden (Inspiration)
        min_distance = int(resp_fs * 1.5)  # Mindestens 1.5s zwischen Atemzügen
        resp_peaks, _ = scipy_signal.find_peaks(
            resp_filtered, distance=min_distance
        )
        
        if len(resp_peaks) < 3:
            return None
        
        # Für jeden Atemzyklus: Max HR - Min HR
        rsa_values = []
        resp_peak_times = resp_peaks / resp_fs
        
        for i in range(len(resp_peak_times) - 1):
            cycle_start = resp_peak_times[i]
            cycle_end = resp_peak_times[i + 1]
            
            # HR-Werte in diesem Atemzyklus
            cycle_mask = (rr_times >= cycle_start) & (rr_times < cycle_end)
            cycle_hr = hr_instantaneous[cycle_mask]
            
            if len(cycle_hr) >= 2:
                rsa_val = np.max(cycle_hr) - np.min(cycle_hr)
                rsa_values.append(rsa_val)
        
        if not rsa_values:
            return None
        
        rsa_array = np.array(rsa_values)
        return {
            "mean": float(np.mean(rsa_array)),
            "std": float(np.std(rsa_array)),
            "median": float(np.median(rsa_array)),
            "values": rsa_array,
        }
    
    except Exception:
        return None


def _porges_bohrer_rsa(
    rpeaks_local: np.ndarray,
    ecg_fs: float,
    resp_rate_hz: float
) -> Optional[float]:
    """
    Porges-Bohrer Methode: Bandpassfilterung der RR-Zeitreihe
    um die Atemfrequenz herum.
    
    RSA = Varianz des gefilterten RR-Signals (ln transformiert).
    """
    try:
        rr_sec = np.diff(rpeaks_local) / ecg_fs
        rr_ms = rr_sec * 1000
        
        if len(rr_ms) < 30:
            return None
        
        # Interpoliere auf gleichmäßiges Zeitraster
        rr_times = np.cumsum(rr_sec)
        interp_fs = 4.0
        t_interp = np.arange(0, rr_times[-1], 1.0 / interp_fs)
        
        if len(t_interp) < 16:
            return None
        
        rr_interp = np.interp(t_interp, rr_times, rr_ms)
        
        # Bandpassfilter um Atemfrequenz (±0.05 Hz)
        band_low = max(resp_rate_hz - 0.05, 0.12)
        band_high = min(resp_rate_hz + 0.05, 0.40)
        
        nyq = 0.5 * interp_fs
        low = band_low / nyq
        high = band_high / nyq
        
        if low >= high or low <= 0 or high >= 1:
            return None
        
        b, a = scipy_signal.butter(2, [low, high], btype='band')
        rr_filtered = scipy_signal.filtfilt(b, a, rr_interp)
        
        # RSA = ln(Varianz)
        variance = np.var(rr_filtered)
        if variance > 0:
            return float(np.log(variance))
        
        return None
    
    except Exception:
        return None


def _compute_cardiorespiratory_coupling(
    rpeaks_local: np.ndarray,
    ecg_fs: float,
    resp_signal: np.ndarray,
    resp_fs: float
) -> Optional[float]:
    """
    Berechnet die Kohärenz zwischen Herzfrequenz und Atmung
    als Maß für die kardiorespiratorische Kopplung.
    
    Returns
    -------
    float or None
        Maximale Kohärenz im Atemfrequenzbereich (0-1).
    """
    try:
        rr_sec = np.diff(rpeaks_local) / ecg_fs
        rr_times = np.cumsum(rr_sec)
        
        if len(rr_sec) < 30:
            return None
        
        # Interpoliere beide Signale auf gemeinsames Zeitraster
        interp_fs = 4.0
        max_time = min(rr_times[-1], len(resp_signal) / resp_fs)
        t_interp = np.arange(0, max_time, 1.0 / interp_fs)
        
        if len(t_interp) < 32:
            return None
        
        # RR-Intervalle interpolieren
        rr_interp = np.interp(t_interp, rr_times, rr_sec * 1000)
        
        # Respirationssignal interpolieren
        resp_times = np.arange(len(resp_signal)) / resp_fs
        resp_interp = np.interp(t_interp, resp_times, resp_signal)
        
        # Kohärenz berechnen
        nperseg = min(len(t_interp), 128)
        freqs, coherence = scipy_signal.coherence(
            rr_interp, resp_interp,
            fs=interp_fs, nperseg=nperseg
        )
        
        # Maximale Kohärenz im Atemfrequenzbereich (0.1-0.5 Hz)
        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not resp_mask.any():
            return None
        
        max_coherence = float(np.max(coherence[resp_mask]))
        return max_coherence
    
    except Exception:
        return None


# ==============================================================================
# HILFSFUNKTIONEN
# ==============================================================================

def _safe_skewness(data: np.ndarray) -> float:
    """Sichere Berechnung der Schiefe."""
    if len(data) < 3:
        return np.nan
    try:
        from scipy.stats import skew
        return float(skew(data, nan_policy='omit'))
    except Exception:
        return np.nan


def _safe_kurtosis(data: np.ndarray) -> float:
    """Sichere Berechnung der Kurtosis."""
    if len(data) < 4:
        return np.nan
    try:
        from scipy.stats import kurtosis
        return float(kurtosis(data, nan_policy='omit'))
    except Exception:
        return np.nan


def _get_empty_ecg_features(segment_idx: int = 0) -> Dict:
    """
    Erstellt ein leeres Feature-Dictionary mit allen ECG-Feature-Namen.
    Wird als Template verwendet, um konsistente Spalten zu gewährleisten.
    """
    features = {
        "segment_idx": segment_idx,
        
        # Qualität
        "ecg_sqi": 0.0,
        "ecg_quality_ok": False,
        
        # Basis-Statistiken
        "ecg_signal_mean": np.nan,
        "ecg_signal_std": np.nan,
        "ecg_signal_range": np.nan,
        "ecg_signal_skewness": np.nan,
        "ecg_signal_kurtosis": np.nan,
        "ecg_duration_sec": np.nan,
        
        # Herzfrequenz
        "hr_mean": np.nan,
        "hr_std": np.nan,
        "hr_min": np.nan,
        "hr_max": np.nan,
        "hr_range": np.nan,
        "hr_median": np.nan,
        "hr_cv": np.nan,
        "rr_mean_ms": np.nan,
        "rr_std_ms": np.nan,
        "rr_median_ms": np.nan,
        "n_valid_rr": 0,
        
        # Time-Domain HRV
        "hrv_sdnn": np.nan,
        "hrv_rmssd": np.nan,
        "hrv_pnn50": np.nan,
        "hrv_pnn20": np.nan,
        "hrv_sdsd": np.nan,
        "hrv_mean_nn": np.nan,
        "hrv_median_nn": np.nan,
        "hrv_cv_nn": np.nan,
        "hrv_cvsd": np.nan,
        "hrv_hti": np.nan,
        "hrv_tinn": np.nan,
        
        # Frequency-Domain HRV
        "hrv_vlf_power": np.nan,
        "hrv_lf_power": np.nan,
        "hrv_hf_power": np.nan,
        "hrv_total_power": np.nan,
        "hrv_lf_hf_ratio": np.nan,
        "hrv_lf_norm": np.nan,
        "hrv_hf_norm": np.nan,
        "hrv_lf_peak_hz": np.nan,
        "hrv_hf_peak_hz": np.nan,
        
        # Nonlinear HRV
        "hrv_sd1": np.nan,
        "hrv_sd2": np.nan,
        "hrv_sd1_sd2_ratio": np.nan,
        "hrv_sample_entropy": np.nan,
        "hrv_approximate_entropy": np.nan,
        "hrv_dfa_alpha1": np.nan,
        "hrv_dfa_alpha2": np.nan,
        "hrv_csi": np.nan,
        "hrv_cvi": np.nan,
        
        # RR-Artefakte
        "rr_total_count": 0,
        "rr_valid_count": 0,
        "rr_artifact_count": 0,
        "rr_artifact_pct": 100.0,
        "rr_longest_valid_run": 0,
        "rr_longest_artifact_run": 0,
        
        # RSA
        "rsa_peak_valley": np.nan,
        "rsa_p2t_mean": np.nan,
        "rsa_p2t_std": np.nan,
        "rsa_gates_mean": np.nan,
        "rsa_resp_rate": np.nan,
        "rsa_coupling_strength": np.nan,
    }
    
    return features


# ==============================================================================
# BATCH-VERARBEITUNG: ALLE SEGMENTE EINES PATIENTEN
# ==============================================================================

def extract_ecg_features_all_segments(
    ecg_segments: list,
    resp_segments: Optional[Dict] = None,
    logger=None
) -> pd.DataFrame:
    """
    Extrahiert ECG-Features für alle Segmente eines Patienten.
    
    Parameters
    ----------
    ecg_segments : list
        Liste von SignalSegment-Objekten (aus segment_signals.py).
    resp_segments : Dict, optional
        Dictionary mit Respirations-Segmenten.
        Keys: Kanalnamen, Values: Liste von SignalSegment-Objekten.
    logger : loguru.Logger, optional
    
    Returns
    -------
    pd.DataFrame
        Feature-Tabelle mit einer Zeile pro Segment.
    """
    all_features = []
    
    # Wähle den besten Respirationskanal (falls vorhanden)
    best_resp_segments = None
    if resp_segments:
        # Nimm den Kanal mit den meisten gültigen Segmenten
        best_resp_name = max(
            resp_segments.keys(),
            key=lambda k: sum(1 for s in resp_segments[k] if s is not None)
        )
        best_resp_segments = resp_segments[best_resp_name]
        if logger:
            logger.info(f"RSA: Verwende Respirationskanal '{best_resp_name}'")
    
    for i, seg in enumerate(ecg_segments):
        if seg is None:
            features = _get_empty_ecg_features(segment_idx=i)
            all_features.append(features)
            continue
        
        # Respirationssignal für dieses Segment
        resp_data = None
        resp_fs = None
        if best_resp_segments and i < len(best_resp_segments):
            resp_seg = best_resp_segments[i]
            if resp_seg is not None:
                resp_data = resp_seg.data
                resp_fs = resp_seg.fs
        
        # Features extrahieren
        features = extract_ecg_features_segment(
            segment_data=seg.data,
            fs=seg.fs,
            rpeaks_local=getattr(seg, 'rpeaks_local', None),
            rr_intervals=getattr(seg, 'rr_intervals', None),
            rr_valid_mask=getattr(seg, 'rr_valid_mask', None),
            sqi=seg.sqi,
            segment_idx=seg.segment_idx,
            resp_signal=resp_data,
            resp_fs=resp_fs,
            logger=logger,
        )
        
        all_features.append(features)
    
    df = pd.DataFrame(all_features)
    
    if logger:
        n_computed = df["ecg_quality_ok"].sum()
        n_total = len(df)
        logger.info(f"ECG Features: {n_computed}/{n_total} Segmente mit "
                     f"vollständiger HRV-Berechnung")
        
        # Zusammenfassung der wichtigsten Features
        if n_computed > 0:
            valid = df[df["ecg_quality_ok"]]
            logger.info(f"  HR mean: {valid['hr_mean'].mean():.1f} ± "
                         f"{valid['hr_mean'].std():.1f} bpm")
            logger.info(f"  SDNN mean: {valid['hrv_sdnn'].mean():.1f} ± "
                         f"{valid['hrv_sdnn'].std():.1f} ms")
            logger.info(f"  RMSSD mean: {valid['hrv_rmssd'].mean():.1f} ± "
                         f"{valid['hrv_rmssd'].std():.1f} ms")
            
            if RSA_ENABLED and not valid["rsa_p2t_mean"].isna().all():
                n_rsa = valid["rsa_p2t_mean"].notna().sum()
                logger.info(f"  RSA berechnet für {n_rsa}/{n_computed} Segmente")
    
    return df
