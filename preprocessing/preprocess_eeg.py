"""
01_preprocessing/preprocess_eeg.py
===================================
EEG-Vorverarbeitung:
- Bandpassfilterung + Notch-Filter
- Artefakterkennung (Amplitude, Flatliner)
- Signal Quality Index pro Kanal
- Kanalauswahl-Strategie (beide mitteln, einzeln, oder verwerfen)
"""

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from typing import Dict, List, Optional, Tuple

from config import (
    EEG_FILTER, EEG_SQI_THRESHOLD, EEG_AMPLITUDE_MAX_UV,
    EEG_AMPLITUDE_MIN_UV, EEG_CORRELATION_THRESHOLD,
    EEG_HOMOLOG_PAIRS, EEG_FREQUENCY_BANDS,
    SEGMENT_LENGTH_SEC
)


def preprocess_eeg_signal(
    eeg_raw: np.ndarray,
    fs: float,
    channel_name: str = "unknown",
    logger=None
) -> Dict:
    """
    Vorverarbeitung eines einzelnen EEG-Kanals.
    
    Parameters
    ----------
    eeg_raw : np.ndarray
        Rohes EEG-Signal.
    fs : float
        Sampling-Rate in Hz.
    channel_name : str
        Name des Kanals.
    logger : loguru.Logger, optional
    
    Returns
    -------
    Dict mit:
        - eeg_cleaned: Gefiltertes Signal
        - sqi_per_segment: SQI pro Segment
        - quality_mask: Boolean-Maske pro Segment
        - info: Zusätzliche Informationen
    """
    result = {
        "eeg_cleaned": None,
        "sqi_per_segment": None,
        "quality_mask": None,
        "info": {"channel_name": channel_name}
    }
    
    if logger:
        logger.info(f"EEG Preprocessing [{channel_name}]: {len(eeg_raw)} samples, "
                     f"fs={fs} Hz, duration={len(eeg_raw)/fs:.1f}s")
    
    # ------------------------------------------------------------------
    # 1. Bandpassfilterung
    # ------------------------------------------------------------------
    try:
        eeg_filtered = _bandpass_filter(
            eeg_raw, fs,
            lowcut=EEG_FILTER["lowcut"],
            highcut=EEG_FILTER["highcut"],
            order=EEG_FILTER["order"]
        )
        
        # Notch-Filter (Netzfrequenz)
        if EEG_FILTER.get("notch"):
            eeg_filtered = _notch_filter(
                eeg_filtered, fs,
                freq=EEG_FILTER["notch"]
            )
        
        result["eeg_cleaned"] = eeg_filtered
        
        if logger:
            logger.debug(f"EEG [{channel_name}] Filterung erfolgreich.")
    except Exception as e:
        if logger:
            logger.error(f"EEG [{channel_name}] Filterung fehlgeschlagen: {e}")
        return result
    
    # ------------------------------------------------------------------
    # 2. SQI pro Segment
    # ------------------------------------------------------------------
    sqi_df = compute_eeg_sqi_segments(
        eeg_filtered, fs,
        segment_length_sec=SEGMENT_LENGTH_SEC,
        channel_name=channel_name,
        logger=logger
    )
    result["sqi_per_segment"] = sqi_df
    
    quality_mask = np.array([
        sqi >= EEG_SQI_THRESHOLD for sqi in sqi_df["sqi"]
    ])
    result["quality_mask"] = quality_mask
    
    if logger:
        n_good = quality_mask.sum()
        n_total = len(quality_mask)
        logger.info(f"EEG [{channel_name}] Qualität: {n_good}/{n_total} Segmente "
                     f"über SQI-Schwelle ({EEG_SQI_THRESHOLD}): "
                     f"{n_good/n_total*100:.1f}%")
    
    return result


def decide_eeg_channel_strategy(
    available_channels: Dict[str, Dict],
    logger=None
) -> Dict[str, Dict]:
    """
    Entscheidet für jedes homologe Kanalpaar die Strategie:
    - "averaged": Beide Kanäle verfügbar und gute Qualität -> Mitteln
    - "single_left" / "single_right": Nur ein Kanal nutzbar
    - "none": Kein Kanal nutzbar
    
    Parameters
    ----------
    available_channels : Dict[str, Dict]
        Dictionary mit Kanalnamen als Keys und preprocessing-Ergebnissen als Values.
        z.B. {"F3-M2": {...}, "F4-M1": {...}, "C3-M2": {...}}
    
    Returns
    -------
    Dict[str, Dict]
        Strategie pro Region mit:
        - strategy: "averaged", "single_left", "single_right", "none"
        - channels_used: Liste der verwendeten Kanalnamen
        - signal: Das resultierende Signal (gemittelt oder einzeln)
        - sqi: Kombinierter SQI
        - quality_mask: Kombinierte Quality-Maske
    """
    strategies = {}
    
    for region, (left_name, right_name) in EEG_HOMOLOG_PAIRS.items():
        left_data = available_channels.get(left_name)
        right_data = available_channels.get(right_name)
        
        left_ok = (
            left_data is not None and 
            left_data.get("eeg_cleaned") is not None and
            left_data.get("quality_mask") is not None and
            left_data["quality_mask"].mean() > 0.3  # Mindestens 30% gute Segmente
        )
        
        right_ok = (
            right_data is not None and 
            right_data.get("eeg_cleaned") is not None and
            right_data.get("quality_mask") is not None and
            right_data["quality_mask"].mean() > 0.3
        )
        
        if left_ok and right_ok:
            # Beide verfügbar -> Prüfe Korrelation
            corr = _compute_channel_correlation(
                left_data["eeg_cleaned"],
                right_data["eeg_cleaned"]
            )
            
            if corr >= EEG_CORRELATION_THRESHOLD:
                # Gute Korrelation -> Mitteln
                averaged_signal = (
                    left_data["eeg_cleaned"] + right_data["eeg_cleaned"]
                ) / 2.0
                
                # Kombinierte Quality-Maske: Beide müssen gut sein
                combined_mask = left_data["quality_mask"] & right_data["quality_mask"]
                
                # Kombinierter SQI: Mittelwert
                combined_sqi = left_data["sqi_per_segment"].copy()
                combined_sqi["sqi"] = (
                    left_data["sqi_per_segment"]["sqi"].values + 
                    right_data["sqi_per_segment"]["sqi"].values
                ) / 2.0
                
                strategies[region] = {
                    "strategy": "averaged",
                    "channels_used": [left_name, right_name],
                    "signal": averaged_signal,
                    "sqi_per_segment": combined_sqi,
                    "quality_mask": combined_mask,
                    "correlation": corr,
                }
                
                if logger:
                    logger.info(f"EEG [{region}]: AVERAGED ({left_name} + {right_name}), "
                                f"correlation={corr:.3f}")
            else:
                # Schlechte Korrelation -> Nutze den besseren Kanal
                left_mean_sqi = left_data["sqi_per_segment"]["sqi"].mean()
                right_mean_sqi = right_data["sqi_per_segment"]["sqi"].mean()
                
                if left_mean_sqi >= right_mean_sqi:
                    best = left_data
                    best_name = left_name
                    strategy_name = "single_left"
                else:
                    best = right_data
                    best_name = right_name
                    strategy_name = "single_right"
                
                strategies[region] = {
                    "strategy": strategy_name,
                    "channels_used": [best_name],
                    "signal": best["eeg_cleaned"],
                    "sqi_per_segment": best["sqi_per_segment"],
                    "quality_mask": best["quality_mask"],
                    "correlation": corr,
                }
                
                if logger:
                    logger.warning(
                        f"EEG [{region}]: LOW CORRELATION ({corr:.3f}), "
                        f"using {strategy_name} ({best_name}, "
                        f"mean SQI={best['sqi_per_segment']['sqi'].mean():.3f})"
                    )
        
        elif left_ok:
            strategies[region] = {
                "strategy": "single_left",
                "channels_used": [left_name],
                "signal": left_data["eeg_cleaned"],
                "sqi_per_segment": left_data["sqi_per_segment"],
                "quality_mask": left_data["quality_mask"],
                "correlation": None,
            }
            if logger:
                logger.info(f"EEG [{region}]: SINGLE ({left_name} only)")
        
        elif right_ok:
            strategies[region] = {
                "strategy": "single_right",
                "channels_used": [right_name],
                "signal": right_data["eeg_cleaned"],
                "sqi_per_segment": right_data["sqi_per_segment"],
                "quality_mask": right_data["quality_mask"],
                "correlation": None,
            }
            if logger:
                logger.info(f"EEG [{region}]: SINGLE ({right_name} only)")
        
        else:
            strategies[region] = {
                "strategy": "none",
                "channels_used": [],
                "signal": None,
                "sqi_per_segment": None,
                "quality_mask": None,
                "correlation": None,
            }
            if logger:
                logger.warning(f"EEG [{region}]: NONE – kein nutzbarer Kanal verfügbar.")
    
    return strategies


def compute_eeg_sqi_segments(
    eeg_signal: np.ndarray,
    fs: float,
    segment_length_sec: float = 300,
    channel_name: str = "unknown",
    logger=None
) -> pd.DataFrame:
    """
    Berechnet EEG Signal Quality Index pro Segment.
    
    Metriken:
    1. Amplituden-SQI: Anteil der Samples innerhalb physiologischer Grenzen
    2. Flatliner-SQI: Erkennung von Null-Varianz-Abschnitten
    3. Spektral-SQI: Prüfung ob typische EEG-Frequenzen vorhanden sind
    """
    segment_length_samples = int(segment_length_sec * fs)
    n_segments = int(np.ceil(len(eeg_signal) / segment_length_samples))
    
    sqi_records = []
    
    for seg_idx in range(n_segments):
        start_sample = seg_idx * segment_length_samples
        end_sample = min(start_sample + segment_length_samples, len(eeg_signal))
        start_sec = start_sample / fs
        end_sec = end_sample / fs
        
        seg = eeg_signal[start_sample:end_sample]
        
        # --- Amplituden-SQI ---
        amp_sqi = _amplitude_sqi(seg)
        
        # --- Flatliner-SQI ---
        flat_sqi = _flatliner_sqi(seg, fs)
        
        # --- Spektral-SQI ---
        spec_sqi = _spectral_sqi(seg, fs)
        
        # --- Kombinierter SQI ---
        combined_sqi = 0.35 * amp_sqi + 0.30 * flat_sqi + 0.35 * spec_sqi
        combined_sqi = np.clip(combined_sqi, 0.0, 1.0)
        
        sqi_records.append({
            "segment_idx": seg_idx,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "sqi": combined_sqi,
            "amp_sqi": amp_sqi,
            "flat_sqi": flat_sqi,
            "spec_sqi": spec_sqi,
            "channel": channel_name,
        })
    
    return pd.DataFrame(sqi_records)


def _bandpass_filter(
    signal_data: np.ndarray, 
    fs: float, 
    lowcut: float, 
    highcut: float, 
    order: int = 4
) -> np.ndarray:
    """Butterworth Bandpassfilter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Sicherheitscheck
    low = max(low, 0.001)
    high = min(high, 0.999)
    
    b, a = scipy_signal.butter(order, [low, high], btype='band')
    filtered = scipy_signal.filtfilt(b, a, signal_data)
    return filtered


def _notch_filter(
    signal_data: np.ndarray, 
    fs: float, 
    freq: float = 50.0, 
    Q: float = 30.0
) -> np.ndarray:
    """Notch-Filter für Netzfrequenz."""
    nyq = 0.5 * fs
    if freq >= nyq:
        return signal_data  # Notch-Frequenz über Nyquist
    
    b, a = scipy_signal.iirnotch(freq, Q, fs)
    filtered = scipy_signal.filtfilt(b, a, signal_data)
    return filtered


def _amplitude_sqi(segment: np.ndarray) -> float:
    """Anteil der Samples innerhalb physiologischer Amplitude."""
    if len(segment) == 0:
        return 0.0
    
    abs_amp = np.abs(segment)
    within_range = (abs_amp <= EEG_AMPLITUDE_MAX_UV) & (abs_amp >= EEG_AMPLITUDE_MIN_UV * 0.1)
    return np.mean(within_range)


def _flatliner_sqi(segment: np.ndarray, fs: float, window_sec: float = 1.0) -> float:
    """Erkennung von Flatliner-Abschnitten (Null-Varianz)."""
    if len(segment) == 0:
        return 0.0
    
    window_samples = int(window_sec * fs)
    if window_samples < 2:
        return 1.0
    
    n_windows = len(segment) // window_samples
    if n_windows == 0:
        return 1.0
    
    flat_count = 0
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        window = segment[start:end]
        if np.std(window) < EEG_AMPLITUDE_MIN_UV:
            flat_count += 1
    
    flat_ratio = flat_count / n_windows
    return 1.0 - flat_ratio  # Hoher Wert = wenig Flatliner


def _spectral_sqi(segment: np.ndarray, fs: float) -> float:
    """
    Prüft ob typische EEG-Frequenzen vorhanden sind.
    Ein gutes EEG sollte Aktivität in Delta-Beta haben, 
    nicht nur Rauschen oder Artefakte.
    """
    if len(segment) < int(fs * 2):
        return 0.0
    
    try:
        freqs, psd = scipy_signal.welch(segment, fs=fs, nperseg=min(len(segment), int(fs * 4)))
        
        # Gesamtleistung im EEG-Bereich (0.5-30 Hz)
        eeg_mask = (freqs >= 0.5) & (freqs <= 30)
        eeg_power = np.trapz(psd[eeg_mask], freqs[eeg_mask])
        
        # Gesamtleistung
        total_power = np.trapz(psd, freqs)
        
        if total_power == 0:
            return 0.0
        
        # Anteil der EEG-Leistung an Gesamtleistung
        eeg_ratio = eeg_power / total_power
        
        # Prüfe ob mehrere Bänder aktiv sind (nicht nur ein Artefakt-Peak)
        band_powers = {}
        for band_name, (fmin, fmax) in EEG_FREQUENCY_BANDS.items():
            band_mask = (freqs >= fmin) & (freqs <= fmax)
            if band_mask.any():
                band_powers[band_name] = np.trapz(psd[band_mask], freqs[band_mask])
            else:
                band_powers[band_name] = 0
        
        # Entropie der Bandverteilung (höher = gleichmäßiger = besser)
        total_band = sum(band_powers.values())
        if total_band > 0:
            probs = np.array([p / total_band for p in band_powers.values()])
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(len(EEG_FREQUENCY_BANDS))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        else:
            norm_entropy = 0
        
        return 0.5 * eeg_ratio + 0.5 * norm_entropy
        
    except Exception:
        return 0.0


def _compute_channel_correlation(
    signal1: np.ndarray, 
    signal2: np.ndarray
) -> float:
    """Berechnet die Korrelation zwischen zwei Kanälen."""
    min_len = min(len(signal1), len(signal2))
    if min_len < 10:
        return 0.0
    
    s1 = signal1[:min_len]
    s2 = signal2[:min_len]
    
    corr = np.corrcoef(s1, s2)[0, 1]
    if np.isnan(corr):
        return 0.0
    
    return abs(corr)
