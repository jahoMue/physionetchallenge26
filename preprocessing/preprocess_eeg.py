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
import neurokit2 as nk
from scipy import signal as scipy_signal
from typing import Dict, List, Optional, Tuple, Set

from config import (
    EEG_FILTER, EEG_SQI_THRESHOLD, EEG_AMPLITUDE_MAX_UV,
    EEG_AMPLITUDE_MIN_UV, EEG_CORRELATION_THRESHOLD,
    EEG_HOMOLOG_PAIRS, EEG_FREQUENCY_BANDS,
    SEGMENT_LENGTH_SEC
)


# ==============================================================================
# NEU: Globale Bad-Channel-Erkennung über alle Kanäle
# ==============================================================================

def detect_bad_channels(
    raw_signals: Dict[str, np.ndarray],
    fs: float,
    bad_threshold: float = 0.5,
    distance_threshold: float = 0.99,
    logger=None
) -> Set[str]:

    bad_channels = set()
    channel_names = list(raw_signals.keys())

    if len(raw_signals) < 2:
        if logger:
            logger.info(
                f"nk.eeg_badchannels übersprungen: Nur {len(raw_signals)} Kanal/Kanäle "
                f"verfügbar (mindestens 2 benötigt für Vergleich)."
            )
        return bad_channels

    try:
        min_len = min(len(sig) for sig in raw_signals.values())
        
        scale_factor = _detect_eeg_unit_scale(raw_signals, logger)
        
        max_samples = int(60 * fs)
        if min_len > max_samples:
            start = (min_len - max_samples) // 2
            end = start + max_samples
        else:
            start = 0
            end = min_len
        
        # ============================================================
        # FIX: nk.eeg_badchannels v0.2.10 erwartet (n_channels, n_samples)
        # Zeile 67: channel = eeg[i, :] → i = Kanal-Index
        # ============================================================
        eeg_array = np.column_stack([
            raw_signals[name][start:end] * scale_factor 
            for name in channel_names
        ]).T  # Shape: (n_channels, n_samples)

        if logger:
            logger.info(
                f"nk.eeg_badchannels: Prüfe {len(raw_signals)} Kanäle "
                f"({channel_names}), shape={eeg_array.shape} "
                f"(n_channels x n_samples), scale_factor={scale_factor:.0f}"
            )
            logger.debug(
                f"  Nach Skalierung: std pro Kanal = "
                f"{[f'{np.std(eeg_array[i, :]):.2f}' for i in range(eeg_array.shape[0])]}"
            )

        import traceback
        try:
            result = nk.eeg_badchannels(
                eeg_array,
                bad_threshold=bad_threshold,
                distance_threshold=distance_threshold,
                show=False
            )

            if logger:
                logger.info(f"nk.eeg_badchannels Ergebnis: type={type(result)}, value={result}")

        except Exception as inner_e:
            if logger:
                logger.warning(f"nk.eeg_badchannels INNER Exception: {type(inner_e).__name__}: {inner_e}")
                logger.warning(f"FULL Traceback:\n{traceback.format_exc()}")
            raise

        # Ergebnis verarbeiten
        if isinstance(result, (list, np.ndarray)):
            result = np.asarray(result)

            if result.dtype == bool or result.dtype == np.bool_:
                for i, is_bad in enumerate(result):
                    if is_bad and i < len(channel_names):
                        bad_channels.add(channel_names[i])

            elif np.issubdtype(result.dtype, np.integer):
                for idx in result:
                    if 0 <= idx < len(channel_names):
                        bad_channels.add(channel_names[idx])

            elif np.issubdtype(result.dtype, np.str_):
                bad_channels = set(result)

        if bad_channels:
            if logger:
                logger.warning(
                    f"nk.eeg_badchannels: Schlechte Kanäle erkannt: {bad_channels}"
                )
        else:
            if logger:
                logger.info("nk.eeg_badchannels: Alle Kanäle OK.")

    except Exception as e:
        if logger:
            logger.warning(f"nk.eeg_badchannels fehlgeschlagen: {e}")
        bad_channels = _fallback_bad_channel_detection(raw_signals, fs, logger)

    return bad_channels



def _detect_eeg_unit_scale(
    raw_signals: Dict[str, np.ndarray],
    logger=None
) -> float:
    """
    Erkennt die Einheit der EEG-Signale und gibt den Skalierungsfaktor
    zurück, um auf Mikrovolt (µV) zu konvertieren.
    
    Typische EEG-Amplituden: 10-100 µV (std ~20-50 µV)
    
    Heuristik basierend auf Standardabweichung:
    - std ~ 1e-5 bis 1e-4  → Volt      → Faktor 1e6
    - std ~ 1e-2 bis 1e-1  → Millivolt → Faktor 1e3
    - std ~ 10 bis 100     → Mikrovolt → Faktor 1
    - std ~ 1e4 bis 1e5    → Nanovolt  → Faktor 1e-3
    
    Returns
    -------
    float
        Skalierungsfaktor um auf µV zu konvertieren.
    """
    # Berechne mittlere Standardabweichung über alle Kanäle
    stds = []
    for name, sig in raw_signals.items():
        # Nehme nur einen Ausschnitt für Effizienz
        sample = sig[:min(len(sig), 100000)]
        stds.append(np.std(sample))
    
    mean_std = np.mean(stds)
    
    if mean_std == 0:
        if logger:
            logger.warning("EEG Einheiten-Erkennung: std=0, nehme Faktor 1")
        return 1.0
    
    # Bestimme Skalierungsfaktor
    if mean_std < 1e-3:
        # Wahrscheinlich Volt (std ~ 1e-5 für typisches EEG)
        scale = 1e6
        unit = "V"
    elif mean_std < 1.0:
        # Wahrscheinlich Millivolt
        scale = 1e3
        unit = "mV"
    elif mean_std < 1000:
        # Wahrscheinlich bereits Mikrovolt
        scale = 1.0
        unit = "µV"
    else:
        # Unbekannt, nehme µV an
        scale = 1.0
        unit = "µV (angenommen)"
    
    if logger:
        logger.info(
            f"EEG Einheiten-Erkennung: mean_std={mean_std:.6e} → "
            f"vermutlich {unit}, Skalierungsfaktor={scale:.0f} "
            f"(→ std nach Skalierung: {mean_std * scale:.1f} µV)"
        )
    
    return scale



def _fallback_bad_channel_detection(
    raw_signals: Dict[str, np.ndarray],
    fs: float,
    logger=None
) -> Set[str]:
    """
    Fallback-Erkennung schlechter Kanäle, falls nk.eeg_badchannels fehlschlägt.
    """
    bad_channels = set()
    
    if logger:
        logger.info("Verwende Fallback Bad-Channel-Erkennung...")
    
    # Einheiten-Skalierung ermitteln
    scale_factor = _detect_eeg_unit_scale(raw_signals, logger)
    
    for ch_name, sig in raw_signals.items():
        is_bad = False
        reasons = []
        
        # Skaliere auf µV für Schwellenwert-Vergleiche
        sig_uv = sig * scale_factor
        
        # 1. Flatliner-Check: Standardabweichung zu niedrig
        std_val = np.std(sig_uv)
        if std_val < EEG_AMPLITUDE_MIN_UV:
            is_bad = True
            reasons.append(f"Flatliner (std={std_val:.3f} µV)")
        
        # 2. Amplituden-Check: Zu viele Samples außerhalb des Bereichs
        abs_amp = np.abs(sig_uv)
        pct_over = np.mean(abs_amp > EEG_AMPLITUDE_MAX_UV)
        if pct_over > 0.3:
            is_bad = True
            reasons.append(f"Amplitude zu hoch ({pct_over*100:.1f}% über {EEG_AMPLITUDE_MAX_UV}µV)")
        
        # 3. Korrelation mit anderen Kanälen (einheitenunabhängig)
        if len(raw_signals) >= 2:
            correlations = []
            for other_name, other_sig in raw_signals.items():
                if other_name == ch_name:
                    continue
                min_len = min(len(sig), len(other_sig))
                if min_len > 100:
                    corr = np.corrcoef(sig[:min_len], other_sig[:min_len])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            if correlations:
                mean_corr = np.mean(correlations)
                if mean_corr < 0.1:
                    is_bad = True
                    reasons.append(f"Niedrige Korrelation (mean={mean_corr:.3f})")
        
        if is_bad:
            bad_channels.add(ch_name)
            if logger:
                logger.warning(f"Fallback: [{ch_name}] als schlecht markiert: {', '.join(reasons)}")
        else:
            if logger:
                logger.debug(f"Fallback: [{ch_name}] OK (std={std_val:.1f} µV)")
    
    return bad_channels



# ==============================================================================
# Einzelkanal-Preprocessing (OHNE nk.eeg_badchannels)
# ==============================================================================

def preprocess_eeg_signal(
    eeg_raw: np.ndarray,
    fs: float,
    channel_name: str = "unknown",
    is_globally_bad: bool = False,
    scale_to_uv: float = None,  # NEU: Skalierungsfaktor
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
    is_globally_bad : bool
        Ob der Kanal von detect_bad_channels() als schlecht markiert wurde.
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
        "info": {
            "channel_name": channel_name,
            "is_globally_bad": is_globally_bad,
        }
    }
    
    if logger:
        logger.info(f"EEG Preprocessing [{channel_name}]: {len(eeg_raw)} samples, "
                     f"fs={fs} Hz, duration={len(eeg_raw)/fs:.1f}s"
                     f"{' [GLOBAL BAD]' if is_globally_bad else ''}")
    # ------------------------------------------------------------------
    # 0. Einheiten-Konvertierung (VOR Filterung)
    # ------------------------------------------------------------------
    if scale_to_uv is not None and scale_to_uv != 1.0:
        eeg_raw = eeg_raw * scale_to_uv
        if logger:
            logger.debug(f"EEG [{channel_name}] skaliert mit Faktor {scale_to_uv:.0f} auf µV")
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
    if is_globally_bad:
        # Wenn global schlecht, alle Segmente als schlecht markieren
        n_segments = int(np.ceil(len(eeg_filtered) / (SEGMENT_LENGTH_SEC * fs)))
        sqi_records = []
        for seg_idx in range(n_segments):
            sqi_records.append({
                "segment_idx": seg_idx,
                "start_sec": seg_idx * SEGMENT_LENGTH_SEC,
                "end_sec": (seg_idx + 1) * SEGMENT_LENGTH_SEC,
                "sqi": 0.0, "amp_sqi": 0.0, "flat_sqi": 0.0, "spec_sqi": 0.0,
                "channel": channel_name,
            })
        sqi_df = pd.DataFrame(sqi_records)
        if logger:
            logger.warning(f"EEG [{channel_name}] als global schlecht markiert – "
                           f"alle {n_segments} Segmente SQI=0.0")
    else:
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
    Entscheidet für jeden verfügbaren EEG-Kanal, ob er verwendet wird.
    Anstatt homologe Paare zu mitteln, wird jeder Kanal einzeln behandelt,
    was für schlafmedizinische Analysen sinnvoller ist.
    
    Parameters
    ----------
    available_channels : Dict[str, Dict]
        Dictionary mit Kanalnamen als Keys und preprocessing-Ergebnissen als Values.
        z.B. {"C3-M2": {...}, "C4-M1": {...}}
    
    Returns
    -------
    Dict[str, Dict]
        Strategie pro Kanal mit:
        - strategy: "single" oder "none"
        - channels_used: Liste der verwendeten Kanalnamen
        - signal: Das Signal des Kanals
        - sqi_per_segment: SQI-Daten des Kanals
        - quality_mask: Quality-Maske des Kanals
    """
    strategies = {}

    for channel_name, channel_data in available_channels.items():
        is_ok = (
            channel_data is not None and
            channel_data.get("eeg_cleaned") is not None and
            channel_data.get("quality_mask") is not None and
            channel_data["quality_mask"].mean() > 0.1  # Mindestens 10% gute Segmente
        )

        if is_ok:
            strategies[channel_name] = {
                "strategy": "single",
                "channels_used": [channel_name],
                "signal": channel_data["eeg_cleaned"],
                "sqi_per_segment": channel_data["sqi_per_segment"],
                "quality_mask": channel_data["quality_mask"],
            }
            if logger:
                logger.info(f"EEG [{channel_name}]: Wird als Einzelkanal verwendet.")
        else:
            strategies[channel_name] = {
                "strategy": "none",
                "channels_used": [],
                "signal": None,
                "sqi_per_segment": None,
                "quality_mask": None,
            }
            if logger:
                logger.warning(f"EEG [{channel_name}]: Kanal wird nicht verwendet (Qualität zu schlecht oder nicht vorhanden).")

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
