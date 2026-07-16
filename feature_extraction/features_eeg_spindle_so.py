"""
03_feature_extraction/features_eeg_spindle_so.py
==================================================
Discrete Spindle Detection, Slow Oscillation (SO) Characterization,
and SO-Spindle Coupling Features for Cognitive Impairment Prediction.

Operates on pre-processed, segmented EEG signals.
"""

import logging
import numpy as np
import warnings
from typing import Dict, Optional, Tuple

from scipy.signal import hilbert, butter, filtfilt, welch
from scipy.stats import circmean, circstd

from config import (
    EEG_SQI_THRESHOLD,
    SIGNAL_DTYPE,
    SPINDLE_FREQ_RANGE,
    SPINDLE_FREQ_SLOW,
    SPINDLE_FREQ_FAST,
    SPINDLE_DURATION_RANGE,
    SPINDLE_MIN_AMPLITUDE_UV,
    SO_FREQ_RANGE,
    SO_DURATION_RANGE,
    SO_MIN_AMPLITUDE_UV,
    COUPLING_ANALYSIS_ENABLED,
    COUPLING_MIN_SPINDLES,
    COUPLING_MIN_SOS,
    SPINDLE_DETECTION_ENABLED,
    SO_DETECTION_ENABLED,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# YASA LOGGER SUPPRESSION
# ==============================================================================

def _suppress_yasa_logger():
    """
    Suppress YASA's internal WARNING-level messages.
    
    Must be called inside each function that invokes YASA, because
    ProcessPoolExecutor spawns fresh processes where module-level
    suppression has not been applied.
    """
    _yasa_log = logging.getLogger("yasa")
    _yasa_log.setLevel(logging.ERROR)


# ==============================================================================
# SIGNAL QUALITY PRE-CHECKS
# ==============================================================================

def _has_sufficient_band_power(
    data: np.ndarray,
    fs: float,
    freq_range: Tuple[float, float],
    min_relative_power: float = 0.01,
) -> bool:
    """
    Check whether the signal has sufficient power in a frequency band
    relative to total power.
    """
    try:
        nperseg = min(len(data), int(fs * 4))
        if nperseg < 16:
            return False
        freqs, psd = welch(data, fs=fs, nperseg=nperseg)

        total_mask = (freqs >= 0.3) & (freqs <= 35.0)
        band_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])

        total_power = np.trapz(psd[total_mask], freqs[total_mask])
        band_power = np.trapz(psd[band_mask], freqs[band_mask])

        if total_power <= 0:
            return False

        return (band_power / total_power) >= min_relative_power
    except Exception:
        return False


# ==============================================================================
# PUBLIC API
# ==============================================================================

def extract_spindle_so_coupling_features(
    segment_data: np.ndarray,
    fs: float,
    region: str = "unknown",
    sqi: float = 0.0,
    segment_idx: int = 0,
    sleep_stage: str = "unknown",
    logger=None,
) -> Dict:
    """
    Extract all spindle, SO, and coupling features for one EEG segment.
    Single entry-point called by features_eeg.py.
    """
    _suppress_yasa_logger()

    prefix_sp = f"eeg_{region}_sp"
    prefix_so = f"eeg_{region}_so"
    prefix_coup = f"eeg_{region}_coup"

    features = {}
    if SPINDLE_DETECTION_ENABLED:
        features.update(_get_empty_spindle_features(prefix_sp))
    if SO_DETECTION_ENABLED:
        features.update(_get_empty_so_features(prefix_so))
    if SPINDLE_DETECTION_ENABLED and SO_DETECTION_ENABLED:
        features.update(_get_empty_coupling_features(prefix_coup))

    # Spindles and SOs only occur during NREM sleep (N1, N2, N3).
    # Skip calculations for Wake (W) and REM to speed up and reduce false positives.
    # If sleep_stage is "unknown" (e.g. annotation missing), run as fallback.
    if sleep_stage in ["W", "REM"]:
        return features

    if segment_data is None or len(segment_data) == 0:
        return features

    if sqi < EEG_SQI_THRESHOLD:
        return features

    duration_sec = len(segment_data) / fs
    if duration_sec < 10.0:
        return features

    data_f64 = segment_data.astype(np.float64)

    # ==================================================================
    # DIAGNOSTIC LOGGING — Remove after debugging
    # ==================================================================
    if logger and segment_idx < 5:
        sig_std = np.std(data_f64)
        sig_ptp = np.ptp(data_f64)

        try:
            nperseg_d = min(len(data_f64), int(fs * 4))
            freqs_d, psd_d = welch(data_f64, fs=fs, nperseg=nperseg_d)

            sigma_mask = (freqs_d >= 12) & (freqs_d <= 16)
            so_mask = (freqs_d >= 0.3) & (freqs_d <= 1.5)
            total_mask = (freqs_d >= 0.3) & (freqs_d <= 35)

            sigma_power = np.trapz(psd_d[sigma_mask], freqs_d[sigma_mask])
            so_power = np.trapz(psd_d[so_mask], freqs_d[so_mask])
            total_power = np.trapz(psd_d[total_mask], freqs_d[total_mask])

            sigma_rel = sigma_power / total_power if total_power > 0 else 0
            so_rel = so_power / total_power if total_power > 0 else 0
        except Exception:
            sigma_rel = -1
            so_rel = -1
            sigma_power = -1
            so_power = -1

        logger.debug(
            f"[SPINDLE_DIAG] Seg {segment_idx} [{region}]: "
            f"std={sig_std:.2f}, ptp={sig_ptp:.2f}, "
            f"fs={fs:.1f}, dur={duration_sec:.1f}s, "
            f"sigma_rel={sigma_rel:.4f}, so_rel={so_rel:.4f}, "
            f"sigma_abs={sigma_power:.6f}, so_abs={so_power:.6f}"
        )
    # ==================================================================
    # END DIAGNOSTIC
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. SPINDLE DETECTION
    # ------------------------------------------------------------------
    sp_summary = None
    if SPINDLE_DETECTION_ENABLED:
        if _has_sufficient_band_power(data_f64, fs, SPINDLE_FREQ_RANGE, 0.005):
            try:
                sp_summary = _detect_spindles(data_f64, fs, logger)
            except Exception as e:
                if logger:
                    logger.debug(
                        f"Segment {segment_idx} [{region}]: Spindle detection failed: {e}"
                    )

        features.update(
            _compute_spindle_features(sp_summary, duration_sec, fs, prefix_sp)
        )

    # ------------------------------------------------------------------
    # 2. SLOW OSCILLATION DETECTION
    # ------------------------------------------------------------------
    so_summary = None
    if SO_DETECTION_ENABLED:
        if _has_sufficient_band_power(data_f64, fs, SO_FREQ_RANGE, 0.01):
            try:
                so_summary = _detect_slow_oscillations(data_f64, fs, logger)
            except Exception as e:
                if logger:
                    logger.debug(
                        f"Segment {segment_idx} [{region}]: SO detection failed: {e}"
                    )

        features.update(
            _compute_so_features(so_summary, duration_sec, prefix_so)
        )

    # ------------------------------------------------------------------
    # 3. SO-SPINDLE COUPLING
    # ------------------------------------------------------------------
    if SPINDLE_DETECTION_ENABLED and SO_DETECTION_ENABLED and COUPLING_ANALYSIS_ENABLED:
        try:
            features.update(
                _compute_coupling_features(
                    data_f64, fs, sp_summary, so_summary,
                    duration_sec, prefix_coup, logger,
                )
            )
        except Exception as e:
            if logger:
                logger.debug(
                    f"Segment {segment_idx} [{region}]: "
                    f"Coupling analysis failed: {e}"
                )

    return features


# ==============================================================================
# SPINDLE DETECTION — REPLACE THE EXISTING FUNCTION  
# (only change: ensure _suppress_yasa_logger is called)
# ==============================================================================

def _detect_spindles(data: np.ndarray, fs: float, logger=None):
    """
    Detect sleep spindles using YASA's ``spindles_detect``.
    """
    _suppress_yasa_logger()
    import yasa

    sp = yasa.spindles_detect(
        data,
        sf=fs,
        freq_sp=SPINDLE_FREQ_RANGE,
        freq_broad=(0.5, 30.0),
        duration=SPINDLE_DURATION_RANGE,
        thresh={
            "rel_pow": 0.15,
            "corr": 0.60,
            "rms": 1.2,
        },
        multi_only=False,
        remove_outliers=True,
        verbose='critical',
    )

    if sp is None:
        return None

    summary = sp.summary()
    if summary is None or len(summary) == 0:
        return None

    return summary



# ==============================================================================
# SLOW OSCILLATION DETECTION — REPLACE THE EXISTING FUNCTION
# ==============================================================================

def _detect_slow_oscillations(data: np.ndarray, fs: float, logger=None):
    """
    Detect slow oscillations using YASA's ``sw_detect``.

    The input signal is already bandpass-filtered at 0.3 Hz (highpass)
    with a 4th-order Butterworth. This significantly attenuates SO
    amplitudes, especially in the 0.3-0.5 Hz range. A real 80 µV PTP
    slow oscillation may appear as only ~50 µV after this filtering.
    
    Thresholds are therefore set very permissively to capture SOs
    that have been amplitude-reduced by the pre-filter.
    """
    _suppress_yasa_logger()
    import yasa

    sw = yasa.sw_detect(
        data,
        sf=fs,
        freq_sw=SO_FREQ_RANGE,
        dur_neg=(0.3, 1.5),
        dur_pos=(0.1, 1.5),
        amp_neg=(10, 400),         # FURTHER RELAXED from (20, 300)
        amp_pos=(5, 250),          # FURTHER RELAXED from (5, 200)
        amp_ptp=(20, 500),         # FURTHER RELAXED from (40, 500) — key change
        coupling=False,
        remove_outliers=True,
        verbose=False,
    )

    if sw is None:
        return None

    summary = sw.summary()
    if summary is None or len(summary) == 0:
        return None

    return summary



# ==============================================================================
# SPINDLE FEATURE COMPUTATION
# ==============================================================================

def _compute_spindle_features(sp_summary, duration_sec, fs, prefix) -> Dict:
    features = _get_empty_spindle_features(prefix)

    if sp_summary is None or len(sp_summary) == 0:
        features[f"{prefix}_count"] = 0
        features[f"{prefix}_density"] = 0.0
        return features

    n_spindles = len(sp_summary)
    duration_min = duration_sec / 60.0

    features[f"{prefix}_count"] = n_spindles
    features[f"{prefix}_density"] = (
        float(n_spindles / duration_min) if duration_min > 0 else 0.0
    )

    if "Duration" in sp_summary.columns:
        dur = sp_summary["Duration"].values
        features[f"{prefix}_duration_mean"] = float(np.nanmean(dur))
        features[f"{prefix}_duration_std"] = (
            float(np.nanstd(dur, ddof=1)) if n_spindles > 1 else 0.0
        )
        features[f"{prefix}_duration_median"] = float(np.nanmedian(dur))

    if "Amplitude" in sp_summary.columns:
        amp = sp_summary["Amplitude"].values
        features[f"{prefix}_amplitude_mean"] = float(np.nanmean(amp))
        features[f"{prefix}_amplitude_std"] = (
            float(np.nanstd(amp, ddof=1)) if n_spindles > 1 else 0.0
        )
        features[f"{prefix}_amplitude_median"] = float(np.nanmedian(amp))

    if "RMS" in sp_summary.columns:
        rms = sp_summary["RMS"].values
        features[f"{prefix}_rms_mean"] = float(np.nanmean(rms))

    if "Frequency" in sp_summary.columns:
        freq = sp_summary["Frequency"].values
        features[f"{prefix}_frequency_mean"] = float(np.nanmean(freq))
        features[f"{prefix}_frequency_std"] = (
            float(np.nanstd(freq, ddof=1)) if n_spindles > 1 else 0.0
        )

        slow_mask = (freq >= SPINDLE_FREQ_SLOW[0]) & (freq < SPINDLE_FREQ_SLOW[1])
        fast_mask = (freq >= SPINDLE_FREQ_FAST[0]) & (freq <= SPINDLE_FREQ_FAST[1])

        features[f"{prefix}_slow_count"] = int(slow_mask.sum())
        features[f"{prefix}_fast_count"] = int(fast_mask.sum())
        features[f"{prefix}_slow_density"] = (
            float(slow_mask.sum() / duration_min) if duration_min > 0 else 0.0
        )
        features[f"{prefix}_fast_density"] = (
            float(fast_mask.sum() / duration_min) if duration_min > 0 else 0.0
        )
        features[f"{prefix}_fast_slow_ratio"] = (
            float(fast_mask.sum() / slow_mask.sum())
            if slow_mask.sum() > 0 else np.nan
        )

    if "Symmetry" in sp_summary.columns:
        sym = sp_summary["Symmetry"].values
        features[f"{prefix}_symmetry_mean"] = float(np.nanmean(sym))

    if "RelPower" in sp_summary.columns:
        rp = sp_summary["RelPower"].values
        features[f"{prefix}_rel_power_mean"] = float(np.nanmean(rp))

    if "Oscillations" in sp_summary.columns:
        osc = sp_summary["Oscillations"].values
        features[f"{prefix}_oscillations_mean"] = float(np.nanmean(osc))

    return features


def _get_empty_spindle_features(prefix: str) -> Dict:
    keys = [
        "count", "density",
        "duration_mean", "duration_std", "duration_median",
        "amplitude_mean", "amplitude_std", "amplitude_median",
        "rms_mean",
        "frequency_mean", "frequency_std",
        "slow_count", "fast_count",
        "slow_density", "fast_density",
        "fast_slow_ratio",
        "symmetry_mean",
        "rel_power_mean",
        "oscillations_mean",
    ]
    return {f"{prefix}_{k}": np.nan for k in keys}


# ==============================================================================
# SLOW OSCILLATION FEATURE COMPUTATION
# ==============================================================================

def _compute_so_features(so_summary, duration_sec, prefix) -> Dict:
    features = _get_empty_so_features(prefix)

    if so_summary is None or len(so_summary) == 0:
        features[f"{prefix}_count"] = 0
        features[f"{prefix}_density"] = 0.0
        return features

    n_so = len(so_summary)
    duration_min = duration_sec / 60.0

    features[f"{prefix}_count"] = n_so
    features[f"{prefix}_density"] = (
        float(n_so / duration_min) if duration_min > 0 else 0.0
    )

    if "Duration" in so_summary.columns:
        dur = so_summary["Duration"].values
        features[f"{prefix}_duration_mean"] = float(np.nanmean(dur))
        features[f"{prefix}_duration_std"] = (
            float(np.nanstd(dur, ddof=1)) if n_so > 1 else 0.0
        )

    if "PTP" in so_summary.columns:
        ptp = so_summary["PTP"].values
        features[f"{prefix}_ptp_amplitude_mean"] = float(np.nanmean(ptp))
        features[f"{prefix}_ptp_amplitude_std"] = (
            float(np.nanstd(ptp, ddof=1)) if n_so > 1 else 0.0
        )
        features[f"{prefix}_ptp_amplitude_median"] = float(np.nanmedian(ptp))

    if "ValNegPeak" in so_summary.columns:
        neg = so_summary["ValNegPeak"].values
        features[f"{prefix}_neg_peak_mean"] = float(np.nanmean(neg))
        features[f"{prefix}_neg_peak_std"] = (
            float(np.nanstd(neg, ddof=1)) if n_so > 1 else 0.0
        )

    if "ValPosPeak" in so_summary.columns:
        pos = so_summary["ValPosPeak"].values
        features[f"{prefix}_pos_peak_mean"] = float(np.nanmean(pos))

    if "Frequency" in so_summary.columns:
        freq = so_summary["Frequency"].values
        features[f"{prefix}_frequency_mean"] = float(np.nanmean(freq))

    if "Slope" in so_summary.columns:
        slope = so_summary["Slope"].values
        features[f"{prefix}_slope_mean"] = float(np.nanmean(slope))
        features[f"{prefix}_slope_std"] = (
            float(np.nanstd(slope, ddof=1)) if n_so > 1 else 0.0
        )
    elif "ValNegPeak" in so_summary.columns and "Duration" in so_summary.columns:
        neg = np.abs(so_summary["ValNegPeak"].values)
        dur = so_summary["Duration"].values
        half_dur = dur / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(half_dur > 0, neg / half_dur, np.nan)
        features[f"{prefix}_slope_mean"] = float(np.nanmean(slope))
        features[f"{prefix}_slope_std"] = (
            float(np.nanstd(slope, ddof=1)) if n_so > 1 else 0.0
        )

    return features


def _get_empty_so_features(prefix: str) -> Dict:
    keys = [
        "count", "density",
        "duration_mean", "duration_std",
        "ptp_amplitude_mean", "ptp_amplitude_std", "ptp_amplitude_median",
        "neg_peak_mean", "neg_peak_std",
        "pos_peak_mean",
        "frequency_mean",
        "slope_mean", "slope_std",
    ]
    return {f"{prefix}_{k}": np.nan for k in keys}


# ==============================================================================
# SO-SPINDLE COUPLING
# ==============================================================================

def _compute_coupling_features(
    data, fs, sp_summary, so_summary, duration_sec, prefix, logger=None,
) -> Dict:
    features = _get_empty_coupling_features(prefix)

    if sp_summary is None or so_summary is None:
        return features
    if len(sp_summary) < COUPLING_MIN_SPINDLES:
        return features
    if len(so_summary) < COUPLING_MIN_SOS:
        return features

    try:
        so_phase = _extract_phase_signal(
            data, fs, SO_FREQ_RANGE[0], SO_FREQ_RANGE[1]
        )
    except Exception:
        return features

    try:
        sigma_env = _extract_amplitude_envelope(
            data, fs, SPINDLE_FREQ_RANGE[0], SPINDLE_FREQ_RANGE[1]
        )
    except Exception:
        sigma_env = None

    if "Peak" in sp_summary.columns:
        spindle_peak_samples = (sp_summary["Peak"].values * fs).astype(int)
    elif "PeakSample" in sp_summary.columns:
        spindle_peak_samples = sp_summary["PeakSample"].values.astype(int)
    elif "Start" in sp_summary.columns and "Duration" in sp_summary.columns:
        mid_sec = sp_summary["Start"].values + sp_summary["Duration"].values / 2
        spindle_peak_samples = (mid_sec * fs).astype(int)
    else:
        return features

    spindle_peak_samples = np.clip(spindle_peak_samples, 0, len(so_phase) - 1)
    coupling_phases = so_phase[spindle_peak_samples]

    valid_mask = ~np.isnan(coupling_phases)
    coupling_phases = coupling_phases[valid_mask]

    n_coupled = len(coupling_phases)
    features[f"{prefix}_count"] = n_coupled

    if n_coupled < 3:
        features[f"{prefix}_rate"] = (
            float(n_coupled / len(sp_summary)) if len(sp_summary) > 0 else 0.0
        )
        return features

    features[f"{prefix}_rate"] = float(n_coupled / len(sp_summary))

    mean_angle_rad = float(circmean(coupling_phases, high=np.pi, low=-np.pi))
    features[f"{prefix}_mean_phase_rad"] = mean_angle_rad
    features[f"{prefix}_mean_phase_deg"] = float(np.degrees(mean_angle_rad))

    mrl = float(np.abs(np.mean(np.exp(1j * coupling_phases))))
    features[f"{prefix}_mrl"] = mrl

    features[f"{prefix}_phase_std_rad"] = float(
        circstd(coupling_phases, high=np.pi, low=-np.pi)
    )

    z_rayleigh = n_coupled * mrl ** 2
    p_rayleigh = float(np.exp(-z_rayleigh))
    features[f"{prefix}_rayleigh_z"] = float(z_rayleigh)
    features[f"{prefix}_rayleigh_p"] = p_rayleigh

    features[f"{prefix}_preferred_phase_quadrant"] = _phase_to_quadrant(
        mean_angle_rad
    )

    if sigma_env is not None:
        try:
            mi = _modulation_index(so_phase, sigma_env, n_bins=18)
            features[f"{prefix}_pac_mi"] = float(mi)
        except Exception:
            pass

    return features


def _extract_phase_signal(data, fs, f_low, f_high):
    nyq = fs / 2.0
    low = max(f_low / nyq, 0.001)
    high = min(f_high / nyq, 0.999)
    order = 2 if f_high <= 2.0 else 4
    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, data)
    analytic = hilbert(filtered)
    return np.angle(analytic)


def _extract_amplitude_envelope(data, fs, f_low, f_high):
    nyq = fs / 2.0
    low = max(f_low / nyq, 0.001)
    high = min(f_high / nyq, 0.999)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, data)
    analytic = hilbert(filtered)
    return np.abs(analytic)


def _modulation_index(phase, amplitude, n_bins=18):
    if len(phase) != len(amplitude):
        min_len = min(len(phase), len(amplitude))
        phase = phase[:min_len]
        amplitude = amplitude[:min_len]

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_indices = np.digitize(phase, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_amp = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.any():
            mean_amp[b] = np.mean(amplitude[mask])

    total = mean_amp.sum()
    if total == 0:
        return 0.0
    p = mean_amp / total
    q = np.ones(n_bins) / n_bins

    eps = 1e-12
    p_safe = p + eps
    p_safe = p_safe / p_safe.sum()
    kl = np.sum(p_safe * np.log(p_safe / q))

    mi = kl / np.log(n_bins)
    return float(np.clip(mi, 0, 1))


def _phase_to_quadrant(angle_rad):
    if -np.pi / 2 <= angle_rad < 0:
        return 1.0
    elif 0 <= angle_rad < np.pi / 2:
        return 2.0
    elif np.pi / 2 <= angle_rad <= np.pi:
        return 3.0
    else:
        return 4.0


def _get_empty_coupling_features(prefix: str) -> Dict:
    keys = [
        "count", "rate",
        "mean_phase_rad", "mean_phase_deg",
        "mrl",
        "phase_std_rad",
        "rayleigh_z", "rayleigh_p",
        "preferred_phase_quadrant",
        "pac_mi",
    ]
    return {f"{prefix}_{k}": np.nan for k in keys}
