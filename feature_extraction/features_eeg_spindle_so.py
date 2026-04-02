"""
03_feature_extraction/features_eeg_spindle_so.py
==================================================
Discrete Spindle Detection, Slow Oscillation (SO) Characterization,
and SO-Spindle Coupling Features for Cognitive Impairment Prediction.

This module operates on **pre-processed, segmented EEG signals** — the same
numpy arrays that features_eeg.py receives — and computes:

1. Spindle Features (YASA spindles_detect)
   - Density, amplitude, duration, frequency, RMS, symmetry
   - Slow vs. fast spindle sub-bands

2. Slow Oscillation Features (YASA sw_detect)
   - Density, amplitude, duration, slope, frequency
   - Negative/positive peak amplitudes

3. SO-Spindle Coupling Features
   - Coupling count & rate
   - Mean coupling phase angle (circular mean)
   - Coupling strength (Mean Resultant Length / MRL)
   - Phase-Amplitude Coupling (PAC) via Modulation Index

All features gracefully degrade to NaN when detection yields
insufficient events (controlled by config thresholds).

References
----------
- Helfrich et al. (2018) Nat Commun: SO-spindle coupling & memory
- Muehlroth et al. (2019) J Neurosci: Coupling & cognitive aging
- Adra et al. (2022): Spindle amplitude thresholds for CI prediction
- Vallat & Walker (2021): YASA — an open-source Python toolbox

Dependencies
------------
- yasa >= 0.7.0 (already in requirements.txt)
- mne >= 1.0 (already in requirements.txt)
- scipy (already in requirements.txt)
"""

import numpy as np
import warnings
from typing import Dict, Optional, Tuple

from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import circmean, circstd

from config import (
    EEG_SQI_THRESHOLD,
    SIGNAL_DTYPE,
    # Spindle detection parameters (added to config.py)
    SPINDLE_FREQ_RANGE,
    SPINDLE_FREQ_SLOW,
    SPINDLE_FREQ_FAST,
    SPINDLE_DURATION_RANGE,
    SPINDLE_MIN_AMPLITUDE_UV,
    # Slow oscillation detection parameters
    SO_FREQ_RANGE,
    SO_DURATION_RANGE,
    SO_MIN_AMPLITUDE_UV,
    # Coupling parameters
    COUPLING_ANALYSIS_ENABLED,
    COUPLING_MIN_SPINDLES,
    COUPLING_MIN_SOS,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# PUBLIC API — Called from features_eeg.py
# ==============================================================================

def extract_spindle_so_coupling_features(
    segment_data: np.ndarray,
    fs: float,
    region: str = "unknown",
    sqi: float = 0.0,
    segment_idx: int = 0,
    logger=None,
) -> Dict:
    """
    Extract all spindle, SO, and coupling features for one EEG segment.

    This is the single entry-point called by features_eeg.py after its
    existing spectral/complexity feature computation.

    Parameters
    ----------
    segment_data : np.ndarray
        Pre-processed EEG signal of the segment (already filtered 0.3-35 Hz).
    fs : float
        Sampling rate in Hz.
    region : str
        Brain region label (e.g. "C3-M2", "frontal", "central").
    sqi : float
        Signal Quality Index of this segment.
    segment_idx : int
        Segment index (for logging / bookkeeping).
    logger : optional
        Loguru logger instance.

    Returns
    -------
    Dict
        Feature dictionary with prefix ``eeg_{region}_sp_``, ``eeg_{region}_so_``,
        ``eeg_{region}_coup_``.
    """
    prefix_sp = f"eeg_{region}_sp"
    prefix_so = f"eeg_{region}_so"
    prefix_coup = f"eeg_{region}_coup"

    # Start with empty features (all NaN)
    features = {}
    features.update(_get_empty_spindle_features(prefix_sp))
    features.update(_get_empty_so_features(prefix_so))
    features.update(_get_empty_coupling_features(prefix_coup))

    # --- Guard clauses ---
    if segment_data is None or len(segment_data) == 0:
        return features

    if sqi < EEG_SQI_THRESHOLD:
        if logger:
            logger.debug(
                f"Segment {segment_idx} [{region}]: SQI too low for "
                f"spindle/SO detection ({sqi:.3f} < {EEG_SQI_THRESHOLD})"
            )
        return features

    duration_sec = len(segment_data) / fs
    # Need at least 5 s for meaningful detection
    if duration_sec < 5.0:
        return features

    # Ensure float64 for YASA (it uses scipy internally)
    data_f64 = segment_data.astype(np.float64)

    # ------------------------------------------------------------------
    # 1. SPINDLE DETECTION
    # ------------------------------------------------------------------
    sp_summary = None
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
    if COUPLING_ANALYSIS_ENABLED:
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
# SPINDLE DETECTION (YASA)
# ==============================================================================

def _detect_spindles(
    data: np.ndarray,
    fs: float,
    logger=None,
):
    """
    Detect sleep spindles using YASA's ``spindles_detect``.

    Returns the summary DataFrame from YASA or None.
    """
    import yasa

    sp = yasa.spindles_detect(
        data,
        sf=fs,
        freq_sp=SPINDLE_FREQ_RANGE,
        duration=SPINDLE_DURATION_RANGE,
        thresh={
            "rel_pow": 0.2,
            "corr": 0.65,
            "rms": 1.5,
        },
        multi_only=False,
        remove_outliers=True,
        verbose=False,
    )

    if sp is None:
        return None

    summary = sp.summary()
    if summary is None or len(summary) == 0:
        return None

    return summary


def _compute_spindle_features(
    sp_summary,
    duration_sec: float,
    fs: float,
    prefix: str,
) -> Dict:
    """
    Derive spindle features from the YASA summary DataFrame.
    """
    features = _get_empty_spindle_features(prefix)

    if sp_summary is None or len(sp_summary) == 0:
        features[f"{prefix}_count"] = 0
        features[f"{prefix}_density"] = 0.0
        return features

    n_spindles = len(sp_summary)
    duration_min = duration_sec / 60.0

    # --- Count & density ---
    features[f"{prefix}_count"] = n_spindles
    features[f"{prefix}_density"] = (
        float(n_spindles / duration_min) if duration_min > 0 else 0.0
    )

    # --- Duration ---
    if "Duration" in sp_summary.columns:
        dur = sp_summary["Duration"].values
        features[f"{prefix}_duration_mean"] = float(np.nanmean(dur))
        features[f"{prefix}_duration_std"] = (
            float(np.nanstd(dur, ddof=1)) if n_spindles > 1 else 0.0
        )
        features[f"{prefix}_duration_median"] = float(np.nanmedian(dur))

    # --- Amplitude ---
    if "Amplitude" in sp_summary.columns:
        amp = sp_summary["Amplitude"].values
        features[f"{prefix}_amplitude_mean"] = float(np.nanmean(amp))
        features[f"{prefix}_amplitude_std"] = (
            float(np.nanstd(amp, ddof=1)) if n_spindles > 1 else 0.0
        )
        features[f"{prefix}_amplitude_median"] = float(np.nanmedian(amp))

    # --- RMS ---
    if "RMS" in sp_summary.columns:
        rms = sp_summary["RMS"].values
        features[f"{prefix}_rms_mean"] = float(np.nanmean(rms))

    # --- Frequency ---
    if "Frequency" in sp_summary.columns:
        freq = sp_summary["Frequency"].values
        features[f"{prefix}_frequency_mean"] = float(np.nanmean(freq))
        features[f"{prefix}_frequency_std"] = (
            float(np.nanstd(freq, ddof=1)) if n_spindles > 1 else 0.0
        )

        # Slow vs fast spindle split
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

    # --- Symmetry ---
    if "Symmetry" in sp_summary.columns:
        sym = sp_summary["Symmetry"].values
        features[f"{prefix}_symmetry_mean"] = float(np.nanmean(sym))

    # --- Relative power (sigma) within detected spindles ---
    if "RelPower" in sp_summary.columns:
        rp = sp_summary["RelPower"].values
        features[f"{prefix}_rel_power_mean"] = float(np.nanmean(rp))

    # --- Oscillations count ---
    if "Oscillations" in sp_summary.columns:
        osc = sp_summary["Oscillations"].values
        features[f"{prefix}_oscillations_mean"] = float(np.nanmean(osc))

    return features


def _get_empty_spindle_features(prefix: str) -> Dict:
    """Return a dict of all spindle feature keys initialised to NaN."""
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
# SLOW OSCILLATION DETECTION (YASA)
# ==============================================================================

def _detect_slow_oscillations(
    data: np.ndarray,
    fs: float,
    logger=None,
):
    """
    Detect slow oscillations using YASA's ``sw_detect``.

    Returns the summary DataFrame from YASA or None.
    """
    import yasa

    sw = yasa.sw_detect(
        data,
        sf=fs,
        freq_sw=SO_FREQ_RANGE,
        dur_neg=SO_DURATION_RANGE,
        dur_pos=SO_DURATION_RANGE,
        amp_neg=(40, 200),
        amp_pos=(10, 150),
        amp_ptp=(SO_MIN_AMPLITUDE_UV, 500),
        coupling=False,  # We do coupling ourselves for more control
        remove_outliers=True,
        verbose=False,
    )

    if sw is None:
        return None

    summary = sw.summary()
    if summary is None or len(summary) == 0:
        return None

    return summary


def _compute_so_features(
    so_summary,
    duration_sec: float,
    prefix: str,
) -> Dict:
    """
    Derive slow oscillation features from the YASA summary DataFrame.
    """
    features = _get_empty_so_features(prefix)

    if so_summary is None or len(so_summary) == 0:
        features[f"{prefix}_count"] = 0
        features[f"{prefix}_density"] = 0.0
        return features

    n_so = len(so_summary)
    duration_min = duration_sec / 60.0

    # --- Count & density ---
    features[f"{prefix}_count"] = n_so
    features[f"{prefix}_density"] = (
        float(n_so / duration_min) if duration_min > 0 else 0.0
    )

    # --- Duration ---
    if "Duration" in so_summary.columns:
        dur = so_summary["Duration"].values
        features[f"{prefix}_duration_mean"] = float(np.nanmean(dur))
        features[f"{prefix}_duration_std"] = (
            float(np.nanstd(dur, ddof=1)) if n_so > 1 else 0.0
        )

    # --- Amplitude (peak-to-trough) ---
    if "PTP" in so_summary.columns:
        ptp = so_summary["PTP"].values
        features[f"{prefix}_ptp_amplitude_mean"] = float(np.nanmean(ptp))
        features[f"{prefix}_ptp_amplitude_std"] = (
            float(np.nanstd(ptp, ddof=1)) if n_so > 1 else 0.0
        )
        features[f"{prefix}_ptp_amplitude_median"] = float(np.nanmedian(ptp))

    # --- Negative peak amplitude ---
    if "ValNegPeak" in so_summary.columns:
        neg = so_summary["ValNegPeak"].values
        features[f"{prefix}_neg_peak_mean"] = float(np.nanmean(neg))
        features[f"{prefix}_neg_peak_std"] = (
            float(np.nanstd(neg, ddof=1)) if n_so > 1 else 0.0
        )

    # --- Positive peak amplitude ---
    if "ValPosPeak" in so_summary.columns:
        pos = so_summary["ValPosPeak"].values
        features[f"{prefix}_pos_peak_mean"] = float(np.nanmean(pos))

    # --- Frequency ---
    if "Frequency" in so_summary.columns:
        freq = so_summary["Frequency"].values
        features[f"{prefix}_frequency_mean"] = float(np.nanmean(freq))

    # --- Slope (negative half-wave) ---
    # Slope = amplitude / duration of negative half-wave
    if "Slope" in so_summary.columns:
        slope = so_summary["Slope"].values
        features[f"{prefix}_slope_mean"] = float(np.nanmean(slope))
        features[f"{prefix}_slope_std"] = (
            float(np.nanstd(slope, ddof=1)) if n_so > 1 else 0.0
        )
    elif "ValNegPeak" in so_summary.columns and "Duration" in so_summary.columns:
        # Manual slope: |neg_peak| / (duration / 2)
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
    """Return a dict of all SO feature keys initialised to NaN."""
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
    data: np.ndarray,
    fs: float,
    sp_summary,
    so_summary,
    duration_sec: float,
    prefix: str,
    logger=None,
) -> Dict:
    """
    Compute SO-spindle coupling features.

    Strategy:
    1. Extract the SO phase signal (bandpass → Hilbert → angle).
    2. For each detected spindle, find the SO phase at the spindle peak.
    3. Compute circular statistics on the coupling phase distribution.
    4. Compute a Modulation Index (PAC) between SO phase and sigma amplitude.
    """
    features = _get_empty_coupling_features(prefix)

    # Need both spindles and SOs
    if sp_summary is None or so_summary is None:
        return features
    if len(sp_summary) < COUPLING_MIN_SPINDLES:
        return features
    if len(so_summary) < COUPLING_MIN_SOS:
        return features

    # ------------------------------------------------------------------
    # A. Extract SO phase signal
    # ------------------------------------------------------------------
    try:
        so_phase = _extract_phase_signal(
            data, fs, SO_FREQ_RANGE[0], SO_FREQ_RANGE[1]
        )
    except Exception as e:
        if logger:
            logger.debug(f"SO phase extraction failed: {e}")
        return features

    # ------------------------------------------------------------------
    # B. Extract sigma amplitude envelope
    # ------------------------------------------------------------------
    try:
        sigma_env = _extract_amplitude_envelope(
            data, fs, SPINDLE_FREQ_RANGE[0], SPINDLE_FREQ_RANGE[1]
        )
    except Exception as e:
        if logger:
            logger.debug(f"Sigma envelope extraction failed: {e}")
        sigma_env = None

    # ------------------------------------------------------------------
    # C. Get spindle peak samples & corresponding SO phases
    # ------------------------------------------------------------------
    if "Peak" in sp_summary.columns:
        # 'Peak' is in seconds in YASA >= 0.6
        spindle_peak_samples = (sp_summary["Peak"].values * fs).astype(int)
    elif "PeakSample" in sp_summary.columns:
        spindle_peak_samples = sp_summary["PeakSample"].values.astype(int)
    else:
        # Fallback: use Start + Duration/2
        if "Start" in sp_summary.columns and "Duration" in sp_summary.columns:
            mid_sec = sp_summary["Start"].values + sp_summary["Duration"].values / 2
            spindle_peak_samples = (mid_sec * fs).astype(int)
        else:
            return features

    # Clip to valid range
    spindle_peak_samples = np.clip(spindle_peak_samples, 0, len(so_phase) - 1)

    # SO phase at each spindle peak
    coupling_phases = so_phase[spindle_peak_samples]

    # Remove NaN phases
    valid_mask = ~np.isnan(coupling_phases)
    coupling_phases = coupling_phases[valid_mask]

    n_coupled = len(coupling_phases)
    features[f"{prefix}_count"] = n_coupled

    if n_coupled < 3:
        # Not enough events for meaningful circular statistics
        features[f"{prefix}_rate"] = (
            float(n_coupled / len(sp_summary)) if len(sp_summary) > 0 else 0.0
        )
        return features

    features[f"{prefix}_rate"] = float(n_coupled / len(sp_summary))

    # ------------------------------------------------------------------
    # D. Circular statistics on coupling phases
    # ------------------------------------------------------------------
    # Mean phase angle (in radians, then convert to degrees)
    mean_angle_rad = float(circmean(coupling_phases, high=np.pi, low=-np.pi))
    features[f"{prefix}_mean_phase_rad"] = mean_angle_rad
    features[f"{prefix}_mean_phase_deg"] = float(np.degrees(mean_angle_rad))

    # Mean Resultant Length (MRL) — coupling strength
    # MRL = |mean(exp(i * phases))| ∈ [0, 1]
    mrl = float(np.abs(np.mean(np.exp(1j * coupling_phases))))
    features[f"{prefix}_mrl"] = mrl

    # Circular standard deviation
    features[f"{prefix}_phase_std_rad"] = float(
        circstd(coupling_phases, high=np.pi, low=-np.pi)
    )

    # Rayleigh test for non-uniformity (p-value)
    # z = n * R^2, where R = MRL
    z_rayleigh = n_coupled * mrl ** 2
    # Approximate p-value: p ≈ exp(-z) for large n
    p_rayleigh = float(np.exp(-z_rayleigh))
    features[f"{prefix}_rayleigh_z"] = float(z_rayleigh)
    features[f"{prefix}_rayleigh_p"] = p_rayleigh

    # Preferred phase quadrant (for interpretability)
    # SO trough = ~0 rad (or ~-π/π), SO rising = ~π/2, SO peak = ~π, SO falling = ~-π/2
    features[f"{prefix}_preferred_phase_quadrant"] = _phase_to_quadrant(
        mean_angle_rad
    )

    # ------------------------------------------------------------------
    # E. Phase-Amplitude Coupling (PAC) — Modulation Index
    # ------------------------------------------------------------------
    if sigma_env is not None:
        try:
            mi = _modulation_index(so_phase, sigma_env, n_bins=18)
            features[f"{prefix}_pac_mi"] = float(mi)
        except Exception:
            pass

    return features


def _extract_phase_signal(
    data: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    """
    Bandpass filter the data and extract instantaneous phase via Hilbert transform.

    Returns phase in radians [-π, π].
    """
    # Design bandpass filter
    nyq = fs / 2.0
    low = max(f_low / nyq, 0.001)
    high = min(f_high / nyq, 0.999)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, data)

    # Hilbert transform → analytic signal → phase
    analytic = hilbert(filtered)
    phase = np.angle(analytic)
    return phase


def _extract_amplitude_envelope(
    data: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
) -> np.ndarray:
    """
    Bandpass filter the data and extract the amplitude envelope via Hilbert transform.
    """
    nyq = fs / 2.0
    low = max(f_low / nyq, 0.001)
    high = min(f_high / nyq, 0.999)
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, data)

    analytic = hilbert(filtered)
    envelope = np.abs(analytic)
    return envelope


def _modulation_index(
    phase: np.ndarray,
    amplitude: np.ndarray,
    n_bins: int = 18,
) -> float:
    """
    Compute the Modulation Index (Tort et al., 2010) for phase-amplitude coupling.

    MI = KL(P, U) / log(N_bins)

    where P is the normalised mean amplitude per phase bin and U is the
    uniform distribution.

    Returns a value in [0, 1]; higher = stronger coupling.
    """
    if len(phase) != len(amplitude):
        min_len = min(len(phase), len(amplitude))
        phase = phase[:min_len]
        amplitude = amplitude[:min_len]

    # Phase bins from -π to π
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_indices = np.digitize(phase, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Mean amplitude per phase bin
    mean_amp = np.zeros(n_bins)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.any():
            mean_amp[b] = np.mean(amplitude[mask])

    # Normalise to probability distribution
    total = mean_amp.sum()
    if total == 0:
        return 0.0
    p = mean_amp / total

    # Uniform distribution
    q = np.ones(n_bins) / n_bins

    # KL divergence: D_KL(P || Q)
    # Avoid log(0) by adding epsilon
    eps = 1e-12
    p_safe = p + eps
    p_safe = p_safe / p_safe.sum()  # re-normalise after epsilon
    kl = np.sum(p_safe * np.log(p_safe / q))

    # Normalise by log(N_bins)
    mi = kl / np.log(n_bins)
    return float(np.clip(mi, 0, 1))


def _phase_to_quadrant(angle_rad: float) -> float:
    """
    Map a phase angle (radians) to a quadrant label (1-4).

    Quadrant 1: SO trough ascending  (-π/2 to 0)    → ~trough
    Quadrant 2: SO ascending to peak (0 to π/2)      → ~rising zero-crossing
    Quadrant 3: SO peak descending   (π/2 to π)      → ~peak
    Quadrant 4: SO descending        (-π to -π/2)    → ~falling zero-crossing

    Returns the quadrant as a float (1.0, 2.0, 3.0, 4.0).
    """
    if -np.pi / 2 <= angle_rad < 0:
        return 1.0
    elif 0 <= angle_rad < np.pi / 2:
        return 2.0
    elif np.pi / 2 <= angle_rad <= np.pi:
        return 3.0
    else:  # -π to -π/2
        return 4.0


def _get_empty_coupling_features(prefix: str) -> Dict:
    """Return a dict of all coupling feature keys initialised to NaN."""
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
