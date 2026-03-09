"""
utils/visualization.py
========================
Visualisierungsmodul für die PhysioNet Challenge 2026 Pipeline.

Dieses Modul bietet Visualisierungen für:

1. Signalqualität & Preprocessing:
   - Roh- und gefilterte Signale (ECG, EEG)
   - SQI-Verlauf über die Nacht
   - R-Peak-Detektion und RR-Intervalle
   - EEG-Kanalvergleich und Strategie-Entscheidung

2. Schlafarchitektur:
   - Hypnogramm
   - Schlafstadien-Verteilung
   - Event-Verteilung (Arousal, Apnea)
   - Schlafzyklen

3. Feature-Analyse:
   - Feature-Verteilungen
   - Korrelationsmatrizen
   - Feature-Vergleich CI vs. Non-CI
   - Nacht-Drittel und Zyklus-Vergleiche

4. Patienten-Übersicht:
   - Dashboard pro Patient
   - Kohorten-Übersicht

Alle Plot-Funktionen geben den Dateipfad des gespeicherten Plots zurück
und schließen die Figure automatisch, um Speicherlecks zu vermeiden.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')  # Non-interactive Backend für Server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from config import (
    OUTPUT_DIR, EEG_FREQUENCY_BANDS, SEGMENT_LENGTH_SEC,
    SLEEP_EVENTS_OF_INTEREST, ECG_SQI_THRESHOLD, EEG_SQI_THRESHOLD
)

# Globale Plot-Einstellungen
sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

# Farbpaletten
STAGE_COLORS = {
    "W": "#E53935",     # Rot
    "REM": "#1E88E5",   # Blau
    "N1": "#43A047",    # Grün
    "N2": "#FDD835",    # Gelb
    "N3": "#8E24AA",    # Lila
    "Unknown": "#9E9E9E",  # Grau
}

EVENT_COLORS = {
    "arousal": "#FF7043",
    "central_apnea": "#AB47BC",
    "obstructive_apnea": "#5C6BC0",
    "mixed_apnea": "#26A69A",
    "hypopnea": "#66BB6A",
}

BAND_COLORS = {
    "delta": "#8E24AA",
    "theta": "#1E88E5",
    "alpha": "#43A047",
    "sigma": "#FDD835",
    "beta": "#E53935",
}

# Ausgabeverzeichnis
PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. SIGNAL-VISUALISIERUNG
# ==============================================================================

def plot_ecg_preprocessing(
    ecg_raw: np.ndarray,
    ecg_cleaned: np.ndarray,
    fs: float,
    rpeaks: Optional[np.ndarray] = None,
    sqi_segments: Optional[pd.DataFrame] = None,
    patient_id: str = "unknown",
    time_range: Optional[Tuple[float, float]] = None,
    output_dir: Path = PLOT_DIR,
) -> Path:
    """
    Visualisiert die ECG-Vorverarbeitung.
    
    Zeigt:
    - Rohsignal vs. gefiltertes Signal
    - R-Peak-Detektion
    - SQI-Verlauf
    - RR-Intervalle
    """
    n_panels = 4 if sqi_segments is not None else 3
    fig, axes = plt.subplots(n_panels, 1, figsize=(16, 3.5 * n_panels),
                              sharex=False)
    
    # Zeitachse
    t = np.arange(len(ecg_raw)) / fs
    
    # Zeitbereich einschränken
    if time_range is not None:
        start_sec, end_sec = time_range
        start_idx = int(start_sec * fs)
        end_idx = int(end_sec * fs)
        t = t[start_idx:end_idx]
        ecg_raw = ecg_raw[start_idx:end_idx]
        ecg_cleaned = ecg_cleaned[start_idx:end_idx]
        title_suffix = f" ({start_sec:.0f}-{end_sec:.0f}s)"
    else:
        # Zeige nur die ersten 60 Sekunden für Übersichtlichkeit
        max_samples = int(60 * fs)
        if len(t) > max_samples:
            t = t[:max_samples]
            ecg_raw = ecg_raw[:max_samples]
            ecg_cleaned = ecg_cleaned[:max_samples]
        title_suffix = " (erste 60s)"
    
    # --- Panel 1: Rohsignal ---
    axes[0].plot(t, ecg_raw, color='#90A4AE', linewidth=0.5, alpha=0.8)
    axes[0].set_ylabel('Amplitude (mV)')
    axes[0].set_title(f'ECG Rohsignal{title_suffix}')
    
    # --- Panel 2: Gefiltertes Signal + R-Peaks ---
    axes[1].plot(t, ecg_cleaned, color='#1E88E5', linewidth=0.6)
    
    if rpeaks is not None:
        # R-Peaks im sichtbaren Bereich
        if time_range is not None:
            start_sample = int(time_range[0] * fs)
            visible_peaks = rpeaks[
                (rpeaks >= start_sample) &
                (rpeaks < start_sample + len(ecg_cleaned))
            ] - start_sample
        else:
            visible_peaks = rpeaks[rpeaks < len(ecg_cleaned)]
        
        if len(visible_peaks) > 0:
            peak_times = visible_peaks / fs
            if time_range:
                peak_times += time_range[0]
            peak_amplitudes = ecg_cleaned[visible_peaks.astype(int)]
            axes[1].plot(peak_times, peak_amplitudes, 'rv',
                         markersize=6, label=f'R-Peaks (n={len(visible_peaks)})')
            axes[1].legend(loc='upper right')
    
    axes[1].set_ylabel('Amplitude (mV)')
    axes[1].set_title('ECG Gefiltert + R-Peak-Detektion')
    
    # --- Panel 3: RR-Intervalle ---
    if rpeaks is not None and len(rpeaks) > 1:
        rr_intervals = np.diff(rpeaks) / fs * 1000  # ms
        rr_times = rpeaks[1:] / fs
        
        axes[2].plot(rr_times / 60, rr_intervals, 'o-',
                     color='#43A047', markersize=2, linewidth=0.8)
        axes[2].axhline(y=np.median(rr_intervals), color='red',
                        linestyle='--', linewidth=1, alpha=0.7,
                        label=f'Median: {np.median(rr_intervals):.0f}ms')
        axes[2].set_ylabel('RR-Intervall (ms)')
        axes[2].set_xlabel('Zeit (min)')
        axes[2].set_title('RR-Intervalle')
        axes[2].legend(loc='upper right')
        axes[2].set_ylim([
            max(200, np.percentile(rr_intervals, 1)),
            min(2000, np.percentile(rr_intervals, 99))
        ])
    else:
        axes[2].text(0.5, 0.5, 'Keine R-Peaks verfügbar',
                     ha='center', va='center', fontsize=14, color='gray')
        axes[2].set_title('RR-Intervalle')
    
    # --- Panel 4: SQI-Verlauf ---
    if sqi_segments is not None and n_panels > 3:
        seg_centers = (sqi_segments["start_sec"] + sqi_segments["end_sec"]) / 2 / 60
        
        # Farbcodierung nach Qualität
        colors = [
            '#43A047' if sqi >= ECG_SQI_THRESHOLD else '#E53935'
            for sqi in sqi_segments["sqi"]
        ]
        
        axes[3].bar(seg_centers, sqi_segments["sqi"],
                    width=SEGMENT_LENGTH_SEC / 60 * 0.9,
                    color=colors, edgecolor='white', linewidth=0.5)
        axes[3].axhline(y=ECG_SQI_THRESHOLD, color='red',
                        linestyle='--', linewidth=1.5,
                        label=f'Schwelle: {ECG_SQI_THRESHOLD}')
        axes[3].set_ylabel('SQI')
        axes[3].set_xlabel('Zeit (min)')
        axes[3].set_title('Signal Quality Index pro Segment')
        axes[3].set_ylim([0, 1.05])
        axes[3].legend(loc='lower right')
        
        # Statistik
        n_good = (sqi_segments["sqi"] >= ECG_SQI_THRESHOLD).sum()
        n_total = len(sqi_segments)
        axes[3].text(0.02, 0.95,
                     f'Gut: {n_good}/{n_total} ({n_good/n_total*100:.0f}%)',
                     transform=axes[3].transAxes, fontsize=10,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(f'ECG Preprocessing – Patient {patient_id}',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    path = output_dir / f"ecg_preprocessing_{patient_id}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_eeg_preprocessing(
    eeg_signals: Dict[str, np.ndarray],
    fs: float,
    eeg_strategies: Optional[Dict] = None,
    sqi_data: Optional[Dict[str, pd.DataFrame]] = None,
    patient_id: str = "unknown",
    duration_sec: float = 30.0,
    start_sec: float = 0.0,
    output_dir: Path = PLOT_DIR,
) -> Path:
    """
    Visualisiert die EEG-Vorverarbeitung und Kanalstrategie.
    
    Zeigt:
    - Alle verfügbaren EEG-Kanäle
    - Kanalstrategie-Entscheidung
    - SQI pro Kanal
    """
    n_channels = len(eeg_signals)
    if n_channels == 0:
        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.text(0.5, 0.5, 'Keine EEG-Kanäle verfügbar',
                ha='center', va='center', fontsize=16, color='gray')
        path = output_dir / f"eeg_preprocessing_{patient_id}.png"
        fig.savefig(path)
        plt.close(fig)
        return path
    
    # Layout: Kanäle + SQI-Übersicht
    has_sqi = sqi_data is not None and len(sqi_data) > 0
    n_rows = n_channels + (1 if has_sqi else 0)
    
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.5 * n_rows), sharex=True)
    if n_rows == 1:
        axes = [axes]
    
    start_sample = int(start_sec * fs)
    end_sample = int((start_sec + duration_sec) * fs)
    t = np.arange(start_sample, end_sample) / fs
    
    # --- EEG-Kanäle ---
    for i, (channel_name, signal) in enumerate(eeg_signals.items()):
        seg = signal[start_sample:min(end_sample, len(signal))]
        t_seg = t[:len(seg)]
        
        # Farbe basierend auf Strategie
        color = '#1E88E5'
        label_suffix = ""
        if eeg_strategies:
            for region, strategy in eeg_strategies.items():
                if channel_name in strategy.get("channels_used", []):
                    if strategy["strategy"] == "averaged":
                        color = '#43A047'
                        label_suffix = " [AVERAGED]"
                    elif strategy["strategy"] in ["single_left", "single_right"]:
                        color = '#FF9800'
                        label_suffix = " [SINGLE]"
                    break
            else:
                color = '#9E9E9E'
                label_suffix = " [UNUSED]"
        
        axes[i].plot(t_seg, seg, color=color, linewidth=0.5, alpha=0.9)
        axes[i].set_ylabel('µV')
        axes[i].set_title(f'{channel_name}{label_suffix}', fontsize=11, loc='left')
        
        # Amplitude-Statistik
        axes[i].text(0.98, 0.95,
                     f'µ={np.mean(seg):.1f}, σ={np.std(seg):.1f}, '
                     f'max={np.max(np.abs(seg)):.1f}µV',
                     transform=axes[i].transAxes, fontsize=8,
                     ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # --- SQI-Übersicht ---
    if has_sqi:
        ax_sqi = axes[-1]
        for channel_name, sqi_df in sqi_data.items():
            seg_centers = (sqi_df["start_sec"] + sqi_df["end_sec"]) / 2 / 60
            ax_sqi.plot(seg_centers, sqi_df["sqi"], 'o-',
                        markersize=3, linewidth=1, label=channel_name)
        
        ax_sqi.axhline(y=EEG_SQI_THRESHOLD, color='red',
                        linestyle='--', linewidth=1.5, alpha=0.7)
        ax_sqi.set_ylabel('SQI')
        ax_sqi.set_xlabel('Zeit (min)')
        ax_sqi.set_title('EEG Signal Quality Index', fontsize=11, loc='left')
        ax_sqi.set_ylim([0, 1.05])
        ax_sqi.legend(loc='lower right', fontsize=8, ncol=3)
    
    axes[-1].set_xlabel('Zeit (s)')
    
    fig.suptitle(f'EEG Preprocessing – Patient {patient_id}',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    path = output_dir / f"eeg_preprocessing_{patient_id}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ==============================================================================
# 2. SCHLAFARCHITEKTUR-VISUALISIERUNG
# ==============================================================================

def plot_hypnogram(
    stages_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame] = None,
    sqi_ecg: Optional[pd.DataFrame] = None,
    sqi_eeg: Optional[Dict[str, pd.DataFrame]] = None,
    patient_id: str = "unknown",
    output_dir: Path = PLOT_DIR,
) -> Path:
    """
    Erstellt ein detailliertes Hypnogramm mit Events und Signalqualität.
    """
    n_panels = 2
    if sqi_ecg is not None or sqi_eeg is not None:
        n_panels = 3
    if events_df is not None and len(events_df) > 0:
        n_panels += 1
    
    fig, axes = plt.subplots(n_panels, 1, figsize=(18, 2.5 * n_panels),
                              sharex=True,
                              gridspec_kw={'height_ratios':
                                           [3] + [1.5] * (n_panels - 1)})
    
    # --- Panel 1: Hypnogramm ---
    ax_hyp = axes[0]
    
    stage_order = {"W": 5, "REM": 4, "N1": 3, "N2": 2, "N3": 1, "Unknown": 0}
    stage_labels = {5: "W", 4: "REM", 3: "N1", 2: "N2", 1: "N3", 0: "?"}
    
    if "start_sec" in stages_df.columns and "stage_label" in stages_df.columns:
        for _, row in stages_df.iterrows():
            stage = row["stage_label"]
            y_val = stage_order.get(stage, 0)
            start_min = row["start_sec"] / 60
            duration_min = row.get("duration_sec", 30) / 60
            
            color = STAGE_COLORS.get(stage, '#9E9E9E')
            ax_hyp.barh(y_val, duration_min, left=start_min,
                        height=0.8, color=color, edgecolor='none')
    
    ax_hyp.set_yticks(list(stage_labels.keys()))
    ax_hyp.set_yticklabels(list(stage_labels.values()))
    ax_hyp.set_ylim([0.3, 5.7])
    ax_hyp.invert_yaxis()
    ax_hyp.set_title(f'Hypnogramm – Patient {patient_id}', fontsize=14)
    ax_hyp.set_ylabel('Schlafstadium')
    
    # Legende
    legend_patches = [
        Patch(facecolor=color, label=stage)
        for stage, color in STAGE_COLORS.items()
        if stage != "Unknown"
    ]
    ax_hyp.legend(handles=legend_patches, loc='upper right',
                  ncol=5, fontsize=9)
    
    panel_idx = 1
    
    # --- Panel 2: Events ---
    if events_df is not None and len(events_df) > 0:
        ax_events = axes[panel_idx]
        panel_idx += 1
        
        for event_type in SLEEP_EVENTS_OF_INTEREST:
            type_events = events_df[events_df["event_type"] == event_type]
            if len(type_events) == 0:
                continue
            
            color = EVENT_COLORS.get(event_type, '#9E9E9E')
            y_pos = list(EVENT_COLORS.keys()).index(event_type) if event_type in EVENT_COLORS else 0
            
            for _, ev in type_events.iterrows():
                start_min = ev["start_sec"] / 60
                dur_min = ev.get("duration_sec", 10) / 60
                ax_events.barh(y_pos, dur_min, left=start_min,
                               height=0.6, color=color, alpha=0.7)
        
        ax_events.set_yticks(range(len(EVENT_COLORS)))
        ax_events.set_yticklabels([
            et.replace("_", " ").title() for et in EVENT_COLORS.keys()
        ], fontsize=8)
        ax_events.set_title('Schlaf-Events', fontsize=11, loc='left')
    
    # --- Panel: Schlafstadien-Verteilung (Pie) ---
    ax_dist = axes[panel_idx]
    panel_idx += 1
    
    if "stage_label" in stages_df.columns:
        stage_counts = stages_df["stage_label"].value_counts()
        total_epochs = len(stages_df)
        
        # Horizontales Balkendiagramm statt Pie
        stages_ordered = ["W", "N1", "N2", "N3", "REM"]
        counts = [stage_counts.get(s, 0) for s in stages_ordered]
        pcts = [c / total_epochs * 100 for c in counts]
        colors = [STAGE_COLORS[s] for s in stages_ordered]
        
        bars = ax_dist.barh(stages_ordered, pcts, color=colors,
                            edgecolor='white', linewidth=0.5)
        
        for bar, pct, count in zip(bars, pcts, counts):
            ax_dist.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                         f'{pct:.1f}% ({count})',
                         va='center', fontsize=9)
        
        ax_dist.set_xlabel('Anteil (%)')
        ax_dist.set_title('Schlafstadien-Verteilung', fontsize=11, loc='left')
        ax_dist.set_xlim([0, max(pcts) * 1.3])
    
    # --- Panel: SQI ---
    if sqi_ecg is not None or sqi_eeg is not None:
        if panel_idx < len(axes):
            ax_sqi = axes[panel_idx]
            
            if sqi_ecg is not None:
                seg_centers = (sqi_ecg["start_sec"] + sqi_ecg["end_sec"]) / 2 / 60
                ax_sqi.plot(seg_centers, sqi_ecg["sqi"], '-',
                            color='#E53935', linewidth=1, label='ECG SQI')
            
            if sqi_eeg is not None:
                for region, sqi_df in sqi_eeg.items():
                    seg_centers = (sqi_df["start_sec"] + sqi_df["end_sec"]) / 2 / 60
                    ax_sqi.plot(seg_centers, sqi_df["sqi"], '-',
                                linewidth=1, label=f'EEG {region} SQI')
            
            ax_sqi.axhline(y=ECG_SQI_THRESHOLD, color='red',
                            linestyle=':', linewidth=1, alpha=0.5)
            ax_sqi.set_ylabel('SQI')
            ax_sqi.set_ylim([0, 1.05])
            ax_sqi.set_title('Signalqualität', fontsize=11, loc='left')
            ax_sqi.legend(loc='lower right', fontsize=8, ncol=4)
    
    axes[-1].set_xlabel('Zeit (min)')
    
    plt.tight_layout()
    path = output_dir / f"hypnogram_{patient_id}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ==============================================================================
# 3. FEATURE-ANALYSE
# ==============================================================================

def plot_feature_distributions(
    feature_table: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_col: str = "target",
    max_features: int = 20,
    patient_id: str = "cohort",
    output_dir: Path = PLOT_DIR,
) -> Path:
    """
    Visualisiert Feature-Verteilungen, getrennt nach Target-Klasse.
    """
    if features is None:
        # Wähle die wichtigsten numerischen Features
        numeric_cols = feature_table.select_dtypes(include=[np.number]).columns
        exclude = ["patient_id", "segment_idx", target_col, "demo_time_to_event"]
        features = [c for c in numeric_cols if c not in exclude][:max_features]
    
    n_features = min(len(features), max_features)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = axes.flatten() if n_rows > 1 else (
        [axes] if n_features == 1 else axes.flatten()
    )
    
    has_target = target_col in feature_table.columns
    
    for i, feat in enumerate(features[:n_features]):
        ax = axes[i]
        
        if has_target:
            for label, color, name in [(0, '#1E88E5', 'No CI'), (1, '#E53935', 'CI')]:
                data = feature_table[feature_table[target_col] == label][feat].dropna()
                if len(data) > 0:
                    ax.hist(data, bins=20, alpha=0.6, color=color,
                            label=name, density=True)
        else:
            data = feature_table[feat].dropna()
            ax.hist(data, bins=20, alpha=0.7, color='#1E88E5', density=True)
        
        ax.set_title(feat, fontsize=8, fontweight='bold')
        ax.tick_params(labelsize=7)
        
        if i == 0 and has_target:
            ax.legend(fontsize=7)
    
    # Leere Subplots ausblenden
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(f'Feature-Verteilungen – {patient_id}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    path = output_dir / f"feature_distributions_{patient_id}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_correlation_matrix(
    feature_table: pd.DataFrame,
    features: Optional[List[str]] = None,
    max_features: int = 40,
    method: str = "spearman",
    patient_id: str = "cohort",
    output_dir: Path = PLOT_DIR,
) -> Path:
    """
    Erstellt eine Korrelationsmatrix der Features.
    """
    if features is None:
        numeric_cols = feature_table.select_dtypes(include=[np.number]).columns
        exclude = ["patient_id", "segment_idx"]
        features = [c for c in numeric_cols if c not in exclude]
    
    features = features[:max_features]
    
    corr_data = feature_table[features].corr(method=method)
    
    # Clustered Heatmap
    fig_size = max(10, len(features) * 0.35)
    
    try:
        g = sns.clustermap(
            corr_data,
            cmap="RdBu_r",
            center=0,
            vmin=-1, vmax=1,
            figsize=(fig_size, fig_size),
            dendrogram_ratio=0.1,
            linewidths=0.1,
            xticklabels=True,
            yticklabels=True,
        )
        g.ax_heatmap.tick_params(labelsize=7)
        g.fig.suptitle(f'Feature-Korrelationsmatrix ({method}) – {patient_id}',
                       fontsize=14, y=1.02)
        
        path = output_dir / f"correlation_matrix_{patient_id}.png"
        g.savefig(path)
        plt.close(g.fig)
    except Exception:
        # Fallback ohne Clustering
        fig, ax = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        sns.heatmap(corr_data, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    ax=ax, xticklabels=True, yticklabels=True,
                    linewidths=0.1)
        ax.tick_params(labelsize=7)
        ax.set_title(f'Feature-Korrelationsmatrix ({method}) – {patient_id}',
                     fontsize=14)
        
        path = output_dir / f"correlation_matrix_{patient_id}.png"
        fig.savefig(path)
        plt.close(fig)
    
    return path


def plot_feature_comparison_ci(
    feature_table: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_col: str = "target",
    max_features: int = 20,
    output_dir: Path = PLOT_DIR,
) -> Path:
    """
    Vergleicht Features zwischen CI und Non-CI Gruppen.
    Zeigt Effektstärken (Cohen's d) und statistische Signifikanz.
    """
    if target_col not in feature_table.columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'Keine Target-Variable verfügbar',
                ha='center', va='center', fontsize=14)
        path = output_dir / "feature_comparison_ci.png"
        fig.savefig(path)
        plt.close(fig)
        return path
    
    if features is None:
        numeric_cols = feature_table.select_dtypes(include=[np.number]).columns
        exclude = ["patient_id", "segment_idx", target_col, "demo_time_to_event"]
        features = [c for c in numeric_cols if c not in exclude]
    
    # Berechne Effektstärken
    effect_sizes = []
    for feat in features:
        ci_data = feature_table[feature_table[target_col] == 1][feat].dropna()
        no_ci_data = feature_table[feature_table[target_col] == 0][feat].dropna()
        
        if len(ci_data) < 3 or len(no_ci_data) < 3:
            continue
        
        # Cohen's d: Differenz der Mittelwerte / gepoolte Standardabweichung [[1]]
        mean_ci = np.mean(ci_data)
        mean_no = np.mean(no_ci_data)
        std_ci = np.std(ci_data, ddof=1)
        std_no = np.std(no_ci_data, ddof=1)
        
        n_ci = len(ci_data)
        n_no = len(no_ci_data)
        
        # Gepoolte Standardabweichung
        pooled_std = np.sqrt(
            ((n_ci - 1) * std_ci**2 + (n_no - 1) * std_no**2) /
            (n_ci + n_no - 2)
        )
        
        if pooled_std > 0:
            cohens_d = (mean_ci - mean_no) / pooled_std
        else:
            cohens_d = 0
        
        # Mann-Whitney U Test (nicht-parametrisch)
        try:
            from scipy.stats import mannwhitneyu
            _, p_value = mannwhitneyu(ci_data, no_ci_data, alternative='two-sided')
        except Exception:
            p_value = 1.0
        
        effect_sizes.append({
            "feature": feat,
            "cohens_d": cohens_d,
            "abs_cohens_d": abs(cohens_d),
            "p_value": p_value,
            "mean_ci": mean_ci,
            "mean_no_ci": mean_no,
            "significant": p_value < 0.05,
        })
    
    if not effect_sizes:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'Keine Features für Vergleich verfügbar',
                ha='center', va='center', fontsize=14)
        path = output_dir / "feature_comparison_ci.png"
        fig.savefig(path)
        plt.close(fig)
        return path
    
    # Sortiere nach absolutem Cohen's d
    effect_df = pd.DataFrame(effect_sizes)
    effect_df = effect_df.sort_values("abs_cohens_d", ascending=False)
    top_effects = effect_df.head(max_features)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(top_effects) * 0.4)),
                              gridspec_kw={'width_ratios': [2, 1]})
    
    # --- Links: Cohen's d Barplot ---
    ax = axes[0]
    top_effects_sorted = top_effects.sort_values("abs_cohens_d", ascending=True)
    
    colors = []
    for _, row in top_effects_sorted.iterrows():
        if row["significant"]:
            colors.append('#E53935' if row["cohens_d"] > 0 else '#1E88E5')
        else:
            colors.append('#BDBDBD')
    
    y_pos = range(len(top_effects_sorted))
    ax.barh(y_pos, top_effects_sorted["cohens_d"].values,
            color=colors, edgecolor='white', linewidth=0.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_effects_sorted["feature"].values, fontsize=8)
    ax.set_xlabel("Cohen's d (Effektstärke)")
    ax.set_title("Feature-Unterschiede: CI vs. Non-CI", fontsize=13)
    ax.axvline(x=0, color='black', linewidth=0.5)
    
    # Effektstärke-Referenzlinien
    for threshold, label in [(0.2, 'klein'), (0.5, 'mittel'), (0.8, 'groß')]:
        ax.axvline(x=threshold, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax.axvline(x=-threshold, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
    
    # Legende
    legend_patches = [
        Patch(facecolor='#E53935', label='Höher bei CI (p<0.05)'),
        Patch(facecolor='#1E88E5', label='Niedriger bei CI (p<0.05)'),
        Patch(facecolor='#BDBDBD', label='Nicht signifikant'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=9)
    
    # --- Rechts: P-Wert Volcano-ähnlich ---
    ax2 = axes[1]
    
    for _, row in effect_df.iterrows():
        color = '#E53935' if row["significant"] else '#BDBDBD'
        size = max(20, min(200, abs(row["cohens_d"]) * 100))
        ax2.scatter(
            row["cohens_d"],
            -np.log10(row["p_value"] + 1e-300),
            c=color, s=size, alpha=0.7, edgecolors='white', linewidth=0.5
        )
    
    ax2.axhline(y=-np.log10(0.05), color='red', linestyle='--',
                linewidth=1, alpha=0.5, label='p=0.05')
    ax2.axvline(x=0, color='black', linewidth=0.5)
    ax2.set_xlabel("Cohen's d")
    ax2.set_ylabel("-log10(p-value)")
    ax2.set_title("Volcano Plot", fontsize=13)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    path = output_dir / "feature_comparison_ci.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_night_third_comparison(
    feature_table: pd.DataFrame,
    features: Optional[List[str]] = None,
    target_col: str = "target",
    output_dir: Path = PLOT_DIR,
) -> Path:
    """
    Vergleicht Features über die drei Nacht-Drittel,
    getrennt nach CI und Non-CI.
    """
    third_col = None
    for candidate in ["ann_night_third"]:
        if candidate in feature_table.columns:
            third_col = candidate
            break
    
    if third_col is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, 'Keine Nacht-Drittel-Information verfügbar',
                ha='center', va='center', fontsize=14, color='gray')
        path = output_dir / "night_third_comparison.png"
        fig.savefig(path)
        plt.close(fig)
        return path
    
    if features is None:
        features = _select_key_visualization_features(feature_table)
    
    n_features = min(len(features), 8)
    n_cols = 2
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows))
    axes = axes.flatten()
    
    has_target = target_col in feature_table.columns
    
    for i, feat in enumerate(features[:n_features]):
        ax = axes[i]
        
        if has_target:
            # Gruppiert nach Drittel und Target
            for label, color, name in [(0, '#1E88E5', 'No CI'), (1, '#E53935', 'CI')]:
                group_data = feature_table[feature_table[target_col] == label]
                means = []
                stds = []
                thirds = [1, 2, 3]
                
                for third in thirds:
                    vals = group_data[group_data[third_col] == third][feat].dropna()
                    means.append(vals.mean() if len(vals) > 0 else np.nan)
                    stds.append(vals.std() if len(vals) > 1 else 0)
                
                offset = -0.15 if label == 0 else 0.15
                ax.bar(np.array(thirds) + offset, means, width=0.3,
                       yerr=stds, color=color, alpha=0.7, label=name,
                       capsize=3, error_kw={'linewidth': 1})
        else:
            means = []
            stds = []
            thirds = [1, 2, 3]
            for third in thirds:
                vals = feature_table[feature_table[third_col] == third][feat].dropna()
                means.append(vals.mean() if len(vals) > 0 else np.nan)
                stds.append(vals.std() if len(vals) > 1 else 0)
            
            ax.bar(thirds, means, width=0.5, yerr=stds,
                   color='#1E88E5', alpha=0.7, capsize=3)
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['1. Drittel', '2. Drittel', '3. Drittel'], fontsize=9)
        ax.set_title(feat, fontsize=10, fontweight='bold')
        
        if i == 0 and has_target:
            ax.legend(fontsize=9)
    
    for j in range(n_features, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle('Feature-Vergleich über Nacht-Drittel',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    path = output_dir / "night_third_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_spectral_profile(
    feature_table: pd.DataFrame,
    region: str = "central",
    target_col: str = "target",
    output_dir: Path = PLOT_DIR,
) -> Path:
    """
    Visualisiert das spektrale Profil (Bandpower) für eine EEG-Region,
    getrennt nach CI und Non-CI.
    """
    prefix = f"eeg_{region}"
    
    bands = list(EEG_FREQUENCY_BANDS.keys())
    band_labels = [b.capitalize() for b in bands]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    has_target = target_col in feature_table.columns
    
    # --- Panel 1: Absolute Bandpower ---
    ax = axes[0]
    x = np.arange(len(bands))
    width = 0.35
    
    if has_target:
        for label, color, name, offset in [
            (0, '#1E88E5', 'No CI', -width/2),
            (1, '#E53935', 'CI', width/2)
        ]:
            group = feature_table[feature_table[target_col] == label]
            means = []
            stds = []
            for band in bands:
                col = f"{prefix}_{band}_power_abs"
                if col in group.columns:
                    vals = group[col].dropna()
                    means.append(vals.mean() if len(vals) > 0 else 0)
                    stds.append(vals.std() if len(vals) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)
            
            ax.bar(x + offset, means, width, yerr=stds,
                   color=color, alpha=0.7, label=name, capsize=3)
    else:
        means = []
        stds = []
        for band in bands:
            col = f"{prefix}_{band}_power_abs"
            if col in feature_table.columns:
                vals = feature_table[col].dropna()
                means.append(vals.mean() if len(vals) > 0 else 0)
                stds.append(vals.std() if len(vals) > 1 else 0)
            else:
                means.append(0)
                stds.append(0)
        
        bar_colors = [BAND_COLORS.get(b, '#9E9E9E') for b in bands]
        ax.bar(x, means, width * 2, yerr=stds, color=bar_colors,
               alpha=0.7, capsize=3)
    
    ax.set_xticks(x)
    ax.set_xticklabels(band_labels)
    ax.set_ylabel('Power (µV²)')
    ax.set_title('Absolute Bandpower', fontsize=12)
    if has_target:
        ax.legend()
    
    # --- Panel 2: Relative Bandpower ---
    ax2 = axes[1]
    
    if has_target:
        for label, color, name, offset in [
            (0, '#1E88E5', 'No CI', -width/2),
            (1, '#E53935', 'CI', width/2)
        ]:
            group = feature_table[feature_table[target_col] == label]
            means = []
            for band in bands:
                col = f"{prefix}_{band}_power_rel"
                if col in group.columns:
                    vals = group[col].dropna()
                    means.append(vals.mean() * 100 if len(vals) > 0 else 0)
                else:
                    means.append(0)
            
            ax2.bar(x + offset, means, width, color=color, alpha=0.7, label=name)
    else:
        means = []
        for band in bands:
            col = f"{prefix}_{band}_power_rel"
            if col in feature_table.columns:
                vals = feature_table[col].dropna()
                means.append(vals.mean() * 100 if len(vals) > 0 else 0)
            else:
                means.append(0)
        
        bar_colors = [BAND_COLORS.get(b, '#9E9E9E') for b in bands]
        ax2.bar(x, means, width * 2, color=bar_colors, alpha=0.7)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(band_labels)
    ax2.set_ylabel('Relative Power (%)')
    ax2.set_title('Relative Bandpower', fontsize=12)
    if has_target:
        ax2.legend()
    
    # --- Panel 3: Schlaf-Ratios ---
    ax3 = axes[2]
    
    ratio_features = [
        (f"{prefix}_theta_alpha_ratio", "θ/α"),
        (f"{prefix}_delta_alpha_ratio", "δ/α"),
        (f"{prefix}_delta_beta_ratio", "δ/β"),
        (f"{prefix}_slowing_ratio", "(δ+θ)/(α+β)"),
        (f"{prefix}_dar", "DAR"),
    ]
    
    available_ratios = [
        (col, label) for col, label in ratio_features
        if col in feature_table.columns
    ]
    
    if available_ratios:
        ratio_x = np.arange(len(available_ratios))
        ratio_labels = [label for _, label in available_ratios]
        
        if has_target:
            for label_val, color, name, offset in [
                (0, '#1E88E5', 'No CI', -width/2),
                (1, '#E53935', 'CI', width/2)
            ]:
                group = feature_table[feature_table[target_col] == label_val]
                means = []
                stds = []
                for col, _ in available_ratios:
                    vals = group[col].dropna()
                    means.append(vals.mean() if len(vals) > 0 else 0)
                    stds.append(vals.std() if len(vals) > 1 else 0)
                
                ax3.bar(ratio_x + offset, means, width, yerr=stds,
                        color=color, alpha=0.7, label=name, capsize=3)
        else:
            means = []
            stds = []
            for col, _ in available_ratios:
                vals = feature_table[col].dropna()
                means.append(vals.mean() if len(vals) > 0 else 0)
                stds.append(vals.std() if len(vals) > 1 else 0)
            
            ax3.bar(ratio_x, means, width * 2, yerr=stds,
                    color='#FF9800', alpha=0.7, capsize=3)
        
        ax3.set_xticks(ratio_x)
        ax3.set_xticklabels(ratio_labels, fontsize=9)
        ax3.set_ylabel('Ratio')
        ax3.set_title('EEG Slowing Ratios', fontsize=12)
        if has_target:
            ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'Keine Ratio-Features verfügbar',
                 ha='center', va='center', fontsize=12, color='gray')
    
    fig.suptitle(f'EEG Spektrales Profil – Region: {region.capitalize()}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = output_dir / f"spectral_profile_{region}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ==============================================================================
# 4. PATIENTEN-DASHBOARD
# ==============================================================================

def plot_patient_dashboard(
    patient_id: str,
    segment_features: pd.DataFrame,
    sleep_summary: Optional[Dict] = None,
    output_dir: Path = PLOT_DIR,
) -> Path:
    """
    Erstellt ein umfassendes Dashboard für einen einzelnen Patienten.
    """
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.35)
    
    # --- 1. Schlafstadien-Verlauf (oben, volle Breite) ---
    ax1 = fig.add_subplot(gs[0, :])
    _plot_stage_timeline(ax1, segment_features)
    
    # --- 2. Herzfrequenz-Verlauf ---
    ax2 = fig.add_subplot(gs[1, :2])
    _plot_hr_timeline(ax2, segment_features)
    
    # --- 3. HRV-Verlauf (SDNN, RMSSD) ---
    ax3 = fig.add_subplot(gs[1, 2:])
    _plot_hrv_timeline(ax3, segment_features)
    
    # --- 4. EEG Bandpower-Verlauf ---
    ax4 = fig.add_subplot(gs[2, :2])
    _plot_bandpower_timeline(ax4, segment_features)
    
    # --- 5. Event-Verlauf ---
    ax5 = fig.add_subplot(gs[2, 2:])
    _plot_event_timeline(ax5, segment_features)
    
    # --- 6. SQI-Verlauf ---
    ax6 = fig.add_subplot(gs[3, :2])
    _plot_sqi_timeline(ax6, segment_features)
    
    # --- 7. Schlafarchitektur-Zusammenfassung ---
    ax7 = fig.add_subplot(gs[3, 2:])
    _plot_sleep_summary_panel(ax7, sleep_summary, patient_id)
    
    fig.suptitle(f'Patienten-Dashboard – {patient_id}',
                 fontsize=18, fontweight='bold', y=0.98)
    
    path = output_dir / f"patient_dashboard_{patient_id}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_stage_timeline(ax, segment_features):
    """Schlafstadien-Verlauf als farbcodierte Timeline."""
    stage_col = None
    for candidate in ["ann_dominant_stage", "dominant_stage"]:
        if candidate in segment_features.columns:
            stage_col = candidate
            break
    
    if stage_col is None:
        ax.text(0.5, 0.5, 'Keine Schlafstadien', ha='center', va='center')
        ax.set_title('Schlafstadien')
        return
    
    stage_order = {"W": 5, "REM": 4, "N1": 3, "N2": 2, "N3": 1, "Unknown": 0}
    
    time_col = None
    for candidate in ["meta_start_sec", "ann_time_sec", "start_sec"]:
        if candidate in segment_features.columns:
            time_col = candidate
            break
    
    if time_col:
        times = segment_features[time_col].values / 60  # Minuten
    else:
        times = np.arange(len(segment_features)) * SEGMENT_LENGTH_SEC / 60
    
    stages = segment_features[stage_col].values
    y_vals = [stage_order.get(s, 0) for s in stages]
    colors = [STAGE_COLORS.get(s, '#9E9E9E') for s in stages]
    
    for i in range(len(times)):
        width = SEGMENT_LENGTH_SEC / 60
        ax.bar(times[i], 1, bottom=y_vals[i] - 0.5, width=width,
               color=colors[i], edgecolor='none')
    
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(['N3', 'N2', 'N1', 'REM', 'W'])
    ax.set_ylim([0.3, 5.7])
    ax.set_xlabel('Zeit (min)')
    ax.set_title('Schlafstadien-Verlauf', fontsize=12)


def _plot_hr_timeline(ax, segment_features):
    """Herzfrequenz-Verlauf."""
    hr_col = "hr_mean"
    if hr_col not in segment_features.columns:
        ax.text(0.5, 0.5, 'Keine HR-Daten', ha='center', va='center')
        ax.set_title('Herzfrequenz')
        return
    
    times = np.arange(len(segment_features)) * SEGMENT_LENGTH_SEC / 60
    hr = segment_features[hr_col].values
    
    valid = ~np.isnan(hr)
    ax.plot(times[valid], hr[valid], '-', color='#E53935',
            linewidth=1, alpha=0.8)
    ax.fill_between(times[valid], hr[valid], alpha=0.1, color='#E53935')
    ax.set_ylabel('HR (bpm)')
    ax.set_xlabel('Zeit (min)')
    ax.set_title('Herzfrequenz', fontsize=12)


def _plot_hrv_timeline(ax, segment_features):
    """HRV-Verlauf (SDNN, RMSSD)."""
    times = np.arange(len(segment_features)) * SEGMENT_LENGTH_SEC / 60
    
    plotted = False
    for col, color, label in [
        ("hrv_sdnn", '#1E88E5', 'SDNN'),
        ("hrv_rmssd", '#43A047', 'RMSSD')
    ]:
        if col in segment_features.columns:
            vals = segment_features[col].values
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(times[valid], vals[valid], '-',
                        color=color, linewidth=1, label=label)
                plotted = True
    
    if plotted:
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Keine HRV-Daten', ha='center', va='center')
    
    ax.set_ylabel('ms')
    ax.set_xlabel('Zeit (min)')
    ax.set_title('HRV-Parameter', fontsize=12)


def _plot_bandpower_timeline(ax, segment_features):
    """EEG Bandpower-Verlauf."""
    times = np.arange(len(segment_features)) * SEGMENT_LENGTH_SEC / 60
    
    plotted = False
    for region in ["central", "frontal", "occipital"]:
        for band, color in BAND_COLORS.items():
            col = f"eeg_{region}_{band}_power_rel"
            if col in segment_features.columns:
                vals = segment_features[col].values
                valid = ~np.isnan(vals)
                if valid.any():
                    ax.plot(times[valid], vals[valid] * 100, '-',
                            color=color, linewidth=0.8,
                            label=f'{band.capitalize()}', alpha=0.8)
                    plotted = True
        if plotted:
            break  # Nur eine Region plotten
    
    if plotted:
        ax.legend(fontsize=8, ncol=5, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Keine EEG-Daten', ha='center', va='center')
    
    ax.set_ylabel('Relative Power (%)')
    ax.set_xlabel('Zeit (min)')
    ax.set_title('EEG Bandpower', fontsize=12)


def _plot_event_timeline(ax, segment_features):
    """Event-Verlauf."""
    times = np.arange(len(segment_features)) * SEGMENT_LENGTH_SEC / 60
    
    plotted = False
    for event_type, color in EVENT_COLORS.items():
        col = f"ann_{event_type}_count"
        if col in segment_features.columns:
            vals = segment_features[col].values
            if np.nansum(vals) > 0:
                ax.bar(times, vals, width=SEGMENT_LENGTH_SEC / 60 * 0.9,
                       color=color, alpha=0.6,
                       label=event_type.replace("_", " ").title())
                plotted = True
    
    if plotted:
        ax.legend(fontsize=8, ncol=3, loc='upper right')
    else:
        ax.text(0.5, 0.5, 'Keine Events', ha='center', va='center')
    
    ax.set_ylabel('Anzahl')
    ax.set_xlabel('Zeit (min)')
    ax.set_title('Schlaf-Events', fontsize=12)


def _plot_sqi_timeline(ax, segment_features):
    """SQI-Verlauf."""
    times = np.arange(len(segment_features)) * SEGMENT_LENGTH_SEC / 60
    
    plotted = False
    for col, color, label in [
        ("ecg_sqi", '#E53935', 'ECG'),
        ("meta_ecg_sqi", '#E53935', 'ECG'),
    ]:
        if col in segment_features.columns:
            vals = segment_features[col].values
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(times[valid], vals[valid], '-',
                        color=color, linewidth=1, label=label)
                plotted = True
                break
    
    for region in ["central", "frontal", "occipital"]:
        col = f"eeg_{region}_sqi"
        if col in segment_features.columns:
            vals = segment_features[col].values
            valid = ~np.isnan(vals)
            if valid.any():
                ax.plot(times[valid], vals[valid], '-',
                        linewidth=1, label=f'EEG {region}')
                plotted = True
    
    if plotted:
        ax.axhline(y=ECG_SQI_THRESHOLD, color='red', linestyle=':',
                    linewidth=1, alpha=0.5)
        ax.legend(fontsize=8, ncol=4, loc='lower right')
    else:
        ax.text(0.5, 0.5, 'Keine SQI-Daten', ha='center', va='center')
    
    ax.set_ylabel('SQI')
    ax.set_xlabel('Zeit (min)')
    ax.set_title('Signalqualität', fontsize=12)
    ax.set_ylim([0, 1.05])


def _plot_sleep_summary_panel(ax, sleep_summary, patient_id):
    """Schlafarchitektur-Zusammenfassung als Textpanel."""
    ax.axis('off')
    
    if sleep_summary is None:
        ax.text(0.5, 0.5, 'Keine Schlafarchitektur-Daten',
                ha='center', va='center', fontsize=14, color='gray')
        return
    
    text_lines = [
        f"Patient: {patient_id}",
        "",
        f"TST:     {sleep_summary.get('total_sleep_time_min', 'N/A'):.1f} min"
            if isinstance(sleep_summary.get('total_sleep_time_min'), (int, float))
            else "TST: N/A",
        f"SE:      {sleep_summary.get('sleep_efficiency', 0)*100:.1f}%"
            if isinstance(sleep_summary.get('sleep_efficiency'), (int, float))
            else "SE: N/A",
        f"WASO:    {sleep_summary.get('waso_min', 'N/A'):.1f} min"
            if isinstance(sleep_summary.get('waso_min'), (int, float))
            else "WASO: N/A",
        f"SOL:     {sleep_summary.get('sleep_onset_latency_min', 'N/A'):.1f} min"
            if isinstance(sleep_summary.get('sleep_onset_latency_min'), (int, float))
            else "SOL: N/A",
        "",
        f"N1:      {sleep_summary.get('n1_pct_tst', 0):.1f}%"
            if isinstance(sleep_summary.get('n1_pct_tst'), (int, float))
            else "N1: N/A",
        f"N2:      {sleep_summary.get('n2_pct_tst', 0):.1f}%"
            if isinstance(sleep_summary.get('n2_pct_tst'), (int, float))
            else "N2: N/A",
        f"N3:      {sleep_summary.get('n3_pct_tst', 0):.1f}%"
            if isinstance(sleep_summary.get('n3_pct_tst'), (int, float))
            else "N3: N/A",
        f"REM:     {sleep_summary.get('rem_pct_tst', 0):.1f}%"
            if isinstance(sleep_summary.get('rem_pct_tst'), (int, float))
            else "REM: N/A",
        "",
        f"AHI:     {sleep_summary.get('ahi', 'N/A'):.1f}"
            if isinstance(sleep_summary.get('ahi'), (int, float))
            else "AHI: N/A",
        f"AI:      {sleep_summary.get('arousal_index', 'N/A'):.1f}"
            if isinstance(sleep_summary.get('arousal_index'), (int, float))
            else "AI: N/A",
        f"REM Lat: {sleep_summary.get('rem_latency_min', 'N/A'):.1f} min"
            if isinstance(sleep_summary.get('rem_latency_min'), (int, float))
            else "REM Lat: N/A",
    ]
    
    text = "\n".join(text_lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))


# ==============================================================================
# 5. KOHORTEN-ÜBERSICHT
# ==============================================================================

def plot_cohort_overview(
    patient_level_features: pd.DataFrame,
    target_col: str = "target",
    output_dir: Path = PLOT_DIR,
) -> Path:
    """
    Erstellt eine Kohorten-Übersicht mit demographischen
    und schlafmedizinischen Kennzahlen.
    """
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)
    
    has_target = target_col in patient_level_features.columns
    
    # --- 1. Target-Verteilung ---
    ax1 = fig.add_subplot(gs[0, 0])
    if has_target:
        counts = patient_level_features[target_col].value_counts()
        labels = ['No CI', 'CI']
        colors_pie = ['#1E88E5', '#E53935']
        ax1.pie(counts.values, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        ax1.set_title(f'Target (n={len(patient_level_features)})', fontsize=12)
    else:
        ax1.text(0.5, 0.5, f'n={len(patient_level_features)}',
                 ha='center', va='center', fontsize=16)
        ax1.set_title('Kohorte', fontsize=12)
    
    # --- 2. Altersverteilung ---
    ax2 = fig.add_subplot(gs[0, 1])
    age_col = None
    for candidate in ["demo_age", "Age"]:
        if candidate in patient_level_features.columns:
            age_col = candidate
            break
    
    if age_col:
        if has_target:
            for label, color, name in [(0, '#1E88E5', 'No CI'), (1, '#E53935', 'CI')]:
                data = patient_level_features[
                    patient_level_features[target_col] == label
                ][age_col].dropna()
                if len(data) > 0:
                    ax2.hist(data, bins=15, alpha=0.6, color=color,
                             label=name, density=True)
            ax2.legend(fontsize=8)
        else:
            patient_level_features[age_col].dropna().hist(
                bins=15, ax=ax2, color='#1E88E5', alpha=0.7
            )
        ax2.set_xlabel('Alter')
        ax2.set_title('Altersverteilung', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'Keine Altersdaten', ha='center', va='center')
        ax2.set_title('Altersverteilung', fontsize=12)
    
    # --- 3. BMI-Verteilung ---
    ax3 = fig.add_subplot(gs[0, 2])
    bmi_col = None
    for candidate in ["demo_bmi", "BMI"]:
        if candidate in patient_level_features.columns:
            bmi_col = candidate
            break
    
    if bmi_col:
        if has_target:
            for label, color, name in [(0, '#1E88E5', 'No CI'), (1, '#E53935', 'CI')]:
                data = patient_level_features[
                    patient_level_features[target_col] == label
                ][bmi_col].dropna()
                if len(data) > 0:
                    ax3.hist(data, bins=15, alpha=0.6, color=color,
                             label=name, density=True)
            ax3.legend(fontsize=8)
        else:
            patient_level_features[bmi_col].dropna().hist(
                bins=15, ax=ax3, color='#43A047', alpha=0.7
            )
        ax3.set_xlabel('BMI')
        ax3.set_title('BMI-Verteilung', fontsize=12)
    else:
        ax3.text(0.5, 0.5, 'Keine BMI-Daten', ha='center', va='center')
        ax3.set_title('BMI-Verteilung', fontsize=12)
    
    # --- 4. Geschlechterverteilung ---
    ax4 = fig.add_subplot(gs[0, 3])
    sex_col = None
    for candidate in ["demo_sex", "Sex"]:
        if candidate in patient_level_features.columns:
            sex_col = candidate
            break
    
    if sex_col:
        if has_target:
            sex_target = patient_level_features.groupby(
                [sex_col, target_col]
            ).size().unstack(fill_value=0)
            sex_labels = {0: 'Männlich', 1: 'Weiblich'}
            
            if len(sex_target) > 0:
                sex_target.index = [sex_labels.get(i, str(i)) for i in sex_target.index]
                sex_target.columns = ['No CI', 'CI']
                sex_target.plot(kind='bar', ax=ax4,
                                color=['#1E88E5', '#E53935'], alpha=0.7)
                ax4.legend(fontsize=8)
        else:
            counts = patient_level_features[sex_col].value_counts()
            labels_map = {0: 'Männlich', 1: 'Weiblich'}
            ax4.bar(
                [labels_map.get(i, str(i)) for i in counts.index],
                counts.values,
                color=['#1E88E5', '#E53935'][:len(counts)]
            )
        ax4.set_title('Geschlechterverteilung', fontsize=12)
        ax4.tick_params(axis='x', rotation=0)
    else:
        ax4.text(0.5, 0.5, 'Keine Geschlechtsdaten', ha='center', va='center')
        ax4.set_title('Geschlechterverteilung', fontsize=12)
    
    # --- 5. Schlafeffizienz ---
    ax5 = fig.add_subplot(gs[1, 0])
    se_col = None
    for candidate in ["sleep_sleep_efficiency", "ann_global_sleep_efficiency",
                       "all_ann_global_sleep_efficiency_mean"]:
        if candidate in patient_level_features.columns:
            se_col = candidate
            break
    
    if se_col:
        data = patient_level_features[se_col].dropna()
        if len(data) > 0:
            if has_target:
                for label, color, name in [(0, '#1E88E5', 'No CI'), (1, '#E53935', 'CI')]:
                    d = patient_level_features[
                        patient_level_features[target_col] == label
                    ][se_col].dropna()
                    if len(d) > 0:
                        ax5.hist(d, bins=15, alpha=0.6, color=color,
                                 label=name, density=True)
                ax5.legend(fontsize=8)
            else:
                ax5.hist(data, bins=15, color='#FF9800', alpha=0.7)
            ax5.set_xlabel('Schlafeffizienz')
            ax5.set_title('Schlafeffizienz', fontsize=12)
    else:
        ax5.text(0.5, 0.5, 'Keine SE-Daten', ha='center', va='center')
        ax5.set_title('Schlafeffizienz', fontsize=12)
    
    # --- 6. AHI-Verteilung ---
    ax6 = fig.add_subplot(gs[1, 1])
    ahi_col = None
    for candidate in ["sleep_ahi", "ann_global_ahi",
                       "all_ann_global_ahi_mean"]:
        if candidate in patient_level_features.columns:
            ahi_col = candidate
            break
    
    if ahi_col:
        data = patient_level_features[ahi_col].dropna()
        if len(data) > 0:
            if has_target:
                for label, color, name in [(0, '#1E88E5', 'No CI'), (1, '#E53935', 'CI')]:
                    d = patient_level_features[
                        patient_level_features[target_col] == label
                    ][ahi_col].dropna()
                    if len(d) > 0:
                        ax6.hist(d, bins=15, alpha=0.6, color=color,
                                 label=name, density=True)
                ax6.legend(fontsize=8)
            else:
                ax6.hist(data, bins=15, color='#AB47BC', alpha=0.7)
            ax6.set_xlabel('AHI (Events/h)')
            ax6.set_title('Apnoe-Hypopnoe-Index', fontsize=12)
    else:
        ax6.text(0.5, 0.5, 'Keine AHI-Daten', ha='center', va='center')
        ax6.set_title('AHI', fontsize=12)
    
    # --- 7. HRV-Vergleich (Boxplots) ---
    ax7 = fig.add_subplot(gs[1, 2:])
    hrv_cols = []
    for candidate in ["all_hrv_sdnn_mean", "all_hrv_rmssd_mean",
                       "all_hrv_hf_power_mean", "all_hrv_lf_hf_ratio_mean",
                       "all_hr_mean_mean"]:
        if candidate in patient_level_features.columns:
            hrv_cols.append(candidate)
    
    if hrv_cols and has_target:
        plot_data = []
        for col in hrv_cols:
            for label, name in [(0, 'No CI'), (1, 'CI')]:
                values = patient_level_features[
                    patient_level_features[target_col] == label
                ][col].dropna()
                for v in values:
                    plot_data.append({
                        'Feature': col.replace('all_', '').replace('_mean', ''),
                        'Value': v,
                        'Group': name,
                    })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.boxplot(data=plot_df, x='Feature', y='Value', hue='Group',
                        ax=ax7, palette={'No CI': '#1E88E5', 'CI': '#E53935'},
                        fliersize=2)
            ax7.tick_params(axis='x', rotation=30, labelsize=8)
            ax7.set_title('HRV-Parameter Vergleich', fontsize=12)
            ax7.legend(fontsize=8)
    else:
        ax7.text(0.5, 0.5, 'Keine HRV-Daten für Vergleich',
                 ha='center', va='center', fontsize=12, color='gray')
        ax7.set_title('HRV-Parameter', fontsize=12)
    
    # --- 8. EEG Slowing Markers ---
    ax8 = fig.add_subplot(gs[2, :2])
    slowing_cols = []
    for candidate in ["all_eeg_central_dar_mean",
                       "all_eeg_central_theta_alpha_ratio_mean",
                       "all_eeg_central_slowing_ratio_mean",
                       "all_eeg_central_spectral_entropy_mean"]:
        if candidate in patient_level_features.columns:
            slowing_cols.append(candidate)
    
    if slowing_cols and has_target:
        plot_data = []
        for col in slowing_cols:
            for label, name in [(0, 'No CI'), (1, 'CI')]:
                values = patient_level_features[
                    patient_level_features[target_col] == label
                ][col].dropna()
                for v in values:
                    short_name = (col.replace('all_eeg_central_', '')
                                  .replace('_mean', ''))
                    plot_data.append({
                        'Feature': short_name,
                        'Value': v,
                        'Group': name,
                    })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.violinplot(data=plot_df, x='Feature', y='Value', hue='Group',
                           ax=ax8, palette={'No CI': '#1E88E5', 'CI': '#E53935'},
                           split=True, inner='quartile', cut=0)
            ax8.tick_params(axis='x', rotation=20, labelsize=9)
            ax8.set_title('EEG Slowing Markers (CI vs. No CI)', fontsize=12)
            ax8.legend(fontsize=8)
    else:
        ax8.text(0.5, 0.5, 'Keine EEG-Slowing-Daten',
                 ha='center', va='center', fontsize=12, color='gray')
        ax8.set_title('EEG Slowing Markers', fontsize=12)
    
    # --- 9. Schlafstadien-Anteile ---
    ax9 = fig.add_subplot(gs[2, 2:])
    stage_pct_cols = {}
    for stage in ["n1", "n2", "n3", "rem", "w"]:
        for candidate in [f"sleep_{stage}_pct_tst", f"ann_global_{stage}_pct",
                           f"all_ann_global_{stage}_pct_mean"]:
            if candidate in patient_level_features.columns:
                stage_pct_cols[stage.upper()] = candidate
                break
    
    if stage_pct_cols and has_target:
        x = np.arange(len(stage_pct_cols))
        width = 0.35
        
        for label, offset, color, name in [
            (0, -width/2, '#1E88E5', 'No CI'),
            (1, width/2, '#E53935', 'CI')
        ]:
            means = []
            stds = []
            for stage, col in stage_pct_cols.items():
                data = patient_level_features[
                    patient_level_features[target_col] == label
                ][col].dropna()
                means.append(data.mean() if len(data) > 0 else 0)
                stds.append(data.std() if len(data) > 1 else 0)
            
            ax9.bar(x + offset, means, width, yerr=stds,
                    color=color, alpha=0.7, label=name, capsize=3)
        
        ax9.set_xticks(x)
        ax9.set_xticklabels(list(stage_pct_cols.keys()), fontsize=10)
        ax9.set_ylabel('% TST')
        ax9.set_title('Schlafstadien-Anteile (CI vs. No CI)', fontsize=12)
        ax9.legend(fontsize=9)
    elif stage_pct_cols:
        means = [patient_level_features[col].mean()
                 for col in stage_pct_cols.values()]
        stage_colors = [STAGE_COLORS.get(s, '#9E9E9E')
                        for s in stage_pct_cols.keys()]
        ax9.bar(list(stage_pct_cols.keys()), means,
                color=stage_colors, alpha=0.7)
        ax9.set_ylabel('% TST')
        ax9.set_title('Schlafstadien-Anteile', fontsize=12)
    else:
        ax9.text(0.5, 0.5, 'Keine Schlafstadien-Daten',
                 ha='center', va='center', fontsize=12, color='gray')
        ax9.set_title('Schlafstadien-Anteile', fontsize=12)
    
    fig.suptitle('Kohorten-Übersicht – PhysioNet Challenge 2026',
                 fontsize=16, fontweight='bold', y=1.01)
    
    path = output_dir / "cohort_overview.png"
    fig.savefig(path)
    plt.close(fig)
    return path


# ==============================================================================
# 6. UTILITY-FUNKTIONEN
# ==============================================================================

def plot_signal_segment(
    signal: np.ndarray,
    fs: float,
    title: str = "Signal",
    channel_name: str = "",
    events: Optional[List[Dict]] = None,
    output_dir: Path = PLOT_DIR,
    filename: Optional[str] = None,
) -> Path:
    """
    Einfacher Plot eines Signalsegments.
    Nützlich für schnelles Debugging.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    
    t = np.arange(len(signal)) / fs
    ax.plot(t, signal, color='#1E88E5', linewidth=0.5)
    
    if events:
        for event in events:
            start = event.get("start_sec", 0)
            end = event.get("end_sec", start + 1)
            color = event.get("color", "red")
            label = event.get("label", "Event")
            ax.axvspan(start, end, alpha=0.2, color=color, label=label)
    
    ax.set_xlabel('Zeit (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(f'{title} – {channel_name}' if channel_name else title)
    
    if events:
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8, loc='upper right')
    
    plt.tight_layout()
    
    if filename is None:
        filename = f"signal_{channel_name or 'unknown'}.png"
    path = output_dir / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all_patient_plots(
    patient_id: str,
    segment_features: pd.DataFrame,
    ecg_preprocessed: Optional[Dict] = None,
    eeg_signals: Optional[Dict] = None,
    eeg_strategies: Optional[Dict] = None,
    stages_df: Optional[pd.DataFrame] = None,
    events_df: Optional[pd.DataFrame] = None,
    sleep_summary: Optional[Dict] = None,
    sqi_ecg: Optional[pd.DataFrame] = None,
    sqi_eeg: Optional[Dict[str, pd.DataFrame]] = None,
    output_dir: Optional[Path] = None,
    logger=None,
) -> Dict[str, Path]:
    """
    Erstellt alle Visualisierungen für einen Patienten.
    
    Returns
    -------
    Dict[str, Path]
        Dictionary mit Plot-Namen und Dateipfaden.
    """
    if output_dir is None:
        output_dir = PLOT_DIR / patient_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    if logger:
        logger.info(f"Erstelle Visualisierungen für Patient {patient_id}")
    
    # --- ECG Preprocessing ---
    if ecg_preprocessed is not None and ecg_preprocessed.get("ecg_cleaned") is not None:
        try:
            path = plot_ecg_preprocessing(
                ecg_raw=ecg_preprocessed.get("ecg_raw",
                                              ecg_preprocessed["ecg_cleaned"]),
                ecg_cleaned=ecg_preprocessed["ecg_cleaned"],
                fs=ecg_preprocessed.get("fs", 256),
                rpeaks=ecg_preprocessed.get("rpeaks"),
                sqi_segments=ecg_preprocessed.get("sqi_per_segment"),
                patient_id=patient_id,
                output_dir=output_dir,
            )
            plot_paths["ecg_preprocessing"] = path
        except Exception as e:
            if logger:
                logger.warning(f"ECG-Plot fehlgeschlagen: {e}")
    
    # --- EEG Preprocessing ---
    if eeg_signals is not None and len(eeg_signals) > 0:
        try:
            path = plot_eeg_preprocessing(
                eeg_signals=eeg_signals,
                fs=list(eeg_signals.values())[0].get("fs", 256)
                    if isinstance(list(eeg_signals.values())[0], dict)
                    else 256,
                eeg_strategies=eeg_strategies,
                sqi_data=sqi_eeg,
                patient_id=patient_id,
                output_dir=output_dir,
            )
            plot_paths["eeg_preprocessing"] = path
        except Exception as e:
            if logger:
                logger.warning(f"EEG-Plot fehlgeschlagen: {e}")
    
    # --- Hypnogramm ---
    if stages_df is not None and len(stages_df) > 0:
        try:
            path = plot_hypnogram(
                stages_df=stages_df,
                events_df=events_df,
                sqi_ecg=sqi_ecg,
                sqi_eeg=sqi_eeg,
                patient_id=patient_id,
                output_dir=output_dir,
            )
            plot_paths["hypnogram"] = path
        except Exception as e:
            if logger:
                logger.warning(f"Hypnogramm-Plot fehlgeschlagen: {e}")
    
    # --- Patienten-Dashboard ---
    if segment_features is not None and len(segment_features) > 0:
        try:
            path = plot_patient_dashboard(
                patient_id=patient_id,
                segment_features=segment_features,
                sleep_summary=sleep_summary,
                output_dir=output_dir,
            )
            plot_paths["patient_dashboard"] = path
        except Exception as e:
            if logger:
                logger.warning(f"Dashboard-Plot fehlgeschlagen: {e}")
    
    if logger:
        logger.info(f"Plots erstellt: {len(plot_paths)} Visualisierungen "
                     f"in {output_dir}")
    
    plt.close('all')
    
    return plot_paths


def generate_all_cohort_plots(
    patient_level_features: pd.DataFrame,
    segment_level_features: Optional[pd.DataFrame] = None,
    target_col: str = "target",
    output_dir: Optional[Path] = None,
    logger=None,
) -> Dict[str, Path]:
    """
    Erstellt alle Kohorten-Visualisierungen.
    
    Returns
    -------
    Dict[str, Path]
        Dictionary mit Plot-Namen und Dateipfaden.
    """
    if output_dir is None:
        output_dir = PLOT_DIR / "cohort"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = {}
    
    if logger:
        logger.info("Erstelle Kohorten-Visualisierungen")
    
    # --- Kohorten-Übersicht ---
    try:
        path = plot_cohort_overview(
            patient_level_features, target_col, output_dir
        )
        plot_paths["cohort_overview"] = path
    except Exception as e:
        if logger:
            logger.warning(f"Kohorten-Übersicht fehlgeschlagen: {e}")
    
    # --- Feature-Verteilungen ---
    try:
        key_features = _select_key_visualization_features(patient_level_features)
        if key_features:
            path = plot_feature_distributions(
                patient_level_features, features=key_features,
                target_col=target_col, patient_id="cohort",
                output_dir=output_dir
            )
            plot_paths["feature_distributions"] = path
    except Exception as e:
        if logger:
            logger.warning(f"Feature-Verteilungen fehlgeschlagen: {e}")
    
    # --- Feature-Vergleich CI vs. Non-CI ---
    try:
        path = plot_feature_comparison_ci(
            patient_level_features, target_col=target_col,
            output_dir=output_dir
        )
        plot_paths["feature_comparison_ci"] = path
    except Exception as e:
        if logger:
            logger.warning(f"Feature-Vergleich fehlgeschlagen: {e}")
    
    # --- Korrelationsmatrix ---
    try:
        key_features = _select_key_visualization_features(
            patient_level_features, max_features=30
        )
        if key_features:
            path = plot_correlation_matrix(
                patient_level_features, features=key_features,
                output_dir=output_dir
            )
            plot_paths["correlation_matrix"] = path
    except Exception as e:
        if logger:
            logger.warning(f"Korrelationsmatrix fehlgeschlagen: {e}")
    
    # --- Spektrales Profil ---
    for region in ["central", "frontal", "occipital"]:
        try:
            region_cols = [c for c in patient_level_features.columns
                           if f"eeg_{region}" in c]
            if region_cols:
                path = plot_spectral_profile(
                    patient_level_features, region=region,
                    target_col=target_col, output_dir=output_dir
                )
                plot_paths[f"spectral_profile_{region}"] = path
        except Exception as e:
            if logger:
                logger.warning(f"Spektrales Profil [{region}] fehlgeschlagen: {e}")
    
    # --- Nacht-Drittel-Vergleich (nur Segment-Level) ---
    if segment_level_features is not None and len(segment_level_features) > 0:
        try:
            path = plot_night_third_comparison(
                segment_level_features, target_col=target_col,
                output_dir=output_dir
            )
            plot_paths["night_third_comparison"] = path
        except Exception as e:
            if logger:
                logger.warning(f"Nacht-Drittel-Vergleich fehlgeschlagen: {e}")
    
    if logger:
        logger.info(f"Kohorten-Plots erstellt: {len(plot_paths)} "
                     f"Visualisierungen in {output_dir}")
    
    plt.close('all')
    
    return plot_paths


def _select_key_visualization_features(
    df: pd.DataFrame,
    max_features: int = 20
) -> List[str]:
    """
    Wählt die wichtigsten Features für Visualisierungen aus.
    """
    key_patterns = [
        # HRV
        "hr_mean", "hrv_sdnn", "hrv_rmssd", "hrv_hf_power",
        "hrv_lf_hf_ratio", "hrv_sample_entropy",
        # RSA
        "rsa_p2t_mean", "rsa_coupling",
        # EEG
        "delta_power_rel", "theta_power_rel", "alpha_power_rel",
        "sigma_power_rel", "swa_power", "spindle_power",
        "spectral_entropy", "dar", "slowing_ratio",
        "theta_alpha_ratio", "sample_entropy",
        "hjorth_complexity", "permutation_entropy",
        # Schlaf
        "sleep_efficiency", "ahi", "arousal_index",
        "n3_pct", "rem_pct", "waso",
        # Demographics
        "demo_age", "demo_bmi",
    ]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude = ["patient_id", "segment_idx", "target", "demo_time_to_event"]
    
    selected = []
    for pattern in key_patterns:
        for col in numeric_cols:
            if pattern in col and col not in exclude and col not in selected:
                selected.append(col)
                break
        if len(selected) >= max_features:
            break
    
    return selected

