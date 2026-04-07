#!/usr/bin/env python
"""
calc_shap_value.py
====================
Calculate SHAP values for the trained XGBoost model.

Usage:
    python calc_shap_value.py

Model directory: outputs/models/model_xgboost_20260331_222925
Features file:   outputs/features/features_patient_level.parquet
"""

import numpy as np
import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats as scipy_stats
from itertools import combinations
import sys


# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_DIR = Path("output/models/model_xgboost_20260331_222925")
FEATURES_PATH = Path("output/features/features_patient_level.parquet")
OUTPUT_DIR = Path("output/shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 1. LOAD MODEL ARTIFACTS
# ==============================================================================

def load_model_artifacts(model_dir: Path) -> dict:
    """
    Load all model artifacts saved during training.
    
    The training pipeline (train_model.py -> _save_model) saves individual
    joblib files and JSON configs into the model directory [[17]].
    """
    artifacts = {}

    # Load model
    artifacts["model"] = joblib.load(model_dir / "model.joblib")

    # Load scaler
    artifacts["scaler"] = joblib.load(model_dir / "scaler.joblib")

    # Load imputer
    artifacts["imputer"] = joblib.load(model_dir / "imputer.joblib")

    # Load feature selector (if exists)
    selector_path = model_dir / "feature_selector.joblib"
    if selector_path.exists():
        artifacts["feature_selector"] = joblib.load(selector_path)
    else:
        artifacts["feature_selector"] = None

    # Load feature names
    with open(model_dir / "feature_names.json", "r") as f:
        feature_data = json.load(f)
        artifacts["feature_names"] = feature_data["all_features"]
        artifacts["selected_features"] = feature_data["selected_features"]

    # Load training config
    config_path = model_dir / "training_config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            artifacts["training_config"] = json.load(f)

    # Load feature importance (for comparison with SHAP)
    importance_path = model_dir / "feature_importance.csv"
    if importance_path.exists():
        artifacts["feature_importance"] = pd.read_csv(importance_path)

    print(f"Model loaded from: {model_dir}")
    print(f"  Model type: {artifacts['training_config'].get('model_type', 'unknown')}")
    print(f"  All features: {len(artifacts['feature_names'])}")
    print(f"  Selected features: {len(artifacts['selected_features'])}")

    return artifacts


# ==============================================================================
# 2. PREPARE DATA (mirrors _predict_single_patient pipeline from team_code.py)
# ==============================================================================

def prepare_data(
    features_path: Path,
    artifacts: dict
) -> pd.DataFrame:
    """
    Load patient-level features and apply the same preprocessing pipeline
    used during inference: impute -> feature select -> scale.
    
    This mirrors the _predict_single_patient function in team_code.py [[11]].
    """
    feature_names = artifacts["feature_names"]
    selected_features = artifacts["selected_features"]
    imputer = artifacts["imputer"]
    feature_selector = artifacts["feature_selector"]
    scaler = artifacts["scaler"]

    # Load patient-level feature table
    patient_level = pd.read_parquet(features_path)
    print(f"Loaded feature table: {patient_level.shape[0]} patients, "
          f"{patient_level.shape[1]} columns")

    # Ensure all expected features exist (fill missing with NaN)
    for f in feature_names:
        if f not in patient_level.columns:
            patient_level[f] = np.nan

    X = patient_level[feature_names].values.astype(np.float64)

    # Step 1: Impute missing values
    X_imputed = imputer.transform(X)

    # Step 2: Feature selection
    if feature_selector is not None:
        try:
            X_selected = feature_selector.transform(X_imputed)
        except Exception:
            # Fallback: manually select columns by name
            sel_idx = [feature_names.index(f) for f in selected_features
                       if f in feature_names]
            X_selected = X_imputed[:, sel_idx]
    else:
        X_selected = X_imputed

    # Step 3: Scale
    X_scaled = scaler.transform(X_selected)

    # Create DataFrame with selected feature names for interpretability
    X_df = pd.DataFrame(X_scaled, columns=selected_features)

    # Preserve patient_id if available
    if "patient_id" in patient_level.columns:
        X_df.index = patient_level["patient_id"].values

    print(f"Preprocessed data: {X_df.shape[0]} patients, "
          f"{X_df.shape[1]} features")

    return X_df, patient_level


# ==============================================================================
# 3. CALCULATE SHAP VALUES
# ==============================================================================
def calculate_shap_values(
    model,
    X_df: pd.DataFrame
) -> shap.Explanation:
    """
    Calculate SHAP values for XGBoost.
    Patches the SHAP source file on disk to fix the base_score parsing bug,
    then restarts the module from scratch.
    """
    print("Calculating SHAP values with TreeExplainer...")

    import shap.explainers._tree as _tree_module
    import importlib

    source_file = _tree_module.__file__
    print(f"  SHAP source file: {source_file}")

    # Read the source
    with open(source_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find and patch ALL lines that do float(learner_model_param["base_score"])
    patched = False
    for i, line in enumerate(lines):
        if ('float(learner_model_param["base_score"])' in line
                and '.strip' not in line):
            original = line
            lines[i] = line.replace(
                'float(learner_model_param["base_score"])',
                'float(str(learner_model_param["base_score"]).strip("[]"))'
            )
            print(f"  Patched line {i+1}:")
            print(f"    OLD: {original.rstrip()}")
            print(f"    NEW: {lines[i].rstrip()}")
            patched = True

    if patched:
        with open(source_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("  Source file written. Reloading module...")

        # Force Python to forget the old compiled module
        # Remove all cached references
        mods_to_remove = [k for k in sys.modules if 'shap' in k]
        for mod_name in mods_to_remove:
            del sys.modules[mod_name]

        # Re-import shap from scratch
        import shap as shap_reloaded
        global shap
        shap = shap_reloaded
        print("  SHAP module reloaded from patched source.")
    else:
        print("  Source already patched or pattern not found.")

    # Now create the explainer with the patched code
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df)

    # Handle binary classification output shape
    if shap_values.values.ndim == 3:
        shap_explanation = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1]
                if shap_values.base_values.ndim == 2
                else shap_values.base_values,
            data=shap_values.data,
            feature_names=X_df.columns.tolist(),
        )
    else:
        shap_explanation = shap_values

    print(f"SHAP values calculated: {shap_explanation.values.shape}")
    print(f"Base value (expected value): {np.mean(shap_explanation.base_values):.4f}")

    return shap_explanation, explainer


# ==============================================================================
# HELPER: Extract site ID from patient_id
# ==============================================================================

def extract_site_ids(patient_ids):
    """
    Extract site ID from patient_id strings.
    The site is encoded in the first 5 characters of the patient_id,
    e.g. 'I0002/sub-I0002150000686_ses-1' -> 'I0002'
    """
    sites = []
    for pid in patient_ids:
        pid_str = str(pid)
        # The site ID is the first 5 characters
        site = pid_str[:5]
        sites.append(site)
    return np.array(sites)


# ==============================================================================
# 4. VISUALIZATIONS
# ==============================================================================

def plot_shap_summary_bar(shap_explanation, X_df, output_dir):
    """Global feature importance bar plot (mean |SHAP|)."""
    plt.figure(figsize=(12, 10))
    shap.plots.bar(shap_explanation, max_display=6, show=False)
    plt.title("SHAP Feature Importance (mean |SHAP value|)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'shap_summary_bar.png'}")


def plot_shap_beeswarm(shap_explanation, X_df, output_dir):
    """Beeswarm plot showing direction and magnitude per feature."""
    max_display = 5

    # Calculate mean absolute shap values to find top features
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)

    # Get feature names from the explanation object
    feature_names = shap_explanation.feature_names

    # Create a DataFrame to sort features by importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    top_features = feature_importance['feature'].head(max_display).tolist()

    plt.figure(figsize=(12, 10))
    # Use the sliced explanation object for plotting to show only top features
    shap.plots.beeswarm(shap_explanation[:, top_features], show=False)
    plt.title("SHAP Beeswarm Plot")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm.svg", bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'shap_beeswarm.svg'}")


def plot_shap_beeswarm_by_site(shap_explanation, X_df, output_dir):
    """
    Beeswarm plot for the top 5 features, colored by site ID.
    
    Each dot represents one patient. Instead of coloring by feature value,
    dots are colored by the site (first 5 characters of patient_id).
    This helps identify whether the visible clusters in the standard
    beeswarm plot correspond to the three data collection sites.
    """
    max_display = 5

    # --- Determine top features ---
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)
    feature_names = shap_explanation.feature_names

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    top_features = feature_importance['feature'].head(max_display).tolist()
    top_indices = [feature_names.index(f) for f in top_features]

    # --- Extract site IDs ---
    patient_ids = X_df.index.values
    sites = extract_site_ids(patient_ids)
    unique_sites = sorted(np.unique(sites))

    print(f"Sites found: {unique_sites}")
    for s in unique_sites:
        print(f"  {s}: {np.sum(sites == s)} patients")

    # --- Define site colors ---
    site_colors_map = {}
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, site in enumerate(unique_sites):
        site_colors_map[site] = color_palette[i % len(color_palette)]

    patient_colors = np.array([site_colors_map[s] for s in sites])

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(12, 8))

    n_features = len(top_features)

    for feat_idx_plot, feat_name in enumerate(reversed(top_features)):
        feat_col_idx = feature_names.index(feat_name)
        shap_vals = shap_explanation.values[:, feat_col_idx]

        y_pos = feat_idx_plot  # y position for this feature row

        # Add jitter to avoid overplotting
        np.random.seed(feat_col_idx)
        jitter = np.random.normal(0, 0.12, size=len(shap_vals))

        for site in unique_sites:
            mask = sites == site
            ax.scatter(
                shap_vals[mask],
                y_pos + jitter[mask],
                c=site_colors_map[site],
                s=12,
                alpha=0.6,
                edgecolors='none',
                label=site if feat_idx_plot == 0 else None,
                rasterized=True,
            )

    # --- Formatting ---
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(list(reversed(top_features)), fontsize=11)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='-')
    ax.set_title("SHAP Beeswarm Plot — Colored by Site", fontsize=14)

    # Legend
    handles = [mpatches.Patch(color=site_colors_map[s], label=s) for s in unique_sites]
    ax.legend(handles=handles, title="Site", loc='lower right', fontsize=10,
              title_fontsize=11, framealpha=0.9)

    plt.tight_layout()
    save_path = output_dir / "shap_beeswarm_by_site.svg"
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

    # Also save as PNG for quick viewing
    fig, ax = plt.subplots(figsize=(12, 8))
    for feat_idx_plot, feat_name in enumerate(reversed(top_features)):
        feat_col_idx = feature_names.index(feat_name)
        shap_vals = shap_explanation.values[:, feat_col_idx]
        y_pos = feat_idx_plot
        np.random.seed(feat_col_idx)
        jitter = np.random.normal(0, 0.12, size=len(shap_vals))
        for site in unique_sites:
            mask = sites == site
            ax.scatter(
                shap_vals[mask],
                y_pos + jitter[mask],
                c=site_colors_map[site],
                s=12,
                alpha=0.6,
                edgecolors='none',
                label=site if feat_idx_plot == 0 else None,
                rasterized=True,
            )
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(list(reversed(top_features)), fontsize=11)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='-')
    ax.set_title("SHAP Beeswarm Plot — Colored by Site", fontsize=14)
    handles = [mpatches.Patch(color=site_colors_map[s], label=s) for s in unique_sites]
    ax.legend(handles=handles, title="Site", loc='lower right', fontsize=10,
              title_fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_beeswarm_by_site.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'shap_beeswarm_by_site.png'}")


# ==============================================================================
# 4b. STATISTICAL ANALYSIS: SHAP values by site
# ==============================================================================

def analyze_shap_by_site(shap_explanation, X_df, output_dir):
    """
    Statistical analysis to test whether SHAP values for the top 5 features
    differ significantly between sites.
    
    For each top feature:
      1. Kruskal-Wallis H-test (non-parametric, k>=2 groups)
      2. Pairwise Mann-Whitney U tests with Bonferroni correction
      3. Effect size: rank-biserial correlation (r = 1 - 2U/(n1*n2))
      4. Descriptive statistics per site
    
    Results are printed and saved to CSV and a text report.
    """
    max_display = 5

    # --- Determine top features ---
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)
    feature_names = shap_explanation.feature_names

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    top_features = feature_importance['feature'].head(max_display).tolist()

    # --- Extract site IDs ---
    patient_ids = X_df.index.values
    sites = extract_site_ids(patient_ids)
    unique_sites = sorted(np.unique(sites))
    n_sites = len(unique_sites)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: SHAP values by Site")
    print("=" * 70)
    print(f"Sites: {unique_sites}")
    for s in unique_sites:
        print(f"  {s}: n={np.sum(sites == s)}")
    print()

    # Number of pairwise comparisons for Bonferroni correction
    n_comparisons = n_sites * (n_sites - 1) // 2

    all_results = []
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("STATISTICAL ANALYSIS: SHAP values by Site")
    report_lines.append("=" * 70)
    report_lines.append(f"Sites: {unique_sites}")
    for s in unique_sites:
        report_lines.append(f"  {s}: n={np.sum(sites == s)}")
    report_lines.append(f"Bonferroni correction: {n_comparisons} pairwise comparisons per feature")
    report_lines.append("")

    for feat_name in top_features:
        feat_col_idx = feature_names.index(feat_name)
        shap_vals = shap_explanation.values[:, feat_col_idx]

        report_lines.append("-" * 70)
        report_lines.append(f"Feature: {feat_name}")
        report_lines.append("-" * 70)
        print(f"\n--- Feature: {feat_name} ---")

        # Group SHAP values by site
        groups = {}
        for site in unique_sites:
            mask = sites == site
            groups[site] = shap_vals[mask]

        # Descriptive statistics
        report_lines.append("  Descriptive statistics:")
        print("  Descriptive statistics:")
        for site in unique_sites:
            vals = groups[site]
            desc = (f"    {site}: n={len(vals)}, "
                    f"mean={np.mean(vals):.4f}, "
                    f"median={np.median(vals):.4f}, "
                    f"std={np.std(vals):.4f}, "
                    f"min={np.min(vals):.4f}, "
                    f"max={np.max(vals):.4f}")
            print(desc)
            report_lines.append(desc)

        # --- Kruskal-Wallis H-test ---
        group_arrays = [groups[s] for s in unique_sites]
        if n_sites >= 2:
            kw_stat, kw_p = scipy_stats.kruskal(*group_arrays)
        else:
            kw_stat, kw_p = np.nan, np.nan

        kw_line = (f"  Kruskal-Wallis H-test: H={kw_stat:.4f}, "
                   f"p={kw_p:.2e}, "
                   f"{'*** SIGNIFICANT' if kw_p < 0.05 else 'not significant'} "
                   f"(alpha=0.05)")
        print(kw_line)
        report_lines.append(kw_line)

        result_row = {
            'feature': feat_name,
            'kruskal_wallis_H': kw_stat,
            'kruskal_wallis_p': kw_p,
            'kruskal_wallis_significant': kw_p < 0.05,
        }

        # Add per-site descriptive stats
        for site in unique_sites:
            vals = groups[site]
            result_row[f'{site}_n'] = len(vals)
            result_row[f'{site}_mean_shap'] = np.mean(vals)
            result_row[f'{site}_median_shap'] = np.median(vals)
            result_row[f'{site}_std_shap'] = np.std(vals)

        # --- Pairwise Mann-Whitney U tests ---
        if kw_p < 0.05 and n_sites >= 2:
            report_lines.append("  Pairwise Mann-Whitney U tests (Bonferroni corrected):")
            print("  Pairwise Mann-Whitney U tests (Bonferroni corrected):")

            for site_a, site_b in combinations(unique_sites, 2):
                u_stat, u_p = scipy_stats.mannwhitneyu(
                    groups[site_a], groups[site_b], alternative='two-sided'
                )
                # Bonferroni correction
                u_p_corrected = min(u_p * n_comparisons, 1.0)

                # Effect size: rank-biserial correlation
                n1, n2 = len(groups[site_a]), len(groups[site_b])
                r_effect = 1 - (2 * u_stat) / (n1 * n2)

                sig_marker = ""
                if u_p_corrected < 0.001:
                    sig_marker = "***"
                elif u_p_corrected < 0.01:
                    sig_marker = "**"
                elif u_p_corrected < 0.05:
                    sig_marker = "*"
                else:
                    sig_marker = "n.s."

                pair_line = (f"    {site_a} vs {site_b}: "
                             f"U={u_stat:.1f}, "
                             f"p={u_p:.2e}, "
                             f"p_corrected={u_p_corrected:.2e} {sig_marker}, "
                             f"effect_size_r={r_effect:.4f}")
                print(pair_line)
                report_lines.append(pair_line)

                result_row[f'mwu_{site_a}_vs_{site_b}_U'] = u_stat
                result_row[f'mwu_{site_a}_vs_{site_b}_p'] = u_p
                result_row[f'mwu_{site_a}_vs_{site_b}_p_corrected'] = u_p_corrected
                result_row[f'mwu_{site_a}_vs_{site_b}_significant'] = u_p_corrected < 0.05
                result_row[f'mwu_{site_a}_vs_{site_b}_effect_r'] = r_effect
        else:
            report_lines.append("  Pairwise tests skipped (Kruskal-Wallis not significant).")
            print("  Pairwise tests skipped (Kruskal-Wallis not significant).")

        all_results.append(result_row)
        report_lines.append("")

    # --- Summary ---
    n_significant = sum(1 for r in all_results if r['kruskal_wallis_significant'])
    summary_line = (f"\nSUMMARY: {n_significant}/{len(all_results)} top features show "
                    f"statistically significant differences in SHAP values between sites.")
    print(summary_line)
    report_lines.append(summary_line)

    if n_significant > 0:
        warning = ("WARNING: Site-dependent SHAP value distributions suggest the model "
                   "may be capturing site-specific effects rather than (or in addition to) "
                   "genuine biomarkers. Consider adding site as a covariate, using "
                   "domain adaptation, or harmonizing features across sites.")
        print(warning)
        report_lines.append(warning)
    else:
        ok_msg = ("OK: No significant site differences detected in SHAP values for the "
                  "top features. The observed clusters are likely not driven by site.")
        print(ok_msg)
        report_lines.append(ok_msg)

    # --- Save results ---
    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / "shap_site_analysis.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    report_path = output_dir / "shap_site_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved: {report_path}")

    return results_df


def plot_shap_waterfall(shap_explanation, patient_idx, output_dir):
    """Waterfall plot for a single patient prediction."""
    plt.figure(figsize=(12, 8))
    shap.plots.waterfall(shap_explanation[patient_idx], max_display=6, show=False)
    plt.title(f"SHAP Waterfall — Patient {patient_idx}")
    plt.tight_layout()
    plt.savefig(
        output_dir / f"shap_waterfall_patient_{patient_idx}.png",
        dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {output_dir / f'shap_waterfall_patient_{patient_idx}.png'}")


def plot_shap_heatmap(shap_explanation, output_dir):
    """Heatmap of SHAP values across all patients."""
    plt.figure(figsize=(16, 10))
    shap.plots.heatmap(shap_explanation, max_display=5, show=False)
    plt.title("SHAP Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'shap_heatmap.png'}")


def plot_shap_dependence(shap_explanation, X_df, feature_name, output_dir):
    """Dependence plot for a specific feature."""
    if feature_name not in X_df.columns:
        print(f"Feature '{feature_name}' not found. Skipping dependence plot.")
        return
    plt.figure(figsize=(10, 6))
    shap.plots.scatter(
        shap_explanation[:, feature_name],
        color=shap_explanation,
        show=False
    )
    plt.title(f"SHAP Dependence — {feature_name}")
    plt.tight_layout()
    safe_name = feature_name.replace("/", "_").replace(" ", "_")
    plt.savefig(
        output_dir / f"shap_dependence_{safe_name}.png",
        dpi=150, bbox_inches="tight"
    )
    plt.close()
    print(f"Saved: {output_dir / f'shap_dependence_{safe_name}.png'}")


# ==============================================================================
# 5. EXPORT SHAP VALUES
# ==============================================================================

def export_shap_values(shap_explanation, X_df, output_dir):
    """Export SHAP values to CSV and parquet for further analysis."""
    # SHAP values DataFrame
    shap_df = pd.DataFrame(
        shap_explanation.values,
        columns=X_df.columns,
        index=X_df.index,
    )
    shap_df.to_csv(output_dir / "shap_values.csv")
    shap_df.to_parquet(output_dir / "shap_values.parquet")
    print(f"Saved: {output_dir / 'shap_values.csv'}")
    print(f"Saved: {output_dir / 'shap_values.parquet'}")

    # Mean absolute SHAP values (global importance ranking)
    mean_abs_shap = pd.DataFrame({
        "feature": X_df.columns,
        "mean_abs_shap": np.mean(np.abs(shap_explanation.values), axis=0),
        "mean_shap": np.mean(shap_explanation.values, axis=0),
        "std_shap": np.std(shap_explanation.values, axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    mean_abs_shap["rank"] = range(1, len(mean_abs_shap) + 1)
    mean_abs_shap["cumulative_importance"] = (
        mean_abs_shap["mean_abs_shap"].cumsum() /
        mean_abs_shap["mean_abs_shap"].sum()
    )

    mean_abs_shap.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    print(f"Saved: {output_dir / 'shap_feature_importance.csv'}")

    # Print top 5 features
    print("\nTop 5 Features by mean |SHAP value|:")
    print("-" * 70)
    for _, row in mean_abs_shap.head(5).iterrows():
        direction = "↑ CI" if row["mean_shap"] > 0 else "↓ CI"
        print(
            f"  {int(row['rank']):3d}. {row['feature']:<45s} "
            f"|SHAP|={row['mean_abs_shap']:.4f}  ({direction})"
        )

    return mean_abs_shap


# ==============================================================================
# HELPER: Extract site ID from patient_id
# ==============================================================================

def extract_site_ids(patient_ids):
    """
    Extract site ID from patient_id strings.
    The site is encoded in the first 5 characters of the patient_id,
    e.g. 'I0002/sub-I0002150000686_ses-1' -> 'I0002'
    """
    sites = []
    for pid in patient_ids:
        pid_str = str(pid)
        site = pid_str[:5]
        sites.append(site)
    return np.array(sites)


# ==============================================================================
# NEW: Site-colored beeswarm plot
# ==============================================================================

def plot_shap_beeswarm_by_site(shap_explanation, X_df, output_dir):
    """
    Beeswarm plot for the top 5 features, colored by site ID.

    Each dot represents one patient. Instead of coloring by feature value,
    dots are colored by the site (first 5 characters of patient_id).
    This helps identify whether the visible clusters in the standard
    beeswarm plot correspond to the three data collection sites.
    """
    import matplotlib.patches as mpatches

    max_display = 5

    # --- Determine top features ---
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)
    feature_names = shap_explanation.feature_names

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    top_features = feature_importance['feature'].head(max_display).tolist()

    # --- Extract site IDs ---
    patient_ids = X_df.index.values
    sites = extract_site_ids(patient_ids)
    unique_sites = sorted(np.unique(sites))

    print(f"\nSites found: {unique_sites}")
    for s in unique_sites:
        print(f"  {s}: {np.sum(sites == s)} patients")

    # --- Define site colors ---
    site_colors_map = {}
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i, site in enumerate(unique_sites):
        site_colors_map[site] = color_palette[i % len(color_palette)]

    # --- Create the plot ---
    fig, ax = plt.subplots(figsize=(12, 8))

    n_features = len(top_features)

    for feat_idx_plot, feat_name in enumerate(reversed(top_features)):
        feat_col_idx = feature_names.index(feat_name)
        shap_vals = shap_explanation.values[:, feat_col_idx]

        y_pos = feat_idx_plot  # y position for this feature row

        # Add jitter to avoid overplotting
        np.random.seed(feat_col_idx)
        jitter = np.random.normal(0, 0.12, size=len(shap_vals))

        for site in unique_sites:
            mask = sites == site
            ax.scatter(
                shap_vals[mask],
                y_pos + jitter[mask],
                c=site_colors_map[site],
                s=12,
                alpha=0.6,
                edgecolors='none',
                label=site if feat_idx_plot == 0 else None,
                rasterized=True,
            )

    # --- Formatting ---
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(list(reversed(top_features)), fontsize=11)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=12)
    ax.axvline(x=0, color='grey', linewidth=0.8, linestyle='-')
    ax.set_title("SHAP Beeswarm Plot — Colored by Site", fontsize=14)

    # Legend
    handles = [mpatches.Patch(color=site_colors_map[s], label=s)
               for s in unique_sites]
    ax.legend(handles=handles, title="Site", loc='lower right', fontsize=10,
              title_fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # Save SVG
    save_path_svg = output_dir / "shap_beeswarm_by_site.svg"
    plt.savefig(save_path_svg, bbox_inches="tight")
    print(f"Saved: {save_path_svg}")

    # Save PNG
    save_path_png = output_dir / "shap_beeswarm_by_site.png"
    plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path_png}")


# ==============================================================================
# NEW: Statistical analysis — SHAP values by site
# ==============================================================================

def analyze_shap_by_site(shap_explanation, X_df, output_dir):
    """
    Statistical analysis to test whether SHAP values for the top 5 features
    differ significantly between sites.

    For each top feature:
      1. Kruskal-Wallis H-test (non-parametric, k>=2 groups)
      2. Pairwise Mann-Whitney U tests with Bonferroni correction
      3. Effect size: rank-biserial correlation (r = 1 - 2U/(n1*n2))
      4. Descriptive statistics per site

    Results are printed and saved to CSV and a text report.
    """
    from scipy import stats as scipy_stats
    from itertools import combinations

    max_display = 5

    # --- Determine top features ---
    mean_abs_shap = np.abs(shap_explanation.values).mean(axis=0)
    feature_names = shap_explanation.feature_names

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    top_features = feature_importance['feature'].head(max_display).tolist()

    # --- Extract site IDs ---
    patient_ids = X_df.index.values
    sites = extract_site_ids(patient_ids)
    unique_sites = sorted(np.unique(sites))
    n_sites = len(unique_sites)

    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS: SHAP values by Site")
    print("=" * 70)
    print(f"Sites: {unique_sites}")
    for s in unique_sites:
        print(f"  {s}: n={np.sum(sites == s)}")
    print()

    # Number of pairwise comparisons for Bonferroni correction
    n_comparisons = n_sites * (n_sites - 1) // 2

    all_results = []
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("STATISTICAL ANALYSIS: SHAP values by Site")
    report_lines.append("=" * 70)
    report_lines.append(f"Sites: {unique_sites}")
    for s in unique_sites:
        report_lines.append(f"  {s}: n={np.sum(sites == s)}")
    report_lines.append(
        f"Bonferroni correction: {n_comparisons} pairwise comparisons "
        f"per feature"
    )
    report_lines.append("")

    for feat_name in top_features:
        feat_col_idx = feature_names.index(feat_name)
        shap_vals = shap_explanation.values[:, feat_col_idx]

        report_lines.append("-" * 70)
        report_lines.append(f"Feature: {feat_name}")
        report_lines.append("-" * 70)
        print(f"\n--- Feature: {feat_name} ---")

        # Group SHAP values by site
        groups = {}
        for site in unique_sites:
            mask = sites == site
            groups[site] = shap_vals[mask]

        # Descriptive statistics
        report_lines.append("  Descriptive statistics:")
        print("  Descriptive statistics:")
        for site in unique_sites:
            vals = groups[site]
            desc = (f"    {site}: n={len(vals)}, "
                    f"mean={np.mean(vals):.4f}, "
                    f"median={np.median(vals):.4f}, "
                    f"std={np.std(vals):.4f}, "
                    f"min={np.min(vals):.4f}, "
                    f"max={np.max(vals):.4f}")
            print(desc)
            report_lines.append(desc)

        # --- Kruskal-Wallis H-test ---
        group_arrays = [groups[s] for s in unique_sites]
        if n_sites >= 2:
            kw_stat, kw_p = scipy_stats.kruskal(*group_arrays)
        else:
            kw_stat, kw_p = np.nan, np.nan

        kw_line = (f"  Kruskal-Wallis H-test: H={kw_stat:.4f}, "
                   f"p={kw_p:.2e}, "
                   f"{'*** SIGNIFICANT' if kw_p < 0.05 else 'not significant'} "
                   f"(alpha=0.05)")
        print(kw_line)
        report_lines.append(kw_line)

        result_row = {
            'feature': feat_name,
            'kruskal_wallis_H': kw_stat,
            'kruskal_wallis_p': kw_p,
            'kruskal_wallis_significant': kw_p < 0.05,
        }

        # Add per-site descriptive stats
        for site in unique_sites:
            vals = groups[site]
            result_row[f'{site}_n'] = len(vals)
            result_row[f'{site}_mean_shap'] = np.mean(vals)
            result_row[f'{site}_median_shap'] = np.median(vals)
            result_row[f'{site}_std_shap'] = np.std(vals)

        # --- Pairwise Mann-Whitney U tests ---
        if kw_p < 0.05 and n_sites >= 2:
            report_lines.append(
                "  Pairwise Mann-Whitney U tests (Bonferroni corrected):"
            )
            print("  Pairwise Mann-Whitney U tests (Bonferroni corrected):")

            for site_a, site_b in combinations(unique_sites, 2):
                u_stat, u_p = scipy_stats.mannwhitneyu(
                    groups[site_a], groups[site_b], alternative='two-sided'
                )
                # Bonferroni correction
                u_p_corrected = min(u_p * n_comparisons, 1.0)

                # Effect size: rank-biserial correlation
                n1, n2 = len(groups[site_a]), len(groups[site_b])
                r_effect = 1 - (2 * u_stat) / (n1 * n2)

                sig_marker = ""
                if u_p_corrected < 0.001:
                    sig_marker = "***"
                elif u_p_corrected < 0.01:
                    sig_marker = "**"
                elif u_p_corrected < 0.05:
                    sig_marker = "*"
                else:
                    sig_marker = "n.s."

                pair_line = (
                    f"    {site_a} vs {site_b}: "
                    f"U={u_stat:.1f}, "
                    f"p={u_p:.2e}, "
                    f"p_corrected={u_p_corrected:.2e} {sig_marker}, "
                    f"effect_size_r={r_effect:.4f}"
                )
                print(pair_line)
                report_lines.append(pair_line)

                result_row[f'mwu_{site_a}_vs_{site_b}_U'] = u_stat
                result_row[f'mwu_{site_a}_vs_{site_b}_p'] = u_p
                result_row[f'mwu_{site_a}_vs_{site_b}_p_corrected'] = u_p_corrected
                result_row[f'mwu_{site_a}_vs_{site_b}_significant'] = (
                    u_p_corrected < 0.05
                )
                result_row[f'mwu_{site_a}_vs_{site_b}_effect_r'] = r_effect
        else:
            report_lines.append(
                "  Pairwise tests skipped (Kruskal-Wallis not significant)."
            )
            print("  Pairwise tests skipped (Kruskal-Wallis not significant).")

        all_results.append(result_row)
        report_lines.append("")

    # --- Summary ---
    n_significant = sum(
        1 for r in all_results if r['kruskal_wallis_significant']
    )
    summary_line = (
        f"\nSUMMARY: {n_significant}/{len(all_results)} top features show "
        f"statistically significant differences in SHAP values between sites."
    )
    print(summary_line)
    report_lines.append(summary_line)

    if n_significant > 0:
        warning = (
            "WARNING: Site-dependent SHAP value distributions suggest the "
            "model may be capturing site-specific effects rather than (or in "
            "addition to) genuine biomarkers. Consider adding site as a "
            "covariate, using domain adaptation, or harmonizing features "
            "across sites (e.g. ComBat)."
        )
        print(warning)
        report_lines.append(warning)
    else:
        ok_msg = (
            "OK: No significant site differences detected in SHAP values "
            "for the top features. The observed clusters are likely not "
            "driven by site."
        )
        print(ok_msg)
        report_lines.append(ok_msg)

    # --- Save results ---
    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / "shap_site_analysis.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    report_path = output_dir / "shap_site_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved: {report_path}")

    return results_df


# ==============================================================================
# 6. MAIN
# ==============================================================================

def main():
    print("=" * 70)
    print("SHAP VALUE CALCULATION")
    print(f"Model: {MODEL_DIR}")
    print(f"Features: {FEATURES_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    # --- Load model artifacts ---
    artifacts = load_model_artifacts(MODEL_DIR)
    model = artifacts["model"]

    # --- Prepare data ---
    X_df, patient_level = prepare_data(FEATURES_PATH, artifacts)

    # --- Calculate SHAP values ---
    shap_explanation, explainer = calculate_shap_values(model, X_df)

    # --- Export SHAP values ---
    mean_abs_shap = export_shap_values(shap_explanation, X_df, OUTPUT_DIR)

    # --- Generate plots ---
    print("\nGenerating SHAP plots...")

    # 1. Bar plot (global importance)
    plot_shap_summary_bar(shap_explanation, X_df, OUTPUT_DIR)

    # 2. Beeswarm plot (direction + magnitude, standard coloring)
    plot_shap_beeswarm(shap_explanation, X_df, OUTPUT_DIR)

    # 3. NEW: Beeswarm plot colored by site
    plot_shap_beeswarm_by_site(shap_explanation, X_df, OUTPUT_DIR)

    # 4. NEW: Statistical analysis of SHAP values by site
    site_analysis_df = analyze_shap_by_site(shap_explanation, X_df, OUTPUT_DIR)

    # 5. Heatmap
    plot_shap_heatmap(shap_explanation, OUTPUT_DIR)

    # 6. Waterfall plots for first 3 patients
    n_waterfall = min(3, len(X_df))
    for i in range(n_waterfall):
        plot_shap_waterfall(shap_explanation, i, OUTPUT_DIR)

    # 7. Dependence plots for top 5 features
    top_features = mean_abs_shap.head(5)["feature"].tolist()
    for feat in top_features:
        plot_shap_dependence(shap_explanation, X_df, feat, OUTPUT_DIR)

    # --- Compare with model's built-in feature importance ---
    if artifacts.get("feature_importance") is not None:
        print("\nComparing SHAP importance vs. model feature importance...")
        model_imp = artifacts["feature_importance"][
            ["feature", "importance_normalized"]
        ].copy()
        model_imp = model_imp.rename(
            columns={"importance_normalized": "model_importance"}
        )

        shap_imp = mean_abs_shap[["feature", "mean_abs_shap"]].copy()
        shap_imp["shap_importance"] = (
            shap_imp["mean_abs_shap"] / shap_imp["mean_abs_shap"].sum()
        )

        comparison = model_imp.merge(
            shap_imp[["feature", "shap_importance"]],
            on="feature", how="outer"
        ).fillna(0).sort_values("shap_importance", ascending=False)

        comparison.to_csv(OUTPUT_DIR / "importance_comparison.csv", index=False)
        print(f"Saved: {OUTPUT_DIR / 'importance_comparison.csv'}")

    print("\n" + "=" * 70)
    print("SHAP analysis complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

