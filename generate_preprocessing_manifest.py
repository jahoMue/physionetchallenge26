#!/usr/bin/env python
"""
generate_preprocessing_manifest.py
==================================

Scan already-preprocessed parquet feature files and generate:

    preprocessing_manifest.json

This manifest is compatible with the cache helpers suggested for team_code.py:

    {
      "<site>/<BidsFolder>_ses-<SessionID>": {
        "pipeline_patient_id": "...",
        "record_name": "...",
        "seg_features_path": "...",
        "pat_features_path": "...",
        "sleep_summary": null,
        ...
      },
      ...
    }

Usage examples
--------------

1) If your previous preprocessing was saved in model/features:

    python generate_preprocessing_manifest.py ^
        --data_folder D:\\Physionet26Data\\training_set ^
        --model_folder output

2) If you know the exact feature folder:

    python generate_preprocessing_manifest.py ^
        --data_folder D:\\Physionet26Data\\training_set ^
        --feature_dir output\\features

3) Require both patient-level and segment-level parquet files:

    python generate_preprocessing_manifest.py ^
        --data_folder D:\\Physionet26Data\\training_set ^
        --feature_dir output\\features ^
        --require both

4) Only preview without writing JSON:

    python generate_preprocessing_manifest.py ^
        --data_folder D:\\Physionet26Data\\training_set ^
        --feature_dir output\\features ^
        --dry_run
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


MANIFEST_FILENAME = "preprocessing_manifest.json"


# -------------------------------------------------------------------------
# ID / path helpers
# -------------------------------------------------------------------------

def safe_cache_key(s: str) -> str:
    """
    Convert an identifier into a filesystem/JSON-friendly key.

    Must match the helper used in team_code.py if you added the previous cache
    patch.
    """
    return (
        str(s)
        .replace("\\", "/")
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def normalise_text(s) -> str:
    return str(s).strip()


def lower_path_token(path: Path) -> str:
    return safe_cache_key(str(path)).lower()


def guess_demographics_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Guess the required PhysioNet demographics columns.

    team_code.py constructs patient IDs from:
        BidsFolder, SiteID, SessionID
    through helper_code.HEADERS in the official workflow [[18]].
    """
    candidates = {
        "bids_folder": [
            "BidsFolder", "bids_folder", "bidsfolder",
            "BDSPPatientID", "patient_id", "PatientID",
        ],
        "site_id": [
            "SiteID", "site_id", "siteid", "Site", "site",
        ],
        "session_id": [
            "SessionID", "session_id", "sessionid", "Session", "session",
        ],
    }

    lower_to_original = {c.lower(): c for c in df.columns}

    chosen = {}

    for key, names in candidates.items():
        found = None

        # Exact / case-insensitive match.
        for name in names:
            if name in df.columns:
                found = name
                break
            if name.lower() in lower_to_original:
                found = lower_to_original[name.lower()]
                break

        if found is None:
            raise ValueError(
                f"Could not find demographics column for {key}. "
                f"Tried {names}. Available columns: {list(df.columns)}"
            )

        chosen[key] = found

    return chosen["bids_folder"], chosen["site_id"], chosen["session_id"]


def build_expected_subjects(demographics_path: Path) -> List[Dict]:
    """
    Build expected subject IDs using the same convention as team_code.py:

        record_name = f"{BidsFolder}_ses-{SessionID}"
        pipeline_patient_id = f"{SiteID}/{record_name}"

    This mirrors the current team_code.py preprocessing loop [[18]].
    """
    df = pd.read_csv(demographics_path)

    bids_col, site_col, sess_col = guess_demographics_columns(df)

    subjects = []

    for _, row in df.iterrows():
        bids = normalise_text(row[bids_col])
        site = normalise_text(row[site_col])
        sess = normalise_text(row[sess_col])

        record_name = f"{bids}_ses-{sess}"
        pipeline_patient_id = f"{site}/{record_name}"

        subjects.append(
            {
                "bids_folder": bids,
                "site_id": site,
                "session_id": sess,
                "record_name": record_name,
                "pipeline_patient_id": pipeline_patient_id,
                "safe_pipeline_patient_id": safe_cache_key(pipeline_patient_id),
                "safe_record_name": safe_cache_key(record_name),
            }
        )

    return subjects


# -------------------------------------------------------------------------
# Parquet inspection
# -------------------------------------------------------------------------

def inspect_parquet(path: Path) -> Dict:
    """
    Inspect one parquet file.

    This function tries to avoid loading full tables where possible, but falls
    back to pandas when metadata access is unavailable.
    """
    info = {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "valid": False,
        "n_rows": None,
        "columns": [],
        "kind": None,          # "segment", "patient", or None
        "patient_ids": [],
        "error": None,
        "size_bytes": None,
        "mtime": None,
    }

    try:
        info["size_bytes"] = int(path.stat().st_size)
        info["mtime"] = float(path.stat().st_mtime)

        if info["size_bytes"] <= 0:
            info["error"] = "empty file"
            return info

        # Prefer pyarrow metadata if available.
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            info["n_rows"] = int(pf.metadata.num_rows)
            info["columns"] = list(pf.schema.names)

            if info["n_rows"] <= 0:
                info["error"] = "zero rows"
                return info

        except Exception:
            # Fallback: read full file.
            df = pd.read_parquet(path)
            info["n_rows"] = int(len(df))
            info["columns"] = list(df.columns)

            if len(df) <= 0:
                info["error"] = "zero rows"
                return info

        columns_lower = {c.lower(): c for c in info["columns"]}

        # Try reading patient_id column only.
        patient_col = None
        for candidate in ["patient_id", "PatientID", "pipeline_patient_id"]:
            if candidate.lower() in columns_lower:
                patient_col = columns_lower[candidate.lower()]
                break

        if patient_col is not None:
            try:
                pid_df = pd.read_parquet(path, columns=[patient_col])
                vals = (
                    pid_df[patient_col]
                    .dropna()
                    .astype(str)
                    .drop_duplicates()
                    .head(10)
                    .tolist()
                )
                info["patient_ids"] = vals
            except Exception:
                pass

        # Classify feature kind.
        path_text = str(path).lower()
        name_text = path.name.lower()
        cols_lower_set = set(columns_lower.keys())

        if "segment_idx" in cols_lower_set:
            info["kind"] = "segment"
        elif any(tok in name_text for tok in ["segment", "segments", "_seg", "seg_"]):
            info["kind"] = "segment"
        elif any(tok in path_text for tok in ["/segments/", "\\segments\\", "/segment/", "\\segment\\"]):
            info["kind"] = "segment"
        elif any(tok in name_text for tok in ["patient", "_pat", "pat_"]):
            info["kind"] = "patient"
        elif any(tok in path_text for tok in ["/patients/", "\\patients\\", "/patient/", "\\patient\\"]):
            info["kind"] = "patient"
        else:
            # Most patient-level feature tables do not have segment_idx.
            # If it has one row or no segment_idx, treat as patient-level.
            info["kind"] = "patient"

        info["valid"] = True
        return info

    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return info


def scan_feature_dir(feature_dir: Path, verbose: bool = False) -> List[Dict]:
    parquet_files = sorted(feature_dir.rglob("*.parquet"))

    if verbose:
        print(f"Found {len(parquet_files)} parquet files under {feature_dir}")

    infos = []

    for i, p in enumerate(parquet_files, start=1):
        info = inspect_parquet(p)
        infos.append(info)

        if verbose:
            status = "OK" if info["valid"] else "BAD"
            kind = info.get("kind") or "unknown"
            print(f"[{i:5d}/{len(parquet_files)}] {status:3s} {kind:7s} {p}")

            if not info["valid"]:
                print(f"      error: {info.get('error')}")

    return infos


# -------------------------------------------------------------------------
# Matching
# -------------------------------------------------------------------------

def score_path_match(subject: Dict, parquet_info: Dict) -> int:
    """
    Score how likely a parquet file belongs to a subject.
    Higher is better.
    """
    p = Path(parquet_info["path"])
    text = lower_path_token(p)

    pipeline_id = subject["pipeline_patient_id"]
    safe_pid = subject["safe_pipeline_patient_id"]
    record_name = subject["record_name"]
    safe_record = subject["safe_record_name"]
    bids = subject["bids_folder"]
    site = subject["site_id"]
    session = subject["session_id"]

    candidates = [
        (pipeline_id.lower(), 100),
        (safe_pid.lower(), 100),
        (record_name.lower(), 80),
        (safe_record.lower(), 80),
        (bids.lower(), 40),
        (site.lower(), 10),
        (f"ses-{session}".lower(), 30),
        (str(session).lower(), 5),
    ]

    score = 0

    for token, points in candidates:
        if token and safe_cache_key(token).lower() in text:
            score += points
        elif token and token in text:
            score += points

    # Strong match if patient_id column explicitly contains the pipeline ID
    # or record name.
    patient_ids = [str(x) for x in parquet_info.get("patient_ids", [])]

    for pid in patient_ids:
        if pid == pipeline_id:
            score += 200
        elif pid == record_name:
            score += 150
        elif record_name in pid:
            score += 120
        elif bids in pid and str(session) in pid:
            score += 80

    return score


def choose_best_file(candidates: List[Dict]) -> Optional[Dict]:
    """
    Choose the newest valid candidate, using score first, then modified time.
    """
    if not candidates:
        return None

    candidates = [c for c in candidates if c.get("valid")]
    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda x: (
            x.get("_match_score", 0),
            x.get("mtime") or 0,
            x.get("size_bytes") or 0,
        ),
        reverse=True,
    )
    return candidates[0]


def build_manifest(
    subjects: List[Dict],
    parquet_infos: List[Dict],
    require: str = "any",
    min_score: int = 70,
    verbose: bool = False,
) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Build manifest entries.

    require:
      - "any": at least patient-level or segment-level features
      - "patient": patient-level file required
      - "segment": segment-level file required
      - "both": both files required
    """
    manifest = {}

    cached = []
    missing = []

    valid_infos = [x for x in parquet_infos if x.get("valid")]

    for subject in subjects:
        pid = subject["pipeline_patient_id"]
        record_name = subject["record_name"]

        seg_candidates = []
        pat_candidates = []

        for info in valid_infos:
            score = score_path_match(subject, info)

            if score < min_score:
                continue

            info_copy = dict(info)
            info_copy["_match_score"] = score

            if info_copy.get("kind") == "segment":
                seg_candidates.append(info_copy)
            elif info_copy.get("kind") == "patient":
                pat_candidates.append(info_copy)

        best_seg = choose_best_file(seg_candidates)
        best_pat = choose_best_file(pat_candidates)

        seg_path = best_seg["path"] if best_seg is not None else None
        pat_path = best_pat["path"] if best_pat is not None else None

        ok = False
        if require == "any":
            ok = bool(seg_path or pat_path)
        elif require == "patient":
            ok = bool(pat_path)
        elif require == "segment":
            ok = bool(seg_path)
        elif require == "both":
            ok = bool(seg_path and pat_path)
        else:
            raise ValueError(f"Unknown --require value: {require}")

        if ok:
            entry = {
                "pipeline_patient_id": pid,
                "record_name": record_name,
                "seg_features_path": seg_path,
                "pat_features_path": pat_path,
                "sleep_summary": None,
                "cached_at_utc": datetime.utcnow().isoformat() + "Z",
                "manifest_generated_by": "generate_preprocessing_manifest.py",
                "match": {
                    "seg_score": (
                        int(best_seg["_match_score"])
                        if best_seg is not None
                        else None
                    ),
                    "pat_score": (
                        int(best_pat["_match_score"])
                        if best_pat is not None
                        else None
                    ),
                    "require": require,
                },
            }

            # Store multiple lookup keys for robust cache lookup.
            manifest[pid] = entry
            manifest[safe_cache_key(pid)] = entry
            manifest[record_name] = entry
            manifest[safe_cache_key(record_name)] = entry

            cached.append(
                {
                    "pipeline_patient_id": pid,
                    "record_name": record_name,
                    "seg_features_path": seg_path,
                    "pat_features_path": pat_path,
                }
            )

            if verbose:
                print(f"CACHED: {pid}")
                if seg_path:
                    print(f"  segment: {seg_path}")
                if pat_path:
                    print(f"  patient: {pat_path}")

        else:
            reason = []
            if not seg_path:
                reason.append("no segment parquet")
            if not pat_path:
                reason.append("no patient parquet")

            missing.append(
                {
                    "pipeline_patient_id": pid,
                    "record_name": record_name,
                    "reason": ", ".join(reason),
                    "best_segment_score": (
                        max([c["_match_score"] for c in seg_candidates])
                        if seg_candidates
                        else None
                    ),
                    "best_patient_score": (
                        max([c["_match_score"] for c in pat_candidates])
                        if pat_candidates
                        else None
                    ),
                }
            )

            if verbose:
                print(f"MISSING: {pid} ({', '.join(reason)})")

    manifest["__meta__"] = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "n_expected_subjects": int(len(subjects)),
        "n_cached_subjects": int(len(cached)),
        "n_missing_subjects": int(len(missing)),
        "n_parquet_files_scanned": int(len(parquet_infos)),
        "n_valid_parquet_files": int(len(valid_infos)),
        "require": require,
        "min_score": int(min_score),
    }

    return manifest, cached, missing


# -------------------------------------------------------------------------
# IO
# -------------------------------------------------------------------------

def atomic_write_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    os.replace(tmp_path, path)


def write_report_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate preprocessing_manifest.json from existing parquet "
            "feature files."
        )
    )

    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Challenge data folder containing demographics.csv.",
    )

    parser.add_argument(
        "--model_folder",
        type=str,
        default=None,
        help=(
            "Model/output folder. If --feature_dir is not supplied, "
            "the script uses <model_folder>/features."
        ),
    )

    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help=(
            "Folder containing existing preprocessed parquet features. "
            "If omitted, uses <model_folder>/features."
        ),
    )

    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help=(
            "Output manifest path. Defaults to "
            "<feature_dir>/preprocessing_manifest.json."
        ),
    )

    parser.add_argument(
        "--require",
        choices=["any", "patient", "segment", "both"],
        default="any",
        help=(
            "Which files are required to mark a subject as cached. "
            "Default: any."
        ),
    )

    parser.add_argument(
        "--min_score",
        type=int,
        default=70,
        help=(
            "Minimum filename/metadata match score. "
            "Lower if your filenames are unusual. Default: 70."
        ),
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan and report, but do not write JSON.",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed matching information.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    data_folder = Path(args.data_folder).resolve()
    demographics_path = data_folder / "demographics.csv"

    if not demographics_path.exists():
        raise FileNotFoundError(f"Missing demographics.csv: {demographics_path}")

    if args.feature_dir is not None:
        feature_dir = Path(args.feature_dir).resolve()
    else:
        if args.model_folder is None:
            raise ValueError(
                "Please provide either --feature_dir or --model_folder."
            )
        feature_dir = Path(args.model_folder).resolve() / "features"

    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")

    if args.output_json is not None:
        output_json = Path(args.output_json).resolve()
    else:
        output_json = feature_dir / MANIFEST_FILENAME

    print("=" * 72)
    print("Generate preprocessing manifest")
    print("=" * 72)
    print(f"Data folder:      {data_folder}")
    print(f"Demographics:     {demographics_path}")
    print(f"Feature dir:      {feature_dir}")
    print(f"Output JSON:      {output_json}")
    print(f"Require:          {args.require}")
    print(f"Min score:        {args.min_score}")
    print(f"Dry run:          {args.dry_run}")
    print("=" * 72)

    subjects = build_expected_subjects(demographics_path)
    print(f"Expected subjects from demographics: {len(subjects)}")

    parquet_infos = scan_feature_dir(feature_dir, verbose=args.verbose)

    manifest, cached, missing = build_manifest(
        subjects=subjects,
        parquet_infos=parquet_infos,
        require=args.require,
        min_score=args.min_score,
        verbose=args.verbose,
    )

    n_valid = sum(1 for x in parquet_infos if x.get("valid"))

    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"Parquet files scanned: {len(parquet_infos)}")
    print(f"Valid parquet files:   {n_valid}")
    print(f"Cached subjects:       {len(cached)} / {len(subjects)}")
    print(f"Missing subjects:      {len(missing)} / {len(subjects)}")

    if missing:
        print()
        print("First missing subjects:")
        for row in missing[:25]:
            print(f"  - {row['pipeline_patient_id']}: {row['reason']}")

        if len(missing) > 25:
            print(f"  ... and {len(missing) - 25} more")

    if args.dry_run:
        print()
        print("Dry run enabled; not writing manifest.")
        return

    atomic_write_json(output_json, manifest)
    print()
    print(f"Wrote manifest: {output_json}")

    cached_report = output_json.with_name("preprocessing_manifest_cached.csv")
    missing_report = output_json.with_name("preprocessing_manifest_missing.csv")

    write_report_csv(cached_report, cached)
    write_report_csv(missing_report, missing)

    print(f"Wrote cached report:  {cached_report}")
    print(f"Wrote missing report: {missing_report}")
    print("Done.")


if __name__ == "__main__":
    main()
