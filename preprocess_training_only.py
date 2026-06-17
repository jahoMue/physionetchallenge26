#!/usr/bin/env python
"""
preprocess_training_only.py
===========================

Standalone script that ONLY preprocesses the PhysioNet Challenge training data.

It:
  - patches config.py paths to point to the provided training data folder;
  - finds patients from demographics.csv;
  - runs main.process_single_patient(...) for each training subject;
  - skips subjects that already have valid cached preprocessing;
  - writes/updates preprocessing_manifest.json;
  - writes simple CSV/JSON preprocessing reports;
  - DOES NOT train a model;
  - DOES NOT call the official train_model.py;
  - DOES NOT edit official Challenge scripts.

Example:

    python preprocess_training_only.py ^
        --data_folder D:\\Physionet26Data\\training_set ^
        --model_folder output ^
        --verbose

or:

    python preprocess_training_only.py \
        --data_folder /path/to/training_set \
        --model_folder output \
        --workers 4 \
        --verbose

    python preprocess_training_only.py --data_folder E:\training_data --model_folder E:\model --workers 4 --verbose

"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd


DEMOGRAPHICS_BASENAME = "demographics.csv"
PREPROCESS_MANIFEST_FILENAME = "preprocessing_manifest.json"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("preprocess_training_only")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Config patching
# ---------------------------------------------------------------------------

def patch_config(data_folder: str, model_folder: str, feature_dir: Optional[str] = None):
    """
    Patch config.py so downstream pipeline modules use the requested data/output
    folders instead of hardcoded local defaults.

    Must be called before importing main.process_single_patient.
    """
    import config

    data_path = Path(data_folder).resolve()
    model_path = Path(model_folder).resolve()

    config.TRAINING_SET_DIR = data_path
    config.PHYSIOLOGICAL_DATA_DIR = data_path / "physiological_data"
    config.ALGORITHMIC_ANNOTATIONS_DIR = data_path / "algorithmic_annotations"
    config.HUMAN_ANNOTATIONS_DIR = data_path / "human_annotations"
    config.DATA_DIR = config.PHYSIOLOGICAL_DATA_DIR
    config.DEMOGRAPHICS_FILE = data_path / DEMOGRAPHICS_BASENAME

    config.OUTPUT_DIR = model_path
    config.FEATURE_DIR = Path(feature_dir).resolve() if feature_dir else model_path / "features"
    config.MODEL_DIR = model_path / "models"
    config.LOG_DIR = model_path / "logs"
    config.PLOT_DIR = model_path / "plots"

    for d in [
        model_path,
        config.FEATURE_DIR,
        config.MODEL_DIR,
        config.LOG_DIR,
        config.PLOT_DIR,
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    config.PLOT_ENABLED = False

    if not hasattr(config, "NUM_WORKERS"):
        config.NUM_WORKERS = max(1, min(os.cpu_count() or 1, 4))

    if not hasattr(config, "SEGMENT_LENGTH_SEC"):
        config.SEGMENT_LENGTH_SEC = 300

    if not hasattr(config, "SEGMENT_OVERLAP_SEC"):
        config.SEGMENT_OVERLAP_SEC = 0


# ---------------------------------------------------------------------------
# Cache / manifest helpers
# ---------------------------------------------------------------------------

def safe_cache_key(s: str) -> str:
    return (
        str(s)
        .replace("\\", "/")
        .replace("/", "__")
        .replace(":", "_")
        .replace(" ", "_")
    )


def manifest_path(feature_output_dir: Path) -> Path:
    return Path(feature_output_dir) / PREPROCESS_MANIFEST_FILENAME


def load_manifest(feature_output_dir: Path) -> Dict:
    path = manifest_path(feature_output_dir)

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            return data

    except Exception:
        pass

    return {}


def save_manifest(feature_output_dir: Path, manifest: Dict):
    feature_output_dir = Path(feature_output_dir)
    feature_output_dir.mkdir(parents=True, exist_ok=True)

    path = manifest_path(feature_output_dir)
    tmp_path = path.with_suffix(".json.tmp")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    os.replace(tmp_path, path)


def parquet_file_looks_valid(path: Optional[str]) -> bool:
    if not path:
        return False

    try:
        p = Path(path)

        if not p.exists() or not p.is_file():
            return False

        if p.stat().st_size <= 0:
            return False

        # Prefer metadata-only check if pyarrow is available.
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(p)
            return pf.metadata.num_rows > 0
        except Exception:
            df = pd.read_parquet(p)
            return df is not None and len(df) > 0

    except Exception:
        return False


def normalise_cached_entry(entry: Dict) -> Optional[Dict]:
    if not isinstance(entry, dict):
        return None

    seg_path = entry.get("seg_features_path")
    pat_path = entry.get("pat_features_path")

    seg_ok = parquet_file_looks_valid(seg_path)
    pat_ok = parquet_file_looks_valid(pat_path)

    if not seg_ok and not pat_ok:
        return None

    return {
        "success": True,
        "seg_features_path": str(Path(seg_path)) if seg_ok else None,
        "pat_features_path": str(Path(pat_path)) if pat_ok else None,
        "sleep_summary": entry.get("sleep_summary", None),
        "from_cache": True,
    }


def find_cached_preprocessing(
    feature_output_dir: Path,
    pipeline_patient_id: str,
    record_name: Optional[str] = None,
    require: str = "any",
) -> Optional[Dict]:
    """
    Find cached preprocessing for one subject.

    require:
      - any: patient-level OR segment-level parquet is enough
      - patient: patient-level parquet required
      - segment: segment-level parquet required
      - both: both parquet files required
    """
    feature_output_dir = Path(feature_output_dir)

    if not feature_output_dir.exists():
        return None

    manifest = load_manifest(feature_output_dir)

    possible_keys = [
        pipeline_patient_id,
        safe_cache_key(pipeline_patient_id),
    ]

    if record_name:
        possible_keys.extend([record_name, safe_cache_key(record_name)])

    for key in possible_keys:
        if key in manifest:
            cached = normalise_cached_entry(manifest[key])
            if cached is not None and cache_satisfies_require(cached, require):
                return cached

    # Fallback scan for old preprocessing runs without manifest.
    return scan_existing_parquets_for_subject(
        feature_output_dir=feature_output_dir,
        pipeline_patient_id=pipeline_patient_id,
        record_name=record_name,
        require=require,
    )


def cache_satisfies_require(cached: Dict, require: str) -> bool:
    has_seg = bool(cached.get("seg_features_path"))
    has_pat = bool(cached.get("pat_features_path"))

    if require == "any":
        return has_seg or has_pat
    if require == "patient":
        return has_pat
    if require == "segment":
        return has_seg
    if require == "both":
        return has_seg and has_pat

    raise ValueError(f"Unknown cache requirement: {require}")


def scan_existing_parquets_for_subject(
    feature_output_dir: Path,
    pipeline_patient_id: str,
    record_name: Optional[str],
    require: str = "any",
) -> Optional[Dict]:
    """
    Conservative fallback scan for pre-existing parquet files that were created
    before preprocessing_manifest.json existed.
    """
    try:
        feature_output_dir = Path(feature_output_dir)

        patient_token = safe_cache_key(pipeline_patient_id).lower()
        record_token = safe_cache_key(record_name).lower() if record_name else None

        matches = []

        for p in feature_output_dir.rglob("*.parquet"):
            rel_text = safe_cache_key(str(p.relative_to(feature_output_dir))).lower()
            full_text = safe_cache_key(str(p)).lower()

            if patient_token in rel_text or patient_token in full_text:
                matches.append(p)
            elif record_token and (record_token in rel_text or record_token in full_text):
                matches.append(p)

        if not matches:
            return None

        seg_candidates = []
        pat_candidates = []

        for p in matches:
            name = p.name.lower()
            full = str(p).lower()

            if (
                "segment" in name
                or "segments" in name
                or "_seg" in name
                or "seg_" in name
                or "/segment" in full.replace("\\", "/")
                or "/segments" in full.replace("\\", "/")
            ):
                seg_candidates.append(p)

            elif (
                "patient" in name
                or "_pat" in name
                or "pat_" in name
                or "/patient" in full.replace("\\", "/")
                or "/patients" in full.replace("\\", "/")
            ):
                pat_candidates.append(p)

            else:
                # If unclear, treat as patient-level only if it looks small.
                # This avoids accidentally using segment-level tables as patient tables.
                pat_candidates.append(p)

        seg_path = newest_valid_parquet(seg_candidates)
        pat_path = newest_valid_parquet(pat_candidates)

        cached = {
            "success": True,
            "seg_features_path": str(seg_path) if seg_path else None,
            "pat_features_path": str(pat_path) if pat_path else None,
            "sleep_summary": None,
            "from_cache": True,
        }

        if cache_satisfies_require(cached, require):
            return cached

    except Exception:
        return None

    return None


def newest_valid_parquet(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None

    paths = sorted(
        paths,
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )

    for p in paths:
        if parquet_file_looks_valid(str(p)):
            return p

    return None


def remember_preprocessing_result(
    feature_output_dir: Path,
    pipeline_patient_id: str,
    record_name: Optional[str],
    result: Dict,
):
    if not result or not result.get("success"):
        return

    seg_path = result.get("seg_features_path")
    pat_path = result.get("pat_features_path")

    if not seg_path and not pat_path:
        return

    entry = {
        "pipeline_patient_id": pipeline_patient_id,
        "record_name": record_name,
        "seg_features_path": str(Path(seg_path).resolve()) if seg_path else None,
        "pat_features_path": str(Path(pat_path).resolve()) if pat_path else None,
        "sleep_summary": result.get("sleep_summary", None),
        "cached_at_utc": datetime.utcnow().isoformat() + "Z",
        "manifest_generated_by": "preprocess_training_only.py",
    }

    manifest = load_manifest(feature_output_dir)

    manifest[pipeline_patient_id] = entry
    manifest[safe_cache_key(pipeline_patient_id)] = entry

    if record_name:
        manifest[record_name] = entry
        manifest[safe_cache_key(record_name)] = entry

    meta = manifest.get("__meta__", {})
    if not isinstance(meta, dict):
        meta = {}

    meta.update(
        {
            "last_updated_utc": datetime.utcnow().isoformat() + "Z",
            "feature_output_dir": str(Path(feature_output_dir).resolve()),
        }
    )

    manifest["__meta__"] = meta

    save_manifest(feature_output_dir, manifest)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def preprocess_one_subject_worker(
    data_folder: str,
    model_folder: str,
    feature_dir: str,
    pipeline_patient_id: str,
    patient_dir_str: str,
    record_name: str,
    segment_length_sec: float,
    overlap_sec: float,
):
    """
    Child-process worker.

    Patches config inside the child process before importing main.
    """
    patch_config(
        data_folder=data_folder,
        model_folder=model_folder,
        feature_dir=feature_dir,
    )

    from main import process_single_patient

    return process_single_patient(
        patient_id=pipeline_patient_id,
        patient_dir=Path(patient_dir_str),
        record_name=record_name,
        segment_length_sec=segment_length_sec,
        overlap_sec=overlap_sec,
        feature_output_dir=Path(feature_dir),
    )


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_reports(
    feature_dir: Path,
    cached_subjects: List[Dict],
    processed_subjects: List[Dict],
    failed_subjects: List[Dict],
    pending_subjects: List[Dict],
    total_subjects: int,
):
    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    cached_csv = feature_dir / "preprocess_only_cached.csv"
    processed_csv = feature_dir / "preprocess_only_processed.csv"
    failed_csv = feature_dir / "preprocess_only_failed.csv"
    pending_csv = feature_dir / "preprocess_only_pending.csv"
    summary_json = feature_dir / "preprocess_only_summary.json"

    pd.DataFrame(cached_subjects).to_csv(cached_csv, index=False)
    pd.DataFrame(processed_subjects).to_csv(processed_csv, index=False)
    pd.DataFrame(failed_subjects).to_csv(failed_csv, index=False)
    pd.DataFrame(pending_subjects).to_csv(pending_csv, index=False)

    summary = {
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "total_subjects": int(total_subjects),
        "cached_subjects": int(len(cached_subjects)),
        "newly_processed_subjects": int(len(processed_subjects)),
        "failed_subjects": int(len(failed_subjects)),
        "pending_subjects": int(len(pending_subjects)),
        "feature_dir": str(feature_dir.resolve()),
        "manifest": str(manifest_path(feature_dir).resolve()),
        "reports": {
            "cached_csv": str(cached_csv.resolve()),
            "processed_csv": str(processed_csv.resolve()),
            "failed_csv": str(failed_csv.resolve()),
            "pending_csv": str(pending_csv.resolve()),
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Only preprocess PhysioNet Challenge training data."
    )

    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        required=True,
        help="Training data folder containing demographics.csv and physiological_data/.",
    )

    parser.add_argument(
        "-m",
        "--model_folder",
        type=str,
        default="output",
        help=(
            "Output/model folder. If --feature_dir is not given, features are "
            "written to <model_folder>/features. Default: output"
        ),
    )

    parser.add_argument(
        "--feature_dir",
        type=str,
        default=None,
        help="Explicit feature output directory. Overrides <model_folder>/features.",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers. Default: config.NUM_WORKERS.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of subjects submitted per process-pool batch. Default: workers.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess subjects even if cached preprocessing already exists.",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only check cache/pending subjects; do not preprocess.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process/check the first N subjects. Useful for testing.",
    )

    parser.add_argument(
        "--require_cache",
        choices=["any", "patient", "segment", "both"],
        default="any",
        help=(
            "Which cached files are required to skip a subject. "
            "Default: any."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(args.verbose)

    data_folder = Path(args.data_folder).resolve()
    model_folder = Path(args.model_folder).resolve()

    if args.feature_dir is not None:
        feature_dir = Path(args.feature_dir).resolve()
    else:
        feature_dir = model_folder / "features"

    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder does not exist: {data_folder}")

    demographics_path = data_folder / DEMOGRAPHICS_BASENAME
    if not demographics_path.exists():
        raise FileNotFoundError(f"Missing demographics.csv: {demographics_path}")

    logger.info("=" * 72)
    logger.info("PhysioNet training-data preprocessing only")
    logger.info("=" * 72)
    logger.info(f"Data folder:   {data_folder}")
    logger.info(f"Model folder:  {model_folder}")
    logger.info(f"Feature dir:   {feature_dir}")
    logger.info(f"Manifest:      {manifest_path(feature_dir)}")
    logger.info(f"Force:         {args.force}")
    logger.info(f"Dry run:       {args.dry_run}")
    logger.info(f"Require cache: {args.require_cache}")
    logger.info("=" * 72)

    patch_config(
        data_folder=str(data_folder),
        model_folder=str(model_folder),
        feature_dir=str(feature_dir),
    )

    import config
    from helper_code import find_patients, DEMOGRAPHICS_FILE, HEADERS

    patient_data_file = os.path.join(str(data_folder), DEMOGRAPHICS_FILE)
    if not os.path.exists(patient_data_file):
        patient_data_file = str(demographics_path)

    patient_metadata_list = find_patients(patient_data_file)

    if args.limit is not None:
        patient_metadata_list = patient_metadata_list[: max(0, int(args.limit))]

    num_records = len(patient_metadata_list)

    if num_records == 0:
        raise RuntimeError("No patients found.")

    logger.info(f"Found {num_records} subjects.")

    segment_length_sec = getattr(config, "SEGMENT_LENGTH_SEC", 300)
    overlap_sec = getattr(config, "SEGMENT_OVERLAP_SEC", 0)

    if args.workers is None:
        workers = int(getattr(config, "NUM_WORKERS", 1))
    else:
        workers = int(args.workers)

    workers = max(1, min(workers, num_records))

    batch_size = int(args.batch_size) if args.batch_size else workers
    batch_size = max(1, batch_size)

    logger.info(f"Workers:       {workers}")
    logger.info(f"Batch size:    {batch_size}")
    logger.info(f"Segment sec:   {segment_length_sec}")
    logger.info(f"Overlap sec:   {overlap_sec}")

    feature_dir.mkdir(parents=True, exist_ok=True)

    records_to_process = []
    cached_subjects = []
    pending_subjects = []
    failed_subjects = []
    processed_subjects = []

    logger.info("Checking preprocessing cache...")

    for i, record in enumerate(patient_metadata_list):
        patient_id_bids = record[HEADERS["bids_folder"]]
        site_id = record[HEADERS["site_id"]]
        session_id = record[HEADERS["session_id"]]

        record_name = f"{patient_id_bids}_ses-{session_id}"
        pipeline_patient_id = f"{site_id}/{record_name}"
        patient_dir = Path(config.PHYSIOLOGICAL_DATA_DIR) / site_id

        item = {
            "index": i,
            "patient_id_bids": patient_id_bids,
            "site_id": site_id,
            "session_id": session_id,
            "record_name": record_name,
            "pipeline_patient_id": pipeline_patient_id,
            "patient_dir": str(patient_dir),
        }

        if not args.force:
            cached = find_cached_preprocessing(
                feature_output_dir=feature_dir,
                pipeline_patient_id=pipeline_patient_id,
                record_name=record_name,
                require=args.require_cache,
            )

            if cached is not None:
                item.update(
                    {
                        "seg_features_path": cached.get("seg_features_path"),
                        "pat_features_path": cached.get("pat_features_path"),
                        "status": "cached",
                    }
                )
                cached_subjects.append(item)

                # If found by fallback scan, make sure it is written to manifest.
                remember_preprocessing_result(
                    feature_output_dir=feature_dir,
                    pipeline_patient_id=pipeline_patient_id,
                    record_name=record_name,
                    result=cached,
                )

                continue

        item["status"] = "pending"
        records_to_process.append(item)
        pending_subjects.append(item)

    logger.info(
        f"Cache status: {len(cached_subjects)}/{num_records} already cached, "
        f"{len(records_to_process)} need preprocessing."
    )

    if args.verbose and cached_subjects:
        preview = [x["pipeline_patient_id"] for x in cached_subjects[:30]]
        logger.info(
            "Cached subjects"
            + (" first 30" if len(cached_subjects) > 30 else "")
            + ": "
            + ", ".join(preview)
        )

    if args.verbose and records_to_process:
        preview = [x["pipeline_patient_id"] for x in records_to_process[:30]]
        logger.info(
            "Pending subjects"
            + (" first 30" if len(records_to_process) > 30 else "")
            + ": "
            + ", ".join(preview)
        )

    if args.dry_run:
        logger.info("Dry run requested. No preprocessing will be performed.")
        write_reports(
            feature_dir=feature_dir,
            cached_subjects=cached_subjects,
            processed_subjects=processed_subjects,
            failed_subjects=failed_subjects,
            pending_subjects=pending_subjects,
            total_subjects=num_records,
        )
        logger.info("Dry-run reports written.")
        return

    total_start = time.time()

    for batch_start in range(0, len(records_to_process), batch_size):
        batch_end = min(batch_start + batch_size, len(records_to_process))
        batch = records_to_process[batch_start:batch_end]

        logger.info(
            f"Starting batch {batch_start // batch_size + 1}: "
            f"{batch_start + 1}-{batch_end} of {len(records_to_process)} pending subjects."
        )

        futures = {}

        with ProcessPoolExecutor(max_workers=workers) as executor:
            for item in batch:
                patient_dir = Path(item["patient_dir"])
                pid = item["pipeline_patient_id"]

                if not patient_dir.exists():
                    msg = f"Directory not found: {patient_dir}"
                    logger.warning(f"{pid}: {msg}")
                    fail_item = dict(item)
                    fail_item["status"] = "failed"
                    fail_item["error"] = msg
                    failed_subjects.append(fail_item)
                    continue

                future = executor.submit(
                    preprocess_one_subject_worker,
                    str(data_folder),
                    str(model_folder),
                    str(feature_dir),
                    item["pipeline_patient_id"],
                    item["patient_dir"],
                    item["record_name"],
                    segment_length_sec,
                    overlap_sec,
                )

                futures[future] = item

            for future in as_completed(futures):
                item = futures[future]
                pid = item["pipeline_patient_id"]

                try:
                    result = future.result()

                    if result and result.get("success"):
                        remember_preprocessing_result(
                            feature_output_dir=feature_dir,
                            pipeline_patient_id=pid,
                            record_name=item["record_name"],
                            result=result,
                        )

                        ok_item = dict(item)
                        ok_item["status"] = "processed"
                        ok_item["seg_features_path"] = result.get("seg_features_path")
                        ok_item["pat_features_path"] = result.get("pat_features_path")
                        processed_subjects.append(ok_item)

                        logger.info(f"SUCCESS: {pid}")

                    else:
                        fail_item = dict(item)
                        fail_item["status"] = "failed"
                        fail_item["error"] = "process_single_patient returned unsuccessful result"
                        failed_subjects.append(fail_item)

                        logger.warning(f"FAILED: {pid}")

                except Exception as e:
                    fail_item = dict(item)
                    fail_item["status"] = "failed"
                    fail_item["error"] = f"{type(e).__name__}: {e}"
                    failed_subjects.append(fail_item)

                    logger.error(f"ERROR: {pid}: {type(e).__name__}: {e}")

                    if args.verbose:
                        traceback.print_exc()

        gc.collect()

        elapsed = time.time() - total_start
        usable = len(cached_subjects) + len(processed_subjects)

        logger.info(
            f"Batch done. Newly processed={len(processed_subjects)}, "
            f"cached={len(cached_subjects)}, failed={len(failed_subjects)}, "
            f"usable total={usable}/{num_records}, elapsed={elapsed:.0f}s."
        )

        # Write reports after every batch so progress survives interruption.
        remaining_pending = records_to_process[batch_end:]

        write_reports(
            feature_dir=feature_dir,
            cached_subjects=cached_subjects,
            processed_subjects=processed_subjects,
            failed_subjects=failed_subjects,
            pending_subjects=remaining_pending,
            total_subjects=num_records,
        )

    elapsed = time.time() - total_start
    usable = len(cached_subjects) + len(processed_subjects)

    logger.info("=" * 72)
    logger.info("Preprocessing complete")
    logger.info("=" * 72)
    logger.info(f"Total subjects:          {num_records}")
    logger.info(f"Already cached skipped:  {len(cached_subjects)}")
    logger.info(f"Newly processed:         {len(processed_subjects)}")
    logger.info(f"Failed:                  {len(failed_subjects)}")
    logger.info(f"Usable total:            {usable}/{num_records}")
    logger.info(f"Elapsed:                 {elapsed:.0f}s")
    logger.info(f"Feature dir:             {feature_dir}")
    logger.info(f"Manifest:                {manifest_path(feature_dir)}")
    logger.info("=" * 72)

    write_reports(
        feature_dir=feature_dir,
        cached_subjects=cached_subjects,
        processed_subjects=processed_subjects,
        failed_subjects=failed_subjects,
        pending_subjects=[],
        total_subjects=num_records,
    )

    logger.info("Reports written.")
    logger.info("Done.")


if __name__ == "__main__":
    main()
