"""Train one frozen CPU water baseline, then evaluate geographic holdouts once.

Run: uv run --with scikit-learn==1.7.2 python scripts/train_sen1_cpu.py
The protocol is immutable; no Rasuwa/CEMS/UNOSAT data is loaded.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import threading
import time

import joblib
import numpy as np
import rasterio
import requests
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from akash_flood.flood_model import FEATURE_NAMES, features_from_db, predict_water, confusion_counts, metrics, chip_grids_match

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_SHA = "57dc4cc412d09ef06889a29d6f3a7956d1c5666dbf57c58750b77575e3b749d2"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path):
    p = Path(path)
    return {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": sha(p)}


def save(path, value):
    p = Path(path); temp = p.with_suffix(p.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n"); temp.replace(p)


def load_protocol(path):
    if sha(path) != PROTOCOL_SHA:
        raise ValueError("Predeclared protocol changed; refuse fitting or holdout evaluation")
    p = json.loads(Path(path).read_text())
    if sklearn.__version__ != p["model"]["scikit_learn_version"] or list(FEATURE_NAMES) != p["features"]:
        raise ValueError("Frozen library version/features differ")
    names, groups = set(), {}
    for split, rows in p["splits"].items():
        groups[split] = set()
        for row in rows:
            if row["s1"] in names or Path(row["s1"]).name != row["s1"] or Path(row["label"]).name != row["label"]:
                raise ValueError("Duplicate or invalid split identity")
            if row["s1"].replace("_S1Hand.tif", "_LabelHand.tif") != row["label"]:
                raise ValueError("Input/label identities differ")
            names.add(row["s1"]); groups[split].add(row["collection"])
    if len(names) != 446 or any(groups[a] & groups[b] for a, b in [("train", "validation"), ("train", "final_test"), ("validation", "final_test")]):
        raise ValueError("Geographic collection split overlaps")
    return p


class Budget:
    def __init__(self, maximum, cached=0):
        self.maximum, self.used, self.network = maximum, cached, 0
        self.lock = threading.Lock()

    def add(self, count):
        with self.lock:
            if self.used + count > self.maximum:
                raise ValueError("Download budget exceeded")
            self.used += count; self.network += count


def download_one(url, target, budget, cap):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".tif.partial")
    # Failed bytes count against this run's network budget. No TLS bypass or
    # alternate-provider object replacement is permitted.
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > cap:
                    raise ValueError("Source object exceeds declared per-file cap")
                size = 0
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(256 * 1024):
                        if not chunk:
                            continue
                        budget.add(len(chunk)); size += len(chunk)
                        if size > cap:
                            raise ValueError("Source object exceeds declared per-file cap")
                        stream.write(chunk)
                if length is not None and int(length) != size:
                    raise ValueError("Truncated source response")
                partial.replace(target)
                return {**record(target), "url": url, "resolved_url": response.url,
                        "downloaded_at": datetime.now(timezone.utc).isoformat(),
                        "response_status": response.status_code, "response_headers": dict(response.headers),
                        "attempts": attempt + 1}
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def download_data(protocol, cache):
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    ledger_path = cache / "downloads.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    expected = []
    for split in ("train", "validation", "final_test"):
        for row in protocol["splits"][split]:
            for kind in ("s1", "label"):
                name = row[kind]; relative = kind + "/" + name
                url = protocol["dataset_base_url"] + "/" + protocol["dataset_directories"][kind] + name
                expected.append((relative, url, cache / relative, 3_000_000 if kind == "s1" else 100_000))
    if set(ledger) - {entry[0] for entry in expected}:
        raise ValueError("Cache ledger contains unexpected dataset objects")
    for key, entry in ledger.items():
        p = cache / key
        if not p.is_file() or p.stat().st_size != entry["bytes"] or sha(p) != entry["sha256"]:
            raise ValueError("Cached data changed; refuse silent replacement")
    cached = sum(r["bytes"] for r in ledger.values())
    budget = Budget(protocol["download_budget_bytes"], cached)
    jobs = [e for e in expected if e[0] not in ledger]
    for key, _, target, _ in jobs:
        if target.exists():
            raise ValueError("Unrecorded cached source exists: " + key)
    print("Downloading", len(jobs), "remaining public objects; cached bytes", cached, flush=True)
    with ThreadPoolExecutor(max_workers=protocol["maximum_download_workers"]) as pool:
        futures = {pool.submit(download_one, url, target, budget, cap): key for key, url, target, cap in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            ledger[futures[future]] = future.result(); save(ledger_path, ledger)
            if done % 20 == 0 or done == len(jobs):
                print("Downloaded", done, "/", len(jobs), "objects; total accounted bytes", budget.used, flush=True)
    save(cache / "download-summary.json", {"completed_at": datetime.now(timezone.utc).isoformat(),
         "protocol_sha256": PROTOCOL_SHA, "objects": len(ledger), "cache_bytes": sum(r["bytes"] for r in ledger.values()),
         "network_bytes_this_run_including_retries": budget.network, "budget_bytes": budget.maximum,
         "anonymous_public_https": True, "gpu_used": False})
    return ledger


def read_chip(cache, row):
    # Reading is deliberately invoked only inside each permitted fitting/eval
    # stage. Downloading frozen sources does not inspect their label arrays.
    with rasterio.open(cache / "s1" / row["s1"]) as ds:
        if ds.count != 2 or ds.shape != (512, 512) or ds.crs != rasterio.crs.CRS.from_epsg(4326):
            raise ValueError("Unexpected official S1 chip band count/grid")
        db = ds.read(); mask = (ds.read_masks() > 0).all(axis=0)
        grid = (ds.crs, ds.transform, ds.shape)
        nodata = "NaN" if ds.nodata is not None and np.isnan(ds.nodata) else ds.nodata
        observation = {"dtype": ds.dtypes, "nodata": nodata, "band_descriptions": ds.descriptions,
                       "numeric_zero_samples": int((db == 0).sum()),
                       "finite_min_db": float(np.min(db[np.isfinite(db)])) if np.isfinite(db).any() else None,
                       "finite_max_db": float(np.max(db[np.isfinite(db)])) if np.isfinite(db).any() else None}
    with rasterio.open(cache / "label" / row["label"]) as ds:
        if ds.count != 1 or not chip_grids_match(grid, (ds.crs, ds.transform, ds.shape)):
            raise ValueError("Unexpected handlabel alignment/nodata")
        # Label semantics are the published raw -1/0/1 encoding. One official
        # training label wrongly declares nodata=0, the documented nonwater
        # class; do not apply its TIFF mask or delete valid negative examples.
        if ds.nodata not in (None, -1) and not (row["label"] == "Ghana_24858_LabelHand.tif" and ds.nodata == 0):
            raise ValueError("Unexpected handlabel nodata metadata")
        observation["label_nodata_metadata"] = ds.nodata
        observation["label_validity"] = "raw class -1 ignored; 0 nonwater and 1 water, no TIFF label mask"
        labels = ds.read(1)
        if not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError("Unexpected handlabel class encoding")
    return db, mask, labels, observation


def train(protocol, cache, output):
    rng = np.random.default_rng(protocol["training_sampler"]["seed"])
    features, labels, chips = [], [], []
    cap = protocol["training_sampler"]["per_chip_pixel_cap"]
    started = time.perf_counter(); cpu_started = time.process_time()
    for index, row in enumerate(protocol["splits"]["train"], 1):
        db, observation_mask, hand, observation = read_chip(cache, row)
        cube, valid = features_from_db(db, observation_mask)
        eligible = np.flatnonzero(valid & (hand != -1))
        selected = rng.choice(eligible, min(cap, len(eligible)), replace=False)
        if len(selected):
            features.append(cube.reshape(-1, len(FEATURE_NAMES))[selected]); labels.append(hand.ravel()[selected].astype("uint8"))
        chips.append({**row, "eligible_labeled_pixels": int(len(eligible)), "sampled_pixels": int(len(selected)),
                      "sampled_water_pixels": int((hand.ravel()[selected] == 1).sum()), "observation_metadata": observation})
        if index % 20 == 0 or index == len(protocol["splits"]["train"]):
            print("Training features", index, "/", len(protocol["splits"]["train"]), flush=True)
    x, y = np.concatenate(features), np.concatenate(labels)
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("Both supervised water classes required")
    print("Fitting fixed CPU model on", len(y), "samples", flush=True)
    model = HistGradientBoostingClassifier(**protocol["model"]["parameters"])
    with threadpool_limits(limits=4):
        model.fit(x, y)
    model_path = output / "water-model.joblib"; joblib.dump(model, model_path, compress=3)
    report = {"trained_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": PROTOCOL_SHA,
              "train_collections": protocol["split_collections"]["train"], "training_chips": len(chips),
              "training_pixels": int(len(y)), "class_counts": {"nonwater": int((y == 0).sum()), "water": int((y == 1).sum())},
              "features": list(FEATURE_NAMES), "model_parameters": model.get_params(), "model": record(model_path),
              "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
              "gpu_used": False, "holdout_labels_loaded_before_model_freeze": False,
              "reference_inputs_loaded": False, "chips": chips}
    save(output / "training-report.json", report)
    return model, report


def evaluate(model, rows, split, cache, output, freeze):
    result_path = output / (split + "-evaluation.json")
    if result_path.exists():
        raise FileExistsError("Holdout evaluation already exists; refuse a repeat")
    if sha(output / "water-model.joblib") != freeze["model"]["sha256"]:
        raise ValueError("Model changed after freeze")
    start = time.perf_counter(); chips = []; totals = {}; grouped = {}
    for index, row in enumerate(rows, 1):
        db, observation_mask, labels, observation = read_chip(cache, row)
        with threadpool_limits(limits=4):
            _, predicted = predict_water(model, db, observation_mask)
        counts = confusion_counts(labels, predicted)
        for key, value in counts.items():
            totals[key] = totals.get(key, 0) + value
        collection = grouped.setdefault(row["collection"], {})
        for key, value in counts.items():
            collection[key] = collection.get(key, 0) + value
        chips.append({**row, "counts": counts, "metrics": metrics(counts)})
        if index % 12 == 0 or index == len(rows):
            print("Evaluating", split, index, "/", len(rows), flush=True)
    report = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "split": split,
              "protocol_sha256": PROTOCOL_SHA, "freeze_sha256": sha(output / "model-evaluation-freeze.json"),
              "model_sha256": freeze["model"]["sha256"], "threshold": .5,
              "chips": chips, "counts": totals, "micro_metrics": metrics(totals),
              "collections": {key: {"counts": c, "metrics": metrics(c)} for key, c in grouped.items()},
              "wall_seconds": time.perf_counter() - start, "gpu_used": False,
              "scope": "Supervised water vs nonwater on held-out geographic collections; permanent/flood water inseparable",
              "reference_inputs_loaded": False, "retuning_used": False}
    save(result_path, report)
    print(split, json.dumps(report["micro_metrics"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "research/sen1-cpu-training-protocol.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/sen1floods11-cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/sen1-cpu-v1")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.out.exists() and not args.download_only:
        raise FileExistsError("Preserve model/evaluation artifacts; no repeat fitting/evaluation")
    ledger = download_data(protocol, args.cache)
    if args.download_only:
        return
    args.out.mkdir(parents=True)
    model, training = train(protocol, args.cache, args.out)
    sources = [ROOT / "src/akash_flood/flood_model.py", Path(__file__), ROOT / "uv.lock"]
    freeze = {"frozen_at": datetime.now(timezone.utc).isoformat(), "protocol": record(args.protocol),
              "model": training["model"], "training_report": record(args.out / "training-report.json"),
              "dataset_download_ledger": record(args.cache / "downloads.json"), "code": [record(p) for p in sources],
              "scikit_learn_version": sklearn.__version__, "python_version": platform.python_version(),
              "threshold": .5, "holdout_labels_loaded": False, "reference_inputs_loaded": False,
              "pre_fit_loader_corrections": record(ROOT / "research/sen1-pre-fit-loader-corrections.json"),
              "source_grid_metadata_audit": record(ROOT / "research/sen1-loader-affine-serialization-audit.json"),
              "forbidden_inputs": protocol["forbidden_inputs"], "gpu_used": False}
    save(args.out / "model-evaluation-freeze.json", freeze)
    evaluate(model, protocol["splits"]["validation"], "validation", args.cache, args.out, freeze)
    # Create a persistent receipt before opening a final-test label array.
    # A failure is recorded rather than silently repeated as a new experiment.
    receipt = args.cache / "final-test-evaluation-receipt.json"
    if receipt.exists():
        raise FileExistsError("A final-test evaluation was already started")
    save(receipt, {"started_at": datetime.now(timezone.utc).isoformat(), "freeze": record(args.out / "model-evaluation-freeze.json"),
                   "status": "started; final-test labels may now be opened"})
    final = evaluate(model, protocol["splits"]["final_test"], "final_test", args.cache, args.out, freeze)
    save(receipt, {"completed_at": datetime.now(timezone.utc).isoformat(), "freeze": record(args.out / "model-evaluation-freeze.json"),
                   "result": record(args.out / "final_test-evaluation.json"), "status": "completed once"})
    save(args.out / "summary.json", {"completed_at": datetime.now(timezone.utc).isoformat(), "model": training["model"],
         "protocol_sha256": PROTOCOL_SHA, "training_pixels": training["training_pixels"],
         "final_test_metrics": final["micro_metrics"], "gpu_used": False,
         "maximum_resident_set_size_raw": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
         "resident_size_platform": platform.system(), "limitations": protocol["limitations"]})


if __name__ == "__main__":
    main()
