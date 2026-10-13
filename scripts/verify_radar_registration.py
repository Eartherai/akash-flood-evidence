"""CPU-only, reference-free relative radar registration diagnostics.

Input: co-gridded before/after single-band VV *linear power*, plus an independent
common binary validity mask. The diagnostic does not alter rasters or establish
absolute geolocation accuracy. Event changes and radar speckle can move or flatten
correlation peaks; every selected tile is reported, including uninformative ones.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import rasterio
from scipy.fft import irfft2, next_fast_len, rfft2


TILE_SIZE = 256
STRIDE = 256
MAX_TILES = 12
MIN_VALID_FRACTION = 0.8
SEARCH_RADIUS = 3
MIN_PAIR_FRACTION = 0.5
MIN_VARIANCE_DB2 = 1e-10
AMBIGUOUS_PEAK_MARGIN = 0.02
MAX_INPUT_PIXELS = 20_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inputs(before_path: Path, after_path: Path, valid_path: Path):
    with rasterio.open(before_path) as before, rasterio.open(after_path) as after, rasterio.open(valid_path) as mask:
        for dataset in (before, after, mask):
            affine = dataset.transform
            if dataset.count != 1 or dataset.crs != rasterio.crs.CRS.from_epsg(32645):
                raise ValueError("Each input must be one EPSG:32645 band")
            if not (math.isclose(affine.a, 10, abs_tol=1e-9) and math.isclose(affine.e, -10, abs_tol=1e-9)
                    and abs(affine.b) < 1e-12 and abs(affine.d) < 1e-12):
                raise ValueError("Each input must be north-up at 10 m")
            if any(not math.isclose((value - 5) / 10, round((value - 5) / 10), abs_tol=1e-7)
                   for value in (affine.c, affine.f)):
                raise ValueError("Input corners must be congruent to 5 m on SNAP center-origin-0 grid")
        signatures = [(d.shape, d.transform, d.crs) for d in (before, after, mask)]
        if not all(signature == signatures[0] for signature in signatures):
            raise ValueError("Before, after and validity rasters must share the exact grid; no resampling is performed")
        if before.width * before.height > MAX_INPUT_PIXELS:
            raise ValueError("Input exceeds diagnostic memory budget; crop all inputs to the same aligned extent")
        before_values, after_values, mask_values = [d.read(1, masked=True) for d in (before, after, mask)]
        if not np.all(np.isin(mask_values.compressed(), [0, 1])):
            raise ValueError("Valid-mask unmasked values must be binary 0/1")
        valid = mask_values.filled(0) == 1
        for label, values in (("before", before_values), ("after", after_values)):
            if np.any(valid & np.ma.getmaskarray(values)):
                raise ValueError(f"Validity mask asserts availability over {label} nodata")
            if not np.all(np.isfinite(values.data[valid]) & (values.data[valid] > 0)):
                raise ValueError(f"{label} must contain finite positive linear power wherever valid")
        metadata = {"shape": list(before.shape), "crs": before.crs.to_string(),
                    "transform": list(before.transform)[:6], "grid_validation": "exact identical affine/CRS/shape",
                    "units_contract": "user declares positive VV linear power; no source radiometry calibration is independently checked"}
        return before_values.data, after_values.data, valid, metadata


def fixed_tile_selection(valid: np.ndarray):
    """Select by geometry/mask coverage before looking at image correlation."""
    if valid.ndim != 2:
        raise ValueError("Validity must be a two-dimensional array")
    candidates = []
    for row in range(0, valid.shape[0] - TILE_SIZE + 1, STRIDE):
        for col in range(0, valid.shape[1] - TILE_SIZE + 1, STRIDE):
            count = int(np.count_nonzero(valid[row:row + TILE_SIZE, col:col + TILE_SIZE]))
            candidates.append({"row": row, "col": col, "valid_pixels": count,
                               "valid_fraction": count / (TILE_SIZE * TILE_SIZE),
                               "eligible": count >= math.ceil(MIN_VALID_FRACTION * TILE_SIZE * TILE_SIZE)})
    eligible = [tile for tile in candidates if tile["eligible"]]
    if len(eligible) <= MAX_TILES:
        selected = eligible
    else:
        # Equally spaced indices in row-major eligible-tile order. No image-value,
        # texture, peak or estimated-shift criterion affects selection.
        indices = np.rint(np.linspace(0, len(eligible) - 1, MAX_TILES)).astype(int)
        selected = [eligible[int(index)] for index in indices]
    return {"full_tile_count": len(candidates), "eligible_tile_count": len(eligible),
            "selected_tile_count": len(selected), "selected": selected,
            "all_candidate_tiles": candidates,
            "excluded_edge_pixels": int(valid.size - len(candidates) * TILE_SIZE * TILE_SIZE)}


def fft_correlation_scores(before_db: np.ndarray, after_db: np.ndarray, valid: np.ndarray):
    """Masked zero-mean NCC for every integer offset in the fixed ±3 px search.

    Offset (dy, dx) compares before[r,c] to after[r+dy,c+dx]. FFT padding makes
    this linear correlation, with neither circular wrapping nor moved output.
    Six cross-correlations supply overlap count, sums, squares and products.
    """
    if before_db.shape != after_db.shape or before_db.shape != valid.shape or valid.ndim != 2:
        raise ValueError("Correlation array shapes differ")
    if not np.any(valid):
        return []
    before_centered = np.where(valid, before_db - np.mean(before_db[valid]), 0).astype("float64")
    after_centered = np.where(valid, after_db - np.mean(after_db[valid]), 0).astype("float64")
    weights = valid.astype("float64")
    padded = tuple(next_fast_len(2 * length - 1) for length in valid.shape)
    before_fft = [rfft2(x, s=padded, workers=1) for x in (weights, before_centered, before_centered ** 2)]
    after_fft = [rfft2(x, s=padded, workers=1) for x in (weights, after_centered, after_centered ** 2)]

    def corr(after_index, before_index):
        return irfft2(after_fft[after_index] * np.conj(before_fft[before_index]), s=padded, workers=1)

    count = corr(0, 0)
    sum_before, sum_after = corr(0, 1), corr(1, 0)
    sum_before_sq, sum_after_sq, product = corr(0, 2), corr(2, 0), corr(1, 1)
    rows = []
    for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            index = (dy % padded[0], dx % padded[1])
            n = max(0, int(round(count[index])))
            before_ss = float(sum_before_sq[index] - sum_before[index] ** 2 / n) if n else 0
            after_ss = float(sum_after_sq[index] - sum_after[index] ** 2 / n) if n else 0
            eligible = (n >= math.ceil(MIN_PAIR_FRACTION * valid.size)
                        and before_ss > MIN_VARIANCE_DB2 * n and after_ss > MIN_VARIANCE_DB2 * n)
            if eligible:
                numerator = float(product[index] - sum_before[index] * sum_after[index] / n)
                score = float(np.clip(numerator / math.sqrt(before_ss * after_ss), -1, 1))
            else:
                score = None
            rows.append({"dy_px": dy, "dx_px": dx, "pair_pixels": n,
                         "pair_fraction": n / valid.size, "ncc": score,
                         "before_variance_db2": max(0, before_ss / n) if n else None,
                         "after_variance_db2": max(0, after_ss / n) if n else None})
    return rows


def diagnose_tile(before_power: np.ndarray, after_power: np.ndarray, valid: np.ndarray):
    before_db, after_db = np.zeros(valid.shape), np.zeros(valid.shape)
    before_db[valid] = 10 * np.log10(before_power[valid])
    after_db[valid] = 10 * np.log10(after_power[valid])
    scores = fft_correlation_scores(before_db, after_db, valid)
    eligible = [row for row in scores if row["ncc"] is not None]
    if not eligible:
        return {"status": "uninformative_texture_or_pair_coverage", "estimated_after_displacement_px": None,
                "all_offset_scores": scores, "flags": ["No offset has sufficient nonconstant paired signal"]}
    eligible.sort(key=lambda row: (-row["ncc"], row["dy_px"] ** 2 + row["dx_px"] ** 2,
                                   abs(row["dy_px"]), abs(row["dx_px"]), row["dy_px"], row["dx_px"]))
    best = eligible[0]
    second = eligible[1] if len(eligible) > 1 else None
    zero = next(row for row in scores if (row["dy_px"], row["dx_px"]) == (0, 0))
    margin = best["ncc"] - second["ncc"] if second else None
    flags = []
    if abs(best["dy_px"]) == SEARCH_RADIUS or abs(best["dx_px"]) == SEARCH_RADIUS:
        flags.append("Peak lies on search boundary; displacement outside ±3 px is not ruled out")
    if margin is not None and margin < AMBIGUOUS_PEAK_MARGIN:
        flags.append("Peak margin below fixed 0.02 diagnostic heuristic; multiple offsets fit similarly")
    return {"status": "relative_content_correlation_diagnostic",
            "estimated_after_displacement_px": {"dy": best["dy_px"], "dx": best["dx_px"]},
            "estimated_after_displacement_m": {"south": best["dy_px"] * 10, "east": best["dx_px"] * 10},
            "best_ncc": best["ncc"], "zero_offset_ncc": zero["ncc"],
            "best_minus_zero_ncc": best["ncc"] - zero["ncc"] if zero["ncc"] is not None else None,
            "best_minus_second_ncc": margin, "best_pair_pixels": best["pair_pixels"],
            "flags": flags, "all_offset_scores": scores}


def run_diagnostics(before_power: np.ndarray, after_power: np.ndarray, valid: np.ndarray):
    if before_power.shape != after_power.shape or before_power.shape != valid.shape:
        raise ValueError("Diagnostic array shapes differ")
    selection = fixed_tile_selection(valid)
    reports = []
    for window in selection["selected"]:
        row, col = window["row"], window["col"]
        index = (slice(row, row + TILE_SIZE), slice(col, col + TILE_SIZE))
        reports.append({**window, **diagnose_tile(before_power[index], after_power[index], valid[index])})
    informative = [tile for tile in reports if tile["estimated_after_displacement_px"] is not None]
    shifts = [tile["estimated_after_displacement_px"] for tile in informative]
    summary = {
        "selected_tiles_reported": len(reports), "informative_tile_count": len(informative),
        "uninformative_tile_count": len(reports) - len(informative),
        "search_boundary_peak_count": sum(any("search boundary" in flag for flag in tile["flags"]) for tile in reports),
        "ambiguous_peak_count": sum(any("Peak margin" in flag for flag in tile["flags"]) for tile in reports),
        "median_integer_dy_px": float(np.median([shift["dy"] for shift in shifts])) if shifts else None,
        "median_integer_dx_px": float(np.median([shift["dx"] for shift in shifts])) if shifts else None,
        "max_absolute_integer_shift_px": max(max(abs(shift["dy"]), abs(shift["dx"])) for shift in shifts) if shifts else None,
        "median_zero_offset_ncc": float(np.median([tile["zero_offset_ncc"] for tile in informative if tile["zero_offset_ncc"] is not None]))
            if any(tile["zero_offset_ncc"] is not None for tile in informative) else None,
        "verification_claim": "Exact grid compatibility only. Estimated content offsets are diagnostics, not a registration pass/fail or absolute-geolocation validation.",
    }
    return {"selection": selection, "tiles": reports, "summary": summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-vv-power", type=Path, required=True)
    parser.add_argument("--after-vv-power", type=Path, required=True)
    parser.add_argument("--valid-mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start_path = args.output.with_suffix(".started.json")
    if args.output.exists() or start_path.exists():
        raise SystemExit("Output or start record already exists; refusing to overwrite diagnostics")
    before, after, valid, grid = read_inputs(args.before_vv_power, args.after_vv_power, args.valid_mask)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = {"before_vv_power": {"path": str(args.before_vv_power), "sha256": sha256(args.before_vv_power)},
               "after_vv_power": {"path": str(args.after_vv_power), "sha256": sha256(args.after_vv_power)},
               "valid_mask": {"path": str(args.valid_mask), "sha256": sha256(args.valid_mask)}}
    record = {"started_at": datetime.now(timezone.utc).isoformat(), "script_sha256": sha256(Path(__file__)),
              "sources": sources, "grid": grid,
              "method": {"intensity_transform": "10log10 of positive VV linear power",
                         "correlation": "linear FFT masked overlap-normalized zero-mean cross-correlation",
                         "tile_size_px": TILE_SIZE, "stride_px": STRIDE, "max_selected_tiles": MAX_TILES,
                         "minimum_tile_valid_fraction": MIN_VALID_FRACTION, "minimum_offset_pair_fraction": MIN_PAIR_FRACTION,
                         "search_radius_px": SEARCH_RADIUS, "minimum_signal_variance_db2": MIN_VARIANCE_DB2,
                         "ambiguous_peak_margin_heuristic": AMBIGUOUS_PEAK_MARGIN,
                         "selection": "All full fixed-lattice tiles counted; eligible mask coverage>=80%; at most12 equally spaced row-major eligible indices chosen before scoring",
                         "tie_break": "highest NCC, then minimum squared displacement, absolute dy/dx, signed dy/dx",
                         "fft_workers": 1},
              "scope": "Relative image-content displacement diagnostics only; no reference map, labels, automatic shifts, GPU, or absolute-geolocation truth.",
              "limitations": ["Different acquisition dates, event change, speckle and terrain radiometry can change or flatten content-correlation peaks.",
                              "A low or boundary correlation peak cannot separate misregistration from changed/ambiguous content.",
                              "Zero estimated integer displacement does not verify subpixel registration.",
                              "No pass/fail threshold has been validated against geolocation control points."]}
    with start_path.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
    started = time.perf_counter()
    record.update(run_diagnostics(before, after, valid))
    record["diagnostic_wall_seconds"] = time.perf_counter() - started
    record["finished_at"] = datetime.now(timezone.utc).isoformat()
    with args.output.open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "selection": {key: record["selection"][key] for key in
          ("full_tile_count", "eligible_tile_count", "selected_tile_count")}, "summary": record["summary"],
          "diagnostic_wall_seconds": record["diagnostic_wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
