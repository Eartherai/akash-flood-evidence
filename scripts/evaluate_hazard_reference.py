"""Evaluation-only comparison with EMSR927 AOI01 mapped landslide extent.

This script must never be imported by a prediction pipeline or copilot. It emits
aggregate diagnostics, not training labels. A candidate protocol can audit the
reference; scoring requires a separately frozen protocol and prediction recipe.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import struct
from typing import Any

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from shapely import from_wkb
from shapely.geometry import mapping
from shapely.ops import transform as transform_geometry, unary_union


MATCHING_SEMANTICS = {"landslide_extent", "mass_movement_extent"}
EXPLORATORY_SEMANTICS = {"radar_change_candidate"}
MAX_DOMAIN_PIXELS = 20_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_gpkg_geometry(blob: bytes):
    """Decode standard GeoPackage binary header before passing WKB to Shapely."""
    if len(blob) < 9 or blob[:2] != b"GP" or blob[2] != 0:
        raise ValueError("Unsupported or malformed GeoPackage geometry header")
    flags = blob[3]
    if flags & 0xF0:
        raise ValueError("Empty, extended, or reserved GeoPackage geometry unsupported")
    envelope_code = (flags >> 1) & 7
    envelope_values = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}
    if envelope_code not in envelope_values:
        raise ValueError("Invalid GeoPackage envelope code")
    byte_order = "<" if flags & 1 else ">"
    srs_id = struct.unpack(byte_order + "i", blob[4:8])[0]
    offset = 8 + 8 * envelope_values[envelope_code]
    if len(blob) <= offset:
        raise ValueError("Truncated GeoPackage geometry")
    geometry = from_wkb(blob[offset:])
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("Reference geometry is empty or invalid; automatic repair refused")
    if geometry.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Reference extent must be polygonal")
    return geometry, srs_id


def load_reference(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual_sha = sha256(path)
    if actual_sha != expected_sha256:
        raise ValueError("Reference GeoPackage checksum differs from protocol")
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        table_names = ["areaOfInterestA_v1", "imageFootprintA_v1", "observedEventA_v1"]
        records = {}
        for table in table_names:
            # Table names are constants, never user-controlled SQL.
            schema = connection.execute(
                "SELECT srs_id FROM gpkg_geometry_columns WHERE table_name = ?", (table,)
            ).fetchone()
            if schema is None or schema[0] != 4326:
                raise ValueError(f"Unexpected reference CRS or missing layer: {table}")
            records[table] = [dict(row) for row in connection.execute(f'SELECT * FROM "{table}"')]
        if len(records["areaOfInterestA_v1"]) != 1:
            raise ValueError("Protocol expects one AOI")
        aoi_record = records["areaOfInterestA_v1"][0]
        if (aoi_record["emsr_id"], aoi_record["area_id"]) != ("EMSR927", "01"):
            raise ValueError("Wrong reference activation or AOI")
        event_records = records["observedEventA_v1"]
        if not event_records or any(
            (row["event_type"], row["obj_desc"], row["det_method"])
            != ("6-Mass Movement", "Landslide", "Photo-interpretation")
            for row in event_records
        ):
            raise ValueError("Reference does not match the declared photo-interpreted landslide semantics")
        source_ids = sorted({row["dmg_src_id"] for row in event_records})
        # Only footprints of the image used for observed-event interpretation.
        image_records = [row for row in records["imageFootprintA_v1"] if row["or_src_id"] in source_ids]
        if not image_records or set(source_ids) != {row["or_src_id"] for row in image_records}:
            raise ValueError("Observed-event source image footprint missing")

        def geometries(rows):
            result = []
            for row in rows:
                geometry, srs_id = decode_gpkg_geometry(row["geom"])
                if srs_id != 4326:
                    raise ValueError("GeoPackage feature CRS differs from layer CRS")
                result.append(geometry)
            return result

        aoi = unary_union(geometries([aoi_record]))
        event = unary_union(geometries(event_records))
        image_footprint = unary_union(geometries(image_records))
        domain = aoi.intersection(image_footprint)
        if domain.is_empty or not domain.is_valid:
            raise ValueError("Mapped AOI/image-footprint domain is empty or invalid")
        return {
            "sha256": actual_sha, "aoi": aoi, "event": event, "domain": domain,
            "metadata": {"activation": "EMSR927", "aoi": "01", "locality": aoi_record["locality"],
                         "event_type": "6-Mass Movement", "object_description": "Landslide",
                         "method": "Photo-interpretation", "event_features": len(event_records),
                         "observed_event_source_ids": source_ids,
                         "source_reported_area_ha": sum(row["area"] for row in event_records)},
        }
    finally:
        connection.close()


def reference_grid(reference: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if config != {"epsg": 32645, "pixel_size_m": 10, "corner_origin_x": 5, "corner_origin_y": 5,
                  "pixel_center_origin_x": 0, "pixel_center_origin_y": 0, "rasterization": "pixel_center"}:
        raise ValueError("Unsupported evaluation grid; changing it requires a new protocol")
    transformer = Transformer.from_crs(4326, 32645, always_xy=True)
    aoi = transform_geometry(transformer.transform, reference["aoi"])
    domain = transform_geometry(transformer.transform, reference["domain"])
    event = transform_geometry(transformer.transform, reference["event"])
    left, bottom, right, top = aoi.bounds
    # SNAP standardGridOriginX/Y=0 refer to centers. GeoTIFF affine coordinates
    # are corners, which are congruent to 5 m on this 10 m lattice.
    left, bottom = math.floor((left - 5) / 10) * 10 + 5, math.floor((bottom - 5) / 10) * 10 + 5
    right, top = math.ceil((right - 5) / 10) * 10 + 5, math.ceil((top - 5) / 10) * 10 + 5
    width, height = int((right - left) / 10), int((top - bottom) / 10)
    if width <= 0 or height <= 0 or width * height > MAX_DOMAIN_PIXELS:
        raise ValueError("Invalid or excessive evaluation domain size")
    affine = Affine(10, 0, left, 0, -10, top)
    shape = (height, width)

    def burn(geometry):
        return rasterize([(mapping(geometry), 1)], out_shape=shape, transform=affine,
                         fill=0, all_touched=False, dtype="uint8").astype(bool)

    aoi_mask, domain_mask = burn(aoi), burn(domain)
    event_mask = burn(event) & domain_mask
    if not np.any(event_mask):
        raise ValueError("Mapped landslide occupies no evaluation pixels")
    return {"shape": shape, "transform": affine, "crs": rasterio.crs.CRS.from_epsg(32645),
            "aoi": aoi_mask, "domain": domain_mask, "event": event_mask,
            "vector_mapped_event_area_m2": event.intersection(domain).area}


def validate_source_grid(dataset) -> None:
    affine = dataset.transform
    if dataset.count != 1 or dataset.crs != rasterio.crs.CRS.from_epsg(32645):
        raise ValueError("Prediction and valid mask must each be one EPSG:32645 band")
    if not (math.isclose(affine.a, 10, abs_tol=1e-9) and math.isclose(affine.e, -10, abs_tol=1e-9)
            and abs(affine.b) < 1e-12 and abs(affine.d) < 1e-12):
        raise ValueError("Rasters must be north-up 10 m; evaluator does not resample predictions")
    if any(not math.isclose((value - 5) / 10, round((value - 5) / 10), abs_tol=1e-7)
           for value in (affine.c, affine.f)):
        raise ValueError("Rasters must align to SNAP center-origin-0 grid (10 m corners congruent to 5 m)")


def place_on_grid(prediction_path: Path, valid_mask_path: Path, grid: dict[str, Any]):
    """Copy aligned cells, padding absent coverage as invalid rather than background."""
    with rasterio.open(prediction_path) as prediction, rasterio.open(valid_mask_path) as mask:
        validate_source_grid(prediction)
        validate_source_grid(mask)
        if (prediction.shape, prediction.transform, prediction.crs) != (mask.shape, mask.transform, mask.crs):
            raise ValueError("Prediction and validity mask grids differ")
        if prediction.width * prediction.height > MAX_DOMAIN_PIXELS:
            raise ValueError("Prediction exceeds evaluator memory budget; crop without changing its grid")
        source_prediction = prediction.read(1, masked=True)
        source_mask = mask.read(1, masked=True)
        unmasked_values = source_mask.compressed()
        if not np.all(np.isin(unmasked_values, [0, 1])):
            raise ValueError("Valid-mask unmasked pixels must be binary 0/1")
        source_valid = source_mask.filled(0) == 1
        if np.any(source_valid & np.ma.getmaskarray(source_prediction)):
            raise ValueError("Valid mask asserts availability at prediction nodata")
        if not np.all(np.isin(source_prediction.data[source_valid], [0, 1])):
            raise ValueError("Prediction must be binary 0/1 wherever valid")
        row_offset = round((grid["transform"].f - prediction.transform.f) / 10)
        col_offset = round((prediction.transform.c - grid["transform"].c) / 10)
        destination = np.zeros(grid["shape"], dtype=bool)
        valid = np.zeros(grid["shape"], dtype=bool)
        dest_r0, dest_c0 = max(0, row_offset), max(0, col_offset)
        dest_r1 = min(grid["shape"][0], row_offset + prediction.height)
        dest_c1 = min(grid["shape"][1], col_offset + prediction.width)
        if dest_r1 > dest_r0 and dest_c1 > dest_c0:
            src = (slice(dest_r0 - row_offset, dest_r1 - row_offset),
                   slice(dest_c0 - col_offset, dest_c1 - col_offset))
            dst = (slice(dest_r0, dest_r1), slice(dest_c0, dest_c1))
            destination[dst] = source_prediction.data[src] == 1
            valid[dst] = source_valid[src]
        metadata = {"source_shape": list(prediction.shape), "source_transform": list(prediction.transform)[:6],
                    "placement": "exact aligned cell copy; absent cells invalid; no resampling"}
        return destination, valid, metadata


def ratio(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def extent_agreement(predicted: np.ndarray, reference: np.ndarray, domain: np.ndarray) -> dict[str, Any]:
    intersection = int(np.count_nonzero(predicted & reference & domain))
    candidate_only = int(np.count_nonzero(predicted & ~reference & domain))
    reference_only = int(np.count_nonzero(~predicted & reference & domain))
    neither = int(np.count_nonzero(~predicted & ~reference & domain))
    return {
        "pixel_contingency": {"intersection": intersection, "candidate_only": candidate_only,
                              "reference_only": reference_only, "neither": neither},
        "iou": ratio(intersection, intersection + candidate_only + reference_only),
        "dice": ratio(2 * intersection, 2 * intersection + candidate_only + reference_only),
        "precision_relative_to_mapped_extent": ratio(intersection, intersection + candidate_only),
        "recall_on_observed_mapped_extent": ratio(intersection, intersection + reference_only),
        "agreement_fraction": ratio(intersection + neither, int(np.count_nonzero(domain))),
    }


def calculate_metrics(predicted: np.ndarray, valid: np.ndarray, grid: dict[str, Any],
                      semantics: str, coverage_gates: dict[str, float]) -> dict[str, Any]:
    if semantics not in MATCHING_SEMANTICS | EXPLORATORY_SEMANTICS:
        raise ValueError("Semantically incompatible prediction: this reference is landslide, not open-water flood")
    if predicted.shape != grid["shape"] or valid.shape != grid["shape"]:
        raise ValueError("Evaluation array shape mismatch")
    aoi, domain, event = grid["aoi"], grid["domain"], grid["event"]
    observed = domain & valid
    aoi_count, domain_count, event_count = [int(np.count_nonzero(x)) for x in (aoi, domain, event)]
    observed_count = int(np.count_nonzero(observed))
    observed_event = int(np.count_nonzero(event & valid))
    candidate_count = int(np.count_nonzero(predicted & observed))
    overlap = int(np.count_nonzero(predicted & event & observed))
    aoi_coverage = ratio(int(np.count_nonzero(aoi & valid)), aoi_count)
    domain_coverage = ratio(observed_count, domain_count)
    event_coverage = ratio(observed_event, event_count)
    passes = (domain_coverage is not None and event_coverage is not None
              and domain_coverage >= coverage_gates["mapped_domain_valid_fraction"]
              and event_coverage >= coverage_gates["mapped_event_valid_fraction"])
    result = {
        "coverage": {"aoi_pixels": aoi_count, "mapped_domain_pixels": domain_count,
                     "mapped_event_pixels": event_count, "valid_domain_pixels": observed_count,
                     "valid_event_pixels": observed_event, "unobservable_event_pixels": event_count - observed_event,
                     "aoi_valid_fraction": aoi_coverage, "mapped_domain_valid_fraction": domain_coverage,
                     "mapped_event_valid_fraction": event_coverage, "predeclared_gates": coverage_gates,
                     "passes_coverage_gates": passes},
        "spatial_overlap_diagnostics": {
            "observed_candidate_pixels": candidate_count, "candidate_event_intersection_pixels": overlap,
            "candidate_outside_mapped_event_pixels": candidate_count - overlap,
            "mapped_event_intersected_fraction_all_domain": ratio(overlap, event_count),
            "mapped_event_intersected_fraction_observed_only": ratio(overlap, observed_event),
            "observed_candidate_area_m2": candidate_count * 100,
            "candidate_event_intersection_area_m2": overlap * 100,
            "mapped_event_raster_area_m2": event_count * 100,
            "mapped_event_vector_area_m2": grid.get("vector_mapped_event_area_m2"),
        },
        "score_status": "coverage_qualified_mapped_extent_agreement" if passes else "partial_coverage_diagnostics_only",
    }
    if semantics in MATCHING_SEMANTICS:
        result["mapped_extent_agreement_on_valid_pixels"] = extent_agreement(predicted, event, observed)
        result["constant_baselines_same_valid_domain"] = {
            "all_background": extent_agreement(np.zeros_like(predicted), event, observed),
            "all_event": extent_agreement(np.ones_like(predicted), event, observed),
        }
    else:
        result["score_status"] = "exploratory_radar_change_overlap_only"
        result["semantic_limitation"] = "Radar change is not a landslide classifier; no IoU, Dice, precision or accuracy score reported."
    return result


def load_protocol(path: Path, scoring: bool) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != "emsr927-hazard-reference-v1":
        raise ValueError("Unknown protocol version")
    if scoring:
        if protocol.get("status") != "frozen" or not protocol.get("frozen_at"):
            raise ValueError("Candidate protocol cannot score predictions; freeze the recipe and protocol first")
        if protocol.get("evaluator_sha256") != sha256(Path(__file__)):
            raise ValueError("Evaluator differs from frozen protocol")
        for field in ("prediction_recipe_sha256", "prediction_sha256", "valid_mask_sha256"):
            digest = protocol.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"Frozen protocol requires exact {field}")
    gates = protocol.get("coverage_gates")
    if not isinstance(gates, dict) or set(gates) != {"mapped_domain_valid_fraction", "mapped_event_valid_fraction"}:
        raise ValueError("Protocol must declare both coverage gates")
    if any(not isinstance(x, (int, float)) or not 0 < x <= 1 for x in gates.values()):
        raise ValueError("Coverage gates must lie in (0, 1]")
    return protocol


def validate_prediction_manifest(path: Path, prediction: Path, valid_mask: Path, protocol: dict[str, Any]):
    manifest = json.loads(path.read_text())
    prediction_sha, valid_mask_sha = sha256(prediction), sha256(valid_mask)
    if manifest.get("prediction_sha256") != prediction_sha or manifest.get("valid_mask_sha256") != valid_mask_sha:
        raise ValueError("Prediction or valid mask checksum differs from manifest")
    if protocol.get("prediction_sha256") != prediction_sha or protocol.get("valid_mask_sha256") != valid_mask_sha:
        raise ValueError("Prediction or valid mask differs from frozen protocol")
    recipe = Path(manifest["prediction_recipe_path"])
    if not recipe.is_absolute():
        recipe = path.parent / recipe
    if sha256(recipe) != protocol["prediction_recipe_sha256"]:
        raise ValueError("Prediction recipe differs from frozen protocol")
    if manifest.get("evaluation_reference_used_for_prediction") is not False:
        raise ValueError("Manifest must explicitly declare that evaluation reference was not used for prediction")
    if manifest.get("valid_mask_derivation") != "sensor_geometry_and_data_availability_only":
        raise ValueError("Validity mask must be declared independent of predicted/reference event presence")
    if manifest.get("prediction_semantics") not in MATCHING_SEMANTICS | EXPLORATORY_SEMANTICS:
        raise ValueError("Prediction semantics do not match the mapped event")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--audit-reference", action="store_true")
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--valid-mask", type=Path)
    parser.add_argument("--prediction-manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".started.json").exists():
        raise SystemExit("Output or start record exists; refusing to replace evaluation evidence")
    if not args.audit_reference and not all([args.prediction, args.valid_mask, args.prediction_manifest]):
        parser.error("scoring requires --prediction, --valid-mask and --prediction-manifest")
    protocol = load_protocol(args.protocol, scoring=not args.audit_reference)
    if args.audit_reference:
        manifest = None
    else:
        manifest = validate_prediction_manifest(args.prediction_manifest, args.prediction, args.valid_mask, protocol)
    reference_path = Path(protocol["reference"]["path"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = {"started_at": datetime.now(timezone.utc).isoformat(), "mode": "reference_audit" if args.audit_reference else "one_time_evaluation",
             "protocol_sha256": sha256(args.protocol), "evaluator_sha256": sha256(Path(__file__))}
    with args.output.with_suffix(".started.json").open("x") as stream:
        json.dump(start, stream, indent=2)
    reference = load_reference(reference_path, protocol["reference"]["sha256"])
    grid = reference_grid(reference, protocol["grid"])
    result = {**start, "reference": {"path": str(reference_path), "sha256": reference["sha256"], **reference["metadata"]},
              "evaluation_grid": {**protocol["grid"], "shape": list(grid["shape"]), "transform": list(grid["transform"])[:6]},
              "boundary": "Evaluation only; no polygon/mask outputs and no production/copilot imports.",
              "interpretation": "Agreement with one photo-interpreted map, not validated flood extent or road passability. Unmarked pixels may contain unmapped phenomena. No confidence intervals or generalization claim from one event."}
    if not args.audit_reference:
        predicted, valid, placement = place_on_grid(args.prediction, args.valid_mask, grid)
        result["prediction"] = {"path": str(args.prediction), "sha256": sha256(args.prediction),
                                "valid_mask_path": str(args.valid_mask), "valid_mask_sha256": sha256(args.valid_mask),
                                "manifest_sha256": sha256(args.prediction_manifest), "recipe_sha256": protocol["prediction_recipe_sha256"],
                                "semantics": manifest["prediction_semantics"], "leakage_attestation": "self-declared, not independently proven", **placement}
        result["metrics"] = calculate_metrics(predicted, valid, grid, manifest["prediction_semantics"], protocol["coverage_gates"])
    else:
        result["reference_audit"] = {"aoi_grid_pixels": int(np.count_nonzero(grid["aoi"])),
                                     "mapped_domain_grid_pixels": int(np.count_nonzero(grid["domain"])),
                                     "mapped_event_grid_pixels": int(np.count_nonzero(grid["event"])),
                                     "mapped_event_vector_area_m2": grid["vector_mapped_event_area_m2"],
                                     "prediction_metrics_computed": False}
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "mode": start["mode"],
                      "score_status": result.get("metrics", {}).get("score_status", "reference_audit_only")}, indent=2))


if __name__ == "__main__":
    main()
