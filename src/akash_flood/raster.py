"""Strict raster contract and geographic exports."""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import shapes
from rasterio.warp import transform_geom


def load_aligned(paths):
    """Refuse silent resampling, absent CRS and inconsistent georeferencing."""
    arrays, validity, profile = [], None, None
    for path in paths:
        with rasterio.open(path) as ds:
            if ds.crs is None or ds.count != 1:
                raise ValueError(f"{Path(path).name}: expected one band and a CRS")
            if profile is not None:
                if (ds.crs != profile["crs"] or ds.shape != (profile["height"], profile["width"])
                        or not ds.transform.almost_equals(profile["transform"], precision=1e-9)):
                    raise ValueError("Rasters must have identical CRS, shape and pixel grid")
            else:
                profile = ds.profile.copy()
            data = ds.read(1, masked=True).astype(np.float64)
            current = ~np.ma.getmaskarray(data) & np.isfinite(data.filled(np.nan))
            validity = current if validity is None else validity & current
            arrays.append(data.filled(np.nan))
    if not arrays:
        raise ValueError("At least one raster is required")
    return arrays, validity, profile


def metric_pixel_area(profile):
    crs = profile["crs"]
    if not crs.is_projected or crs.linear_units not in ("metre", "meter"):
        raise ValueError("Area reporting requires a projected raster with metre units")
    t = profile["transform"]
    return abs(t.a * t.e - t.b * t.d)


def export_labels(path, classes, profile):
    p = profile.copy()
    p.update(dtype="uint8", count=1, nodata=255, compress="deflate")
    with rasterio.open(path, "w", **p) as ds:
        ds.write(classes, 1)
        ds.update_tags(classes="0=no_new_candidate,1=new_water_candidate,2=preexisting_dark,255=unassessed")


def candidate_geojson(classes, profile):
    features = []
    for geom, _ in shapes(classes, mask=classes == 1, transform=profile["transform"]):
        features.append({"type": "Feature", "id": f"flood-candidate-{len(features)+1}",
                         "properties": {"kind": "new_water_candidate", "verified_damage": False},
                         "geometry": transform_geom(profile["crs"], "EPSG:4326", geom)})
    return {"type": "FeatureCollection", "features": features}
