"""CPU screening baseline. Outputs are candidates, not verified flood damage.

Inputs must already be calibrated, terrain-corrected, co-registered Sentinel-1
backscatter in dB. Raw GRD amplitude must never be passed to this detector.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class FloodConfig:
    water_threshold_db: float = -17.0
    change_threshold_db: float = -3.0
    max_slope_deg: float = 5.0
    min_component_pixels: int = 8

    def __post_init__(self):
        if not np.isfinite(self.water_threshold_db):
            raise ValueError("Water threshold must be finite")
        if not np.isfinite(self.change_threshold_db) or self.change_threshold_db >= 0:
            raise ValueError("Change threshold must be a negative finite dB value")
        if not 0 <= self.max_slope_deg <= 90:
            raise ValueError("Slope threshold must be between 0 and 90 degrees")
        if self.min_component_pixels < 1:
            raise ValueError("Minimum component size must be positive")


def classify_change(pre_db, post_db, *, valid=None, slope_deg=None,
                    radar_invalid=None, config=None):
    """Return uint8 labels: 0 no candidate, 1 new-water candidate,
    2 pre-existing dark/water candidate, 255 unassessed.

    A single pre-event observation cannot establish permanent water. Slope is
    only a screening mask, not a physical SAR layover/shadow mask.
    """
    config = config or FloodConfig()
    pre = np.asarray(pre_db, dtype=np.float64)
    post = np.asarray(post_db, dtype=np.float64)
    if pre.ndim != 2 or pre.shape != post.shape:
        raise ValueError("Input arrays must be matching 2-D rasters")
    eligible = np.isfinite(pre) & np.isfinite(post)
    for label, array in [("valid", valid), ("slope_deg", slope_deg),
                         ("radar_invalid", radar_invalid)]:
        if array is not None and np.asarray(array).shape != pre.shape:
            raise ValueError(f"{label} must match input shape")
    if valid is not None:
        eligible &= np.asarray(valid, dtype=bool)
    if slope_deg is not None:
        slope = np.asarray(slope_deg)
        eligible &= np.isfinite(slope) & (slope >= 0) & (slope <= config.max_slope_deg)
    if radar_invalid is not None:
        eligible &= ~np.asarray(radar_invalid, dtype=bool)

    pre_dark = pre < config.water_threshold_db
    candidates = (eligible & ~pre_dark & (post < config.water_threshold_db)
                  & ((post - pre) <= config.change_threshold_db))
    labels, _ = ndimage.label(candidates)  # 4-connected; do not bridge diagonals
    counts = np.bincount(labels.ravel())
    keep = counts >= config.min_component_pixels
    keep[0] = False
    candidates = keep[labels]
    classes = np.full(pre.shape, 255, dtype=np.uint8)
    classes[eligible] = 0
    classes[eligible & pre_dark] = 2
    classes[candidates] = 1
    return classes, {
        "thresholds": asdict(config),
        "total_pixels": int(pre.size),
        "assessed_pixels": int(eligible.sum()),
        "unassessed_pixels": int((~eligible).sum()),
        "new_water_candidate_pixels": int(candidates.sum()),
        "preexisting_dark_pixels": int((eligible & pre_dark).sum()),
        "interpretation": "Screening candidates; not validated inundation or structural damage",
        "has_slope_mask": slope_deg is not None,
        "has_physical_radar_invalid_mask": radar_invalid is not None,
    }
