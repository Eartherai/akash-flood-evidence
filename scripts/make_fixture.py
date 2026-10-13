"""Create small, explicitly synthetic inputs for engineering checks only."""
import json
from pathlib import Path
import sys

import numpy as np
from pyproj import Transformer
import rasterio
from rasterio.transform import from_origin


def create(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    grid = from_origin(340000, 3120000, 10, 10)
    profile = dict(driver="GTiff", height=64, width=64, count=1, dtype="float32",
                   crs="EPSG:32645", transform=grid, nodata=-9999)
    pre = np.full((64, 64), -10, dtype=np.float32)
    post = pre.copy()
    post[18:48, 28:36] = -24
    pre[:, 45:48] = post[:, 45:48] = -23  # pre-existing water fixture
    for name, array in [("pre", pre), ("post", post), ("slope", np.zeros_like(pre))]:
        with rasterio.open(out / f"{name}.tif", "w", **profile) as dst:
            dst.write(array, 1)
    transform = Transformer.from_crs(32645, 4326, always_xy=True).transform
    positions = [(50, 320), (220, 320), (410, 320), (590, 320)]
    nodes = []
    for i, (x, y) in enumerate(positions, 1):
        lon, lat = transform(340000+x, 3120000-y)
        nodes.append(dict(type="node", id=i, lon=lon, lat=lat))
    ways = [dict(type="way", id=10, nodes=[1, 2, 3, 4], tags={"highway": "residential"})]
    for way_id, x, y in [(100, 300, 240), (101, 100, 240)]:
        ids = []
        for dx, dy in [(0, 0), (20, 0), (20, 20), (0, 20)]:
            identity = len(nodes) + 1
            lon, lat = transform(340000+x+dx, 3120000-y-dy)
            nodes.append(dict(type="node", id=identity, lon=lon, lat=lat))
            ids.append(identity)
        ways.append(dict(type="way", id=way_id, nodes=ids+[ids[0]], tags={"building": "yes"}))
    (out / "osm.json").write_text(json.dumps({"elements": nodes + ways}))
    def point(i, name):
        n = nodes[i-1]
        return dict(id=name, name=name, lon=n["lon"], lat=n["lat"])
    metadata = dict(relative_orbit=85, orbit_state="ascending", polarization="VV", units="dB",
                    calibrated=True, terrain_corrected=True, backscatter="sigma0",
                    processing_record="Synthetic array fixture; not satellite processing")
    config = dict(event_date="2026-08-26", data_kind="synthetic_fixture",
                  before=dict(**metadata, path="pre.tif", scene_id="SYNTHETIC-BEFORE", datetime="2026-08-16T12:00:00Z"),
                  after=dict(**metadata, path="post.tif", scene_id="SYNTHETIC-AFTER", datetime="2026-08-28T12:00:00Z"),
                  slope="slope.tif", osm="osm.json", osm_snapshot="2026-07-27",
                  settlements=[point(1, "Fixture West"), point(3, "Fixture East")],
                  destinations=[point(4, "Fixture Hospital")], max_snap_m=25)
    (out / "run.json").write_text(json.dumps(config, indent=2))
    return out / "run.json"


if __name__ == "__main__":
    print(create(sys.argv[1] if len(sys.argv)>1 else "data/synthetic-fixture"))
