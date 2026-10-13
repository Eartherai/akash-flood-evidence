# Saved geographic context viewer

Run from this folder with Python's standard-library server:

```sh
python3 -m http.server 8806 --bind 127.0.0.1 --directory .
```

Open http://127.0.0.1:8806/ . Serve over localhost; file:// cannot load/verify JSON modules. All HTML, modules, data and receipts are local. There is no basemap, external font, paid API, satellite acquisition, graph solve or model inference.

data.json is an exact byte-copy of the preserved compact observation-access payload. context.json adds only joins to hash-bound actual historical target/source points and snapped graph-node coordinates from saved segments. No inferred point-to-road connector is drawn. Actual source and target points are distinct from graph endpoints; operation and physical entrances remain unvalidated.

verification.json binds each original input/new output and lists exact point/route extents, status/scope counts, geometry assertions and the null empty-route case. The frozen protocol's 70 records are hashed before and after this build. Hashes establish preservation, not hazard truth. Browser and human usability/Nepali review are not performed by the builder.

Original/new viewer code is MIT on the explicit owner assertion; CODE-LICENSE.txt does not grant rights over data. Derived historical OpenStreetMap source/road data remains subject to ODbL and attribution. Contains modified Copernicus Sentinel data 2026 and 2024; Copernicus DEM/WorldDEM and Sen1Floods11 retain their upstream terms. No raw/reference geometry, model weights or vendor runtime is added here. The graph data is conditional evidence and not operational guidance.

This is a new standalone output. Earlier viewers, protocols, rasters, graph results, copilot checks and kits are unchanged. It improves geographic inspection only, not hazard accuracy or the arbitrary-area raw-data requirement.
