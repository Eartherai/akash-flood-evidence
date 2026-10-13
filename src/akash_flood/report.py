"""Map-linked report using computed numbers only; not an AI copilot."""
from html import escape
from pathlib import Path

import folium
from shapely.geometry import shape


def write_report(out, manifest, metrics, hazards, roads, access, settlements, destinations, *, buildings=None):
    out = Path(out)
    buildings = buildings or {"type": "FeatureCollection", "features": []}
    coords = [p for p in settlements + destinations]
    center = ([coords[0]["lat"], coords[0]["lon"]] if coords else [28.2, 85.36])
    m = folium.Map(location=center, zoom_start=13, tiles=None, control_scale=True)
    # No present-day basemap mixed with historic evidence. Optional contextual
    # basemap can be added later with its time/role explicitly labeled.
    if hazards["features"]:
        folium.GeoJson(hazards, name="New water candidates", style_function=lambda f: {
            "fillColor": "#2289ba", "color": "#14628b", "weight": 1, "fillOpacity": .5}).add_to(m)
    if roads["features"]:
        folium.GeoJson(roads, name="Intersecting road segments", style_function=lambda f: {
            "color": "#d1642c" if f["properties"]["scenario_closed"] else "#816a24", "weight": 4},
            tooltip=folium.GeoJsonTooltip(fields=["way_id", "overlap_m", "bridge", "scenario_closed"])).add_to(m)
    if buildings["features"]:
        folium.GeoJson(buildings, name="Intersecting building footprints", style_function=lambda f: {
            "color": "#813d76", "fillColor": "#bc83b2", "fillOpacity": .7, "weight": 2},
            tooltip=folium.GeoJsonTooltip(fields=["osm_id", "overlap_m2", "confirmed_damage"])).add_to(m)
    by_id = {a["id"]: a for a in access}
    for p in coords:
        status = by_id.get(p["id"], {}).get("status", "destination")
        folium.CircleMarker([p["lat"], p["lon"]], radius=6, color="#202a38", fill=True,
                            fill_color="#ed8b46" if status == "newly_disconnected" else "#fff",
                            fill_opacity=1, tooltip=escape(p["name"]),
                            popup=f"{escape(p['name'])}: {escape(status)}").add_to(m)
    extents = [shape(f["geometry"]).bounds for f in hazards["features"] + roads["features"] + buildings["features"]]
    extents.extend((p["lon"], p["lat"], p["lon"], p["lat"]) for p in coords)
    if extents:
        m.fit_bounds([[min(b[1] for b in extents), min(b[0] for b in extents)],
                      [max(b[3] for b in extents), max(b[2] for b in extents)]], padding=(60, 60), max_zoom=17)
    folium.LayerControl().add_to(m)
    m.save(out / "map.html")
    synthetic = manifest["data_kind"] == "synthetic_fixture"
    label = "SYNTHETIC ENGINEERING TEST — not a real flood result" if synthetic else "UNVALIDATED ANALYSIS — candidates and closure scenarios"
    # This label also appears in the standalone map, so sharing it loses no context.
    standalone = (out / "map.html").read_text()
    standalone = standalone.replace("<body>", f'<body><div style="position:fixed;z-index:9999;top:10px;left:55px;background:#fff5dc;padding:10px;max-width:75%;font:14px sans-serif">{label}</div>')
    (out / "map.html").write_text(standalone)
    rows = "".join(f"<tr><td>{escape(a['name'])}</td><td>{escape(a['status'].replace('_',' '))}</td><td>{len(a['before'])}</td><td>{len(a['after'])}</td></tr>" for a in access)
    limitations = "".join(f"<li>{escape(item)}</li>" for item in manifest["limitations"])
    conf = manifest["config"]
    (out / "report.html").write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Akash Flood | Evidence report</title>
<style>body{{font:16px/1.5 system-ui;margin:0;color:#192b3b;background:#f2f5f8}}main{{max-width:1100px;margin:auto;padding:28px}}h1{{margin:4px 0;font-size:32px}}.notice{{background:#fff0c8;border-left:4px solid #bb711a;padding:12px}}.cards{{display:flex;gap:18px;margin:20px 0;flex-wrap:wrap}}.card{{background:white;padding:18px;flex:1;border:1px solid #d4dfe6}}strong{{display:block;font-size:27px}}small{{color:#415767}}iframe{{width:100%;height:450px;border:1px solid #aabac7;background:#e9edf0}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{text-align:left;padding:10px;border-bottom:1px solid #d5dfe5}}a{{color:#145784}}@media print{{iframe{{height:250px}}main{{padding:10px}}.details{{display:none}}}}</style>
<main><small>AKASH FLOOD / COMPUTED EVIDENCE</small><h1>Flood exposure and road access</h1>
<p class="notice">{label}</p><p>Event: {escape(conf['event_date'])} · Run: {manifest['run_id']}</p>
<div class="cards"><div class="card">New-water candidates<strong>{metrics['candidate_area_km2']:.4f} km²</strong></div><div class="card">Assessed area<strong>{metrics['assessed_area_km2']:.4f} km²</strong></div><div class="card">Disconnected under scenario<strong>{metrics.get('newly_disconnected_under_scenario','Not assessed')}</strong></div></div>
<p>Building footprints intersecting candidates: <b>{metrics.get('building_features_exposed', 'Not assessed')}</b>. This is an exposure count, not a damage estimate. <a href="exposed_buildings.geojson">Footprint evidence</a> · <a href="building_geometry_audit.json">Geometry exclusions</a>.</p>
<iframe title="Candidate and exposure evidence map" src="map.html"></iframe>
<p>Scenario: non-bridge/non-tunnel road segments intersecting a candidate are removed from the graph. This does not establish physical road closure.</p>
<table><thead><tr><th>Settlement</th><th>Modeled access</th><th>Destinations before</th><th>Destinations after</th></tr></thead><tbody>{rows}</tbody></table>
<p>Observation dates: {escape(conf['before']['datetime'])} / {escape(conf['after']['datetime'])}. OSM snapshot: {escape(conf.get('osm_snapshot','not supplied'))}.</p>
<div class="details"><h2>Evidence files</h2><p><a href="run_manifest.json">Input provenance</a> · <a href="metrics.json">Computed metrics</a> · <a href="hazards.geojson">Candidate polygons</a> · <a href="exposed_roads.geojson">Road intersections</a> · <a href="settlement_access.json">Routes and destination comparisons</a></p><h2>Limits of this run</h2><ul>{limitations}</ul></div>
<p><small>Educational prototype. Contains modified Copernicus Sentinel data 2026 when real Sentinel inputs are used. © OpenStreetMap contributors when OSM data are used. Synthetic fixtures contain no real observations.</small></p></main></html>''')
