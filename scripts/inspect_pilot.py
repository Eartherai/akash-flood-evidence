"""Render real historical OSM inputs and pre-event connectivity; no flood claims."""
from collections import Counter
from html import escape
import json
from pathlib import Path
import shutil

import folium
from shapely.geometry import mapping

from akash_flood.access import analyze_access, build_road_graph
from akash_flood.buildings import building_footprints
from akash_flood.osm import named_points
from akash_flood.terrain import preview_layer

data = Path('data/rasuwa-pilot')
out = Path('outputs/rasuwa-input-audit')
out.mkdir(parents=True, exist_ok=True)
catalog = json.loads((data/'catalog.json').read_text())
osm = json.loads((data/'osm.json').read_text())
provenance = json.loads((data/'osm.provenance.json').read_text())
buildings, building_audit = building_footprints(osm)
g = build_road_graph(osm)
settlements, destinations = named_points(osm)
access = analyze_access(g, settlements, destinations, set(), max_snap_m=500)
summary = dict(osm_elements=len(osm['elements']), graph_nodes=len(g),
               directed_edges=g.number_of_edges(), settlements=len(settlements),
               node_tagged_destinations=len(destinations),
               baseline_status=dict(Counter(row['status'] for row in access)),
               missing_graph_nodes=len(g.graph['missing_node_ids']),
               snapshot=provenance['snapshot'], analysis='baseline_only_no_hazard_or_closures')
summary['mapped_building_features'] = building_audit['mapped_building_features']
summary['excluded_building_features'] = len(building_audit['excluded'])
(out/'building_inventory.geojson').write_text(json.dumps(buildings))
(out/'building_geometry_audit.json').write_text(json.dumps(building_audit, indent=2))
(out/'baseline_summary.json').write_text(json.dumps(summary, indent=2))
(out/'baseline_access.json').write_text(json.dumps(access, indent=2, ensure_ascii=False))
(out/'osm.provenance.json').write_text(json.dumps(provenance, indent=2))
pair=catalog['selected_pair']
(out/'scene_pair.json').write_text(json.dumps(pair, indent=2))
m=folium.Map(location=[28.2,85.36],zoom_start=11,tiles=None,control_scale=True)
terrain_summary = ''
terrain_path = data/'terrain'/'slope-30m.tif'
if terrain_path.exists():
 terrain = json.loads((data/'terrain'/'terrain.provenance.json').read_text())
 west,south,east,north=catalog['bbox']
 folium.raster_layers.ImageOverlay(preview_layer(terrain_path, catalog['bbox']),
     bounds=[[south,west],[north,east]], name='Copernicus terrain slope (not hazard)',
     mercator_project=True, opacity=.85, zindex=1).add_to(m)
 shutil.copyfile(data/'terrain'/'terrain.provenance.json', out/'terrain.provenance.json')
 terrain_summary = f"<p>Copernicus GLO-30 slope: pale means flatter, dark means steeper (0–60°). Median slope is {terrain['slope_percentiles_deg']['p50']:.1f}°; only {100*terrain['grid_fraction_slope_at_most_5_deg']:.2f}% of valid cells meet the prototype's 5° cutoff. Those statistics cover the projected terrain grid and its halo. Excluded terrain is unassessed, not evidence of no flood. <a href='terrain.provenance.json'>Terrain provenance</a>.</p>"
seen=set();features=[]
for u,v,k,e in g.edges(keys=True,data=True):
 physical=(e['way_id'],min(u,v),max(u,v))
 if physical in seen:continue
 seen.add(physical)
 features.append(dict(type='Feature',geometry=mapping(e['geometry']),properties={'way_id':e['way_id'],'bridge':e['bridge']}))
folium.GeoJson(dict(type='FeatureCollection',features=features),name='Historical motor roads',
               style_function=lambda f:{'color':'#547487','weight':2},
               tooltip=folium.GeoJsonTooltip(fields=['way_id','bridge'])).add_to(m)
folium.GeoJson(buildings, name='Historical building footprints', show=False,
               style_function=lambda f:{'color':'#78617e','weight':1,'fillOpacity':.4},
               tooltip=folium.GeoJsonTooltip(fields=['osm_id','building'])).add_to(m)
statuses={row['id']:row['status'] for row in access}
colors={'connected':'#167061','unmapped':'#c58322','baseline_disconnected':'#8f5675'}
for p in settlements+destinations:
 status=statuses.get(p['id'],'destination')
 folium.CircleMarker([p['lat'],p['lon']],radius=5,fill=True,fill_opacity=1,
                     color=colors.get(status,'#243449'),
                     tooltip=f"{escape(p['name'])}: {status.replace('_',' ')}").add_to(m)
west,south,east,north=catalog['bbox']
folium.Rectangle([[south,west],[north,east]],color='#125a92',fill=False,weight=2,
                 tooltip='Satellite pilot area; road network extends beyond this box').add_to(m)
m.fit_bounds([[south,west],[north,east]])
folium.LayerControl().add_to(m)
m.save(out/'map.html')
rows=''.join(f"<tr><td>{escape(a['name'])}</td><td>{a['status'].replace('_',' ')}</td><td>{len(a['before'])}</td></tr>" for a in access)
(out/'index.html').write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Akash Flood | Real input audit</title>
<style>body{{font:16px/1.5 system-ui;background:#f2f5f8;color:#203344;margin:0}}main{{max-width:1100px;margin:auto;padding:28px}}h1{{font-size:32px;margin:0}}.notice{{padding:12px;background:#e1edf5;border-left:4px solid #29698f}}.cards{{display:flex;gap:15px;flex-wrap:wrap;margin:20px 0}}.card{{background:white;padding:16px;border:1px solid #ccd7df;flex:1}}strong{{display:block;font-size:27px}}iframe{{width:100%;height:540px;border:1px solid #a2b7c7}}table{{width:100%;border-collapse:collapse}}td,th{{padding:8px;text-align:left;border-bottom:1px solid #ccd7df}}code{{overflow-wrap:anywhere}}a{{color:#165d8e}}</style>
<main><p>AKASH FLOOD / REAL INPUT AUDIT</p><h1>Rasuwa: data and road-network check</h1>
<p class="notice">Real historical OpenStreetMap inputs. Baseline connectivity only. Flood mapping and post-event road closures have not been evaluated.</p>
<div class="cards"><div class="card">Historical elements<strong>{summary['osm_elements']:,}</strong></div><div class="card">Motor-road graph nodes<strong>{summary['graph_nodes']:,}</strong></div><div class="card">Settlement nodes<strong>{len(settlements)}</strong></div></div>
<p>Historical building footprints: <b>{summary['mapped_building_features']:,}</b>; geometry exclusions: <b>{summary['excluded_building_features']}</b>. Enable the building layer in the map to inspect it. These counts cover the expanded OSM extraction area and are not flood-exposure counts.</p>
<iframe title="Historical Rasuwa road map" src="map.html"></iframe>
{terrain_summary}
<p>Green: modeled baseline access. Amber: outside snap limit or unmapped destination. Purple: no baseline route. Dark: town or hospital destination. These are not flood effects.</p>
<p>Snapshot requested: <code>{provenance['snapshot']}</code>. No present-day basemap is used as evidence. Only node-tagged settlements and facilities are included in this first audit; polygon facilities, road restrictions and map boundaries still need review.</p>
<h2>Matched radar acquisitions</h2><p>Before: <code>{escape(pair['before']['id'])}</code></p><p>After: <code>{escape(pair['after']['id'])}</code></p><p>Relative orbit 85, ascending. Catalogue metadata verified; pixel download, calibration and terrain correction remain pending.</p>
<p><a href="scene_pair.json">Scene metadata</a> · <a href="osm.provenance.json">OSM query provenance</a> · <a href="baseline_summary.json">Computed summary</a> · <a href="baseline_access.json">Baseline routes</a></p>
<p><a href="building_inventory.geojson">Building footprints</a> · <a href="building_geometry_audit.json">Building geometry audit</a>. The original historical query selected building ways; unqueried building relations may still be missing.</p>
<table><tr><th>Settlement</th><th>Baseline status</th><th>Reachable destinations</th></tr>{rows}</table>
<p>© OpenStreetMap contributors, ODbL. Sentinel catalogue: Copernicus Data Space Ecosystem. Educational prototype.</p></main></html>''')
print(json.dumps(summary,indent=2))
