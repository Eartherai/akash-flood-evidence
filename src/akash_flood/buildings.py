"""Historical OSM building footprints and candidate overlap, not damage detection."""
from pyproj import CRS, Transformer
from shapely.geometry import MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform, unary_union


def _rings(chains):
    """Join relation members by OSM node ID; reject gaps and ambiguous branches."""
    pending = [list(chain) for chain in chains]
    rings = []
    while pending:
        ring = pending.pop(0)
        if len(ring) < 2:
            raise ValueError("member has fewer than two nodes")
        while ring[0] != ring[-1]:
            matches = [(i, chain) for i, chain in enumerate(pending)
                       if chain[0] == ring[-1] or chain[-1] == ring[-1]]
            if len(matches) != 1:
                raise ValueError("unclosed or ambiguous relation ring")
            i, chain = matches[0]
            pending.pop(i)
            if chain[-1] == ring[-1]:
                chain = chain[::-1]
            ring.extend(chain[1:])
        if len(set(ring)) < 3:
            raise ValueError("ring has fewer than three distinct nodes")
        rings.append(ring)
    return rings


def building_footprints(osm):
    """Return one feature per mapped building way/relation, with exclusion audit.

    Building parts and point-only buildings do not establish building footprints.
    Relation member ways are suppressed even if the relation is incomplete, so
    partial geometry is never promoted to a complete building.
    """
    elements = {(e["type"], e["id"]): e for e in osm.get("elements", [])}
    candidates = [e for e in elements.values() if e.get("tags", {}).get("building")
                  not in (None, "no")]
    member_ids = {m["ref"] for e in candidates if e["type"] == "relation"
                  for m in e.get("members", []) if m["type"] == "way"}
    features, excluded, suppressed = [], [], []

    def coordinates(node_ids):
        return [(elements["node", n]["lon"], elements["node", n]["lat"]) for n in node_ids]

    for element in candidates:
        kind, identity = element["type"], element["id"]
        osm_id = f"{kind}/{identity}"
        if kind == "way" and identity in member_ids:
            suppressed.append(osm_id)
            continue
        try:
            if kind == "way":
                rings = _rings([element["nodes"]])
                geometry = Polygon(coordinates(rings[0]))
            elif kind == "relation" and element["tags"].get("type") == "multipolygon":
                chains = {"outer": [], "inner": []}
                for member in element.get("members", []):
                    role = member.get("role") or "outer"
                    if member["type"] != "way" or role not in chains:
                        raise ValueError("unsupported multipolygon member")
                    chains[role].append(elements["way", member["ref"]]["nodes"])
                outers = [Polygon(coordinates(r)) for r in _rings(chains["outer"])]
                inners = [Polygon(coordinates(r)) for r in _rings(chains["inner"])]
                if not outers or any(not p.is_valid for p in outers + inners):
                    raise ValueError("invalid relation rings")
                holes = [[] for _ in outers]
                for inner in inners:
                    owners = [i for i, outer in enumerate(outers) if outer.contains(inner)]
                    if len(owners) != 1:
                        raise ValueError("inner ring has no unique outer ring")
                    holes[owners[0]].append(inner.exterior.coords)
                polygons = [Polygon(outer.exterior.coords, holes[i]) for i, outer in enumerate(outers)]
                geometry = polygons[0] if len(polygons) == 1 else MultiPolygon(polygons)
            else:
                raise ValueError("no supported polygon footprint")
            if geometry.is_empty or not geometry.is_valid or geometry.area <= 0:
                raise ValueError("invalid footprint; no automatic geometry repair")
        except (KeyError, ValueError) as error:
            excluded.append({"osm_id": osm_id, "reason": str(error)})
            continue
        tags = element.get("tags", {})
        features.append({"type": "Feature", "id": osm_id, "geometry": mapping(geometry),
                         "properties": {"osm_id": osm_id, "building": tags["building"],
                                        "name": tags.get("name", ""), "confirmed_damage": False}})
    audit = {"mapped_building_features": len(features), "excluded": excluded,
             "suppressed_relation_members": suppressed,
             "count_unit": "OSM building polygon features, not verified physical structures"}
    return {"type": "FeatureCollection", "features": features}, audit


def intersect_buildings(footprints, hazards, metric_crs):
    crs = CRS.from_user_input(metric_crs)
    if not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info):
        raise ValueError("Building areas require a projected CRS with metre units")
    project = Transformer.from_crs(4326, crs, always_xy=True).transform
    hazard = unary_union([shape(f["geometry"]) for f in hazards["features"]])
    projected_hazard = transform(project, hazard)
    features = []
    for feature in footprints["features"]:
        geometry = shape(feature["geometry"])
        if hazard.is_empty or not geometry.intersects(hazard):
            continue
        projected = transform(project, geometry)
        overlap = projected.intersection(projected_hazard).area
        if overlap <= 1e-6:  # Boundary-only contact and projection round-off are not exposure.
            continue
        properties = dict(feature["properties"], footprint_area_m2=round(projected.area, 2),
                          overlap_m2=round(overlap, 2), overlap_fraction=overlap / projected.area,
                          exposure_basis="intersection with unvalidated new-water candidate",
                          confirmed_damage=False)
        features.append(dict(feature, properties=properties))
    return {"type": "FeatureCollection", "features": features}
