// Original viewer code: MIT. Saved geographic data retains its upstream terms.
export const statusOrder = ['robust_connected', 'ambiguous', 'unmapped', 'baseline_disconnected'];
export const statusLabels = {
  robust_connected: 'Conditional graph connection', ambiguous: 'Undetermined',
  unmapped: 'No mapped association', baseline_disconnected: 'Baseline disconnected'
};
export function scopedRows(c, scope) {
  if (!['aoi', 'context', 'all'].includes(scope)) throw new Error('Unsupported source-point scope');
  return c.settlements.filter(s => scope === 'all' || (scope === 'aoi' ? s.inside_water_aoi : s.inside_extraction_context));
}
export function matchingRows(c, scope, status, query = '') {
  const q = query.trim().toLocaleLowerCase();
  return scopedRows(c, scope).filter(s => (status === 'all' || s.status === status) &&
    (!q || `${s.name} ${s.id}`.toLocaleLowerCase().includes(q)));
}
export function statusCounts(rows) {
  return Object.fromEntries(statusOrder.map(k => [k, rows.filter(s => s.status === k).length]));
}
export function selectedRow(rows, id) { return rows.find(s => s.id === id) ?? null; }
export function savedRoute(row, witness = 'expanded') { return row?.[`${witness}_route`] ?? null; }
export function routeSegments(c, row, witness = 'expanded') {
  const route = savedRoute(row, witness);
  if (!route) return [];
  return route.edge_ids.map(id => {
    if (!c.segments[id]) throw new Error(`Missing supplied segment ${id}`);
    return {id, ...c.segments[id]};
  });
}
export function segmentSupport(s) {
  const positive = [s.assessed_length_m, s.unknown_within_aoi_length_m, s.outside_aoi_length_m].map(v => v > 0);
  if (positive.filter(Boolean).length > 1) return 'mixed';
  return positive[2] ? 'outside' : positive[1] ? 'unknown' : 'assessed';
}
export function extent(points) {
  if (!points.length) throw new Error('No supplied coordinates for this extent');
  for (const p of points) if (!Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite)) throw new Error('Invalid supplied geographic coordinate');
  return [Math.min(...points.map(p => p[0])), Math.min(...points.map(p => p[1])),
    Math.max(...points.map(p => p[0])), Math.max(...points.map(p => p[1]))];
}
export function paddedBounds(points) {
  const b = extent(points), dx = Math.max(b[2] - b[0], .002), dy = Math.max(b[3] - b[1], .002);
  return [b[0] - dx * .07, b[1] - dy * .07, b[2] + dx * .07, b[3] + dy * .07];
}
export function boxPoints(b) { return [[b[0], b[1]], [b[2], b[3]]]; }
export function displayProjection(bounds, width = 1000, height = 640) {
  const cosine = Math.cos((bounds[1] + bounds[3]) * Math.PI / 360);
  const sx = (width - 120) / ((bounds[2] - bounds[0]) * cosine);
  const sy = (height - 100) / (bounds[3] - bounds[1]);
  const scale = Math.min(sx, sy), left = 60 + ((width - 120) - (bounds[2] - bounds[0]) * cosine * scale) / 2;
  const bottom = height - 55 - ((height - 100) - (bounds[3] - bounds[1]) * scale) / 2;
  return {x: lon => left + (lon - bounds[0]) * cosine * scale,
    y: lat => bottom - (lat - bounds[1]) * scale, scale, cosine};
}
export function routePath(segments, project, category) {
  // Each supplied segment begins with M. No inferred inter-segment or point-to-road connector.
  return segments.filter(s => segmentSupport(s) === category).map(s => s.coordinates.map((p, i) =>
    `${i ? 'L' : 'M'}${project.x(p[0]).toFixed(5)},${project.y(p[1]).toFixed(5)}`).join(' ')).join(' ');
}
