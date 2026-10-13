const $ = (id) => document.getElementById(id);
const number = (n) => new Intl.NumberFormat('en-US').format(n);
const percent = (value) => `${(100 * value).toFixed(2)}%`;
const sourceHash = 'd5a4b09abfbc724a464c701ac0962af6d056e1c158f1ed111166b9211970547c';
const svgNS = 'http://www.w3.org/2000/svg';
// Follow the same public anchor in browsers whose embedded viewport does not
// reliably scroll on a native fragment click. Preserve normal modified clicks.
function followAnchor(hash, updateHistory = false) {
  if (!hash || hash === '#') return;
  let id;
  try { id = decodeURIComponent(hash.slice(1)); } catch { return; }
  const target = document.getElementById(id);
  if (!target) return;
  if (updateHistory && location.hash !== hash) history.pushState(null, '', hash);
  if (!target.hasAttribute('tabindex')) target.setAttribute('tabindex', '-1');
  target.scrollIntoView({block:'start', behavior:'auto'});
  target.focus({preventScroll:true});
}
document.addEventListener('click', (event) => {
  if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
  const anchor = event.target.closest?.('a[href^="#"]');
  if (!anchor) return;
  const hash = anchor.getAttribute('href');
  let target;
  try { target = document.getElementById(decodeURIComponent(hash.slice(1))); } catch { return; }
  if (!target) return;
  event.preventDefault();
  followAnchor(hash, true);
});
window.addEventListener('hashchange', () => followAnchor(location.hash));

let data, derivation, routes, routeLoading, view = 'radar', selectedId = 'node/269569477', mapBounds;
const routeColors = {assessed:'#117574',unknown:'#996116',outside:'#788896',mixed:'#704f85'};
const states = {ambiguous:'Undetermined',unmapped:'No mapped association',baseline_disconnected:'Baseline disconnected',robust_connected:'Conditional graph connection'};
const safe = (text) => String(text ?? '').replace(/[&<>"']/g, (character) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[character]));
function showError(message) { $('error').textContent = message; $('error').hidden = false; }
async function readBoundJSON(path, expected) {
  const response = await fetch(path, {credentials:'omit'});
  if (!response.ok) throw new Error(`${path} could not be loaded (${response.status}).`);
  const bytes = await response.arrayBuffer();
  if (expected && globalThis.crypto?.subtle) {
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const hash = [...new Uint8Array(digest)].map((x) => x.toString(16).padStart(2,'0')).join('');
    if (hash !== expected) throw new Error(`${path} does not match its saved display binding.`);
  }
  return JSON.parse(new TextDecoder().decode(bytes));
}
const sourceLink = (pointer, text = 'Open the source facts') => `<a href="case/case-facts.json">${safe(text)}</a><span class="id">${safe(pointer)}</span>`;
const metric = (label, value) => `<div><span>${safe(label)}</span><b>${safe(value)}</b></div>`;
function renderRaster(mode) {
  const optical = mode === 'optical';
  const record = data[mode], counts = record.counts;
  const assessed = counts[optical ? 'joint_assessed_pixels' : 'assessed_pixels'];
  const fraction = assessed / counts.aoi_pixels;
  $('map-title').textContent = optical ? 'Optical surface-change candidates' : 'Radar water-state candidates';
  $('map-tag').textContent = optical ? 'Native 20 m cells' : 'Native 10 m cells';
  $('raster').src = optical ? 'case/optical-overview.png' : 'case/map.png';
  $('raster').alt = optical ? 'Recorded optical classification: unknown cells in gray, assessed noncandidate cells in green, provisional surface-change candidates in brown' : 'Recorded radar water-state classification: unchanged water in pale teal, new-water candidates in dark teal, lost-water candidates in orange, unknown cells in gray';
  $('raster-legend').innerHTML = optical ? `<span><i class="key" style="background:#c1c8cf"></i>Unknown</span><span><i class="key" style="background:#e8f0e6"></i>Assessed, no candidate</span><span><i class="key" style="background:#934e23"></i>Surface-change candidate</span>` : `<span><i class="key" style="background:#c1c8cf"></i>Unknown</span><span><i class="key" style="background:#edf2ee"></i>Assessed nonwater</span><span><i class="key" style="background:#99bbb8"></i>Unchanged water</span><span><i class="key" style="background:#087e82"></i>New-water candidate</span><span><i class="key" style="background:#d87c43"></i>Lost-water candidate</span>`;
  $('map-scope').textContent = 'Original saved classification image, displayed in full. Radar and optical have separate observation masks; there is no cross-grid pixel fusion.';
  $('evidence-content').innerHTML = `<h3>${optical ? 'Most optical cells stay unknown.' : 'Observation support comes first.'}</h3><p class="summary">${optical ? 'Cloud and shadow masking leaves a small jointly assessed area in this fixed scene pair.' : 'Only cells supported by both radar dates enter the recorded water-state comparison.'}</p><div class="metric-large"><b>${percent(fraction)}</b><span>assessed</span></div><div class="support-track" role="img" aria-label="${percent(fraction)} assessed; ${percent(1-fraction)} unknown"><i style="width:${100*fraction}%"></i></div><div class="metric-pair"><span>${number(assessed)} assessed</span><b>${number(counts.unknown_pixels)} unknown</b></div><div class="count-list">${metric('Native AOI cells',number(counts.aoi_pixels))}${optical ? metric('Surface-change candidate cells',number(counts.candidate_pixels)) + metric('Candidate footprint',`${number(record.candidate_area_m2)} m²`) : metric('New-water candidate cells',number(counts.new_water_candidate_pixels)) + metric('Lost-water candidate cells',number(counts.lost_water_candidate_pixels)) + metric('Unchanged-water cells',number(counts.unchanged_water_pixels))}</div><p class="source-date"><b>Recorded acquisition times (UTC)</b>Before: ${safe(record.source_dates.before)}<br>After: ${safe(record.source_dates.after)}</p><p class="caption">${optical ? 'The 424,000 m² footprint is a spectral candidate area, not measured debris or damage.' : 'The new-water count is provisional transfer output, not a locally validated flood area.'}</p>`;
  $('warning-text').textContent = optical ? 'Vegetation loss or exposed surface can reflect agriculture, clearing, moisture, season or sensor effects. No local debris accuracy is established.' : 'The binary-water model does not separate permanent water from floodwater. This case does not establish local hazard accuracy.';
}
function currentPoints() {
  const scope = $('scope').value, query = $('point-search').value.trim().toLocaleLowerCase();
  return data.settlements.filter((s) => (scope === 'all' || (scope === 'aoi' ? s.inside_water_aoi : s.inside_extraction_context)) && (!query || `${s.name} ${s.id}`.toLocaleLowerCase().includes(query)));
}
function updatePointSelector() {
  const points = currentPoints();
  if (!points.some((point) => point.id === selectedId)) selectedId = points[0]?.id ?? null;
  $('settlement').replaceChildren(...points.map((point) => {
    const option = document.createElement('option'); option.value = point.id; option.textContent = `${point.name} / ${states[point.status]}`; return option;
  }));
  if (!points.length) { const option = document.createElement('option'); option.textContent = 'No matching recorded points'; option.value=''; $('settlement').append(option); }
  $('settlement').value = selectedId ?? '';
  $('settlement').disabled = !points.length;
  const scope = $('scope').value;
  const counts = Object.fromEntries(['ambiguous','unmapped','baseline_disconnected','robust_connected'].map((state) => [state, points.filter((s) => s.status===state).length]));
  $('point-count').textContent = `${points.length} matching source points. ${counts.ambiguous} undetermined, ${counts.unmapped} without mapped association, ${counts.baseline_disconnected} baseline disconnected, ${counts.robust_connected} conditional graph connections. ${scope==='all' ? 'All source points are selected.' : scope==='aoi' ? 'Point denominator: inside water AOI.' : 'Point denominator: inside extraction context.'} Circles: undetermined. Squares: unmapped. Triangles: baseline disconnected.`;
}
function defaultBounds() {
  const points = currentPoints().map((p) => p.point_coordinates).filter((p) => Array.isArray(p) && p.every(Number.isFinite));
  if (!points.length) return data.case.bbox.slice();
  return paddedBounds(points, data.case.bbox);
}
function paddedBounds(points, include = null) {
  const coordinates = include ? [...points,[include[0],include[1]],[include[2],include[3]]] : points;
  const xs = coordinates.map((p) => p[0]), ys = coordinates.map((p) => p[1]);
  let west=Math.min(...xs), east=Math.max(...xs), south=Math.min(...ys), north=Math.max(...ys);
  const dx=Math.max(east-west,.015),dy=Math.max(north-south,.015);
  return [west-dx*.08,south-dy*.08,east+dx*.08,north+dy*.08];
}
function element(name, attributes={}) {
  const node=document.createElementNS(svgNS,name);
  for(const [key,value] of Object.entries(attributes)) node.setAttribute(key,String(value));
  return node;
}
function renderRoadMap() {
  if (!data) return;
  const svg=$('road-map');svg.replaceChildren();
  const bounds=mapBounds??defaultBounds(), latitude=(bounds[1]+bounds[3])/2, cosine=Math.cos(latitude*Math.PI/180);
  const dx=(bounds[2]-bounds[0])*cosine,dy=bounds[3]-bounds[1],scale=Math.min(704/dx,450/dy),offsetX=(800-dx*scale)/2,offsetY=(540-dy*scale)/2;
  const project=(point)=>[offsetX+(point[0]-bounds[0])*cosine*scale,540-offsetY-(point[1]-bounds[1])*scale];
  const clip=element('clipPath',{id:'map-clip'});clip.append(element('rect',{x:26,y:20,width:748,height:486}));const defs=element('defs');defs.append(clip);svg.append(defs);
  const content=element('g',{'clip-path':'url(#map-clip)'});
  for(let index=0;index<=4;index++) {
    const x=26+748*index/4,y=20+486*index/4;
    svg.append(element('line',{x1:x,x2:x,y1:20,y2:506,stroke:'#d2e0e6','stroke-width':1}));
    svg.append(element('line',{x1:26,x2:774,y1:y,y2:y,stroke:'#d2e0e6','stroke-width':1}));
  }
  function box(coordinates,dashed) { const first=project([coordinates[0],coordinates[3]]),last=project([coordinates[2],coordinates[1]]);content.append(element('rect',{x:first[0],y:first[1],width:last[0]-first[0],height:last[1]-first[1],fill:'none',stroke:dashed?'#6c8696':'#173e4b','stroke-width':dashed?1.5:2,'stroke-dasharray':dashed?'7 5':'none'})); }
  box(data.case.context_bbox,true);box(data.case.bbox,false);
  const selected=data.settlements.find((s)=>s.id===selectedId);
  if(routes && selectedId && routes[selectedId]) {
    for(const category of Object.keys(routeColors)) {
      const segments=routes[selectedId].filter((s)=>s.category===category);
      const path=segments.map((s)=>s.coordinates.map((p,i)=>`${i?'L':'M'}${project(p).join(',')}`).join(' ')).join(' ');
      if(path) content.append(element('path',{d:path,fill:'none',stroke:routeColors[category],'stroke-width':3,'stroke-dasharray':category==='unknown'||category==='outside'?'5 3':'none','stroke-linecap':'round'}));
    }
  }
  for(const point of currentPoints()) {
    if(!Array.isArray(point.point_coordinates))continue;
    const [x,y]=project(point.point_coordinates),isSelected=point.id===selectedId;
    const group=element('g');const title=element('title');title.textContent=`${point.name}: ${states[point.status]} (${point.id})`;group.append(title);
    const props={fill:isSelected?'#173e4b':'#fff',stroke:'#173e4b','stroke-width':isSelected?3:1.5};
    let symbol;
    if(point.status==='unmapped') symbol=element('rect',{...props,x:x-5,y:y-5,width:10,height:10});
    else if(point.status==='baseline_disconnected')symbol=element('path',{...props,d:`M${x},${y-7}L${x+7},${y+6}L${x-7},${y+6}Z`});
    else symbol=element('circle',{...props,cx:x,cy:y,r:isSelected?7:5});
    group.append(symbol);content.append(group);
    if(isSelected) {const label=element('text',{x:x+13,y:y-12,fill:'#173e4b','font-size':14,'font-weight':650,'paint-order':'stroke',stroke:'#edf4f6','stroke-width':4});label.textContent=point.name;content.append(label);}
  }
  svg.append(content);
  const latitudeLabel=element('text',{x:26,y:528,fill:'#506571','font-size':12});latitudeLabel.textContent=`Longitude ${bounds[0].toFixed(3)}° to ${bounds[2].toFixed(3)}° E`;svg.append(latitudeLabel);
  const longitudeLabel=element('text',{x:775,y:528,fill:'#506571','font-size':12,'text-anchor':'end'});longitudeLabel.textContent=`Latitude ${bounds[1].toFixed(3)}° to ${bounds[3].toFixed(3)}° N`;svg.append(longitudeLabel);
}
function renderSelectedPoint() {
  const point = data.settlements.find((s)=>s.id===selectedId);
  $('fit-route').disabled=!point?.support;
  if(!point) { $('evidence-content').innerHTML='<h3>No matching recorded point</h3><p class="summary">Clear the search or choose another point scope to inspect the saved records.</p>';renderRoadMap();return; }
  const support=point.support, fractions=support?.fractions_of_projected_centerline;
  const share=(key)=>fractions?.[key] == null ? 'No nonempty witness' : percent(fractions[key]);
  const summary=point.status==='ambiguous'?'The saved graph has a route to a fixed destination when uncertain road groups are open, and no route to that destination when they are closed.':point.status==='unmapped'?'The historical graph has no eligible mapped association for this point or its destination. It does not establish whether a physical road exists.':'The baseline historical graph already lacks a route. This result is not new flood-caused isolation.';
  $('evidence-content').innerHTML=`<h3 class="point-title">${safe(point.name)}</h3><p class="id">${safe(point.id)}</p><span class="point-state">${safe(states[point.status])}</span><p class="summary">${summary}</p>${support?`<div class="count-list">${metric('Assessed centerline share',share('assessed'))}${metric('Unknown inside AOI',share('unknown_within_aoi'))}${metric('Outside AOI',share('outside_aoi'))}${metric('Recorded graph path',`${(support.graph_distance_m/1000).toFixed(2)} km`)}${metric('Unique physical segments',number(support.unique_physical_segment_count))}${metric('Closed-scenario cut edges',number(point.cut_boundary_edge_count??0))}</div><p class="caption">Route support measures projected centerline length. Distance uses the saved graph weights. Assessed classifications still do not establish physical passage.</p>`:'<p class="caption">No saved route to draw. Unknown evidence is not filled with an illustrative connection.</p>'}<details><summary>Trace this point to its source</summary><p>${sourceLink(point.source_pointer)}</p>${point.nearest_baseline_destination?`<p>Fixed target: ${safe(point.nearest_baseline_destination.destination_id)}. The map draws saved road-node geometry, without invented entrance connectors.</p>`:''}<p>Point coordinates: ${safe(point.point_coordinates?.join(', ')??'Unavailable')}. Witness search uses hop count; it is not a fastest or safest route.</p></details>`;
  $('warning-text').textContent='There are no confirmed closures in this record. Open/closed scenarios express uncertainty in supplied road groups; they are not observations of road failure.';
  renderRoadMap();
}
async function loadRoutes() {
  if(routes) return;
  if(routeLoading) return routeLoading;
  $('receipt-status').textContent='Loading saved route coordinates (4.3 MB)';
  routeLoading=readBoundJSON('routes.json',derivation.derived_files['routes.json'].sha256).then((result)=>{routes=result;$('receipt-status').textContent=globalThis.crypto?.subtle?'Saved facts and routes verified':'Saved source bindings available';if(view==='roads')renderRoadMap();}).catch((error)=>{routeLoading=null;showError(`${error.message} The point facts and original case record remain available.`);});
  return routeLoading;
}
function selectView(next) {
  view=next;
  for(const button of document.querySelectorAll('[data-view]'))button.setAttribute('aria-pressed',String(button.dataset.view===view));
  $('raster-view').hidden=view==='roads';$('road-view').hidden=view!=='roads';
  if(view==='roads') {
    $('map-title').textContent='A saved witness, with observation gaps';$('map-tag').textContent='Recorded geographic coordinates';
    updatePointSelector();mapBounds=defaultBounds();renderSelectedPoint();loadRoutes();
  } else renderRaster(view);
}
function renderAnswer() {
  if(!data)return;
  const r=data.radar.counts,o=data.optical.counts,c=data.conditional_scenario_counts.inside_water_aoi;
  const answers={
    unknown:{title:'The unknown areas differ between sensors.',text:`Radar has ${number(r.unknown_pixels)} unknown native cells (${percent(r.unknown_pixels/r.aoi_pixels)}). Optical has ${number(o.unknown_pixels)} unknown native cells (${percent(o.unknown_pixels/o.aoi_pixels)}). These have different cell sizes and observation masks; their counts are not interchangeable.`,pointer:'/radar/counts/unknown_pixels; /optical/counts/unknown_pixels'},
    closure:{title:'No confirmed road closures are supplied.',text:`The source records ${data.confirmed_closures} confirmed closures. Graph scenarios treat candidate-intersecting, unknown-within-AOI and outside-AOI road groups as uncertain. A disconnected scenario does not prove flood-caused closure or physical isolation.`,pointer:'/confirmed_closures; /observation_uncertainty_applied'},
    dates:{title:'The road snapshot precedes the event reference.',text:`Historical OSM is dated ${data.historical_snapshot}. The event reference is ${data.case.event_date}. A prior snapshot prevents later map edits entering this case; it does not establish whether a mapped road was usable.`,pointer:'/historical_snapshot; /case/event_date'},
    settlements:{title:'Access stays undetermined for this AOI.',text:`Among the ${c.ambiguous+c.unmapped+c.baseline_disconnected} historical source points inside the water AOI, ${c.ambiguous} are undetermined, ${c.unmapped} have no eligible mapped association and ${c.baseline_disconnected} were already disconnected in the baseline graph. No robust conditional connection or confirmed physical isolation is established.`,pointer:'/conditional_scenario_counts/inside_water_aoi'},
    fusion:{title:'The native grids were not fused.',text:'The saved record explicitly sets native_grids_resampled_or_fused to false. Radar remains on its native 10 m grid and optical on its native 20 m grid. Their separate masks and candidate interpretations are preserved.',pointer:'/native_grids_resampled_or_fused'},
    candidate:{title:'Candidates are observations to investigate.',text:`The radar comparison contains ${number(r.new_water_candidate_pixels)} new-water candidate cells. Optical contains ${number(o.candidate_pixels)} surface-change candidate cells across ${number(data.optical.candidate_area_m2)} m². Neither count establishes locally validated flood, debris or road-damage accuracy.`,pointer:'/radar/counts/new_water_candidate_pixels; /optical/counts/candidate_pixels'}
  };
  const answer=answers[$('question').value];
  $('answer').innerHTML=`<h3>${safe(answer.title)}</h3><p>${safe(answer.text)}</p><a href="case/case-facts.json">Source fields: ${safe(answer.pointer)}</a>`;
}
for(const button of document.querySelectorAll('[data-view]'))button.addEventListener('click',()=>{if(data)selectView(button.dataset.view);});
$('scope').addEventListener('change',()=>{updatePointSelector();mapBounds=defaultBounds();renderSelectedPoint();});
$('point-search').addEventListener('input',()=>{updatePointSelector();mapBounds=defaultBounds();renderSelectedPoint();});
$('settlement').addEventListener('change',()=>{selectedId=$('settlement').value;renderSelectedPoint();});
$('question').addEventListener('change',renderAnswer);
$('fit-scope').addEventListener('click',()=>{mapBounds=defaultBounds();renderRoadMap();});
$('fit-route').addEventListener('click',async()=>{await loadRoutes();const route=routes?.[selectedId];if(!route?.length)return;const point=data.settlements.find((s)=>s.id===selectedId);mapBounds=paddedBounds([...route.flatMap((s)=>s.coordinates),...(point?.point_coordinates?[point.point_coordinates]:[])]);renderRoadMap();});
function zoom(factor) {const bounds=mapBounds??defaultBounds(),x=(bounds[0]+bounds[2])/2,y=(bounds[1]+bounds[3])/2,dx=(bounds[2]-bounds[0])*factor/2,dy=(bounds[3]-bounds[1])*factor/2;mapBounds=[x-dx,y-dy,x+dx,y+dy];renderRoadMap();}
$('zoom-in').addEventListener('click',()=>zoom(.75));$('zoom-out').addEventListener('click',()=>zoom(1.333333));
try {
  derivation=await readBoundJSON('derivation-manifest.json');
  if(derivation.source.sha256!==sourceHash)throw new Error('The display manifest no longer refers to the recorded case source.');
  data=await readBoundJSON('evidence.json',derivation.derived_files['evidence.json'].sha256);
  if(data.source_binding.sha256!==sourceHash)throw new Error('Compact facts are bound to a different case.');
  $('receipt-status').textContent=globalThis.crypto?.subtle?'Saved display facts verified':'Saved source bindings available';
  $('binding').textContent=`Full case-facts.json SHA-256: ${data.source_binding.sha256}. Source size: ${number(data.source_binding.bytes)} bytes. This display does not run inference, graph solving or acquisition.`;
  selectView('radar');renderAnswer();
  if(location.hash)followAnchor(location.hash);
} catch(error) {showError(`${error.message} Open the original case record to inspect the preserved output.`);$('receipt-status').textContent='Display unavailable; original record preserved';$('evidence-content').innerHTML='<h3>Open the preserved case record</h3><p class="summary">This compact display could not load. The original saved record and its source bindings are available directly.</p><a href="case/">Read original case evidence</a>';}
