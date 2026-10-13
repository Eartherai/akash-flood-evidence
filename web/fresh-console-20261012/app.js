'use strict';
const form = document.querySelector('#case-form');
const $ = (id) => document.getElementById(id);
let session = null;
let current = null;
let timer = null;
let pending = false;
const stageNames = {validate_sources:'Verify source identities',discover:'Discover satellite scenes',retrieve_historic_osm:'Fetch historical OSM',plan_case:'Select compatible scenes',acquire_native:'Acquire raw radar sources',prepare_dem:'Prepare Copernicus elevation',process_s1:'Calibrate and terrain-correct radar',infer_water:'Infer provisional water state',map_exposure:'Map provisional exposure',build_report:'Build source-bound report',freeze_optical_sources:'Freeze optical sources',acquire_and_screen_optical:'Fetch and screen optical observations',freeze_observation_scenario:'Freeze access scenario',analyze_observation_access:'Analyze observation and access',build_native_layer_report:'Build layer report',build_combined_case_report:'Build combined case report'};
const statusNames = {planned:'Plan ready',running:'Running',completed:'Completed',failed:'Failed',plan_failed:'Plan failed',cancelled:'Cancelled',interrupted:'Interrupted'};
const numNames = ['max_days','native_budget_mb','dem_budget_mb','optical_budget_mb','stage_timeout_seconds','max_total_seconds','optical_timeout_seconds'];
async function api(path, body) {
  const options = {cache:'no-store',headers:{}};
  if (body !== undefined) { options.method='POST'; options.headers={'Content-Type':'application/json','X-Akash-CSRF':session.csrf_token}; options.body=JSON.stringify(body); }
  const response = await fetch(path,options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function error(message) { $('form-error').textContent=message; $('form-error').hidden=!message;if(message)$('form-error').focus(); }
function parameters() {
  const fields = form.elements;
  const result={bbox:['west','south','east','north'].map((key)=>Number(fields[key].value)),event_date:fields.event_date.value,osm_snapshot:fields.osm_snapshot.value,workflow:fields.workflow.value};
  for (const key of numNames) result[key]=Number(fields[key].value);
  return result;
}
function buttons() {
  const running=current?.status==='running';
  $('plan-button').disabled=pending || running || !session;
  $('run-button').disabled=pending || !current || current.status!=='planned' || Boolean(current.plan?.blockers?.length);
  $('cancel-button').hidden=!running;
  $('cancel-button').disabled=pending || Boolean(current?.cancel_requested_at);
  $('receipt-button').disabled=!current;
}
function render(job) {
  current=job;
  $('empty-state').hidden=true; $('job-content').hidden=false;
  $('status').textContent=statusNames[job.status] || job.status; $('status').dataset.state=job.status;
  const p=job.parameters;
  $('case-summary').textContent=`${p.event_date} · ${p.bbox.map(v=>Number(v).toFixed(3)).join(', ')} · OSM before ${p.osm_snapshot}`;
  $('job-id').textContent=`Case ${job.id.slice(5)}. This plan runs at most once.`;
  const blockers=job.plan?.blockers || [];
  const reasons=[...blockers];
  if (job.failure_reason) reasons.push(job.failure_reason);
  if (job.receipt?.status==='failed') reasons.push(`Runner stopped at ${job.receipt.stopped_at_stage || 'an unrecorded stage'}. ${job.receipt.stages?.at(-1)?.failure_reason || ''}`);
  $('blockers').replaceChildren(); $('blockers').hidden=!reasons.length;
  if (reasons.length) { const list=document.createElement('ul'); for(const text of reasons){const li=document.createElement('li');li.textContent=text;list.append(li);} $('blockers').append(list); }
  const planned=job.plan?.stages || [];
  const actual=new Map((job.receipt?.stages || []).map(stage=>[stage.name,stage]));
  const completed=[...actual.values()].filter(stage=>stage.status==='completed').length;
  const active=job.active_stage_observation?.name;
  const elapsed=job.elapsed_seconds ?? job.receipt?.elapsed_seconds;
  $('progress-note').textContent=job.status==='planned' ? `${planned.length} declared stages. ${blockers.length ? 'Resolve the blockers before execution.' : 'No network requests or stage subprocesses started by this plan.'}` : `${completed} of ${planned.length} stages have completed runner receipts.${elapsed!==undefined ? ` Elapsed ${Number(elapsed).toFixed(1)} s.` : ''}${job.cancel_requested_at && job.status==='running' ? ' Stopping the actual process and its stage children.' : ''}`;
  $('stage-list').replaceChildren();
  for(const stage of planned) {
    const item=actual.get(stage.name);
    const state=item?.status || (active===stage.name ? 'logs_present' : 'pending');
    const li=document.createElement('li');li.dataset.state=state;
    const body=document.createElement('div');const name=document.createElement('div');name.className='stage-name';name.textContent=stageNames[stage.name] || stage.name.replaceAll('_',' ');body.append(name);
    const detail=document.createElement('p');detail.className='stage-detail';detail.textContent=item?.failure_reason || (item?.elapsed_seconds!==undefined ? `${Number(item.elapsed_seconds).toFixed(2)} s · ${item.artifacts?.length || 0} verified output records` : stage.network_scope);body.append(detail);
    const status=document.createElement('span');status.className='stage-state';status.textContent=state==='logs_present' ? 'Logs observed' : state==='pending' ? 'Pending' : state.charAt(0).toUpperCase()+state.slice(1);
    li.append(body,status);$('stage-list').append(li);
  }
  const logs=job.logs || {};
  $('stdout-log').textContent=[logs.stage_stdout,logs.stdout].filter(Boolean).join('\n') || (job.status==='planned' ? 'No process has started.' : 'Waiting for actual process output.');
  $('stderr-log').textContent=[logs.stage_stderr,logs.stderr].filter(Boolean).join('\n');$('stderr-log').hidden=!$('stderr-log').textContent;
  $('receipt-json').textContent=JSON.stringify(job,null,2);
  $('report-link').hidden=!job.combined_report_url;
  if(job.combined_report_url) $('open-report').href=job.combined_report_url;
  $('run-note').textContent=job.status==='completed' ? 'The declared engineering stages completed. Inspect coverage and limitations; this is not unseen hazard validation or a passability finding.' : job.status==='planned' ? 'Running fetches the declared sources within your budgets. No training or threshold changes are included.' : 'This execution uses its own fresh output directory. Errors and incomplete stages remain visible; there is no automatic retry.';
  buttons();
  clearTimeout(timer);
  if(job.status==='running') timer=setTimeout(poll,1800);
}
async function poll() {
  if(!current) return;
  try { render(await api(`/api/jobs/${current.id}`)); }
  catch(e) { $('progress-note').textContent=`Connection lost: ${e.message}. No completion is inferred. Reconnecting to the actual receipt.`;timer=setTimeout(poll,3000); }
}
async function listCases() {
  const data=await api('/api/jobs');$('recent-jobs').replaceChildren();
  if(!data.jobs.length){const p=document.createElement('p');p.className='hint';p.textContent='No case plans yet.';$('recent-jobs').append(p);return;}
  for(const job of data.jobs.slice(0,10)){const row=document.createElement('div');row.className='recent-row';const b=document.createElement('button');b.type='button';b.textContent=`${job.parameters.event_date} · ${job.parameters.bbox.map(v=>Number(v).toFixed(3)).join(', ')}`;b.addEventListener('click',async()=>{try{render(await api(`/api/jobs/${job.id}`));}catch(e){error(e.message);}});const s=document.createElement('span');s.textContent=statusNames[job.status] || job.status;row.append(b,s);$('recent-jobs').append(row);}
}
form.addEventListener('submit',async(event)=>{event.preventDefault();if(!form.reportValidity())return;pending=true;buttons();error('');try{render(await api('/api/plan',parameters()));await listCases();}catch(e){error(e.message);}finally{pending=false;buttons();}});
$('run-button').addEventListener('click',async()=>{pending=true;buttons();error('');try{render(await api('/api/run',{id:current.id}));await listCases();}catch(e){error(e.message);}finally{pending=false;buttons();}});
$('cancel-button').addEventListener('click',async()=>{pending=true;buttons();error('');try{await api('/api/cancel',{id:current.id});render(await api(`/api/jobs/${current.id}`));}catch(e){error(e.message);}finally{pending=false;buttons();}});
$('receipt-button').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(current,null,2)+'\n'],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`${current.id}-console-receipt.json`;a.click();URL.revokeObjectURL(url);});
$('refresh-button').addEventListener('click',()=>listCases().catch(e=>error(e.message)));
form.elements.workflow.addEventListener('change',()=>{$('pipeline-hint').textContent=form.elements.workflow.value==='multimodal' ? 'Adds actual optical acquisition, observation support and an access scenario. These do not establish local debris accuracy or observed road closures.' : 'Runs raw radar through provisional water state, mapped exposure and a source-bound report.';});
(async()=>{try{session=await api('/api/session');const d=session.defaults;for(const [i,key] of ['west','south','east','north'].entries())form.elements[key].value=d.bbox[i];for(const key of ['event_date','osm_snapshot','workflow',...numNames])form.elements[key].value=d[key];for(const option of form.elements.workflow.options)option.disabled=!session.workflows[option.value]?.available;if(Object.keys(session.explicit_source_reuse || {}).length){$('cache-note').hidden=false;const kinds=Object.keys(session.explicit_source_reuse);$('cache-note').textContent='Explicit saved source inputs: '+kinds.join(', ')+'. The runner verifies their identities and historical cutoff/extent. Reused sources are not newly downloaded data. Omit these startup options to request fresh acquisition.';}await listCases();buttons();}catch(e){error(`Local console unavailable: ${e.message}`);buttons();}})();
