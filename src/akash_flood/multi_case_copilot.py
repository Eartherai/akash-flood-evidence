"""Case-isolated CPU fact selection over the preserved Rasuwa/Kathmandu evidence.

Qwen selects IDs, never displayed measurements or prose. Every source clause is
checksum-bound. The Kathmandu road supplement is category-limited; buildings
are unavailable. This is a development assistant, not hazard/language validation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import sys

from .copilot import MODEL
from .focused_copilot import FocusedCaseFactStore, ask_focused_model
from .fresh_copilot import STATUS, _question, _source_json

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CaseConfig:
    slug: str
    label: str
    water: Path
    report_binding: Path
    exposure: Path | None = None
    optical: Path | None = None
    supplement: Path | None = None


def default_cases(root=ROOT):
    root = Path(root)
    return {
        "rasuwa": CaseConfig("rasuwa", "Rasuwa", root / "outputs/fresh-one-command-replay/water/manifest.json",
            root / "outputs/fresh-report-attributed/source-and-output-manifest.json",
            root / "outputs/fresh-one-command-replay/exposure/source-and-output-manifest.json",
            root / "outputs/s2-surface-change-resumed/manifest.json"),
        "kathmandu": CaseConfig("kathmandu", "Kathmandu", root / "outputs/kathmandu-unseen-water/water/manifest.json",
            root / "outputs/kathmandu-access-report-final-2026-10-08/source-and-output-manifest.json",
            supplement=root / "outputs/kathmandu-node-drive-access-2026-10-08/source-and-output-manifest.json"),
    }


def _supplement_reader():
    path = ROOT / "scripts/render_node_drive_access.py"
    spec = importlib.util.spec_from_file_location("akash_multi_case_supplement_validator", path)
    module = importlib.util.module_from_spec(spec)
    # Its established validators are sibling script modules.
    inserted = str(path.parent) not in sys.path
    if inserted:
        sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(str(path.parent))
    return module, path


class MultiCaseFactStore(FocusedCaseFactStore):
    @classmethod
    def from_config(cls, config):
        self = cls.from_paths(config.water, config.exposure, config.optical)
        self.config = config
        binding_path, binding = _source_json(config.report_binding)
        self._bind_record(self._helpers.record(binding_path), binding_path.parent)
        self._verify_records(binding, binding_path.parent)
        self.assets = {name: self._helpers.bound(item, binding_path.parent)
                       for name, item in binding["outputs"].items()}
        self.report_path = self.assets["report.json"]
        report = self._helpers.read_json(self.report_path)
        if report.get("case") != self.case or report.get("counts") != self.manifest["counts"] or report.get("source_dates") != self.manifest["source_dates"]:
            raise ValueError("Saved report case/counts/dates differ from active evidence")
        if config.supplement is None:
            inputs = [self._helpers.record(p) for p in (config.water, config.exposure, config.optical) if p is not None]
            if binding.get("schema") != "akash-fresh-report-binding-v1" or binding.get("status") != "completed" or binding.get("inputs") != inputs:
                raise ValueError("Rasuwa report must bind the exact active evidence")
        else:
            if config.exposure is not None or binding.get("schema") != "akash-node-drive-access-render-v1":
                raise ValueError("Require a separately bound category-limited supplement")
            inputs = [self._helpers.record(config.water), self._helpers.record(config.supplement)]
            if binding.get("inputs") != inputs:
                raise ValueError("Kathmandu rendered supplement binds different inputs")
            validator, validator_path = _supplement_reader()
            self._bind_record(self._helpers.record(validator_path), validator_path.parent)
            supplement_path, supplement = _source_json(config.supplement)
            self._bind_record(self._helpers.record(supplement_path), supplement_path.parent)
            self._verify_records(supplement, supplement_path.parent)
            water, _, _, summary, proof, files, _, settlement_points, destination_points = validator.read_supplement(config.water, supplement_path)
            if water != self.manifest or report["access_scenario"] != summary["access_scenario"]:
                raise ValueError("Supplement report scenario differs from verified source")
            if any(report[k] != summary[k] for k in ("roads", "bridges", "buildings", "category_availability")):
                raise ValueError("Supplement report categories differ from verified source")
            def inside(point, bbox):
                return bbox[0] <= point['lon'] <= bbox[2] and bbox[1] <= point['lat'] <= bbox[3]
            point_scope = {'settlements_in_aoi': sum(inside(p, self.case['bbox']) for p in settlement_points),
                'settlements_in_context': sum(inside(p, self.case['context_bbox']) for p in settlement_points),
                'incidental_settlements_outside_context': [p for p in settlement_points if not inside(p, self.case['context_bbox'])],
                'all_source_settlements': len(settlement_points), 'all_source_destinations': len(destination_points)}
            if report.get('point_scope') != point_scope:
                raise ValueError("Rendered point-scope counts differ from the verified historical source")
            del self.facts["missing_exposure"]
            self._supplement_facts(summary, proof, files, report)
        self._add("case_identity", f"Active case: {config.label}; event date {self.case['event_date']}; source case ID {self.case['case_id']}. Switch the case selector to ask about the other saved case.",
            f"सक्रिय घटना: {config.label}; घटना मिति {self.case['event_date']}; स्रोत घटना ID {self.case['case_id']}। अर्को भण्डारित घटनाबारे सोध्न घटना चयन बदल्नुहोस्।",
            "/case/case_id;/case/event_date", ("case", "event", "identity", "घटना"))
        self._bind_record(self._helpers.record(Path(__file__)), ROOT)
        self.source_bindings = [self._bindings[key] for key in sorted(self._bindings)]
        self.verification = ("Saved water/optical domains recounted at activation; recorded sources checksum-verified. "
            "Kathmandu nearest targets and all source-edge certificates rechecked at activation. "
            "All pinned sources checked before/after each CPU question; no hazard inference or checking labels loaded.")
        return self

    def _supplement_facts(self, summary, proof, files, report):
        summary_path, proof_path = files["summary.json"], files["access-proof.json"]
        scenario = summary["access_scenario"]
        self._add("osm_snapshot", f"The Kathmandu historical OSM snapshot requested {summary['historical_osm']['snapshot']}. Driving roads and node-tagged settlement/hospital POIs were requested; buildings and polygon-only facilities were excluded. Independent historical-server authenticity and source completeness are unvalidated.",
            f"काठमाडौंको ऐतिहासिक OSM स्न्यापशटले {summary['historical_osm']['snapshot']} अनुरोध गरेको छ। सवारी सडक तथा नोड-ट्याग भएका बस्ती र अस्पताल बिन्दु अनुरोध गरिएका थिए; भवन र बहुभुज मात्र भएका सुविधा समावेश छैनन्। ऐतिहासिक सर्भरको स्वतन्त्र प्रामाणिकता र स्रोत पूर्णता प्रमाणीकरण भएका छैनन्।",
            "/historical_osm", ("osm", "snapshot", "historical", "date", "source"), path=summary_path)
        self._add("missing_buildings", "Kathmandu building inventory is unavailable: buildings were not requested. No building intersection or damage count can be inferred; unavailable does not mean zero. Polygon-only facilities are also excluded.",
            "काठमाडौंको भवन सूची उपलब्ध छैन: भवन अनुरोध गरिएका थिएनन्। भवन छेदन वा क्षतिको गणना निकाल्न मिल्दैन; उपलब्ध नभएकोलाई शून्य मान्न मिल्दैन। बहुभुज मात्र भएका सुविधा पनि समावेश छैनन्।",
            "/buildings;/category_availability/buildings;/historical_osm/categories/buildings", ("building", "count", "unavailable", "missing", "भवन", "कति"), path=summary_path)
        new = [summary[k]["features_with_new_water_candidate_overlap"] for k in ("roads", "bridges")]
        lost = [summary[k]["features_with_lost_water_candidate_overlap"] for k in ("roads", "bridges")]
        self._add("exposure_intersections", f"New-water candidates intersect {new[0]:,} physical driving-road segments and {new[1]:,} bridge-tagged segments. Lost-water candidates intersect {lost[0]:,} roads and {lost[1]:,} bridge-tagged segments. Bridges are a road subset. The 10 m half-width corridor is an association rule; intersections do not establish physical closures or damage. Building intersections are unavailable.",
            f"नयाँ पानी उम्मेदवारसँग {new[0]:,} भौतिक सवारी सडक खण्ड र {new[1]:,} पुल-ट्याग खण्डको छेदन छ। हराएको पानी उम्मेदवारसँग {lost[0]:,} सडक र {lost[1]:,} पुल-ट्याग खण्डको छेदन छ। पुल सडककै उपसमूह हुन्। 10 मिटर अर्ध-चौडाइको कोरिडोर सम्बन्ध नियम हो; छेदनले वास्तविक बन्द सडक वा क्षति स्थापित गर्दैन। भवन छेदन उपलब्ध छैन।",
            "/roads/features_with_new_water_candidate_overlap;/bridges/features_with_new_water_candidate_overlap;/roads/features_with_lost_water_candidate_overlap;/bridges/features_with_lost_water_candidate_overlap;/road_association", ("road", "bridge", "intersection", "intersections", "candidate", "count", "सडक", "पुल", "छेदन"), path=summary_path)
        n, d = scenario["n_settlements"], scenario["n_destinations"]
        self._add("access_counts", f"All {n:,} source settlements have a saved route to their fixed baseline destination in both scenario extremes. There are {d:,} mapped node-tagged destination candidates. This conditional robust-connected graph status does not establish observed access, safe passage or hospital operation.",
            f"स्रोतका सबै {n:,} बस्तीबाट दुवै सर्तगत चरम अवस्थामा निश्चित आधारभूत गन्तव्यसम्म भण्डारित मार्ग छ। नोड-ट्याग भएका {d:,} नक्सागत गन्तव्य उम्मेदवार छन्। यो सर्तगत ग्राफको robust_connected अवस्थाले अवलोकन गरिएको पहुँच, सुरक्षित यात्रा वा अस्पताल सञ्चालन स्थापित गर्दैन।",
            "/access_scenario/status_counts;/access_scenario/n_settlements;/access_scenario/n_destinations", ("access", "settlement", "count", "status", "बस्ती", "पहुँच", "कति"), path=summary_path)
        uncertain, directed = scenario["n_uncertain_physical_segments"], scenario["n_uncertain_directed_edges"]
        self._add("access_assumptions", f"The hypothetical scenario varies {uncertain:,} new-water-intersecting physical segments ({directed:,} directed edges) jointly open/closed, with every state combination allowed. All other baseline edges remain assumed available, including unknown and outside-AOI edges. No observed closures or probabilities are assigned; debris and unspecified failures are excluded.",
            f"काल्पनिक सर्तगत अवस्थाले नयाँ पानीसँग छेदन भएका {uncertain:,} भौतिक खण्ड ({directed:,} निर्देशित किनारा) लाई संयुक्त खुला वा बन्द मान्छ; सबै अवस्था संयोजन अनुमति छन्। अज्ञात र अनुरोध क्षेत्रबाहिरका किनारासहित बाँकी सबै आधारभूत किनारा उपलब्ध भएको मानिन्छ। अवलोकन गरिएका बन्द सडक वा सम्भाव्यता तोकिएका छैनन्; भग्नावशेष र नतोकिएका विफलता समावेश छैनन्।",
            "/scenario;/uncertain_links", ("access", "assumption", "scenario", "unknown", "outside", "open", "closed", "मान्यता"), path=proof_path)
        self._add("access_method", f"Each settlement retains its nearest baseline-reachable mapped town/city/hospital by directed road length plus fixed node-association distances, with a {proof['max_snap_m']} m association limit. Targets do not switch after scenario closures. This does not validate facility operation, entrances, off-road connections or travel time.",
            f"प्रत्येक बस्तीको निश्चित आधारभूत अवस्थामा पुग्न सकिने नक्सागत नगर वा अस्पताल निर्देशित सडक लम्बाइ र निश्चित नोड-सम्बन्ध दूरी जोडेर छानिन्छ; सम्बन्ध सीमा {proof['max_snap_m']} मिटर हो। सर्तगत बन्द अवस्थापछि गन्तव्य बदलिँदैन। यसले सुविधा सञ्चालन, प्रवेश, सडकबाहिरको सम्बन्ध वा यात्रा समय प्रमाणीकरण गर्दैन।",
            "/destination_rule;/max_snap_m;/destination_switching_after_closure;/facility_scope", ("access", "method", "nearest", "destination", "hospital", "distance", "गन्तव्य"), path=proof_path)
        checks = proof["certificate_checks"]
        self._add("access_certificates", f"The source-edge validator rechecked {checks['paths']} saved path witnesses and {checks['cuts']} cut certificates at activation. This checks the declared graph scenario; it is not field validation of roads or facilities.",
            f"स्रोत-किनारा प्रमाणीकरणले सक्रिय गर्दा {checks['paths']} भण्डारित मार्ग साक्षी र {checks['cuts']} कट प्रमाणपत्र पुनः जाँचेको छ। यसले तोकिएको ग्राफ अवस्था जाँच्छ; सडक वा सुविधाको स्थलगत प्रमाणीकरण होइन।",
            "/certificate_checks", ("access", "certificate", "path", "proof", "cut", "प्रमाणपत्र"), path=proof_path)
        self._add("access_limits", "A robust graph route does not establish physical passability, safe evacuation, a functioning hospital, damage or observed closures. Missing roads/facilities, source boundaries, point associations and the baseline availability assumption limit this scenario.",
            "सबै सर्तमा रहेको ग्राफ मार्गले वास्तविक पारगम्यता, सुरक्षित उद्धार, सञ्चालनमा रहेको अस्पताल, क्षति वा अवलोकन गरिएको बन्द सडक स्थापित गर्दैन। छुटेका सडक वा सुविधा, स्रोत सीमा, बिन्दु सम्बन्ध र आधारभूत उपलब्धताको मान्यताले यस अवस्थालाई सीमित गर्छन्।",
            "/facility_scope;/scenario/exclusions", ("access", "limits", "safe", "closure", "damage", "सीमा"), path=proof_path)
        scope = report["point_scope"]
        self._add("access_point_scope", f"Of {scope['all_source_settlements']} source settlements, {scope['settlements_in_aoi']} are inside the water AOI and {scope['settlements_in_context']} inside the extraction context. {len(scope['incidental_settlements_outside_context'])} incidental way-node dependency is outside that context. All are retained in graph analysis; AOI membership is not a hazard finding.",
            f"स्रोतका {scope['all_source_settlements']} बस्तीमध्ये {scope['settlements_in_aoi']} पानी क्षेत्रभित्र र {scope['settlements_in_context']} निकासी सन्दर्भभित्र छन्। {len(scope['incidental_settlements_outside_context'])} अतिरिक्त way-node निर्भरता सन्दर्भबाहिर छ। सबैलाई ग्राफ विश्लेषणमा राखिएको छ; क्षेत्रभित्र हुनु जोखिमको निष्कर्ष होइन।",
            "/point_scope", ("access", "inside", "outside", "aoi", "context", "बाहिर", "भित्र"), path=self.report_path)
        destinations = self._helpers.read_json(files["destinations.geojson"])["features"]
        dest = {p["properties"]["id"]: (i, p["properties"]["name"]) for i, p in enumerate(destinations)}
        locations = self._helpers.read_json(files["settlements.geojson"])["features"]
        loc = {p["properties"]["id"]: (i, p["properties"]["within_requested_aoi"]) for i, p in enumerate(locations)}
        for i, row in enumerate(proof["settlements"]):
            sid, name = row["id"], row["name"]
            base = "settlement_" + __import__("hashlib").sha256(sid.encode()).hexdigest()[:12]
            selection = row["nearest_baseline_destination"]
            if row["status"] != "robust_connected" or selection is None or sid not in loc or selection["destination_id"] not in dest:
                raise ValueError("Unexpected Kathmandu source settlement/target")
            di, dn = dest[selection["destination_id"]]
            self.settlement_ids[sid] = {"name": name, "fact_ids": [base, base + "_destination"]}
            place_en, place_ne = ("inside", "भित्र") if loc[sid][1] else ("outside", "बाहिर")
            self._add(base, f"For {name} ({sid}), a graph route to its fixed destination exists in both scenario extremes (robust_connected). Its source point is {place_en} the water AOI. This is conditional connectivity, not observed passability.",
                f"{name} ({sid}) बाट दुवै सर्तगत चरम अवस्थामा निश्चित गन्तव्यसम्म ग्राफ मार्ग छ (robust_connected)। स्रोत बिन्दु पानी क्षेत्र{place_ne} छ। यो सर्तगत जोडिने अवस्था हो, अवलोकन गरिएको पारगम्यता होइन।",
                f"/settlements/{i}/status", (name, sid, "settlement", "access", "status", "बस्ती"), path=proof_path,
                references=((files["settlements.geojson"], f"/features/{loc[sid][0]}/properties/within_requested_aoi"),))
            self._add(base + "_destination", f"For {name} ({sid}), the fixed baseline destination is {dn} ({selection['destination_id']}): {selection['baseline_road_distance_m']:,.3f} m directed road length and {selection['baseline_distance_including_association_m']:,.3f} m including fixed associations. These are modeled lengths; facility operation and travel time are unknown.",
                f"{name} ({sid}) को निश्चित आधारभूत गन्तव्य {dn} ({selection['destination_id']}) हो: निर्देशित सडक लम्बाइ {selection['baseline_road_distance_m']:,.3f} मिटर र निश्चित सम्बन्धसहित {selection['baseline_distance_including_association_m']:,.3f} मिटर। यी मोडेलका लम्बाइ हुन्; सुविधा सञ्चालन र यात्रा समय अज्ञात छन्।",
                f"/settlements/{i}/nearest_baseline_destination", (name, sid, "destination", "nearest", "hospital", "distance", "गन्तव्य"), path=proof_path,
                references=((files["destinations.geojson"], f"/features/{di}/properties/name"),))

    def policy_reason(self, question):
        folded = _question(question)
        other = ("kathmandu", "काठमाडौं", "काठमाडौँ") if self.config.slug == "rasuwa" else ("rasuwa", "रसुवा")
        if any(term in folded for term in other) or any(year != self.case["event_date"][:4] for year in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", folded)):
            return "requested_case_differs_from_selected_case"
        if "missing_buildings" in self.facts and any(term in folded for term in ("building", "भवन")) and any(term in folded for term in ("count", "how many", "कति", "गणना", "शून्य", "zero", "intersection", "छेदन")):
            return "building_inventory_unavailable_not_zero"
        if any(term in folded for term in ("physically closed", "observed road closure", "roads are closed", "actually blocked", "वास्तविक बन्द", "प्रत्यक्ष बन्द", "सडक बन्द छन्")):
            return "no_observed_damage_passability_or_operational_safety"
        if any(term in folded for term in ("accuracy", "iou", "precision", "recall", "शुद्धता", "सटीकता")) and not any(term in folded for term in ("not accuracy", "not establish accuracy", "शुद्धता होइन", "unvalidated", "unavailable")):
            return "no_local_hazard_accuracy_measurement"
        if any(term in folded for term in ("बनाएर", "आँकडा बनाऊ", "तथ्य बदल", "invent", "fabricat", "make up", "ignore")):
            return "instruction_override_or_fabrication_request"
        return super().policy_reason(question)

    def ranked_facts(self, question, limit=8):
        reason = self.policy_reason(question)
        if reason:
            contextual = {"requested_case_differs_from_selected_case": "case_identity",
                "building_inventory_unavailable_not_zero": "missing_buildings",
                "no_local_hazard_accuracy_measurement": "water_limits",
                "no_observed_damage_passability_or_operational_safety": "access_limits",
                "instruction_override_or_fabrication_request": "case_identity"}.get(reason, "scope")
            return [self.facts[contextual], self.facts["scope"]] if contextual != "scope" else [self.facts["scope"]]
        facts = super().ranked_facts(question, limit)
        if "missing_buildings" in self.facts and any(x in _question(question) for x in ("building", "भवन")):
            return [self.facts["missing_buildings"], self.facts["scope"]]
        return facts

    retrieve = ranked_facts

    def model_payload(self, question, language="en"):
        result = super().model_payload(question, language)
        context = json.loads(result["messages"][1]["content"])
        context["active_case"]["label"] = self.config.label
        # Scope is always rendered as mandatory context. Actual v2 responses
        # unnecessarily selected it as an extra answer ID. Supported questions
        # retain only evidence-specific selectable IDs; refusals keep the
        # established scope candidate and explicit post-generation guard.
        if self.policy_reason(question) is None and any(f['id'] != 'scope' for f in context['candidate_facts']):
            context['candidate_facts'] = [f for f in context['candidate_facts'] if f['id'] != 'scope']
            result['candidate_ids'] = [identifier for identifier in result['candidate_ids'] if identifier != 'scope']
        result["messages"][1]["content"] = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        # v1 added a nullable reason field and another conditional instruction.
        # Actual Qwen over-refused all four supported questions. Keep the
        # established focused prompt and apply the same explicit policy after
        # generation; preserve raw plans independently of that policy.
        return result

    prompt = model_payload

    def resolve_model_plan(self, plan, *, language, candidate_ids):
        result = super().resolve_model_plan(plan, language=language, candidate_ids=candidate_ids)
        result["selected_case"] = self.config.slug
        result["case_label"] = self.config.label
        return result

    resolve = resolve_model_plan

    def verify_unchanged(self):
        for item in self.source_bindings:
            if self._helpers.record(Path(item["path"])) != item:
                raise ValueError("A case source changed; restart after verified reconstruction")


def ask_multi_case_model(store, question, *, language="en", endpoint="http://127.0.0.1:8782", timeout=60):
    store.verify_unchanged()
    result = ask_focused_model(store, question, language=language, endpoint=endpoint, timeout=timeout)
    store.verify_unchanged()
    # Declined requests still expose the bound reason clause in the UI. It is
    # explicit deterministic policy context, not the model's selected answer.
    reason = result["policy_guard"]["reason"]
    if reason:
        facts = [f for f in store.retrieve(question) if f.id != "scope"]
        result["policy_context_claims"] = [{"fact_id": f.id, "text": getattr(f, language),
            "source_pointer": f.source_pointer, "source_path": f.source_path,
            "source_sha256": f.source_sha256, "source_references": list(f.source_references)} for f in facts]
    else:
        result["policy_context_claims"] = []
    result["ai_component"] = "Actual Qwen CPU fact-ID/refusal generation; verified source clauses and explicit refusal guard"
    result["model"] = MODEL
    return result
