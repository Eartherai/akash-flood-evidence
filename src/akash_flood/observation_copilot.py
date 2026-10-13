"""Optional source-clause extension for the separately frozen road-support audit.

Existing case factories/facts/prompt remain unchanged. This extension reads only
saved metadata; it performs no graph solve, model invocation or checking-map read.
Nepali clauses are AI-authored and have not received human language review.
"""
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import re

from .fresh_copilot import FreshFact, _question, _source_json
from .multi_case_copilot import CaseConfig, MultiCaseFactStore, default_cases, ROOT

SUMMARY_RELATIVE = "research/observation-access-2026-10-11/summary.json"
CASE_SLUGS = ("rasuwa", "kathmandu")
COUNTS = ("robust_connected", "ambiguous", "unmapped", "baseline_disconnected")
CLAIMS = {"observed_closures": False, "safe_routes": False, "local_hazard_accuracy": False,
          "model_inference": False, "training": False, "network": False, "gpu": False,
          "reference_geometry_loaded": False}
OBSERVATION_TERMS = ("observation-aware", "observation aware", "coverage-aware", "coverage aware",
    "expanded", "unobserved", "unknown roads", "unknown road", "unsupported", "road coverage", "route coverage",
    "route support", "empty route", "zero-length", "विस्तारित", "अज्ञात सडक", "अवलोकन नभएका",
    "अवलोकन-आधारित", "असमर्थित", "मार्ग कभरेज", "सडक कभरेज", "खाली मार्ग", "शून्य लम्बाइ")


@dataclass(frozen=True)
class ObservationCaseConfig(CaseConfig):
    observation_receipt: Path | None = None
    observation_summary: Path | None = None
    observation_source_root: Path = ROOT


def _check_record(item, root):
    if not isinstance(item, dict) or set(item) != {"path", "bytes", "sha256"}:
        raise ValueError("Observation source record requires path, bytes and SHA256")
    if not isinstance(item["path"], str) or not isinstance(item["sha256"], str):
        raise ValueError("Invalid observation source path/hash")
    relative = Path(item["path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("Observation evidence must use contained relative paths")
    if isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int) or item["bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
        raise ValueError("Invalid observation source size/hash")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()): raise ValueError("Observation source escapes root")
    with path.open("rb") as stream: digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if path.stat().st_size != item["bytes"] or digest != item["sha256"]:
        raise ValueError("Observation evidence hash/size changed")
    return path


def observation_cases(audit_root, *, source_root=ROOT, cases=None):
    """Require both completed source-bound audit cases before activating either."""
    source_root, audit_root = Path(source_root).resolve(), Path(audit_root).resolve()
    cases = cases or default_cases(source_root)
    if set(cases) != set(CASE_SLUGS): raise ValueError("Observation extension requires both explicit cases")
    summary_path, summary = _source_json(source_root / SUMMARY_RELATIVE)
    if summary.get("schema") != "akash-observation-access-summary-v1" or set(summary.get("cases", {})) != set(CASE_SLUGS):
        raise ValueError("Require the exact two-case observation summary")
    _check_record(summary["independent_verification"], source_root)
    result = {}
    for slug in CASE_SLUGS:
        item = summary["cases"][slug]
        receipt_path = _check_record(item["receipt"], source_root)
        if receipt_path != audit_root / (slug + "-v1") / "receipt.json":
            raise ValueError("Receipt is outside the selected observation audit root")
        _, receipt = _source_json(receipt_path)
        if (receipt.get("schema") != "akash-observation-access-receipt-v1" or receipt.get("status") != "completed"
                or receipt.get("all_frozen_inputs_unchanged") is not True or receipt.get("claims") != CLAIMS):
            raise ValueError("Require completed reference-free observation receipts")
        if receipt.get("protocol") != summary["protocol"]: raise ValueError("Observation protocols differ")
        _check_record(receipt["protocol"], source_root)  # Opaque protocol binding; no training/reference data are opened.
        output_paths = [_check_record(record, source_root) for record in receipt["outputs"]]
        expected = {receipt_path.parent / "expanded-scenario.json", receipt_path.parent / "original-witness-support.json"}
        if set(output_paths) != expected or len(output_paths) != 2 or item["full_evidence"] != receipt["outputs"]:
            raise ValueError("Require both exact new observation outputs")
        values = {k: getattr(cases[slug], k) for k in CaseConfig.__dataclass_fields__}
        result[slug] = ObservationCaseConfig(**values, observation_receipt=receipt_path,
            observation_summary=summary_path, observation_source_root=source_root)
    return result


def _count_line(counts):
    if any(k not in COUNTS or isinstance(v, bool) or not isinstance(v, int) or v < 0 for k,v in counts.items()):
        raise ValueError("Invalid observation graph status counts")
    return ", ".join(f"{counts.get(k, 0):,} {k}" for k in COUNTS)


class ObservationCaseFactStore(MultiCaseFactStore):
    @classmethod
    def from_verified_store(cls, store, config):
        """Extend an already activated original store without repeating graphs."""
        self = cls()
        self.__dict__.update(store.__dict__)
        self.facts = dict(store.facts)
        self._bindings = dict(store._bindings)
        self.config = config
        self._observation_facts(config)
        return self

    @classmethod
    def from_config(cls, config):
        self = super().from_config(config)
        if getattr(config, "observation_receipt", None) is not None:
            self._observation_facts(config)
        return self

    def _observation_add(self, fact_id, en, ne, pointer, tags, path):
        path = Path(path).resolve()
        item = self._bindings[str(path)]
        reference = {"source_pointer": pointer, "source_path": str(path), "source_sha256": item["sha256"]}
        if fact_id in self.facts: raise ValueError("Duplicate observation fact")
        self.facts[fact_id] = FreshFact(fact_id, en, ne, pointer, tuple(tags), item["sha256"], str(path), (reference,))

    def _observation_facts(self, config):
        root = Path(config.observation_source_root).resolve()
        summary_path, summary = _source_json(config.observation_summary)
        receipt_path, receipt = _source_json(config.observation_receipt)
        selected = summary["cases"][config.slug]
        if receipt["case"] != self.case or _check_record(selected["receipt"], root) != receipt_path:
            raise ValueError("Observation receipt differs from selected case")
        if receipt.get("status") != "completed" or receipt.get("claims") != CLAIMS or receipt.get("all_frozen_inputs_unchanged") is not True:
            raise ValueError("Uncompleted or unsafe observation receipt")
        for path in (summary_path, receipt_path):
            self._bind_record(self._helpers.record(path), path.parent)
        for record in receipt["outputs"] + [receipt["protocol"]]:
            path = _check_record(record, root)
            self._bind_record(self._helpers.record(path), path.parent)
        independent_path = _check_record(summary["independent_verification"], root)
        self._bind_record(self._helpers.record(independent_path), independent_path.parent)
        expanded_path, expanded = _source_json(receipt_path.parent / "expanded-scenario.json")
        witness_path, witnesses = _source_json(receipt_path.parent / "original-witness-support.json")
        if expanded["case"] != self.case or witnesses["case"] != self.case:
            raise ValueError("Cross-case observation output")
        if (expanded.get("schema") != "akash-observation-access-bounds-v1"
                or witnesses.get("schema") != "akash-original-witness-support-v1"
                or expanded.get("confirmed_closures") != 0 or expanded.get("certain_closed_edge_ids") != []
                or expanded.get("destination_switching_after_closure") is not False
                or expanded["scenario_policy"].get("observed_nonwater_is_passable") is not False
                or expanded["scenario_policy"].get("unknown_is_observed_closed") is not False
                or expanded["scenario_policy"].get("positive_area_tolerance_m2") != 1e-6
                or expanded["scenario_policy"].get("certain_closures") != []):
            raise ValueError("Require conditional unknown-road observation scenario")
        rows = expanded["settlements"]
        actual = dict(Counter(r["status"] for r in rows))
        previous = dict(Counter(r["previous_status"] for r in rows))
        if (len({r["id"] for r in rows}) != len(rows) or actual != receipt["expanded_status_counts"]
                or actual != expanded["status_counts"] or actual != selected["expanded_status_counts"]
                or previous != receipt["original_status_counts"] or previous != selected["original_status_counts"]):
            raise ValueError("Observation row status/count inconsistency")
        binding_path = config.exposure or config.supplement
        _, old_binding = _source_json(binding_path)
        old_path = self._helpers.bound(old_binding["outputs"]["access-proof.json"], binding_path.parent)
        _, old = _source_json(old_path)
        old_rows = {r["id"]:r for r in old["settlements"]}
        if set(old_rows) != {r["id"] for r in rows}: raise ValueError("Observation omitted or invented settlement")
        for row in rows:
            prior = old_rows[row["id"]]
            if any(row[k] != prior[k] for k in ("name", "snap_node_id", "nearest_baseline_destination")) or row["previous_status"] != prior["status"]:
                raise ValueError("Observation changed source point/fixed target")
            p = row["point_coordinates"]
            def inside(b): return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]
            if row["inside_water_aoi"] != inside(self.case["bbox"]) or row["inside_extraction_context"] != inside(self.case["context_bbox"]):
                raise ValueError("Observation AOI scope inconsistency")
        scopes = {k:dict(Counter(r["status"] for r in rows if pred(r))) for k,pred in (
            ("all_source",lambda r:True),("inside_water_aoi",lambda r:r["inside_water_aoi"]),
            ("inside_extraction_context",lambda r:r["inside_extraction_context"]),
            ("incidental_outside_context",lambda r:not r["inside_extraction_context"]))}
        if scopes != receipt["scope_status_counts"] or scopes != expanded["scope_status_counts"] or scopes != selected["scope_status_counts"]:
            raise ValueError("Observation scoped totals differ")
        weighted = [0.0,0.0,0.0];total = 0.0
        keys = ("assessed", "unknown_within_aoi", "outside_aoi")
        for w in witnesses["witnesses"]:
            if w["id"] not in old_rows: raise ValueError("Witness from another case")
            s = w["support"]; n = s["projected_centerline_total_m"]
            values = [s["lengths_m"][k] for k in keys]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in [n] + values):
                raise ValueError("Invalid witness support length")
            if not math.isclose(sum(values),n,abs_tol=1e-4,rel_tol=1e-12): raise ValueError("Witness support partition differs")
            for k,v in zip(keys,values):
                f=s["fractions_of_projected_centerline"][k]
                if (n==0 and f is not None) or (n>0 and not math.isclose(f,v/n,abs_tol=1e-12)): raise ValueError("Invalid witness support fraction")
            weighted=[a+b for a,b in zip(weighted,values)];total+=n
        fractions={k:v/total if total else None for k,v in zip(keys,weighted)}
        expected=selected["original_witness_statistics"]
        if witnesses["witness_count"]!=len(witnesses["witnesses"]) or expected["count"]!=len(witnesses["witnesses"]) or any(
                (fractions[k] is None and expected["traversal_weighted_fractions"][k] is not None)
                or (fractions[k] is not None and (expected["traversal_weighted_fractions"][k] is None
                or not math.isclose(fractions[k],expected["traversal_weighted_fractions"][k],abs_tol=1e-12))) for k in keys):
            raise ValueError("Witness aggregate differs from bound summary")
        zero = [r["id"] for r in rows if (r.get("path_support") or {}).get("projected_centerline_total_m")==0]
        if zero!=selected["zero_length_graph_witness_ids"]:raise ValueError("Empty route case mismatch")
        for row in rows:
            support=row.get("path_support")
            if support and support["projected_centerline_total_m"]==0 and any(v is not None for v in support["fractions_of_projected_centerline"].values()):
                raise ValueError("Empty graph route fractions must remain null")
        def add(identifier,en,ne,pointer,tags,path=receipt_path):
            self._observation_add(identifier,en,ne,pointer,tags,path)
        old_text,new_text=_count_line(previous),_count_line(actual)
        add("newscenario_counts",f"{config.label}: the original candidate-only scenario has {old_text}; the expanded observation-aware scenario has {new_text}. Original fixed baseline targets remain unchanged. These are conditional graph states, not observed closures or safe access.",
            f"{config.label}: मूल उम्मेदवार-मात्र सर्तगत अवस्थामा {old_text}; विस्तारित अवलोकन-आधारित अवस्थामा {new_text} छन्। मूल निश्चित आधारभूत गन्तव्य बदलिएका छैनन्। यी सर्तगत ग्राफ अवस्था हुन्, अवलोकन गरिएका बन्द सडक वा सुरक्षित पहुँच होइनन्।",
            "/original_status_counts;/expanded_status_counts",("expanded","observation","counts","status","विस्तारित","कति"))
        add("newscenario_assumptions",f"The expanded scenario jointly varies {receipt['uncertain_physical_group_count']:,} physical segments ({receipt['uncertain_directed_edge_count']:,} directed edges): original candidate-selected roads plus roads with unknown-within-AOI or outside-AOI corridor support. Every open/closed combination is permitted, with zero certain closures and unchanged fixed destinations. Assessed nonwater does not prove passability; unknown is not observed closed.",
            f"विस्तारित सर्तगत अवस्थामा {receipt['uncertain_physical_group_count']:,} भौतिक खण्ड ({receipt['uncertain_directed_edge_count']:,} निर्देशित किनारा) संयुक्त रूपमा बदलिन्छन्: मूल उम्मेदवार-चयन सडक तथा अज्ञात वा क्षेत्रबाहिरको कोरिडोर समर्थन भएका सडक। सबै खुला/बन्द संयोजन अनुमति छन्; निश्चित बन्द सडक शून्य र गन्तव्य उही छन्। मूल्याङ्कन गरिएको गैर-पानीले पारगम्यता प्रमाणित गर्दैन; अज्ञातलाई अवलोकन गरिएको बन्द मानिँदैन।",
            "/uncertain_physical_group_count;/uncertain_directed_edge_count;/claims",("expanded","unknown","unobserved","assumption","अज्ञात सडक","मान्यता"))
        scope_text="; ".join(k+": "+_count_line(v) for k,v in scopes.items())
        add("newscenario_scope",f"Expanded observation scenario by source-point scope: {scope_text}. Source settlements outside the water AOI are retained; incidental dependencies outside the extraction context are separate. AOI membership is not a hazard finding.",
            f"विस्तारित अवलोकन अवस्थाको स्रोत-बिन्दु दायरा: {scope_text}। पानी क्षेत्रबाहिरका स्रोत बस्ती राखिएका छन्; निकासी सन्दर्भबाहिरका अतिरिक्त निर्भरता छुट्टै छन्। क्षेत्रभित्र हुनु जोखिमको निष्कर्ष होइन।",
            "/scope_status_counts",("expanded","inside","outside","aoi","scope","भित्र","बाहिर"))
        percent=" / ".join(f"{fractions[k]*100:.2f}% {k}" if fractions[k] is not None else f"unavailable {k}" for k in keys)
        add("newscenario_witness_coverage",f"Across all {len(witnesses['witnesses'])} original saved witnesses, traversal-weighted projected centreline support is {percent}. Shared road sections and repeated traversals count each time. This is not unique network coverage, accuracy, failure probability or a newly optimized route.",
            f"मूल भण्डारित {len(witnesses['witnesses'])} मार्ग साक्षीको दोहोरिने यात्राअनुसार भारित प्रक्षेपित केन्द्ररेखा समर्थन {percent} छ। साझा सडक खण्ड र दोहोरिने यात्रा हरेक पटक गनिन्छन्। यो अद्वितीय सञ्जाल कभरेज, शुद्धता, विफलताको सम्भावना वा नयाँ अनुकूलित मार्ग होइन।",
            "/cases/"+config.slug+"/original_witness_statistics",("expanded","coverage","unsupported","length","witness","कभरेज","लम्बाइ"),summary_path)
        empty_text=", ".join(zero) if zero else "none"
        add("newscenario_empty_route",f"Expanded scenario zero-length graph witness IDs: {empty_text}. An empty source-to-target-node path has null support fractions, never 100% observed coverage. It does not validate the physical point-to-road connection, facility entrance or operation.",
            f"विस्तारित अवस्थामा शून्य लम्बाइका ग्राफ मार्ग साक्षी ID: {empty_text}। एउटै स्रोत-गन्तव्य नोडको खाली मार्गमा समर्थन अंश null हुन्छन्, 100% अवलोकन गरिएको कभरेज होइन। यसले वास्तविक बिन्दु-सडक सम्बन्ध, सुविधा प्रवेश वा सञ्चालन प्रमाणीकरण गर्दैन।",
            "/cases/"+config.slug+"/zero_length_graph_witness_ids",("expanded","empty","zero-length","null","खाली मार्ग","शून्य लम्बाइ"),summary_path)
        self.observation_settlement_ids={}
        for i,row in enumerate(rows):
            identifier="newscenario_settlement_"+hashlib.sha256(row['id'].encode()).hexdigest()[:12]
            self.observation_settlement_ids[row['id']]=identifier
            target=row['nearest_baseline_destination'];target_id=target['destination_id'] if target else "unavailable"
            add(identifier,f"Expanded observation scenario for {row['name']} ({row['id']}): original candidate-only status {row['previous_status']}; expanded status {row['status']}; same fixed target {target_id}. Inside water AOI: {row['inside_water_aoi']}; inside extraction context: {row['inside_extraction_context']}. This is a conditional graph result, not actual passability.",
                f"{row['name']} ({row['id']}) को विस्तारित अवलोकन अवस्था: मूल उम्मेदवार-मात्र अवस्था {row['previous_status']}; विस्तारित अवस्था {row['status']}; उही निश्चित गन्तव्य {target_id}। पानी क्षेत्रभित्र: {row['inside_water_aoi']}; निकासी सन्दर्भभित्र: {row['inside_extraction_context']}। यो सर्तगत ग्राफ निष्कर्ष हो, वास्तविक पारगम्यता होइन।",
                f"/settlements/{i}/status;/settlements/{i}/previous_status;/settlements/{i}/nearest_baseline_destination",(row['name'],row['id'],"expanded","status","विस्तारित"),expanded_path)
            support=row.get('path_support')
            if support is None:
                support_en="No graph-path support record is available for this unmapped/baseline-disconnected row."
                support_ne="यो नक्सामा सम्बन्ध नभएको वा आधारभूत रूपमा नजोडिएको पङ्क्तिका लागि ग्राफ-मार्ग समर्थन अभिलेख उपलब्ध छैन।"
            elif support['projected_centerline_total_m']==0:
                support_en="The source and fixed target share a graph node: zero-length witness, with null support fractions, never 100% observed coverage. Physical point/facility connection is unvalidated."
                support_ne="स्रोत र निश्चित गन्तव्य एउटै ग्राफ नोडमा छन्: शून्य लम्बाइको साक्षीमा समर्थन अंश null हुन्छन्, 100% अवलोकन गरिएको कभरेज होइन। वास्तविक बिन्दु-सुविधा सम्बन्ध प्रमाणीकरण भएको छैन।"
            else:
                support_en="Projected centreline support of the expanded scenario's saved witness: "+" / ".join(f"{support['fractions_of_projected_centerline'][k]*100:.2f}% {k}" for k in keys)+f"; denominator {support['projected_centerline_total_m']:,.3f} m. This is source observation support, not passability."
                support_ne="विस्तारित अवस्थाको भण्डारित साक्षीको प्रक्षेपित केन्द्ररेखा समर्थन: "+" / ".join(f"{support['fractions_of_projected_centerline'][k]*100:.2f}% {k}" for k in keys)+f"; हर {support['projected_centerline_total_m']:,.3f} मिटर। यो स्रोत अवलोकन समर्थन हो, पारगम्यता होइन।"
            add(identifier+'_support',f"For {row['name']} ({row['id']}), {support_en}",f"{row['name']} ({row['id']}) का लागि {support_ne}",f"/settlements/{i}/path_support",(row['name'],row['id'],"expanded","support","coverage","length","कभरेज","लम्बाइ"),expanded_path)
        # Only this opt-in snapshot labels legacy graph clauses explicitly.
        # The original store/source files and the no-option service stay exact.
        for identifier,fact in list(self.facts.items()):
            if identifier.startswith(('access_','settlement_')):
                self.facts[identifier]=replace(fact,en="Original candidate-only scenario: "+fact.en,
                    ne="मूल उम्मेदवार-मात्र सर्तगत अवस्था: "+fact.ne)
        self.observation_receipt_path=receipt_path
        self.observation_summary_path=summary_path
        self._bind_record(self._helpers.record(Path(__file__)),ROOT)
        self.source_bindings=[self._bindings[k] for k in sorted(self._bindings)]
        self.verification+=" Optional observation metadata: both completed receipts and all exact new output hashes checked; row counts/scopes/fixed targets/support arithmetic cross-checked. New graph analysis is not rerun; no model/reference geometry loaded by this extension."

    def policy_reason(self, question):
        folded=_question(question)
        if any(term in folded for term in ("functioning hospital","hospital functioning","passable road","roads passable","सञ्चालनमा रहेको अस्पताल")):
            return "no_observed_damage_passability_or_operational_safety"
        return super().policy_reason(question)

    def ranked_facts(self,question,limit=8):
        folded=_question(question)
        if not hasattr(self,"observation_receipt_path") or self.policy_reason(question):
            return super().ranked_facts(question,limit)
        if isinstance(limit,bool) or not isinstance(limit,int) or not 1<=limit<=16:raise ValueError("Retrieval limit must be 1–16")
        named=self._named_settlements(question)
        if not any(t in folded for t in OBSERVATION_TERMS):
            legacy=super().ranked_facts(question,limit)
            if any(f.id.startswith(('access_','settlement_')) for f in legacy):
                compare=[self.facts[self.observation_settlement_ids[sid]] for sid in named] if named else [self.facts['newscenario_counts']]
                legacy=compare+legacy+[self.facts['newscenario_assumptions']]
            return list({f.id:f for f in legacy}.values())
        selected=[self.facts[self.observation_settlement_ids[sid]] for sid in named]
        if not named:selected.append(self.facts['newscenario_counts'])
        if any(x in folded for x in ('coverage','support','length','fraction','witness','unsupported','कभरेज','लम्बाइ','अंश')):
            selected.extend(self.facts[self.observation_settlement_ids[sid]+'_support'] for sid in named)
            if not named:selected.append(self.facts['newscenario_witness_coverage'])
        if any(x in folded for x in ('inside','outside','aoi','context','scope','भित्र','बाहिर','दायरा')):selected.append(self.facts['newscenario_scope'])
        if any(x in folded for x in ('empty','zero-length','null','खाली','शून्य')):selected.append(self.facts['newscenario_empty_route'])
        selected=list({f.id:f for f in selected}.values())[:limit]
        return selected+[self.facts['newscenario_assumptions'],self.facts['scope']]

    retrieve=ranked_facts

    def resolve_model_plan(self, plan, *, language, candidate_ids):
        result = super().resolve_model_plan(plan, language=language, candidate_ids=candidate_ids)
        named = {key for key in candidate_ids if key.startswith("newscenario_settlement_")}
        if result["status"] == "answer" and named and not set(plan["fact_ids"]) & named:
            raise ValueError("A named observation answer requires its retrieved source fact")
        if hasattr(self, "observation_receipt_path"):
            result["observation_scenario_receipt"] = str(self.observation_receipt_path)
            result["observation_claim_scope"] = "Saved conditional graph scenario and source observation support; no observed closure/passability or operating facility"
        return result

    resolve=resolve_model_plan
