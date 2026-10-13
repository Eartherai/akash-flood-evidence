"""Export the pinned, small saved Rasuwa report/evidence viewer; no inference.

Default is an offline dry run. --execute writes a new deterministic ZIP and a
separate creation receipt. It never rewrites scientific source files or follows
paths inside their original JSON manifests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from serve_saved_case import (ALLOWED_FILES, CASE_ID, FALSE_FLAGS, SCHEMA, SOURCE_PATHS,
                             checked_path, MAX_TOTAL_BYTES)


ROOT = Path(__file__).resolve().parents[1]
ZIP_ROOT = "akash-saved-case-replay-2026-10-07"
ZIP_TIMESTAMP = (2026, 10, 7, 0, 0, 0)
SOURCE_FILES = (
    ('outputs/fresh-report-attributed/access-proof.json', 'report/access-proof.json', 2639642, 'fdf540c55e54612b84dd5003ff47781d73aca4757ed9626d92b9994a9b196737'),
    ('outputs/fresh-report-attributed/map.png', 'report/map.png', 211767, '8f54c5b2cc69fdd90b82b6bb1a32f69829eaa1d89e112610d589ae5b58db75fc'),
    ('outputs/fresh-report-attributed/optical-manifest.json', 'report/optical-manifest.json', 28621, 'ce6c33f9324a6df189b4c5b465e1c871454db60d834d0480281215ab1865e525'),
    ('outputs/fresh-report-attributed/optical-overview.png', 'report/optical-overview.png', 19338, '9850fe6ee63bc4724c775165845624a640830bb6cbba8f143e897a57b55e7d4f'),
    ('outputs/fresh-report-attributed/optical-protocol.json', 'report/optical-protocol.json', 10112, '8d91b89edcbff5e7fe874d3e9a055e35462ab697d4c4af96916194108601dc80'),
    ('outputs/fresh-report-attributed/report.html', 'report/report.html', 2417131, '81020868919b73277ed62b1a0f3fa8fc2f39858708ac8f40a3422e30802cacd7'),
    ('outputs/fresh-report-attributed/report.json', 'report/report.json', 2608167, '6a623d9ec3a3e99e60375e2982df75cad4bfbf8edc76c7f0485ecb11e2eb671f'),
    ('outputs/fresh-report-attributed/source-and-output-manifest.json', 'report/source-and-output-manifest.json', 4262, 'a96011900e460ff9ec8d51c34cc355e88e9080d7a5c7976b545cdf7a9c7db826'),
    ('output/pdf/akash-technical-evidence-report.pdf', 'documents/akash-technical-evidence-report.pdf', 17840, '1f53a06a1fc245795eaefcd5a58e63f9c2f74fb4fadbbaa2693675cf36246340'),
    ('output/pdf/akash-technical-evidence-report.provenance.json', 'documents/akash-technical-evidence-report.provenance.json', 6345, '8d3829e239078ee2890eff004c9582e54b3673a5263fc02fb31e6d214ae371e4'),
    ('output/pdf/akash-attributed-situation-report.pdf', 'documents/akash-attributed-situation-report.pdf', 269656, '86fe0613edeaac0ba060481032ebedba3e94da25435b25476389d25326e0a0d4'),
    ('output/pdf/akash-attributed-situation-report.provenance.json', 'documents/akash-attributed-situation-report.provenance.json', 956, 'f9eb22f2cfdc025e64bf1199ae5faf45e7ffa544606695722301dcb45d9c7eff'),
    ('outputs/fresh-one-command-replay/execution-receipt.json', 'evidence/execution-receipt.json', 22541, 'e5e27e20e557d4ae13d36ea0e8552364b1515f1110457f7534d0174912730824'),
    ('outputs/fresh-one-command-replay/reproducibility-audit.json', 'evidence/reproducibility-audit.json', 898, '7b236e31d7c2fc39379ad9075fe870e538f274427c01fa9681ff8bf453184d5e'),
    ('outputs/fresh-one-command-replay/water/manifest.json', 'evidence/water/manifest.json', 6011, 'df63f70e87c746d1f50fb0cade62c2ee088860b78907f69f7b98a1d863dbdeb5'),
    ('outputs/fresh-one-command-replay/water/policy.json', 'evidence/water/policy.json', 1248, 'b966abc263a8e05aa5c9569749b9d05762e4960da12300ea7e554c1211271027'),
    ('outputs/fresh-one-command-replay/processing/run-metadata.json', 'evidence/processing/run-metadata.json', 34125, '30848a2de5d7941b0f32432c957bde3d657622a60c6a3d98d7db45c89cdcb187'),
    ('outputs/fresh-one-command-replay/exposure/summary.json', 'evidence/exposure/summary.json', 8441, 'c274390cf35554bfb84c2efd42bb153335473b815ca1de8f99847f2ea69dc032'),
    ('outputs/fresh-one-command-replay/exposure/scenario-policy.json', 'evidence/exposure/scenario-policy.json', 1646, 'ce88e9e3b8e8a0bda5b23eda4ae84e92d3eb50ceda4f9d2bc71904fcd9d8a12a'),
    ('outputs/fresh-one-command-replay/exposure/source-and-output-manifest.json', 'evidence/exposure/source-and-output-manifest.json', 10751, '5009053814cb1e4e218c911489d7ceb3b379745be8cedfdc03177bc39f031e38'),
    ('data/rasuwa-pilot/osm.json', 'source/osm.json', 7424485, '9f96f2e0f090ee6f235011b2f3be8836a6e585fc877bc1c73d9f848f434240b0'),
    ('data/rasuwa-pilot/osm.provenance.json', 'source/osm.provenance.json', 810, '3fbf66db2ff0de23d1f15727e399d1b8235caed04735a050e0bb7ce22577aa60'),
)

README = """# Akash saved-case replay - 7 October 2026

This is an offline report and evidence viewer for the saved Rasuwa case
fresh-d22bcbfb6d7bc937. It uses Python 3.11 or newer and its standard library.
No package installation, GPU, satellite download or AI model is needed.

Extract the ZIP, then open a terminal inside its extracted top-level directory:

    python3 serve_saved_case.py --verify-only
    python3 serve_saved_case.py --port 8790

Open http://127.0.0.1:8790/report/report.html in a browser. Select a settlement
to display its saved route/cut, or use the report's source download links.
The viewer binds only 127.0.0.1. Stop it with Control-C. An occupied port can
be changed with --port. No network requests are made by the viewer or report;
your browser makes only localhost requests to the viewer.

Files in documents/ are the six-page technical report, one-page situation
report and their visual-review provenance. Technical checking results are
review material only; this viewer has no fact retrieval or AI endpoint.
The case facts/proof are in report/, original source/stage records in evidence/,
and the pre-event OSM extract plus original query receipt in source/.

The new manifest.json uses relative paths, sizes and SHA256 hashes.
manifest.sha256 verifies the manifest, and all 25 payload files are checked
before serving; every served file is checked again. Original scientific JSON
copies preserve their bytes and original absolute paths as historical labels.
Those old paths are not executed or resolved by this viewer. Dependencies
named in the original records are not all included in this small export.

For archive authenticity, compare the ZIP SHA256 with the separately supplied
export receipt or published checksum. The internal manifest detects changed
bytes against that release; it is not a digital signature or independent
verification that the original scientific conclusions are correct.

This is saved_case_replay. It performs NO raw pipeline replay, model inference,
training, new graph analysis or bilingual copilot calls. The separately built
case-aware copilot still requires its full verified source case and the real
optional Qwen/llama.cpp runtime. No successful AI call is implied here.

The included new/lost water-state and optical surface-change classifications
are provisional. Local flood/debris accuracy is unvalidated. The original
optical gate failed and most optical area remains unknown. Candidate road
states are hypothetical; all other baseline edges are assumed available.
Routes/cuts do not prove observed closure, damage, facility operation or safety.
The export does not establish full Track B compliance, fresh-machine raw
processing, a complete scientific source-chain rerun or a winning submission.

The original historical OSM database is included as a source offer; missing
relations and map completeness remain unresolved. Original data copyrights
and ODbL apply. Read NOTICES.txt. New project code licensing and inherited
asset redistribution permissions remain to be finalized before publication;
this locally reviewable bundle is not evidence of public release authorization.
"""

NOTICES = """Akash saved-case evidence export

Contains modified Copernicus Sentinel data 2026.
Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence
and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA;
all rights reserved.
© OpenStreetMap contributors. OSM data is available under the Open Database
Licence (ODbL) 1.0: https://www.openstreetmap.org/copyright and
https://opendatacommons.org/licenses/odbl/1-0/ . The original pre-event OSM
extract and query provenance are included under source/.

Training source actually used in the original water model:
Derrick Bonafilia, Beth Tellman, Tyler Anderson and Erica Issenberg.
Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood
Algorithms for Sentinel-1. CVPR Workshops, 2020.
https://openaccess.thecvf.com/content_CVPRW_2020/html/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.html
The organizer permitted this training dataset. No training imagery, labels or
model weights are included; no blanket upstream redistribution grant is claimed.

The original scientific workflow used ESA SNAP and an explicitly reviewed
local DEM sampler correction plus an NGA EGM2008 vertical-reference grid.
No SNAP runtime, vendor jars, DEM/geoid payload or correction runtime is shipped.
Technical report checking statistics credit European Union, Copernicus Emergency
Management Service data. No EMS/CEMS/UNOSAT vectors, checking pixels or labels
are included. The technical PDF is separate review material, not AI facts.

Qwen and llama.cpp are not bundled or invoked. Their licenses apply separately
to the optional copilot runtime. This viewer uses the Python standard library.
AI assistance was used for research, code and documentation; the AI is not a
human teammate. No invented accuracy, observed damage or winning guarantee.

The new project code license and inherited-asset redistribution decisions are
not finalized. Preserve source copyrights and notices; do not infer a blanket
permissive license from this bundle or the user's archive license field.
"""


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def collect_sources(root: Path, expected=SOURCE_FILES):
    root = Path(root).resolve(strict=True)
    if len(expected) != 22 or {row[1] for row in expected} != set(SOURCE_PATHS):
        raise ValueError("Require the fixed 22-file source allowlist")
    payload, rows = {}, []
    for source, target, size, digest in expected:
        path = checked_path(root, source)
        if path.stat().st_size != size:
            raise ValueError("Pinned source size mismatch: " + source)
        content = path.read_bytes()
        if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Pinned source checksum mismatch: " + source)
        if target in payload:
            raise ValueError("Duplicate target path")
        payload[target] = content
        rows.append({"path": target, "bytes": size, "sha256": digest,
                     "role": "preserved_source_artifact", "original_source_path": str(path)})
    return payload, rows


def prepare_payload(root=ROOT, expected=SOURCE_FILES):
    payload, rows = collect_sources(root, expected)
    source_bytes = sum(row["bytes"] for row in rows)
    generated = {"README.md": README.encode("utf-8"), "NOTICES.txt": NOTICES.encode("utf-8"),
                 "serve_saved_case.py": Path(__file__).with_name("serve_saved_case.py").read_bytes()}
    for name, content in generated.items():
        payload[name] = content
        rows.append({"path": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                     "role": "viewer_documentation"})
    if set(payload) != ALLOWED_FILES or sum(len(v) for v in payload.values()) > MAX_TOTAL_BYTES:
        raise ValueError("Unexpected saved-case payload")
    document = {"schema": SCHEMA, "mode": "saved_case_replay", "case_id": CASE_ID,
                "release_date": "2026-10-07", "zip_root": ZIP_ROOT,
                "deterministic_zip_timestamp": list(ZIP_TIMESTAMP),
                "source_file_count": 22, "source_payload_bytes": source_bytes,
                "payload_bytes": sum(len(v) for v in payload.values()),
                **{flag: False for flag in FALSE_FLAGS},
                "full_track_b_requirements_satisfied": False,
                "scientific_receipts_modified": False,
                "source_chain_dependencies_all_included": False,
                "network_requests_by_exporter_or_viewer": 0,
                "license_status": "new code license and inherited redistribution decisions pending; local review only",
                "checking_material_policy": "technical PDF for separate review only; no AI or inference input",
                "files": sorted(rows, key=lambda row: row["path"])}
    payload["manifest.json"] = json_bytes(document)
    manifest_hash = hashlib.sha256(payload["manifest.json"]).hexdigest()
    payload["manifest.sha256"] = (manifest_hash + "  manifest.json\n").encode("ascii")
    return payload, document, manifest_hash


def write_archive(output: Path, payload):
    output = Path(output).absolute()
    if output.suffix != ".zip" or output.exists() or output.is_symlink():
        raise FileExistsError("Use a fresh .zip output path; preserve earlier archives")
    output.parent.mkdir(parents=True, exist_ok=True)
    # No extraction, filesystem traversal or arbitrary repository glob is used.
    with output.open("xb") as stream:
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name, content in sorted(payload.items()):
                info = zipfile.ZipInfo(ZIP_ROOT + "/" + name, date_time=ZIP_TIMESTAMP)
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return output


def export(root, output, execute=False, expected=SOURCE_FILES):
    output = Path(output).absolute()
    receipt_path = output.with_name(output.name + ".provenance.json")
    if output.exists() or receipt_path.exists() or output.is_symlink() or receipt_path.is_symlink():
        raise FileExistsError("Use fresh archive and receipt paths")
    payload, document, manifest_hash = prepare_payload(root, expected)
    result = {"mode": "saved_case_replay", "status": "dry_run", "execute": execute,
              "case_id": CASE_ID, "source_file_count": 22, "archive_file_count": len(payload),
              "source_payload_bytes": document["source_payload_bytes"],
              "payload_bytes": document["payload_bytes"], "manifest_sha256": manifest_hash,
              **{flag: False for flag in FALSE_FLAGS}, "network_requests": 0,
              "scientific_receipts_modified": False, "output": str(output)}
    if not execute:
        return result
    path = write_archive(output, payload)
    with path.open("rb") as stream:
        archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    result.update(status="exported", created_at=datetime.now(timezone.utc).isoformat(),
                  zip_bytes=path.stat().st_size, zip_sha256=archive_hash,
                  zip_timestamp=list(ZIP_TIMESTAMP), file_manifest=document["files"],
                  exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with receipt_path.open("xb") as stream:
        stream.write(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "output/release" / (ZIP_ROOT + ".zip"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    result = export(args.root, args.out, args.execute)
    print(json.dumps({k: v for k, v in result.items() if k != "file_manifest"}, indent=2))


if __name__ == "__main__":
    main()
