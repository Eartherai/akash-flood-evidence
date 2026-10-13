"""Build a narrowly scoped SNAP 14.0.0 external-DEM center correction.

The original vendor jars and the geodetically valid DEM are never changed.
Only FileElevationModel.getElevation's interface method reference is changed
from computeCornerBasedIndex to computeIndex. The vendor reader already maps
geographic sample centers to image coordinates (column+.5,row+.5).

This is a local GPL-3.0-or-later modification, not an ESA-validated release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
from zipfile import ZipFile, ZipInfo, ZIP_STORED

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAME = "org/esa/snap/dem/dataio/FileElevationModel.class"
VENDOR_JAR_SHA = "f9236856773277ae42b12baa0bf2149c064d7945bb6d9c3c0440adf17da41232"
VENDOR_CLASS_SHA = "e4e221a7d7c74052f061042db9bdd8e9230cc3764b0e5ac6976115dfea26305a"
UPSTREAM_COMMIT = "059c4901869eacd2136590a1c21530abbd31d2f3"
UPSTREAM_URL = f"https://github.com/senbox-org/snap-engine/tree/{UPSTREAM_COMMIT}"
OWNER = "org/esa/snap/core/dataop/resamp/Resampling"
DESCRIPTOR = "(DDIILorg/esa/snap/core/dataop/resamp/Resampling$Index;)V"
OLD_NAME, NEW_NAME = "computeCornerBasedIndex", "computeIndex"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    p = Path(path)
    return {"path": str(p.resolve()), "bytes": p.stat().st_size, "sha256": digest(p.read_bytes())}


def constant_pool(data):
    """Read classfile constants without executing or changing vendor code."""
    if data[:4] != b"\xca\xfe\xba\xbe":
        raise ValueError("Not a Java class file")
    count = struct.unpack_from(">H", data, 8)[0]
    entries, offsets, pos, i = {}, {}, 10, 1
    sizes = {3: 4, 4: 4, 5: 8, 6: 8, 7: 2, 8: 2, 9: 4, 10: 4, 11: 4,
             12: 4, 15: 3, 16: 2, 17: 4, 18: 4, 19: 2, 20: 2}
    while i < count:
        start = pos
        tag = data[pos]; pos += 1
        if tag == 1:
            length = struct.unpack_from(">H", data, pos)[0]; pos += 2
            value = data[pos:pos + length].decode("utf-8"); pos += length
        elif tag in (7, 8, 16, 19, 20):
            value = struct.unpack_from(">H", data, pos)[0]; pos += 2
        elif tag in (9, 10, 11, 12, 17, 18):
            value = struct.unpack_from(">HH", data, pos); pos += 4
        elif tag in sizes:
            value = data[pos:pos + sizes[tag]]; pos += sizes[tag]
        else:
            raise ValueError("Unsupported Java constant tag")
        if pos > len(data):
            raise ValueError("Truncated class file")
        entries[i] = (tag, value); offsets[i] = (start, pos)
        i += 2 if tag in (5, 6) else 1
    return entries, offsets, pos


def utf(cp, index):
    tag, value = cp[index]
    if tag != 1:
        raise ValueError("Expected UTF8 constant")
    return value


def methods(data, cp, pos):
    """Return method bytecode so the patch's call site is independently bound."""
    def u2():
        nonlocal pos
        v = struct.unpack_from(">H", data, pos)[0]; pos += 2
        return v

    def attributes():
        nonlocal pos
        result = {}
        for _ in range(u2()):
            name = utf(cp, u2())
            size = struct.unpack_from(">I", data, pos)[0]; pos += 4
            result[name] = data[pos:pos + size]; pos += size
        return result

    pos += 6  # access flags, this class, superclass
    interface_count = u2()
    pos += interface_count * 2
    for _ in range(u2()):
        pos += 6; attributes()
    result = []
    for _ in range(u2()):
        u2(); name, descriptor = utf(cp, u2()), utf(cp, u2())
        attrs = attributes()
        code = attrs.get("Code")
        if code:
            length = struct.unpack_from(">I", code, 4)[0]
            result.append((name, descriptor, code[8:8 + length]))
    return result


def instructions(code):
    """Iterate exact bytecode instruction boundaries, including switches."""
    two = {0x10, 0x12, *range(0x15, 0x1a), *range(0x36, 0x3b), 0xa9, 0xbc}
    three = {0x11, 0x13, 0x14, 0x84, *range(0x99, 0xa9), *range(0xb2, 0xb9),
             0xbb, 0xbd, 0xc0, 0xc1, 0xc6, 0xc7}
    pos = 0
    while pos < len(code):
        start = pos; opcode = code[pos]
        if opcode in two:
            size = 2
        elif opcode in three:
            size = 3
        elif opcode in (0xb9, 0xba, 0xc8, 0xc9):
            size = 5
        elif opcode == 0xc5:
            size = 4
        elif opcode == 0xc4:
            size = 6 if code[pos + 1] == 0x84 else 4
        elif opcode in (0xaa, 0xab):
            p = (pos + 4) & ~3
            if opcode == 0xaa:
                low, high = struct.unpack_from(">ii", code, p + 4)
                size = p - pos + 12 + 4 * (high - low + 1)
            else:
                n = struct.unpack_from(">i", code, p + 4)[0]
                size = p - pos + 8 + 8 * n
        else:
            size = 1
        if size < 1 or pos + size > len(code):
            raise ValueError("Truncated bytecode instruction")
        yield start, opcode, code[pos:pos + size]
        pos += size


def patch_class(data, expected_sha=VENDOR_CLASS_SHA):
    if digest(data) != expected_sha:
        raise ValueError("Vendor class checksum changed; no speculative patch")
    cp, offsets, end = constant_pool(data)
    old_indices = [i for i, (tag, value) in cp.items() if tag == 1 and value == OLD_NAME]
    if len(old_indices) != 1:
        raise ValueError("Require exactly one old method-name constant")
    old = old_indices[0]
    nats = [i for i, (tag, value) in cp.items()
            if tag == 12 and value[0] == old and utf(cp, value[1]) == DESCRIPTOR]
    if len(nats) != 1:
        raise ValueError("Require exactly one expected NameAndType")
    refs = [i for i, (tag, value) in cp.items() if tag == 11 and value[1] == nats[0]
            and cp[value[0]][0] == 7 and utf(cp, cp[value[0]][1]) == OWNER]
    if len(refs) != 1:
        raise ValueError("Require exactly one Resampling interface reference")
    sites = []
    for name, descriptor, code in methods(data, cp, end):
        for position, opcode, encoded in instructions(code):
            if opcode == 0xb9 and struct.unpack_from(">H", encoded, 1)[0] == refs[0]:
                sites.append({"method": name, "descriptor": descriptor, "bytecode_offset": position})
    if len(sites) != 1 or sites[0]["method"] != "getElevation" or sites[0]["descriptor"] != "(Lorg/esa/snap/core/datamodel/GeoPos;)D":
        raise ValueError("Require one getElevation call site; no wider patch")
    a, b = offsets[old]
    new_utf = NEW_NAME.encode("utf-8")
    patched = data[:a] + bytes([1]) + struct.pack(">H", len(new_utf)) + new_utf + data[b:]
    new_cp, _, new_end = constant_pool(patched)
    if methods(data, cp, end) != methods(patched, new_cp, new_end):
        raise ValueError("Patch changed method bodies")
    if len(cp) != len(new_cp) or sum(cp[i] != new_cp[i] for i in cp) != 1:
        raise ValueError("Patch changed more than one constant")
    return patched, {"class": CLASS_NAME, "interface_owner": OWNER, "old_method": OLD_NAME,
                     "new_method": NEW_NAME, "interface_descriptor": DESCRIPTOR,
                     "call_sites": sites, "changed_constant_count": 1, "method_bodies_unchanged": True}


def build(vendor, license_path, source_path, out):
    vendor, license_path, source_path, out = map(Path, (vendor, license_path, source_path, out))
    if out.exists():
        raise FileExistsError("Preserve earlier correction artifacts; use a fresh directory")
    if digest(vendor.read_bytes()) != VENDOR_JAR_SHA:
        raise ValueError("Vendor SNAP 14.0.0 DEM jar checksum changed")
    with ZipFile(vendor) as z:
        original = z.read(CLASS_NAME)
    corrected, scope = patch_class(original)
    source = source_path.read_text()
    old = f"resampling.{OLD_NAME}(pix.x, pix.y, RASTER_WIDTH, RASTER_HEIGHT, newIndex);"
    if source.count(old) != 1 or "GNU General Public License" not in source:
        raise ValueError("Require corresponding attributed SNAP 14.0.0 source")
    patched_source = source.replace(old, f"resampling.{NEW_NAME}(pix.x, pix.y, RASTER_WIDTH, RASTER_HEIGHT, newIndex);")
    out.mkdir(parents=True)
    (out / "FileElevationModel.vendor.java").write_text(source)
    (out / "FileElevationModel.java").write_text(patched_source)
    license_data = license_path.read_bytes()
    (out / "COPYING.txt").write_bytes(license_data)
    notice = ("Local modification to SNAP 14.0.0 FileElevationModel\n"
              "Copyright (C) 2014 Array Systems Computing Inc.; GPL-3.0-or-later.\n"
              "Upstream corresponding source: " + UPSTREAM_URL + "\n"
              "Only external DEM sampling call computeCornerBasedIndex is changed to computeIndex.\n"
              "The input geographic sample locations and elevation values are unchanged.\n"
              "This is a local research correction; no ESA validation or endorsement is claimed.\n"
              "Prepend this jar only for explicitly recorded corrected runs.\n")
    (out / "NOTICE.txt").write_text(notice)
    jar = out / "external-dem-center-overlay.jar"
    with ZipFile(jar, "w", compression=ZIP_STORED) as z:
        for name, data in [(CLASS_NAME, corrected), ("META-INF/NOTICE.txt", notice.encode()),
                           ("META-INF/COPYING.txt", license_data),
                           ("source/org/esa/snap/dem/dataio/FileElevationModel.java", patched_source.encode())]:
            info = ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = ZIP_STORED
            z.writestr(info, data)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "status": "prepared; independent JVM proof required",
              "vendor_jar": record(vendor), "vendor_class_sha256": digest(original),
              "patched_class_sha256": digest(corrected), "overlay_jar": record(jar),
              "patch_builder": record(Path(__file__)), "scope": scope,
              "upstream_corresponding_source_commit": UPSTREAM_COMMIT, "upstream_source_url": UPSTREAM_URL,
              "sources": [record(out / "FileElevationModel.vendor.java"), record(out / "FileElevationModel.java")],
              "license": "GPL-3.0-or-later", "license_file": record(out / "COPYING.txt"),
              "original_runtime_modified": False, "DEM_values_or_geographic_centers_modified": False,
              "reason": "GeoTIFF reader yields center-based image coordinates; installed corner-based helper adds a second half pixel",
              "limitations": ["Local modification is not an official SNAP release or ESA-validated patch",
                              "Output sampling still requires independent comparison at nominal output centers",
                              "EGM96 object may still initialize as a runtime dependency, although correction is disabled",
                              "No hazard classification or absolute satellite geolocation accuracy is established"]}
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-jar", type=Path, default=ROOT / "data/snap-runtime/portable/snap/modules/org-esa-snap-snap-dem.jar")
    parser.add_argument("--license", type=Path, default=ROOT / "data/snap-runtime/portable/LICENSE.txt")
    parser.add_argument("--source", type=Path, default=ROOT / "research/copdem-review-evidence/FileElevationModel-14.0.0.java")
    parser.add_argument("--out", type=Path, default=ROOT / "data/snap-external-dem-pixel-correction")
    args = parser.parse_args()
    print(json.dumps(build(args.vendor_jar, args.license, args.source, args.out), indent=2))
