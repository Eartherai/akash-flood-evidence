from pathlib import Path
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_snap_external_dem as patch


def fixture(*, method="getElevation", owner=patch.OWNER, duplicate_name=False, calls=1,
            method_descriptor="(Lorg/esa/snap/core/datamodel/GeoPos;)D"):
    # Minimal structural class fixture. Runtime semantics are checked separately
    # by the actual vendor/corrected JVMs on independent analytic raster values.
    constants = [
        (1, "org/esa/snap/dem/dataio/FileElevationModel"), (7, 1),
        (1, "java/lang/Object"), (7, 3), (1, method), (1, method_descriptor),
        (1, "Code"), (1, owner), (7, 8), (1, patch.OLD_NAME),
        (1, patch.DESCRIPTOR), (12, (10, 11)), (11, (9, 12)),
    ]
    if duplicate_name:
        constants.append((1, patch.OLD_NAME))
    data = b"\xca\xfe\xba\xbe" + struct.pack(">HHH", 0, 65, len(constants) + 1)
    for tag, value in constants:
        data += bytes([tag])
        if tag == 1:
            value = value.encode(); data += struct.pack(">H", len(value)) + value
        elif tag == 7:
            data += struct.pack(">H", value)
        else:
            data += struct.pack(">HH", *value)
    code = b"\xb9\x00\x0d\x08\x00" * calls + b"\xaf"
    attribute = struct.pack(">HHI", 8, 2, len(code)) + code + struct.pack(">HH", 0, 0)
    data += struct.pack(">HHHHH", 0x21, 2, 4, 0, 0)  # flags, class, super, interfaces, fields
    # One public method and one Code attribute.
    data += struct.pack(">HHHHH", 1, 1, 5, 6, 1)
    data += struct.pack(">HI", 7, len(attribute)) + attribute + struct.pack(">H", 0)
    return data


def test_narrow_patch_changes_one_name_but_preserves_every_method_body():
    original = fixture()
    corrected, scope = patch.patch_class(original, patch.digest(original))
    before, _, before_end = patch.constant_pool(original)
    after, _, after_end = patch.constant_pool(corrected)
    assert before[10] == (1, patch.OLD_NAME)
    assert after[10] == (1, patch.NEW_NAME)
    assert sum(before[i] != after[i] for i in before) == 1
    assert patch.methods(original, before, before_end) == patch.methods(corrected, after, after_end)
    assert scope["call_sites"] == [{"method": "getElevation", "descriptor": "(Lorg/esa/snap/core/datamodel/GeoPos;)D", "bytecode_offset": 0}]


def test_patch_refuses_any_unpinned_vendor_class():
    with pytest.raises(ValueError, match="checksum"):
        patch.patch_class(fixture())


@pytest.mark.parametrize("settings, message", [
    ({"duplicate_name": True}, "one old method-name"),
    ({"owner": "other/Resampling"}, "Resampling interface reference"),
    ({"calls": 2}, "one getElevation call site"),
    ({"calls": 0}, "one getElevation call site"),
    ({"method": "otherMethod"}, "one getElevation call site"),
    ({"method_descriptor": "()D"}, "one getElevation call site"),
])
def test_patch_refuses_wider_or_wrong_operator_scope(settings, message):
    data = fixture(**settings)
    with pytest.raises(ValueError, match=message):
        patch.patch_class(data, patch.digest(data))


def test_builder_never_overwrites_prior_evidence(tmp_path):
    with pytest.raises(FileExistsError, match="fresh directory"):
        patch.build(tmp_path / "missing.jar", tmp_path / "missing-license", tmp_path / "missing.java", tmp_path)


def test_bytecode_parser_does_not_find_invocations_inside_constant_operands():
    code = b"\x11\xb9\x00\x10\x0d\xb1"  # sipush operand contains invokeinterface byte
    assert [(p, op) for p, op, _ in patch.instructions(code)] == [(0, 0x11), (3, 0x10), (5, 0xb1)]
