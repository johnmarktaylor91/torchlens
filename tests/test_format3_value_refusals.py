"""The 3.0 writer's value refusals: what refuses typed and what is disclosed instead.

The 2.x save path dry-ran ``metadata.pkl`` through the default-deny unpickler and
refused an unportable annotation (``annotation_value_unportable``). The 3.0 writer
has no pickle: a value with no typed encoding is stringified with a disclosure
(manifest ``disclosures``), and only a value JSON cannot hold at all, or a tensor
with no supported payload codec, refuses with ``metadata_value_unportable``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest
import torch

from torchlens._io import ArtifactError
from torchlens._io.format3.decode import decode_value
from torchlens._io.format3.payloads import ChunkWriter
from torchlens._io.format3.values import ValueEncoder
from torchlens.types import UnencodableValue


def test_self_referencing_value_refuses_typed() -> None:
    """A container that contains itself has no JSON form; the encoder refuses it."""

    loop: list[object] = [1]
    loop.append(loop)
    with pytest.raises(ArtifactError, match="self-referencing") as caught:
        ValueEncoder().encode(loop, "trace[0].annotations")
    assert caught.value.fields["code"] == "metadata_value_unportable"


def test_unknown_payload_codec_refuses_typed(tmp_path: Path) -> None:
    """A chunk writer for a codec outside the 3.0 codec list refuses before writing."""

    with pytest.raises(ArtifactError, match="Unknown payload codec") as caught:
        ChunkWriter(tmp_path, codec="pickle_v0")
    assert caught.value.fields["code"] == "metadata_value_unportable"
    assert not any(tmp_path.iterdir())


def test_value_with_no_typed_encoding_is_disclosed_not_refused() -> None:
    """An opaque object is stringified with a disclosure naming its position."""

    class Opaque:
        def __repr__(self) -> str:
            return "Opaque()"

    encoder = ValueEncoder()
    encoded = encoder.encode({"k": Opaque()}, "trace[0].annotations")
    assert encoder.disclosures, encoded
    disclosure = encoder.disclosures[0]
    assert disclosure["repr"] == "Opaque()"
    assert disclosure["at"].startswith("trace[0].annotations")


def test_unencodable_value_loads_back_as_the_public_stand_in() -> None:
    """The decoder returns ``tl.types.UnencodableValue`` in place of the value, never a dict."""

    class Opaque:
        def __repr__(self) -> str:
            return "Opaque()"

    encoded = ValueEncoder().encode({"k": Opaque()}, "op[3].func_config")
    stand_in = decode_value(encoded, at="op[3].func_config")["k"]
    assert isinstance(stand_in, UnencodableValue)
    assert stand_in.repr_text == "Opaque()"
    assert stand_in.type_name.endswith("Opaque")
    assert stand_in.at == "op[3].func_config['k']"
    assert hash(stand_in) == hash(UnencodableValue(stand_in.type_name, "Opaque()", stand_in.at))


@pytest.mark.parametrize(
    "cell",
    [
        {"$tl": "no_such_tag"},
        {"$tl": "tuple"},
        {"$tl": "tuple", "items": 3},
        {"$tl": "dtype", "name": "float99"},
        {"$tl": "tuple", "items": [], "subtype": "unlisted.Tuple"},
        {"$tl": "tuple", "items": ["bad-dimension"], "subtype": "torch.Size"},
        {"$tl": "dict", "items": [], "subtype": "unlisted.Mapping"},
        {
            "$tl": "dict",
            "items": [],
            "subtype": "collections.defaultdict",
            "default_factory": "eval",
        },
    ],
)
def test_cell_failing_its_type_check_refuses_record_schema_invalid(cell: dict) -> None:
    """A saved cell off the typed vocabulary refuses with its own code (ruled 2026-11-02)."""

    with pytest.raises(ArtifactError) as caught:
        decode_value(cell, at="op[0].func_config")
    assert caught.value.fields["code"] == "record_schema_invalid"
    assert "op[0].func_config" in str(caught.value)


@pytest.mark.parametrize("factory", [list, dict, set, int, float, str, tuple])
def test_builtin_defaultdict_factory_has_a_closed_wire_identity(factory: type) -> None:
    """Every allowed factory is data in the carrier; encoding never calls it."""

    encoder = ValueEncoder()
    cell = encoder.encode(defaultdict(factory, {"seen": 3}), "op[0].func_config")
    assert cell == {
        "$tl": "dict",
        "subtype": "collections.defaultdict",
        "default_factory": factory.__name__,
        "items": [["seen", 3]],
    }
    assert not encoder.disclosures


def test_foreign_defaultdict_factory_is_disclosed_without_losing_contents() -> None:
    """A foreign factory is never called or serialized as an executable reference."""

    calls = []

    def factory() -> list:
        calls.append(True)
        return []

    encoder = ValueEncoder()
    cell = encoder.encode(defaultdict(factory, {"seen": 3}), "op[0].func_config")
    assert cell["items"] == [["seen", 3]]
    assert cell["default_factory"]["$tl"] == "unencodable"
    assert cell["default_factory"]["at"] == "op[0].func_config.default_factory"
    assert len(encoder.disclosures) == 1
    assert not calls


@pytest.mark.parametrize("value", [torch.Size([2, 3]), torch.return_types.max((2, 3))])
def test_known_tuple_subtype_retains_its_closed_identity(value: tuple) -> None:
    """Size and result tuples preserve their constructor identity and all members."""

    cell = ValueEncoder().encode(value, "op[0].func_config")
    assert cell["$tl"] == "tuple"
    assert cell["items"] == [2, 3]
    assert cell["subtype"] == f"{type(value).__module__}.{type(value).__qualname__}"


@pytest.mark.parametrize("text", ["Opaque()", "x" * 300])
def test_loaded_unencodable_value_resaves_identically(text: str) -> None:
    """A loaded disclosure keeps its original type, bounded text, full length and location."""

    class Opaque:
        def __repr__(self) -> str:
            return text

    at = "trace.annotations['opaque']"
    first_encoder = ValueEncoder()
    first = first_encoder.encode(Opaque(), at)
    loaded = decode_value(first, at=at)
    again_encoder = ValueEncoder()
    assert again_encoder.encode(loaded, at) == first
    assert again_encoder.disclosures == first_encoder.disclosures
