"""The 3.0 writer's value refusals: what refuses typed and what is disclosed instead.

The 2.x save path dry-ran ``metadata.pkl`` through the default-deny unpickler and
refused an unportable annotation (``annotation_value_unportable``). The 3.0 writer
has no pickle: a value with no typed encoding is stringified with a disclosure
(manifest ``disclosures``), and only a value JSON cannot hold at all, or a tensor
with no supported payload codec, refuses with ``metadata_value_unportable``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from torchlens._io import ArtifactError
from torchlens._io.format3.payloads import ChunkWriter
from torchlens._io.format3.values import ValueEncoder


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
