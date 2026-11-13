"""The JAX ecosystem rows' pins are complete and checked (graduation item G3).

No framework and no network: these run in every environment and keep the
roster's checkpoint evidence, the pinned preprocessing and the file check honest.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from tests.conformance import _jax_checkpoints as checkpoints
from tests.conformance._rosters import jax as jax_roster

_PINS = (checkpoints.BERT_L2_H256, checkpoints.VIT_B16_IN21K)


@pytest.mark.parametrize("pin", _PINS, ids=lambda pin: pin.repo)
def test_pin_is_a_full_commit_digest_and_license(pin: checkpoints.FlaxCheckpoint) -> None:
    """Each pin names a 40-hex commit, 64-hex digests, a size and a license."""
    assert re.fullmatch(r"[0-9a-f]{40}", pin.revision)
    assert re.fullmatch(r"[0-9a-f]{64}", pin.sha256)
    assert re.fullmatch(r"[0-9a-f]{64}", pin.reference_sha256)
    assert pin.size > 0
    assert pin.license == "apache-2.0"
    assert pin.filename == "flax_model.msgpack"


def test_roster_ecosystem_rows_are_pinned_and_unparked() -> None:
    """Both JAX ecosystem rows run, are pretrained, and carry their pin as evidence."""
    rows = [spec for spec in jax_roster.ROW_SPECS if spec["group"] == "ecosystem"]
    assert [spec["family"] for spec in rows] == [
        "bert_uncased_l2_h256_flax",
        "vit_base_patch16_in21k_flax",
    ]
    assert jax_roster.PARKED_ECOSYSTEM_FAMILIES == ()
    assert [spec["checkpoint_evidence"] for spec in rows] == [pin.evidence for pin in _PINS]
    assert [spec["builder"] for spec in rows] == [checkpoints.build_bert, checkpoints.build_vit]
    for spec in rows:
        assert spec["realism"] == "pretrained"
        assert spec["random_init"] is False


def test_ecosystem_roster_returns_both_rows_behind_the_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the ecosystem switch on, the scored roster ends with the two pinned rows."""
    from tests.conformance._rosters import ECOSYSTEM_ENV

    monkeypatch.setenv(ECOSYSTEM_ENV, "1")
    families = [row.family for row in jax_roster.roster()]
    assert families[-2:] == ["bert_uncased_l2_h256_flax", "vit_base_patch16_in21k_flax"]


def test_verify_file_refuses_wrong_size_and_wrong_bytes(tmp_path: Path) -> None:
    """A file of the wrong size, or the right size with other bytes, is refused."""
    payload = b"torchlens pinned checkpoint stand-in"
    good = checkpoints.FlaxCheckpoint(
        repo="example/repo",
        revision="0" * 40,
        filename="flax_model.msgpack",
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        license="apache-2.0",
        reference_filename="model.safetensors",
        reference_sha256="0" * 64,
    )
    path = tmp_path / "flax_model.msgpack"
    path.write_bytes(payload)
    checkpoints.verify_file(str(path), good)
    path.write_bytes(payload + b"!")
    with pytest.raises(checkpoints.CheckpointMismatchError, match="bytes, pinned"):
        checkpoints.verify_file(str(path), good)
    path.write_bytes(payload[::-1])
    with pytest.raises(checkpoints.CheckpointMismatchError, match="sha256"):
        checkpoints.verify_file(str(path), good)


def test_vit_preprocessing_is_pinned_and_deterministic() -> None:
    """The seeded image normalises to a (1, 224, 224, 3) float32 batch in [-1, 1]."""
    pixels = checkpoints.vit_pixels()
    assert pixels.shape == (1, 224, 224, 3)
    assert pixels.dtype == np.float32
    assert float(pixels.min()) >= -1.0
    assert float(pixels.max()) <= 1.0
    np.testing.assert_array_equal(pixels, checkpoints.vit_pixels())
    first = checkpoints.vit_image_bytes()[0, 0]
    expected = (first.astype(np.float32) / np.float32(255) - np.float32(0.5)) / np.float32(0.5)
    np.testing.assert_allclose(pixels[0, 0, 0], expected, atol=1e-6)


def test_bert_input_ids_are_one_unpadded_segment() -> None:
    """The pinned ids open with [CLS] (101), close with [SEP] (102) and hold no padding."""
    ids = checkpoints.BERT_INPUT_IDS
    assert ids[0] == 101
    assert ids[-1] == 102
    assert 0 not in ids
