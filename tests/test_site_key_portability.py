"""site_key portability exit gates G1-G5 (S3 discipline): all persistence
runs UNDER the test-only activation switch -- never an active tlspec-v7
write. These gates re-run as acceptance at the wave-3 coordinated bump.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io import PreReleaseArtifactError
from torchlens._io.prerelease import activate_prerelease_fields
from torchlens._postprocess._site_key import SiteKeyMinter
from torchlens.errors import MetadataInvariantError
from torchlens.validation import check_metadata_invariants


class _Tied(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = self.act(self.lin(x))
        return x


def _keys(trace: tl.Trace) -> dict[str, str | None]:
    return {label: trace.ops[label].site_key for label in trace.op_labels}


# ---------------------------------------------------------------------------
# G1: switch-on round trip preserves every key verbatim (both writers)
# ---------------------------------------------------------------------------


def test_g1_roundtrip_bundle_writer(tmp_path) -> None:
    trace = tl.trace(_Tied(), torch.randn(2, 8))
    saved_keys = _keys(trace)
    assert all(isinstance(key, str) for key in saved_keys.values())
    path = tmp_path / "sites.tlspec"
    with activate_prerelease_fields():
        trace.save(str(path))
        loaded = tl.load(str(path))
    assert _keys(loaded) == saved_keys


def test_g1_roundtrip_streaming_writer(tmp_path) -> None:
    path = tmp_path / "sites_stream.tlspec"
    with activate_prerelease_fields():
        trace = tl.trace(_Tied(), torch.randn(2, 8), storage=tl.to_disk(str(path)))
        saved_keys = _keys(trace)
        loaded = tl.load(str(path))
    assert all(isinstance(key, str) for key in saved_keys.values())
    assert _keys(loaded) == saved_keys


# ---------------------------------------------------------------------------
# G2: recompute parity on the LOADED artifact (I-S4) -- recomputing keys
# from loaded raw facts via the canonical site-axis function reproduces the
# stored strings byte-for-byte, over the SERIALIZED representation.
# ---------------------------------------------------------------------------


def test_g2_loaded_recompute_byte_parity(tmp_path) -> None:
    trace = tl.trace(_Tied(), torch.randn(2, 8))
    path = tmp_path / "parity.tlspec"
    with activate_prerelease_fields():
        trace.save(str(path))
        loaded = tl.load(str(path))
    minter = SiteKeyMinter()
    for label in loaded.op_labels:
        op = loaded.ops[label]
        recomputed = minter.mint(
            tuple(op.module_call_stack or ()),
            op.type,
            getattr(op, "multi_output_index", None),
        )
        assert recomputed == op.site_key, (label, recomputed, op.site_key)


# ---------------------------------------------------------------------------
# G3: forged keys on a loaded artifact fail validation
# ---------------------------------------------------------------------------


def test_g3_forged_key_fails_is1_and_is2(tmp_path) -> None:
    trace = tl.trace(_Tied(), torch.randn(2, 8))
    path = tmp_path / "forged.tlspec"
    with activate_prerelease_fields():
        trace.save(str(path))
        loaded = tl.load(str(path))
    assert check_metadata_invariants(loaded)
    # Malformed forgery -> I-S1.
    loaded.ops["relu_1_2:1"].site_key = "not-a-site-key"
    with pytest.raises(MetadataInvariantError, match="I-S1"):
        check_metadata_invariants(loaded)
    # Colliding forgery (duplicate key within one call instance -- input and
    # output both live in the root pseudo-instance) -> I-S2.
    with activate_prerelease_fields():
        collided = tl.load(str(path))
    collided.ops["output_1:1"].site_key = collided.ops["input_1:1"].site_key
    with pytest.raises(MetadataInvariantError, match="I-S2"):
        check_metadata_invariants(collided)


def test_g3_label_bearing_forged_key_fails_sweep(tmp_path) -> None:
    trace = tl.trace(_Tied(), torch.randn(2, 8))
    path = tmp_path / "label_forged.tlspec"
    with activate_prerelease_fields():
        trace.save(str(path))
        loaded = tl.load(str(path))
    loaded.ops["relu_1_2:1"].site_key = "s1|act_raw|relu||9"
    with pytest.raises(MetadataInvariantError, match="survived postprocessing"):
        check_metadata_invariants(loaded)


# ---------------------------------------------------------------------------
# G4: switch-off refusal -- old-v7 and switched artifacts are never
# indistinguishable (includes the streaming/manifest (c)-residual check:
# no sidecar path yields usable data without hitting the marker chokepoint)
# ---------------------------------------------------------------------------


def test_g4_marked_bundle_refuses_switch_off(tmp_path) -> None:
    trace = tl.trace(_Tied(), torch.randn(2, 8))
    path = tmp_path / "marked.tlspec"
    with activate_prerelease_fields():
        trace.save(str(path))
    with pytest.raises(PreReleaseArtifactError, match="pre-release field marker"):
        tl.load(str(path))


def test_g4_marked_streaming_artifact_refuses_switch_off(tmp_path) -> None:
    # The (c)-residual named by the memo: the streaming writer's
    # manifest/sidecar path must not yield usable data without
    # reconstructing a marker-carrying record.
    path = tmp_path / "marked_stream.tlspec"
    with activate_prerelease_fields():
        tl.trace(_Tied(), torch.randn(2, 8), storage=tl.to_disk(str(path)))
    with pytest.raises(PreReleaseArtifactError, match="pre-release field marker"):
        tl.load(str(path))


# ---------------------------------------------------------------------------
# G5: legacy load -- the real (switch-off) v7 artifact was written by
# torchlens 2.x, which this release does not read: it refuses typed
# ---------------------------------------------------------------------------


def test_g5_legacy_v7_artifact_refuses_as_2x() -> None:
    from pathlib import Path

    from _refuses_as_2x import assert_refuses_as_2x

    fixture = Path(__file__).parent / "fixtures" / "tlspec_v7" / "tiny_v7.tlspec"
    assert_refuses_as_2x(fixture)
