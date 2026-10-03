"""site_key portability exit gates G1-G5 (S3 discipline): all persistence
runs UNDER the test-only activation switch -- never an active tlspec-v7
write. These gates re-run as acceptance at the wave-3 coordinated bump.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._io import PreReleaseArtifactError
from torchlens._io.prerelease import activate_prerelease_fields
from torchlens.postprocess._site_key import SiteKeyMinter
from torchlens.validation import MetadataInvariantError, check_metadata_invariants


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
        tl.save(trace, str(path))
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
        tl.save(trace, str(path))
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
        tl.save(trace, str(path))
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
        tl.save(trace, str(path))
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
        tl.save(trace, str(path))
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
# G5: legacy load -- a real (switch-off) v7 artifact loads with
# site_key=None everywhere; consumers behave per the consumer matrix
# ---------------------------------------------------------------------------


def test_g5_legacy_v7_artifact_loads_keyless_and_refuses_typed() -> None:
    import warnings
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "tlspec_v7" / "tiny_v7.tlspec"
    with warnings.catch_warnings():
        # The checked-in pre-bump artifact loads with the age advisory.
        warnings.simplefilter("ignore")
        loaded = tl.load(str(fixture))  # real v7 write: site_key was DROPped
    assert set(_keys(loaded).values()) == {None}
    # Out of the invariant's declared domain: validation skips, never reds.
    assert check_metadata_invariants(loaded)
    # Consumer matrix: accessors and the join profile refuse typed.
    with pytest.raises(InvalidArgumentError) as site_exc:
        _ = loaded["linear_1_1"].site_key
    assert site_exc.value.fields["code"] == "site_key_unavailable"
    with pytest.raises(InvalidArgumentError) as peers_exc:
        _ = loaded["linear_1_1"].site_peers
    assert peers_exc.value.fields["code"] == "site_key_unavailable"
    from torchlens.postprocess._site_join import site_profile

    with pytest.raises(InvalidArgumentError) as profile_exc:
        site_profile(loaded)
    assert profile_exc.value.fields["code"] == "site_key_unavailable"
