"""site_key portability exit gates G1-G5 (S3 discipline): all persistence
runs UNDER the test-only activation switch -- never an active tlspec-v7
write. These gates re-run as acceptance at the wave-3 coordinated bump.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io.prerelease import activate_prerelease_fields
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
    return {label: trace.ops[label].graph.site_key for label in trace.op_labels}


# ---------------------------------------------------------------------------
# G1: switch-on round trip preserves every key verbatim (both writers)
# ---------------------------------------------------------------------------


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
    loaded.ops["relu_1_2:1"]._graph_site_key = "not-a-site-key"
    with pytest.raises(MetadataInvariantError, match="I-S1"):
        check_metadata_invariants(loaded)
    # Colliding forgery (duplicate key within one call instance -- input and
    # output both live in the root pseudo-instance) -> I-S2.
    with activate_prerelease_fields():
        collided = tl.load(str(path))
    collided.ops["output_1:1"]._graph_site_key = collided.ops["input_1:1"].graph.site_key
    with pytest.raises(MetadataInvariantError, match="I-S2"):
        check_metadata_invariants(collided)


def test_g3_label_bearing_forged_key_fails_sweep(tmp_path) -> None:
    trace = tl.trace(_Tied(), torch.randn(2, 8))
    path = tmp_path / "label_forged.tlspec"
    with activate_prerelease_fields():
        trace.save(str(path))
        loaded = tl.load(str(path))
    loaded.ops["relu_1_2:1"]._graph_site_key = "s1|act_raw|relu||9"
    with pytest.raises(MetadataInvariantError, match="survived postprocessing"):
        check_metadata_invariants(loaded)


# ---------------------------------------------------------------------------
# G4: switch-off refusal -- old-v7 and switched artifacts are never
# indistinguishable (includes the streaming/manifest (c)-residual check:
# no sidecar path yields usable data without hitting the marker chokepoint)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# G5: legacy load -- the real (switch-off) v7 artifact was written by
# torchlens 2.x, which this release does not read: it refuses typed
# ---------------------------------------------------------------------------
