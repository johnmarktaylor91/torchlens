"""r35 stage-4 (corr2_5 + hon1_5, Option A): non-persistent buffer declared state.

The required ``runnable_nonpersistent_buffer_v1`` family carries capture-time
values of used non-persistent buffers (declared state, contract sections 5/11).
Attestation eligibility partitions by persistence: persistent slots compare
against ``capture_state_digests``; used non-persistent slots must originate from
the load-validated embedded family.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    StateSource,
)

pytestmark = pytest.mark.smoke


class _MixedBufferModel(nn.Module):
    """One persistent and one used NON-persistent registered buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("shift", torch.tensor([1.0, 2.0, 3.0]), persistent=True)
        self.register_buffer("scale", torch.tensor([2.0, 2.0, 2.0]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


def _save(model: nn.Module, x: Any, path: Path, **save_kwargs: Any) -> Path:
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    with pytest.warns(UserWarning, match="NON-persistent") if _fresh_warning() else _null():
        tl.save(trace, str(path), level="runnable", **save_kwargs)
    return path


def _fresh_warning() -> bool:
    import torchlens._io.bundle as bundle

    return not bundle._NONPERSISTENT_DISCLOSURE_WARNED


def _null():
    import contextlib

    return contextlib.nullcontext()


def test_r35_mixed_buffer_original_run_attests(tmp_path: Path) -> None:
    """corr2_5 repro: exact non-persistent replay ATTESTS instead of n/a."""

    model = _MixedBufferModel().eval()
    # Mutate the non-persistent buffer pre-capture so its value differs from a
    # fresh constructor value (the hon1_5 configuration).
    with torch.no_grad():
        model.scale.mul_(3.0)
    x = torch.tensor([1.0, 2.0, 3.0])
    path = _save(
        model,
        x,
        tmp_path / "mixed.tlspec",
        include_weights=True,
        include_activations=True,
    )
    result = tl.load(str(path)).run(inputs=x)
    assert result.report.state_source is StateSource.EMBEDDED_CAPTURE_STATE
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert torch.equal(result.output, model(x))


def test_r35_changed_persistent_state_is_not_applicable(tmp_path: Path) -> None:
    """A changed persistent buffer keeps the eligibility partition honest."""

    model = _MixedBufferModel().eval()
    x = torch.tensor([1.0, 2.0, 3.0])
    path = _save(
        model,
        x,
        tmp_path / "changedstate.tlspec",
        include_weights=True,
        include_activations=True,
    )
    loaded = tl.load(str(path))
    loaded.load_state_dict({"shift": torch.tensor([9.0, 9.0, 9.0])})
    result = loaded.run(inputs=x)
    assert result.report.state_source is StateSource.USER_STATE_DICT
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_r35_user_state_cannot_supply_nonpersistent_buffers(tmp_path: Path) -> None:
    """``load_state_dict`` refuses non-persistent names (family is authoritative)."""

    from torchlens.errors import StateBindingError

    model = _MixedBufferModel().eval()
    x = torch.tensor([1.0, 2.0, 3.0])
    path = _save(model, x, tmp_path / "usernp.tlspec", include_weights=True)
    loaded = tl.load(str(path))
    with pytest.raises(StateBindingError):
        loaded.load_state_dict(
            {"shift": torch.tensor([1.0, 2.0, 3.0]), "scale": torch.tensor([5.0, 5.0, 5.0])}
        )


def test_r35_mandatory_family_presence_and_tamper_rejection(tmp_path: Path) -> None:
    """The family is present by default; a tampered blob fails load typed."""

    model = _MixedBufferModel().eval()
    x = torch.tensor([1.0, 2.0, 3.0])
    path = _save(model, x, tmp_path / "family.tlspec")
    manifest = json.loads((path / "manifest.json").read_text())
    layer = manifest["run"]["payload_layers"]["nonpersistent_buffers"]
    assert layer == {"present": True, "schema": "runnable_nonpersistent_buffer_v1"}
    entries = [
        entry
        for entry in manifest["tensors"]
        if entry.get("kind") == "runnable_nonpersistent_buffer"
    ]
    assert [entry["label"] for entry in entries] == ["scale"]
    # Tamper the blob bytes: load must fail its checksum validation.
    blob_path = path / entries[0]["relative_path"]
    blob_bytes = bytearray(blob_path.read_bytes())
    blob_bytes[-1] ^= 0xFF
    blob_path.write_bytes(bytes(blob_bytes))
    with pytest.raises(Exception, match="[Cc]hecksum|corrupt|mismatch"):
        tl.load(str(path))


def test_r35_unused_nonpersistent_buffer_needs_no_family(tmp_path: Path) -> None:
    """A registered-but-UNUSED non-persistent buffer requires no payload."""

    class _UnusedBufferModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("unused", torch.ones(4), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1

    x = torch.tensor([1.0, 2.0])
    trace = tl.trace(
        _UnusedBufferModel().eval(),
        x,
        capture=CaptureOptions(intervention_ready=True, cache=False),
    )
    path = tmp_path / "unused.tlspec"
    tl.save(trace, str(path), level="runnable")
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["run"]["payload_layers"]["nonpersistent_buffers"]["present"] is False
    result = tl.load(str(path)).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
