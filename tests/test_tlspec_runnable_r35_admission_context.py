"""r35 stage-2 regressions: contract-before-touch admission (corr2_6, I5),
physical input fingerprints (hon1_3 H-a + H_B_RESOLUTION R2), and per-call /
ambient execution-context restore (corr2_8, decision E).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._runnable import (
    DivergencePolicy,
    PathFaithfulness,
)
from torchlens._utils._torch_compat import HAS_NAMED_TENSOR_API
from torchlens.errors import PathDivergenceError, RunPreconditionError
from torchlens.options import CaptureOptions


class _AddOneModel(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1


class _ConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _save(model: nn.Module, args: Any, path: Path, **save_kwargs: Any) -> Path:
    trace = tl.trace(
        model,
        args,
        capture=CaptureOptions(replayable=True, capture_container_structure=True, cache=False),
    )
    trace.save(str(path), level="runnable", **save_kwargs)
    return path


# ---------------------------------------------------------------------------
# corr2_6 -- contract-before-touch admission.
# ---------------------------------------------------------------------------


def _exotic_inputs() -> dict[str, torch.Tensor]:
    dense = torch.randn(2, 3)
    inputs = {
        "meta": torch.empty(2, 3, device="meta"),
        "sparse_coo": dense.to_sparse(),
        "sparse_csr": dense.to_sparse_csr(),
        "nested": torch.nested.nested_tensor([torch.randn(3), torch.randn(3)]),
    }
    if HAS_NAMED_TENSOR_API:
        inputs["named"] = torch.randn(2, 3).refine_names("rows", "cols")
    return inputs


# Kept: shares a parametrize list that is not trimmable per id with kept params.
@pytest.mark.parametrize("kind", sorted(_exotic_inputs()))
@pytest.mark.parametrize("policy", ["raise", "return"])
def test_r35_exotic_inputs_fail_typed_before_any_byte_touch(
    tmp_path: Path, kind: str, policy: DivergencePolicy
) -> None:
    """Hard preconditions raise the TYPED divergence error under BOTH policies.

    corr2_6: previously meta/sparse/named inputs reached
    ``.cpu().contiguous().view(uint8)`` and surfaced raw
    NotImplementedError/RuntimeError before their recorded contract could fire.
    ``on_divergence="return"`` never bypasses a hard executability precondition.
    """

    x = torch.randn(2, 3)
    path = _save(
        _AddOneModel().eval(),
        x,
        tmp_path / f"{kind}.tlspec",
        include_weights=True,
        include_activations=True,
    )
    loaded = tl.load(str(path))
    exotic = _exotic_inputs()[kind]
    with pytest.raises(PathDivergenceError) as excinfo:
        loaded.run(inputs=exotic, on_divergence=policy)
    check = excinfo.value.fields.get("contract_check")
    assert check is not None and check.name.startswith("input_layout:")


# Kept: sole branch coverage of _runnable/state.py.
def test_r35_digest_helper_refuses_exotic_typed() -> None:
    """I5 defense in depth: the digest helper itself rejects exotic tensors."""

    from torchlens._runnable.state import runnable_tensor_byte_digest

    for value in _exotic_inputs().values():
        with pytest.raises(RunPreconditionError):
            runnable_tensor_byte_digest(value)
    assert len(runnable_tensor_byte_digest(torch.randn(2, 2))) == 64


# ---------------------------------------------------------------------------
# hon1_3 (H-a) -- physical input fingerprints; layout twins never false-trip.
# ---------------------------------------------------------------------------


def _conv_twins(xc: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "channels_last": xc.clone().contiguous(memory_format=torch.channels_last),
        "storage_offset": torch.cat([torch.zeros_like(xc), xc])
        .reshape(-1)[xc.numel() :]
        .reshape(xc.shape),
        "sliced_noncontiguous": torch.stack([xc, torch.zeros_like(xc)], dim=-1)[..., 0],
    }


# ---------------------------------------------------------------------------
# corr2_8 + decision E -- execution-context capture and restore.
# ---------------------------------------------------------------------------


# Kept: sole branch coverage of _runnable/state_context.py.
def test_r35_autocast_capture_replays_outside_autocast(tmp_path: Path) -> None:
    """corr2_8 repro: a bf16-autocast capture replays faithfully with no ambient."""

    model = nn.Sequential(nn.Linear(4, 3)).eval()
    x = torch.randn(2, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        trace = tl.trace(
            model,
            x,
            capture=CaptureOptions(replayable=True, capture_container_structure=True, cache=False),
        )
    path = tmp_path / "autocast.tlspec"
    trace.save(str(path), level="runnable", include_weights=True)
    result = tl.load(str(path)).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.output.dtype is torch.bfloat16


# Kept: sole branch coverage of _runnable/state_context.py.
def test_r35_disabled_capture_replays_inside_ambient_autocast(tmp_path: Path) -> None:
    """An autocast-DISABLED capture is immune to the caller's ambient autocast."""

    model = nn.Sequential(nn.Linear(4, 3)).eval()
    x = torch.randn(2, 4)
    path = _save(model, x, tmp_path / "noamp.tlspec", include_weights=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = tl.load(str(path)).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.output.dtype is torch.float32
