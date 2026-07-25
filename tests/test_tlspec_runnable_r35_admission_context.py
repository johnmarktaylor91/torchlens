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
from torchlens.errors import PathDivergenceError, RunPreconditionError
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
)
from torchlens.utils._torch_compat import HAS_NAMED_TENSOR_API

pytestmark = pytest.mark.smoke


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
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    tl.save(trace, str(path), level="runnable", **save_kwargs)
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


@pytest.mark.parametrize("kind", sorted(_exotic_inputs()))
@pytest.mark.parametrize("policy", [DivergencePolicy.RAISE, DivergencePolicy.RETURN_DIVERGED])
def test_r35_exotic_inputs_fail_typed_before_any_byte_touch(
    tmp_path: Path, kind: str, policy: DivergencePolicy
) -> None:
    """Hard preconditions raise the TYPED divergence error under BOTH policies.

    corr2_6: previously meta/sparse/named inputs reached
    ``.cpu().contiguous().view(uint8)`` and surfaced raw
    NotImplementedError/RuntimeError before their recorded contract could fire.
    ``return_diverged`` never bypasses a hard executability precondition.
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


def test_r35_dense_inputs_keep_transaction_behavior(tmp_path: Path) -> None:
    """Ordinary dense supported inputs keep the existing transactional flow."""

    x = torch.randn(2, 3)
    path = _save(_AddOneModel().eval(), x, tmp_path / "dense.tlspec", include_activations=True)
    result = tl.load(str(path)).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


def test_r35_digest_helper_refuses_exotic_typed() -> None:
    """I5 defense in depth: the digest helper itself rejects exotic tensors."""

    from torchlens._runnable_state import runnable_tensor_byte_digest

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


@pytest.mark.parametrize("twin_kind", sorted(_conv_twins(torch.zeros(1, 3, 4, 4))))
def test_r35_physical_layout_twin_never_false_trips(tmp_path: Path, twin_kind: str) -> None:
    """A byte-identical physical twin of the original input never false-trips.

    hon1_3: previously a channels_last twin raised NumericAttestationError with
    DIVERGED on a faithful replay. The fingerprint compares the EXECUTED-clone
    basis (H_B_RESOLUTION R2): a twin whose physical difference SURVIVES the
    defensive clone (channels_last memory format -- it can change kernel
    reduction order) is changed-input-for-attestation, verified +
    not_applicable; a twin whose difference the clone ERASES (storage offset,
    non-dense slicing -- the executed value is physically identical to the
    capture execution) honestly stays eligible and must byte-attest exactly.
    Neither may ever raise the corruption tripwire on a faithful replay.
    """

    torch.manual_seed(7)
    xc = torch.randn(1, 3, 4, 4)
    path = _save(
        _ConvModel().eval(),
        xc,
        tmp_path / f"twin_{twin_kind}.tlspec",
        include_weights=True,
        include_activations=True,
    )
    twin = _conv_twins(xc)[twin_kind]
    assert torch.equal(xc, twin)
    result = tl.load(str(path)).run(inputs=twin)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    if twin_kind == "channels_last":
        assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    else:
        # Clone-erased physical difference: execution is byte-identical to the
        # capture execution, so the byte-exact tripwire stays armed and passes.
        assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_r35_original_physical_input_stays_attested(tmp_path: Path) -> None:
    """The exact original input (logical AND physical) still attests."""

    torch.manual_seed(7)
    xc = torch.randn(1, 3, 4, 4)
    path = _save(
        _ConvModel().eval(),
        xc,
        tmp_path / "orig.tlspec",
        include_weights=True,
        include_activations=True,
    )
    result = tl.load(str(path)).run(inputs=xc)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


def test_r35_matmul_carveout_unchanged_no_conv_sdpa_extension() -> None:
    """H_B_RESOLUTION: NO conv/transpose-conv/SDPA extension of the ULP carve-out."""

    from torchlens._runnable_execution import _LAYOUT_SENSITIVE_BLAS_QUALNAMES

    assert _LAYOUT_SENSITIVE_BLAS_QUALNAMES == frozenset(
        {
            "linear",
            "matmul",
            "mm",
            "bmm",
            "mv",
            "dot",
            "vdot",
            "inner",
            "outer",
            "ger",
            "addmm",
            "addbmm",
            "baddbmm",
            "addmv",
            "addr",
            "einsum",
            "tensordot",
        }
    )
    assert not any(
        "conv" in name or "attention" in name for name in _LAYOUT_SENSITIVE_BLAS_QUALNAMES
    )


# ---------------------------------------------------------------------------
# corr2_8 + decision E -- execution-context capture and restore.
# ---------------------------------------------------------------------------


def test_r35_autocast_capture_replays_outside_autocast(tmp_path: Path) -> None:
    """corr2_8 repro: a bf16-autocast capture replays faithfully with no ambient."""

    model = nn.Sequential(nn.Linear(4, 3)).eval()
    x = torch.randn(2, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        trace = tl.trace(
            model,
            x,
            capture=CaptureOptions(
                intervention_ready=True, capture_container_structure=True, cache=False
            ),
        )
    path = tmp_path / "autocast.tlspec"
    tl.save(trace, str(path), level="runnable", include_weights=True)
    result = tl.load(str(path)).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.output.dtype is torch.bfloat16


def test_r35_disabled_capture_replays_inside_ambient_autocast(tmp_path: Path) -> None:
    """An autocast-DISABLED capture is immune to the caller's ambient autocast."""

    model = nn.Sequential(nn.Linear(4, 3)).eval()
    x = torch.randn(2, 4)
    path = _save(model, x, tmp_path / "noamp.tlspec", include_weights=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = tl.load(str(path)).run(inputs=x)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.output.dtype is torch.float32


def test_r35_caller_ambient_state_restored_after_run_and_after_failure(
    tmp_path: Path,
) -> None:
    """The caller's ambient backend context survives success AND divergence."""

    x = torch.randn(2, 3)
    path = _save(_AddOneModel().eval(), x, tmp_path / "restore.tlspec", include_weights=True)
    loaded = tl.load(str(path))
    caller_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("medium")
    try:
        loaded.run(inputs=x)
        assert torch.get_float32_matmul_precision() == "medium"
        with pytest.raises(PathDivergenceError):
            loaded.run(inputs=torch.randn(5, 7))
        assert torch.get_float32_matmul_precision() == "medium"
        assert torch.get_default_dtype() is torch.float32
    finally:
        torch.set_float32_matmul_precision(caller_precision)


def test_r35_recorded_ambient_context_is_applied_during_run(tmp_path: Path) -> None:
    """The recorded capture ambient value (not the caller's) governs the run."""

    x = torch.randn(2, 3)
    caller_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        path = _save(_AddOneModel().eval(), x, tmp_path / "amb.tlspec", include_weights=True)
    finally:
        torch.set_float32_matmul_precision(caller_precision)
    loaded = tl.load(str(path))
    descriptor = loaded.__dict__["_runnable_descriptor"]
    assert descriptor.ambient_context.float32_matmul_precision == "high"
    # Run under a DIFFERENT caller precision; the run must restore the caller's
    # value afterwards while the recorded one governed execution.
    torch.set_float32_matmul_precision("highest")
    try:
        result = loaded.run(inputs=x)
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(caller_precision)
