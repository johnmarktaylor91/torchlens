"""Tensor->Python-scalar escape honesty for sparse runnable execution.

Round-8 hardening (the r8 "money bug"). A tensor->Python-scalar escape
(``.item()`` / ``int()`` / ``float()`` / ``aten._local_scalar_dense``) used AS A
VALUE bakes the derived scalar into a downstream op as a literal constant. The
escape breaks the tensor graph, so the sparse DAG never recomputes that constant.
Before this fix, a run on a CHANGED input used the STALE baked literal yet reported
``path_faithfulness=VERIFIED`` + ``numeric_attestation=ATTESTED`` -- a silently
wrong output AND wrong saved activations. Example: ``s = x.mean().item(); return
x * s`` captured with ``x=[2, 2]`` bakes ``s=2``; run with ``x=[10, 10]`` a live
forward is ``[100, 100]`` but the sparse replay returns the stale ``[20, 20]``.

These tests lock the honest contract: a run whose baked tensor-derived-scalar
literal is stale for the given input (the scalar-producing sink recomputes a
different value) reports ``UNVERIFIABLE`` + ``NOT_APPLICABLE`` (never a false
``VERIFIED`` / ``ATTESTED``), while the ORIGINAL/unchanged input still reports
``VERIFIED`` + ``ATTESTED`` (the baked constant is correct there). Deterministic
scalar-free models and genuinely dead scalar sinks (a reduction whose value is
discarded, never baked) are UNCHANGED.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
    WitnessCompleteness,
)


class DDScale(nn.Module):
    """Scale by a data-dependent Python scalar extracted via ``.item()``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean().item()
        return x * s


class TempScale(nn.Module):
    """Normalize by a scalar temperature extracted via ``.item()``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.std().item()
        return torch.softmax(x / t, dim=-1)


class ArgmaxSelect(nn.Module):
    """Gather with an ``int(argmax())`` Python index escape."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i = int(x.argmax().item())
        return x * x[i]


class ScalarFree(nn.Module):
    """Deterministic model with only genuine literal constants (no escape)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x * 2.0) + 1.0


class DeadScalarSink(nn.Module):
    """Compute a scalar reduction, discard it (never baked), return a real op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _ = (x * 3.0).sum()  # scalar sink whose value never escapes / is baked
        return x + 1.0


def _save_runnable(model: nn.Module, capture_input: torch.Tensor, path: Path) -> Path:
    """Capture (unchanged fast oracle) and save a runnable artifact with attestation."""

    trace = tl.trace(
        model,
        capture_input,
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable", include_activations=True)
    return path


def _assert_not_blessed(report) -> None:
    """A stale tensor-derived-scalar run must never masquerade as verified/attested."""

    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert report.numeric_attestation is not NumericAttestationStatus.ATTESTED
    assert report.poisoned
    assert not any(
        check.name.startswith("numeric_attestation:") for check in report.contract_checks
    )


def _assert_verified_attested(report) -> None:
    """An original-input run of the same model must still verify and attest."""

    assert report.path_faithfulness is PathFaithfulness.VERIFIED
    assert report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert not report.poisoned


@pytest.mark.smoke
def test_item_scale_changed_input_is_unverifiable_not_attested(tmp_path: Path) -> None:
    path = _save_runnable(DDScale(), torch.tensor([2.0, 2.0]), tmp_path / "dd.tlspec")

    result = tl.load(path).run(inputs=torch.tensor([10.0, 10.0]))

    _assert_not_blessed(result.report)
    # The stale baked literal (s=2) yields [20, 20]; the honest report refuses to
    # bless it even though the DAG produced an output.
    assert torch.allclose(result.output, torch.tensor([20.0, 20.0]))


@pytest.mark.smoke
def test_item_scale_original_input_still_verified_attested(tmp_path: Path) -> None:
    path = _save_runnable(DDScale(), torch.tensor([2.0, 2.0]), tmp_path / "dd.tlspec")

    result = tl.load(path).run(inputs=torch.tensor([2.0, 2.0]))

    _assert_verified_attested(result.report)
    assert torch.allclose(result.output, torch.tensor([4.0, 4.0]))


@pytest.mark.smoke
def test_temperature_scaling_changed_input_is_unverifiable(tmp_path: Path) -> None:
    path = _save_runnable(TempScale(), torch.tensor([1.0, 2.0, 3.0, 4.0]), tmp_path / "t.tlspec")

    result = tl.load(path).run(inputs=torch.tensor([10.0, 0.0, -10.0, 5.0]))

    _assert_not_blessed(result.report)


def test_temperature_scaling_original_input_verified_attested(tmp_path: Path) -> None:
    capture = torch.tensor([1.0, 2.0, 3.0, 4.0])
    path = _save_runnable(TempScale(), capture, tmp_path / "t.tlspec")

    result = tl.load(path).run(inputs=capture.clone())

    _assert_verified_attested(result.report)


@pytest.mark.smoke
def test_argmax_index_changed_input_is_unverifiable(tmp_path: Path) -> None:
    path = _save_runnable(ArgmaxSelect(), torch.tensor([1.0, 5.0, 2.0]), tmp_path / "a.tlspec")

    result = tl.load(path).run(inputs=torch.tensor([9.0, 1.0, 2.0]))

    _assert_not_blessed(result.report)


def test_argmax_index_original_input_verified_attested(tmp_path: Path) -> None:
    capture = torch.tensor([1.0, 5.0, 2.0])
    path = _save_runnable(ArgmaxSelect(), capture, tmp_path / "a.tlspec")

    result = tl.load(path).run(inputs=capture.clone())

    _assert_verified_attested(result.report)


@pytest.mark.smoke
def test_scalar_free_model_changed_input_unchanged(tmp_path: Path) -> None:
    path = _save_runnable(ScalarFree(), torch.tensor([1.0, -2.0]), tmp_path / "sf.tlspec")

    result = tl.load(path).run(inputs=torch.tensor([3.0, -5.0]))

    # No tensor-derived-scalar dependency: a deterministic scalar-free model stays
    # VERIFIED on a changed input (the taken path is faithful).
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    live = ScalarFree()(torch.tensor([3.0, -5.0]))
    assert torch.allclose(result.output, live)


def test_dead_scalar_sink_not_over_triggered(tmp_path: Path) -> None:
    # A scalar reduction whose value is discarded (never .item()-escaped, never
    # baked) must NOT downgrade the run: no descriptor witness is emitted.
    trace = tl.trace(
        DeadScalarSink(),
        torch.tensor([1.0, 2.0]),
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    from torchlens._io.runnable import build_sparse_run_descriptor

    descriptor = build_sparse_run_descriptor(trace)
    assert descriptor.witness_completeness is WitnessCompleteness.COMPLETE
    assert not any(
        witness.kind.value == "tensor_derived_scalar_literal"
        for witness in descriptor.control_witnesses
    )

    path = _save_runnable(DeadScalarSink(), torch.tensor([1.0, 2.0]), tmp_path / "dead.tlspec")
    result = tl.load(path).run(inputs=torch.tensor([7.0, 8.0]))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned
    live = DeadScalarSink()(torch.tensor([7.0, 8.0]))
    assert torch.allclose(result.output, live)


def test_item_scale_descriptor_emits_tensor_derived_scalar_witness(tmp_path: Path) -> None:
    from torchlens._io.runnable import build_sparse_run_descriptor

    trace = tl.trace(
        DDScale(),
        torch.tensor([2.0, 2.0]),
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    descriptor = build_sparse_run_descriptor(trace)

    scalar_witnesses = [
        witness
        for witness in descriptor.control_witnesses
        if witness.kind.value == "tensor_derived_scalar_literal"
    ]
    assert len(scalar_witnesses) == 1
    witness = scalar_witnesses[0]
    assert witness.site_label == "slot:mean_1_1:1"
    # witness_completeness stays COMPLETE so the ORIGINAL input can still attest;
    # the downgrade is input-conditional at run time, not a static descriptor flag.
    assert descriptor.witness_completeness is WitnessCompleteness.COMPLETE
