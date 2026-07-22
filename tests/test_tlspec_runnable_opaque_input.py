"""Opaque / non-finite non-tensor model-input honesty for sparse runnable execution.

Round-4 hardening. The round-3 fix witnessed *grammar-encodable* non-tensor Python
inputs (bool/int/finite-float/str/None) so a changed control input diverges instead of
false-verifying. But a non-tensor leaf *outside* the frozen literal grammar -- an enum,
``bytes``, ``set``, ``complex``, numpy scalar, or a non-finite ``inf``/``nan`` float --
was witnessed value-free and then never compared, leaving ``witness_completeness`` at
``COMPLETE``. A changed opaque leaf could therefore steer unobserved Python control flow
while the run reported ``VERIFIED`` + ``ATTESTED`` over a numerically wrong replayed path
(the silent-wrong-result tripwire failure).

These tests lock the honest contract: an opaque non-tensor input leaf makes the run
``UNVERIFIABLE`` (never ``VERIFIED``) and numeric attestation ``NOT_APPLICABLE`` (never
``ATTESTED``), whether the leaf changed or not, while the encodable path keeps its exact
round-3 diverge-on-change / verify+attest-on-unchanged behaviour.
"""

from __future__ import annotations

import enum
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    NumericAttestationStatus,
    PathFaithfulness,
)


class _Mode(enum.Enum):
    A = 1
    B = 2


class EnumBranch(nn.Module):
    """Select an arm from an opaque ``enum.Enum`` control input."""

    def forward(self, x: torch.Tensor, mode: _Mode) -> torch.Tensor:
        if mode is _Mode.A:
            return x + 1.0
        return x * 10.0


class FloatFlagBranch(nn.Module):
    """Select an arm from a float control input (may be non-finite at capture)."""

    def forward(self, x: torch.Tensor, scale: float) -> torch.Tensor:
        if scale > 1.0:
            return x + 1.0
        return x * 10.0


class BytesBranch(nn.Module):
    """Select an arm from an opaque ``bytes`` control input."""

    def forward(self, x: torch.Tensor, tag: bytes) -> torch.Tensor:
        if tag == b"add":
            return x + 1.0
        return x * 10.0


class SetBranch(nn.Module):
    """Select an arm from an opaque ``set`` control input."""

    def forward(self, x: torch.Tensor, flags: set) -> torch.Tensor:
        if "add" in flags:
            return x + 1.0
        return x * 10.0


def _save_runnable(model: nn.Module, capture_inputs: list, path: Path) -> Path:
    """Capture (unchanged fast oracle) and save a runnable artifact with attestation."""

    trace = tl.trace(
        model,
        list(capture_inputs),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable", include_activations=True)
    return path


def _assert_not_blessed(report) -> None:
    """An opaque-leaf run must never masquerade as verified or attested."""

    assert report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert report.numeric_attestation is not NumericAttestationStatus.ATTESTED
    assert report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert report.poisoned
    assert not any(
        check.name.startswith("numeric_attestation:") for check in report.contract_checks
    )


@pytest.mark.smoke
def test_enum_leaf_refuses_runnable_save_typed(tmp_path: Path) -> None:
    """r69 B supersedes the save-then-UNVERIFIABLE lane for enum leaves.

    An ``enum.Enum`` input leaf is a SEMANTIC-typed scalar: its type/member
    identity steers host control flow (``mode is _Mode.A``), which no persisted
    value fact can carry. The runnable save now refuses typed
    (``missing_input_container_contract`` / ``semantic_scalar_type``) instead of
    producing an artifact that could ever replay the wrong arm; analysis saves
    stay available. Strictly stronger than the former UNVERIFIABLE ceiling.
    """

    model = EnumBranch()
    x = torch.tensor([2.0, 3.0])
    with pytest.raises(tl.errors.RunnablePreflightError) as excinfo:
        _save_runnable(model, [x, _Mode.A], tmp_path / "enum.tlspec")
    diagnostics = str(excinfo.value.fields.get("diagnostics"))
    assert "missing_input_container_contract" in diagnostics
    assert "semantic_scalar_type" in diagnostics


def test_non_finite_inf_leaf_is_unverifiable_not_attested(tmp_path: Path) -> None:
    """A non-finite ``inf`` float leaf is opaque to the frozen grammar: unverifiable."""

    model = FloatFlagBranch()
    x = torch.tensor([2.0, 3.0])
    path = _save_runnable(model, [x, float("inf")], tmp_path / "inf.tlspec")

    changed = tl.load(path).run(inputs=[x, 0.5])
    _assert_not_blessed(changed.report)
    assert not torch.equal(changed.output, model(x, 0.5))

    unchanged = tl.load(path).run(inputs=[x, float("inf")])
    _assert_not_blessed(unchanged.report)


def test_non_finite_nan_leaf_is_unverifiable_not_attested(tmp_path: Path) -> None:
    """A ``nan`` float leaf is non-finite hence opaque: unverifiable, never attested."""

    model = FloatFlagBranch()
    x = torch.tensor([2.0, 3.0])
    path = _save_runnable(model, [x, float("nan")], tmp_path / "nan.tlspec")

    # nan > 1.0 is False at capture, recording the x*10 arm.
    changed = tl.load(path).run(inputs=[x, 5.0])
    _assert_not_blessed(changed.report)
    unchanged = tl.load(path).run(inputs=[x, float("nan")])
    _assert_not_blessed(unchanged.report)


def test_bytes_leaf_is_unverifiable_not_attested(tmp_path: Path) -> None:
    """An opaque ``bytes`` control input makes the run unverifiable."""

    model = BytesBranch()
    x = torch.tensor([2.0, 3.0])
    path = _save_runnable(model, [x, b"add"], tmp_path / "bytes.tlspec")

    changed = tl.load(path).run(inputs=[x, b"mul"])
    _assert_not_blessed(changed.report)
    assert not torch.equal(changed.output, model(x, b"mul"))


def test_set_leaf_is_unverifiable_not_attested(tmp_path: Path) -> None:
    """An opaque ``set`` control input makes the run unverifiable."""

    model = SetBranch()
    x = torch.tensor([2.0, 3.0])
    path = _save_runnable(model, [x, {"add"}], tmp_path / "set.tlspec")

    changed = tl.load(path).run(inputs=[x, {"mul"}])
    _assert_not_blessed(changed.report)
    assert not torch.equal(changed.output, model(x, {"mul"}))


class FiniteFloatBranch(nn.Module):
    """Encodable float control input (round-3 path) -- must stay diverge/verify."""

    def forward(self, value: torch.Tensor, scale: float) -> torch.Tensor:
        if scale > 1.0:
            return value * 2
        return value - 5


@pytest.mark.smoke
def test_encodable_float_path_unregressed(tmp_path: Path) -> None:
    """The grammar-encodable (finite-float) path keeps exact round-3 behaviour.

    Guards against over-broad UNVERIFIABLE downgrade: a finite-float leaf must
    still DIVERGE on change and VERIFY + ATTEST when unchanged.
    """

    model = FiniteFloatBranch()
    value = torch.tensor([3.0, 4.0])
    path = _save_runnable(model, [value, 2.0], tmp_path / "finite.tlspec")

    same = tl.load(path).run(inputs=[value, 2.0])
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert same.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(same.output, value * 2)

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[value, 0.5])


class SignedZeroBranch(nn.Module):
    """Branch on the sign bit of a finite Python float (``math.copysign``)."""

    def forward(self, value: torch.Tensor, direction: float) -> torch.Tensor:
        import math

        if math.copysign(1.0, direction) < 0:
            return value + 10
        return value - 10


@pytest.mark.smoke
def test_signed_zero_change_diverges(tmp_path: Path) -> None:
    """A changed float sign-bit (``-0.0`` -> ``+0.0``) must DIVERGE, not false-verify.

    ``-0.0 == +0.0`` is ``True`` in Python, so a value-``==`` witness would bless the
    wrong replayed path. The witness compares IEEE bit patterns, so the sign flip is a
    divergence.
    """

    model = SignedZeroBranch()
    value = torch.tensor([2.0])
    path = _save_runnable(model, [value, -0.0], tmp_path / "signed_zero.tlspec")

    # Capture recorded the negative-zero arm (value + 10); positive zero takes the
    # other arm, so replaying the recorded path would be numerically wrong.
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[value, +0.0])

    # Unchanged negative zero still verifies + attests: bit patterns match.
    same = tl.load(path).run(inputs=[value, -0.0])
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert same.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(same.output, value + 10)


def test_unchanged_nan_leaf_does_not_falsely_diverge(tmp_path: Path) -> None:
    """An unchanged ``nan`` float input must not DIVERGE (``nan == nan`` is ``False``).

    ``nan`` is non-finite hence opaque to the frozen grammar (round-4 finding 1), so the
    run is UNVERIFIABLE rather than VERIFIED -- but critically it must *run* and never be
    reported as a spurious divergence from a value-``==`` self-comparison.
    """

    model = FloatFlagBranch()
    value = torch.tensor([2.0, 3.0])
    path = _save_runnable(model, [value, float("nan")], tmp_path / "nan_selfsame.tlspec")

    result = tl.load(path).run(inputs=[value, float("nan")])
    assert result.report.path_faithfulness is not PathFaithfulness.DIVERGED
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_normal_finite_float_change_still_diverges(tmp_path: Path) -> None:
    """A normal finite-float control change keeps diverging (no over-broad relaxation)."""

    model = FiniteFloatBranch()
    value = torch.tensor([3.0, 4.0])
    path = _save_runnable(model, [value, 2.0], tmp_path / "normal.tlspec")

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[value, 0.5])


def test_normal_finite_float_unchanged_still_verifies(tmp_path: Path) -> None:
    """A normal finite-float unchanged input keeps verifying + attesting exactly."""

    model = FiniteFloatBranch()
    value = torch.tensor([3.0, 4.0])
    path = _save_runnable(model, [value, 2.0], tmp_path / "normal.tlspec")

    same = tl.load(path).run(inputs=[value, 2.0])
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert same.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(same.output, value * 2)
