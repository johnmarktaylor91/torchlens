"""Non-tensor Python model-input contract for sparse runnable execution.

These tests lock the honesty rule that a sparse runnable ``run`` may never report
``VERIFIED``/``ATTESTED`` after silently replaying the recorded taken path for a
*changed* non-tensor Python input (bool/int/float). A changed control input can
make the recorded path numerically wrong, so it must diverge (or, under
``return_diverged``, be reported diverged/poisoned and never attested) while an
unchanged input still verifies and attests exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
)


class BoolBranch(nn.Module):
    """Select addition or subtraction from a non-tensor Python bool."""

    def forward(self, left: torch.Tensor, right: torch.Tensor, add_values: bool) -> torch.Tensor:
        """Add or subtract according to the runtime Python bool."""

        if add_values:
            return left + right
        return left - right


class IntBranch(nn.Module):
    """Select an arithmetic arm from a non-tensor Python int."""

    def forward(self, value: torch.Tensor, mode: int) -> torch.Tensor:
        """Route on an integer mode selector."""

        if mode == 0:
            return value + 1
        return value * 10


class FloatBranch(nn.Module):
    """Select an arithmetic arm from a non-tensor Python float."""

    def forward(self, value: torch.Tensor, scale: float) -> torch.Tensor:
        """Route on a float threshold."""

        if scale > 1.0:
            return value * 2
        return value - 5


class MixedSingleTensorModel(nn.Module):
    """One tensor argument beside one non-tensor Python argument."""

    def forward(self, value: torch.Tensor, flag: bool) -> torch.Tensor:
        """Shift the tensor up or down from the Python bool."""

        if flag:
            return value + 100
        return value - 100


def _save_runnable(model: nn.Module, capture_inputs: list, path: Path) -> Path:
    """Capture and save a runnable artifact carrying activation attestation."""

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


@pytest.mark.smoke
def test_changed_bool_input_diverges_and_is_not_attested(tmp_path: Path) -> None:
    """A changed bool input must raise under the default rollback policy."""

    model = BoolBranch()
    left = torch.tensor([1.0, 4.0])
    right = torch.tensor([2.0, 2.0])
    path = _save_runnable(model, [left, right, True], tmp_path / "bool.tlspec")

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[left, right, False])

    diverged = tl.load(path).run(
        inputs=[left, right, False],
        on_divergence=DivergencePolicy.RETURN_DIVERGED,
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is not NumericAttestationStatus.ATTESTED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert diverged.report.poisoned
    # The honest report must never masquerade the silent-wrong result as correct.
    fresh = model(left, right, False)
    assert torch.equal(fresh, left - right)


@pytest.mark.smoke
def test_unchanged_bool_input_still_verifies_and_attests(tmp_path: Path) -> None:
    """An identical bool input must verify, attest, and return the right output."""

    model = BoolBranch()
    left = torch.tensor([1.0, 4.0])
    right = torch.tensor([2.0, 2.0])
    path = _save_runnable(model, [left, right, True], tmp_path / "bool.tlspec")

    result = tl.load(path).run(inputs=[left, right, True])
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert not result.report.poisoned
    assert all(check.passed for check in result.report.contract_checks)
    torch.testing.assert_close(result.output, left + right)


def test_changed_int_input_diverges_unchanged_verifies(tmp_path: Path) -> None:
    """Int control inputs follow the same changed-diverges/unchanged-verifies rule."""

    model = IntBranch()
    value = torch.tensor([1.0, 2.0])
    path = _save_runnable(model, [value, 0], tmp_path / "int.tlspec")

    same = tl.load(path).run(inputs=[value, 0])
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert same.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(same.output, value + 1)

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[value, 1])
    diverged = tl.load(path).run(inputs=[value, 1], on_divergence=DivergencePolicy.RETURN_DIVERGED)
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_changed_float_input_diverges_unchanged_verifies(tmp_path: Path) -> None:
    """Float control inputs follow the same changed-diverges/unchanged-verifies rule."""

    model = FloatBranch()
    value = torch.tensor([3.0, 4.0])
    path = _save_runnable(model, [value, 2.0], tmp_path / "float.tlspec")

    same = tl.load(path).run(inputs=[value, 2.0])
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert same.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(same.output, value * 2)

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[value, 0.5])
    diverged = tl.load(path).run(
        inputs=[value, 0.5], on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_mixed_tensor_and_python_arg_binds_without_crashing(tmp_path: Path) -> None:
    """One tensor arg plus one Python arg binds on original inputs and diverges on a change.

    Regression for the binder shortcut that treated the whole ``[tensor, python]``
    runtime argument list as the single tensor site, producing a spurious
    input-tree mismatch even for the original inputs.
    """

    model = MixedSingleTensorModel()
    value = torch.tensor([5.0])
    path = _save_runnable(model, [value, True], tmp_path / "mixed.tlspec")

    original = tl.load(path).run(inputs=[value, True])
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert original.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(original.output, value + 100)

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[value, False])
    diverged = tl.load(path).run(
        inputs=[value, False], on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


# --------------------------------------------------------------------------- #
# r27-H1: an EXTRA runtime non-tensor input leaf (an added dict key the model
# branches on, or a longer list) steers unwitnessed control flow. The per-leaf
# value check only visits RECORDED leaves, so it is blind to an extra leaf; the
# non-tensor leaf-path SET check (mirroring the tensor-leaf set equality) now
# diverges on any added/removed non-tensor leaf. A fresh model on the given
# inputs differs, so the run must never report VERIFIED/ATTESTED.
# --------------------------------------------------------------------------- #


class _FlagInDictModel(nn.Module):
    """Branch on the PRESENCE of an extra non-tensor dict key."""

    def forward(self, d: dict) -> torch.Tensor:
        """Steer control flow on whether an extra ``'flag'`` key is present."""

        x = d["x"]
        if "flag" in d:
            return x * 10.0
        return x + 1.0


class _ExtraListElementModel(nn.Module):
    """Branch on the LENGTH of a runtime input list nested in a dict.

    The list lives inside a dict so ``tl.trace`` does not unpack it as positional
    arguments; an extra trailing non-tensor element is a new non-tensor leaf.
    """

    def forward(self, d: dict) -> torch.Tensor:
        """Steer control flow on whether an extra trailing list element exists."""

        x = d["x"]
        tags = d.get("tags", [])
        if len(tags) > 0:
            return x * 5.0
        return x + 2.0


def _save_dictish(model: nn.Module, capture_input, path: Path) -> Path:
    """Capture and save a runnable artifact for a single container-input model."""

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


@pytest.mark.smoke
def test_h1_extra_dict_key_diverges_identical_verifies(tmp_path: Path) -> None:
    """An extra dict key must diverge; the identical dict input still VERIFIES."""

    x = torch.tensor([1.0, 2.0])
    path = _save_dictish(_FlagInDictModel(), {"x": x}, tmp_path / "flag_dict.tlspec")

    # No over-trigger: the identical (single-key) input still verifies + attests.
    same = tl.load(path).run(inputs={"x": x})
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert same.report.numeric_attestation is NumericAttestationStatus.ATTESTED

    # An extra non-tensor leaf steers an unwitnessed branch -> fail closed.
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs={"x": x, "flag": True})
    diverged = tl.load(path).run(
        inputs={"x": x, "flag": True}, on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert diverged.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


@pytest.mark.smoke
def test_h1_longer_list_diverges_identical_verifies(tmp_path: Path) -> None:
    """A longer runtime input list (extra non-tensor element) must diverge."""

    x = torch.tensor([3.0, 4.0])
    path = _save_dictish(
        _ExtraListElementModel(), {"x": x, "tags": []}, tmp_path / "extra_list.tlspec"
    )

    same = tl.load(path).run(inputs={"x": x, "tags": []})
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED

    diverged = tl.load(path).run(
        inputs={"x": x, "tags": ["extra"]}, on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert diverged.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


# ======================================================================================
# r42 corr1_1 / corr1_2 (CLASS 3) immunizers -- extra top-level arity + dataclass inputs
# ======================================================================================

import dataclasses  # noqa: E402

from torchlens.runnable import RunnableErrorCode  # noqa: E402


class _TwoArgModel(nn.Module):
    """Strict two-positional-argument model (no varargs)."""

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Add two tensors."""

        return x + y


class _KwOnlyModel(nn.Module):
    """Strict keyword-only model (no ``**kwargs``)."""

    def forward(self, x: torch.Tensor, *, scale: int = 2) -> torch.Tensor:
        """Scale one tensor by a keyword-only int."""

        return (x * scale).relu()


@pytest.mark.smoke
def test_extra_positional_arg_diverges_never_verified(tmp_path: Path) -> None:
    """r42 corr1_1: an EXTRA top-level positional arg is INPUT_ARITY_EXTRA, never VERIFIED."""

    x = torch.tensor([1.0])
    y = torch.tensor([2.0])
    path = _save_runnable(_TwoArgModel(), [x, y], tmp_path / "extra_pos.tlspec")
    with pytest.raises(PathDivergenceError) as exc:
        tl.load(path).run(inputs=[x, y, torch.tensor([3.0])])
    assert exc.value.fields.get("code") == RunnableErrorCode.INPUT_ARITY_EXTRA.value
    diverged = tl.load(path).run(
        inputs=[x, y, torch.tensor([3.0])], on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.first_mismatch is not None
    assert diverged.report.first_mismatch.code is RunnableErrorCode.INPUT_ARITY_EXTRA
    # Identical original inputs stay VERIFIED (no over-trigger).
    ok = tl.load(path).run(inputs=[x, y])
    assert ok.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_extra_keyword_arg_diverges_never_verified(tmp_path: Path) -> None:
    """r42 corr1_1: an EXTRA top-level keyword not present at capture is INPUT_ARITY_EXTRA."""

    x = torch.tensor([-1.0, 2.0])
    trace = tl.trace(
        _KwOnlyModel(),
        x,
        input_kwargs={"scale": 2},
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    path = tmp_path / "extra_kw.tlspec"
    trace.save(path, level="runnable", include_activations=True)
    with pytest.raises(PathDivergenceError) as exc:
        tl.load(path).run(inputs={"args": [x], "kwargs": {"scale": 2, "extra": 0}})
    assert exc.value.fields.get("code") == RunnableErrorCode.INPUT_ARITY_EXTRA.value
    ok = tl.load(path).run(inputs={"args": [x], "kwargs": {"scale": 2}})
    assert ok.report.path_faithfulness is PathFaithfulness.VERIFIED


@dataclasses.dataclass
class _TensorPair:
    """Tensor-only dataclass model input."""

    x: torch.Tensor
    y: torch.Tensor


@dataclasses.dataclass
class _MixedPair:
    """Dataclass with a tensor field and a non-tensor control field."""

    x: torch.Tensor
    flag: bool


@dataclasses.dataclass
class _OpaqueFieldPair:
    """Dataclass with a tensor field and a genuinely-opaque (unencodable) field."""

    x: torch.Tensor
    tag: object


class _DataclassAddModel(nn.Module):
    """Add the two tensor fields of a dataclass input."""

    def forward(self, pair: _TensorPair) -> torch.Tensor:
        """Return ``pair.x + pair.y``."""

        return pair.x + pair.y


class _DataclassFlagModel(nn.Module):
    """Branch on a dataclass non-tensor field."""

    def forward(self, pair: _MixedPair) -> torch.Tensor:
        """Add or subtract a constant per the dataclass bool field."""

        return pair.x + 1.0 if pair.flag else pair.x - 1.0


class _DataclassOpaqueModel(nn.Module):
    """Read the tensor field of a dataclass carrying an opaque field."""

    def forward(self, pair: _OpaqueFieldPair) -> torch.Tensor:
        """Return ``pair.x * 2`` (the opaque ``tag`` field is unwitnessable)."""

        return pair.x * 2.0


@pytest.mark.smoke
def test_tensor_only_dataclass_input_verifies_and_attests(tmp_path: Path) -> None:
    """r42 corr1_2: a tensor-only dataclass input is fully witnessable -> VERIFIED (+ATTESTED)."""

    pair = _TensorPair(torch.randn(2), torch.randn(2))
    path = _save_runnable(_DataclassAddModel(), [pair], tmp_path / "dc_tensor.tlspec")
    result = tl.load(path).run(inputs=pair)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED


@pytest.mark.smoke
def test_dataclass_changed_bool_field_diverges(tmp_path: Path) -> None:
    """r42 corr1_2: a dataclass NON-tensor field is still witnessed -- a changed value diverges."""

    pair = _MixedPair(torch.randn(2), True)
    path = _save_runnable(_DataclassFlagModel(), [pair], tmp_path / "dc_flag.tlspec")
    changed = _MixedPair(pair.x, False)
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=changed)
    ok = tl.load(path).run(inputs=pair)
    assert ok.report.path_faithfulness is PathFaithfulness.VERIFIED


@pytest.mark.smoke
def test_dataclass_opaque_field_never_verified(tmp_path: Path) -> None:
    """r42 corr1_2 (no-weakening guard): a dataclass with a genuinely-opaque field is never
    blessed. The tensor-only descent does NOT mask a genuinely-unwitnessable field -- the run
    stays fail-closed (never VERIFIED, never ATTESTED, poisoned)."""

    pair = _OpaqueFieldPair(torch.randn(2), object())
    path = _save_runnable(_DataclassOpaqueModel(), [pair], tmp_path / "dc_opaque.tlspec")
    result = tl.load(path).run(inputs=pair, on_divergence=DivergencePolicy.RETURN_DIVERGED)
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED
    assert result.report.poisoned
