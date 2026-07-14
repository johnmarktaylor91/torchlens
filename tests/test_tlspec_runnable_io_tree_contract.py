"""Input/output-tree container contract honesty for sparse runnable execution.

Round-5 hardening. The round-4 fix covered opaque *values*, but the input/output
tree contract still mishandled whole classes of container shapes:

* F1 -- non-tensor input leaves under non-``str``/``int`` mapping keys (tuple, bool,
  float, enum, ...) were silently skipped, so a changed scalar under such a key
  false-verified the wrong replayed path; and the single-tensor arity shortcut
  crashed on the identical captured input once such a dict was present.
* F2 -- a torch structseq (``torch.return_types.*``) nested inside another
  container crashed reconstruction.
* F3 -- a capture with both positional tensors and keyword leaves had no
  representable ``run(inputs=)`` binding.
* F4 -- a ``bool`` output-dict key passed producer preflight but the loader
  rejected it, so the artifact advertised runnable then lost it on load.
* F5 -- a ``numpy.float64`` literal falsely diverged on the identical input
  (recorded round-trips to plain ``float`` while ``_literal_leaf_equal`` was
  type-strict).

The unifying principle: a non-tensor leaf is either WITNESSED correctly, or its
subtree is marked OPAQUE -> the run reports UNVERIFIABLE (never silently
skipped); reconstruction/binding never spuriously crashes or diverges on the
IDENTICAL captured input, while a genuinely CHANGED leaf still diverges.
"""

from __future__ import annotations

import enum
from pathlib import Path
from typing import Any, NamedTuple

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


def _save_runnable(
    model: nn.Module,
    capture_inputs: Any,
    path: Path,
    *,
    input_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Capture (unchanged fast oracle) and save a runnable artifact."""

    trace = tl.trace(
        model,
        capture_inputs,
        input_kwargs=input_kwargs or {},
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable", include_activations=True)
    return path


# ---------------------------------------------------------------------------
# F1 -- dict leaves under exotic keys must be witnessed or opaque, never skipped
# ---------------------------------------------------------------------------


class _TupleKeyBranch(nn.Module):
    """Branch on an encodable scalar stored under a tuple mapping key."""

    def forward(self, x: torch.Tensor, options: dict[Any, int]) -> torch.Tensor:
        if options[("mode", 1)] == 0:
            return x + 1.0
        return x * 10.0


class _BoolKeyBranch(nn.Module):
    """Branch on an encodable scalar stored under a bool mapping key."""

    def forward(self, x: torch.Tensor, options: dict[Any, int]) -> torch.Tensor:
        if options[True] == 0:
            return x + 1.0
        return x * 10.0


class _FloatKeyBranch(nn.Module):
    """Branch on an encodable scalar stored under a float mapping key."""

    def forward(self, x: torch.Tensor, options: dict[Any, int]) -> torch.Tensor:
        if options[1.5] == 0:
            return x + 1.0
        return x * 10.0


class _EnumKey(enum.Enum):
    FAST = enum.auto()


class _EnumKeyBranch(nn.Module):
    """Branch on a scalar stored under an OPAQUE enum mapping key."""

    def forward(self, x: torch.Tensor, options: dict[Any, int]) -> torch.Tensor:
        if options[_EnumKey.FAST] == 0:
            return x + 1.0
        return x * 10.0


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("model_cls", "key"),
    [
        (_TupleKeyBranch, ("mode", 1)),
        (_BoolKeyBranch, True),
        (_FloatKeyBranch, 1.5),
    ],
)
def test_encodable_exotic_key_scalar_diverges_on_change(
    tmp_path: Path, model_cls: type[nn.Module], key: Any
) -> None:
    """A changed scalar under a tuple/bool/float key must DIVERGE, not false-verify."""

    model = model_cls()
    x = torch.tensor([2.0, 3.0])
    path = _save_runnable(model, [x, {key: 0}], tmp_path / "exotic_key.tlspec")

    # Unchanged -> honest VERIFIED.
    same = tl.load(path).run(inputs=[x, {key: 0}])
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(same.output, model(x, {key: 0}))

    # Changed hidden scalar -> the recorded path is stale, must diverge.
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[x, {key: 1}])

    diverged = tl.load(path).run(
        inputs=[x, {key: 1}], on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert diverged.report.poisoned


@pytest.mark.smoke
def test_opaque_enum_key_subtree_is_unverifiable_not_skipped(tmp_path: Path) -> None:
    """A scalar under an OPAQUE enum key must downgrade to UNVERIFIABLE, never skip."""

    model = _EnumKeyBranch()
    x = torch.tensor([2.0, 3.0])
    path = _save_runnable(model, [x, {_EnumKey.FAST: 0}], tmp_path / "enum_key.tlspec")

    # Even the unchanged identical input cannot be proven -> UNVERIFIABLE, not VERIFIED.
    result = tl.load(path).run(
        inputs=[x, {_EnumKey.FAST: 0}], on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert result.report.poisoned


@pytest.mark.smoke
def test_single_tensor_plus_exotic_key_dict_identical_no_crash(tmp_path: Path) -> None:
    """The arity shortcut must not misfire for single-tensor + exotic-key-dict inputs."""

    class _Scale(nn.Module):
        def forward(self, value: torch.Tensor, cfg: dict[Any, float]) -> torch.Tensor:
            return value * cfg[(1, 2)]

    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    path = _save_runnable(_Scale(), [t, {(1, 2): 2.0}], tmp_path / "single_tensor.tlspec")

    result = tl.load(path).run(inputs=[t, {(1, 2): 2.0}])
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.allclose(result.output, t * 2.0)


# ---------------------------------------------------------------------------
# F2 -- nested torch structseq reconstructs at any container depth
# ---------------------------------------------------------------------------


class _WrappedResult(NamedTuple):
    stats: Any
    shifted: torch.Tensor


class _NestedStructSeq(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def forward(self, x: torch.Tensor) -> Any:
        stats = torch.max(x, dim=0)
        shifted = x + 1
        if self.kind == "dict":
            return {"stats": stats, "shifted": shifted}
        if self.kind == "list":
            return [stats, shifted]
        return _WrappedResult(stats, shifted)


@pytest.mark.smoke
@pytest.mark.parametrize("kind", ["dict", "list", "namedtuple"])
def test_nested_structseq_runs_verified(tmp_path: Path, kind: str) -> None:
    """A structseq nested in a dict/list/namedtuple runs VERIFIED with output == live."""

    model = _NestedStructSeq(kind)
    x = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
    path = _save_runnable(model, x, tmp_path / f"nested_{kind}.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED

    live = model(x)
    if kind == "dict":
        assert torch.equal(result.output["stats"].values, live["stats"].values)
        assert torch.equal(result.output["stats"].indices, live["stats"].indices)
        assert torch.equal(result.output["shifted"], live["shifted"])
    elif kind == "list":
        assert torch.equal(result.output[0].values, live[0].values)
        assert torch.equal(result.output[1], live[1])
    else:
        assert torch.equal(result.output.stats.values, live.stats.values)
        assert torch.equal(result.output.shifted, live.shifted)


# ---------------------------------------------------------------------------
# F3 -- mixed positional + keyword inputs have a representable run(inputs=) form
# ---------------------------------------------------------------------------


class _MixedInputs(nn.Module):
    def forward(self, x: torch.Tensor, *, add: bool) -> torch.Tensor:
        if add:
            return x + 1
        return x * 10


@pytest.mark.smoke
def test_mixed_positional_keyword_inputs_bind_and_diverge(tmp_path: Path) -> None:
    """A combined ``{'args': [...], 'kwargs': {...}}`` binds; a changed kwarg diverges."""

    model = _MixedInputs()
    x = torch.tensor([2.0, 3.0])
    path = _save_runnable(model, [x], tmp_path / "mixed.tlspec", input_kwargs={"add": True})

    # Unchanged mixed input -> VERIFIED, output == live.
    result = tl.load(path).run(inputs={"args": [x], "kwargs": {"add": True}})
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(result.output, x + 1)

    # Changed keyword leaf -> the recorded taken path (add branch) is stale.
    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs={"args": [x], "kwargs": {"add": False}})


@pytest.mark.smoke
def test_live_mixed_positional_keyword_inputs_bind_and_diverge() -> None:
    """Live ``run`` accepts the same mixed-input spelling as loaded runnable traces."""

    model = _MixedInputs()
    x = torch.tensor([2.0, 3.0])
    trace = tl.trace(
        model,
        [x],
        input_kwargs={"add": True},
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )

    result = trace.run(inputs={"args": [x], "kwargs": {"add": True}})
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(result.output, x + 1)

    with pytest.raises(ValueError, match="computational graph changed"):
        trace.run(inputs={"args": [x], "kwargs": {"add": False}})


# ---------------------------------------------------------------------------
# F4 -- bool output-dict key agrees between producer and loader
# ---------------------------------------------------------------------------


class _BoolOutputKey(nn.Module):
    def forward(self, x: torch.Tensor) -> dict[Any, torch.Tensor]:
        return {True: x + 1}


@pytest.mark.smoke
def test_bool_output_dict_key_stays_runnable(tmp_path: Path) -> None:
    """A ``{True: ...}`` output must not advertise runnable then lose it on load."""

    model = _BoolOutputKey()
    x = torch.tensor([1.0, 2.0])
    path = _save_runnable(model, x, tmp_path / "bool_out.tlspec")

    result = tl.load(path).run(inputs=x.clone())
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    live = model(x)
    assert set(result.output) == set(live)
    assert torch.equal(result.output[True], live[True])


# ---------------------------------------------------------------------------
# F5 -- numpy.float64 literal saves, loads, and verifies like a Python float
# ---------------------------------------------------------------------------


class _Scale(nn.Module):
    def forward(self, v: torch.Tensor, s: Any) -> torch.Tensor:
        return v * s


@pytest.mark.smoke
def test_numpy_float64_literal_identical_verifies_changed_diverges(tmp_path: Path) -> None:
    """A ``numpy.float64`` op literal must save, load, and verify normally."""

    np = pytest.importorskip("numpy")
    model = _Scale()
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    path = _save_runnable(model, [t, np.float64(2.0)], tmp_path / "np_scalar.tlspec")

    # NumPy scalar metadata is normalized before it reaches the bundle pickle,
    # so the normal safe loader accepts the artifact without a trust opt-in.
    identical = tl.load(path).run(inputs=[t, np.float64(2.0)])
    assert identical.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.allclose(identical.output, t * 2.0)

    # A plain float with the SAME value must also verify (value-equality across
    # numeric float subclasses).
    plain = tl.load(path).run(inputs=[t, 2.0])
    assert plain.report.path_faithfulness is PathFaithfulness.VERIFIED

    # A changed numeric value must still diverge honestly.
    changed = tl.load(path).run(
        inputs=[t, np.float64(9.0)], on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert changed.report.path_faithfulness is not PathFaithfulness.VERIFIED
    assert changed.report.poisoned


# ---------------------------------------------------------------------------
# F6 -- NumPy scalar control inputs follow the Python-scalar divergence contract
# ---------------------------------------------------------------------------


class _IntegerControlBranch(nn.Module):
    """Choose one recorded arm from an integer control input."""

    def forward(self, value: torch.Tensor, mode: int) -> torch.Tensor:
        """Return distinct values for the two control-flow arms."""

        if mode == 0:
            return value + 1
        return value * 10


@pytest.mark.smoke
def test_numpy_scalar_control_input_diverges_like_python_int(tmp_path: Path) -> None:
    """Changed NumPy integer controls must diverge rather than become unverifiable."""

    np = pytest.importorskip("numpy")
    value = torch.tensor([2.0, 3.0])
    path = _save_runnable(
        _IntegerControlBranch(),
        [value, np.int64(0)],
        tmp_path / "numpy_control.tlspec",
    )

    unchanged = tl.load(path).run(inputs=[value, np.int64(0)])
    assert unchanged.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(unchanged.output, value + 1)

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[value, np.int64(1)])

    diverged = tl.load(path).run(
        inputs=[value, np.int64(1)],
        on_divergence=DivergencePolicy.RETURN_DIVERGED,
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.poisoned
