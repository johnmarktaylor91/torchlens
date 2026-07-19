"""Exhaustive (source x mechanism x use) escape-witness fail-closed matrix.

This is the anti-whack-a-mole guard for the unified runnable escape witness. It
enumerates the whole matrix of tensor->host escapes and asserts the honesty tripwire
for EVERY cell:

* SOURCE class -- model INPUT, INTERNAL op output, BOUND param, BOUND buffer, and
  UNBOUND param/buffer.
* ESCAPE mechanism -- census-VISIBLE ``.item()`` / ``float()`` / ``__index__``
  (``aten._local_scalar_dense``) AND census-INVISIBLE ``.tolist()`` / ``.numpy()``
  (dispatcher-invisible; observed at the torch-function layer).
* USE -- baked as an op-arg literal VERBATIM, baked after HOST ARITHMETIC, and pure-
  Python CONTROL flow.

Honest rule per cell: a run on the ORIGINAL input + capture-equivalent state is
VERIFIED; a run on a CHANGED input or CHANGED staged state whose escape SOURCE would
differ is never VERIFIED (UNVERIFIABLE or DIVERGED). A model with NO escape stays
VERIFIED on any input (no over-trigger). The single documented fail-closed exception is
an UNATTRIBUTABLE pruned bool control predicate (a ``.data``-alias branch covered by no
net), which is honestly UNVERIFIABLE even on the original -- exactly like a pruned-RNG
control escape.
"""

from __future__ import annotations

import ctypes
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens as _torchlens
from torchlens.errors import RunnablePreflightError
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, RunnableErrorCode

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
    random_seed=0,
)


def _save_load(model: nn.Module, cap_x: torch.Tensor, tmp: Path, *, acts: bool = False) -> tl.Trace:
    """Capture, save a runnable ``.tlspec`` with weights, and reload it."""
    trace = tl.trace(model, cap_x, capture=_CAPTURE)
    path = tmp / "escape.tlspec"
    shutil.rmtree(path, ignore_errors=True)
    trace.save(path, level="runnable", include_weights=True, include_activations=acts)
    return tl.load(path)


# --------------------------------------------------------------------------------------
# INPUT-source and INTERNAL-source cells: the escape SOURCE changes with the model input.
# --------------------------------------------------------------------------------------


class _InputItemArith(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + x.item() * 2.0


class _InputItemVerbatim(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(min=x.item())


class _InputFloatControl(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if float(x) > 0.0:
            return x + 10.0
        return x - 10.0


class _InputTolistArith(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + sum(x.tolist())


class _InputNumpyArith(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + float(x.detach().numpy().sum())


class _InputNumpyReconstruct(nn.Module):
    # C1 seam: census-invisible numpy->tolist reconstruction of the raw input.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.tensor(x.numpy().tolist())


class _InternalItemArith(nn.Module):
    def __init__(self) -> None:
        torch.manual_seed(0)
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return h * h.flatten()[0].item()


class _InternalTolistArith(nn.Module):
    def __init__(self) -> None:
        torch.manual_seed(0)
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return h * sum(h[0].tolist())


class _InternalNumpyArith(nn.Module):
    def __init__(self) -> None:
        torch.manual_seed(0)
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        return h + float(h.detach().numpy().sum())


class _InternalFloatControl(nn.Module):
    def __init__(self) -> None:
        torch.manual_seed(0)
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        if float(h.flatten()[0]) > 0.0:
            return h + 5.0
        return h - 5.0


class _InputEqualControl(nn.Module):
    # aten.equal returns a raw Python bool DIRECTLY (no _local_scalar_dense) -> the general
    # tensor->non-tensor census rule is what witnesses the input source here.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.equal(x, x.abs()):
            return x * 2.0
        return x + 7.0


class _InputEqualValueBake(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + float(torch.equal(x, x.abs()))


class _InputAllcloseControl(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.allclose(x, x.round()):
            return x * 3.0
        return x - 4.0


class _InputIsNonzeroControl(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_nonzero(x.sum()):
            return x + 1.0
        return x - 1.0


class _InputDlpackArith(nn.Module):
    # Tensor.__dlpack__ (via np.from_dlpack) is dispatcher-invisible AND bypasses the
    # method patch unless __dlpack__ is patched: the scoped patch records the source.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * float(np.from_dlpack(x.detach())[0])


_ONE = torch.tensor([1.0])
_ONE_CHANGED = torch.tensor([9.0])
_TWO = torch.tensor([1.0, 2.0])
_TWO_CHANGED = torch.tensor([10.0, 20.0])
_VEC_IN = torch.ones(2, 4)
_VEC_IN_CHANGED = torch.full((2, 4), 3.0)
_EQ_CAP = torch.tensor([2.0, 2.0])
_EQ_CHANGED = torch.tensor([-10.0, -10.0])
_ROUND_CAP = torch.tensor([2.0, 3.0])
_ROUND_CHANGED = torch.tensor([2.5, 3.5])

# (name, model_factory, capture_input, changed_input)
_INPUT_CELLS: list[tuple[str, Callable[[], nn.Module], torch.Tensor, torch.Tensor]] = [
    ("input/item/arith", _InputItemArith, _ONE, _ONE_CHANGED),
    ("input/item/verbatim", _InputItemVerbatim, _ONE, _ONE_CHANGED),
    ("input/float/control", _InputFloatControl, _ONE, torch.tensor([-9.0])),
    ("input/tolist/arith", _InputTolistArith, _TWO, _TWO_CHANGED),
    ("input/numpy/arith", _InputNumpyArith, _TWO, _TWO_CHANGED),
    ("input/numpy_reconstruct/arith", _InputNumpyReconstruct, _TWO, _TWO_CHANGED),
    # r12: tensor->non-tensor host escapes that never emit _local_scalar_dense.
    ("input/equal/control", _InputEqualControl, _EQ_CAP, _EQ_CHANGED),
    ("input/equal/value_bake", _InputEqualValueBake, _EQ_CAP, _EQ_CHANGED),
    ("input/allclose/control", _InputAllcloseControl, _ROUND_CAP, _ROUND_CHANGED),
    ("input/is_nonzero/control", _InputIsNonzeroControl, _TWO, torch.tensor([1.0, -1.0])),
    # r12: dlpack zero-copy export (__dlpack__ method patch).
    ("input/dlpack/arith", _InputDlpackArith, _TWO, _TWO_CHANGED),
    ("internal/item/arith", _InternalItemArith, _VEC_IN, _VEC_IN_CHANGED),
    ("internal/tolist/arith", _InternalTolistArith, _VEC_IN, _VEC_IN_CHANGED),
    ("internal/numpy/arith", _InternalNumpyArith, _VEC_IN, _VEC_IN_CHANGED),
    ("internal/float/control", _InternalFloatControl, _VEC_IN, -_VEC_IN_CHANGED),
]


# The internal/float/control cell intentionally does float() on a requires_grad activation to
# exercise a value escape into control flow; torch's "converting to scalar" UserWarning is expected
# noise here, not a torchlens signal, so it must not escalate under filterwarnings=error.
@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")
@pytest.mark.parametrize(
    "name,factory,cap_x,changed_x", _INPUT_CELLS, ids=[c[0] for c in _INPUT_CELLS]
)
def test_input_escape_cell_changed_input_is_never_verified(
    name: str,
    factory: Callable[[], nn.Module],
    cap_x: torch.Tensor,
    changed_x: torch.Tensor,
    tmp_path: Path,
) -> None:
    """A changed input restales the escape source -> never VERIFIED."""
    loaded = _save_load(factory(), cap_x, tmp_path)
    result = loaded.run(inputs=changed_x, seed=0, on_divergence="return_diverged")
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


@pytest.mark.filterwarnings("ignore:Converting a tensor with requires_grad")
@pytest.mark.parametrize(
    "name,factory,cap_x,changed_x", _INPUT_CELLS, ids=[c[0] for c in _INPUT_CELLS]
)
def test_input_escape_cell_original_input_is_verified(
    name: str,
    factory: Callable[[], nn.Module],
    cap_x: torch.Tensor,
    changed_x: torch.Tensor,
    tmp_path: Path,
) -> None:
    """The original input recomputes the byte-identical escape source -> VERIFIED."""
    loaded = _save_load(factory(), cap_x, tmp_path)
    result = loaded.run(inputs=cap_x, seed=0, on_divergence="return_diverged")
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# STATE-source cells: the escape SOURCE changes with staged state (load_state_dict).
# --------------------------------------------------------------------------------------


class _BoundParamItemArith(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.thr = nn.Parameter(torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.thr + self.thr.item()


class _BoundBufferItemArith(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.tensor(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.b + self.b.item()


class _BoundBufferItemControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("gate", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * self.gate
        if self.gate.item() > 0.5:
            return h + 10.0
        return h - 10.0


class _BoundBufferNumpyArith(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.b + float(self.b.numpy().sum())


class _UnboundBufferItemControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("gate", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ``gate`` feeds NO traced op -- it steers control only through the host.
        if self.gate.item() > 0.5:
            return x + 10.0
        return x - 10.0


class _UnboundParamFloatControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.thr = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if float(self.thr) > 0.5:
            return x + 7.0
        return x - 7.0


_STATE_X = torch.tensor([3.0])
_STATE_X_VEC = torch.ones(2)

# (name, model_factory, input, changed_state_dict)
_STATE_CELLS: list[tuple[str, Callable[[], nn.Module], torch.Tensor, dict[str, torch.Tensor]]] = [
    ("bound_param/item/arith", _BoundParamItemArith, _STATE_X, {"thr": torch.tensor(5.0)}),
    ("bound_buffer/item/arith", _BoundBufferItemArith, _STATE_X, {"b": torch.tensor(5.0)}),
    (
        "bound_buffer/item/control",
        _BoundBufferItemControl,
        _STATE_X_VEC,
        {"gate": torch.tensor(0.0)},
    ),
    (
        "bound_buffer/numpy/arith",
        _BoundBufferNumpyArith,
        torch.tensor([3.0, 4.0]),
        {"b": torch.tensor([5.0, 6.0])},
    ),
    (
        "unbound_buffer/item/control",
        _UnboundBufferItemControl,
        _STATE_X_VEC,
        {"gate": torch.tensor(0.0)},
    ),
    (
        "unbound_param/float/control",
        _UnboundParamFloatControl,
        _STATE_X_VEC,
        {"thr": torch.tensor(0.0)},
    ),
]


@pytest.mark.parametrize(
    "name,factory,x,changed_state", _STATE_CELLS, ids=[c[0] for c in _STATE_CELLS]
)
def test_state_escape_cell_changed_state_is_never_verified(
    name: str,
    factory: Callable[[], nn.Module],
    x: torch.Tensor,
    changed_state: dict[str, torch.Tensor],
    tmp_path: Path,
) -> None:
    """A changed staged param/buffer restales the escape source -> never VERIFIED."""
    loaded = _save_load(factory(), x, tmp_path)
    loaded.load_state_dict(changed_state)
    result = loaded.run(inputs=x, seed=0, on_divergence="return_diverged")
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


@pytest.mark.parametrize(
    "name,factory,x,changed_state", _STATE_CELLS, ids=[c[0] for c in _STATE_CELLS]
)
def test_state_escape_cell_capture_equivalent_state_is_verified(
    name: str,
    factory: Callable[[], nn.Module],
    x: torch.Tensor,
    changed_state: dict[str, torch.Tensor],
    tmp_path: Path,
) -> None:
    """The embedded capture-equivalent state re-digests identically -> VERIFIED."""
    loaded = _save_load(factory(), x, tmp_path)
    result = loaded.run(inputs=x, seed=0, on_divergence="return_diverged")
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# Documented fail-closed exception: an UNATTRIBUTABLE pruned bool control predicate.
# --------------------------------------------------------------------------------------


class _DataAliasBoolControl(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("gate", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x * self.gate
        # ``.data`` severs the graph link; the gt predicate is orphan-pruned. r37
        # mechanism A resolves its LEAF origins ({state: gate}) so the predicate is
        # positively witnessed by the gate's capture-time state digest.
        if bool(self.gate.data > 0.5):
            return h + 10.0
        return h - 10.0


def test_pruned_data_bool_predicate_is_state_witnessed(tmp_path: Path) -> None:
    """r37: the pruned ``.data`` bool predicate is witnessed by its state leaf origin.

    Pre-r37 this class was a documented fail-closed exception (unattributable ->
    UNVERIFIABLE even on the original). Origin propagation attributes it positively:
    the ORIGINAL state re-digests identically -> VERIFIED (honest recovery), while a
    CHANGED staged state restales the digest -> never VERIFIED (the tripwire half,
    unchanged).
    """
    loaded = _save_load(_DataAliasBoolControl(), _STATE_X_VEC, tmp_path)
    original = loaded.run(inputs=_STATE_X_VEC, seed=0, on_divergence="return_diverged")
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    changed = _save_load(_DataAliasBoolControl(), _STATE_X_VEC, tmp_path)
    changed.load_state_dict({"gate": torch.tensor(0.0)})
    changed_result = changed.run(inputs=_STATE_X_VEC, seed=0, on_divergence="return_diverged")
    assert changed_result.report.path_faithfulness is not PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# No over-trigger: an escape-free deterministic model stays VERIFIED on any input.
# --------------------------------------------------------------------------------------


class _CleanLinear(nn.Module):
    def __init__(self) -> None:
        torch.manual_seed(0)
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).relu()


class _CleanElementwise(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * 2.0 + 1.0).tanh()


class _ShapeReadClean(nn.Module):
    """r12 regression guard: a model that reads tensor SHAPE/METADATA constantly but has
    NO value escape must stay VERIFIED. ``x.size()`` / ``x.shape`` / ``x.numel()`` /
    ``x.dim()`` return non-tensor host values, but they derive from LAYOUT, not data
    values, so the narrowed value-escape rule must NOT witness them (the over-broad
    "any non-tensor output = escape" rule falsely downgraded this class to UNVERIFIABLE
    AND pathologically slowed capture)."""

    def __init__(self) -> None:
        torch.manual_seed(0)
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        assert x.dim() == 2 and x.numel() == b * x.shape[1]
        h = self.lin(x).relu()
        return h.reshape(b, -1) * float(h.shape[1])


_ESCAPE_FREE: list[tuple[str, Callable[[], nn.Module], torch.Tensor, torch.Tensor]] = [
    ("clean_linear", _CleanLinear, _VEC_IN, _VEC_IN_CHANGED),
    ("clean_elementwise", _CleanElementwise, _TWO, _TWO_CHANGED),
    ("shape_read_clean", _ShapeReadClean, _VEC_IN, _VEC_IN_CHANGED),
]


@pytest.mark.parametrize(
    "name,factory,cap_x,changed_x", _ESCAPE_FREE, ids=[c[0] for c in _ESCAPE_FREE]
)
def test_escape_free_model_is_verified_on_any_input(
    name: str,
    factory: Callable[[], nn.Module],
    cap_x: torch.Tensor,
    changed_x: torch.Tensor,
    tmp_path: Path,
) -> None:
    """An escape-free model never over-triggers the escape witness."""
    original = _save_load(factory(), cap_x, tmp_path).run(inputs=cap_x, seed=0)
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    changed = _save_load(factory(), cap_x, tmp_path).run(inputs=changed_x, seed=0)
    assert changed.report.path_faithfulness is PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# r12 F-H2: the internal-vs-user filter is allowlist-BY-CONSTRUCTION (an explicit marker),
# not a spoofable co_filename stack inference. A user escape helper carrying a torchlens
# co_filename is STILL a user escape and must be witnessed.
# --------------------------------------------------------------------------------------


def _make_fake_internal_model() -> nn.Module:
    """Build a model whose escape helper's frame resolves inside the torchlens package."""
    namespace: dict[str, object] = {}
    exec(  # noqa: S102 - test-only construction of a spoofed co_filename frame
        compile(
            "def sneaky(t):\n    return t.sum().item()\n",
            _torchlens.__file__,
            "exec",
        ),
        namespace,
    )
    sneaky = namespace["sneaky"]

    class _FakeInternalEscape(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Transformed bake defeats any value-equality net; only witnessing the SOURCE
            # keeps this honest, which requires NOT classifying the escape as internal.
            return x * (sneaky(x) * 3.0 + 1.0)  # type: ignore[operator]

    return _FakeInternalEscape()


def test_fake_internal_escape_is_witnessed_not_classified_internal(tmp_path: Path) -> None:
    """A spoofed-``co_filename`` user escape is recorded, so a changed input is not VERIFIED."""
    original = _save_load(_make_fake_internal_model(), _TWO, tmp_path).run(inputs=_TWO, seed=0)
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    changed = _save_load(_make_fake_internal_model(), _TWO, tmp_path).run(
        inputs=_TWO_CHANGED, seed=0, on_divergence="return_diverged"
    )
    assert changed.report.path_faithfulness is not PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# r12 F-C2: a non-mutating identity return (x.cpu() on CPU) carries a capture label but is
# NOT in-place; it must save+run without a false MUTATION_VERSION_MISMATCH, while genuine
# in-place ops are still detected and replay faithfully.
# --------------------------------------------------------------------------------------


class _IdentityCpu(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.cpu()


class _IdentityContiguous(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous() + 1.0


class _GenuineInplaceAdd(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 1.0
        y.add_(2.0)
        return y


class _GenuineInplaceCopy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x * 1.0
        y.copy_(x + 5.0)
        return y


class _InplaceOnRawInput(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x.mul_(3.0)
        return x + 1.0


_INPLACE_CELLS: list[tuple[str, Callable[[], nn.Module], torch.Tensor, torch.Tensor]] = [
    ("identity/cpu", _IdentityCpu, _TWO, _TWO_CHANGED),
    ("identity/contiguous", _IdentityContiguous, _TWO, _TWO_CHANGED),
    ("inplace/add_", _GenuineInplaceAdd, _TWO, _TWO_CHANGED),
    ("inplace/copy_", _GenuineInplaceCopy, _TWO, _TWO_CHANGED),
    ("inplace/raw_input_mul_", _InplaceOnRawInput, _TWO, _TWO_CHANGED),
]


@pytest.mark.parametrize(
    "name,factory,cap_x,changed_x", _INPLACE_CELLS, ids=[c[0] for c in _INPLACE_CELLS]
)
def test_identity_and_inplace_ops_run_without_false_mutation_mismatch(
    name: str,
    factory: Callable[[], nn.Module],
    cap_x: torch.Tensor,
    changed_x: torch.Tensor,
    tmp_path: Path,
) -> None:
    """Identity returns and genuine in-place ops both save+run and VERIFY (no false crash)."""
    original = _save_load(factory(), cap_x, tmp_path).run(inputs=cap_x, seed=0)
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    changed = _save_load(factory(), cap_x, tmp_path).run(inputs=changed_x, seed=0)
    assert changed.report.path_faithfulness is PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# R13-H1: a genuine in-place op through an UNLABELLED ``.data`` alias mutates storage the
# sparse DAG cannot model; the op is orphan-pruned and the write silently dropped. The run
# must NEVER report VERIFIED with the mutation lost -- it is honestly UNVERIFIABLE on both the
# original and a changed input. The whole in-place family (add_/mul_/copy_/masked_fill_) is
# covered. A genuine in-place on a LABELLED alias stays VERIFIED (covered above).
# --------------------------------------------------------------------------------------


class _DataAliasAdd(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.data.add_(5.0)
        return y * 2.0


class _DataAliasMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.data.mul_(3.0)
        return y * 2.0


class _DataAliasCopy(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.data.copy_(torch.tensor([9.0, 9.0, 9.0, 9.0]))
        return y * 2.0


class _DataAliasMaskedFill(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.data.masked_fill_(torch.tensor([True, False, True, False]), 7.0)
        return y * 2.0


_DATA_ALIAS_MUTATION_CELLS: list[tuple[str, Callable[[], nn.Module]]] = [
    ("data.add_", _DataAliasAdd),
    ("data.mul_", _DataAliasMul),
    ("data.copy_", _DataAliasCopy),
    ("data.masked_fill_", _DataAliasMaskedFill),
]


@pytest.mark.parametrize(
    "name,factory", _DATA_ALIAS_MUTATION_CELLS, ids=[c[0] for c in _DATA_ALIAS_MUTATION_CELLS]
)
def test_data_alias_inplace_mutation_is_never_verified_with_dropped_write(
    name: str, factory: Callable[[], nn.Module], tmp_path: Path
) -> None:
    """A dropped ``.data``-alias mutation is UNVERIFIABLE, never a false VERIFIED-with-wrong-output."""
    cap_x = torch.tensor([1.0, 0.5, 2.0, 0.25])
    changed_x = torch.tensor([3.0, 4.0, 5.0, 6.0])
    for run_x in (cap_x, changed_x):
        result = _save_load(factory(), cap_x.clone(), tmp_path).run(
            inputs=run_x.clone(), seed=0, on_divergence="return_diverged"
        )
        assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# R13-H2: a host WRITE-BACK through a mutable zero-copy ``.numpy()`` / ``__array__`` alias
# mutates the source bytes with no dispatch and no version bump, so the sparse replay
# recomputes the pre-write value. The run must be UNVERIFIABLE (not a false VERIFIED) even on
# the original input, while a READ-only conversion stays VERIFIED (no over-trigger).
# --------------------------------------------------------------------------------------


class _NumpyViewWriteBack(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.detach().numpy()[0] = 99.0
        return y * 2.0


class _ArrayViewWriteBack(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        np.asarray(y.detach())[0] = 99.0
        return y * 2.0


class _NumpyReadOnly(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        return y + float(y.detach().numpy().sum())


@pytest.mark.parametrize(
    "factory", [_NumpyViewWriteBack, _ArrayViewWriteBack], ids=["numpy", "__array__"]
)
def test_host_writeback_through_mutable_alias_is_never_verified(
    factory: Callable[[], nn.Module], tmp_path: Path
) -> None:
    """A host write-back through a mutable numpy/__array__ alias is UNVERIFIABLE on the original."""
    x = torch.tensor([1.0, 0.5, 2.0, 0.25])
    result = _save_load(factory(), x.clone(), tmp_path).run(
        inputs=x.clone(), seed=0, on_divergence="return_diverged"
    )
    assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


def test_readonly_numpy_escape_stays_verified_on_original(tmp_path: Path) -> None:
    """A READ-only numpy conversion (no write-back) still VERIFIES on the original input."""
    x = torch.tensor([1.0, 0.5, 2.0, 0.25])
    result = _save_load(_NumpyReadOnly(), x.clone(), tmp_path).run(inputs=x.clone(), seed=0)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED


# --------------------------------------------------------------------------------------
# R13-H3: ``torch.utils.dlpack.to_dlpack`` == ``torch._C._to_dlpack`` is a C binding that
# bypasses the ``Tensor.__dlpack__`` method patch. Its exported tensor VALUE can be baked into
# a downstream literal, so the module function must be observed as a host escape: a changed
# input whose exported source differs is never VERIFIED.
# --------------------------------------------------------------------------------------


class _DlpackCapsuleShim:
    # Wrap the raw capsule so np.from_dlpack accepts it (pure host code, no dispatch).
    def __init__(self, capsule: object) -> None:
        self._capsule = capsule

    def __dlpack__(self, **kwargs: object) -> object:
        return self._capsule

    def __dlpack_device__(self) -> tuple[int, int]:
        return (1, 0)


class _ToDlpackExport(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        capsule = torch.utils.dlpack.to_dlpack(y.detach())
        arr = np.from_dlpack(_DlpackCapsuleShim(capsule))
        return x + float(arr.sum())


def test_to_dlpack_c_binding_export_is_witnessed_as_escape(tmp_path: Path) -> None:
    """``to_dlpack`` (C binding) is observed as a host escape: a changed input is never VERIFIED."""
    cap_x = torch.tensor([1.0, 0.5, 2.0, 0.25])
    changed_x = torch.tensor([3.0, 4.0, 5.0, 6.0])
    original = _save_load(_ToDlpackExport(), cap_x.clone(), tmp_path).run(
        inputs=cap_x.clone(), seed=0, on_divergence="return_diverged"
    )
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    changed = _save_load(_ToDlpackExport(), cap_x.clone(), tmp_path).run(
        inputs=changed_x.clone(), seed=0, on_divergence="return_diverged"
    )
    assert changed.report.path_faithfulness is not PathFaithfulness.VERIFIED


# ======================================================================================
# R14 mutation-residual regressions (cross-lab confirmed): host WRITE / mutation paths
# that slipped the r13 detection and would false-VERIFY with the wrong output (H1/H2/H3),
# plus a storage-rebinding op that crashed at run (C3). Each must be UNVERIFIABLE (or
# save-refused), NEVER VERIFIED-with-wrong-output; read-only exposures must NOT
# over-trigger, and genuine tracked in-place must still replay + VERIFY.
# ======================================================================================

_R14_X = torch.tensor([1.0, 0.5, 2.0, 0.25])
_R14_CHANGED = torch.tensor([3.0, 4.0, 5.0, 6.0])


# --- R14-H1: a host numpy/.data write-back on a source ALSO mutated by a TRACKED in-place op
# bumps the version, which pre-r14 skipped the byte compare -> false VERIFIED. All three
# orderings (write-back before / between / after tracked in-place) must be UNVERIFIABLE.
class _WritebackThenInplace(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.detach().numpy()[0] = 99.0
        y.add_(2.0)
        return y * 2.0


class _WritebackBetweenInplace(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.add_(2.0)
        y.detach().numpy()[0] = 99.0
        y.mul_(3.0)
        return y * 2.0


class _LateWriteViaKeptAlias(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        arr = y.detach().numpy()
        y.add_(2.0)
        arr[0] = 99.0  # host write via a kept alias, AFTER a tracked version bump
        return y * 2.0


_H1_CELLS: list[tuple[str, Callable[[], nn.Module]]] = [
    ("writeback_then_inplace", _WritebackThenInplace),
    ("writeback_between_inplace", _WritebackBetweenInplace),
    ("late_write_via_kept_alias", _LateWriteViaKeptAlias),
]


@pytest.mark.parametrize("name,factory", _H1_CELLS, ids=[c[0] for c in _H1_CELLS])
def test_r14_h1_writeback_with_tracked_inplace_is_never_verified(
    name: str, factory: Callable[[], nn.Module], tmp_path: Path
) -> None:
    """A numpy write-back layered with a tracked in-place op is UNVERIFIABLE (both inputs)."""
    for run_x in (_R14_X, _R14_CHANGED):
        result = _save_load(factory(), _R14_X.clone(), tmp_path).run(
            inputs=run_x.clone(), seed=0, on_divergence="return_diverged"
        )
        assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


# --- R14-H2: a mutation through a LABELLED VIEW of the invisible .data alias is orphan-pruned
# yet was not recorded (the r13 flag keyed on an unlabelled target). Must be UNVERIFIABLE.
class _DataViewSetitem(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.data.view(-1)[0] = 9.0
        return y * 2.0


class _DataReshapeSetitem(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.data.reshape(-1)[0] = 9.0
        return y * 2.0


class _DataFlattenSetitem(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.data.flatten()[0] = 9.0
        return y * 2.0


_H2_CELLS: list[tuple[str, Callable[[], nn.Module]]] = [
    ("data.view.setitem", _DataViewSetitem),
    ("data.reshape.setitem", _DataReshapeSetitem),
    ("data.flatten.setitem", _DataFlattenSetitem),
]


@pytest.mark.parametrize("name,factory", _H2_CELLS, ids=[c[0] for c in _H2_CELLS])
def test_r14_h2_labelled_view_of_data_alias_mutation_is_never_verified(
    name: str, factory: Callable[[], nn.Module], tmp_path: Path
) -> None:
    """A dropped mutation through a labelled view of .data is UNVERIFIABLE (both inputs)."""
    for run_x in (_R14_X, _R14_CHANGED):
        result = _save_load(factory(), _R14_X.clone(), tmp_path).run(
            inputs=run_x.clone(), seed=0, on_divergence="return_diverged"
        )
        assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


# --- R14-H3: a host write through a storage-pointer bridge (untyped_storage / storage /
# data_ptr + ctypes) has no dispatch, no version bump, no escape record. Must be UNVERIFIABLE.
class _UntypedStorageFill(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.detach().untyped_storage().fill_(0)
        return y * 2.0


class _TypedStorageSetitem(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.storage()[0] = 99.0
        return y * 2.0


class _CtypesDataPtrWrite(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        ctypes.c_float.from_address(y.data_ptr()).value = 99.0
        return y * 2.0


_H3_CELLS: list[tuple[str, Callable[[], nn.Module]]] = [
    ("untyped_storage.fill_", _UntypedStorageFill),
    ("storage.setitem", _TypedStorageSetitem),
    ("ctypes.data_ptr.write", _CtypesDataPtrWrite),
]


@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated")
@pytest.mark.parametrize("name,factory", _H3_CELLS, ids=[c[0] for c in _H3_CELLS])
def test_r14_h3_storage_bridge_host_write_is_never_verified(
    name: str, factory: Callable[[], nn.Module], tmp_path: Path
) -> None:
    """A host write through a storage-pointer bridge is UNVERIFIABLE (both inputs)."""
    for run_x in (_R14_X, _R14_CHANGED):
        result = _save_load(factory(), _R14_X.clone(), tmp_path).run(
            inputs=run_x.clone(), seed=0, on_divergence="return_diverged"
        )
        assert result.report.path_faithfulness is not PathFaithfulness.VERIFIED


# --- R14-C3: a storage-rebinding op (set_ / resize_ family) is not faithfully representable;
# it must be REFUSED at SAVE with a typed diagnostic, never crash at run and never false VERIFY.
class _StorageSet(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.set_(y.clone())
        return y * 2.0


class _StorageResize(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.resize_(4)
        return y * 2.0


_C3_CELLS: list[tuple[str, Callable[[], nn.Module]]] = [
    ("set_", _StorageSet),
    ("resize_", _StorageResize),
]


@pytest.mark.parametrize("name,factory", _C3_CELLS, ids=[c[0] for c in _C3_CELLS])
def test_r14_c3_storage_rebind_op_is_refused_at_save(
    name: str, factory: Callable[[], nn.Module], tmp_path: Path
) -> None:
    """set_/resize_ fail closed at SAVE with a typed diagnostic (no run-time crash)."""
    trace = tl.trace(factory(), _R14_X.clone(), capture=_CAPTURE)
    with pytest.raises(RunnablePreflightError) as excinfo:
        trace.save(tmp_path / "storage_rebind.tlspec", level="runnable")
    diagnostics = excinfo.value.fields["diagnostics"]
    assert any(d.detection_stage == "producer_storage_unsafe_op" for d in diagnostics)
    assert any(d.code is RunnableErrorCode.UNTRUSTED_CUSTOM_IMPORT for d in diagnostics)


# --- No over-trigger: a READ-ONLY storage METADATA exposure reads no value/pointer and must
# stay VERIFIED on any input; a genuine tracked in-place on a labelled alias still replays.
# (r15-H1: ``data_ptr()`` is NO LONGER a safe read-only exposure -- the raw pointer is
# unobservable -> UNVERIFIABLE; only pure metadata like ``untyped_storage().nbytes()`` stays
# VERIFIED. See ``test_r15_data_ptr_pointer_escape_is_unverifiable``.)
class _ReadonlyStorageBridge(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        _ = y.detach().untyped_storage().nbytes()
        _ = y.detach().untyped_storage().size()
        return y * 2.0


class _LabelledAliasInplace(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + 1.0
        y.detach().add_(2.0)  # tracked in-place on a LABELLED (detach) alias -> replayed
        return y * 2.0


@pytest.mark.parametrize(
    "factory",
    [_ReadonlyStorageBridge, _LabelledAliasInplace],
    ids=["readonly_storage_bridge", "labelled_alias_inplace"],
)
def test_r14_no_over_trigger_stays_verified_on_any_input(
    factory: Callable[[], nn.Module], tmp_path: Path
) -> None:
    """Read-only storage exposure and genuine labelled-alias in-place stay VERIFIED, no over-trigger."""
    original = _save_load(factory(), _R14_X.clone(), tmp_path).run(inputs=_R14_X.clone(), seed=0)
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    changed = _save_load(factory(), _R14_X.clone(), tmp_path).run(
        inputs=_R14_CHANGED.clone(), seed=0
    )
    assert changed.report.path_faithfulness is PathFaithfulness.VERIFIED
