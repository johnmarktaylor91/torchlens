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

import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens as _torchlens
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

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
        # ``.data`` severs the graph link; the gt predicate is orphan-pruned and its
        # bool source carries no resolvable label -> witnessed by NO net -> INCOMPLETE.
        if bool(self.gate.data > 0.5):
            return h + 10.0
        return h - 10.0


def test_pruned_unattributable_bool_is_unverifiable_even_on_original(tmp_path: Path) -> None:
    """A pruned, unlabelled bool control predicate fails closed on ANY run (documented)."""
    loaded = _save_load(_DataAliasBoolControl(), _STATE_X_VEC, tmp_path)
    original = loaded.run(inputs=_STATE_X_VEC, seed=0, on_divergence="return_diverged")
    assert original.report.path_faithfulness is not PathFaithfulness.VERIFIED
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


_ESCAPE_FREE: list[tuple[str, Callable[[], nn.Module], torch.Tensor, torch.Tensor]] = [
    ("clean_linear", _CleanLinear, _VEC_IN, _VEC_IN_CHANGED),
    ("clean_elementwise", _CleanElementwise, _TWO, _TWO_CHANGED),
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
