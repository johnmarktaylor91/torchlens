"""An untraced module output credits only the dispatch that built it.

When a module returns a tensor with no live label, module exit synthesizes a
functionless boundary Op and marks the module-forward token capture-accounted.
That credit covers the opaque construction of the exact boundary tensors. A
stale raw torch call elsewhere in the same module body (a callable bound
before TorchLens wrapped torch, as in IQL's ``hidden_activation=torch.relu``
default) is not represented by the boundary, so validation must fail on it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from _stale_holders import CountedRoot, OpaqueCallable, provenance_warnings
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.backends.torch import rescue
from torchlens.backends.torch.escape_detection import ExpectedOriginalToken
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.user_funcs import _validate_forward_pass_torch


@pytest.fixture
def _isolated_witness_mode() -> Iterator[None]:
    """Restore the process-level diagnostic modes a census test arms.

    ``wrap_torch(completeness_witness=True)`` is process-wide: left armed, it
    makes every later capture on the worker settle through the witness, so a
    later meta capture trips the weights-free settlement invariant.
    """

    saved_escape = _state._escape_detector_mode
    saved_witness = _state._completeness_witness_mode
    unwrap_torch()
    yield
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)


def _raw(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the original torch callable behind an installed wrapper, in an opaque holder.

    Capture preparation rebinds pristine torch functions held directly on a
    model, so a bare original would no longer escape; the custom callable
    object is a holder it never rebinds, which keeps the escape these
    completeness tripwires must catch.
    """

    wrap_torch()
    return OpaqueCallable(_state._decorated_to_orig.get(id(func), func))


class _OpaqueOutputChild(nn.Module):
    """Traced linear and relu, then a direct-aten output: one boundary, nothing dropped."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a direct-aten tanh of traced work."""

        return torch.ops.aten.tanh.default(torch.relu(self.fc(x)))


class _OpaqueTupleChild(nn.Module):
    """Two direct-aten outputs: each is its own boundary tensor."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two untraced outputs built by single direct-aten ops."""

        return torch.ops.aten.tanh.default(x), torch.ops.aten.sigmoid.default(x)


class _StaleHiddenChild(nn.Module):
    """IQL shape: stale relu hidden activation, stale tanh output activation."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        self.hidden_activation = _raw(torch.relu)
        self.output_activation = _raw(torch.tanh)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run fc1, stale relu, fc2, stale tanh."""

        return self.output_activation(self.fc2(self.hidden_activation(self.fc1(x))))


class _StaleInnerLeaf(nn.Module):
    """A stale relu intermediate under a direct-aten output, two levels down."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.act = _raw(torch.relu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a direct-aten tanh of a stale relu."""

        return torch.ops.aten.tanh.default(self.fc(self.act(x)))


class _Middle(nn.Module):
    """Plain container so the boundary-backed module sits at depth two."""

    def __init__(self) -> None:
        super().__init__()
        self.leaf = _StaleInnerLeaf()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the leaf and a traced op on its output."""

        return self.leaf(x) + 1.0


class _Parent(nn.Module):
    """Consume a child's output in a traced op."""

    def __init__(self, child: nn.Module) -> None:
        super().__init__()
        self.child = child

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the child and scale its output (summing tuple outputs)."""

        out = self.child(x)
        if isinstance(out, tuple):
            return (out[0] + out[1]) * 2.0
        return out * 2.0


def _validate(model: nn.Module, *, grad: bool = True) -> bool:
    for param in model.parameters():
        param.requires_grad_(grad)
    torch.manual_seed(0)
    return _validate_forward_pass_torch(
        model.eval(), [torch.randn(3, 4)], {}, random_seed=0, validate_metadata=True
    )


def _assert_completeness_failure(model: nn.Module, *, grad: bool = True) -> None:
    assert not _validate(model, grad=grad)
    failure = tl.validation.last_validation_failure()
    assert failure is not None
    assert "bfs_completeness" in failure.summary()


# The stale op's output reaches the next traced op with no recorded parent; that
# provenance disclosure is expected alongside the completeness failure. An opaque
# module RETURN is disclosed the same way (module-exit adoption record) while the
# boundary still credits the dispatch that built it.
_NO_PROVENANCE = "ignore:TorchLens found tensor arguments with no graph:UserWarning"


@pytest.mark.filterwarnings(_NO_PROVENANCE)
def test_single_opaque_output_op_still_validates() -> None:
    """The boundary still credits the one direct-aten op that built the module output."""

    assert _validate(_Parent(_OpaqueOutputChild())), tl.validation.last_validation_failure()


@pytest.mark.filterwarnings(_NO_PROVENANCE)
def test_tuple_of_opaque_outputs_still_validates() -> None:
    """Every tensor of a tuple result that is a boundary output is credited."""

    assert _validate(_Parent(_OpaqueTupleChild())), tl.validation.last_validation_failure()


@pytest.mark.filterwarnings(_NO_PROVENANCE)
def test_stale_hidden_activation_under_stale_output_fails_completeness() -> None:
    """IQL shape: the stale relu is not hidden by the stale tanh's boundary."""

    _assert_completeness_failure(_Parent(_StaleHiddenChild()))


@pytest.mark.filterwarnings(_NO_PROVENANCE)
def test_stale_op_in_nested_boundary_module_fails_completeness() -> None:
    """A stale op inside a depth-two module with a synthesized output boundary fails."""

    _assert_completeness_failure(_Parent(_Middle()))


@pytest.mark.filterwarnings(_NO_PROVENANCE)
@pytest.mark.usefixtures("_isolated_witness_mode")
def test_census_names_only_the_stale_relu_in_iql_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """The primary capture's census names the dropped relu, never the boundary's tanh."""

    # Keep the primary capture: the rescue re-run would replace its diagnostics.
    monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)
    wrap_torch(completeness_witness=True)
    torch.manual_seed(0)
    trace = tl.trace(_Parent(_StaleHiddenChild()).eval(), torch.randn(3, 4))
    operators = [row["operator"] for row in trace.completeness_diagnostics]
    assert "aten.relu.default" in operators
    assert "aten.tanh.default" not in operators
    assert trace.capture_verified is False


_ATEN = torch.ops.aten


class _BodyChild(nn.Module):
    """Run ``body(self, x)``: one linear layer, a stale relu, and the case's own forward."""

    def __init__(self, body: Callable[[_BodyChild, torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.stale_relu = _raw(torch.relu)
        self.body = body

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to the case body."""

        return self.body(self, x)


def _direct_aten_view_output(m: _BodyChild, x: torch.Tensor) -> torch.Tensor:
    return _ATEN.t.default(m.fc(x))


def _direct_aten_split_output(m: _BodyChild, x: torch.Tensor) -> torch.Tensor:
    return _ATEN.split.Tensor(m.fc(x), 2, 1)[0]


def _stale_op_builds_view_base(m: _BodyChild, x: torch.Tensor) -> torch.Tensor:
    return _ATEN.t.default(m.stale_relu(m.fc(x)))


def _stale_op_reads_detached_output(m: _BodyChild, x: torch.Tensor) -> torch.Tensor:
    out = _ATEN.tanh.default(m.fc(x))
    m.side = m.stale_relu(_ATEN.detach.default(out))
    return out


def _freed_stale_intermediates(m: _BodyChild, x: torch.Tensor) -> torch.Tensor:
    hidden = m.fc(x)
    for _ in range(40):
        tmp = m.stale_relu(hidden)
        del tmp
    return _ATEN.tanh.default(hidden)


def _composite_aten_output(m: _BodyChild, x: torch.Tensor) -> torch.Tensor:
    return _ATEN.linear.default(x.view(1, 3, 4), m.fc.weight, m.fc.bias)


@pytest.mark.filterwarnings(_NO_PROVENANCE)
@pytest.mark.parametrize("grad", [True, False], ids=["grad", "no_grad"])
@pytest.mark.parametrize("body", [_direct_aten_view_output, _direct_aten_split_output])
def test_direct_aten_view_or_split_output_validates(body: Any, grad: bool) -> None:
    """A direct-aten view or multi-output op whose result is the module output is credited."""

    model = _Parent(_BodyChild(body))
    assert _validate(model, grad=grad), tl.validation.last_validation_failure()


@pytest.mark.filterwarnings(_NO_PROVENANCE)
@pytest.mark.parametrize("grad", [True, False], ids=["grad", "no_grad"])
@pytest.mark.parametrize(
    "body",
    [_stale_op_builds_view_base, _stale_op_reads_detached_output, _freed_stale_intermediates],
)
def test_alias_edges_do_not_credit_stale_ops(body: Any, grad: bool) -> None:
    """Only the boundary object and its pure aliases are credited, never their neighbours.

    The base of a boundary view, a consumer of a detached boundary, and freed stale
    intermediates whose addresses a later boundary tensor may reuse all stay flagged.
    """

    _assert_completeness_failure(_Parent(_BodyChild(body)), grad=grad)


@pytest.mark.filterwarnings(_NO_PROVENANCE)
@pytest.mark.usefixtures("_isolated_witness_mode")
def test_freed_stale_intermediates_are_each_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """All 40 freed stale relus are census rows: an ``id()``-reuse mis-credit drops one."""

    # Keep the primary capture: the rescue re-run would replace its diagnostics.
    monkeypatch.setattr(rescue, "_escape_signal", lambda trace: None)
    wrap_torch(completeness_witness=True)
    torch.manual_seed(0)
    model = _Parent(_BodyChild(_freed_stale_intermediates)).eval()
    trace = tl.trace(model, torch.randn(3, 4))
    operators = [row["operator"] for row in trace.completeness_diagnostics]
    assert operators.count("aten.relu.default") == 40, operators


@pytest.mark.parametrize(
    ("wrapper_name", "scoped"),
    [
        ("module_forward:exhaustive", True),
        ("module_forward:predicate", True),
        ("module_forward_hook:user", False),
        ("torch.relu", False),
    ],
)
def test_output_scoped_credit_is_derived_from_the_wrapper_name(
    wrapper_name: str, scoped: bool
) -> None:
    """The flag follows ``wrapper_name`` at every construction site; it cannot be passed."""

    token = ExpectedOriginalToken(
        raw_id=0, original=len, wrapper_name=wrapper_name, wrapper_frame_id=0, owner_thread_id=0
    )
    assert token.boundary_credit_is_output_scoped is scoped
    with pytest.raises(TypeError):
        ExpectedOriginalToken(  # type: ignore[call-arg]
            raw_id=0,
            original=len,
            wrapper_name=wrapper_name,
            wrapper_frame_id=0,
            owner_thread_id=0,
            boundary_credit_is_output_scoped=not scoped,
        )


@pytest.mark.filterwarnings(_NO_PROVENANCE)
@pytest.mark.parametrize("grad", [True, False], ids=["grad", "no_grad"])
def test_multi_dispatch_opaque_output_fails_completeness(grad: bool) -> None:
    """One raw op means one aten dispatch: a composite op's inner dispatches stay flagged.

    ``aten.linear`` on a 3-d input decomposes above the Python dispatch key into
    several dispatches (``view``, ``t``, ``addmm``, ...); only the last returns the
    boundary tensor, so the others are census diagnostics and validation fails.
    """

    _assert_completeness_failure(_Parent(_BodyChild(_composite_aten_output)), grad=grad)


def _bare(func: Callable[..., Any]) -> Callable[..., Any]:
    """The original torch callable itself, held directly as these cases did before the rebind."""

    wrap_torch()
    return _state._decorated_to_orig.get(id(func), func)


@pytest.mark.parametrize("grad", [True, False], ids=["grad", "no_grad"])
@pytest.mark.parametrize(
    "build",
    [
        lambda: _Parent(_StaleHiddenChild()),
        lambda: _Parent(_Middle()),
        lambda: _Parent(_BodyChild(_stale_op_builds_view_base)),
        lambda: _Parent(_BodyChild(_freed_stale_intermediates)),
    ],
    ids=["iql_hidden_and_output", "nested_boundary", "view_base", "freed_intermediates"],
)
def test_attribute_held_originals_are_rebound_and_validate(
    build: Callable[[], nn.Module], grad: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The attribute-held cases above with the bare original, as they were first written.

    Capture preparation rebinds each held original to its wrapper for the
    capture, so the stale ops are no escape: one forward, the relu captured,
    no provenance warning, and validation passes.
    """

    monkeypatch.setitem(globals(), "_raw", _bare)
    model = CountedRoot(build()).eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, torch.randn(3, 4))
    assert model.calls == [1]
    assert provenance_warnings(caught) == []
    assert trace.rescue_rerun is None
    assert "relu" in [op.func_name for op in trace.ops]
    assert _validate(build(), grad=grad), tl.validation.last_validation_failure()
