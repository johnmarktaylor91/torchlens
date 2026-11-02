"""Parameters the captured forward itself creates: lazy modules built inside forward (stress F3).

``self.lazy = nn.LazyLinear(4)`` inside forward builds a module model preparation never saw.
Torch materializes its pending slots in place from the lazy pre-hook, ``reset_parameters``
fills them, and ``F.linear`` consumes them with no session provenance: the capture warned,
tried a rescue it had to discard, and forward validation failed its ``source_provenance``
check. The wrapper now registers such a parameter late, at the wrapped call that first
consumes it, through the same session registry preparation uses, and discloses the creation
as a ``parameter_created_in_forward`` capture advisory.

The carve-out is exactly the r77 shape: only a Parameter the wrapper watched torch
materialize from a pending lazy slot during this forward, that is a live slot of the
captured model's own module tree at consumption time, registers. A lazy module that is not
attached to the model, a foreign model's lazy module, and a Parameter attached to the tree
from a captured value (``self.p = nn.Parameter(y.detach())``) all keep the honest
``source_provenance`` failure. A hand-written ``nn.init.uniform_`` on an ordinary prepared
parameter inside forward is still model dataflow (a recorded, consumed mutation op), and an
unaccounted dispatch beside a lazy materialization still fails the completeness tripwire.
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens._capture_honesty import ADVISORY_PARAMETER_CREATED_IN_FORWARD
from torchlens.backends.torch.completeness_witness import LAZY_MODULE_MATERIALIZE_WRAPPER
from torchlens.backends.torch.wrappers import unwrap_torch, wrap_torch
from torchlens.validation import last_validation_failure
from torchlens.validation.diagnostics import (
    CHECK_ARG_LOGGING,
    CHECK_COMPLETENESS,
    CHECK_SOURCE_PROVENANCE,
)

WIDTH = 8
BATCH = 3
PROVENANCE_WARNING = "no graph/source provenance"


class LazyBuiltInForward(nn.Module):
    """``nn.LazyLinear`` created and materialized inside forward, never seen by preparation."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.lazy: nn.Module | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lazy = nn.LazyLinear(4)
        return self.lazy(torch.relu(self.pre(x)))


class LazyFirstCall(nn.Module):
    """``nn.LazyLinear`` recreated every forward: the slot preparation registered is replaced."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.lazy: nn.Module = nn.LazyLinear(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lazy = nn.LazyLinear(4)
        return self.lazy(torch.relu(self.pre(x)))


class LazyAtPrep(nn.Module):
    """``nn.LazyLinear`` registered before capture and materialized by the captured forward."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.lazy = nn.LazyLinear(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lazy(torch.relu(self.pre(x)))


class LazyNotAttached(nn.Module):
    """A lazy module built inside forward but never attached to the model."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lazy = nn.LazyLinear(4)
        return lazy(torch.relu(self.pre(x)))


class ForeignLazyModule(nn.Module):
    """The forward runs ANOTHER model's lazy module: its parameters are foreign."""

    def __init__(self, foreign: nn.Module) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)
        object.__setattr__(self, "_foreign", foreign)  # held outside the module tree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._foreign(torch.relu(self.pre(x)))


class CapturedValueAttachedAsParameter(nn.Module):
    """``self.p = nn.Parameter(y.detach())`` inside forward: attached, but never a pending slot."""

    def __init__(self) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.pre(x)
        self.p = nn.Parameter(y.detach(), requires_grad=False)
        return self.p * 1.0 + y


class InitInForward(nn.Module):
    """A hand-written initializer on an ORDINARY prepared parameter inside forward."""

    def __init__(self, initializer: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        torch.manual_seed(1)
        self.weight = nn.Parameter(torch.randn(4, WIDTH))
        self.initializer = initializer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.initializer(self.weight)
        return F.linear(x, self.weight)


class LazyThenStaleCall(nn.Module):
    """A stale (unwrapped) torch call feeding a lazy module built inside forward."""

    def __init__(self, stale: Callable[..., Any]) -> None:
        super().__init__()
        self.pre = nn.Linear(WIDTH, WIDTH)
        self.lazy: nn.Module | None = None
        self.stale = stale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lazy = nn.LazyLinear(4)
        return self.lazy(self.stale(torch.relu(self.pre(x))))


@pytest.fixture
def _isolated_witness_mode() -> Iterator[None]:
    """Restore the process-level diagnostic modes a census test arms."""

    saved_escape = _state._escape_detector_mode
    saved_witness = _state._completeness_witness_mode
    unwrap_torch()
    yield
    unwrap_torch()
    wrap_torch(escape_detector=saved_escape, completeness_witness=saved_witness)


def _raw(func: Callable[..., Any]) -> Callable[..., Any]:
    """Return the original torch callable behind an installed wrapper, in an opaque holder."""

    wrap_torch()
    return OpaqueCallable(_state._decorated_to_orig.get(id(func), func))


def _x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, WIDTH)


def _trace_without_provenance_warning(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, [x])
    provenance = [w for w in caught if PROVENANCE_WARNING in str(w.message)]
    assert provenance == [], [str(w.message)[:200] for w in provenance]
    return trace


def _creation_rows(trace: tl.Trace) -> list[dict[str, Any]]:
    rows = (trace.annotations or {}).get("capture_advisories") or []
    return [row for row in rows if row.get("kind") == ADVISORY_PARAMETER_CREATED_IN_FORWARD]


def _op_consuming(trace: tl.Trace, address: str) -> Any:
    consumers = [layer for layer in trace.layers if address in [p.address for p in layer.params]]
    assert len(consumers) == 1, [
        (layer.label, [p.address for p in layer.params]) for layer in consumers
    ]
    return consumers[0]


def _assert_fails_at(model: nn.Module, x: torch.Tensor, *check_names: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert tl.validate(model, x, scope="forward") is False
    failure = last_validation_failure()
    assert failure is not None
    assert failure.check_name in check_names, failure.summary()


@pytest.mark.smoke
def test_lazy_module_built_in_forward_registers_its_parameters_with_provenance() -> None:
    x = _x()
    trace = _trace_without_provenance_warning(LazyBuiltInForward().eval(), x)

    assert "lazy.weight" in trace.params and "lazy.bias" in trace.params
    assert tuple(trace.params["lazy.weight"].shape) == (4, WIDTH)
    assert tuple(trace.params["lazy.bias"].shape) == (4,)
    assert trace.params["lazy.weight"].num_params == 4 * WIDTH
    linear = _op_consuming(trace, "lazy.weight")
    assert linear.type == "linear"
    assert {p.address for p in linear.params} == {"lazy.weight", "lazy.bias"}
    # The init writes created the first value: logged, then orphan-pruned, never a mutation.
    assert all(layer.type != "uniform" for layer in trace.layers)

    (row,) = _creation_rows(trace)
    assert row["count"] == 2
    assert "lazy.weight" in row["message"] and "lazy.bias" in row["message"]
    assert "replaces" not in row["message"]


def _assert_validates(model: nn.Module, x: torch.Tensor) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        verdict = tl.validate(model, x, scope="forward")
    failure = last_validation_failure()
    assert verdict is True, failure.summary() if failure is not None else "no failure recorded"


@pytest.mark.smoke
def test_lazy_module_built_in_forward_validates() -> None:
    _assert_validates(LazyBuiltInForward().eval(), _x())


def test_lazy_module_recreated_in_forward_replaces_the_prep_slot_and_validates() -> None:
    x = _x()
    model = LazyFirstCall().eval()
    trace = _trace_without_provenance_warning(model, x)

    # The slot preparation registered held a pending parameter the forward displaced; the
    # record now names the live object with its real geometry.
    assert tuple(trace.params["lazy.weight"].shape) == (4, WIDTH)
    assert trace.params["lazy.weight"].num_params == 4 * WIDTH
    assert {p.address for p in _op_consuming(trace, "lazy.weight").params} == {
        "lazy.weight",
        "lazy.bias",
    }
    (row,) = _creation_rows(trace)
    assert row["count"] == 2
    assert "replaces the pending slot registered at preparation" in row["message"]
    # Validation re-traces the model it is handed; a model with still-pending slots
    # at preparation is refused typed, so validate the materialized instance, as
    # the stress oracle's eager pre-run does.
    _assert_validates(model, x)


def test_lazy_module_registered_at_prep_records_no_creation_advisory() -> None:
    x = _x()
    model = LazyAtPrep().eval()
    trace = _trace_without_provenance_warning(model, x)
    assert _creation_rows(trace) == []
    assert tuple(trace.params["lazy.weight"].shape) == (4, WIDTH)
    _assert_validates(model, x)


@pytest.mark.smoke
@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda: LazyNotAttached(), id="not_attached"),
        pytest.param(lambda: ForeignLazyModule(nn.LazyLinear(4)), id="foreign_model"),
    ],
)
def test_lazy_parameters_outside_the_model_tree_gain_no_provenance(
    build: Callable[[], nn.Module],
) -> None:
    x = _x()
    with pytest.warns(UserWarning, match=PROVENANCE_WARNING):
        trace = tl.trace(build().eval(), [x])
    assert {pl.address for pl in trace.params} == {"pre.weight", "pre.bias"}
    assert _creation_rows(trace) == []
    _assert_fails_at(build().eval(), x, CHECK_SOURCE_PROVENANCE)


@pytest.mark.smoke
def test_parameter_built_from_a_captured_value_gains_no_provenance() -> None:
    """r77 shape: attached to the tree inside forward, but never a pending lazy slot."""

    x = _x()
    with pytest.warns(UserWarning, match=PROVENANCE_WARNING):
        trace = tl.trace(CapturedValueAttachedAsParameter().eval(), [x])
    assert "p" not in trace.params
    assert _creation_rows(trace) == []
    # The source-less argument fails whichever provenance check reaches it first.
    _assert_fails_at(
        CapturedValueAttachedAsParameter().eval(), x, CHECK_ARG_LOGGING, CHECK_SOURCE_PROVENANCE
    )


def test_hand_written_init_on_a_prepared_parameter_is_model_dataflow() -> None:
    """``nn.init.uniform_(self.weight)`` on an ordinary parameter is a recorded, consumed op."""

    x = _x()
    model = InitInForward(lambda w: nn.init.uniform_(w, -0.1, 0.1))
    trace = tl.trace(copy.deepcopy(model), x)
    # ``nn.init.uniform_`` records its write as a chain of ``uniform`` ops on the
    # parameter (the first names ``weight`` as its parameter input; a later one
    # consumes the labeled result); the ``linear`` then reads the chain's output.
    uniform_ops = [layer for layer in trace.layers if layer.type == "uniform"]
    assert uniform_ops, [layer.label for layer in trace.layers]
    assert any([p.address for p in layer.params] == ["weight"] for layer in uniform_ops), [
        (layer.label, [p.address for p in layer.params]) for layer in uniform_ops
    ]
    (linear,) = [layer for layer in trace.layers if layer.type == "linear"]
    parent_labels = [trace[parent].label for parent in linear.parents]
    assert any(layer.label in parent_labels for layer in uniform_ops), (
        [layer.label for layer in uniform_ops],
        parent_labels,
    )
    assert "weight" not in [p.address for p in linear.params]
    assert _creation_rows(trace) == []
    assert all(
        entry.get("owner_wrapper") != LAZY_MODULE_MATERIALIZE_WRAPPER
        for entry in getattr(trace, "completeness_decompositions", ())
    )


def test_hand_written_constant_init_on_a_prepared_parameter_validates() -> None:
    x = _x()
    model = InitInForward(lambda w: w.fill_(0.5))
    trace = tl.trace(copy.deepcopy(model), x)
    mutations = [layer for layer in trace.layers if layer.type == "fill"]
    assert mutations, [layer.label for layer in trace.layers]
    assert all([p.address for p in layer.params] == ["weight"] for layer in mutations)
    _assert_validates(copy.deepcopy(model), x)


@pytest.mark.smoke
def test_unaccounted_dispatch_beside_a_lazy_materialization_still_fails_completeness(
    _isolated_witness_mode: None,
) -> None:
    """The audited materialization owner covers only the pending slot's own calls."""

    stale = _raw(torch.tanh)
    _assert_fails_at(LazyThenStaleCall(stale).eval(), _x(), CHECK_COMPLETENESS)
