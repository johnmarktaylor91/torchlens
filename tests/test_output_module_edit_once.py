"""A module-scoped edit on a module whose output the model returns applies exactly once.

Post-postprocess, every value the model returns gets a synthetic ``output_N`` node that
aliases the producing op and carries that op's module stamp for display. Post-hoc
``tl.module(M)`` / ``tl.in_module(M)`` used to resolve to the producer AND that alias, so
``fork().do(...)`` re-applied the edit at the alias (``tl.scale(0.5)`` gave x0.25). The
legacy rerun ``fork.run(model, x2)`` doubled ``tl.in_module(M)`` the same way at the live
module boundary, which aliases the op inside ``M`` that produced it.

Every door must agree with an eager forward hook on ``M``: post-hoc ``fork().do``, a
legacy rerun of that fork on new input, ``tl.trace(intervene=)`` and ``spec.bind(model)``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.intervention.errors import ControlFlowDivergenceWarning, MultiMatchWarning


class _MLP(nn.Module):
    """Two Linear layers; the last one's output is the model output."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Output of ``fc2``.
        """

        return self.fc2(torch.relu(self.fc1(x)))


class _ReluTail(nn.Module):
    """A Linear followed by an ``nn.ReLU`` module whose output is returned."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the network.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            Output of ``act``.
        """

        return self.act(self.fc1(x))


class _PairHead(nn.Module):
    """A head module that returns a tuple of two projections."""

    def __init__(self) -> None:
        """Build the projections."""

        super().__init__()
        self.a = nn.Linear(8, 3)
        self.b = nn.Linear(8, 2)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project ``h`` twice.

        Parameters
        ----------
        h:
            Hidden batch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Both projections.
        """

        return self.a(h), self.b(h)


class _TupleHeadNet(nn.Module):
    """A model whose last module returns a tuple, returned unchanged by the model."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.head = _PairHead()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The head's tuple output.
        """

        return self.head(torch.relu(self.fc1(x)))


class _TupleOutNet(nn.Module):
    """A model returning ``(fc2(h), h)``: ``fc1``'s output is both consumed and returned."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the network.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``fc2``'s output and the hidden ``fc1`` output.
        """

        h = self.fc1(x)
        return self.fc2(h), h


#: ``(case id, model factory, module address)``. Every module listed here produces a value
#: the model returns, so its output carries a synthetic ``output_N`` alias.
_CASES: tuple[tuple[str, Callable[[], nn.Module], str], ...] = (
    ("mlp_last_linear", _MLP, "fc2"),
    ("relu_module_tail", _ReluTail, "act"),
    ("tuple_from_last_module", _TupleHeadNet, "head"),
    ("tuple_out_last_linear", _TupleOutNet, "fc2"),
    ("tuple_out_consumed_and_returned", _TupleOutNet, "fc1"),
)
_KINDS = ("module", "in_module")


def _scale_tree(value: Any, factor: float) -> Any:
    """Scale every tensor leaf of a (possibly tuple) module output.

    Parameters
    ----------
    value:
        Tensor or tuple of tensors.
    factor:
        Multiplier.

    Returns
    -------
    Any
        Same structure with each tensor scaled.
    """

    if isinstance(value, tuple):
        return tuple(_scale_tree(item, factor) for item in value)
    return value * factor


def _leaves(value: Any) -> list[torch.Tensor]:
    """Flatten a model output into its tensor leaves, in return order.

    Parameters
    ----------
    value:
        Tensor or tuple of tensors.

    Returns
    -------
    list[torch.Tensor]
        Tensor leaves.
    """

    if isinstance(value, tuple):
        return [leaf for item in value for leaf in _leaves(item)]
    return [value]


def _oracle(model: nn.Module, address: str, x: torch.Tensor) -> list[torch.Tensor]:
    """Return the model output with an eager forward hook halving ``address``'s output.

    Parameters
    ----------
    model:
        Model under test.
    address:
        Submodule address to edit.
    x:
        Input batch.

    Returns
    -------
    list[torch.Tensor]
        Output tensor leaves.
    """

    handle = model.get_submodule(address).register_forward_hook(
        lambda _module, _args, out: _scale_tree(out, 0.5)
    )
    try:
        with torch.no_grad():
            return _leaves(model(x))
    finally:
        handle.remove()


def _trace_outputs(trace: Any) -> list[torch.Tensor]:
    """Return a trace's model-output values in return order.

    Parameters
    ----------
    trace:
        Captured or edited trace.

    Returns
    -------
    list[torch.Tensor]
        Output node values.
    """

    return [op.out for op in trace.output_ops]


def _assert_same(got: list[torch.Tensor], want: list[torch.Tensor], door: str) -> None:
    """Assert two output lists agree leaf by leaf.

    Parameters
    ----------
    got:
        Door output leaves.
    want:
        Oracle output leaves.
    door:
        Door name for the failure message.
    """

    assert len(got) == len(want), door
    for index, (got_leaf, want_leaf) in enumerate(zip(got, want)):
        torch.testing.assert_close(
            got_leaf.detach(), want_leaf.detach(), msg=f"{door}: output {index} differs"
        )


def _selector(kind: str, address: str) -> Any:
    """Build ``tl.module(address)`` or ``tl.in_module(address)``.

    Parameters
    ----------
    kind:
        ``"module"`` or ``"in_module"``.
    address:
        Module address.

    Returns
    -------
    Any
        The selector.
    """

    return tl.module(address) if kind == "module" else tl.in_module(address)


def _ready_trace(model: nn.Module, x: torch.Tensor) -> Any:
    """Capture an intervention-ready trace.

    Parameters
    ----------
    model:
        Model to capture.
    x:
        Input batch.

    Returns
    -------
    Any
        The trace.
    """

    return tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))


def _setup(factory: Callable[[], nn.Module]) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """Build a seeded eval-mode model and two input batches.

    Parameters
    ----------
    factory:
        Model class.

    Returns
    -------
    tuple[nn.Module, torch.Tensor, torch.Tensor]
        Model, capture input, and a second input for reruns.
    """

    torch.manual_seed(0)
    model = factory().eval()
    return model, torch.randn(3, 8), torch.randn(3, 8)


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(("case", "factory", "address"), _CASES, ids=[c[0] for c in _CASES])
def test_output_module_edit_applies_once_on_every_door(
    case: str, factory: Callable[[], nn.Module], address: str, kind: str
) -> None:
    """Post-hoc, rerun, capture and bind doors all match the eager forward-hook oracle."""

    model, x, x2 = _setup(factory)
    want, want2 = _oracle(model, address, x), _oracle(model, address, x2)
    selector = _selector(kind, address)

    fork = _ready_trace(model, x).fork()
    with warnings.catch_warnings():
        # A tuple-returning module legitimately fans out to its independent leaves.
        warnings.simplefilter("ignore", MultiMatchWarning)
        fork.do(selector, tl.scale(0.5))
    _assert_same(_trace_outputs(fork), want, f"{case}: fork().do post-hoc")

    with warnings.catch_warnings():
        # The rerun graph legitimately differs from the un-edited capture (other lane).
        warnings.simplefilter("ignore", ControlFlowDivergenceWarning)
        fork.run(model, x2)
    _assert_same(_trace_outputs(fork), want2, f"{case}: fork.run(model, x2) after do")

    captured = tl.trace(model, x, intervene=tl.when(selector, tl.scale(0.5)))
    _assert_same(_trace_outputs(captured), want, f"{case}: tl.trace(intervene=)")

    with torch.no_grad():
        bound = tl.when(selector, tl.scale(0.5)).bind(model)(x)
    _assert_same(_leaves(bound), want, f"{case}: spec.bind(model)(x)")


@pytest.mark.parametrize("kind", _KINDS)
@pytest.mark.parametrize(("case", "factory", "address"), _CASES, ids=[c[0] for c in _CASES])
def test_module_selectors_never_resolve_to_the_output_alias(
    case: str, factory: Callable[[], nn.Module], address: str, kind: str
) -> None:
    """``find_sites`` returns the producing ops only; the output alias keeps its stamp."""

    model, x, _x2 = _setup(factory)
    trace = _ready_trace(model, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MultiMatchWarning)
        sites = list(trace.find_sites(_selector(kind, address)))
    assert sites, case
    assert not any(getattr(site, "is_output", False) for site in sites), [
        site.layer_label for site in sites
    ]
    # Display stamp unchanged: the alias still names the module calls its producer sits in.
    from torchlens.ir.selector_eval import module_address_matches

    stamped = [
        op
        for op in trace.output_ops
        if any(module_address_matches(call, address) for call in op.output_of_module_calls)
    ]
    assert stamped, f"{case}: output alias lost its display stamp"


def test_single_leaf_output_module_selects_one_site_without_multimatch() -> None:
    """A leaf module that produces the model output resolves to exactly one site."""

    model, x, _x2 = _setup(_MLP)
    trace = _ready_trace(model, x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MultiMatchWarning)
        table = trace.resolve_sites(tl.module("fc2"))
    assert len(list(table)) == 1


def test_multimatch_warning_names_alias_compounding_not_fan_out() -> None:
    """A selector matching a producer and its output alias warns that the edit compounds."""

    model, x, _x2 = _setup(_MLP)
    trace = _ready_trace(model, x)
    alias = trace.output_ops[0]
    producer = alias.parents[0]
    selector = tl.label(producer) | tl.label(alias.layer_label)
    with pytest.warns(MultiMatchWarning) as record:
        trace.resolve_sites(selector)
    message = " ".join(str(item.message) for item in record)
    assert "compound" in message
    assert "fan out" not in message
    assert alias.layer_label in message and producer in message


def test_multimatch_warning_keeps_fan_out_for_independent_sites() -> None:
    """Independent sites (no alias pair) keep the fan-out wording."""

    model, x, _x2 = _setup(_TupleHeadNet)
    trace = _ready_trace(model, x)
    with pytest.warns(MultiMatchWarning, match="will fan out"):
        trace.resolve_sites(tl.module("head"))


class _ChunkNet(nn.Module):
    """A multi-output op (``torch.chunk``) whose two leaves feed a product."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split ``x`` in two and multiply the halves.

        Parameters
        ----------
        x:
            Input batch with an even feature count.

        Returns
        -------
        torch.Tensor
            Product of the two halves.
        """

        a, b = torch.chunk(x, 2, dim=1)
        return a * b


def test_bind_applies_edits_to_every_leaf_of_a_multi_output_op() -> None:
    """``bind`` rebuilds a tuple-valued op output with its edited leaves."""

    torch.manual_seed(0)
    x = torch.randn(3, 8)
    a, b = torch.chunk(x, 2, dim=1)
    want = (a * 0.5) * (b * 0.5)
    with torch.no_grad():
        bound = tl.when(tl.func("chunk"), tl.scale(0.5)).bind(_ChunkNet())(x)
    torch.testing.assert_close(bound, want)
    captured = tl.trace(_ChunkNet(), x, intervene=tl.when(tl.func("chunk"), tl.scale(0.5)))
    torch.testing.assert_close(captured.output_ops[0].out, want)


def test_region_exiting_into_a_model_output_refuses_typed() -> None:
    """A region whose exit is a returned value refuses instead of silently not editing it.

    Before the alias exclusion, ``tl.in_module("fc2")`` pulled ``output_1`` into the
    region interior, so the region had no exit and the edit never reached the model
    output. The exit into the alias cannot be spliced, so it now refuses by name.
    """

    from torchlens.intervention.errors import RegionError

    model, x, _x2 = _setup(_MLP)
    fork = _ready_trace(model, x).fork()
    target = fork.subgraph(tl.in_module("fc2")).as_region()
    with pytest.raises(RegionError) as excinfo:
        fork.do(target, tl.scale(0.5))
    assert excinfo.value.fields["code"] == "region_exit_address_underivable"
    assert "output_1" in str(excinfo.value)
