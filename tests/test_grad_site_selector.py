"""Gradient addresses: ``tl.grad`` construction, universes, composition and plans.

Criteria A1 to A8 of the gradient-sites change: ``tl.grad(site)`` names the
gradient of a forward site's OUTPUT, a bare selector names the activation and
the autograd-node selectors name grad_fn records. The three universes never
share a site, so ``&`` across them refuses, ``|`` keeps one branch per
universe, and a bare label never reaches the backward universe.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._selection import selectors as _selectors
from torchlens._selection.resolver import (
    resolve_site_plan,
    split_site_branches,
)
from torchlens._selection.selectors import (
    GradSelector,
    grad_fn_label,
)
from torchlens.intervention.errors import SelectorCompositionError
from torchlens.intervention.types import TargetSpec


class _BlockModel(nn.Module):
    """Linear+ReLU block followed by a head, so module and func selectors differ."""

    def __init__(self) -> None:
        """Initialize the model."""

        super().__init__()
        self.block = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model."""

        return torch.relu(self.head(self.block(x)))


def _trace(*, backward: bool = False) -> Any:
    """Return a finished trace of the block model, optionally after one backward."""

    torch.manual_seed(0)
    model = _BlockModel()
    x = torch.randn(2, 3, requires_grad=True)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"))
    if backward:
        trace.log_backward(trace[trace.output_layers[0]].out.sum(), retain_graph=True)
    return trace


def _relu_labels(trace: Any) -> tuple[str, ...]:
    """Return the labels of the trace's two relu ops in execution order."""

    labels = trace.find_sites(tl.func("relu")).labels()
    assert len(labels) == 2
    return labels


def _code(exc: pytest.ExceptionInfo[BaseException]) -> str:
    """Return the refusal code a raised error carries."""

    fields = getattr(exc.value, "fields", {})
    assert fields.get("remedy"), "every refusal carries a remedy"
    return str(fields.get("code"))


def test_grad_accepts_every_forward_site_spelling() -> None:
    """A2: each accepted spelling resolves to the bare inner's ops, tagged gradient."""

    trace = _trace()
    first_relu, _ = _relu_labels(trace)
    op_record = resolve_site_plan(trace, tl.label(first_relu))[0].sites[0]
    layer_record = trace.layers[op_record.layer_label]
    spellings: list[Any] = [
        first_relu,
        tl.func("relu"),
        tl.module("block"),
        tl.in_module("block") & tl.func("relu"),
        op_record,
        layer_record,
        trace.find_sites(tl.func("relu")),
    ]
    for spelling in spellings:
        bare = resolve_site_plan(trace, spelling)
        assert [branch.universe for branch in bare] == ["activation"]
        plan = resolve_site_plan(trace, tl.grad(spelling))
        assert [branch.universe for branch in plan] == ["gradient"], spelling
        assert plan[0].signal == "backward"
        assert plan[0].pass_filter is None
        assert [op.label for op in plan[0].sites] == [op.label for op in bare[0].sites]


@pytest.mark.parametrize(
    ("build", "code"),
    [
        (lambda: tl.grad(tl.grad("x") | "y"), "grad_selector_nested"),
        (lambda: tl.grad(grad_fn_label("x")), "grad_selector_inner_not_forward"),
        (lambda: tl.grad(tl.head(0)), "grad_facet_unsupported"),
        (lambda: tl.grad({"a": 1}), "grad_selector_inner_invalid"),
        (lambda: tl.grad(3), "grad_selector_inner_invalid"),
        (lambda: tl.grad(tl.label("a").__selection__()), "grad_selection_unsupported"),
        (lambda: tl.grad("a").__selection__(), "grad_selection_unsupported"),
    ],
    ids=[
        "nested_in_union",
        "grad_fn_label",
        "head",
        "mapping",
        "int",
        "selection",
        "lift",
    ],
)
def test_grad_construction_refusals(build: Any, code: str) -> None:
    """A3: every refused inner raises SelectorCompositionError with code and remedy."""

    with pytest.raises(SelectorCompositionError) as exc:
        build()
    assert _code(exc) == code


def test_grad_refuses_a_pass_qualifier_hidden_in_a_deserialized_inner() -> None:
    """A3: a pass qualifier inside the inner refuses even when no operator ran.

    ``tl.grad("x" & tl.in_backward_pass(2))`` never reaches ``tl.grad``: the
    ``&`` refuses first (A5). A deserialized spec builds the same tree without
    the operator, so ``tl.grad`` itself must refuse it.
    """

    spec = TargetSpec(
        selector_kind="and",
        selector_value=(
            TargetSpec(selector_kind="label", selector_value="x"),
            TargetSpec(selector_kind="backward_pass", selector_value=2),
        ),
    )
    with pytest.raises(SelectorCompositionError) as exc:
        tl.grad(spec)
    assert _code(exc) == "grad_selector_inner_not_forward"
    assert "tl.grad(s) & tl.in_backward_pass(2)" in exc.value.fields["remedy"]


def test_mixed_union_keeps_both_branches() -> None:
    """A4: tl.grad(s) | s plans a gradient and an activation branch on the same op."""

    trace = _trace()
    first_relu, _ = _relu_labels(trace)
    plan = resolve_site_plan(trace, tl.grad(first_relu) | first_relu)
    assert [branch.universe for branch in plan] == ["gradient", "activation"]
    assert [branch.signal for branch in plan] == ["backward", "forward"]
    for branch in plan:
        assert [op.label for op in branch.sites] == [trace[first_relu].label]
    table = trace.find_sites(tl.grad(first_relu) | first_relu)
    assert len(table) == 2
    frame = pytest.importorskip("pandas") and table.to_dataframe()
    assert list(frame["signal"]) == ["backward", "forward"]


@pytest.mark.parametrize(
    ("build", "code"),
    [
        (lambda: (tl.grad("a") | "a") & tl.in_backward_pass(2), "selector_cross_universe_and"),
        (lambda: ~(tl.grad("a") | "a"), "selector_cross_universe_not"),
    ],
    ids=[
        "mixed_and_pass",
        "invert_mixed",
    ],
)
def test_cross_universe_composition_refusals(build: Any, code: str) -> None:
    """A5: & across universes refuses with a coded remedy."""

    with pytest.raises(SelectorCompositionError) as exc:
        build()
    assert _code(exc) == code


@pytest.mark.smoke
def test_signal_mismatch_remedy_names_the_gradient_spelling() -> None:
    """A5: the remedy spells tl.grad('relu_3') & tl.in_backward_pass(2)."""

    with pytest.raises(SelectorCompositionError) as exc:
        _ = "relu_3" & tl.in_backward_pass(2)
    assert "tl.grad('relu_3') & tl.in_backward_pass(2)" in exc.value.fields["remedy"]
    assert str(exc.value).endswith("tl.grad('relu_3') & tl.in_backward_pass(2).")


def test_pass_qualifier_becomes_the_gradient_branch_filter() -> None:
    """tl.grad(s) & tl.in_backward_pass(2) is one gradient branch filtered to pass 2."""

    (branch,) = split_site_branches(tl.grad("a") & tl.in_backward_pass(2))
    assert branch.universe == "gradient"
    assert branch.selector == "a"
    assert branch.accepts_pass(2) and not branch.accepts_pass(1)
    (unfiltered,) = split_site_branches(tl.grad("a"))
    assert unfiltered.accepts_pass(1) and unfiltered.accepts_pass(7)
    with pytest.raises(SelectorCompositionError) as exc:
        split_site_branches(~(tl.grad("a") & tl.in_backward_pass(2)))
    assert _code(exc) == "selector_signal_mismatch"


def test_grad_complement_and_union_normalise_inside_the_gradient_universe() -> None:
    """A6: ~tl.grad(s) == tl.grad(~s); tl.grad(a) | tl.grad(b) == tl.grad(a | b)."""

    trace = _trace()
    first_relu, second_relu = _relu_labels(trace)
    negated = ~tl.grad(tl.label(first_relu))
    assert isinstance(negated, GradSelector)
    assert negated == tl.grad(~tl.label(first_relu))
    plan = resolve_site_plan(trace, negated)
    assert [branch.universe for branch in plan] == ["gradient"]
    every_op = [op.label for op in trace._layer_list]
    assert [op.label for op in plan[0].sites] == [
        label for label in every_op if label != trace[first_relu].label
    ]
    assert isinstance(~tl.grad(first_relu), GradSelector)

    union = resolve_site_plan(trace, tl.grad(first_relu) | tl.grad(second_relu))
    joined = resolve_site_plan(trace, tl.grad(tl.label(first_relu) | tl.label(second_relu)))
    assert [b.universe for b in union] == [b.universe for b in joined] == ["gradient"]
    assert [op.label for op in union[0].sites] == [op.label for op in joined[0].sites]


def test_grad_bridges_to_the_grad_fn_behind_its_op() -> None:
    """The one bridge: tl.grad's inner evaluated against the op behind a GradFn."""

    from torchlens._ir.selector_eval import evaluate

    trace = _trace(backward=True)
    first_relu, _ = _relu_labels(trace)
    relu_grad_fns = [
        site
        for site in trace._grad_fn_logs.values()
        if site.has_op and site.op_label == trace[first_relu].layer_label
    ]
    assert relu_grad_fns
    for site in relu_grad_fns:
        assert evaluate(tl.grad(tl.label(first_relu)), site, lifecycle="site")
        assert not evaluate(tl.label(first_relu), site, lifecycle="site")


def test_grad_target_spec_round_trip() -> None:
    """A gradient address persists as kind "grad" wrapping the inner target."""

    from torchlens._ir.selector_eval import normalize_selector_like

    for address in (tl.grad("relu_3"), tl.grad(tl.func("relu") & tl.in_module("block"))):
        spec = address.to_target_spec()
        assert spec.selector_kind == "grad"
        assert normalize_selector_like(spec, lifecycle="site") == address
        assert normalize_selector_like(spec.freeze(), lifecycle="site") == address


def test_root_exports_grad_and_drop_gradient_only_helpers() -> None:
    """A8: grad and clip_norm are exported; the six helpers are gone with no alias."""

    assert "grad" in tl.__all__
    assert "clip_norm" in tl.__all__
    assert tl.grad is _selectors.grad
    for removed in (
        "zero_grad",
        "scale_grad",
        "clamp_grad",
        "add_grad_noise",
        "clip_grad",
        "hook_grad",
    ):
        assert removed not in tl.__all__
        with pytest.raises(AttributeError):
            getattr(tl, removed)
