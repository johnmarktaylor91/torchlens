"""PYTEST_DONT_REWRITE

End-to-end coverage for the remaining Phase 8 conditional integration matrix.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Sequence

import pytest
import torch
import torch.nn as nn

pd = pytest.importorskip("pandas")

import torchlens.backends.torch.ops as output_tensors  # noqa: E402
import torchlens.backends.torch.sources as source_tensors  # noqa: E402
import torchlens.postprocess.ast_branches as ast_branches  # noqa: E402
import torchlens.postprocess.graph_traversal as graph_traversal  # noqa: E402
import torchlens.utils.introspection as introspection  # noqa: E402
from torchlens import trace as trace_fn  # noqa: E402
from torchlens.data_classes.layer import Layer  # noqa: E402
from torchlens.data_classes.op import Op  # noqa: E402
from torchlens.data_classes.trace import ConditionalEvent, Trace  # noqa: E402


class NestedIfThenIfModel(nn.Module):
    """Model with an inner ``if`` nested inside the outer THEN arm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if x.mean() > 0:
            y = torch.relu(x)
            if y.mean() > 0:
                z = torch.sigmoid(y)
            else:
                z = torch.tanh(y)
        else:
            z = torch.square(x)
        return z


class NestedInElseModel(nn.Module):
    """Model with an inner ``if`` nested inside the outer ELSE arm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if x.mean() > 0:
            y = torch.relu(x)
        else:
            y = torch.neg(x)
            if y.mean() > 0:
                y = torch.sigmoid(y)
            else:
                y = torch.tanh(y)
        return y


class MultilinePredicateModel(nn.Module):
    """Model whose predicate spans multiple physical lines."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if x.mean() > 0 and x.std() > 0:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class BranchUsesOnlyParameterModel(nn.Module):
    """Model whose branch entry comes directly from a parameter ancestor."""

    def __init__(self) -> None:
        """Initialise the branch-local parameter."""
        super().__init__()
        self.bias = nn.Parameter(torch.full((2, 2), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        baseline = torch.neg(x)
        if x.mean() > 0:
            y = torch.add(self.bias, baseline)
        else:
            y = torch.sub(baseline, 1)
        return y


class BranchUsesOnlyConstantModel(nn.Module):
    """Model whose branch entry comes directly from a constant-producing op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        baseline = torch.neg(x)
        if x.mean() > 0:
            y = torch.add(baseline, 1)
        else:
            y = torch.sub(baseline, 1)
        return y


class MultiArmEntryNestedModel(nn.Module):
    """Nested model whose one edge enters both an outer and inner arm."""

    def __init__(self) -> None:
        """Initialise the branch-local parameter."""
        super().__init__()
        self.bias = nn.Parameter(torch.full((2, 2), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        shared = torch.neg(x)
        if x.mean() > 0:
            if x.sum() > 0:
                y = torch.add(shared, 1)
            else:
                y = torch.sub(shared, 1)
        else:
            y = torch.zeros_like(x)
        return torch.add(y, x)


class IfBoolCastModel(nn.Module):
    """Model whose ``if`` predicate is wrapped in ``bool(...)``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if bool(x.mean() > 0):
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class ReconvergingBranchesModel(nn.Module):
    """Model whose THEN and ELSE arms reconverge before the output."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after branch reconvergence.
        """
        if x.sum() > 0:
            y = torch.mul(x, 2)
        else:
            y = torch.div(x, 2)
        return torch.add(y, 1)


class NestedHelperSameNameModel(nn.Module):
    """Model with two same-named nested helpers that both branch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sum of two helper outputs.
        """

        def outer_one(t: torch.Tensor) -> torch.Tensor:
            """Run the first helper scope.

            Parameters
            ----------
            t:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Helper output.
            """

            def helper() -> torch.Tensor:
                """Run the first nested helper.

                Returns
                -------
                torch.Tensor
                    Branch-selected tensor.
                """
                if t.mean() > 0:
                    return torch.relu(t)
                return torch.sigmoid(t)

            return helper()

        def outer_two(t: torch.Tensor) -> torch.Tensor:
            """Run the second helper scope.

            Parameters
            ----------
            t:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Helper output.
            """

            def helper() -> torch.Tensor:
                """Run the second nested helper.

                Returns
                -------
                torch.Tensor
                    Branch-selected tensor.
                """
                if torch.neg(t).mean() < 0:
                    return torch.tanh(t)
                return torch.square(t)

            return helper()

        return torch.add(outer_one(x), outer_two(x))


class NestedQualnameModel(nn.Module):
    """Model with a method and nested helper that share the same function name."""

    def helper(self, x: torch.Tensor) -> torch.Tensor:
        """Run the method-scoped helper.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected tensor.
        """
        if torch.neg(x).mean() < 0:
            return torch.tanh(x)
        return torch.square(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Sum of nested-helper and method-helper outputs.
        """

        def helper(t: torch.Tensor) -> torch.Tensor:
            """Run the nested helper.

            Parameters
            ----------
            t:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Branch-selected tensor.
            """
            if t.mean() > 0:
                return torch.relu(t)
            return torch.sigmoid(t)

        return torch.add(helper(x), self.helper(x))


class SameLineNestedDefModel(nn.Module):
    """Model used to force D14 fail-closed scope resolution in integration."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Helper output tensor.
        """

        def helper() -> torch.Tensor:
            """Run the nested helper.

            Returns
            -------
            torch.Tensor
                Branch-selected tensor.
            """
            if x.mean() > 0:
                return torch.relu(x)
            return torch.sigmoid(x)

        return helper()


def _forward_decorator(
    func: Callable[[nn.Module, torch.Tensor], torch.Tensor],
) -> Callable[[nn.Module, torch.Tensor], torch.Tensor]:
    """Wrap ``forward`` in a thin decorator that preserves the original call.

    Parameters
    ----------
    func:
        Forward method to wrap.

    Returns
    -------
    Callable[[nn.Module, torch.Tensor], torch.Tensor]
        Wrapper function.
    """

    def wrapper(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Invoke the wrapped forward method.

        Parameters
        ----------
        self:
            Bound module instance.
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Wrapped forward result.
        """
        return func(self, x)

    return wrapper


class DecoratedForwardModel(nn.Module):
    """Model whose ``forward`` method is decorated."""

    @_forward_decorator
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the decorated conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if x.mean() > 0:
            return torch.relu(x)
        return torch.sigmoid(x)


class BoolCastOnlyModel(nn.Module):
    """Model that uses ``bool(...)`` outside any control-flow branch."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with a Python-bool offset.
        """
        flag = bool(x.mean() > 0)
        return torch.add(x, float(flag))


class TernaryIfExpModel(nn.Module):
    """Minimal model that exercises ``ifexp`` attribution."""

    def __init__(self) -> None:
        """Initialise the ternary arm modules."""
        super().__init__()
        self.then_branch = nn.Linear(2, 2)
        self.else_branch = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ternary forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        return self.then_branch(x) if x.mean() > 0 else self.else_branch(x)


class ComprehensionIfModel(nn.Module):
    """Model with tensor bools consumed by a comprehension filter."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Reduced stacked tensor.
        """
        selected = [value for value in (x, torch.neg(x)) if value.mean() > 0]
        return torch.stack(selected).sum(dim=0)


class WhileLoopModel(nn.Module):
    """Model with tensor bools consumed by a ``while`` loop."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Iteratively reduced tensor.
        """
        count = 0
        while x.mean() > 0.2 and count < 3:
            x = torch.mul(x, 0.5)
            count += 1
        return x


class NotIfModel(nn.Module):
    """Model that negates a tensor-driven ``if`` predicate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if not (x.mean() < 0):
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class AndOrIfModel(nn.Module):
    """Model with a compound ``and``/``or`` tensor predicate."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if (x.mean() > 0 and x.std() > 0) or (x.max() > 10):
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class WalrusIfModel(nn.Module):
    """Model whose tensor predicate is stored via the walrus operator."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if (mean_val := x.mean()) > 0:
            y = torch.relu(torch.mul(mean_val, x))
        else:
            y = torch.sigmoid(x)
        return y


class NestedTernaryModel(nn.Module):
    """Model with one ``ifexp`` nested inside another."""

    def __init__(self) -> None:
        """Initialise the ternary arm modules."""
        super().__init__()
        self.inner_then = nn.Linear(2, 2)
        self.inner_else = nn.Linear(2, 2)
        self.outer_else = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the nested ternary forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        return (
            (self.inner_then(x) if x.mean() > 0 else self.inner_else(x))
            if x.sum() > 0
            else self.outer_else(x)
        )


class TernaryInsideIfModel(nn.Module):
    """Model with an ``ifexp`` nested inside a regular ``if`` THEN arm."""

    def __init__(self) -> None:
        """Initialise the ternary arm modules."""
        super().__init__()
        self.then_branch = nn.Linear(2, 2)
        self.else_branch = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the mixed conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if x.sum() > 0:
            y = self.then_branch(x) if x.mean() > 0 else self.else_branch(x)
        else:
            y = x
        return y


class TernaryWithBoolCastModel(nn.Module):
    """Model with a ternary predicate wrapped in ``bool(...)``."""

    def __init__(self) -> None:
        """Initialise the ternary arm modules."""
        super().__init__()
        self.then_branch = nn.Linear(2, 2)
        self.else_branch = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ternary forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        return self.then_branch(x) if bool(x.mean() > 0) else self.else_branch(x)


class TernaryPy310FailClosedModel(nn.Module):
    """Same-line ternary used for Python-version-gated column-offset assertions."""

    def __init__(self) -> None:
        """Initialise the ternary arm modules."""
        super().__init__()
        self.then_branch = nn.Linear(2, 2)
        self.else_branch = nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ternary forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        return self.then_branch(x) if x.mean() > 0 else self.else_branch(x)


class TernaryMultiOpOneLineModel(nn.Module):
    """Model with two multi-op ternary arms on the same physical line."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ternary forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        return (x * 2 + 1) if x.mean() > 0 else (x / 2 - 1)


class PythonBoolModel(nn.Module):
    """Model whose predicate is a pure Python bool."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if self.training:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class ItemScalarizationModel(nn.Module):
    """Model whose predicate scalarises through ``item()``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if x.mean().item() > 0:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class ShapePredicateModel(nn.Module):
    """Model whose predicate inspects tensor shape metadata."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if x.shape[0] > 0:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class SaveSourceContextOffAssertModel(nn.Module):
    """Model that combines ``assert`` with ``save_code_context=False``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Activated output tensor.
        """
        assert x.mean() > 0
        return torch.relu(x)


class KeepUnsavedLayersFalseModel(nn.Module):
    """Model used to verify conditional cleanup after pruning unsaved layers."""

    def __init__(self) -> None:
        """Initialise the branch-local modules."""
        super().__init__()
        self.then_branch = nn.Sequential(nn.Linear(2, 2), nn.ReLU())
        self.else_branch = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        baseline = torch.neg(x)
        if x.mean() > 0:
            y = self.then_branch(baseline)
        else:
            y = self.else_branch(baseline)
        return torch.add(y, 1)


class ToPandasConditionalModel(nn.Module):
    """Model used to exercise actual conditional rows in ``to_pandas()``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the conditional forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected output tensor.
        """
        if x.mean() > 0:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


class PublicIfElseArmModel(nn.Module):
    """Model used to verify public conditional arm fired statuses."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple if/else forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch-selected tensor.
        """

        if x.mean() > 0:
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y


def _log_model(
    model: nn.Module,
    x: torch.Tensor,
    *,
    save_code_context: bool = True,
    layers_to_save: str | Sequence[str] | None = "all",
) -> Trace:
    """Capture a ``Trace`` for a small integration-test model.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Input tensor.
    save_code_context:
        Whether rich source loading is enabled during capture.
    layers_to_save:
        Activation-saving selection passed through to ``trace``.

    Returns
    -------
    Trace
        Fully postprocessed model log.
    """
    return trace_fn(
        model,
        x,
        save_code_context=save_code_context,
        layers_to_save=layers_to_save,
    )


def _get_terminal_bool_layers(trace: Trace) -> list[Op]:
    """Return terminal scalar bool layers from a model log.

    Parameters
    ----------
    trace:
        Logged model execution.

    Returns
    -------
    list[Op]
        Terminal scalar bool layers in execution order.
    """
    return [layer for layer in trace.layer_list if layer.is_terminal_bool and layer.is_scalar_bool]


def _find_only_layer(
    trace: Trace,
    func_name: str,
    branch_stack: list[tuple[int, str]] | None = None,
) -> Op:
    """Find the unique layer matching a function name and optional branch stack.

    Parameters
    ----------
    trace:
        Logged model execution.
    func_name:
        Function name to match.
    branch_stack:
        Optional exact conditional branch stack to require.

    Returns
    -------
    Op
        Matching layer.
    """
    matching_layers = [layer for layer in trace.layer_list if layer.func_name == func_name]
    if branch_stack is not None:
        matching_layers = [
            layer for layer in matching_layers if layer.conditional_branch_stack == branch_stack
        ]
    assert len(matching_layers) == 1, (
        f"Expected one {func_name!r} layer for stack {branch_stack}, found {len(matching_layers)}"
    )
    return matching_layers[0]


def _find_only_layer_any_name(
    trace: Trace,
    func_names: Sequence[str],
) -> Op:
    """Find the unique layer whose function name matches any provided option.

    Parameters
    ----------
    trace:
        Logged model execution.
    func_names:
        Acceptable function names.

    Returns
    -------
    Op
        Matching layer.
    """
    matching_layers = [layer for layer in trace.layer_list if layer.func_name in set(func_names)]
    assert len(matching_layers) == 1
    return matching_layers[0]


def _find_only_layer_log(
    trace: Trace,
    func_name: str,
    predicate: Callable[[Layer], bool],
) -> Layer:
    """Find one aggregate ``Layer`` matching the provided predicate.

    Parameters
    ----------
    trace:
        Logged model execution.
    func_name:
        Function name to match.
    predicate:
        Additional filter over candidate layer logs.

    Returns
    -------
    Layer
        Matching aggregate layer log.
    """
    matching_layers = [
        layer_log
        for layer_log in trace.layer_logs.values()
        if layer_log.func_name == func_name and predicate(layer_log)
    ]
    assert len(matching_layers) == 1
    return matching_layers[0]


def _get_root_event(trace: Trace) -> ConditionalEvent:
    """Return the sole root conditional event in a model log.

    Parameters
    ----------
    trace:
        Logged model execution.

    Returns
    -------
    ConditionalEvent
        Root event with no parent conditional.
    """
    root_events = [
        event for event in trace.conditional_records if event.parent_conditional_id is None
    ]
    assert len(root_events) == 1
    return root_events[0]


def _get_child_event(trace: Trace, parent_id: int) -> ConditionalEvent:
    """Return the sole child event whose parent id matches ``parent_id``.

    Parameters
    ----------
    trace:
        Logged model execution.
    parent_id:
        Parent conditional id.

    Returns
    -------
    ConditionalEvent
        Nested conditional event.
    """
    child_events = [
        event for event in trace.conditional_records if event.parent_conditional_id == parent_id
    ]
    assert len(child_events) == 1
    return child_events[0]


def _collect_branch_child_labels(
    branch_children_by_cond: dict[int, dict[str, list[str]]],
) -> set[str]:
    """Collect child labels from a nested ``conditional_arm_children`` mapping.

    Parameters
    ----------
    branch_children_by_cond:
        Conditional child mapping from a layer or layer log.

    Returns
    -------
    set[str]
        Flat set of every referenced child label.
    """
    return {
        child_label
        for branch_children in branch_children_by_cond.values()
        for child_labels in branch_children.values()
        for child_label in child_labels
    }


def _collect_model_conditional_labels(trace: Trace) -> set[str]:
    """Collect every layer label referenced by model-level conditional metadata.

    Parameters
    ----------
    trace:
        Logged model execution.

    Returns
    -------
    set[str]
        Flat set of all referenced layer labels.
    """
    referenced_labels: set[str] = set()
    for parent_label, child_label in trace.conditional_branch_edges:
        referenced_labels.add(parent_label)
        referenced_labels.add(child_label)
    for parent_label, child_label in trace.conditional_then_entry_edges:
        referenced_labels.add(parent_label)
        referenced_labels.add(child_label)
    for _, _, parent_label, child_label in trace.conditional_elif_entry_edges:
        referenced_labels.add(parent_label)
        referenced_labels.add(child_label)
    for _, parent_label, child_label in trace.conditional_else_entry_edges:
        referenced_labels.add(parent_label)
        referenced_labels.add(child_label)
    for edge_list in trace.conditional_arm_entry_edges.values():
        for parent_label, child_label in edge_list:
            referenced_labels.add(parent_label)
            referenced_labels.add(child_label)
    for event in trace.conditional_records:
        referenced_labels.update(event.bool_layers)
    return referenced_labels


def _assert_branchless_log(trace: Trace) -> None:
    """Assert that a model log contains no conditional attribution metadata.

    Parameters
    ----------
    trace:
        Logged model execution.
    """
    assert trace.conditional_records == []
    assert trace.conditional_branch_edges == []
    assert trace.conditional_arm_entry_edges == {}
    assert trace.conditional_then_entry_edges == []
    assert trace.conditional_elif_entry_edges == []
    assert trace.conditional_else_entry_edges == []


def test_nested_if_then_if_model_materializes_nested_branch_stack() -> None:
    """Nested THEN->THEN execution records two events and a two-deep branch stack."""
    trace = _log_model(NestedIfThenIfModel(), torch.ones(2, 2))

    root_event = _get_root_event(trace)
    inner_event = _get_child_event(trace, root_event.id)
    relu_layer = _find_only_layer(trace, "relu")
    sigmoid_layer = _find_only_layer(trace, "sigmoid")

    assert len(trace.conditional_records) == 2
    assert root_event.kind == "if_chain"
    assert inner_event.kind == "if_chain"
    assert inner_event.parent_branch_kind == "then"
    assert relu_layer.conditional_branch_stack == [(root_event.id, "then")]
    assert sigmoid_layer.conditional_branch_stack == [
        (root_event.id, "then"),
        (inner_event.id, "then"),
    ]


def test_nested_in_else_model_materializes_else_to_inner_then_stack() -> None:
    """Nested ELSE->THEN execution records the outer ELSE parent branch."""
    trace = _log_model(NestedInElseModel(), -torch.ones(2, 2))

    root_event = _get_root_event(trace)
    inner_event = _get_child_event(trace, root_event.id)
    neg_layer = _find_only_layer(trace, "neg")
    sigmoid_layer = _find_only_layer(trace, "sigmoid")

    assert len(trace.conditional_records) == 2
    assert inner_event.parent_branch_kind == "else"
    assert neg_layer.conditional_branch_stack == [(root_event.id, "else")]
    assert sigmoid_layer.conditional_branch_stack == [
        (root_event.id, "else"),
        (inner_event.id, "then"),
    ]


def test_multiline_predicate_model_tracks_one_if_chain_event() -> None:
    """A multiline predicate still materialises one attributed ``if_chain`` event."""
    input_tensor = torch.tensor([[0.5, 1.5], [0.25, 2.0]])
    trace = _log_model(MultilinePredicateModel(), input_tensor)

    event = _get_root_event(trace)
    bool_layers = _get_terminal_bool_layers(trace)
    relu_layer = _find_only_layer(trace, "relu")

    assert len(trace.conditional_records) == 1
    assert event.kind == "if_chain"
    assert set(event.branch_ranges) == {"then", "else"}
    assert len(bool_layers) == 2
    assert all(layer.conditional_context_kind == "if_test" for layer in bool_layers)
    assert all(layer.is_terminal_conditional_bool is True for layer in bool_layers)
    assert relu_layer.conditional_branch_stack == [(event.id, "then")]


def test_public_conditionals_materialize_else_arm_fired_status() -> None:
    """Public conditional records expose fired ELSE and unfired THEN arms."""
    then_trace = _log_model(PublicIfElseArmModel(), torch.ones(2, 2))
    else_trace = _log_model(PublicIfElseArmModel(), -torch.ones(2, 2))

    then_conditional = then_trace.conditionals[0]
    else_conditional = else_trace.conditionals[0]

    assert then_conditional.fired_arm_kind == "then"
    assert [(arm.kind, arm.fired) for arm in then_conditional.arms] == [
        ("then", True),
        ("else", False),
    ]
    assert else_conditional.fired_arm_kind == "else"
    assert [(arm.kind, arm.fired) for arm in else_conditional.arms] == [
        ("then", False),
        ("else", True),
    ]


def test_branch_uses_only_parameter_model_records_parameter_entry_edge() -> None:
    """A parameter-only branch body still records a THEN arm-entry edge."""
    trace = _log_model(BranchUsesOnlyParameterModel(), torch.ones(2, 2))

    event = _get_root_event(trace)
    then_edges = trace.conditional_arm_entry_edges[(event.id, "then")]

    assert len(then_edges) == 1
    parent_label, child_label = then_edges[0]
    assert trace[parent_label].conditional_branch_stack == []
    assert trace[child_label].conditional_branch_stack == [(event.id, "then")]


def test_branch_uses_only_constant_model_records_constant_entry_edge() -> None:
    """A constant-only branch body still records a THEN arm-entry edge."""
    trace = _log_model(BranchUsesOnlyConstantModel(), torch.ones(2, 2))

    event = _get_root_event(trace)
    then_edges = trace.conditional_arm_entry_edges[(event.id, "then")]

    assert len(then_edges) == 1
    parent_label, child_label = then_edges[0]
    assert trace[parent_label].conditional_branch_stack == []
    assert trace[child_label].conditional_branch_stack == [(event.id, "then")]


def test_multi_arm_entry_nested_model_duplicates_entry_edge_per_conditional() -> None:
    """One nested branch-entry edge is recorded for both gained THEN arms."""
    trace = _log_model(MultiArmEntryNestedModel(), torch.ones(2, 2))

    outer_event = _get_root_event(trace)
    inner_event = _get_child_event(trace, outer_event.id)
    outer_then_edges = set(trace.conditional_arm_entry_edges[(outer_event.id, "then")])
    inner_then_edges = set(trace.conditional_arm_entry_edges[(inner_event.id, "then")])
    shared_edges = outer_then_edges & inner_then_edges

    assert len(shared_edges) == 1
    parent_label, child_label = next(iter(shared_edges))

    parent_layer = trace[parent_label]
    assert parent_layer.conditional_arm_children[outer_event.id]["then"] == [child_label]
    assert parent_layer.conditional_arm_children[inner_event.id]["then"] == [child_label]
    assert parent_layer.conditional_then_children == [child_label]


def test_if_bool_cast_model_marks_wrapper_kind_without_losing_branch_attribution() -> None:
    """``bool(...)`` wrappers inside ``if`` tests remain branch-participating."""
    trace = _log_model(IfBoolCastModel(), torch.ones(2, 2))

    event = _get_root_event(trace)
    bool_layer = _get_terminal_bool_layers(trace)[0]
    relu_layer = _find_only_layer(trace, "relu")

    assert bool_layer.conditional_context_kind == "if_test"
    assert bool_layer.is_terminal_conditional_bool is True
    assert bool_layer.conditional_wrapper_kind == "bool_cast"
    assert bool_layer.terminal_conditional_id == event.id
    assert relu_layer.conditional_branch_stack == [(event.id, "then")]


def test_reconverging_branches_model_clears_branch_stack_after_merge() -> None:
    """Ops after reconvergence have empty stacks even though their parent was in-branch."""
    positive_log = _log_model(ReconvergingBranchesModel(), torch.ones(2, 2))
    negative_log = _log_model(ReconvergingBranchesModel(), -torch.ones(2, 2))

    positive_add = _find_only_layer(positive_log, "add")
    negative_add = _find_only_layer(negative_log, "add")
    positive_parent = positive_log[positive_add.parents[0]]
    negative_parent = negative_log[negative_add.parents[0]]

    assert positive_add.conditional_branch_stack == []
    assert negative_add.conditional_branch_stack == []
    assert positive_parent.conditional_branch_stack != []
    assert negative_parent.conditional_branch_stack != []


def test_nested_helper_same_name_model_distinguishes_helpers_by_scope() -> None:
    """Two same-named nested helpers materialise separate conditional ids."""
    trace = _log_model(NestedHelperSameNameModel(), torch.ones(2, 2))

    relu_layer = _find_only_layer(trace, "relu")
    tanh_layer = _find_only_layer(trace, "tanh")
    bool_layers = _get_terminal_bool_layers(trace)

    assert len(trace.conditional_records) == 2
    assert len({layer.terminal_conditional_id for layer in bool_layers}) == 2
    assert relu_layer.conditional_branch_stack == [(bool_layers[0].terminal_conditional_id, "then")]
    assert tanh_layer.conditional_branch_stack == [(bool_layers[1].terminal_conditional_id, "then")]


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="func_qualname is only available on Python 3.11+"
)
def test_nested_qualname_model_distinguishes_method_and_nested_helper() -> None:
    """Distinct qualnames keep a method helper separate from a nested helper."""
    trace = _log_model(NestedQualnameModel(), torch.ones(2, 2))

    root_events = sorted(trace.conditional_records, key=lambda event: event.function_qualname)
    relu_layer = _find_only_layer(trace, "relu")
    tanh_layer = _find_only_layer(trace, "tanh")

    assert len(root_events) == 2
    assert root_events[0].function_qualname != root_events[1].function_qualname
    nested_event = next(
        event for event in root_events if "<locals>.helper" in event.function_qualname
    )
    method_event = next(event for event in root_events if "<locals>" not in event.function_qualname)
    assert relu_layer.conditional_branch_stack == [(nested_event.id, "then")]
    assert tanh_layer.conditional_branch_stack == [(method_event.id, "then")]


def test_same_line_nested_def_model_fails_closed_when_scope_resolution_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous fallback scope resolution leaves branch stacks empty instead of guessing."""

    original_get_stack = introspection._get_code_context
    original_get_file_index = ast_branches.get_file_index

    def _stack_without_qualname(*args: object, **kwargs: object) -> list[object]:
        """Return a call stack with ``func_qualname`` cleared on every frame."""
        stack = original_get_stack(*args, **kwargs)
        for frame in stack:
            frame.func_qualname = None
        return stack

    def _ambiguous_file_index(filename: str) -> object:
        """Return a file index with a duplicate helper scope for the test file."""
        index = original_get_file_index(filename)
        if index is None or filename != __file__:
            return index
        helper_scopes = [
            scope
            for scope in index.scopes
            if scope.qualname == "SameLineNestedDefModel.forward.<locals>.helper"
        ]
        if len(helper_scopes) >= 1 and not any(
            scope.qualname == "shadow.<locals>.helper" for scope in index.scopes
        ):
            index.scopes.append(replace(helper_scopes[0], qualname="shadow.<locals>.helper"))
        return index

    monkeypatch.setattr(introspection, "_get_code_context", _stack_without_qualname)
    monkeypatch.setattr(output_tensors, "_get_code_context", _stack_without_qualname)
    monkeypatch.setattr(source_tensors, "_get_code_context", _stack_without_qualname)
    monkeypatch.setattr(graph_traversal, "_get_code_context", _stack_without_qualname)
    monkeypatch.setattr(ast_branches, "get_file_index", _ambiguous_file_index)

    trace = _log_model(SameLineNestedDefModel(), torch.ones(2, 2))
    relu_layer = _find_only_layer(trace, "relu")

    assert relu_layer.conditional_branch_stack == []
    assert trace.conditional_arm_entry_edges == {}


def test_decorated_forward_model_preserves_or_gracefully_skips_branch_attribution() -> None:
    """Decorated ``forward`` either attributes correctly or degrades without false labels."""
    trace = _log_model(DecoratedForwardModel(), torch.ones(2, 2))
    relu_layer = _find_only_layer(trace, "relu")

    if trace.conditional_records:
        event = _get_root_event(trace)
        assert relu_layer.conditional_branch_stack == [(event.id, "then")]
    else:
        assert relu_layer.conditional_branch_stack == []
        assert trace.conditional_arm_entry_edges == {}


def test_bool_cast_only_model_is_not_treated_as_branch() -> None:
    """Standalone ``bool(...)`` casts stay non-branching."""
    trace = _log_model(BoolCastOnlyModel(), torch.ones(2, 2))
    bool_layer = _get_terminal_bool_layers(trace)[0]

    assert bool_layer.conditional_context_kind == "bool_cast"
    assert bool_layer.is_terminal_conditional_bool is False
    assert bool_layer.terminal_conditional_id is None
    _assert_branchless_log(trace)


def test_ternary_ifexp_model_attributes_then_arm_as_ifexp() -> None:
    """A minimal ternary materialises an ``ifexp`` event and branch stack."""
    trace = _log_model(TernaryIfExpModel(), torch.ones(1, 2))

    event = _get_root_event(trace)
    bool_layer = _get_terminal_bool_layers(trace)[0]
    linear_layer = _find_only_layer(trace, "linear", [(event.id, "then")])

    assert event.kind == "ifexp"
    assert set(event.branch_ranges) == {"then", "else"}
    assert bool_layer.conditional_context_kind == "ifexp"
    assert bool_layer.is_terminal_conditional_bool is True
    assert linear_layer.conditional_branch_stack == [(event.id, "then")]


def test_comprehension_if_model_classifies_bool_without_materialising_branch_metadata() -> None:
    """Comprehension filters classify bools but do not create branch events."""
    trace = _log_model(ComprehensionIfModel(), torch.ones(2, 2))
    bool_layers = _get_terminal_bool_layers(trace)

    assert len(bool_layers) == 2
    assert all(layer.conditional_context_kind == "comprehension_filter" for layer in bool_layers)
    assert all(layer.is_terminal_conditional_bool is False for layer in bool_layers)
    _assert_branchless_log(trace)


def test_while_loop_model_classifies_bool_without_materialising_branch_metadata() -> None:
    """``while`` predicates are classified but never attributed as branches."""
    trace = _log_model(WhileLoopModel(), torch.full((2, 2), 0.5))
    bool_layers = _get_terminal_bool_layers(trace)

    assert len(bool_layers) >= 1
    assert all(layer.conditional_context_kind == "while" for layer in bool_layers)
    assert all(layer.is_terminal_conditional_bool is False for layer in bool_layers)
    _assert_branchless_log(trace)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="match guards require Python 3.10+")
def test_match_guard_model_classifies_bool_without_materialising_branch_metadata(
    tmp_path: Path,
) -> None:
    """``match`` guards classify bools as ``match_guard`` and stop there."""
    source_path = tmp_path / "match_guard_model.py"
    source_path.write_text(
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class MatchGuardModel(nn.Module):",
                '    """Model whose tensor bool is consumed by a match guard."""',
                "",
                "    def forward(self, x):",
                "        match 1:",
                "            case 1 if x.mean() > 0:",
                "                return torch.relu(x)",
                "            case _:",
                "                return torch.sigmoid(x)",
            ]
        ),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("match_guard_model", source_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.MatchGuardModel()
    trace = _log_model(model, torch.ones(2, 2))
    bool_layer = _get_terminal_bool_layers(trace)[0]

    assert bool_layer.conditional_context_kind == "match_guard"
    assert bool_layer.is_terminal_conditional_bool is False
    _assert_branchless_log(trace)


def test_not_if_model_attributes_branch_despite_boolean_negation() -> None:
    """Negating an ``if`` predicate still produces a normal ``if_chain`` event."""
    trace = _log_model(NotIfModel(), torch.ones(2, 2))

    event = _get_root_event(trace)
    relu_layer = _find_only_layer(trace, "relu")

    assert event.kind == "if_chain"
    assert relu_layer.conditional_branch_stack == [(event.id, "then")]


def test_and_or_if_model_keeps_all_branching_bools_on_one_event() -> None:
    """Compound ``and``/``or`` predicates keep all terminal bools on one event."""
    input_tensor = torch.tensor([[0.5, 1.5], [0.25, 2.0]])
    trace = _log_model(AndOrIfModel(), input_tensor)

    event = _get_root_event(trace)
    bool_layers = _get_terminal_bool_layers(trace)
    relu_layer = _find_only_layer(trace, "relu")

    assert len(bool_layers) >= 2
    assert all(layer.conditional_context_kind == "if_test" for layer in bool_layers)
    assert {layer.terminal_conditional_id for layer in bool_layers} == {event.id}
    assert relu_layer.conditional_branch_stack == [(event.id, "then")]


def test_walrus_if_model_attributes_branch_normally() -> None:
    """Walrus-bound tensor predicates still drive normal branch attribution."""
    trace = _log_model(WalrusIfModel(), torch.ones(2, 2))

    event = _get_root_event(trace)
    relu_layer = _find_only_layer(trace, "relu")

    assert event.kind == "if_chain"
    assert relu_layer.conditional_branch_stack == [(event.id, "then")]


def test_nested_ternary_model_records_parent_child_ifexp_events() -> None:
    """Nested ternaries keep parent-child links between ``ifexp`` events."""
    trace = _log_model(NestedTernaryModel(), torch.ones(1, 2))

    outer_event = _get_root_event(trace)
    inner_event = _get_child_event(trace, outer_event.id)
    linear_layer = _find_only_layer(
        trace,
        "linear",
        [(outer_event.id, "then"), (inner_event.id, "then")],
    )

    assert len(trace.conditional_records) == 2
    assert outer_event.kind == "ifexp"
    assert inner_event.kind == "ifexp"
    assert inner_event.parent_branch_kind == "then"
    assert linear_layer.conditional_branch_stack == [
        (outer_event.id, "then"),
        (inner_event.id, "then"),
    ]


def test_ternary_inside_if_model_records_mixed_if_and_ifexp_stack() -> None:
    """An inner ternary keeps both the outer ``if_chain`` and inner ``ifexp`` stack."""
    trace = _log_model(TernaryInsideIfModel(), torch.ones(1, 2))

    outer_event = next(event for event in trace.conditional_records if event.kind == "if_chain")
    inner_event = next(event for event in trace.conditional_records if event.kind == "ifexp")
    linear_layer = _find_only_layer(
        trace,
        "linear",
        [(outer_event.id, "then"), (inner_event.id, "then")],
    )

    assert inner_event.parent_conditional_id == outer_event.id
    assert inner_event.parent_branch_kind == "then"
    assert linear_layer.conditional_branch_stack == [
        (outer_event.id, "then"),
        (inner_event.id, "then"),
    ]


def test_ternary_with_bool_cast_model_marks_conditional_wrapper_kind() -> None:
    """``bool(...)`` wrappers inside ternaries keep ``ifexp`` attribution."""
    trace = _log_model(TernaryWithBoolCastModel(), torch.ones(1, 2))

    event = _get_root_event(trace)
    bool_layer = _get_terminal_bool_layers(trace)[0]
    linear_layer = _find_only_layer(trace, "linear", [(event.id, "then")])

    assert event.kind == "ifexp"
    assert bool_layer.conditional_context_kind == "ifexp"
    assert bool_layer.conditional_wrapper_kind == "bool_cast"
    assert bool_layer.is_terminal_conditional_bool is True
    assert linear_layer.conditional_branch_stack == [(event.id, "then")]


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="degraded ternary mode is only meaningful before Python 3.11",
)
def test_ternary_py310_fail_closed_model_drops_same_line_arm_attribution() -> None:
    """Pre-3.11 same-line ternaries fail closed when no column offsets are available."""
    trace = _log_model(TernaryPy310FailClosedModel(), torch.ones(1, 2))

    linear_layers = [layer for layer in trace.layer_list if layer.func_name == "linear"]

    assert len(trace.conditional_records) == 1
    assert trace.conditional_arm_entry_edges == {}
    assert all(layer.conditional_branch_stack == [] for layer in linear_layers)


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="column offsets are only captured on Python 3.11+"
)
def test_ternary_py311_records_non_none_column_offsets_for_ifexp_frames() -> None:
    """Python 3.11+ captures ``col_offset`` on ternary frames."""
    trace = _log_model(TernaryPy310FailClosedModel(), torch.ones(1, 2))
    bool_layer = _get_terminal_bool_layers(trace)[0]

    assert any(frame.col_offset is not None for frame in bool_layer.code_context)
    assert trace.conditional_arm_entry_edges


def test_ternary_multi_op_one_line_model_attributes_each_arm_by_column_offset() -> None:
    """Same-line multi-op ternary arms are separated by column offsets."""
    positive_log = _log_model(TernaryMultiOpOneLineModel(), torch.ones(2, 2))
    negative_log = _log_model(TernaryMultiOpOneLineModel(), -torch.ones(2, 2))

    positive_event = _get_root_event(positive_log)
    negative_event = _get_root_event(negative_log)

    positive_mul = _find_only_layer_any_name(positive_log, ("mul", "__mul__"))
    positive_add = _find_only_layer_any_name(positive_log, ("add", "__add__"))
    negative_div = _find_only_layer_any_name(negative_log, ("div", "__truediv__"))
    negative_sub = _find_only_layer_any_name(negative_log, ("sub", "__sub__"))

    assert positive_mul.conditional_branch_stack == [(positive_event.id, "then")]
    assert positive_add.conditional_branch_stack == [(positive_event.id, "then")]
    assert negative_div.conditional_branch_stack == [(negative_event.id, "else")]
    assert negative_sub.conditional_branch_stack == [(negative_event.id, "else")]


@pytest.mark.parametrize(
    ("model", "input_tensor"),
    [
        (PythonBoolModel(), torch.ones(2, 2)),
        (ItemScalarizationModel(), torch.ones(2, 2)),
        (ShapePredicateModel(), torch.ones(2, 2)),
    ],
)
def test_documented_false_negative_scalar_predicates_do_not_materialise_events(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> None:
    """Python, ``item()``, and shape predicates remain documented false negatives."""
    trace = _log_model(model, input_tensor)
    branch_layer = next(
        layer
        for layer in trace.layer_list
        if layer.func_name in {"relu", "sigmoid"} and not layer.is_output
    )

    _assert_branchless_log(trace)
    assert branch_layer.conditional_branch_stack == []


def test_torch_where_model_stays_outside_branch_attribution() -> None:
    """Functional conditionals like ``torch.where`` do not create branch events."""

    class TorchWhereModel(nn.Module):
        """Model whose conditional behavior comes from ``torch.where``."""

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the functional conditional forward pass.

            Parameters
            ----------
            x:
                Input tensor.

            Returns
            -------
            torch.Tensor
                Output tensor from ``torch.where``.
            """
            return torch.where(x > 0.5, x, torch.zeros_like(x))

    model = TorchWhereModel()
    trace = _log_model(model, torch.full((2, 2), 0.75))

    _assert_branchless_log(trace)
    assert _get_terminal_bool_layers(trace) == []


def test_save_code_context_off_assert_model_has_no_false_positive_if_edges() -> None:
    """Disabled source loading plus ``assert`` must not fabricate branch metadata."""
    trace = _log_model(
        SaveSourceContextOffAssertModel(),
        torch.ones(2, 2),
        save_code_context=False,
    )
    bool_layer = _get_terminal_bool_layers(trace)[0]

    assert bool_layer.conditional_context_kind == "assert"
    assert bool_layer.is_terminal_conditional_bool is False
    assert len(bool_layer.code_context) > 0
    assert all(frame.source_loading_enabled is False for frame in bool_layer.code_context)
    _assert_branchless_log(trace)


def test_selective_save_preserves_all_conditional_surfaces() -> None:
    """Selective activation saving keeps conditional metadata addressable."""

    input_tensor = torch.ones(1, 2)
    log = _log_model(
        KeepUnsavedLayersFalseModel(),
        input_tensor,
        save_code_context=True,
        layers_to_save=[-1],
    )

    assert log.num_saved_ops == 2
    assert set(log.saved_ops.keys()) == {"add_1_6:1", "output_1:1"}
    assert _collect_model_conditional_labels(log).issubset(set(log.layer_labels))
    assert all(
        parent_label in log.layer_logs and child_label in log.layer_logs
        for parent_label, child_label, _, _ in log.conditional_edge_call_indices
    )


def test_to_pandas_conditional_model_populates_live_conditional_columns() -> None:
    """Real conditional execution populates the exported conditional DataFrame columns."""
    trace = _log_model(ToPandasConditionalModel(), torch.ones(2, 2), layers_to_save="all")
    layer_df = trace.to_pandas()
    bool_layer = _get_terminal_bool_layers(trace)[0]
    branch_layer = _find_only_layer(trace, "relu")

    bool_row = layer_df.loc[layer_df["layer_label"] == bool_layer.layer_label].iloc[0]
    branch_row = layer_df.loc[layer_df["layer_label"] == branch_layer.layer_label].iloc[0]

    assert isinstance(layer_df, pd.DataFrame)
    assert bool(bool_row["is_terminal_conditional_bool"]) is True
    assert bool_row["conditional_context_kind"] == "if_test"
    assert int(bool_row["terminal_conditional_id"]) == 0
    assert int(branch_row["conditional_branch_depth"]) == 1
    assert branch_row["conditional_branch_stack"] == "cond_0:then"
    assert branch_row["conditional_then_children"] == []
