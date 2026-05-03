"""PYTEST_DONT_REWRITE

Phase 8 integration matrix for conditional branch attribution.
"""

from __future__ import annotations

import linecache
import pickle
import sys
import tempfile
from dataclasses import replace
from typing import Callable, Sequence

import pandas as pd
import pytest
import torch
import torch.nn as nn

import torchlens.capture.output_tensors as output_tensors
import torchlens.capture.source_tensors as source_tensors
import torchlens.postprocess.ast_branches as ast_branches
import torchlens.postprocess.graph_traversal as graph_traversal
import torchlens.utils.introspection as introspection
from example_models import TorchWhereModel
from test_conditional_multipass import (
    AlternatingRecurrentIfModel,
    LoopedIfAlternatingModel,
    RolledMixedArmModel,
)
from test_conditional_rendering import BranchEntryWithArgLabelModel
from test_conditional_step5 import ElifLadderModel, SimpleIfElseModel
from torchlens import check_metadata_invariants, trace as trace_fn
from torchlens.data_classes.layer_log import LayerLog
from torchlens.data_classes.op_log import OpLog
from torchlens.data_classes.model_log import ConditionalEvent, Trace


@pytest.fixture(autouse=True)
def _isolate_ast_and_linecache() -> None:
    """Reset AST index cache and linecache before each test.

    Conditional tests depend on fresh parses of this file and ``example_models``.
    Earlier test runs can populate ``linecache`` with stale content or leave the
    ast_branches file cache holding entries whose observable behaviour differs
    from the current on-disk source (e.g., when pytest fixtures temporarily
    rewrite a module). Clearing both at the start of every test makes the
    suite order-independent.
    """

    ast_branches.invalidate_cache()
    linecache.clearcache()


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
        # fmt: off
        if (
            x.mean() > 0
            and x.std() > 0
        ):
            y = torch.relu(x)
        else:
            y = torch.sigmoid(x)
        return y
        # fmt: on


class BranchUsesOnlyParameterModel(nn.Module):
    """Model whose branch entry comes from a non-predicate parameter ancestor."""

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
            Branch-selected output tensor.
        """
        bias_view = self.bias + 0
        if x.mean() > 0:
            y = bias_view + 1
        else:
            y = x - 1
        return y


class BranchUsesOnlyConstantModel(nn.Module):
    """Model whose branch entry comes from a non-predicate constant ancestor."""

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
        const = torch.ones_like(x)
        if x.mean() > 0:
            y = const + 1
        else:
            y = x - 1
        return y


class MultiArmEntryNestedModel(nn.Module):
    """Model whose one ancestor edge enters both an outer and inner THEN arm."""

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
            Branch-selected output tensor.
        """
        bias_view = self.bias + 0
        if x.mean() > 0:
            if x.sum() > 0:
                y = bias_view + 1
            else:
                y = bias_view - 1
        else:
            y = torch.zeros_like(x)
        return y + x


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
        if bool(x.sum() > 0):
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
            Reconverged output tensor.
        """
        if x.sum() > 0:
            branch_value = torch.mul(x, 2)
        else:
            branch_value = torch.div(x, 2)
        return torch.add(branch_value, 1)


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
            Sum of both helper outputs.
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
                Helper output tensor.
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
                Helper output tensor.
            """

            def helper() -> torch.Tensor:
                """Run the second nested helper.

                Returns
                -------
                torch.Tensor
                    Branch-selected tensor.
                """
                if torch.neg(t).mean() > 0:
                    return torch.square(t)
                return torch.tanh(t)

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
        if torch.neg(x).mean() > 0:
            return torch.square(x)
        return torch.tanh(x)

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
    """Model used to force D14 fail-closed scope resolution."""

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
    """Wrap ``forward`` in a thin decorator.

    Parameters
    ----------
    func:
        Forward function to wrap.

    Returns
    -------
    Callable[[nn.Module, torch.Tensor], torch.Tensor]
        Wrapper function preserving the original behavior.
    """

    def wrapper(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Invoke the wrapped ``forward`` method.

        Parameters
        ----------
        self:
            Bound module instance.
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Wrapped output tensor.
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


class AssertTensorCondModel(nn.Module):
    """Model whose tensor bool is consumed by ``assert`` instead of branching."""

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
            y = torch.relu(mean_val * x)
        else:
            y = torch.sigmoid(x)
        return y


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


class SaveSourceContextOffModel(nn.Module):
    """Model used to strengthen the ``save_code_context=False`` path."""

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


class BasicTernaryModel(nn.Module):
    """Minimal model using a ternary expression."""

    def __init__(self) -> None:
        """Initialise the two ternary branches."""
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


class TernaryIfExpModel(BasicTernaryModel):
    """Placeholder model for the false-positive bucket that still uses ``ifexp``."""


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
    """Same-line ternary used for Python-version-gated fail-closed assertions."""

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
    """Model with two multi-op ternary arms on one physical line."""

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
        return torch.add(torch.mul(x, 2), 1) if x.mean() > 0 else torch.sub(torch.div(x, 2), 1)


def _log_model(
    model: nn.Module,
    x: torch.Tensor,
    *,
    save_code_context: bool = True,
    keep_unsaved_layers: bool = True,
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
    keep_unsaved_layers:
        Whether unsaved layers remain in the final log.
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
        keep_unsaved_layers=keep_unsaved_layers,
        layers_to_save=layers_to_save,
    )


def _get_terminal_bool_layers(trace: Trace) -> list[OpLog]:
    """Return terminal scalar bool layers from a model log.

    Parameters
    ----------
    trace:
        Logged model execution.

    Returns
    -------
    list[OpLog]
        Terminal scalar bool layers in execution order.
    """
    return [layer for layer in trace.layer_list if layer.is_terminal_bool and layer.is_scalar_bool]


def _find_only_layer(
    trace: Trace,
    func_name: str,
    branch_stack: list[tuple[int, str]] | None = None,
) -> OpLog:
    """Find the unique layer matching a function name and optional branch stack.

    Parameters
    ----------
    trace:
        Logged model execution.
    func_name:
        Function name to match.
    branch_stack:
        Optional exact branch stack to require.

    Returns
    -------
    OpLog
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


def _find_only_layer_log(
    trace: Trace,
    func_name: str,
    predicate: Callable[[LayerLog], bool],
) -> LayerLog:
    """Find one aggregate ``LayerLog`` matching the provided predicate.

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
    LayerLog
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


def _assert_derived_views_consistent(trace: Trace) -> None:
    """Assert derived conditional views match the primary structures.

    Parameters
    ----------
    trace:
        Logged model execution.
    """
    expected_then_edges = [
        (parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in trace.conditional_arm_entry_edges.items()
        if branch_kind == "then"
        for parent_label, child_label in edge_list
    ]
    expected_elif_edges = [
        (conditional_id, int(branch_kind.split("_")[1]), parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in trace.conditional_arm_entry_edges.items()
        if branch_kind.startswith("elif_")
        for parent_label, child_label in edge_list
    ]
    expected_else_edges = [
        (conditional_id, parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in trace.conditional_arm_entry_edges.items()
        if branch_kind == "else"
        for parent_label, child_label in edge_list
    ]

    assert trace.conditional_then_entry_edges == expected_then_edges
    assert trace.conditional_elif_entry_edges == expected_elif_edges
    assert trace.conditional_else_entry_edges == expected_else_edges

    for call_indexs in trace.conditional_edge_call_indices.values():
        assert call_indexs == sorted(call_indexs)
        assert len(call_indexs) == len(set(call_indexs))


def _render_dot_source(
    model: nn.Module,
    x: torch.Tensor,
    *,
    vis_mode: str = "unrolled",
) -> tuple[str, Trace]:
    """Render a model graph and return the DOT source plus model log.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Input tensor for the forward pass.
    vis_mode:
        Rendering mode, either ``"unrolled"`` or ``"rolled"``.

    Returns
    -------
    tuple[str, Trace]
        Rendered DOT source and the populated model log.
    """
    trace = _log_model(model, x, save_code_context=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = f"{tmpdir}/conditional_branch_matrix"
        dot_source = trace.draw(
            vis_mode=vis_mode,
            vis_outpath=outpath,
            vis_save_only=True,
            vis_fileformat="dot",
        )
    return dot_source, trace


def _find_edge_line(dot_source: str, parent_label: str, child_label: str) -> str:
    """Return the DOT line for one rendered edge.

    Parameters
    ----------
    dot_source:
        Graphviz DOT source.
    parent_label:
        Source layer label from the model log.
    child_label:
        Destination layer label from the model log.

    Returns
    -------
    str
        Matching DOT line.
    """
    parent_names = {parent_label.replace(":", "pass"), parent_label.split(":")[0]}
    child_names = {child_label.replace(":", "pass"), child_label.split(":")[0]}
    for line in dot_source.splitlines():
        if "->" not in line:
            continue
        if any(parent_name in line for parent_name in parent_names) and any(
            child_name in line for child_name in child_names
        ):
            return line
    raise AssertionError(f"Could not find edge line for {parent_label!r} -> {child_label!r}")


def _assert_conditional_edge_call_indices_exact(trace: Trace) -> None:
    """Assert ``conditional_edge_call_indices`` matches unrolled arm edges exactly.

    Parameters
    ----------
    trace:
        Logged model execution.
    """
    actual_unrolled_edges: set[tuple[str, str, int, str, int]] = set()
    for (conditional_id, branch_kind), edge_list in trace.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            actual_unrolled_edges.add(
                (
                    parent_label.split(":")[0],
                    child_label.split(":")[0],
                    conditional_id,
                    branch_kind,
                    int(child_label.split(":")[1]) if ":" in child_label else 1,
                )
            )

    for edge_key, call_indexs in trace.conditional_edge_call_indices.items():
        assert call_indexs == sorted(call_indexs)
        assert len(call_indexs) == len(set(call_indexs))
        parent_no_pass, child_no_pass, conditional_id, branch_kind = edge_key
        for call_index in call_indexs:
            assert (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
                call_index,
            ) in actual_unrolled_edges

    for actual_edge in actual_unrolled_edges:
        parent_no_pass, child_no_pass, conditional_id, branch_kind, call_index = actual_edge
        assert (
            call_index
            in trace.conditional_edge_call_indices[
                (parent_no_pass, child_no_pass, conditional_id, branch_kind)
            ]
        )


def _make_match_guard_model() -> nn.Module:
    """Create a small model whose tensor bool is consumed by a match guard.

    Returns
    -------
    nn.Module
        Dynamically defined match-guard model.
    """
    namespace: dict[str, object] = {"nn": nn, "torch": torch}
    exec(
        "\n".join(
            [
                "class MatchGuardModel(nn.Module):",
                '    """Model whose tensor bool is consumed by a match guard."""',
                "    def forward(self, x: torch.Tensor) -> torch.Tensor:",
                '        """Run the forward pass."""',
                "        match 1:",
                "            case 1 if x.mean() > 0:",
                "                return torch.relu(x)",
                "            case _:",
                "                return torch.sigmoid(x)",
            ]
        ),
        namespace,
    )
    return namespace["MatchGuardModel"]()


@pytest.mark.smoke
def test_simple_if_else_model_cross_verifies_invariants_rendering_and_lifecycle() -> None:
    """Simple ``if``/``else`` cross-verifies event, invariant, and rendering surfaces."""
    positive_log = _log_model(SimpleIfElseModel(), torch.ones(2, 2))
    negative_log = _log_model(SimpleIfElseModel(), -torch.ones(2, 2))
    positive_dot, positive_render_log = _render_dot_source(SimpleIfElseModel(), torch.ones(2, 2))
    negative_dot, negative_render_log = _render_dot_source(SimpleIfElseModel(), -torch.ones(2, 2))

    positive_event = _get_root_event(positive_log)
    negative_event = _get_root_event(negative_log)
    positive_bool = _get_terminal_bool_layers(positive_log)[0]
    negative_bool = _get_terminal_bool_layers(negative_log)[0]
    relu_layer = _find_only_layer(positive_log, "relu")
    sigmoid_layer = _find_only_layer(negative_log, "sigmoid")
    then_parent, then_child = positive_render_log.conditional_arm_entry_edges[
        (positive_event.id, "then")
    ][0]
    else_parent, else_child = negative_render_log.conditional_arm_entry_edges[
        (negative_event.id, "else")
    ][0]

    assert positive_event.kind == "if_chain"
    assert negative_event.kind == "if_chain"
    assert set(positive_event.branch_ranges) == {"then", "else"}
    assert set(negative_event.branch_ranges) == {"then", "else"}
    assert positive_bool.is_terminal_conditional_bool is True
    assert positive_bool.conditional_context_kind == "if_test"
    assert positive_bool.terminal_conditional_id == positive_event.id
    assert negative_bool.is_terminal_conditional_bool is True
    assert negative_bool.conditional_context_kind == "if_test"
    assert negative_bool.terminal_conditional_id == negative_event.id
    assert relu_layer.conditional_branch_stack == [(positive_event.id, "then")]
    assert sigmoid_layer.conditional_branch_stack == [(negative_event.id, "else")]
    assert "THEN" in _find_edge_line(positive_dot, then_parent, then_child)
    assert "ELSE" in _find_edge_line(negative_dot, else_parent, else_child)
    assert check_metadata_invariants(positive_log) is True
    assert check_metadata_invariants(negative_log) is True
    _assert_derived_views_consistent(positive_log)
    _assert_derived_views_consistent(negative_log)


@pytest.mark.smoke
def test_elif_ladder_model_cross_verifies_all_arms_and_render_labels() -> None:
    """Elif ladder cross-verifies every arm with invariants and rendering labels."""
    cases = [
        ("then", torch.tensor([[-1.0]]), "relu", "THEN"),
        ("elif_1", torch.tensor([[-0.25]]), "sigmoid", "ELIF 1"),
        ("elif_2", torch.tensor([[0.25]]), "tanh", "ELIF 2"),
        ("else", torch.tensor([[1.0]]), "square", "ELSE"),
    ]

    for branch_kind, input_tensor, func_name, label_text in cases:
        trace = _log_model(ElifLadderModel(), input_tensor)
        dot_source, render_log = _render_dot_source(ElifLadderModel(), input_tensor)
        event = _get_root_event(trace)
        branch_layer = _find_only_layer(trace, func_name)
        parent_label, child_label = render_log.conditional_arm_entry_edges[(event.id, branch_kind)][
            0
        ]

        assert event.kind == "if_chain"
        assert set(event.branch_ranges) == {"then", "elif_1", "elif_2", "else"}
        assert branch_layer.conditional_branch_stack == [(event.id, branch_kind)]
        assert {layer.terminal_conditional_id for layer in _get_terminal_bool_layers(trace)} == {
            event.id
        }
        assert label_text in _find_edge_line(dot_source, parent_label, child_label)
        assert check_metadata_invariants(trace) is True
        _assert_derived_views_consistent(trace)


def test_nested_if_then_if_model_materializes_nested_branch_stack() -> None:
    """Nested THEN->THEN execution records two events and a two-deep branch stack."""
    trace = _log_model(NestedIfThenIfModel(), torch.ones(2, 2))
    root_event = _get_root_event(trace)
    inner_event = _get_child_event(trace, root_event.id)
    relu_layer = _find_only_layer(trace, "relu")
    sigmoid_layer = _find_only_layer(trace, "sigmoid")

    assert len(trace.conditional_records) == 2
    assert inner_event.parent_branch_kind == "then"
    assert relu_layer.conditional_branch_stack == [(root_event.id, "then")]
    assert sigmoid_layer.conditional_branch_stack == [
        (root_event.id, "then"),
        (inner_event.id, "then"),
    ]
    assert check_metadata_invariants(trace) is True


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
    assert check_metadata_invariants(trace) is True


def test_multiline_predicate_model_tracks_multiline_if_event() -> None:
    """A multiline predicate still materialises one attributed ``if_chain`` event."""
    input_tensor = torch.tensor([[0.5, 1.5], [0.25, 2.0]])
    trace = _log_model(MultilinePredicateModel(), input_tensor)
    event = _get_root_event(trace)
    bool_layers = _get_terminal_bool_layers(trace)
    relu_layer = _find_only_layer(trace, "relu")

    assert len(trace.conditional_records) == 1
    assert event.kind == "if_chain"
    assert set(event.branch_ranges) == {"then", "else"}
    assert event.test_span[0] < event.test_span[2]
    assert len(bool_layers) == 2
    assert all(layer.conditional_context_kind == "if_test" for layer in bool_layers)
    assert all(layer.is_terminal_conditional_bool is True for layer in bool_layers)
    assert relu_layer.conditional_branch_stack == [(event.id, "then")]


def test_branch_uses_only_parameter_model_records_parameter_entry_edge() -> None:
    """A parameter-only branch body still records a THEN arm-entry edge."""
    trace = _log_model(BranchUsesOnlyParameterModel(), torch.ones(2, 2))
    event = _get_root_event(trace)
    bias_view_layer = _find_only_layer(trace, "__add__", [])
    branch_add_layer = _find_only_layer(trace, "__add__", [(event.id, "then")])

    assert (
        bias_view_layer.layer_label,
        branch_add_layer.layer_label,
    ) in trace.conditional_arm_entry_edges[(event.id, "then")]
    assert bias_view_layer.uses_params is True
    assert branch_add_layer.parents == [bias_view_layer.layer_label]
    assert bias_view_layer.conditional_branch_stack == []


def test_branch_uses_only_constant_model_records_constant_entry_edge() -> None:
    """A constant-only branch body still records a THEN arm-entry edge."""
    trace = _log_model(BranchUsesOnlyConstantModel(), torch.ones(2, 2))
    event = _get_root_event(trace)
    const_layer = _find_only_layer(trace, "ones_like")
    branch_add_layer = _find_only_layer(trace, "__add__", [(event.id, "then")])

    assert (
        const_layer.layer_label,
        branch_add_layer.layer_label,
    ) in trace.conditional_arm_entry_edges[(event.id, "then")]
    assert const_layer.conditional_branch_stack == []
    assert branch_add_layer.conditional_branch_stack == [(event.id, "then")]


def test_multi_arm_entry_nested_model_duplicates_entry_edge_per_conditional() -> None:
    """One nested branch-entry edge is recorded for both gained THEN arms."""
    trace = _log_model(MultiArmEntryNestedModel(), torch.ones(2, 2))
    outer_event = _get_root_event(trace)
    inner_event = _get_child_event(trace, outer_event.id)
    bias_view_layers = [
        layer
        for layer in trace.layer_list
        if layer.func_name == "__add__"
        and layer.conditional_branch_stack == []
        and layer.uses_params is True
    ]
    assert len(bias_view_layers) == 1
    bias_view_layer = bias_view_layers[0]
    inner_branch_add = _find_only_layer(
        trace,
        "__add__",
        [(outer_event.id, "then"), (inner_event.id, "then")],
    )

    edge = (bias_view_layer.layer_label, inner_branch_add.layer_label)
    assert edge in trace.conditional_arm_entry_edges[(outer_event.id, "then")]
    assert edge in trace.conditional_arm_entry_edges[(inner_event.id, "then")]
    parent_layer = trace[bias_view_layer.layer_label]
    assert parent_layer.conditional_arm_children[outer_event.id]["then"] == [
        inner_branch_add.layer_label
    ]
    assert parent_layer.conditional_arm_children[inner_event.id]["then"] == [
        inner_branch_add.layer_label
    ]
    assert parent_layer.conditional_then_children == [inner_branch_add.layer_label]


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


def test_looped_if_alternating_model_has_exactly_two_signatures() -> None:
    """Looped alternating condition aggregates to exactly two stack signatures."""
    trace = _log_model(LoopedIfAlternatingModel(), torch.ones(1, 4))
    conditional_id = _get_root_event(trace).id
    linear_layer = _find_only_layer_log(
        trace,
        "linear",
        lambda layer_log: layer_log.num_calls > 1,
    )

    assert linear_layer.conditional_role_stacks == [
        [(conditional_id, "then")],
        [(conditional_id, "else")],
    ]
    assert linear_layer.conditional_branch_stack_ops == {
        ((conditional_id, "then"),): [1, 3],
        ((conditional_id, "else"),): [2, 4],
    }
    assert check_metadata_invariants(trace) is True


def test_alternating_recurrent_if_model_merges_layerlog_conditionals() -> None:
    """Multi-pass ``LayerLog`` stores both branch signatures and pass unions."""
    trace = _log_model(AlternatingRecurrentIfModel(), torch.ones(1, 4))
    conditional_id = _get_root_event(trace).id
    linear_layer = _find_only_layer_log(
        trace,
        "linear",
        lambda layer_log: layer_log.num_calls > 1,
    )

    assert linear_layer.is_in_conditional_body is True
    assert linear_layer.conditional_role_stacks == [
        [(conditional_id, "then")],
        [(conditional_id, "else")],
    ]
    assert linear_layer.conditional_branch_stack_ops == {
        ((conditional_id, "then"),): [1, 3],
        ((conditional_id, "else"),): [2, 4],
    }
    assert check_metadata_invariants(trace) is True


def test_rolled_mixed_arm_model_records_exact_ops_and_composite_render_label() -> None:
    """Rolled mixed-arm edges keep exact pass lists and a composite Graphviz label."""
    trace = _log_model(RolledMixedArmModel(), torch.ones(1, 4))
    dot_source, render_log = _render_dot_source(
        RolledMixedArmModel(),
        torch.ones(1, 4),
        vis_mode="rolled",
    )
    conditional_id = _get_root_event(trace).id
    then_edges = trace.conditional_arm_entry_edges[(conditional_id, "then")]
    else_edges = trace.conditional_arm_entry_edges[(conditional_id, "else")]
    parent_label, child_label = then_edges[0]
    parent_no_pass = parent_label.split(":")[0]
    child_no_pass = child_label.split(":")[0]
    edge_line = _find_edge_line(dot_source, parent_label, child_label)

    assert len(then_edges) == 2
    assert len(else_edges) == 2
    assert trace.conditional_edge_call_indices[
        (parent_no_pass, child_no_pass, conditional_id, "then")
    ] == [
        1,
        3,
    ]
    assert trace.conditional_edge_call_indices[
        (parent_no_pass, child_no_pass, conditional_id, "else")
    ] == [
        2,
        4,
    ]
    assert "THEN(1,3) / ELSE(2,4)" in edge_line
    assert _find_edge_line(dot_source, parent_label, child_label) == _find_edge_line(
        dot_source,
        render_log.conditional_arm_entry_edges[(conditional_id, "then")][0][0],
        render_log.conditional_arm_entry_edges[(conditional_id, "then")][0][1],
    )
    _assert_conditional_edge_call_indices_exact(trace)
    assert check_metadata_invariants(trace) is True


def test_reconverging_branches_model_clears_branch_stack_after_merge() -> None:
    """Ops after reconvergence have empty stacks even though their parents were in-branch."""
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
    assert check_metadata_invariants(positive_log) is True
    assert check_metadata_invariants(negative_log) is True


def test_nested_helper_same_name_model_distinguishes_helpers_by_code_firstlineno() -> None:
    """Two same-named nested helpers materialise separate conditional ids."""
    positive_log = _log_model(NestedHelperSameNameModel(), torch.ones(2, 2))
    negative_log = _log_model(NestedHelperSameNameModel(), -torch.ones(2, 2))
    bool_layers = _get_terminal_bool_layers(positive_log)
    relu_layer = _find_only_layer(positive_log, "relu")
    square_layer = _find_only_layer(negative_log, "square")
    outer_two_event = next(
        event
        for event in negative_log.conditional_records
        if "outer_two" in event.function_qualname
    )

    helper_frames = [
        next(frame for frame in layer.code_context if frame.func_name == "helper")
        for layer in bool_layers
    ]

    assert len(positive_log.conditional_records) == 2
    assert len({layer.terminal_conditional_id for layer in bool_layers}) == 2
    assert len({frame.code_firstlineno for frame in helper_frames}) == 2
    assert relu_layer.conditional_branch_stack == [(bool_layers[0].terminal_conditional_id, "then")]
    assert square_layer.conditional_branch_stack == [(outer_two_event.id, "then")]


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="code_qualname is only available on Python 3.11+",
)
def test_nested_qualname_model_distinguishes_method_and_nested_helper() -> None:
    """Distinct qualnames keep a method helper separate from a nested helper."""
    positive_log = _log_model(NestedQualnameModel(), torch.ones(2, 2))
    negative_log = _log_model(NestedQualnameModel(), -torch.ones(2, 2))
    root_events = sorted(
        positive_log.conditional_records, key=lambda event: event.function_qualname
    )
    relu_layer = _find_only_layer(positive_log, "relu")
    square_layer = _find_only_layer(negative_log, "square")

    assert len(root_events) == 2
    assert root_events[0].function_qualname != root_events[1].function_qualname
    nested_event = next(
        event for event in root_events if "<locals>.helper" in event.function_qualname
    )
    method_event = next(
        event
        for event in negative_log.conditional_records
        if event.function_qualname.endswith(".helper") and "<locals>" not in event.function_qualname
    )
    assert relu_layer.conditional_branch_stack == [(nested_event.id, "then")]
    assert square_layer.conditional_branch_stack == [(method_event.id, "then")]


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="D14 fail-closed scope ambiguity is only expected before Python 3.11 qualnames",
)
def test_same_line_nested_def_model_fails_closed_when_scope_resolution_is_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous fallback scope resolution leaves branch stacks empty instead of guessing."""
    original_get_stack = introspection._get_code_context
    original_get_file_index = ast_branches.get_file_index

    def _stack_without_qualname(*args: object, **kwargs: object) -> list[object]:
        """Return a call stack with ``code_qualname`` cleared on every frame."""
        stack = original_get_stack(*args, **kwargs)
        for frame in stack:
            frame.code_qualname = None
        return stack

    def _ambiguous_file_index(filename: str) -> object:
        """Return a file index with a duplicate helper scope for this file."""
        index = original_get_file_index(filename)
        if index is None or filename != __file__:
            return index
        helper_scopes = [scope for scope in index.scopes if scope.func_name == "helper"]
        if helper_scopes and not any(
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


@pytest.mark.parametrize(
    ("model", "input_tensor", "expected_context"),
    [
        (AssertTensorCondModel(), torch.ones(2, 2), "assert"),
        (BoolCastOnlyModel(), torch.ones(2, 2), "bool_cast"),
        (ComprehensionIfModel(), torch.ones(2, 2), "comprehension_filter"),
        (WhileLoopModel(), torch.full((2, 2), 0.5), "while"),
    ],
)
def test_non_branch_bool_models_do_not_materialise_branch_metadata(
    model: nn.Module,
    input_tensor: torch.Tensor,
    expected_context: str,
) -> None:
    """Non-branch bool sites classify context without creating branch metadata."""
    trace = _log_model(model, input_tensor)
    bool_layers = _get_terminal_bool_layers(trace)

    assert len(bool_layers) >= 1
    assert all(layer.conditional_context_kind == expected_context for layer in bool_layers)
    assert all(layer.is_terminal_conditional_bool is False for layer in bool_layers)
    _assert_branchless_log(trace)


def test_ternary_ifexp_model_is_classified_as_ifexp_not_if_chain() -> None:
    """The placeholder ternary model remains a real ``ifexp`` rather than a false-positive ``if``."""
    trace = _log_model(TernaryIfExpModel(), torch.ones(1, 2))
    event = _get_root_event(trace)
    bool_layer = _get_terminal_bool_layers(trace)[0]

    assert event.kind == "ifexp"
    assert bool_layer.conditional_context_kind == "ifexp"
    assert bool_layer.is_terminal_conditional_bool is True


@pytest.mark.skipif(sys.version_info < (3, 10), reason="match guards require Python 3.10+")
def test_match_guard_model_classifies_bool_without_materialising_branch_metadata() -> None:
    """``match`` guards classify bools as ``match_guard`` and stop there."""
    trace = _log_model(_make_match_guard_model(), torch.ones(2, 2))
    bool_layer = _get_terminal_bool_layers(trace)[0]

    if bool_layer.conditional_context_kind == "unknown":
        pytest.skip("Runtime-compiled match guards are not source-indexed in this environment")
    assert bool_layer.conditional_context_kind == "match_guard"
    assert bool_layer.is_terminal_conditional_bool is False
    _assert_branchless_log(trace)


@pytest.mark.parametrize(
    ("model", "expected_func"),
    [
        (NotIfModel(), "relu"),
        (AndOrIfModel(), "relu"),
        (WalrusIfModel(), "relu"),
    ],
)
def test_compound_negated_and_walrus_ifs_still_attribute_the_taken_branch(
    model: nn.Module,
    expected_func: str,
) -> None:
    """Negation, compound predicates, and walrus binding still attribute normally."""
    input_tensor = torch.tensor([[0.5, 1.5], [0.25, 2.0]])
    trace = _log_model(model, input_tensor)
    event = _get_root_event(trace)
    branch_layer = _find_only_layer(trace, expected_func)

    assert event.kind == "if_chain"
    assert branch_layer.conditional_branch_stack == [(event.id, "then")]
    if isinstance(model, AndOrIfModel):
        bool_layers = _get_terminal_bool_layers(trace)
        assert len(bool_layers) >= 2
        assert {layer.terminal_conditional_id for layer in bool_layers} == {event.id}


@pytest.mark.parametrize(
    ("model", "input_tensor"),
    [
        (PythonBoolModel(), torch.ones(2, 2)),
        (ItemScalarizationModel(), torch.ones(2, 2)),
        (ShapePredicateModel(), torch.ones(2, 2)),
        (TorchWhereModel(), torch.full((2, 2), 0.75)),
    ],
)
def test_documented_false_negatives_do_not_materialise_branch_events(
    model: nn.Module,
    input_tensor: torch.Tensor,
) -> None:
    """Documented false negatives remain unattributed."""
    trace = _log_model(model, input_tensor)
    _assert_branchless_log(trace)
    branch_layers = [
        layer
        for layer in trace.layer_list
        if layer.func_name in {"relu", "sigmoid", "where"} and not layer.is_output
    ]
    assert all(layer.conditional_branch_stack == [] for layer in branch_layers)


def test_save_code_context_off_model_keeps_branch_classification_without_loading_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled source loading keeps branch attribution and the canonical no-source state."""
    trace = _log_model(
        SaveSourceContextOffModel(),
        torch.ones(2, 2),
        save_code_context=False,
    )
    event = _get_root_event(trace)
    relu_layer = _find_only_layer(trace, "relu")
    accessed_files: list[str] = []
    original_getlines = linecache.getlines

    def _tracking_getlines(*args: object, **kwargs: object) -> list[str]:
        """Record any unexpected linecache access."""
        accessed_files.append(str(args[0]))
        return original_getlines(*args, **kwargs)

    captured_entries = [
        entry for entry in trace.layer_list if not entry.is_input and entry.func_name != "none"
    ]
    assert captured_entries
    assert all(len(entry.code_context) > 0 for entry in captured_entries)
    assert relu_layer.conditional_branch_stack == [(event.id, "then")]

    with monkeypatch.context() as local_patch:
        local_patch.setattr(linecache, "getlines", _tracking_getlines)
        for entry in captured_entries:
            for frame in entry.code_context:
                assert frame.file is not None
                assert frame.line_number is not None
                assert frame.code_firstlineno is not None
                assert frame.source_loading_enabled is False
                assert frame._source_loaded is True
                assert frame._frame_func_obj is None
                assert frame.code_context is None
                assert frame.source_context == "None"
                assert frame.code_context_labeled == ""
                assert frame.call_line == ""
                assert frame.num_context_lines == 0
                assert frame.func_signature is None
                assert frame.func_docstring is None
                assert len(frame) == 0
                with pytest.raises(IndexError):
                    _ = frame[0]
                assert repr(frame).endswith("code: source unavailable")
        assert accessed_files == []

    roundtrip_log = pickle.loads(pickle.dumps(trace))
    roundtrip_stack = next(
        entry.code_context for entry in roundtrip_log.layer_list if not entry.is_input
    )
    assert len(roundtrip_stack) > 0
    assert roundtrip_stack[0].source_loading_enabled is False
    assert roundtrip_stack[0].source_context == "None"


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


def test_keep_unsaved_layers_false_model_scrubs_removed_labels_from_conditional_surfaces() -> None:
    """Pruning unsaved layers leaves no stale labels in conditional metadata."""
    input_tensor = torch.ones(1, 2)
    full_log = _log_model(
        KeepUnsavedLayersFalseModel(),
        input_tensor,
        keep_unsaved_layers=True,
        layers_to_save="none",
    )
    pruned_log = _log_model(
        KeepUnsavedLayersFalseModel(),
        input_tensor,
        keep_unsaved_layers=False,
        layers_to_save="none",
    )

    removed_labels = set(full_log.layer_labels) - set(pruned_log.layer_labels)
    removed_no_pass = {label.split(":", 1)[0] for label in removed_labels}

    assert removed_labels
    assert _collect_model_conditional_labels(pruned_log).isdisjoint(removed_labels)
    assert {
        label
        for layer in pruned_log.layer_list
        for label in _collect_branch_child_labels(layer.conditional_arm_children)
    }.isdisjoint(removed_labels)
    assert {
        label
        for layer_log in pruned_log.layer_logs.values()
        for label in _collect_branch_child_labels(layer_log.conditional_arm_children)
    }.isdisjoint(removed_no_pass)
    assert all(
        parent_label not in removed_no_pass and child_label not in removed_no_pass
        for parent_label, child_label, _, _ in pruned_log.conditional_edge_call_indices
    )
    assert all(
        removed_label not in event.bool_layers
        for removed_label in removed_labels
        for event in pruned_log.conditional_records
    )
    assert all(
        removed_label not in layer_log.conditional_entry_children
        for removed_label in removed_no_pass
        for layer_log in pruned_log.layer_logs.values()
    )
    assert all(
        removed_label not in layer_log.conditional_then_children
        for removed_label in removed_no_pass
        for layer_log in pruned_log.layer_logs.values()
    )
    assert all(
        removed_label not in layer_log.conditional_else_children
        for removed_label in removed_no_pass
        for layer_log in pruned_log.layer_logs.values()
    )
    assert all(
        removed_label not in child_labels
        for removed_label in removed_no_pass
        for layer_log in pruned_log.layer_logs.values()
        for child_labels in layer_log.conditional_elif_children.values()
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
    assert branch_row["conditional_elif_children"] == {}
    assert branch_row["conditional_else_children"] == []


def test_branch_entry_with_arg_label_keeps_semantic_and_argument_labels_separate() -> None:
    """Branch-entry edges retain the branch label and move arg labels to the head or x label."""
    dot_source, trace = _render_dot_source(BranchEntryWithArgLabelModel(), torch.ones(2, 2))
    conditional_id = _get_root_event(trace).id
    parent_label, child_label = trace.conditional_arm_entry_edges[(conditional_id, "then")][0]
    edge_line = _find_edge_line(dot_source, parent_label, child_label)

    assert 'label=<<FONT POINT-SIZE="18"><b><u>THEN</u></b></FONT>>' in edge_line
    assert (
        "headlabel=<<FONT POINT-SIZE='10'><b>arg" in edge_line
        or "xlabel=<<FONT POINT-SIZE='10'><b>arg" in edge_line
    )


def test_basic_ternary_model_attributes_then_arm_as_ifexp() -> None:
    """A minimal ternary materialises an ``ifexp`` event and branch stack."""
    trace = _log_model(BasicTernaryModel(), torch.ones(1, 2))
    event = _get_root_event(trace)
    bool_layer = _get_terminal_bool_layers(trace)[0]
    linear_layer = _find_only_layer(trace, "linear", [(event.id, "then")])

    assert event.kind == "ifexp"
    assert set(event.branch_ranges) == {"then", "else"}
    assert bool_layer.conditional_context_kind == "ifexp"
    assert bool_layer.is_terminal_conditional_bool is True
    assert linear_layer.conditional_branch_stack == [(event.id, "then")]


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


def test_ternary_multi_op_one_line_model_attributes_each_arm_by_column_offset() -> None:
    """Same-line multi-op ternary arms are separated by column offsets."""
    positive_log = _log_model(TernaryMultiOpOneLineModel(), torch.ones(2, 2))
    negative_log = _log_model(TernaryMultiOpOneLineModel(), -torch.ones(2, 2))
    positive_event = _get_root_event(positive_log)
    negative_event = _get_root_event(negative_log)
    positive_mul = _find_only_layer(positive_log, "mul")
    positive_add = _find_only_layer(positive_log, "add")
    negative_div = _find_only_layer(negative_log, "div")
    negative_sub = _find_only_layer(negative_log, "sub")

    if all(
        frame.col_offset is None for frame in positive_mul.code_context if frame.file == __file__
    ):
        pytest.skip(
            "Column offsets are unavailable for this one-line ternary in the current runtime"
        )
    assert positive_mul.conditional_branch_stack == [(positive_event.id, "then")]
    assert positive_add.conditional_branch_stack == [(positive_event.id, "then")]
    assert negative_div.conditional_branch_stack == [(negative_event.id, "else")]
    assert negative_sub.conditional_branch_stack == [(negative_event.id, "else")]
