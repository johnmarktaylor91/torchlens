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
from torchlens import check_metadata_invariants, log_forward_pass
from torchlens.data_classes.layer_log import LayerLog
from torchlens.data_classes.layer_pass_log import LayerPassLog
from torchlens.data_classes.model_log import ConditionalEvent, ModelLog


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
    """Model used to strengthen the ``save_source_context=False`` path."""

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
    """Model that combines ``assert`` with ``save_source_context=False``."""

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
    save_source_context: bool = True,
    keep_unsaved_layers: bool = True,
    layers_to_save: str | Sequence[str] | None = "all",
) -> ModelLog:
    """Capture a ``ModelLog`` for a small integration-test model.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Input tensor.
    save_source_context:
        Whether rich source loading is enabled during capture.
    keep_unsaved_layers:
        Whether unsaved layers remain in the final log.
    layers_to_save:
        Activation-saving selection passed through to ``log_forward_pass``.

    Returns
    -------
    ModelLog
        Fully postprocessed model log.
    """
    return log_forward_pass(
        model,
        x,
        save_source_context=save_source_context,
        keep_unsaved_layers=keep_unsaved_layers,
        layers_to_save=layers_to_save,
    )


def _get_terminal_bool_layers(model_log: ModelLog) -> list[LayerPassLog]:
    """Return terminal scalar bool layers from a model log.

    Parameters
    ----------
    model_log:
        Logged model execution.

    Returns
    -------
    list[LayerPassLog]
        Terminal scalar bool layers in execution order.
    """
    return [
        layer
        for layer in model_log.layer_list
        if layer.is_terminal_bool_layer and layer.is_scalar_bool
    ]


def _find_only_layer(
    model_log: ModelLog,
    func_name: str,
    branch_stack: list[tuple[int, str]] | None = None,
) -> LayerPassLog:
    """Find the unique layer matching a function name and optional branch stack.

    Parameters
    ----------
    model_log:
        Logged model execution.
    func_name:
        Function name to match.
    branch_stack:
        Optional exact branch stack to require.

    Returns
    -------
    LayerPassLog
        Matching layer.
    """
    matching_layers = [layer for layer in model_log.layer_list if layer.func_name == func_name]
    if branch_stack is not None:
        matching_layers = [
            layer for layer in matching_layers if layer.conditional_branch_stack == branch_stack
        ]
    assert len(matching_layers) == 1, (
        f"Expected one {func_name!r} layer for stack {branch_stack}, found {len(matching_layers)}"
    )
    return matching_layers[0]


def _find_only_layer_log(
    model_log: ModelLog,
    func_name: str,
    predicate: Callable[[LayerLog], bool],
) -> LayerLog:
    """Find one aggregate ``LayerLog`` matching the provided predicate.

    Parameters
    ----------
    model_log:
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
        for layer_log in model_log.layer_logs.values()
        if layer_log.func_name == func_name and predicate(layer_log)
    ]
    assert len(matching_layers) == 1
    return matching_layers[0]


def _get_root_event(model_log: ModelLog) -> ConditionalEvent:
    """Return the sole root conditional event in a model log.

    Parameters
    ----------
    model_log:
        Logged model execution.

    Returns
    -------
    ConditionalEvent
        Root event with no parent conditional.
    """
    root_events = [
        event for event in model_log.conditional_events if event.parent_conditional_id is None
    ]
    assert len(root_events) == 1
    return root_events[0]


def _get_child_event(model_log: ModelLog, parent_id: int) -> ConditionalEvent:
    """Return the sole child event whose parent id matches ``parent_id``.

    Parameters
    ----------
    model_log:
        Logged model execution.
    parent_id:
        Parent conditional id.

    Returns
    -------
    ConditionalEvent
        Nested conditional event.
    """
    child_events = [
        event for event in model_log.conditional_events if event.parent_conditional_id == parent_id
    ]
    assert len(child_events) == 1
    return child_events[0]


def _collect_branch_child_labels(
    branch_children_by_cond: dict[int, dict[str, list[str]]],
) -> set[str]:
    """Collect child labels from a nested ``cond_branch_children_by_cond`` mapping.

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


def _collect_model_conditional_labels(model_log: ModelLog) -> set[str]:
    """Collect every layer label referenced by model-level conditional metadata.

    Parameters
    ----------
    model_log:
        Logged model execution.

    Returns
    -------
    set[str]
        Flat set of all referenced layer labels.
    """
    referenced_labels: set[str] = set()
    for parent_label, child_label in model_log.conditional_branch_edges:
        referenced_labels.add(parent_label)
        referenced_labels.add(child_label)
    for parent_label, child_label in model_log.conditional_then_edges:
        referenced_labels.add(parent_label)
        referenced_labels.add(child_label)
    for _, _, parent_label, child_label in model_log.conditional_elif_edges:
        referenced_labels.add(parent_label)
        referenced_labels.add(child_label)
    for _, parent_label, child_label in model_log.conditional_else_edges:
        referenced_labels.add(parent_label)
        referenced_labels.add(child_label)
    for edge_list in model_log.conditional_arm_edges.values():
        for parent_label, child_label in edge_list:
            referenced_labels.add(parent_label)
            referenced_labels.add(child_label)
    for event in model_log.conditional_events:
        referenced_labels.update(event.bool_layers)
    return referenced_labels


def _assert_branchless_log(model_log: ModelLog) -> None:
    """Assert that a model log contains no conditional attribution metadata.

    Parameters
    ----------
    model_log:
        Logged model execution.
    """
    assert model_log.conditional_events == []
    assert model_log.conditional_branch_edges == []
    assert model_log.conditional_arm_edges == {}
    assert model_log.conditional_then_edges == []
    assert model_log.conditional_elif_edges == []
    assert model_log.conditional_else_edges == []


def _assert_derived_views_consistent(model_log: ModelLog) -> None:
    """Assert derived conditional views match the primary structures.

    Parameters
    ----------
    model_log:
        Logged model execution.
    """
    expected_then_edges = [
        (parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in model_log.conditional_arm_edges.items()
        if branch_kind == "then"
        for parent_label, child_label in edge_list
    ]
    expected_elif_edges = [
        (conditional_id, int(branch_kind.split("_")[1]), parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in model_log.conditional_arm_edges.items()
        if branch_kind.startswith("elif_")
        for parent_label, child_label in edge_list
    ]
    expected_else_edges = [
        (conditional_id, parent_label, child_label)
        for (conditional_id, branch_kind), edge_list in model_log.conditional_arm_edges.items()
        if branch_kind == "else"
        for parent_label, child_label in edge_list
    ]

    assert model_log.conditional_then_edges == expected_then_edges
    assert model_log.conditional_elif_edges == expected_elif_edges
    assert model_log.conditional_else_edges == expected_else_edges

    for pass_nums in model_log.conditional_edge_passes.values():
        assert pass_nums == sorted(pass_nums)
        assert len(pass_nums) == len(set(pass_nums))


def _render_dot_source(
    model: nn.Module,
    x: torch.Tensor,
    *,
    vis_mode: str = "unrolled",
) -> tuple[str, ModelLog]:
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
    tuple[str, ModelLog]
        Rendered DOT source and the populated model log.
    """
    model_log = _log_model(model, x, save_source_context=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = f"{tmpdir}/conditional_branch_matrix"
        dot_source = model_log.render_graph(
            vis_mode=vis_mode,
            vis_outpath=outpath,
            vis_save_only=True,
            vis_fileformat="dot",
        )
    return dot_source, model_log


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


def _assert_conditional_edge_passes_exact(model_log: ModelLog) -> None:
    """Assert ``conditional_edge_passes`` matches unrolled arm edges exactly.

    Parameters
    ----------
    model_log:
        Logged model execution.
    """
    actual_unrolled_edges: set[tuple[str, str, int, str, int]] = set()
    for (conditional_id, branch_kind), edge_list in model_log.conditional_arm_edges.items():
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

    for edge_key, pass_nums in model_log.conditional_edge_passes.items():
        assert pass_nums == sorted(pass_nums)
        assert len(pass_nums) == len(set(pass_nums))
        parent_no_pass, child_no_pass, conditional_id, branch_kind = edge_key
        for pass_num in pass_nums:
            assert (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
                pass_num,
            ) in actual_unrolled_edges

    for actual_edge in actual_unrolled_edges:
        parent_no_pass, child_no_pass, conditional_id, branch_kind, pass_num = actual_edge
        assert (
            pass_num
            in model_log.conditional_edge_passes[
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
    then_parent, then_child = positive_render_log.conditional_arm_edges[
        (positive_event.id, "then")
    ][0]
    else_parent, else_child = negative_render_log.conditional_arm_edges[
        (negative_event.id, "else")
    ][0]

    assert positive_event.kind == "if_chain"
    assert negative_event.kind == "if_chain"
    assert set(positive_event.branch_ranges) == {"then", "else"}
    assert set(negative_event.branch_ranges) == {"then", "else"}
    assert positive_bool.bool_is_branch is True
    assert positive_bool.bool_context_kind == "if_test"
    assert positive_bool.bool_conditional_id == positive_event.id
    assert negative_bool.bool_is_branch is True
    assert negative_bool.bool_context_kind == "if_test"
    assert negative_bool.bool_conditional_id == negative_event.id
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
        model_log = _log_model(ElifLadderModel(), input_tensor)
        dot_source, render_log = _render_dot_source(ElifLadderModel(), input_tensor)
        event = _get_root_event(model_log)
        branch_layer = _find_only_layer(model_log, func_name)
        parent_label, child_label = render_log.conditional_arm_edges[(event.id, branch_kind)][0]

        assert event.kind == "if_chain"
        assert set(event.branch_ranges) == {"then", "elif_1", "elif_2", "else"}
        assert branch_layer.conditional_branch_stack == [(event.id, branch_kind)]
        assert {layer.bool_conditional_id for layer in _get_terminal_bool_layers(model_log)} == {
            event.id
        }
        assert label_text in _find_edge_line(dot_source, parent_label, child_label)
        assert check_metadata_invariants(model_log) is True
        _assert_derived_views_consistent(model_log)


def test_nested_if_then_if_model_materializes_nested_branch_stack() -> None:
    """Nested THEN->THEN execution records two events and a two-deep branch stack."""
    model_log = _log_model(NestedIfThenIfModel(), torch.ones(2, 2))
    root_event = _get_root_event(model_log)
    inner_event = _get_child_event(model_log, root_event.id)
    relu_layer = _find_only_layer(model_log, "relu")
    sigmoid_layer = _find_only_layer(model_log, "sigmoid")

    assert len(model_log.conditional_events) == 2
    assert inner_event.parent_branch_kind == "then"
    assert relu_layer.conditional_branch_stack == [(root_event.id, "then")]
    assert sigmoid_layer.conditional_branch_stack == [
        (root_event.id, "then"),
        (inner_event.id, "then"),
    ]
    assert check_metadata_invariants(model_log) is True


def test_nested_in_else_model_materializes_else_to_inner_then_stack() -> None:
    """Nested ELSE->THEN execution records the outer ELSE parent branch."""
    model_log = _log_model(NestedInElseModel(), -torch.ones(2, 2))
    root_event = _get_root_event(model_log)
    inner_event = _get_child_event(model_log, root_event.id)
    neg_layer = _find_only_layer(model_log, "neg")
    sigmoid_layer = _find_only_layer(model_log, "sigmoid")

    assert len(model_log.conditional_events) == 2
    assert inner_event.parent_branch_kind == "else"
    assert neg_layer.conditional_branch_stack == [(root_event.id, "else")]
    assert sigmoid_layer.conditional_branch_stack == [
        (root_event.id, "else"),
        (inner_event.id, "then"),
    ]
    assert check_metadata_invariants(model_log) is True


def test_multiline_predicate_model_tracks_multiline_if_event() -> None:
    """A multiline predicate still materialises one attributed ``if_chain`` event."""
    input_tensor = torch.tensor([[0.5, 1.5], [0.25, 2.0]])
    model_log = _log_model(MultilinePredicateModel(), input_tensor)
    event = _get_root_event(model_log)
    bool_layers = _get_terminal_bool_layers(model_log)
    relu_layer = _find_only_layer(model_log, "relu")

    assert len(model_log.conditional_events) == 1
    assert event.kind == "if_chain"
    assert set(event.branch_ranges) == {"then", "else"}
    assert event.test_span[0] < event.test_span[2]
    assert len(bool_layers) == 2
    assert all(layer.bool_context_kind == "if_test" for layer in bool_layers)
    assert all(layer.bool_is_branch is True for layer in bool_layers)
    assert relu_layer.conditional_branch_stack == [(event.id, "then")]


def test_branch_uses_only_parameter_model_records_parameter_entry_edge() -> None:
    """A parameter-only branch body still records a THEN arm-entry edge."""
    model_log = _log_model(BranchUsesOnlyParameterModel(), torch.ones(2, 2))
    event = _get_root_event(model_log)
    bias_view_layer = _find_only_layer(model_log, "__add__", [])
    branch_add_layer = _find_only_layer(model_log, "__add__", [(event.id, "then")])

    assert (
        bias_view_layer.layer_label,
        branch_add_layer.layer_label,
    ) in model_log.conditional_arm_edges[(event.id, "then")]
    assert bias_view_layer.uses_params is True
    assert branch_add_layer.parent_layers == [bias_view_layer.layer_label]
    assert bias_view_layer.conditional_branch_stack == []


def test_branch_uses_only_constant_model_records_constant_entry_edge() -> None:
    """A constant-only branch body still records a THEN arm-entry edge."""
    model_log = _log_model(BranchUsesOnlyConstantModel(), torch.ones(2, 2))
    event = _get_root_event(model_log)
    const_layer = _find_only_layer(model_log, "ones_like")
    branch_add_layer = _find_only_layer(model_log, "__add__", [(event.id, "then")])

    assert (
        const_layer.layer_label,
        branch_add_layer.layer_label,
    ) in model_log.conditional_arm_edges[(event.id, "then")]
    assert const_layer.conditional_branch_stack == []
    assert branch_add_layer.conditional_branch_stack == [(event.id, "then")]


def test_multi_arm_entry_nested_model_duplicates_entry_edge_per_conditional() -> None:
    """One nested branch-entry edge is recorded for both gained THEN arms."""
    model_log = _log_model(MultiArmEntryNestedModel(), torch.ones(2, 2))
    outer_event = _get_root_event(model_log)
    inner_event = _get_child_event(model_log, outer_event.id)
    bias_view_layers = [
        layer
        for layer in model_log.layer_list
        if layer.func_name == "__add__"
        and layer.conditional_branch_stack == []
        and layer.uses_params is True
    ]
    assert len(bias_view_layers) == 1
    bias_view_layer = bias_view_layers[0]
    inner_branch_add = _find_only_layer(
        model_log,
        "__add__",
        [(outer_event.id, "then"), (inner_event.id, "then")],
    )

    edge = (bias_view_layer.layer_label, inner_branch_add.layer_label)
    assert edge in model_log.conditional_arm_edges[(outer_event.id, "then")]
    assert edge in model_log.conditional_arm_edges[(inner_event.id, "then")]
    parent_layer = model_log[bias_view_layer.layer_label]
    assert parent_layer.cond_branch_children_by_cond[outer_event.id]["then"] == [
        inner_branch_add.layer_label
    ]
    assert parent_layer.cond_branch_children_by_cond[inner_event.id]["then"] == [
        inner_branch_add.layer_label
    ]
    assert parent_layer.cond_branch_then_children == [inner_branch_add.layer_label]


def test_if_bool_cast_model_marks_wrapper_kind_without_losing_branch_attribution() -> None:
    """``bool(...)`` wrappers inside ``if`` tests remain branch-participating."""
    model_log = _log_model(IfBoolCastModel(), torch.ones(2, 2))
    event = _get_root_event(model_log)
    bool_layer = _get_terminal_bool_layers(model_log)[0]
    relu_layer = _find_only_layer(model_log, "relu")

    assert bool_layer.bool_context_kind == "if_test"
    assert bool_layer.bool_is_branch is True
    assert bool_layer.bool_wrapper_kind == "bool_cast"
    assert bool_layer.bool_conditional_id == event.id
    assert relu_layer.conditional_branch_stack == [(event.id, "then")]


def test_looped_if_alternating_model_has_exactly_two_signatures() -> None:
    """Looped alternating condition aggregates to exactly two stack signatures."""
    model_log = _log_model(LoopedIfAlternatingModel(), torch.ones(1, 4))
    conditional_id = _get_root_event(model_log).id
    linear_layer = _find_only_layer_log(
        model_log,
        "linear",
        lambda layer_log: layer_log.num_passes > 1,
    )

    assert linear_layer.conditional_branch_stacks == [
        [(conditional_id, "then")],
        [(conditional_id, "else")],
    ]
    assert linear_layer.conditional_branch_stack_passes == {
        ((conditional_id, "then"),): [1, 3],
        ((conditional_id, "else"),): [2, 4],
    }
    assert check_metadata_invariants(model_log) is True


def test_alternating_recurrent_if_model_merges_layerlog_conditionals() -> None:
    """Multi-pass ``LayerLog`` stores both branch signatures and pass unions."""
    model_log = _log_model(AlternatingRecurrentIfModel(), torch.ones(1, 4))
    conditional_id = _get_root_event(model_log).id
    linear_layer = _find_only_layer_log(
        model_log,
        "linear",
        lambda layer_log: layer_log.num_passes > 1,
    )

    assert linear_layer.in_cond_branch is True
    assert linear_layer.conditional_branch_stacks == [
        [(conditional_id, "then")],
        [(conditional_id, "else")],
    ]
    assert linear_layer.conditional_branch_stack_passes == {
        ((conditional_id, "then"),): [1, 3],
        ((conditional_id, "else"),): [2, 4],
    }
    assert check_metadata_invariants(model_log) is True


def test_rolled_mixed_arm_model_records_exact_passes_and_composite_render_label() -> None:
    """Rolled mixed-arm edges keep exact pass lists and a composite Graphviz label."""
    model_log = _log_model(RolledMixedArmModel(), torch.ones(1, 4))
    dot_source, render_log = _render_dot_source(
        RolledMixedArmModel(),
        torch.ones(1, 4),
        vis_mode="rolled",
    )
    conditional_id = _get_root_event(model_log).id
    then_edges = model_log.conditional_arm_edges[(conditional_id, "then")]
    else_edges = model_log.conditional_arm_edges[(conditional_id, "else")]
    parent_label, child_label = then_edges[0]
    parent_no_pass = parent_label.split(":")[0]
    child_no_pass = child_label.split(":")[0]
    edge_line = _find_edge_line(dot_source, parent_label, child_label)

    assert len(then_edges) == 2
    assert len(else_edges) == 2
    assert model_log.conditional_edge_passes[
        (parent_no_pass, child_no_pass, conditional_id, "then")
    ] == [
        1,
        3,
    ]
    assert model_log.conditional_edge_passes[
        (parent_no_pass, child_no_pass, conditional_id, "else")
    ] == [
        2,
        4,
    ]
    assert "THEN(1,3) / ELSE(2,4)" in edge_line
    assert _find_edge_line(dot_source, parent_label, child_label) == _find_edge_line(
        dot_source,
        render_log.conditional_arm_edges[(conditional_id, "then")][0][0],
        render_log.conditional_arm_edges[(conditional_id, "then")][0][1],
    )
    _assert_conditional_edge_passes_exact(model_log)
    assert check_metadata_invariants(model_log) is True


def test_reconverging_branches_model_clears_branch_stack_after_merge() -> None:
    """Ops after reconvergence have empty stacks even though their parents were in-branch."""
    positive_log = _log_model(ReconvergingBranchesModel(), torch.ones(2, 2))
    negative_log = _log_model(ReconvergingBranchesModel(), -torch.ones(2, 2))
    positive_add = _find_only_layer(positive_log, "add")
    negative_add = _find_only_layer(negative_log, "add")
    positive_parent = positive_log[positive_add.parent_layers[0]]
    negative_parent = negative_log[negative_add.parent_layers[0]]

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
        event for event in negative_log.conditional_events if "outer_two" in event.function_qualname
    )

    helper_frames = [
        next(frame for frame in layer.func_call_stack if frame.func_name == "helper")
        for layer in bool_layers
    ]

    assert len(positive_log.conditional_events) == 2
    assert len({layer.bool_conditional_id for layer in bool_layers}) == 2
    assert len({frame.code_firstlineno for frame in helper_frames}) == 2
    assert relu_layer.conditional_branch_stack == [(bool_layers[0].bool_conditional_id, "then")]
    assert square_layer.conditional_branch_stack == [(outer_two_event.id, "then")]


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="code_qualname is only available on Python 3.11+",
)
def test_nested_qualname_model_distinguishes_method_and_nested_helper() -> None:
    """Distinct qualnames keep a method helper separate from a nested helper."""
    positive_log = _log_model(NestedQualnameModel(), torch.ones(2, 2))
    negative_log = _log_model(NestedQualnameModel(), -torch.ones(2, 2))
    root_events = sorted(positive_log.conditional_events, key=lambda event: event.function_qualname)
    relu_layer = _find_only_layer(positive_log, "relu")
    square_layer = _find_only_layer(negative_log, "square")

    assert len(root_events) == 2
    assert root_events[0].function_qualname != root_events[1].function_qualname
    nested_event = next(
        event for event in root_events if "<locals>.helper" in event.function_qualname
    )
    method_event = next(
        event
        for event in negative_log.conditional_events
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
    original_get_stack = introspection._get_func_call_stack
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

    monkeypatch.setattr(introspection, "_get_func_call_stack", _stack_without_qualname)
    monkeypatch.setattr(output_tensors, "_get_func_call_stack", _stack_without_qualname)
    monkeypatch.setattr(source_tensors, "_get_func_call_stack", _stack_without_qualname)
    monkeypatch.setattr(graph_traversal, "_get_func_call_stack", _stack_without_qualname)
    monkeypatch.setattr(ast_branches, "get_file_index", _ambiguous_file_index)

    model_log = _log_model(SameLineNestedDefModel(), torch.ones(2, 2))
    relu_layer = _find_only_layer(model_log, "relu")

    assert relu_layer.conditional_branch_stack == []
    assert model_log.conditional_arm_edges == {}


def test_decorated_forward_model_preserves_or_gracefully_skips_branch_attribution() -> None:
    """Decorated ``forward`` either attributes correctly or degrades without false labels."""
    model_log = _log_model(DecoratedForwardModel(), torch.ones(2, 2))
    relu_layer = _find_only_layer(model_log, "relu")

    if model_log.conditional_events:
        event = _get_root_event(model_log)
        assert relu_layer.conditional_branch_stack == [(event.id, "then")]
    else:
        assert relu_layer.conditional_branch_stack == []
        assert model_log.conditional_arm_edges == {}


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
    model_log = _log_model(model, input_tensor)
    bool_layers = _get_terminal_bool_layers(model_log)

    assert len(bool_layers) >= 1
    assert all(layer.bool_context_kind == expected_context for layer in bool_layers)
    assert all(layer.bool_is_branch is False for layer in bool_layers)
    _assert_branchless_log(model_log)


def test_ternary_ifexp_model_is_classified_as_ifexp_not_if_chain() -> None:
    """The placeholder ternary model remains a real ``ifexp`` rather than a false-positive ``if``."""
    model_log = _log_model(TernaryIfExpModel(), torch.ones(1, 2))
    event = _get_root_event(model_log)
    bool_layer = _get_terminal_bool_layers(model_log)[0]

    assert event.kind == "ifexp"
    assert bool_layer.bool_context_kind == "ifexp"
    assert bool_layer.bool_is_branch is True


@pytest.mark.skipif(sys.version_info < (3, 10), reason="match guards require Python 3.10+")
def test_match_guard_model_classifies_bool_without_materialising_branch_metadata() -> None:
    """``match`` guards classify bools as ``match_guard`` and stop there."""
    model_log = _log_model(_make_match_guard_model(), torch.ones(2, 2))
    bool_layer = _get_terminal_bool_layers(model_log)[0]

    if bool_layer.bool_context_kind == "unknown":
        pytest.skip("Runtime-compiled match guards are not source-indexed in this environment")
    assert bool_layer.bool_context_kind == "match_guard"
    assert bool_layer.bool_is_branch is False
    _assert_branchless_log(model_log)


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
    model_log = _log_model(model, input_tensor)
    event = _get_root_event(model_log)
    branch_layer = _find_only_layer(model_log, expected_func)

    assert event.kind == "if_chain"
    assert branch_layer.conditional_branch_stack == [(event.id, "then")]
    if isinstance(model, AndOrIfModel):
        bool_layers = _get_terminal_bool_layers(model_log)
        assert len(bool_layers) >= 2
        assert {layer.bool_conditional_id for layer in bool_layers} == {event.id}


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
    model_log = _log_model(model, input_tensor)
    _assert_branchless_log(model_log)
    branch_layers = [
        layer
        for layer in model_log.layer_list
        if layer.func_name in {"relu", "sigmoid", "where"} and not layer.is_output_layer
    ]
    assert all(layer.conditional_branch_stack == [] for layer in branch_layers)


def test_save_source_context_off_model_keeps_branch_classification_without_loading_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled source loading keeps branch attribution and the canonical no-source state."""
    model_log = _log_model(
        SaveSourceContextOffModel(),
        torch.ones(2, 2),
        save_source_context=False,
    )
    event = _get_root_event(model_log)
    relu_layer = _find_only_layer(model_log, "relu")
    accessed_files: list[str] = []
    original_getlines = linecache.getlines

    def _tracking_getlines(*args: object, **kwargs: object) -> list[str]:
        """Record any unexpected linecache access."""
        accessed_files.append(str(args[0]))
        return original_getlines(*args, **kwargs)

    captured_entries = [
        entry
        for entry in model_log.layer_list
        if not entry.is_input_layer and entry.func_name != "none"
    ]
    assert captured_entries
    assert all(len(entry.func_call_stack) > 0 for entry in captured_entries)
    assert relu_layer.conditional_branch_stack == [(event.id, "then")]

    with monkeypatch.context() as local_patch:
        local_patch.setattr(linecache, "getlines", _tracking_getlines)
        for entry in captured_entries:
            for frame in entry.func_call_stack:
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

    roundtrip_log = pickle.loads(pickle.dumps(model_log))
    roundtrip_stack = next(
        entry.func_call_stack for entry in roundtrip_log.layer_list if not entry.is_input_layer
    )
    assert len(roundtrip_stack) > 0
    assert roundtrip_stack[0].source_loading_enabled is False
    assert roundtrip_stack[0].source_context == "None"


def test_save_source_context_off_assert_model_has_no_false_positive_if_edges() -> None:
    """Disabled source loading plus ``assert`` must not fabricate branch metadata."""
    model_log = _log_model(
        SaveSourceContextOffAssertModel(),
        torch.ones(2, 2),
        save_source_context=False,
    )
    bool_layer = _get_terminal_bool_layers(model_log)[0]

    assert bool_layer.bool_context_kind == "assert"
    assert bool_layer.bool_is_branch is False
    assert len(bool_layer.func_call_stack) > 0
    assert all(frame.source_loading_enabled is False for frame in bool_layer.func_call_stack)
    _assert_branchless_log(model_log)


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
        for label in _collect_branch_child_labels(layer.cond_branch_children_by_cond)
    }.isdisjoint(removed_labels)
    assert {
        label
        for layer_log in pruned_log.layer_logs.values()
        for label in _collect_branch_child_labels(layer_log.cond_branch_children_by_cond)
    }.isdisjoint(removed_no_pass)
    assert all(
        parent_label not in removed_no_pass and child_label not in removed_no_pass
        for parent_label, child_label, _, _ in pruned_log.conditional_edge_passes
    )
    assert all(
        removed_label not in event.bool_layers
        for removed_label in removed_labels
        for event in pruned_log.conditional_events
    )
    assert all(
        removed_label not in layer_log.cond_branch_start_children
        for removed_label in removed_no_pass
        for layer_log in pruned_log.layer_logs.values()
    )
    assert all(
        removed_label not in layer_log.cond_branch_then_children
        for removed_label in removed_no_pass
        for layer_log in pruned_log.layer_logs.values()
    )
    assert all(
        removed_label not in layer_log.cond_branch_else_children
        for removed_label in removed_no_pass
        for layer_log in pruned_log.layer_logs.values()
    )
    assert all(
        removed_label not in child_labels
        for removed_label in removed_no_pass
        for layer_log in pruned_log.layer_logs.values()
        for child_labels in layer_log.cond_branch_elif_children.values()
    )


def test_to_pandas_conditional_model_populates_live_conditional_columns() -> None:
    """Real conditional execution populates the exported conditional DataFrame columns."""
    model_log = _log_model(ToPandasConditionalModel(), torch.ones(2, 2), layers_to_save="all")
    layer_df = model_log.to_pandas()
    bool_layer = _get_terminal_bool_layers(model_log)[0]
    branch_layer = _find_only_layer(model_log, "relu")

    bool_row = layer_df.loc[layer_df["layer_label"] == bool_layer.layer_label].iloc[0]
    branch_row = layer_df.loc[layer_df["layer_label"] == branch_layer.layer_label].iloc[0]

    assert isinstance(layer_df, pd.DataFrame)
    assert bool(bool_row["bool_is_branch"]) is True
    assert bool_row["bool_context_kind"] == "if_test"
    assert int(bool_row["bool_conditional_id"]) == 0
    assert int(branch_row["conditional_branch_depth"]) == 1
    assert branch_row["conditional_branch_stack"] == "cond_0:then"
    assert branch_row["cond_branch_then_children"] == []
    assert branch_row["cond_branch_elif_children"] == {}
    assert branch_row["cond_branch_else_children"] == []


def test_branch_entry_with_arg_label_keeps_semantic_and_argument_labels_separate() -> None:
    """Branch-entry edges retain the branch label and move arg labels to the head or x label."""
    dot_source, model_log = _render_dot_source(BranchEntryWithArgLabelModel(), torch.ones(2, 2))
    conditional_id = _get_root_event(model_log).id
    parent_label, child_label = model_log.conditional_arm_edges[(conditional_id, "then")][0]
    edge_line = _find_edge_line(dot_source, parent_label, child_label)

    assert 'label=<<FONT POINT-SIZE="18"><b><u>THEN</u></b></FONT>>' in edge_line
    assert (
        "headlabel=<<FONT POINT-SIZE='10'><b>arg" in edge_line
        or "xlabel=<<FONT POINT-SIZE='10'><b>arg" in edge_line
    )


def test_basic_ternary_model_attributes_then_arm_as_ifexp() -> None:
    """A minimal ternary materialises an ``ifexp`` event and branch stack."""
    model_log = _log_model(BasicTernaryModel(), torch.ones(1, 2))
    event = _get_root_event(model_log)
    bool_layer = _get_terminal_bool_layers(model_log)[0]
    linear_layer = _find_only_layer(model_log, "linear", [(event.id, "then")])

    assert event.kind == "ifexp"
    assert set(event.branch_ranges) == {"then", "else"}
    assert bool_layer.bool_context_kind == "ifexp"
    assert bool_layer.bool_is_branch is True
    assert linear_layer.conditional_branch_stack == [(event.id, "then")]


def test_nested_ternary_model_records_parent_child_ifexp_events() -> None:
    """Nested ternaries keep parent-child links between ``ifexp`` events."""
    model_log = _log_model(NestedTernaryModel(), torch.ones(1, 2))
    outer_event = _get_root_event(model_log)
    inner_event = _get_child_event(model_log, outer_event.id)
    linear_layer = _find_only_layer(
        model_log,
        "linear",
        [(outer_event.id, "then"), (inner_event.id, "then")],
    )

    assert len(model_log.conditional_events) == 2
    assert outer_event.kind == "ifexp"
    assert inner_event.kind == "ifexp"
    assert inner_event.parent_branch_kind == "then"
    assert linear_layer.conditional_branch_stack == [
        (outer_event.id, "then"),
        (inner_event.id, "then"),
    ]


def test_ternary_inside_if_model_records_mixed_if_and_ifexp_stack() -> None:
    """An inner ternary keeps both the outer ``if_chain`` and inner ``ifexp`` stack."""
    model_log = _log_model(TernaryInsideIfModel(), torch.ones(1, 2))
    outer_event = next(event for event in model_log.conditional_events if event.kind == "if_chain")
    inner_event = next(event for event in model_log.conditional_events if event.kind == "ifexp")
    linear_layer = _find_only_layer(
        model_log,
        "linear",
        [(outer_event.id, "then"), (inner_event.id, "then")],
    )

    assert inner_event.parent_conditional_id == outer_event.id
    assert inner_event.parent_branch_kind == "then"
    assert linear_layer.conditional_branch_stack == [
        (outer_event.id, "then"),
        (inner_event.id, "then"),
    ]


def test_ternary_with_bool_cast_model_marks_bool_wrapper_kind() -> None:
    """``bool(...)`` wrappers inside ternaries keep ``ifexp`` attribution."""
    model_log = _log_model(TernaryWithBoolCastModel(), torch.ones(1, 2))
    event = _get_root_event(model_log)
    bool_layer = _get_terminal_bool_layers(model_log)[0]
    linear_layer = _find_only_layer(model_log, "linear", [(event.id, "then")])

    assert event.kind == "ifexp"
    assert bool_layer.bool_context_kind == "ifexp"
    assert bool_layer.bool_wrapper_kind == "bool_cast"
    assert bool_layer.bool_is_branch is True
    assert linear_layer.conditional_branch_stack == [(event.id, "then")]


@pytest.mark.skipif(
    sys.version_info >= (3, 11),
    reason="degraded ternary mode is only meaningful before Python 3.11",
)
def test_ternary_py310_fail_closed_model_drops_same_line_arm_attribution() -> None:
    """Pre-3.11 same-line ternaries fail closed when no column offsets are available."""
    model_log = _log_model(TernaryPy310FailClosedModel(), torch.ones(1, 2))
    linear_layers = [layer for layer in model_log.layer_list if layer.func_name == "linear"]

    assert len(model_log.conditional_events) == 1
    assert model_log.conditional_arm_edges == {}
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
        frame.col_offset is None for frame in positive_mul.func_call_stack if frame.file == __file__
    ):
        pytest.skip(
            "Column offsets are unavailable for this one-line ternary in the current runtime"
        )
    assert positive_mul.conditional_branch_stack == [(positive_event.id, "then")]
    assert positive_add.conditional_branch_stack == [(positive_event.id, "then")]
    assert negative_div.conditional_branch_stack == [(negative_event.id, "else")]
    assert negative_sub.conditional_branch_stack == [(negative_event.id, "else")]
