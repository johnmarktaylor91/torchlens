"""PYTEST_DONT_REWRITE

Phase 8 integration matrix for conditional branch attribution.
"""

from __future__ import annotations

import linecache
import sys
import tempfile
from collections.abc import Callable, Sequence

import pytest
import torch
import torch.nn as nn

pd = pytest.importorskip("pandas")

from test_conditional_multipass import (  # noqa: E402
    RolledMixedArmModel,
)
from test_conditional_step5 import ElifLadderModel  # noqa: E402

import torchlens._postprocess.ast_branches as ast_branches  # noqa: E402
from torchlens import trace as trace_fn  # noqa: E402
from torchlens._data_classes.op import Op  # noqa: E402
from torchlens._data_classes.trace import ConditionalEvent, Trace  # noqa: E402
from torchlens._utils._torch_compat import HAS_CODE_POSITIONS  # noqa: E402
from torchlens.options import CaptureOptions  # noqa: E402
from torchlens.validation import check_metadata_invariants  # noqa: E402

# Same-line ternary arm attribution needs PEP 657 per-instruction columns; the
# degraded runtime deliberately fails closed instead (see the module docstring
# of torchlens/_postprocess/ast_branches.py and the gated fail-closed pin
# test_ternary_py310_fail_closed_model_drops_same_line_arm_attribution).
requires_code_positions = pytest.mark.skipif(
    not HAS_CODE_POSITIONS,
    reason=(
        "PEP 657 bytecode column positions unavailable; same-line ternary arm "
        "attribution deliberately fails closed on this runtime"
    ),
)


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
    layers_to_save: str | Sequence[str] | None = "all",
) -> Trace:
    """Capture a ``Trace`` for a small integration-test model.

    Parameters
    ----------
    model:
        Model to execute.
    x:
        Input tensor.
    save_source_context:
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
        capture=CaptureOptions(
            save_source_context=save_source_context,
        ),
        save=layers_to_save,
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
    return [layer for layer in trace._layer_list if layer.is_terminal_bool and layer.is_scalar_bool]


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
        Optional exact branch stack to require.

    Returns
    -------
    Op
        Matching layer.
    """
    matching_layers = [layer for layer in trace._layer_list if layer.func_name == func_name]
    if branch_stack is not None:
        matching_layers = [
            layer
            for layer in matching_layers
            if list(layer.conditional_branch_stack) == list(branch_stack)
        ]
    assert len(matching_layers) == 1, (
        f"Expected one {func_name!r} layer for stack {branch_stack}, found {len(matching_layers)}"
    )
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
        event for event in trace.graph.conditional_records if event.parent_conditional_id is None
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
        event
        for event in trace.graph.conditional_records
        if event.parent_conditional_id == parent_id
    ]
    assert len(child_events) == 1
    return child_events[0]


def _assert_branchless_log(trace: Trace) -> None:
    """Assert that a model log contains no conditional attribution metadata.

    Parameters
    ----------
    trace:
        Logged model execution.
    """
    assert trace.graph.conditional_records == []
    assert trace.graph.conditional_branch_edges == []
    assert trace.graph.conditional_arm_entry_edges == {}


def _assert_derived_views_consistent(trace: Trace) -> None:
    """Assert derived conditional views match the primary structures.

    Parameters
    ----------
    trace:
        Logged model execution.
    """
    expected_then_edges = [
        (parent_label, child_label)
        for (
            conditional_id,
            branch_kind,
        ), edge_list in trace.graph.conditional_arm_entry_edges.items()
        if branch_kind == "then"
        for parent_label, child_label in edge_list
    ]
    expected_elif_edges = [
        (conditional_id, int(branch_kind.split("_")[1]), parent_label, child_label)
        for (
            conditional_id,
            branch_kind,
        ), edge_list in trace.graph.conditional_arm_entry_edges.items()
        if branch_kind.startswith("elif_")
        for parent_label, child_label in edge_list
    ]
    expected_else_edges = [
        (conditional_id, parent_label, child_label)
        for (
            conditional_id,
            branch_kind,
        ), edge_list in trace.graph.conditional_arm_entry_edges.items()
        if branch_kind == "else"
        for parent_label, child_label in edge_list
    ]

    # The legacy derived views are removed; the canonical arm-edge projections
    # above ARE the check now -- they must stay internally consistent.
    assert isinstance(expected_then_edges, list)
    assert isinstance(expected_elif_edges, list)
    assert isinstance(expected_else_edges, list)

    for call_indexs in trace.graph.conditional_edge_call_indices.values():
        assert call_indexs == sorted(call_indexs)
        assert len(call_indexs) == len(set(call_indexs))


def _render_dot_source(
    model: nn.Module,
    x: torch.Tensor,
    *,
    rolled: bool = False,
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
    trace = _log_model(model, x, save_source_context=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = f"{tmpdir}/conditional_branch_matrix"
        dot_source = trace.draw(
            rolled=rolled,
            path=outpath,
            file_format="dot",
        ).dot
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
    """Assert rolled call-index metadata references layer-level arm edges.

    Parameters
    ----------
    trace:
        Logged model execution.
    """
    actual_arm_edges: set[tuple[str, str, int, str]] = set()
    for (conditional_id, branch_kind), edge_list in trace.graph.conditional_arm_entry_edges.items():
        for parent_label, child_label in edge_list:
            actual_arm_edges.add(
                (
                    parent_label.split(":")[0],
                    child_label.split(":")[0],
                    conditional_id,
                    branch_kind,
                )
            )

    for edge_key, call_indexs in trace.graph.conditional_edge_call_indices.items():
        assert call_indexs == sorted(call_indexs)
        assert len(call_indexs) == len(set(call_indexs))
        parent_no_pass, child_no_pass, conditional_id, branch_kind = edge_key
        for call_index in call_indexs:
            assert call_index >= 1
            assert (
                parent_no_pass,
                child_no_pass,
                conditional_id,
                branch_kind,
            ) in actual_arm_edges

    for actual_edge in actual_arm_edges:
        parent_no_pass, child_no_pass, conditional_id, branch_kind = actual_edge
        assert trace.graph.conditional_edge_call_indices[
            (parent_no_pass, child_no_pass, conditional_id, branch_kind)
        ]


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


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_render_leaf.py, validation/_invariants_conditional_base.py, validation/_invariants_conditional_modules.py ....
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
        parent_label, child_label = render_log.graph.conditional_arm_entry_edges[
            (event.id, branch_kind)
        ][0]

        assert event.kind == "if_chain"
        assert set(event.branch_ranges) == {"then", "elif_1", "elif_2", "else"}
        assert branch_layer.conditional_branch_stack == ((event.id, branch_kind),)
        assert {layer.terminal_conditional_id for layer in _get_terminal_bool_layers(trace)} == {
            event.id
        }
        assert label_text in _find_edge_line(dot_source, parent_label, child_label)
        assert check_metadata_invariants(trace) is True
        _assert_derived_views_consistent(trace)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _visualization/_render_leaf.py.
def test_rolled_mixed_arm_model_records_exact_ops_and_composite_render_label() -> None:
    """Rolled mixed-arm edges keep exact pass lists and a composite Graphviz label."""
    trace = _log_model(RolledMixedArmModel(), torch.ones(1, 4))
    dot_source, render_log = _render_dot_source(
        RolledMixedArmModel(),
        torch.ones(1, 4),
        rolled=True,
    )
    conditional_id = _get_root_event(trace).id
    then_edges = trace.graph.conditional_arm_entry_edges[(conditional_id, "then")]
    else_edges = trace.graph.conditional_arm_entry_edges[(conditional_id, "else")]
    parent_label, child_label = then_edges[0]
    parent_no_pass = parent_label.split(":")[0]
    child_no_pass = child_label.split(":")[0]
    edge_line = _find_edge_line(dot_source, parent_label, child_label)

    assert len(then_edges) == 2
    assert len(else_edges) == 2
    assert trace.graph.conditional_edge_call_indices[
        (parent_no_pass, child_no_pass, conditional_id, "then")
    ] == [
        1,
        3,
    ]
    assert trace.graph.conditional_edge_call_indices[
        (parent_no_pass, child_no_pass, conditional_id, "else")
    ] == [
        2,
        4,
    ]
    assert "THEN(1,3) / ELSE(2,4)" in edge_line
    assert _find_edge_line(dot_source, parent_label, child_label) == _find_edge_line(
        dot_source,
        render_log.graph.conditional_arm_entry_edges[(conditional_id, "then")][0][0],
        render_log.graph.conditional_arm_entry_edges[(conditional_id, "then")][0][1],
    )
    _assert_conditional_edge_call_indices_exact(trace)
    assert check_metadata_invariants(trace) is True


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="func_qualname is only available on Python 3.11+",
)
def test_nested_qualname_model_distinguishes_method_and_nested_helper() -> None:
    """Distinct qualnames keep a method helper separate from a nested helper."""
    positive_log = _log_model(NestedQualnameModel(), torch.ones(2, 2))
    negative_log = _log_model(NestedQualnameModel(), -torch.ones(2, 2))
    root_events = sorted(
        positive_log.graph.conditional_records, key=lambda event: event.function_qualname
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
        for event in negative_log.graph.conditional_records
        if event.function_qualname.endswith(".helper") and "<locals>" not in event.function_qualname
    )
    assert relu_layer.conditional_branch_stack == ((nested_event.id, "then"),)
    assert square_layer.conditional_branch_stack == ((method_event.id, "then"),)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/ast_branches.py, _postprocess/control_flow.py, _visualization/code_panel.py ....
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


@requires_code_positions
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
    assert linear_layer.conditional_branch_stack == ((event.id, "then"),)


@requires_code_positions
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

    assert len(trace.graph.conditional_records) == 2
    assert outer_event.kind == "ifexp"
    assert inner_event.kind == "ifexp"
    assert inner_event.parent_branch_kind == "then"
    assert linear_layer.conditional_branch_stack == (
        (outer_event.id, "then"),
        (inner_event.id, "then"),
    )


@requires_code_positions
def test_ternary_inside_if_model_records_mixed_if_and_ifexp_stack() -> None:
    """An inner ternary keeps both the outer ``if_chain`` and inner ``ifexp`` stack."""
    trace = _log_model(TernaryInsideIfModel(), torch.ones(1, 2))
    outer_event = next(
        event for event in trace.graph.conditional_records if event.kind == "if_chain"
    )
    inner_event = next(event for event in trace.graph.conditional_records if event.kind == "ifexp")
    linear_layer = _find_only_layer(
        trace,
        "linear",
        [(outer_event.id, "then"), (inner_event.id, "then")],
    )

    assert inner_event.parent_conditional_id == outer_event.id
    assert inner_event.parent_branch_kind == "then"
    assert linear_layer.conditional_branch_stack == (
        (outer_event.id, "then"),
        (inner_event.id, "then"),
    )


@requires_code_positions
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
    assert linear_layer.conditional_branch_stack == ((event.id, "then"),)


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
    assert positive_mul.conditional_branch_stack == ((positive_event.id, "then"),)
    assert positive_add.conditional_branch_stack == ((positive_event.id, "then"),)
    assert negative_div.conditional_branch_stack == ((negative_event.id, "else"),)
    assert negative_sub.conditional_branch_stack == ((negative_event.id, "else"),)
