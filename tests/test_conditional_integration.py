"""PYTEST_DONT_REWRITE

End-to-end coverage for the remaining Phase 8 conditional integration matrix.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pd = pytest.importorskip("pandas")

import torchlens as tl  # noqa: E402
import torchlens._postprocess.ast_branches as ast_branches  # noqa: E402
import torchlens._postprocess.graph_traversal as graph_traversal  # noqa: E402
import torchlens._utils.introspection as introspection  # noqa: E402
import torchlens.backends.torch._ops_shared_fields as _ops_shared_fields  # noqa: E402
import torchlens.backends.torch.ops as output_tensors  # noqa: E402
import torchlens.backends.torch.sources as source_tensors  # noqa: E402
from torchlens import trace as trace_fn  # noqa: E402
from torchlens._data_classes.op import Op  # noqa: E402
from torchlens._data_classes.trace import ConditionalEvent, Trace  # noqa: E402
from torchlens._utils._torch_compat import HAS_CODE_POSITIONS  # noqa: E402

# Same-line ternary arm attribution needs PEP 657 per-instruction columns; the
# degraded runtime deliberately fails closed instead (see the module docstring
# of torchlens/_postprocess/ast_branches.py and the gated fail-closed pin
# test_ternary_py310_fail_closed_model_drops_same_line_arm_attribution). Mirrors
# the marker in tests/test_conditional_branches.py.
requires_code_positions = pytest.mark.skipif(
    not HAS_CODE_POSITIONS,
    reason=(
        "PEP 657 bytecode column positions unavailable; same-line ternary arm "
        "attribution deliberately fails closed on this runtime"
    ),
)


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
        capture=tl.options.CaptureOptions(save_source_context=save_source_context),
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
    return [
        layer
        for layer in trace._layer_list
        if layer.control_flow.is_terminal_bool and layer.control_flow.is_scalar_bool
    ]


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
    matching_layers = [layer for layer in trace._layer_list if layer.func_name == func_name]
    if branch_stack is not None:
        matching_layers = [
            layer
            for layer in matching_layers
            if list(layer.control_flow.branch_stack) == list(branch_stack)
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
    matching_layers = [layer for layer in trace._layer_list if layer.func_name in set(func_names)]
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
        event for event in trace.graph.conditional_records if event.parent_conditional_id is None
    ]
    assert len(root_events) == 1
    return root_events[0]


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


@pytest.mark.skipif(
    sys.version_info < (3, 11), reason="func_qualname is only available on Python 3.11+"
)
def test_nested_qualname_model_distinguishes_method_and_nested_helper() -> None:
    """Distinct qualnames keep a method helper separate from a nested helper."""
    trace = _log_model(NestedQualnameModel(), torch.ones(2, 2))

    root_events = sorted(trace.graph.conditional_records, key=lambda event: event.function_qualname)
    relu_layer = _find_only_layer(trace, "relu")
    tanh_layer = _find_only_layer(trace, "tanh")

    assert len(root_events) == 2
    assert root_events[0].function_qualname != root_events[1].function_qualname
    nested_event = next(
        event for event in root_events if "<locals>.helper" in event.function_qualname
    )
    method_event = next(event for event in root_events if "<locals>" not in event.function_qualname)
    assert relu_layer.control_flow.branch_stack == ((nested_event.id, "then"),)
    assert tanh_layer.control_flow.branch_stack == ((method_event.id, "then"),)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/ast_branches.py.
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
    monkeypatch.setattr(_ops_shared_fields, "_get_code_context", _stack_without_qualname)
    monkeypatch.setattr(source_tensors, "_get_code_context", _stack_without_qualname)
    monkeypatch.setattr(graph_traversal, "_get_code_context", _stack_without_qualname)
    monkeypatch.setattr(ast_branches, "get_file_index", _ambiguous_file_index)

    trace = _log_model(SameLineNestedDefModel(), torch.ones(2, 2))
    relu_layer = _find_only_layer(trace, "relu")

    assert relu_layer.control_flow.branch_stack == ()
    assert trace.graph.conditional_arm_entry_edges == {}


@requires_code_positions
def test_ternary_ifexp_model_attributes_then_arm_as_ifexp() -> None:
    """A minimal ternary materialises an ``ifexp`` event and branch stack."""
    trace = _log_model(TernaryIfExpModel(), torch.ones(1, 2))

    event = _get_root_event(trace)
    bool_layer = _get_terminal_bool_layers(trace)[0]
    linear_layer = _find_only_layer(trace, "linear", [(event.id, "then")])

    assert event.kind == "ifexp"
    assert set(event.branch_ranges) == {"then", "else"}
    assert bool_layer.control_flow.context_kind == "ifexp"
    assert bool_layer.control_flow.is_terminal_conditional_bool is True
    assert linear_layer.control_flow.branch_stack == ((event.id, "then"),)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/ast_branches.py.
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

    assert bool_layer.control_flow.context_kind == "match_guard"
    assert bool_layer.control_flow.is_terminal_conditional_bool is False
    _assert_branchless_log(trace)


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="column offsets are only captured on Python 3.11+",
)
def test_ternary_py311_records_non_none_column_offsets_for_ifexp_frames() -> None:
    """Python 3.11+ captures ``col_offset`` on ternary frames."""
    trace = _log_model(TernaryPy310FailClosedModel(), torch.ones(1, 2))
    bool_layer = _get_terminal_bool_layers(trace)[0]

    assert any(frame.col_offset is not None for frame in bool_layer.source.frames)
    assert trace.graph.conditional_arm_entry_edges


@requires_code_positions
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

    assert positive_mul.control_flow.branch_stack == ((positive_event.id, "then"),)
    assert positive_add.control_flow.branch_stack == ((positive_event.id, "then"),)
    assert negative_div.control_flow.branch_stack == ((negative_event.id, "else"),)
    assert negative_sub.control_flow.branch_stack == ((negative_event.id, "else"),)
