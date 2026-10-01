"""Shared-root plant for ground-truth output enumeration (b9 R74/75-1).

Sol's live probe: a planted drop-the-last-leaf mutation in the ONE backends
output walker made capture AND the validation oracle omit output #2 of a
2-tuple model, and ``validate_forward_pass`` returned True -- a dropped
capture output survived the tripwire because both sides resolved through the
same callable. The fix is the validation-owned independent traversal
(``torchlens.validation._output_walk``) cross-checked against the adapter's
enumeration inside ``validate_forward_pass``.

This file is the handoff contract: the plant re-applies sol's exact
mutation shape and validation must now FAIL; the structural test keeps the
two roots from ever re-merging.
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch
import torch.nn as nn

import torchlens as tl
import torchlens.backends.torch.ops as torch_ops
from torchlens.validation import _output_walk

pytestmark = pytest.mark.smoke


class _TupleOut(nn.Module):
    """Model returning a 2-tuple, the shape of sol's original plant."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return two derived tensors.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Two independently derived leaves.
        """

        y = torch.relu(x)
        return y, y + 1


def test_validation_survives_pristine_tuple_output():
    """Control: the 2-tuple model validates before any plant."""

    assert tl.validation.validate_forward_pass(_TupleOut(), [torch.randn(3)], input_kwargs={})


def test_planted_walker_leaf_drop_fails_validation(monkeypatch: pytest.MonkeyPatch):
    """Sol's shared-root plant now goes RED instead of validating True.

    The plant wraps the real backends walker and drops the LAST yielded
    leaf -- the exact mutation that previously made both capture and the
    oracle agree on the truncated enumeration.
    """

    real_walker = torch_ops._walk_output_tensors_with_paths

    def _dropping_walker(output):
        """Yield the real walk minus its final leaf.

        Parameters
        ----------
        output:
            Model output tree.

        Returns
        -------
        list
            Truncated walk results.
        """

        rows = list(real_walker(output))
        return rows[:-1]

    monkeypatch.setattr(torch_ops, "_walk_output_tensors_with_paths", _dropping_walker)
    with pytest.warns(RuntimeWarning, match="output-enumeration defect"):
        result = tl.validation.validate_forward_pass(_TupleOut(), [torch.randn(3)], input_kwargs={})
    assert result is False, (
        "a dropped output leaf in the shared walker still validated True: "
        "the independent cross-check is disarmed"
    )


def test_independent_walker_never_imports_the_capture_adapter():
    """The two enumeration roots must stay distinct callables and modules.

    If both sides ever resolve to the same callable or the independent
    module grows a ``torchlens.backends`` import, the R74/75-1 defect class
    (one defect silencing both sides) is structurally possible again.
    """

    assert _output_walk.independent_output_tensor_ids is not (
        torch_ops._walk_output_tensors_with_paths
    )
    tree = ast.parse(inspect.getsource(_output_walk))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert "backends" not in name, (
                f"validation/_output_walk.py imports {name!r}: the independent "
                "traversal must never share the capture adapter's root"
            )


class _DeepNestOut(nn.Module):
    """Model returning one shallow leaf plus one leaf nested past the old ceiling."""

    _NEST_DEPTH = 12

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, object]:
        """Return a shallow tensor and a deeply list-nested sibling.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        tuple[torch.Tensor, object]
            Shallow leaf and a ``_NEST_DEPTH``-level nested list holding the
            second leaf.
        """

        y = torch.relu(x)
        deep: object = x + 1
        for _ in range(self._NEST_DEPTH):
            deep = [deep]
        return y, deep


def test_independent_walker_reaches_capture_depth_leaves():
    """A leaf nested past the OLD ceiling of 8 is still enumerated.

    The walker's ceiling used to be 8 while capture supports 200, so any
    leaf nested at depth 9+ was invisible to the cross-check and the
    comparator was one-sided.
    """

    leaf = torch.ones(2)
    deep: object = leaf
    for _ in range(12):
        deep = [deep]
    assert id(leaf) in _output_walk.independent_output_tensor_ids(deep)


def test_walker_ceiling_locksteps_capture_output_ceiling():
    """The independent ceiling must never fall below capture's output ceiling.

    Below it, the ``missed_by_adapter`` direction of the cross-check cannot
    see -- and can never flag -- a dropped leaf capture could reach. The
    constant stays a literal in ``_output_walk`` (independent second root);
    this test is the lockstep.
    """

    from torchlens.ir.container_registry import OUTPUT_TREE_MAX_DEPTH

    assert _output_walk._MAX_DEPTH >= OUTPUT_TREE_MAX_DEPTH


def test_planted_deep_leaf_drop_fails_validation(monkeypatch: pytest.MonkeyPatch):
    """The surviving mutant: a dropped DEEP leaf must fail validation.

    With the walker ceiling at 8 and capture's at 200, the drop-last-leaf
    plant on a depth-12 output validated True because the independent walk
    could not reach the dropped leaf (live surviving mutant, grind-p3 T11).
    """

    real_walker = torch_ops._walk_output_tensors_with_paths

    def _dropping_walker(output):
        """Yield the real walk minus its final leaf.

        Parameters
        ----------
        output:
            Model output tree.

        Returns
        -------
        list
            Truncated walk results.
        """

        rows = list(real_walker(output))
        return rows[:-1]

    monkeypatch.setattr(torch_ops, "_walk_output_tensors_with_paths", _dropping_walker)
    with pytest.warns(RuntimeWarning, match="output-enumeration defect"):
        result = tl.validation.validate_forward_pass(
            _DeepNestOut(), [torch.randn(3)], input_kwargs={}
        )
    assert result is False, (
        "a dropped output leaf nested past the old walker ceiling still "
        "validated True: the cross-check is one-sided"
    )


def test_validation_survives_pristine_deep_nest_output():
    """Control: the depth-12 nested output validates with no plant."""

    assert tl.validation.validate_forward_pass(_DeepNestOut(), [torch.randn(3)], input_kwargs={})
