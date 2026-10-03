"""grind-p3 T11: validation/backward.py input/output walkers are guarded.

``_clone_inputs_with_grad`` and ``_sum_tensors`` were unguarded recursion with
no memo: a legal deep nest or a self-referential container died in a raw
``RecursionError`` from library internals, and a DAG-shaped tree that reuses
one sub-container under several paths expanded EXPONENTIALLY in shared-
substructure depth (x2 per level). Both walkers now share the boundary
nesting ceilings, refuse typed, and memoize per call -- preserving the exact
values the unmemoized walks produced.
"""

from __future__ import annotations

import pytest
import torch

from torchlens._errors import InvalidArgumentError
from torchlens._input_walk import INPUT_TREE_MAX_DEPTH
from torchlens.validation.backward import _clone_inputs_with_grad, _sum_tensors


def _deep_list(depth: int, leaf: object) -> object:
    """Build one ``depth``-level nested list around ``leaf``."""

    value = leaf
    for _ in range(depth):
        value = [value]
    return value


class TestCloneInputsWithGrad:
    """DAG/cycle/depth behavior of the backward-validation input cloner."""

    def test_shared_container_cloned_once_and_stays_aliased(self):
        """A container reused under two paths is cloned ONCE (the memo).

        The unmemoized walk cloned a shared node once per PATH -- exponential
        in shared-substructure depth -- and broke the aliasing topology the
        model itself would see.
        """

        shared = [torch.ones(2)]
        pair = [shared, shared]
        cloned = _clone_inputs_with_grad(pair)
        assert cloned[0] is cloned[1]
        assert cloned[0] is not shared
        assert torch.equal(cloned[0][0], shared[0])

    def test_deep_shared_dag_is_linear_not_exponential(self):
        """A depth-40 shared DAG completes (2^40 walks pre-memo)."""

        node: list[object] = [torch.ones(1)]
        for _ in range(40):
            node = [node, node]
        cloned = _clone_inputs_with_grad(node)
        assert cloned[0] is cloned[1]

    def test_reference_cycle_is_reproduced_not_recursion_error(self):
        """A self-referential list clones without a raw ``RecursionError``."""

        cyclic: list[object] = [torch.ones(2)]
        cyclic.append(cyclic)
        cloned = _clone_inputs_with_grad(cyclic)
        assert cloned[1] is cloned
        assert isinstance(cloned[0], torch.Tensor)
        assert cloned[0] is not cyclic[0]

    def test_over_deep_nest_refuses_typed(self):
        """Nesting past the shared input ceiling refuses typed, never raw."""

        deep = _deep_list(INPUT_TREE_MAX_DEPTH + 5, torch.ones(1))
        with pytest.raises(InvalidArgumentError) as excinfo:
            _clone_inputs_with_grad(deep)
        assert excinfo.value.fields["code"] == "input_tree_depth_exceeded"

    def test_grad_enabling_semantics_unchanged(self):
        """Floating clones require grad; ints and non-containers pass through."""

        cloned = _clone_inputs_with_grad({"x": torch.ones(2), "n": 3})
        assert cloned["x"].requires_grad
        assert cloned["n"] == 3


class TestSumTensors:
    """DAG/cycle/depth behavior of the default backward-validation loss."""

    @pytest.mark.smoke
    def test_shared_subtree_sum_is_occurrence_weighted(self):
        """The memoized sum equals the historical occurrence-weighted value."""

        t = torch.arange(4.0)
        shared = [t]
        tree = [shared, shared, [shared]]
        assert torch.equal(_sum_tensors(tree), t.sum() * 3)

    def test_deep_shared_dag_sum_is_linear(self):
        """A depth-40 shared DAG sums in O(nodes) with the exact DAG value."""

        node: list[object] = [torch.ones(1)]
        for _ in range(40):
            node = [node, node]
        assert float(_sum_tensors(node)) == float(2**40)

    def test_cyclic_output_refuses_typed(self):
        """A self-referential output container refuses typed, never raw."""

        cyclic: list[object] = [torch.ones(2)]
        cyclic.append(cyclic)
        with pytest.raises(InvalidArgumentError) as excinfo:
            _sum_tensors(cyclic)
        assert excinfo.value.fields["code"] == "output_tree_cycle"

    def test_over_deep_output_refuses_typed(self):
        """Nesting past the output ceiling refuses typed, never raw."""

        from torchlens.ir.container_registry import OUTPUT_TREE_MAX_DEPTH

        deep = _deep_list(OUTPUT_TREE_MAX_DEPTH + 5, torch.ones(1))
        with pytest.raises(InvalidArgumentError) as excinfo:
            _sum_tensors(deep)
        assert excinfo.value.fields["code"] == "output_tree_depth_exceeded"

    def test_tensor_free_output_still_raises_value_error(self):
        """The historical no-tensor refusal is preserved."""

        with pytest.raises(ValueError, match="at least one tensor"):
            _sum_tensors(["not-a-tensor"])
        with pytest.raises(ValueError, match="at least one tensor"):
            _sum_tensors([])
