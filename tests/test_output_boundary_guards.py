"""r-b4 R27-4: model-OUTPUT container walkers are depth-bounded and cycle-guarded.

A 3000-deep or self-referential forward output used to die mid-capture in a raw
``RecursionError`` (probe: 994 torchlens frames) from the unguarded output
walkers (``_build_container_spec``, ``_walk_tensor_occurrences``,
``_walk_supported_output_container``, ``_prove_runnable_output_lossless``).
They now share ``OUTPUT_TREE_MAX_DEPTH`` and degrade over-deep/cyclic subtrees
to the existing honest ``kind="opaque"`` lane (reconstructable=False; the
runnable prover refuses typed) instead of crashing. Cycle guards are
path-scoped, so DAG-shaped outputs stay fully walked.
"""

from __future__ import annotations

import torch
from torch import nn

import torchlens as tl
from torchlens._ir.container_registry import (
    OUTPUT_TREE_MAX_DEPTH,
    walk_container,
)


def _deep_list(depth: int, leaf: object) -> list[object]:
    """Build one ``depth``-level nested list around ``leaf``."""

    value: list[object] = [leaf]
    for _ in range(depth - 1):
        value = [value]
    return value


class _DeepOut(nn.Module):
    """Forward returning a nested list far beyond the output ceiling."""

    def forward(self, x: torch.Tensor) -> object:
        return _deep_list(OUTPUT_TREE_MAX_DEPTH + 300, x + 1)


class _CyclicOut(nn.Module):
    """Forward returning a self-referential list."""

    def forward(self, x: torch.Tensor) -> object:
        out: list[object] = [x + 1]
        out.append(out)
        return out


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/torch/_outputs.py, campaign mutant kills.
def test_deep_output_captures_instead_of_recursion_error() -> None:
    """A too-deep forward output degrades honestly instead of crashing capture."""

    log = tl.trace(_DeepOut(), torch.ones(2))
    assert len(log) > 0


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_cyclic_output_captures_instead_of_recursion_error() -> None:
    """A self-referential forward output degrades honestly instead of crashing."""

    log = tl.trace(_CyclicOut(), torch.ones(2))
    assert len(log) > 0


# Kept by shrink-r4 (per-id subset rule): sole coverage of _ir/_container_type_probes.py, _ir/container_registry.py.
def test_overdeep_spec_degrades_to_opaque_and_unreconstructable() -> None:
    """The registry spec builder records the over-deep subtree as opaque."""

    deep = _deep_list(OUTPUT_TREE_MAX_DEPTH + 50, torch.ones(1))
    result = walk_container(deep, role="output", capability="full")
    assert result is not None
    assert result.reconstructable is False


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_runnable_prover_refuses_deep_and_cyclic_typed() -> None:
    """Refuse-unless-proved: the losslessness prover names the guard reasons."""

    import torchlens.backends.torch._ops_containers

    deep = _deep_list(OUTPUT_TREE_MAX_DEPTH + 50, torch.ones(1))
    proved, reason = torchlens.backends.torch._ops_containers._prove_runnable_output_lossless(deep)
    assert proved is False
    assert reason == "output_tree_depth_exceeded"

    cyclic: list[object] = [torch.ones(1)]
    cyclic.append(cyclic)
    proved, reason = torchlens.backends.torch._ops_containers._prove_runnable_output_lossless(
        cyclic
    )
    assert proved is False
    assert reason.startswith("output_tree_cycle:")

    shared = [torch.ones(1)]
    proved, reason = torchlens.backends.torch._ops_containers._prove_runnable_output_lossless(
        [shared, shared]
    )
    assert proved is True


# Kept by shrink-r4 (per-id subset rule): sole coverage of _ir/container_registry.py.
def test_moderate_output_nesting_still_reconstructable() -> None:
    """Outputs well under the ceiling keep full reconstructable specs."""

    nested = {"a": (torch.ones(1), [torch.ones(1) * 2])}
    result = walk_container(nested, role="output", capability="full")
    assert result is not None
    assert result.reconstructable is True
