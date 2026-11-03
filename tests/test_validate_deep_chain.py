"""Regression tests for ``tl.validate`` on a plain ``Linear`` + ``ReLU`` chain.

A ``nn.Sequential`` of six or more ``Linear`` + ``ReLU`` pairs failed
``tl.validate(..., scope="forward", validate_metadata=True)`` at every width,
dtype and seed, on the ``equivalence_symmetry`` metadata invariant. Replay and
perturbation passed; the capture was correct.

The torch capture keys ``Trace.graph.op_equivalence_classes`` by the pre-suffix class
and stamps each op with that class plus its module path joined without a
separator. The ReLU at module ``"1"`` is stamped ``relu_<hash>_module1`` +
``"1"``, which is the registry key of the ReLU at module ``"11"`` (first
present at depth 6). The invariant looked the stamped string up as a registry
key, so it compared the honest ReLU at ``"1"`` against the foreign group of the
ReLU at ``"11"``. It now takes the group that holds the op under a key that
prefixes the op's stamped class (an op's own key always does), and falls back
to the stamped-class lookup only when no such group holds the op.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.validation.invariants import MetadataInvariantError, check_metadata_invariants

# Smallest failing chain: ReLUs sit at modules "1", "3", ..., "11".
_DEPTH = 6
_WIDTH = 8


def _linear_relu_chain(depth: int = _DEPTH, width: int = _WIDTH) -> nn.Sequential:
    """Build a ``Sequential`` of ``depth`` ``Linear`` + ``ReLU`` pairs.

    Parameters
    ----------
    depth:
        Number of ``Linear`` + ``ReLU`` pairs.
    width:
        Feature width of every ``Linear``.

    Returns
    -------
    nn.Sequential
        The chain, seeded for determinism.
    """

    torch.manual_seed(0)
    layers: list[nn.Module] = []
    for _ in range(depth):
        layers += [nn.Linear(width, width), nn.ReLU()]
    return nn.Sequential(*layers)


def _relu_at(trace: tl.Trace, module_address: str) -> tl.Op:
    """Return the ReLU op whose innermost module is ``module_address``.

    Parameters
    ----------
    trace:
        Captured chain.
    module_address:
        Address of the ``nn.ReLU`` module in the ``Sequential``.

    Returns
    -------
    tl.Op
        The single ReLU op in that module.
    """

    return next(
        op
        for op in trace._layer_list
        if op.type == "relu" and list(op.modules) == [f"{module_address}:1"]
    )


def test_validate_passes_on_linear_relu_chain_at_collision_depth() -> None:
    """The forward verdict is True once a ReLU sits at module ``"11"``."""

    model = _linear_relu_chain()
    x = torch.randn(2, _WIDTH)
    assert tl.validate(model, x, scope="forward", seed=0, validate_metadata=True) is True


def test_equivalence_symmetry_resolves_registry_key_without_suffix_collision() -> None:
    """A stamped class that spells another op's registry key is not a lookup hit.

    The precondition pins the collision itself, so the test stays meaningful
    only while the capture still produces it.
    """

    trace = tl.trace(_linear_relu_chain(), torch.randn(2, _WIDTH))
    try:
        relu_1 = _relu_at(trace, "1")
        relu_11 = _relu_at(trace, "11")
        foreign_group = trace.graph.op_equivalence_classes.get(relu_1.equivalence_class)
        assert foreign_group is not None and relu_11.label in foreign_group, (
            "precondition: the ReLU at module '1' is stamped with the registry key "
            "of the ReLU at module '11'"
        )
        assert relu_1.label not in foreign_group
        assert relu_1.equivalent_ops == {relu_1.label}
        check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_equivalence_symmetry_still_fires_on_scrubbed_registry_group() -> None:
    """Removing the colliding op's label from its own group still trips the check."""

    trace = tl.trace(_linear_relu_chain(), torch.randn(2, _WIDTH))
    try:
        relu_1 = _relu_at(trace, "1")
        own_key = next(
            key
            for key, group in trace.graph.op_equivalence_classes.items()
            if relu_1.label in group
        )
        assert own_key != relu_1.equivalence_class
        trace.graph.op_equivalence_classes[own_key] = set()
        with pytest.raises(MetadataInvariantError, match="equivalence_symmetry"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_equivalence_symmetry_still_fires_on_one_sided_rebind() -> None:
    """Rebinding the colliding op's view to the foreign group trips the check."""

    trace = tl.trace(_linear_relu_chain(), torch.randn(2, _WIDTH))
    try:
        relu_1 = _relu_at(trace, "1")
        relu_11 = _relu_at(trace, "11")
        relu_1.equivalent_ops = {relu_11.label}
        with pytest.raises(MetadataInvariantError, match="equivalence_symmetry"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()
