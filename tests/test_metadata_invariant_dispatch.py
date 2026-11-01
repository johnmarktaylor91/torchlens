"""Behavior-parity tests for metadata invariant backend dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.validation import invariants

# The pre-refactor sequences below are the dispatch-parity baseline. cert10
# ADDED three checks (receptive_field_metadata for every backend,
# pass_count_consistency for torch, and backend_neutral_graph_topology for
# non-torch) without dropping or reordering any pre-refactor check; the expected
# sequences include those additions. L1 wave 0 (6f417b43) ADDED
# site_key_invariants to both backends -- torch: after loop_detection_invariants
# (site keys are minted at grouping time, so recurrence metadata is a
# precondition) and before graph_topology; non-torch: after
# backend_neutral_module_mode_invariants -- again without dropping or
# reordering any pre-refactor check.
PRE_REFACTOR_TORCH_SEQUENCE = (
    "backend_identity_invariants",
    "trace_self_consistency",
    "region_replay_provenance",
    "backward_graph_invariants",
    "backend_neutral_accessor_refs",
    "receptive_field_metadata",
    "special_layer_lists",
    # loop_detection_invariants must precede graph_topology: recurrence
    # metadata is a precondition for pass-sensitive Layer accessors, so the
    # owning invariant has to report before topology walks Layer labels
    # (grind r1 trust, SF-31 corruption_loop root cause).
    "loop_detection_invariants",
    "graph_topology",
    "edge_use_parent_arg_consistency",
    "capture_edge_survival",
    "op_log_fields",
    "payload_metadata_invariants",
    "recurrence_invariants",
    "branching_invariants",
    "conditional_invariants",
    "layer_pass_layer_log_xrefs",
    "module_layer_containment",
    "module_hierarchy",
    "param_xrefs",
    "buffer_xrefs",
    "equivalence_symmetry",
    "graph_ordering",
    "pass_count_consistency",
    "distance_invariants",
    "graph_connectivity",
    # ancestry_closure recomputes the four reachability closures from parents/children;
    # it runs AFTER graph_connectivity so a dropped op is reported by the dangling-node
    # contract that owns it, not by the closure check that also notices.
    "ancestry_closure",
    "module_containment_logic",
    "lookup_key_consistency",
    "func_call_id_consistency",
)

PRE_REFACTOR_NON_TORCH_SEQUENCE = (
    "backend_identity_invariants",
    "trace_self_consistency",
    "region_replay_provenance",
    "non_torch_backward_inert",
    "backend_neutral_accessor_refs",
    "receptive_field_metadata",
    "backend_neutral_module_mode_invariants",
    "backend_neutral_graph_topology",
    "graph_ordering",
    "lookup_key_consistency",
)

EXPECTED_TORCH_SEQUENCE = (
    *PRE_REFACTOR_TORCH_SEQUENCE[:8],
    "site_key_invariants",
    *PRE_REFACTOR_TORCH_SEQUENCE[8:12],
    "primitive_op_invariants",
    *PRE_REFACTOR_TORCH_SEQUENCE[12:],
)
EXPECTED_NON_TORCH_SEQUENCE = (
    *PRE_REFACTOR_NON_TORCH_SEQUENCE[:4],
    "non_torch_primitive_op_inert",
    *PRE_REFACTOR_NON_TORCH_SEQUENCE[4:7],
    "site_key_invariants",
    *PRE_REFACTOR_NON_TORCH_SEQUENCE[7:],
)


def _is_ordered_subsequence(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    """Return whether ``needle`` appears in order within ``haystack``.

    Parameters
    ----------
    needle
        Historical invariant sequence.
    haystack
        Current invariant sequence.

    Returns
    -------
    bool
        Whether every historical entry survives in the same order.
    """

    cursor = iter(haystack)
    return all(any(candidate == item for candidate in cursor) for item in needle)


def test_metadata_invariant_dispatch_preserves_torch_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert torch dispatch executes the pre-refactor torch check sequence."""

    executed = _install_dispatch_spies(monkeypatch)
    trace = SimpleNamespace(_provenance_backend="torch")

    assert invariants.check_metadata_invariants(trace)

    assert tuple(executed) == EXPECTED_TORCH_SEQUENCE
    assert _is_ordered_subsequence(PRE_REFACTOR_TORCH_SEQUENCE, tuple(executed))


def test_metadata_invariant_dispatch_preserves_non_torch_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert non-torch dispatch executes the pre-refactor non-torch subset."""

    executed = _install_dispatch_spies(monkeypatch)
    trace = SimpleNamespace(_provenance_backend="mlx")

    assert invariants.check_metadata_invariants(trace)

    assert tuple(executed) == EXPECTED_NON_TORCH_SEQUENCE
    assert _is_ordered_subsequence(PRE_REFACTOR_NON_TORCH_SEQUENCE, tuple(executed))


def _install_dispatch_spies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace metadata invariant checks with execution-recording spies.

    Parameters
    ----------
    monkeypatch:
        Pytest monkeypatch fixture used to restore the contract table.

    Returns
    -------
    list[str]
        Mutable list populated with check names in execution order.
    """

    executed: list[str] = []
    spy_contracts = tuple(
        invariants.MetadataInvariantContract(
            name=contract.name,
            check=_make_spy(contract.name, executed),
            applies_to=contract.applies_to,
            requires_capability=contract.requires_capability,
        )
        for contract in invariants.METADATA_INVARIANT_CONTRACTS
    )
    monkeypatch.setattr(invariants, "METADATA_INVARIANT_CONTRACTS", spy_contracts)
    return executed


def _make_spy(name: str, executed: list[str]) -> invariants.MetadataInvariantFunc:
    """Return a metadata-invariant spy for ``name``.

    Parameters
    ----------
    name:
        Contract name to append when the spy runs.
    executed:
        Mutable execution log.

    Returns
    -------
    invariants.MetadataInvariantFunc
        Callable with the metadata invariant signature.
    """

    def _spy(trace: Any) -> None:
        """Record one metadata invariant dispatch.

        Parameters
        ----------
        trace:
            Trace-like object supplied by the dispatcher.
        """

        del trace
        executed.append(name)

    return _spy
