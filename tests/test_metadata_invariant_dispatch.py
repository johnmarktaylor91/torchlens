"""Behavior-parity tests for metadata invariant backend dispatch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.validation import invariants


PRE_REFACTOR_TORCH_SEQUENCE = (
    "trace_self_consistency",
    "region_replay_provenance",
    "backward_graph_invariants",
    "special_layer_lists",
    "graph_topology",
    "op_log_fields",
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
    "loop_detection_invariants",
    "distance_invariants",
    "graph_connectivity",
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
    "backend_neutral_module_mode_invariants",
    "graph_ordering",
    "lookup_key_consistency",
)


def test_metadata_invariant_dispatch_preserves_torch_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert torch dispatch executes the pre-refactor torch check sequence."""

    executed = _install_dispatch_spies(monkeypatch)
    trace = SimpleNamespace(backend="torch")

    assert invariants.check_metadata_invariants(trace)

    assert tuple(executed) == PRE_REFACTOR_TORCH_SEQUENCE
    assert set(executed) == set(PRE_REFACTOR_TORCH_SEQUENCE)


def test_metadata_invariant_dispatch_preserves_non_torch_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert non-torch dispatch executes the pre-refactor non-torch subset."""

    executed = _install_dispatch_spies(monkeypatch)
    trace = SimpleNamespace(backend="mlx")

    assert invariants.check_metadata_invariants(trace)

    assert tuple(executed) == PRE_REFACTOR_NON_TORCH_SEQUENCE
    assert set(executed) == set(PRE_REFACTOR_NON_TORCH_SEQUENCE)


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
