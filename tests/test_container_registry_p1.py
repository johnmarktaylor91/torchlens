"""P1 tests for the output container registry."""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

import torchlens as tl
from torchlens.ir.container import ContainerSpec, DictKey
from torchlens.ir.container_registry import (
    ContainerLeafOccurrence,
    ContainerRecord,
    ContainerRegistry,
    FuncSite,
    IdentityEntry,
    Phase,
    Role,
)


class RepeatedTensorOutput(nn.Module):
    """Return the same tensor at two output-container paths."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        dict[str, torch.Tensor]
            Dict with repeated tensor leaves.
        """

        y = x.relu()
        return {"a": y, "b": y}


@dataclass
class PairOutput:
    """Weak-referenceable output dataclass for teardown tests."""

    left: torch.Tensor
    right: torch.Tensor


class PairOutputModel(nn.Module):
    """Return a dataclass output container."""

    last_output_ref: weakref.ReferenceType[PairOutput] | None = None

    def forward(self, x: torch.Tensor) -> PairOutput:
        """Run the model.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        PairOutput
            Dataclass output.
        """

        output = PairOutput(x + 1, x + 2)
        self.last_output_ref = weakref.ref(output)
        return output


def test_output_registry_keeps_repeated_tensor_path_occurrences() -> None:
    """The registry stores occurrences, not a deduplicated tensor set."""

    trace = tl.trace(
        RepeatedTensorOutput(),
        torch.tensor([-1.0, 2.0]),
        capture_output_structure=True,
    )

    model_records = [
        record for record in trace._containers.values() if Role.MODEL_OUTPUT in record.roles
    ]
    assert len(model_records) == 1
    snapshot = model_records[0].snapshots[-1]

    assert [occ.occ_index for occ in snapshot.leaf_occurrences] == [0, 1]
    assert [tuple(occ.path) for occ in snapshot.leaf_occurrences] == [
        (DictKey("a"),),
        (DictKey("b"),),
    ]
    assert len({occ.tensor_identity for occ in snapshot.leaf_occurrences}) == 1


def test_container_registry_identity_guard_handles_reused_id_slot() -> None:
    """A stale id-map slot is not reused unless the live object is identical."""

    registry = ContainerRegistry()
    first = []
    second = []
    registry.records[7] = ContainerRecord(
        ordinal=7,
        object_kind="builtins.list",
        label="builtins.list#7",
        first_seen_event_index=1,
    )
    registry.id_to_entry[id(second)] = IdentityEntry(obj=first, ordinal=7)

    record = registry.register_snapshot(
        second,
        site=FuncSite(func_call_id=2, position="return"),
        role=Role.CALL_OUTPUT,
        phase=Phase.POST_CALL,
        observed_at_event_index=2,
        spec=ContainerSpec(kind="list", length=0),
        leaf_occurrences=(),
        reconstructable=True,
    )

    assert record.ordinal != 7
    assert registry.id_to_entry[id(second)].obj is second


def test_no_live_container_registry_state_survives_final_streamed_or_cached_trace(
    tmp_path: Path,
) -> None:
    """Final, streamed, and cached traces do not retain live registry state."""

    model = PairOutputModel()
    final_trace = tl.trace(
        model,
        torch.tensor([1.0]),
        capture_output_structure=True,
    )
    final_ref = model.last_output_ref
    assert final_ref is not None
    _assert_no_live_registry_state(final_trace)
    gc.collect()
    assert final_ref() is None

    streamed_model = PairOutputModel()
    streamed_trace = tl.trace(
        streamed_model,
        torch.tensor([1.0]),
        capture_output_structure=True,
        storage=tl.to_disk(tmp_path / "streamed.tlspec"),
    )
    streamed_ref = streamed_model.last_output_ref
    assert streamed_ref is not None
    _assert_no_live_registry_state(streamed_trace)
    gc.collect()
    assert streamed_ref() is None

    cached_model = PairOutputModel()
    cached_trace = tl.trace(
        cached_model,
        torch.tensor([1.0]),
        capture_output_structure=True,
        cache=True,
        cache_dir=tmp_path / "cache",
    )
    cached_ref = cached_model.last_output_ref
    assert cached_ref is not None
    _assert_no_live_registry_state(cached_trace)
    gc.collect()
    assert cached_ref() is None


def test_output_container_views_match_legacy_fields() -> None:
    """Registry-backed views preserve op.container and final reconstruction behavior."""

    trace = tl.trace(PairOutputModel(), torch.tensor([1.0]), capture_output_structure=True)
    output_op = trace.ops[trace.output_layers[0]]

    container = output_op.container
    assert isinstance(container, tl.Container)
    assert output_op.output_containers == (container,)
    assert container.kind == output_op.container_spec.kind
    assert container.leaves == tuple(trace.ops[label] for label in trace.output_layers)

    rebuilt = trace.reconstruct_output()
    assert isinstance(rebuilt, PairOutput)
    assert torch.equal(rebuilt.left, torch.tensor([2.0]))
    assert torch.equal(rebuilt.right, torch.tensor([3.0]))


def test_flag_off_has_no_container_registry_attr() -> None:
    """Default flag-off traces do not expose portable registry records."""

    trace = tl.trace(RepeatedTensorOutput(), torch.tensor([1.0]))

    assert "_containers" not in trace.__dict__
    assert "_container_ordinals_by_output_op_label" not in trace.__dict__
    assert "_build_state" not in trace.__dict__


def _assert_no_live_registry_state(trace: tl.Trace) -> None:
    """Assert no capture-only registry state survives on a trace."""

    assert "_build_state" not in trace.__dict__
    assert "_container_ordinals_by_output_op_label" not in trace.__dict__
    for record in trace._containers.values():
        for snapshot in record.snapshots:
            assert all(
                isinstance(occ, ContainerLeafOccurrence) for occ in snapshot.leaf_occurrences
            )
            for occurrence in snapshot.leaf_occurrences:
                assert not isinstance(occurrence.tensor_identity, int)
