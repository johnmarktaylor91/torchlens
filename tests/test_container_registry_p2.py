"""P2 tests for input-side container capture."""

from __future__ import annotations

import torch
import pytest
from torch import nn

import torchlens as tl
from torchlens.ir.container import DictKey, TupleIndex
from torchlens.ir.container_registry import FuncSite, Phase, Role


class ScalarConfigModel(nn.Module):
    """Use scalar config containers that should not become records."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run reshape, permute, and conv config tuple operations."""

        weight = torch.ones(1, 1, 3, 3)
        y = torch.reshape(x, (1, 1, 2, 3))
        y = torch.permute(y, (0, 1, 3, 2))
        return torch.nn.functional.conv2d(y, weight, stride=(1, 1), padding=(1, 1))


class NestedInputModel(nn.Module):
    """Consume nested tensor-bearing containers."""

    def forward(self, payload: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        """Stack nested payload tensors."""

        return torch.stack(payload["items"]).sum()


class StackModel(nn.Module):
    """Call ``torch.stack`` with a list input container."""

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Stack three tensors."""

        return torch.stack([a, b, c])


class StackReturnModel(nn.Module):
    """Return a stack result inside a final-output container."""

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
        """Stack two tensors and return the result in a dict."""

        return {"stacked": torch.stack([a, b])}


class MutatingCacheModel(nn.Module):
    """Mutate and return the same cache container."""

    def forward(self, cache: list[torch.Tensor], x: torch.Tensor) -> list[torch.Tensor]:
        """Append ``x`` to ``cache`` and return it."""

        cache.append(x)
        return cache


class ThreadedCacheModel(nn.Module):
    """Thread an unchanged past-key-values container through repeated modules."""

    def __init__(self) -> None:
        """Initialize identity layers."""

        super().__init__()
        self.layers = nn.ModuleList([CacheLayer() for _ in range(3)])

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Pass the cache through at least three layer boundaries."""

        for layer in self.layers:
            x, past_key_values = layer(x, past_key_values)
        return x, past_key_values


class CacheLayer(nn.Module):
    """Layer that consumes and returns an unchanged cache container."""

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        """Return ``x`` and ``past_key_values`` unchanged."""

        return x + 1, past_key_values


def test_scalar_config_containers_emit_no_input_records() -> None:
    """Shape/dim/conv config tuples stay metadata, not input container records."""

    trace = tl.trace(
        ScalarConfigModel(),
        torch.randn(1, 6),
        capture_container_structure=True,
    )

    input_records = _records_with_role(trace, Role.CALL_INPUT)
    assert not input_records


def test_nested_tensor_bearing_input_records_recursively() -> None:
    """A nested dict-of-lists input container creates input records."""

    payload = {"items": [torch.tensor([1.0]), torch.tensor([2.0])]}
    trace = tl.trace(NestedInputModel(), payload, capture_container_structure=True)

    records = _records_with_role(trace, Role.MODEL_INPUT)
    assert len(records) == 1
    snapshot = records[0].snapshots[0]
    assert snapshot.phase == Phase.PRE_CALL
    assert [occ.path for occ in snapshot.leaf_occurrences] == [
        (DictKey("items"), TupleIndex(0)),
        (DictKey("items"), TupleIndex(1)),
    ]
    assert trace.input_structure.kind == "dict"


def test_same_site_input_output_mutation_has_distinct_snapshots() -> None:
    """A mutated cache records PRE input and POST output snapshots that differ."""

    cache = [torch.tensor([1.0])]
    trace = tl.trace(
        MutatingCacheModel(),
        (cache, torch.tensor([2.0])),
        capture_container_structure=True,
    )

    records = [
        record
        for record in trace._containers.values()
        if Role.CALL_INPUT in record.roles and Role.CALL_OUTPUT in record.roles
    ]
    assert records
    snapshots = records[0].snapshots
    input_snapshot = next(snapshot for snapshot in snapshots if snapshot.role == Role.CALL_INPUT)
    output_snapshot = next(snapshot for snapshot in snapshots if snapshot.role == Role.CALL_OUTPUT)
    assert input_snapshot.phase == Phase.PRE_CALL
    assert output_snapshot.phase == Phase.POST_CALL
    assert input_snapshot.spec != output_snapshot.spec
    assert input_snapshot.spec.length == 1
    assert output_snapshot.spec.length == 2


def test_threaded_past_key_values_dedups_identical_snapshots() -> None:
    """Threaded cache observations share one snapshot body plus site aliases."""

    pkv = tuple((torch.ones(1), torch.zeros(1)) for _ in range(2))
    trace = tl.trace(
        ThreadedCacheModel(),
        (torch.tensor([3.0]), pkv),
        capture_container_structure=True,
    )

    records = [
        record
        for record in trace._containers.values()
        if Role.CALL_INPUT in record.roles
        and any(snapshot.site_aliases for snapshot in record.snapshots)
    ]
    assert records
    input_snapshots = [
        snapshot
        for record in records
        for snapshot in record.snapshots
        if snapshot.role == Role.CALL_INPUT
    ]
    snapshot_count = len(input_snapshots)
    alias_count = sum(len(snapshot.site_aliases) for snapshot in input_snapshots)
    assert snapshot_count == 1
    assert alias_count >= 2


def test_stack_list_arg_registers_as_consumed_input_container() -> None:
    """The list passed to ``torch.stack`` is reachable from that op."""

    trace = tl.trace(
        StackModel(),
        (torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])),
        capture_container_structure=True,
    )

    stack_op = next(op for op in trace.ops if op.func_name == "stack")
    containers = stack_op.input_containers
    assert len(containers) == 1
    assert containers[0].kind == "list"
    assert len(containers[0].leaves) == 3


def test_op_containers_unions_input_and_output_views() -> None:
    """The union view includes input and output containers for one op."""

    trace = tl.trace(
        StackReturnModel(),
        (torch.tensor([1.0]), torch.tensor([2.0])),
        capture_container_structure=True,
    )

    output_op = trace.ops[trace.output_layers[0]]
    containers = output_op.containers

    assert len(output_op.input_containers) == 1
    assert len(output_op.output_containers) == 1
    assert {container.kind for container in containers} == {"list", "dict"}


def test_input_at_selects_top_level_and_nested_input_leaf() -> None:
    """Nested input selector resolves leaf ops matching the captured structure."""

    payload = {
        "attention_mask": torch.tensor([1.0]),
        "past_key_values": ((torch.tensor([2.0]), torch.tensor([3.0])),),
    }
    trace = tl.trace(NestedInputSelectModel(), payload, capture_container_structure=True)

    top = trace.resolve_sites(tl.input_at("attention_mask")).first()
    nested = trace.resolve_sites(tl.input_at("past_key_values", 0, 1)).first()

    assert torch.equal(top.out, payload["attention_mask"])
    assert torch.equal(nested.out, payload["past_key_values"][0][1])


class NestedInputSelectModel(nn.Module):
    """Read top-level and nested input leaves."""

    def forward(self, payload: dict[str, object]) -> torch.Tensor:
        """Return a sum involving two selected input leaves."""

        attention_mask = payload["attention_mask"]
        past_key_values = payload["past_key_values"]
        assert isinstance(attention_mask, torch.Tensor)
        assert isinstance(past_key_values, tuple)
        return attention_mask + past_key_values[0][1]


def test_role_general_reconstructs_input_and_call_output_containers() -> None:
    """Live reconstruction works for non-final roles."""

    payload = {"items": [torch.tensor([1.0]), torch.tensor([2.0])]}
    input_trace = tl.trace(NestedInputModel(), payload, capture_container_structure=True)
    rebuilt_input = input_trace.reconstruct_container(role=Role.MODEL_INPUT)

    assert torch.equal(rebuilt_input["items"][0], payload["items"][0])
    assert torch.equal(rebuilt_input["items"][1], payload["items"][1])

    cache = [torch.tensor([1.0])]
    output_trace = tl.trace(
        MutatingCacheModel(),
        (cache, torch.tensor([2.0])),
        capture_container_structure=True,
    )
    rebuilt_output = output_trace.reconstruct_container(role=Role.CALL_OUTPUT)

    assert isinstance(rebuilt_output, list)
    assert torch.equal(rebuilt_output[0], torch.tensor([1.0]))
    assert torch.equal(rebuilt_output[1], torch.tensor([2.0]))


def test_multi_snapshot_record_requires_selector_and_reconstructs_selected_role() -> None:
    """Multi-snapshot records reject ambiguous spec reads and support role selection."""

    cache = [torch.tensor([1.0])]
    trace = tl.trace(
        MutatingCacheModel(),
        (cache, torch.tensor([2.0])),
        capture_container_structure=True,
    )
    record = next(
        record
        for record in trace._containers.values()
        if Role.CALL_INPUT in record.roles and Role.CALL_OUTPUT in record.roles
    )

    with pytest.raises(ValueError, match="spec_at"):
        _ = record.spec

    input_spec = record.spec_at(role=Role.CALL_INPUT)
    output_spec = record.spec_at(role=Role.CALL_OUTPUT)
    rebuilt = trace.reconstruct_container(role=Role.CALL_OUTPUT)

    assert input_spec.length == 1
    assert output_spec.length == 2
    assert torch.equal(rebuilt[1], torch.tensor([2.0]))


def test_aliasing_empty_container_and_tensor_dict_key_degradation() -> None:
    """Aliased leaves are repeated, empty containers skip, tensor keys degrade."""

    from torchlens.ir.container_registry import walk_container

    tensor = torch.tensor([1.0])
    payload = {"alias": [tensor, tensor], tensor: torch.tensor([2.0])}
    aliased = walk_container(payload["alias"], role=Role.CALL_INPUT, capability="full_spec")
    assert aliased is not None
    assert len(aliased.leaf_occurrences) == 2
    assert len({occ.tensor_identity for occ in aliased.leaf_occurrences}) == 1

    assert walk_container({"items": []}, role=Role.MODEL_INPUT, capability="full_spec") is None
    degraded = walk_container(payload, role=Role.MODEL_INPUT, capability="full_spec")
    assert degraded is not None
    assert all(
        path[0] != DictKey(tensor) for path in (occ.path for occ in degraded.leaf_occurrences)
    )


def _records_with_role(trace: tl.Trace, role: Role) -> list[object]:
    """Return container records carrying ``role``."""

    return [record for record in getattr(trace, "_containers", {}).values() if role in record.roles]
