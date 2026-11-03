"""Ancestry/reachability invariants and the label surfaces they guard (B3 L4).

Four defects, one gap:

* The whole derived ancestry class had NO tripwire. Between two labs, eleven planted
  corruptions across ``input_ancestors``, ``output_descendants``, ``root_ancestors``,
  ``internal_source_ancestors``, ``internal_source_parents`` and ``distance_from_input``
  produced zero hits, while every neighbouring relation class fired and named itself.
* Parent/child symmetry was blind INSIDE a recurrence group: ``label_aliases`` folded the
  whole ``recurrent_ops`` group into the alias set, so an edge repointed from ``layer:N``
  to ``layer:M`` satisfied both directions and ``graph_ordering`` saw no violation --
  exactly the corruption loop grouping and removal/collapse repointing produce.
* Synthetic output nodes inherited ``internal_source_parents`` from their clone source,
  naming labels that are not parents at all, on every BatchNorm/buffer/factory model.
* Step 6's late ``buffer -> buffer_source`` edges never re-propagated ancestry, so ops
  downstream of a written buffer reported no input ancestry despite demonstrably
  depending on the model input.

Plus the pair that has to land together: ``buffer_source`` survived as a DANGLING raw
label, and the "no raw labels survive" tripwire pinned ``^l_\\d+$`` -- a spelling
TorchLens no longer emits -- so it could never fire on anything.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.errors import MetadataInvariantError
from torchlens.validation._invariants_entry import check_metadata_invariants
from torchlens.validation._invariants_equivalence import _check_graph_ordering


class _TwoInputBatchNorm(nn.Module):
    """Two inputs, two outputs, and live BatchNorm buffers."""

    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm1d(4)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix a normalized input with a second input."""

        return self.bn(a) + b, b * 2


class _Cell(nn.Module):
    """One recurrent step."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Linear then ReLU."""

        return torch.relu(self.lin(x))


class _Recurrent(nn.Module):
    """Three passes through one shared cell."""

    def __init__(self) -> None:
        super().__init__()
        self.cell = _Cell()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the cell three times."""

        for _ in range(3):
            x = self.cell(x)
        return x


class _NonRecurrent(nn.Module):
    """Three distinct linears (the non-recurrent control)."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Linear(4, 4)
        self.b = nn.Linear(4, 4)
        self.c = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Chain three linears."""

        return self.c(torch.relu(self.b(torch.relu(self.a(x)))))


class _WriteThenRead(nn.Module):
    """Write into a buffer slice, then read the buffer (the step-6 edge path)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Copy into a slice of the buffer and reduce it."""

        self.b[:2].copy_(x)
        return self.b.sum()


class _StaticRead(nn.Module):
    """Read-only buffer (the synthetic-output-node ISP fixture)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("b", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add a static buffer to the input."""

        return self.b + x


def _entry(trace: Any, layer_label: str) -> Any:
    """Return the ``layer_list`` record the invariants actually read."""

    for record in trace._layer_list:
        if record.layer_label == layer_label:
            return record
    raise AssertionError(f"{layer_label!r} not in layer_list")


def _fresh_two_io() -> Any:
    """Trace the two-input BatchNorm fixture."""

    return tl.trace(_TwoInputBatchNorm().train(), [torch.randn(4, 4), torch.randn(4, 4)])


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("batchnorm_1_2", "input_ancestors", {"input_2"}),
        ("batchnorm_1_2", "internal_source_parents", ["input_2"]),
        ("batchnorm_1_2", "internal_source_parents", []),
    ],
)
def test_planted_ancestry_corruption_is_named(target: str, field: str, value: Any) -> None:
    """Every ancestry corruption class must be caught, not silently accepted."""

    trace = _fresh_two_io()
    check_metadata_invariants(trace)  # baseline is clean
    setattr(_entry(trace, target), field, value)
    with pytest.raises(MetadataInvariantError) as excinfo:
        check_metadata_invariants(trace)
    assert excinfo.value.check_name == "ancestry_closure"


def test_cross_pass_child_repoint_is_caught_inside_a_recurrence_group() -> None:
    """A child edge repointed to the WRONG PASS must fail the symmetry check."""

    trace = tl.trace(_Recurrent(), torch.randn(1, 4))
    check_metadata_invariants(trace)
    record = None
    for candidate in trace._layer_list:
        if candidate.layer_label == "linear_1_1" and candidate.pass_number == 2:
            record = candidate
            break
    assert record is not None
    assert tuple(record.children) == ("relu_1_2:2",)
    record.children = ["relu_1_2:3"]
    with pytest.raises(MetadataInvariantError) as excinfo:
        check_metadata_invariants(trace)
    assert excinfo.value.check_name == "graph_topology"
    # The message must name the exact PASS, not just the group.
    assert "linear_1_1:2" in str(excinfo.value)


def test_buffer_write_descendants_keep_their_input_ancestry() -> None:
    """An op downstream of a written buffer must report the input it depends on."""

    trace = tl.trace(_WriteThenRead(), torch.ones(2))
    sum_op = trace["sum_1_3"]
    assert sum_op.parents == ("buffer_2",)
    assert "input_1" in sum_op.input_ancestors
    assert sum_op.has_input_ancestor
    assert "input_1" in trace["output_1"].input_ancestors


def test_buffer_write_descendant_ancestry_survives_the_validation_path() -> None:
    """The full replay+metadata path (depths off) must agree with the edges."""

    assert tl.validation.validate_forward_pass(
        _WriteThenRead(), torch.ones(2), seed=123, validate_metadata=True
    )


def test_forged_raw_label_reddens_the_survival_tripwire() -> None:
    """The tripwire must match the raw spelling TorchLens actually emits."""

    trace = tl.trace(_WriteThenRead(), torch.ones(2))
    _check_graph_ordering(trace)  # clean baseline

    relation_trace = tl.trace(_WriteThenRead(), torch.ones(2))
    _entry(relation_trace, "sum_1_3").parents = ["buffer_2_raw"]
    with pytest.raises(MetadataInvariantError) as relation_error:
        _check_graph_ordering(relation_trace)
    assert "buffer_2_raw" in str(relation_error.value)

    scalar_trace = tl.trace(_WriteThenRead(), torch.ones(2))
    _entry(scalar_trace, "buffer_2").buffer_source = "copy_1_2_raw"
    with pytest.raises(MetadataInvariantError) as scalar_error:
        _check_graph_ordering(scalar_trace)
    assert "buffer_source" in str(scalar_error.value)


# ---------------------------------------------------------------------------
# Generative corruption sweep (R73, round 3): the hand plants above pin the
# known corruption classes; this sweep GENERATES seeded random corruptions so
# coverage does not depend on the hand-list. Its first run found the
# fabricated-distance hole pinned below.
# ---------------------------------------------------------------------------

_ANCESTRY_SET_FIELDS = (
    "input_ancestors",
    "output_descendants",
    "root_ancestors",
    "internal_source_ancestors",
    "internal_source_parents",
)


def test_seeded_generative_ancestry_corruptions_are_caught() -> None:
    """Seeded random ancestry corruptions must trip the invariants.

    Every generated corruption must raise (the former fabricated-distance
    tolerance closed with the R73 hole fix), and the sweep must demonstrably
    do work (a floor on the caught count guards against the generator
    degenerating into no-op mutations).
    """

    import random

    rng = random.Random(42)
    caught = 0
    unledgered: list[str] = []
    for _ in range(20):
        trace = _fresh_two_io()
        record = rng.choice(list(trace._layer_list))
        kind = rng.choice(["add", "remove", "clear", "bool", "dist"])
        if kind in ("add", "remove", "clear"):
            field = rng.choice(_ANCESTRY_SET_FIELDS)
            old = set(getattr(record, field))
            if kind == "add":
                new = old | {"phantom_9_9"}
            elif kind == "remove" and old:
                new = set(old)
                new.discard(sorted(new)[0])
            else:
                new = set()
            if new == old:
                continue
            setattr(record, field, new)
            description = f"{record.layer_label}.{field}:{kind}"
        elif kind == "bool":
            record.has_internal_source_ancestor = not record.has_internal_source_ancestor
            description = f"{record.layer_label}.has_internal_source_ancestor:flip"
        else:
            record.min_distance_from_input = 77
            record.max_distance_from_input = 77
            description = f"{record.layer_label}.distance:77"
        try:
            check_metadata_invariants(trace)
        except MetadataInvariantError:
            caught += 1
        else:
            unledgered.append(description)
    assert not unledgered, f"generated ancestry corruptions were silently accepted: {unledgered}"
    assert caught >= 10, f"generative sweep degenerated: only {caught} corruptions caught"
