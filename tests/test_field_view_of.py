"""The field views' many-unit form: ``op.projective_field.of`` and ``op.receptive_field.of``.

``.of(units, within=...)`` is the public spelling of unit-level influence
(JMT, 2026-11-02, influence.1). It is a thin wrapper over the private engine
in ``torchlens.receptive_field._influence``: the seed is ``tl.units`` over the
view's own op, ``within`` takes labels, module addresses, records and
selections, and the result is a trace-bound ``tl.Selection``. These tests pin
the wrapper: equality with the union of the per-unit ``.at()`` lifts in both
directions, equality with the private builder, list ``within`` with module
records, the default endpoints, selection algebra, a ``do()`` round trip, a
subgraph, and the typed refusals. The engine's own geometry is covered in
``tests/test_selection_influence.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._selection.graph import _influence
from torchlens.errors import SelectionError


class _Stack(nn.Module):
    """input (16x16) -> conv1 k3 p1 -> relu -> conv2 k3 s2 p1 (8x8) -> conv3 k3 (6x6)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, padding=1)
        self.conv2 = nn.Conv2d(2, 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv2(torch.relu(self.conv1(x))))


_PERM = [3, 1, 2, 0, 7, 5, 6, 4]


class _Routed(nn.Module):
    """A conv followed by row routing through an index list (data-dependent rule)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[:, :, _PERM, :]


def _armed(model: nn.Module, inputs: torch.Tensor):
    """Capture with autograd retained so the gradient method can probe."""

    capture = tl.options.CaptureOptions(backward_ready=True)
    return tl.trace(model, inputs.requires_grad_(True), capture=capture, save_mode="reference")


@pytest.fixture(scope="module")
def stack():
    """Armed capture of the conv stack on a 16x16 input."""

    torch.manual_seed(0)
    trace = _armed(_Stack(), torch.randn(1, 1, 16, 16))
    try:
        yield trace
    finally:
        trace.cleanup()


def _out(trace, address: str):
    """Return the only output op of the module at ``address``."""

    module = trace.modules[address]
    (label,) = module.calls[0].output_ops
    return trace.ops[label]


def _row(selection, op):
    """Return the one entry of ``selection`` at ``op``."""

    rows = [entry for entry in selection if f"{entry.key[0]}:{entry.key[1]}" == op.label]
    assert len(rows) == 1, f"expected one row for {op.label}, got {len(rows)}"
    return rows[0]


_UNITS = [(4, 5), (4, 6), (5, 5), (5, 6)]


def _full(units, channel: int = 0):
    """Lift windowed (row, col) coordinates to full (batch, channel, row, col) units."""

    return [(0, channel, *unit) for unit in units]


def _at_union(seed_op, units, target, direction: str) -> torch.Tensor:
    """OR the shipped per-unit ``.at()`` lifts at ``target`` over ``units``."""

    trace = seed_op.source_trace
    folded = torch.zeros(tuple(target.shape), dtype=torch.bool)
    for unit in units:
        if direction == "projective":
            box = seed_op.projective_field.at(unit, target=target)
        elif bool(getattr(target, "is_input", False)):
            box = seed_op.receptive_field.at(unit, input_op=target)
        else:
            box = seed_op.receptive_field.at(unit, source=target)
        folded |= _row(trace[box.__selection__()], target).mask
    return folded


@pytest.mark.parametrize("direction", ["projective", "receptive"])
def test_of_is_the_union_of_the_per_unit_at_lifts(stack, direction: str) -> None:
    """``.of`` over a patch equals the union over its units of ``.at`` at every target."""

    if direction == "projective":
        seed_op = _out(stack, "conv1")
        within = ["conv2", "conv3"]
        targets = [_out(stack, "conv2"), _out(stack, "conv3")]
        region = stack["conv1"].projective_field.of(_full(_UNITS), within=within)
    else:
        seed_op = _out(stack, "conv2")
        units = [(2, 2), (2, 3), (3, 3)]
        targets = [stack.input_ops[0], _out(stack, "conv1")]
        within = [stack.input_ops[0], "conv1"]
        region = stack["conv2"].receptive_field.of(_full(units), within=within)
    assert isinstance(region, tl.Selection)
    assert [f"{entry.key[0]}:{entry.key[1]}" for entry in region] == [op.label for op in targets]
    for target in targets:
        entry = _row(region, target)
        seed_units = _UNITS if direction == "projective" else [(2, 2), (2, 3), (3, 3)]
        assert torch.equal(entry.mask, _at_union(seed_op, seed_units, target, direction))
        assert entry.provenance.basis == "geometric"
        assert entry.provenance.field_status in {"exact", "upper_bound"}


@pytest.mark.parametrize("direction", ["projective", "receptive"])
def test_of_equals_the_private_builder(stack, direction: str) -> None:
    """The wrapper adds nothing: same rows, masks and provenance as the engine query."""

    seed_op = _out(stack, "conv2")
    within = [_out(stack, "conv3")] if direction == "projective" else [_out(stack, "conv1")]
    view = getattr(stack["conv2"], f"{direction}_field")
    region = view.of(_full([(2, 2), (3, 4)], channel=1), within=within)
    engine = stack[
        _influence(
            tl.units(seed_op.label, _full([(2, 2), (3, 4)], channel=1)),
            within=within,
            direction=direction,
        )
    ]
    assert [entry.key for entry in region] == [entry.key for entry in engine]
    for mine, theirs in zip(region, engine, strict=True):
        assert torch.equal(mine.mask, theirs.mask)
        assert mine.provenance == theirs.provenance


def test_bool_mask_units_equal_coordinate_units(stack) -> None:
    """A bool mask over the op's full output is the same seed as its coordinates."""

    seed_op = _out(stack, "conv1")
    mask = torch.zeros(tuple(seed_op.shape), dtype=torch.bool)
    for unit in _full(_UNITS):
        mask[unit] = True
    by_mask = stack["conv1"].projective_field.of(mask, within="conv3")
    by_coords = stack["conv1"].projective_field.of(_full(_UNITS), within="conv3")
    assert torch.equal(by_mask[0].mask, by_coords[0].mask)


def test_within_accepts_module_records_and_mixed_lists(stack) -> None:
    """Module records, module addresses and op labels name the same targets."""

    conv2, conv3 = _out(stack, "conv2"), _out(stack, "conv3")
    view = stack["conv1"].projective_field
    by_address = view.of(_full(_UNITS), within=["conv3", "conv2"])
    by_record = view.of(_full(_UNITS), within=[stack.modules["conv3"], conv2.label])
    by_call = view.of(_full(_UNITS), within=(stack.modules["conv2"].calls[0], conv3))
    for other in (by_record, by_call):
        assert [entry.key for entry in other] == [entry.key for entry in by_address]
        for mine, theirs in zip(other, by_address, strict=True):
            assert torch.equal(mine.mask, theirs.mask)
    # Targets come back in execution order, whatever order `within` lists them in.
    assert [f"{entry.key[0]}:{entry.key[1]}" for entry in by_address] == [
        conv2.label,
        conv3.label,
    ]


def test_default_within_is_the_model_outputs_or_inputs(stack) -> None:
    """``within=None``: model outputs for projective, model inputs for receptive."""

    outward = stack["conv1"].projective_field.of(_full(_UNITS))
    assert [f"{entry.key[0]}:{entry.key[1]}" for entry in outward] == [
        op.label for op in stack.output_ops
    ]
    inward = stack["conv2"].receptive_field.of(_full([(3, 3)]))
    assert len(inward) == 1
    assert _row(inward, stack.input_ops[0]).num_selected > 0


def test_results_compose_with_the_selection_algebra(stack) -> None:
    """``&``, ``|``, ``-`` and ``~`` act element-wise on the shared target."""

    conv3 = _out(stack, "conv3")
    view = stack["conv1"].projective_field
    # conv1 (6, 6) reaches conv3 rows/cols 1..3 and (8, 8) reaches 2..4: overlapping, not equal.
    left = view.of(_full([(6, 6)]), within="conv3")
    right = view.of(_full([(8, 8)]), within="conv3")
    a, b = _row(left, conv3).mask, _row(right, conv3).mask
    assert bool((a & b).any()) and not torch.equal(a, b)
    assert torch.equal(_row(left & right, conv3).mask, a & b)
    assert torch.equal(_row(left | right, conv3).mask, a | b)
    assert torch.equal(_row(left - right, conv3).mask, a & ~b)
    assert torch.equal(_row(~left, conv3).mask, ~a)
    # An unresolved selector composes too, resolving against the bound trace.
    corner = tl.units(conv3.label, [(0, 0, 0, 0)])
    assert torch.equal(_row(left | corner, conv3).mask, a | _row(stack[corner], conv3).mask)


def test_do_round_trip_edits_exactly_the_reached_units() -> None:
    """``fork.do(region, zero_ablate())`` zeroes the reached units and nothing else."""

    torch.manual_seed(5)
    model = _Stack()
    trace = tl.trace(
        model, torch.randn(1, 1, 16, 16), capture=tl.options.CaptureOptions(replayable=True)
    )
    try:
        fork = trace.fork()
        region = fork["conv1"].projective_field.of(_full([(6, 6)]), within="conv2")
        target = _out(fork, "conv2")
        mask = _row(region, target).mask
        assert 0 < int(mask.sum()) < mask.numel()
        fork.do(region, tl.zero_ablate())
        edited = fork[target.label].out
        baseline = trace[target.label].out
        assert bool((edited[mask] == 0).all())
        assert torch.equal(edited[~mask], baseline[~mask])
        assert not torch.equal(fork[fork.output_ops[0].label].out, trace.output_ops[0].out)
        # The same region resolved on the parent bridges to a fork explicitly.
        parent_region = trace["conv1"].projective_field.of(_full([(6, 6)]), within="conv2")
        other = trace.fork()
        other.do(parent_region.align_to(other), tl.zero_ablate())
        assert torch.equal(other[target.label].out, edited)
    finally:
        trace.cleanup()


def test_subgraph_of_a_region(stack) -> None:
    """``trace.subgraph`` takes the region like any other ACT selection."""

    region = stack["conv1"].projective_field.of(_full(_UNITS), within=["conv2", "conv3"])
    view = stack.subgraph(region)
    assert set(view.labels) == {_out(stack, "conv2").label, _out(stack, "conv3").label}


def test_gradient_method_is_contained_in_the_geometric_answer(stack) -> None:
    """``method="gradient"`` passes through; its support sits inside the geometry.

    The seed is conv2, whose path to conv3 is linear: a seed before the ReLU can
    have an empty empirical field wherever its pre-activation is negative.
    """

    conv3 = _out(stack, "conv3")
    view = stack["conv2"].projective_field
    units = _full([(3, 3), (3, 4)])
    geometric = _row(view.of(units, within="conv3"), conv3)
    probed = _row(view.of(units, within="conv3", method="gradient"), conv3)
    assert probed.provenance.basis == "gradient"
    assert probed.provenance.field_status == "empirical"
    assert probed.num_selected > 0
    assert not bool((probed.mask & ~geometric.mask).any())


def test_probe_budget_passes_through_and_refuses_typed(stack) -> None:
    """Units times sites over ``probe_budget`` refuses before any probe runs."""

    view = stack["conv1"].projective_field
    with pytest.raises(SelectionError) as excinfo:
        view.of(_full(_UNITS), within=["conv2", "conv3"], method="gradient", probe_budget=3)
    assert excinfo.value.fields["code"] == "influence_probe_budget"
    assert excinfo.value.fields["probes"] == 8
    assert excinfo.value.fields["probe_budget"] == 3


def test_data_dependent_geometry_refuses_typed_and_gradient_answers() -> None:
    """Routing has no per-unit geometry: typed refusal, and the gradient method answers."""

    torch.manual_seed(3)
    trace = _armed(_Routed(), torch.randn(1, 1, 8, 8))
    try:
        view = trace.input_ops[0].projective_field
        with pytest.raises(SelectionError) as excinfo:
            view.of([(0, 0, 4, 4)])
        assert excinfo.value.fields["code"] == "influence_geometry_unavailable"
        assert excinfo.value.fields["field_status"] == "data_dependent"
        assert "method='gradient'" in str(excinfo.value)
        probed = view.of([(0, 0, 4, 4)], method="gradient")
        assert probed[0].num_selected > 0
    finally:
        trace.cleanup()


def test_unknown_region_and_bad_seed_refuse_typed(stack) -> None:
    """A region the trace lacks and a wrong-shape mask refuse as unresolvable."""

    view = stack["conv1"].projective_field
    with pytest.raises(SelectionError) as excinfo:
        view.of(_full(_UNITS), within="no_such_layer")
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "site_not_in_trace"
    with pytest.raises(SelectionError) as excinfo:
        view.of(torch.ones(2, 2, dtype=torch.bool), within="conv2")
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "mask_shape_mismatch"


def test_bad_arguments_raise_value_error(stack) -> None:
    """Closed vocabularies and the units contract are checked before resolution."""

    view = stack["conv1"].projective_field
    with pytest.raises(ValueError, match="method"):
        view.of(_full(_UNITS), method="saliency")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="probe_budget"):
        view.of(_full(_UNITS), probe_budget=0)
    with pytest.raises(ValueError, match="bool mask"):
        view.of(torch.zeros(1, 2, 16, 16))
