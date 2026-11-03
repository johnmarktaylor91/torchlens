"""Unit-level influence ENGINE: geometry, status, gradient probes, refusals.

Exercises the private producer in ``torchlens.receptive_field._influence``
(no public spelling yet) on four fixtures: a three-layer conv stack with
hand-computed hulls in both directions, a conv skip model where the skip
widens the union past the conv-only box and a one-row batch seed stays on
its row, a global-mixing block whose global axis is covered whole, and a
data-dependent routing case that refuses geometry typed and answers under
the gradient method. Every geometric answer is cross-checked against the
union over seed units of the shipped per-unit ``.at()`` lift, and every
gradient answer is contained in the geometric one. Every test here runs well
under a second on the gate workers, so none carries a tier mark.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.nn import functional

import torchlens as tl
from torchlens._selection.graph import _influence
from torchlens._selection.provenance import SelectionProvenance
from torchlens._selection.selection import _join_relation
from torchlens.errors import SelectionError
from torchlens.receptive_field import _influence as influence
from torchlens.receptive_field._types import ReceptiveFieldStatus

# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


class _ConvStack(nn.Module):
    """Three convolutions with stride and padding chosen for hand-computed hulls.

    input (16x16) -> conv1 k3 s1 p1 (16x16) -> relu -> conv2 k3 s2 p1 (8x8)
    -> conv3 k3 s1 p0 (6x6)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.conv2 = nn.Conv2d(1, 1, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(1, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv3(self.conv2(torch.relu(self.conv1(x))))


class _ConvBranchy(nn.Module):
    """Two conv branches of different reach merging into a conv head.

    input -> conv k3 p1 -> relu -\\
                                  add -> head conv k3 p1
    input -> conv k5 p2 -> tanh -/
    """

    def __init__(self) -> None:
        super().__init__()
        self.narrow = nn.Conv2d(1, 1, 3, padding=1)
        self.wide = nn.Conv2d(1, 1, 5, padding=2)
        self.head = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(self.narrow(x))
        b = torch.tanh(self.wide(x))
        return self.head(a + b)


class _GlobalMix(nn.Module):
    """A conv whose output is scaled by its own row mean: the column axis goes global."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y * y.mean(dim=-1, keepdim=True)


class _Attention(nn.Module):
    """Pure self-attention: every axis is pointwise or full, none windowed."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return functional.scaled_dot_product_attention(value, value, value)


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
    trace = _armed(_ConvStack(), torch.randn(1, 1, 16, 16))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def branchy():
    """Armed capture of the conv skip model on a two-row batch."""

    torch.manual_seed(1)
    trace = _armed(_ConvBranchy(), torch.randn(2, 1, 8, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def mixed():
    """Plain capture of the global-mixing block."""

    torch.manual_seed(2)
    trace = tl.trace(_GlobalMix(), torch.randn(1, 1, 8, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def routed():
    """Armed capture of the data-dependent routing model."""

    torch.manual_seed(3)
    trace = _armed(_Routed(), torch.randn(1, 1, 8, 8))
    try:
        yield trace
    finally:
        trace.cleanup()


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _op(trace, prefix: str, index: int = 0):
    """Return the ``index``-th executed op whose label starts with ``prefix``."""

    matches = [op for op in trace._layer_list if op.label.startswith(prefix)]
    assert matches, f"no op labelled {prefix}* in {[op.label for op in trace._layer_list]}"
    return matches[index]


def _entry(resolved, op):
    """Return the resolved row for ``op`` (exactly one must exist)."""

    rows = [entry for entry in resolved if f"{entry.key[0]}:{entry.key[1]}" == op.label]
    assert len(rows) == 1, f"expected one row for {op.label}, got {len(rows)}"
    return rows[0]


def _slices_mask(shape: tuple[int, ...], *slices: slice) -> torch.Tensor:
    """Build a dense bool mask from one slice per axis."""

    mask = torch.zeros(shape, dtype=torch.bool)
    mask[slices] = True
    return mask


def _patch(rows: range, cols: range) -> list[tuple[int, int, int, int]]:
    """Return the full-shape coordinates of a batch-0 channel-0 patch."""

    return [(0, 0, row, col) for row in rows for col in cols]


def _lifted_units(trace, seed_op, units, target, direction: str):
    """Resolve the shipped per-unit ``.at()`` lift at ``target`` for every seed unit."""

    rows = []
    for unit in units:
        if direction == "projective":
            box = seed_op.projective_field.at(unit, target=target)
        elif bool(getattr(target, "is_input", False)):
            box = seed_op.receptive_field.at(unit, input_op=target)
        else:
            box = seed_op.receptive_field.at(unit, source=target)
        rows.append(_entry(trace[box.__selection__()], target))
    return rows


def _union_of_units(trace, seed_op, units, target, direction: str) -> torch.Tensor:
    """OR the shipped per-unit lifts over the seed units at ``target``."""

    folded = torch.zeros(tuple(target.shape), dtype=torch.bool)
    for row in _lifted_units(trace, seed_op, units, target, direction):
        folded |= row.mask
    return folded


def _lifted_relation(trace, seed_op, units, target, direction: str) -> str:
    """Join the shipped per-unit lifts' relations the way the engine folds them."""

    relation = "exact"
    for row in _lifted_units(trace, seed_op, units, target, direction):
        relation = _join_relation(relation, row.provenance.relation)
    return relation


# ---------------------------------------------------------------------------
# Fixture 1: conv stack, hand-computed hulls, both directions.
# ---------------------------------------------------------------------------


def test_receptive_hull_to_input_matches_hand_computation(stack) -> None:
    """A 2x2 seed at conv3 rows/cols 2..3 reaches input rows/cols 2..12."""

    conv3 = _op(stack, "conv2d_", 2)
    seed = tl.units(conv3.label, _patch(range(2, 4), range(2, 4)))
    resolved = stack[_influence(seed, direction="receptive")]
    target = stack.input_ops[0]
    entry = _entry(resolved, target)

    expected = _slices_mask((1, 1, 16, 16), slice(None), slice(None), slice(2, 13), slice(2, 13))
    assert torch.equal(entry.mask, expected)
    assert entry.provenance.relation == "exact"
    assert entry.provenance.basis == "geometric"
    assert entry.provenance.field_status == "exact"
    assert len(list(resolved)) == 1


def test_receptive_hull_to_interior_layers_matches_hand_computation(stack) -> None:
    """The same seed reaches conv2 rows 2..5 and conv1 rows 3..11."""

    conv1 = _op(stack, "conv2d_", 0)
    conv2 = _op(stack, "conv2d_", 1)
    conv3 = _op(stack, "conv2d_", 2)
    seed = tl.units(conv3.label, _patch(range(2, 4), range(2, 4)))
    resolved = stack[_influence(seed, within=[conv1, conv2], direction="receptive")]

    at_conv2 = _entry(resolved, conv2).mask
    at_conv1 = _entry(resolved, conv1).mask
    assert torch.equal(
        at_conv2, _slices_mask((1, 1, 8, 8), slice(None), slice(None), slice(2, 6), slice(2, 6))
    )
    assert torch.equal(
        at_conv1,
        _slices_mask((1, 1, 16, 16), slice(None), slice(None), slice(3, 12), slice(3, 12)),
    )
    assert [f"{entry.key[0]}:{entry.key[1]}" for entry in resolved] == [conv1.label, conv2.label]


def test_projective_hull_from_input_matches_hand_computation(stack) -> None:
    """An input patch at rows/cols 7..8 reaches conv1 6..9, conv2 3..5, conv3 1..5."""

    source = stack.input_ops[0]
    conv1 = _op(stack, "conv2d_", 0)
    conv2 = _op(stack, "conv2d_", 1)
    conv3 = _op(stack, "conv2d_", 2)
    seed = tl.units(source.label, _patch(range(7, 9), range(7, 9)))
    resolved = stack[_influence(seed, within=[conv1, conv2, conv3])]

    expected = {
        conv1.label: _slices_mask(
            (1, 1, 16, 16), slice(None), slice(None), slice(6, 10), slice(6, 10)
        ),
        conv2.label: _slices_mask((1, 1, 8, 8), slice(None), slice(None), slice(3, 6), slice(3, 6)),
        conv3.label: _slices_mask((1, 1, 6, 6), slice(None), slice(None), slice(1, 6), slice(1, 6)),
    }
    units = [(7, 7), (7, 8), (8, 7), (8, 8)]
    for op in (conv1, conv2, conv3):
        entry = _entry(resolved, op)
        assert torch.equal(entry.mask, expected[op.label]), op.label
        # The engine inherits the shipped grading: the relation equals the join of
        # the per-unit lifts' relations (the stride-2 transposition is graded
        # upper_bound by the field engine, so no exactness is invented here).
        assert entry.provenance.relation == _lifted_relation(stack, source, units, op, "projective")
        assert entry.provenance.field_status in {"exact", "upper_bound"}


def test_projective_default_target_is_the_model_output(stack) -> None:
    """``within=None`` projective means the model outputs, as the view default."""

    source = stack.input_ops[0]
    seed = tl.units(source.label, _patch(range(7, 9), range(7, 9)))
    resolved = stack[_influence(seed)]
    output = stack.output_ops[0]
    entry = _entry(resolved, output)
    assert torch.equal(
        entry.mask, _slices_mask((1, 1, 6, 6), slice(None), slice(None), slice(1, 6), slice(1, 6))
    )


@pytest.mark.parametrize("direction", ["receptive", "projective"])
def test_union_of_per_unit_lifts_equals_the_set_answer(stack, direction: str) -> None:
    """The engine's set answer is the union over seed units of the shipped lift."""

    conv1 = _op(stack, "conv2d_", 0)
    conv2 = _op(stack, "conv2d_", 1)
    conv3 = _op(stack, "conv2d_", 2)
    if direction == "receptive":
        seed_op, units = conv3, [(2, 2), (2, 3), (3, 2), (3, 3)]
        targets = [stack.input_ops[0], conv1, conv2]
    else:
        seed_op, units = stack.input_ops[0], [(7, 7), (7, 8), (8, 7), (8, 8)]
        targets = [conv1, conv2, conv3]
    seed = tl.units(seed_op.label, [(0, 0, *unit) for unit in units])
    resolved = stack[_influence(seed, within=targets, direction=direction)]
    for target in targets:
        entry = _entry(resolved, target)
        expected = _union_of_units(stack, seed_op, units, target, direction)
        assert torch.equal(entry.mask, expected), target.label
        assert entry.provenance.relation == _lifted_relation(
            stack, seed_op, units, target, direction
        ), target.label


# ---------------------------------------------------------------------------
# Fixture 2: conv skip model; the wide branch widens the union; batch stays put.
# ---------------------------------------------------------------------------


def test_skip_branch_widens_the_union_past_the_narrow_box(branchy) -> None:
    """One input unit reaches head rows 1..7 through the k5 branch, 2..6 through k3 alone."""

    source = branchy.input_ops[0]
    head = _op(branchy, "conv2d_", 2)
    relu = _op(branchy, "relu_")
    seed = tl.units(source.label, [(0, 0, 4, 4)])
    resolved = branchy[_influence(seed, within=[relu, head])]

    at_head = _entry(resolved, head).mask
    at_relu = _entry(resolved, relu).mask
    assert torch.equal(
        at_head, _slices_mask((2, 1, 8, 8), slice(0, 1), slice(None), slice(1, 8), slice(1, 8))
    )
    assert torch.equal(
        at_relu, _slices_mask((2, 1, 8, 8), slice(0, 1), slice(None), slice(3, 6), slice(3, 6))
    )


def test_batch_row_seed_stays_on_its_row(branchy) -> None:
    """Batch is pointwise: a batch-1 seed never touches batch 0 at any target."""

    source = branchy.input_ops[0]
    seed = tl.units(source.label, [(1, 0, 0, 0)])
    resolved = branchy[_influence(seed)]
    entry = _entry(resolved, branchy.output_ops[0])
    mask = entry.mask
    assert not mask[0].any()
    assert torch.equal(mask[1], _slices_mask((1, 8, 8), slice(None), slice(0, 4), slice(0, 4)))


def test_two_seed_sites_fold_per_target(branchy) -> None:
    """Seeds at two sites union per target through the join table."""

    source = branchy.input_ops[0]
    relu = _op(branchy, "relu_")
    head = _op(branchy, "conv2d_", 2)
    seed = tl.units(source.label, [(0, 0, 0, 0)]) | tl.units(relu.label, [(0, 0, 7, 7)])
    resolved = branchy[_influence(seed, within=head)]
    mask = _entry(resolved, head).mask
    expected = _slices_mask((2, 1, 8, 8), slice(0, 1), slice(None), slice(0, 4), slice(0, 4))
    expected |= _slices_mask((2, 1, 8, 8), slice(0, 1), slice(None), slice(6, 8), slice(6, 8))
    assert torch.equal(mask, expected)


def test_seed_at_the_target_site_is_the_identity(branchy) -> None:
    """When a target is a seed site, its own mask comes back unchanged."""

    relu = _op(branchy, "relu_")
    seed = tl.units(relu.label, [(0, 0, 2, 2)])
    resolved = branchy[_influence(seed, within=relu)]
    entry = _entry(resolved, relu)
    assert torch.equal(entry.mask, _entry(branchy[seed], relu).mask)
    assert entry.provenance.relation == "exact"


# ---------------------------------------------------------------------------
# Fixture 3: a global-mixing block; the full axis covers the whole extent.
# ---------------------------------------------------------------------------
#
# The field engine's ``whole_input`` status means every non-pointwise axis is
# full, which leaves no windowed axis to seed, and per-unit geometry refuses
# such descriptors (pure attention below). A block with one global axis beside
# a windowed one is graded ``exact`` with the global axis covered whole.


def test_global_axis_is_covered_whole_beside_a_windowed_axis(mixed) -> None:
    """A row-mean broadcast makes every column reachable while rows stay windowed."""

    output = mixed.output_ops[0]
    seed = tl.units(output.label, _patch(range(3, 5), range(3, 5)))
    entry = _entry(mixed[_influence(seed, direction="receptive")], mixed.input_ops[0])
    assert torch.equal(
        entry.mask, _slices_mask((1, 1, 8, 8), slice(None), slice(None), slice(2, 6), slice(None))
    )
    assert entry.provenance.relation == "exact"
    assert entry.provenance.field_status == "exact"


def test_whole_site_seed_on_a_global_block_is_a_whole_site_mask(mixed) -> None:
    """Seeding the whole output site reaches the whole input site."""

    entry = _entry(
        mixed[_influence(mixed.output_ops[0].label, direction="receptive")],
        mixed.input_ops[0],
    )
    assert bool(entry.mask.all())
    assert entry.provenance.relation == "exact"
    assert entry.provenance.field_status == "exact"


def test_status_fold_is_worst_of_in_the_documented_order() -> None:
    """EXACT < WHOLE_INPUT < UPPER_BOUND; an empty fold reads ``none``."""

    worst = influence._worst_status
    assert worst(()) == "none"
    assert worst((ReceptiveFieldStatus.EXACT,)) == "exact"
    assert worst((ReceptiveFieldStatus.EXACT, ReceptiveFieldStatus.WHOLE_INPUT)) == "whole_input"
    assert (
        worst((ReceptiveFieldStatus.WHOLE_INPUT, ReceptiveFieldStatus.UPPER_BOUND)) == "upper_bound"
    )


# ---------------------------------------------------------------------------
# Fixture 4 and refusals: geometry refuses typed; gradient answers.
# ---------------------------------------------------------------------------


def test_data_dependent_routing_refuses_geometry_typed(routed) -> None:
    """A data-dependent rule on the path refuses with the geometry reason."""

    source = routed.input_ops[0]
    seed = tl.units(source.label, [(0, 0, 4, 4)])
    with pytest.raises(SelectionError) as excinfo:
        routed[_influence(seed)]
    assert excinfo.value.fields["code"] == "influence_geometry_unavailable"
    assert excinfo.value.fields["reason"] == "influence_geometry_unavailable"
    assert excinfo.value.fields["field_status"] == "data_dependent"
    assert "method='gradient'" in str(excinfo.value)


def test_attention_without_windowed_axes_refuses_geometry_typed() -> None:
    """Pure attention has no windowed axis; geometry refuses typed, not silently."""

    torch.manual_seed(4)
    trace = tl.trace(_Attention(), torch.randn(2, 2, 3, 2))
    try:
        seed = tl.units(trace.input_ops[0].label, [(0, 0, 0, 0)])
        with pytest.raises(SelectionError) as excinfo:
            trace[_influence(seed)]
        assert excinfo.value.fields["code"] == "influence_geometry_unavailable"
    finally:
        trace.cleanup()


def test_gradient_method_routes_through_the_index_list(routed) -> None:
    """Gradient probes OR into the rows the index list sends the conv rows to."""

    source = routed.input_ops[0]
    seed = tl.units(source.label, [(0, 0, 4, 4)])
    entry = _entry(routed[_influence(seed, method="gradient")], routed.output_ops[0])
    expected = torch.zeros((1, 1, 8, 8), dtype=torch.bool)
    for position, row in enumerate(_PERM):
        if 3 <= row <= 5:
            expected[0, 0, position, 3:6] = True
    assert torch.equal(entry.mask, expected)
    assert entry.provenance.relation == "exact"
    assert entry.provenance.basis == "gradient"
    assert entry.provenance.field_status == "empirical"
    assert "probes=1" in entry.provenance.source


@pytest.mark.parametrize("direction", ["receptive", "projective"])
def test_gradient_answer_is_contained_in_the_geometric_answer(stack, direction: str) -> None:
    """Empirical support never leaves the geometric hull (zero-tolerance containment)."""

    conv3 = _op(stack, "conv2d_", 2)
    if direction == "receptive":
        seed = tl.units(conv3.label, _patch(range(2, 4), range(2, 4)))
        target = stack.input_ops[0]
    else:
        seed = tl.units(stack.input_ops[0].label, _patch(range(7, 9), range(7, 9)))
        target = stack.output_ops[0]
    geometric = _entry(stack[_influence(seed, direction=direction)], target).mask
    gradient = _entry(stack[_influence(seed, direction=direction, method="gradient")], target).mask
    assert bool(gradient.any())
    assert not bool((gradient & ~geometric).any())


def test_gradient_batch_row_seed_stays_on_its_row(branchy) -> None:
    """Gradient probes report no cross-batch influence for a pointwise batch."""

    seed = tl.units(branchy.input_ops[0].label, [(1, 0, 4, 4)])
    entry = _entry(branchy[_influence(seed, method="gradient")], branchy.output_ops[0])
    assert not entry.mask[0].any()
    assert bool(entry.mask[1].any())
    assert "cross_batch_influence=False" in entry.provenance.source


def test_probe_budget_refuses_before_any_probe(stack) -> None:
    """Seed units times target sites over the budget refuses typed."""

    conv3 = _op(stack, "conv2d_", 2)
    seed = tl.units(conv3.label, _patch(range(0, 6), range(0, 6)))
    with pytest.raises(SelectionError) as excinfo:
        stack[_influence(seed, direction="receptive", method="gradient", probe_budget=4)]
    assert excinfo.value.fields["code"] == "influence_probe_budget"
    assert excinfo.value.fields["reason"] == "influence_probe_budget"
    assert excinfo.value.fields["probes"] == 36
    assert excinfo.value.fields["probe_budget"] == 4


def test_typed_disclosure_survives_complement_and_agreeing_composition(stack) -> None:
    """``~`` keeps basis and field status; ``|`` keeps them only when both sides agree."""

    seed = tl.units(stack.input_ops[0].label, _patch(range(7, 9), range(7, 9)))
    target = stack.output_ops[0]
    geometric = stack[_influence(seed)]
    gradient = stack[_influence(seed, method="gradient")]
    status = _entry(geometric, target).provenance.field_status
    flipped = _entry(~geometric, target).provenance
    assert (flipped.basis, flipped.field_status) == ("geometric", status)
    agreeing = _entry(geometric | geometric, target).provenance
    assert (agreeing.basis, agreeing.field_status) == ("geometric", status)
    mixed = _entry(geometric | gradient, target).provenance
    assert (mixed.basis, mixed.field_status) == (None, None)


def test_provenance_basis_and_field_status_vocabularies_are_closed() -> None:
    """An off-vocabulary basis or field status cannot be constructed."""

    with pytest.raises(ValueError, match="basis must be one of"):
        SelectionProvenance(relation="exact", basis="sampled")
    with pytest.raises(ValueError, match="field_status must be one of"):
        SelectionProvenance(relation="exact", field_status="approximate")


# ---------------------------------------------------------------------------
# Negatives and algebra.
# ---------------------------------------------------------------------------


def test_empty_seed_yields_zero_masks_at_every_target(stack) -> None:
    """An all-empty seed is disclosure: zero-mask exact entries, family kept."""

    conv1 = _op(stack, "conv2d_", 0)
    conv2 = _op(stack, "conv2d_", 1)
    conv3 = _op(stack, "conv2d_", 2)
    empty = tl.units(conv3.label, [(0, 0, 0, 0)]) - tl.units(conv3.label, [(0, 0, 0, 0)])
    resolved = stack[_influence(empty, within=[conv1, conv2], direction="receptive")]
    rows = list(resolved)
    assert [f"{row.key[0]}:{row.key[1]}" for row in rows] == [conv1.label, conv2.label]
    for row in rows:
        assert not row.mask.any()
        assert row.provenance.relation == "exact"
        assert row.provenance.field_status == "none"


def test_param_seed_refuses_kind_incompatible(stack) -> None:
    """A PARAM seed is not an ACT region."""

    with pytest.raises(SelectionError) as excinfo:
        _influence(tl.param("conv1.weight"))
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"


def test_unknown_target_refuses_site_not_in_trace(stack) -> None:
    """A ``within`` label the trace never executed refuses typed."""

    conv3 = _op(stack, "conv2d_", 2)
    seed = tl.units(conv3.label, [(0, 0, 0, 0)])
    with pytest.raises(SelectionError) as excinfo:
        stack[_influence(seed, within="conv2d_9_9:1", direction="receptive")]
    assert excinfo.value.fields["reason"] == "site_not_in_trace"


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"direction": "sideways"}, "direction"),
        ({"method": "auto"}, "method"),
        ({"probe_budget": 0}, "probe_budget"),
        ({"probe_budget": True}, "probe_budget"),
    ],
)
def test_bad_parameters_refuse_at_construction(kwargs, fragment: str) -> None:
    """Direction, method and budget are validated before any trace is touched."""

    with pytest.raises(ValueError, match=fragment):
        _influence("relu_1_2:1", **kwargs)


def test_checkerboard_seed_over_budget_collapses_to_upper_bound(stack) -> None:
    """A ragged seed past the rectangle budget answers from its bounding box."""

    conv1 = _op(stack, "conv2d_", 0)
    board = torch.zeros((1, 1, 16, 16), dtype=torch.bool)
    board[0, 0, ::2, ::2] = True
    board[0, 0, 1::2, 1::2] = True
    assert int(board.sum()) > influence._RECTANGLE_BUDGET
    seed = tl.units(conv1.label, board)
    resolved = stack[_influence(seed, direction="receptive")]
    entry = _entry(resolved, stack.input_ops[0])
    assert bool(entry.mask.all())
    assert entry.provenance.relation == "upper_bound"
    flipped = _entry(stack[~_influence(seed, direction="receptive")], stack.input_ops[0])
    assert flipped.provenance.relation == "lower_bound"


def test_rectangle_decomposition_is_exact_under_budget() -> None:
    """A dense L-shaped mask decomposes into a few exact rectangles."""

    from torchlens._selection.mask import _mask_from_dense

    dense = torch.zeros((6, 6), dtype=torch.bool)
    dense[0:2, :] = True
    dense[:, 0:2] = True
    rectangles, exact = influence._rectangles(_mask_from_dense((6, 6), dense))
    assert exact
    rebuilt = torch.zeros((6, 6), dtype=torch.bool)
    for rectangle in rectangles:
        rebuilt[tuple(slice(start, stop) for start, stop in rectangle)] = True
    assert torch.equal(rebuilt, dense)
    assert len(rectangles) <= 3


def test_algebra_with_between_and_units(stack) -> None:
    """The result composes like any other selection: ``&`` restricts, ``|`` adds."""

    conv1 = _op(stack, "conv2d_", 0)
    conv2 = _op(stack, "conv2d_", 1)
    conv3 = _op(stack, "conv2d_", 2)
    seed = tl.units(conv3.label, _patch(range(2, 4), range(2, 4)))
    reach = _influence(seed, within=[conv1, conv2], direction="receptive")

    restricted = stack[reach & tl.between(conv2.label, conv3.label)]
    assert [f"{row.key[0]}:{row.key[1]}" for row in restricted] == [conv2.label]
    assert torch.equal(
        _entry(restricted, conv2).mask,
        _slices_mask((1, 1, 8, 8), slice(None), slice(None), slice(2, 6), slice(2, 6)),
    )

    added = stack[reach | tl.units(conv2.label, [(0, 0, 0, 0)])]
    mask = _entry(added, conv2).mask
    assert bool(mask[0, 0, 0, 0])
    assert int(mask.sum()) == 17


def test_term_repr_is_constructor_shaped(stack) -> None:
    """The private term discloses its arguments compactly."""

    conv3 = _op(stack, "conv2d_", 2)
    selector = _influence(conv3.label, within=["conv2d_1_1:1"], direction="receptive")
    text = repr(selector)
    assert "influence(" in text
    assert "direction='receptive'" in text
    assert "method='geometric'" in text
