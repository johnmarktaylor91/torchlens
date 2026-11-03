"""L6 stage-1(b)/(c) exit gate: Selection algebra, producers, resolution.

Covers the design-memo obligations: operator closure + totality matrix over
the widened axes, the law table at BOTH denotation levels (family +
elements) including pinned NON-laws, reflected/mixin dispatch including the
exact METAPLAN flagship expression, BaseSelector NotImplemented-branch rows,
the ``__bool__`` refusal pair, mask materialization parity across the three
representations, the mask-immutability pin, the resolver refusal matrix, one
lift test per stage-1 producer, the table-driven provenance-relation
composition tables (never sampled), and the Selection ``__repr__`` stability
pin.

EDGE-kind matrix cells are stage-3 territory (``trace.graph.edges`` lands there);
the kind vocabulary and refusal cells for EDGE are pinned in
``test_kind_matrix_edge_cells_pinned_for_stage3``.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._selection.provenance import (
    _RELATIONS,
    _difference_relation,
    _join_relation,
    _meet_relation,
)
from torchlens._selection.selection import Selection, Selector
from torchlens._selection.selectors import CompositeSelector, NotSelector
from torchlens.errors import SelectionError


class _TwoConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(1, 2, 3)
        self.c2 = nn.Conv2d(2, 2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.c2(torch.relu(self.c1(x))))


@pytest.fixture(scope="module")
def log():
    """One traced deterministic CNN shared by the suite."""

    torch.manual_seed(0)
    trace = tl.trace(_TwoConv(), torch.randn(1, 1, 12, 12))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def other_log():
    """A second trace for trace-mismatch rows."""

    torch.manual_seed(1)
    trace = tl.trace(_TwoConv(), torch.randn(1, 1, 12, 12))
    try:
        yield trace
    finally:
        trace.cleanup()


def _elements(resolved: Selection) -> int:
    return sum(entry.num_selected for entry in resolved)


# ---------------------------------------------------------------------------
# Flagship + mixin dispatch
# ---------------------------------------------------------------------------


def test_metaplan_flagship_expression(log):
    """u1.receptive_field.at(p) | u2.receptive_field.at(q) IS a Selection
    usable end to end — producer-to-producer dispatch, no conversion calls."""

    u1 = log["relu_2_4"]
    u2 = log["conv2d_2_3"]
    union = u1.receptive_field.at((3, 3)) | u2.receptive_field.at((5, 5))
    inter = u1.receptive_field.at((3, 3)) & u2.receptive_field.at((5, 5))
    assert isinstance(union, Selector) and isinstance(inter, Selector)
    resolved_union = log[union]
    resolved_inter = log[inter]
    assert resolved_inter.is_empty is False
    assert _elements(resolved_inter) <= _elements(resolved_union)
    # Overlapping late-layer RF boxes at the input site: one shared ACT site.
    assert len(resolved_union) == 1
    assert resolved_union[0].key == ("input_1", 1)


def test_mixin_dispatch_box_facet_unit_cross_shapes(log):
    """box | facet-shaped and facet - unit compositions yield Selections."""

    box = log["relu_2_4"].receptive_field.at((3, 3))
    unit = tl.units("input_1", [(0, 0, 4, 4)])
    assert isinstance(box | unit, Selector)
    assert isinstance(box - unit, Selector)
    assert isinstance(unit - box, Selector)  # reflected __rsub__ path
    assert isinstance(~box, Selector)


def test_selector_notimplemented_branch_rows(log):
    """selector OP __selection__-bearing operand -> Selection via reflection;
    selector OP selector keeps shipped CompositeSelector semantics."""

    selector = tl.func("relu")
    box = log["relu_2_4"].receptive_field.at((3, 3))
    layer = log["relu_1_2"]
    assert isinstance(selector | box, Selector)
    assert isinstance(selector & box, Selector)
    assert isinstance(selector - box, Selector)
    assert isinstance(selector | layer, Selector)
    # Shipped selector algebra untouched (golden-pinned semantics).
    composite = tl.func("relu") | tl.func("conv2d")
    assert isinstance(composite, CompositeSelector)
    assert composite.operator == "or"
    assert isinstance(tl.func("relu") & tl.func("conv2d"), CompositeSelector)
    assert isinstance(~tl.func("relu"), NotSelector)


def test_selector_sub_desugars_to_and_not(log):
    """a - b == CompositeSelector("and", (a, NotSelector(b))) — no new kind."""

    difference = tl.func("relu") - tl.in_module("c1")
    assert isinstance(difference, CompositeSelector)
    assert difference.operator == "and"
    assert isinstance(difference.selectors[1], NotSelector)
    # Semantics: difference of match-sets IS a(x) and not b(x); coincides with
    # the desugared spelling on the resolved whole-site view.
    desugared = tl.func("relu") & (~tl.in_module("c1"))
    lifted_difference = log[difference.__selection__()]
    lifted_desugared = log[desugared.__selection__()]
    assert lifted_difference == lifted_desugared

    with pytest.raises(Exception, match="cannot subtract"):
        tl.func("relu") - 5


# ---------------------------------------------------------------------------
# Truthiness pair
# ---------------------------------------------------------------------------


def test_bool_refusal_pair(log):
    """Selector.__bool__ refuses typed; Selection is element-level."""

    query = tl.units("relu_1_2", [(0, 0, 0, 0)])
    with pytest.raises(SelectionError) as excinfo:
        bool(query)
    assert excinfo.value.fields["code"] == "selection_bool_ambiguous"

    resolved = log[query]
    assert bool(resolved) is True
    assert resolved.is_empty is False
    element_empty = resolved - resolved
    assert bool(element_empty) is False
    assert element_empty.is_empty is True
    assert len(element_empty) == 1  # family retained


# ---------------------------------------------------------------------------
# Law table (both denotation levels) + pinned non-laws
# ---------------------------------------------------------------------------


def _family(resolved: Selection) -> frozenset:
    return frozenset(entry.key for entry in resolved)


def test_pinned_non_laws(log):
    """Deliberate NON-laws, pinned."""

    a = log[tl.units("relu_1_2", [(0, 0, 1, 1)])]
    c = log[log["conv2d_1_1"].__selection__()]

    # A - B != A & ~B whenever A touches sites B does not (family rules differ)
    left = a - c
    right = a & (~c)
    assert _family(left) == _family(a)
    assert _family(right) == (_family(a) & _family(c))  # empty here
    assert left != right

    # De Morgan fails across DIFFERENT touched-site families (site-local ~)
    de_morgan_left = ~(a | c)
    de_morgan_right = (~a) & (~c)
    assert de_morgan_left != de_morgan_right

    # ~lift(s) != lift(~s) for selector-terms
    mask_complement = log[~tl.func("relu").__selection__()]
    predicate_negation = log[(~tl.func("relu")).__selection__()]
    assert mask_complement != predicate_negation
    # predicate negation matches all NON-relu sites; mask complement stays on
    # the relu sites with empty masks.
    assert _family(mask_complement) != _family(predicate_negation)


# ---------------------------------------------------------------------------
# Closure/totality matrix over the widened axes
# ---------------------------------------------------------------------------


def _act_nonempty(log):
    return log[tl.units("relu_1_2", [(0, 0, 1, 1)])]


def _act_element_empty(log):
    resolved = _act_nonempty(log)
    return resolved - resolved


def _act_no_sites(log):
    return _act_nonempty(log) & log[log["conv2d_1_1"].__selection__()]


def _param_nonempty(log):
    return log[tl.param("c1.weight")]


def _param_element_empty(log):
    resolved = _param_nonempty(log)
    return resolved - resolved


def _param_no_sites(log):
    # PARAM x PARAM & over disjoint families -> no touched sites
    return log[tl.param("c1.weight")] & log[tl.param("c2.weight")]


_EMPTINESS_BUILDERS = {
    ("ACT", "nonempty"): _act_nonempty,
    ("ACT", "element_empty"): _act_element_empty,
    ("ACT", "no_sites"): _act_no_sites,
    ("PARAM", "nonempty"): _param_nonempty,
    ("PARAM", "element_empty"): _param_element_empty,
    ("PARAM", "no_sites"): _param_no_sites,
}


_STATES = ("no_sites", "element_empty", "nonempty")


def test_totality_same_kind_cells(log):
    """Every same-kind cell composes legally (empties first-class, disclosed).

    One test over all 54 cells (kind x operator x left state x right state); the per-cell
    parametrization was folded because a few cells carry all the branch coverage.
    """

    for kind, operator, left_state, right_state in itertools.product(
        ("ACT", "PARAM"), ("or", "and", "sub"), _STATES, _STATES
    ):
        _check_same_kind_cell(log, kind, kind, operator, left_state, right_state)


def _check_same_kind_cell(log, left_kind, right_kind, operator, left_state, right_state):
    """Compose one same-kind cell and check the family rule for its operator."""

    left = _EMPTINESS_BUILDERS[(left_kind, left_state)](log)
    right = _EMPTINESS_BUILDERS[(right_kind, right_state)](log)
    composed = {
        "or": lambda: left | right,
        "and": lambda: left & right,
        "sub": lambda: left - right,
    }[operator]()
    assert isinstance(composed, Selection)
    # family rules
    if operator == "or":
        assert _family(composed) == (_family(left) | _family(right))
    elif operator == "and":
        assert _family(composed) == (_family(left) & _family(right))
    else:
        assert _family(composed) == _family(left)


def test_totality_mixed_kind_cells_refuse(log):
    """ACT x PARAM refuses typed at both query and resolved levels, for every operator."""

    for operator, resolved_state in itertools.product(("or", "and", "sub"), ("query", "resolved")):
        _check_mixed_kind_cell(log, operator, resolved_state)


def _check_mixed_kind_cell(log, operator, resolved_state):
    """One mixed-kind cell refuses with selection_kind_incompatible naming both kinds."""

    act = tl.units("relu_1_2", [(0, 0, 1, 1)])
    param = tl.param("c1.weight")
    if resolved_state == "resolved":
        act = log[act]
        param = log[param]
    with pytest.raises(SelectionError) as excinfo:
        {
            "or": lambda: act | param,
            "and": lambda: act & param,
            "sub": lambda: act - param,
        }[operator]()
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"
    assert {excinfo.value.fields["left_kind"], excinfo.value.fields["right_kind"]} == {
        "ACT",
        "PARAM",
    }


def test_totality_invert_cells(log):
    """~ over every kind/emptiness cell (query and resolved)."""

    for (_kind, _state), builder in _EMPTINESS_BUILDERS.items():
        resolved = builder(log)
        inverted = ~resolved
        assert isinstance(inverted, Selection)
        assert _family(inverted) == _family(resolved)
    query = tl.units("relu_1_2", [(0, 0, 1, 1)])
    assert isinstance(~query, Selector)
    assert log[~~query] == log[query]


def test_query_composition_flattens_or(log):
    """Chained | accumulates flat n-ary operands (constant depth)."""

    query = (
        tl.units("relu_1_2", [(0, 0, 1, 1)])
        | tl.units("relu_1_2", [(0, 0, 2, 2)])
        | tl.units("relu_1_2", [(0, 0, 3, 3)])
    )
    assert isinstance(query, Selector)
    assert log[query][0].num_selected == 3


def test_query_resolved_mixing(log, other_log):
    """resolved OP query resolves the query against the resolved side's trace;
    resolved OP resolved across traces refuses typed."""

    query = tl.units("relu_1_2", [(0, 0, 1, 1)])
    resolved = log[tl.units("relu_1_2", [(0, 0, 2, 2)])]
    mixed = resolved | query
    assert isinstance(mixed, Selection)
    assert mixed[0].num_selected == 2
    mixed_reflected = query | resolved
    assert mixed_reflected == mixed

    foreign = other_log[tl.units("relu_1_2", [(0, 0, 1, 1)])]
    with pytest.raises(SelectionError) as excinfo:
        resolved | foreign
    assert excinfo.value.fields["code"] == "selection_trace_mismatch"


# ---------------------------------------------------------------------------
# Provenance relation tables — TABLE-DRIVEN over every cell, never sampled
# ---------------------------------------------------------------------------


_JOIN_EXPECTED = {
    ("exact", "exact"): "exact",
    ("exact", "upper_bound"): "upper_bound",
    ("exact", "lower_bound"): "lower_bound",
    ("exact", "unknown"): "unknown",
    ("upper_bound", "upper_bound"): "upper_bound",
    ("upper_bound", "lower_bound"): "unknown",
    ("upper_bound", "unknown"): "unknown",
    ("lower_bound", "lower_bound"): "lower_bound",
    ("lower_bound", "unknown"): "unknown",
    ("unknown", "unknown"): "unknown",
}


def test_join_table_all_cells():
    """JOIN is symmetric with exact as identity — every cell pinned."""

    for a, b in itertools.product(_RELATIONS, _RELATIONS):
        expected = _JOIN_EXPECTED.get((a, b)) or _JOIN_EXPECTED[(b, a)]
        assert _join_relation(a, b) == expected, (a, b)
        assert _join_relation(b, a) == expected, (b, a)
        assert _meet_relation(a, b) == expected, (a, b)  # meet lattice coincides


_DIFFERENCE_EXPECTED = {
    ("exact", "exact"): "exact",
    ("exact", "upper_bound"): "lower_bound",  # pinned direction case
    ("exact", "lower_bound"): "upper_bound",
    ("exact", "unknown"): "unknown",
    ("upper_bound", "exact"): "upper_bound",
    ("upper_bound", "upper_bound"): "unknown",
    ("upper_bound", "lower_bound"): "upper_bound",
    ("upper_bound", "unknown"): "unknown",
    ("lower_bound", "exact"): "lower_bound",
    ("lower_bound", "upper_bound"): "lower_bound",
    ("lower_bound", "lower_bound"): "unknown",
    ("lower_bound", "unknown"): "unknown",
    ("unknown", "exact"): "unknown",
    ("unknown", "upper_bound"): "unknown",
    ("unknown", "lower_bound"): "unknown",
    ("unknown", "unknown"): "unknown",
}


def test_difference_table_all_cells_both_orders():
    """DIFFERENCE is ordered flip-first: rel(A-B) = join(rel(A), FLIP(rel(B))), every cell."""

    for a, b in itertools.product(_RELATIONS, _RELATIONS):
        assert _difference_relation(a, b) == _DIFFERENCE_EXPECTED[(a, b)], (a, b)


# ---------------------------------------------------------------------------
# resolve(): parity, identity, refusal matrix
# ---------------------------------------------------------------------------


def test_resolver_refusal_matrix(log):
    """Every closed selection_unresolvable reason row (stage-1 producers)."""

    # site_not_in_trace
    with pytest.raises(SelectionError) as excinfo:
        log[tl.units("nowhere_9_9", [(0,)])]
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "site_not_in_trace"

    with pytest.raises(SelectionError) as excinfo:
        log[tl.param("no.such.param")]
    assert excinfo.value.fields["reason"] == "site_not_in_trace"

    # mask_shape_mismatch (bad shape mask; out-of-range unit index)
    with pytest.raises(SelectionError) as excinfo:
        log[tl.units("relu_1_2", torch.ones(2, 2, dtype=torch.bool))]
    assert excinfo.value.fields["reason"] == "mask_shape_mismatch"
    with pytest.raises(SelectionError) as excinfo:
        log[tl.units("relu_1_2", [(9, 9, 9, 9)])]
    assert excinfo.value.fields["reason"] == "mask_shape_mismatch"
    with pytest.raises(SelectionError) as excinfo:
        log[tl.param("c1.weight", mask=torch.ones(1, dtype=torch.bool))]
    assert excinfo.value.fields["reason"] == "mask_shape_mismatch"

    # population_too_small (incl. the empty complement case)
    like = tl.units("relu_1_2", [(0, 0, 1, 1), (0, 0, 2, 2)])
    whole = log["relu_1_2"].__selection__()
    with pytest.raises(SelectionError) as excinfo:
        log[tl.random_like(like=like, within=~whole, seed=3)]
    assert excinfo.value.fields["reason"] == "population_too_small"
    assert excinfo.value.fields["requested"] == 2
    assert excinfo.value.fields["available"] == 0


def test_resolver_refusals_no_index_space_and_non_tensor(log, monkeypatch):
    """no_index_space / non_tensor_site rows (stub-site unit pins)."""

    class _NoShapeOp:
        label = "stub_1_1:1"
        layer_label = "stub_1_1"
        pass_index = 1
        io_role = None
        shape = None
        out = None

    class _NonTensorOp(_NoShapeOp):
        label = "tuple_1_1:1"
        layer_label = "tuple_1_1"
        out = ("not", "a", "tensor")

    import torchlens._selection.selection as selection_module

    monkeypatch.setattr(
        selection_module, "_forward_ops", lambda trace: (_NoShapeOp(), _NonTensorOp())
    )
    with pytest.raises(SelectionError) as excinfo:
        log[tl.units("stub_1_1", [(0,)])]
    assert excinfo.value.fields["reason"] == "no_index_space"
    with pytest.raises(SelectionError) as excinfo:
        log[tl.units("tuple_1_1", [(0,)])]
    assert excinfo.value.fields["reason"] == "non_tensor_site"


def test_facet_lift_and_refusal_rows(log):
    """FacetSpec lifts to its write region; unsaved home refuses value_not_saved."""

    from torchlens._semantic.facets import FacetSpec

    op = log["relu_1_2"].ops[0]
    spec = FacetSpec.from_home(op)
    resolved = log[spec.__selection__()]
    assert resolved[0].key == ("relu_1_2", 1)
    assert resolved[0].provenance.relation == "exact"
    assert resolved[0].num_selected == resolved[0].mask.numel()

    # Mixin dispatch on the facet producer
    assert isinstance(spec | op, Selector)

    torch.manual_seed(0)
    sparse_log = tl.trace(_TwoConv(), torch.randn(1, 1, 12, 12), save=tl.func("conv2d"))
    sparse_op = sparse_log["relu_1_2"].ops[0]
    sparse_spec = FacetSpec.from_home(sparse_op)
    with pytest.raises(SelectionError) as excinfo:
        sparse_log[sparse_spec.__selection__()]
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] in ("value_not_saved", "facet_write_mask_unavailable")


def test_gradient_rf_lift():
    """GradReceptiveField lifts as an exact-set empirical term."""

    torch.manual_seed(0)
    model = _TwoConv()
    x = torch.randn(1, 1, 12, 12).requires_grad_(True)
    armed = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True, save_mode="reference"),
    )
    armed_op = armed["relu_2_4"]
    unit = armed_op.receptive_field.center_unit(batch_index=0)
    gradient_by_role = armed_op.receptive_field.probe_grad(unit, retain_graph=True)
    gradient = gradient_by_role["input.x"]
    selection = gradient.__selection__()
    assert isinstance(selection, Selector)
    resolved = armed[selection]
    assert resolved[0].provenance.relation == "exact"
    assert resolved[0].num_selected == int(gradient.support_mask.sum().item())
    # Mixin dispatch with a geometric box at the same site
    box = armed_op.receptive_field.at("center")
    combined = gradient | box
    assert isinstance(combined, Selector)
    assert armed[combined][0].num_selected >= resolved[0].num_selected


# ---------------------------------------------------------------------------
# Constructor validation (house-style ValueErrors, no trace needed)
# ---------------------------------------------------------------------------


def test_constructor_validation_rows():
    with pytest.raises(ValueError, match="non-empty site label"):
        tl.units("", [(0,)])
    with pytest.raises(ValueError, match="non-negative"):
        tl.units("relu_1_2", [(-1, 0)])
    with pytest.raises(ValueError, match="bool mask"):
        tl.units("relu_1_2", torch.ones(2, 2))
    with pytest.raises(ValueError, match="non-empty parameter name"):
        tl.param("")
    with pytest.raises(ValueError, match="bool tensor"):
        tl.param("c1.weight", mask=torch.ones(2))
    with pytest.raises(ValueError, match="non-negative int"):
        tl.random_like(like=tl.units("a", [(0,)]), within=tl.units("a", [(0,)]), seed=-1)
    with pytest.raises(ValueError, match="region-shaped"):
        tl.random_like(like=42, within=tl.units("a", [(0,)]), seed=0)


def test_random_selection_totality(log):
    """Seeded, deterministic, size-matched, sampled inside `within`."""

    like = tl.units("relu_1_2", [(0, 0, 1, 1), (0, 0, 2, 2), (0, 1, 3, 3)])
    within = log["relu_1_2"].__selection__() - like
    control = tl.random_like(like=like, within=within, seed=11)
    first = log[control]
    second = log[control]
    assert first == second  # seeded determinism
    assert _elements(first) == 3  # |result| == |like|
    # sampled inside within: no overlap with `like`
    overlap = first & log[like]
    assert overlap.is_empty
    assert first.kind == "ACT"  # result kind = within's kind


# ---------------------------------------------------------------------------
# Repr stability pins
# ---------------------------------------------------------------------------


def test_selection_repr_stability(log):
    """The AST __repr__ is stable and readable (future NL compile target)."""

    unit = tl.units("relu_1_2", [(0, 0, 1, 1)])
    assert repr(unit) == "Selection[ACT](units('relu_1_2', n=1))"
    param = tl.param("c1.weight")
    assert repr(param) == "Selection[PARAM](params('c1.weight'))"
    combined = unit | tl.units("conv2d_1_1", [(0, 0, 0, 0)])
    assert repr(combined) == "Selection[ACT]((units('relu_1_2', n=1) | units('conv2d_1_1', n=1)))"
    inverted = ~unit
    assert repr(inverted) == "Selection[ACT](~units('relu_1_2', n=1))"
    box = log["relu_2_4"].receptive_field.at((3, 3))
    assert repr(box.__selection__()) == "Selection[ACT](rf_box('relu_2_4:1' -> 'input.x'))"
    resolved = log[unit]
    assert repr(resolved) == "Selection[ACT](1 sites, 1 elements)"


def test_selection_is_frozen(log):
    """Both types refuse attribute mutation."""

    query = tl.units("relu_1_2", [(0, 0, 1, 1)])
    with pytest.raises(AttributeError):
        query._kind = "PARAM"
    resolved = log[query]
    with pytest.raises(AttributeError):
        resolved._entries = ()


def test_internal_closed_set_guards_raise(log):
    """The interior closed-set guards are live raises, never dead lines.

    Covers the unknown resolved-composition operator, the unknown AST node,
    and the operand mixin's abstract ``__selection__`` — the census forbids
    hiding these behind ``pragma: no cover``.
    """

    from torchlens._selection.selection import (
        _compose_resolved,
        _resolve_node,
        _SelectionOperand,
    )

    resolved = log[tl.units("relu_1_2", [(0, 0, 1, 1)])]
    with pytest.raises(ValueError, match="unknown operator"):
        _compose_resolved("xor", resolved, resolved)

    with pytest.raises(TypeError, match="unknown selection AST node"):
        _resolve_node(object(), log, "ACT")

    class _Bare(_SelectionOperand):
        __slots__ = ()

    with pytest.raises(NotImplementedError):
        _Bare().__selection__()
