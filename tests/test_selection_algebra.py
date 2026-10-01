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

EDGE-kind matrix cells are stage-3 territory (``trace.edges`` lands there);
the kind vocabulary and refusal cells for EDGE are pinned in
``test_kind_matrix_edge_cells_pinned_for_stage3``.
"""

from __future__ import annotations

import itertools

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.intervention.selectors import CompositeSelector, NotSelector
from torchlens.selection import (
    _FLIP,
    _RELATIONS,
    ResolvedSelection,
    Selection,
    SelectionError,
    SiteEntry,
    _difference_relation,
    _join_relation,
    _meet_relation,
)


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


def _elements(resolved: ResolvedSelection) -> int:
    return sum(entry.selected_count for entry in resolved)


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
    assert isinstance(union, Selection) and isinstance(inter, Selection)
    resolved_union = union.resolve(log)
    resolved_inter = inter.resolve(log)
    assert resolved_inter.empty is False
    assert _elements(resolved_inter) <= _elements(resolved_union)
    # Overlapping late-layer RF boxes at the input site: one shared ACT site.
    assert len(resolved_union) == 1
    assert resolved_union[0].site_key == ("input_1", 1)


def test_mixin_dispatch_box_facet_unit_cross_shapes(log):
    """box | facet-shaped and facet - unit compositions yield Selections."""

    box = log["relu_2_4"].receptive_field.at((3, 3))
    unit = tl.units("input_1", [(0, 0, 4, 4)])
    assert isinstance(box | unit, Selection)
    assert isinstance(box - unit, Selection)
    assert isinstance(unit - box, Selection)  # reflected __rsub__ path
    assert isinstance(~box, Selection)


def test_selector_notimplemented_branch_rows(log):
    """selector OP __selection__-bearing operand -> Selection via reflection;
    selector OP selector keeps shipped CompositeSelector semantics."""

    selector = tl.func("relu")
    box = log["relu_2_4"].receptive_field.at((3, 3))
    layer = log["relu_1_2"]
    assert isinstance(selector | box, Selection)
    assert isinstance(selector & box, Selection)
    assert isinstance(selector - box, Selection)
    assert isinstance(selector | layer, Selection)
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
    lifted_difference = difference.__selection__().resolve(log)
    lifted_desugared = desugared.__selection__().resolve(log)
    assert lifted_difference == lifted_desugared

    with pytest.raises(Exception, match="cannot subtract"):
        tl.func("relu") - 5


def test_selector_subtraction_non_selector_operand_refuses_typed():
    """`selector - <non-selector>` refuses with the documented code."""

    from torchlens._errors import ArgumentTypeError

    with pytest.raises(ArgumentTypeError) as excinfo:
        _ = tl.func("relu") - 3
    assert excinfo.value.fields["code"] == "selector_subtraction_operand_invalid"


def test_op_and_layer_lift(log):
    """Op lifts one pass; Layer lifts ALL passes (whole-output producers)."""

    layer = log["relu_1_2"]
    op = layer.ops[0]
    resolved_layer = layer.__selection__().resolve(log)
    resolved_op = op.__selection__().resolve(log)
    assert resolved_layer == resolved_op  # single-pass model: same denotation
    assert resolved_op[0].site_key == ("relu_1_2", 1)
    assert resolved_op[0].provenance.relation == "exact"
    assert resolved_op[0].selected_count == resolved_op[0].mask.numel()


# ---------------------------------------------------------------------------
# Truthiness pair
# ---------------------------------------------------------------------------


def test_bool_refusal_pair(log):
    """Selection.__bool__ refuses typed; ResolvedSelection is element-level."""

    query = tl.units("relu_1_2", [(0, 0, 0, 0)])
    with pytest.raises(SelectionError) as excinfo:
        bool(query)
    assert excinfo.value.fields["code"] == "selection_bool_ambiguous"

    resolved = query.resolve(log)
    assert bool(resolved) is True
    assert resolved.empty is False
    element_empty = resolved - resolved
    assert bool(element_empty) is False
    assert element_empty.empty is True
    assert len(element_empty) == 1  # family retained


# ---------------------------------------------------------------------------
# Law table (both denotation levels) + pinned non-laws
# ---------------------------------------------------------------------------


def _family(resolved: ResolvedSelection) -> frozenset:
    return frozenset(entry.site_key for entry in resolved)


def test_law_table(log):
    """Laws that HOLD, at both levels."""

    a = tl.units("relu_1_2", [(0, 0, 1, 1), (0, 1, 2, 2)]).resolve(log)
    b = tl.units("relu_1_2", [(0, 1, 2, 2), (0, 0, 3, 3)]).resolve(log)
    c = log["conv2d_1_1"].__selection__().resolve(log)

    # commutativity + associativity of | and &
    assert (a | b) == (b | a)
    assert (a & b) == (b & a)
    assert ((a | b) | c) == (a | (b | c))
    # NOTE: (a & b) & c crosses sites: relu vs conv -> different sites; & of
    # disjoint families is the empty-family selection, associativity holds.
    assert ((a & b) & c) == (a & (b & c))
    # distributivity
    assert (a & (b | c)) == ((a & b) | (a & c))
    # idempotence
    assert (a | a) == a
    assert (a & a) == a
    # A - A == element-empty over fam(A), NOT the no-sites selection
    minus_self = a - a
    assert minus_self.empty and _family(minus_self) == _family(a)
    # involution THROUGH the element-empty intermediate
    full_site = log["relu_1_2"].__selection__().resolve(log)
    complemented = ~full_site
    assert complemented.empty and _family(complemented) == _family(full_site)
    assert (~complemented) == full_site
    assert (~~a) == a
    # symmetric difference spelling
    sym = (a - b) | (b - a)
    union_minus_inter = (a | b) - (a & b)
    assert sym == union_minus_inter


def test_pinned_non_laws(log):
    """Deliberate NON-laws, pinned."""

    a = tl.units("relu_1_2", [(0, 0, 1, 1)]).resolve(log)
    c = log["conv2d_1_1"].__selection__().resolve(log)

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
    mask_complement = (~tl.func("relu").__selection__()).resolve(log)
    predicate_negation = (~tl.func("relu")).__selection__().resolve(log)
    assert mask_complement != predicate_negation
    # predicate negation matches all NON-relu sites; mask complement stays on
    # the relu sites with empty masks.
    assert _family(mask_complement) != _family(predicate_negation)


def test_complement_empty_cells(log):
    """The two empty cells, keyed by level."""

    a = tl.units("relu_1_2", [(0, 0, 1, 1)]).resolve(log)
    c = log["conv2d_1_1"].__selection__().resolve(log)
    no_sites = a & c  # disjoint families -> no touched sites
    assert len(no_sites) == 0
    assert (~no_sites) == no_sites  # complement of no-touched-sites

    element_empty = a - a
    restored = ~element_empty
    assert _family(restored) == _family(a)
    assert not restored.empty
    assert restored[0].selected_count == restored[0].mask.numel()  # FULL masks


# ---------------------------------------------------------------------------
# Closure/totality matrix over the widened axes
# ---------------------------------------------------------------------------


def _act_nonempty(log):
    return tl.units("relu_1_2", [(0, 0, 1, 1)]).resolve(log)


def _act_element_empty(log):
    resolved = _act_nonempty(log)
    return resolved - resolved


def _act_no_sites(log):
    return _act_nonempty(log) & log["conv2d_1_1"].__selection__().resolve(log)


def _param_nonempty(log):
    return tl.params("c1.weight").resolve(log)


def _param_element_empty(log):
    resolved = _param_nonempty(log)
    return resolved - resolved


def _param_no_sites(log):
    # PARAM x PARAM & over disjoint families -> no touched sites
    return tl.params("c1.weight").resolve(log) & tl.params("c2.weight").resolve(log)


_EMPTINESS_BUILDERS = {
    ("ACT", "nonempty"): _act_nonempty,
    ("ACT", "element_empty"): _act_element_empty,
    ("ACT", "no_sites"): _act_no_sites,
    ("PARAM", "nonempty"): _param_nonempty,
    ("PARAM", "element_empty"): _param_element_empty,
    ("PARAM", "no_sites"): _param_no_sites,
}


@pytest.mark.parametrize("left_kind,right_kind", [("ACT", "ACT"), ("PARAM", "PARAM")])
@pytest.mark.parametrize("operator", ["or", "and", "sub"])
@pytest.mark.parametrize("left_state", ["no_sites", "element_empty", "nonempty"])
@pytest.mark.parametrize("right_state", ["no_sites", "element_empty", "nonempty"])
def test_totality_same_kind_cells(log, left_kind, right_kind, operator, left_state, right_state):
    """Every same-kind cell composes legally (empties first-class, disclosed)."""

    left = _EMPTINESS_BUILDERS[(left_kind, left_state)](log)
    right = _EMPTINESS_BUILDERS[(right_kind, right_state)](log)
    composed = {
        "or": lambda: left | right,
        "and": lambda: left & right,
        "sub": lambda: left - right,
    }[operator]()
    assert isinstance(composed, ResolvedSelection)
    # family rules
    if operator == "or":
        assert _family(composed) == (_family(left) | _family(right))
    elif operator == "and":
        assert _family(composed) == (_family(left) & _family(right))
    else:
        assert _family(composed) == _family(left)


@pytest.mark.parametrize("operator", ["or", "and", "sub"])
@pytest.mark.parametrize("resolved_state", ["query", "resolved"])
def test_totality_mixed_kind_cells_refuse(log, operator, resolved_state):
    """ACT x PARAM refuses typed at both query and resolved levels."""

    act = tl.units("relu_1_2", [(0, 0, 1, 1)])
    param = tl.params("c1.weight")
    if resolved_state == "resolved":
        act = act.resolve(log)
        param = param.resolve(log)
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
        assert isinstance(inverted, ResolvedSelection)
        assert _family(inverted) == _family(resolved)
    query = tl.units("relu_1_2", [(0, 0, 1, 1)])
    assert isinstance(~query, Selection)
    assert (~~query).resolve(log) == query.resolve(log)


def test_kind_matrix_edge_cells_pinned_for_stage3():
    """EDGE matrix cells are pinned as stage-3 territory: the closed kind
    vocabulary already carries EDGE, no stage-1 constructor exists, and
    ACT x EDGE / PARAM x EDGE refuse via the same closed matrix when the
    edge-term producer lands (trace.edges, stage 3)."""

    from torchlens.selection import _SELECTION_KINDS

    assert _SELECTION_KINDS == ("ACT", "PARAM", "EDGE")


def test_query_composition_flattens_or(log):
    """Chained | accumulates flat n-ary operands (constant depth)."""

    query = (
        tl.units("relu_1_2", [(0, 0, 1, 1)])
        | tl.units("relu_1_2", [(0, 0, 2, 2)])
        | tl.units("relu_1_2", [(0, 0, 3, 3)])
    )
    assert isinstance(query, Selection)
    assert query.resolve(log)[0].selected_count == 3


def test_query_resolved_mixing(log, other_log):
    """resolved OP query resolves the query against the resolved side's trace;
    resolved OP resolved across traces refuses typed."""

    query = tl.units("relu_1_2", [(0, 0, 1, 1)])
    resolved = tl.units("relu_1_2", [(0, 0, 2, 2)]).resolve(log)
    mixed = resolved | query
    assert isinstance(mixed, ResolvedSelection)
    assert mixed[0].selected_count == 2
    mixed_reflected = query | resolved
    assert mixed_reflected == mixed

    foreign = tl.units("relu_1_2", [(0, 0, 1, 1)]).resolve(other_log)
    with pytest.raises(SelectionError) as excinfo:
        resolved | foreign
    assert excinfo.value.fields["code"] == "selection_trace_mismatch"


# ---------------------------------------------------------------------------
# Provenance relation tables — TABLE-DRIVEN over every cell, never sampled
# ---------------------------------------------------------------------------


def test_flip_table():
    assert _FLIP == {
        "exact": "exact",
        "upper_bound": "lower_bound",
        "lower_bound": "upper_bound",
        "unknown": "unknown",
    }


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


@pytest.mark.parametrize("a,b", list(itertools.product(_RELATIONS, _RELATIONS)))
def test_join_table_all_cells(a, b):
    """JOIN is symmetric with exact as identity — every cell pinned."""

    expected = _JOIN_EXPECTED.get((a, b)) or _JOIN_EXPECTED[(b, a)]
    assert _join_relation(a, b) == expected
    assert _join_relation(b, a) == expected
    assert _meet_relation(a, b) == expected  # meet lattice coincides


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


@pytest.mark.parametrize("a,b", list(itertools.product(_RELATIONS, _RELATIONS)))
def test_difference_table_all_cells_both_orders(a, b):
    """DIFFERENCE is ordered flip-first: rel(A-B) = join(rel(A), FLIP(rel(B)))."""

    assert _difference_relation(a, b) == _DIFFERENCE_EXPECTED[(a, b)]


def test_relation_composition_flows_through_operators(log):
    """Composed entries carry the table-derived relation."""

    exact = tl.units("input_1", [(0, 0, 4, 4)]).resolve(log)  # relation exact
    box = log["relu_2_4"].receptive_field.at((3, 3))
    hull = box.__selection__().resolve(log)
    hull_relation = hull[0].provenance.relation  # exact or upper_bound
    joined = exact | hull
    assert joined[0].provenance.relation == _join_relation("exact", hull_relation)
    difference = exact - hull
    assert difference[0].provenance.relation == _difference_relation("exact", hull_relation)
    flipped = ~hull
    assert flipped[0].provenance.relation == _FLIP[hull_relation]


# ---------------------------------------------------------------------------
# resolve(): parity, identity, refusal matrix
# ---------------------------------------------------------------------------


def test_mask_representation_parity(log):
    """whole-site sentinel == dense equivalent; slice-box == dense equivalent."""

    op = log["relu_1_2"].ops[0]
    whole = op.__selection__().resolve(log)
    dense_mask = torch.ones(op.shape, dtype=torch.bool)
    dense = tl.units("relu_1_2", dense_mask).resolve(log)
    assert whole == dense

    box = log["relu_2_4"].receptive_field.at((3, 3))
    from_slices = box.__selection__().resolve(log)
    dense_box = torch.zeros(box.input_shape, dtype=torch.bool)
    dense_box[box.slices()] = True
    from_dense = tl.units("input_1", dense_box).resolve(log)
    assert from_slices[0].site_key == from_dense[0].site_key
    assert torch.equal(from_slices[0].mask, from_dense[0].mask)


def test_mask_immutability_pin(log):
    """Mutating a returned mask alters NOTHING: digest, equality, later reads."""

    resolved = tl.units("relu_1_2", [(0, 0, 1, 1)]).resolve(log)
    entry = resolved[0]
    digest_before = resolved.resolve_digest
    handed_out = entry.mask
    assert handed_out is not entry.mask  # fresh materialization per access
    handed_out.fill_(True)
    assert entry.selected_count == 1
    assert resolved.resolve_digest == digest_before
    assert resolved == tl.units("relu_1_2", [(0, 0, 1, 1)]).resolve(log)


def test_repeated_resolve_identity_stability(log):
    """Repeated resolves are equal with equal digests (deterministic)."""

    query = tl.units("relu_1_2", [(0, 0, 1, 1)]) | tl.func("conv2d")
    first = query.resolve(log)
    second = query.resolve(log)
    assert first == second
    assert first.resolve_digest == second.resolve_digest


def test_multi_site_value_type(log):
    """Iteration follows GRAPH ORDER (F10); per-site access by position.

    Re-pin: entries read in the trace's execution order, not lexicographic
    site-key order (lovely bug 14 -- ``relu_1_10`` no longer sorts before
    ``relu_1_2``).
    """

    resolved = (tl.func("relu") | tl.func("conv2d")).__selection__().resolve(log)
    keys = [entry.site_key for entry in resolved]
    execution_order = {
        (op.layer_label, op.pass_index): index for index, op in enumerate(log.layer_list)
    }
    steps = [execution_order[key] for key in keys]
    assert steps == sorted(steps), "resolved ACT entries must read in execution order"
    assert resolved[0].site_key == keys[0]
    assert len(resolved) == len(keys)
    assert all(isinstance(entry, SiteEntry) for entry in resolved)


def test_empty_selection_legal_end_to_end(log):
    """Zero matches is disclosure, never an error."""

    resolved = tl.label("no_such_layer_anywhere").__selection__().resolve(log)
    assert len(resolved) == 0
    assert resolved.empty is True
    assert bool(resolved) is False


def test_resolver_refusal_matrix(log):
    """Every closed selection_unresolvable reason row (stage-1 producers)."""

    # site_not_in_trace
    with pytest.raises(SelectionError) as excinfo:
        tl.units("nowhere_9_9", [(0,)]).resolve(log)
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "site_not_in_trace"

    with pytest.raises(SelectionError) as excinfo:
        tl.params("no.such.param").resolve(log)
    assert excinfo.value.fields["reason"] == "site_not_in_trace"

    # mask_shape_mismatch (bad shape mask; out-of-range unit index)
    with pytest.raises(SelectionError) as excinfo:
        tl.units("relu_1_2", torch.ones(2, 2, dtype=torch.bool)).resolve(log)
    assert excinfo.value.fields["reason"] == "mask_shape_mismatch"
    with pytest.raises(SelectionError) as excinfo:
        tl.units("relu_1_2", [(9, 9, 9, 9)]).resolve(log)
    assert excinfo.value.fields["reason"] == "mask_shape_mismatch"
    with pytest.raises(SelectionError) as excinfo:
        tl.params("c1.weight", mask=torch.ones(1, dtype=torch.bool)).resolve(log)
    assert excinfo.value.fields["reason"] == "mask_shape_mismatch"

    # population_too_small (incl. the empty complement case)
    like = tl.units("relu_1_2", [(0, 0, 1, 1), (0, 0, 2, 2)])
    whole = log["relu_1_2"].__selection__()
    with pytest.raises(SelectionError) as excinfo:
        tl.random_selection(like=like, within=~whole, seed=3).resolve(log)
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

    import torchlens.selection as selection_module

    monkeypatch.setattr(
        selection_module, "_forward_ops", lambda trace: (_NoShapeOp(), _NonTensorOp())
    )
    with pytest.raises(SelectionError) as excinfo:
        tl.units("stub_1_1", [(0,)]).resolve(log)
    assert excinfo.value.fields["reason"] == "no_index_space"
    with pytest.raises(SelectionError) as excinfo:
        tl.units("tuple_1_1", [(0,)]).resolve(log)
    assert excinfo.value.fields["reason"] == "non_tensor_site"


def test_facet_lift_and_refusal_rows(log):
    """FacetSpec lifts to its write region; unsaved home refuses value_not_saved."""

    from torchlens.semantic.facets import FacetSpec

    op = log["relu_1_2"].ops[0]
    spec = FacetSpec.from_home(op)
    resolved = spec.__selection__().resolve(log)
    assert resolved[0].site_key == ("relu_1_2", 1)
    assert resolved[0].provenance.relation == "exact"
    assert resolved[0].selected_count == resolved[0].mask.numel()

    # Mixin dispatch on the facet producer
    assert isinstance(spec | op, Selection)

    torch.manual_seed(0)
    sparse_log = tl.trace(_TwoConv(), torch.randn(1, 1, 12, 12), save=tl.func("conv2d"))
    sparse_op = sparse_log["relu_1_2"].ops[0]
    sparse_spec = FacetSpec.from_home(sparse_op)
    with pytest.raises(SelectionError) as excinfo:
        sparse_spec.__selection__().resolve(sparse_log)
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] in ("value_not_saved", "facet_write_mask_unavailable")


def test_gradient_rf_lift():
    """GradientReceptiveField lifts as an exact-set empirical term."""

    torch.manual_seed(0)
    model = _TwoConv()
    x = torch.randn(1, 1, 12, 12).requires_grad_(True)
    armed = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(backward_ready=True),
        save_mode="reference",
    )
    armed_op = armed["relu_2_4"]
    unit = armed_op.receptive_field.center_unit(batch_index=0)
    gradient_by_role = armed_op.receptive_field.gradient(unit, retain_graph=True)
    gradient = gradient_by_role["input.x"]
    selection = gradient.__selection__()
    assert isinstance(selection, Selection)
    resolved = selection.resolve(armed)
    assert resolved[0].provenance.relation == "exact"
    assert resolved[0].selected_count == int(gradient.support_mask.sum().item())
    # Mixin dispatch with a geometric box at the same site
    box = armed_op.receptive_field.at("center")
    combined = gradient | box
    assert isinstance(combined, Selection)
    assert combined.resolve(armed)[0].selected_count >= resolved[0].selected_count


def test_box_lift_relation_and_shape_guard(log):
    """Hull lift relation: exact only when box.exact and no sparse axis."""

    box = log["relu_2_4"].receptive_field.at((3, 3))
    resolved = box.__selection__().resolve(log)
    expected = "exact" if (box.exact and not box.sparse_possible) else "upper_bound"
    assert resolved[0].provenance.relation == expected
    assert resolved[0].site_key == ("input_1", 1)


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
        tl.params("")
    with pytest.raises(ValueError, match="bool tensor"):
        tl.params("c1.weight", mask=torch.ones(2))
    with pytest.raises(ValueError, match="non-negative int"):
        tl.random_selection(like=tl.units("a", [(0,)]), within=tl.units("a", [(0,)]), seed=-1)
    with pytest.raises(ValueError, match="region-shaped"):
        tl.random_selection(like=42, within=tl.units("a", [(0,)]), seed=0)


def test_random_selection_totality(log):
    """Seeded, deterministic, size-matched, sampled inside `within`."""

    like = tl.units("relu_1_2", [(0, 0, 1, 1), (0, 0, 2, 2), (0, 1, 3, 3)])
    within = log["relu_1_2"].__selection__() - like
    control = tl.random_selection(like=like, within=within, seed=11)
    first = control.resolve(log)
    second = control.resolve(log)
    assert first == second  # seeded determinism
    assert _elements(first) == 3  # |result| == |like|
    # sampled inside within: no overlap with `like`
    overlap = first & like.resolve(log)
    assert overlap.empty
    assert first.kind == "ACT"  # result kind = within's kind


# ---------------------------------------------------------------------------
# Repr stability pins
# ---------------------------------------------------------------------------


def test_selection_repr_stability(log):
    """The AST __repr__ is stable and readable (future NL compile target)."""

    unit = tl.units("relu_1_2", [(0, 0, 1, 1)])
    assert repr(unit) == "Selection[ACT](units('relu_1_2', n=1))"
    param = tl.params("c1.weight")
    assert repr(param) == "Selection[PARAM](params('c1.weight'))"
    combined = unit | tl.units("conv2d_1_1", [(0, 0, 0, 0)])
    assert repr(combined) == "Selection[ACT]((units('relu_1_2', n=1) | units('conv2d_1_1', n=1)))"
    inverted = ~unit
    assert repr(inverted) == "Selection[ACT](~units('relu_1_2', n=1))"
    box = log["relu_2_4"].receptive_field.at((3, 3))
    assert repr(box.__selection__()) == "Selection[ACT](rf_box('relu_2_4:1' -> 'input.x'))"
    resolved = unit.resolve(log)
    assert repr(resolved) == "ResolvedSelection[ACT](1 sites, 1 elements)"


def test_selection_is_frozen(log):
    """Both types refuse attribute mutation."""

    query = tl.units("relu_1_2", [(0, 0, 1, 1)])
    with pytest.raises(AttributeError):
        query._kind = "PARAM"
    resolved = query.resolve(log)
    with pytest.raises(AttributeError):
        resolved._entries = ()


def test_selection_error_is_catalogued():
    """SelectionError rides the intervention error catalog surface."""

    from torchlens.intervention import SelectionError as catalogued

    assert catalogued is SelectionError
    assert issubclass(SelectionError, ValueError)
    assert SelectionError.severity == "recoverable"


def test_internal_closed_set_guards_raise(log):
    """The interior closed-set guards are live raises, never dead lines.

    Covers the unknown resolved-composition operator, the unknown AST node,
    and the operand mixin's abstract ``__selection__`` — the census forbids
    hiding these behind ``pragma: no cover``.
    """

    from torchlens.selection import (
        _compose_resolved,
        _resolve_node,
        _SelectionOperand,
    )

    resolved = tl.units("relu_1_2", [(0, 0, 1, 1)]).resolve(log)
    with pytest.raises(ValueError, match="unknown operator"):
        _compose_resolved("xor", resolved, resolved)

    with pytest.raises(TypeError, match="unknown selection AST node"):
        _resolve_node(object(), log, "ACT")

    class _Bare(_SelectionOperand):
        __slots__ = ()

    with pytest.raises(NotImplementedError):
        _Bare().__selection__()
