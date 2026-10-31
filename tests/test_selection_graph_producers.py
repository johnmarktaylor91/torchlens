"""Graph-structural producers + TraceSlice: acceptance and honesty gates.

Covers the graph-lane obligations: ``neighborhood`` / ``between`` return
composable ``Selection`` queries whose member sets are verified against
hand-computed hop sets and path regions on a branchy model, family
semantics (element masks never shrink a graph region), pinned ``exact``
provenance, parity with the shipped influence-geometry ``between_labels``
machinery, the typed refusal matrix (kind, unknown sites, bad parameters),
multi-pass (recurrence-grouped) traversal over pass-qualified nodes, the
``TraceSlice`` presenter contract (members, internal edges, EXPLICIT
boundary declaration, entry/exit ops, teaching lookup misses, frozen
surface, algebra lift, the ``do()`` gallery row), and the
``slice_save_unsupported`` save refusal.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._io.bundle import save as _save_bundle
from torchlens._selection import SelectionError


class _Branchy(nn.Module):
    """Two parallel branches merging into a head: a real path structure.

    input -> lin1 -> relu -\\
                            add -> head
    input -> lin2 -> tanh -/
    """

    def __init__(self) -> None:
        super().__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.relu(self.lin1(x))
        b = torch.tanh(self.lin2(x))
        return self.head(a + b)


class _Loopy(nn.Module):
    """One linear+relu applied three times (recurrence-grouped passes)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.relu(self.lin(x))
        return x


_INPUT = "input_1:1"
_LIN1 = "linear_1_1:1"
_RELU = "relu_1_2:1"
_LIN2 = "linear_2_3:1"
_TANH = "tanh_1_4:1"
_ADD = "add_1_5:1"
_HEAD = "linear_3_6:1"
_OUTPUT = "output_1:1"
_ALL = (_INPUT, _LIN1, _RELU, _LIN2, _TANH, _ADD, _HEAD, _OUTPUT)


@pytest.fixture(scope="module")
def model():
    """One shared deterministic branchy model."""

    torch.manual_seed(0)
    return _Branchy()


@pytest.fixture(scope="module")
def log(model):
    """Primary structural capture."""

    torch.manual_seed(1)
    trace = tl.trace(model, torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def loopy_log():
    """Multi-pass capture (three passes of one recurrence-grouped layer)."""

    torch.manual_seed(2)
    trace = tl.trace(_Loopy(), torch.randn(2, 4))
    try:
        yield trace
    finally:
        trace.cleanup()


def _labels(resolved) -> set[str]:
    """Return the resolved family as pass-qualified op labels."""

    return {
        f"{layer_label}:{pass_index}"
        for layer_label, pass_index in (entry.key for entry in resolved)
    }


# ---------------------------------------------------------------------------
# neighborhood: hop sets.
# ---------------------------------------------------------------------------


def test_neighborhood_zero_hops_is_the_seed_family(log) -> None:
    """hops=0 selects exactly the seed's touched sites."""

    resolved = tl.neighborhood("relu_1_2", hops=0).resolve(log)
    assert _labels(resolved) == {_RELU}


def test_neighborhood_one_hop_both_directions(log) -> None:
    """One hop reaches exactly the recorded parents and children."""

    resolved = tl.neighborhood("relu_1_2", hops=1).resolve(log)
    assert _labels(resolved) == {_LIN1, _RELU, _ADD}


def test_neighborhood_directional_hops(log) -> None:
    """upstream follows parent edges only; downstream child edges only."""

    upstream = tl.neighborhood("relu_1_2", hops=2, direction="upstream").resolve(log)
    assert _labels(upstream) == {_INPUT, _LIN1, _RELU}
    downstream = tl.neighborhood("relu_1_2", hops=2, direction="downstream").resolve(log)
    assert _labels(downstream) == {_RELU, _ADD, _HEAD}


def test_neighborhood_saturates_at_the_component(log) -> None:
    """A hop budget beyond the diameter reaches the whole component, stably."""

    resolved = tl.neighborhood("add_1_5", hops=50).resolve(log)
    assert _labels(resolved) == set(_ALL)


def test_neighborhood_masks_are_whole_site_exact(log) -> None:
    """Graph membership is structural: whole-site masks, relation exact."""

    resolved = tl.neighborhood("relu_1_2", hops=1).resolve(log)
    for entry in resolved:
        assert entry.provenance.relation == "exact"
        assert entry.num_selected == entry.mask.numel()
        assert "neighborhood(hops=1" in entry.provenance.source


def test_neighborhood_family_semantics_ignore_element_masks(log) -> None:
    """A one-element seed still seeds its WHOLE site (touched is touched)."""

    seed = tl.units(_RELU, [(0, 0)])
    resolved = tl.neighborhood(seed, hops=0).resolve(log)
    assert _labels(resolved) == {_RELU}
    assert resolved[0].num_selected == resolved[0].mask.numel()


def test_neighborhood_accepts_op_and_layer_seeds(log) -> None:
    """Op and Layer handles lift as seeds through __selection__."""

    layer = log["relu_1_2"]
    from_layer = tl.neighborhood(layer, hops=0).resolve(log)
    assert _labels(from_layer) == {_RELU}
    from_op = tl.neighborhood(layer.ops[0], hops=0).resolve(log)
    assert _labels(from_op) == {_RELU}


def test_neighborhood_composes_with_the_algebra(log) -> None:
    """Producers return Selections: the full operator surface applies."""

    union = tl.neighborhood("relu_1_2", hops=0) | tl.neighborhood("tanh_1_4", hops=0)
    assert _labels(union.resolve(log)) == {_RELU, _TANH}
    restricted = tl.neighborhood("relu_1_2", hops=1) & tl.units(_ADD, [(0, 0)])
    resolved = restricted.resolve(log)
    assert _labels(resolved) == {_ADD}
    assert resolved[0].num_selected == 1


# ---------------------------------------------------------------------------
# neighborhood: refusals.
# ---------------------------------------------------------------------------


def test_neighborhood_validates_hops_and_direction() -> None:
    """Bad hops / direction refuse at construction with plain ValueError."""

    with pytest.raises(ValueError, match="non-negative int"):
        tl.neighborhood("relu_1_2", hops=-1)
    with pytest.raises(ValueError, match="non-negative int"):
        tl.neighborhood("relu_1_2", hops=True)
    with pytest.raises(ValueError, match="direction"):
        tl.neighborhood("relu_1_2", direction="sideways")
    with pytest.raises(ValueError, match="site label string"):
        tl.neighborhood(None)
    with pytest.raises(ValueError, match="non-empty"):
        tl.neighborhood("")


def test_neighborhood_refuses_param_seed_typed() -> None:
    """PARAM operands refuse selection_kind_incompatible at construction."""

    with pytest.raises(SelectionError) as excinfo:
        tl.neighborhood(tl.param("lin1.weight"))
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"


def test_neighborhood_unknown_seed_refuses_typed(log) -> None:
    """Unknown seed sites refuse selection_unresolvable / site_not_in_trace."""

    with pytest.raises(SelectionError) as excinfo:
        tl.neighborhood("nonexistent_9_9", hops=1).resolve(log)
    assert excinfo.value.fields["code"] == "selection_unresolvable"
    assert excinfo.value.fields["reason"] == "site_not_in_trace"


# ---------------------------------------------------------------------------
# between: influence regions.
# ---------------------------------------------------------------------------


def test_between_selects_exactly_the_path_region(log) -> None:
    """The region is the ops on at least one directed source-to-sink path."""

    resolved = tl.between("linear_1_1", "add_1_5").resolve(log)
    assert _labels(resolved) == {_LIN1, _RELU, _ADD}


def test_between_excludes_the_other_branch(log) -> None:
    """Ops not on any source-to-sink path stay out (the tanh branch)."""

    resolved = tl.between("input_1", "add_1_5").resolve(log)
    assert _labels(resolved) == {_INPUT, _LIN1, _RELU, _LIN2, _TANH, _ADD}
    narrow = tl.between("relu_1_2", "linear_3_6").resolve(log)
    assert _TANH not in _labels(narrow)
    assert _labels(narrow) == {_RELU, _ADD, _HEAD}


def test_between_same_endpoint_is_the_single_op(log) -> None:
    """between(x, x) is the one-op region (endpoints included)."""

    resolved = tl.between("relu_1_2", "relu_1_2").resolve(log)
    assert _labels(resolved) == {_RELU}


def test_between_no_path_is_the_empty_selection(log) -> None:
    """No directed path resolves EMPTY: disclosure, never an error."""

    resolved = tl.between("tanh_1_4", "relu_1_2").resolve(log)
    assert len(resolved) == 0
    assert not resolved
    parallel = tl.between("relu_1_2", "tanh_1_4").resolve(log)
    assert len(parallel) == 0


def test_between_accepts_endpoint_lists(log) -> None:
    """Multi-source / multi-sink endpoint groups union their regions."""

    resolved = tl.between(["linear_1_1", "linear_2_3"], "linear_3_6").resolve(log)
    assert _labels(resolved) == {_LIN1, _RELU, _LIN2, _TANH, _ADD, _HEAD}


def test_between_agrees_with_influence_geometry_machinery(log) -> None:
    """Producer members equal the shipped between_labels region (one machinery)."""

    from torchlens.receptive_field._path import between_labels

    expected = set(between_labels(log, _LIN1, _HEAD))
    resolved = tl.between("linear_1_1", "linear_3_6").resolve(log)
    assert _labels(resolved) == expected


def test_between_relation_is_exact_with_source_disclosure(log) -> None:
    """Structural membership claims are exact and disclosed."""

    resolved = tl.between("linear_1_1", "add_1_5").resolve(log)
    for entry in resolved:
        assert entry.provenance.relation == "exact"
        assert "between" in entry.provenance.source


def test_between_refusal_matrix(log) -> None:
    """Empty groups, PARAM kinds, and unknown sites refuse typed."""

    with pytest.raises(ValueError, match="at least one region"):
        tl.between([], "add_1_5")
    with pytest.raises(SelectionError) as excinfo:
        tl.between(tl.param("lin1.weight"), "add_1_5")
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"
    with pytest.raises(SelectionError) as unresolved:
        tl.between("nonexistent_9_9", "add_1_5").resolve(log)
    assert unresolved.value.fields["reason"] == "site_not_in_trace"


# ---------------------------------------------------------------------------
# Multi-pass (recurrence-grouped) traversal.
# ---------------------------------------------------------------------------


def test_multipass_nodes_are_pass_qualified(loopy_log) -> None:
    """between spans passes: pass 1 -> pass 3 of one layer crosses pass 2."""

    resolved = tl.between("linear_1_1:1", "linear_1_1:3").resolve(loopy_log)
    labels = _labels(resolved)
    assert {"linear_1_1:1", "linear_1_1:2", "linear_1_1:3"} <= labels
    assert {"relu_1_2:1", "relu_1_2:2"} <= labels
    assert "relu_1_2:3" not in labels


def test_multipass_bare_label_seed_means_all_passes(loopy_log) -> None:
    """A bare layer label seed is the all-passes Layer spelling."""

    resolved = tl.neighborhood("linear_1_1", hops=0).resolve(loopy_log)
    assert _labels(resolved) == {"linear_1_1:1", "linear_1_1:2", "linear_1_1:3"}


# ---------------------------------------------------------------------------
# TraceSlice: the sub-DAG view presenter.
# ---------------------------------------------------------------------------


def test_slice_members_match_the_producer_region(log) -> None:
    """trace.between and tl.between denote the SAME region (one machinery)."""

    view = log.between("linear_1_1", "add_1_5")
    resolved = tl.between("linear_1_1", "add_1_5").resolve(log)
    assert set(view.labels) == _labels(resolved)
    assert view.labels == (_LIN1, _RELU, _ADD)  # execution order


def test_slice_internal_edges_are_exact(log) -> None:
    """Internal edges are exactly the member-to-member dataflow edges."""

    view = log.between("linear_1_1", "add_1_5")
    assert view.edges == ((_LIN1, _RELU), (_RELU, _ADD))


def test_slice_declares_its_boundary_explicitly(log) -> None:
    """Every crossing edge is named: external dependencies never vanish."""

    view = log.between("linear_1_1", "add_1_5")
    assert view.boundary_in_edges == ((_INPUT, _LIN1), (_TANH, _ADD))
    assert view.boundary_out_edges == ((_ADD, _HEAD),)


def test_slice_entry_and_exit_ops(log) -> None:
    """source_ops / sink_ops are the members without in-slice parents/children."""

    view = log.between("linear_1_1", "add_1_5")
    assert [op.label for op in view.source_ops] == [_LIN1]
    assert [op.label for op in view.sink_ops] == [_ADD]


def test_slice_iteration_and_membership(log) -> None:
    """Iteration yields member Ops in execution order; __contains__ is honest."""

    view = log.between("linear_1_1", "add_1_5")
    assert len(view) == 3
    assert [op.label for op in view] == [_LIN1, _RELU, _ADD]
    assert _RELU in view
    assert "relu_1_2" in view  # bare layer spelling
    assert _TANH not in view
    assert view.ops[0] in view


def test_slice_getitem_teaches_on_misses(log) -> None:
    """Lookup misses raise teaching KeyErrors that name the failure mode."""

    view = log.between("linear_1_1", "add_1_5")
    assert view[_RELU].label == _RELU
    assert view["relu_1_2"].label == _RELU  # bare, unique among members
    with pytest.raises(KeyError, match="not a member of"):
        view["tanh_1_4"]
    with pytest.raises(KeyError, match="not an op of this slice's underlying trace"):
        view["nonexistent_9_9"]
    with pytest.raises(KeyError, match="label string"):
        view[0]


def test_slice_getitem_multipass_bare_label_is_ambiguous(loopy_log) -> None:
    """A bare label naming several member passes teaches the qualified spelling."""

    view = loopy_log.between("linear_1_1:1", "linear_1_1:3")
    with pytest.raises(KeyError, match="pass-qualified"):
        view["linear_1_1"]


def test_slice_empty_region_is_disclosure(log) -> None:
    """No directed path yields an EMPTY slice that says so, not an error."""

    view = log.between("tanh_1_4", "relu_1_2")
    assert view.empty
    assert len(view) == 0
    assert view.boundary_in_edges == ()
    assert "0 ops" in repr(view)
    assert "no directed path" in view.summary()


def test_slice_summary_disclosure(log) -> None:
    """The summary names entry/exit ops and the boundary edges."""

    text = log.between("linear_1_1", "add_1_5").summary()
    assert "entry ops: linear_1_1:1" in text
    assert "exit ops: add_1_5:1" in text
    assert "2 edges in, 1 edges out" in text
    assert f"in: {_TANH} -> {_ADD}" in text
    assert f"out: {_ADD} -> {_HEAD}" in text


def test_slice_is_frozen(log) -> None:
    """The presenter refuses mutation after freeze."""

    view = log.between("linear_1_1", "add_1_5")
    with pytest.raises(AttributeError, match="frozen"):
        view.labels = ()


def test_slice_offers_no_capture_capabilities(log) -> None:
    """A slice is a presenter: no save/run/validate/draw surfaces exist."""

    view = log.between("linear_1_1", "add_1_5")
    for capability in ("save", "run", "validate", "log_backward", "draw", "fork"):
        assert not hasattr(view, capability)


def test_slice_save_refuses_typed(log, tmp_path) -> None:
    """Trace.save on a slice refuses slice_save_unsupported with the remedy."""

    view = log.between("linear_1_1", "add_1_5")
    with pytest.raises(SelectionError) as excinfo:
        _save_bundle(view, str(tmp_path / "slice_bundle"))
    assert excinfo.value.fields["code"] == "slice_save_unsupported"
    assert "source_trace" in str(excinfo.value)


def test_slice_lifts_back_into_the_algebra(log, loopy_log) -> None:
    """__selection__ + the operator mixin compose slices with selections."""

    view = log.between("linear_1_1", "add_1_5")
    lifted = view.__selection__()
    assert isinstance(lifted, tl.Selector)  # a QUERY, like Op/Layer lifts
    assert _labels(lifted.resolve(log)) == set(view.labels)
    composed = (view & tl.units(_RELU, [(0, 0)])).resolve(log)
    assert _labels(composed) == {_RELU}
    union = (view | tl.neighborhood("tanh_1_4", hops=0)).resolve(log)
    assert _labels(union) == {_LIN1, _RELU, _ADD, _TANH}
    missing = log.between("linear_1_1", "add_1_5").__selection__()
    with pytest.raises(SelectionError) as excinfo:
        missing.resolve(loopy_log)  # no add_1_5 site on the loopy trace
    assert excinfo.value.fields["reason"] == "site_not_in_trace"


def test_subgraph_is_the_general_door(log) -> None:
    """Any ACT region presents as a slice: neighborhood, string, Op, units."""

    from_producer = log.subgraph(tl.neighborhood("add_1_5", hops=1))
    assert set(from_producer.labels) == {_RELU, _TANH, _ADD, _HEAD}
    from_string = log.subgraph("relu_1_2")
    assert from_string.labels == (_RELU,)
    from_layer = log.subgraph(log["relu_1_2"])
    assert from_layer.labels == (_RELU,)
    from_units = log.subgraph(tl.units(_ADD, [(0, 0)]))
    assert from_units.labels == (_ADD,)  # family semantics: whole site


def test_subgraph_refusal_matrix(log) -> None:
    """Non-ACT regions and non-selection garbage refuse typed."""

    with pytest.raises(SelectionError) as excinfo:
        log.subgraph(tl.param("lin1.weight"))
    assert excinfo.value.fields["code"] == "selection_kind_incompatible"
    with pytest.raises(ValueError, match="selection-shaped"):
        log.subgraph(42)


def test_slice_do_gallery_row(model) -> None:
    """End-to-end: ablate a slice's member sites on a fork via do()."""

    torch.manual_seed(3)
    log = tl.trace(
        model,
        torch.randn(2, 4),
        capture=tl.options.CaptureOptions(intervention_ready=True, layers_to_save="all"),
    )
    try:
        view = log.between("linear_1_1", "relu_1_2")
        fork = log.fork()
        fork.do(view, tl.zero_ablate())
        assert torch.all(fork[_RELU].out == 0)
        assert torch.all(fork[_LIN1].out == 0)
        assert not torch.all(log[_RELU].out == 0)  # capture truth intact
    finally:
        log.cleanup()
