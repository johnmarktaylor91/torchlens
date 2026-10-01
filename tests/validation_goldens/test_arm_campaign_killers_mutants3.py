"""Direct per-arm killers for the M1 raise-arm mutation campaign (mutants3 lane).

Continues ``test_arm_campaign_killers_mutants2.py``'s per-arm coverage: each test
below targets exactly ONE ``raise MetadataInvariantError`` arm of its checker
(mutation_driver.py's per-arm family, id ``<contract>#aNN``) with a duck-typed
fake of the exact object the checker reads, proving every EARLIER arm on the
same checker stays silent so the mutant's own arm is what trips (the project's
"Validation Integrity (LOCKED)" doctrine: root-cause the real violation path,
never broaden a tolerance).

This lane's roster is the arm-shard 1/4, 2/4, and 3/4 survivor list measured
against the mutants2 lane's branch point (shard 4's own survivors are covered
by ``test_arm_campaign_killers_mutants2.py``): one duck-typed fake per raise
arm, each proving every earlier arm on the same checker stays silent.

Two tiny indexable fakes are needed because several checkers index the trace
or a module accessor directly (``ml[label]``, ``mod_accessor[address]``)
rather than only reading plain attributes, which a bare ``SimpleNamespace``
cannot support -- mirrored from the mutants2 killer file so this module stays
independently runnable and reviewable.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.validation.invariants import (
    MetadataInvariantError,
    _check_backend_identity_invariants,
    _check_backend_neutral_graph_topology,
    _check_buffer_xrefs,
    _check_distance_invariants,
    _check_edge_use_parent_arg_invariants,
    _check_equivalence_symmetry,
    _check_graph_connectivity,
    _check_graph_ordering,
    _check_graph_topology,
    _check_layer_pass_to_layer_log_xrefs,
    _check_lookup_key_consistency,
    _check_loop_detection_invariants,
    _check_module_containment_logic,
    _check_module_hierarchy,
    _check_module_layer_containment,
    _check_non_torch_backward_inert,
    _check_non_torch_primitive_op_inert,
    _check_op_log_fields,
    _check_param_xrefs,
    _check_pass_count_consistency,
    _check_recurrence_invariants,
    _check_special_layer_lists,
    _check_trace_self_consistency,
    check_func_call_id_invariant,
)


class _FakeTrace(SimpleNamespace):
    """``SimpleNamespace`` that also resolves ``self[label]`` like ``Trace``.

    Several checkers index the trace directly (``ml[label]``) rather than
    only reading plain attributes; a bare ``SimpleNamespace`` has no
    ``__getitem__``, so this thin subclass resolves a label against whatever
    ``layer_list`` the test populates (matching on either ``layer_label`` or
    the pass-qualified ``label``).
    """

    def __getitem__(self, label: str) -> Any:
        for candidate in getattr(self, "layer_list", ()):
            if getattr(candidate, "layer_label", None) == label:
                return candidate
            if getattr(candidate, "label", None) == label:
                return candidate
        raise KeyError(label)


class _AddrIndexed(list):
    """List of module-like fakes also indexable by ``.address`` (mirrors ``ml.modules``)."""

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            for item in self:
                if getattr(item, "address", None) == key:
                    return item
            raise KeyError(key)
        return list.__getitem__(self, key)


# ---------------------------------------------------------------------------
# backend_identity_invariants
# ---------------------------------------------------------------------------


def test_backend_identity_invariants_fires_on_missing_backend() -> None:
    """Arm 0: Trace.backend must be a non-empty string."""

    fake_trace = SimpleNamespace(backend="")
    with pytest.raises(MetadataInvariantError, match="must be a non-empty string"):
        _check_backend_identity_invariants(fake_trace)  # type: ignore[arg-type]


def test_backend_identity_invariants_fires_on_unregistered_backend() -> None:
    """Arm 1: an unregistered backend name must raise."""

    fake_trace = SimpleNamespace(backend="not_a_real_backend")
    with pytest.raises(MetadataInvariantError, match="is not registered"):
        _check_backend_identity_invariants(fake_trace)  # type: ignore[arg-type]


def test_backend_identity_invariants_fires_on_unsupported_module_identity_mode() -> None:
    """Arm 2: a module_identity_mode unsupported by the backend must raise."""

    fake_trace = SimpleNamespace(backend="torch", module_identity_mode="not_a_real_mode")
    with pytest.raises(MetadataInvariantError, match="is not supported by backend"):
        _check_backend_identity_invariants(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# backend_neutral_graph_topology
# ---------------------------------------------------------------------------


def test_backend_neutral_graph_topology_fires_on_parent_outside_labels() -> None:
    """Arm 0: a parent label outside the trace's labels must raise."""

    layer = SimpleNamespace(
        layer_label="relu_1", label="relu_1", parents=["missing_parent"], children=[]
    )
    fake_trace = _FakeTrace(layer_list=[layer])
    with pytest.raises(MetadataInvariantError, match="has parent .*outside trace labels"):
        _check_backend_neutral_graph_topology(fake_trace)  # type: ignore[arg-type]


def test_backend_neutral_graph_topology_fires_on_missing_reciprocal_child() -> None:
    """Arm 1: a parent that does not list this layer as a child must raise."""

    parent = SimpleNamespace(layer_label="conv_1", label="conv_1", parents=[], children=[])
    child = SimpleNamespace(layer_label="relu_1", label="relu_1", parents=["conv_1"], children=[])
    fake_trace = _FakeTrace(layer_list=[parent, child])
    with pytest.raises(MetadataInvariantError, match="reciprocal child is missing"):
        _check_backend_neutral_graph_topology(fake_trace)  # type: ignore[arg-type]


def test_backend_neutral_graph_topology_fires_on_child_outside_labels() -> None:
    """Arm 2: a child label outside the trace's labels must raise."""

    layer = SimpleNamespace(
        layer_label="relu_1", label="relu_1", parents=[], children=["missing_child"]
    )
    fake_trace = _FakeTrace(layer_list=[layer])
    with pytest.raises(MetadataInvariantError, match="has child .*outside trace labels"):
        _check_backend_neutral_graph_topology(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# buffer_xrefs
# ---------------------------------------------------------------------------


def test_buffer_xrefs_fires_on_buffer_layer_outside_layer_labels() -> None:
    """Arm 0: buffer_layers containing a label outside layer_labels must raise."""

    fake_trace = SimpleNamespace(buffer_layers=["ghost_buffer"], layer_labels=[])
    with pytest.raises(MetadataInvariantError, match="buffer_layers contains"):
        _check_buffer_xrefs(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# distance_invariants
# ---------------------------------------------------------------------------


def test_distance_invariants_fires_on_inverted_output_distance_bounds() -> None:
    """Arm 1: min_distance_to_output must not exceed max_distance_to_output."""

    layer = SimpleNamespace(
        layer_label="relu_1",
        min_distance_from_input=None,
        max_distance_from_input=None,
        min_distance_to_output=5,
        max_distance_to_output=2,
    )
    fake_trace = SimpleNamespace(
        mark_layer_depths=True, input_layers=[], output_layers=[], layer_list=[layer]
    )
    with pytest.raises(MetadataInvariantError, match="min_distance_to_output=5"):
        _check_distance_invariants(fake_trace)  # type: ignore[arg-type]


def test_distance_invariants_fires_on_output_descendant_flag_mismatch() -> None:
    """Arm 5: has_output_descendant must match whether output_descendants is non-empty."""

    layer = SimpleNamespace(
        layer_label="relu_1",
        min_distance_from_input=0,
        max_distance_from_input=0,
        min_distance_to_output=0,
        max_distance_to_output=0,
        has_input_ancestor=False,
        input_ancestors=set(),
        has_output_descendant=True,
        output_descendants=set(),
    )
    fake_trace = SimpleNamespace(
        mark_layer_depths=True, input_layers=[], output_layers=[], layer_list=[layer]
    )
    with pytest.raises(MetadataInvariantError, match="has_output_descendant=True"):
        _check_distance_invariants(fake_trace)  # type: ignore[arg-type]


def test_distance_invariants_fires_on_input_ancestors_outside_input_layers() -> None:
    """Arm 6: input_ancestors must be a subset of input_layers."""

    layer = SimpleNamespace(
        layer_label="relu_1",
        min_distance_from_input=0,
        max_distance_from_input=0,
        min_distance_to_output=0,
        max_distance_to_output=0,
        has_input_ancestor=True,
        input_ancestors={"ghost_input"},
        has_output_descendant=False,
        output_descendants=set(),
    )
    fake_trace = SimpleNamespace(
        mark_layer_depths=True, input_layers=[], output_layers=[], layer_list=[layer]
    )
    with pytest.raises(MetadataInvariantError, match="input_ancestors contains labels not in"):
        _check_distance_invariants(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# edge_use_parent_arg_consistency
# ---------------------------------------------------------------------------


def test_edge_use_parent_arg_consistency_fires_on_invalid_arg_kind() -> None:
    """Arm 1: an edge-use record's arg_kind must be a known domain value."""

    record = SimpleNamespace(
        edge_use="arg", arg_kind="not_a_real_kind", parent_label="p", child_label="c"
    )
    layer = SimpleNamespace(layer_label="relu_1", _edge_uses=[record], parent_arg_positions={})
    fake_trace = SimpleNamespace(layer_list=[layer], layer_dict_all_keys={})
    with pytest.raises(MetadataInvariantError, match="invalid edge arg_kind"):
        _check_edge_use_parent_arg_invariants(fake_trace)  # type: ignore[arg-type]


def test_edge_use_parent_arg_consistency_fires_on_unresolved_edge_use_parent() -> None:
    """Arm 2: an edge-use record's parent label must resolve."""

    record = SimpleNamespace(
        edge_use="arg", arg_kind="positional", parent_label="ghost_parent", child_label="c"
    )
    layer = SimpleNamespace(layer_label="relu_1", _edge_uses=[record], parent_arg_positions={})
    fake_trace = SimpleNamespace(layer_list=[layer], layer_dict_all_keys={})
    with pytest.raises(MetadataInvariantError, match="unresolved parent"):
        _check_edge_use_parent_arg_invariants(fake_trace)  # type: ignore[arg-type]


def test_edge_use_parent_arg_consistency_fires_on_non_mapping_parent_arg_positions() -> None:
    """Arm 4: parent_arg_positions must be a mapping."""

    layer = SimpleNamespace(
        layer_label="relu_1", _edge_uses=[], parent_arg_positions=["not", "a", "map"]
    )
    fake_trace = SimpleNamespace(layer_list=[layer], layer_dict_all_keys={})
    with pytest.raises(MetadataInvariantError, match="non-mapping parent_arg_positions"):
        _check_edge_use_parent_arg_invariants(fake_trace)  # type: ignore[arg-type]


def test_edge_use_parent_arg_consistency_fires_on_non_mapping_arg_domain_entries() -> None:
    """Arm 5: parent_arg_positions['args']/['kwargs'] must each be a mapping."""

    layer = SimpleNamespace(
        layer_label="relu_1", _edge_uses=[], parent_arg_positions={"args": ["not", "a", "map"]}
    )
    fake_trace = SimpleNamespace(layer_list=[layer], layer_dict_all_keys={})
    with pytest.raises(MetadataInvariantError, match="is not a mapping"):
        _check_edge_use_parent_arg_invariants(fake_trace)  # type: ignore[arg-type]


def test_edge_use_parent_arg_consistency_fires_on_non_string_arg_position_value() -> None:
    """Arm 6: a parent_arg_positions entry value must be a label string."""

    layer = SimpleNamespace(
        layer_label="relu_1", _edge_uses=[], parent_arg_positions={"args": {0: 12345}}
    )
    fake_trace = SimpleNamespace(layer_list=[layer], layer_dict_all_keys={})
    with pytest.raises(MetadataInvariantError, match="is not a label string"):
        _check_edge_use_parent_arg_invariants(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# equivalence_symmetry
# ---------------------------------------------------------------------------


def test_equivalence_symmetry_fires_on_non_set_equivalence_class_value() -> None:
    """Arm 0: op_equivalence_classes values must be sets."""

    fake_trace = SimpleNamespace(
        op_labels=["relu_1"],
        op_equivalence_classes={"type_a": ["relu_1"]},
        layer_list=[],
        layer_logs={},
    )
    with pytest.raises(MetadataInvariantError, match="is not a set"):
        _check_equivalence_symmetry(fake_trace)  # type: ignore[arg-type]


def test_equivalence_symmetry_fires_on_whole_trace_label_cross_check() -> None:
    """Arm 2: every equivalence-class label is re-checked across the whole mapping.

    The redundant whole-trace scan reads ``.values()`` separately from the
    per-entry scan's ``.items()``; this fake's container exposes an EXTRA
    label only through ``.values()``, proving the second scan has real teeth
    if a non-dict equivalence-class container ever let the two accessors
    observe different snapshots.
    """

    class _ShiftingEquivalenceClasses(dict):
        def values(self):  # type: ignore[override]
            return [set(v) | {"ghost_op"} for v in dict.values(self)]

    fake_trace = SimpleNamespace(
        op_labels=["relu_1"],
        op_equivalence_classes=_ShiftingEquivalenceClasses({"type_a": {"relu_1"}}),
        layer_list=[],
        layer_logs={},
    )
    with pytest.raises(MetadataInvariantError, match="contains labels not in op_labels"):
        _check_equivalence_symmetry(fake_trace)  # type: ignore[arg-type]


def test_equivalence_symmetry_fires_on_non_set_op_equivalent_ops() -> None:
    """Arm 3: Op.equivalent_ops must be a set or frozenset."""

    op = SimpleNamespace(label="op_1_1", equivalence_class=None, equivalent_ops=["op_1_1"])
    fake_trace = SimpleNamespace(
        op_labels=["op_1_1"], op_equivalence_classes={}, layer_list=[op], layer_logs={}
    )
    with pytest.raises(MetadataInvariantError, match="equivalent_ops is not a set"):
        _check_equivalence_symmetry(fake_trace)  # type: ignore[arg-type]


def test_equivalence_symmetry_fires_on_op_equivalent_ops_outside_op_labels() -> None:
    """Arm 4: Op.equivalent_ops members must be valid op labels."""

    op = SimpleNamespace(label="op_1_1", equivalence_class=None, equivalent_ops={"ghost_op"})
    fake_trace = SimpleNamespace(
        op_labels=["op_1_1"], op_equivalence_classes={}, layer_list=[op], layer_logs={}
    )
    with pytest.raises(MetadataInvariantError, match=r"op_1_1\.equivalent_ops contains"):
        _check_equivalence_symmetry(fake_trace)  # type: ignore[arg-type]


def test_equivalence_symmetry_fires_on_non_set_layer_equivalent_ops() -> None:
    """Arm 6: Layer.equivalent_ops must be a set or frozenset."""

    layer = SimpleNamespace(layer_label="layer_1", ops={}, equivalent_ops=["not", "a", "set"])
    fake_trace = SimpleNamespace(
        op_labels=[], op_equivalence_classes={}, layer_list=[], layer_logs={"layer_1": layer}
    )
    with pytest.raises(MetadataInvariantError, match="Layer layer_1.equivalent_ops is not a set"):
        _check_equivalence_symmetry(fake_trace)  # type: ignore[arg-type]


def test_equivalence_symmetry_fires_on_layer_equivalent_ops_outside_op_labels() -> None:
    """Arm 7: Layer.equivalent_ops members must be valid op labels."""

    layer = SimpleNamespace(layer_label="layer_1", ops={}, equivalent_ops={"ghost_op"})
    fake_trace = SimpleNamespace(
        op_labels=[], op_equivalence_classes={}, layer_list=[], layer_logs={"layer_1": layer}
    )
    with pytest.raises(MetadataInvariantError, match="Layer layer_1.equivalent_ops contains"):
        _check_equivalence_symmetry(fake_trace)  # type: ignore[arg-type]


def test_equivalence_symmetry_fires_on_pass_disagreement() -> None:
    """Arm 8: every pass of a Layer must agree on equivalent_ops."""

    op1 = SimpleNamespace(equivalent_ops={"op_1_1"})
    op2 = SimpleNamespace(equivalent_ops={"op_1_2"})
    layer = SimpleNamespace(layer_label="layer_1", ops={1: op1, 2: op2}, equivalent_ops={"op_1_1"})
    fake_trace = SimpleNamespace(
        op_labels=["op_1_1", "op_1_2"],
        op_equivalence_classes={},
        layer_list=[],
        layer_logs={"layer_1": layer},
    )
    with pytest.raises(MetadataInvariantError, match="ops disagree on equivalent_ops"):
        _check_equivalence_symmetry(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# func_call_id_consistency
# ---------------------------------------------------------------------------


def test_func_call_id_consistency_fires_on_non_integer_func_call_id() -> None:
    """Arm 1: a populated func_call_id must be an int."""

    layer = SimpleNamespace(
        layer_label="relu_1",
        is_input=False,
        is_output=False,
        is_buffer=False,
        func_name="relu",
        func_call_id="not_an_int",
    )
    fake_trace = SimpleNamespace(layer_list=[layer])
    with pytest.raises(MetadataInvariantError, match="non-integer func_call_id"):
        check_func_call_id_invariant(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# graph_connectivity
# ---------------------------------------------------------------------------


def test_graph_connectivity_fires_on_internal_source_consuming_unattributed_data() -> None:
    """Arm 1: an internal-source op that consumed unattributed tensor data must raise."""

    layer = SimpleNamespace(
        layer_label="mystery_op",
        label="mystery_op",
        is_internal_source=True,
        parents=[],
        func=lambda: None,
        is_output=False,
        unattributed_tensor_args=("ghost_tensor",),
    )
    fake_trace = SimpleNamespace(
        layer_labels=["mystery_op"],
        input_layers=[],
        buffer_layers=[],
        layer_list=[layer],
        _orphan_labels=[],
    )
    with pytest.raises(MetadataInvariantError, match="consumed unattributed tensor data"):
        _check_graph_connectivity(fake_trace)  # type: ignore[arg-type]


def test_graph_connectivity_fires_on_leaked_retained_orphan() -> None:
    """Arm 4: a retained orphan island's label must not leak into active labels."""

    orphan = SimpleNamespace(label="retained_op", layer_label="retained_op", is_orphan=True)
    fake_trace = SimpleNamespace(
        layer_labels=["retained_op"],
        input_layers=[],
        buffer_layers=[],
        layer_list=[],
        _orphan_labels=[],
        _orphan_logs=[orphan],
        op_labels=["retained_op"],
    )
    with pytest.raises(MetadataInvariantError, match="Retained orphan labels survive"):
        _check_graph_connectivity(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# graph_ordering
# ---------------------------------------------------------------------------


def test_graph_ordering_fires_on_duplicate_raw_index() -> None:
    """Arm 0: raw_index must be unique across layers."""

    layer1 = SimpleNamespace(layer_label="a", raw_index=1)
    layer2 = SimpleNamespace(layer_label="b", raw_index=1)
    fake_trace = SimpleNamespace(layer_list=[layer1, layer2])
    with pytest.raises(MetadataInvariantError, match="Duplicate raw_index"):
        _check_graph_ordering(fake_trace)  # type: ignore[arg-type]


def test_graph_ordering_fires_on_non_monotonic_raw_index() -> None:
    """Arm 1: raw_index must increase monotonically in layer_list order."""

    layer1 = SimpleNamespace(layer_label="a", raw_index=5)
    layer2 = SimpleNamespace(layer_label="b", raw_index=3)
    fake_trace = SimpleNamespace(layer_list=[layer1, layer2])
    with pytest.raises(MetadataInvariantError, match="not monotonically increasing"):
        _check_graph_ordering(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# graph_topology
# ---------------------------------------------------------------------------


def _topology_layer(**overrides: Any) -> SimpleNamespace:
    """Return a minimally-valid graph_topology layer fake, overridden per test."""

    base = {
        "pass_index": 1,
        "num_passes": 1,
        "parents": [],
        "children": [],
        "has_children": False,
        "parent_arg_positions": {},
        "is_input": False,
        "out_versions_by_child": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_graph_topology_fires_on_parent_label_not_canonical() -> None:
    """Arm 0: a stored parent label must exact-match a canonical label."""

    layer = _topology_layer(layer_label="relu_1", parents=["ghost_parent"])
    fake_trace = _FakeTrace(
        layer_labels=["relu_1"], op_labels=["relu_1"], output_layers=[], layer_list=[layer]
    )
    with pytest.raises(MetadataInvariantError, match="not a canonical layer or op label"):
        _check_graph_topology(fake_trace)  # type: ignore[arg-type]


def test_graph_topology_fires_on_missing_reciprocal_parent_for_child() -> None:
    """Arm 3: a listed child must list this layer back as a parent."""

    parent = _topology_layer(layer_label="conv_1", children=["relu_1"], has_children=True)
    child = _topology_layer(layer_label="relu_1")
    fake_trace = _FakeTrace(
        layer_labels=["conv_1", "relu_1"],
        op_labels=["conv_1", "relu_1"],
        output_layers=[],
        layer_list=[parent, child],
    )
    with pytest.raises(MetadataInvariantError, match="does not list .* as parent"):
        _check_graph_topology(fake_trace)  # type: ignore[arg-type]


def test_graph_topology_fires_on_has_children_flag_mismatch() -> None:
    """Arm 4: has_children must match whether non-output children exist."""

    child = _topology_layer(layer_label="relu_1", parents=["conv_1"])
    parent = _topology_layer(layer_label="conv_1", children=["relu_1"], has_children=False)
    fake_trace = _FakeTrace(
        layer_labels=["conv_1", "relu_1"],
        op_labels=["conv_1", "relu_1"],
        output_layers=[],
        layer_list=[parent, child],
    )
    with pytest.raises(MetadataInvariantError, match="has_children="):
        _check_graph_topology(fake_trace)  # type: ignore[arg-type]


def test_graph_topology_fires_on_foreign_parent_arg_position_domain() -> None:
    """Arm 5: parent_arg_positions must only use 'args'/'kwargs' top-level keys."""

    layer = _topology_layer(layer_label="relu_1", parent_arg_positions={"ghost_domain": {}})
    fake_trace = _FakeTrace(
        layer_labels=["relu_1"], op_labels=["relu_1"], output_layers=[], layer_list=[layer]
    )
    with pytest.raises(MetadataInvariantError, match="foreign top-level"):
        _check_graph_topology(fake_trace)  # type: ignore[arg-type]


def test_graph_topology_fires_on_input_layer_with_parents() -> None:
    """Arm 7: an input layer must have no parents."""

    parent = _topology_layer(layer_label="conv_1", children=["input_1"], has_children=True)
    layer = _topology_layer(layer_label="input_1", parents=["conv_1"], is_input=True)
    fake_trace = _FakeTrace(
        layer_labels=["conv_1", "input_1"],
        op_labels=["conv_1", "input_1"],
        output_layers=[],
        layer_list=[parent, layer],
    )
    with pytest.raises(MetadataInvariantError, match="has parents="):
        _check_graph_topology(fake_trace)  # type: ignore[arg-type]


def test_graph_topology_fires_on_out_versions_by_child_outside_children() -> None:
    """Arm 8: out_versions_by_child keys must be a subset of children."""

    layer = _topology_layer(layer_label="relu_1", out_versions_by_child={"ghost_child": 1})
    fake_trace = _FakeTrace(
        layer_labels=["relu_1"], op_labels=["relu_1"], output_layers=[], layer_list=[layer]
    )
    with pytest.raises(MetadataInvariantError, match="out_versions_by_child has keys"):
        _check_graph_topology(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# layer_pass_layer_log_xrefs
# ---------------------------------------------------------------------------


def test_layer_pass_layer_log_xrefs_fires_on_pass_index_mismatch() -> None:
    """Arm 2: an Op's pass_index must match its dict key in Layer.ops."""

    op = SimpleNamespace(pass_index=2, layer_label="relu_1")
    layer = SimpleNamespace(layer_label="relu_1", num_passes=1, ops={1: op})
    fake_trace = SimpleNamespace(layer_logs={"relu_1": layer})
    with pytest.raises(MetadataInvariantError, match=r"pass key=1 but Op.pass_index=2"):
        _check_layer_pass_to_layer_log_xrefs(fake_trace)  # type: ignore[arg-type]


def test_layer_pass_layer_log_xrefs_fires_on_op_layer_label_mismatch() -> None:
    """Arm 3: an Op's layer_label must match its parent Layer's layer_label."""

    op = SimpleNamespace(pass_index=1, layer_label="wrong_label")
    layer = SimpleNamespace(layer_label="relu_1", num_passes=1, ops={1: op})
    fake_trace = SimpleNamespace(layer_logs={"relu_1": layer})
    with pytest.raises(MetadataInvariantError, match="parent Layer.layer_label"):
        _check_layer_pass_to_layer_log_xrefs(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# lookup_key_consistency
# ---------------------------------------------------------------------------


def test_lookup_key_consistency_fires_on_forward_num_missing_from_reverse() -> None:
    """Arm 0: a forward key's layer num must exist in the reverse map."""

    fake_trace = SimpleNamespace(
        _lookup_keys_to_layer_num_dict={"k1": 5},
        _layer_num_to_lookup_keys_dict={},
        _raw_to_final_layer_labels={},
        _final_to_raw_layer_labels={},
        layer_labels=[],
        op_labels=[],
    )
    with pytest.raises(MetadataInvariantError, match="not in _layer_num_to_lookup_keys_dict"):
        _check_lookup_key_consistency(fake_trace)  # type: ignore[arg-type]


def test_lookup_key_consistency_fires_on_reverse_key_missing_from_forward() -> None:
    """Arm 2: a reverse-mapped key must exist in the forward map."""

    fake_trace = SimpleNamespace(
        _lookup_keys_to_layer_num_dict={},
        _layer_num_to_lookup_keys_dict={5: ["k1"]},
        _raw_to_final_layer_labels={},
        _final_to_raw_layer_labels={},
        layer_labels=[],
        op_labels=[],
    )
    with pytest.raises(MetadataInvariantError, match="not in _lookup_keys_to_layer_num_dict"):
        _check_lookup_key_consistency(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# loop_detection_invariants
# ---------------------------------------------------------------------------


def test_loop_detection_invariants_fires_on_shortcut_func_mismatch() -> None:
    """Arm 0: a later pass sharing an already-validated group must still agree on func_name."""

    layer1 = SimpleNamespace(
        label="relu_1:1",
        layer_label="relu_1",
        recurrent_ops=["relu_1:1", "relu_1:2"],
        equivalence_class="type_a",
        is_input=True,
        is_buffer=False,
        is_output=False,
        func_name="relu",
        num_passes=2,
        pass_index=1,
    )
    layer2 = SimpleNamespace(
        label="relu_1:2",
        layer_label="relu_1",
        recurrent_ops=["relu_1:1", "relu_1:2"],
        equivalence_class="type_a",
        is_input=False,
        is_buffer=False,
        is_output=False,
        func_name="sigmoid",
        num_passes=2,
        pass_index=2,
    )
    fake_trace = SimpleNamespace(
        layer_list=[layer1, layer2],
        op_labels=["relu_1:1", "relu_1:2"],
        layer_logs={},
        layer_dict_all_keys={"relu_1:1": layer1, "relu_1:2": layer2},
        op_equivalence_classes={},
    )
    with pytest.raises(MetadataInvariantError, match="recurrent_ops func mismatch"):
        _check_loop_detection_invariants(fake_trace)  # type: ignore[arg-type]


def test_loop_detection_invariants_fires_on_shortcut_num_passes_mismatch() -> None:
    """Arm 1: a later pass sharing an already-validated group must match num_passes."""

    layer1 = SimpleNamespace(
        label="relu_1:1",
        layer_label="relu_1",
        recurrent_ops=["relu_1:1", "relu_1:2"],
        equivalence_class="type_a",
        is_input=False,
        is_buffer=False,
        is_output=False,
        func_name="relu",
        num_passes=2,
        pass_index=1,
    )
    layer2 = SimpleNamespace(
        label="relu_1:2",
        layer_label="relu_1",
        recurrent_ops=["relu_1:1", "relu_1:2"],
        equivalence_class="type_a",
        is_input=False,
        is_buffer=False,
        is_output=False,
        func_name="relu",
        num_passes=99,
        pass_index=2,
    )
    fake_trace = SimpleNamespace(
        layer_list=[layer1, layer2],
        op_labels=["relu_1:1", "relu_1:2"],
        layer_logs={},
        layer_dict_all_keys={"relu_1:1": layer1, "relu_1:2": layer2},
        op_equivalence_classes={},
    )
    with pytest.raises(MetadataInvariantError, match="num_passes=99"):
        _check_loop_detection_invariants(fake_trace)  # type: ignore[arg-type]


def test_loop_detection_invariants_fires_on_unknown_recurrent_ops_member() -> None:
    """Arm 3: every recurrent_ops member must be a known op label."""

    op = SimpleNamespace(label="relu_1", layer_label="relu_1", recurrent_ops=["relu_1", "ghost_op"])
    fake_trace = SimpleNamespace(
        layer_list=[op],
        op_labels=["relu_1"],
        layer_logs={},
        layer_dict_all_keys={},
        op_equivalence_classes={},
    )
    with pytest.raises(MetadataInvariantError, match="not in op_labels"):
        _check_loop_detection_invariants(fake_trace)  # type: ignore[arg-type]


def test_loop_detection_invariants_fires_on_asymmetric_recurrent_ops() -> None:
    """Arm 5: every recurrent_ops member must report the identical group back."""

    anchor = SimpleNamespace(
        label="relu_1", layer_label="relu_1", recurrent_ops=["relu_1", "relu_2"]
    )
    other = SimpleNamespace(label="relu_2", layer_label="relu_1", recurrent_ops=["relu_2"])
    fake_trace = SimpleNamespace(
        layer_list=[anchor],
        op_labels=["relu_1", "relu_2"],
        layer_logs={},
        layer_dict_all_keys={"relu_1": anchor, "relu_2": other},
        op_equivalence_classes={},
    )
    with pytest.raises(MetadataInvariantError, match="Asymmetric recurrent_ops"):
        _check_loop_detection_invariants(fake_trace)  # type: ignore[arg-type]


def test_loop_detection_invariants_fires_on_equivalence_class_mismatch() -> None:
    """Arm 7: every recurrent_ops member must share the anchor's equivalence_class."""

    anchor = SimpleNamespace(
        label="relu_1",
        layer_label="relu_1",
        recurrent_ops=["relu_1", "relu_2"],
        equivalence_class="type_a",
    )
    other = SimpleNamespace(
        label="relu_2",
        layer_label="relu_1",
        recurrent_ops=["relu_1", "relu_2"],
        equivalence_class="type_b",
    )
    fake_trace = SimpleNamespace(
        layer_list=[anchor],
        op_labels=["relu_1", "relu_2"],
        layer_logs={},
        layer_dict_all_keys={"relu_1": anchor, "relu_2": other},
        op_equivalence_classes={},
    )
    with pytest.raises(MetadataInvariantError, match="type mismatch"):
        _check_loop_detection_invariants(fake_trace)  # type: ignore[arg-type]


def test_loop_detection_invariants_fires_on_full_path_func_mismatch() -> None:
    """Arm 8: a computational anchor's recurrent_ops members must share func_name."""

    anchor = SimpleNamespace(
        label="relu_1",
        layer_label="relu_1",
        recurrent_ops=["relu_1", "relu_2"],
        equivalence_class="type_a",
        is_input=False,
        is_buffer=False,
        is_output=False,
        func_name="relu",
    )
    other = SimpleNamespace(
        label="relu_2",
        layer_label="relu_1",
        recurrent_ops=["relu_1", "relu_2"],
        equivalence_class="type_a",
        func_name="sigmoid",
    )
    fake_trace = SimpleNamespace(
        layer_list=[anchor],
        op_labels=["relu_1", "relu_2"],
        layer_logs={},
        layer_dict_all_keys={"relu_1": anchor, "relu_2": other},
        op_equivalence_classes={},
    )
    with pytest.raises(MetadataInvariantError, match="recurrent_ops func mismatch"):
        _check_loop_detection_invariants(fake_trace)  # type: ignore[arg-type]


def test_loop_detection_invariants_fires_on_cross_pass_equivalence_type_drift() -> None:
    """Arm 12: all recurrent_ops group members must share one equivalence type."""

    anchor = SimpleNamespace(
        label="relu_1",
        layer_label="relu_1",
        recurrent_ops=["relu_1", "relu_2"],
        equivalence_class="type_a",
        is_input=False,
        is_buffer=False,
        is_output=False,
        func_name="relu",
        num_passes=2,
        pass_index=1,
        uses_params=False,
    )
    other = SimpleNamespace(
        label="relu_2",
        layer_label="relu_1",
        recurrent_ops=["relu_1", "relu_2"],
        equivalence_class="type_a",
        func_name="relu",
        pass_index=2,
        uses_params=False,
    )
    fake_trace = SimpleNamespace(
        layer_list=[anchor],
        op_labels=["relu_1", "relu_2"],
        layer_logs={},
        layer_dict_all_keys={"relu_1": anchor, "relu_2": other},
        op_equivalence_classes={"relu_type_x": {"relu_1"}, "sigmoid_type_y": {"relu_2"}},
    )
    with pytest.raises(MetadataInvariantError, match="spans multiple equivalence types"):
        _check_loop_detection_invariants(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# module_containment_logic
# ---------------------------------------------------------------------------


def test_module_containment_logic_fires_on_leaf_module_mismatch() -> None:
    """Arm 3: a layer's last nested module must match its module field."""

    mod_self = SimpleNamespace(
        address="self", address_parent=None, address_depth=0, address_children=[]
    )
    layer = SimpleNamespace(layer_label="relu_1", modules=["encoder"], module="decoder")
    fake_trace = SimpleNamespace(modules=_AddrIndexed([mod_self]), layer_list=[layer])
    with pytest.raises(MetadataInvariantError, match="last nested module"):
        _check_module_containment_logic(fake_trace)  # type: ignore[arg-type]


def test_module_containment_logic_fires_on_unknown_nested_module_address() -> None:
    """Arm 4: every nested module path entry must be a known module address."""

    mod_self = SimpleNamespace(
        address="self", address_parent=None, address_depth=0, address_children=[]
    )
    layer = SimpleNamespace(layer_label="relu_1", modules=["ghost_module"], module="ghost_module")
    fake_trace = SimpleNamespace(modules=_AddrIndexed([mod_self]), layer_list=[layer])
    with pytest.raises(MetadataInvariantError, match="unknown module address"):
        _check_module_containment_logic(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# module_hierarchy
# ---------------------------------------------------------------------------


def test_module_hierarchy_fires_on_missing_root_module() -> None:
    """Arm 0: the module accessor must contain a 'self' root module."""

    fake_trace = SimpleNamespace(modules=_AddrIndexed([]))
    with pytest.raises(MetadataInvariantError, match="'self' module not found"):
        _check_module_hierarchy(fake_trace)  # type: ignore[arg-type]


def test_module_hierarchy_fires_on_missing_address_child_listing() -> None:
    """Arm 1: a module's address_parent must list it back in address_children."""

    mod_self = SimpleNamespace(
        address="self", address_parent=None, address_children=[], ops={}, num_calls=0
    )
    mod_encoder = SimpleNamespace(
        address="encoder",
        address_parent="self",
        address_children=[],
        has_multiple_addresses=False,
        ops={},
        num_calls=0,
    )
    fake_trace = SimpleNamespace(modules=_AddrIndexed([mod_self, mod_encoder]))
    with pytest.raises(MetadataInvariantError, match="doesn't list it in address_children"):
        _check_module_hierarchy(fake_trace)  # type: ignore[arg-type]


def test_module_hierarchy_fires_on_child_address_parent_mismatch() -> None:
    """Arm 2: a module's address_children must report it as their address_parent."""

    mod_self = SimpleNamespace(
        address="self", address_parent=None, address_children=["encoder"], ops={}, num_calls=0
    )
    mod_encoder = SimpleNamespace(
        address="encoder",
        address_parent="wrong_parent",
        address_children=[],
        has_multiple_addresses=False,
        ops={},
        num_calls=0,
    )
    fake_trace = SimpleNamespace(modules=_AddrIndexed([mod_self, mod_encoder]))
    with pytest.raises(MetadataInvariantError, match="child's address_parent="):
        _check_module_hierarchy(fake_trace)  # type: ignore[arg-type]


def test_module_hierarchy_fires_on_pass_key_mismatch() -> None:
    """Arm 4: a module's call pass keys must be exactly {1..num_calls}."""

    mod_self = SimpleNamespace(
        address="self",
        address_parent=None,
        address_children=[],
        ops={2: SimpleNamespace(call_parent=None, call_children=[])},
        num_calls=1,
    )
    fake_trace = SimpleNamespace(modules=_AddrIndexed([mod_self]))
    with pytest.raises(MetadataInvariantError, match="pass keys="):
        _check_module_hierarchy(fake_trace)  # type: ignore[arg-type]


def test_module_hierarchy_fires_on_unresolvable_call_parent() -> None:
    """Arm 5: a ModuleCall's call_parent must resolve in the module accessor."""

    mpl = SimpleNamespace(call_parent="ghost_parent", call_children=[])
    mod_self = SimpleNamespace(
        address="self", address_parent=None, address_children=[], ops={1: mpl}, num_calls=1
    )
    fake_trace = SimpleNamespace(modules=_AddrIndexed([mod_self]))
    with pytest.raises(MetadataInvariantError, match="call_parent="):
        _check_module_hierarchy(fake_trace)  # type: ignore[arg-type]


def test_module_hierarchy_fires_on_unresolvable_call_child() -> None:
    """Arm 6: a ModuleCall's call_children must all resolve in the module accessor."""

    mpl = SimpleNamespace(call_parent=None, call_children=["ghost_child"])
    mod_self = SimpleNamespace(
        address="self", address_parent=None, address_children=[], ops={1: mpl}, num_calls=1
    )
    fake_trace = SimpleNamespace(modules=_AddrIndexed([mod_self]))
    with pytest.raises(MetadataInvariantError, match="call_children"):
        _check_module_hierarchy(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# module_layer_containment
# ---------------------------------------------------------------------------


def test_module_layer_containment_fires_on_num_layers_mismatch() -> None:
    """Arm 1: Module.num_layers must equal len(layer_labels)."""

    mod = SimpleNamespace(address="self", layer_labels=["relu_1"], num_layers=5, ops={})
    fake_trace = SimpleNamespace(
        modules=_AddrIndexed([mod]),
        op_labels=["relu_1"],
        layer_labels=["relu_1"],
        layer_logs={"relu_1": SimpleNamespace()},
        layer_list=[],
    )
    with pytest.raises(MetadataInvariantError, match="num_layers=5"):
        _check_module_layer_containment(fake_trace)  # type: ignore[arg-type]


def test_module_layer_containment_fires_on_invalid_modulecall_op_label() -> None:
    """Arm 2: a ModuleCall's ops must be valid op or layer labels."""

    mcall = SimpleNamespace(ops=["ghost_op"], num_layers=1, input_layers=[], output_layers=[])
    mod = SimpleNamespace(address="self", layer_labels=[], num_layers=0, ops={1: mcall})
    fake_trace = SimpleNamespace(
        modules=_AddrIndexed([mod]),
        op_labels=[],
        layer_labels=[],
        layer_logs={},
        layer_list=[],
    )
    with pytest.raises(MetadataInvariantError, match="ops contains"):
        _check_module_layer_containment(fake_trace)  # type: ignore[arg-type]


def test_module_layer_containment_fires_on_modulecall_num_layers_mismatch() -> None:
    """Arm 3: ModuleCall.num_layers must equal len(ops)."""

    mcall = SimpleNamespace(ops=["relu_1"], num_layers=5, input_layers=[], output_layers=[])
    mod = SimpleNamespace(address="self", layer_labels=[], num_layers=0, ops={1: mcall})
    fake_trace = SimpleNamespace(
        modules=_AddrIndexed([mod]),
        op_labels=["relu_1"],
        layer_labels=[],
        layer_logs={},
        layer_list=[],
    )
    with pytest.raises(MetadataInvariantError, match="num_layers=5"):
        _check_module_layer_containment(fake_trace)  # type: ignore[arg-type]


def test_module_layer_containment_fires_on_unresolvable_layer_module() -> None:
    """Arm 5: a layer's module address must resolve in the module accessor."""

    layer = SimpleNamespace(layer_label="relu_1", module="ghost_module")
    fake_trace = SimpleNamespace(
        modules=_AddrIndexed([]),
        op_labels=[],
        layer_labels=[],
        layer_logs={},
        layer_list=[layer],
    )
    with pytest.raises(MetadataInvariantError, match="not found in module accessor"):
        _check_module_layer_containment(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# non_torch_backward_inert
# ---------------------------------------------------------------------------


def test_non_torch_backward_inert_fires_on_grad_fn_order() -> None:
    """Arm 2: a non-torch trace must not populate grad_fn_order."""

    fake_trace = SimpleNamespace(has_backward_pass=False, grad_fn_logs=None, grad_fn_order=["x"])
    with pytest.raises(MetadataInvariantError, match="grad_fn_order"):
        _check_non_torch_backward_inert(fake_trace)  # type: ignore[arg-type]


def test_non_torch_backward_inert_fires_on_backward_pass_logs() -> None:
    """Arm 3: a non-torch trace must not populate backward_pass_logs."""

    fake_trace = SimpleNamespace(
        has_backward_pass=False, grad_fn_logs=None, grad_fn_order=None, backward_pass_logs=["x"]
    )
    with pytest.raises(MetadataInvariantError, match="backward_pass_logs"):
        _check_non_torch_backward_inert(fake_trace)  # type: ignore[arg-type]


def test_non_torch_backward_inert_fires_on_backward_root_grad_fn_object_ids() -> None:
    """Arm 4: a non-torch trace must not populate backward_root_grad_fn_object_ids."""

    fake_trace = SimpleNamespace(
        has_backward_pass=False,
        grad_fn_logs=None,
        grad_fn_order=None,
        backward_pass_logs=None,
        backward_root_grad_fn_object_ids=[123],
    )
    with pytest.raises(MetadataInvariantError, match="backward_root_grad_fn_object_ids"):
        _check_non_torch_backward_inert(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# non_torch_primitive_op_inert
# ---------------------------------------------------------------------------


def test_non_torch_primitive_op_inert_fires_on_primitive_op_store() -> None:
    """Arm 1: a non-torch trace's core must not carry a primitive_op kind-row store."""

    core = SimpleNamespace(kind_rows={"primitive_op": []}, backward_epochs=[])
    fake_trace = SimpleNamespace(_primitive_op_profile=None, _trace_core=core)
    with pytest.raises(MetadataInvariantError, match="primitive_op store"):
        _check_non_torch_primitive_op_inert(fake_trace)  # type: ignore[arg-type]


def test_non_torch_primitive_op_inert_fires_on_backward_primitive_op_store() -> None:
    """Arm 2: a non-torch trace's backward epochs must not carry a primitive_op store."""

    epoch = SimpleNamespace(stores={"primitive_op": []})
    core = SimpleNamespace(kind_rows={}, backward_epochs=[epoch])
    fake_trace = SimpleNamespace(_primitive_op_profile=None, _trace_core=core)
    with pytest.raises(MetadataInvariantError, match="backward primitive_op store"):
        _check_non_torch_primitive_op_inert(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# op_log_fields
# ---------------------------------------------------------------------------


def test_op_log_fields_fires_on_pass_index_below_one() -> None:
    """Arm 2: pass_index must be >= 1."""

    layer = SimpleNamespace(
        layer_label="relu_1", has_saved_activation=False, out=None, pass_index=0, num_passes=1
    )
    fake_trace = SimpleNamespace(layer_list=[layer], _loaded_from_bundle=False)
    with pytest.raises(MetadataInvariantError, match="pass_index=0"):
        _check_op_log_fields(fake_trace)  # type: ignore[arg-type]


def test_op_log_fields_fires_on_num_passes_below_pass_index() -> None:
    """Arm 3: num_passes must be >= pass_index."""

    layer = SimpleNamespace(
        layer_label="relu_1", has_saved_activation=False, out=None, pass_index=3, num_passes=1
    )
    fake_trace = SimpleNamespace(layer_list=[layer], _loaded_from_bundle=False)
    with pytest.raises(MetadataInvariantError, match="num_passes=1"):
        _check_op_log_fields(fake_trace)  # type: ignore[arg-type]


def test_op_log_fields_fires_on_empty_func_name() -> None:
    """Arm 6: a computational layer's func_name must be non-empty."""

    layer = SimpleNamespace(
        layer_label="relu_1",
        has_saved_activation=False,
        out=None,
        pass_index=1,
        num_passes=1,
        func_name="",
        intervention_replaced=False,
        is_internal_source=False,
        func=lambda: None,
        is_input=False,
        is_output=False,
        is_buffer=False,
    )
    fake_trace = SimpleNamespace(layer_list=[layer], _loaded_from_bundle=False)
    with pytest.raises(MetadataInvariantError, match="func_name is empty"):
        _check_op_log_fields(fake_trace)  # type: ignore[arg-type]


def test_op_log_fields_fires_on_raw_index_below_one() -> None:
    """Arm 9: raw_index must be >= 1."""

    layer = SimpleNamespace(
        layer_label="relu_1",
        has_saved_activation=False,
        out=None,
        pass_index=1,
        num_passes=1,
        func_name="relu",
        intervention_replaced=False,
        is_internal_source=False,
        func=lambda: None,
        is_input=False,
        is_output=False,
        is_buffer=False,
        step_index=1,
        raw_index=0,
        modules=(),
        module=None,
        module_call_stack=(),
        module_call_depth=0,
        label="relu_1",
    )
    fake_trace = SimpleNamespace(layer_list=[layer], _loaded_from_bundle=False)
    with pytest.raises(MetadataInvariantError, match="raw_index=0"):
        _check_op_log_fields(fake_trace)  # type: ignore[arg-type]


def test_op_log_fields_fires_on_multipass_label_without_colon() -> None:
    """Arm 13: a multi-pass op's label must carry a ':pass' suffix."""

    layer = SimpleNamespace(
        layer_label="relu_1",
        has_saved_activation=False,
        out=None,
        pass_index=1,
        num_passes=2,
        func_name="relu",
        intervention_replaced=False,
        is_internal_source=False,
        func=lambda: None,
        is_input=False,
        is_output=False,
        is_buffer=False,
        step_index=1,
        raw_index=1,
        modules=(),
        module=None,
        module_call_stack=(),
        module_call_depth=0,
        label="relu_1",
    )
    fake_trace = SimpleNamespace(layer_list=[layer], _loaded_from_bundle=False)
    with pytest.raises(MetadataInvariantError, match="has no ':'"):
        _check_op_log_fields(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# param_xrefs
# ---------------------------------------------------------------------------


def test_param_xrefs_fires_on_used_by_ops_outside_op_labels() -> None:
    """Arm 0: Param.used_by_ops must only name valid op labels."""

    param = SimpleNamespace(
        address="layer.weight",
        used_by_ops=["ghost_op"],
        used_by_layers=[],
        all_addresses={"layer.weight"},
    )
    fake_trace = SimpleNamespace(
        param_logs=[param], layer_labels=[], op_labels=[], layer_list=[], layers_with_params={}
    )
    with pytest.raises(MetadataInvariantError, match="used_by_ops contains"):
        _check_param_xrefs(fake_trace)  # type: ignore[arg-type]


def test_param_xrefs_fires_on_uses_params_without_param_logs() -> None:
    """Arm 3: a layer with uses_params=True must have non-empty _param_logs."""

    layer = SimpleNamespace(layer_label="relu_1", uses_params=True, _param_logs=[])
    fake_trace = SimpleNamespace(
        param_logs=[], layer_labels=[], op_labels=[], layer_list=[layer], layers_with_params={}
    )
    with pytest.raises(MetadataInvariantError, match="_param_logs is empty"):
        _check_param_xrefs(fake_trace)  # type: ignore[arg-type]


def test_param_xrefs_fires_on_layers_with_params_outside_layer_labels() -> None:
    """Arm 4: layers_with_params values must be valid layer labels."""

    fake_trace = SimpleNamespace(
        param_logs=[],
        layer_labels=[],
        op_labels=[],
        layer_list=[],
        layers_with_params={"layer.weight": ["ghost_layer"]},
    )
    with pytest.raises(MetadataInvariantError, match="contains 'ghost_layer'"):
        _check_param_xrefs(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# pass_count_consistency
# ---------------------------------------------------------------------------


def test_pass_count_consistency_fires_on_op_pass_index_mismatch() -> None:
    """Arm 1: a stored op's pass_index must match its key in Layer.ops."""

    op = SimpleNamespace(pass_index=2, num_passes=1, label="relu_1:1")
    layer_log = SimpleNamespace(ops={1: op}, num_passes=1)
    fake_trace = SimpleNamespace(layer_logs={"relu_1": layer_log}, layer_num_calls={})
    with pytest.raises(MetadataInvariantError, match="stores op with pass_index=2"):
        _check_pass_count_consistency(fake_trace)  # type: ignore[arg-type]


def test_pass_count_consistency_fires_on_op_num_passes_mismatch() -> None:
    """Arm 2: a stored op's num_passes must match its Layer's num_passes."""

    op = SimpleNamespace(pass_index=1, num_passes=99, label="relu_1:1")
    layer_log = SimpleNamespace(ops={1: op}, num_passes=1)
    fake_trace = SimpleNamespace(layer_logs={"relu_1": layer_log}, layer_num_calls={})
    with pytest.raises(MetadataInvariantError, match="num_passes=99"):
        _check_pass_count_consistency(fake_trace)  # type: ignore[arg-type]


def test_pass_count_consistency_fires_on_layer_num_calls_mismatch() -> None:
    """Arm 3: layer_num_calls must match Layer.num_passes."""

    layer_log = SimpleNamespace(ops={}, num_passes=0)
    fake_trace = SimpleNamespace(layer_logs={"relu_1": layer_log}, layer_num_calls={"relu_1": 5})
    with pytest.raises(MetadataInvariantError, match="Layer.num_passes=0"):
        _check_pass_count_consistency(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# recurrence_invariants
# ---------------------------------------------------------------------------


def test_recurrence_invariants_fires_on_is_recurrent_mismatch() -> None:
    """Arm 0: is_recurrent must match whether any layer has more than one pass."""

    fake_trace = SimpleNamespace(
        layer_num_calls={"relu_1": 3}, is_recurrent=False, layer_labels=["relu_1"], layer_logs={}
    )
    with pytest.raises(MetadataInvariantError, match="is_recurrent=False"):
        _check_recurrence_invariants(fake_trace)  # type: ignore[arg-type]


def test_recurrence_invariants_fires_on_max_layer_op_count_mismatch() -> None:
    """Arm 1: max_layer_op_count must equal max(layer_num_calls)."""

    fake_trace = SimpleNamespace(
        layer_num_calls={"relu_1": 3},
        is_recurrent=True,
        max_layer_op_count=99,
        layer_labels=["relu_1"],
        layer_logs={},
    )
    with pytest.raises(MetadataInvariantError, match="max_layer_op_count=99"):
        _check_recurrence_invariants(fake_trace)  # type: ignore[arg-type]


def test_recurrence_invariants_fires_on_unknown_layer_num_calls_key() -> None:
    """Arm 2: layer_num_calls keys must be valid layer_labels."""

    fake_trace = SimpleNamespace(
        layer_num_calls={"ghost_layer": 1}, is_recurrent=False, layer_labels=[], layer_logs={}
    )
    with pytest.raises(MetadataInvariantError, match="not in layer_labels"):
        _check_recurrence_invariants(fake_trace)  # type: ignore[arg-type]


def test_recurrence_invariants_fires_on_layer_log_pass_keys_mismatch() -> None:
    """Arm 4: a top-level Layer's ops keys must be exactly {1..num_passes}."""

    layer_log = SimpleNamespace(num_passes=2, ops={1: object()})
    fake_trace = SimpleNamespace(
        layer_num_calls={}, is_recurrent=False, layer_labels=[], layer_logs={"relu_1": layer_log}
    )
    with pytest.raises(MetadataInvariantError, match="ops keys="):
        _check_recurrence_invariants(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# special_layer_lists
# ---------------------------------------------------------------------------


def test_special_layer_lists_fires_on_input_layers_outside_layer_labels() -> None:
    """Arm 0: input_layers must only name valid layer labels."""

    layer = SimpleNamespace(layer_label="relu_1", label="relu_1", is_input=True)
    fake_trace = SimpleNamespace(
        input_layers=["ghost_input"],
        output_layers=[],
        buffer_layers=[],
        internal_source_ops=[],
        internal_sink_ops=[],
        op_labels=["relu_1"],
        layer_labels=["relu_1"],
        layer_list=[layer],
    )
    with pytest.raises(
        MetadataInvariantError, match="input_layers contains labels not in layer_labels"
    ):
        _check_special_layer_lists(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# trace_self_consistency
# ---------------------------------------------------------------------------


def test_trace_self_consistency_fires_on_op_labels_length_mismatch() -> None:
    """Arm 0: len(op_labels) must equal len(layer_list)."""

    fake_trace = SimpleNamespace(op_labels=["a", "b"], layer_list=[SimpleNamespace()])
    with pytest.raises(MetadataInvariantError, match=r"len\(op_labels\)=2"):
        _check_trace_self_consistency(fake_trace)  # type: ignore[arg-type]


def test_trace_self_consistency_fires_on_duplicate_op_labels() -> None:
    """Arm 1: op_labels must not contain duplicates."""

    fake_trace = SimpleNamespace(
        op_labels=["relu_1", "relu_1"], layer_list=[SimpleNamespace(), SimpleNamespace()]
    )
    with pytest.raises(MetadataInvariantError, match="Duplicate op_labels"):
        _check_trace_self_consistency(fake_trace)  # type: ignore[arg-type]


def test_trace_self_consistency_fires_on_num_param_tensors_mismatch() -> None:
    """Arm 3: num_param_tensors must equal len(param_logs)."""

    layer = SimpleNamespace(is_input=False, is_output=False, is_buffer=False)
    fake_trace = SimpleNamespace(
        op_labels=["relu_1"],
        layer_list=[layer],
        num_ops=1,
        param_logs=[],
        num_param_tensors=5,
        _orphan_logs=(),
    )
    with pytest.raises(MetadataInvariantError, match="num_param_tensors=5"):
        _check_trace_self_consistency(fake_trace)  # type: ignore[arg-type]


def test_trace_self_consistency_fires_on_num_params_mismatch() -> None:
    """Arm 4: num_params must equal sum(param_logs num_params)."""

    layer = SimpleNamespace(is_input=False, is_output=False, is_buffer=False)
    param = SimpleNamespace(num_params=10)
    fake_trace = SimpleNamespace(
        op_labels=["relu_1"],
        layer_list=[layer],
        num_ops=1,
        param_logs=[param],
        num_param_tensors=1,
        num_params=99,
        _orphan_logs=(),
    )
    with pytest.raises(MetadataInvariantError, match="num_params=99"):
        _check_trace_self_consistency(fake_trace)  # type: ignore[arg-type]


def test_trace_self_consistency_fires_on_trainable_frozen_sum_mismatch() -> None:
    """Arm 5: num_params_trainable + num_params_frozen must equal num_params."""

    layer = SimpleNamespace(is_input=False, is_output=False, is_buffer=False)
    fake_trace = SimpleNamespace(
        op_labels=["relu_1"],
        layer_list=[layer],
        num_ops=1,
        param_logs=[],
        num_param_tensors=0,
        num_params=0,
        num_params_trainable=3,
        num_params_frozen=4,
        _orphan_logs=(),
    )
    with pytest.raises(MetadataInvariantError, match="!= total"):
        _check_trace_self_consistency(fake_trace)  # type: ignore[arg-type]


def test_trace_self_consistency_fires_on_negative_capture_duration() -> None:
    """Arm 7: capture_duration must be non-negative."""

    layer = SimpleNamespace(is_input=False, is_output=False, is_buffer=False)
    fake_trace = SimpleNamespace(
        op_labels=["relu_1"],
        layer_list=[layer],
        num_ops=1,
        param_logs=[],
        num_param_tensors=0,
        num_params=0,
        num_params_trainable=0,
        num_params_frozen=0,
        output_layers=["relu_1"],
        capture_duration=-1,
        _orphan_logs=(),
    )
    with pytest.raises(MetadataInvariantError, match="capture_duration=-1"):
        _check_trace_self_consistency(fake_trace)  # type: ignore[arg-type]


def test_trace_self_consistency_fires_on_start_after_end_time() -> None:
    """Arm 8: capture_start_time must not exceed capture_end_time."""

    layer = SimpleNamespace(is_input=False, is_output=False, is_buffer=False)
    fake_trace = SimpleNamespace(
        op_labels=["relu_1"],
        layer_list=[layer],
        num_ops=1,
        param_logs=[],
        num_param_tensors=0,
        num_params=0,
        num_params_trainable=0,
        num_params_frozen=0,
        output_layers=["relu_1"],
        capture_duration=0,
        capture_start_time=5,
        capture_end_time=1,
        _orphan_logs=(),
    )
    with pytest.raises(MetadataInvariantError, match="capture_start_time=5"):
        _check_trace_self_consistency(fake_trace)  # type: ignore[arg-type]


def test_trace_self_consistency_fires_on_num_tensors_below_saved_ops() -> None:
    """Arm 9: num_tensors must be >= num_saved_ops."""

    layer = SimpleNamespace(is_input=False, is_output=False, is_buffer=False)
    fake_trace = SimpleNamespace(
        op_labels=["relu_1"],
        layer_list=[layer],
        num_ops=1,
        param_logs=[],
        num_param_tensors=0,
        num_params=0,
        num_params_trainable=0,
        num_params_frozen=0,
        output_layers=["relu_1"],
        capture_duration=0,
        capture_start_time=0,
        capture_end_time=1,
        num_tensors=0,
        num_saved_ops=5,
        _orphan_logs=(),
    )
    with pytest.raises(MetadataInvariantError, match="num_tensors=0"):
        _check_trace_self_consistency(fake_trace)  # type: ignore[arg-type]
