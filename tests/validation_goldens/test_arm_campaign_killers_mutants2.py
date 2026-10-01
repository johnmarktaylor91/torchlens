"""Direct per-arm killers for the M1 raise-arm mutation campaign (mutants2 lane).

Each test below targets exactly ONE ``raise MetadataInvariantError`` arm of its
checker (mutation_driver.py's per-arm family, id ``<contract>#aNN``) with a
duck-typed fake of the exact object the checker reads: every EARLIER arm must
stay silent on the fake so the mutant's own arm is what trips, matching the
project's "Validation Integrity (LOCKED)" doctrine (root-cause the real
violation path, never broaden a tolerance). See
``tests/validation_goldens/test_validation_exemption_hardening.py`` for the
sibling W3/W4/M1 killers this file continues; this module exists separately
so the mutants2 lane's additions are easy to re-run and review on their own.

Two tiny indexable fakes are needed because several checkers index the trace
or a module accessor directly (``ml[label]``, ``mod_accessor[address]``)
rather than only reading plain attributes, which a bare ``SimpleNamespace``
cannot support.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from torchlens.validation.invariants import (
    MetadataInvariantError,
    _check_backend_identity_invariants,
    _check_backend_neutral_accessor_refs,
    _check_distance_invariants,
    _check_edge_use_parent_arg_invariants,
    _check_equivalence_symmetry,
    _check_graph_ordering,
    _check_graph_topology,
    _check_lookup_key_consistency,
    _check_loop_detection_invariants,
    _check_module_containment_logic,
    _check_module_hierarchy,
    _check_module_layer_containment,
    _check_op_log_fields,
    _check_param_xrefs,
    _check_special_layer_lists,
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
# backend_identity_invariants / backend_neutral_accessor_refs
# ---------------------------------------------------------------------------


def test_backend_identity_invariants_fires_on_invalid_param_source() -> None:
    """Arm 3: an unrecognized ``param_source`` domain value must raise."""

    fake_trace = SimpleNamespace(
        backend="torch",
        module_identity_mode="torch_module",
        param_source="bogus_source",
        num_param_tensors=0,
    )

    with pytest.raises(MetadataInvariantError, match=r"param_source=.*is invalid"):
        _check_backend_identity_invariants(fake_trace)  # type: ignore[arg-type]


def test_backend_neutral_accessor_refs_fires_on_non_string_backend_address() -> None:
    """Arm 2: a populated ``backend_address`` must be a string."""

    record = SimpleNamespace(
        layer_label="relu_1_1",
        resolver_status="resolved",
        dtype_ref=None,
        device_ref=None,
        backend_address=12345,
    )
    fake_trace = SimpleNamespace(layer_list=[record], layer_logs={}, param_logs={})

    with pytest.raises(MetadataInvariantError, match="non-string backend_address"):
        _check_backend_neutral_accessor_refs(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# distance_invariants
# ---------------------------------------------------------------------------


def test_distance_invariants_fires_on_nonzero_output_distance() -> None:
    """Arm 3: an output layer must have distance_to_output == 0."""

    layer = SimpleNamespace(
        layer_label="out_1_1",
        min_distance_from_input=2,
        max_distance_from_input=2,
        min_distance_to_output=5,
        max_distance_to_output=5,
        has_input_ancestor=False,
        input_ancestors=frozenset(),
        has_output_descendant=False,
        output_descendants=frozenset(),
    )
    fake_trace = SimpleNamespace(
        mark_layer_depths=True,
        input_layers=[],
        output_layers=["out_1_1"],
        layer_list=[layer],
    )

    with pytest.raises(MetadataInvariantError, match="distance_from_output should be 0"):
        _check_distance_invariants(fake_trace)  # type: ignore[arg-type]


def test_distance_invariants_fires_on_output_descendant_outside_output_layers() -> None:
    """Arm 7: output_descendants must be a subset of output_layers.

    (M1 raise-arm campaign: ``distance_invariants#a07`` survivor -- the
    nonzero-output-distance killer above trips arm 3 first on any layer that
    is itself an output, so this arm needs a non-output layer whose
    descendant set names a label absent from ``output_layers``.)
    """

    layer = SimpleNamespace(
        layer_label="mid_1_1",
        min_distance_from_input=None,
        max_distance_from_input=None,
        min_distance_to_output=None,
        max_distance_to_output=None,
        has_input_ancestor=False,
        input_ancestors=frozenset(),
        has_output_descendant=True,
        output_descendants=frozenset({"ghost_output_9_9"}),
    )
    fake_trace = SimpleNamespace(
        mark_layer_depths=True,
        input_layers=[],
        output_layers=["real_output_1_1"],
        layer_list=[layer],
    )

    with pytest.raises(MetadataInvariantError, match="output_descendants contains labels not in"):
        _check_distance_invariants(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# edge_use_parent_arg_consistency
# ---------------------------------------------------------------------------


def test_edge_use_parent_arg_consistency_fires_on_unresolved_child_label() -> None:
    """Arm 3: an edge-use record's child label must resolve on the trace."""

    edge_record = SimpleNamespace(
        edge_use="arg",
        arg_kind="positional",
        parent_label="parent_1_1",
        child_label="unresolved_child_9_9",
    )
    layer = SimpleNamespace(
        layer_label="child_1_1",
        _edge_uses=[edge_record],
        parent_arg_positions={},
    )
    fake_trace = SimpleNamespace(
        layer_list=[layer],
        layer_dict_all_keys={"parent_1_1": True},
        _raw_to_final_layer_labels={},
    )

    with pytest.raises(MetadataInvariantError, match="unresolved child"):
        _check_edge_use_parent_arg_invariants(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# equivalence_symmetry
# ---------------------------------------------------------------------------


def test_equivalence_symmetry_fires_on_registry_label_outside_op_labels() -> None:
    """Arm 1: an op_equivalence_classes member must exist in op_labels."""

    fake_trace = SimpleNamespace(op_labels=[], op_equivalence_classes={"t": {"ghost_op"}})

    with pytest.raises(MetadataInvariantError, match=r"op_equivalence_classes\[.*\] contains"):
        _check_equivalence_symmetry(fake_trace)  # type: ignore[arg-type]


def test_equivalence_symmetry_fires_on_layer_pass_mismatch() -> None:
    """Arm 9 (last): a Layer's equivalent_ops must match its pass's equivalent_ops.

    (M1 raise-arm campaign: ``equivalence_symmetry#a09`` survivor -- every
    earlier arm (registry shape, per-Op checks) must stay silent, so the only
    corruption is at the LAYER level disagreeing with its own single pass.)
    """

    op = SimpleNamespace(label="op_1_1", equivalence_class=None, equivalent_ops={"op_1_1"})
    layer = SimpleNamespace(
        layer_label="layer_1_1",
        ops={1: op},
        equivalent_ops={"op_2_2"},
    )
    fake_trace = SimpleNamespace(
        op_labels=["op_1_1", "op_2_2"],
        op_equivalence_classes={},
        layer_list=[op],
        layer_logs={"layer_1_1": layer},
    )

    with pytest.raises(MetadataInvariantError, match="pass equivalent_ops"):
        _check_equivalence_symmetry(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# func_call_id_consistency
# ---------------------------------------------------------------------------


def test_func_call_id_consistency_fires_on_duplicate_container_path() -> None:
    """Arm 3: two group members must not share a populated container_path."""

    layer1 = SimpleNamespace(
        layer_label="relu_1_1",
        is_input=False,
        is_output=False,
        is_buffer=False,
        func_name="relu",
        is_internal_source=False,
        func_call_id=1,
        container_spec=None,
        container_path=("block", "0"),
    )
    layer2 = SimpleNamespace(
        layer_label="relu_1_2",
        is_input=False,
        is_output=False,
        is_buffer=False,
        func_name="relu",
        is_internal_source=False,
        func_call_id=1,
        container_spec=None,
        container_path=("block", "0"),
    )
    fake_trace = SimpleNamespace(layer_list=[layer1, layer2])

    with pytest.raises(MetadataInvariantError, match="duplicate container_path"):
        check_func_call_id_invariant(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# graph_ordering / graph_topology
# ---------------------------------------------------------------------------


def test_graph_ordering_fires_on_duplicate_step_index() -> None:
    """Arm 2: step_index must be unique among computational layers."""

    layer1 = SimpleNamespace(layer_label="op_1_1", raw_index=0, step_index=1, parents=[])
    layer2 = SimpleNamespace(layer_label="op_2_2", raw_index=1, step_index=1, parents=[])
    fake_trace = SimpleNamespace(
        layer_list=[layer1, layer2],
        input_layers=[],
        buffer_layers=[],
        output_layers=[],
    )

    with pytest.raises(MetadataInvariantError, match="Duplicate step_index"):
        _check_graph_ordering(fake_trace)  # type: ignore[arg-type]


def test_graph_topology_fires_on_unresolvable_child_label() -> None:
    """Arm 2: a stored child label must resolve to a canonical label."""

    layer = SimpleNamespace(
        layer_label="op_1_1",
        label="op_1_1",
        pass_index=None,
        num_passes=None,
        parents=[],
        children=["ghost_child_9_9"],
    )
    fake_trace = _FakeTrace(
        layer_list=[layer],
        layer_labels=["op_1_1"],
        op_labels=["op_1_1"],
        output_layers=[],
    )

    with pytest.raises(MetadataInvariantError, match="stores child label"):
        _check_graph_topology(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# lookup_key_consistency
# ---------------------------------------------------------------------------


def test_lookup_key_consistency_fires_on_missing_reverse_key() -> None:
    """Arm 1: a forward key must appear in its reverse bucket."""

    fake_trace = SimpleNamespace(
        _lookup_keys_to_layer_num_dict={"k1": 1},
        _layer_num_to_lookup_keys_dict={1: ["other_key"]},
    )

    with pytest.raises(MetadataInvariantError, match=r"not in _layer_num_to_lookup_keys_dict\["):
        _check_lookup_key_consistency(fake_trace)  # type: ignore[arg-type]


def test_lookup_key_consistency_fires_on_final_to_raw_missing_forward_entry() -> None:
    """Arm 5: every raw label in the reverse map must exist in the forward map.

    (M1 raise-arm campaign: ``lookup_key_consistency#a05`` survivor -- the
    forward-direction killer above only ever trips the earlier arms, so this
    needs an EMPTY forward lookup pair and a populated reverse-only entry.)
    """

    fake_trace = SimpleNamespace(
        _lookup_keys_to_layer_num_dict={},
        _layer_num_to_lookup_keys_dict={},
        _raw_to_final_layer_labels={},
        _final_to_raw_layer_labels={"final_1_1": "raw_ghost_1_1"},
    )

    with pytest.raises(MetadataInvariantError, match="not in _raw_to_final_layer_labels"):
        _check_lookup_key_consistency(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# loop_detection_invariants
# ---------------------------------------------------------------------------


def test_loop_detection_invariants_fires_on_empty_recurrent_ops() -> None:
    """Arm 2: a layer's recurrent_ops must be non-empty."""

    layer = SimpleNamespace(label="op_1_1", layer_label="layer_1_1", recurrent_ops=())
    fake_trace = SimpleNamespace(
        op_labels=[],
        layer_logs={},
        layer_dict_all_keys={},
        layer_list=[layer],
    )

    with pytest.raises(MetadataInvariantError, match="empty recurrent_ops"):
        _check_loop_detection_invariants(fake_trace)  # type: ignore[arg-type]


def test_loop_detection_invariants_fires_on_mismatched_member_layer_label() -> None:
    """Arm 6: every recurrent_ops member must share the anchor's layer_label.

    (M1 raise-arm campaign: ``loop_detection_invariants#a06`` survivor --
    membership, self-inclusion, and symmetry must all stay silent, so the
    group's two members agree on everything except ``layer_label``.)
    """

    slo = ["op_1_1", "op_2_1"]
    anchor = SimpleNamespace(
        label="op_1_1",
        layer_label="layer_A",
        recurrent_ops=slo,
        is_input=False,
        is_buffer=False,
        is_output=False,
        equivalence_class="eqA",
        func_name="relu",
    )
    member = SimpleNamespace(
        label="op_2_1",
        layer_label="layer_B",
        recurrent_ops=slo,
        equivalence_class="eqA",
        func_name="relu",
    )
    fake_trace = SimpleNamespace(
        op_labels=["op_1_1", "op_2_1"],
        layer_logs={"op_1_1": anchor, "op_2_1": member},
        layer_dict_all_keys={},
        layer_list=[anchor],
    )

    with pytest.raises(MetadataInvariantError, match="recurrent_ops inconsistency"):
        _check_loop_detection_invariants(fake_trace)  # type: ignore[arg-type]


def test_loop_detection_invariants_fires_on_noncontiguous_pass_numbering() -> None:
    """Arm 10: a recurrent group's pass_index set must be exactly {1..N}.

    (M1 raise-arm campaign: ``loop_detection_invariants#a10`` survivor --
    membership, symmetry, shared layer_label/equivalence_class/func_name, and
    the num_passes count must all stay silent, so the two members instead
    collide on the SAME pass_index.)
    """

    slo = ["op_1_1", "op_2_1"]
    anchor = SimpleNamespace(
        label="op_1_1",
        layer_label="layer_A",
        recurrent_ops=slo,
        is_input=False,
        is_buffer=False,
        is_output=False,
        equivalence_class="eqA",
        func_name="relu",
        num_passes=2,
        pass_index=1,
    )
    member = SimpleNamespace(
        label="op_2_1",
        layer_label="layer_A",
        recurrent_ops=slo,
        equivalence_class="eqA",
        func_name="relu",
        pass_index=1,
    )
    fake_trace = SimpleNamespace(
        op_labels=["op_1_1", "op_2_1"],
        layer_logs={"op_1_1": anchor, "op_2_1": member},
        layer_dict_all_keys={},
        layer_list=[anchor],
    )

    with pytest.raises(MetadataInvariantError, match="Pass numbering for group"):
        _check_loop_detection_invariants(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# module_containment_logic
# ---------------------------------------------------------------------------


def test_module_containment_logic_fires_on_wrong_root_depth() -> None:
    """Arm 1: the root module 'self' must have address_depth == 0."""

    root = SimpleNamespace(address="self", address_depth=5, address_parent=None)
    fake_trace = SimpleNamespace(modules=_AddrIndexed([root]))

    with pytest.raises(MetadataInvariantError, match="Root module 'self' has address_depth"):
        _check_module_containment_logic(fake_trace)  # type: ignore[arg-type]


def test_module_containment_logic_fires_on_duplicate_nested_module_address() -> None:
    """Arm 5: a layer's nested module path may not repeat an address.

    (M1 raise-arm campaign: ``module_containment_logic#a05`` survivor -- the
    address-tree checks (acyclic, depth) must stay silent, so the corruption
    is confined to one layer's call-nesting path repeating one address.)
    """

    root = SimpleNamespace(address="self", address_depth=0, address_parent=None)
    block = SimpleNamespace(address="self.block", address_depth=2, address_parent="self")
    layer = SimpleNamespace(
        layer_label="leaf_1_1",
        modules=["self.block:1", "self.block:1"],
        module="self.block:1",
    )
    fake_trace = SimpleNamespace(modules=_AddrIndexed([root, block]), layer_list=[layer])

    with pytest.raises(MetadataInvariantError, match="duplicate module address"):
        _check_module_containment_logic(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# module_hierarchy
# ---------------------------------------------------------------------------


def test_module_hierarchy_fires_on_pass_count_mismatch() -> None:
    """Arm 3: a module's recorded pass count must match len(ops)."""

    root = SimpleNamespace(
        address="self",
        address_parent=None,
        address_children=[],
        ops={1: SimpleNamespace()},
        num_calls=2,
    )
    fake_trace = SimpleNamespace(modules=_AddrIndexed([root]))

    with pytest.raises(MetadataInvariantError, match=r"len\(ops\)="):
        _check_module_hierarchy(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# module_layer_containment
# ---------------------------------------------------------------------------


def test_module_layer_containment_fires_on_layer_label_missing_from_layer_logs() -> None:
    """Arm 0: a module's layer_labels must exist in trace.layer_logs."""

    mod_log = SimpleNamespace(
        address="self.block",
        layer_labels=["ghost_layer_9_9"],
        num_layers=1,
        ops={},
    )
    fake_trace = SimpleNamespace(
        modules=_AddrIndexed([mod_log]),
        op_labels=[],
        layer_labels=[],
        layer_logs={},
        layer_list=[],
    )

    with pytest.raises(MetadataInvariantError, match="not in trace.layer_logs"):
        _check_module_layer_containment(fake_trace)  # type: ignore[arg-type]


def test_module_layer_containment_fires_on_output_layers_outside_known_labels() -> None:
    """Arm 4: a ModuleCall's input/output_layers must be known labels.

    (M1 raise-arm campaign: ``module_layer_containment#a04`` survivor --
    layer_labels resolution, the num_layers count, the ops-membership check,
    and the num_layers/len(ops) check must all stay silent, so only the
    output_layers roster names a label outside op_labels/layer_labels.)
    """

    mpl = SimpleNamespace(
        ops=["op_1_1"],
        num_layers=1,
        input_layers=["op_1_1"],
        output_layers=["ghost_out_9_9"],
    )
    mod_log = SimpleNamespace(
        address="self",
        layer_labels=["op_1_1"],
        num_layers=1,
        ops={1: mpl},
    )
    fake_trace = SimpleNamespace(
        modules=_AddrIndexed([mod_log]),
        op_labels=["op_1_1"],
        layer_labels=["op_1_1"],
        layer_logs={"op_1_1": object()},
        layer_list=[],
    )

    with pytest.raises(MetadataInvariantError, match="has labels not in layers"):
        _check_module_layer_containment(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# op_log_fields
# ---------------------------------------------------------------------------


def test_op_log_fields_fires_on_loaded_artifact_nonnull_func() -> None:
    """Arm 4: a loaded artifact's ``func`` field must be None (FieldPolicy.DROP)."""

    layer = SimpleNamespace(
        layer_label="relu_1_1",
        has_saved_activation=False,
        pass_index=1,
        num_passes=1,
        func_name="relu",
        intervention_replaced=False,
        is_internal_source=False,
        is_input=False,
        is_buffer=False,
        is_output=False,
        func=lambda: None,
    )
    fake_trace = SimpleNamespace(layer_list=[layer], _loaded_from_bundle=True)

    with pytest.raises(MetadataInvariantError, match="loaded artifact carries a non-None func"):
        _check_op_log_fields(fake_trace)  # type: ignore[arg-type]


def test_op_log_fields_fires_on_subzero_step_index() -> None:
    """Arm 8: a computational layer's step_index must be >= 1.

    (M1 raise-arm campaign: ``op_log_fields#a08`` survivor -- the shape/dtype,
    pass-numbering, and func-callable/func_name/sentinel checks must all stay
    silent, so only step_index itself is out of range.)
    """

    layer = SimpleNamespace(
        layer_label="relu_1_1",
        has_saved_activation=False,
        pass_index=1,
        num_passes=1,
        func_name="relu",
        intervention_replaced=False,
        is_internal_source=False,
        is_input=False,
        is_buffer=False,
        is_output=False,
        func=lambda: None,
        step_index=0,
    )
    fake_trace = SimpleNamespace(layer_list=[layer], _loaded_from_bundle=False)

    with pytest.raises(MetadataInvariantError, match=r"step_index=.*< 1"):
        _check_op_log_fields(fake_trace)  # type: ignore[arg-type]


def test_op_log_fields_fires_on_module_call_depth_mismatch() -> None:
    """Arm 12: module_call_depth must equal len(module_call_stack).

    (M1 raise-arm campaign: ``op_log_fields#a12`` survivor -- every earlier
    field check must stay silent on a bookkeeping (input) layer, so only the
    depth/stack-length cross-check is corrupted.)
    """

    layer = SimpleNamespace(
        layer_label="input_1_1",
        has_saved_activation=False,
        pass_index=1,
        num_passes=1,
        func_name="input",
        intervention_replaced=False,
        is_internal_source=False,
        is_input=True,
        is_buffer=False,
        is_output=False,
        raw_index=1,
        modules=(),
        module=None,
        module_call_stack=(),
        module_call_depth=3,
    )
    fake_trace = SimpleNamespace(layer_list=[layer])

    with pytest.raises(MetadataInvariantError, match=r"module_call_depth=.*!= len"):
        _check_op_log_fields(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# param_xrefs
# ---------------------------------------------------------------------------


def test_param_xrefs_fires_on_used_by_layers_outside_layer_labels() -> None:
    """Arm 1: a Param's used_by_layers must be known layer labels."""

    param = SimpleNamespace(address="w1", used_by_ops=[], used_by_layers=["ghost_layer_9_9"])
    fake_trace = SimpleNamespace(param_logs=[param], layer_labels=[], op_labels=[])

    with pytest.raises(MetadataInvariantError, match="used_by_layers contains"):
        _check_param_xrefs(fake_trace)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# special_layer_lists
# ---------------------------------------------------------------------------


def test_special_layer_lists_fires_on_input_layer_with_false_flag() -> None:
    """Arm 1: every label in input_layers must have is_input=True on its Op."""

    layer = SimpleNamespace(layer_label="op_1_1", is_input=False)
    fake_trace = _FakeTrace(
        input_layers=["op_1_1"],
        layer_labels=["op_1_1"],
        layer_list=[layer],
    )

    with pytest.raises(MetadataInvariantError, match="is in input_layers but is_input=False"):
        _check_special_layer_lists(fake_trace)  # type: ignore[arg-type]
