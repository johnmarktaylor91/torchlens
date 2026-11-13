"""Cross-backend agreement for the INCIDENTAL bare-label all-keys binding.

There was never a documented contract for what ``layer_dict_all_keys[bare
layer_label]`` resolves to on a multi-pass layer -- bare-label addressing of
multi-pass layers refuses on every path that matters
(``multipass_bare_label_ambiguous``). But the neutral preview finalizer and
the jax backend used to bind the bare label to the FIRST pass while torch's
raw-index artifact resolves to the LAST pass: an undocumented cross-backend
disagreement waiting to be mistaken for behavior. These tests pin the
alignment: every backend's incidental binding is last-pass-wins, and every
pass carries the bare label in its ``lookup_keys`` (torch parity).
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

from torchlens._postprocess.loop_grouping_adapter import RecurrenceAssignment
from torchlens._utils._record_group_table import op_attr
from torchlens.backends._finalize import _apply_recurrence_relabel_epilogue, _finalize_single_op


def _stub_trace() -> SimpleNamespace:
    """Return a minimal trace stub with the lookup indexes finalize touches."""

    return SimpleNamespace(
        _layer_list=[],
        _layer_dict_main_keys={},
        _layer_dict_all_keys={},
        op_labels=[],
        layer_labels=[],
        _layer_num_calls={},
        _lookup_keys_to_layer_num_dict={},
        _layer_num_to_lookup_keys_dict=defaultdict(list),
    )


def _assignment(pass_index: int) -> RecurrenceAssignment:
    """Return a 2-pass assignment for the shared ``cell`` layer."""

    return RecurrenceAssignment(
        layer_label="cell",
        recurrent_labels=("cell_raw_0", "cell_raw_1"),
        pass_index=pass_index,
        num_passes=2,
        equivalence_key="eq:cell",
        site_key=None,
    )


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_finalize.py.
def test_neutral_finalizer_bare_label_is_last_pass_and_on_every_pass() -> None:
    """The shared finalizer's bare-label binding is last-pass-wins."""

    trace = _stub_trace()
    # The finalize loop applies each op's final labels (apply_final_label) before
    # _finalize_single_op attaches lookup keys, so the fakes carry them already.
    first = SimpleNamespace(label="cell:1", layer_label="cell")
    second = SimpleNamespace(label="cell:2", layer_label="cell")
    _finalize_single_op(trace, first, "cell_raw_0", 0, _assignment(1))
    _finalize_single_op(trace, second, "cell_raw_1", 1, _assignment(2))

    assert trace._layer_dict_all_keys["cell:1"] is first
    assert trace._layer_dict_all_keys["cell:2"] is second
    # Incidental raw-index artifact: last pass wins, matching torch.
    assert trace._layer_dict_all_keys["cell"] is second
    # Torch parity: EVERY pass lists the bare label among its lookup keys.
    assert "cell" in first.lookup_keys
    assert "cell" in second.lookup_keys


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/_recurrence.py.
def test_relabel_epilogue_fixes_children_even_without_grouping() -> None:
    """N5: every op's edges must be relabeled, not just multi-pass members.

    Before the fix, ``_apply_recurrence_relabel_epilogue`` only rewrote
    ``parents``/``children`` for multi-pass group members (``changed`` was
    filtered to ``num_passes != 1``) and only ran at all when recurrence
    assignments were present. That was safe while a singleton op's
    ``layer_label`` equaled its raw label (the historical preview bug this
    module's other tests pin the fix for) -- but once the raw ``_raw``
    sentinel is stripped from every op's final label, a children/parents
    edge that still names the raw string becomes a dangling reference. This
    must be fixed with ``assignments=None`` (the ``recurrence_detection=False``
    path), the case with no recurrence grouping at all.

    Torch parity (second-pass convention): both ops here are single-pass, so
    ``parents``/``children``/``input_layers``/``output_layers`` and the
    lineage sets resolve through the CONDITIONAL mapping to their BARE
    ``layer_label`` -- never the pass-qualified ``label`` -- while
    ``recurrent_ops`` (always fully pass-qualified) still reads ``:1``.
    """

    producer = SimpleNamespace(
        label="producer_1_1:1",
        layer_label="producer_1_1",
        num_passes=1,
        **{op_attr("equivalence_class"): "producer"},
        parents=[],
        children=["consumer_1_2_raw"],
        root_ancestors=frozenset({"producer_1_1_raw"}),
        input_ancestors={"producer_1_1_raw"},
        output_descendants=set(),
        internal_source_ancestors=frozenset(),
    )
    consumer = SimpleNamespace(
        label="consumer_1_2:1",
        layer_label="consumer_1_2",
        num_passes=1,
        **{op_attr("equivalence_class"): "consumer"},
        parents=["producer_1_1_raw"],
        children=[],
        root_ancestors=frozenset(),
        input_ancestors={"producer_1_1_raw"},
        output_descendants={"consumer_1_2_raw"},
        internal_source_ancestors=frozenset(),
    )
    raw_dict = {"producer_1_1_raw": producer, "consumer_1_2_raw": consumer}
    trace = SimpleNamespace(
        _raw_graph_ws=SimpleNamespace(raw_layer_dict=raw_dict),
        _graph_op_equivalence_classes={},
        input_layers=["producer_1_1_raw"],
        output_layers=["consumer_1_2_raw"],
    )

    _apply_recurrence_relabel_epilogue(trace, None, None)

    assert consumer.parents == ["producer_1_1"]
    assert producer.children == ["consumer_1_2"]
    assert trace.input_layers == ["producer_1_1"]
    assert trace.output_layers == ["consumer_1_2"]
    assert getattr(producer, op_attr("recurrent_ops")) == ["producer_1_1:1"]
    assert getattr(consumer, op_attr("recurrent_ops")) == ["consumer_1_2:1"]
    # N5: lineage sets (seeded with raw labels at capture time or during the
    # pre-relabel depth flood) must be relabeled too, with their original
    # container type (frozenset vs set) preserved.
    assert producer.root_ancestors == frozenset({"producer_1_1"})
    assert isinstance(producer.root_ancestors, frozenset)
    assert producer.input_ancestors == {"producer_1_1"}
    assert isinstance(producer.input_ancestors, set) and not isinstance(
        producer.input_ancestors, frozenset
    )
    assert consumer.output_descendants == {"consumer_1_2"}
