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

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends._finalize import _apply_recurrence_relabel_epilogue, _finalize_single_op
from torchlens.postprocess.loop_grouping_adapter import RecurrenceAssignment


def _stub_trace() -> SimpleNamespace:
    """Return a minimal trace stub with the lookup indexes finalize touches."""

    return SimpleNamespace(
        layer_list=[],
        layer_dict_main_keys={},
        layer_dict_all_keys={},
        op_labels=[],
        layer_labels=[],
        layer_num_calls={},
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


def test_neutral_finalizer_bare_label_is_last_pass_and_on_every_pass() -> None:
    """The shared finalizer's bare-label binding is last-pass-wins."""

    trace = _stub_trace()
    first = SimpleNamespace()
    second = SimpleNamespace()
    _finalize_single_op(trace, first, "cell_raw_0", 0, _assignment(1))
    _finalize_single_op(trace, second, "cell_raw_1", 1, _assignment(2))

    assert trace.layer_dict_all_keys["cell:1"] is first
    assert trace.layer_dict_all_keys["cell:2"] is second
    # Incidental raw-index artifact: last pass wins, matching torch.
    assert trace.layer_dict_all_keys["cell"] is second
    # Torch parity: EVERY pass lists the bare label among its lookup keys.
    assert "cell" in first.lookup_keys
    assert "cell" in second.lookup_keys


def test_torch_bare_label_artifact_is_last_pass() -> None:
    """Pin the torch side of the parity so the agreement cannot drift."""

    class Loop(nn.Module):
        """Three applications of one shared cell."""

        def __init__(self) -> None:
            """Build the shared cell."""

            super().__init__()
            self.cell = nn.Linear(4, 4, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Apply the cell three times."""

            h = x
            for _ in range(3):
                h = torch.relu(self.cell(h))
            return h

    torch.manual_seed(0)
    log = tl.trace(Loop(), torch.randn(2, 4))
    multi_pass_bare = [
        key
        for key, record in log.layer_dict_all_keys.items()
        if isinstance(key, str) and ":" not in key and record.num_passes > 1
    ]
    assert multi_pass_bare, "expected multi-pass layers in the loop trace"
    for key in multi_pass_bare:
        record = log.layer_dict_all_keys[key]
        if key != record.layer_label:
            continue  # raw/address spellings, not the bare layer label
        assert record.pass_index == record.num_passes, (
            f"torch bare-label artifact for {key!r} no longer resolves to the last pass"
        )


@pytest.mark.smoke
def test_singleton_fallback_strips_raw_capture_sentinel() -> None:
    """N5: the ``assignment=None`` singleton fallback must not leak ``_raw``.

    ``_finalize_single_op`` falls back to the raw backend label (e.g.
    ``"input_1_1_raw"``, the exact shape ``ir.capture_events.reserve_label``
    mints) as the layer label whenever recurrence grouping produced no
    assignment -- the common case for every non-recurrent op on every
    preview backend (tf, jax, tinygrad, paddle, mlx). Leaving the internal
    ``_raw`` capture sentinel in place made ``trace.layer_labels`` carry raw
    labels straight into the public surface, tripping the ``graph_ordering``
    "Raw label survived postprocessing" invariant on essentially every
    preview capture.
    """

    trace = _stub_trace()
    op_log = SimpleNamespace()
    _finalize_single_op(trace, op_log, "input_1_1_raw", 0, None)

    assert op_log.layer_label == "input_1_1"
    assert op_log.layer_label_short == "input_1_1"
    assert op_log.label == "input_1_1:1"
    assert op_log._label_raw == "input_1_1_raw"
    # Torch parity (second-pass convention): ``_layer_label_raw`` carries the
    # finalized layer label, same as ``layer_label`` -- only ``_label_raw``
    # keeps the ``_raw`` capture sentinel intact.
    assert op_log._layer_label_raw == "input_1_1"
    assert trace.layer_labels == ["input_1_1"]
    # Raw labels stay resolvable through the lookup keys, just not as the
    # presented layer label.
    assert trace.layer_dict_all_keys["input_1_1_raw"] is op_log


@pytest.mark.smoke
def test_grouped_leader_strips_raw_capture_sentinel() -> None:
    """A multi-pass group's leader label must also lose its ``_raw`` suffix.

    ``RecurrenceAssignment.layer_label`` is documented as a raw node label:
    ``group_recurrent_nodes`` picks the group LEADER's own raw label
    (``_raw`` suffix intact) for every member, so the grouped path needs the
    same strip as the singleton fallback.
    """

    trace = _stub_trace()
    assignment = RecurrenceAssignment(
        layer_label="cell_1_1_raw",
        recurrent_labels=("cell_1_1_raw", "cell_2_2_raw"),
        pass_index=1,
        num_passes=2,
        equivalence_key="eq:cell",
        site_key=None,
    )
    op_log = SimpleNamespace()
    _finalize_single_op(trace, op_log, "cell_1_1_raw", 0, assignment)

    assert op_log.layer_label == "cell_1_1"
    assert op_log.label == "cell_1_1:1"
    assert trace.layer_labels == ["cell_1_1"]


@pytest.mark.smoke
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
        equivalence_class="producer",
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
        equivalence_class="consumer",
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
        op_equivalence_classes={},
        input_layers=["producer_1_1_raw"],
        output_layers=["consumer_1_2_raw"],
    )

    _apply_recurrence_relabel_epilogue(trace, None, None)

    assert consumer.parents == ["producer_1_1"]
    assert producer.children == ["consumer_1_2"]
    assert trace.input_layers == ["producer_1_1"]
    assert trace.output_layers == ["consumer_1_2"]
    assert producer.recurrent_ops == ["producer_1_1:1"]
    assert consumer.recurrent_ops == ["consumer_1_2:1"]
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
