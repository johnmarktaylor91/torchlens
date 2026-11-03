"""The cost of a multi-receiver ``torchlens.mechinterp.patch_edges`` call, as deterministic counters.

Counters per call: replays (runs of the replay engine's cone executor),
component-graph builds, and trace forks. Declared target: one replay per
topological stage (one layer's q/k/v receivers, one MLP, ``logits``; at most
``2L + 1`` on an ``L``-layer GPT-2), one graph build per stage after the
first, one fork, so no scratch fork per q/k/v receiver. Values are pinned
elsewhere and unchanged (``tests/test_circ3_multi_receiver_patch.py``
against plain hooks, and bit for bit against the single-receiver
implementation); here the local norm-and-projection recompute that replaced
the scratch fork is checked bit for bit against an edge substitution plus a
push on a fork.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch
from test_circ3_multi_receiver_patch import _tiny_gpt2

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.intervention import _edge_substitution as edge_substitution, _replay as replay_module
from torchlens.intervention._edge_substitution import _recompute_with_substitution
from torchlens.mechinterp import _edges
from torchlens.mechinterp._edge_patch import _head_spec, _input_edge_record

pytest.importorskip("transformers")
pytestmark = pytest.mark.heavy

_CAPTURE = tl.options.CaptureOptions(replayable=True)


@pytest.fixture(scope="module")
def tiny():
    model = _tiny_gpt2()
    clean = tl.trace(
        model, torch.tensor([[3, 7, 1, 9, 4], [2, 6, 1, 8, 5]]), capture=_CAPTURE, save="all"
    )
    corrupt = tl.trace(
        model, torch.tensor([[5, 7, 2, 9, 8], [4, 6, 3, 8, 1]]), capture=_CAPTURE, save="all"
    )
    try:
        yield clean, mi.component_graph(clean), mi.component_graph(corrupt)
    finally:
        clean.cleanup()
        corrupt.cleanup()


def counted_patch(monkeypatch, clean, edges, corrupt):
    """Run ``patch_edges`` and return ``(patched, counts)`` of replays, graph builds and forks."""

    counts: Counter = Counter()

    def counting(name, inner):
        def wrapper(*args, **kwargs):
            counts[name] += 1
            return inner(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(
        replay_module, "_run_replay", counting("replays", replay_module._run_replay)
    )
    monkeypatch.setattr(_edges, "component_graph", counting("graph_builds", _edges.component_graph))
    monkeypatch.setattr(type(clean), "fork", counting("forks", type(clean).fork))
    patched = mi.patch_edges(clean, edges, corrupted_graph=corrupt)
    return patched, counts


def test_a_target_upstream_of_the_edge_returns_the_reason_and_computes_nothing(tiny) -> None:
    clean, graph, _corrupt = tiny
    late = graph.receivers["a1.h0<q>"]
    early_home = str(_head_spec(clean, graph.receivers["a0.h0<q>"]).home_label)
    reason = _recompute_with_substitution(
        clean, _input_edge_record(clean, late), [late.value], early_home
    )
    assert isinstance(reason, str)
    assert "is not downstream of edge" in reason


def test_patch_edges_refuses_with_the_reason_when_a_projection_is_not_recomputed(
    tiny, monkeypatch
) -> None:
    clean, graph, corrupt = tiny
    edges = [edge for edge in graph.edges.values() if edge.receiver.name == "a1.h0<q>"]
    monkeypatch.setattr(
        edge_substitution,
        "_recompute_with_substitution",
        lambda *args, **kwargs: "the path passes buffer record 'buffer_1'",
    )
    with pytest.raises(tl.errors.MechInterpError) as excinfo:
        mi.patch_edges(clean, edges, corrupted_graph=corrupt)
    assert excinfo.value.fields["code"] == "mi_edge_selection_unresolvable"
    assert "buffer record 'buffer_1'" in str(excinfo.value)
