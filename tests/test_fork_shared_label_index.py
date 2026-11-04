"""Structural label indexes shared by a trace and its forks.

The README's patching sweep runs ``italy.fork().do(module & head, patch_from(france))``
once per attention head. Every replay used to build the selector label index,
the replay label maps and the function-call groups for its fresh fork, though
a clean fork's labels equal its parent's. These tests pin zero builds for a
clean fork once one fork of the same trace has built them, check that a label
write on one fork (in place or by cell) or on the parent reaches only that
trace's indexes and replays, and keep the label field set the sharing rule
compares in step with what the builders actually read.
"""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl
from torchlens._ir import selector_eval
from torchlens._ir._selector_universe import _string_labels
from torchlens._selection.errors import SiteResolutionError
from torchlens.intervention import _replay_cone, _replay_context

_N_LAYER, _N_HEAD, _N_EMBD = 2, 2, 8

_BUILDERS = {
    selector_eval._label_site_index.__code__: "label_site_index",
    _replay_cone._label_key_map.__code__: "label_key_map",
    _replay_cone._final_label_key_map.__code__: "final_label_key_map",
    _replay_context._func_call_groups.__code__: "func_call_groups",
}


def _model() -> GPT2LMHeadModel:
    torch.manual_seed(0)
    config = GPT2Config(
        n_layer=_N_LAYER,
        n_head=_N_HEAD,
        n_embd=_N_EMBD,
        n_positions=16,
        vocab_size=32,
        attn_implementation="eager",
    )
    return GPT2LMHeadModel(config).eval()


_IDS_F = torch.tensor([[1, 5, 7, 3, 9]])
_IDS_I = torch.tensor([[1, 5, 7, 4, 9]])


@pytest.fixture(scope="module")
def pair() -> Iterator[tuple[GPT2LMHeadModel, tl.Trace, tl.Trace]]:
    model = _model()
    france = tl.trace(model, _IDS_F)
    try:
        italy = tl.trace(model, _IDS_I)
        try:
            yield model, france, italy
        finally:
            italy.cleanup()
    finally:
        france.cleanup()


def _head(block: int, head: int) -> Any:
    return tl.module(f"transformer.h.{block}.attn") & tl.head(head, "v")


def _logits(trace: tl.Trace, france: tl.Trace, block: int, head: int) -> torch.Tensor:
    patched = trace.do(_head(block, head), tl.patch_from(france))
    return patched["lm_head"].out.detach().clone()


def _profiled(fn: Callable[[], Any]) -> tuple[Counter[str], int]:
    """Run ``fn`` under a profiler: builder calls by name, and all Python calls."""

    builds: Counter[str] = Counter()
    total = [0]

    def profile(frame: Any, event: str, arg: Any) -> None:
        if event == "call" or event == "c_call":
            total[0] += 1
            if event == "call":
                name = _BUILDERS.get(frame.f_code)
                if name is not None:
                    builds[name] += 1

    sys.setprofile(profile)
    try:
        fn()
    finally:
        sys.setprofile(None)
    return builds, total[0]


def _build_privately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn sharing off: every trace builds its own indexes, as before sharing existed."""

    from torchlens._trace_core import label_cells

    monkeypatch.setattr(label_cells, "shared_entry", lambda ops: None)
    monkeypatch.setattr(_replay_cone, "shared_entry", lambda ops: None)


def test_clean_forks_build_no_structural_index(pair: Any) -> None:
    """After one fork of a trace replays, a clean sibling fork builds no index."""

    _, france, italy = pair
    _logits(italy.fork(), france, 0, 0)  # first use builds the shared indexes
    builds, _ = _profiled(lambda: _logits(italy.fork(), france, 1, 1))
    assert dict(builds) == {}
    nested = italy.fork().fork()
    builds, _ = _profiled(lambda: _logits(nested, france, 0, 1))
    assert dict(builds) == {}


def test_sharing_cuts_python_calls_per_fork_replay(
    pair: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean fork replay through the shared indexes costs fewer Python calls than building."""

    _, france, italy = pair
    _logits(italy.fork(), france, 0, 0)
    shared_builds, shared_calls = _profiled(lambda: _logits(italy.fork(), france, 1, 1))
    _build_privately(monkeypatch)
    private_builds, private_calls = _profiled(lambda: _logits(italy.fork(), france, 1, 1))
    assert dict(shared_builds) == {}
    assert set(private_builds) == set(_BUILDERS.values())
    assert shared_calls < private_calls


def _fresh_views(trace: tl.Trace) -> tuple[Any, ...]:
    """The four structures built privately for ``trace`` (the reference answer)."""

    ops = trace._layer_list
    site_index = selector_eval._label_site_index(ops)
    assert site_index is not None
    return (
        {label: [id(op) for op in sites] for label, sites in site_index.items()},
        _replay_cone._label_key_map(trace),
        _replay_cone._final_label_key_map(trace),
        {
            key: [id(op) for op in group]
            for key, group in _replay_context._func_call_groups(trace).items()
        },
    )


def _shared_views(trace: tl.Trace) -> tuple[Any, ...]:
    """The same four structures answered through the shared tables."""

    from torchlens._trace_core.label_cells import shared_site_index

    ops = trace._layer_list
    site_index = shared_site_index(ops, selector_eval._label_site_index)
    assert site_index is not None
    groups = _replay_cone._shared_func_call_groups(trace)
    return (
        {label: [id(op) for op in site_index[label]] for label in site_index},
        _replay_cone._shared_label_key_map(trace),
        _replay_cone._shared_final_label_key_map(trace),
        {key: [id(op) for op in groups[key]] for key in groups},
    )


def test_label_writes_reach_only_their_own_trace(pair: Any) -> None:
    """A label write on one fork or on the parent never shows through another trace."""

    from torchlens._selection.resolver import _label_index_scope

    model, france, _ = pair
    parent = tl.trace(model, _IDS_I)
    reference = tl.trace(model, _IDS_I)
    try:
        sib_a, sib_b = parent.fork(), parent.fork()
        _logits(sib_a.fork(), france, 0, 0)  # the shared entry exists before any write
        assert _shared_views(sib_b) == _fresh_views(sib_b)

        in_place = sib_a._layer_list[3]
        in_place.lookup_keys.append("zz_fork_alias")  # no cell write at all
        by_cell = sib_a._layer_list[5]
        by_cell._internal_set("lookup_keys", [*by_cell.lookup_keys, "zz_fork_cell_alias"])
        parent._layer_list[4].lookup_keys.append("zz_parent_alias")  # parent, after forks exist

        for trace in (parent, sib_a, sib_b, sib_a.fork(), sib_b.fork()):
            assert _shared_views(trace) == _fresh_views(trace)
        assert _replay_cone._shared_label_key_map(sib_a)["zz_fork_alias"] == (in_place.label,)
        for trace in (parent, sib_b):
            keys = _replay_cone._shared_label_key_map(trace)
            assert "zz_fork_alias" not in keys and "zz_fork_cell_alias" not in keys
        assert "zz_parent_alias" not in _replay_cone._shared_label_key_map(sib_b)
        assert "zz_parent_alias" in _replay_cone._shared_label_key_map(parent)

        with _label_index_scope():
            assert sib_a[tl.label("zz_fork_alias")] is in_place
            assert parent[tl.label("zz_parent_alias")] is parent._layer_list[4]
            with pytest.raises(SiteResolutionError):
                sib_b[tl.label("zz_fork_alias")]

        want = _logits(reference.fork(), france, 1, 0)
        assert torch.equal(_logits(sib_b, france, 1, 0), want)
        assert torch.equal(_logits(parent, france, 1, 0), want)
        assert torch.equal(_logits(sib_a, france, 1, 0), want)
    finally:
        reference.cleanup()
        parent.cleanup()


def _uncovered(read: set[str], declared: set[str]) -> set[str]:
    """Fields the builders read that the sharing rule does not compare."""

    return read - declared


def test_label_field_set_matches_every_builder_read(
    pair: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drift guard: the compared label fields are exactly the fields the builders read."""

    from torchlens._data_classes.op import _OP_STORE_LAYOUT, Op, _OpField
    from torchlens._trace_core.label_cells import LABEL_FIELDS

    _, _, italy = pair
    fork = italy.fork()
    ops = fork._layer_list
    reads: set[str] = set()
    real_getattribute = Op.__getattribute__

    def recording(self: Any, name: str) -> Any:
        reads.add(name)
        return real_getattribute(self, name)

    monkeypatch.setattr(Op, "__getattribute__", recording)
    selector_eval._label_site_index(ops)
    _replay_cone._label_key_map(fork)
    _replay_cone._final_label_key_map(fork)
    _replay_context._func_call_groups(fork)
    for op in ops:
        _string_labels(op, "site")
    monkeypatch.undo()

    stored = set(_OP_STORE_LAYOUT.names)
    declared = set(LABEL_FIELDS)
    # Every read is a stored field (a computed property would escape the cell snapshot).
    assert reads - stored <= {"__class__"}
    field_reads = reads & stored
    assert _uncovered(field_reads, declared) == set()
    assert field_reads == declared
    # Each is a plain cell descriptor, so the cell IS what the builder reads.
    assert all(type(vars(Op)[name]) is _OpField for name in declared)
    # Red-capable: dropping any one entry is caught.
    for name in sorted(declared):
        assert _uncovered(field_reads, declared - {name}) == {name}
