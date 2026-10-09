"""Marker identity + internal separation (torchnative W0.2/W0.3, TN-D11/D13/D14).

The honest-Nsight launch copy depends on two XS repairs the panel measured as
missing: NVTX range names must carry CALL identity (twenty ``conv2d`` calls
were twenty indistinguishable ranges, TN-D13), and TorchLens's OWN
bookkeeping calls -- 49.2% of the markers on a real GPT-2 capture were our
per-output ``register_hook`` installs (TN-D11) -- must be separated under
``torchlens::internal::`` instead of being published as model work. FLIP-1's
join markers ride the same bracket with the exact ``func_call_id`` key.

All smoke-tier: tiny models, monkeypatched NVTX, no CUDA required.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl


@pytest.fixture()
def nvtx_calls(monkeypatch) -> list[str]:
    """Capture NVTX range names without CUDA."""

    calls: list[str] = []
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda name: calls.append(str(name)))
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    return calls


def test_nvtx_names_carry_call_identity(nvtx_calls) -> None:
    """TN-D13: repeated same-type calls produce DISTINGUISHABLE ranges."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4)).eval()
    x = torch.randn(2, 4)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(emit_nvtx=True))
    log.cleanup()
    names = [c for c in nvtx_calls if c.startswith("torchlens::")]
    assert names, "emit_nvtx=True emitted no torchlens:: ranges"
    assert len(set(names)) == len(names), f"non-unique NVTX range names: {names}"
    linear_ranges = [n for n in names if "linear" in n]
    assert len(linear_ranges) >= 2
    assert len(set(linear_ranges)) == len(linear_ranges)
    # Identity is carried explicitly: name#<func_call_id>.
    assert all("#" in n for n in names)


def test_nvtx_internal_bookkeeping_separated(nvtx_calls) -> None:
    """TN-D11: TorchLens's own register_hook installs never publish as model work.

    The per-output gradient-hook install runs under ``pause_logging()`` like
    every TorchLens-internal torch call (Critical Invariant 2), so it reaches
    no wrapper and pushes no range at all, under torch's name or ours.
    """

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).train()
    x = torch.randn(2, 4, requires_grad=True)
    log = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(emit_nvtx=True, backward_ready=True),
        save_mode="reference",
    )
    has_grad_fn = all(
        op.grad_fn_object_id is not None for op in log.ops if op.func_name == "linear"
    )
    log.cleanup()
    assert has_grad_fn, "backward_ready capture recorded no grad_fn on its linear ops"
    names = [c for c in nvtx_calls if c.startswith("torchlens::")]
    assert [n for n in names if "register_hook" in n] == [], (
        "TorchLens's paused gradient-hook install reached the NVTX sink: "
        f"{[n for n in names if 'register_hook' in n]}"
    )
    model_ranges = [n for n in names if not n.startswith("torchlens::internal::")]
    assert any("linear" in n for n in model_ranges)


def test_internal_read_marker_names_are_typed_internal() -> None:
    """A wrapped call made inside an internal read is named under ``torchlens::internal::``."""

    from torchlens.backends.torch._op_markers import _op_marker_labels
    from torchlens.backends.torch.completeness_witness import internal_scalar_read

    assert _op_marker_labels("linear", 7) == ("torchlens::linear#7", "torchlens::op::7")
    with internal_scalar_read():
        assert _op_marker_labels("register_hook", 8) == (
            "torchlens::internal::register_hook#8",
            "torchlens::internal::8",
        )


def test_no_markers_without_a_sink(nvtx_calls) -> None:
    """Strictly opt-in: plain captures push nothing into either sink."""

    model = nn.Linear(4, 4).eval()
    log = tl.trace(model, torch.randn(2, 4))
    log.cleanup()
    assert not [c for c in nvtx_calls if c.startswith("torchlens::")]


def test_session_records_op_join_markers() -> None:
    """FLIP-1 marker side: an active owned session sees exact op keys.

    The ``record_function`` marker name is ``torchlens::op::<func_call_id>``
    -- an exact key onto the persisted op-record field, never a function
    name (name matching is forbidden from the join).
    """

    from torchlens import observability as obs

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU()).eval()
    x = torch.randn(2, 4)
    with obs.session() as sess:
        log = tl.trace(model, x)
    assert sess.result is not None and sess.result.facts["mode"] == "owned"
    extraction = obs.extract_events(sess.closed_profiler)
    assert extraction.path in ("in_memory", "chrome_stream")
    key_markers = [
        e for e in extraction.events if e.kind == "marker" and e.name.startswith("torchlens::op::")
    ]
    assert key_markers, "no torchlens::op:: markers reached the owned profiler"
    marker_ids = {int(e.name.rsplit("::", 1)[-1]) for e in key_markers}
    op_ids = {int(op.func_call_id) for op in log.ops if op.func_call_id is not None}
    assert marker_ids & op_ids, "marker keys do not intersect captured op func_call_ids"
    log.cleanup()


@pytest.mark.smoke
def test_native_chrome_artifact_and_sidecar(tmp_path) -> None:
    """The NATIVE chrome trace is preserved with the exact-ID mapping sidecar."""

    import json

    from torchlens import observability as obs

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU()).eval()
    path = tmp_path / "native.json"
    result = obs.native_profile(model, torch.randn(2, 8), native_chrome_path=path)
    native = json.loads(path.read_text())
    assert "traceEvents" in native  # torch's own artifact, not rewritten
    sidecar = json.loads((tmp_path / "native.json.torchlens-map.json").read_text())
    assert sidecar["schema"] == "torchlens.native_chrome_map.v1"
    assert "never projected" in sidecar["clock_note"]
    assert any(m["owner_class"] == "forward_op" for m in sidecar["markers"])
    result.trace.cleanup()


def test_borrowed_session_never_closed_and_joinable() -> None:
    """Borrowed mode: markers land in the caller's profiler; we never close it."""

    from torchlens import observability as obs

    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU()).eval()
    with torch.profiler.profile() as prof, obs.session(mode="borrowed", profiler=prof) as sess:
        log = tl.trace(model, torch.randn(2, 8))
    # The borrowed profiler closed AFTER the session exited without touching it.
    join = obs.join_session(sess, log)
    assert join.facts["n_markers"] >= 2
    assert join.availability == "empty"  # CPU-only: honest, with reason
    assert "device" in (join.coverage.reason or "")
    log.cleanup()
