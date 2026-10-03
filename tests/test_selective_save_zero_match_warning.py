"""A save predicate matching ZERO sites must warn, never pass silently.

TF is fail-closed on selector reachability, but the shared non-torch
selective-save resolver accepted a predicate matching nothing with status
COMPLETE and no diagnostic: a typo'd label or function name silently produced
a trace with zero saved activations.  The resolver now warns at predicate-
resolution completion.  (The torch capture-path and paddle intervene-side
equivalents live outside this resolver and are tracked separately.)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import torchlens as tl
from torchlens.postprocess._selective_save import apply_static_label_save_policy


def _stub_trace() -> Any:
    """Return a minimal trace-like object the resolver can filter."""

    ops = [
        SimpleNamespace(
            label="relu_1_1",
            _label_raw="relu_1_1",
            layer_label="relu_1_1",
            layer_label_short="relu_1",
            func_name="relu",
            out=object(),
            has_saved_activation=True,
            is_orphan=False,
            activation_memory=16,
        ),
        SimpleNamespace(
            label="add_1_2",
            _label_raw="add_1_2",
            layer_label="add_1_2",
            layer_label_short="add_1",
            func_name="add",
            out=object(),
            has_saved_activation=True,
            is_orphan=False,
            activation_memory=16,
        ),
    ]
    return SimpleNamespace(layer_list=ops, module_calls=())


@pytest.mark.smoke
def test_zero_match_save_predicate_warns() -> None:
    """A predicate matching no op must surface a warning, not silence."""

    trace = _stub_trace()
    with pytest.warns(UserWarning, match="matched zero"):
        apply_static_label_save_policy(trace, tl.label("no_such_label"), backend_name="mlx")
    assert trace.num_saved_ops == 0


@pytest.mark.smoke
def test_matching_save_predicate_does_not_warn() -> None:
    """A predicate matching at least one op stays silent."""

    import warnings

    trace = _stub_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_static_label_save_policy(trace, tl.label("relu_1_1"), backend_name="mlx")
    assert trace.num_saved_ops == 1


@pytest.mark.smoke
def test_none_predicate_does_not_warn() -> None:
    """The full-save default (no predicate) is untouched."""

    import warnings

    trace = _stub_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        apply_static_label_save_policy(trace, None, backend_name="mlx")
    assert all(op.has_saved_activation for op in trace.layer_list)


# ---------------------------------------------------------------------------
# R17 (round 3): TF joins the zero-match save= disclosure family
# ---------------------------------------------------------------------------


def test_shared_zero_match_warning_names_the_backend() -> None:
    """The extracted shared helper is the one disclosure text for all backends."""

    from torchlens.postprocess._selective_save import warn_zero_match_save_predicate

    with pytest.warns(UserWarning, match="tf trace\\(save=...\\) predicate matched zero"):
        warn_zero_match_save_predicate("tf")


@pytest.mark.smoke
def test_tf_eager_session_counts_save_predicate_matches() -> None:
    """The eager retention gate counts matches for the entry-level disclosure.

    Import-safe without TensorFlow: the session constructor stores plain
    attributes and ``_should_save_payload`` consults only the predicate.
    """

    from types import SimpleNamespace

    from torchlens.backends.tf.op_callback_capture import TFEagerCaptureSession

    session = TFEagerCaptureSession(
        tf=None,
        callable_obj=None,
        args=(),
        kwargs={},
        module_tree=None,
        save_payloads=True,
        save_predicate=lambda ctx: ctx.func_name == "relu",
    )
    assert session.save_predicate_match_count == 0
    assert session._should_save_payload(SimpleNamespace(func_name="matmul")) is False
    assert session.save_predicate_match_count == 0
    assert session._should_save_payload(SimpleNamespace(func_name="relu")) is True
    assert session.save_predicate_match_count == 1


def test_tf_capture_entries_call_the_shared_zero_match_disclosure() -> None:
    """Source lockstep: both TF entries (eager + funcgraph) warn on zero match.

    TF gates retention per-op instead of running the shared post-finalization
    resolver, so a typo'd ``save=`` completed COMPLETE with zero payloads and
    no diagnostic -- the one preview outside the disclosure family. Live TF
    runs need the preview env; this pin keeps both call sites from silently
    disappearing on torch-only hosts.
    """

    import pathlib

    import torchlens as tl

    torchlens_dir = pathlib.Path(tl.__file__).resolve().parent
    backend_source = (torchlens_dir / "backends" / "tf" / "backend.py").read_text(encoding="utf-8")
    assert 'warn_zero_match_save_predicate("tf")' in backend_source
    assert "save_predicate_match_count == 0" in backend_source
    funcgraph_source = (torchlens_dir / "backends" / "tf" / "funcgraph.py").read_text(
        encoding="utf-8"
    )
    assert 'warn_zero_match_save_predicate("tf")' in funcgraph_source
    assert "predicate_matches == 0" in funcgraph_source
