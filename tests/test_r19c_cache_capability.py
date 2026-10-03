"""Regression tests for the capture-cache path (r19c hardening).

Covers a previously-uncovered gap: nothing combined ``cache=True`` with an
activation/grad transform (F2) or exercised a capability-option cache
collision (F3), and a pre-forward setup failure leaked capture-global state
(SOL-A5-002). The ``trace()`` docstring also carried three false
self-referential "deprecated alias" lines (DOC1).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl


def _tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.ReLU())


def _act_transform(value: torch.Tensor) -> torch.Tensor:
    # Module-level (picklable) so the cached trace can be serialized.
    return value.detach() + 1.0


def _grad_transform(value: torch.Tensor) -> torch.Tensor:
    return value.detach() * 2.0


# --------------------------------------------------------------------------- F2
@pytest.mark.smoke
def test_cache_with_activation_transform_roundtrips(tmp_path):
    """cache=True + activation_transform must not raise and must cache-hit.

    Fail-before: ``AttributeError: can't set attribute 'transformed_out'`` from
    a raw ``setattr`` on the read-only ``Layer.transformed_out`` proxy inside
    ``_prepare_log_for_capture_cache``.
    """
    model = _tiny_model()
    x = torch.randn(2, 4)
    cache_dir = str(tmp_path / "cache")

    first = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=cache_dir),
        save=tl.options.SaveOptions(activation_transform=_act_transform),
    )
    assert first.capture_cache_hit is False

    second = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=cache_dir),
        save=tl.options.SaveOptions(activation_transform=_act_transform),
    )
    assert second.capture_cache_hit is True


@pytest.mark.smoke
def test_cache_with_grad_transform_roundtrips(tmp_path):
    """The sibling grad_transform path must also survive cache serialization."""
    model = _tiny_model()
    x = torch.randn(2, 4)
    cache_dir = str(tmp_path / "cache")

    first = tl.trace(
        model,
        x,
        grad_transform=_grad_transform,
        capture=tl.options.CaptureOptions(
            cache=True, cache_dir=cache_dir, save_grads=True, backward_ready=True
        ),
    )
    assert first.capture_cache_hit is False

    second = tl.trace(
        model,
        x,
        grad_transform=_grad_transform,
        capture=tl.options.CaptureOptions(
            cache=True, cache_dir=cache_dir, save_grads=True, backward_ready=True
        ),
    )
    assert second.capture_cache_hit is True


# --------------------------------------------------------------------------- F3
@pytest.mark.smoke
def test_cache_key_distinguishes_intervention_ready(tmp_path):
    """A capability change must MISS the cache -- never silently return a stale trace.

    Fail-before: the cache key omitted intervention_ready, so the second call
    (intervention_ready=True) returned the earlier cached trace whose
    intervention_ready was False -- silent wrongness with no warning.
    """
    model = _tiny_model()
    x = torch.randn(2, 4)
    cache_dir = str(tmp_path / "cache")

    first = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(
            cache=True, cache_dir=cache_dir, intervention_ready=False
        ),
    )
    assert first.capture_cache_hit is False
    assert first.intervention_ready is False

    second = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=cache_dir, intervention_ready=True),
    )
    # Different capability => must not reuse the intervention_ready=False trace.
    assert second.capture_cache_hit is False
    assert second.intervention_ready is True


@pytest.mark.smoke
def test_cache_key_distinguishes_save_raw_input(tmp_path):
    """Sibling payload-policy option: save_raw_input must also key the cache."""
    model = _tiny_model()
    x = torch.randn(2, 4)
    cache_dir = str(tmp_path / "cache")

    first = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=cache_dir, save_raw_input=False),
    )
    assert first.capture_cache_hit is False

    second = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=cache_dir, save_raw_input=True),
    )
    assert second.capture_cache_hit is False


@pytest.mark.smoke
def test_cache_hit_preserved_for_identical_capability(tmp_path):
    """Fix must not break caching: identical options still cache-hit."""
    model = _tiny_model()
    x = torch.randn(2, 4)
    cache_dir = str(tmp_path / "cache")

    first = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=cache_dir, intervention_ready=True),
    )
    assert first.capture_cache_hit is False
    assert first.intervention_ready is True

    second = tl.trace(
        model,
        x,
        capture=tl.options.CaptureOptions(cache=True, cache_dir=cache_dir, intervention_ready=True),
    )
    assert second.capture_cache_hit is True
    assert second.intervention_ready is True


# ------------------------------------------------------------------------- DOC1
@pytest.mark.smoke
def test_trace_docstring_has_no_self_referential_aliases():
    """trace() docstring must not brand a live canonical kwarg its own 'alias'.

    Fail-before: three stale rename-cruft lines said e.g. ``grad_transform:
    Alias for ``grad_transform``.`` -- each named ITSELF; no second spelling
    exists in the signature.
    """
    import inspect

    # Python 3.13 dedents docstrings at compile time (every leading-whitespace
    # column that is common to every line is stripped from co_consts), so the
    # raw __doc__ no longer carries the source's 4-space numpydoc indentation
    # there -- `inspect.cleandoc` normalizes BOTH representations to the same
    # canonical (zero-indent-for-the-first-level) form, so the check below is
    # Python-version-agnostic rather than tied to the literal source spelling.
    doc = inspect.cleandoc(tl.trace.__doc__ or "")
    signature_params = set(inspect.signature(tl.trace).parameters)

    # activation_transform and recurrence_detection moved into the grouped
    # options with the flat-kwarg removal; grad_transform stays a real param.
    for name in ("grad_transform",):
        # No second spelling exists, so any "alias for <itself>" line is a lie.
        assert f"{name}: Alias for ``{name}``" not in doc
        assert f"{name}: Deprecated alias for ``{name}``" not in doc
        # The real canonical parameter still exists and stays documented
        # exactly once, as its own numpydoc entry (a line starting with
        # "name:", after cleandoc normalization).
        assert name in signature_params
        assert doc.count(f"\n{name}:") == 1
    for removed in ("activation_transform", "recurrence_detection"):
        assert removed not in signature_params


# -------------------------------------------------------------------- SOL-A5-002
@pytest.mark.smoke
def test_pre_forward_failure_resets_capture_runtime_context(monkeypatch):
    """A pre-forward setup failure must not leak capture-global runtime state.

    Fail-before: configure_capture_runtime_context() ran before the outer
    try/finally; a raise in the Trace ctor (or later pre-forward setup) left
    _capture_replay_templates=True and the relationship model/input identity
    stale until the next trace() happened to reset at its start.
    """
    from torchlens import _state, user_funcs as uf

    real_trace_cls = uf.Trace

    class _ExplodingTrace(real_trace_cls):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            raise RuntimeError("boom-in-ctor")

    model = _tiny_model()
    x = torch.randn(2, 4)

    _state.reset_capture_runtime_context()
    try:
        monkeypatch.setattr(uf, "Trace", _ExplodingTrace)
        with pytest.raises(RuntimeError, match="boom-in-ctor"):
            tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
        # The pre-forward failure window must have reset the capture-global state.
        assert getattr(_state, "_capture_replay_templates") is False
        assert getattr(_state, "_relationship_model_id") is None
        assert getattr(_state, "_relationship_model_class") is None
        assert getattr(_state, "_relationship_input_id") is None
    finally:
        _state.reset_capture_runtime_context()


def test_unpicklable_capture_degrades_to_uncached(tmp_path):
    """cache=True must never destroy a successful capture (b6 R25, 3rd round).

    A stock ``nn.MultiheadAttention`` capture holds a python-level
    ``Tensor.*`` method in ``Op.func`` while the class attribute is the
    installed wrapper, so pickle's by-name identity check fails; the cache
    store propagated the bare ``PicklingError`` out of ``tl.trace`` itself.
    The store must degrade to "not cached" with a warning instead.
    """

    import warnings

    class _MHAWrap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mha = nn.MultiheadAttention(8, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.mha(x, x, x)
            return out

    model = _MHAWrap().eval()
    x = torch.randn(3, 2, 8)
    cache_dir = str(tmp_path / "cache")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(
            model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=cache_dir)
        )
    assert trace.num_ops > 0
    degrade = [w for w in caught if "Not caching this capture" in str(w.message)]
    assert degrade, "expected the not-cached degrade warning"
    assert "unaffected" in str(degrade[0].message)

    # The failed store must not poison later captures either (still warning,
    # never raising -- pytest's warnings-as-errors needs the explicit expect).
    with pytest.warns(UserWarning, match="Not caching this capture"):
        second = tl.trace(
            model, x, capture=tl.options.CaptureOptions(cache=True, cache_dir=cache_dir)
        )
    assert second.num_ops == trace.num_ops
