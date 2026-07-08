"""Regression tripwire for internal self-deprecation.

"Internal self-deprecation" means torchlens' own source code calling one of its
own deprecated public aliases (as opposed to *user* code deliberately exercising
a documented back-compat alias, which is expected and fine). This bug class was
fixed piecemeal at roughly six source sites across hardening rounds 3-5 (the
``experimental/dagua/_bridge.py`` skip-list for deprecated ``Trace`` properties,
``semantic/patching.py`` calling ``Trace.run()`` instead of the deprecated
``Trace.rerun()``, and earlier "periphery: stop torchlens self-warnings"
cleanups -- see ``.research/hardening-megasprint/cert5/peripheral_claude.md``).

That bucket's round-5 audit found the class had **no regression tripwire**:
``pyproject.toml``'s ``filterwarnings`` promotes torchlens-sourced
``DeprecationWarning``/``UserWarning`` to errors, but then unconditionally
downgrades every warning matching the deprecation-alias message pattern back to
non-fatal (``"default:`.*` is deprecated; use `.*` instead\\..*:DeprecationWarning"``).
That rule cannot distinguish "test code deliberately exercises a documented
alias" from "torchlens internally hit its own deprecated alias by accident", so
a silent regression of any of those ~6 fixes would NOT fail the suite today.

This module closes that gap directly: each test runs one representative
torchlens operation inside ``warnings.catch_warnings(record=True)`` with
``simplefilter("always")`` (bypassing pytest's ini-level downgrade for that
block only) and asserts that none of the captured warnings are torchlens
firing a self-deprecation warning against itself. Every test in this file
calls torchlens exclusively through canonical (non-deprecated) argument
names/methods inside the monitored block, so any self-deprecation warning
that surfaces there can only have come from torchlens' own internals -- there
is deliberately no "this test file is allowed to trigger one" exemption (see
detection note below for why one would be unsound). A test that legitimately
wants to exercise a deprecated alias directly should do so *outside* the
monitored ``with`` block, the way the save/load and dagua-audit tests below
build their fixtures before entering it.

Detection note: ``warn_deprecated_alias`` (``torchlens/_deprecations.py``)
calls ``warnings.warn(..., stacklevel=6)``, a depth tuned for its most common
call pattern (a deprecated flat kwarg handled a few frames below a top-level
``tl.trace``/``tl.record`` call). For deprecated calls several frames deeper
or shallower inside torchlens -- e.g. ``build_render_audit`` walking fields
via ``getattr`` in a loop (deeper), or ``_run_patch`` calling ``Trace.rerun``
one hop below a facets-patching entry point (shallower) -- that fixed
stacklevel resolves to the WRONG frame: verified empirically that it can
overshoot to a torchlens caller several levels up, or land squarely on *this
test file* itself, even though the deprecated call was made deep inside
torchlens with no test-level involvement. So the plain reported ``filename``
cannot be trusted as evidence of non-involvement, and an origin-based
"except this test file" exemption would silently mask exactly this kind of
regression (confirmed live: reintroducing the ``patched_log.rerun(...)`` bug
in ``semantic/patching.py`` produced a warning whose reported filename WAS
this test file, purely due to stacklevel arithmetic). ``_internal_self_deprecations``
therefore also matches the exact self-deprecation message shape
``warn_deprecated_alias`` emits (the same message pattern
``pyproject.toml``'s ``filterwarnings`` uses to downgrade these warnings)
with no origin-based carve-out, which is sound given the "canonical names
only inside monitored blocks" discipline above.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens.experimental import dagua
from torchlens.options import CaptureOptions
from torchlens.semantic import FacetSpec

import example_models

_TORCHLENS_ROOT = Path(tl.__file__).resolve().parent

# Mirrors warn_deprecated_alias's message shape (torchlens/_deprecations.py) and
# the equivalent regex pyproject.toml's filterwarnings uses to downgrade it.
_SELF_DEPRECATION_MESSAGE_RE = re.compile(r"^`.*` is deprecated; use `.*` instead\.")


def _internal_self_deprecations(
    records: list[warnings.WarningMessage],
) -> list[warnings.WarningMessage]:
    """Return the subset of ``records`` that torchlens fired against itself.

    A record counts as an internal self-deprecation when its category is
    ``DeprecationWarning`` or ``UserWarning`` and EITHER its reported warn-site
    is inside the installed ``torchlens`` package, OR its message matches
    ``warn_deprecated_alias``'s distinctive self-deprecation shape. Both
    signals are needed (see module docstring): the reported ``filename`` is
    derived from a fixed ``stacklevel`` that does not reliably land inside
    ``torchlens/`` for every call depth, so the message-shape check is the
    primary, depth-independent signal; the origin check catches any residual
    case the message pattern misses. There is intentionally no
    "except this file" exemption -- see module docstring for why one would be
    unsound here.
    """

    hits = []
    for record in records:
        if not issubclass(record.category, (DeprecationWarning, UserWarning)):
            continue
        try:
            origin = Path(record.filename).resolve()
        except (OSError, ValueError):
            origin = None
        origin_is_internal = origin is not None and (
            origin == _TORCHLENS_ROOT or _TORCHLENS_ROOT in origin.parents
        )
        looks_self_deprecated = bool(_SELF_DEPRECATION_MESSAGE_RE.search(str(record.message)))
        if origin_is_internal or looks_self_deprecated:
            hits.append(record)
    return hits


def _assert_no_internal_self_deprecation(
    records: list[warnings.WarningMessage], label: str
) -> None:
    hits = _internal_self_deprecations(records)
    assert not hits, (
        f"{label}: torchlens internally fired {len(hits)} deprecated-API "
        "warning(s) against its own code (not user/test code):\n"
        + "\n".join(f"  {hit.filename}:{hit.lineno}: {hit.message}" for hit in hits)
    )


class _SelfDepProbeModel(nn.Module):
    """Plain feed-forward model, no facets/dagua involvement."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + 1)


def test_basic_trace_no_internal_self_deprecation() -> None:
    """A plain ``tl.trace`` call fires zero torchlens-internal deprecations."""

    model = _SelfDepProbeModel()
    x = torch.randn(1, 5)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        log = tl.trace(model, x, save=tl.func("relu"))
    try:
        _assert_no_internal_self_deprecation(records, "tl.trace")
    finally:
        log.cleanup()


def test_record_to_trace_no_internal_self_deprecation() -> None:
    """``tl.record`` + ``Recording.to_trace()`` fire zero internal deprecations."""

    model = _SelfDepProbeModel()
    x = torch.randn(1, 5)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        recording = tl.record(model, x, save=tl.func("relu"))
        log = recording.to_trace()
    try:
        _assert_no_internal_self_deprecation(records, "tl.record + Recording.to_trace")
    finally:
        log.cleanup()


class _SelfDepProbeHeadResult(nn.Module):
    """Return two explicit per-head result tensors stacked on a new axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack((x, torch.zeros_like(x)), dim=2)


class _SelfDepProbeAttention(nn.Module):
    """Tiny attention-like module with a writable per-head result facet.

    Deliberately named/shaped distinctly from ``tests/semantic/test_facets_p4.py``'s
    ``PatchableAttention`` (different class name, different submodule name) so
    this module's ``@tl.facets.register`` registration cannot collide with -- or
    shadow -- that file's registration in the process-global facet registry.
    """

    def __init__(self) -> None:
        super().__init__()
        self.result_source = _SelfDepProbeHeadResult()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.result_source(x).sum(dim=2)


class _SelfDepProbeModel2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _SelfDepProbeAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(x)


@tl.facets.register(
    class_name="_SelfDepProbeAttention",
    target_scope="module",
    facets=("result", "n_heads", "head"),
)
def _self_dep_probe_attention_recipe(module: Any) -> dict[str, Any]:
    """Expose a writable per-head result facet for ``_SelfDepProbeAttention``."""

    trace = module.trace
    result_module = trace.modules[f"{module.address}.result_source"]
    result_op = trace.ops[result_module.calls[0].output_ops[0]]
    return {
        "result": FacetSpec.from_home(result_op, recipe_id="self_dep_probe_attention"),
        "n_heads": 2,
        "head": module.facets.head,
    }


def _self_dep_probe_metric(log: Any) -> torch.Tensor:
    return log[log.output_layers[0]].out[0, 0, 0]


def test_semantic_patching_no_internal_self_deprecation() -> None:
    """Activation patching (which forks + reruns a trace) fires zero internal deprecations.

    ``activation_patch_attention_heads`` -> ``_run_patch`` is exactly the code
    path fixed in ``semantic/patching.py`` to call ``Trace.run()`` instead of
    the deprecated ``Trace.rerun()`` alias.
    """

    model = _SelfDepProbeModel2()
    clean = torch.zeros(1, 3, 4)
    corrupted = torch.zeros(1, 3, 4)
    clean[0, 0, 0] = 2.0
    clean = clean.requires_grad_()
    corrupted = corrupted.requires_grad_()

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        scores = tl.facets.patching.activation_patch_attention_heads(
            model, clean, corrupted, _self_dep_probe_metric
        )
    assert scores.shape == (1, 2)
    _assert_no_internal_self_deprecation(records, "activation_patch_attention_heads")


def test_dagua_render_audit_no_internal_self_deprecation() -> None:
    """``build_render_audit`` (the round-3 deprecated-field skip-list fix) fires zero warnings.

    ``build_render_audit`` walks ``Trace``/``Op``/``Module`` public fields via
    ``_list_public_fields``, which must skip the deprecated
    ``conditional_{then,elif,else}_entry_edges`` properties rather than
    accessing them and triggering their own deprecation warning.
    """

    model = example_models.SimpleFF().eval()
    log = tl.trace(model, torch.rand(5, 5), capture=CaptureOptions(layers_to_save=None))
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            audit = dagua.build_render_audit(log).to_dict()
        assert "trace_unused" in audit
        _assert_no_internal_self_deprecation(records, "dagua.build_render_audit")
    finally:
        log.cleanup()


def test_save_load_no_internal_self_deprecation(tmp_path: Path) -> None:
    """``tl.save``/``tl.load`` round-trip fires zero internal deprecations."""

    model = _SelfDepProbeModel()
    x = torch.randn(1, 5)
    log = tl.trace(model, x, save=tl.func("relu"))
    out_path = tmp_path / "self_dep_probe.tlspec"
    try:
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            tl.save(log, out_path)
            reloaded = tl.load(out_path)
        _assert_no_internal_self_deprecation(records, "tl.save + tl.load")
    finally:
        log.cleanup()
        if hasattr(reloaded, "cleanup"):
            reloaded.cleanup()
