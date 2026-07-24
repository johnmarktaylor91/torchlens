"""Artifact self-coherence closures (r83 C3, r82 secA S3 + S4).

S3 -- the callable a loaded artifact EXECUTES comes from ``manifest.json ->
run.callable_registry[].key``. The SAME call is named independently by
``manifest.json -> sites[].function_path`` and by ``metadata.pkl ->
layer_list[].func_name`` / ``.func_id.qualname``. Nothing reconciled them, so
editing ONE JSON field (``Tensor.__add__`` -> ``__sub__``) left an internally
self-contradictory artifact that RAN and reported ``VERIFIED`` with different
numbers while three other persisted fields still said ``add``.

Not weaponizable -- an attacker would simply capture the wrong program honestly,
which is out of scope by contract (coherent reauthoring) -- so this is
defence-in-depth against artifact CORRUPTION and partial rewrites, where a typed
refusal beats a silent ``VERIFIED``. The attestation lane already anchors it when
``include_activations=True`` makes it eligible; the gap was the DEFAULT artifact,
whose run is ``not_applicable``.

S4 -- ``Trace._pending_live_fire_records`` is reinstated by ``Trace.__setstate__``
and listed in ``_io/rehydrate.py``, but was never registered in
``PORTABLE_STATE_SPEC``. Any trace that went through ``__setstate__`` -- which
``cache=True`` makes routine -- tripped the field-catalog tripwire, so
``save(level="runnable")`` failed outright on the SECOND capture with
``TorchLensIOError: Trace._pending_live_fire_records is missing from
PORTABLE_STATE_SPEC``. A user-facing save failure on a normal flow, not merely a
test red. The broader cross-effort field-catalog drift stays deferred.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness

_CAPTURE = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=False)


class _Simple(nn.Module):
    """Linear + buffer add: one function call and one dunder method call."""

    def __init__(self) -> None:
        """Build a linear layer and a registered buffer."""

        super().__init__()
        torch.manual_seed(0)
        self.lin = nn.Linear(4, 4)
        self.register_buffer("b", torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a function dispatch followed by a dunder-method dispatch."""

        return self.lin(x) + self.b


def _save(path: Path, *, cache: bool = False) -> Path:
    """Capture ``_Simple`` and save a runnable artifact."""

    capture = CaptureOptions(intervention_ready=True, capture_container_structure=True, cache=cache)
    trace = tl.trace(_Simple(), torch.randn(2, 4), capture=capture)
    trace.save(path, level="runnable", include_weights=True)
    return path


def _retarget_registry(source: Path, target: Path, frm: str, to: str) -> Path:
    """Copy an artifact and rewrite one ``callable_registry`` qualname."""

    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)
    manifest = json.loads((target / "manifest.json").read_text())
    hits = 0
    for entry in manifest["run"]["callable_registry"]:
        if entry["key"]["qualname"] == frm:
            entry["key"]["qualname"] = to
            hits += 1
    assert hits == 1, f"expected exactly one {frm!r} registry entry, found {hits}"
    (target / "manifest.json").write_text(json.dumps(manifest))
    return target


# --------------------------------------------------------------------------- #
# S3 -- the tampered registry must not run.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "frm, to",
    [
        ("__add__", "__sub__"),
        ("__add__", "__mul__"),
        ("__add__", "__truediv__"),
        ("linear", "bilinear"),
    ],
    ids=["add_to_sub", "add_to_mul", "add_to_truediv", "linear_to_bilinear"],
)
def test_self_contradictory_callable_registry_is_refused(tmp_path: Path, frm: str, to: str) -> None:
    """A single-field registry edit must be refused, never silently executed.

    Pre-fix the three signature-COMPATIBLE swaps all RAN and reported
    ``VERIFIED`` with numerically different output; only the arity-incompatible
    ``F.linear -> F.bilinear`` was caught, and only later, by signature drift.
    """

    clean = _save(tmp_path / "clean.tlspec")
    tampered = _retarget_registry(clean, tmp_path / "tampered.tlspec", frm, to)

    loaded = tl.load(tampered)
    with pytest.raises(Exception) as excinfo:
        loaded.run(inputs=torch.randn(2, 4))
    assert type(excinfo.value).__name__ != "AssertionError"


@pytest.mark.smoke
def test_tampered_registry_leaves_a_typed_diagnostic(tmp_path: Path) -> None:
    """The refusal must carry the typed diagnostic, not just fail opaquely."""

    from torchlens.runnable import ReadinessStatus, RunnableErrorCode

    clean = _save(tmp_path / "clean.tlspec")
    tampered = _retarget_registry(clean, tmp_path / "tampered.tlspec", "__add__", "__sub__")

    loaded = tl.load(tampered)
    report = loaded.readiness
    assert report.status is ReadinessStatus.UNAVAILABLE
    codes = {diagnostic.code for diagnostic in report.diagnostics}
    assert RunnableErrorCode.CONTEXT_FIELD_INVALID in codes
    assert any("callable_registry" in str(diagnostic.message) for diagnostic in report.diagnostics)


@pytest.mark.smoke
def test_untampered_artifact_still_runs_verified(tmp_path: Path) -> None:
    """ZERO COLLATERAL: the anchor runs on every load and must pass honest ones.

    The reconciliation is exact for method, function, dunder and in-place
    dispatch; only two spellings differ between the two records at all
    (``__neg__``/``neg``, ``__pow__``/``pow``) and the operator-dunder
    normalization collapses exactly those.
    """

    path = _save(tmp_path / "clean.tlspec")
    result = tl.load(path).run(inputs=torch.randn(2, 4))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


class _MixedDispatch(nn.Module):
    """Exercises the dispatch spellings the two records name differently."""

    def __init__(self) -> None:
        """Register the state the ops consume."""

        super().__init__()
        self.register_buffer("b", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix dunder, negation, power, in-place and method dispatch."""

        y = x + self.b
        y = -y
        y = y**2
        y = y.clone()
        y += self.b
        return y.reshape(-1).contiguous()


@pytest.mark.smoke
def test_dunder_and_inplace_spellings_do_not_false_refuse(tmp_path: Path) -> None:
    """``__neg__``/``neg`` and ``__pow__``/``pow`` must normalize, not refuse.

    These are the ONLY two measured pairs where the site keeps the operator
    dunder while the registry records the torch method. In-place spellings
    (``__iadd__``, ``relu_``) keep their distinguishing characters, so they can
    never normalize onto their out-of-place siblings.
    """

    path = tmp_path / "mixed.tlspec"
    trace = tl.trace(_MixedDispatch(), torch.randn(4), capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    result = tl.load(path).run(inputs=torch.randn(4))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


# --------------------------------------------------------------------------- #
# S4 -- cache=True + a repeat capture must save.
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_cache_true_repeat_capture_saves_runnable(tmp_path: Path) -> None:
    """Pre-fix the SECOND capture raised the field-catalog tripwire outright.

    ``TorchLensIOError: Trace._pending_live_fire_records is missing from
    PORTABLE_STATE_SPEC`` -- a user-facing save failure on a normal flow.
    """

    for index in range(3):
        _save(tmp_path / f"cached_{index}.tlspec", cache=True)


@pytest.mark.smoke
def test_cached_repeat_capture_still_replays_verified(tmp_path: Path) -> None:
    """The saved cached artifact must also load and replay faithfully."""

    _save(tmp_path / "warm.tlspec", cache=True)
    path = _save(tmp_path / "cached.tlspec", cache=True)
    result = tl.load(path).run(inputs=torch.randn(2, 4))
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert not result.report.poisoned


@pytest.mark.smoke
def test_pending_live_fire_records_is_registered() -> None:
    """The field must be in the catalog, so the tripwire stays armed for others."""

    from torchlens.data_classes.trace import FieldPolicy, Trace

    spec: dict[str, Any] = Trace.PORTABLE_STATE_SPEC
    assert spec["_pending_live_fire_records"] is FieldPolicy.DROP
