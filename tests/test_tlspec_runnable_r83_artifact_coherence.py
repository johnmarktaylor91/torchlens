"""Artifact self-coherence closures (r83 C3, r82 secA S3 + S4).

S3 -- the callable a loaded artifact EXECUTES comes from ``runnable/descriptor.json ->
callable_registry[].key``. The SAME call is named independently by
the site and op JSON records under ``records/``
(``function_path``, ``func_name`` / ``func_id.qualname``). Nothing reconciled them, so
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
``_PORTABLE_STATE_SPEC``. Any trace that went through ``__setstate__`` -- which
``cache=True`` makes routine -- tripped the field-catalog tripwire, so
``save(level="runnable")`` failed outright on the SECOND capture with
``ArtifactError: Trace._pending_live_fire_records is missing from
_PORTABLE_STATE_SPEC``. A user-facing save failure on a normal flow, not merely a
test red. The broader cross-effort field-catalog drift stays deferred.
"""

from __future__ import annotations

import json
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from tests._format3_envelope import (
    read_runnable_descriptor,
    write_runnable,
)
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(replayable=True, capture_container_structure=True, cache=False)


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

    capture = CaptureOptions(replayable=True, capture_container_structure=True, cache=cache)
    trace = tl.trace(_Simple(), torch.randn(2, 4), capture=capture)
    trace.save(path, level="runnable", include_weights=True)
    return path


def _recompute_runtime_fingerprints(run: dict[str, Any]) -> None:
    """Rewrite each call's ``runtime_fingerprint`` per the loader recipe.

    Naive descriptor tampers are now caught at PARSE time by c9904d19's
    ``_verify_runtime_fingerprints`` check (sha256 over callable key, argument
    names, arity, output slot shapes/dtypes, and execution context), which
    would pre-empt the deeper gates this suite pins. Recomputing the
    fingerprints after the tamper keeps these tests exercising the deeper
    resolver gates (``SEMANTIC_DRIFT`` / ``UNRESOLVED_QUALNAME``), which is
    their point.
    """

    keys_by_registry_id = {entry["registry_id"]: entry["key"] for entry in run["callable_registry"]}
    slots_by_id = {slot["slot_id"]: slot for slot in run["tensor_slots"]}
    for call in run["calls"]:
        key = keys_by_registry_id[call["registry_id"]]
        context = call["execution_context"]
        payload = {
            "callable": {
                "namespace": key["namespace"],
                "qualname": key["qualname"],
                "dispatch_kind": key["dispatch_kind"],
                "version": key["version"],
                "import_path": key.get("import_path"),
            },
            "argument_names": list(call["argument_names"]),
            "num_positional_args": int(call["num_positional_args"]),
            "num_keyword_args": int(call["num_keyword_args"]),
            "outputs": [
                {
                    "shape": list(slots_by_id[slot_id]["shape"]),
                    "dtype": slots_by_id[slot_id]["dtype"],
                }
                for slot_id in call["output_slot_ids"]
            ],
            "execution_context": {
                "autocast": [
                    {
                        "device_type": entry["device_type"],
                        "enabled": entry["enabled"],
                        "dtype": entry.get("dtype"),
                    }
                    for entry in context["autocast"]
                ],
                "grad_enabled": context["grad_enabled"],
                "inference_mode": context["inference_mode"],
            },
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        call["runtime_fingerprint"] = sha256(serialized.encode("utf-8")).hexdigest()


def _retarget_registry(source: Path, target: Path, frm: str, to: str) -> Path:
    """Copy an artifact and rewrite one ``callable_registry`` qualname."""

    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)
    run = read_runnable_descriptor(target)
    hits = 0
    for entry in run["callable_registry"]:
        if entry["key"]["qualname"] == frm:
            entry["key"]["qualname"] = to
            hits += 1
    assert hits == 1, f"expected exactly one {frm!r} registry entry, found {hits}"
    _recompute_runtime_fingerprints(run)
    write_runnable(target, run)
    return target


# --------------------------------------------------------------------------- #
# S3 -- the tampered registry must not run.
# --------------------------------------------------------------------------- #


# Kept: shares a parametrize list that is not trimmable per id with kept params.
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

    The reconciliation lives in the RESOLVER's success sink, so it fires only
    for a swap that RESOLVES to a real, signature-compatible callable -- and the
    refusal flows through the resolver's aggregate readiness/``ReattachError``
    path, exactly like every other resolver refusal (see
    ``test_unresolvable_key_keeps_the_resolver_reattach_path``). The
    arity-incompatible ``F.linear -> F.bilinear`` still refuses first via
    signature drift; its refusal is unchanged.
    """

    clean = _save(tmp_path / "clean.tlspec")
    tampered = _retarget_registry(clean, tmp_path / "tampered.tlspec", frm, to)

    loaded = tl.load(tampered)
    with pytest.raises(Exception) as excinfo:
        loaded.run(inputs=torch.randn(2, 4))
    assert type(excinfo.value).__name__ != "AssertionError"


# Kept: sole branch coverage of _io/runnable_load.py; a sole mutation kill.
def test_unresolvable_key_keeps_the_resolver_reattach_path(tmp_path: Path) -> None:
    """An UNRESOLVABLE key must keep the resolver's richer aggregate-report path.

    The reconciliation must NOT steal the diagnostic from an unresolvable key
    (``torch.definitely_absent``): that case is already handled by the resolver
    with an ``UNRESOLVED_QUALNAME`` diagnostic and a ``ReattachError`` carrying
    the full aggregate report -- NOT the coarse analysis-only refusal. This pins
    the distinction the r83 S3 relocation exists to preserve (r83 regression 2).
    """

    from torchlens._runnable import ReadinessStatus, RunnableErrorCode
    from torchlens.errors import ReattachError

    clean = _save(tmp_path / "clean.tlspec")
    tampered = _retarget_registry(
        clean, tmp_path / "tampered.tlspec", "__add__", "definitely_absent"
    )

    loaded = tl.load(tampered)
    report = loaded.readiness
    assert report.status is ReadinessStatus.UNAVAILABLE
    # The resolver owns this case with an "unresolvable callable" diagnostic (the
    # exact code -- UNRESOLVED_QUALNAME vs CALLABLE_REMOVED -- depends on the
    # namespace ladder rung). What matters is it is NOT the reconciliation's
    # SEMANTIC_DRIFT, so the reconciliation did not pre-empt the resolver.
    codes = {diagnostic.code for diagnostic in report.diagnostics}
    assert RunnableErrorCode.SEMANTIC_DRIFT not in codes, "must not pre-empt the resolver"
    assert codes & {
        RunnableErrorCode.UNRESOLVED_QUALNAME,
        RunnableErrorCode.CALLABLE_REMOVED,
    }, f"expected a resolver unresolvable-callable diagnostic, got {codes}"
    # And run() must raise the resolver's ReattachError carrying the full report,
    # NOT the coarse analysis-only RunCapabilityUnavailableError.
    with pytest.raises(ReattachError) as captured:
        loaded.run(inputs=torch.randn(2, 4))
    assert captured.value.fields["readiness"] is loaded.readiness


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


# --------------------------------------------------------------------------- #
# S4 -- cache=True + a repeat capture must save.
# --------------------------------------------------------------------------- #


# Kept: sole branch coverage of _capture/prepare/cache.py, _capture/prepare/door.py, _capture_state_helpers.py and more.
def test_cache_true_repeat_capture_saves_runnable(tmp_path: Path) -> None:
    """Pre-fix the SECOND capture raised the field-catalog tripwire outright.

    ``ArtifactError: Trace._pending_live_fire_records is missing from
    _PORTABLE_STATE_SPEC`` -- a user-facing save failure on a normal flow.
    """

    for index in range(3):
        _save(tmp_path / f"cached_{index}.tlspec", cache=True)


# Kept: its run phase reached no library code in the per-id coverage run, so it has no measurement (shrink rule 3).
def test_pending_live_fire_records_is_registered() -> None:
    """The field must be in the catalog, so the tripwire stays armed for others."""

    from torchlens._data_classes.trace import FieldPolicy, Trace

    spec: dict[str, Any] = Trace._PORTABLE_STATE_SPEC
    assert spec["_pending_live_fire_records"] is FieldPolicy.DROP
