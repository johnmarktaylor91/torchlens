"""Round-45 immunizer: exhaustive classification of Tensor getset-descriptor properties.

The safe pure-read tensor-property surface (``x.T`` / ``x.mT`` / ``x.H`` / ``x.mH`` /
``x.real`` / ``x.imag``) is the single canonical set
``torchlens.utils._callable_safety._PURE_TENSOR_PROPERTY_NAMES``, shared by the capture-side
keyer (``torchlens.backends.torch.ops``), the load-side resolver
(``torchlens._io.runnable_load._safe_tensor_property_getter``) and the security gate's
recognized-operator predicate. r45 made it STRUCTURALLY computed -- a descriptor is admitted
iff its getter returns a storage-sharing, autograd-PRESERVING, non-mutating view -- so a
future torch tensor property is auto-classified instead of being silently over-denied (the
r44 corr1_1 / secF_1 ``.H`` / ``.mH`` finding).

This module is the next-sibling immunizer: it enumerates EVERY live
``torch._C.TensorBase`` getset descriptor and, via an INDEPENDENT structural oracle
(deliberately not importing the shipped ``_pure_view``), asserts each descriptor is either
admitted-as-pure-view or denied-with-a-concrete-reason. It FAILS on any descriptor that is
unclassified, or misclassified in either direction (over-deny of a genuine pure view, or
over-admit of a mutating / autograd-detaching / metadata property). ``.H`` / ``.mH`` are
pinned admitted; ``.data`` is pinned denied (it shares storage but DETACHES from autograd and
is a live lvalue mutation channel); a future torch property that is not a pure view is RED.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import build_sparse_run_descriptor
from torchlens._io.runnable_load import _safe_tensor_property_getter
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, ReadinessStatus
from torchlens.utils._callable_safety import (
    _PURE_TENSOR_PROPERTY_NAMES,
    _iter_tensor_getset_descriptor_names,
    _pure_view,
    _unwrap_capture_wrapper,
    is_pure_forward_callable,
)

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)

# The r45 converged classification for torch 2.x. Pinned so an unexpected torch runtime is
# a LOUD, inspectable failure rather than a silent surface change.
_EXPECTED_ADMITTED = frozenset({"T", "mT", "H", "mH", "real", "imag"})
# A representative slice of properties that MUST stay denied (metadata / layout / autograd
# state / the autograd-detaching ``.data`` alias). Not exhaustive of the deny side -- the
# structural oracle below covers the rest -- but pins the ones most likely to regress.
_EXPECTED_DENIED_SAMPLE = frozenset(
    {"data", "grad", "grad_fn", "requires_grad", "shape", "dtype", "device", "layout", "names"}
)

# Exhaustive reason vocabulary for the INDEPENDENT structural oracle. Every live descriptor
# must resolve to exactly one of these -- an "unknown" reason can never arise, so a
# descriptor that the oracle cannot label would be a code bug, not an "unclassified" pass.
_KNOWN_REASONS = frozenset(
    {
        "pure_view",
        "non_tensor",
        "mutates_source",
        "not_storage_sharing",
        "autograd_detached",
        "undefined_all_dtypes",
    }
)


def _indep_classify(name: str) -> str:
    """Independently classify a Tensor getset descriptor by structural probe.

    Deliberately re-implements the pure-view rule (rather than importing the shipped
    ``_pure_view``) so the shipped set is checked against an independent oracle: a
    regression that re-freezes the set to a hand-list, or admits a non-view property, is
    caught because this oracle recomputes from the live torch runtime. Mirrors the shipped
    predicate exactly: the FIRST defined dtype whose getter violates a clause denies with
    that clause's reason; admission requires >=1 defined dtype and every defined dtype
    passing every clause.
    """

    saw_defined = False
    for probe in (
        torch.randn(2, 3, requires_grad=True),
        torch.randn(2, 3, dtype=torch.complex64, requires_grad=True),
    ):
        before = probe.detach().clone()
        version = probe._version
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = getattr(probe, name)
        except Exception:
            continue
        saw_defined = True
        if not isinstance(result, torch.Tensor):
            return "non_tensor"
        if probe._version != version or not torch.equal(probe.detach(), before):
            return "mutates_source"
        try:
            shares_storage = (
                result.untyped_storage().data_ptr() == probe.untyped_storage().data_ptr()
            )
        except Exception:
            return "not_storage_sharing"
        if not shares_storage:
            return "not_storage_sharing"
        if probe.requires_grad and not result.requires_grad:
            return "autograd_detached"
    if not saw_defined:
        return "undefined_all_dtypes"
    return "pure_view"


def test_r45_every_tensor_getset_descriptor_is_classified() -> None:
    """Enumerate ALL TensorBase getset descriptors; each is admitted-pure-view or denied.

    The core immunizer: the shipped canonical set must equal the independent structural
    oracle's pure-view set, so it catches over-DENY (a genuine pure view refused -- the r44
    ``.H`` / ``.mH`` bug) AND over-ADMIT (a mutating / detaching / metadata property leaking
    in). Every descriptor must carry a known denial reason, so nothing is silently
    unclassified.
    """

    descriptors = tuple(_iter_tensor_getset_descriptor_names())
    descs = set(descriptors)

    # Enumeration is real and stable (torch 2.x exposes ~47 tensor getset descriptors).
    assert len(descriptors) == len(descs), "duplicate descriptor names enumerated"
    assert len(descs) >= 40, f"suspiciously few tensor getset descriptors: {len(descs)}"

    shipped_admitted = set(_PURE_TENSOR_PROPERTY_NAMES)
    # The canonical set is drawn from live descriptors -- never a stale hand-list name.
    assert shipped_admitted <= descs, shipped_admitted - descs

    reasons = {name: _indep_classify(name) for name in descs}
    # No descriptor escapes the reason vocabulary -> nothing is "unclassified".
    assert set(reasons.values()) <= _KNOWN_REASONS, set(reasons.values()) - _KNOWN_REASONS

    indep_admitted = {name for name, reason in reasons.items() if reason == "pure_view"}
    denied = descs - shipped_admitted

    # Policy == independent oracle (both directions): the single strongest assertion.
    assert shipped_admitted == indep_admitted, {
        "over_denied": indep_admitted - shipped_admitted,
        "over_admitted": shipped_admitted - indep_admitted,
    }

    # Every descriptor is classified exactly once (admitted XOR denied), and every denied
    # descriptor carries a concrete, non-"pure_view" reason.
    for name in descs:
        assert (name in shipped_admitted) ^ (name in denied), name
    for name in denied:
        assert reasons[name] != "pure_view", name
        assert reasons[name] in _KNOWN_REASONS, (name, reasons[name])

    # Explicit intent pins (documented behavior; independent of the oracle above).
    assert _EXPECTED_ADMITTED <= shipped_admitted, _EXPECTED_ADMITTED - shipped_admitted
    assert shipped_admitted == _EXPECTED_ADMITTED, shipped_admitted ^ _EXPECTED_ADMITTED
    assert {"H", "mH"} <= shipped_admitted
    assert "data" in denied
    assert reasons["data"] == "autograd_detached"  # the discriminator, structurally.
    for name in _EXPECTED_DENIED_SAMPLE:
        assert name in descs, f"expected-denied property vanished from torch: {name}"
        assert name in denied, f"a denied property was wrongly admitted: {name}"


def test_r45_shipped_pure_view_predicate_agrees_with_shipped_set() -> None:
    """The shipped ``_pure_view`` predicate agrees with the shipped canonical set.

    Self-consistency belt: the set IS computed from ``_pure_view``, so a divergence would
    mean the set was hand-frozen while the predicate was left live (a regression shape).
    """

    for name in _iter_tensor_getset_descriptor_names():
        assert _pure_view(name) == (name in _PURE_TENSOR_PROPERTY_NAMES), name
    # ``.data`` is denied by the autograd-preservation clause, not a carve-out.
    assert _pure_view("data") is False
    for name in ("T", "mT", "H", "mH", "real", "imag"):
        assert _pure_view(name) is True, name


@pytest.mark.parametrize("name", sorted(_EXPECTED_ADMITTED))
def test_r45_admitted_property_getter_resolves_and_is_pure_forward(name: str) -> None:
    """Each admitted property yields a loader getter that clears the pure-forward gate."""

    getter = _safe_tensor_property_getter(FunctionRegistryKey("torch.Tensor", name, "method"))
    assert getter is not None, f"loader refused safe property key: {name}"
    assert is_pure_forward_callable(getter), f"safe tensor property WRONGLY DENIED: {name}"
    # A synthetic getter can never be admitted on identity -- it needs the property-name rung.
    real = _unwrap_capture_wrapper(getter)
    assert callable(real)


@pytest.mark.parametrize("name", sorted(_EXPECTED_DENIED_SAMPLE))
def test_r45_denied_property_getter_refused_by_loader(name: str) -> None:
    """Denied properties (``.data`` / autograd / metadata) get NO synthetic getter."""

    assert (
        _safe_tensor_property_getter(FunctionRegistryKey("torch.Tensor", name, "method")) is None
    ), name


class _HModel(nn.Module):
    """Read the conjugate-transpose view property ``x.H``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x.H + 1``."""

        return x.H + 1


class _MHModel(nn.Module):
    """Read the batched conjugate-transpose view property ``x.mH``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x.mH + 1``."""

        return x.mH + 1


@pytest.mark.parametrize(
    ("model_factory", "prop"),
    ((_HModel, "H"), (_MHModel, "mH")),
)
def test_r45_H_mH_producer_keys_are_tensor_property_methods(
    model_factory: type[nn.Module], prop: str
) -> None:
    """``.H`` / ``.mH`` capture as ``("torch.Tensor", <name>, "method")`` -- not custom keys.

    The r44 corr1_1 / secF_1 producer defect: ``.H`` fell through to an unresolvable
    ``custom`` ``getset_descriptor.__get__`` key and ``.mH`` failed save with
    ``unsupported_tensor_constant``. Both now key exactly like ``.T`` / ``.mT``.
    """

    x = torch.arange(6.0).reshape(2, 3)
    trace = tl.trace(model_factory(), x.clone(), capture=_CAPTURE)
    descriptor = build_sparse_run_descriptor(trace)
    keys = [entry.key for entry in descriptor.callable_registry]

    assert FunctionRegistryKey("torch.Tensor", prop, "method") in keys, keys
    # No unresolvable raw descriptor key survives capture.
    assert not any(key.qualname == "getset_descriptor.__get__" for key in keys), keys


@pytest.mark.parametrize(
    ("model_factory", "prop", "x"),
    (
        (_HModel, "H", torch.arange(6.0).reshape(2, 3)),
        (_MHModel, "mH", torch.arange(6.0).reshape(2, 3)),
        (
            _HModel,
            "H",
            torch.tensor(
                [[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]], dtype=torch.complex64
            ),
        ),
        (
            _MHModel,
            "mH",
            torch.tensor(
                [[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]], dtype=torch.complex64
            ),
        ),
    ),
)
def test_r45_H_mH_round_trip_verified(
    tmp_path: Path, model_factory: type[nn.Module], prop: str, x: torch.Tensor
) -> None:
    """``.H`` / ``.mH`` round-trip runnable save/load/run to VERIFIED on real and complex."""

    trace = tl.trace(model_factory(), x.clone(), capture=_CAPTURE)
    path = tmp_path / f"{prop}.tlspec"
    trace.save(path, level="runnable")
    loaded = tl.load(path)
    result = loaded.run(x.clone(), seed=0, on_divergence="return_diverged")

    assert loaded.readiness.status is ReadinessStatus.READY
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, getattr(x, prop) + 1)
