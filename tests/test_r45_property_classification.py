"""Round-45 immunizer: exhaustive classification of Tensor getset-descriptor properties.

The safe pure-read tensor-property surface (``x.T`` / ``x.mT`` / ``x.H`` / ``x.mH`` /
``x.real`` / ``x.imag``) is the single canonical set
``torchlens._utils._callable_safety._PURE_TENSOR_PROPERTY_NAMES``, shared by the capture-side
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

from torchlens._utils._callable_safety import (
    _PURE_TENSOR_PROPERTY_NAMES,
    _iter_tensor_getset_descriptor_names,
    _pure_view,
)
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


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
