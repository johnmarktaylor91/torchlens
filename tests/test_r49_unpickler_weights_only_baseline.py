"""Round-49 whole-class immunizer: SafeBundleUnpickler is a STRICT SUBSET of torch's
``weights_only`` baseline (secA_1).

A default (trust=False) bundle-metadata unpickle must NOT be able to
    (a) CONSTRUCT a torch storage -- ``UntypedStorage(N)`` allocates N raw bytes at
        ``tl.load()`` time (a multi-GiB out-of-memory DoS) and is the sole source of an
        uninitialized-heap storage; nor
    (b) REBIND tensor storage -- reach ``Tensor.__setstate__`` / ``Tensor.set_`` via the
        pickle BUILD opcode (an uninitialized-heap tensor), which is INVISIBLE to
        ``find_class`` because BUILD acts on an object already on the stack.

Both vectors are closed by two layers in ``torchlens/_io/_safe_unpickle.py``:
  Layer 1: torch storage constructors are denied by IDENTITY at ``find_class``
           (``_is_torch_storage_type``), including the two real
           ``torch.storage.{Typed,Untyped}Storage`` classes that torch's OWN baseline
           (``_weights_only_unpickler._get_allowed_globals()``) admits.
  Layer 2: ``SafeBundleUnpickler`` is a pure-Python ``pickle._Unpickler`` whose REBUILT
           opcode ``dispatch`` gates BUILD / REDUCE / NEWOBJ / NEWOBJ_EX. (Overriding
           the methods on a C ``pickle.Unpickler`` -- or even on ``pickle._Unpickler``
           WITHOUT rebuilding ``dispatch`` -- is INERT; that trap is pinned below.)

The whole-class invariant is CONSTRUCTOR-SEMANTICS, not a string subset: NO ``(module,
name)`` the unpickler admits resolves to a CONSTRUCTABLE torch storage, and no BUILD can
rebind a tensor/storage. A string-subset-of-baseline assertion is proven too weak -- the
baseline set itself lists the two real storage classes -- so a future permissive drift
toward a constructable storage/tensor ctor makes this immunizer go RED.
"""

from __future__ import annotations

import io
import pickle

import pytest
import torch
from torch import nn

from torchlens._io._safe_unpickle import (
    _SAFE_EXPLICIT_GLOBALS,
    _SAFE_TORCHLENS_TYPES,
    SafeBundleUnpickler,
    _is_torch_storage_instance,
    _is_torch_storage_type,
    _PurePythonBundleUnpickler,
)
from torchlens.options import CaptureOptions

_CAP = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


# --------------------------------------------------------------------------- #
# 1. Storage-family construction is denied (exhaustive over the storage surface).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 2. Layer-2 opcode belts fire even when a storage type reaches the stack by a
#    route OTHER than find_class (defense-in-depth, white-box).
# --------------------------------------------------------------------------- #


def test_reduce_belt_denies_storage_ctor_bypassing_find_class() -> None:
    """``load_reduce`` refuses constructing a storage placed directly on the stack."""

    u = _PurePythonBundleUnpickler(io.BytesIO(b""))
    u.stack = [torch.UntypedStorage, (16,)]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError, match="via REDUCE"):
        u.load_reduce()


# --------------------------------------------------------------------------- #
# 3. BUILD cannot rebind tensor/storage state (the __setstate__/set_ vector).
# --------------------------------------------------------------------------- #


def test_build_belt_denies_tensor_rebind_white_box() -> None:
    """``load_build`` refuses a BUILD whose target is a torch Tensor / Storage."""

    u = _PurePythonBundleUnpickler(io.BytesIO(b""))
    # BUILD stack layout: inst = stack[-2], state = stack[-1].
    u.stack = [torch.zeros(2), {"malicious": "state"}]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError, match="BUILD applied to a torch"):
        u.load_build()

    storage = torch.zeros(2).untyped_storage()
    assert _is_torch_storage_instance(storage)
    u.stack = [storage, {"x": 1}]  # type: ignore[attr-defined]
    with pytest.raises(pickle.UnpicklingError, match="BUILD applied to a torch"):
        u.load_build()


# --------------------------------------------------------------------------- #
# 4. Interposition non-vacuity: the rebuilt dispatch actually points at the
#    subclass overrides (guards the "method-override-alone-is-inert" trap).
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 5. Constructor-semantics drift guard (the anti-drift immunizer). NOT a string
#    subset of the baseline (which itself lists the storage classes).
# --------------------------------------------------------------------------- #


def _resolve_or_none(module: str, name: str) -> object:
    try:
        return SafeBundleUnpickler(io.BytesIO(b"")).find_class(module, name)
    except Exception:  # noqa: BLE001 - a denied/unresolvable name resolves to nothing
        return None


def test_admit_set_constructs_no_storage_strict_subset_of_baseline() -> None:
    """No (module,name) the unpickler admits resolves to a CONSTRUCTABLE storage.

    Enumerated over BOTH torch's ``weights_only`` baseline global set (which DOES admit
    the real ``torch.storage.{Typed,Untyped}Storage`` classes) AND TorchLens's own
    explicit allowlists. The admitted-constructable-storage set must be EMPTY -- proving
    a STRICT SUBSET of the baseline by constructor semantics, and making any future
    re-admission of a storage ctor a RED test.
    """

    from torch import _weights_only_unpickler as wou

    baseline = wou._get_allowed_globals()
    baseline_storage = {k: v for k, v in baseline.items() if _is_torch_storage_type(v)}
    # Sanity: the baseline really does admit storages (else the subset claim is vacuous).
    assert baseline_storage, "torch baseline unexpectedly lists no constructable storage"

    admitted_storage: list[str] = []
    for key in baseline_storage:
        module, _, name = key.rpartition(".")
        resolved = _resolve_or_none(module, name)
        if _is_torch_storage_type(resolved):
            admitted_storage.append(key)

    # TorchLens's own admit allowlists must never carry a storage ctor either.
    for module, name in set(_SAFE_EXPLICIT_GLOBALS) | set(_SAFE_TORCHLENS_TYPES):
        resolved = _resolve_or_none(module, name)
        if _is_torch_storage_type(resolved):
            admitted_storage.append(f"{module}.{name}")

    assert admitted_storage == [], (
        "SafeBundleUnpickler admits a CONSTRUCTABLE storage ctor -- it is no longer a "
        f"strict subset of the weights_only baseline: {admitted_storage}"
    )


# --------------------------------------------------------------------------- #
# 6. Over-trigger pins: legitimate bundles still load AND run under both layers.
# --------------------------------------------------------------------------- #


class _StatefulLinear(nn.Module):
    """Runnable-eligible parameterized graph with a persistent buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.register_buffer("scale", torch.ones(4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(value)) * self.scale


class _ControlFlow(nn.Module):
    """Control-flow graph; a portable save embeds a predicate tensor + method refs."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.sum() > 0:
            return torch.relu(value) * 2
        return torch.sigmoid(value) - 1
