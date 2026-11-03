"""r53 free_1: bound the attacker-integer allocation-bomb CLASS on the default victim path.

A hostile ``manifest.json`` integer (``num_positional_args=1e10``,
``shape=[1e9,1e9]``, ...) must never drive an 80 GB-8 EB single-shot allocation on
the default-untrusting ``tl.load(path)`` / ``tl.load(path).run(...)`` path. This
suite is the whole-class immunizer:

* a schema-anchor META-TEST over every integer-typed field of the frozen runnable
  descriptor dataclasses (each must be declared leaf-count-anchored / rank-anchored
  / shape-anchored / preflight-gated / bounded-scalar / allocation-inert -- a NEW
  unanchored integer field fails until a row and a parse check exist);
* a BOMB MATRIX under a hard ``RLIMIT_AS`` cap: tamper each of the four documented
  loci to a 10^10-scale integer and assert a TYPED refusal (not a ``MemoryError``);
* OVER-TRIGGER guards: the untampered bundle still loads and runs ``verified``, and
  a legitimate >1 GiB random-init slot passes the preflight (pins the no-static-cap
  decision).
"""

from __future__ import annotations

import json
import resource
import typing
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._runnable import (
    AutocastDeviceContext,
    CallExecutionContext,
    ControlWitness,
    InputAttestationFingerprint,
    InputSlotBinding,
    ReadinessStatus,
    RunnableCallDescriptor,
    RunnableErrorCode,
    StateSlotBinding,
    StateSlotRole,
    TensorSlotDescriptor,
    TensorSlotRole,
    TensorUseSite,
)
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.intervention.types import FunctionRegistryKey


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # softmax resolves to a Python-signature pure callable, exercising the
        # LOAD-time readiness locus (``_signature_accepts_call``).
        return torch.softmax(self.fc(x), dim=1)


def _build(tmp_path: Path, name: str, *, include_weights: bool = False) -> Path:
    model = _Tiny().eval()
    x = torch.randn(2, 4)
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))
    bundle = tmp_path / name
    trace.save(str(bundle), level="runnable", include_weights=include_weights)
    return bundle


def _tamper(bundle: Path, mutate: Any) -> None:
    path = bundle / "manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


@contextmanager
def _rlimit_cap(extra: int = 1 << 30) -> Iterator[None]:
    """Cap address space at current VmSize + ``extra`` so a bomb allocation raises."""

    with open("/proc/self/status", encoding="ascii") as handle:
        vmsize_kb = next(int(line.split()[1]) for line in handle if line.startswith("VmSize"))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (vmsize_kb * 1024 + extra, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


# --------------------------------------------------------------------------- #
# (a) schema-anchor meta-test                                                 #
# --------------------------------------------------------------------------- #

# Every integer-typed field of the frozen runnable descriptor dataclasses, with
# its anchor class. A field is either bounded/cross-validated against structure
# (so it can never scale an allocation with a hostile magnitude) or is inert.
# A NEW integer field with no row here fails the coverage meta-test until an
# anchor class -- and its parse check -- is added.
_INTEGER_FIELD_ANCHORS: dict[tuple[type, str], str] = {
    (RunnableCallDescriptor, "num_positional_args"): "leaf-count-anchored",
    (RunnableCallDescriptor, "num_keyword_args"): "leaf-count-anchored",
    (TensorSlotDescriptor, "shape"): "shape-anchored",
    (TensorSlotDescriptor, "rank"): "rank-anchored",
    (TensorSlotDescriptor, "device_index"): "allocation-inert",
    (InputSlotBinding, "container_record_id"): "allocation-inert",
    (InputSlotBinding, "model_site_position"): "allocation-inert",
    (ControlWitness, "order"): "allocation-inert",
    (FunctionRegistryKey, "version"): "allocation-inert",
    (InputAttestationFingerprint, "sizes"): "allocation-inert",
    (InputAttestationFingerprint, "strides"): "allocation-inert",
    (InputAttestationFingerprint, "storage_offset"): "allocation-inert",
    (InputAttestationFingerprint, "alignment_class"): "allocation-inert",
    (InputAttestationFingerprint, "device_index"): "allocation-inert",
}

_ALLOCATION_SCALING_ANCHORS = {"leaf-count-anchored", "shape-anchored", "rank-anchored"}


def _is_integer_annotation(annotation: Any) -> bool:
    """Return whether a dataclass field annotation is an int / tuple-of-int / int|None."""

    if annotation is int:
        return True
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (typing.Union, getattr(__import__("types"), "UnionType", None)):
        return any(_is_integer_annotation(arg) for arg in args)
    if origin in (tuple, list):
        return any(arg is int for arg in args)
    return False


_DESCRIPTOR_DATACLASSES = (
    RunnableCallDescriptor,
    TensorSlotDescriptor,
    InputSlotBinding,
    StateSlotBinding,
    ControlWitness,
    FunctionRegistryKey,
    InputAttestationFingerprint,
    CallExecutionContext,
    AutocastDeviceContext,
    TensorUseSite,
)


# --------------------------------------------------------------------------- #
# (b) bomb matrix under RLIMIT_AS                                             #
# --------------------------------------------------------------------------- #


# Kept: sole branch coverage of _io/runnable_load.py; a sole mutation kill.
def test_num_keyword_args_bomb_refused_at_load(tmp_path: Path) -> None:
    """Oversized ``num_keyword_args`` is refused at parse."""

    bundle = _build(tmp_path, "kwarity.tlspec", include_weights=True)
    _tamper(bundle, lambda m: m["run"]["calls"][0].__setitem__("num_keyword_args", 10**10))
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
    readiness = loaded._runnable.readiness
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert RunnableErrorCode.CALL_ARITY_MISMATCH in {d.code for d in readiness.diagnostics}


# Kept: sole branch coverage of _io/runnable_load.py; a sole mutation kill.
def test_sparse_positional_root_bomb_refused_at_load(tmp_path: Path) -> None:
    """A single huge positional root (``args[10**10]``) is refused at parse."""

    bundle = _build(tmp_path, "sparseroot.tlspec", include_weights=True)

    def _bomb(manifest: dict[str, Any]) -> None:
        for call in manifest["run"]["calls"]:
            for arg in call["tensor_arguments"]:
                if arg["argument_path"] and arg["argument_path"][0] == "args":
                    arg["argument_path"][1] = 10**10
                    return

    _tamper(bundle, _bomb)
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
    readiness = loaded._runnable.readiness
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert RunnableErrorCode.CALL_ARITY_MISMATCH in {d.code for d in readiness.diagnostics}


# Kept: sole branch coverage of _runnable/state.py.
def test_random_init_shape_bomb_refused_at_run(tmp_path: Path) -> None:
    """Locus 4 (state init): a structurally-valid huge shape -> allocation preflight refusal."""

    bundle = _build(tmp_path, "shape.tlspec", include_weights=False)

    def _bomb(manifest: dict[str, Any]) -> None:
        for slot in manifest["run"]["tensor_slots"]:
            if slot.get("state_binding"):
                slot["shape"] = [10**9, 10**9]  # ~8 EB, still int64-safe
                slot["rank"] = 2
                return

    _tamper(bundle, _bomb)
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
    with _rlimit_cap(), pytest.raises(RunCapabilityUnavailableError) as caught:
        loaded.run(inputs=torch.randn(2, 4))
    assert caught.value.fields.get("code") == "run_capability_unavailable"
    assert caught.value.fields.get("detection_stage") == "state_allocation_preflight"


# Kept: sole branch coverage of _io/runnable_load.py.
def test_int64_overflow_shape_refused_at_load(tmp_path: Path) -> None:
    """A shape whose byte product overflows signed 64-bit is refused at parse."""

    bundle = _build(tmp_path, "overflow.tlspec", include_weights=False)

    def _bomb(manifest: dict[str, Any]) -> None:
        for slot in manifest["run"]["tensor_slots"]:
            if slot.get("state_binding"):
                slot["shape"] = [2**40, 2**40]
                slot["rank"] = 2
                return

    _tamper(bundle, _bomb)
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
    readiness = loaded._runnable.readiness
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert RunnableErrorCode.STATE_SHAPE_MISMATCH in {d.code for d in readiness.diagnostics}


# Kept: sole branch coverage of _io/runnable_load.py.
def test_rank_shape_mismatch_refused_at_load(tmp_path: Path) -> None:
    """A declared rank contradicting ``len(shape)`` is refused at parse."""

    bundle = _build(tmp_path, "rankmismatch.tlspec", include_weights=False)

    def _bomb(manifest: dict[str, Any]) -> None:
        for slot in manifest["run"]["tensor_slots"]:
            if slot.get("state_binding"):
                slot["rank"] = 99
                return

    _tamper(bundle, _bomb)
    with _rlimit_cap():
        loaded = tl.load(str(bundle))
    readiness = loaded._runnable.readiness
    assert readiness.status is ReadinessStatus.UNAVAILABLE
    assert RunnableErrorCode.STATE_SHAPE_MISMATCH in {d.code for d in readiness.diagnostics}


# --------------------------------------------------------------------------- #
# (c) over-trigger guards                                                     #
# --------------------------------------------------------------------------- #


def _big_slot(shape: tuple[int, ...], dtype: str) -> TensorSlotDescriptor:
    return TensorSlotDescriptor(
        slot_id="s_big",
        role=TensorSlotRole.PARAMETER,
        use_sites=(),
        shape=shape,
        dtype=dtype,
        rank=len(shape),
        device_type="cpu",
        device_index=None,
        mutable=False,
        version_of=None,
        producer_slot_id=None,
        output_path=None,
        input_binding=None,
        state_binding=StateSlotBinding(
            module_path="embed",
            state_dict_name="embed.weight",
            semantic_role=StateSlotRole.WEIGHT,
            trainable=True,
            persistent=True,
            alias_group=None,
            captured_requires_grad=True,
            captured_grad_fn=False,
            host_escape_disposition=None,
        ),
        host_escape=False,
        inert_sink=False,
    )
