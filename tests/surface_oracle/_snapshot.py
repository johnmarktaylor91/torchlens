"""Canonical public-surface snapshot machinery for the surface oracle."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

import torch

import torchlens as tl
from torchlens import constants as tl_constants
from torchlens.data_classes.backward_pass import BackwardPass
from torchlens.data_classes.buffer import Buffer
from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.data_classes.grad_fn import GradFn
from torchlens.data_classes.grad_fn_call import GradFnCall
from torchlens.data_classes.layer import Layer
from torchlens.data_classes.module import Module, ModuleCall
from torchlens.data_classes.op import Op
from torchlens.data_classes.param import Param
from torchlens.data_classes.trace import Trace

SNAPSHOT_VERSION = 1

_REPO_ROOT = str(Path(__file__).resolve().parents[2])

#: Name tokens whose fields hold measured (machine/run-dependent) values.
#: These are normalized to a marker instead of frozen into the goldens.
_MEASURED_NAME_TOKENS = frozenset(
    {
        "time",
        "timestamp",
        "duration",
        "memory",
        "rng",
        "peak",
        "nonce",
        "uuid",
        "pid",
        "clock",
        "rss",
        "elapsed",
        "timing",
        "timings",
    }
)

#: Attributes whose values embed process-global counters (the per-process
#: trace counter and fork counter). Digit runs are normalized so the golden
#: does not depend on how many traces ran earlier in the process.
_COUNTER_NORMALIZED_ATTRIBUTES = frozenset({"trace_label", "state_history"})

#: Nested mapping keys treated as measured wherever they appear (runtime
#: object identities inside structured values such as ``state_history``).
_NESTED_MEASURED_KEYS = frozenset({"source_id"})

#: Attribute names skipped entirely, each with a documented reason. Skips are
#: presence-recorded (the name appears with a marker) so a skipped attribute
#: silently disappearing still fails the oracle.
SKIPPED_ATTRIBUTES: dict[str, str] = {
    "receptive_field": "lazy influence-geometry solver; view object, solve cost unbounded",
    "projective_field": "lazy influence-geometry solver; view object, solve cost unbounded",
    "_code_context_cache": "internal cache keyed by runtime code-object ids",
}

_FIELD_ORDER_BY_CLASS: dict[type, tuple[str, ...]] = {
    Trace: tuple(tl_constants.MODEL_LOG_FIELD_ORDER),
    Op: tuple(tl_constants.LAYER_PASS_LOG_FIELD_ORDER),
    Layer: tuple(tl_constants.LAYER_LOG_FIELD_ORDER),
    Module: tuple(tl_constants.MODULE_LOG_FIELD_ORDER),
    ModuleCall: tuple(tl_constants.MODULE_PASS_LOG_FIELD_ORDER),
    Param: tuple(tl_constants.PARAM_LOG_FIELD_ORDER),
    Buffer: tuple(tl_constants.BUFFER_LOG_FIELD_ORDER),
    GradFn: tuple(tl_constants.GRAD_FN_LOG_FIELD_ORDER),
    GradFnCall: tuple(tl_constants.GRAD_FN_PASS_LOG_FIELD_ORDER),
    BackwardPass: tuple(tl_constants.BACKWARD_PASS_FIELD_ORDER),
    FuncCallLocation: tuple(tl_constants.FUNC_CALL_LOCATION_FIELD_ORDER),
}

_RECORD_CLASSES = tuple(_FIELD_ORDER_BY_CLASS)

_MAX_DEPTH = 8


def public_surface_names(cls: type) -> tuple[str, ...]:
    """Return the snapshot attribute names for one record class.

    The set is the class's declared FIELD_ORDER (the portable schema,
    including private schema fields) followed by every additional public
    ``@property`` defined anywhere on the MRO, sorted for stability.

    Parameters
    ----------
    cls:
        Record class registered in the oracle.

    Returns
    -------
    tuple[str, ...]
        Ordered attribute names to snapshot.
    """

    field_order = _FIELD_ORDER_BY_CLASS[cls]
    seen = set(field_order)
    extra_properties = sorted(
        {
            name
            for mro_cls in cls.__mro__
            for name, value in vars(mro_cls).items()
            if isinstance(value, property) and not name.startswith("_") and name not in seen
        }
    )
    return field_order + tuple(extra_properties)


def _is_measured_name(name: str) -> bool:
    """Return whether an attribute name denotes a measured quantity."""

    lowered = name.lower()
    if "object_id" in lowered or "thread_id" in lowered:
        return True
    return any(token in _MEASURED_NAME_TOKENS for token in lowered.split("_"))


def _normalize_string(value: str) -> str:
    """Normalize machine-specific path prefixes inside string values."""

    if _REPO_ROOT in value:
        value = value.replace(_REPO_ROOT, "<repo>")
    if "/site-packages/" in value:
        _, _, tail = value.partition("/site-packages/")
        value = f"<site>/{tail}"
    if "/tmp/" in value or value.startswith("/tmp"):
        value = "<tmp-path>"
    return value


def _tensor_digest(value: torch.Tensor) -> dict[str, Any]:
    """Return a deterministic digest record for a tensor value."""

    detached = value.detach()
    meta = {
        "__tensor__": True,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "requires_grad": bool(value.requires_grad),
    }
    try:
        array_bytes = detached.cpu().contiguous().numpy().tobytes()
    except (TypeError, RuntimeError):
        array_bytes = repr(detached.cpu().flatten().tolist()).encode()
    meta["sha256"] = hashlib.sha256(array_bytes).hexdigest()
    return meta


def _record_ref(value: Any) -> dict[str, Any]:
    """Return a stable reference marker for a nested record object."""

    label = None
    for candidate in ("label", "layer_label", "address", "name"):
        try:
            candidate_value = getattr(value, candidate, None)
        except Exception:  # noqa: BLE001 - refs must never fail the snapshot
            candidate_value = None
        if isinstance(candidate_value, str):
            label = candidate_value
            break
    return {"__ref__": type(value).__name__, "label": label}


def canonicalize(value: Any, depth: int = 0) -> Any:
    """Convert a value into a deterministic JSON-serializable form.

    Parameters
    ----------
    value:
        Arbitrary attribute value from a record object.
    depth:
        Current recursion depth; values below ``_MAX_DEPTH`` recurse.

    Returns
    -------
    Any
        JSON-serializable canonical form.
    """

    if depth > _MAX_DEPTH:
        return "<max-depth>"
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return _normalize_string(value)
    if isinstance(value, bytes):
        return {"__bytes__": hashlib.sha256(value).hexdigest()}
    if isinstance(value, torch.Tensor):
        return _tensor_digest(value)
    if isinstance(value, (torch.dtype, torch.device, torch.Size)):
        return str(value)
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, _RECORD_CLASSES):
        return _record_ref(value)
    if isinstance(value, dict):
        items = [
            (json.dumps(canonicalize(key, depth + 1), sort_keys=True), key, entry)
            for key, entry in value.items()
        ]
        items.sort(key=lambda triple: triple[0])
        canonical_mapping: dict[str, Any] = {}
        for canon_key, raw_key, entry in items:
            key_name = raw_key if isinstance(raw_key, str) else canon_key
            if (
                key_name in _NESTED_MEASURED_KEYS
                or key_name.endswith("source_id")
                or _is_measured_name(key_name)
            ):
                canonical_mapping[canon_key] = "<measured>"
            else:
                canonical_mapping[canon_key] = canonicalize(entry, depth + 1)
        return canonical_mapping
    if isinstance(value, (set, frozenset)):
        members = [json.dumps(canonicalize(member, depth + 1), sort_keys=True) for member in value]
        return {"__set__": sorted(members)}
    if isinstance(value, (list, tuple)):
        return [canonicalize(member, depth + 1) for member in value]
    if isinstance(value, slice):
        return f"slice({value.start},{value.stop},{value.step})"
    if callable(value):
        qualname = getattr(value, "__qualname__", type(value).__qualname__)
        return f"<callable {qualname}>"
    return f"<{type(value).__module__}.{type(value).__qualname__}>"


def _normalize_counters_deep(canonical: Any) -> Any:
    """Replace digit runs in strings throughout a canonical subtree."""

    if isinstance(canonical, str):
        return re.sub(r"\d+", "N", canonical)
    if isinstance(canonical, list):
        return [_normalize_counters_deep(member) for member in canonical]
    if isinstance(canonical, dict):
        return {key: _normalize_counters_deep(entry) for key, entry in canonical.items()}
    return canonical


def snapshot_object(obj: Any) -> dict[str, Any]:
    """Snapshot the full public surface of one record object.

    Parameters
    ----------
    obj:
        Record instance whose class is registered in the oracle.

    Returns
    -------
    dict[str, Any]
        Attribute-name-to-canonical-value mapping.
    """

    surface: dict[str, Any] = {}
    for name in public_surface_names(type(obj)):
        if name in SKIPPED_ATTRIBUTES:
            surface[name] = f"<skipped: {SKIPPED_ATTRIBUTES[name]}>"
            continue
        if _is_measured_name(name):
            try:
                getattr(obj, name)
            except Exception as error:  # noqa: BLE001 - surface truth
                surface[name] = f"<measured raises {type(error).__name__}>"
            else:
                surface[name] = "<measured>"
            continue
        try:
            value = getattr(obj, name)
        except Exception as error:  # noqa: BLE001 - raising IS the contract
            surface[name] = f"<raises {type(error).__name__}>"
            continue
        canonical_value = canonicalize(value)
        if name in _COUNTER_NORMALIZED_ATTRIBUTES:
            canonical_value = _normalize_counters_deep(canonical_value)
        surface[name] = canonical_value
    return surface


def snapshot_trace_surface(trace: Trace) -> dict[str, Any]:
    """Snapshot the trace plus every reachable record family.

    Parameters
    ----------
    trace:
        Finished trace to snapshot.

    Returns
    -------
    dict[str, Any]
        Canonical snapshot with one section per record family.
    """

    snapshot: dict[str, Any] = {
        "__version__": SNAPSHOT_VERSION,
        "trace": snapshot_object(trace),
    }
    families = {
        # Every row, keyed by label as ``trace.ops.items()`` was before sumfam
        # F1: the counted ``trace.ops`` view leaves out input, output and
        # buffer rows, and the surface oracle snapshots every record.
        "ops": {op.label: op for op in trace.layer_list},
        "layers": trace.layers,
        "modules": trace.modules,
        "module_calls": trace.module_calls,
        "params": trace.params,
        "buffers": trace.buffers,
        "grad_fns": trace.autograd.grad_fns,
        "backward_passes": trace.autograd.backward_passes,
    }
    for family_name, accessor in families.items():
        if accessor is None:
            snapshot[family_name] = None
            continue
        snapshot[family_name] = {
            str(key): snapshot_object(record) for key, record in accessor.items()
        }
    return snapshot


def canonical_dump(snapshot: dict[str, Any]) -> str:
    """Serialize a snapshot to its canonical byte-comparable JSON string."""

    return json.dumps(snapshot, sort_keys=True, indent=1, ensure_ascii=True)


def load_public_trace(path: Path) -> Trace:
    """Load a saved trace bundle for the tlspec stage."""

    return tl.load(str(path))
