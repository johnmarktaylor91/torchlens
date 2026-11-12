"""TensorFlow compatibility probes for private eager-capture APIs."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Any

from ..._utils._torch_compat import TorchCapabilityWarning
from .._errors import BackendUnsupportedError

TFCapabilitySnapshot = dict[str, bool]
"""Stable mapping from TensorFlow capability flag name to availability."""

_warned_missing_capabilities: set[str] = set()


def _import_op_callbacks_module() -> Any | None:
    """Return TensorFlow's private op-callback module when available.

    Returns
    -------
    Any | None
        ``tensorflow.python.framework.op_callbacks`` or ``None``.
    """

    try:
        from tensorflow.python.framework import op_callbacks
    except ImportError:
        return None
    return op_callbacks


def _probe_op_callbacks() -> bool:
    """Return whether TensorFlow exposes eager op callbacks, without importing it.

    The module-level flag is read by ``tl.report.doctor()``, which must
    never import a preview framework. When TensorFlow is already
    loaded the real import is free and decides; otherwise the private module
    is located on disk through TensorFlow's package spec (``find_spec`` on the
    top-level name imports nothing). A located module that later fails to
    import is caught at use time: ``get_op_callbacks_module`` flips the flag
    through ``mark_tf_capability_missing``.

    Returns
    -------
    bool
        True when ``tensorflow.python.framework.op_callbacks`` imports (TF
        loaded) or its source ships in the installed package (TF not loaded).
    """

    if "tensorflow" in sys.modules:
        return _import_op_callbacks_module() is not None
    try:
        spec = importlib.util.find_spec("tensorflow")
    except (ImportError, ValueError):
        return False
    locations = getattr(spec, "submodule_search_locations", None) if spec is not None else None
    if not locations:
        return False
    suffixes = importlib.machinery.all_suffixes()
    return any(
        (Path(location) / "python" / "framework" / f"op_callbacks{suffix}").is_file()
        for location in locations
        for suffix in suffixes
    )


HAS_TF_OP_CALLBACKS: bool = _probe_op_callbacks()

_CAPABILITY_ATTRS: tuple[str, ...] = ("HAS_TF_OP_CALLBACKS",)


def mark_tf_capability_missing(capability_name: str, detail: str) -> None:
    """Mark a TensorFlow capability absent and emit at most one warning.

    Parameters
    ----------
    capability_name:
        Name of the module-level ``HAS_*`` flag to flip to ``False``.
    detail:
        Short user-facing detail describing the graceful degradation.

    Returns
    -------
    None
        The matching module-level flag is updated in place.
    """

    if capability_name not in _CAPABILITY_ATTRS:
        raise ValueError(f"unknown TensorFlow capability flag: {capability_name}")
    globals()[capability_name] = False
    if capability_name in _warned_missing_capabilities:
        return
    _warned_missing_capabilities.add(capability_name)
    warnings.warn(
        f"TorchLens TensorFlow capability {capability_name} is unavailable; {detail}",
        TorchCapabilityWarning,
        stacklevel=3,
    )


def get_tf_capability_snapshot() -> TFCapabilitySnapshot:
    """Return the current TensorFlow capability flags as a stable snapshot.

    Returns
    -------
    TFCapabilitySnapshot
        Mapping from ``HAS_*`` flag name to boolean availability. EMPTY when
        TensorFlow itself is not installed: this module imports
        cleanly without TF (the tf import is deferred inside
        ``_import_op_callbacks_module``), so torch-only installs used to merge
        a permanent ``HAS_TF_OP_CALLBACKS=False`` into every doctor/compat
        snapshot -- a false degradation alarm for an optional backend the user
        never installed.
    """

    try:
        tf_installed = importlib.util.find_spec("tensorflow") is not None
    except (ImportError, ValueError):
        tf_installed = False
    if not tf_installed:
        return {}
    return {name: bool(globals()[name]) for name in _CAPABILITY_ATTRS}


def get_op_callbacks_module() -> Any:
    """Return TensorFlow's private eager op-callback module.

    Returns
    -------
    Any
        TensorFlow op-callback module.

    Raises
    ------
    BackendUnsupportedError
        If the private op-callback module is unavailable. TensorFlow eager
        capture cannot record per-op events without it; graph-only fallback
        capture remains separate.
    """

    op_callbacks = _import_op_callbacks_module()
    if op_callbacks is None:
        mark_tf_capability_missing(
            "HAS_TF_OP_CALLBACKS",
            "TensorFlow eager op capture is unavailable",
        )
        raise BackendUnsupportedError(
            "TensorFlow eager capture requires tensorflow.python.framework.op_callbacks."
        )
    return op_callbacks


__all__ = [
    "HAS_TF_OP_CALLBACKS",
    "TFCapabilitySnapshot",
    "get_op_callbacks_module",
    "get_tf_capability_snapshot",
    "mark_tf_capability_missing",
]


#: TensorFlow's generated op-wrapper modules. Every Python-issued eager op runs
#: through one of these functions, so wrapping them presents nearly every op to
#: the intervention layer (op callbacks themselves are read-only).
_GEN_OP_MODULE_NAMES = (
    "tensorflow.python.ops.gen_math_ops",
    "tensorflow.python.ops.gen_nn_ops",
    "tensorflow.python.ops.gen_array_ops",
    "tensorflow.python.ops.gen_image_ops",
    "tensorflow.python.ops.gen_linalg_ops",
    "tensorflow.python.ops.gen_special_math_ops",
)

#: Internal TensorFlow op modules that re-export generated wrappers by name.
_OP_NAMESPACE_MODULE_NAMES = (
    "tensorflow.python.ops.math_ops",
    "tensorflow.python.ops.array_ops",
    "tensorflow.python.ops.nn_ops",
    "tensorflow.python.ops.nn_impl",
)


def _import_present(names: tuple[str, ...]) -> tuple[Any, ...]:
    """Import each named private TensorFlow module that exists (feature detection)."""

    modules = []
    for name in names:
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            continue
    return tuple(modules)


def get_tf_gen_op_modules() -> tuple[Any, ...]:
    """Return the generated op-wrapper modules this TensorFlow provides.

    Returns
    -------
    tuple[Any, ...]
        Imported ``gen_*_ops`` modules; empty when TensorFlow lays them out
        differently. Nothing degrades silently then: the curated public entry
        points still apply, and the fail-closed reachability audit refuses a
        site whose ops no wrapper presented.
    """

    return _import_present(_GEN_OP_MODULE_NAMES)


def get_tf_op_namespace_modules() -> tuple[Any, ...]:
    """Return internal op modules that hold direct references to generated wrappers.

    Returns
    -------
    tuple[Any, ...]
        Imported internal op modules that exist in this TensorFlow.
    """

    return _import_present(_OP_NAMESPACE_MODULE_NAMES)
