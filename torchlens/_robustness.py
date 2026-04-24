"""Tensor-variant detection and pre-flight guards for ``log_forward_pass``.

TorchLens was designed around standard dense ``torch.Tensor`` /
``torch.nn.Parameter`` objects on real (CPU/CUDA/MPS) devices.  A number of
tensor variants break the logging pipeline in different ways:

================  =================================================  =================
Variant           Why TorchLens cannot handle it today                Detection outcome
================  =================================================  =================
Meta tensor       No storage, so activation saving returns garbage;  raise RuntimeError
                  ``.clone()`` yields another meta tensor, etc.
Sparse tensor     ``safe_copy``/print-override paths assume dense    raise RuntimeError
                  layouts; postprocess indexing uses ``.numel()``
                  which double-counts sparse entries.
Symbolic shape   Dimensions that are ``torch.SymInt`` /              raise RuntimeError
                  ``torch.SymFloat`` break shape-dependent metadata
                  (flops, tensor memory, counter alignment).
Quantized model   Partial support: logging works but FLOPs are        warn (keep going)
                  computed as zero/wrong for quantized ops.
================  =================================================  =================

This module centralises detection.  Callers (``log_forward_pass``,
``log_model_metadata``, ``validate_forward_pass``) invoke
:func:`check_model_and_input_variants` near entry, *before* decoration or
session setup, so failures happen up front with a clear error message
instead of partway through an 18-step pipeline.
"""

from __future__ import annotations

import warnings
from typing import Any, Iterable, Iterator, List, Tuple

import torch
from torch import nn


# ---------------------------------------------------------------------------
# Per-tensor detectors
# ---------------------------------------------------------------------------


def _is_meta_tensor(t: torch.Tensor) -> bool:
    """True if ``t`` lives on the meta device (no backing storage)."""
    try:
        return t.device.type == "meta"
    except Exception:
        return False


def _is_sparse_tensor(t: torch.Tensor) -> bool:
    """True for any sparse layout (COO, CSR, CSC, BSR, BSC)."""
    # ``layout`` exists on every torch.Tensor; sparse variants are not ``strided``.
    try:
        layout = t.layout
    except Exception:
        return False
    return layout is not torch.strided


def _has_symbolic_shape(t: torch.Tensor) -> bool:
    """True if any dimension is a ``torch.SymInt`` / ``torch.SymFloat``.

    Concrete ``int`` dims are safe.  Symbolic dims arise under
    ``torch._dynamo.mark_dynamic`` / ``torch.export`` traces and break
    metadata collection.
    """
    SymInt = getattr(torch, "SymInt", None)
    SymFloat = getattr(torch, "SymFloat", None)
    if SymInt is None and SymFloat is None:
        return False
    try:
        shape = t.shape
    except Exception:
        return False
    for dim in shape:
        if SymInt is not None and isinstance(dim, SymInt):
            return True
        if SymFloat is not None and isinstance(dim, SymFloat):
            return True
    return False


# ---------------------------------------------------------------------------
# Model-level detectors
# ---------------------------------------------------------------------------


# Quantized module class names — string-match to avoid importing
# ``torch.ao.quantization`` modules when the user doesn't have them compiled in.
_QUANTIZED_MODULE_NAME_PREFIXES: Tuple[str, ...] = (
    "torch.ao.nn.quantized",
    "torch.nn.quantized",
    "torch.ao.nn.intrinsic.quantized",
    "torch.ao.nn.qat",
    "torch.nn.qat",
)


def _is_quantized_module(module: nn.Module) -> bool:
    """True if ``module``'s class lives in a quantization namespace."""
    mod_name = type(module).__module__ or ""
    return any(mod_name.startswith(prefix) for prefix in _QUANTIZED_MODULE_NAME_PREFIXES)


def _model_has_quantized_modules(model: nn.Module) -> bool:
    """True if any submodule is a quantized ``nn`` module."""
    for sub in model.modules():
        if _is_quantized_module(sub):
            return True
    return False


# ---------------------------------------------------------------------------
# Input-tree walk
# ---------------------------------------------------------------------------


def _iter_tensors(obj: Any, _seen: set | None = None) -> Iterator[torch.Tensor]:
    """Yield every ``torch.Tensor`` reachable via list/tuple/dict traversal.

    Doesn't descend into ``nn.Module`` instances (those are handled by
    ``model.parameters()`` / ``model.buffers()``) and dedupes by ``id()`` so
    shared tensors are only visited once.
    """
    if _seen is None:
        _seen = set()
    if isinstance(obj, torch.Tensor):
        if id(obj) in _seen:
            return
        _seen.add(id(obj))
        yield obj
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            yield from _iter_tensors(item, _seen)
        return
    if isinstance(obj, dict):
        for item in obj.values():
            yield from _iter_tensors(item, _seen)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


class UnsupportedTensorVariantError(RuntimeError):
    """Raised when ``log_forward_pass`` is called on a model/input combination
    that TorchLens cannot reliably log (see module docstring for the matrix).
    """


def _docs_pointer() -> str:
    """Human-readable pointer to the limitations documentation."""
    return (
        "See the 'Tensor variants and unsupported contexts' section of "
        "README.md / docs/LIMITATIONS.md for supported alternatives."
    )


def check_model_and_input_variants(
    model: nn.Module,
    input_args: Any = None,
    input_kwargs: dict | None = None,
) -> None:
    """Pre-flight check for ``log_forward_pass``.

    Raises :class:`UnsupportedTensorVariantError` when a fundamentally
    incompatible tensor variant is detected on the model or its inputs.
    Emits :class:`UserWarning` for variants with partial / degraded support
    (quantization) so the user knows what to treat with skepticism in the log.

    Args:
        model: The ``nn.Module`` about to be logged.
        input_args: Positional arguments that will be passed to
            ``model.forward`` (may contain nested containers of tensors).
        input_kwargs: Keyword arguments to ``model.forward``.
    """
    if input_kwargs is None:
        input_kwargs = {}

    offenses: List[Tuple[str, str]] = []

    # Treat a bare tensor and a container of tensors identically — ``_iter_tensors``
    # yields tensors directly for a tensor, or recurses into list/tuple/dict.
    if input_args is None:
        args_payload: Any = []
    elif isinstance(input_args, torch.Tensor):
        args_payload = input_args
    else:
        args_payload = input_args

    # Input-side tensors.
    for t in _iter_tensors(args_payload):
        if _is_meta_tensor(t):
            offenses.append(
                (
                    "meta tensor in input",
                    "Meta tensors have no backing storage, so activation saving "
                    "cannot produce usable values.",
                )
            )
        if _is_sparse_tensor(t):
            offenses.append(
                (
                    f"sparse tensor ({t.layout}) in input",
                    "TorchLens' copy/print/FLOPs paths assume dense strided layouts.",
                )
            )
        if _has_symbolic_shape(t):
            offenses.append(
                (
                    "symbolic (SymInt/SymFloat) tensor shape in input",
                    "TorchLens requires concrete integer shapes for metadata and "
                    "counter alignment.",
                )
            )
    for t in _iter_tensors(dict(input_kwargs)):
        if _is_meta_tensor(t):
            offenses.append(("meta tensor in keyword input", ""))
        if _is_sparse_tensor(t):
            offenses.append((f"sparse tensor ({t.layout}) in keyword input", ""))
        if _has_symbolic_shape(t):
            offenses.append(("symbolic tensor shape in keyword input", ""))

    # Model params + buffers (dedupe across both generators).
    seen_ids: set[int] = set()
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen_ids:
            continue
        seen_ids.add(id(t))
        if _is_meta_tensor(t):
            offenses.append(
                (
                    "meta tensor among model parameters/buffers",
                    "Meta-init models (e.g. HuggingFace device_map='meta') must be "
                    "materialized on a real device before logging.",
                )
            )
            break  # one message is enough — don't list every param.

    if offenses:
        # Dedupe while preserving order of first appearance.
        seen: set = set()
        unique: List[Tuple[str, str]] = []
        for name, why in offenses:
            if name in seen:
                continue
            seen.add(name)
            unique.append((name, why))
        bullet_list = "\n".join(f"  - {name}" + (f": {why}" if why else "") for name, why in unique)
        raise UnsupportedTensorVariantError(
            "torchlens.log_forward_pass cannot run on this model/input "
            "combination. Detected unsupported tensor variant(s):\n"
            f"{bullet_list}\n"
            f"\n{_docs_pointer()}"
        )

    # Warnings (non-fatal).
    if _model_has_quantized_modules(model):
        warnings.warn(
            "TorchLens detected quantized submodules. Activation capture "
            "generally works, but FLOPs counts are not computed correctly for "
            "quantized ops and activation dtype handling is best-effort. "
            f"{_docs_pointer()}",
            UserWarning,
            stacklevel=3,
        )
