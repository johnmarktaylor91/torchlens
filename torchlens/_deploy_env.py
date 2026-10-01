"""Deployment-envelope detection helpers (lane F37, R5).

The population TorchLens is aimed at loads large models through Hugging Face
Accelerate: ``device_map="auto"`` dispatch, CPU/disk offload, and 4/8-bit
quantization. Offloaded modules hold META parameters between forwards -- the
real values live in the hook's ``weights_map`` and are materialized onto the
execution device by ``AlignDevicesHook.pre_forward`` for exactly the duration
of the module call. A meta parameter is therefore NOT always an
unmaterialized model: when an offload hook provably backs it, the forward
pass sees real values and eager capture is sound.

This module centralizes the STRUCTURAL detection of that evidence. Nothing
here imports ``accelerate``; detection matches the hook contract shape
(attribute names ``_hf_hook`` / ``offload`` / ``weights_map`` /
``place_submodules`` / ``offload_buffers``) exactly the way
``torchlens/compat/_report.py`` detects the same hooks for its report rows.
Fail-closed doctrine: a meta tensor with NO offload hook in scope stays a
refusal at the capture entry gate (``torchlens._robustness``); only tensors
inside a hook's declared offload scope are admitted, and when the hook's key
inventory is readable the tensor's key must additionally be present in it.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch.nn as nn

__all__ = [
    "offload_backed_state_paths",
    "model_has_offload_hooks",
    "accelerate_execution_devices",
    "restore_state_dict_resilient",
]


def restore_state_dict_resilient(model: nn.Module, state_dict: Any) -> None:
    # Cognitive complexity ~22 accepted by design: the fallback is ONE
    # fail-closed evidence ladder (live slot -> geometry match -> sidecar
    # byte-proof -> re-raise) whose rungs only make sense read together;
    # splitting it would scatter the restore guarantee across helpers.
    """Restore a module's OWN cloned ``state_dict``, wrapped-param modules included.

    Validation clones ``model.state_dict()`` before capture and restores it
    afterwards. ``load_state_dict`` is the primary mechanism, but
    wrapped-parameter modules (bitsandbytes ``Linear8bitLt``/``Linear4bit``)
    refuse a round-trip of their OWN state dict on CPU (quantized-checkpoint
    guards, sidecar quantization keys). The fallback restores by exact-slot
    in-place copy and PROVES every slotless key is already at its captured
    value; anything unrestorable re-raises the original error -- the restore
    guarantee is never silently weakened.
    """

    import torch

    try:
        model.load_state_dict(state_dict)
        return
    except Exception as exc:  # noqa: BLE001 - wrapped-param modules raise arbitrary guard errors; the ladder below proves restorability or re-raises this exact error
        load_error = exc
    live: dict[str, Any] = dict(model.named_parameters())
    live.update(dict(model.named_buffers()))
    current = dict(model.state_dict())
    with torch.no_grad():
        for name, saved in state_dict.items():
            if not isinstance(saved, torch.Tensor):
                if name in current and current[name] == saved:
                    continue
                raise load_error  # non-tensor state drifted; cannot restore in place
            target = live.get(name)
            if target is not None:
                if (
                    target.shape == saved.shape
                    and target.dtype == saved.dtype
                    and target.device == saved.device
                ):
                    target.copy_(saved)
                    continue
                raise load_error  # geometry drifted; in-place restore unsound
            cur = current.get(name)
            if cur is None:
                raise load_error  # saved key has no live twin; cannot restore
            if cur.device.type == "meta" or saved.device.type == "meta":
                continue  # no value evidence exists or is needed on meta state
            if cur.shape == saved.shape and cur.dtype == saved.dtype and torch.equal(cur, saved):
                continue  # derived/sidecar key proven already at its captured value
            raise load_error  # sidecar quantization state drifted; unrestorable


def _iter_hooks(hook: Any) -> Iterator[Any]:
    """Yield leaf hooks; accelerate's ``SequentialHook`` holds a ``.hooks`` list."""

    child_hooks = getattr(hook, "hooks", None)
    if isinstance(child_hooks, (list, tuple)):
        for child in child_hooks:
            yield from _iter_hooks(child)
        return
    yield hook


def _offload_hooks(module: nn.Module) -> Iterator[Any]:
    """Yield the module's offload-declaring hooks (``offload=True``), if any."""

    hook = getattr(module, "_hf_hook", None)
    if hook is None:
        return
    for leaf in _iter_hooks(hook):
        if bool(getattr(leaf, "offload", False)) or bool(getattr(leaf, "offload_buffers", False)):
            yield leaf


def _weights_map_keys(hook: Any) -> frozenset[str] | None:
    """Best-effort relative key inventory of the hook's ``weights_map``.

    Returns ``None`` when the inventory is not readable (foreign mapping
    type); the caller then falls back to hook-scope evidence alone. Key
    iteration never loads tensor payloads (``__getitem__`` is what triggers
    the disk read, ``keys()`` is metadata-only on accelerate's
    ``PrefixedDataset`` / ``OffloadedWeightsLoader``).
    """

    weights_map = getattr(hook, "weights_map", None)
    if weights_map is None:
        return None
    try:
        keys = weights_map.keys()  # Mapping protocol; metadata-only.
        return frozenset(str(k) for k in keys)
    except Exception:  # noqa: BLE001 - foreign mapping types may fail arbitrarily; unreadable inventory falls back to hook-scope evidence (documented best-effort contract)
        return None


def offload_backed_state_paths(model: nn.Module) -> frozenset[str]:
    # Cognitive complexity ~21 accepted by design: one cohesive scan whose
    # nested conditions ARE the admission evidence rules (hook scope x
    # param/buffer declaration x key-inventory membership); extraction
    # would separate the rules from the fail-closed doctrine they encode.
    """Full state-dict-style names of offload-hook-backed params/buffers.

    A name is included exactly when (a) an offload-declaring accelerate-style
    hook (``offload=True`` for parameters, ``offload_buffers=True`` for
    buffers) sits on the owning module, (b) the tensor is inside that hook's
    declared scope (direct tensors, or the subtree under
    ``place_submodules=True``), and (c) when the hook's ``weights_map`` key
    inventory is readable, the tensor's hook-relative key is present in it.
    Everything else -- including a meta buffer under a hook that only
    declares parameter offload -- is excluded, so the capture entry gate
    keeps refusing genuinely valueless meta state.
    """

    backed: set[str] = set()
    for prefix, module in model.named_modules():
        for hook in _offload_hooks(module):
            recurse = bool(getattr(hook, "place_submodules", False))
            keys = _weights_map_keys(hook)
            offload_params = bool(getattr(hook, "offload", False))
            offload_buffers = bool(getattr(hook, "offload_buffers", False))
            scoped: list[tuple[str, Any]] = []
            if offload_params:
                scoped.extend(module.named_parameters(recurse=recurse))
            if offload_buffers:
                scoped.extend(module.named_buffers(recurse=recurse))
            for rel_name, _t in scoped:
                full_name = f"{prefix}.{rel_name}" if prefix else rel_name
                # accelerate's PrefixedDataset serves FULL state-dict names
                # from keys(); older/other mappings may hold hook-relative
                # names. Either spelling is key-inventory evidence.
                if keys is not None and full_name not in keys and rel_name not in keys:
                    continue
                backed.add(full_name)
    return frozenset(backed)


def model_has_offload_hooks(model: nn.Module) -> bool:
    """True when any module carries an offload-declaring accelerate-style hook."""

    return any(True for module in model.modules() for _ in _offload_hooks(module))


def accelerate_execution_devices(model: nn.Module) -> frozenset[str]:
    """Distinct execution-device strings declared by accelerate-style hooks."""

    devices: set[str] = set()
    for module in model.modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None:
            continue
        for leaf in _iter_hooks(hook):
            device = getattr(leaf, "execution_device", None)
            if device is not None:
                devices.add(str(device))
    return frozenset(devices)
