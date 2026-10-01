"""Accelerate offload/dispatch hook shims for eager capture (lane F37, R5).

Accelerate's ``AlignDevicesHook`` machinery runs INSIDE the module forwards
TorchLens logs: ``pre_forward`` moves args to the execution device and
materializes offloaded (meta) weights from the hook's ``weights_map``;
``post_forward`` returns weights to meta and moves outputs back. Left alone,
that infrastructure work is captured as model ops (spurious ``to_*`` nodes)
and the freshly materialized parameters carry no TorchLens session metadata,
so every weight consumption reads as an unattributed foreign tensor and the
escape detector ceilings the capture.

The shims installed here follow the capture-side hook doctrine (every
internal tensor read under ``pause_logging``):

1. ``pre_forward``/``post_forward`` execute under ``pause_logging`` --
   weight materialization and device alignment are infrastructure, never
   model computation, and are excluded from the op graph.
2. After ``pre_forward``, freshly materialized parameters are RE-STAMPED
   with the exact prep-time session metadata (barcode + address) recorded
   for their meta originals, so weight consumptions attribute normally and
   parameter identity (tied weights included) survives offload.

Shims are installed per capture session by ``_prepare_model_session`` and
removed by ``_cleanup_model_session`` (failure path included). Instance
attributes shadow the hook's class methods; uninstall deletes the shadow,
restoring the untouched original. Nothing here imports ``accelerate`` --
detection is structural, matching ``torchlens/_deploy_env.py``.

CPU-side scope note (C-DEPLOY): on a single execution device the hook's arg
moves are identity no-ops, so pausing them is exact. On genuinely
multi-device maps a cross-device move creates an unlogged tensor whose
provenance the existing disclosure machinery reports; the GPU cluster row
(C-DEPLOY) is the acceptance authority for those paths.
"""

from __future__ import annotations

import contextlib
import weakref
from typing import TYPE_CHECKING, Any

import torch.nn as nn

from ..._state import pause_logging
from ._tl import get_param_meta, set_param_meta

if TYPE_CHECKING:
    from ...data_classes.trace import Trace

__all__ = [
    "install_offload_hook_shims",
    "uninstall_offload_hook_shims",
]

_SHIM_MARKER = "_tl_offload_shim"


def _is_shimmable_hook(hook: Any) -> bool:
    """Structurally match an accelerate-style module hook."""

    return callable(getattr(hook, "pre_forward", None)) and callable(
        getattr(hook, "post_forward", None)
    )


def _restamp_materialized_params(trace: Trace, module: nn.Module, hook: Any) -> None:
    # Cognitive complexity ~17 accepted by design: one per-module scan whose
    # guard chain (meta skip, foreign-meta skip, unknown-address skip) IS the
    # never-crash-capture contract; each early continue is a distinct honest
    # outcome and belongs beside the others.
    """Re-stamp hook-materialized params with their prep-time session metadata.

    The prep pass stamped the META originals; ``pre_forward`` replaced them
    with fresh real-valued objects at the same addresses. Each fresh object
    inherits the barcode/address its address' ``Param`` record minted at
    prep, and joins the session inventory so cleanup clears the stamp.
    ``Param._param_ref`` deliberately stays on the prep-time object: pinning
    materialized weights would hold every offloaded shard in RAM at once,
    defeating the offload the user configured.
    """

    param_accessor = getattr(trace, "param_logs", None)
    records = getattr(param_accessor, "_dict", None)
    if not records:
        return  # predicate-mode capture: no session param logs to mirror.
    from .model_prep import _module_address

    recurse = bool(getattr(hook, "place_submodules", False))
    base = _module_address(module)
    inventory = getattr(trace, "_session_param_inventory", None)
    for rel_name, param in module.named_parameters(recurse=recurse):
        if param is None or param.device.type == "meta":
            continue
        try:
            if get_param_meta(param) is not None:
                continue  # already stamped (not a fresh materialization)
        except Exception:  # noqa: BLE001, S112 - foreign param metadata may fail arbitrarily; skipping is the disclosed never-crash-capture contract
            continue  # foreign metadata: never overwrite, never crash capture
        address = f"{base}.{rel_name}" if base else rel_name
        record = records.get(address)
        if record is None:
            continue  # unknown address: leave unattributed (disclosed path)
        set_param_meta(
            param,
            barcode=record.barcode,
            address=address,
            requires_grad_before=bool(record.is_trainable),
        )
        if inventory is not None:
            inventory.append(param)
        rebinds = getattr(trace, "_offload_param_rebinds", None)
        if rebinds is not None:
            # Weak-valued: attribution needs the object only while the module
            # call is live; a strong ref here would pin every offloaded shard
            # in RAM at once and defeat the offload the user configured.
            rebinds[address] = param


def install_offload_hook_shims(trace: Trace, model: nn.Module) -> None:
    """Shadow accelerate-style hooks' pre/post_forward for this session."""

    shims: list[tuple[Any, str]] = []
    for module in model.modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None or not _is_shimmable_hook(hook):
            continue
        if getattr(hook, _SHIM_MARKER, False):
            continue  # nested capture safety: never double-shim
        if getattr(trace, "_offload_param_rebinds", None) is None:
            # Session-scoped, created only when a hook exists; dropped at
            # uninstall so no live-trace attr survives the session.
            trace._offload_param_rebinds = weakref.WeakValueDictionary()
        orig_pre = hook.pre_forward
        orig_post = hook.post_forward

        def tl_pre_forward(
            mod: nn.Module,
            *args: Any,
            _orig_pre: Any = orig_pre,
            **kwargs: Any,
        ) -> Any:
            """Run the hook's real ``pre_forward`` unlogged, then re-stamp weights.

            Weight materialization and device alignment are infrastructure,
            never model computation, so the original hook runs under
            ``pause_logging``; the freshly materialized parameters then
            inherit their prep-time session metadata so weight consumptions
            attribute normally.
            """

            with pause_logging():
                out = _orig_pre(mod, *args, **kwargs)
            _restamp_materialized_params(trace, mod, getattr(mod, "_hf_hook", None))
            return out

        def tl_post_forward(
            mod: nn.Module,
            output: Any,
            _orig_post: Any = orig_post,
        ) -> Any:
            """Run the hook's real ``post_forward`` unlogged.

            Returning weights to meta and moving outputs back to their
            declared device are infrastructure moves; excluding them keeps
            spurious ``to_*`` nodes out of the op graph.
            """

            with pause_logging():
                return _orig_post(mod, output)

        hook.pre_forward = tl_pre_forward
        hook.post_forward = tl_post_forward
        setattr(hook, _SHIM_MARKER, True)
        shims.append((hook, "pre_forward"))
        shims.append((hook, "post_forward"))
    if shims:
        # Session-scoped bookkeeping only: set exactly when hooks exist and
        # DELETED at uninstall, so no live-trace attribute ever reaches the
        # save-time portability spec check.
        trace._offload_hook_shims = shims


def uninstall_offload_hook_shims(trace: Trace) -> None:
    """Delete the session's instance-attribute shadows (restores class methods)."""

    shims = getattr(trace, "_offload_hook_shims", None)
    if not shims:
        return
    seen: set[int] = set()
    for hook, attr_name in shims:
        try:
            if attr_name in hook.__dict__:
                delattr(hook, attr_name)
            if id(hook) not in seen and _SHIM_MARKER in hook.__dict__:
                delattr(hook, _SHIM_MARKER)
            seen.add(id(hook))
        except Exception:  # noqa: BLE001, S112 - foreign hook __dict__ access may fail arbitrarily; teardown must finish (same never-mask class as the ledgered teardown guards)
            continue  # never let hook cleanup break session teardown
    for attr in ("_offload_hook_shims", "_offload_param_rebinds"):
        with contextlib.suppress(AttributeError):
            delattr(trace, attr)
