# ruff: noqa
"""Model and process state snapshots for differential state-leak probes."""

from __future__ import annotations

import hashlib
import random
import sys
import types
from typing import Any

import numpy as np
import torch
from torch import nn

_HOOK_DICTS = (
    "_forward_hooks",
    "_forward_pre_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_pre_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)

_GLOBAL_MODULES = (
    "torchlens._state",
    "torchlens.intervention.runtime",
    "torchlens.intervention.hooks",
    "torchlens.intervention.binding",
    "torchlens.intervention.site_keys",
    "torchlens.intervention.rerun",
    "torchlens.intervention.replay",
    "torchlens.intervention.helpers",
    "torchlens.user_funcs",
    "torchlens._capture_intervention",
    "torchlens._episode_spec",
    "torchlens._fast_run",
)


def _digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous()
    if data.dtype == torch.bfloat16:
        data = data.float()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()[:12]


def _summ(value: Any) -> Any:
    if isinstance(value, (bool, int, float, str)) or value is None:
        return repr(value)[:80]
    if isinstance(value, (list, tuple, set, frozenset, dict)):
        return f"{type(value).__name__}[{len(value)}]"
    if hasattr(value, "__len__") and type(value).__name__ in {
        "WeakSet",
        "WeakKeyDictionary",
        "WeakValueDictionary",
        "OrderedDict",
        "deque",
    }:
        try:
            return f"{type(value).__name__}[{len(value)}]"
        except Exception:  # noqa: BLE001
            return type(value).__name__
    if type(value).__name__ == "ContextVar":
        try:
            got = value.get()
        except LookupError:
            return "ContextVar<unset>"
        return f"ContextVar<{_summ(got)}>"
    return f"<{type(value).__name__}>"


def module_globals() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in _GLOBAL_MODULES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        for key, value in vars(mod).items():
            if key.startswith("__") or isinstance(
                value, (types.ModuleType, types.FunctionType, type)
            ):
                continue
            if (
                callable(value)
                and not hasattr(value, "__len__")
                and type(value).__name__ != "ContextVar"
            ):
                continue
            # the call-id counter advances on every capture by design
            if key in {"_func_call_id_iter", "_wrap_epoch"}:
                continue
            out[f"{name}.{key}"] = _summ(value)
    return out


def model_state(model: nn.Module) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for mname, mod in model.named_modules():
        tag = mname or "<root>"
        for hd in _HOOK_DICTS:
            d = getattr(mod, hd, None)
            if d:
                state[f"{tag}.{hd}"] = len(d)
        state[f"{tag}.training"] = mod.training
        keys = sorted(k for k in vars(mod) if k not in {"_parameters", "_buffers", "_modules"})
        state[f"{tag}.__dict__keys"] = ",".join(keys)
        if "forward" in vars(mod):
            state[f"{tag}.forward_in_instance_dict"] = True
        state[f"{tag}.class"] = type(mod).__qualname__
    for pname, p in model.named_parameters():
        state[f"param.{pname}"] = (_digest(p), p.requires_grad, p.grad is None, type(p).__name__)
    for bname, b in model.named_buffers():
        state[f"buffer.{bname}"] = (_digest(b), b.requires_grad)
    return state


def process_state() -> dict[str, Any]:
    from torch.nn.modules import module as mm
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    st: dict[str, Any] = {
        "global_forward_hooks": len(mm._global_forward_hooks),
        "global_forward_pre_hooks": len(mm._global_forward_pre_hooks),
        "global_backward_hooks": len(mm._global_backward_hooks),
        "global_module_registration_hooks": len(
            getattr(mm, "_global_module_registration_hooks", {})
        ),
        "global_parameter_registration_hooks": len(
            getattr(mm, "_global_parameter_registration_hooks", {})
        ),
        "global_buffer_registration_hooks": len(
            getattr(mm, "_global_buffer_registration_hooks", {})
        ),
        "torch_function_stack": torch._C._len_torch_function_stack(),
        "dispatch_mode_stack": len(_get_current_dispatch_mode_stack()),
        "grad_enabled": torch.is_grad_enabled(),
        "inference_mode": torch.is_inference_mode_enabled(),
        "default_dtype": str(torch.get_default_dtype()),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "autocast_cpu": torch.is_autocast_cpu_enabled(),
        "torch_rng": _digest(torch.get_rng_state()),
        "py_rng": hashlib.sha256(repr(random.getstate()).encode()).hexdigest()[:12],
        "np_rng": hashlib.sha256(repr(np.random.get_state()[1][:8]).encode()).hexdigest()[:12],
        "num_threads": torch.get_num_threads(),
        "anomaly": torch.is_anomaly_enabled(),
    }
    # identity of a few wrapped torch callables (wrappers persist after first capture by design)
    for name, fn in (
        ("torch.relu", torch.relu),
        ("F.linear", torch.nn.functional.linear),
        ("Tensor.__add__", torch.Tensor.__add__),
        ("torch.matmul", torch.matmul),
        ("F.softmax", torch.nn.functional.softmax),
    ):
        st[f"fn_id.{name}"] = id(fn)
    st.update(module_globals())
    return st


def full_state(model: nn.Module) -> dict[str, Any]:
    st = {f"model.{k}": v for k, v in model_state(model).items()}
    st.update({f"proc.{k}": v for k, v in process_state().items()})
    return st


def diff(
    before: dict[str, Any], after: dict[str, Any], *, ignore: tuple[str, ...] = ()
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        if any(part in key for part in ignore):
            continue
        a, b = before.get(key, "<absent>"), after.get(key, "<absent>")
        if a != b:
            out[key] = [a, b]
    return out
