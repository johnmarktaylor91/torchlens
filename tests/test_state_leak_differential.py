"""Differential state-leak guard: every repeated or derived call equals a fresh oracle.

Each entry point that stores or applies an intervention is compared against two
oracles that share no TorchLens state with the call under test:

* the plain ``register_forward_hook`` ground truth for the same edit, and
* a fresh capture on a fresh model copy.

Every call is also wrapped in a model/process state diff (hook dicts, instance
``forward`` attributes, parameters, buffers, ``requires_grad``, train/eval,
global module hooks, torch function/dispatch mode stacks, grad mode and the
TorchLens active-capture globals). Repetition is the point: a rerun, a second
``do()`` or a second bound call must give exactly what the first call gave
(or the documented composition), never an accumulated multiple.

Known open bugs are ``xfail(strict=True)`` so the guard flips loudly when the
fix lands (run with ``--runxfail`` to see them fail).
"""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Callable, Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state

_HOOK_DICTS = (
    "_forward_hooks",
    "_forward_pre_hooks",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class _TwicePass(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)
        self.out = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.fc(torch.relu(self.fc(x)))))


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()[:16]


def model_state(model: nn.Module) -> dict[str, Any]:
    """Return every user-visible piece of module state TorchLens could leave behind."""

    state: dict[str, Any] = {}
    for name, module in model.named_modules():
        for hook_dict in _HOOK_DICTS:
            state[f"{name}.{hook_dict}"] = len(getattr(module, hook_dict, {}) or {})
        state[f"{name}.training"] = module.training
        state[f"{name}.instance_forward"] = "forward" in vars(module)
    for name, param in model.named_parameters():
        state[f"param.{name}"] = (_digest(param), param.requires_grad, param.grad is None)
    for name, buf in model.named_buffers():
        state[f"buffer.{name}"] = _digest(buf)
    return state


def process_state() -> dict[str, Any]:
    """Return process-global state a finished TorchLens call must leave as it found it."""

    from torch.nn.modules import module as module_mod
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    return {
        "global_forward_hooks": len(module_mod._global_forward_hooks),
        "global_forward_pre_hooks": len(module_mod._global_forward_pre_hooks),
        "torch_function_stack": torch._C._len_torch_function_stack(),
        "dispatch_mode_stack": len(_get_current_dispatch_mode_stack()),
        "grad_enabled": torch.is_grad_enabled(),
        "logging_enabled": _state._logging_enabled,
        "active_trace": _state._active_trace is None,
        "active_hook_plan": _state._active_hook_plan is None,
        "active_intervention_spec": _state._active_intervention_spec is None,
    }


@pytest.fixture
def mlp() -> Iterator[tuple[nn.Module, torch.Tensor, torch.Tensor]]:
    torch.manual_seed(0)
    model = _MLP().eval()
    x, x2 = torch.randn(3, 8), torch.randn(3, 8)
    tl.trace(model, x)  # first capture prepares the model (persistent by design)
    before_model, before_proc = model_state(model), process_state()
    yield model, x, x2
    assert model_state(model) == before_model, "a TorchLens call left state on the model"
    assert process_state() == before_proc, "a TorchLens call left process-global state behind"


def hooked(
    model: nn.Module, site: str, edit: Callable[[torch.Tensor], torch.Tensor], x: Any
) -> Any:
    """Plain-PyTorch ground truth: the same edit as a forward hook."""

    handle = model.get_submodule(site).register_forward_hook(lambda _m, _a, out: edit(out))
    try:
        with torch.no_grad():
            return model(x)
    finally:
        handle.remove()


def out_of(trace: Any) -> torch.Tensor:
    return trace.output_ops[0].out


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a, b, atol=1e-5, rtol=0.0)


_DIRECTION = torch.randn(16, generator=torch.Generator().manual_seed(7))


def _steer() -> Any:
    return tl.steer(_DIRECTION, magnitude=1.0, feature_axis=-1)


def _steer_edit(k: float) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda out: out + k * _DIRECTION


# ---------------------------------------------------------------------------
# Repetition oracles: the n-th call equals the first call.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.parametrize("n_calls", [3])
def test_bound_executor_repeats_exactly(mlp: Any, n_calls: int) -> None:
    model, x, _ = mlp
    bound = tl.when(tl.module("fc2"), _steer()).bind(model)
    want = hooked(model, "fc2", _steer_edit(1.0), x)
    for _ in range(n_calls):
        with torch.no_grad():
            assert _close(bound(x), want)


@pytest.mark.smoke
def test_capture_and_record_with_one_spec_repeat_exactly(mlp: Any) -> None:
    model, x, _ = mlp
    spec = tl.when(tl.module("fc2"), _steer())
    want = hooked(model, "fc2", _steer_edit(1.0), x)
    digest = spec.spec_digest
    for _ in range(3):
        assert _close(out_of(tl.trace(model, x, intervene=spec)), want)
        assert _close(out_of(tl.record(model, x, default_op=True, intervene=spec).to_trace()), want)
    assert spec.spec_digest == digest


@pytest.mark.xfail(
    strict=True, reason="B0: legacy rerun doubles staged hooks (wip/rerun-hook-doubling)"
)
@pytest.mark.parametrize("door", ["intervene", "attach_hooks"])
def test_legacy_rerun_repeats_exactly(mlp: Any, door: str) -> None:
    model, x, x2 = mlp
    if door == "intervene":
        trace = tl.trace(model, x, intervene=tl.when(tl.module("fc2"), _steer()))
    else:
        trace = tl.trace(model, x)
        trace.attach_hooks(tl.module("fc2"), _steer(), confirm_mutation=True)
    want = hooked(model, "fc2", _steer_edit(1.0), x2)
    for _ in range(3):
        trace.run(model, x2)
        assert _close(out_of(trace), want)


@pytest.mark.xfail(
    strict=True, reason="B0 family: do(engine='rerun') re-applies earlier edits (1x, 3x, 7x)"
)
def test_rerun_engine_do_composes_once_per_call(mlp: Any) -> None:
    model, x, _ = mlp
    fork = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
    rerun = tl.options.InterventionOptions(engine="rerun")
    for k in (1.0, 2.0, 3.0):
        fork.do(tl.module("fc2"), _steer(), model=model, x=x, intervention=rerun)
        assert _close(out_of(fork), hooked(model, "fc2", _steer_edit(k), x))


def test_replay_engine_do_composes_once_per_call(mlp: Any) -> None:
    model, x, x2 = mlp
    parent = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    parent_out = out_of(parent).clone()
    fork = parent.fork()
    for k in (1.0, 2.0):
        fork.do(tl.module("fc2"), _steer())
        assert _close(out_of(fork), hooked(model, "fc2", _steer_edit(k), x))
    assert torch.equal(out_of(parent), parent_out)


# ---------------------------------------------------------------------------
# Door agreement: every door applies one spec at the same sites, once.
# ---------------------------------------------------------------------------


def _all_doors(
    model: nn.Module, x: torch.Tensor, selector: Any, helper: Callable[[], Any]
) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        bound = tl.when(selector, helper()).bind(model)(x)
    fork = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
    fork.do(selector, helper())
    return {
        "trace": out_of(tl.trace(model, x, intervene=tl.when(selector, helper()))),
        "record": out_of(
            tl.record(model, x, default_op=True, intervene=tl.when(selector, helper())).to_trace()
        ),
        "bind": bound,
        "fork_do": out_of(fork),
    }


@pytest.mark.xfail(
    strict=True, reason="F5: post-hoc tl.module() also matches the synthetic output node"
)
def test_last_module_edit_applies_once_on_every_door(mlp: Any) -> None:
    model, x, _ = mlp
    want = hooked(model, "fc3", lambda out: out * 0.5, x)
    outs = _all_doors(model, x, tl.module("fc3"), lambda: tl.scale(0.5))
    assert {door: _close(out, want) for door, out in outs.items()} == dict.fromkeys(outs, True)


@pytest.mark.xfail(strict=True, reason="F4: live module matcher ignores the pass qualifier")
def test_pass_qualified_module_edit_agrees_on_every_door() -> None:
    torch.manual_seed(0)
    model = _TwicePass().eval()
    x = torch.randn(2, 8)
    calls = [0]

    def second_pass_only(_m: nn.Module, _a: Any, out: torch.Tensor) -> torch.Tensor:
        calls[0] += 1
        return out * 0.0 if calls[0] == 2 else out

    handle = model.fc.register_forward_hook(second_pass_only)
    with torch.no_grad():
        want = model(x)
    handle.remove()
    outs = _all_doors(model, x, tl.module("fc:2"), tl.zero_ablate)
    assert {door: _close(out, want) for door, out in outs.items()} == dict.fromkeys(outs, True)


# ---------------------------------------------------------------------------
# Derived objects: copies and round trips behave like the fresh oracle.
# ---------------------------------------------------------------------------


@pytest.mark.smoke
@pytest.mark.xfail(
    strict=True, reason="F1: prepared forward wrappers close over the original module"
)
def test_deepcopy_of_a_traced_model_runs_its_own_weights(mlp: Any) -> None:
    model, x, _ = mlp
    clone = copy.deepcopy(model)
    with torch.no_grad():
        for param in clone.parameters():
            param.mul_(2.0)
    fresh = _MLP().eval()
    fresh.load_state_dict(clone.state_dict())
    with torch.no_grad():
        assert torch.equal(clone(x), fresh(x))
    trace = tl.trace(clone, x)
    assert torch.equal(out_of(trace), fresh(x))


@pytest.mark.xfail(
    strict=True, reason="F2: trace save drops the staged spec; loaded rerun is silently clean"
)
@pytest.mark.parametrize("door", ["intervene", "attach_hooks", "fork_do"])
def test_saved_intervened_trace_reruns_like_the_live_one(
    mlp: Any, door: str, tmp_path: Any
) -> None:
    model, x, x2 = mlp
    if door == "intervene":
        trace = tl.trace(model, x, intervene=tl.when(tl.module("fc2"), tl.scale(0.5)))
    elif door == "attach_hooks":
        trace = tl.trace(model, x)
        trace.attach_hooks(tl.module("fc2"), tl.scale(0.5), confirm_mutation=True)
    else:
        trace = tl.trace(
            model, x, capture=tl.options.CaptureOptions(intervention_ready=True)
        ).fork()
        trace.do(tl.module("fc2"), tl.scale(0.5))
    path = os.path.join(tmp_path, "t.tlspec")
    trace.save(path)
    loaded = tl.load(path)
    # Either the loaded trace replays the edit, or it refuses typed. Silence is the bug.
    try:
        loaded.run(model, x2)
    except tl.errors.TorchLensError:
        return
    assert _close(out_of(loaded), hooked(model, "fc2", lambda out: out * 0.5, x2))


def test_failed_capture_leaves_no_runtime_state(mlp: Any) -> None:
    model, x, _ = mlp

    def interrupt(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
        raise KeyboardInterrupt

    for call in (
        lambda: tl.trace(model, x, intervene=tl.when(tl.module("fc2"), interrupt)),
        lambda: tl.when(tl.module("fc2"), interrupt).bind(model)(x),
        lambda: tl.record(
            model, x, default_op=True, intervene=tl.when(tl.module("fc2"), interrupt)
        ),
    ):
        with pytest.raises(KeyboardInterrupt):
            call()
        assert out_of(tl.trace(model, x)).equal(model(x).detach())
