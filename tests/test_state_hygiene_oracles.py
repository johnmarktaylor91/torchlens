"""Standing state-hygiene guard for every door that stores or applies an intervention.

Four oracles, applied per entry point, all computed OUTSIDE TorchLens (an
analytic forward or a plain ``register_forward_hook`` ground truth), so a rerun
that is merely self-consistent with the hook plan it ran cannot pass:

1. REPEAT EQUALS FRESH: the n-th call of a door gives what one fresh call gives,
   never an accumulated multiple.
2. DOORS AGREE: ``tl.trace``, ``tl.record``, ``bind``, ``fork().do`` and the
   legacy rerun apply one spec at the same sites, once.
3. LEFT AS FOUND: after the call the model carries no extra hooks, flags,
   grads or parameter changes, and the process-global capture slots are free.
4. SPEC UNCHANGED: engines never mutate the trace's staged spec, including on a
   failed or interrupted run.

Every test runs under the project's warnings-as-errors filters with no carve-out:
a rerun of a correct graph is silent, and a planted real divergence still warns
(or raises when strict). Model-state and deepcopy-aliasing cases live with the
model-hygiene guard. These are observable-result checks, never a validation
change.
"""

from __future__ import annotations

import hashlib
import warnings
from collections import Counter
from collections.abc import Callable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state
from torchlens.errors import RunCapabilityUnavailableError
from torchlens.intervention.errors import (
    ControlFlowDivergenceError,
    ControlFlowDivergenceWarning,
    EngineDispatchError,
    LiveModeLabelError,
    ReplayPreconditionError,
)

_HOOK_DICTS = (
    "_forward_hooks",
    "_forward_pre_hooks",
    "_backward_hooks",
    "_backward_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
)


class _Net(nn.Module):
    """fc1 -> relu -> fc2."""

    def __init__(self) -> None:
        """Build the two linears."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.fc2(torch.relu(self.fc1(x)))


class _MLP(nn.Module):
    """fc1 -> relu -> fc2 -> relu -> fc3."""

    def __init__(self) -> None:
        """Build the three linears."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))


class _BranchNet(nn.Module):
    """fc1 -> (relu if the input mean is positive, else sigmoid) -> fc2."""

    def __init__(self) -> None:
        """Build the two linears."""

        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Branch on the input, not on any edited activation."""

        hidden = self.fc1(x)
        hidden = torch.relu(hidden) if bool(x.mean() > 0) else torch.sigmoid(hidden)
        return self.fc2(hidden)


class _Tied(nn.Module):
    """Tied input/output weights, one module called twice."""

    def __init__(self) -> None:
        """Tie ``head`` to ``emb``."""

        super().__init__()
        self.emb = nn.Linear(2, 2, bias=False)
        self.head = nn.Linear(2, 2, bias=False)
        self.head.weight = self.emb.weight
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ``act`` twice between the tied linears."""

        return self.head(self.act(self.act(self.emb(x))))


def _digest(tensor: torch.Tensor) -> str:
    """Return a short content digest of one tensor."""

    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()[:16]


def _model_state(model: nn.Module) -> dict[str, Any]:
    """Return every user-visible piece of module state a call could leave behind."""

    state: dict[str, Any] = {}
    for name, module in model.named_modules():
        for hook_dict in _HOOK_DICTS:
            state[f"{name}.{hook_dict}"] = len(getattr(module, hook_dict, {}) or {})
        state[f"{name}.training"] = module.training
    for name, param in model.named_parameters():
        state[f"param.{name}"] = (_digest(param), param.requires_grad, param.grad is None)
    for name, buf in model.named_buffers():
        state[f"buffer.{name}"] = _digest(buf)
    return state


def _process_state() -> dict[str, Any]:
    """Return process-global state a finished call must leave as it found it."""

    from torch.nn.modules import module as module_mod
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    return {
        "global_forward_hooks": len(module_mod._global_forward_hooks),
        "global_forward_pre_hooks": len(module_mod._global_forward_pre_hooks),
        "torch_function_stack": torch._C._len_torch_function_stack(),
        "dispatch_mode_stack": len(_get_current_dispatch_mode_stack()),
        "grad_enabled": torch.is_grad_enabled(),
        "logging_enabled": _state._logging_enabled,
        "capture_slots_free": _capture_globals_released(),
    }


def _capture_globals_released() -> bool:
    """Return whether every process-global capture slot is released."""

    return (
        _state._active_trace is None
        and _state._active_hook_plan is None
        and _state._active_intervention_spec is None
        and _state._capture_reserved_by is None
    )


def _staged(trace: Any) -> tuple[Any, ...]:
    """Return the trace's staged spec content (fire records excluded)."""

    spec = trace._intervention_spec
    if spec is None:
        return ()
    frozen = spec.freeze()
    return (
        frozen.targets,
        frozen.helper,
        frozen.value,
        frozen.hook,
        frozen.target_value_specs,
        frozen.hook_specs,
        frozen.metadata,
    )


def _readout(trace: Any, module: str = "fc2") -> torch.Tensor:
    """Return one module's saved output (the ``_Net`` readout by default)."""

    return trace.find_sites(tl.module(module)).first().out


def _hooked(model: nn.Module, edits: dict[str, Callable[[Any], Any]], x: Any) -> torch.Tensor:
    """Plain-PyTorch ground truth: each edit as a forward hook on its module."""

    handles = [
        model.get_submodule(site).register_forward_hook(lambda _m, _a, out, edit=edit: edit(out))
        for site, edit in edits.items()
    ]
    try:
        with torch.no_grad():
            return model(x)
    finally:
        for handle in handles:
            handle.remove()


def _scale_fc1() -> Any:
    """Return the one-clause spec that doubles the fc1 output."""

    return tl.when(tl.module("fc1"), tl.scale(2.0))


_DIRECTION = torch.randn(16, generator=torch.Generator().manual_seed(7))


def _steer() -> Any:
    """Return a non-idempotent additive steer on a 16-wide site."""

    return tl.steer(_DIRECTION, magnitude=1.0, feature_axis=-1)


def _steer_edit(k: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return the ground-truth edit for ``k`` applications of ``_steer``."""

    return lambda out: out + k * _DIRECTION


@pytest.fixture
def net() -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """A seeded model, an input, and the analytic doubled-fc1 readout."""

    torch.manual_seed(0)
    model = _Net().eval()
    x = torch.randn(4, 8)
    with torch.no_grad():
        expected = model.fc2(2.0 * torch.relu(model.fc1(x)))
        plain = model(x)
    # The edit must move the readout, or exactness below proves nothing.
    assert not torch.allclose(expected, plain)
    return model, x, expected


@pytest.fixture
def mlp() -> Iterator[tuple[nn.Module, torch.Tensor, torch.Tensor]]:
    """A prepared three-layer model bracketed by a model and process state diff."""

    torch.manual_seed(0)
    model = _MLP().eval()
    x, x2 = torch.randn(3, 8), torch.randn(3, 8)
    tl.trace(model, x)  # first capture prepares the model (persistent by design)
    before_model, before_process = _model_state(model), _process_state()
    yield model, x, x2
    assert _model_state(model) == before_model, "a TorchLens call left state on the model"
    assert _process_state() == before_process, "a TorchLens call left process-global state"


# ---------------------------------------------------------------------------
# Repeat equals fresh.
# ---------------------------------------------------------------------------


def test_chunked_rerun_steers_every_row_once(net) -> None:
    """F1: a chunked rerun's rows all match one fresh steered forward.

    The first chunked rerun of a full-batch capture compares chunk 0 (batch 2)
    with the capture (batch 4); under the detector's contract a batch-size
    change is a shape divergence (``test_append_semantics`` pins the same for a
    full rerun at a new batch size), so that one call discloses it. A repeat
    reproduces the chunked graph and is silent.
    """

    model, x, expected = net
    trace = tl.trace(model, x, intervene=_scale_fc1())
    staged = _staged(trace)
    chunked = tl.options.ReplayOptions(chunk_size=2)
    with pytest.warns(ControlFlowDivergenceWarning, match="raw-event shape hash diverged"):
        trace.run(model, x, replay=chunked)
    torch.testing.assert_close(_readout(trace), expected)
    trace.run(model, x, replay=chunked)
    torch.testing.assert_close(_readout(trace), expected)
    assert _staged(trace) == staged, "a chunked rerun changed the staged spec"


@pytest.mark.parametrize(
    "door", [pytest.param("attach_hooks", marks=pytest.mark.smoke), "intervene"]
)
def test_legacy_rerun_repeats_exactly(mlp, door: str) -> None:
    """B0: every legacy rerun on a new input equals one hooked forward, silently."""

    model, x, x2 = mlp
    if door == "intervene":
        trace = tl.trace(model, x, intervene=tl.when(tl.module("fc2"), _steer()))
    else:
        trace = tl.trace(model, x)
        trace.attach_hooks(tl.module("fc2"), _steer(), confirm_mutation=True)
    staged = _staged(trace)
    want = _hooked(model, {"fc2": _steer_edit(1.0)}, x2)
    for _ in range(3):
        trace.run(model, x2)
        torch.testing.assert_close(trace.output_ops[0].out, want)
        assert _staged(trace) == staged


def test_bound_executor_repeats_exactly(mlp) -> None:
    """A bound executor gives the hooked answer on every call."""

    model, x, _ = mlp
    bound = tl.when(tl.module("fc2"), _steer()).bind(model)
    want = _hooked(model, {"fc2": _steer_edit(1.0)}, x)
    for _ in range(3):
        with torch.no_grad():
            torch.testing.assert_close(bound(x), want)


def test_capture_and_record_with_one_spec_repeat_exactly(mlp) -> None:
    """One spec object reused across captures and records applies once each time."""

    model, x, _ = mlp
    spec = tl.when(tl.module("fc2"), _steer())
    want = _hooked(model, {"fc2": _steer_edit(1.0)}, x)
    for _ in range(3):
        torch.testing.assert_close(tl.trace(model, x, intervene=spec).output_ops[0].out, want)
        recorded = tl.record(model, x, default_op=True, intervene=spec).to_trace()
        torch.testing.assert_close(recorded.output_ops[0].out, want)


def test_rerun_engine_do_composes_once_per_call(mlp) -> None:
    """Seat 2 F5: ``do(engine="rerun")`` adds one edit per call (1x, 2x, 3x), never 1x, 3x, 7x."""

    model, x, _ = mlp
    fork = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
    rerun = tl.options.InterventionOptions(engine="rerun")
    for k in (1.0, 2.0, 3.0):
        fork.do(tl.module("fc2"), _steer(), model=model, x=x, intervention=rerun)
        torch.testing.assert_close(
            fork.output_ops[0].out, _hooked(model, {"fc2": _steer_edit(k)}, x)
        )


def test_replay_engine_do_composes_once_per_call(mlp) -> None:
    """A replay ``do()`` composes once per call and leaves its parent untouched."""

    model, x, _ = mlp
    parent = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    parent_out = parent.output_ops[0].out.clone()
    fork = parent.fork()
    for k in (1.0, 2.0):
        fork.do(tl.module("fc2"), _steer())
        torch.testing.assert_close(
            fork.output_ops[0].out, _hooked(model, {"fc2": _steer_edit(k)}, x)
        )
    assert torch.equal(parent.output_ops[0].out, parent_out)


@pytest.mark.parametrize("exit_door", ["context", "remove"])
def test_attach_rerun_then_detach_restores_baseline(net, exit_door: str) -> None:
    """F2 / seat 3 F6: attach, rerun, remove, rerun equals a plain run."""

    model, x, expected = net
    trace = tl.trace(model, x)
    baseline = _readout(trace).clone()
    if exit_door == "context":
        with trace.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True):
            trace.run(model, x)
            trace.run(model, x)
            torch.testing.assert_close(_readout(trace), expected)
    else:
        handle = trace.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True)
        trace.run(model, x)
        trace.run(model, x)
        torch.testing.assert_close(_readout(trace), expected)
        handle.remove()
    assert len(trace._intervention_spec.hook_specs) == 0, "detach left hook copies behind"
    trace.run(model, x)
    torch.testing.assert_close(_readout(trace), baseline)


def test_callable_set_rerun_repeat_equals_fresh(net) -> None:
    """F5: a callable ``set`` replacement applies once per rerun, never compounds."""

    model, x, expected = net
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    # set() only stages the replacement; each run applies it.
    trace.set(tl.module("fc1"), lambda out: out * 2.0, confirm_mutation=True)
    staged = _staged(trace)
    for _ in range(3):
        trace.run(model, x)
        torch.testing.assert_close(_readout(trace), expected)
        assert _staged(trace) == staged


def test_capture_time_func_selector_is_rerunnable(net) -> None:
    """F7: a ``tl.when(tl.func(...))`` capture reruns with the same edit."""

    model, x, expected = net
    trace = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.scale(2.0)))
    torch.testing.assert_close(_readout(trace), expected)
    staged = _staged(trace)
    for _ in range(2):
        trace.run(model, x)
        torch.testing.assert_close(_readout(trace), expected)
        assert _staged(trace) == staged


def test_func_selector_append_rerun_applies_the_edit_once(net) -> None:
    """F7: append reruns of a ``tl.func`` capture re-arm the predicate exactly once."""

    model, x, expected = net
    trace = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.scale(2.0)))
    staged = _staged(trace)
    batch = x.shape[0]
    for _ in range(2):
        trace.run(model, x, replay=tl.options.ReplayOptions(append=True))
        torch.testing.assert_close(_readout(trace)[-batch:], expected)
        assert _staged(trace) == staged


def test_module_selector_rerun_on_tied_user_hooked_model_is_constant() -> None:
    """Seat 3 F7: tied weights, a module called twice and user hooks: reruns stay at one steer."""

    torch.manual_seed(0)
    model, x = _Tied(), torch.ones(1, 2)
    pre = model.act.register_forward_pre_hook(lambda _m, args: (args[0] * 3,))
    post = model.emb.register_forward_hook(lambda _m, _a, out: out + 1)
    try:
        spec = tl.when(tl.module("act"), tl.add(1))
        expected = _hooked(model, {"act": lambda out: out + 1}, x)
        before = _model_state(model)
        trace = tl.trace(model, x, intervene=spec)
        torch.testing.assert_close(trace.output_ops[0].out, expected)
        for _ in range(3):
            trace.run(model, x)
            torch.testing.assert_close(trace.output_ops[0].out, expected)
        assert _model_state(model) == before, "the user's own hooks or weights changed"
    finally:
        pre.remove()
        post.remove()


# ---------------------------------------------------------------------------
# Doors agree.
# ---------------------------------------------------------------------------


def test_middle_module_edit_agrees_on_every_door(mlp) -> None:
    """``tl.trace``, ``tl.record``, ``bind``, ``fork().do`` and a rerun give one answer."""

    model, x, _ = mlp
    want = _hooked(model, {"fc2": lambda out: out * 0.5}, x)
    with torch.no_grad():
        bound = tl.when(tl.module("fc2"), tl.scale(0.5)).bind(model)(x)
    fork = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
    fork.do(tl.module("fc2"), tl.scale(0.5))
    rerun = tl.trace(model, x)
    rerun.attach_hooks(tl.module("fc2"), tl.scale(0.5), confirm_mutation=True)
    rerun.run(model, x)
    outs = {
        "trace": tl.trace(model, x, intervene=tl.when(tl.module("fc2"), tl.scale(0.5))),
        "record": tl.record(
            model, x, default_op=True, intervene=tl.when(tl.module("fc2"), tl.scale(0.5))
        ).to_trace(),
        "fork_do": fork,
        "rerun": rerun,
    }
    agree = {door: torch.allclose(t.output_ops[0].out, want) for door, t in outs.items()}
    agree["bind"] = torch.allclose(bound, want)
    assert agree == dict.fromkeys(agree, True)


# ---------------------------------------------------------------------------
# Spec unchanged; failures stage nothing.
# ---------------------------------------------------------------------------


class _InjectedFailure(RuntimeError):
    """Raised by a patched submodule forward."""


def _failing_forward(failure: type[BaseException]) -> Callable[..., torch.Tensor]:
    """Return a forward that raises ``failure`` mid-forward."""

    def raising_forward(*_args: Any, **_kwargs: Any) -> torch.Tensor:
        raise failure("injected mid-forward")

    return raising_forward


@pytest.mark.parametrize(
    "failure", [_InjectedFailure, pytest.param(KeyboardInterrupt, marks=pytest.mark.smoke)]
)
def test_failed_rerun_leaves_staged_spec_unchanged(net, failure: type[BaseException]) -> None:
    """F3: a rerun that raises or is interrupted mid-forward stages nothing."""

    model, x, expected = net
    trace = tl.trace(model, x, save=tl.module("fc2"), intervene=_scale_fc1())
    staged = _staged(trace)
    before = _model_state(model)
    original = model.fc2.forward
    model.fc2.forward = _failing_forward(failure)
    try:
        with pytest.raises(failure):
            trace.run(model, x)
    finally:
        model.fc2.forward = original
    assert _staged(trace) == staged, "a failed rerun mutated the staged spec"
    assert _model_state(model) == before
    assert _capture_globals_released()
    trace.run(model, x)
    torch.testing.assert_close(_readout(trace), expected)
    assert _staged(trace) == staged


@pytest.mark.parametrize("failure", [_InjectedFailure, KeyboardInterrupt])
def test_failed_rerun_engine_do_then_retry_applies_once(net, failure: type[BaseException]) -> None:
    """Seat 3 F5: a failed ``fork().do(engine="rerun")`` leaves nothing; a retry steers once."""

    model, x, expected = net
    parent = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    baseline = _readout(parent).clone()
    rerun = tl.options.InterventionOptions(engine="rerun")
    original = model.fc2.forward
    forks = (parent.fork(), parent.fork())
    for fork in forks:
        model.fc2.forward = _failing_forward(failure)
        try:
            with pytest.raises(failure):
                fork.do(tl.module("fc1"), tl.scale(2.0), model=model, x=x, intervention=rerun)
        finally:
            model.fc2.forward = original
        assert len(fork._intervention_spec.hook_specs) == 0, "a failed do() left a hook staged"
    plain_fork, retry_fork = forks
    plain_fork.run(model, x)
    torch.testing.assert_close(_readout(plain_fork), baseline)
    retry_fork.do(tl.module("fc1"), tl.scale(2.0), model=model, x=x, intervention=rerun)
    torch.testing.assert_close(_readout(retry_fork), expected)
    torch.testing.assert_close(_readout(parent), baseline)


def test_portable_spec_saved_after_reruns_carries_no_duplicates(net, tmp_path) -> None:
    """F4: a spec saved after reruns holds exactly the staged entries."""

    from torchlens.io import load_intervention_spec

    model, x, _ = net
    trace = tl.trace(model, x, intervene=_scale_fc1())
    staged_hooks = len(trace._intervention_spec.hook_specs)
    trace.run(model, x)
    trace.run(model, x)
    path = tmp_path / "spec.tlspec"
    trace.save_intervention(path, level="portable")
    loaded = load_intervention_spec(path)
    assert len(loaded.hook_specs) == staged_hooks
    assert len(loaded.target_value_specs) == len(trace._intervention_spec.target_value_specs)


def test_fork_leaves_parent_spec_untouched(net) -> None:
    """F8: ``fork()`` never creates or changes the parent's staged spec."""

    model, x, expected = net
    parent = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
    parent_spec = parent._intervention_spec
    staged = _staged(parent)
    fork = parent.fork()
    fork.do(tl.module("fc1"), tl.scale(2.0))
    torch.testing.assert_close(_readout(fork), expected)
    assert parent._intervention_spec is parent_spec
    assert _staged(parent) == staged
    assert fork._intervention_spec is not parent_spec
    # A parent with no spec object at all stays without one.
    parent._intervention_spec = None
    parent.fork()
    assert parent._intervention_spec is None


def test_func_selector_rerun_refuses_a_partially_detached_recipe() -> None:
    """F7: a rerun that cannot reproduce the staged per-op entries refuses typed."""

    torch.manual_seed(0)
    model = _MLP().eval()
    x = torch.randn(4, 8)
    trace = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.scale(2.0)))
    door_labels = sorted(
        hook_spec.site_target.selector_value for hook_spec in trace._intervention_spec.hook_specs
    )
    assert len(door_labels) == 2
    trace.detach_hooks(tl.label(door_labels[0]), confirm_mutation=True)
    staged = _staged(trace)
    readout = _readout(trace, "fc3").clone()
    with pytest.raises(ControlFlowDivergenceError) as excinfo:
        trace.run(model, x)
    assert excinfo.value.fields["code"] == "rerun_predicate_restage_mismatch"
    assert "fired at" in str(excinfo.value)
    assert _staged(trace) == staged
    assert torch.equal(_readout(trace, "fc3"), readout)
    assert _capture_globals_released()


def test_predicate_rerun_names_a_changed_steer_as_the_cause() -> None:
    """Same ops, different rule identity: the refusal blames the changed steer, not the labels."""

    from torchlens.intervention._rerun_predicate import RerunSpecPlan, settle_predicate_rerun
    from torchlens.intervention.types import HookSpec, InterventionSpec, TargetSpec

    def door_hook(rule_id: str) -> HookSpec:
        return HookSpec(
            site_target=TargetSpec("label", "relu_1_2"),
            hook=_steer,
            metadata={
                "created_by": "intervene_predicate",
                "spec_rule_id": rule_id,
                "direction": None,
                "timing": "post",
            },
        )

    staged = InterventionSpec(hook_specs=[door_hook("r-captured")])
    plan = RerunSpecPlan(
        capture_spec=InterventionSpec(),
        intervene_predicate=object(),
        staged_keys=Counter(
            [("relu_1_2", "r-captured", None, "post", getattr(_steer, "__qualname__", None))]
        ),
    )
    rerun_log = SimpleNamespace(
        _intervention_spec=InterventionSpec(hook_specs=[door_hook("r-edited")])
    )
    with pytest.raises(ControlFlowDivergenceError) as excinfo:
        settle_predicate_rerun(plan, staged, rerun_log)
    assert excinfo.value.fields["code"] == "rerun_predicate_restage_mismatch"
    assert "changed since capture" in str(excinfo.value)
    assert len(staged.records) == 0


def test_set_label_rerun_refuses_with_a_set_remedy(mlp) -> None:
    """Seat 2 F7: ``set(label, value)`` + rerun refuses before the forward, naming ``set()``."""

    model, x, x2 = mlp
    trace = tl.trace(model, x)
    label = trace.find_sites(tl.module("fc2"))[0].layer_label
    trace.set(label, torch.full_like(trace[label].out, 0.3), confirm_mutation=True)
    staged = _staged(trace)
    readout = trace.output_ops[0].out.clone()
    with pytest.raises(LiveModeLabelError) as excinfo:
        trace.run(model, x2)
    assert excinfo.value.fields["code"] == "rerun_staged_label_unmatchable"
    assert "Trace.set()" in str(excinfo.value)
    assert "tl.module(" in str(excinfo.value)
    assert _staged(trace) == staged
    assert torch.equal(trace.output_ops[0].out, readout)


# ---------------------------------------------------------------------------
# Left as found.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "door",
    ["trace", "run", "chunked", "record", "bind", "fork_do", "attach_detach", "failing_capture"],
)
def test_entry_point_leaves_model_and_globals_pristine(net, door: str) -> None:
    """Every door leaves the user's model and the capture globals as it found them."""

    model, x, _ = net
    tl.trace(model, x)  # one-time lazy wrap and persistent model prep
    before, before_process = _model_state(model), _process_state()
    if door == "trace":
        tl.trace(model, x, intervene=_scale_fc1())
    elif door == "run":
        tl.trace(model, x, intervene=_scale_fc1()).run(model, x)
    elif door == "chunked":
        with pytest.warns(ControlFlowDivergenceWarning, match="raw-event shape hash diverged"):
            tl.trace(model, x, intervene=_scale_fc1()).run(
                model, x, replay=tl.options.ReplayOptions(chunk_size=2)
            )
    elif door == "record":
        tl.record(model, x, save=tl.func("relu"), intervene=_scale_fc1())
    elif door == "bind":
        with torch.no_grad():
            _scale_fc1().bind(model)(x)
    elif door == "fork_do":
        fork = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
        fork.do(tl.module("fc1"), tl.scale(2.0))
        fork.run(model, x)
    elif door == "attach_detach":
        trace = tl.trace(model, x)
        handle = trace.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True)
        trace.run(model, x)
        handle.remove()
    elif door == "failing_capture":

        def boom(_out: torch.Tensor, *, hook: Any) -> torch.Tensor:
            raise ValueError("boom")

        with pytest.raises(ValueError):
            tl.trace(model, x, intervene=tl.when(tl.module("fc1"), boom))
    assert _model_state(model) == before
    assert _process_state() == before_process


def test_interrupted_capture_record_and_bind_leave_no_runtime_state(mlp) -> None:
    """A KeyboardInterrupt inside any capture door leaves the next capture plain."""

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
        with torch.no_grad():
            assert tl.trace(model, x).output_ops[0].out.equal(model(x))


# ---------------------------------------------------------------------------
# Divergence disclosure precision.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("door", ["intervene", "attach_hooks", "set", "fork_do", "detach"])
def test_every_rerun_of_a_correct_graph_is_silent(net, door: str) -> None:
    """Seat 2 F7: a staged edit is a value change, never a control-flow divergence."""

    model, x, expected = net
    ready = tl.options.CaptureOptions(intervention_ready=True)
    handle = None
    if door == "intervene":
        trace = tl.trace(model, x, intervene=_scale_fc1())
    elif door == "fork_do":
        trace = tl.trace(model, x, capture=ready).fork()
        trace.do(tl.module("fc1"), tl.scale(2.0))
    elif door == "set":
        trace = tl.trace(model, x, capture=ready)
        trace.set(tl.module("fc1"), lambda out: out * 2.0, confirm_mutation=True)
    else:
        trace = tl.trace(model, x)
        handle = trace.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True)
    if door == "detach":
        assert handle is not None
        trace.run(model, x)
        handle.remove()
        with torch.no_grad():
            expected = model(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ControlFlowDivergenceWarning)
        for _ in range(3):
            trace.run(model, x)
            torch.testing.assert_close(_readout(trace), expected)
            assert trace.last_run["divergence_count"] == 0


@pytest.mark.parametrize("staged_edit", [False, True])
def test_planted_divergence_still_warns_and_raises_when_strict(staged_edit: bool) -> None:
    """A real control-flow change still discloses, with or without a staged edit."""

    torch.manual_seed(0)
    model = _BranchNet().eval()
    positive, negative = torch.ones(2, 8), -torch.ones(2, 8)
    trace = tl.trace(model, positive)
    if staged_edit:
        trace.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True)
    with pytest.warns(ControlFlowDivergenceWarning, match="raw-event shape hash diverged"):
        trace.run(model, negative)
    assert trace.last_run["divergence_count"] == 1
    strict = tl.trace(model, positive)
    if staged_edit:
        strict.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True)
    with pytest.raises(ControlFlowDivergenceError, match="raw-event shape hash diverged"):
        strict.run(model, negative, replay=tl.options.ReplayOptions(strict=True))


def test_loaded_trace_divergence_check_compares_like_with_like(tmp_path) -> None:
    """Seat 3 F3: a loaded trace has no raw-event hash, so its graph is compared graph to graph.

    Comparing the rerun's raw-event hash with the loaded graph hash never
    matched, so every rerun of a loaded trace blamed control flow. A loaded
    rerun of the same graph is now silent and a real branch change still warns.
    """

    torch.manual_seed(0)
    model = _BranchNet().eval()
    positive, negative = torch.ones(2, 8), -torch.ones(2, 8)
    path = tmp_path / "t.tlspec"
    tl.trace(model, positive).save(path)
    loaded = tl.load(path)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ControlFlowDivergenceWarning)
        loaded.run(model, positive)
    assert loaded.last_run["divergence_count"] == 0
    reloaded = tl.load(path)
    with pytest.warns(ControlFlowDivergenceWarning, match="raw-event shape hash diverged"):
        reloaded.run(model, negative)


def _raw_event(
    label: str,
    layer_type: str,
    parents: tuple[str, ...],
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
) -> SimpleNamespace:
    """Build the minimal raw op event the raw-event shape hash reads."""

    return SimpleNamespace(
        label_raw=label,
        kind="op",
        layer_type=layer_type,
        function=SimpleNamespace(func_name=layer_type, func_qualname=layer_type),
        output=SimpleNamespace(
            tensor=torch.empty(shape, dtype=dtype), container_path=None, container_spec=None
        ),
        parents=[SimpleNamespace(parent_label_raw=parent) for parent in parents],
        modules=(),
    )


def test_raw_event_hash_folds_only_value_preserving_replacements() -> None:
    """The fold is narrow: a same-shape replacement folds; a reshaping one or a source does not."""

    from torchlens.utils.hashing import compute_raw_event_shape_hash

    def digest(*middle: SimpleNamespace, relu_parent: str = "linear") -> str:
        events = [
            _raw_event("input", "input", (), (4, 8)),
            _raw_event("linear", "linear", ("input",), (4, 16)),
            *middle,
            _raw_event("relu", "relu", (relu_parent,), (4, 16)),
        ]
        return compute_raw_event_shape_hash(SimpleNamespace(op_events=events))

    plain = digest()
    edit = _raw_event("edit", "interventionreplacement", ("linear",), (4, 16))
    assert digest(edit, relu_parent="edit") == plain
    reshaped = _raw_event("edit", "interventionreplacement", ("linear",), (4, 8))
    assert digest(reshaped, relu_parent="edit") != plain
    recast = _raw_event("edit", "interventionreplacement", ("linear",), (4, 16), torch.float64)
    assert digest(recast, relu_parent="edit") != plain
    source = _raw_event("src", "internalsource", (), (4, 16))
    assert digest(source) != plain


# ---------------------------------------------------------------------------
# Loaded traces: every door that consumes the staged spec refuses or is correct.
# ---------------------------------------------------------------------------


def _intervened_trace(model: nn.Module, x: torch.Tensor, origin: str) -> Any:
    """Return a trace whose recorded fc2 values were halved through ``origin``."""

    if origin == "intervene":
        return tl.trace(model, x, intervene=tl.when(tl.module("fc2"), tl.scale(0.5)))
    if origin == "attach_hooks":
        trace = tl.trace(model, x)
        trace.attach_hooks(tl.module("fc2"), tl.scale(0.5), confirm_mutation=True)
        trace.run(model, x)
        return trace
    fork = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True)).fork()
    fork.do(tl.module("fc2"), tl.scale(0.5))
    return fork


def _reload(trace: Any, tmp_path: Any) -> Any:
    """Save ``trace`` with the default (analysis) save and load it back."""

    path = tmp_path / "t.tlspec"
    trace.save(path)
    return tl.load(path)


_NOT_PERSISTED = (EngineDispatchError, "run_intervention_spec_not_persisted")
_RERUN = tl.options.InterventionOptions(engine="rerun", confirm_mutation=True)
_REPLAY = tl.options.InterventionOptions(engine="replay", confirm_mutation=True)

_LOADED_DOORS: dict[str, tuple[Callable[..., Any], tuple[type[BaseException], str | None]]] = {
    "run": (lambda t, m, x: t.run(m, x), _NOT_PERSISTED),
    "run_append": (
        lambda t, m, x: t.run(m, x, replay=tl.options.ReplayOptions(append=True)),
        _NOT_PERSISTED,
    ),
    "run_chunked": (
        lambda t, m, x: t.run(m, x, replay=tl.options.ReplayOptions(chunk_size=2)),
        _NOT_PERSISTED,
    ),
    "fork_run": (lambda t, m, x: t.fork().run(m, x), _NOT_PERSISTED),
    "attach_then_run": (
        lambda t, m, x: (
            t.attach_hooks(tl.module("fc1"), tl.scale(2.0), confirm_mutation=True),
            t.run(m, x),
        ),
        _NOT_PERSISTED,
    ),
    "do_rerun": (
        lambda t, m, x: t.do(tl.module("fc1"), tl.scale(2.0), model=m, x=x, intervention=_RERUN),
        _NOT_PERSISTED,
    ),
    # The default save is analysis-only: no replay funcs and no run descriptor,
    # so the replay and unified doors cannot run at all on a loaded trace.
    "do_replay": (
        lambda t, m, x: t.do(tl.module("fc1"), tl.scale(2.0), intervention=_REPLAY),
        (ReplayPreconditionError, None),
    ),
    "run_inputs": (
        lambda t, m, x: t.run(inputs=x),
        (RunCapabilityUnavailableError, "run_capability_unavailable"),
    ),
}


@pytest.mark.parametrize(
    ("origin", "door"),
    [
        pytest.param("intervene", "run", marks=pytest.mark.smoke),
        ("attach_hooks", "run"),
        ("fork_do", "run"),
        *(("intervene", door) for door in _LOADED_DOORS if door != "run"),
    ],
)
def test_loaded_intervened_trace_door_refuses_typed(mlp, tmp_path, origin: str, door: str) -> None:
    """Seat 1 F6 / seat 2 F2 / seat 3 F3: no door reruns a loaded trace silently un-intervened."""

    model, x, x2 = mlp
    loaded = _reload(_intervened_trace(model, x, origin), tmp_path)
    readout = loaded.output_ops[0].out.clone()
    call, (error_type, code) = _LOADED_DOORS[door]
    with pytest.raises(error_type) as excinfo:
        call(loaded, model, x2)
    if code is not None:
        assert excinfo.value.fields["code"] == code
    assert torch.equal(loaded.output_ops[0].out, readout), "a refused door changed the trace"


def test_loaded_plain_trace_reruns_with_a_new_edit(mlp, tmp_path) -> None:
    """The refusal is narrow: a loaded UN-intervened trace still reruns a new edit, repeatedly."""

    model, x, x2 = mlp
    loaded = _reload(tl.trace(model, x), tmp_path)
    handle = loaded.attach_hooks(tl.module("fc2"), tl.scale(0.5), confirm_mutation=True)
    want = _hooked(model, {"fc2": lambda out: out * 0.5}, x2)
    for _ in range(2):
        loaded.run(model, x2)
        torch.testing.assert_close(loaded.output_ops[0].out, want)
    handle.remove()
    loaded.run(model, x2)
    with torch.no_grad():
        torch.testing.assert_close(loaded.output_ops[0].out, model(x2))
