"""Decision-matrix coverage for forward-pre-hook input provenance."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.modules.module as torch_module
from torch import nn

import torchlens as tl
from torchlens.backends.torch import prehook_provenance as provenance
from torchlens._io import BlobRef
from torchlens.fastlog import PredicateError
from torchlens.validation.core import validate_saved_outs


class _Unary(nn.Module):
    """One-operation module used by provenance tests."""

    def forward(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        """Scale the input after pre-hooks run."""

        return torch.relu(x * scale)


class _Nested(nn.Module):
    """Root with one named child module."""

    def __init__(self, child: nn.Module | None = None) -> None:
        """Initialize the child."""

        super().__init__()
        self.child = child if child is not None else _Unary()

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Delegate to the child."""

        return self.child(x, **kwargs)


def _call(trace: tl.Trace, label: str = "child:1") -> tl.ModuleCall:
    """Return one ModuleCall by label."""

    return trace.module_calls[label]


def _tensor(snapshot: tl.ModuleInputSnapshot, path: tuple[Any, ...] = ("args", 0)) -> torch.Tensor:
    """Return the retained tensor payload at ``path``."""

    observation = next(item for item in snapshot.tensor_observations if item.path == path)
    assert isinstance(observation.payload, torch.Tensor)
    return observation.payload


def _interposition_state(model: nn.Module) -> tuple[Any, ...]:
    """Capture exact user-visible state covered by the restoration contract."""

    module_states = []
    for module in model.modules():
        module_states.append(
            (
                tuple(module._forward_pre_hooks.items()),
                tuple(module._forward_pre_hooks_with_kwargs.items()),
                tuple(module._forward_hooks.items()),
                tuple(module._forward_hooks_with_kwargs.items()),
                tuple(module._forward_hooks_always_called.items()),
                "_call_impl" in module.__dict__,
                module.__dict__.get("_call_impl"),
            )
        )
    return (
        tuple(module_states),
        nn.Module.register_forward_pre_hook,
        torch_module.register_module_forward_pre_hook,
        "forward" in model.__dict__,
        model.__dict__.get("forward"),
    )


def test_in_place_mutation_truthful_snapshots_and_validation_replay() -> None:
    """Matrix 1: in-place mutation retains old/new bytes and replay authority stays Ops."""

    model = _Nested()

    def mutate(_module: nn.Module, args: tuple[Any, ...]) -> None:
        """Increment the first tensor in place."""

        args[0].add_(1)

    model.child.register_forward_pre_hook(mutate)
    x = torch.tensor([-2.0, 1.0])
    trace = tl.trace(model, x, save_arg_values=True, layers_to_save="all")
    call = _call(trace)

    assert torch.equal(call.inputs_before_pre_hooks.args[0], x)
    assert torch.equal(call.inputs_after_pre_hooks.args[0], x + 1)
    assert call.forward_pre_hook_effects[0].change_kinds == ("in_place_tensor",)
    assert call.forward_args is None

    ground_truth = model(x.clone()).detach()
    status = validate_saved_outs(trace, [ground_truth], validate_metadata=True)
    assert status.state == "passed"


def test_observational_hook_records_unchanged_without_dedicated_payload() -> None:
    """Matrix 2: a no-op hook is distinct from no hook and reuses producer payload."""

    model = _Nested()
    seen: list[torch.Size] = []

    def observe(_module: nn.Module, args: tuple[Any, ...]) -> None:
        """Read shape without changing inputs."""

        seen.append(args[0].shape)

    model.child.register_forward_pre_hook(observe)
    trace = tl.trace(model, torch.ones(2))
    call = _call(trace)

    assert seen == [torch.Size([2])]
    assert call.forward_pre_hook_effects[0].changed is False
    assert call.inputs_before_pre_hooks.tensor_observations[0].payload_origin == (
        "immutable_producer_snapshot"
    )
    assert (
        _tensor(call.inputs_before_pre_hooks).data_ptr()
        == _tensor(call.inputs_after_pre_hooks).data_ptr()
    )


def test_non_kwargs_replacement_is_tuple_normalized_and_attributed() -> None:
    """Matrix 3: a single tensor return becomes one positional argument."""

    model = _Nested()

    def replace(_module: nn.Module, args: tuple[Any, ...]) -> torch.Tensor:
        """Return a replacement tensor rather than a tuple."""

        return args[0] * 3

    model.child.register_forward_pre_hook(replace)
    trace = tl.trace(model, torch.tensor([2.0]))
    call = _call(trace)

    assert isinstance(call.inputs_after_pre_hooks.args, tuple)
    assert len(call.inputs_after_pre_hooks.args) == 1
    assert torch.equal(call.inputs_after_pre_hooks.args[0], torch.tensor([6.0]))
    assert call.forward_pre_hook_effects[0].change_kinds == ("identity",)
    assert call.inputs_after_pre_hooks.tensor_observations[0].source_op is not None


def test_in_place_mutation_and_replacement_record_both_effects() -> None:
    """A pre-hook that mutates then replaces records both change kinds."""

    model = _Nested()

    def mutate_and_replace(_module: nn.Module, args: tuple[Any, ...]) -> tuple[torch.Tensor, ...]:
        """Mutate the original tensor before returning a fresh replacement."""

        tensor = args[0]
        assert isinstance(tensor, torch.Tensor)
        tensor.add_(2)
        return (tensor.clone() * 3,)

    model.child.register_forward_pre_hook(mutate_and_replace)
    trace = tl.trace(model, torch.tensor([1.0]))
    effect = _call(trace).forward_pre_hook_effects[0]

    assert effect.change_kinds == ("identity", "in_place_tensor")


def test_kwargs_transition_paths_and_invalid_return_are_canonical() -> None:
    """Matrix 4: kwargs replacement is typed and invalid returns remain PyTorch errors."""

    model = _Nested()

    def replace_kwargs(
        _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Mutate positional input and replace one keyword."""

        args[0].add_(2)
        return args, {**kwargs, "scale": 4.0}

    model.child.register_forward_pre_hook(replace_kwargs, with_kwargs=True)
    trace = tl.trace(model, torch.tensor([1.0]), input_kwargs={"scale": 2.0})
    call = _call(trace)
    effect = call.forward_pre_hook_effects[0]

    assert torch.equal(call.inputs_after_pre_hooks.args[0], torch.tensor([3.0]))
    assert call.inputs_after_pre_hooks.kwargs == {"scale": 4.0}
    assert ("kwargs", ("key", "scale")) in effect.changed_paths

    bad_model = _Nested()

    def invalid(_module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any]) -> str:
        """Return an invalid kwargs-hook value."""

        return "bad"

    bad_model.child.register_forward_pre_hook(invalid, with_kwargs=True)
    expected = "forward pre-hook must return None"
    with pytest.raises(RuntimeError, match=expected):
        bad_model(torch.ones(1))
    with pytest.raises(RuntimeError, match=expected):
        tl.trace(bad_model, torch.ones(1))


def test_global_prepend_append_order_and_per_hook_attribution() -> None:
    """Matrix 5: global and local hooks share exact PyTorch execution order."""

    model = _Nested()
    order: list[str] = []

    def hook(name: str, delta: float) -> Callable[..., Any]:
        """Build one named replacement hook."""

        def apply(_module: nn.Module, args: tuple[Any, ...]) -> tuple[torch.Tensor]:
            """Append execution order and replace the tensor."""

            order.append(name)
            return (args[0] + delta,)

        apply.__name__ = name
        return apply

    global_handle = torch_module.register_module_forward_pre_hook(hook("G", 1))
    model.child.register_forward_pre_hook(hook("A", 3))
    model.child.register_forward_pre_hook(hook("B", 4))
    model.child.register_forward_pre_hook(hook("P", 2), prepend=True)
    try:
        trace = tl.trace(model, torch.zeros(1))
    finally:
        global_handle.remove()

    child_order = [name for name in order if name in {"G", "P", "A", "B"}][-4:]
    assert child_order == ["G", "P", "A", "B"]
    effects = _call(trace).forward_pre_hook_effects
    assert [effect.hook.name for effect in effects] == ["G", "P", "A", "B"]
    assert [effect.ordinal for effect in effects] == [1, 2, 3, 4]


def test_parent_adds_child_hook_before_first_child_call() -> None:
    """Matrix 6: post-prep public registration is wrapped before first execution."""

    class ParentRegisters(_Nested):
        """Register a child hook inside root forward."""

        def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            """Register then invoke the child."""

            self.child.register_forward_pre_hook(lambda _m, args: (args[0] + 5,))
            return self.child(x, **kwargs)

    model = ParentRegisters()
    trace = tl.trace(model, torch.ones(1))

    assert torch.equal(_call(trace).inputs_after_pre_hooks.args[0], torch.tensor([6.0]))
    installed = next(iter(model.child._forward_pre_hooks.values()))
    assert not hasattr(installed, "_torchlens_registration_id_cell")


def test_hook_added_by_hook_uses_pytorch_snapshot_iteration() -> None:
    """Matrix 7: a hook added during iteration starts on the next invocation."""

    class Twice(nn.Module):
        """Call one shared child twice."""

        def __init__(self) -> None:
            """Initialize the shared child."""

            super().__init__()
            self.child = _Unary()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Invoke the child twice."""

            return self.child(x) + self.child(x)

    model = Twice()
    added = False

    def first(module: nn.Module, _args: tuple[Any, ...]) -> None:
        """Register a second hook once."""

        nonlocal added
        if not added:
            module.register_forward_pre_hook(lambda _m, args: (args[0] + 1,))
            added = True

    model.child.register_forward_pre_hook(first)
    trace = tl.trace(model, torch.ones(1))

    assert len(_call(trace, "child:1").forward_pre_hook_effects) == 1
    assert len(_call(trace, "child:2").forward_pre_hook_effects) == 2


def test_removal_and_addition_survive_three_way_cleanup() -> None:
    """Matrix 8: real handle removal and user additions survive exact rollback."""

    model = _Nested()
    original_handle: Any = None
    new_hook: Callable[..., Any] | None = None

    def original(_module: nn.Module, _args: tuple[Any, ...]) -> None:
        """Remove self and install a replacement hook."""

        nonlocal new_hook
        original_handle.remove()

        def replacement(_replacement_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
            """Replace the removed hook through public registration."""

            return (args[0] + 1,)

        new_hook = replacement
        model.child.register_forward_pre_hook(new_hook)

    original_handle = model.child.register_forward_pre_hook(original)
    tl.trace(model, torch.ones(1))

    assert original_handle.id not in model.child._forward_pre_hooks
    assert new_hook is not None
    assert new_hook in model.child._forward_pre_hooks.values()


def test_hook_exception_records_partial_and_restores_every_interposition() -> None:
    """Matrix 9: mutation-then-raise propagates and leaves the model reusable."""

    model = _Nested()

    def explode(_module: nn.Module, args: tuple[Any, ...]) -> None:
        """Mutate then raise the sentinel exception."""

        args[0].add_(1)
        raise LookupError("hook exploded")

    handle = model.child.register_forward_pre_hook(explode)
    before = _interposition_state(model)
    with pytest.raises(LookupError, match="hook exploded") as exc_info:
        tl.trace(model, torch.ones(1))
    assert hasattr(exc_info.value, "partial_log")
    assert _interposition_state(model) == before

    handle.remove()
    assert torch.equal(model(torch.ones(1)), torch.ones(1))


def test_spectral_norm_module_state_hook_is_not_input_change() -> None:
    """Matrix 10: weight-mutating spectral norm is observational for inputs."""

    linear = torch.nn.utils.spectral_norm(nn.Linear(2, 2))
    model = nn.Sequential(linear)
    trace = tl.trace(model, torch.ones(1, 2))
    effect = _call(trace, "0:1").forward_pre_hook_effects[0]

    assert effect.changed is False
    assert effect.change_kinds == ()


def test_hookless_trace_skips_registration_refresh_hot_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hookless module calls do not scan registries or all prepared modules."""

    refresh_calls = 0
    original = provenance.refresh_registration_bypasses

    def count_refreshes(trace: tl.Trace, module: nn.Module | None = None) -> None:
        """Count registration refreshes while preserving their behavior."""

        nonlocal refresh_calls
        refresh_calls += 1
        original(trace, module)

    monkeypatch.setattr(provenance, "refresh_registration_bypasses", count_refreshes)
    tl.trace(nn.Sequential(*[nn.ReLU() for _ in range(20)]), torch.ones(1))

    assert refresh_calls == 0


def test_global_hook_refreshes_observers_once_per_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stable global hook does not trigger an all-module refresh per call."""

    refresh_calls = 0
    original = provenance._refresh_observers

    def count_refreshes(ledger: provenance.PreHookProvenanceLedger) -> None:
        """Count full observer refreshes while preserving their behavior."""

        nonlocal refresh_calls
        refresh_calls += 1
        original(ledger)

    monkeypatch.setattr(provenance, "_refresh_observers", count_refreshes)
    handle = torch_module.register_module_forward_pre_hook(lambda _module, _args: None)
    try:
        tl.trace(nn.Sequential(*[nn.ReLU() for _ in range(20)]), torch.ones(1))
    finally:
        handle.remove()

    assert refresh_calls == 1


def test_root_pre_hook_populates_self_call() -> None:
    """Matrix 11: root provenance binds to ``self:1``."""

    model = _Unary()
    model.register_forward_pre_hook(lambda _m, args: (args[0] + 2,))
    trace = tl.trace(model, torch.ones(1))

    root = _call(trace, "self:1")
    assert torch.equal(root.inputs_before_pre_hooks.args[0], torch.ones(1))
    assert torch.equal(root.inputs_after_pre_hooks.args[0], torch.tensor([3.0]))


def test_shared_recurrent_calls_have_independent_records() -> None:
    """Matrix 12: repeated shared-module calls bind independent call indices."""

    class Shared(nn.Module):
        """Call one shared module with distinct inputs."""

        def __init__(self) -> None:
            """Initialize the shared child."""

            super().__init__()
            self.child = _Unary()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Invoke the child twice."""

            return self.child(x) + self.child(x + 10)

    model = Shared()
    model.child.register_forward_pre_hook(lambda _m, args: (args[0] + 1,))
    trace = tl.trace(model, torch.zeros(1))

    assert torch.equal(_call(trace, "child:1").inputs_before_pre_hooks.args[0], torch.zeros(1))
    assert torch.equal(
        _call(trace, "child:2").inputs_before_pre_hooks.args[0], torch.tensor([10.0])
    )


def test_reentrant_same_module_uses_lifo_tokens() -> None:
    """Matrix 13: recursive invocation binds inner and outer tokens independently."""

    model = _Nested()
    in_hook = False

    def recurse(module: nn.Module, args: tuple[Any, ...]) -> None:
        """Invoke the same module once from its pre-hook."""

        nonlocal in_hook
        if in_hook:
            return
        in_hook = True
        try:
            module(args[0] + 5)
        finally:
            in_hook = False

    model.child.register_forward_pre_hook(recurse)
    trace = tl.trace(model, torch.ones(1))

    assert len(trace.module_calls) == 3
    assert len(_call(trace, "child:1").forward_pre_hook_effects) == 1
    assert len(_call(trace, "child:2").forward_pre_hook_effects) == 1


def test_fast_second_pass_preserves_exhaustive_provenance_once() -> None:
    """Matrix 14: selective two-pass capture does not overwrite provenance."""

    model = _Nested()
    count = 0

    def mutate(_module: nn.Module, args: tuple[Any, ...]) -> tuple[torch.Tensor]:
        """Count executions and replace input."""

        nonlocal count
        count += 1
        return (args[0] + 1,)

    model.child.register_forward_pre_hook(mutate)
    trace = tl.trace(model, torch.ones(1))
    trace.save_new_outs(model, torch.ones(1), layers_to_save="all")

    assert count == 2
    assert len(_call(trace).forward_pre_hook_effects) == 1


def test_record_to_trace_ram_and_disk_preserve_provenance(tmp_path: Path) -> None:
    """Matrix 15: predicate Recording materializes provenance in RAM and disk modes."""

    for streaming in (
        None,
        tl.StreamingOptions(bundle_path=tmp_path / "recording.tlfast", retain_in_memory=False),
    ):
        model = _Nested()
        model.child.register_forward_pre_hook(lambda _m, args: (args[0] + 1,))
        recording = tl.fastlog.record(model, torch.ones(1), default_op=True, streaming=streaming)
        trace = recording.to_trace()
        assert torch.equal(_call(trace).inputs_after_pre_hooks.args[0], torch.tensor([2.0]))


def test_version_edges_and_alias_deduplication() -> None:
    """Matrix 16: views, write-restore, same writes, and aliases are attributed."""

    class Pair(nn.Module):
        """Accept two aliased tensor paths."""

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            """Add both inputs."""

            return x + y

    class AliasRoot(nn.Module):
        """Pass one tensor to two child argument paths."""

        def __init__(self) -> None:
            """Initialize the pair child."""

            super().__init__()
            self.child = Pair()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Call the child with aliased arguments."""

            return self.child(x, x)

    model = AliasRoot()

    def mutate(_module: nn.Module, args: tuple[Any, ...]) -> None:
        """Perform same-value and write-restore mutations through a view."""

        original = args[0].clone()
        args[0].view(-1).copy_(args[0].view(-1))
        args[0].add_(1)
        args[0].copy_(original)

    model.child.register_forward_pre_hook(mutate)
    x = torch.ones(2)
    trace = tl.trace(model, x)
    call = _call(trace)
    observations = call.inputs_before_pre_hooks.tensor_observations

    assert call.forward_pre_hook_effects[0].changed is True
    assert call.forward_pre_hook_effects[0].change_kinds == ("in_place_tensor",)
    assert len({item.alias_group for item in observations}) == 1
    assert observations[0].payload.data_ptr() == observations[1].payload.data_ptr()


def test_unavailable_tensor_version_produces_tri_state_unknown() -> None:
    """Matrix 16 unavailable-version row: missing evidence never becomes false."""

    class VersionlessTensor(torch.Tensor):
        """Tensor subclass whose version counter is intentionally unreadable."""

        @property
        def _version(self) -> int:
            """Raise as inference/subclass tensors may do."""

            raise RuntimeError("version unavailable")

    tensor = torch.ones(1).as_subclass(VersionlessTensor)
    before = provenance._observe_state((tensor,), {})
    after = provenance._observe_state((tensor,), {})
    changed, kinds, paths, reasons = provenance._compare_observations(before, after)

    assert changed is None
    assert kinds == ()
    assert paths == ()
    assert reasons == ("tensor_version_unavailable",)


def test_wrappers_expose_original_and_unrelated_registration_is_unmodified() -> None:
    """Amendments 4-5: unrelated modules delegate and live wrappers expose originals."""

    model = _Nested()
    unrelated = _Unary()
    visibility: list[bool] = []
    unrelated_hook: Callable[..., Any] | None = None

    def local_hook(_module: nn.Module, _args: tuple[Any, ...]) -> None:
        """Inspect the live wrapper and register on an unrelated module."""

        nonlocal unrelated_hook
        current = next(iter(model.child._forward_pre_hooks.values()))
        visibility.append(getattr(current, "__wrapped__", None) is local_hook)

        def external(_external_module: nn.Module, _external_args: tuple[Any, ...]) -> None:
            """Remain an exact unwrapped callable outside the prepared set."""

        unrelated_hook = external
        unrelated.register_forward_pre_hook(external)

    model.child.register_forward_pre_hook(local_hook)
    tl.trace(model, torch.ones(1))

    assert visibility == [True]
    assert unrelated_hook is not None
    assert next(iter(unrelated._forward_pre_hooks.values())) is unrelated_hook


def test_pickle_tlspec_lazy_and_v5_compatibility(tmp_path: Path) -> None:
    """Matrix 17: pickle and eager/lazy tlspec preserve or explicitly defer payloads."""

    model = _Nested()
    model.child.register_forward_pre_hook(lambda _m, args: (args[0] + 1,))
    trace = tl.trace(model, torch.ones(1))
    pickled_call = pickle.loads(pickle.dumps(_call(trace)))
    assert torch.equal(pickled_call.inputs_before_pre_hooks.args[0], torch.ones(1))

    path = tmp_path / "prehook.tlspec"
    tl.save(trace, path)
    eager = tl.load(path, lazy=False)
    lazy = tl.load(path, lazy=True, materialize_nested=False)
    assert torch.equal(_call(eager).inputs_after_pre_hooks.args[0], torch.tensor([2.0]))
    assert isinstance(_call(lazy).inputs_before_pre_hooks.args[0], BlobRef)


def test_restoration_matrix_success_forward_error_halt_and_predicate_error() -> None:
    """Matrix 18: success, forward error, halt, and predicate failure restore state."""

    class MaybeRaise(_Nested):
        """Nested model with optional user-forward failure."""

        fail = False

        def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            """Optionally fail after child execution."""

            out = self.child(x, **kwargs)
            if self.fail:
                raise ArithmeticError("forward failed")
            return out

    model = MaybeRaise()
    model.child.register_forward_pre_hook(lambda _m, _args: None)
    before = _interposition_state(model)

    tl.trace(model, torch.ones(1))
    assert _interposition_state(model) == before

    model.fail = True
    with pytest.raises(ArithmeticError, match="forward failed"):
        tl.trace(model, torch.ones(1))
    assert _interposition_state(model) == before
    model.fail = False

    tl.trace(model, torch.ones(1), halt=lambda ctx: ctx.kind == "op")
    assert _interposition_state(model) == before

    with pytest.raises(PredicateError):
        tl.record(model, torch.ones(1), save=lambda _ctx: 1.5)
    assert _interposition_state(model) == before


def test_preparation_failure_rolls_back_partial_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matrix 18 prep-failure row: installation failure cannot strand wrappers."""

    model = _Nested()
    model.child.register_forward_pre_hook(lambda _m, _args: None)
    before = _interposition_state(model)
    original = provenance.install_prehook_provenance

    def fail_after_install(*args: Any, **kwargs: Any) -> Any:
        """Install successfully, then inject a preparation failure."""

        original(*args, **kwargs)
        raise RuntimeError("prep injection")

    monkeypatch.setattr(provenance, "install_prehook_provenance", fail_after_install)
    with pytest.raises(RuntimeError, match="prep injection"):
        tl.trace(model, torch.ones(1))
    assert _interposition_state(model) == before


def test_hookless_model_installs_no_observers_or_hook_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Matrix 19: no-hook capture installs zero observer/wrapper frames."""

    observer_calls = 0
    wrapper_calls = 0
    original_observer = provenance._ensure_observer
    original_wrapper = provenance._make_pre_hook_wrapper

    def count_observer(*args: Any, **kwargs: Any) -> Any:
        """Count observer installations."""

        nonlocal observer_calls
        observer_calls += 1
        return original_observer(*args, **kwargs)

    def count_wrapper(*args: Any, **kwargs: Any) -> Any:
        """Count hook wrapper constructions."""

        nonlocal wrapper_calls
        wrapper_calls += 1
        return original_wrapper(*args, **kwargs)

    monkeypatch.setattr(provenance, "_ensure_observer", count_observer)
    monkeypatch.setattr(provenance, "_make_pre_hook_wrapper", count_wrapper)
    tl.trace(nn.Sequential(nn.ReLU(), nn.ReLU()), torch.ones(1))

    assert observer_calls == 0
    assert wrapper_calls == 0


def test_requires_grad_metadata_mutation_is_detected_without_version_increment() -> None:
    """Amendment 1: requires-grad metadata changes are checked every transition."""

    model = _Unary()

    def change_metadata(_module: nn.Module, args: tuple[Any, ...]) -> None:
        """Change autograd metadata without changing tensor values."""

        args[0].requires_grad_()

    model.register_forward_pre_hook(change_metadata)
    trace = tl.trace(model, torch.ones(1))
    effect = _call(trace, "self:1").forward_pre_hook_effects[0]

    assert effect.changed is True
    assert effect.change_kinds == ("tensor_metadata",)


def test_full_backward_hook_rewrap_does_not_redefine_snapshot_b() -> None:
    """Amendment 2: B is the last wrapper state, not backward-hook forward-entry views."""

    model = _Unary()
    model.register_forward_pre_hook(lambda _m, args: (args[0] + 2,))
    backward_handle = model.register_full_backward_hook(lambda _m, grad_in, _grad_out: grad_in)
    try:
        trace = tl.trace(model, torch.ones(1, requires_grad=True), backward_ready=True)
    finally:
        backward_handle.remove()

    call = _call(trace, "self:1")
    assert torch.equal(call.inputs_after_pre_hooks.args[0], torch.tensor([3.0]))
    assert call.forward_pre_hook_effects[0].changed is True


def test_registration_bypass_downgrades_but_keeps_after_snapshot_truthful() -> None:
    """Amendment 3: subclass registration bypass yields explicit incomplete attribution."""

    class BypassUnary(_Unary):
        """Bypass the session-scoped class registration patch."""

        def register_forward_pre_hook(self, hook: Callable[..., Any], **kwargs: Any) -> Any:
            """Delegate through the pre-bound original function."""

            original = nn.Module.register_forward_pre_hook.__wrapped__
            return original(self, hook, **kwargs)

    class RegisterDuringForward(_Nested):
        """Register a bypass hook immediately before the child invocation."""

        def __init__(self) -> None:
            """Initialize the bypass child."""

            super().__init__(BypassUnary())

        def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            """Register through the subclass override and call the child."""

            self.child.register_forward_pre_hook(lambda _m, args: (args[0] + 7,))
            return self.child(x, **kwargs)

    trace = tl.trace(RegisterDuringForward(), torch.ones(1))
    call = _call(trace)

    assert call.inputs_before_pre_hooks is None
    assert torch.equal(call.inputs_after_pre_hooks.args[0], torch.tensor([8.0]))
    assert call.inputs_after_pre_hooks.capture_complete is False
    assert "registration_interposition_bypassed" in (call.inputs_after_pre_hooks.incomplete_reasons)
