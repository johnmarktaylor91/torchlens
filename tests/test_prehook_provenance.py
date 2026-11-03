"""Decision-matrix coverage for forward-pre-hook input provenance."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.modules.module as torch_module
from torch import nn

import torchlens as tl
from torchlens.backends.torch import prehook_provenance as provenance
from torchlens.errors import PredicateError, TraceCaptureError


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


def _call(trace: tl.Trace, label: str = "child:1") -> tl.types.ModuleCall:
    """Return one ModuleCall by label."""

    return trace.module_calls[label]


def _tensor(
    snapshot: tl.types.ModuleInputSnapshot, path: tuple[Any, ...] = ("args", 0)
) -> torch.Tensor:
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
    assert call.forward_pre_hook_effects[0].has_changes is False
    assert call.inputs_before_pre_hooks.tensor_observations[0].payload_origin == (
        "immutable_producer_snapshot"
    )
    assert (
        _tensor(call.inputs_before_pre_hooks).data_ptr()
        == _tensor(call.inputs_after_pre_hooks).data_ptr()
    )


def test_opaque_object_tensor_mutation_downgrades_provenance_fail_closed() -> None:
    """Opaque objects carrying tensors must downgrade provenance completeness."""

    class _Box:
        """Container with a tensor hidden behind an attribute."""

        def __init__(self, x: torch.Tensor) -> None:
            """Store the hidden tensor."""

            self.x = x

    class _OpaqueChild(nn.Module):
        """Consume a tensor stored on an opaque input object."""

        def forward(self, box: _Box) -> torch.Tensor:
            """Apply relu to the hidden tensor."""

            return torch.relu(box.x)

    class _OpaqueRoot(nn.Module):
        """Expose the child module through one named submodule call."""

        def __init__(self) -> None:
            """Initialize the child module."""

            super().__init__()
            self.child = _OpaqueChild()

        def forward(self, box: _Box) -> torch.Tensor:
            """Delegate to the child module."""

            return self.child(box)

    model = _OpaqueRoot()

    def mutate(_module: nn.Module, args: tuple[Any, ...]) -> None:
        """Mutate the tensor hidden inside the opaque object."""

        box = args[0]
        assert isinstance(box, _Box)
        box.x.add_(1)

    model.child.register_forward_pre_hook(mutate)
    trace = tl.trace(
        model,
        _Box(torch.tensor([-2.0, 1.0])),
        capture=tl.options.CaptureOptions(save_args=True),
        save="all",
    )
    call = _call(trace)
    effect = call.forward_pre_hook_effects[0]

    assert effect.has_changes is None
    assert effect.change_kinds == ()
    assert effect.incomplete_reasons == ("opaque_object_untraversed",)
    assert call.inputs_before_pre_hooks is not None
    assert call.inputs_before_pre_hooks.is_capture_complete is False
    assert call.inputs_before_pre_hooks.incomplete_reasons == ("opaque_object_untraversed",)
    assert call.inputs_after_pre_hooks.is_capture_complete is False
    assert call.inputs_after_pre_hooks.incomplete_reasons == ("opaque_object_untraversed",)


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
    with pytest.raises(TraceCaptureError, match=expected) as _tce_info2:
        tl.trace(bad_model, torch.ones(1))
    assert isinstance(_tce_info2.value.__cause__, RuntimeError)


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
    with pytest.raises(TraceCaptureError, match="hook exploded") as exc_info:
        tl.trace(model, torch.ones(1))
    assert isinstance(exc_info.value.__cause__, LookupError)
    assert hasattr(exc_info.value, "partial_trace")
    assert _interposition_state(model) == before

    handle.remove()
    assert torch.equal(model(torch.ones(1)), torch.ones(1))


def test_root_pre_hook_populates_self_call() -> None:
    """Matrix 11: root provenance binds to ``self:1``."""

    model = _Unary()
    model.register_forward_pre_hook(lambda _m, args: (args[0] + 2,))
    trace = tl.trace(model, torch.ones(1))

    root = _call(trace, "self:1")
    assert torch.equal(root.inputs_before_pre_hooks.args[0], torch.ones(1))
    assert torch.equal(root.inputs_after_pre_hooks.args[0], torch.tensor([3.0]))


def test_record_to_trace_ram_and_disk_preserve_provenance(tmp_path: Path) -> None:
    """Matrix 15: predicate Recording materializes provenance in RAM and disk modes."""

    for streaming in (
        None,
        tl.options.StorageOptions(path=tmp_path / "recording.tlfast", retain_in_memory=False),
    ):
        model = _Nested()
        model.child.register_forward_pre_hook(lambda _m, args: (args[0] + 1,))
        recording = tl.record(
            model,
            torch.ones(1),
            storage=streaming,
            default_op=True,
        )
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

    assert call.forward_pre_hook_effects[0].has_changes is True
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
    with pytest.raises(TraceCaptureError, match="forward failed") as _tce_info3:
        tl.trace(model, torch.ones(1))
    assert isinstance(_tce_info3.value.__cause__, ArithmeticError)
    assert _interposition_state(model) == before
    model.fail = False

    tl.trace(model, torch.ones(1), stop_after=lambda ctx: ctx.kind == "op")
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
    with pytest.raises(TraceCaptureError, match="prep injection") as _tce_info4:
        tl.trace(model, torch.ones(1))
    assert isinstance(_tce_info4.value.__cause__, RuntimeError)
    assert _interposition_state(model) == before


def test_requires_grad_metadata_mutation_is_detected_without_version_increment() -> None:
    """Amendment 1: requires-grad metadata changes are checked every transition."""

    model = _Unary()

    def change_metadata(_module: nn.Module, args: tuple[Any, ...]) -> None:
        """Change autograd metadata without changing tensor values."""

        args[0].requires_grad_()

    model.register_forward_pre_hook(change_metadata)
    trace = tl.trace(model, torch.ones(1))
    effect = _call(trace, "self:1").forward_pre_hook_effects[0]

    assert effect.has_changes is True
    assert effect.change_kinds == ("tensor_metadata",)


def test_full_backward_hook_rewrap_does_not_redefine_snapshot_b() -> None:
    """Amendment 2: B is the last wrapper state, not backward-hook forward-entry views."""

    model = _Unary()
    model.register_forward_pre_hook(lambda _m, args: (args[0] + 2,))
    backward_handle = model.register_full_backward_hook(lambda _m, grad_in, _grad_out: grad_in)
    try:
        trace = tl.trace(
            model,
            torch.ones(1, requires_grad=True),
            capture=tl.options.CaptureOptions(backward_ready=True),
        )
    finally:
        backward_handle.remove()

    call = _call(trace, "self:1")
    assert torch.equal(call.inputs_after_pre_hooks.args[0], torch.tensor([3.0]))
    assert call.forward_pre_hook_effects[0].has_changes is True


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
    assert call.inputs_after_pre_hooks.is_capture_complete is False
    assert "registration_interposition_bypassed" in (call.inputs_after_pre_hooks.incomplete_reasons)
    assert call.forward_pre_hook_effects == ()
    assert call.has_pre_hook_input_change is None
