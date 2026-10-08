"""Every capture entry point hands the model back exactly as it went in.

Regression pins for two model-state leaks:

* After any ``tl.trace`` or ``tl.record``, every non-root submodule kept an
  instance-level TorchLens ``forward`` wrapper that closed over the ORIGINAL
  module and its bound forward. ``copy.deepcopy`` treats functions as atomic,
  so a deepcopy made after one inspection (an EMA teacher, a frozen reference,
  "try an edit on a copy") ran the original's weights, sent its gradients to
  the original, and crashed ``tl.trace(copy)`` with ``KeyError``. Whole-model
  ``pickle`` / ``torch.save`` failed outright.
* A ``KeyboardInterrupt`` or ``SystemExit`` during ``tl.trace`` left the
  wrapper on the model where a ``RuntimeError`` did not.

The failure matrix crosses every capture entry point with a normal return, a
``RuntimeError``, a ``KeyboardInterrupt`` and a ``SystemExit`` raised inside
the model, and asserts a model-state fingerprint is unchanged, the exception
type survives, and a deepcopy taken afterwards is independent of the
original. The oracles compare against a model TorchLens never touched, never
against the same call repeated.
"""

from __future__ import annotations

import copy
import io
import pickle
import random
import warnings
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _state

_X = torch.tensor([[1.0, -2.0, 0.5]])
_FILL = 0.5


class _Gate(nn.Module):
    """Pass-through that raises ``exc`` when set (the failure injection point)."""

    def __init__(self) -> None:
        """Start with no injected failure."""

        super().__init__()
        self.exc: type[BaseException] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` or raise the injected exception type."""

        if self.exc is not None:
            raise self.exc("injected model failure")
        return x


class _Doubler(nn.Module):
    """Module whose instance-level ``forward`` is a bound method of itself."""

    def __init__(self) -> None:
        """Build the linear map and pin the instance forward."""

        super().__init__()
        self.lin = nn.Linear(3, 3)
        self.forward = self.doubled  # type: ignore[method-assign]

    def doubled(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear map and double it."""

        return self.lin(x) * 2.0


class _Model(nn.Module):
    """Nested model: a shared block, a user instance forward and a gate."""

    def __init__(self) -> None:
        """Build the children."""

        super().__init__()
        self.block = nn.Sequential(nn.Linear(3, 3), nn.Tanh())
        self.alt = _Doubler()
        self.gate = _Gate()
        self.head = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run block, alt and gate, then the head."""

        hidden = self.block(x)
        hidden = self.gate(self.alt(hidden)) + hidden
        return self.head(hidden)

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """Bound-method entry point that calls the model once."""

        return self(x)


def _fresh() -> _Model:
    """Return a deterministically initialized model TorchLens never touched."""

    torch.manual_seed(7)
    return _Model().eval()


def _filled(model: nn.Module) -> nn.Module:
    """Fill every parameter of ``model`` with ``_FILL`` in place and return it."""

    with torch.no_grad():
        for param in model.parameters():
            param.fill_(_FILL)
    return model


def _value_fingerprint(value: Any) -> Any:
    """Identity-or-value fingerprint for one ``module.__dict__`` entry."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return ("value", value)
    if isinstance(value, dict):
        return ("dict", tuple((key, id(item)) for key, item in value.items()))
    return ("id", type(value).__qualname__, id(value))


def _model_fingerprint(model: nn.Module) -> dict[str, Any]:
    """Snapshot module, parameter and process state a capture could leak.

    Parameters
    ----------
    model:
        Model under test.

    Returns
    -------
    dict[str, Any]
        Comparable snapshot: every module's instance ``__dict__`` (hook
        dicts, parameter/buffer/child registries and plain attributes by
        identity), every parameter's identity, storage, version and grad
        state, the global module hooks, the torch function-mode stack, the
        TorchLens logging flags, grad and inference mode, and the RNG streams.
    """

    import torch.nn.modules.module as module_mod

    modules = {
        name: tuple((key, _value_fingerprint(value)) for key, value in sorted(vars(module).items()))
        for name, module in model.named_modules(remove_duplicate=False)
    }
    # Values, not the version counter: tl.validate snapshots and restores
    # parameter storage in place, which bumps ``_version`` with equal bytes.
    params = {
        name: (
            id(param),
            param.data_ptr(),
            param.detach().clone(),
            param.requires_grad,
            param.grad is None,
            param.is_leaf,
        )
        for name, param in model.named_parameters(remove_duplicate=False)
    }
    global_hooks = tuple(
        (name, tuple(getattr(module_mod, name, {}).keys()))
        for name in (
            "_global_forward_hooks",
            "_global_forward_pre_hooks",
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
            "_global_forward_hooks_always_called",
            "_global_module_registration_hooks",
            "_global_parameter_registration_hooks",
            "_global_buffer_registration_hooks",
        )
    )
    len_mode_stack = getattr(torch._C, "_len_torch_function_stack", None)
    return {
        "modules": modules,
        "params": params,
        "global_hooks": global_hooks,
        "function_mode_stack": None if len_mode_stack is None else len_mode_stack(),
        "logging": (_state._logging_enabled, _state._active_trace is None),
        "grad_mode": (torch.is_grad_enabled(), torch.is_inference_mode_enabled()),
        "torch_rng": torch.get_rng_state().tolist(),
        "python_rng": random.getstate(),
    }


def _assert_fingerprints_equal(
    before: dict[str, Any], after: dict[str, Any], *, skip: frozenset[str] = frozenset()
) -> None:
    """Assert two fingerprints match, naming the first differing component.

    Parameters
    ----------
    before, after:
        Fingerprints from :func:`_model_fingerprint`.
    skip:
        Components a door documents as changed (never model state).
    """

    for key in before:
        if key in skip:
            continue
        if key == "modules":
            for name, entries in before[key].items():
                assert after[key].get(name) == entries, (
                    f"module {name or '<root>'!r} instance state changed: "
                    f"before keys {[entry[0] for entry in entries]}, "
                    f"after keys {[entry[0] for entry in after[key].get(name, ())]}"
                )
            assert set(after[key]) == set(before[key]), "module tree changed"
        elif key == "params":
            assert set(after[key]) == set(before[key]), "parameter set changed"
            for name, (*ident, value) in (
                (name, (*entry[:2], *entry[3:], entry[2])) for name, entry in before[key].items()
            ):
                after_entry = after[key][name]
                after_ident = (*after_entry[:2], *after_entry[3:])
                assert after_ident == tuple(ident), f"parameter {name!r} identity/flags changed"
                assert torch.equal(after_entry[2], value), f"parameter {name!r} value changed"
        else:
            assert after[key] == before[key], f"{key} changed across the call"


def _eager_out(model: nn.Module) -> torch.Tensor:
    """Run ``model`` eagerly without grad and return a detached clone."""

    with torch.no_grad():
        return model(_X).detach().clone()


def _assert_copy_independent(model: nn.Module) -> None:
    """Assert a deepcopy of ``model`` computes, and trains, on its own weights.

    The oracle is a fresh, never-captured model with the same fill, so a copy
    that silently runs the original's weights or sends gradients to the
    original fails.
    """

    reference = _eager_out(model)
    clone = _filled(copy.deepcopy(model))
    expected = _eager_out(_filled(_fresh()))
    assert torch.equal(_eager_out(clone), expected), "the copy ran the original's weights"
    assert torch.equal(_eager_out(model), reference), "filling the copy changed the original"

    clone(_X).sum().backward()
    assert all(param.grad is not None for param in clone.parameters()), (
        "backward through the copy left the copy's grads empty"
    )
    assert all(param.grad is None for param in model.parameters()), (
        "backward through the copy filled the original's grads"
    )

    copy_trace = tl.trace(clone, _X)
    assert torch.equal(copy_trace[copy_trace.output_layers[0]].out.detach(), expected)
    assert torch.equal(_eager_out(clone), expected)


def _spec() -> Any:
    """A module-boundary add spec (non-idempotent, so a stale hook shows)."""

    return tl.when(tl.module("block"), tl.add(1.0))


def _door_trace(model: _Model) -> None:
    """Plain ``tl.trace``."""

    tl.trace(model, _X)


def _door_trace_intervene(model: _Model) -> None:
    """``tl.trace`` with a live intervention."""

    tl.trace(model, _X, intervene=_spec())


def _door_trace_bound_method(model: _Model) -> None:
    """``tl.trace`` of a bound method (TL-authored wrapper root)."""

    tl.trace(model.run, _X)


def _door_record(model: _Model) -> None:
    """``tl.record`` with an opaque ``save=``."""

    tl.record(model, _X, save=lambda ctx: True, return_output=True)


def _door_record_intervene(model: _Model) -> None:
    """``tl.record`` with a live intervention."""

    tl.record(model, _X, save=lambda ctx: True, intervene=_spec(), return_output=True)


def _door_bind(model: _Model) -> None:
    """``spec.bind(model)(x)``."""

    with torch.no_grad():
        _spec().bind(model)(_X)


def _door_validate(model: _Model) -> None:
    """``tl.validate`` (deep-copies and captures internally)."""

    tl.validate(model, _X, scope="forward")


_DOORS: dict[str, Callable[[_Model], None]] = {
    "trace": _door_trace,
    "trace_intervene": _door_trace_intervene,
    "trace_bound_method": _door_trace_bound_method,
    "record": _door_record,
    "record_intervene": _door_record_intervene,
    "bind": _door_bind,
    "validate": _door_validate,
}

# tl.validate documents that it seeds the global RNGs for its ground-truth and
# replay runs and does not restore them; every other component, and every
# component of every other door, must come back unchanged.
_DOCUMENTED_CHANGES: dict[str, frozenset[str]] = {
    "validate": frozenset({"torch_rng", "python_rng"}),
}

_FAILURES: dict[str, type[BaseException] | None] = {
    "normal": None,
    "runtime_error": RuntimeError,
    "keyboard_interrupt": KeyboardInterrupt,
    "system_exit": SystemExit,
}


@pytest.mark.parametrize("failure", list(_FAILURES))
@pytest.mark.parametrize("door", list(_DOORS))
def test_entry_point_returns_the_model_unchanged(door: str, failure: str) -> None:
    """The model state fingerprint is identical before and after every call.

    A failing call must also preserve the exception type, and a deepcopy taken
    afterwards must be independent of the original.
    """

    model = _fresh()
    exc_type = _FAILURES[failure]
    model.gate.exc = exc_type
    before = _model_fingerprint(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if exc_type is None:
            _DOORS[door](model)
        else:
            with pytest.raises(exc_type, match="injected model failure"):
                _DOORS[door](model)
    _assert_fingerprints_equal(
        before, _model_fingerprint(model), skip=_DOCUMENTED_CHANGES.get(door, frozenset())
    )

    model.gate.exc = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _assert_copy_independent(model)


@pytest.mark.parametrize("door", ["trace", "record"])
def test_captured_model_pickles_like_a_never_captured_model(door: str) -> None:
    """Whole-model pickle and torch.save work after a capture, TorchLens-free.

    The serialized bytes must not reference TorchLens at all (a never-captured
    model's pickle loads in a process without TorchLens), and the round trip
    must compute what the original computes.
    """

    model = _fresh()
    _DOORS[door](model)
    expected = _eager_out(model)

    blob = pickle.dumps(model)
    assert b"torchlens" not in blob, "the pickle references TorchLens objects"
    restored = pickle.loads(blob)
    assert torch.equal(_eager_out(restored), expected)
    assert torch.equal(_eager_out(_filled(restored)), _eager_out(_filled(_fresh())))
    assert torch.equal(_eager_out(model), expected), "filling the round trip changed the original"

    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False)
    assert torch.equal(_eager_out(loaded), expected)


def test_copy_of_captured_model_supports_state_dict_and_dtype_moves() -> None:
    """state_dict round trips and ``.to()`` on a copy never reach the original."""

    model = _fresh()
    tl.trace(model, _X)
    expected = _eager_out(model)

    clone = copy.deepcopy(model)
    clone.load_state_dict(_filled(_fresh()).state_dict())
    assert torch.equal(_eager_out(clone), _eager_out(_filled(_fresh())))
    assert torch.equal(_eager_out(model), expected)

    as_double = copy.deepcopy(model).to(torch.float64)
    oracle = _fresh().to(torch.float64)
    with torch.no_grad():
        assert torch.equal(as_double(_X.double()), oracle(_X.double()))
    assert all(param.dtype == torch.float32 for param in model.parameters())
    assert torch.equal(_eager_out(model), expected)


def test_user_instance_forward_survives_capture_by_identity() -> None:
    """A user instance-level forward is put back as the same object."""

    model = _fresh()
    pinned = model.alt.__dict__["forward"]
    tl.trace(model, _X)
    assert model.alt.__dict__["forward"] is pinned
    assert "forward" not in model.block.__dict__
    tl.release_model(model)
    assert model.alt.__dict__["forward"] is pinned


def test_captured_model_called_inside_another_capture() -> None:
    """A model captured earlier and called as an unregistered helper still traces."""

    helper = _fresh()
    tl.trace(helper, _X)

    class _Outer(nn.Module):
        """Calls ``helper`` without registering it as a child."""

        def __init__(self) -> None:
            """Hold the helper in a list so it is not a submodule."""

            super().__init__()
            self.helpers = [helper]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run the helper."""

            return self.helpers[0](x) * 3.0

    with warnings.catch_warnings():
        # The helper's weights are not registered on _Outer, so the capture
        # discloses them as sourceless tensors; that is the documented
        # behavior for outside tensors, not this test's subject.
        warnings.simplefilter("ignore")
        outer_trace = tl.trace(_Outer(), _X)
    expected = _eager_out(helper) * 3.0
    assert torch.equal(outer_trace[outer_trace.output_layers[0]].out.detach(), expected)
