"""Tensors returned by a warm ``functools.lru_cache`` are pre-forward constants.

Transformers builds DETR-family sine position embeddings in a static method
decorated with an ``lru_cache`` keyed on the input shape. On a warmed model the
module's forward returns the tensor the cache built during an earlier pass: no op
runs, and the tensor is reachable from no module attribute. The capture adopted it
at the module exit as a functionless internal source and source-provenance
validation failed the trace (``module_boundary_adoption``). The pre-forward
ownership snapshot and the session-end clear now read the contents of C
``lru_cache`` wrappers reachable from a model's forward callables (their closures,
defaults and referenced globals, and the class-level callables a module's methods
name), so a cached tensor is recorded as a ``constant`` source and a cold cache
filled during one capture leaves no stale label for the next. A tensor first made
mid-forward still fails validation.
"""

from __future__ import annotations

import functools
import warnings

import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens.validation import last_validation_failure


def _compile_compatible_cache(maxsize: int) -> object:
    """The shape of transformers' ``compile_compatible_method_lru_cache``."""

    def decorator(func):  # type: ignore[no-untyped-def]
        cached = functools.lru_cache(maxsize=maxsize)(func)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            return cached(*args, **kwargs)

        return wrapper

    return decorator


class _SinePositions(nn.Module):
    """A position embedding whose builder is a cached static method."""

    @staticmethod
    @_compile_compatible_cache(maxsize=1)  # type: ignore[misc]
    def build(shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        steps = torch.arange(1, shape[-1] + 1, dtype=dtype)
        return torch.stack((steps.sin(), steps.cos())).expand(shape[0], 2, shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.build(x.shape, x.dtype)


class _CachedPositionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.positions = _SinePositions()
        self.proj = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.positions(x).sum(1, keepdim=True)


@functools.lru_cache(maxsize=4)
def _cached_grid(width: int) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, width)


class _GlobalCachedGrid(nn.Module):
    """A forward that reads a module-level cached function's tensor directly."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) * _cached_grid(x.shape[-1])


def _warm(model: nn.Module, x: torch.Tensor) -> nn.Module:
    model.eval()
    with torch.no_grad():
        model(x)
    return model


def _types(trace: object) -> list[str]:
    return [op.type for op in trace._layer_list]  # type: ignore[attr-defined]


def test_warm_cached_static_method_output_is_a_constant_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 6)
    model = _warm(_CachedPositionModel(), x)
    trace = tl.trace(model, x)
    assert "constant" in _types(trace)
    assert "internalsource" not in _types(trace)
    assert tl.validate(model, x, scope="forward") is True


def test_cold_cache_filled_during_capture_leaves_no_stale_label() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, 7)  # a width no other test warms, so the cache starts cold
    model = _CachedPositionModel().eval()
    model.proj = nn.Linear(7, 7)
    first = tl.trace(model, x)
    assert "constant" not in _types(first)  # the builder ran inside the capture
    second = tl.trace(model, x)
    assert "constant" in _types(second)
    assert tl.validate(model, x, scope="forward") is True


def test_module_level_cached_function_output_is_a_constant_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(3, 5)
    model = _warm(_GlobalCachedGrid(), x)
    trace = tl.trace(model, x)
    assert "constant" in _types(trace)
    assert tl.validate(model, x, scope="forward") is True


class _MidForwardEscapeOut(nn.Module):
    """A module that returns a tensor first made mid-forward by a stale reference."""

    def __init__(self, stale_mul: OpaqueCallable) -> None:
        super().__init__()
        self.stale_mul = stale_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stale_mul(x, 2.0)


class _EscapeHost(nn.Module):
    def __init__(self, stale_mul: OpaqueCallable) -> None:
        super().__init__()
        self.inner = _MidForwardEscapeOut(stale_mul)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.inner(x))


def test_mid_forward_untraced_module_output_still_fails_validation() -> None:
    from torchlens import _state
    from torchlens.backends.torch.wrappers import wrap_torch

    wrap_torch()
    stale_mul = OpaqueCallable(_state._decorated_to_orig.get(id(torch.mul), torch.mul))
    model = _EscapeHost(stale_mul)
    x = torch.randn(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the escape's provenance warning is not under test
        assert tl.validate(model, x, scope="forward") is False
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "source_provenance"
    assert "module_boundary_adoption" in failure.message
