"""Module state a forward reaches through its helper functions is a pre-forward constant.

Two Menagerie models failed source-provenance validation at an elementwise op whose
first argument was a tensor built in an earlier pass and kept in module-level state
that the forward reaches only through helper functions:

- TinyHITNet (``hitnet``): ``forward`` calls ``hyp_up``, which calls ``disp_up``,
  which reads its warp coefficients from ``make_warp_coef``, a module-level
  ``functools.lru_cache`` function; ``coef_x * dx`` failed at ``mul``.
- DCVC-DC: a module ``forward`` calls ``flow_warp``, which calls ``torch_warp``,
  which adds the flow to a sampling grid cached in the module-global list of dicts
  ``backward_grid``; ``grid + flow`` failed at ``add``.

The pre-forward ownership snapshot and the session-end clear read only the globals
the forward names directly, so neither saw the cache, the warm tensor was an
unattributed argument, and a tensor a capture built into a cold cache kept a stale
label into the next capture. Both sweeps now follow the user helper functions a
forward names (module globals and methods of the module's own class) for a bounded
number of hops, so these tensors are recorded as ``constant`` sources. A tensor
first made mid-forward and stored into the same state still fails validation.
"""

from __future__ import annotations

import functools
import warnings

import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens.validation import last_validation_failure


@functools.lru_cache
@torch.no_grad()
def _make_warp_coef(scale: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """TinyHITNet's cached coefficient builder: an ``lru_cache`` over a no-grad function."""

    index = torch.arange(scale, device=device) - (scale - 1) / 2
    coef_y, coef_x = torch.meshgrid(index, index, indexing="ij")
    return coef_x.reshape(1, -1, 1, 1), coef_y.reshape(1, -1, 1, 1)


def _disp_up(d: torch.Tensor, dx: torch.Tensor, dy: torch.Tensor, scale: int) -> torch.Tensor:
    """Read the cached coefficients; the forward never names the cached function."""

    coef_x, coef_y = _make_warp_coef(scale, d.device)
    return d + coef_x * dx + coef_y * dy


def _hyp_up(hyp: torch.Tensor) -> torch.Tensor:
    """The helper the forward names, one hop above the coefficient reader."""

    return _disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], 2)


class _HitnetStandIn(nn.Module):
    """``forward -> _hyp_up -> _disp_up -> _make_warp_coef`` (an ``lru_cache``)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _hyp_up(self.conv(x))


def _warp_and_aggregate(hyp: torch.Tensor) -> torch.Tensor:
    """A helper reached only from a method of the module's class."""

    return _disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], 3)


class _MethodRouteStandIn(nn.Module):
    """``forward -> self.forward_once -> _warp_and_aggregate -> _disp_up``."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)

    def forward_once(self, hyp: torch.Tensor) -> torch.Tensor:
        return _warp_and_aggregate(hyp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_once(self.conv(x))


#: DCVC-DC's ``backward_grid``: one dict per device, keyed on the flow's size string.
_BACKWARD_GRID: list[dict[str, torch.Tensor]] = [{} for _ in range(2)]


def _add_grid_cache(flow: torch.Tensor) -> None:
    """Build the sampling grid for this flow size once and keep it in the global cache."""

    if str(flow.size()) not in _BACKWARD_GRID[-1]:
        n, _, h, w = flow.size()
        hor = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(n, -1, h, -1)
        ver = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(n, -1, -1, w)
        _BACKWARD_GRID[-1][str(flow.size())] = torch.cat([hor, ver], 1)


def _torch_warp(feature: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Add the flow to the cached grid (the op that failed) and sample with it."""

    _add_grid_cache(flow)
    grid = _BACKWARD_GRID[-1][str(flow.size())] + flow
    return nn.functional.grid_sample(feature, grid.permute(0, 2, 3, 1), align_corners=True)


def _flow_warp(im: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """The helper the forward names, one hop above the grid reader."""

    return _torch_warp(im, flow)


class _DcvcStandIn(nn.Module):
    """``forward -> _flow_warp -> _torch_warp -> _BACKWARD_GRID`` (a global dict cache)."""

    def __init__(self) -> None:
        super().__init__()
        self.flow = nn.Conv2d(3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _flow_warp(x, self.flow(x))


def _warm(model: nn.Module, x: torch.Tensor) -> nn.Module:
    model.eval()
    with torch.no_grad():
        model(x)
    return model


def _types(trace: object) -> list[str]:
    return [op.type for op in trace._layer_list]  # type: ignore[attr-defined]


def test_lru_cache_two_helper_hops_from_forward_is_a_constant_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 3, 4, 4)
    model = _warm(_HitnetStandIn(), x)
    trace = tl.trace(model, x)
    assert "constant" in _types(trace)
    assert tl.validate(model, x, scope="forward") is True


def test_lru_cache_reached_through_a_class_method_is_a_constant_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 3, 5, 5)
    model = _warm(_MethodRouteStandIn(), x)
    trace = tl.trace(model, x)
    assert "constant" in _types(trace)
    assert tl.validate(model, x, scope="forward") is True


def test_global_dict_cache_two_helper_hops_from_forward_is_a_constant_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 3, 6, 6)
    model = _warm(_DcvcStandIn(), x)
    trace = tl.trace(model, x)
    assert "constant" in _types(trace)
    assert tl.validate(model, x, scope="forward") is True


def test_cold_global_cache_filled_during_capture_leaves_no_stale_label() -> None:
    torch.manual_seed(0)
    x = torch.randn(1, 3, 7, 9)  # a size no other test warms, so the cache starts cold
    model = _DcvcStandIn().eval()
    first = tl.trace(model, x)
    assert "constant" not in _types(first)  # the grid was built inside the capture
    second = tl.trace(model, x)
    assert "constant" in _types(second)
    assert tl.validate(model, x, scope="forward") is True


#: A global cache a helper refills on every call with a tensor made mid-forward.
_MID_FORWARD_CACHE: dict[str, torch.Tensor] = {}
#: An unwrapped ``torch.mul`` the planted helper uses (set by the test).
_STALE_MUL: list[OpaqueCallable] = []


def _refill_from_input(x: torch.Tensor) -> None:
    """Store a tensor made from the input by an op no wrapper sees."""

    _MID_FORWARD_CACHE["grid"] = _STALE_MUL[0](x, 2.0)


def _read_refilled(x: torch.Tensor) -> torch.Tensor:
    _refill_from_input(x)
    return _MID_FORWARD_CACHE["grid"] + x


def _read_refilled_outer(x: torch.Tensor) -> torch.Tensor:
    return _read_refilled(x)


class _MidForwardCacheModel(nn.Module):
    """``forward -> _read_refilled_outer -> _read_refilled``, whose cache entry is new each pass."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _read_refilled_outer(self.proj(x))


def test_mid_forward_tensor_in_helper_reached_cache_still_fails_validation() -> None:
    from torchlens import _state
    from torchlens.backends.torch.wrappers import wrap_torch

    wrap_torch()
    _STALE_MUL[:] = [OpaqueCallable(_state._decorated_to_orig.get(id(torch.mul), torch.mul))]
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = _warm(_MidForwardCacheModel(), x)  # the snapshot pins last pass's entry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the provenance warning is not under test
        assert tl.validate(model, x, scope="forward") is False
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "source_provenance"
    assert "unattributed_tensor_args" in failure.message
