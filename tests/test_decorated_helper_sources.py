"""Module state reached through decorated helpers, child-class methods, and distribution outputs.

Four Menagerie models failed the graph-connectivity invariant ("is_internal_source but
consumed unattributed tensor data") at the first op reading a tensor kept in
module-level state that the pre-forward snapshot never reached:

- L-GATr (m17701): ``EquiLinear.forward -> equi_linear`` (wrapped by a
  ``functools.wraps`` autocast decorator) ``-> _equi_linear_dense ->
  _compute_pin_equi_linear_basis`` (an ``lru_cache`` returning the module global
  ``_BASIS_SUBGROUP``); ``basis.flatten`` failed.
- Equiformer (m913): a ``@beartype`` forward ``-> get_D_to_from_z_axis -> irr_repr ->
  wigner_d_matrix`` reading the module-global dict ``Jd``.
- AnySat (m14703): ``iRPE.forward -> self._get_rp_bucket -> get_bucket_ids_2d ->
  get_bucket_ids_2d_without_skip`` (both ``@torch.no_grad()``) reading the global
  dict cache ``BUCKET_IDS_BUF``.
- CrystalGRW (m15493): ``EquiformerV2.forward -> self.SO3_rotation[i].set_wigner``
  (a method of a CHILD module's class) ``-> self.RotationToWignerDMatrix ->
  wigner_D`` reading the module-global list ``_Jd``.

The helper walk spent a hop on every decorator and lost the decorated function
(it lives in a closure cell the wrapper never names), and it resolved ``x.method``
names only against the forward's own class. Decorators built with
``functools.wraps`` are now transparent, and names resolve against every module
class in the model.

Two more failed with ``OutputAttributionError`` on output addresses
``0._exp.mean`` (Poisson VAE, m19409) and ``rec.pdf.dist.variance``
(StackedCapsuleAE, m2931): the outputs hold a ``torch.distributions`` object inside
a container TorchLens walks by attribute, and the walk evaluated the
distribution's derived properties, making a fresh tensor after the forward. The
walk now reads a distribution's instance state only.

Planted-bug tests keep the checks' teeth: a cache refilled mid-forward by a hidden
op behind a decorator still fails validation, and a distribution parameter made by
a hidden op still fails output attribution in validation.
"""

from __future__ import annotations

import functools
import warnings

import pytest
import torch
from _stale_holders import OpaqueCallable
from torch import nn

import torchlens as tl
from torchlens._errors import OutputAttributionError


def _passthrough_decorator(func):  # type: ignore[no-untyped-def]
    """A ``functools.wraps`` decorator like L-GATr's autocast pin and ``@beartype``."""

    @functools.wraps(func)
    def decorated(*args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    return decorated


def _warm(model: nn.Module, x: torch.Tensor) -> nn.Module:
    model.eval()
    with torch.no_grad():
        model(x)
    return model


def _types(trace: object) -> list[str]:
    return [op.type for op in trace._layer_list]  # type: ignore[attr-defined]


# L-GATr: a decorated helper, an undecorated helper, then an lru_cache over a global.
_BASIS = torch.randn(4, 4)


@functools.lru_cache
def _basis(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the module global itself when device and dtype already match."""

    return _BASIS.to(device=device, dtype=dtype)


def _equi_dense(x: torch.Tensor) -> torch.Tensor:
    basis = _basis(x.device, x.dtype)
    return x @ basis.flatten(-1).reshape(4, 4)


@_passthrough_decorator
def _equi_linear(x: torch.Tensor) -> torch.Tensor:
    return _equi_dense(x)


class _LgatrStandIn(nn.Module):
    """``forward -> @wraps(_equi_linear) -> _equi_dense -> _basis`` (an ``lru_cache``)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _equi_linear(self.proj(x))


def test_lru_cache_behind_a_wraps_decorated_helper_is_a_constant_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = _warm(_LgatrStandIn(), x)
    assert "constant" in _types(tl.trace(model, x))
    assert tl.validate(model, x, scope="forward") is True


# L-GATr's sparse path: the cached tables are read inside a custom autograd.Function.
_GP = torch.randn(4, 4)
_GP_SIGNS = _GP.sign().squeeze(0)


@functools.lru_cache
def _sparse_gp_signs(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _GP_SIGNS.to(device=device, dtype=dtype)


class _GeometricProductSparse(torch.autograd.Function):
    """Reads the cached signs in its static ``forward``, which ``apply`` runs."""

    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        signs = _sparse_gp_signs(x.device, x.dtype)
        return signs * x

    @staticmethod
    def setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def,override]
        return None

    @staticmethod
    def backward(ctx, grad):  # type: ignore[no-untyped-def,override]
        return grad


def _gp_sparse(x: torch.Tensor) -> torch.Tensor:
    return _GeometricProductSparse.apply(x)


@_passthrough_decorator
def _geometric_product(x: torch.Tensor) -> torch.Tensor:
    return _gp_sparse(x)


class _LgatrSparseStandIn(nn.Module):
    """``forward -> @wraps _geometric_product -> _gp_sparse -> Function.apply -> lru cache``."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _geometric_product(self.proj(x))


def test_cache_read_inside_a_custom_autograd_function_is_a_constant_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 4)
    model = _warm(_LgatrSparseStandIn(), x)
    assert "constant" in _types(tl.trace(model, x))
    assert tl.validate(model, x, scope="forward") is True


# Equiformer: a decorated forward, then three plain helpers down to a global dict.
_JD = {1: torch.randn(4, 4), 2: torch.randn(4, 4)}


def _wigner(x: torch.Tensor, degree: int) -> torch.Tensor:
    return x @ _JD[degree].type(x.dtype)


def _irr_repr(x: torch.Tensor, degree: int) -> torch.Tensor:
    return _wigner(x, degree)


def _get_d(x: torch.Tensor) -> torch.Tensor:
    return _irr_repr(x, 1) + _irr_repr(x, 2)


class _EquiformerStandIn(nn.Module):
    """``@wraps forward -> _get_d -> _irr_repr -> _wigner -> _JD`` (a global dict)."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    @_passthrough_decorator
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _get_d(self.proj(x))


def test_global_dict_three_helpers_below_a_decorated_forward_is_a_constant_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = _warm(_EquiformerStandIn(), x)
    assert "constant" in _types(tl.trace(model, x))
    assert tl.validate(model, x, scope="forward") is True


# AnySat: a class method, then two torch.no_grad() helpers down to a global dict cache.
_BUCKET_BUF: dict[int, tuple[torch.Tensor, int]] = {}


@torch.no_grad()
def _bucket_ids_without_skip(width: int) -> torch.Tensor:
    if width not in _BUCKET_BUF:
        _BUCKET_BUF[width] = (torch.arange(width * width).reshape(width, width), width)
    value = _BUCKET_BUF[width]
    return value[0][:width, :width].reshape(-1)


@torch.no_grad()
def _bucket_ids(width: int) -> torch.Tensor:
    return _bucket_ids_without_skip(width)


class _AnysatStandIn(nn.Module):
    """``forward -> self._get_bucket -> @no_grad _bucket_ids -> @no_grad ... -> _BUCKET_BUF``."""

    def __init__(self) -> None:
        super().__init__()
        self.table = nn.Embedding(16, 4)

    def _get_bucket(self, width: int) -> torch.Tensor:
        return _bucket_ids(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.table(self._get_bucket(4)).sum(0)


def test_global_cache_behind_two_no_grad_helpers_is_a_constant_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = _warm(_AnysatStandIn(), x)
    assert "constant" in _types(tl.trace(model, x))
    assert tl.validate(model, x, scope="forward") is True


# CrystalGRW: the parent forward calls a method of a CHILD module's class.
_WIGNER_JD = [torch.randn(4, 4), torch.randn(4, 4)]


def _wigner_d(x: torch.Tensor) -> torch.Tensor:
    return x @ _WIGNER_JD[1].to(dtype=x.dtype, device=x.device)


class _So3Rotation(nn.Module):
    """A child whose own forward never calls the method the parent uses."""

    def __init__(self) -> None:
        super().__init__()
        self.wigner: torch.Tensor | None = None

    def _rotation_to_wigner(self, x: torch.Tensor) -> torch.Tensor:
        return _wigner_d(x)

    def set_wigner(self, x: torch.Tensor) -> None:
        self.wigner = self._rotation_to_wigner(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2


class _CrystalStandIn(nn.Module):
    """``forward -> self.rot.set_wigner -> self._rotation_to_wigner -> _wigner_d -> _WIGNER_JD``."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)
        self.rot = _So3Rotation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        self.rot.set_wigner(h)
        assert self.rot.wigner is not None
        return self.rot(self.rot.wigner)


def test_global_reached_through_a_child_class_method_is_a_constant_source() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = _warm(_CrystalStandIn(), x)
    assert "constant" in _types(tl.trace(model, x))
    assert tl.validate(model, x, scope="forward") is True


# Planted bug: a cache refilled mid-forward by a hidden op, behind a decorator.
_REFILLED: dict[str, torch.Tensor] = {}
_STALE_MUL: list[OpaqueCallable] = []


def _refill(x: torch.Tensor) -> torch.Tensor:
    _REFILLED["grid"] = _STALE_MUL[0](x, 2.0)
    return _REFILLED["grid"] + x


@_passthrough_decorator
def _refill_decorated(x: torch.Tensor) -> torch.Tensor:
    return _refill(x)


class _DecoratedRefillModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _refill_decorated(self.proj(x))


def _install_stale_mul() -> None:
    from torchlens import _state
    from torchlens.backends.torch.wrappers import wrap_torch

    wrap_torch()
    _STALE_MUL[:] = [OpaqueCallable(_state._decorated_to_orig.get(id(torch.mul), torch.mul))]


def test_mid_forward_tensor_behind_a_decorated_helper_still_fails_validation() -> None:
    _install_stale_mul()
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = _warm(_DecoratedRefillModel(), x)  # the snapshot pins last pass's entry
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the capture-gap warnings are not under test
        assert tl.validate(model, x, scope="forward") is False


# Poisson VAE / StackedCapsuleAE: a distribution inside an object walked by attribute.
class _PoissonHolder:
    """A plain (non-distribution) object holding a torch ``Exponential``, as P-VAE returns."""

    def __init__(self, rate: torch.Tensor) -> None:
        self.rate = rate
        self._exp = torch.distributions.Exponential(rate, validate_args=False)


class _PvaeStandIn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> list[object]:
        h = self.enc(x)
        return [_PoissonHolder(torch.exp(h)), h]


def test_distribution_inside_an_attribute_walked_output_is_not_evaluated() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = _PvaeStandIn().eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the unsupported-container fallback warning
        trace = tl.trace(model, x)  # raised OutputAttributionError at '0._exp.mean'
        assert trace is not None
        assert tl.validate(model, x, scope="forward") is True


class _HiddenRateModel(nn.Module):
    """Planted bug: the distribution's stored parameter comes from an op no wrapper sees."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> list[object]:
        h = self.enc(x)
        return [_PoissonHolder(_STALE_MUL[0](h, 2.0)), h]


def test_hidden_op_distribution_parameter_still_fails_output_attribution() -> None:
    _install_stale_mul()
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    model = _HiddenRateModel().eval()
    with warnings.catch_warnings(), pytest.raises(OutputAttributionError, match="'0.rate'"):
        warnings.simplefilter("ignore")  # the capture-gap warnings are not under test
        tl.validate(model, x, scope="forward")
