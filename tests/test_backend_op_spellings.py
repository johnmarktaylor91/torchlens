"""Conformance of the per-backend op spelling lists (``backends/<b>/_spellings.py``).

Every name a backend wraps or records is listed (mapped to a canonical torch
spelling) or marked ``UNMAPPED`` with a family; every canonical spelling is a
name torch's wrappers log. The lists only label ops and rank structural
candidates; nothing here pairs ops.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterable
from types import ModuleType
from typing import Any

import pytest
import torch.nn.functional as functional

from tests.conformance._rosters import (
    jax as jax_roster,
    mlx as mlx_roster,
    paddle as paddle_roster,
    tf as tf_roster,
)
from tests.conformance._rosters._tinygrad_inventories import _PYTHON_INVENTORIES
from torchlens._capture.arg_positions import _normalize_func_name
from torchlens._constants import get_orig_torch_funcs
from torchlens.backends import _op_spelling
from torchlens.backends._op_spelling import (
    CANONICAL_FAMILY,
    FAMILIES,
    SPELLING_BACKENDS,
    SYNTHETIC_CANONICAL,
    Spelling,
    op_spelling,
)
from torchlens.backends.mlx import wrappers as mlx_wrappers
from torchlens.backends.paddle import wrappers as paddle_wrappers
from torchlens.backends.paddle.conv_layers import PADDLE_CONV_LAYER_OPS

#: The activation functions ``wrappers._wrap_installed`` names inline under ``mlx.nn``.
_MLX_NN_INLINE_ACTIVATIONS = ("relu", "gelu", "sigmoid", "tanh", "softmax", "silu")

#: Canonical spellings a supported torch may lack (``F.rms_norm`` arrived in torch 2.4).
_TORCH_VERSION_GATED = frozenset({"rmsnorm"})


def _table(backend: str) -> ModuleType:
    """Import one backend's spelling list module."""

    return importlib.import_module(f"torchlens.backends.{backend}._spellings")


def _listed(backend: str) -> set[str]:
    """Every name ``backend``'s list names, mapped or unmapped."""

    module = _table(backend)
    return set(module.SPELLINGS) | set(module.UNMAPPED)


def _unlisted(backend: str, names: set[str]) -> list[str]:
    """The names ``op_spelling`` answers for ``backend`` only by fallback."""

    return sorted(name for name in names if not op_spelling(backend, name).listed)


@pytest.mark.parametrize("backend", SPELLING_BACKENDS)
def test_list_is_well_formed(backend: str) -> None:
    module = _table(backend)
    assert not set(module.SPELLINGS) & set(module.UNMAPPED), "a name is mapped and unmapped"
    assert set(module.SPELLINGS.values()) <= set(CANONICAL_FAMILY), "canonical spelling unknown"
    assert set(module.UNMAPPED.values()) <= FAMILIES, "unmapped family unknown"


def test_every_family_is_a_known_family() -> None:
    assert set(CANONICAL_FAMILY.values()) <= FAMILIES
    assert set(CANONICAL_FAMILY) >= SYNTHETIC_CANONICAL


def test_mlx_static_inventory_is_listed() -> None:
    inventory = (
        set(mlx_wrappers.MLX_CORE_FUNCTION_OPS)
        | set(mlx_wrappers.MLX_FAST_FUNCTION_OPS)
        | set(mlx_wrappers.MLX_ARRAY_DUNDER_OPS.values())
        | set(mlx_wrappers.MLX_ARRAY_PROPERTY_OPS.values())
        | set(mlx_wrappers.MLX_ARRAY_METHOD_OPS)
        | set(mlx_wrappers.MLX_LAYER_CLASS_OPS.values())
        | set(_MLX_NN_INLINE_ACTIVATIONS)
    )
    assert inventory - _listed("mlx") == set()


def test_paddle_static_inventory_is_listed() -> None:
    inventory = (
        {paddle_wrappers._op_name("paddle", name) for name in paddle_wrappers._TOP_LEVEL_CORE_OPS}
        | {
            paddle_wrappers._op_name("paddle.nn.functional", name)
            for name in paddle_wrappers._FUNCTIONAL_CORE_OPS
        }
        | {
            paddle_wrappers._op_name("paddle.Tensor", name)
            for name in paddle_wrappers._TENSOR_CORE_METHODS
        }
        | set(PADDLE_CONV_LAYER_OPS.values())
    )
    assert inventory - _listed("paddle") == set()


@pytest.mark.parametrize(
    ("backend", "rows"),
    [
        ("jax", jax_roster.roster),
        ("tf", tf_roster.roster),
        ("mlx", mlx_roster.roster),
        ("paddle", paddle_roster.roster),
    ],
)
def test_roster_op_kinds_are_listed(backend: str, rows: Callable[[], Iterable[Any]]) -> None:
    recorded = {kind for row in rows() for kind in (row.expected_op_kinds or ())}
    assert recorded, f"{backend} roster declares no op kinds"
    assert _unlisted(backend, recorded) == []


def test_tinygrad_recorded_inventories_are_listed() -> None:
    # The tinygrad roster declares its op kinds per device; the PYTHON device's
    # recorded inventories are the ones checked against each model definition.
    recorded = {kind for kinds in _PYTHON_INVENTORIES.values() for kind in (kinds or ())}
    assert recorded, "tinygrad declares no recorded inventories"
    assert _unlisted("tinygrad", recorded) == []


def test_canonical_spellings_are_torch_spellings() -> None:
    torch_names = {
        _normalize_func_name(name) for _, name in get_orig_torch_funcs(include_torchvision=False)
    }
    missing = set(CANONICAL_FAMILY) - SYNTHETIC_CANONICAL - torch_names
    gated_absent = {
        name for name in _TORCH_VERSION_GATED if not _torch_has(name.replace("norm", "_norm"))
    }
    assert missing - gated_absent == set()


def test_torch_aliases_are_torch_spellings() -> None:
    torch_names = {
        _normalize_func_name(name) for _, name in get_orig_torch_funcs(include_torchvision=False)
    }
    assert set(_table("torch").SPELLINGS) - torch_names == set()


def _torch_has(functional_name: str) -> bool:
    """Whether ``torch.nn.functional`` defines ``functional_name`` in this torch."""

    return hasattr(functional, functional_name)


@pytest.mark.parametrize(
    ("backend", "func_name", "expected"),
    [
        ("torch", "layer_norm", Spelling("layernorm", "norm", True)),
        ("torch", "__add__", Spelling("add", "arith", True)),
        ("torch", "__radd__", Spelling("add", "arith", True)),
        ("torch", "__truediv__", Spelling("div", "arith", True)),
        ("torch", "linear", Spelling("linear", "matmul", True)),
        ("mlx", "layer_norm", Spelling("layernorm", "norm", True)),
        ("mlx", "layernorm", Spelling("layernorm", "norm", True)),
        ("mlx", "subtract", Spelling("sub", "arith", True)),
        ("mlx", "rope", Spelling("rope", "other", True)),
        ("paddle", "functional.linear", Spelling("linear", "matmul", True)),
        ("paddle", "functional.layer_norm", Spelling("layernorm", "norm", True)),
        ("paddle", "tensor.__add__", Spelling("add", "arith", True)),
        ("paddle", "tensor.__rsub__", Spelling("rsub", "arith", True)),
        ("jax", "dot_general", Spelling("matmul", "matmul", True)),
        ("jax", "reduce_window_max", Spelling("reducewindowmax", "pool", True)),
        ("tf", "BiasAdd", Spelling("add", "arith", True)),
        ("tinygrad", "cmplt", Spelling("lt", "compare", True)),
        ("tinygrad", "reduce", Spelling("reduce", "reduce", True)),
        ("jax", "region:pjit", Spelling("region:pjit", "region", True)),
        ("tf", "input", Spelling("input", "source", True)),
    ],
)
def test_op_spelling_answers(backend: str, func_name: str, expected: Spelling) -> None:
    assert op_spelling(backend, func_name) == expected


@pytest.mark.parametrize(
    ("backend", "func_name", "canonical"),
    [
        ("torch", "_some_private_kernel", "someprivatekernel"),
        ("mlx", "made_up_op", "madeupop"),
        ("jax", "custom_primitive", "customprimitive"),
        ("not_a_backend", "Layer_Norm", "layernorm"),
    ],
)
def test_unlisted_names_fall_back_to_their_own_name(
    backend: str, func_name: str, canonical: str
) -> None:
    assert op_spelling(backend, func_name) == Spelling(canonical, "other", False)


def test_unlisted_mlx_name_never_borrows_a_torch_family() -> None:
    # A non-torch name that happens to equal a canonical spelling is not taken
    # as that op unless the backend's list says so.
    assert op_spelling("mlx", "bmm") == Spelling("bmm", "other", False)


def test_lookup_loads_only_the_backend_it_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_op_spelling, "_TABLES", {})
    op_spelling("jax", "add")
    assert set(_op_spelling._TABLES) == {"jax"}


@pytest.mark.backend_mlx
@pytest.mark.rare
def test_mlx_derived_activations_are_listed() -> None:
    pytest.importorskip("mlx")
    import mlx.nn as nn

    activations = mlx_wrappers.mlx_activations_module(nn)
    names = set(mlx_wrappers.mlx_activation_function_names(activations)) | {
        cls.__name__.lower() for cls in mlx_wrappers.mlx_activation_layer_classes(activations, nn)
    }
    assert names, "no MLX activations derived"
    assert names - _listed("mlx") == set()
