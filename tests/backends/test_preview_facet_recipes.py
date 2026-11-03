"""``trace(facets=...)`` attaches facet recipes on every preview backend (RC-266).

Recipes are backend-neutral, so each preview backend snapshots them onto
``trace.facet_functions`` through the shared helper instead of refusing. The
models come from the conformance rosters each backend already answers for.
"""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
from tests.conformance._preview_adapters import (
    PREVIEW_FRAMEWORKS,
    _roster_jax,
    _roster_mlx,
    _roster_tf,
    _roster_tinygrad,
    framework_unavailable_reason,
)
from torchlens.errors import BackendUnsupportedError

_ROSTERS = {
    "jax": _roster_jax,
    "tf": _roster_tf,
    "mlx": _roster_mlx,
    "tinygrad": _roster_tinygrad,
}

_BACKENDS = (
    pytest.param("jax", marks=pytest.mark.backend_jax),
    pytest.param("tf", marks=pytest.mark.tf_backend),
    pytest.param("mlx", marks=pytest.mark.backend_mlx),
    pytest.param("tinygrad", marks=pytest.mark.backend_tinygrad),
)


def _recipe(record: Any) -> dict[str, Any]:
    """Return one facet from any operation record."""

    return {"is_relu": "relu" in str(getattr(record, "label", ""))}


def _first_roster_model(backend: str) -> tuple[Any, Any]:
    reason = framework_unavailable_reason(PREVIEW_FRAMEWORKS[backend])
    if reason is not None:
        pytest.skip(reason)
    return _ROSTERS[backend]()[0].builder()


@pytest.mark.parametrize("backend", _BACKENDS)
def test_preview_backend_attaches_facet_recipes(backend: str) -> None:
    model, example_input = _first_roster_model(backend)
    trace = tl.trace(model, example_input, backend=backend, facets=[_recipe])
    snapshot = trace.facet_functions
    assert snapshot is not None
    assert any("_recipe" in entry.public.name for entry in snapshot.functions)
    # A single callable is admitted too, as on torch.
    single = tl.trace(model, example_input, backend=backend, facets=_recipe)
    assert single.facet_functions is not None


@pytest.mark.parametrize("backend", _BACKENDS)
def test_preview_backend_refuses_non_callable_recipes_typed(backend: str) -> None:
    model, example_input = _first_roster_model(backend)
    with pytest.raises(BackendUnsupportedError, match="recipe callables") as info:
        tl.trace(model, example_input, backend=backend, facets=["not-callable"])
    assert info.value.code == "trace_option_unsupported"
    assert info.value.option == "facets"
