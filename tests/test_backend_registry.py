"""Backend registry and public ``backend=`` routing tests."""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends import (
    BackendAmbiguityError,
    BackendCapabilities,
    BackendMismatchError,
    BackendSpec,
    register_backend_spec,
    unregister_backend_spec,
)


class _TinyModel(nn.Module):
    """Small torch model for backend routing tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a simple torch operation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Doubled tensor.
        """

        return x * 2


class _FakeModel:
    """Marker model accepted by fake backend specs."""


def _fake_can_handle(
    model: object,
    input_args: object,
    input_kwargs: dict[Any, Any] | None,
) -> bool:
    """Return whether ``model`` is the fake marker model.

    Parameters
    ----------
    model:
        Candidate model.
    input_args:
        Positional inputs, unused.
    input_kwargs:
        Keyword inputs, unused.

    Returns
    -------
    bool
        ``True`` for fake marker models.
    """

    del input_args, input_kwargs
    return isinstance(model, _FakeModel)


def _fake_capture_trace(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return a visible fake capture result.

    Parameters
    ----------
    *args, **kwargs:
        Public trace arguments.

    Returns
    -------
    dict[str, Any]
        Fake capture result.
    """

    return {"backend": "fake", "args": args, "kwargs": kwargs}


def _fake_validate_entry(*args: Any, **kwargs: Any) -> bool:
    """Return a visible fake validation result.

    Parameters
    ----------
    *args, **kwargs:
        Public validation arguments.

    Returns
    -------
    bool
        Always ``True``.
    """

    del args, kwargs
    return True


def _fake_validate_trace(*args: Any, **kwargs: Any) -> bool:
    """Return a visible fake trace-validation result.

    Parameters
    ----------
    *args, **kwargs:
        Trace validation arguments.

    Returns
    -------
    bool
        Always ``True``.
    """

    del args, kwargs
    return True


def _register_fake_backend(name: str = "fake", *, priority: int = 50) -> None:
    """Register a fake backend spec for tests.

    Parameters
    ----------
    name:
        Backend name.
    priority:
        Auto-resolution priority.

    Returns
    -------
    None
        The fake spec is registered.
    """

    register_backend_spec(
        BackendSpec(
            name=name,
            can_handle=_fake_can_handle,
            capture_trace=_fake_capture_trace,
            validate_entry=_fake_validate_entry,
            validate_trace=_fake_validate_trace,
            capabilities=BackendCapabilities(
                backward_capture=False,
                validation_replay=True,
                fastlog=False,
                interventions=False,
                rng_replay=False,
                payload_materialization=False,
                streaming=False,
                module_identity_modes=("function_root",),
                save_levels=("audit",),
            ),
            priority=priority,
        ),
    )


def test_explicit_torch_backend_matches_legacy_trace() -> None:
    """Explicit ``backend='torch'`` keeps torch capture reachable."""

    model = _TinyModel()
    x = torch.ones(1)
    legacy = tl.trace(model, x, layers_to_save="all", random_seed=1)
    explicit = tl.trace(model, x, layers_to_save="all", random_seed=1, backend="torch")
    assert explicit.backend == legacy.backend == "torch"
    assert explicit.layer_labels == legacy.layer_labels


def test_public_trace_dispatches_through_backend_spec() -> None:
    """Public ``trace`` dispatch stays owned by the backend spec."""

    source = inspect.getsource(tl.trace)
    assert "capture_trace(**public_trace_kwargs)" in source
    assert "resolved_spec.name" not in source


def test_explicit_backend_mismatch_is_deterministic() -> None:
    """Explicit torch selection rejects non-torch models before capture."""

    with pytest.raises(BackendMismatchError, match="backend='torch' cannot handle"):
        tl.trace(object(), torch.ones(1), backend="torch")


def test_fake_backend_explicit_trace_and_validate() -> None:
    """Registered fake backend drives public trace and validation entries."""

    _register_fake_backend()
    try:
        result = tl.trace(_FakeModel(), object(), backend="fake")
        assert result["backend"] == "fake"
        assert tl.validate(_FakeModel(), object(), scope="forward", backend="fake")
    finally:
        unregister_backend_spec("fake")


def test_backend_none_ambiguity_is_deterministic() -> None:
    """Equal-priority detector collisions fail with a canonical error."""

    _register_fake_backend("fake_a", priority=99)
    _register_fake_backend("fake_b", priority=99)
    try:
        with pytest.raises(BackendAmbiguityError, match="fake_a, fake_b"):
            tl.trace(_FakeModel(), object())
    finally:
        unregister_backend_spec("fake_a")
        unregister_backend_spec("fake_b")
