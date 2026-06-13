"""Backend registry and public ``backend=`` routing tests."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
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
    SerializationPolicy,
    register_backend_spec,
    unregister_backend_spec,
)
from torchlens.validation import check_metadata_invariants
from torchlens.validation.invariants import MetadataInvariantError


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


def _fake_capture_trace(*args: Any, **kwargs: Any) -> tl.Trace:
    """Return a real trace relabeled as the fake backend.

    Parameters
    ----------
    *args, **kwargs:
        Public trace arguments.

    Returns
    -------
    tl.Trace
        Fake backend trace.
    """

    del args, kwargs
    trace = tl.trace(
        _TinyModel().eval(),
        torch.ones(1),
        layers_to_save="all",
        random_seed=1,
        backend="torch",
    )
    trace.backend = "fake"
    trace.module_identity_mode = "function_root"
    trace.param_source = "none"
    trace.model_class_name = "_FakeModel"
    trace.model_label = "_FakeModel"
    trace.model_class_qualname = "tests.test_backend_registry._FakeModel"
    trace.trace_label = "fake-backend-trace"
    for layer in trace.layer_list:
        layer.resolver_status = "metadata_only"
        layer.backend_address = f"fake:{layer.layer_label}"
    for layer in trace.layer_logs.values():
        layer.resolver_status = "metadata_only"
        layer.backend_address = f"fake:{layer.layer_label}"
    return trace


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
    """Run fake trace metadata validation.

    Parameters
    ----------
    *args, **kwargs:
        Trace validation arguments.

    Returns
    -------
    bool
        ``True`` when fake metadata invariants pass.
    """

    trace = args[0]
    validate_metadata = kwargs.get("validate_metadata", True)
    if validate_metadata:
        check_metadata_invariants(trace)
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
            serialization_policy=SerializationPolicy(
                payload_policy="metadata_only",
                body_format="audit_only",
                manifest_schema_versions=(2,),
                runtime_name="fake",
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


def test_public_backend_literal_branches_stay_in_registry_or_backends() -> None:
    """Public code has no new hard-coded backend literal branches."""

    project_root = Path(__file__).resolve().parents[1]
    allowed_dirs = {
        project_root / "torchlens" / "backends",
    }
    allowed_files = {
        project_root / "torchlens" / "_io" / "bundle.py",
        project_root / "torchlens" / "_io" / "tlspec.py",
    }
    backend_literals = {"torch", "mlx", "jax", "tinygrad", "fake"}
    offenders: list[str] = []

    for source_path in sorted((project_root / "torchlens").rglob("*.py")):
        if any(source_path.is_relative_to(allowed_dir) for allowed_dir in allowed_dirs):
            continue
        if source_path in allowed_files:
            continue
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        source = source_path.read_text(encoding="utf-8")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            expression = ast.get_source_segment(source, node) or ""
            if "backend" not in expression:
                continue
            compared_literals = {
                item.value
                for item in [node.left, *node.comparators]
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
            if compared_literals & backend_literals:
                relpath = source_path.relative_to(project_root)
                offenders.append(f"{relpath}:{node.lineno}: {expression}")

    assert offenders == []


def test_explicit_backend_mismatch_is_deterministic() -> None:
    """Explicit torch selection rejects non-torch models before capture."""

    with pytest.raises(BackendMismatchError, match="backend='torch' cannot handle"):
        tl.trace(object(), torch.ones(1), backend="torch")


def test_fake_backend_explicit_trace_and_validate() -> None:
    """Registered fake backend drives public trace and validation entries."""

    _register_fake_backend()
    try:
        result = tl.trace(_FakeModel(), object(), backend="fake")
        assert isinstance(result, tl.Trace)
        assert result.backend == "fake"
        assert result.module_identity_mode == "function_root"
        assert result.param_source == "none"
        assert result.validate_forward_pass([]) is True
        assert tl.validate(_FakeModel(), object(), scope="forward", backend="fake")
    finally:
        unregister_backend_spec("fake")


def test_fake_backend_trace_save_load_accessors_and_invariants(tmp_path: Path) -> None:
    """Fake backend trace round-trips metadata and exposes neutral accessors."""

    _register_fake_backend()
    try:
        trace = tl.trace(_FakeModel(), object(), backend="fake")
        path = tmp_path / "fake.tlspec"

        trace.save(path, level="audit")
        loaded = tl.load(path)

        assert isinstance(loaded, tl.Trace)
        assert loaded.backend == "fake"
        assert loaded.module_identity_mode == "function_root"
        assert loaded.param_source == "none"
        assert loaded[0].resolver_status == "metadata_only"
        assert loaded[0].backend_address.startswith("fake:")
        assert check_metadata_invariants(loaded) is True
    finally:
        unregister_backend_spec("fake")


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda trace: setattr(trace, "module_identity_mode", "torch_module"), "module_identity"),
        (lambda trace: setattr(trace, "has_backward_pass", True), "has_backward_pass"),
        (lambda trace: trace.grad_fn_logs.__setitem__(1, object()), "grad_fn_logs"),
        (lambda trace: setattr(trace[0], "resolver_status", "lost"), "resolver_status"),
        (lambda trace: trace.output_layers.clear(), "output layer"),
    ],
)
def test_fake_backend_invariant_corruptions_fail(
    mutate: Any,
    match: str,
) -> None:
    """Non-torch invariant gates reject representative corruptions."""

    _register_fake_backend()
    try:
        trace = tl.trace(_FakeModel(), object(), backend="fake")
        mutate(trace)

        with pytest.raises(MetadataInvariantError, match=match):
            check_metadata_invariants(trace)
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
