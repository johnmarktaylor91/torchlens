"""Backend trace-option honesty checks.

Two generators define the contract: every registered backend crossed with
every public spine trace option (declared options pass through unchanged,
undeclared ones refuse typed), and every preview option policy crossed with
every option the policy may refuse (the shared helper refuses exactly the
options the policy names). Each generator has one property test that walks
the whole product and names every cell that breaks the rule, plus a sampled
parametrization that runs one cell per axis value and both verdicts per row
and per column, so a failure still points at a single cell id.
"""

from __future__ import annotations

from typing import Any

import pytest

import torchlens as tl
from torchlens._capture.prepare.dispatch import _filter_trace_kwargs_for_backend
from torchlens.backends import BackendSpec
from torchlens.backends._backend_protocol import REFUSABLE_CAPABILITY_FLAGS
from torchlens.backends._options import (
    JAX_PREVIEW_TRACE_OPTION_POLICY,
    MLX_PREVIEW_TRACE_OPTION_POLICY,
    PADDLE_PREVIEW_TRACE_OPTION_POLICY,
    TF_PREVIEW_TRACE_OPTION_POLICY,
    TINYGRAD_PREVIEW_TRACE_OPTION_POLICY,
    TRACE_OPTION_CAPABILITY_GATES,
    PreviewTraceOptionPolicy,
    reject_unsupported_trace_options,
)
from torchlens.backends.registry import PUBLIC_OPTION_SPINE_TRACE_OPTIONS
from torchlens.errors import BackendUnsupportedError

pytestmark = pytest.mark.backend_parity

_PUBLIC_SPINE_OPTION_VALUES: dict[str, Any] = {
    "jax_control_flow": "unroll",
    "jax_max_control_flow_unroll": 4,
    "module_identity_mode": "object_module",
    "payload_policy": "full",
    "capture_container_structure": True,
    "inference_only": True,
    "chunk_size": 2,
    "chunk_paths": [("args", 0)],
    "facets": (lambda record: {},),
}

_PREVIEW_POLICIES: tuple[PreviewTraceOptionPolicy, ...] = (
    JAX_PREVIEW_TRACE_OPTION_POLICY,
    MLX_PREVIEW_TRACE_OPTION_POLICY,
    PADDLE_PREVIEW_TRACE_OPTION_POLICY,
    TF_PREVIEW_TRACE_OPTION_POLICY,
    TINYGRAD_PREVIEW_TRACE_OPTION_POLICY,
)

_PREVIEW_DEFAULT_OPTIONS: dict[str, Any] = {
    "layers_to_save": "all",
    "input_kwargs": {},
    "activation_transform": None,
    "detach_saved_activations": False,
    "save_grads": None,
    "save_args": False,
    "save_source_context": False,
    "save_rng": False,
    "backward_ready": False,
    "transform": None,
    "output_device": "same",
    "layer_visualizers": None,
    "save_visualizations": False,
    "save_raw_activations": True,
    "lookback": 0,
    "lookback_payload_policy": "metadata_only",
}

_PREVIEW_UNSUPPORTED_VALUES: tuple[tuple[str, Any], ...] = (
    ("layers_to_save", ["relu_1_1"]),
    ("input_kwargs", {"bias": object()}),
    ("activation_transform", lambda value: value),
    ("detach_saved_activations", True),
    ("save_grads", True),
    ("save_args", True),
    ("save_source_context", True),
    ("save_rng", True),
    ("backward_ready", True),
    ("transform", lambda value: value),
    ("output_device", "cpu"),
    ("layer_visualizers", {"relu": object()}),
    ("save_visualizations", True),
    ("save_raw_activations", False),
    ("lookback", 1),
    ("lookback_payload_policy", "detached_raw"),
)


_SPECS: tuple[BackendSpec, ...] = tuple(tl.backends.specs())


def _spine_declared(spec: BackendSpec, option_name: str) -> bool:
    """Whether ``spec`` declares the public spine option ``option_name``."""

    return option_name in spec.capabilities.trace_options


def _preview_rejects(policy: PreviewTraceOptionPolicy, option_name: str) -> bool:
    """Whether ``policy`` names ``option_name`` as one it refuses."""

    return (
        (option_name == "input_kwargs" and policy.input_kwargs_message is not None)
        or (option_name == "layers_to_save" and policy.full_save_message is not None)
        or option_name in (policy.rejected_truthy_messages or {})
        or (option_name == "output_device" and policy.output_device_message is not None)
        or (
            option_name == "save_raw_activations"
            and policy.save_raw_activations_false_message is not None
        )
        or (
            option_name in {"lookback", "lookback_payload_policy"}
            and policy.save_window_message is not None
        )
    )


#: Options whose refusal names another option: the save-window check refuses
#: ``lookback`` and ``lookback_payload_policy`` together and names ``lookback``.
_REFUSAL_NAMES: dict[str, str] = {"lookback_payload_policy": "lookback"}


def _expected_refusal_code(option_name: str) -> str:
    """The contracted code, read from the two data tables rather than the library helper.

    An option a capability flag gates refuses with that flag's code; any other
    option refuses with ``trace_option_unsupported``.
    """

    gate = TRACE_OPTION_CAPABILITY_GATES.get(option_name)
    return "trace_option_unsupported" if gate is None else REFUSABLE_CAPABILITY_FLAGS[gate]


def _typed_refusal_problem(
    error: BackendUnsupportedError, option_name: str, backend: str
) -> str | None:
    """Why ``error`` is not the contracted typed refusal for one option, else ``None``."""

    fields = error.fields
    option_name = _REFUSAL_NAMES.get(option_name, option_name)
    named = fields.get("option") == option_name or option_name in (fields.get("options") or ())
    expected = _expected_refusal_code(option_name)
    if fields.get("code") != expected or not named:
        return f"refused with {fields!r}, expected code {expected!r}"
    if fields.get("backend", backend) != backend:
        return f"refusal names backend {fields.get('backend')!r}"
    return None


def _spine_violation(spec: BackendSpec, option_name: str) -> str | None:
    """How one (backend, spine option) cell breaks the honesty rule, else ``None``."""

    value = _PUBLIC_SPINE_OPTION_VALUES[option_name]
    kwargs = {option_name: value}
    try:
        _filter_trace_kwargs_for_backend(kwargs, spec)
    except BackendUnsupportedError as error:
        if _spine_declared(spec, option_name):
            return f"declared option refused: {error.fields!r}"
        return _typed_refusal_problem(error, option_name, str(spec.name))
    if not _spine_declared(spec, option_name):
        return "undeclared option admitted"
    if kwargs != {option_name: value}:
        return f"declared option rewritten to {kwargs!r}"
    return None


def _preview_violation(
    policy: PreviewTraceOptionPolicy, option_name: str, value: Any
) -> str | None:
    """How one (preview policy, option value) cell breaks the shared-helper rule, else ``None``."""

    options = dict(_PREVIEW_DEFAULT_OPTIONS)
    options[option_name] = value
    try:
        reject_unsupported_trace_options(options, policy)
    except BackendUnsupportedError as error:
        if not _preview_rejects(policy, option_name):
            return f"option the policy admits was refused: {error.fields!r}"
        return _typed_refusal_problem(error, option_name, policy.backend_name)
    if _preview_rejects(policy, option_name):
        return "option the policy names was admitted"
    return None


def _guarded(check: Any, *cell: Any) -> str | None:
    """Run one cell's check; an unexpected exception is that cell's problem, not the sweep's."""

    try:
        return check(*cell)
    except Exception as error:  # the property names the cell that raised
        return f"raised {type(error).__name__}: {error}"


def _covering_cells(
    rows: tuple[Any, ...], columns: tuple[Any, ...], verdict: Any
) -> list[tuple[Any, Any]]:
    """One cell per row and per column, then both verdicts for every row and column.

    The diagonal ``rows[i % R] x columns[i % C]`` visits every axis value once;
    a row (or column) whose cells so far all share one verdict gains the first
    partner with the other verdict, so each axis value is exercised both ways
    wherever the product has both.
    """

    size = max(len(rows), len(columns))
    cells = [(rows[i % len(rows)], columns[i % len(columns)]) for i in range(size)]
    for row in rows:
        for wanted in (True, False):
            if any(r is row and verdict(r, c) == wanted for r, c in cells):
                continue
            extra = next((c for c in columns if verdict(row, c) == wanted), None)
            if extra is not None:
                cells.append((row, extra))
    for column in columns:
        for wanted in (True, False):
            if any(c is column and verdict(r, c) == wanted for r, c in cells):
                continue
            partner = next((r for r in rows if verdict(r, column) == wanted), None)
            if partner is not None:
                cells.append((partner, column))
    return cells


_SPINE_CELLS = _covering_cells(_SPECS, tuple(PUBLIC_OPTION_SPINE_TRACE_OPTIONS), _spine_declared)
_PREVIEW_CELLS = _covering_cells(
    _PREVIEW_POLICIES,
    _PREVIEW_UNSUPPORTED_VALUES,
    lambda policy, pair: _preview_rejects(policy, pair[0]),
)


def test_every_backend_spine_option_cell_is_declared_or_rejected_typed() -> None:
    """Property: across the whole backend x spine-option product, no cell breaks the rule."""

    broken = {
        f"{spec.name}-{option_name}": problem
        for spec in _SPECS
        for option_name in PUBLIC_OPTION_SPINE_TRACE_OPTIONS
        if (problem := _guarded(_spine_violation, spec, option_name)) is not None
    }
    assert not broken, broken


def test_every_preview_policy_option_cell_is_refused_exactly_when_named() -> None:
    """Property: across the whole preview-policy x option product, no cell breaks the rule."""

    broken = {
        f"{policy.backend_name}-{option_name}": problem
        for policy in _PREVIEW_POLICIES
        for option_name, value in _PREVIEW_UNSUPPORTED_VALUES
        if (problem := _guarded(_preview_violation, policy, option_name, value)) is not None
    }
    assert not broken, broken


@pytest.mark.parametrize(
    ("spec", "option_name"),
    _SPINE_CELLS,
    ids=[f"{spec.name}-{option_name}" for spec, option_name in _SPINE_CELLS],
)
def test_registered_backend_public_trace_options_are_declared_or_rejected(
    spec: BackendSpec,
    option_name: str,
) -> None:
    """Sampled cell: a backend declares a public spine option or refuses it typed."""

    assert _spine_violation(spec, option_name) is None


@pytest.mark.parametrize(
    ("policy", "option_value"),
    _PREVIEW_CELLS,
    ids=[f"{policy.backend_name}-{pair[0]}" for policy, pair in _PREVIEW_CELLS],
)
def test_preview_backend_options_are_centrally_rejected(
    policy: PreviewTraceOptionPolicy,
    option_value: tuple[str, Any],
) -> None:
    """Sampled cell: the shared policy helper refuses exactly the options the policy names."""

    option_name, value = option_value
    assert _preview_violation(policy, option_name, value) is None
