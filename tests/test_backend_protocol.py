"""The one backend protocol: registry wiring, closed registries, shells.

Runs in a torch-only environment: the backend shells must construct and
answer ``runtime_available`` without their framework installed.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Any

import pytest

import torchlens as tl
from tests.conformance._protocol_cases import C0_PROTOCOL_CASES, run_c0_protocol_cases
from torchlens.backends import (
    REFUSAL_CODES,
    VALIDATION_REASONS,
    BackendCapabilities,
    BackendProtocol,
    BackendSpec,
    FrameworkRequirement,
)
from torchlens.backends._backend_protocol import (
    BACKEND_PROTOCOL_MEMBERS,
    SpecBackedProtocol,
    protocol_problems,
)
from torchlens.backends.registry import (
    BackendRegistryError,
    SerializationPolicy,
    get_backend_spec,
    registered_backend_specs,
    unregister_backend_spec,
)

ALL_BACKENDS = ("torch", "tf", "jax", "mlx", "paddle", "tinygrad")
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def test_refusal_codes_and_reasons_are_snake_case_and_disjoint() -> None:
    """Closed registries hold lowercase snake_case codes and do not overlap."""

    assert all(_SNAKE.match(code) for code in REFUSAL_CODES)
    assert all(_SNAKE.match(code) for code in VALIDATION_REASONS)
    assert not REFUSAL_CODES & VALIDATION_REASONS


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_protocols/paddle.py.
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_every_registered_backend_supplies_a_protocol_object(backend: str) -> None:
    """All six specs register a protocol factory whose object satisfies the protocol."""

    spec = get_backend_spec(backend)
    assert spec.protocol is not None
    candidate = spec.protocol()
    assert isinstance(candidate, BackendProtocol)
    assert protocol_problems(candidate, expected_name=backend) == ()
    assert candidate.name == backend
    assert candidate.materialization in {"eager", "lazy"}
    assert candidate.refusal_codes() <= REFUSAL_CODES
    assert candidate.capabilities() is spec.capabilities
    assert tuple(candidate.trace_options()) == tuple(spec.capabilities.trace_options)


def test_no_backend_carries_a_maturity_label() -> None:
    """The ``preview``/``stable`` label is retired: no spec field, no protocol member."""

    assert "maturity" not in {f.name for f in dataclasses.fields(BackendSpec)}
    assert "maturity" not in BACKEND_PROTOCOL_MEMBERS
    for backend in ALL_BACKENDS:
        assert not hasattr(get_backend_spec(backend), "maturity")
        assert not hasattr(get_backend_spec(backend).protocol(), "maturity")


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_framework_requirement_is_internally_consistent(backend: str) -> None:
    """Range, tested version and pin agree; the extra names the pyproject extra."""

    requirement = get_backend_spec(backend).protocol().framework
    assert isinstance(requirement, FrameworkRequirement)
    assert requirement._consistency_problems() == ()
    assert requirement.distributions
    if backend == "torch":
        assert requirement.extra == ""
    else:
        assert requirement.extra == backend


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_backend_protocol.py.
def test_framework_requirement_range_logic() -> None:
    """``admits`` honours the floor, the ceiling and exclusions; problems are named."""

    requirement = FrameworkRequirement(
        distributions=("x",),
        minimum="0.26",
        below="0.27",
        tested_through="0.26.3",
        pinned="0.26.3",
        extra="x",
        excluded=("0.26.5",),
    )
    assert requirement._admits("0.26.0")
    assert requirement._admits("0.26.9")
    assert not requirement._admits("0.27")
    assert not requirement._admits("0.25.9")
    assert not requirement._admits("0.26.5")
    assert not requirement._admits("not-a-version")
    bad = FrameworkRequirement(("x",), "1.0", "2.0", "2.5", "0.9", "x")
    problems = bad._consistency_problems()
    assert any("tested_through" in problem for problem in problems)
    assert any("pinned" in problem for problem in problems)
    open_ended = FrameworkRequirement(("x",), "1.0", None, "", None, "x")
    assert any("upper end" in problem for problem in open_ended._consistency_problems())


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_runtime_available_never_raises_and_answers_with_a_code(backend: str) -> None:
    """Absent or out-of-range frameworks answer with a registered code, never an exception."""

    answer = get_backend_spec(backend).protocol().runtime_available()
    assert answer is None or answer in {"runtime_missing", "runtime_version_unsupported"}
    if backend == "torch":
        assert answer is None


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_protocols/torch.py.
def test_torch_protocol_host_copy_keeps_scalar_shape_and_widens_bfloat16() -> None:
    """``host_copy`` returns a detached NumPy copy with the shape kept (F3 guard)."""

    import numpy as np
    import torch

    candidate = get_backend_spec("torch").protocol()
    scalar = candidate.host_copy(torch.tensor(3.0))
    assert isinstance(scalar, np.ndarray) and scalar.shape == ()
    wide = candidate.host_copy(torch.ones(2, dtype=torch.bfloat16))
    assert wide.dtype == np.float32 and wide.shape == (2,)
    assert candidate.is_tensor(torch.ones(1)) and not candidate.is_tensor(np.ones(1))
    candidate.realize_once([torch.ones(1)])  # eager: a no-op


def _make_spec(name: str, *, protocol: Any = None) -> BackendSpec:
    """A throwaway spec with no framework entries, for registration checks."""

    def _no(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("never called")

    return BackendSpec(
        name=name,
        can_handle=lambda model, args, kwargs: False,
        capture_trace=_no,
        validate_entry=_no,
        validate_trace=_no,
        capabilities=BackendCapabilities(
            backward_capture=False,
            has_validation_replay=False,
            fastlog=False,
            can_intervene=False,
            rng_replay=False,
            payload_materialization=False,
            has_streaming=False,
        ),
        serialization_policy=SerializationPolicy(runtime_name=name),
        priority=999,
        protocol=protocol,
    )


class _Incomplete:
    name = "incomplete"


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/_backend_protocol.py.
def test_registration_refuses_an_incomplete_protocol_object() -> None:
    """A protocol factory whose object misses members fails at registration."""

    spec = _make_spec("incomplete", protocol=lambda: _Incomplete())
    with pytest.raises(TypeError, match="missing member"):
        tl.backends.register(spec)
    assert all(s.name != "incomplete" for s in registered_backend_specs())


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/registry.py.
def test_registration_refuses_a_duplicate_name_typed() -> None:
    """Registering over a built-in without ``replace=True`` refuses typed."""

    with pytest.raises(BackendRegistryError) as excinfo:
        tl.backends.register(_make_spec("torch"))
    assert excinfo.value.fields["code"] == "backend_already_registered"
    assert get_backend_spec("torch").priority != 999


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/registry.py.
def test_registration_refuses_a_protocol_whose_name_disagrees_with_the_spec() -> None:
    """The protocol object reports the name its spec registers."""

    shell = SpecBackedProtocol(
        name="someone_else",
        framework=FrameworkRequirement(("x",), "1.0", None, "1.0", None, "x"),
        materialization="eager",
        trace_options=(),
        refusal_codes=(),
    )
    with pytest.raises(TypeError, match="differs from the spec name"):
        tl.backends.register(_make_spec("disagree", protocol=lambda: shell))
    assert all(s.name != "disagree" for s in registered_backend_specs())


# Kept by shrink-r4 (per-id subset rule): sole coverage of backends/registry.py.
def test_registration_accepts_a_complete_shell_and_the_c0_cells_pass() -> None:
    """A spec-backed shell registers; its four protocol-level C0 cells read passed."""

    shell = SpecBackedProtocol(
        name="shelled",
        framework=FrameworkRequirement(("torchlens",), "0.0", None, "0.0", None, "shelled"),
        materialization="eager",
        trace_options=(),
        refusal_codes=("trace_option_unsupported",),
    )
    tl.backends.register(_make_spec("shelled", protocol=lambda: shell))
    try:
        rows = run_c0_protocol_cases("shelled")
        assert tuple(row.case_id for row in rows) == C0_PROTOCOL_CASES
        assert all(row.outcome == "passed" for row in rows), [(r.case_id, r.detail) for r in rows]
    finally:
        unregister_backend_spec("shelled")


def test_protocol_members_list_matches_the_protocol_class() -> None:
    """The runtime member list and the Protocol class name the same surface."""

    declared = set(BackendProtocol.__annotations__) | {
        name
        for name, value in vars(BackendProtocol).items()
        if callable(value) and not name.startswith("_")
    }
    assert declared == set(BACKEND_PROTOCOL_MEMBERS)
