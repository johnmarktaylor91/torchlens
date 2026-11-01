"""The one backend protocol: registry wiring, closed registries, shells.

Runs in a torch-only environment: the preview shells must construct and
answer ``runtime_available`` without their framework installed.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

import torchlens as tl
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
    MATURITIES,
    REFUSABLE_CAPABILITY_FLAGS,
    SpecBackedProtocol,
    protocol_problems,
    refusal_code_for_capability,
)
from torchlens.backends.registry import (
    BackendRegistryError,
    SerializationPolicy,
    get_backend_spec,
    registered_backend_specs,
    unregister_backend_spec,
)
from torchlens.conformance import C0_PROTOCOL_CASES, run_conformance
from torchlens.conformance._protocol_cases import run_c0_protocol_cases

ALL_BACKENDS = ("torch", "tf", "jax", "mlx", "paddle", "tinygrad")
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def test_refusal_codes_and_reasons_are_snake_case_and_disjoint() -> None:
    """Closed registries hold lowercase snake_case codes and do not overlap."""

    assert all(_SNAKE.match(code) for code in REFUSAL_CODES)
    assert all(_SNAKE.match(code) for code in VALIDATION_REASONS)
    assert not REFUSAL_CODES & VALIDATION_REASONS


def test_every_capability_field_that_can_be_false_has_a_refusal_code() -> None:
    """Each boolean capability field maps to a registered ``*_unsupported`` code."""

    boolean_fields = {
        name
        for name, annotation in BackendCapabilities.__annotations__.items()
        if "bool" in str(annotation)
    }
    assert boolean_fields == set(REFUSABLE_CAPABILITY_FLAGS)
    for flag in boolean_fields:
        code = refusal_code_for_capability(flag)
        assert code in REFUSAL_CODES
        assert code.endswith("_unsupported")
    with pytest.raises(KeyError):
        refusal_code_for_capability("not_a_flag")


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_every_registered_backend_supplies_a_protocol_object(backend: str) -> None:
    """All six specs register a protocol factory whose object satisfies the protocol."""

    spec = get_backend_spec(backend)
    assert spec.protocol is not None
    candidate = spec.protocol()
    assert isinstance(candidate, BackendProtocol)
    assert protocol_problems(candidate, expected_name=backend) == ()
    assert candidate.name == backend
    assert candidate.maturity == spec.maturity
    assert candidate.maturity in MATURITIES
    assert candidate.materialization in {"eager", "lazy"}
    assert candidate.refusal_codes() <= REFUSAL_CODES
    assert candidate.capabilities() is spec.capabilities
    assert tuple(candidate.trace_options()) == tuple(spec.capabilities.trace_options)


def test_torch_is_the_stable_reference_and_previews_are_preview() -> None:
    """Maturity labels match the launch decision: torch stable, the rest preview."""

    assert get_backend_spec("torch").maturity == "stable"
    for backend in ALL_BACKENDS[1:]:
        assert get_backend_spec(backend).maturity == "preview"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_framework_requirement_is_internally_consistent(backend: str) -> None:
    """Range, tested version and pin agree; the extra names the pyproject extra."""

    requirement = get_backend_spec(backend).protocol().framework
    assert isinstance(requirement, FrameworkRequirement)
    assert requirement.consistency_problems() == ()
    assert requirement.distributions
    if backend == "torch":
        assert requirement.extra == ""
    else:
        assert requirement.extra == backend


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
    assert requirement.admits("0.26.0")
    assert requirement.admits("0.26.9")
    assert not requirement.admits("0.27")
    assert not requirement.admits("0.25.9")
    assert not requirement.admits("0.26.5")
    assert not requirement.admits("not-a-version")
    bad = FrameworkRequirement(("x",), "1.0", "2.0", "2.5", "0.9", "x")
    problems = bad.consistency_problems()
    assert any("tested_through" in problem for problem in problems)
    assert any("pinned" in problem for problem in problems)
    open_ended = FrameworkRequirement(("x",), "1.0", None, "", None, "x")
    assert any("upper end" in problem for problem in open_ended.consistency_problems())


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_runtime_available_never_raises_and_answers_with_a_code(backend: str) -> None:
    """Absent or out-of-range frameworks answer with a registered code, never an exception."""

    answer = get_backend_spec(backend).protocol().runtime_available()
    assert answer is None or answer in {"runtime_missing", "runtime_version_unsupported"}
    if backend == "torch":
        assert answer is None


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


def test_torch_protocol_reference_output_runs_pristine() -> None:
    """The reference forward matches a direct call and runs with wrappers removed."""

    import torch

    model = torch.nn.Linear(2, 1)
    x = torch.ones(1, 2)
    candidate = get_backend_spec("torch").protocol()
    trace = candidate.capture(model, x, None)
    assert trace.backend == "torch"
    reference = candidate.reference_output(model, x, None)
    torch.testing.assert_close(reference, model(x))


def _make_spec(name: str, *, protocol: Any = None, maturity: str = "preview") -> BackendSpec:
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
            interventions=False,
            rng_replay=False,
            payload_materialization=False,
            has_streaming=False,
        ),
        serialization_policy=SerializationPolicy(runtime_name=name),
        priority=999,
        protocol=protocol,
        maturity=maturity,
    )


class _Incomplete:
    name = "incomplete"
    maturity = "preview"


def test_registration_refuses_an_incomplete_protocol_object() -> None:
    """A protocol factory whose object misses members fails at registration."""

    spec = _make_spec("incomplete", protocol=lambda: _Incomplete())
    with pytest.raises(TypeError, match="missing member"):
        tl.backends.register(spec)
    assert all(s.name != "incomplete" for s in registered_backend_specs())


def test_registration_refuses_a_maturity_outside_the_closed_set() -> None:
    """``maturity`` is ``preview`` or ``stable``; anything else refuses typed."""

    with pytest.raises(BackendRegistryError) as excinfo:
        tl.backends.register(_make_spec("mislabelled", maturity="beta"))
    assert excinfo.value.fields["code"] == "backend_maturity_invalid"


def test_registration_refuses_a_protocol_whose_maturity_disagrees_with_the_spec() -> None:
    """The protocol object and the spec state one label."""

    shell = SpecBackedProtocol(
        name="disagree",
        maturity="stable",
        framework=FrameworkRequirement(("x",), "1.0", None, "1.0", None, "x"),
        materialization="eager",
        trace_options=(),
        refusal_codes=(),
    )
    with pytest.raises(TypeError, match="maturity"):
        tl.backends.register(_make_spec("disagree", protocol=lambda: shell, maturity="preview"))


def test_registration_accepts_a_complete_shell_and_the_c0_cells_pass() -> None:
    """A spec-backed shell registers; its four protocol-level C0 cells read passed."""

    shell = SpecBackedProtocol(
        name="shelled",
        maturity="preview",
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


def test_c0_protocol_cells_do_not_exist_for_unregistered_providers() -> None:
    """The fake plant substrate has no registered spec, so it defines none of these slots."""

    assert run_c0_protocol_cases("fake") == []
    assert run_c0_protocol_cases("no-such-backend") == []
    report = run_conformance("fake")
    assert not {row.case_id for row in report.results} & set(C0_PROTOCOL_CASES)


def test_torch_conformance_c0_includes_the_protocol_cells_and_passes() -> None:
    """The reference backend passes every C0 cell, protocol cells included."""

    report = run_conformance("torch")
    ids = [row.case_id for row in report.results]
    assert all(case_id in ids for case_id in C0_PROTOCOL_CASES)
    assert all(row.outcome == "passed" for row in report.results), [
        (row.case_id, row.detail) for row in report.results if row.outcome != "passed"
    ]


def test_protocol_members_list_matches_the_protocol_class() -> None:
    """The runtime member list and the Protocol class name the same surface."""

    declared = set(BackendProtocol.__annotations__) | {
        name
        for name, value in vars(BackendProtocol).items()
        if callable(value) and not name.startswith("_")
    }
    assert declared == set(BACKEND_PROTOCOL_MEMBERS)
