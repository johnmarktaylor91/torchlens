"""Capability-table load-bearing conformance.

The registered ``BackendCapabilities`` table must be the single authority for
backend feature support, fail-closed in BOTH directions on EVERY backend:

- Flag ``False`` refuses the corresponding public surface typed — torch
  included, so an in-place flip of torch's table refuses ``intervene=``,
  ``random_seed=``, ``backward_ready=``, and ``tl.record()`` instead of
  silently running them.
- Flag ``True`` on a preview whose declarative policy rejects the option is a
  self-contradictory registration and refuses
  ``BackendCapabilityConformanceError`` whether or not an implementation
  factory is bound: a binding the backend's capture path never dispatches
  must not admit the option (it would be silently ignored — the sol probe).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends import (
    GATED_CAPABILITY_FLAGS,
    BackendCapabilities,
    BackendCapabilityConformanceError,
    BackendUnsupportedError,
    get_backend_spec,
    register_backend_spec,
    registered_backend_specs,
    require_capability_implementation,
)
from torchlens.backends._options import (
    JAX_EXTRA_KWARG_POLICY,
    JAX_PREVIEW_TRACE_OPTION_POLICY,
    MLX_EXTRA_KWARG_POLICY,
    MLX_PREVIEW_TRACE_OPTION_POLICY,
    PADDLE_EXTRA_KWARG_POLICY,
    PADDLE_PREVIEW_TRACE_OPTION_POLICY,
    TF_EXTRA_KWARG_POLICY,
    TF_PREVIEW_TRACE_OPTION_POLICY,
    TINYGRAD_EXTRA_KWARG_POLICY,
    TINYGRAD_PREVIEW_TRACE_OPTION_POLICY,
    TRACE_OPTION_CAPABILITY_GATES,
    reject_extra_trace_kwargs,
    reject_unsupported_trace_options,
)
from torchlens.data_classes._backend_capability_guards import raise_if_no_backward_capture

pytestmark = [pytest.mark.backend_parity, pytest.mark.smoke]

_TORCHLENS_ROOT = Path(tl.__file__).resolve().parent

_PREVIEW_NAMES = ("mlx", "jax", "tinygrad", "paddle", "tf")

EXPECTED_GATED_CAPABILITIES_MEMBERSHIP_UNUSED = frozenset({("tf", "interventions")})
"""Preview ``(name, gated-flag)`` pairs lifted with REAL dispatch plus
conformance coverage (parity-C: tf static-label interventions through the
writable wrap layer). Lifting a preview capability is a one-line diff here;
every pair not listed stays fail-closed in both directions."""

_EXTRA_POLICIES = {
    "mlx": MLX_EXTRA_KWARG_POLICY,
    "jax": JAX_EXTRA_KWARG_POLICY,
    "tinygrad": TINYGRAD_EXTRA_KWARG_POLICY,
    "paddle": PADDLE_EXTRA_KWARG_POLICY,
    "tf": TF_EXTRA_KWARG_POLICY,
}

_OPTION_POLICIES = {
    "mlx": MLX_PREVIEW_TRACE_OPTION_POLICY,
    "jax": JAX_PREVIEW_TRACE_OPTION_POLICY,
    "tinygrad": TINYGRAD_PREVIEW_TRACE_OPTION_POLICY,
    "paddle": PADDLE_PREVIEW_TRACE_OPTION_POLICY,
    "tf": TF_PREVIEW_TRACE_OPTION_POLICY,
}

_OPTION_POLICY_DEFAULTS: dict[str, Any] = {
    "layers_to_save": "all",
    "input_kwargs": {},
    "output_device": "same",
    "save_raw_activations": True,
    "lookback": 0,
    "lookback_payload_policy": "metadata_only",
}

#: Declared truth for every gated capability flag on every backend. A lift is
#: a ONE-LINE diff here (plus its real mechanism + conformance coverage);
#: anything not listed True must refuse its public surface typed.
EXPECTED_GATED_CAPABILITIES: dict[str, dict[str, bool]] = {
    "torch": {
        "backward_capture": True,
        "fastlog": True,
        "interventions": True,
        "rng_replay": True,
        "streaming": True,
        # Lifted 2026-08 (L7a wave 0): structure-only capture with the typed
        # chokepoint in torchlens.capture.structure_only
        # (tests/test_structure_only_entry.py); DOCUMENTED-UNSTABLE pending
        # naming-session/S2 ratification.
        "structure_only_capture": True,
    },
    "mlx": {
        "backward_capture": False,
        "fastlog": False,
        # Lifted 2026-08: static-label intervene=/halt= dispatch for real
        # (tests/test_mlx_interventions.py); value-dependent predicates and
        # recipes= keep typed refusals inside the dispatch path.
        "interventions": True,
        "rng_replay": False,
        "streaming": False,
        "structure_only_capture": False,
    },
    "jax": {
        "backward_capture": False,
        "fastlog": False,
        "interventions": False,
        "rng_replay": False,
        "streaming": False,
        "structure_only_capture": False,
    },
    "tinygrad": {
        "backward_capture": False,
        "fastlog": False,
        "interventions": False,
        "rng_replay": False,
        "streaming": False,
        "structure_only_capture": False,
    },
    "paddle": {
        "backward_capture": False,
        "fastlog": False,
        # Lifted 2026-08: live intervene=/halt= dispatch through the eager
        # capture wrapper, forward-only (tests/test_paddle_backend_interventions.py);
        # value-dependent predicates and recipes= keep typed refusals.
        "interventions": True,
        "rng_replay": False,
        "streaming": False,
        "structure_only_capture": False,
    },
    "tf": {
        "backward_capture": False,
        "fastlog": False,
        # Lifted 2026-08: static-label intervene= for eager entries via the
        # two-level writable wrap layer (tests/backends/test_tf_interventions.py);
        # halt= and value-dependent predicates keep typed refusals.
        "interventions": True,
        "rng_replay": False,
        "streaming": False,
        "structure_only_capture": False,
    },
}


def test_every_capability_flag_has_a_production_consumer() -> None:
    """No decorative flags: every field is read by production code or a gate map."""

    gated_fields = set(TRACE_OPTION_CAPABILITY_GATES.values())
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(_TORCHLENS_ROOT.rglob("*.py"))
    )
    for field in dataclasses.fields(BackendCapabilities):
        directly_read = f"capabilities.{field.name}" in source
        assert directly_read or field.name in gated_fields, (
            f"BackendCapabilities.{field.name} gates nothing in production code; "
            "a flag no code reads can silently lie."
        )


def _spec_with_flag(name: str, flag: str, *, implementation: bool) -> Any:
    """Return the registered spec with ``flag`` flipped True, optionally bound."""

    original = get_backend_spec(name)
    implementations = dict(original.capability_implementations or {})
    if implementation:
        implementations[flag] = lambda: object()
    else:
        implementations.pop(flag, None)
    return dataclasses.replace(
        original,
        capabilities=dataclasses.replace(original.capabilities, **{flag: True}),
        capability_implementations=implementations or None,
    )


@pytest.mark.parametrize("name", _PREVIEW_NAMES)
def test_extra_kwarg_gates_are_fail_closed_biconditional(name: str) -> None:
    """intervene/storage/streaming: False rejects; a True flip refuses typed
    with OR without a binding — the preview capture path never dispatches it,
    so a bound-but-unconsumed implementation must not admit the option."""

    spec = get_backend_spec(name)
    policy = _EXTRA_POLICIES[name]
    sentinel = object()
    for option, flag in (
        ("intervene", "interventions"),
        ("storage", "streaming"),
        ("streaming", "streaming"),
    ):
        if EXPECTED_GATED_CAPABILITIES[name][flag]:
            # Lifted capability: the flag is True with a real binding, the
            # capture path pops the option before extra-kwarg rejection (so the
            # policy tables no longer govern it), and real dispatch + flag-False
            # refusal are covered by the backend's own E2E suite.
            assert getattr(spec.capabilities, flag)
            assert require_capability_implementation(spec, flag) is not None
            assert option not in (policy.inert_values or {})
            continue
        assert not getattr(spec.capabilities, flag)
        with pytest.raises(BackendUnsupportedError):
            reject_extra_trace_kwargs({option: sentinel}, policy, spec=spec)
        bare_flip = _spec_with_flag(name, flag, implementation=False)
        with pytest.raises(BackendCapabilityConformanceError):
            reject_extra_trace_kwargs({option: sentinel}, policy, spec=bare_flip)
        implemented = _spec_with_flag(name, flag, implementation=True)
        with pytest.raises(BackendCapabilityConformanceError, match="never\\s+dispatches"):
            reject_extra_trace_kwargs({option: sentinel}, policy, spec=implemented)


@pytest.mark.parametrize("name", _PREVIEW_NAMES)
def test_option_policy_gates_are_fail_closed_biconditional(name: str) -> None:
    """save_grads/backward_ready/save_rng_states refuse in every flag state:
    False via the policy message, True via the undispatched-binding refusal."""

    spec = get_backend_spec(name)
    policy = _OPTION_POLICIES[name]
    cases = [("save_grads", "backward_capture"), ("backward_ready", "backward_capture")]
    if "save_rng_states" in (policy.rejected_truthy_messages or {}):
        cases.append(("save_rng_states", "rng_replay"))
    for option, flag in cases:
        options = dict(_OPTION_POLICY_DEFAULTS)
        options[option] = True
        assert not getattr(spec.capabilities, flag)
        with pytest.raises(BackendUnsupportedError):
            reject_unsupported_trace_options(options, policy, spec=spec)
        bare_flip = _spec_with_flag(name, flag, implementation=False)
        with pytest.raises(BackendCapabilityConformanceError):
            reject_unsupported_trace_options(options, policy, spec=bare_flip)
        implemented = _spec_with_flag(name, flag, implementation=True)
        with pytest.raises(BackendCapabilityConformanceError, match="never\\s+dispatches"):
            reject_unsupported_trace_options(options, policy, spec=implemented)


def test_registration_refuses_bare_capability_flips() -> None:
    """Re-registering a spec whose True gated flag has no binding refuses typed."""

    for name in _PREVIEW_NAMES:
        for flag in sorted(GATED_CAPABILITY_FLAGS):
            original_value = getattr(get_backend_spec(name).capabilities, flag)
            with pytest.raises(BackendCapabilityConformanceError):
                register_backend_spec(
                    _spec_with_flag(name, flag, implementation=False), replace=True
                )
            # The refused registration must leave the registered truth intact.
            assert getattr(get_backend_spec(name).capabilities, flag) == original_value


def test_in_place_capability_flip_refuses_end_to_end() -> None:
    """Review probe: mutating the frozen table in place must refuse typed at trace()."""

    spec = get_backend_spec("mlx")
    object.__setattr__(spec.capabilities, "streaming", True)
    try:
        with pytest.raises(BackendCapabilityConformanceError):
            require_capability_implementation(spec, "streaming")
    finally:
        object.__setattr__(spec.capabilities, "streaming", False)


def test_record_gate_reads_the_capability_table() -> None:
    """tl.record's torch-only gate is owned by capabilities.fastlog + its binding."""

    model = nn.Linear(2, 2)
    inputs = torch.randn(1, 2)
    for name in _PREVIEW_NAMES:
        assert not get_backend_spec(name).capabilities.fastlog
        with pytest.raises(BackendUnsupportedError, match="torch-only"):
            tl.record(model, inputs, backend=name)


def test_record_bare_fastlog_flip_never_runs_torch_recorder() -> None:
    """Review probe: fastlog=True flipped in place must refuse typed, never return a
    torch Recording for a non-torch backend."""

    model = nn.Linear(2, 2)
    inputs = torch.randn(1, 2)
    spec = get_backend_spec("tinygrad")
    object.__setattr__(spec.capabilities, "fastlog", True)
    try:
        with pytest.raises(BackendCapabilityConformanceError):
            tl.record(model, inputs, backend="tinygrad")
    finally:
        object.__setattr__(spec.capabilities, "fastlog", False)
        tl.release_model(model)


def test_record_foreign_fastlog_implementation_refuses() -> None:
    """A registered non-torch fastlog binding still refuses: record() only runs
    the torch Recorder and must not silently substitute it."""

    model = nn.Linear(2, 2)
    inputs = torch.randn(1, 2)
    original = get_backend_spec("tinygrad")
    opened = _spec_with_flag("tinygrad", "fastlog", implementation=True)
    register_backend_spec(opened, replace=True)
    try:
        with pytest.raises(BackendUnsupportedError, match="no non-torch dispatch path"):
            tl.record(model, inputs, backend="tinygrad")
    finally:
        register_backend_spec(original, replace=True)
        tl.release_model(model)


def test_torch_capability_bindings_resolve() -> None:
    """Torch's declared True flags all resolve to real implementing surfaces."""

    spec = get_backend_spec("torch")
    for flag in sorted(GATED_CAPABILITY_FLAGS):
        assert getattr(spec.capabilities, flag), flag
        assert require_capability_implementation(spec, flag) is not None


def test_runnable_producer_gate_reads_save_levels() -> None:
    """The runnable producer refusal keys on 'runnable' in save_levels."""

    from torchlens._io.runnable import build_sparse_run_descriptor
    from torchlens.runnable import RunnableErrorCode

    assert "runnable" in get_backend_spec("torch").capabilities.save_levels
    for name in _PREVIEW_NAMES:
        assert "runnable" not in get_backend_spec(name).capabilities.save_levels

    model = nn.Linear(2, 2)
    trace = tl.trace(model, torch.randn(1, 2))
    try:
        descriptor = build_sparse_run_descriptor(trace)
        torch_codes = {d.code for d in descriptor.preflight.diagnostics}
        assert RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY not in torch_codes

        trace.backend = "paddle"
        descriptor = build_sparse_run_descriptor(trace)
        preview_codes = {d.code for d in descriptor.preflight.diagnostics}
        assert RunnableErrorCode.UNSUPPORTED_BACKEND_REPLAY in preview_codes
    finally:
        trace.backend = "torch"
        tl.release_model(model)


class _StubTrace:
    def __init__(self, backend: str) -> None:
        self.backend = backend


@pytest.mark.parametrize("name", ("jax", "mlx", "tinygrad", "paddle", "tf"))
def test_backward_accessor_guard_redirects_to_derived_grads(name: str) -> None:
    """Derived-grads backends refuse with the derived-gradient redirect."""

    with pytest.raises(ValueError, match="derived_grads"):
        raise_if_no_backward_capture(_StubTrace(name), plural_subject="backward_passes")


def test_backward_accessor_guard_without_derived_surface_has_no_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend declaring no derived-gradient surface must not get the redirect.

    Every registered preview now declares ``intermediate_derived_grads=True``,
    so the no-surface branch is exercised through a flag-flipped spec: the
    guard must keep refusing typed without pointing at an accessor that does
    not exist.
    """

    from torchlens.data_classes import _backend_capability_guards as guards

    original = get_backend_spec("tf")
    flipped = dataclasses.replace(
        original,
        capabilities=dataclasses.replace(original.capabilities, intermediate_derived_grads=False),
    )
    monkeypatch.setattr(guards, "get_backend_spec", lambda _name: flipped)
    with pytest.raises(ValueError, match="declares no derived-gradient surface"):
        raise_if_no_backward_capture(_StubTrace("tf"), plural_subject="backward_passes")


def test_backward_accessor_guard_torch_passes_unknown_refuses() -> None:
    """torch passes; an unregistered backend refuses typed instead of
    restoring the silently-empty accessor behavior the guard removed."""

    raise_if_no_backward_capture(_StubTrace("torch"), plural_subject="backward_passes")
    with pytest.raises(ValueError, match="not a registered backend"):
        raise_if_no_backward_capture(_StubTrace("not-a-backend"), plural_subject="backward_passes")


def test_backward_accessors_raise_for_all_non_backward_backends() -> None:
    """paddle/tf join jax/mlx/tinygrad: no silent-empty backward accessors."""

    model = nn.Linear(2, 2)
    trace = tl.trace(model, torch.randn(1, 2))
    try:
        for name in _PREVIEW_NAMES:
            trace.backend = name
            with pytest.raises(ValueError):
                _ = trace.backward_passes
            with pytest.raises(ValueError):
                _ = trace.saved_grad_ops
            with pytest.raises(ValueError):
                _ = trace.layer_list[0].grads
        trace.backend = "torch"
        _ = trace.backward_passes
        _ = trace.saved_grad_ops
        _ = trace.layer_list[0].grads
    finally:
        trace.backend = "torch"
        tl.release_model(model)


@pytest.mark.parametrize(
    ("flag", "trace_kwargs"),
    [
        ("interventions", {"intervene": "PREDICATE"}),
        ("rng_replay", {"capture": tl.options.CaptureOptions(random_seed=0)}),
        ("rng_replay", {"capture": tl.options.CaptureOptions(save_rng_states=True)}),
        ("backward_capture", {"capture": tl.options.CaptureOptions(backward_ready=True)}),
        ("backward_capture", {"capture": tl.options.CaptureOptions(save_grads="all")}),
        ("streaming", {"storage": "STORAGE"}),
    ],
)
def test_torch_flag_false_refuses_the_surface(flag: str, trace_kwargs: dict) -> None:
    """Review probe (reverse direction): flipping a torch capability flag False in
    place must refuse the corresponding trace() surface typed — the table is
    load-bearing on torch too, not only on previews."""

    model = nn.Linear(2, 2)
    inputs = torch.randn(1, 2)
    resolved_kwargs: dict[str, Any] = {}
    for key, value in trace_kwargs.items():
        if value == "PREDICATE":
            resolved_kwargs[key] = tl.when(tl.func("linear"), tl.zero_ablate())
        elif value == "STORAGE":
            resolved_kwargs[key] = object()
        else:
            resolved_kwargs[key] = value
    spec = get_backend_spec("torch")
    object.__setattr__(spec.capabilities, flag, False)
    try:
        with pytest.raises(BackendUnsupportedError, match=f"{flag}=False"):
            tl.trace(model, inputs, **resolved_kwargs)
    finally:
        object.__setattr__(spec.capabilities, flag, True)
        tl.release_model(model)


def test_torch_flag_false_refuses_capture_options_spelling() -> None:
    """The capture=CaptureOptions(...) spelling is gated by the same table."""

    model = nn.Linear(2, 2)
    inputs = torch.randn(1, 2)
    spec = get_backend_spec("torch")
    object.__setattr__(spec.capabilities, "backward_capture", False)
    try:
        with pytest.raises(BackendUnsupportedError, match="backward_capture=False"):
            tl.trace(model, inputs, capture=tl.options.CaptureOptions(backward_ready=True))
    finally:
        object.__setattr__(spec.capabilities, "backward_capture", True)
        tl.release_model(model)


def test_torch_fastlog_flag_false_refuses_record() -> None:
    """tl.record() with no backend argument still consults torch's fastlog flag."""

    model = nn.Linear(2, 2)
    inputs = torch.randn(1, 2)
    spec = get_backend_spec("torch")
    object.__setattr__(spec.capabilities, "fastlog", False)
    try:
        with pytest.raises(BackendUnsupportedError, match="fastlog=False"):
            tl.record(model, inputs, save=tl.func("linear"))
    finally:
        object.__setattr__(spec.capabilities, "fastlog", True)
        tl.release_model(model)


def test_registered_capability_tables_match_declared_matrix() -> None:
    """Every registered gated flag equals the declared expected matrix.

    A capability lift must edit ``EXPECTED_GATED_CAPABILITIES`` (one line)
    alongside its real mechanism and conformance coverage; a drive-by flag
    flip fails here.
    """

    for spec in registered_backend_specs():
        expected = EXPECTED_GATED_CAPABILITIES[str(spec.name)]
        for flag in sorted(GATED_CAPABILITY_FLAGS):
            assert getattr(spec.capabilities, flag) == expected[flag], (
                f"{spec.name}.{flag} diverges from EXPECTED_GATED_CAPABILITIES"
            )
        if str(spec.name) == "torch":
            assert "runnable" in spec.capabilities.save_levels
        else:
            assert "runnable" not in spec.capabilities.save_levels
