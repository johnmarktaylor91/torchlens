"""The one backend protocol every TorchLens backend fills, torch included.

``BackendProtocol`` is the trace-producing contract the conformance suite and
the shared backend rules talk to: framework requirement, capability table,
capture, validate, the oracle's independent reference forward, host copies,
the realize-once hook for lazy frameworks and the closed refusal-code
registry. It is deliberately small. The 37-method orchestration adapter in
``_protocol.py`` (``CaptureBackend``) stays torch's internal seam until the
capture-pipeline rebuild makes torch implement this protocol natively; today
torch satisfies it through a delegating adapter and the other backends
through delegating shells, so nothing about any backend's capture changes by
registering one.

Two contracts the protocol carries for every backend:

- Typed refusal: anything a backend cannot do raises
  ``BackendUnsupportedError`` with ``code`` drawn from :data:`REFUSAL_CODES`,
  before any framework side effect. A silent wrong result is never an option;
  a recorded gap (``capture_verified=False`` with a reason from the registry)
  is the only other allowed outcome and the conformance suite scores it as a
  failed cell.
- Named validation verdicts: ``validate`` returns the replay status object
  whose per-op decisions carry a reason from :data:`VALIDATION_REASONS`;
  a bare boolean is a contract violation.

Spellings here are DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Literal, Protocol, runtime_checkable

from packaging.version import InvalidVersion, Version

from .._validation_status import (
    _VALIDATION_REASON_DEFINITIONS as _VALIDATION_REASON_DEFINITIONS,
    VALIDATION_REASONS as VALIDATION_REASONS,
)
from ._errors import BackendUnsupportedError

__tl_layer__ = "L2"

#: ``registry.get_backend_spec``, bound by the registry at import. The shells
#: read specs through this hook so this leaf never imports the registry that
#: imports it (``protocol_problems``).
_spec_lookup: Callable[[str], Any] | None = None


def _install_spec_lookup(lookup: Callable[[str], Any]) -> None:
    """Bind the spec lookup ``SpecBackedProtocol._spec`` resolves through."""

    global _spec_lookup
    _spec_lookup = lookup


Materialization = Literal["eager", "lazy"]

#: Whether op outputs are concrete when a wrapped call returns (``eager``) or
#: deferred until an evaluation (``lazy``); lazy backends obey realize-once.
MATERIALIZATIONS: frozenset[str] = frozenset({"eager", "lazy"})

#: Closed registry of refusal codes any backend may raise. A raise with a code
#: outside it fails the backend's ``c0_refusal_codes_registered`` cell. One
#: code per cause; capability and option refusals end in ``_unsupported``.
REFUSAL_CODES: frozenset[str] = frozenset(
    {
        # one per BackendCapabilities flag that can be False
        "backward_capture_unsupported",
        "fastlog_unsupported",
        "interventions_unsupported",
        "rng_replay_unsupported",
        "streaming_unsupported",
        "structure_only_capture_unsupported",
        "validation_replay_unsupported",
        "payload_materialization_unsupported",
        "intermediate_derived_grads_unsupported",
        # option and mode refusals
        "trace_option_unsupported",
        "module_identity_mode_unsupported",
        "save_level_unsupported",
        "explicit_seed_unsupported",
        # runtime and input refusals
        "runtime_missing",
        "runtime_version_unsupported",
        "foreign_tensor_input",
        # the shared constants and realize-once rules
        "constant_unrealizable",
        "rng_state_in_graph",
        "unwrapped_op_in_capture",
        # codes the backends already raise today (kept as met)
        "value_dependent_branch_unsupported",
        "followed_by_unsupported",
        "torchscript_boundary",
        "scripted_function_not_captured",
        "foreign_kernel_boundary",
        "unsupported_tensor_variant",
        "output_unsupported_tensor_variant",
        "subclass_ctor_under_dispatch_mode_unsupported",
        "predicate_return_invalid",
        "structure_only_substrate_mismatch",
        "lookback_payload_policy_conflict",
        "legacy_constructor_uncaptured",
        "input_rung_requires_module_root",
        "held_tensor_scan_truncated",
        "meta_kernel_unavailable",
        "emit_nvtx_unavailable",
        "dlpack_call_not_captured",
        "output_attribution_failed",
        "wrappers_removed_before_capture",
        "unwrap_during_active_capture",
        "release_during_active_capture",
        # backend-specific refusals, one per cause
        "runtime_mode_unsupported",
        "inplace_op_unsupported",
        "input_mutation_unsupported",
        "tensor_output_required",
        "tinygrad_input_required",
        "live_payloads_required",
        "mid_capture_execution_unsupported",
        "traced_transform_entry_unsupported",
        "intervention_with_derived_grads_unsupported",
        "intervention_result_invalid",
    }
)


#: ``BackendCapabilities`` field -> the refusal code its ``False`` value owes
#: (the suite's ``c1_refusal_<flag>`` cells). Keep in step with the dataclass.
REFUSABLE_CAPABILITY_FLAGS: dict[str, str] = {
    "backward_capture": "backward_capture_unsupported",
    "fastlog": "fastlog_unsupported",
    "can_intervene": "interventions_unsupported",
    "rng_replay": "rng_replay_unsupported",
    "has_streaming": "streaming_unsupported",
    "structure_only_capture": "structure_only_capture_unsupported",
    "has_validation_replay": "validation_replay_unsupported",
    "payload_materialization": "payload_materialization_unsupported",
    "intermediate_derived_grads": "intermediate_derived_grads_unsupported",
}


def refusal_code_for_capability(flag: str) -> str:
    """The contracted refusal code for a False capability ``flag``.

    Parameters
    ----------
    flag:
        A ``BackendCapabilities`` field name in :data:`REFUSABLE_CAPABILITY_FLAGS`.

    Returns
    -------
    str
        The code, a member of :data:`REFUSAL_CODES`.

    Raises
    ------
    KeyError
        If ``flag`` is not a refusable capability flag.
    """

    return REFUSABLE_CAPABILITY_FLAGS[flag]


def refuse_backward_capture_unsupported(spec: Any) -> None:
    """Refuse a backward capture on a backend without the ``backward_capture`` capability."""

    if spec.capabilities.backward_capture:
        return
    raise BackendUnsupportedError(
        f"Backend {spec.name!r} does not support backward capture.",
        code=refusal_code_for_capability("backward_capture"),
        remedy="use trace.autograd.derived_grads when this backend exposes leaf-level derived gradients",
        backend=spec.name,
    )


@dataclass(frozen=True)
class FrameworkRequirement:
    """What a backend needs from its framework, stated once.

    Parameters
    ----------
    distributions:
        Installed distribution names that version the framework, in lookup
        order (``("tensorflow", "tensorflow-cpu")``).
    minimum:
        Inclusive version floor.
    below:
        Exclusive version ceiling, or ``None`` when none is declared; then
        ``tested_through`` states the range's upper evidence.
    tested_through:
        Highest version the backend's suite has run on.
    pinned:
        The exact version the per-change CI leg installs, or ``None`` while
        the leg is not pinned yet (a graduation blocker, never an error here).
    extra:
        The pyproject extra that installs the framework (``""`` for a base
        dependency).
    excluded:
        Versions inside the range that are refused (broken wheels).

    Examples
    --------
    >>> tl.backends.spec("torch").protocol().framework.distributions
    ('torch',)
    >>> req = FrameworkRequirement(distributions=("jax",), minimum="0.4", below=None,
    ...     tested_through="0.4.30", pinned=None, extra="jax")
    >>> (req.minimum, req.excluded)
    ('0.4', ())
    """

    distributions: tuple[str, ...]
    minimum: str
    below: str | None
    tested_through: str
    pinned: str | None
    extra: str
    excluded: tuple[str, ...] = ()

    def _installed_version(self) -> str | None:
        """The installed framework version, or ``None`` when absent."""

        for distribution in self.distributions:
            try:
                return metadata.version(distribution)
            except metadata.PackageNotFoundError:
                continue
        return None

    def _admits(self, version: str) -> bool:
        """Whether ``version`` lies in ``[minimum, below)`` and is not excluded.

        Parameters
        ----------
        version:
            Version string to test.

        Returns
        -------
        bool
            ``False`` for an unparsable version.
        """

        try:
            candidate = Version(version)
        except InvalidVersion:
            return False
        if candidate < Version(self.minimum):
            return False
        if self.below is not None and candidate >= Version(self.below):
            return False
        return all(candidate != Version(excluded) for excluded in self.excluded)

    def _consistency_problems(self) -> tuple[str, ...]:
        """Internal contradictions, each as one sentence (empty when consistent)."""

        problems: list[str] = []
        if not self.distributions:
            problems.append("distributions is empty")
        if self.below is None and not self.tested_through:
            problems.append("neither below nor tested_through states the range's upper end")
        if self.tested_through and not self._admits(self.tested_through):
            problems.append(f"tested_through {self.tested_through} lies outside the range")
        if self.pinned is not None and not self._admits(self.pinned):
            problems.append(f"pinned {self.pinned} lies outside the range")
        return tuple(problems)


@runtime_checkable
class BackendProtocol(Protocol):
    """The contract every registered backend fills (SPEC section 1.1).

    Members are the ones every backend already has as functions somewhere;
    the protocol names them once so the conformance suite, the constants rule
    and the realize-once rule can drive any backend the same way.

    Examples
    --------
    >>> protocol = tl.backends.spec("torch").protocol()
    >>> isinstance(protocol, BackendProtocol)
    True
    >>> (protocol.name, protocol.is_tensor(x))
    ('torch', True)
    """

    name: str
    framework: FrameworkRequirement
    materialization: str

    def capabilities(self) -> Any:
        """The declared ``BackendCapabilities`` table (the registry's truth)."""
        ...

    def runtime_available(self) -> str | None:
        """``None`` when the framework imports inside the range, else a refusal code."""
        ...

    def can_handle(self, model: Any, args: Any, kwargs: dict[str, Any] | None) -> bool:
        """Side-effect-light detector; rejects foreign tensor leaves."""
        ...

    def capture(self, model: Any, args: Any, kwargs: dict[str, Any] | None, **options: Any) -> Any:
        """The public capture entry; refuses typed before any framework call."""
        ...

    def reference_output(self, model: Any, args: Any, kwargs: dict[str, Any] | None) -> Any:
        """The framework's own forward, independent of TorchLens wrappers."""
        ...

    def validate(self, trace: Any, ground_truth: Any) -> Any:
        """Replay validation returning the status object, never a bare bool."""
        ...

    def is_tensor(self, value: Any) -> bool:
        """Whether ``value`` is one of this framework's array types."""
        ...

    def host_copy(self, value: Any) -> Any:
        """A detached NumPy copy, shape and dtype kept (0-d stays 0-d)."""
        ...

    def realize_once(self, values: Sequence[Any]) -> None:
        """Lazy backends: evaluate every value in one pass; eager: no-op."""
        ...

    def trace_options(self) -> tuple[str, ...]:
        """Trace options this backend implements."""
        ...

    def refusal_codes(self) -> frozenset[str]:
        """Every refusal code this backend can raise (a subset of the registry)."""
        ...


#: Runtime attributes a registered protocol object must expose.
BACKEND_PROTOCOL_MEMBERS: tuple[str, ...] = (
    "name",
    "framework",
    "materialization",
    "capabilities",
    "runtime_available",
    "can_handle",
    "capture",
    "reference_output",
    "validate",
    "is_tensor",
    "host_copy",
    "realize_once",
    "trace_options",
    "refusal_codes",
)


def protocol_problems(candidate: Any, *, expected_name: str | None = None) -> tuple[str, ...]:
    """Why ``candidate`` fails the protocol, each as one sentence.

    Parameters
    ----------
    candidate:
        Object returned by a ``BackendSpec.protocol`` factory.
    expected_name:
        The spec name the object must report, when known.

    Returns
    -------
    tuple[str, ...]
        Empty when the object satisfies the protocol.
    """

    problems = [
        f"missing member {m!r}" for m in BACKEND_PROTOCOL_MEMBERS if not hasattr(candidate, m)
    ]
    if problems:
        return tuple(problems)
    if expected_name is not None and candidate.name != expected_name:
        problems.append(f"name {candidate.name!r} differs from the spec name {expected_name!r}")
    if candidate.materialization not in MATERIALIZATIONS:
        problems.append(
            f"materialization {candidate.materialization!r} is not one of "
            f"{sorted(MATERIALIZATIONS)}"
        )
    if not isinstance(candidate.framework, FrameworkRequirement):
        problems.append("framework is not a FrameworkRequirement")
    unknown = sorted(set(candidate.refusal_codes()) - REFUSAL_CODES)
    if unknown:
        problems.append(f"refusal codes outside the registry: {unknown}")
    return tuple(problems)


class SpecBackedProtocol:
    """Delegating base for backends whose entries live on their ``BackendSpec``.

    The shells for every backend subclass this and add the framework-specific
    members (``is_tensor``, ``host_copy``, ``realize_once``, ``runtime_available``).
    Delegation is lazy through the registry so constructing a shell never
    imports a framework.

    Parameters
    ----------
    name:
        Registered backend name.
    framework:
        The framework requirement.
    materialization:
        ``"eager"`` or ``"lazy"``.
    trace_options:
        Options this backend implements.
    refusal_codes:
        Codes this backend raises today.
    """

    def __init__(
        self,
        *,
        name: str,
        framework: FrameworkRequirement,
        materialization: str,
        trace_options: Iterable[str],
        refusal_codes: Iterable[str],
    ) -> None:
        self.name = name
        self.framework = framework
        self.materialization = materialization
        self._trace_options = tuple(trace_options)
        self._refusal_codes = frozenset(refusal_codes)

    def _spec(self) -> Any:
        """The registered ``BackendSpec`` this protocol is backed by."""

        if _spec_lookup is None:  # pragma: no cover - registry import installs it
            raise RuntimeError(
                "backend spec lookup is not bound; import torchlens.backends before "
                "constructing a protocol shell"
            )
        return _spec_lookup(self.name)

    def capabilities(self) -> Any:
        """The registered ``BackendCapabilities`` (the only capability truth)."""

        return self._spec().capabilities

    def runtime_available(self) -> str | None:
        """Version-range check over the installed distribution; subclasses add the import."""

        installed = self.framework._installed_version()
        if installed is None:
            return "runtime_missing"
        if not self.framework._admits(installed):
            return "runtime_version_unsupported"
        return None

    def can_handle(self, model: Any, args: Any, kwargs: dict[str, Any] | None) -> bool:
        """Delegate to the spec detector."""

        return bool(self._spec().can_handle(model, args, kwargs))

    def capture(self, model: Any, args: Any, kwargs: dict[str, Any] | None, **options: Any) -> Any:
        """Delegate to the spec's public capture entry."""

        return self._spec().capture_trace(model, args, kwargs, **options)

    def reference_output(self, model: Any, args: Any, kwargs: dict[str, Any] | None) -> Any:
        """Call the model directly; a tuple is positional arguments, like ``tl.trace``."""

        call_kwargs = dict(kwargs or {})
        if isinstance(args, tuple):
            return model(*args, **call_kwargs)
        return model(args, **call_kwargs)

    def validate(self, trace: Any, ground_truth: Any) -> Any:
        """Delegate to the spec's trace validation entry."""

        return self._spec().validate_trace(trace, ground_truth)

    def is_tensor(self, value: Any) -> bool:
        """Shells override with the framework's array type check."""

        raise NotImplementedError(f"{self.name} shell must implement is_tensor")

    def host_copy(self, value: Any) -> Any:
        """NumPy copy through the value's ``numpy()`` or ``__array__``; shape kept."""

        import numpy as np

        to_numpy = getattr(value, "numpy", None)
        return np.array(to_numpy() if callable(to_numpy) else value, copy=True)

    def realize_once(self, values: Sequence[Any]) -> None:
        """Eager default: nothing to evaluate."""

        return None

    def trace_options(self) -> tuple[str, ...]:
        """Options this backend implements."""

        return self._trace_options

    def refusal_codes(self) -> frozenset[str]:
        """Codes this backend raises today."""

        return self._refusal_codes


ProtocolFactory = Callable[[], BackendProtocol]
