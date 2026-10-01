"""C07X identity-fact load validators (tlspec v9 entry-dark slots).

The Trace-level root entry-point fact (``Trace.root_entry_point``, written
unconditionally at capture; foldA D10) and the two entry-dark Op facts the
C07X amendment declares (``Op.episode_step``, F-EPISODE writes;
``Op.tl_authored_root``, F41 writes), each with fail-closed load
validation. Split from ``_io/forgery_validation.py`` at the C07X amendment
(R43 size discipline); the one consumer is
``validate_persisted_forgery_surfaces``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


def _refuse_identity(message: str, *, code: str, field: str, reason: str, remedy: str) -> None:
    """Route one identity-fact refusal through the shared teaching refusal."""

    from .forgery_validation import _refuse

    _refuse(message, code=code, field=field, reason=reason, remedy=remedy)


def _is_int(value: Any) -> bool:
    """True for a real int (bools excluded)."""

    return isinstance(value, int) and not isinstance(value, bool)


def _ops_of(trace: Trace) -> tuple[Any, ...]:
    """The trace's op records, through the shared census helper."""

    from .forgery_validation import _trace_ops

    return _trace_ops(trace)


#: Closed root entry-point kind tokens (C07X item (iv), foldA D10/D11):
#: ``module_call`` (nn.Module/callable-model roots), ``bound_method`` (F41's
#: bound-method roots), ``function_call`` (preview function roots).
_ROOT_ENTRY_POINT_KINDS = frozenset({"module_call", "bound_method", "function_call"})


#: First tlspec stamp on which an ABSENT ``Trace.root_entry_point`` refuses.
#: The C07X writers stamp the fact unconditionally, but they ride the SAME v9
#: window as pre-C07X v9 writers whose artifacts are checked in (e.g.
#: ``tests/agent_surface_goldens/clean.tlspec``: tlspec 9, torchlens 2.34.1,
#: no root_entry_point), so within v9 the slot is ENTRY-DARK by construction
#: and absence cannot be told from deletion. The tripwire is armed for the
#: next coordinated bump (AUD-CODE 3.0e; W051-IO).
_ROOT_ENTRY_POINT_REQUIRED_FROM_TLSPEC = 10


def _refuse_absent_root_entry_point_on_current_schema(trace: Trace) -> None:
    """Refuse a v9+ governed artifact whose root entry-point fact is absent (AUD-CODE 3.0e).

    Only inside the governed ``.tlspec`` load window: plain session pickling of
    live records keeps its historical tolerance, and artifacts stamped below
    ``_ROOT_ENTRY_POINT_REQUIRED_FROM_TLSPEC`` keep the legacy ``None`` reading.
    """

    from .state_contract import governed_load_active

    stamp = getattr(trace, "tlspec_version", None)
    if not governed_load_active() or not isinstance(stamp, int) or isinstance(stamp, bool):
        return
    if stamp < _ROOT_ENTRY_POINT_REQUIRED_FROM_TLSPEC:
        return
    _refuse_identity(
        f"Trace.root_entry_point is absent on a tlspec_version={stamp} artifact; every "
        f"writer since tlspec {_ROOT_ENTRY_POINT_REQUIRED_FROM_TLSPEC} stamps the root "
        "invocation descriptor unconditionally (C07X), so absence is a deleted or "
        "forged identity fact rather than the legacy reading.",
        code="artifact_root_entry_point_invalid",
        field="Trace.root_entry_point",
        reason="absent_on_current_schema",
        remedy=(
            "re-save the artifact with one current TorchLens version; do not delete "
            "the root entry-point identity fact"
        ),
    )


def _validate_root_entry_point(trace: Trace) -> None:
    """Validate the Trace-level root entry-point identity fact (C07X).

    Grammar: ``"<kind>:<qualified_identity>"`` with a closed kind vocabulary
    and a non-empty identity. ``None`` is legal only as the legacy-artifact
    reading (pre-C07X saves); every current capture writes the fact
    unconditionally (foldA D10), and the rerun identity gate consumes it
    FAIL-CLOSED (F41).
    """

    value = getattr(trace, "root_entry_point", None)
    if value is None:
        _refuse_absent_root_entry_point_on_current_schema(trace)
        return
    reason = None
    if not isinstance(value, str):
        reason = "not_a_string"
    else:
        kind, separator, identity = value.partition(":")
        if not separator or kind not in _ROOT_ENTRY_POINT_KINDS or not identity:
            reason = "descriptor_grammar"
    if reason is not None:
        _refuse_identity(
            f"Trace.root_entry_point {value!r} violates the closed root "
            "invocation descriptor grammar "
            "('module_call|bound_method|function_call' + ':' + identity).",
            code="artifact_root_entry_point_invalid",
            field="Trace.root_entry_point",
            reason=reason,
            remedy=(
                "re-save the artifact with one current TorchLens version; do "
                "not hand-edit the root entry-point identity fact"
            ),
        )


def _validate_episode_step_stamps(trace: Trace) -> None:
    """Validate per-op episode-step stamps (tlspec v9 entry-dark, C07X).

    ``None`` on every op today (the F-EPISODE read-axis lane writes the
    stamp). A present value must be a non-negative int, and stamps are only
    coherent on a product carrying an episode declaration -- a stamped op on
    a plain capture is a forged or drifted artifact.
    """

    stamped_labels: list[str] = []
    for op in _ops_of(trace):
        value = getattr(op, "episode_step", None)
        if value is None:
            continue
        label = str(getattr(op, "label", "<unknown>"))
        if not _is_int(value) or value < 0:
            _refuse_identity(
                f"op {label!r} episode_step {value!r} must be a non-negative int or None.",
                code="artifact_episode_step_invalid",
                field="Op.episode_step",
                reason="stamp_type",
                remedy=(
                    "re-save the artifact with one current TorchLens version; "
                    "do not hand-edit episode-step stamps"
                ),
            )
        stamped_labels.append(label)
    if stamped_labels:
        from ..capture._episode_ledger import capture_kind_for

        if capture_kind_for(trace) != "episode":
            _refuse_identity(
                f"ops {stamped_labels[:3]!r} carry episode_step stamps on a "
                "capture with no episode declaration.",
                code="artifact_episode_step_invalid",
                field="Op.episode_step",
                reason="stamp_without_declaration",
                remedy=(
                    "re-capture with episode=EpisodeSpec(...) or drop the "
                    "forged stamps by re-saving from the source capture"
                ),
            )


def _validate_tl_authored_root(trace: Trace) -> None:
    """Validate the TL-authored-root marker (tlspec v9 entry-dark, C07X).

    ``None`` on every module-root op; F41 writes ``True`` on the root op
    record of its TL-authored bound-method wrapper. A present value must be
    a bool, and ``True`` is only coherent beside a ``bound_method`` root
    entry-point fact (the marker discloses the wrapper the fact names).
    """

    marked_labels: list[str] = []
    for op in _ops_of(trace):
        value = getattr(op, "tl_authored_root", None)
        if value is None:
            continue
        label = str(getattr(op, "label", "<unknown>"))
        if not isinstance(value, bool):
            _refuse_identity(
                f"op {label!r} tl_authored_root {value!r} must be a bool or None.",
                code="artifact_tl_authored_root_invalid",
                field="Op.tl_authored_root",
                reason="marker_type",
                remedy=(
                    "re-save the artifact with one current TorchLens version; "
                    "do not hand-edit the TL-authored-root marker"
                ),
            )
        if value:
            marked_labels.append(label)
    if marked_labels:
        root_fact = getattr(trace, "root_entry_point", None)
        if not (isinstance(root_fact, str) and root_fact.startswith("bound_method:")):
            _refuse_identity(
                f"ops {marked_labels[:3]!r} carry tl_authored_root=True but the "
                f"trace's root entry-point fact is {root_fact!r}, not a "
                "bound_method descriptor -- the marker discloses a TL-authored "
                "wrapper root that this capture does not declare.",
                code="artifact_tl_authored_root_invalid",
                field="Op.tl_authored_root",
                reason="marker_without_bound_method_root",
                remedy=(
                    "re-save from the source capture; do not hand-edit the TL-authored-root marker"
                ),
            )
