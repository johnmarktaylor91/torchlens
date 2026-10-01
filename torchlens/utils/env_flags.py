"""Closed-vocabulary parsing for torchlens-owned boolean environment knobs.

THE one boolean env parser (round-7 b7 R47): every torchlens-owned on/off
knob must parse against the same closed vocabulary, because the historical
exact-``"1"`` / raw-truthiness spellings turned a typo into a silently
different configuration -- ``TORCHLENS_COLLAPSE_STRICT=true`` left a
verification tripwire DISARMED while the exporter believed it was armed, and
``TORCHLENS_DEBUG_FORK_COPY=0`` ENABLED the debug behavior it names off.
The vocabulary and refuse-on-unrecognized behavior mirror the postprocess
audit knob parser (``torchlens/postprocess/__init__.py``), the reviewed
precedent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .._errors import InvalidArgumentError

__all__ = ["ENV_FLAG_REGISTRY", "EnvFlagSpec", "closed_bool_env"]

_TRUE_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_VALUES = frozenset(("0", "false", "no", "off"))


def closed_bool_env(name: str, *, default: bool = False) -> bool:
    """Parse a torchlens-owned boolean env var against a closed vocabulary.

    Parameters
    ----------
    name:
        Environment variable name (``TORCHLENS_*``).
    default:
        Value when the variable is unset or empty (the only implicit spelling).

    Returns
    -------
    bool
        ``True`` for ``1/true/yes/on``, ``False`` for ``0/false/no/off``
        (case-insensitive, surrounding whitespace ignored).

    Raises
    ------
    InvalidArgumentError
        When the value is set but unrecognized (``env_flag_invalid``). A knob
        whose typo silently selects one of its two states is a disarmed
        tripwire; refusing is the only honest reading.
    """

    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value == "":
        return default
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise InvalidArgumentError(
        f"{name}={raw!r} is not a recognized value",
        code="env_flag_invalid",
        remedy=(
            "use '1'/'true'/'yes'/'on' to enable, '0'/'false'/'no'/'off' to "
            "disable explicitly, or unset the variable"
        ),
        argument=name,
    )


@dataclass(frozen=True)
class EnvFlagSpec:
    """One registered torchlens-owned environment knob.

    Parameters
    ----------
    name:
        Environment variable name (``TORCHLENS_*``).
    kind:
        ``"bool"`` (parsed by :func:`closed_bool_env` or a ledgered inline
        closed-vocabulary parser), ``"enum"`` (closed value set in
        ``values``), or ``"path"`` (filesystem path, no boolean semantics).
    reader:
        Dotted module path of the consuming module (documentation, not
        dispatch).
    description:
        One-line user-facing meaning of the TRUE / set state.
    values:
        Closed value vocabulary for ``kind="enum"`` (empty string = unset).
    """

    name: str
    kind: str
    reader: str
    description: str
    values: tuple[str, ...] = field(default_factory=tuple)


#: THE registry of torchlens-owned environment knobs (WT1 A-VI item 34).
#: Every ``TORCHLENS_*`` environment read in the package must have a row
#: here, and every ``kind="bool"`` knob must parse through
#: :func:`closed_bool_env` (or a reason-ledgered inline closed-vocabulary
#: parser) so a typo can never silently select a state and "true" can never
#: mean OFF. The registration lint lives in
#: ``tests/test_entry_facade_env_flags.py`` and walks the package for env
#: reads, so an unregistered knob is red, not forgotten.
ENV_FLAG_REGISTRY: dict[str, EnvFlagSpec] = {
    spec.name: spec
    for spec in (
        EnvFlagSpec(
            name="TORCHLENS_AUTO",
            kind="bool",
            reader="torchlens.user_funcs / torchlens.experimental",
            description=(
                "Requests the unsupported implicit capture mode; TRUE refuses "
                "capture entry with auto_environment_unsupported."
            ),
        ),
        EnvFlagSpec(
            name="TORCHLENS_CACHE_DIR",
            kind="path",
            reader="torchlens._capture_state_helpers",
            description="Overrides the capture cache directory (default ~/.cache/torchlens).",
        ),
        EnvFlagSpec(
            name="TORCHLENS_COLLAPSE_STRICT",
            kind="bool",
            reader="torchlens.visualization._render_common",
            description="Arms the strict collapse-verification tripwire during rendering.",
        ),
        EnvFlagSpec(
            name="TORCHLENS_COLLAPSE_WATCHDOG",
            kind="bool",
            reader="torchlens.visualization.collapse_estimator",
            description=(
                "Collapse quality-planner wall-clock watchdog master switch "
                "(default TRUE; FALSE disables the watchdog everywhere)."
            ),
        ),
        EnvFlagSpec(
            name="TORCHLENS_DETERMINISTIC",
            kind="bool",
            reader="torchlens.visualization.collapse_estimator",
            description=(
                "Declares a determinism-sensitive run: TRUE disables the "
                "collapse watchdog so wall-clock can never change results."
            ),
        ),
        EnvFlagSpec(
            name="TORCHLENS_DEBUG_FORK_COPY",
            kind="bool",
            reader="torchlens.data_classes._trace_fork",
            description="Enables fork-copy debug diagnostics.",
        ),
        EnvFlagSpec(
            name="TORCHLENS_DEFER_GRAD_PAYLOADS",
            kind="bool",
            reader="torchlens.utils.tensor_utils",
            description="Defers gradient payload clones (opt-in).",
        ),
        EnvFlagSpec(
            name="TORCHLENS_EAGER_PAYLOAD_CLONE",
            kind="bool",
            reader="torchlens.utils.tensor_utils",
            description="Disables deferred payload clones (TRUE forces eager cloning).",
        ),
        EnvFlagSpec(
            name="TORCHLENS_PLUGINS",
            kind="enum",
            reader="torchlens.ecosystem.plugins",
            description=(
                "Plugin activation kill switch: 'none' refuses every "
                "activation path (beats explicit consent); unset = normal "
                "explicit activation."
            ),
            values=("", "none"),
        ),
        EnvFlagSpec(
            name="TORCHLENS_POSTPROCESS_ASSERTIONS",
            kind="bool",
            reader="torchlens.postprocess",
            description=(
                "Arms the postprocess write-audit windows (inline closed-vocabulary "
                "parser, ledgered: it adds the -O/-OO stripped-assertions refusal "
                "and the audit-specific refusal code)."
            ),
        ),
        EnvFlagSpec(
            name="TORCHLENS_POSTPROCESS_WRITE_AUDIT",
            kind="enum",
            reader="torchlens.postprocess",
            description="Write-audit mode: unset = enforce, 'record' = record.",
            values=("", "record"),
        ),
        EnvFlagSpec(
            name="TORCHLENS_SUMMARY_STYLE",
            kind="enum",
            reader="torchlens.report._summary_charset",
            description=(
                "Charset override for summary display boundaries (detection "
                "ladder rung 2): 'ascii' or 'unicode'; unset/other = detect, "
                "failing toward ASCII."
            ),
            values=("", "ascii", "unicode"),
        ),
        EnvFlagSpec(
            name="TORCHLENS_POSTPROCESS_READ_AUDIT",
            kind="enum",
            reader="torchlens.postprocess",
            description="Read-audit mode: unset = off, 'record', or 'enforce'.",
            values=("", "record", "enforce"),
        ),
        EnvFlagSpec(
            name="TORCHLENS_SUPPRESS_TORCH_CAPABILITY_WARNINGS",
            kind="bool",
            reader="torchlens.utils._torch_compat",
            description=(
                "TRUE suppresses the one-per-flag torch capability degradation "
                "warning (the degradation stays visible in compat/doctor reports)."
            ),
        ),
        EnvFlagSpec(
            name="TORCHLENS_VALIDATE_PEAK_MEMORY",
            kind="bool",
            reader="torchlens.validation.consolidated",
            description="TRUE additionally measures peak memory during validation runs.",
        ),
        EnvFlagSpec(
            name="TORCHLENS_WATCH_DISABLE",
            kind="bool",
            reader="torchlens.trackers._watch",
            description=(
                "Kill switch for tl watch sessions: TRUE disables collection "
                "(off-only; can never activate instrumentation). A disabled "
                "watcher still writes its torchlens/run rows saying it was "
                "disabled and by what."
            ),
        ),
    )
}
