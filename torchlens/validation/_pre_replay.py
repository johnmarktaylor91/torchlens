"""Pre-replay verdicts: ops validation settles before any replay is attempted.

Split out of ``core.py``, which hands in what lives above this module: the
``ValidationCheckResult`` type and the intervention-replacement predicate
(``core._is_intentional_intervention_replacement``, which reads the backward-flow
ledger). One structural proof and one typed refusal live here:

* :func:`_is_provable_functionless_source_or_boundary` (ledger
  ``functionless_source_or_boundary``): a functionless op is a source or boundary by
  captured metadata, not a lost computational function;
* an **unresolved foreign data round trip** (``backends.torch._foreign_data``): data
  left torch and came back after host computation TorchLens did not see; the recorded
  value is never replayed as a constant (validation integrity), so the op fails replay
  with the typed reason ``foreign_data_roundtrip_unreplayable``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..backends.torch._foreign_data import FOREIGN_DATA_REPLAY_REASON, is_unresolved_foreign_data
from .diagnostics import CHECK_REPLAY, ValidationFailure, record_validation_failure

if TYPE_CHECKING:
    from .._data_classes.op import Op


Predicate = Callable[["Op"], bool]


@dataclass(frozen=True)
class CoreBindings:
    """What ``core`` hands down: its result type and the intervention-replacement predicate."""

    result_type: Any
    """``ValidationCheckResult``."""
    is_intervention_replacement: Predicate
    """``core._is_intentional_intervention_replacement`` (reads the backward-flow ledger)."""


def pre_replay_verdict(
    core: CoreBindings, trace: Any, layer: Op, op_label: str, perturb: bool
) -> Any | None:
    """Return the replay check's verdict when ``layer`` must not be replayed, else ``None``.

    Parameters
    ----------
    core:
        The result type and replacement predicate ``core`` binds.
    trace:
        Trace being validated.
    layer:
        Operation pass being validated.
    op_label:
        Label under which the op is validated.
    perturb:
        Whether this is a perturbation attempt rather than a replay.
    """

    result_type = core.result_type
    if layer.func is None:
        if _is_provable_functionless_source_or_boundary(core.is_intervention_replacement, layer):
            return result_type.exempted("functionless_source_or_boundary")
        return result_type.failed_result("functionless_computational_op")
    if is_unresolved_foreign_data(layer.func):
        if perturb:
            return result_type.unverified(FOREIGN_DATA_REPLAY_REASON)
        _record_unresolved_foreign_data_failure(trace, layer, op_label)
        return result_type.failed_result(FOREIGN_DATA_REPLAY_REASON)
    return None


def _record_unresolved_foreign_data_failure(trace: Any, layer: Op, op_label: str) -> None:
    """Record the typed failure of a foreign data round trip that cannot be re-run."""

    config = dict(getattr(layer, "transform_config", None) or {})
    parents = tuple(getattr(layer, "parents", ()) or ())
    record_validation_failure(
        trace,
        ValidationFailure(
            check_name=CHECK_REPLAY,
            op_label=op_label,
            func_name=getattr(layer, "func_name", None),
            message=(
                f"foreign data round trip {op_label} ({config.get('entry')}) re-entered torch "
                f"after host computation TorchLens did not see (left through "
                f"{', '.join(config.get('exits', ()))} from {', '.join(parents)}); the recorded "
                "value is not replayed as a constant. Wrap the foreign step in an "
                "autograd.Function or build the constant before the forward pass."
            ),
            extra={
                "reason": FOREIGN_DATA_REPLAY_REASON,
                "entry": config.get("entry"),
                "exits": list(config.get("exits", ())),
                "parents": list(parents),
            },
        ),
    )


def _is_provable_functionless_source_or_boundary(
    is_intervention_replacement: Predicate, layer: Op
) -> bool:
    """Return whether a functionless op is a structural source or boundary.

    Parameters
    ----------
    is_intervention_replacement:
        ``core._is_intentional_intervention_replacement`` (first, so ``core`` binds it).
    layer:
        Operation pass being validated.

    Returns
    -------
    bool
        True only when captured metadata proves the absence of a callable is
        expected by design, rather than a lost computational function.
    """

    if getattr(layer, "func", None) is not None:
        return False
    if is_intervention_replacement(layer):
        return True
    parents = tuple(getattr(layer, "parents", ()) or ())
    children = tuple(getattr(layer, "children", ()) or ())
    parent_arg_positions = getattr(layer, "parent_arg_positions", {}) or {}
    has_parent_arg_positions = any(
        bool(parent_arg_positions.get(arg_type)) for arg_type in ("args", "kwargs")
    )
    saved_args = getattr(layer, "saved_args", None)
    has_saved_args = bool(saved_args)
    func_call_id = getattr(layer, "func_call_id", None)
    is_buffer_boundary = (
        func_call_id is None
        and bool(getattr(layer, "buffer_source", None))
        and bool(getattr(layer, "buffer_write_kind", None))
    )
    if is_buffer_boundary:
        return True
    is_source = (
        not parents and not has_parent_arg_positions and not has_saved_args and func_call_id is None
    )
    if is_source:
        return True
    is_output_boundary = bool(parents) and not children and has_parent_arg_positions
    return is_output_boundary
