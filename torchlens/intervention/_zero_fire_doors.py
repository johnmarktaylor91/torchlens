"""Door-specific zero-fire disclosure: the bound-model and rerun doors.

The policy itself lives in :mod:`._zero_fire`; this module builds each door's
coded warning or refusal, including the ``reason`` fields for a module whose
output was not a tensor (capture door parity).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any, NoReturn

from .._errors import TorchLensWarning
from .._ir.distribution_container import is_distribution
from ._zero_fire import (
    current_zero_fire_policy,
    external_stacklevel,
    live_distribution_leaf_paths,
    non_tensor_module_output_detail,
    non_tensor_reason_fields,
    raise_rerun_zero_fire,
    recorded_non_tensor_details,
    settle_zero_fire,
)


def note_non_tensor_module_output(details: dict[str, str], address: str, output: Any) -> bool:
    """Whether a module-output rule has no tensor to apply to; record why when so.

    The capture door applies a module rule to the tensor leaves of a tensor,
    tuple, list or dict output only; a ``torch.distributions`` output (whose
    parameters capture records as their own output ops) is not a tensor there,
    so the rule fires at zero sites and the reason names the parameter paths.
    """

    if not is_distribution(output):
        return False
    paths = live_distribution_leaf_paths(output)
    details.setdefault(
        address, non_tensor_module_output_detail(address, type(output).__qualname__, paths)
    )
    return True


def _reason_suffix(reason_fields: dict[str, Any]) -> str:
    """Return the ``Reason (...)`` sentence for a message, or an empty string."""

    if not reason_fields:
        return ""
    return f" Reason ({reason_fields['reason']}): {reason_fields['reason_detail']}."


def refuse_bound_zero_fire(policy: Any, report: Any, non_tensor_details: dict[str, str]) -> None:
    """Apply the zero-fire policy to one settled bind report (the fact is on the report).

    Raises
    ------
    BindingRuntimeError
        ``bind_zero_fire`` when a rule never fired and the policy is
        ``"error"``; the default ``"warn"`` emits a coded warning instead.
    """

    from .errors import BindingRuntimeError

    if not report.zero_fire_rule_ids:
        return
    names = ", ".join(report.zero_fire_rule_ids)
    reason_fields = non_tensor_reason_fields(list(non_tensor_details.values()))
    because = _reason_suffix(reason_fields)

    def _raise() -> NoReturn:
        """Refuse the call under ``on_zero_fire='error'``."""

        raise BindingRuntimeError(
            f"rule(s) [{names}] never fired during this bound {report.door}; "
            "on_zero_fire='error' refuses after the call (the model's side "
            "effects may already have occurred; the full report is retained "
            f"on .last_report).{because}",
            code="bind_zero_fire",
            remedy="fix the WHERE terms so every rule fires, or settle with the "
            "default spec.bind(model, on_zero_fire='warn') or 'disclose'",
        )

    def _warn() -> None:
        """Emit the coded ``bind_zero_fire`` warning under ``on_zero_fire='warn'``."""

        warnings.warn(
            TorchLensWarning(
                f"rule(s) [{names}] never fired during this bound {report.door}; the "
                "call returned the model's output with those rules unapplied "
                f"(recorded on .last_report.zero_fire_rule_ids).{because} Remedy: fix "
                "the WHERE terms, or pass on_zero_fire='disclose' to record without "
                "warning",
                code="bind_zero_fire",
                zero_fire_rule_ids=list(report.zero_fire_rule_ids),
                **reason_fields,
            ),
            stacklevel=external_stacklevel(),
        )

    settle_zero_fire(policy, warn=_warn, raise_error=_raise)


def _unfired_non_tensor_reason(
    new_log: Any, hook_plan: Sequence[Any], unfired: Sequence[str], identifier: Callable[[Any], str]
) -> dict[str, Any]:
    """Zero-fire ``reason`` fields for unfired module entries whose output is not a tensor."""

    unfired_ids = set(unfired)
    targets = {
        str(getattr(getattr(entry, "site_target", None), "selector_value", ""))
        for entry in hook_plan
        if identifier(entry) in unfired_ids
    }
    return non_tensor_reason_fields(recorded_non_tensor_details(new_log, targets))


def disclose_unfired_rerun_entries(
    new_log: Any, hook_plan: Sequence[Any], unfired: list[str], identifier: Callable[[Any], str]
) -> None:
    """Settle the rerun door's zero-fire disclosure for ``unfired`` plan entries.

    First CONTRACTED S-18 row (compo wave 0): the disclosure standing between a
    user and a silently un-applied intervention carries a stable machine-readable
    code and a structured remedy; consumers branch on ``fields["code"]``, never on
    message text. The typed refusals fire first, whatever the on_zero_fire policy
    says; the policy settles the rest (default: this coded warning). Under
    "error" the refusal raises here, before the swap, so the trace is unchanged.
    """

    message = (
        "Rerun hook plan entries fired at zero sites on the new inputs: "
        f"{unfired!r}. The rerun completed, but those interventions were "
        "no-ops. Remedy: resolve the target sites against the rerun trace "
        "(trace[selector]) before re-applying, or route the edit "
        "through the push engine (fork().do(...)), which validates sites "
        "at plan time"
    )
    reason_fields = _unfired_non_tensor_reason(new_log, hook_plan, unfired, identifier)
    if reason_fields:
        message = f"{message}.{_reason_suffix(reason_fields)}".rstrip(".")

    def _warn() -> None:
        """Emit the coded ``rerun_zero_fire`` warning under ``on_zero_fire='warn'``."""

        warnings.warn(
            TorchLensWarning(
                message,
                code="rerun_zero_fire",
                unfired_plan_ids=list(unfired),
                **reason_fields,
            ),
            stacklevel=external_stacklevel(),
        )

    settle_zero_fire(
        current_zero_fire_policy(),
        warn=_warn,
        raise_error=lambda: raise_rerun_zero_fire(message, list(unfired)),
    )
