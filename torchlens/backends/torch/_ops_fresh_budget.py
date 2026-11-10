"""Save-budget admission and reconciliation for the direct payload path.

:func:`admit_fresh` charges a fresh plain output's byte size exactly as
``SaveBudget.admit`` does, and :func:`commit_fresh` reconciles it exactly as
``SaveBudget.commit`` does, without building the reservation record when both
steps take their common case:

* admission: the device's ledger already exists under ``str(device)`` (a cuda
  device without an index is canonicalised by ``SaveBudget._ledger_for``, so it
  goes through ``admit``) and the charge stays within its limit. A first save on
  a device (which measures the device and may warn) and a charge that would
  cross the limit go through ``SaveBudget.admit``, which creates the ledger, or
  raises the refusal with its own rollback, as it always has.
* reconciliation: the payload is a new storage of exactly the admitted bytes on
  the admitted device, not yet charged. ``commit`` would subtract the estimate
  and charge the same bytes back under a new storage entry, so the totals, the
  saved count and the refusal outcome are unchanged and only the entry and its
  release watcher are added. Any other payload goes through ``commit``.

``tests/test_fresh_payload_path.py`` compares the ledgers, the refusals and the
release credits with the general path.
"""

from __future__ import annotations

from typing import Any

import torch

from ..._save_budget import (
    SaveBudget,
    _BudgetReservation,
    _DeviceLedger,
    _RetainedStorageEntry,
)

__all__ = ("admit_fresh", "commit_fresh")


def _budget_label(fields_dict: dict[str, Any]) -> str:
    """Return the label the general path admits under (used in a refusal)."""

    return str(fields_dict["_layer_label_raw"] or fields_dict.get("_label_raw") or "<unlabeled>")


def admit_fresh(
    budget: SaveBudget, device: torch.device, num_bytes: int, fields_dict: dict[str, Any]
) -> _DeviceLedger | _BudgetReservation | None:
    """Charge ``num_bytes`` on ``device`` as ``SaveBudget.admit`` would.

    Returns
    -------
    _DeviceLedger | _BudgetReservation | None
        The charged ledger (common case), the reservation ``admit`` returned, or
        ``None`` for an empty payload, which ``admit`` does not reserve.

    Raises
    ------
    SaveBudgetExceededError
        Through ``SaveBudget.admit``, when the charge would cross the limit.
    """

    if num_bytes <= 0:
        return None
    key = str(device)
    ledgers = budget.ledgers
    if key not in ledgers or (device.type == "cuda" and device.index is None):
        return budget.admit(_budget_label(fields_dict), device, num_bytes)
    ledger = ledgers[key]
    committed = ledger.committed_bytes + num_bytes
    limit = ledger.limit_bytes
    if limit is not None and committed > limit:
        return budget.admit(_budget_label(fields_dict), device, num_bytes)
    ledger.committed_bytes = committed
    ledger.num_saved += 1
    return ledger


def commit_fresh(
    budget: SaveBudget,
    admitted: _DeviceLedger | _BudgetReservation,
    payload: torch.Tensor,
    identity: tuple[str, int, int] | None,
    device: torch.device,
    num_bytes: int,
    fields_dict: dict[str, Any],
) -> None:
    """Reconcile one admission with its payload as ``SaveBudget.commit`` would.

    Parameters
    ----------
    budget:
        The trace's accountant.
    admitted:
        What :func:`admit_fresh` returned (not ``None``).
    payload:
        The saved payload.
    identity:
        The payload's ``(device string, storage data pointer, storage bytes)``, or
        ``None`` when unreadable.
    device:
        The admitted device (the source's).
    num_bytes:
        The admitted byte size.
    fields_dict:
        The output's fields (for the refusal label).
    """

    if type(admitted) is _BudgetReservation:
        budget.commit(admitted, (payload, None))
        return
    # ``admitted`` is the ledger keyed ``str(device)`` (never a cuda device without
    # an index), so an identity on that device string is a payload on that ledger.
    assert type(admitted) is _DeviceLedger
    ledger_key = str(device)
    if (
        identity is None
        or identity[0] != ledger_key
        or identity[2] != num_bytes
        or identity in admitted.retained_storage
    ):
        reservation = _BudgetReservation(
            label=_budget_label(fields_dict), device=device, num_bytes=num_bytes, site="primary"
        )
        budget.commit(reservation, (payload, None))
        return
    admitted.retained_storage[identity] = _RetainedStorageEntry(
        physical_bytes=num_bytes, live_refs=1
    )
    budget._watch_payload(payload, ledger_key, identity)
