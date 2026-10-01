"""The D24 waiver schema: dated, owned, counted, expiring, never silent-wrong.

One schema across all three panels: dated, owned, issue-linked, 90-day
expiry, counted from rows never prose, second named reviewer for same-PR
additions, no grace for new surfaces -- and a waiver may NEVER authorize a
known plausible silent wrong answer (fixed or typed-refusal, only). The
ledger is empty at wave 0; the schema and its refusals are live from the
first merge so the first real waiver cannot invent its own rules.
"""

from __future__ import annotations

import csv
import datetime
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"

#: Hard ceiling on waiver lifetime (D24).
MAX_WAIVER_DAYS = 90


@dataclass(frozen=True)
class Waiver:
    """One waiver row.

    Parameters
    ----------
    waiver_id:
        Stable id (``WVR-...``).
    created:
        ISO creation date.
    expires:
        ISO expiry date, at most 90 days after ``created``.
    owner:
        Named owner.
    issue:
        Issue link/id.
    scope:
        Comma-joined row keys the waiver covers (rows, never prose).
    reason:
        One-line rationale.
    second_reviewer:
        Named second reviewer (required for same-PR additions).
    silent_wrong_risk:
        MUST be ``none``: a waiver never authorizes a known plausible
        silent wrong answer.
    """

    waiver_id: str
    created: str
    expires: str
    owner: str
    issue: str
    scope: str
    reason: str
    second_reviewer: str
    silent_wrong_risk: str


def validate_waiver(waiver: Waiver, today: datetime.date) -> tuple[str, ...]:
    """Validate one waiver row against the D24 schema.

    Parameters
    ----------
    waiver:
        The row under validation.
    today:
        Evaluation date (injected so expiry tests are deterministic).

    Returns
    -------
    tuple[str, ...]
        Findings; empty means the waiver is valid and unexpired.
    """

    findings: list[str] = []
    try:
        created = datetime.date.fromisoformat(waiver.created)
        expires = datetime.date.fromisoformat(waiver.expires)
    except ValueError:
        return (f"{waiver.waiver_id}: unparseable dates",)
    if (expires - created).days > MAX_WAIVER_DAYS:
        findings.append(f"{waiver.waiver_id}: lifetime exceeds {MAX_WAIVER_DAYS} days")
    if expires < today:
        findings.append(f"{waiver.waiver_id}: EXPIRED {waiver.expires}")
    for column in ("owner", "issue", "scope", "reason"):
        if not getattr(waiver, column).strip():
            findings.append(f"{waiver.waiver_id}: empty {column}")
    if waiver.silent_wrong_risk.strip().lower() != "none":
        findings.append(
            f"{waiver.waiver_id}: silent_wrong_risk={waiver.silent_wrong_risk!r} -- a "
            "waiver may NEVER authorize a known plausible silent wrong answer; fix "
            "the defect or make it a typed refusal (D24)"
        )
    return tuple(findings)


def load_waivers(path: Path | None = None) -> tuple[Waiver, ...]:
    """Load the waiver ledger.

    Parameters
    ----------
    path:
        Ledger path; defaults to the committed ``data/waivers.tsv``.

    Returns
    -------
    tuple[Waiver, ...]
        Parsed rows (validation is the gate's job so plants can assert on
        exact findings).
    """

    ledger = path if path is not None else DATA_DIR / "waivers.tsv"
    with ledger.open(newline="") as handle:
        return tuple(
            Waiver(**record)
            for record in csv.DictReader(
                (line for line in handle if not line.startswith("#")), delimiter="\t", restval=""
            )
        )
