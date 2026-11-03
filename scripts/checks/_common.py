"""Shared result type and file helpers for the policy checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

OK = "ok"
FAIL = "fail"
SKIP = "skip"

BASELINES = Path("scripts/checks/baselines.json")


@dataclass(frozen=True)
class Outcome:
    """One check's verdict and the one line printed for it."""

    status: str
    message: str
    details: tuple[str, ...] = ()

    @property
    def failed(self) -> bool:
        return self.status == FAIL


def load_baselines(repo: Path) -> dict[str, dict[str, object]]:
    """Read ``scripts/checks/baselines.json`` under ``repo``."""

    data = json.loads((repo / BASELINES).read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not key.startswith("_")}


def ratchet(name: str, measured: dict[str, int], baseline: dict[str, object]) -> Outcome:
    """Fail when any measured number exceeds its recorded baseline.

    A number below its baseline passes and is reported, so the baseline can be
    lowered in the same change. A measured key with no baseline fails: new
    counters start with a recorded number.
    """

    over, under = [], []
    for key, value in sorted(measured.items()):
        limit = baseline.get(key)
        if not isinstance(limit, int):
            over.append(f"{key}={value} has no baseline")
        elif value > limit:
            over.append(f"{key}={value} > baseline {limit}")
        elif value < limit:
            under.append(f"{key}={value} < baseline {limit} (lower the baseline)")
    shown = ", ".join(f"{k}={v}" for k, v in sorted(measured.items()))
    if over:
        return Outcome(FAIL, f"{name}: over baseline: {'; '.join(over)}", tuple(under))
    return Outcome(OK, f"{name}: {shown}", tuple(under))
