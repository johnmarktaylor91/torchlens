"""Regenerate the committed oracle baselines from the live tree.

Run from the repo root::

    python tests/oracles/_regen.py [--check]

This is the CONSCIOUS-UPDATE tool the gate teaching messages point at: when a
gate reports a surface/census delta, the author of the change reruns this,
reviews the diff (the diff IS the public-surface change review), and commits
the baseline beside the change. ``--check`` exits nonzero when any baseline
would change, without writing.

KNOWN-GAP rows are NEVER regenerated -- the manifest is hand-curated,
monotone-down, and only shrinks (D25). This script writes walk-derived
baselines only. The CLASS layer has no baseline file at all: it gates on the
license rules in ``_surface.CLASS_MEMBER_LICENSES`` (frozen member
inventories cannot compose across sibling branches; train T03).
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

from oracles import _censuses, _surface  # noqa: E402
from oracles._deprecation import load_deprecated_doors  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"


def _render_module_classification() -> str:
    """Render the module-layer classification baseline.

    Returns
    -------
    str
        TSV text.
    """

    import torchlens

    doors = frozenset(row.door for row in load_deprecated_doors())
    entries = _surface.walk_module_surface(torchlens, doors)
    lines = ["name\tclassification"]
    lines += [f"{entry.name}\t{entry.classification}" for entry in entries]
    return "\n".join(lines) + "\n"


def _render_numeral_baseline() -> str:
    """Render the dated legacy numeral baseline (census root 8).

    Returns
    -------
    str
        TSV text.
    """

    result = _censuses.census_numerals(_REPO_ROOT)
    lines = ["corpus\tfile\tnumeral"]
    lines += ["\t".join(row) for row in result.rows]
    return "\n".join(lines) + "\n"


def _render_census_baseline(name: str) -> str:
    """Render one single-column census baseline.

    Parameters
    ----------
    name:
        Census attribute name on ``_censuses`` (e.g. ``census_options_fields``).

    Returns
    -------
    str
        TSV text.
    """

    result = getattr(_censuses, name)()
    lines = ["key"]
    lines += ["\t".join(row) if isinstance(row, tuple) else row for row in result.rows]
    return "\n".join(lines) + "\n"


BASELINES: dict[str, object] = {
    "surface_module_classification.tsv": _render_module_classification,
    "numeral_baseline.tsv": _render_numeral_baseline,
    "options_fields.tsv": lambda: _render_census_baseline("census_options_fields"),
    "signature_params.tsv": lambda: _render_census_baseline("census_signature_params"),
    "memoization_sites.tsv": lambda: _render_census_baseline("census_memoization_sites"),
}


def main(argv: list[str]) -> int:
    """Regenerate (or check) every walk-derived baseline.

    Parameters
    ----------
    argv:
        CLI args; ``--check`` verifies without writing.

    Returns
    -------
    int
        0 when clean; 1 when ``--check`` found drift.
    """

    check = "--check" in argv
    DATA_DIR.mkdir(exist_ok=True)
    drift = 0
    for filename, render in BASELINES.items():
        text = render()  # type: ignore[operator]
        target = DATA_DIR / filename
        current = target.read_text() if target.exists() else None
        if current == text:
            print(f"unchanged: {filename}")
            continue
        if check:
            print(f"DRIFT: {filename}")
            drift = 1
        else:
            target.write_text(text)
            print(f"wrote: {filename}")
    return drift


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
