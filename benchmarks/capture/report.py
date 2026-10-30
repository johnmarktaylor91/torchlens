"""Generate markdown tables from capture benchmark result documents.

``census`` documents give the "what hooks miss" table by architecture family, with a
second share that leaves out view, indexing and shape operations (so the count is not
carried by bookkeeping operations), and the fused or opaque boundaries seen. ``capture``
documents give the Suite A matrix, the Suite B rows with what each keeps, and the
fixture sweep with host time per operation. Unsupported, refused and failing cells are
printed with their reasons.

    python benchmarks/capture/report.py census.json capture.json > tables.md
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

VIEW_LIKE = frozenset(
    {
        "view",
        "reshape",
        "permute",
        "transpose",
        "__getitem__",
        "flatten",
        "unsqueeze",
        "squeeze",
        "expand",
        "contiguous",
        "size",
        "t",
        "chunk",
        "split",
        "unbind",
        "view_as",
        "expand_as",
        "unflatten",
        "narrow",
        "select",
        "repeat",
        "type",
        "to",
        "float",
        "detach",
        "clone",
    }
)


def substantive_share(counts: dict[str, Any]) -> float | None:
    """Share of non-view operations that no module hook can reach."""

    total = sum(n for f, n in counts["func_counts"].items() if f not in VIEW_LIKE)
    missed = sum(n for f, n in counts["missed_funcs"].items() if f not in VIEW_LIKE)
    return missed / total if total else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def census_tables(doc: dict[str, Any]) -> list[str]:
    """Return the census table lines."""

    ok = [c for c in doc["cells"] if c["status"] == "ok"]
    families: dict[str, list[dict[str, Any]]] = {}
    for cell in ok:
        families.setdefault(cell["family"], []).append(cell)
    lines = [
        "| Family | Models | Median share unreachable | Pooled | Median, views excluded | "
        "Models with any | Replay passed |",
        "|---|---|---|---|---|---|---|",
    ]
    for family, rows in sorted(families.items()):
        shares = [r["counts"]["share_unreachable_any_hook"] for r in rows]
        subst = [s for r in rows if (s := substantive_share(r["counts"])) is not None]
        comp = sum(r["counts"]["computations"] for r in rows)
        missed = sum(r["counts"]["unreachable_any_hook"] for r in rows)
        passed = sum(1 for r in rows if r["validation"]["passed"])
        lines.append(
            f"| {family} | {len(rows)} | {_pct(statistics.median(shares))} | {_pct(missed / comp)} "
            f"| {_pct(statistics.median(subst) if subst else None)} "
            f"| {sum(1 for s in shares if s > 0)} | {passed}/{len(rows)} |"
        )
    shares = [r["counts"]["share_unreachable_any_hook"] for r in ok]
    subst = [s for r in ok if (s := substantive_share(r["counts"])) is not None]
    lines.append(
        f"| all | {len(ok)} | {_pct(statistics.median(shares))} | "
        f"{_pct(sum(r['counts']['unreachable_any_hook'] for r in ok) / sum(r['counts']['computations'] for r in ok))} "
        f"| {_pct(statistics.median(subst))} | {sum(1 for s in shares if s > 0)} "
        f"| {sum(1 for r in ok if r['validation']['passed'])}/{len(ok)} |"
    )
    failed = [c for c in doc["cells"] if c["status"] != "ok" or not c["validation"]["passed"]]
    lines += ["", f"Not captured or replay failed: {len(failed)}"]
    for cell in failed:
        reason = cell.get("reason") or cell.get("validation", {}).get("failure", "")
        lines.append(f"- {cell['subject']}: {str(reason)[:200]}")
    return [*lines, "", *boundary_lines(ok)]


def boundary_lines(cells: list[dict[str, Any]]) -> list[str]:
    """List fused and opaque boundaries: models affected and calls seen."""

    models: Counter[str] = Counter()
    calls: Counter[str] = Counter()
    kinds: dict[str, str] = {}
    for cell in cells:
        for name, (count, kind) in cell["counts"]["fused_boundaries"].items():
            models[name] += 1
            calls[name] += count
            kinds[name] = kind
        for name, count in cell.get("opaque_calls", {}).items():
            key = f"opaque: {name}"
            models[key] += 1
            calls[key] += count
            kinds[key] = "recorded as one operation; its inside is not captured"
    lines = ["| Boundary | What is not visible | Models | Calls |", "|---|---|---|---|"]
    for name, n in models.most_common():
        lines.append(f"| `{name}` | {kinds[name]} | {n} | {calls[name]} |")
    return lines


def capture_tables(doc: dict[str, Any]) -> list[str]:
    """Return the Suite A, Suite B and fixture sweep lines."""

    groups = {g["cell_id"]: g for g in doc.get("groups", [])}
    lines = [f"A/A noise: {json.dumps(doc.get('aa_noise'))}", ""]
    lines += ["| Suite | Case | Arm | Subject | Status | Median ms | ms/op | Exact | Note |"]
    lines += ["|---|---|---|---|---|---|---|---|---|"]
    seen: set[str] = set()
    for cell in doc["cells"]:
        if cell["cell_id"] in seen:
            continue
        seen.add(cell["cell_id"])
        group = groups.get(cell["cell_id"], {})
        median = group.get("median_ms")
        exact = cell.get("exact", {})
        exact_text = (
            ""
            if not exact
            else ("bit" if exact.get("bit_identical") else str(exact.get("matched")))
        )
        note = cell.get("reason") or cell.get("keeps") or ""
        if group.get("noisy"):
            note = f"NOISY (spread {group['process_spread']:.0%}); {note}"
        per_op = cell.get("ms_per_op")
        lines.append(
            f"| {cell['suite']} | {cell['case']} | {cell['arm']} | {cell['subject']} | {cell['status']} "
            f"| {'' if median is None else f'{median:.2f}'} | {'' if per_op is None else f'{per_op:.3f}'} "
            f"| {exact_text} | {str(note)[:140]} |"
        )
    return lines


def main() -> None:
    """Print tables for every document named on the command line."""

    for path in sys.argv[1:]:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        print(f"## {Path(path).name} ({doc['suite']})\n")
        lines = census_tables(doc) if doc["suite"] == "census" else capture_tables(doc)
        print("\n".join(lines))
        print()


if __name__ == "__main__":
    main()
