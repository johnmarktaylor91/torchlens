"""Regenerate every generated ledger in one deterministic step.

Run from anywhere in a checkout::

    python scripts/regen_ledgers.py                 # write the regenerated ledgers
    python scripts/regen_ledgers.py --check         # exit 1 when any ledger would change
    python scripts/regen_ledgers.py --report        # per-ledger row counts, before and after
    python scripts/regen_ledgers.py --base REF --verify-deltas A.tsv B.tsv

The ledgers are the committed files a regenerator derives from the tree (the
where-index page, the file-cap allowlist, the census and ratchet TSVs, the
oracle baselines and the producer static ledger). They run in a fixed
dependency order: a later ledger may read an earlier one (the harvest and
numeral censuses read ``docs/maintainers/where.md``). The step writes those
files and nothing else, and the same tree always gives the same bytes.

Refusal (exit 2). A regen that would RAISE a ceiling or ratchet, lower a
floor, or add an allowlist row is a defect in the change under review, never
a ledger edit: nothing is written and the ledger and row are named. With
``--base REF`` the reference value is the one at ``REF`` (pass the integration
base, so a branch that edited a ratchet upward is still caught); without it
the committed value is the reference. The S-17/S-18 totals (and their
universe rows) may rise only when every added site is coded and classified,
that is when the matching uncoded and unclassified counts do not rise.

Union-merge duplicates. A keyed TSV that a ``merge=union`` merge left with a
key on two rows is collapsed to one row holding the regenerated value (the
base value plus the combined deltas); each collapse is reported with the
values it replaced and the base value.

Host-sensitive ledgers (the numeral baseline) regenerate only in the pinned
environment (python 3.10, torch 2.13.0). Ledgers that import torch are skipped
when torch is absent. Rendered images (``docs/images/``) are never touched.

Deltas. A branch declares the ledger rows it moves in a tab-separated file
with the header ``ledger key train_value lane_value reason`` (``-`` marks an
absent row). ``--verify-deltas`` compares the regen against ``--base`` and
names every change no file declares and every declaration the regen does not
show. Numeric values add up across files (base plus the sum of each file's
``lane - train``). Ledgers marked ``auto`` (the where-index page, allowlist
tightenings, the numeral baseline, provenance records) need no declaration.

Exit codes: 0 clean or written; 1 ``--check`` drift or undeclared deltas;
2 refusal, an unresolvable duplicate, or a bad delta file.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PINNED = ((3, 10), "2.13.0")
NEVER_TOUCHED = ("docs/images/",)
ABSENT = "-"
DELTA_HEADER = ("ledger", "key", "train_value", "lane_value", "reason")


Context = dict[str, int]
Guard = Callable[[dict[str, str], dict[str, str], Context, Context], list[str]]


class LedgerRefusal(Exception):
    """A regen would loosen a guard or cannot resolve a ledger safely."""


@dataclass(frozen=True)
class Ledger:
    """One generated ledger: where it lives and how it is regenerated and guarded.

    ``policy`` is ``inventory`` (any change, declared), ``guarded`` (rows checked
    by ``guard``, declared) or ``auto`` (regenerated freely, never declared).
    ``requires`` is ``""``, ``"torch"`` or ``"pinned"``. ``free_shrink`` exempts
    lowered or dropped rows from declaration (allowlist tightening). ``union``
    marks a ``merge=union`` file whose duplicate rows are collapsed and reported.
    """

    name: str
    path: str
    render: Callable[[Path], str]
    rows: Callable[[str], dict[str, list[str]]]
    policy: str = "inventory"
    requires: str = ""
    guard: Guard | None = None
    free_shrink: bool = False
    union: bool = False


@dataclass
class Outcome:
    """What the run did to one ledger."""

    ledger: Ledger
    before: str | None = None
    after: str | None = None
    skipped: str = ""
    dedupes: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.after is not None and self.after != self.before


# --------------------------------------------------------------------------- rows


def keyed_tsv_rows(text: str) -> dict[str, list[str]]:
    """Parse ``key<TAB>value`` rows into key -> every value seen (duplicates kept)."""

    rows: dict[str, list[str]] = {}
    header_seen = False
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if not header_seen:
            header_seen = True
            continue
        key, _, value = line.partition("\t")
        rows.setdefault(key.strip(), []).append(value.strip())
    return rows


def line_rows(text: str) -> dict[str, list[str]]:
    """Every non-comment line after the header is a key that is simply present."""

    rows: dict[str, list[str]] = {}
    lines = [line for line in text.splitlines() if line.strip() and not line.startswith("#")]
    for line in lines[1:]:
        rows.setdefault(line, []).append("present")
    return rows


def allowlist_rows(text: str) -> dict[str, list[str]]:
    """File-cap allowlist entries: ``path[::qualname]`` -> recorded line count."""

    rows: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, _, count = line.rpartition("\t")
            rows.setdefault(key, []).append(count)
    return rows


def markdown_table_rows(text: str) -> dict[str, list[str]]:
    """Table rows of a generated markdown page; each distinct row is a key."""

    rows: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("| ") and not line.startswith("| ---"):
            rows.setdefault(line, []).append("present")
    return rows


def static_scan_rows(text: str) -> dict[str, list[str]]:
    """Producer static scan: one key per (file, field, flags) with its site count."""

    rows: dict[str, list[str]] = {}
    for entry in json.loads(text) if text.strip() else []:
        flags = ",".join(
            f"{k}={entry[k]}" for k in sorted(entry) if k not in {"file", "field", "sites"}
        )
        rows.setdefault(f"{entry['file']}::{entry['field']}[{flags}]", []).append(
            str(entry["sites"])
        )
    return rows


def mutator_rows(text: str) -> dict[str, list[str]]:
    """Producer mutator inventory: ``channel::file`` -> site count."""

    rows: dict[str, list[str]] = {}
    for channel, files in sorted((json.loads(text) if text.strip() else {}).items()):
        for file, count in sorted(files.items()):
            rows.setdefault(f"{channel}::{file}", []).append(str(count))
    return rows


def provenance_rows(text: str) -> dict[str, list[str]]:
    """Provenance records, keyed by their position (an append-only log)."""

    records = [chunk.strip() for chunk in text.split("\n---\n") if chunk.strip()]
    return {str(index): [record] for index, record in enumerate(records)}


def single(rows: dict[str, list[str]]) -> dict[str, str]:
    """Collapse each key to its last value (callers check duplicates first)."""

    return {key: values[-1] for key, values in rows.items()}


# ------------------------------------------------------------------- rendering


def rewrite_keyed_tsv(text: str, live: dict[str, str]) -> str:
    """Rewrite a keyed TSV with ``live`` values, keeping comments, header and order.

    Duplicate keys collapse to one row; keys absent from ``live`` are dropped and
    new keys are appended in ``live`` order.
    """

    out: list[str] = []
    written: set[str] = set()
    header_seen = False
    for line in text.splitlines():
        if not line.strip() or line.startswith("#") or not header_seen:
            header_seen = header_seen or bool(line.strip() and not line.startswith("#"))
            out.append(line)
            continue
        key = line.partition("\t")[0].strip()
        if key in live and key not in written:
            out.append(f"{key}\t{live[key]}")
            written.add(key)
    out.extend(f"{key}\t{value}" for key, value in live.items() if key not in written)
    return "\n".join(out) + "\n"


def _read(repo: Path, rel: str) -> str:
    path = repo / rel
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _import_paths(repo: Path) -> None:
    for entry in (repo / "scripts" / "checks", repo / "tests", repo):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))


def render_file_cap(repo: Path) -> str:
    import file_cap

    measured, allowed = file_cap.measure(repo), file_cap.read_allowlist(repo)
    kept = {key: min(lines, allowed.get(key, lines)) for key, lines in measured.items()}
    body = "".join(f"{key}\t{lines}\n" for key, lines in sorted(kept.items()))
    return (
        file_cap.HEADER.format(file_cap=file_cap.FILE_CAP, function_cap=file_cap.FUNCTION_CAP)
        + body
    )


def render_where(repo: Path) -> str:
    sys.path.insert(0, str(repo / "tools" / "docs"))
    import where_index

    return where_index.render(where_index.collect_packages())


def render_harvest(repo: Path) -> str:
    from tests.composition_expectations.test_harvest_spike import harvest_counts

    path = "tests/composition_expectations/data/harvest_counts.tsv"
    return rewrite_keyed_tsv(_read(repo, path), {k: str(v) for k, v in harvest_counts().items()})


def _diagnostic_counts() -> dict[str, int]:
    from tests.composition_expectations import _censuses
    from tests.composition_expectations.test_diagnostic_site_contract import _classification_rows
    from tests.composition_expectations.test_warning_disclosures import (
        WARNING_CONTRACT_DOC,
        _contract_table_codes,
    )

    raises, warns = _censuses.raise_sites(), _censuses.warn_sites()
    classified = set(_classification_rows())
    table = _contract_table_codes(WARNING_CONTRACT_DOC.read_text(encoding="utf-8"))
    return {
        "s17_total_sites": len(raises),
        "s17_uncoded_sites": sum(1 for site in raises if not site.has_code),
        "s17_unclassified_sites": sum(1 for site in raises if site.site_key not in classified),
        "s18_total_sites": len(warns),
        "s18_uncoded_sites": sum(1 for site in warns if not site.has_code),
        "s18_contract_rows": len(table),
    }


def render_ratchets(repo: Path) -> str:
    path = "tests/composition_expectations/data/diagnostic_ratchets.tsv"
    live = {k: str(v) for k, v in _diagnostic_counts().items()}
    return rewrite_keyed_tsv(_read(repo, path), live)


def render_universes(repo: Path) -> str:
    """Universe cardinalities; a universe that needs torch keeps its committed value."""

    from tests.composition_expectations import universes

    path = "tests/composition_expectations/data/universe_cardinalities.tsv"
    committed = single(keyed_tsv_rows(_read(repo, path)))
    live: dict[str, str] = {}
    for universe in universes.UNIVERSES:
        if universe.derive is None:
            live[universe.universe_id] = "OPEN"
            continue
        try:
            live[universe.universe_id] = str(len(universe.derive()))
        except ModuleNotFoundError as error:
            if error.name != "torch" or universe.universe_id not in committed:
                raise
            live[universe.universe_id] = committed[universe.universe_id]
    return rewrite_keyed_tsv(_read(repo, path), live)


def render_oracle(filename: str) -> Callable[[Path], str]:
    def render(repo: Path) -> str:
        from oracles import _regen

        return _regen.BASELINES[filename]()  # type: ignore[operator]

    return render


def _producer() -> object:
    from tests.producer_parity import test_ledger

    return test_ledger


def render_static_scan(repo: Path) -> str:
    ledger = _producer()
    sites = ledger.static_scan(ledger._PACKAGE_ROOT)  # type: ignore[attr-defined]
    return ledger.artifact_payload(ledger.normalized_static_scan(sites))  # type: ignore[attr-defined]


def render_mutators(repo: Path) -> str:
    ledger = _producer()
    mutators = ledger.mutator_inventory(ledger._PACKAGE_ROOT)  # type: ignore[attr-defined]
    return ledger.artifact_payload(ledger.normalized_mutators(mutators))  # type: ignore[attr-defined]


# ------------------------------------------------------------------------ guards


def _ints(rows: dict[str, str]) -> dict[str, int]:
    return {key: int(value) for key, value in rows.items() if value.lstrip("-").isdigit()}


def guard_allowlist(
    reference: dict[str, str], live: dict[str, str], _ref: Context, _live: Context
) -> list[str]:
    """No allowlist row is added and no recorded count rises."""

    ref = _ints(reference)
    problems = []
    for key, lines in sorted(_ints(live).items()):
        if key not in ref:
            problems.append(f"{key}: new allowlist row ({lines} lines over the cap)")
        elif lines > ref[key]:
            problems.append(f"{key}: {ref[key]} -> {lines} (ceiling raised)")
    return problems


#: shrink-only metrics; a total may rise only when its companions do not.
_CEILINGS = ("s17_uncoded_sites", "s17_unclassified_sites", "s18_uncoded_sites")
_FLOORS = ("s18_contract_rows",)
_TOTALS = {
    "s17_total_sites": ("s17_uncoded_sites", "s17_unclassified_sites"),
    "s18_total_sites": ("s18_uncoded_sites",),
    "U-RAISE-SITES": ("s17_uncoded_sites", "s17_unclassified_sites"),
    "U-WARN-SITES": ("s18_uncoded_sites",),
}


def guard_counts(
    reference: dict[str, str], live: dict[str, str], ctx_ref: Context, ctx_live: Context
) -> list[str]:
    """Ceilings never rise, floors never fall, totals rise only fully coded and classified.

    ``ctx_ref`` and ``ctx_live`` carry the counts of the ledgers already run, so
    the universe rows see the S-17/S-18 companions from the ratchet ledger.
    """

    ref, now = _ints(reference), _ints(live)
    companions, companions_now = {**ctx_ref, **ref}, {**ctx_live, **now}
    problems = []
    for key in sorted(set(ref) & set(now)):
        if key in _CEILINGS and now[key] > ref[key]:
            problems.append(f"{key}: {ref[key]} -> {now[key]} (ratchet raised)")
        elif key in _FLOORS and now[key] < ref[key]:
            problems.append(f"{key}: {ref[key]} -> {now[key]} (floor lowered)")
        elif key in _TOTALS and now[key] > ref[key]:
            grown = [
                f"{name} {companions[name]} -> {companions_now[name]}"
                for name in _TOTALS[key]
                if companions_now.get(name, 0) > companions.get(name, companions_now.get(name, 0))
            ]
            if grown:
                problems.append(
                    f"{key}: {ref[key]} -> {now[key]} with uncoded or unclassified growth "
                    f"({'; '.join(grown)}): code and classify the new sites"
                )
    return problems


LEDGERS: tuple[Ledger, ...] = (
    Ledger(
        "file-cap",
        "scripts/checks/file_cap_allowlist.txt",
        render_file_cap,
        allowlist_rows,
        "guarded",
        guard=guard_allowlist,
        free_shrink=True,
    ),
    Ledger("where", "docs/maintainers/where.md", render_where, markdown_table_rows, "auto"),
    Ledger(
        "harvest",
        "tests/composition_expectations/data/harvest_counts.tsv",
        render_harvest,
        keyed_tsv_rows,
        union=True,
    ),
    Ledger(
        "ratchets",
        "tests/composition_expectations/data/diagnostic_ratchets.tsv",
        render_ratchets,
        keyed_tsv_rows,
        "guarded",
        guard=guard_counts,
        union=True,
    ),
    Ledger(
        "universes",
        "tests/composition_expectations/data/universe_cardinalities.tsv",
        render_universes,
        keyed_tsv_rows,
        "guarded",
        guard=guard_counts,
        union=True,
    ),
    Ledger(
        "numerals",
        "tests/oracles/data/numeral_baseline.tsv",
        render_oracle("numeral_baseline.tsv"),
        line_rows,
        "auto",
        "pinned",
        union=True,
    ),
    Ledger(
        "surface",
        "tests/oracles/data/surface_module_classification.tsv",
        render_oracle("surface_module_classification.tsv"),
        keyed_tsv_rows,
        requires="torch",
        union=True,
    ),
    Ledger(
        "options-fields",
        "tests/oracles/data/options_fields.tsv",
        render_oracle("options_fields.tsv"),
        line_rows,
        requires="torch",
    ),
    Ledger(
        "signature-params",
        "tests/oracles/data/signature_params.tsv",
        render_oracle("signature_params.tsv"),
        line_rows,
        requires="torch",
    ),
    Ledger(
        "memoization-sites",
        "tests/oracles/data/memoization_sites.tsv",
        render_oracle("memoization_sites.tsv"),
        line_rows,
        requires="torch",
    ),
    Ledger(
        "static-scan",
        "tests/producer_parity/ledger/static_scan.json",
        render_static_scan,
        static_scan_rows,
        requires="torch",
        union=True,
    ),
    Ledger(
        "mutators",
        "tests/producer_parity/ledger/mutators.json",
        render_mutators,
        mutator_rows,
        requires="torch",
    ),
)
PROVENANCE = Ledger(
    "provenance",
    "tests/producer_parity/ledger/PROVENANCE",
    lambda repo: "",
    provenance_rows,
    "auto",
    "torch",
)
_PRODUCER_LEDGERS = ("static-scan", "mutators")


# ------------------------------------------------------------------------ engine


def environment() -> dict[str, str]:
    """Torch version (empty when absent) and whether this is the pinned environment."""

    try:
        torch = importlib.metadata.version("torch") if importlib.util.find_spec("torch") else ""
    except importlib.metadata.PackageNotFoundError:
        torch = ""
    pinned = sys.version_info[:2] == PINNED[0] and torch.split("+")[0] == PINNED[1]
    return {"torch": torch, "pinned": "yes" if pinned else ""}


def skip_reason(ledger: Ledger, env: dict[str, str]) -> str:
    if ledger.requires == "pinned" and not env["pinned"]:
        version = (
            f"python {sys.version_info[0]}.{sys.version_info[1]}, torch {env['torch'] or 'absent'}"
        )
        return f"host-sensitive: regenerates only in python 3.10 + torch 2.13.0 ({version})"
    if ledger.requires in {"torch", "pinned"} and not env["torch"]:
        return "needs torch"
    return ""


def git(repo: Path, *args: str) -> str | None:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def reference_text(repo: Path, ledger: Ledger, base: str | None, committed: str) -> str:
    if base is None:
        return committed
    text = git(repo, "show", f"{base}:{ledger.path}")
    return "" if text is None else text


def collapse(rows: dict[str, list[str]], prefer_low: bool = True) -> dict[str, str]:
    """One value per key; a union duplicate keeps its lowest number (shrink-only doctrine)."""

    out = {}
    for key, values in rows.items():
        numeric = [v for v in values if v.lstrip("-").isdigit()]
        if len(values) > 1 and len(numeric) == len(values):
            out[key] = str(min(map(int, numeric)) if prefer_low else max(map(int, numeric)))
        else:
            out[key] = values[-1]
    return out


def safe_rows(ledger: Ledger, text: str) -> tuple[dict[str, list[str]], str]:
    try:
        return ledger.rows(text), ""
    except (ValueError, KeyError, TypeError) as error:
        return {}, f"unparseable ({type(error).__name__}: {error}); likely a union merge"


@dataclass
class Run:
    """Every ledger outcome plus the refusals and notes of one run."""

    outcomes: list[Outcome] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)


def run_ledger(
    repo: Path, ledger: Ledger, base: str | None, ctx: tuple[Context, Context], run: Run
) -> Outcome:
    outcome = Outcome(ledger, before=_read(repo, ledger.path))
    before_rows, problem = safe_rows(ledger, outcome.before or "")
    if problem:
        outcome.dedupes.append(f"committed file {problem}: regenerated whole")
    ref_text = reference_text(repo, ledger, base, outcome.before or "")
    ref_rows, ref_problem = safe_rows(ledger, ref_text)
    if ref_problem and ledger.guard is not None:
        run.refusals.append(f"{ledger.path}: reference {ref_problem}; cannot check the guard")
    outcome.after = ledger.render(repo)
    after_rows = single(ledger.rows(outcome.after))
    for key, values in sorted(before_rows.items()) if ledger.union else ():
        if len(values) > 1:
            base_value = collapse(ref_rows).get(key, ABSENT) if base else "n/a"
            outcome.dedupes.append(
                f"{key}: union duplicate rows {values[:4]} -> {after_rows.get(key, ABSENT)} (base {base_value})"
            )
    reference = collapse(ref_rows)
    if ledger.guard is not None:
        problems = ledger.guard(reference, after_rows, ctx[0], ctx[1])
        run.refusals.extend(f"{ledger.path}: {problem}" for problem in problems)
    ctx[0].update(_ints(reference))
    ctx[1].update(_ints(after_rows))
    (repo / ledger.path).parent.mkdir(parents=True, exist_ok=True)
    (repo / ledger.path).write_text(outcome.after, encoding="utf-8")
    return outcome


def provenance_record(repo: Path, reason: str, env: dict[str, str]) -> str:
    head = (git(repo, "rev-parse", "HEAD") or "unknown").strip()
    when = (git(repo, "log", "-1", "--format=%cI", "HEAD") or "unknown").strip()
    major, minor = sys.version_info[:2]
    return (
        "---\n"
        "generator: scripts/regen_ledgers.py (producer static half)\n"
        "flag: none (deterministic ledger step)\n"
        f"reason: {reason}\n"
        f"env: py{major}.{minor}-torch{env['torch'].split('+')[0]}\n"
        f"torch: {env['torch']}\n"
        f"source: {head}\n"
        f"recorded: {when}\n"
    )


def regenerate(
    repo: Path,
    ledgers: Iterable[Ledger],
    *,
    base: str | None = None,
    reason: str = "",
    keep: bool = True,
    env: dict[str, str] | None = None,
) -> Run:
    """Regenerate ``ledgers`` in order; restore every file unless ``keep`` and no refusal.

    Later ledgers read earlier ones from disk, so files are written as the run
    goes and restored from the snapshot when the run is a check or refused.
    """

    env = env if env is not None else environment()
    run, ctx = Run(), ({}, {})
    snapshot: dict[Path, str | None] = {}
    try:
        for ledger in ledgers:
            if ledger.path.startswith(NEVER_TOUCHED):
                raise LedgerRefusal(f"{ledger.path}: rendered images are never regenerated here")
            skipped = skip_reason(ledger, env)
            if skipped:
                run.outcomes.append(Outcome(ledger, skipped=skipped))
                continue
            path = repo / ledger.path
            snapshot.setdefault(path, path.read_text(encoding="utf-8") if path.is_file() else None)
            run.outcomes.append(run_ledger(repo, ledger, base, ctx, run))
        _append_provenance(repo, run, reason, env, snapshot)
    finally:
        if run.refusals or not keep:
            for path, text in snapshot.items():
                if text is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(text, encoding="utf-8")
    return run


def _append_provenance(
    repo: Path, run: Run, reason: str, env: dict[str, str], snapshot: dict[Path, str | None]
) -> None:
    moved = [
        o.ledger.name for o in run.outcomes if o.changed and o.ledger.name in _PRODUCER_LEDGERS
    ]
    if not moved:
        return
    path = repo / PROVENANCE.path
    before = path.read_text(encoding="utf-8") if path.is_file() else ""
    snapshot.setdefault(path, before)
    head = (git(repo, "rev-parse", "--short=9", "HEAD") or "?").strip()
    why = reason or f"ledger-owner regen at {head}: {', '.join(moved)} moved"
    after = (
        before
        + ("" if not before or before.endswith("\n") else "\n")
        + provenance_record(repo, why, env)
    )
    path.write_text(after, encoding="utf-8")
    run.outcomes.append(Outcome(PROVENANCE, before=before, after=after))


# ------------------------------------------------------------------------ deltas


@dataclass(frozen=True)
class Delta:
    ledger: str
    key: str
    train: str
    lane: str
    reason: str
    source: str


def load_deltas(paths: Iterable[Path]) -> list[Delta]:
    """Read lane delta files; a malformed file raises :class:`LedgerRefusal`."""

    deltas = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip() or line.startswith("#"):
                continue
            cells = line.split("\t")
            if tuple(cells) == DELTA_HEADER:
                continue
            if len(cells) != len(DELTA_HEADER) or not cells[4].strip():
                raise LedgerRefusal(
                    f"{path}:{number}: want 5 tab-separated cells with a reason: {line!r}"
                )
            deltas.append(Delta(*(cell.strip() for cell in cells), source=f"{path}:{number}"))
    return deltas


def _numeric(value: str) -> int | None:
    if value == ABSENT:
        return 0
    return int(value) if value.lstrip("-").isdigit() else None


def expected_value(base_value: str, declared: list[Delta]) -> str:
    """Base plus each lane's numeric delta, or the agreed non-numeric lane value."""

    numbers = [_numeric(d.train) for d in declared] + [_numeric(d.lane) for d in declared]
    if _numeric(base_value) is not None and all(n is not None for n in numbers):
        total = _numeric(base_value) + sum(_numeric(d.lane) - _numeric(d.train) for d in declared)  # type: ignore[operator]
        return ABSENT if total == 0 and all(d.lane == ABSENT for d in declared) else str(total)
    values = {d.lane for d in declared}
    if len(values) > 1:
        return "conflict: " + " vs ".join(f"{d.lane} ({d.source})" for d in declared)
    return values.pop()


def verify_deltas(repo: Path, run: Run, base: str, deltas: list[Delta]) -> list[str]:
    """Name every regen change no delta declares and every declaration the regen lacks."""

    by_name = {o.ledger.name: o for o in run.outcomes}
    by_name.update({o.ledger.path: o for o in run.outcomes})
    problems = [
        f"{d.source}: unknown ledger {d.ledger!r}" for d in deltas if d.ledger not in by_name
    ]
    for outcome in run.outcomes:
        ledger = outcome.ledger
        if ledger.policy == "auto" or outcome.skipped or outcome.after is None:
            continue
        declared: dict[str, list[Delta]] = {}
        for delta in deltas:
            if delta.ledger in {ledger.name, ledger.path}:
                declared.setdefault(delta.key, []).append(delta)
        base_rows = collapse(safe_rows(ledger, reference_text(repo, ledger, base, ""))[0])
        new_rows = single(ledger.rows(outcome.after))
        for key in sorted(set(base_rows) | set(new_rows) | set(declared)):
            old, new = base_rows.get(key, ABSENT), new_rows.get(key, ABSENT)
            if key not in declared:
                free = ledger.free_shrink and (
                    new == ABSENT or (_numeric(new) or 0) < (_numeric(old) or 0)
                )
                if old != new and not free:
                    problems.append(f"{ledger.name}: undeclared change {key}: {old} -> {new}")
                continue
            want = expected_value(old, declared[key])
            if want != new:
                sources = ", ".join(d.source for d in declared[key])
                problems.append(
                    f"{ledger.name}: {key}: declared {want} ({sources}) but the regen gives {new}"
                )
    return problems


# ------------------------------------------------------------------------ output


def _row_diff(outcome: Outcome) -> tuple[int, int, int]:
    before = single(safe_rows(outcome.ledger, outcome.before or "")[0])
    after = single(outcome.ledger.rows(outcome.after or ""))
    added = len(set(after) - set(before))
    removed = len(set(before) - set(after))
    moved = sum(1 for key in set(before) & set(after) if before[key] != after[key])
    return added, removed, moved


def summary_lines(run: Run, report: bool) -> list[str]:
    lines = []
    for outcome in run.outcomes:
        name = f"{outcome.ledger.name:<18} {outcome.ledger.path}"
        if outcome.skipped:
            lines.append(f"skip   {name}: {outcome.skipped}")
            continue
        added, removed, moved = _row_diff(outcome)
        state = "change" if outcome.changed else "same  "
        detail = f" (+{added} -{removed} ~{moved} rows)" if outcome.changed else ""
        lines.append(f"{state} {name}{detail}")
        lines.extend(f"         dedupe {note}" for note in outcome.dedupes)
        if report:
            lines.extend(_report_lines(outcome))
    return lines


def _report_lines(outcome: Outcome) -> list[str]:
    before = single(safe_rows(outcome.ledger, outcome.before or "")[0])
    after = single(outcome.ledger.rows(outcome.after or ""))
    lines = [f"         rows {len(before)} -> {len(after)}"]
    if outcome.ledger.policy == "guarded":
        old, new = _ints(before), _ints(after)
        lines.append(f"         shrink-only total {sum(old.values())} -> {sum(new.values())}")
        if outcome.ledger.name != "file-cap":
            lines.extend(f"         {key} {old.get(key)} -> {value}" for key, value in new.items())
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 when any ledger would change")
    mode.add_argument(
        "--report", action="store_true", help="row counts and shrink-only totals, writes nothing"
    )
    mode.add_argument(
        "--verify-deltas",
        nargs="+",
        type=Path,
        metavar="DELTA_TSV",
        help="compare the regen with --base against the declared lane deltas",
    )
    parser.add_argument(
        "--base", help="reference ref for guards, dedupes and deltas (the integration base)"
    )
    parser.add_argument(
        "--reason", default="", help="provenance reason when the producer ledger moves"
    )
    parser.add_argument(
        "--only", nargs="+", metavar="NAME", help="run only these ledgers (in order)"
    )
    args = parser.parse_args(argv)
    if args.verify_deltas and not args.base:
        parser.error("--verify-deltas needs --base")
    _import_paths(REPO)
    ledgers = [ledger for ledger in LEDGERS if not args.only or ledger.name in args.only]
    keep = not (args.check or args.report or args.verify_deltas)
    try:
        deltas = load_deltas(args.verify_deltas) if args.verify_deltas else []
        run = regenerate(REPO, ledgers, base=args.base, reason=args.reason, keep=keep)
    except LedgerRefusal as refusal:
        print(f"REFUSED: {refusal}")
        return 2
    print("\n".join(summary_lines(run, report=args.report or bool(args.verify_deltas))))
    if run.refusals:
        print(
            "REFUSED (nothing written): a regen would loosen a guard; fix the source, not the ledger"
        )
        print("\n".join(f"  {line}" for line in run.refusals))
        return 2
    if args.verify_deltas:
        problems = verify_deltas(REPO, run, args.base, deltas)
        print("\n".join(f"DELTA {line}" for line in problems) or "deltas: every change declared")
        return 1 if problems else 0
    changed = [o.ledger.name for o in run.outcomes if o.changed]
    if args.check:
        print(
            f"check: {len(changed)} ledger(s) would change: {', '.join(changed)}"
            if changed
            else "check: clean"
        )
        return 1 if changed else 0
    if not args.report:
        print(f"wrote {len(changed)} ledger(s)" + (f": {', '.join(changed)}" if changed else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
