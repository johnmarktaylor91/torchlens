"""Ledger files that merge as a union fail loudly on the shapes a union merge leaves.

``merge=union`` keeps both sides' lines when two branches change the same spot,
and git's merge first collapses the lines the two sides share. For one-line
keyed rows that leaves two rows for one key, and two branches editing
NEIGHBOURING keys bring the base row back as well (``k1'``, ``k2``, ``k1``,
``k2'``), so a last-wins loader silently reverts one branch's change. For
multi-line records the shared lines collapse: two JSON objects inserted at one
spot become one object holding two ``"file"`` keys, and two appended
PROVENANCE entries become one entry missing the second header.

Each check below parses one artifact strictly enough to refuse those shapes,
runs against the committed file, and is proven red on a real ``git merge``
with the union driver in a scratch repository. The registry test keeps the
union set in ``.gitattributes`` equal to the files these checks make safe: a
new ``merge=union`` entry needs a check here first. Generated and prose files
(``static_scan.json``, ``PROVENANCE``, the glossary, ``AGENTS.md``,
``CLAUDE.md``) do not merge as a union; a concurrent edit there is a normal
conflict (regenerate the JSON; keep both PROVENANCE entries whole).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.composition_expectations._keyed_tsv import parse_keyed_tsv

REPO_ROOT = Path(__file__).resolve().parents[1]

Check = Callable[[str, str], None]


def _tsv_rows(text: str) -> list[tuple[int, list[str]]]:
    """Return ``(line number, cells)`` for every data row, header and comments skipped."""

    rows: list[tuple[int, list[str]]] = []
    header_seen = False
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        if not header_seen:
            header_seen = True
            continue
        rows.append((number, line.split("\t")))
    return rows


def unique_tsv_key(key_columns: int | None) -> Check:
    """Build a check refusing a repeated primary key (the first ``key_columns`` cells).

    ``None`` keys on the whole row, for ledgers whose rows are plain members of a set.
    """

    def check(text: str, name: str) -> None:
        seen: dict[tuple[str, ...], int] = {}
        for number, cells in _tsv_rows(text):
            key = tuple(cells if key_columns is None else cells[:key_columns])
            if key in seen:
                raise AssertionError(
                    f"{name}: key {key!r} appears on lines {seen[key]} and {number}. A union "
                    "merge kept both branches' rows; keep one, recomputed on the merged tree."
                )
            seen[key] = number

    return check


def keyed_count_tsv(header_key: str) -> Check:
    """The count baselines read through their loaders' own duplicate-refusing parser."""

    def check(text: str, name: str) -> None:
        parse_keyed_tsv(text, filename=name, header_key=header_key)

    return check


def _refuse_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    keys = [key for key, _ in pairs]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    if repeated:
        raise AssertionError(
            f"JSON object repeats key(s) {repeated}: a union merge fused two objects; "
            "json.load would keep the last value and drop a row silently"
        )
    return dict(pairs)


_STATIC_SCAN_FIELDS = ("ambiguous", "field", "file", "has_default", "sites", "via_getattr")


def check_static_scan(text: str, name: str) -> None:
    """``static_scan.json`` is the sorted, unique, canonical output of its generator."""

    rows = json.loads(text, object_pairs_hook=_refuse_duplicate_json_keys)
    assert isinstance(rows, list), f"{name}: top level is not a list"
    for index, row in enumerate(rows):
        assert tuple(sorted(row)) == _STATIC_SCAN_FIELDS, f"{name}: row {index} keys {sorted(row)}"
    keys = [
        (row["file"], row["field"], row["via_getattr"], row["has_default"], row["ambiguous"])
        for row in rows
    ]
    repeated = sorted({key for key in keys if keys.count(key) > 1})
    assert not repeated, f"{name}: rows repeat {repeated[:5]}"
    assert keys == sorted(keys), f"{name}: rows are not in the generator's sorted order"
    # The serialization of tests/producer_parity/test_ledger.py::artifact_payload.
    canonical = json.dumps(rows, indent=1, sort_keys=True) + "\n"
    assert text == canonical, f"{name}: bytes differ from the generator's canonical form"


_PROVENANCE_WHOLE = (
    ("generator", "flag", "reason", "env", "torch", "source", "recorded"),
    # The four oldest entries predate the ``source`` field.
    ("generator", "flag", "reason", "env", "torch", "recorded"),
)
_PROVENANCE_HEADERLESS = ("reason", "env", "torch", "source", "recorded")

#: The two entries that were committed with no header and no separator when first
#: written (ad09379e6 and cbc939019, 2026-08-19); no commit holds the missing lines.
#: Every other entry lost to a union merge was restored verbatim from history.
_PROVENANCE_BORN_HEADERLESS = frozenset(
    {"2026-08-19T07:52:35.681515+00:00", "2026-08-19T07:59:25.698510+00:00"}
)


def _provenance_entries(block: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Split one ``---``-delimited block into entries, each ending at ``recorded``."""

    entries: list[list[tuple[str, str]]] = [[]]
    for field in block:
        entries[-1].append(field)
        if field[0] == "recorded":
            entries.append([])
    return entries if entries[-1] else entries[:-1]


def check_provenance(text: str, name: str) -> None:
    """Every PROVENANCE entry is whole (header to ``recorded``) and alone in its block."""

    assert text.endswith("\n") and not text.startswith("---"), f"{name}: framing damaged"
    for number, block_text in enumerate(text[:-1].split("\n---\n"), 1):
        block = [line.partition(": ")[::2] for line in block_text.split("\n")]
        entries = _provenance_entries(block)
        for position, entry in enumerate(entries):
            fields = tuple(field for field, _ in entry)
            recorded = entry[-1][1]
            if position == 0 and fields in _PROVENANCE_WHOLE:
                continue
            if fields == _PROVENANCE_HEADERLESS and recorded in _PROVENANCE_BORN_HEADERLESS:
                continue
            raise AssertionError(
                f"{name}: block {number}, entry recorded {recorded!r} has fields {fields}; "
                "each entry needs its own generator/flag header and a --- before it. A merge "
                "fused two appended entries: keep both entries whole, separated by ---."
            )


def check_dict_literal_keys(text: str, name: str) -> None:
    """No dict literal repeats a constant key (Python would keep the last silently)."""

    for node in ast.walk(ast.parse(text)):
        if not isinstance(node, ast.Dict):
            continue
        seen: dict[object, tuple[int, str]] = {}
        for key, value in zip(node.keys, node.values, strict=True):
            if not isinstance(key, ast.Constant):
                continue
            shown = ast.unparse(value)
            if key.value in seen:
                first_line, first_value = seen[key.value]
                raise AssertionError(
                    f"{name}: dict key {key.value!r} appears on lines {first_line} "
                    f"({first_value}) and {key.lineno} ({shown}); Python keeps the last, so a "
                    "union merge chose a ceiling nobody chose. Keep one row, set deliberately."
                )
            seen[key.value] = (key.lineno, shown)


_TABLE_CELL_SPLIT = re.compile(r"(?<!\\)\|")
_CODE_SPAN = re.compile(r"`[^`]*`")
_CODE_CELL = re.compile(r"^`([a-z0-9_]+)`$")


def check_contract_tables(text: str, name: str) -> None:
    """Contract tables keep their header's width and never repeat a code row."""

    width = 0
    codes: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), 1):
        if not line.startswith("|"):
            width = 0
            continue
        # A pipe inside a code span is cell content here (the error contract writes the
        # selector operator as code); only bare pipes split cells.
        masked = _CODE_SPAN.sub(lambda span: span.group(0).replace("|", "\0"), line.strip())
        cells = [cell.strip().replace("\0", "|") for cell in _TABLE_CELL_SPLIT.split(masked)[1:-1]]
        if width == 0:
            width = len(cells)
        assert len(cells) == width, f"{name}:{number}: {len(cells)} cells, header has {width}"
        match = _CODE_CELL.match(cells[0])
        if match is None:
            continue
        code = match.group(1)
        assert code not in codes, f"{name}: code {code!r} rows on lines {codes[code]} and {number}"
        codes[code] = number


_DATA = "tests/composition_expectations/data"
_ORACLES = "tests/oracles/data"
_LEDGER = "tests/producer_parity/ledger"

#: Files that merge as a union, each with the check that makes the union safe.
UNION_CHECKS: dict[str, Check] = {
    f"{_DATA}/diagnostic_ratchets.tsv": keyed_count_tsv("metric"),
    f"{_DATA}/universe_cardinalities.tsv": keyed_count_tsv("universe_id"),
    f"{_DATA}/harvest_counts.tsv": keyed_count_tsv("metric"),
    f"{_DATA}/refusal_site_classification.tsv": unique_tsv_key(1),
    f"{_ORACLES}/surface_module_classification.tsv": unique_tsv_key(1),
    f"{_ORACLES}/numeral_baseline.tsv": unique_tsv_key(None),
    "docs/reference/error_refusal_contract.md": check_contract_tables,
    "docs/reference/warning_contract.md": check_contract_tables,
    "tests/test_file_size_ratchet.py": check_dict_literal_keys,
    "tests/test_ruff_ratchet_governance.py": check_dict_literal_keys,
}

#: Former union files whose concurrent edits must now conflict; still checked, since a
#: hand resolution can leave the same shapes.
CONFLICT_CHECKS: dict[str, Check] = {
    f"{_LEDGER}/static_scan.json": check_static_scan,
    f"{_LEDGER}/PROVENANCE": check_provenance,
}

ALL_CHECKS = {**UNION_CHECKS, **CONFLICT_CHECKS}


def _union_paths(attributes: str) -> set[str]:
    return {
        line.split()[0]
        for line in attributes.splitlines()
        if not line.startswith("#") and "merge=union" in line.split()[1:]
    }


def test_union_set_is_exactly_the_checked_set() -> None:
    attributes = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    union = _union_paths(attributes)
    assert union == set(UNION_CHECKS), (
        f"merge=union entries without a check here: {sorted(union - set(UNION_CHECKS))}; "
        f"checked files no longer union: {sorted(set(UNION_CHECKS) - union)}"
    )


@pytest.mark.parametrize("path", sorted(ALL_CHECKS))
def test_committed_ledger_passes_its_check(path: str) -> None:
    ALL_CHECKS[path]((REPO_ROOT / path).read_text(encoding="utf-8"), path)


Edit = Callable[[str], str]


def _neighbours(lines: list[str], pattern: str) -> tuple[int, int]:
    """Index the first two consecutive lines that both match ``pattern``."""

    hits = [index for index, line in enumerate(lines) if re.fullmatch(pattern, line)]
    pairs = [(a, b) for a, b in zip(hits, hits[1:], strict=False) if b == a + 1]
    assert pairs, f"no two neighbouring rows match {pattern!r}"
    return pairs[0]


def _bump(pattern: str, row: int, delta: int, comment: str = "") -> Edit:
    """Add ``delta`` to the count ending row ``row`` (0 or 1) of the first neighbouring pair.

    Rows are picked by shape, never by value, so the scenario survives every
    re-trued count; ``comment`` adds a line above the row, as a ceiling raise does.
    """

    def edit(text: str) -> str:
        lines = text.split("\n")
        index = _neighbours(lines, pattern)[row]
        lines[index] = re.sub(r"\d+(?=,?$)", lambda m: str(int(m.group()) + delta), lines[index])
        if comment:
            lines.insert(index, comment)
        return "\n".join(lines)

    return edit


def _bump_first(pattern: str, delta: int) -> Edit:
    """Add ``delta`` to the count on the first line matching ``pattern``."""

    def edit(text: str) -> str:
        lines = text.split("\n")
        index = next(i for i, line in enumerate(lines) if re.fullmatch(pattern, line))
        lines[index] = re.sub(r"\d+(?=,?$)", lambda m: str(int(m.group()) + delta), lines[index])
        return "\n".join(lines)

    return edit


def _relabel(pattern: str, cell: int, label: str, sep: str) -> Edit:
    """Prefix cell ``cell`` of the first line matching ``pattern`` with ``label``."""

    def edit(text: str) -> str:
        lines = text.split("\n")
        index = next(i for i, line in enumerate(lines) if re.fullmatch(pattern, line))
        cells = lines[index].split(sep)
        cells[cell] = f"{label}{cells[cell]}" if sep == "\t" else f" {label}{cells[cell]}"
        lines[index] = sep.join(cells)
        return "\n".join(lines)

    return edit


def _insert_after_first(line: str, extra: str) -> Edit:
    def edit(text: str) -> str:
        index = text.index(line) + len(line)
        return text[:index] + extra + text[index:]

    return edit


def _append(extra: str) -> Edit:
    return lambda text: text + extra


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
        + ["-c", "commit.gpgsign=false", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def union_merge(tmp_path: Path, path: str, ours: Edit, theirs: Edit) -> str:
    """Merge two edits of the committed ``path`` with git's union driver; return the result."""

    repo = tmp_path / "repo"
    target = repo / path
    target.parent.mkdir(parents=True)
    _git(tmp_path, "init", "-q", "-b", "base", str(repo))
    _git(repo, "config", "core.hooksPath", "/dev/null")
    (repo / ".gitattributes").write_text(f"{path} merge=union\n", encoding="utf-8")
    base = (REPO_ROOT / path).read_text(encoding="utf-8")
    for branch, edit in (("base", None), ("theirs", theirs), ("ours", ours)):
        if edit is not None:
            _git(repo, "checkout", "-q", "-b", branch, "base")
        target.write_text(base if edit is None else edit(base), encoding="utf-8")
        _git(repo, "add", "-A")
        assert _git(repo, "commit", "-q", "-m", branch).returncode == 0
    merged = _git(repo, "merge", "-q", "--no-edit", "theirs")
    assert merged.returncode == 0, f"union merge conflicted: {merged.stdout}{merged.stderr}"
    return target.read_text(encoding="utf-8")


_ENTRY = (
    "---\ngenerator: producer_parity/test_ledger.py (static half)\n"
    "flag: TORCHLENS_REFRESH_PRODUCER_LEDGER=1\nreason: lane {lane} refresh\n"
    "env: py3.10-torch2.13.0\ntorch: 2.13.0+cpu\nsource: {lane} (clean)\n"
    "recorded: 2026-10-14T0{hour}:00:00+00:00\n"
)
_JSON_ROW = (
    ' {{\n  "ambiguous": false,\n  "field": "zz_{lane}",\n  "file": "torchlens/__init__.py",\n'
    '  "has_default": false,\n  "sites": {sites},\n  "via_getattr": false\n }},\n'
)
_COUNT_ROW = r"[^#\t][^\t]*\t\d+"
_DICT_ROW = r'    "[^"]+": \d+,'
_TSV_ROW = r"(?!site_key\t|name\t)[^#\t][^\t]*\t.+"
_CODE_ROW = r"\| `[a-z0-9_]+` \|.*"

#: (scenario id, file, our edit, their edit): the concurrent edits each check exists for.
CORRUPTING_MERGES = (
    (
        "count-tsv-same-key",
        f"{_DATA}/diagnostic_ratchets.tsv",
        _bump(_COUNT_ROW, 0, 1),
        _bump(_COUNT_ROW, 0, 2),
    ),
    (
        "count-tsv-neighbouring-keys",
        f"{_DATA}/diagnostic_ratchets.tsv",
        _bump(_COUNT_ROW, 0, 1),
        _bump(_COUNT_ROW, 1, 1),
    ),
    (
        "cardinality-same-key",
        f"{_DATA}/universe_cardinalities.tsv",
        _bump(_COUNT_ROW, 0, 1),
        _bump(_COUNT_ROW, 0, 2),
    ),
    (
        "harvest-same-key",
        f"{_DATA}/harvest_counts.tsv",
        _bump(_COUNT_ROW, 0, 1),
        _bump(_COUNT_ROW, 0, 2),
    ),
    (
        "refusal-class-same-row",
        f"{_DATA}/refusal_site_classification.tsv",
        _relabel(_TSV_ROW, 1, "lane-a-", "\t"),
        _relabel(_TSV_ROW, 1, "lane-b-", "\t"),
    ),
    (
        "surface-same-name",
        f"{_ORACLES}/surface_module_classification.tsv",
        _relabel(_TSV_ROW, 1, "lane-a-", "\t"),
        _relabel(_TSV_ROW, 1, "lane-b-", "\t"),
    ),
    (
        "error-contract-same-row",
        "docs/reference/error_refusal_contract.md",
        _relabel(_CODE_ROW, 2, "lane A:", "|"),
        _relabel(_CODE_ROW, 2, "lane B:", "|"),
    ),
    (
        "warning-contract-same-row",
        "docs/reference/warning_contract.md",
        _relabel(_CODE_ROW, 2, "lane A:", "|"),
        _relabel(_CODE_ROW, 2, "lane B:", "|"),
    ),
    (
        "size-ratchet-same-key",
        "tests/test_file_size_ratchet.py",
        _bump(_DICT_ROW, 0, 50, "    # raised by lane A."),
        _bump(_DICT_ROW, 0, 100, "    # raised by lane B."),
    ),
    (
        "size-ratchet-neighbouring-keys",
        "tests/test_file_size_ratchet.py",
        _bump(_DICT_ROW, 0, -50),
        _bump(_DICT_ROW, 1, -50),
    ),
    (
        "ruff-ratchet-neighbouring-keys",
        "tests/test_ruff_ratchet_governance.py",
        _bump(_DICT_ROW, 0, -1),
        _bump(_DICT_ROW, 1, -1),
    ),
    (
        "static-scan-two-new-rows",
        f"{_LEDGER}/static_scan.json",
        _insert_after_first("[\n", _JSON_ROW.format(lane="a", sites=1)),
        _insert_after_first("[\n", _JSON_ROW.format(lane="b", sites=2)),
    ),
    (
        "static-scan-same-row-count",
        f"{_LEDGER}/static_scan.json",
        _bump_first(r'  "sites": \d+,', 1),
        _bump_first(r'  "sites": \d+,', 2),
    ),
    (
        "provenance-two-appends",
        f"{_LEDGER}/PROVENANCE",
        _append(_ENTRY.format(lane="a", hour=1)),
        _append(_ENTRY.format(lane="b", hour=2)),
    ),
)


@pytest.mark.parametrize(
    ("path", "ours", "theirs"),
    [case[1:] for case in CORRUPTING_MERGES],
    ids=[case[0] for case in CORRUPTING_MERGES],
)
def test_union_merge_corruption_fails_the_check(
    tmp_path: Path, path: str, ours: Edit, theirs: Edit
) -> None:
    merged = union_merge(tmp_path, path, ours, theirs)
    with pytest.raises(AssertionError):
        ALL_CHECKS[path](merged, path)


def test_numeral_baseline_check_refuses_a_repeated_row() -> None:
    """Union cannot repeat a numeral row (identical lines collapse); a hand edit can."""

    text = "corpus\tfile\tnumeral\ndocs-md\tAGENTS.md\t1,421\ndocs-md\tAGENTS.md\t1,421\n"
    with pytest.raises(AssertionError, match="appears on lines 2 and 3"):
        unique_tsv_key(None)(text, "planted.tsv")


def test_provenance_exemption_covers_only_the_two_born_headerless_entries() -> None:
    """A headerless entry is refused unless it is one of the two recorded at birth."""

    whole = _ENTRY.format(lane="a", hour=1)[4:]
    headerless = "reason: r\nenv: e\ntorch: t\nsource: s\nrecorded: {}\n"
    born = headerless.format("2026-08-19T07:52:35.681515+00:00")
    check_provenance(whole + born, "p")
    with pytest.raises(AssertionError, match="fused two appended entries"):
        check_provenance(whole + headerless.format("2026-10-14T03:00:00+00:00"), "p")
