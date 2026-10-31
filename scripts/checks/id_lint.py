"""Internal-id lint: count private planning ids left in package and docs text.

Maintainers' planning names its work with private ids (walk rulings, tracker
items, review rounds, finding numbers, sprint and lane features, memo items).
Those ids mean nothing to a reader of this public package. Each family below
is one regex; the check counts matches per family and per file and fails
when a family's total rises above its number in
``scripts/checks/baselines.json``. The baselines only shrink, to zero.

Scanned text: comments and string literals of ``torchlens/**/*.py`` (tool
directives such as ``# noqa: D401`` are dropped first, and a string literal
that is exactly one id, such as ``code="N1"``, is data and is skipped), and
the full text of ``docs/**`` files with suffix ``.md``, ``.rst``, ``.txt`` or
``.py``. Narrowing a family's regex lowers its count and needs the same
one-line reason in the commit as raising a baseline.

``python scripts/checks/id_lint.py --by-file`` lists counts per file.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import Outcome, load_baselines, ratchet  # noqa: E402

#: One regex per id family, each with examples in its comment.
FAMILIES: dict[str, str] = {
    # U36.2, U00, R2-08, forks.11: design-walk units, rows and topic rulings.
    "walk_ruling": (
        r"\bU[0-9]{1,3}(?:\.[0-9]+)?\b|\bR[0-9]-[0-9]{2}\b"
        r"|\b(?:forks|neuro|names|scope|errors|defaults|naming)\.[0-9]+\b"
    ),
    # AGT-2360: work-tracker items.
    "work_item": r"\bAGT-[0-9]+\b",
    # r37, r-b4, grind-r5, w15, b3, corr1_1: review and grind round tags.
    "review_round": r"\b(?:grind-r|r-b|corr|r|w|b)[0-9]+[a-z]?(?:_[0-9]+)?\b",
    # R26-2, B8-1a, SF-04, INV-1, FORK-4, FLIP-2, D-11: numbered findings and forks.
    "finding": r"\b(?:[RB][0-9]+|SF|INV|FORK|FLIP|D|F|S|GC|W[0-9]*)-[0-9]+[a-z]?\b",
    # F22, lane F42, W3, WT1, C01: sprint lanes, features and work tranches.
    "sprint_feature": r"\blane [A-Z][A-Z0-9-]*\b|\bF[0-9]{1,3}\b|\bWT?[0-9]{1,3}(?:\.[0-9]+)?\b|\bC[0-9]{1,2}\b",
    # memo 3.1, D18, M8, S7: architecture-memo sections and their decision items.
    "memo_item": (
        r"\bmemo (?:[A-Z]-?)?[0-9]+(?:\.[0-9]+)*\b|\bD[0-9]{1,3}\b"
        r"|\bM[0-9]{1,2}\b(?! ?(?:Mac|Pro|Max|Ultra|chip))|\bS[0-9]{1,2}\b"
    ),
}
_COMPILED = {name: re.compile(pattern) for name, pattern in FAMILIES.items()}
_DIRECTIVE = re.compile(r"#\s*(?:noqa|type:\s*ignore|pragma|pylint|fmt|isort|ruff)\b.*")
_DOC_SUFFIXES = frozenset({".md", ".rst", ".txt", ".py"})
_STRING_PREFIX = re.compile(r"^[rRbBuUfF]*('''|\"\"\"|'|\")")
_TEXT_TOKENS = frozenset(
    kind
    for kind in (tokenize.COMMENT, tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", None))
    if kind is not None
)


def _string_body(token: str) -> str:
    match = _STRING_PREFIX.match(token)
    if match is None:
        return token
    quote = match.group(1)
    return token[match.end() : len(token) - len(quote)]


def _python_text(source: str) -> Iterator[str]:
    """Yield the comment and string-literal text of a Python source."""

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type not in _TEXT_TOKENS:
            continue
        if token.type == tokenize.COMMENT:
            yield _DIRECTIVE.sub("", token.string)
            continue
        body = _string_body(token.string).strip()
        if not any(regex.fullmatch(body) for regex in _COMPILED.values()):
            yield body


def count_text(text: str) -> Counter[str]:
    """Return id counts per family in one block of text."""

    return Counter({name: len(regex.findall(text)) for name, regex in _COMPILED.items()})


def _scanned_files(repo: Path) -> Iterator[tuple[Path, bool]]:
    for path in sorted((repo / "torchlens").rglob("*.py")):
        yield path, True
    docs = repo / "docs"
    if docs.is_dir():
        for path in sorted(docs.rglob("*")):
            if path.is_file() and path.suffix in _DOC_SUFFIXES:
                yield path, False


def measure_by_file(repo: Path) -> dict[str, Counter[str]]:
    """Return ``{relative path: family counts}`` for files with any id."""

    found: dict[str, Counter[str]] = {}
    for path, is_package in _scanned_files(repo):
        source = path.read_text(encoding="utf-8")
        text = "\n".join(_python_text(source)) if is_package else source
        counts = +count_text(text)
        if counts:
            found[path.relative_to(repo).as_posix()] = counts
    return found


def measure(repo: Path) -> dict[str, int]:
    """Return per-family totals and the overall total."""

    totals: Counter[str] = Counter(dict.fromkeys(FAMILIES, 0))
    for counts in measure_by_file(repo).values():
        totals.update(counts)
    return {"total": sum(totals.values()), **dict(totals)}


def check(repo: Path) -> Outcome:
    """Fail when the total or any family rises above its baseline."""

    return ratchet("id-lint", measure(repo), load_baselines(repo)["id_lint"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--by-file", action="store_true", help="print counts per file")
    args = parser.parse_args(argv)
    if args.by_file:
        rows = sorted(measure_by_file(args.repo).items(), key=lambda kv: -sum(kv[1].values()))
        for rel, counts in rows:
            print(f"{sum(counts.values()):6} {rel} {dict(counts)}")
        return 0
    outcome = check(args.repo)
    print(f"[{outcome.status.upper():4}] {outcome.message}")
    return 1 if outcome.failed else 0


if __name__ == "__main__":
    sys.exit(main())
