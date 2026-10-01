"""Test-naming vestigial governance (r7 R80): round tokens stop growing.

402 test functions carry a round/sprint token (``_rNN``, ``_pN``, ``_certN``,
``_fwN``, ...) and 10 test-file stems collide once the token is stripped — a
PURPOSE-named file coexisting with a ROUND-named file of the same purpose, so
a contributor cannot tell which owns the concern. Nothing prevented new ones
from landing: the count only ever grew between deliberate sweeps.

Two locks, both SHRINK-ONLY:

1. a no-growth ceiling on round-token test-FUNCTION names — rename-by-purpose
   sweeps lower it, new tokens cannot land silently;
2. a frozen ledger of the known stem collisions — the dedup sweep (relayed;
   file renames are deliberately not done mid-fixwave, they conflict with
   parallel lanes) shrinks it, new collisions cannot enter.

The ``test_r*_rce`` security corpus is RENAME-ONLY per the naming matrix (the
token is a corpus marker); it counts toward the ceiling like everything else
so the ceiling only moves in deliberate, reviewed sweeps.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_TESTS_DIR = Path(__file__).resolve().parent

#: Matches the sprint/round token vocabulary measured by the r7 b10 audit.
_ROUND_TOKEN = re.compile(r"_(r\d+|round\d+|b\d+|p\d+|phase\d+|cert\d+|fw\d+)(?=_|$)")

#: No-growth ceiling, measured 2026-08-16 at the fixwave-7 tip with this
#: file's own scanner. SHRINK-ONLY: lower it alongside a rename-by-purpose
#: sweep; if a NEW name trips this, name the test by its purpose instead of
#: its sprint round (or, for a genuine token-bearing domain name like a
#: model size suffix, pick a spelling the token regex does not match).
_ROUND_TOKEN_FUNCTION_CEILING = 402

#: Known stem collisions once round tokens are stripped (r7 R80-2), keyed by
#: the collapsed stem. SHRINK-ONLY: the dedup sweep merges or renames these;
#: a NEW collision means a round-named twin of an existing purpose landed.
_KNOWN_STEM_COLLISIONS = frozenset(
    {
        "semantic/test_facets",
        "test_capture_unification",
        "test_intervention",
        "test_container_registry",
        "test_intervention_hardening",
        "test_io_security",
        "test_tlspec_runnable_alloc_bomb",
        "test_tlspec_runnable_producer",
        "test_tlspec_runnable_seed_typing",
    }
)


def _iter_test_files() -> list[Path]:
    """Return every test module under tests/."""

    return sorted(_TESTS_DIR.rglob("test_*.py"))


def count_round_token_test_functions() -> list[str]:
    """Return ``file::function`` keys for test functions carrying a round token."""

    found: list[str] = []
    for path in _iter_test_files():
        rel = path.relative_to(_TESTS_DIR).as_posix()
        for match in re.finditer(
            r"^def (test_[a-z0-9_]*)", path.read_text(encoding="utf-8"), re.MULTILINE
        ):
            if _ROUND_TOKEN.search(match.group(1)):
                found.append(f"{rel}::{match.group(1)}")
    return found


def collect_stem_collisions() -> dict[str, list[str]]:
    """Group test-file stems that collide once round tokens are stripped."""

    groups: dict[str, list[str]] = defaultdict(list)
    for path in _iter_test_files():
        rel = path.relative_to(_TESTS_DIR).as_posix()
        stripped = _ROUND_TOKEN.sub("", path.stem)
        key = str(Path(rel).parent / stripped) if "/" in rel else stripped
        key = key.removeprefix("./")
        groups[key].append(rel)
    return {stem: files for stem, files in groups.items() if len(files) > 1}


def test_round_token_test_function_count_never_grows() -> None:
    """The round-token test-name census stays at or below its ceiling."""

    found = count_round_token_test_functions()
    assert len(found) <= _ROUND_TOKEN_FUNCTION_CEILING, (
        f"round-token test names grew to {len(found)} (ceiling "
        f"{_ROUND_TOKEN_FUNCTION_CEILING}): name new tests by PURPOSE, not "
        "sprint round — the newest entries are likely "
        f"{sorted(found)[-5:]}"
    )


def test_no_new_test_file_stem_collisions() -> None:
    """Stripped-stem collisions stay within the frozen known set."""

    collisions = collect_stem_collisions()
    known_only = {
        stem
        for stem in collisions
        if stem not in ("test_independent_fact_pins", "test_no_bare_detach")
        # sub-directory namespacing legitimately reuses these two stems
    }
    new = sorted(known_only - _KNOWN_STEM_COLLISIONS)
    assert not new, (
        "new test-file stem collisions (a round-named twin of an existing "
        f"purpose file): { {stem: collisions[stem] for stem in new} } — merge "
        "into the purpose-named file or pick a genuinely distinct name"
    )


def test_round_token_scanner_is_red_capable(tmp_path: Path) -> None:
    """The token regex catches planted round names and passes purpose names."""

    assert _ROUND_TOKEN.search("test_capture_r53_goldens")
    assert _ROUND_TOKEN.search("test_facets_p4")
    assert _ROUND_TOKEN.search("test_intervention_cert10")
    assert _ROUND_TOKEN.search("test_settle_fw6_ledger")
    assert not _ROUND_TOKEN.search("test_receptive_field_center_unit")
    assert not _ROUND_TOKEN.search("test_rank_render")
