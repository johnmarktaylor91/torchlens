"""The executor's own stage prompts must match their PLAN-pinned digests.

``PLAN.md`` section 20 pins ``stage1_discovery.md`` and ``stage2_author.md``,
the two prompts that drive every author session on the production path. The
shipped ``tools/verify_prompts.py`` checks only the pool fragments
(``claude_crawler_author_v2.txt``, ``codex_accuracy_checker_v2.txt``) that the
headless executor replaced, so until this module existed the executor pins were
PROSE: a stage prompt could be edited and its PLAN row left stale, or the row
bumped and the prompt left alone, and nothing anywhere would notice. That is
precisely the undetectable drift section 20 says it closed.

The pin is re-derived here with the SAME two primitives the shipped tool
composes -- :func:`~menagerie.crawler.tools.verify_prompts._pinned_digest` for
the PLAN row and :func:`~menagerie.crawler.identity.hash_bytes` for the file --
rather than a second PLAN parser, so a change to how a pin row is read cannot
leave this guard agreeing with a rule nobody else follows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from menagerie.crawler.identity import hash_bytes
from menagerie.crawler.tools.verify_prompts import _pinned_digest

#: Every prompt PLAN.md section 20 pins for the headless executor. BOTH are
#: covered on purpose: a guard that froze exactly one prompt would pass while
#: the other drifted, which is the failure this module exists to prevent.
EXECUTOR_PROMPT_NAMES = ("stage1_discovery.md", "stage2_author.md")

_CRAWLER_ROOT = Path(__file__).resolve().parents[1]
_PLAN_PATH = _CRAWLER_ROOT / "PLAN.md"
_PROMPT_DIR = _CRAWLER_ROOT / "prompts" / "executor"


def _plan_bytes() -> bytes:
    """Return the committed PLAN.md bytes carrying the pinned digest rows."""

    return _PLAN_PATH.read_bytes()


@pytest.mark.smoke
@pytest.mark.parametrize("prompt_name", EXECUTOR_PROMPT_NAMES)
def test_an_executor_stage_prompt_matches_its_pinned_digest(prompt_name: str) -> None:
    """Each shipped stage prompt hashes to exactly the digest PLAN.md pins."""

    prompt_path = _PROMPT_DIR / prompt_name
    assert prompt_path.is_file(), f"pinned executor prompt is missing: {prompt_name}"
    assert _pinned_digest(_plan_bytes(), prompt_name) == hash_bytes(prompt_path.read_bytes()), (
        f"{prompt_name} does not match its PLAN.md section 20 pin -- either the prompt "
        "drifted or the pin was not bumped alongside the edit"
    )


@pytest.mark.smoke
def test_the_two_executor_pins_are_distinct_digests() -> None:
    """The pinned rows must be two real, different digests.

    Without this, a PLAN edit that pointed both rows at one digest -- or a
    parser that silently returned the same row twice -- would leave the
    per-prompt assertions passing against a single value.
    """

    plan = _plan_bytes()
    digests = [_pinned_digest(plan, name) for name in EXECUTOR_PROMPT_NAMES]
    assert len(set(digests)) == len(EXECUTOR_PROMPT_NAMES), (
        "PLAN.md section 20 pins the executor prompts to a shared digest"
    )


@pytest.mark.smoke
@pytest.mark.parametrize("prompt_name", EXECUTOR_PROMPT_NAMES)
def test_the_pin_bites_when_a_stage_prompt_drifts(tmp_path: Path, prompt_name: str) -> None:
    """One appended byte must break the pin.

    The tamper is applied to a COPY so the assertion exercises the real
    comparison without mutating the tree. If this ever passes, the guard above
    is decorative.
    """

    original = (_PROMPT_DIR / prompt_name).read_bytes()
    drifted_path = tmp_path / prompt_name
    drifted_path.write_bytes(original + b"x")
    pinned = _pinned_digest(_plan_bytes(), prompt_name)
    assert pinned == hash_bytes(original), "fixture must start from the pinned bytes"
    assert pinned != hash_bytes(drifted_path.read_bytes()), (
        f"appending a byte to {prompt_name} did not move its digest"
    )
