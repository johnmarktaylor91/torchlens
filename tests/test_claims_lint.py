"""Executable claims-lint rows from ``docs/migration/_claims.json``.

A lint row may carry a ``check`` object that turns its prose ``lint`` rule into
a scan of the public copy:

* ``scope``: repo-relative globs of the files the row governs. The token
  ``CHANGELOG.md#unreleased`` means the part of ``CHANGELOG.md`` above the first
  released section heading. Released CHANGELOG sections are excluded on purpose:
  they are a historical record of what each release said and are never reworded,
  so a retired figure quoted there is history, not current copy.
* ``rules``: each rule has a ``forbid`` pattern and optionally a ``context``
  pattern and ``require`` patterns. A paragraph violates the rule when it matches
  ``context`` (if given) and ``forbid``, unless it also matches every ``require``
  pattern (the scoped form the row allows).

Patterns are case-insensitive Python regexes applied to one paragraph at a time,
with emphasis markers and backticks removed and whitespace collapsed, so a
sentence wrapped across lines still matches. Markdown table rows and headings are
paragraphs of their own.

The thirteen seed lint rows predate executable checks and stay prose-only; every
row added after them must carry a check, must flag at least one planted
violation per rule (``PLANTED`` below) and must pass on the real tree.
"""

from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLAIMS = REPO_ROOT / "docs" / "migration" / "_claims.json"
UNRELEASED_SUFFIX = "#unreleased"
RELEASED_HEADING = re.compile(r"^## v?\d", re.MULTILINE)
LIST_ITEM = re.compile(r"(?:[-*+]|\d+[.)])\s")
FLAGS = re.IGNORECASE

#: The seed rows, written before lint rows carried a ``check``. Never shrink
#: this set by deleting a row; a seed row may gain a check and leave it.
SEED_PROSE_LINTS = frozenset(
    {
        "perf-no-single-multiplier",
        "perf-fastlog-attribution",
        "episode-capture-caveats",
        "menagerie-phrasing",
        "macs-unit-suffix",
        "timing-uniqueness-scope",
        "halt-selector-shape",
        "attribution-self-check-scope",
        "attention-picture-scope",
        "halt-module-selector-named",
        "public-spelling-disclosure",
        "provenance-scope-matches-number",
        "flops-name-both-conventions",
    }
)

#: Planted cases per executable row: (violations, passes). Every violation must
#: be flagged, every pass must not be, and each rule of the row must be the one
#: that flags at least one violation.
PLANTED: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "torchview-yolov8-retired": (
        (
            "Unlike TorchLens, torchview fails on YOLOv8.",
            "torchview's most-commented issue (#63) is a detector failure.",
        ),
        ("torchview raises a KeyError on Hugging Face decoders under its default settings.",),
    ),
    "tlens-verified-count-context": (
        ("TransformerLens verifies 1,205 of 15,670 models; TorchLens verifies thousands.",),
        (
            "TransformerLens reports 1,205 of 15,670 Hugging Face models passing its "
            "logit-comparison check; TorchLens counts menagerie entries whose capture "
            "replays op by op. Different checks over different model sets.",
        ),
    ),
    "baukit-license-wording": (
        ("baukit is unlicensed, so nothing is copied from it.",),
        ("baukit has no license file (its setup file declares MIT).",),
    ),
    "deepjuice-dependency-scope": (
        (
            "DeepJuice uses TorchLens for feature extraction.",
            "TorchLens reaches GPU parity with DeepJuice.",
            "Net2Brain's RDMs include duplicate pairs from ReLU layers.",
        ),
        ("DeepJuice lists TorchLens as a dependency.",),
    ),
    "rsa-brainscore-destinations": (
        (
            "TorchLens runs Brain-Score benchmarks end to end.",
            "Our RDMs reach RSA parity with rsatoolbox.",
        ),
        ("TorchLens hands RDMs to rsatoolbox and model activations to Brain-Score.",),
    ),
    "torchsnooper-not-snoop": (
        (
            "snoop is dead, so line tracing needs TorchLens.",
            "torchsnooper adds 32x overhead on a GPT-2 forward.",
        ),
        (
            "torchsnooper is unmaintained; snoop is active.",
            "torchsnooper ran 32 to 389 times slower than the plain forward in one session.",
        ),
    ),
    "torcheck-no-superiority": (
        (
            "The checks kit catches more bugs than torcheck.",
            "Compared with torcheck, the step overhead is 3% per step.",
        ),
        ("torcheck's checks map onto registry rows listed below.",),
    ),
    "torchexplorer-breakage-scope": (
        ("torchexplorer is broken on current PyTorch.",),
        (
            "torchexplorer crashes on in-place models under its default settings and on "
            "transformer models called with keyword arguments.",
        ),
    ),
    "three-dashboards-retired": (
        (
            "One TensorBoard emission lands in three dashboards.",
            "A single emission reaches every tracker you use.",
        ),
        (
            "One engine, two first-party sinks, and documented relays with a measured "
            "fidelity table.",
        ),
    ),
    "tensorboard-feeds-not-beats": (
        (
            "TorchLens beats TensorBoard at graph display.",
            "TensorBoard's graph view breaks on Hugging Face models.",
        ),
        (
            "TorchLens feeds TensorBoard; the stock torch add_graph path fails on current "
            "Hugging Face models.",
        ),
    ),
    "activation-watch-scope": (
        (
            "No tracker can watch activations.",
            "Nobody else watches activations during training.",
            "TensorBoard and wandb do not watch activations.",
        ),
        (
            "TensorBoard and stock wandb do not watch activations; Neptune's TorchWatcher "
            "did, and is credited here by name.",
        ),
    ),
    "device-time-posture": (
        (
            "TorchLens measures GPU kernel time for every op.",
            "Capture overhead is about 14x.",
        ),
        (
            "Locate the slow op with TorchLens, then measure it clean with torch's "
            "benchmark Timer; the GPU join that would report device time has not run yet.",
        ),
    ),
    "hidden-states-grain-not-access": (
        ("TorchLens replaces output_hidden_states.",),
        (
            "output_hidden_states gives one tensor per block; TorchLens gives every "
            "operation at a uniform grain.",
        ),
    ),
    "replay-no-single-multiplier": (
        (
            "Nobody else has replay.",
            "Record plus intervene costs 30x native.",
            "The bind tier runs at 2x native.",
        ),
        (
            "Record plus intervene measured about 28x to 42x native across three "
            "sessions; TransformerLens has an official partial forward.",
        ),
    ),
    "weightsfree-in-seconds": (
        ("Weights-free capture traces any model in seconds.",),
        (
            "Weights-free capture builds the graph in about 15 seconds for 8B-class "
            "transformers, on CPU.",
        ),
    ),
    "weightsfree-launch-limits": (
        (
            "Weights-free capture is zero cost.",
            "It works on transformers 4.x.",
            'Load with from_pretrained(device_map="meta") to skip the weights.',
            "Weights-free capture runs 70B models on a laptop.",
        ),
        (
            'from_pretrained(device_map="meta") waits for its offline test row.',
            "Per-version results are in the transformers table.",
        ),
    ),
    "agent-surface-scope": (
        (
            "TorchLens is the first library with an agent surface.",
            "The JSON dump fits in context.",
        ),
        (
            "TorchLens is the first interpretability package with a read-only MCP and "
            "structured view over saved artifacts; nnsight ships its own agent surface "
            "(an MCP server and agent docs).",
        ),
    ),
    "restricted-execution-milestone": (
        (
            "Restricted execution is a committed milestone.",
            "Counterfactual execution ships in Q1 2027.",
        ),
        ("Restricted execution ships when its gates pass.",),
    ),
    "menagerie-figure-collision": (
        (
            "The menagerie is 100% verified.",
            "All 11,670 menagerie architectures are algorithmically verified.",
            "The menagerie catalog holds 9,200 entries.",
        ),
        ("The menagerie catalog holds 8,500+ rows across ~3,600 families, measured 2026-08-16.",),
    ),
    "verified-capture-scope": (
        (
            "Faithful capture is proven, not assumed.",
            "Replay precludes silent wrong answers.",
        ),
        ("Faithful capture is proven, not assumed, for verified captures.",),
    ),
    "model-diagnostics-weights-caveat": (
        ("TorchLens is the model-diagnostics tool for PyTorch.",),
        (
            "TorchLens is the model-diagnostics tool for activations and execution; "
            "weight-side diagnostics such as spectra belong to weightwatcher.",
        ),
    ),
    "root-limitations-current": (
        (
            "Cross-rank merging (tl.merge_ranks()) is not available in this release.",
            "| `bitsandbytes_8bit_4bit` | `known_broken` when detected | outside the contract |",
            "A compiled callable cannot be unwrapped, so its interior is not logged.",
        ),
        (
            "tl.merge_ranks(...) merges rank-local captures.",
            "| `bitsandbytes_8bit_4bit` | `pass` when detected (severity `warning`) | parity |",
            "On older torch without set_stance, a compiled callable's interior is not logged.",
        ),
    ),
    "nnsight-matched-uniqueness": (
        (
            "Only TorchLens sees every operation, not just modules.",
            "TorchLens is the only interpretability tool for diffusers pipelines.",
        ),
        (
            "TorchLens records the full executed graph of every operation, not just "
            "module outputs; nnsight's .source reaches operations inside a forward.",
        ),
    ),
}


@cache
def _claims() -> dict[str, Any]:
    return json.loads(CLAIMS.read_text(encoding="utf-8"))


def _checked_rows() -> list[dict[str, Any]]:
    return [row for row in _claims()["lints"] if "check" in row]


def _normalize(text: str) -> str:
    text = text.replace("`", "").replace("**", "")
    text = re.sub(r"(?<!\w)\*(?=\S)|(?<=\S)\*(?!\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _paragraphs(text: str) -> list[str]:
    """Split markdown into normalized paragraphs.

    Table rows and headings stand alone; a list item starts a new paragraph.
    """

    out: list[str] = []
    block: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if block and LIST_ITEM.match(stripped):
            out.append(_normalize(" ".join(block)))
            block = []
        if not stripped or stripped.startswith(("|", "#")):
            if block:
                out.append(_normalize(" ".join(block)))
                block = []
            if stripped:
                out.append(_normalize(stripped))
            continue
        block.append(stripped)
    if block:
        out.append(_normalize(" ".join(block)))
    return out


@cache
def _scope_paragraphs(pattern: str) -> tuple[tuple[str, str], ...]:
    """Return (location, paragraph) pairs for one scope entry."""

    if pattern.endswith(UNRELEASED_SUFFIX):
        path = REPO_ROOT / pattern[: -len(UNRELEASED_SUFFIX)]
        text = path.read_text(encoding="utf-8")
        released = RELEASED_HEADING.search(text)
        sources = [(path, text[: released.start()] if released else text)]
    else:
        sources = [
            (path, path.read_text(encoding="utf-8")) for path in sorted(REPO_ROOT.glob(pattern))
        ]
    pairs = []
    for path, text in sources:
        rel = path.relative_to(REPO_ROOT).as_posix()
        pairs.extend((rel, para) for para in _paragraphs(text))
    return tuple(pairs)


def _rule_hits(rule: dict[str, Any], paragraph: str) -> bool:
    context = rule.get("context")
    if context and not re.search(context, paragraph, FLAGS):
        return False
    if not re.search(rule["forbid"], paragraph, FLAGS):
        return False
    required = rule.get("require", [])
    return not (required and all(re.search(req, paragraph, FLAGS) for req in required))


def _violated_rules(check: dict[str, Any], paragraph: str) -> list[int]:
    return [index for index, rule in enumerate(check["rules"]) if _rule_hits(rule, paragraph)]


def test_rows_added_after_the_seed_are_executable() -> None:
    lint_ids = {row["id"] for row in _claims()["lints"]}
    missing_seed = SEED_PROSE_LINTS - lint_ids
    assert not missing_seed, f"seed lint rows were deleted: {sorted(missing_seed)}"
    prose_only = {row["id"] for row in _claims()["lints"] if "check" not in row}
    unexpected = prose_only - SEED_PROSE_LINTS
    assert not unexpected, f"lint rows without an executable check: {sorted(unexpected)}"


def test_every_check_is_well_formed_and_planted() -> None:
    checked = {row["id"]: row["check"] for row in _checked_rows()}
    missing = sorted(set(checked) - set(PLANTED))
    stale = sorted(set(PLANTED) - set(checked))
    assert not missing and not stale, f"planted cases out of step: missing {missing}, stale {stale}"
    for row_id, check in checked.items():
        assert set(check) == {"scope", "rules"}, (row_id, sorted(check))
        assert check["scope"] and check["rules"], row_id
        for pattern in check["scope"]:
            assert _scope_paragraphs(pattern), f"{row_id}: scope {pattern!r} matches no text"
        for rule in check["rules"]:
            assert "forbid" in rule, (row_id, rule)
            assert set(rule) <= {"forbid", "context", "require"}, (row_id, rule)
            for regex in [rule["forbid"], rule.get("context") or "", *rule.get("require", [])]:
                re.compile(regex, FLAGS)


@pytest.mark.parametrize("row_id", sorted(PLANTED))
def test_planted_violations_are_caught(row_id: str) -> None:
    check = next(row["check"] for row in _checked_rows() if row["id"] == row_id)
    violations, passes = PLANTED[row_id]
    assert violations and passes, row_id
    rules_seen: set[int] = set()
    for text in violations:
        hits = [index for para in _paragraphs(text) for index in _violated_rules(check, para)]
        assert hits, f"{row_id} missed its planted violation: {text!r}"
        rules_seen.update(hits)
    assert rules_seen == set(range(len(check["rules"]))), (
        f"{row_id}: rules {sorted(set(range(len(check['rules']))) - rules_seen)} "
        "catch no planted violation"
    )
    for text in passes:
        hits = [index for para in _paragraphs(text) for index in _violated_rules(check, para)]
        assert not hits, f"{row_id} flagged its allowed form (rules {hits}): {text!r}"


@pytest.mark.parametrize("row_id", sorted(PLANTED))
def test_public_copy_passes_the_row(row_id: str) -> None:
    check = next(row["check"] for row in _checked_rows() if row["id"] == row_id)
    found = []
    for pattern in check["scope"]:
        for location, para in _scope_paragraphs(pattern):
            for index in _violated_rules(check, para):
                found.append(f"{location} (rule {index}): {para[:200]}")
    assert not found, f"{row_id} violations:\n" + "\n".join(found)


def test_completeness_dividend_copy_cites_only_backed_instances() -> None:
    """Each backed instance names executed tests that exist; pending ones stay out of copy."""

    row = next(r for r in _claims()["claims"] if r["id"] == "claim-6-completeness-dividend")
    wording = row["allowed_public_wording"].lower()
    assert any(ref.startswith("run ") for ref in row["evidence_refs"]), row["evidence_refs"]
    statuses = {instance["status"] for instance in row["instances"]}
    assert statuses <= {"backed", "pending"}, statuses
    for instance in row["instances"]:
        phrase = instance["phrase"].lower()
        if instance["status"] == "pending":
            assert not instance["evidence"] and instance["pending_on"], instance["instance"]
            assert phrase not in wording, f"pending instance in copy: {instance['instance']}"
            continue
        assert phrase in wording, f"backed instance missing from copy: {instance['instance']}"
        assert instance["evidence"], instance["instance"]
        for node in instance["evidence"]:
            path, _, name = node.partition("::")
            source = (REPO_ROOT / path).read_text(encoding="utf-8")
            test_name = name.split("::")[-1]
            assert re.search(rf"^\s*def {re.escape(test_name)}\(", source, re.MULTILINE), node
