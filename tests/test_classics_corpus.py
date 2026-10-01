"""Classics corpus: coverage-chosen hand-built models checked end to end.

The corpus (``tests/classics_corpus/``) is a sample of hand-built historical and
unusual architectures, chosen greedily for coverage of capture features (op
functions, module types, recurrence, conditionals, in-place ops, dtypes, input
and output containers, buffers, shared parameters, model size). Every entry is

1. captured with ``tl.trace``,
2. round-tripped through a portable ``.tlspec`` save and load,
3. forward-replay validated with metadata invariants (``tl.validate``), and
4. checked for runnable resolver readiness: zero unresolved or ambiguous torch
   registry keys, the resolver release gate of
   ``docs/reference/runnable_tlspec_contract.md`` section 13.

That full check is the comprehensive tier and carries ``slow``;
``pytest tests/test_classics_corpus.py`` runs it for every entry. The smoke
subset (fragile paths: recurrence, conditionals, in-place ops, unusual dtypes
and containers, shared parameters) additionally runs forward validation alone
under ``smoke``, sized so the whole family stays inside the smoke tier's
aggregate duration budget.
"""

from __future__ import annotations

import gc
import re
import warnings
from collections import Counter
from pathlib import Path

import pytest

import torchlens as tl
from tests.classics_corpus._loader import (
    MANIFEST_SCHEMA,
    MODELS_DIR,
    TIERS,
    VALIDATION_SEED,
    CorpusEntry,
    build_entry,
    corpus_entries,
    file_sha256,
    load_manifest,
)
from torchlens._io.runnable import build_sparse_run_descriptor, preflight_sparse_run_descriptor
from torchlens.options import CaptureOptions
from torchlens.runnable import ResolverStatus, RunnableErrorCode
from torchlens.validation import last_validation_failure

SMOKE_TIER_SIZE = (10, 20)
"""Inclusive bounds on the smoke subset size.

The 2026-09-30 ruling asked for about 15-20 models; the smoke family's aggregate
duration budget (``tests/conftest.py``) holds the subset to the low end.
"""


def _entry_param(entry: CorpusEntry) -> object:
    """Wrap one entry as a pytest param identified by its entry id."""

    return pytest.param(entry, id=entry.id)


def _validate_entry(entry: CorpusEntry, model: object, inputs: object) -> None:
    """Fail with the structured failure summary unless forward validation passes."""

    verdict = tl.validate(
        model, inputs, scope="forward", random_seed=VALIDATION_SEED, validate_metadata=True
    )
    if not verdict:
        failure = last_validation_failure()
        summary = failure.summary() if failure is not None else "no failure record"
        pytest.fail(f"{entry.id}: forward validation failed: {summary}")


@pytest.mark.smoke
def test_manifest_pins_every_vendored_model() -> None:
    """The manifest and ``models/`` agree, and every file matches its sha256 pin."""

    manifest = load_manifest()
    assert manifest["schema"] == MANIFEST_SCHEMA
    files = {row["module"]: row for row in manifest["files"]}
    on_disk = {path.stem for path in MODELS_DIR.glob("*.py")}
    assert set(files) == on_disk, (
        f"unpinned files: {sorted(on_disk - set(files))}; "
        f"missing files: {sorted(set(files) - on_disk)}"
    )
    for module, row in files.items():
        assert file_sha256(MODELS_DIR / f"{module}.py") == row["sha256"], (
            f"{module}.py differs from its pinned source; re-vendor it byte-identical "
            "and update the manifest, never edit it in place"
        )

    entries = corpus_entries()
    ids = Counter(entry.id for entry in entries)
    assert not [name for name, count in ids.items() if count > 1], "duplicate entry ids"
    assert {entry.module for entry in entries} == on_disk, "a vendored file has no entry"
    for entry in entries:
        assert entry.tier in TIERS, entry
        assert entry.features and entry.why, f"{entry.id} lacks its coverage record"
    smoke = corpus_entries("smoke")
    low, high = SMOKE_TIER_SIZE
    assert low <= len(smoke) <= high, f"smoke subset has {len(smoke)} entries"


@pytest.mark.smoke
@pytest.mark.parametrize("entry", [_entry_param(entry) for entry in corpus_entries("smoke")])
def test_classics_smoke_entry_validates(entry: CorpusEntry) -> None:
    """Forward-validate one smoke-subset entry (disclosures kept non-fatal, as below)."""

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        model, inputs = build_entry(entry)
        _validate_entry(entry, model, inputs)


@pytest.mark.slow
@pytest.mark.parametrize("entry", [_entry_param(entry) for entry in corpus_entries()])
def test_classics_entry_end_to_end(entry: CorpusEntry, tmp_path: Path) -> None:
    """Capture, save/load, validate, and resolver-check one corpus entry.

    The corpus models are ordinary user code, so capture legitimately emits
    TorchLens disclosures (provenance notes, container fallbacks, portable
    metadata flattening) and torch's own deprecation notes. The suite promotes
    TorchLens warnings to errors so that tests assert them locally; here they
    are deliberately kept visible but not fatal, because the four verdicts below
    are what this test asserts.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("default")
        _check_entry(entry, tmp_path)


def _check_entry(entry: CorpusEntry, tmp_path: Path) -> None:
    """Run the four corpus checks for one entry."""

    model, inputs = build_entry(entry)

    trace = tl.trace(model, inputs)
    labels = [op.label for op in trace.ops]
    assert labels, f"{entry.id}: empty trace"
    spec = tmp_path / "entry.tlspec"
    tl.save(trace, spec, level="portable")
    loaded = tl.load(spec)
    assert [op.label for op in loaded.ops] == labels, f"{entry.id}: save/load changed the ops"
    del trace, loaded
    gc.collect()

    _validate_entry(entry, model, inputs)

    readiness_trace = tl.trace(
        model,
        inputs,
        capture=CaptureOptions(
            intervention_ready=True, capture_container_structure=True, cache=False
        ),
    )
    report, _attachments = preflight_sparse_run_descriptor(
        build_sparse_run_descriptor(readiness_trace)
    )
    gaps = []
    for record in report.resolver_records:
        codes = {diagnostic.code for diagnostic in record.diagnostics}
        if RunnableErrorCode.AMBIGUOUS_QUALNAME in codes:
            gaps.append(f"ambiguous {record.recorded_key}")
        elif record.status is ResolverStatus.UNAVAILABLE:
            gaps.append(f"unresolved {record.recorded_key}")
    assert not gaps, f"{entry.id}: resolver release gate: {gaps}"


@pytest.mark.smoke
def test_readme_tier_counts_match_manifest() -> None:
    """The corpus README states the tier sizes the manifest actually holds."""

    readme = (MODELS_DIR.parent / "README.md").read_text(encoding="utf-8")
    smoke = len(corpus_entries("smoke"))
    total = len(corpus_entries())
    match = re.search(r"(\d+) entries \((\d+) smoke", readme)
    assert match is not None, "README must state '<N> entries (<M> smoke'"
    assert (int(match.group(1)), int(match.group(2))) == (total, smoke)
