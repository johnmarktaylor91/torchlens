"""B1-18: the H2 hot-AST release is a TOTAL postprocess epilogue.

``ast_branches`` keeps a process-wide, 256-file-bounded ``_file_cache`` split
into a cold tier (spans, projected calls, retained source) and a hot tier
(``_HeavyAst``: parsed module AST, parent map, scope nodes) for whole source
files -- including torch-library files. The H2 retention seal drops the hot
tier at the postprocess epilogue so no capture leaves multi-megabyte ASTs
pinned between captures.

That drop used to be a success-path STATEMENT at the end of ``postprocess()``,
so any failure inside steps 0-20 or the core freeze skipped it and the hot tier
survived. Distinct failing files accumulate up to the cache bound, which
resurrects exactly the retention class the seal exists to close. It now runs in
a ``finally``.

Releasing is safe on every exit by design: the cold tier survives, so a later
query for an unprojected scope re-parses from RETAINED source (never disk).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens.postprocess as pp
from torchlens.postprocess import ast_branches

# Markers are per-test, not module-level: every case here is smoke-fast EXCEPT the
# `-O` gate proof, which spawns a fresh interpreter that pays a full cold torchlens
# import (measured 37s). Marking that one `heavy` keeps the smoke partition honest;
# combining `smoke` with `heavy` would fail tests/test_marker_lint.py.


class _Conditional(nn.Module):
    """A model with a real branch, so step 5 builds AST indexes."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        y = self.fc(x)
        if y.sum() > 0:
            return torch.relu(y)
        return torch.sigmoid(y)


def _hot_tier_files() -> list[str]:
    """Return cached files currently holding a parsed-AST hot tier."""

    return sorted(
        name for name, index in ast_branches._file_cache.items() if index._heavy is not None
    )


@pytest.fixture(autouse=True)
def _clean_file_cache():
    """Isolate each test from process-wide AST-cache carryover."""

    ast_branches._file_cache.clear()
    yield
    ast_branches._file_cache.clear()


def test_successful_postprocess_releases_the_hot_tier() -> None:
    """The historical success-path behavior is unchanged."""

    tl.trace(
        _Conditional().eval(),
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(save_code_context=True),
    )
    assert _hot_tier_files() == []


def test_failed_postprocess_still_releases_the_hot_tier(monkeypatch) -> None:
    """The defect: a raising pipeline used to keep whole ASTs pinned.

    Fault injection at a late step (module-log building, step 16), after step 5
    has populated the AST cache. Measured before the fix, this left the hot
    tier holding the model's own source file AND
    ``torch/nn/modules/linear.py``.
    """

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected postprocess failure")

    monkeypatch.setattr(pp, "_build_module_logs", _boom)
    monkeypatch.setattr(pp.finalization, "_build_module_logs", _boom, raising=False)

    with pytest.raises(RuntimeError, match="injected postprocess failure"):
        tl.trace(
            _Conditional().eval(),
            torch.ones(1, 3),
            capture=tl.options.CaptureOptions(save_code_context=True),
        )
    assert _hot_tier_files() == []


def test_repeated_postprocess_failures_do_not_accumulate_hot_tiers(monkeypatch) -> None:
    """The accumulation-to-the-cache-bound half of the finding."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected postprocess failure")

    monkeypatch.setattr(pp, "_build_module_logs", _boom)
    monkeypatch.setattr(pp.finalization, "_build_module_logs", _boom, raising=False)

    for _ in range(3):
        with pytest.raises(RuntimeError, match="injected postprocess failure"):
            tl.trace(
                _Conditional().eval(),
                torch.ones(1, 3),
                capture=tl.options.CaptureOptions(save_code_context=True),
            )
    assert _hot_tier_files() == []


def test_the_release_is_reached_through_a_finally_not_a_tail_statement() -> None:
    """Source lockstep on the totality itself.

    The runtime tests above would also pass if someone re-added the release as
    a duplicated statement on each exit path -- which is how the zero-layer
    early return came to be missed in the first place. Pin the structure.
    """

    import inspect

    source = inspect.getsource(pp.postprocess)
    assert "finally:" in source
    finally_body = source.split("finally:", 1)[1]
    assert "release_parsed_asts()" in finally_body
    # The body function must NOT carry its own tail release.
    body_source = inspect.getsource(pp._postprocess_body)
    assert "release_parsed_asts()" not in body_source


def test_released_hot_tier_still_answers_lazy_source_queries() -> None:
    """Releasing is safe: the cold tier keeps post-capture queries working.

    This is why the release can live in ``finally`` at all. If a released hot
    tier broke ``Op.arg_expressions``, moving the drop onto the failure path
    would have traded a leak for a correctness bug.
    """

    trace = tl.trace(
        _Conditional().eval(),
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(save_code_context=True),
    )
    assert _hot_tier_files() == []
    # Exercise the lazy re-parse path on every op that has one.
    expressions = [getattr(op, "arg_expressions", None) for op in trace.ops]
    assert any(value for value in expressions)


# ---------------------------------------------------------------------------
# B1-23b: the per-step audit window opens INSIDE the try
# ---------------------------------------------------------------------------


def _audit_registry_sizes() -> tuple[int, ...]:
    """Return the sizes of every process-wide audit side table."""

    from torchlens._trace_core import op_store

    return (
        len(op_store._AUDIT_COLLECTORS),
        len(op_store._AUDIT_FINGERPRINTS),
        len(op_store._AUDIT_ROW_RELEASES),
        len(op_store._AUDIT_READS),
    )


def test_a_raise_while_arming_the_audit_window_is_unwound(monkeypatch) -> None:
    """B1-23b: `_open_step_write_audit` ran BEFORE the try.

    ``begin_cell_write_audit`` registers its collectors under ``id(store)`` and
    only THEN swaps the store class, and its fingerprint sweep runs arbitrary
    ``__repr__`` code at the leaves. A raise part-way through therefore used to
    escape with collectors registered (and possibly the class swapped) and
    nothing to unwind them -- an id-keyed leak a later store at the same
    address would inherit.
    """

    from torchlens._trace_core import op_store

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    real_begin = op_store.begin_cell_write_audit
    calls = {"n": 0}

    def _begin_then_explode(store, **kwargs):
        calls["n"] += 1
        # Arm for real (registering the collectors), then fail like a leaf
        # __repr__ would.
        real_begin(store, **kwargs)
        raise RuntimeError("injected failure while arming the audit window")

    monkeypatch.setattr(op_store, "begin_cell_write_audit", _begin_then_explode)

    before = _audit_registry_sizes()
    with pytest.raises(RuntimeError, match="while arming the audit window"):
        tl.trace(_Conditional().eval(), torch.ones(1, 3))
    assert calls["n"] >= 1, "the injected arming failure never ran"
    assert _audit_registry_sizes() == before, (
        "a raise while arming the audit window leaked process-wide audit state"
    )


def test_the_window_open_sits_inside_the_try(monkeypatch) -> None:
    """Source lockstep on the unwind structure itself."""

    import inspect

    from torchlens.postprocess import _executor

    source = inspect.getsource(_executor.run_pipeline)
    body = source.split("for spec in STEP_REGISTRY:", 1)[1]
    try_at = body.index("try:")
    open_at = body.index("_open_step_write_audit")
    assert try_at < open_at, "the audit window opens outside the try (B1-23b)"


@pytest.mark.smoke
def test_audited_capture_still_runs_clean_end_to_end(monkeypatch) -> None:
    """Moving the open did not break the audit it guards."""

    monkeypatch.setenv("TORCHLENS_POSTPROCESS_ASSERTIONS", "1")
    before = _audit_registry_sizes()
    trace = tl.trace(
        _Conditional().eval(),
        torch.ones(1, 3),
        capture=tl.options.CaptureOptions(save_code_context=True),
    )
    assert trace.num_ops > 0
    assert _audit_registry_sizes() == before
    assert _hot_tier_files() == []


# ---------------------------------------------------------------------------
# B1-23a: REFUTED, and pinned so it is not re-filed
# ---------------------------------------------------------------------------


@pytest.mark.heavy
def test_arming_the_audit_under_dash_O_is_a_hard_error_not_a_silent_no_op() -> None:
    """B1-23a's premise is false: `-O` cannot silently strip these asserts.

    The finding held that `_assert_no_open_window`'s bare `assert` is stripped
    under `python -O`. It cannot be reached in that state:
    `_postprocess_assertions_enabled` HARD-ERRORS when the audit is armed with
    assertions off, and returns False otherwise -- and `run_pipeline` calls the
    check only when it returns True. So assertions-off means the audit never
    runs and the check is never called. The whole audit family is
    assert-based by design; this gate is what makes that safe, so the spelling
    was reviewed and left alone rather than made inconsistent.
    """

    import subprocess
    import sys

    program = (
        "import os;"
        "os.environ['TORCHLENS_POSTPROCESS_ASSERTIONS']='1';"
        "import torchlens.postprocess as pp;"
        "pp._postprocess_assertions_enabled()"
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", program],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "assertions are disabled" in result.stderr
