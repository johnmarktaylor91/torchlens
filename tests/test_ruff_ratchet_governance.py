"""Ruff ratchet governance: the family ratchet and deferral ledger get TEETH.

R70 (round 3+4): the ``[tool.ruff.lint]`` block SELLS a monotone family
ratchet and an honest per-code deferral ledger, but both were prose — no test
pinned ``select``'s contents (a commit narrowing it passed every gate) and
every ledgered site count had drifted (SIM105 grew 32% while "deferred", B023
— the config's own "REAL bug class" — exceeded its stated ceiling). Two
mechanical locks:

1. ``select`` must remain a SUPERSET of the frozen family floor — the ratchet
   can only grow.
2. Every deferred code carries a NO-GROWTH ceiling, measured here with the
   pinned ruff over the exact CI scope in ONE combined isolated run. Debt may
   shrink (lower the ceiling in the same change — welcomed); it cannot grow.

D417 (parameter-documentation debt, R69) rides the same mechanism: it is not
ledger-deferred (the D family is not in ``select``) but its count was rotting
measurably (140 → 141 → 143 across three hunt passes) with nobody measuring.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: The family ratchet floor. GROW-ONLY: entries are never removed; new
#: families join here in the same change that lands them in pyproject.
_SELECT_FAMILY_FLOOR = frozenset({"E4", "E7", "E9", "F", "I", "B", "SIM", "UP", "C4"})

#: No-growth ceilings per deferred code, measured 2026-08-15 with the pinned
#: ruff 0.15.4 over the CI scope (isolated mode + the config's extend-exclude
#: set, so per-file-ignores do not mask sites). SHRINK-ONLY: lower a ceiling
#: alongside a real cleanup; raising one is the exact silent-growth this test
#: exists to prevent (SIM105 grew 74 -> 99 while ledgered as "deferred").
#:
#: 2026-08-16 R70 r7 INSTRUMENT CORRECTION (fixwave-7): the count parser was
#: blind to .ipynb diagnostics (the "cell N" path segment contains a space,
#: so `^\S+:` never matched) -- every ceiling below was frozen against a
#: measurement that silently excluded all 88 notebook sites. The parser is
#: fixed and every ceiling re-trued to the corrected 2026-08-16 measurement
#: at the fixwave-7 tip. Four ceilings RISE here as an explicit re-ledger of
#: pre-existing, newly-visible notebook debt -- NOT growth (the sites predate
#: the ceilings; enumerated per code below). Five ceilings SHRINK to the true
#: count in the same pass. SHRINK-ONLY from these corrected values.
_DEFERRED_CODE_CEILINGS: dict[str, int] = {
    # r7 R23 reconcile: hunt-6 counts 102 (fable, full lint scope) vs 99
    # (opus, torchlens+tests+scripts) were BOTH correct -- the delta is
    # exactly the 3 examples/notebooks sites. 2026-08-16 re-true: with the
    # ipynb-aware parser the isolated-mode CI-scope count IS 102 (99 .py +
    # 3 notebook sites); ceiling shrunk 109 -> 102. Cite the measurement
    # MODE with any future count or the dispute recurs.
    "B905": 102,
    "B028": 2,
    # 2026-08-16 fixwave-7: all 17 B023 sites fixed (loop vars bound via
    # keyword defaults / class attributes at definition time) -- the
    # config's own "REAL bug class" ledger row is retired. Any new site is
    # a genuine late-binding hazard; fix it, never raise this.
    "B023": 0,
    "B904": 17,
    "B018": 14,
    "B007": 16,
    "B008": 3,
    # 2026-08-16 re-true (ipynb-aware parser): shrunk to the corrected
    # counts -- SIM108 88 -> 86, SIM105 99 -> 92 (90 .py + 2 ipynb),
    # SIM102 40 -> 39.
    "SIM108": 86,
    # 2026-08-16 fixwave-7 settle: 92 -> 93. The one new deferral is the
    # post-fork pre-exec PDEATHSIG guard in utils/_subprocess.py, where
    # contextlib.suppress would ALLOCATE in the no-allocation fork window
    # its own comment forbids; the wave's other new site (reaper test
    # cleanup) was converted to suppress instead of ledgered.
    "SIM105": 93,
    "SIM102": 39,
    "SIM117": 45,
    "SIM115": 7,
    "UP031": 16,
    # Not ledger-deferred (family not in select) but measurably rotting: R69's
    # parameter-documentation debt, torchlens/ only by design (tests/ has no
    # param-doc policy).
    "D417": 143,
    # The complexity family (r4 b5-fable R44r2-1 HIGH + b5-opus R44-1/2
    # convergent): completely ungated through fixwave-3 — C901 grew 408->441
    # and PLR0912 227->253 with zero tripwire while the round-1 named
    # hotspots were properly fixed. Ceilings frozen at the 2026-08-15 tip
    # measurement (pinned ruff 0.15.4, isolated defaults, torchlens/ only —
    # interior quality debt, not a test-style policy). SHRINK-ONLY like every
    # row above; splitting a hot-path god-function should lower the ceiling
    # in the same change.
    # 2026-08-15 fixwave-5 settle: re-frozen at the post-wave tip (441->445,
    # 189->192, 253->257, 417->418, 146->147) -- the growth is the wave's
    # typed-refusal and teardown-guard branches landing in already-hot
    # functions, not new god-functions. SHRINK-ONLY from here.
    # 2026-08-16 fixwave-6 integration settle: re-frozen at the merged-tree
    # tip (445->450, 192->194, 257->258, 418->419, 147->148; SIM117 41->45
    # above) -- these ceilings were frozen on the fix/infra-r7 lane while the
    # other fixwave-6 lanes merged their reviewed fix branches to main in
    # parallel. SHRINK-ONLY from here.
    # 2026-08-16 fixwave-7 integration settle: re-frozen at the merged-tree
    # tip (450->451, 194->195, 258->262, 419->422, 148->152) -- per-site
    # diff against the fixwave-6 tip shows the growth is the wave's typed
    # refusals, admission-ordering claims, and collapse-ceiling branches
    # landing in already-hot functions (run_and_log_inputs_through_model,
    # _merge_iso_groups_to_layers, from_dict, _check_graph_topology, ...),
    # not new god-functions. SHRINK-ONLY from here.
    # 451->453 (2026-08-17 L8 C2 recording settle): the funcol boundary wrap
    # family (_make_funcol_wrap/wrapped_funcol) mirrors the ledgered c10d wrap
    # shape -- inherently branchy armed/nested/binding/capturing dispatch.
    # Debloat pass keeps the pre-sprint count as its target.
    # C901 453 -> 455, PLR0911 195 -> 196, PLR0912 262 -> 264, PLR0915
    # 155 -> 156 (below): C02's sound stats kernel routes per dtype family
    # and per gate (_float_kernel/run_kernel/_distribution_zone) -- the
    # branching IS the per-family policy table; deliberate, reviewed.
    "C901": 455,
    "PLR0911": 196,
    "PLR0912": 264,
    # 422->423 (same L8 settle): _build_funcol_payload carries the C0 payload
    # argument surface (mirrors the ledgered _build_payload in collectives).
    # 423->425 (2026-08-19 semantic-builds lane): the two new PUBLIC entry
    # points logit_lens (facet/layers/lens/validate/rtol/atol) and
    # bisect_precision (input_kwargs/reference_dtype/rtol/atol/seed) keep the
    # torch.allclose rtol=/atol= convention as separate keywords -- bundling
    # them into a tolerance object would trade a familiar user surface for a
    # lint count. Private helpers were reduced instead of ledgered.
    # 425->426 (2026-08-19 comparative-producers lane): tl.top_changed is a
    # public producer constructor whose surface is the designed API (reference +
    # population + exactly-one-of k/fraction + by/largest); collapsing
    # k/fraction into one dual-typed arg would trade an arg-count point for
    # one-name-two-meanings ambiguity. Two independent lanes each raised this
    # ceiling with a stated reason; the count is their UNION, not a pick-one.
    # 426->427 (2026-08-19 subspace-producer lane): tl.subspace is a public
    # producer constructor whose surface is the designed API (within + basis +
    # mandatory origin= provenance + method/dim/tol); folding origin/method
    # into a provenance object would add a construction step to the honesty
    # requirement the producer exists to enforce.
    # 427->428 (2026-08-19 async-disk lane): tl.to_disk gains the documented
    # async_writes/max_pending_bytes knobs; the flat options factory IS the
    # deliberate public API shape (grouping two user knobs into a sub-object to
    # dodge the count would be a worse surface). FOUR independent lanes each
    # raised this ceiling this sprint with a stated reason; the value is their
    # UNION, measured at merge, never a hand-derived subtotal or a pick-one.
    # 2026-08-27 megasprint C06 (fix cycle 2): the lane's two new offenders
    # (HistoryCollector.__init__ / .step) were refactored below the ceiling
    # instead of ledgered -- plumbing knobs bundle into WatchSettings and the
    # per-step optimizer-truth disclosures into StepTruth (both frozen
    # dataclasses). Ceiling stays at the pre-lane 428.
    "PLR0913": 428,
    # 152->155 (same L8 settle): wrapped_funcol + the criterion-3 census body
    # + capture_completeness_witness gained reviewed statements with plane-P.
    "PLR0915": 156,  # C02 stats kernel (see the C901 note above)
    # The broad-catch / silent-swallow family (grind-r5 b1 R22-1, 4th round,
    # + b7 fable/opus corroboration): the population grew 369 -> 463 AST
    # handlers across the sprint with zero tripwire while every neighbour
    # family got ceilinged -- "every one a place a capture bug can hide".
    # Ceilings frozen at the 2026-08-15 fix/capture-r6 measurement (pinned
    # ruff 0.15.4, isolated, CI scope + config extend-excludes). SHRINK-ONLY.
    # 2026-08-16 R70 r7 explicit re-ledger (instrument correction, see the
    # header note): the parser-blind measurement missed 72 BLE001 and 2 S110
    # notebook sites (audit-notebook demo cells that deliberately catch
    # Exception to DISPLAY refusal behavior, plus two guarded-import
    # try/except/pass cells). True corrected counts: BLE001 523 (451 .py +
    # 72 ipynb), S110 41 (39 .py + 2 ipynb). These are pre-existing sites
    # made visible, not growth; the .py populations did not move. Relayed to
    # the docs lane for notebook-side cleanup; SHRINK-ONLY from here.
    # 2026-08-16 fixwave-7 settle: BLE001 523 -> 528, S110 41 -> 43. The
    # new sites are the wave's deliberate best-effort handlers: the
    # subprocess group-reaper / PDEATHSIG guards (utils/_subprocess.py --
    # post-fork and teardown paths that must never raise), the minted-helper
    # rebuild gate (test_intervention_spec_pickle.py, name-resolution proof
    # that tolerates constructor rejection by design), and belt/rescue
    # teardown guards. Each reviewed; none swallows a capture verdict.
    # 528->539 (2026-08-17 L8 C2 recording settle): the funcol/plane-P/plane-W
    # observers are fail-open BY CONTRACT (an observation failure must degrade
    # to a disclosed gap, never crash or perturb the capture), so their guard
    # excepts are deliberately broad, mirroring the ledgered c10d wrap and
    # witness families. Debloat pass target: pre-sprint 528.
    # 539 -> 540: 2026-08-26 A06 -- the failed-capture preparation release
    # (user_funcs._release_preparation_after_failed_capture) deliberately
    # catches Exception so a secondary release failure can NEVER mask the
    # user's capture exception; it re-surfaces coded
    # (failed_capture_release_incomplete), the same never-mask class as the
    # existing teardown sites.
    # 540->541 (2026-08-26 megasprint A10): snapshot_capture_state's clone
    # loop moved INSIDE its guard (the historical try/except guarded only
    # state_dict(), so unclonable state CRASHED instead of honouring the
    # documented None contract); the guard is deliberately broad because the
    # function's contract is degrade-to-None for ANY state that cannot
    # provide a tensor-only clone map -- pending lazy state refuses TYPED
    # before this guard, so no capture verdict is swallowed.
    "BLE001": 541,
    # 43->44 (same L8 settle): _record_plane_p's swallow-and-continue is the
    # observer fail-open contract stated above.
    "S110": 44,
    # 35->36 (fixwave-5 settle): one new guarded-iteration continue landed
    # with the wave's defensive sweeps; re-frozen at the post-wave tip.
    "S112": 36,
    # grind-r5 b7 R24 (SF-23): every in-package assert strips under
    # ``python -O``, so each new site is a potential optimized-mode semantic
    # split (the roster contradiction guard was the proven instance --
    # torch.tensor silently vanished from the wrapper roster). torchlens/
    # only: tests legitimately assert.
    "S101": 97,
    # The four "proven harmful HERE" permanent [tool.ruff.lint] ignore
    # entries (b9 R70 round 5, second pass): permanent exemption is the
    # right POLICY for these (B009/B010 autofixes traded 214 cosmetic
    # rewrites for 59 new mypy errors; SIM118 caused 11 smoke failures of
    # silent type confusion) but permanent exemption is not unbounded
    # growth. Measured 2026-08-15 with the pinned ruff over the test's own
    # scope/exclude mode; test_ignored_codes_all_carry_ceilings keeps the
    # NEXT ignore entry from entering unmeasured.
    "B009": 120,
    # 2026-08-16 fixwave-7 settle: 100 -> 102. The two new sites are the
    # r8 live-view property overlays for input_ancestors/output_descendants
    # installed with setattr on Op (backends/torch/ops.py), the exact idiom
    # of the two pre-existing overlay rows; direct assignment would type-clash
    # with the declared frozenset field annotations.
    "B010": 102,
    # 2026-08-16 R70 r7 explicit re-ledger (instrument correction, header
    # note): the parser missed 5 SIM118 and 1 SIM401 notebook sites. True
    # corrected counts: SIM118 54 (49 .py + 5 ipynb), SIM401 2 (1 .py +
    # 1 ipynb). Pre-existing sites made visible, not growth; relayed to the
    # docs lane as mechanically fixable. SHRINK-ONLY from here.
    "SIM118": 54,
    "SIM401": 2,
}

#: Codes measured over torchlens/ only (see the D417 and complexity notes
#: above).
_PACKAGE_ONLY_CODES = frozenset(
    {"D417", "C901", "PLR0911", "PLR0912", "PLR0913", "PLR0915", "S101"}
)

_CI_SCOPE = ("torchlens", "tests", "scripts", "tools", "benchmarks", "examples", "notebooks")


def _pyproject_text() -> str:
    """Return the pyproject.toml text."""

    return (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _configured_extend_excludes() -> list[str]:
    """Parse ``[tool.ruff] extend-exclude`` so the isolated run stays in lockstep."""

    match = re.search(
        r"^extend-exclude\s*=\s*\[(.*?)\]", _pyproject_text(), re.MULTILINE | re.DOTALL
    )
    assert match, "pyproject.toml lost [tool.ruff] extend-exclude"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_ruff_select_ratchet_never_narrows() -> None:
    """[tool.ruff.lint] select must stay a superset of the family floor."""

    match = re.search(r"^select\s*=\s*\[(.*?)\]", _pyproject_text(), re.MULTILINE | re.DOTALL)
    assert match, "pyproject.toml lost [tool.ruff.lint] select entirely"
    selected = set(re.findall(r'"([^"]+)"', match.group(1)))
    missing = sorted(_SELECT_FAMILY_FLOOR - selected)
    assert not missing, (
        f"[tool.ruff.lint] select dropped ratcheted families: {missing}. The "
        "family ratchet is grow-only (R70); restore them — narrowing select is "
        "never a fix."
    )


def _count_by_code(concise_output: str) -> Counter[str]:
    """Count violations per code from ruff concise output (pure, testable)."""

    return Counter(
        re.findall(r"^\S+?(?::cell \d+)?:\d+:\d+: ([A-Z]+\d+)", concise_output, re.MULTILINE)
    )


def _measure_deferred_codes() -> Counter[str]:
    """Run the pinned ruff ONCE per scope over every ceilinged code."""

    excludes: list[str] = []
    for pattern in _configured_extend_excludes():
        excludes.extend(("--extend-exclude", pattern))
    counts: Counter[str] = Counter()
    scoped = {
        "ci-scope": [code for code in _DEFERRED_CODE_CEILINGS if code not in _PACKAGE_ONLY_CODES],
        "package-only": sorted(_PACKAGE_ONLY_CODES),
    }
    for scope_name, codes in scoped.items():
        if not codes:
            continue
        paths = _CI_SCOPE if scope_name == "ci-scope" else ("torchlens",)
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruff",
                "check",
                *paths,
                "--isolated",
                "--select",
                ",".join(codes),
                "--output-format",
                "concise",
                "--no-cache",
                *excludes,
            ],
            capture_output=True,
            text=True,
            cwd=_PROJECT_ROOT,
            timeout=300,
        )
        assert completed.returncode in (0, 1), (
            f"ruff invocation failed ({scope_name}): {completed.stderr[-1000:]}"
        )
        counts.update(_count_by_code(completed.stdout))
    return counts


def test_deferred_lint_debt_never_grows() -> None:
    """Every ceilinged code's measured count stays at or below its ceiling."""

    counts = _measure_deferred_codes()
    grown = [
        f"{code}: measured {counts.get(code, 0)} > ceiling {ceiling}"
        for code, ceiling in sorted(_DEFERRED_CODE_CEILINGS.items())
        if counts.get(code, 0) > ceiling
    ]
    assert not grown, (
        "deferred lint debt GREW (the ledger promises deferral, not license):\n  "
        + "\n  ".join(grown)
        + "\nFix the new sites (or, for a deliberate exception, raise the ceiling "
        "in tests/test_ruff_ratchet_governance.py with a stated reason in the "
        "same change)."
    )


def test_deferred_ceiling_parser_is_red_capable() -> None:
    """The count parser attributes planted concise output correctly."""

    planted = (
        "torchlens/a.py:1:1: B023 Function definition does not bind loop variable\n"
        "torchlens/a.py:9:5: B023 Function definition does not bind loop variable\n"
        "tests/b.py:2:3: SIM105 Use `contextlib.suppress`\n"
        # Notebook diagnostics carry a "cell N" path segment WITH A SPACE. The
        # original parser (`^\S+:...`) silently dropped every one of them --
        # BLE001 measured 451 while the true CI-scope count was 523 (R70 r7).
        "notebooks/audit/c.ipynb:cell 4:1:2: BLE001 Do not catch blind exception: `Exception`\n"
        "examples/d.ipynb:cell 12:9:5: B023 Function definition does not bind loop variable\n"
    )
    counts = _count_by_code(planted)
    assert counts == Counter({"B023": 3, "SIM105": 1, "BLE001": 1})
    assert counts.get("B023", 0) > 1  # a ceiling of 1 would trip on this plant


def _configured_ignore_codes() -> set[str]:
    """Parse the ``[tool.ruff.lint] ignore`` code list from pyproject."""

    match = re.search(r"^ignore\s*=\s*\[(.*?)^\]", _pyproject_text(), re.MULTILINE | re.DOTALL)
    assert match, "pyproject.toml lost [tool.ruff.lint] ignore"
    return set(re.findall(r'^\s*"([A-Z]+\d+)"', match.group(1), re.MULTILINE))


def test_ignored_codes_all_carry_ceilings() -> None:
    """`ignore` is a bounded set: every entry has a no-growth ceiling.

    b9 R70 round 5 (second pass): four "proven harmful HERE" permanent
    exemptions sat in `ignore` with no ceiling, so their site counts were
    unmeasured and any NEW code added to `ignore` entered unmeasured by
    default. Permanent exemption is a policy choice; unbounded growth never
    is.
    """

    unceilinged = sorted(_configured_ignore_codes() - set(_DEFERRED_CODE_CEILINGS))
    assert not unceilinged, (
        f"[tool.ruff.lint] ignore entries with no ceiling row: {unceilinged} — "
        "measure each with the pinned ruff and add a shrink-only ceiling in "
        "_DEFERRED_CODE_CEILINGS in the same change"
    )


#: File-level blanket ruff-noqa directives ("# ruff" + ": noqa") in
#: torchlens/ (b9 R70 round
#: 5, F/O pair): each one is a whole-file blind spot no per-code ceiling can
#: see — a NEW dead import in ops.py / _runnable_execution.py /
#: completeness_witness.py (three of the most change-heavy capture files) is
#: permanently invisible to F401. SHRINK-ONLY: converting a blanket to
#: per-line noqa (the repo's own preferred pattern, see
#: validation/invariants.py) lowers this; adding a file may never pass
#: silently.
_BLANKET_NOQA_CEILING = 9


def test_blanket_noqa_directives_never_grow() -> None:
    """The file-level `# ruff: noqa` census stays at or below its ceiling."""

    blanket = sorted(
        str(path.relative_to(_PROJECT_ROOT))
        for path in (_PROJECT_ROOT / "torchlens").rglob("*.py")
        if re.search("^" + "# ruff" + ": noqa", path.read_text(encoding="utf-8"), re.MULTILINE)
    )
    assert len(blanket) <= _BLANKET_NOQA_CEILING, (
        f"file-level blanket ruff-noqa directives grew to {len(blanket)} (ceiling "
        f"{_BLANKET_NOQA_CEILING}): {blanket} — use per-line noqa so F401 "
        "stays armed for genuinely dead code"
    )
