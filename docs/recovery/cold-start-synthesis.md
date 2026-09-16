# Cold-start optimization -- CROSS-LAB SYNTHESIS (Fable + sol, 2026-07-11)

Two independent adversarial planning passes (Fable = PLAN_fable.md, sol = PLAN_sol.md) + JMT/CC direct
measurement CONVERGED. This is the shovel-ready substrate for the build (fires after spine + tlspec).

## VERIFIED FACTS (both labs + direct measurement agree)
- **Premise was STALE.** fastlog / intervention / user_funcs / data_classes are ALREADY lazy on current
  main (`_LAZY_ATTRS`, pinned by `test_import_hygiene.py`). The tracker's "~15 eager islands" is mostly done.
- **torchlens' OWN marginal overhead is tiny: ~52-57ms** over bare `import torch` (sol median 52ms; CC
  importtime ~57ms). Remaining real eager cost: `captured_run`+`ir` ~40ms, `options` ~7ms, a few ms more.
- **The total is TORCH-BOUND.** `import torch` alone = ~1.24s (CC warm best-of-3) up to ~1.7-1.85s (Fable's
  box) -- machine/cache dependent. The tracker's "2.5s" was cold-cache. torchlens code CANNOT reliably move
  the total under 1.5s; that number is torch's, not ours.

## THE ONE DECISION FOR JMT (non-blocking; rides in this plan)
The `<1.5s TOTAL` gate is torch-bound + machine-dependent => not a deterministic CI target. Both labs
recommend: **define the gate as torchlens INCREMENTAL overhead over torch** (deterministic, CI-guardable via
an eager-module-set pin). CC RECOMMENDATION: gate = torchlens-marginal **<= 10ms** after the final stage
(investigate >25ms); keep whole-process `<1.5s` as a reported DIAGNOSTIC only. REJECT also-deferring
`import torch` (Fable/sol both note it observably changes `sys.modules` -- wrong for a torch tool).

## REAL WORK (small -- a cleanup, NOT a sprint)
Shave torchlens-marginal 52ms -> ~5-10ms by lazifying the last eager islands in `torchlens/__init__.py`
(`captured_run`, `ir`, `options`, `observers`, `quantities`, `errors`) onto the facade, in measured stages.

## RISKS BOTH LABS FOUND (must be handled in the build -- ZERO observable change is the bar)
1. **Side-effect submodule binding (LIVE in the repo's own tests).** Lazifying kills module-level bindings
   like `tl.options.CaptureOptions` (25+ test sites) and `torchlens.visualization.summary`
   (`test_api_surface_deprecation.py:71-73`). FIX: `(path, None)` facade parity entries for ALL side-effect-
   bound modules (incl. private `_state`/`_io`/`_errors`/`_literals`/`_deprecations`) + attribute-
   reachability regression test. Note (sol line 241/496): some names already DON'T resolve after bare
   `import torchlens as tl` until a submodule import -- preserve current behavior exactly, don't "fix" it.
2. **Static-typing / IDE downgrade.** Moved names become `Any` under mypy/pyright/jedi. FIX: `TYPE_CHECKING`
   mirror imports (`X as X`) for all moved public names + a reveal_type probe gate.
3. **`test_import_hygiene.py` pins the facade-collision dict + eager module set EXACTLY.** Adding facades
   trips it; extend the pin with honestly-audited collisions in the SAME commit (never loosen to pass).
4. **`sys.modules` membership changes** (the source of the speedup) -- ensure no DOCUMENTED public module is
   made unreachable; add the desired-cold-allowlist test.
5. Deferred-ImportError risk audited to ~zero (all deps hard; safetensors/graphviz aren't loaded at import
   today). Wrapping-timing invariance proven trivial (backends not in the eager set; wrap path already lazy).

## DOC DRIFT TO FIX IN THE BUILD (Fable)
- `torchlens.__all__` is **92**, not 90 (CLAUDE.md / glossary / MEMORY say 90).
- CLAUDE.md's two eager-import descriptions + tracker import lines are stale (say the now-lazy modules are eager).

## VERIFICATION (merged from both)
Fresh-process measurement protocol (11 scored samples, median/p90, raw importtime as CI artifact); the
torchlens-marginal <=10ms deterministic backstop; "all 92 __all__ names resolve" test; `dir()` completeness;
wrapping-timing test (torch clean after import, wrapped after first capture); mypy/pyright clean; ruff; smoke.
