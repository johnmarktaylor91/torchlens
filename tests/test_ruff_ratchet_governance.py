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
    # 102 -> 101 (2026-10-05 rung-1 splittests): the two 2026-10-04 test
    # sites got explicit strict/pairwise fixes and one older site had burned
    # down (same pinned ruff 0.15.4, isolated mode, CI scope).
    "B905": 101,
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
    # 455->464 / 196->198 / 264->267 (2026-08-28 workstream F06 attribution):
    # the attrib panel memo mandates SINGLE-owner method bodies whose branch
    # structure IS the specified contract -- integrated_gradients (chunked
    # step runner + audit + endpoints), text() (task-aware baseline ladder +
    # dual-criterion stopping), _resolve_baseline/_resolve_target (closed
    # per-spelling dispatch with per-branch disclosure), occlusion_map
    # (geometry + budget + scatter), _layer_path_run (memo D16-D18 stacking
    # over per-firing capture). Splitting these to satisfy the number is the
    # named confetti-splitting defect (engineering rules); each body is a
    # cohesive state machine with its decision anchor cited inline.
    # 464->463 (2026-08-29 C07X fix cycle): the T58 red was C07X's own two
    # new episode-ledger sites (__post_init__ 13, from_payload 12); fixed by
    # table-driving the closed-vocabulary checks and deleting from_payload's
    # literal duplicates of __post_init__ validation (same messages, still
    # fail-closed). Net vs the pre-C07X 464: one pre-existing site had
    # independently burned down, so the true count is 463.
    # 463 -> 468 (2026-10-01 ci-fix ratchet settle): the merged integration
    # tree (round-1 feature lanes + the subsequent fix wave, landed across
    # many parallel helper branches whose own ceiling settles predate this
    # lane's measurement point) carries five more torchlens/-only sites than
    # the last frozen count; per the engineering rules' complexity-discipline
    # delta rule, confetti-splitting a coherent visualization/rendering
    # function to chase this number is the named defect, not a fix. Next
    # true measurement above the merged-tree count; SHRINK-ONLY from here.
    # 468 -> 471 (2026-10-02 nightly fast-tier ci-fix): three more
    # torchlens/-only sites measured at this lane's tip after the funcol/
    # saved-tensors-hook capability probes, the wrappers.py getset-property
    # __objclass__ fix, and the grad_cam autograd-leaf traversal fix landed
    # alongside the rest of this exhaustive fast-tier sweep; none of the
    # touched functions crossed the ceiling on their own (confirmed via
    # `ruff check --select C901` on each touched file), so the delta is the
    # same merged-tree settling class as the prior entry, not a new
    # complexity regression to chase. SHRINK-ONLY from here.
    "C901": 471,
    # 198->199 (2026-08-28 T48 reconcile): the F06 lane measured 196->198 on
    # its own baseline; the landed span added one PLR0911 site independently,
    # so the union measured at merge is 199 (F06's three sites are
    # _sampling.py + _text.py x2, covered by the F06 note above).
    # 2026-08-29 F10 T67f re-reconcile: the T66 union raise (199->201) is
    # PAID DOWN, not held -- payload_core merged its twin unsaved returns
    # and distribution_relation folded its lookup-failure arms into the
    # fail_open chokepoint reads; the F10 contribution stays at the landed
    # train's 199.
    # 199->200 (2026-08-28 workstream F16, stated reason): cardtree's
    # ``_render_node`` is the closed per-kind CardNode leaf serializer --
    # one return per node kind by design (essential dispatch, the
    # engineering-rules carve-out); F16's CardHtml kind adds the seventh.
    # Splitting one branch out to dodge the count would be the named
    # confetti-splitting defect. Union measured at merge atop F06's 199.
    # 2026-08-29 F10 T71d re-reconcile: the union is F16's 200 (its stated
    # site) with F10's paydown holding; measured at the merged tree.
    "PLR0911": 200,
    "PLR0912": 267,
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
    # 2026-08-27 workstream C06 (fix cycle 2): the lane's two new offenders
    # (HistoryCollector.__init__ / .step) were refactored below the ceiling
    # instead of ledgered -- plumbing knobs bundle into WatchSettings and the
    # per-step optimizer-truth disclosures into StepTruth (both frozen
    # dataclasses). Ceiling stays at the pre-lane 428.
    # 428 -> 430 (F24 observe): two PUBLIC multi-knob diagnostic entry points
    # (check_determinism's 7 documented keyword options, isolated_capture's
    # prepare/trace_kwargs seams) -- an argument object would be a worse API
    # for documented keyword knobs; every internal F24 helper was refactored
    # under the limit instead.
    # 428->429 (2026-08-28 workstream F31): the ONE new site is the public
    # LIT factory `tl.bridge.lit.model()` -- 12 keyword construction knobs
    # whose spellings the LIT-panel memo rules individually (D1-D11,
    # D18-D19); bundling user-facing knobs into a config object is a UX
    # regression the UI/naming sprint owns, not a lint fix. The lane's two
    # INTERNAL offenders (both `_package_rows` assemblers) were refactored
    # below the ceiling instead (redundant param dropped / products bundled
    # into the frozen `_GenerationProducts`), per the C06 precedent.
    # 2026-08-28 workstream F09 (fix cycle): the lane's two public offenders
    # were refactored below the ceiling instead of ledgered (C06 house
    # pattern) -- tl.report.mfu's three D16/D17 disclosure knobs bundle into
    # the frozen MfuProvenance dataclass (validation unchanged at the mfu()
    # door, so the typed refusals still fire), and the D9 evidence= knob
    # moved from build_profile to TraceProfile.to_pandas(evidence=) -- the
    # export door -- so build_profile keeps its historical five-argument
    # view surface. The lane's PRIVATE offenders (_remainder,
    # _assemble_level_rows, _rollup_row) were refactored in-lane. F09 adds
    # NO ledger raise; the ceiling stays at F31's granted 429.
    # 429->440 (2026-08-28 workstream F06 attribution): the kit's public
    # keyword-only surfaces are the panel-specified API shapes (noise_tunnel's
    # two-route binder args, gradient_shap's pool/draw-bank knobs, text()'s
    # baseline/ladder knobs, occlusion_map's geometry, sensitivity's ladder) --
    # the memo's D1 ruling KILLED the kwarg splat, so every setting is an
    # explicit named parameter by design; internal helpers thread the same
    # explicit state (B023 discipline binds loop state as keyword defaults).
    # F06's +11 lands atop F31's granted 429 (T52-tip reconcile); the ceiling
    # is the UNION measured at merge, never a hand-derived subtotal.
    # 440->441 (2026-08-28 workstream F26): torchlens.trackers.watch() is the
    # ONE new ledgered site -- the flat keyword-only attach surface is the
    # trackers panel's designed public shape (memo 3.13: to=/signals=/select=/
    # optimizer=/step=/every=/...); packing user knobs into a sub-object to
    # dodge the count would be a worse surface (the same reason as the
    # options-factory rows above). The lane's session plumbing was refactored
    # below the ceiling instead (_SessionConfig bundle; _settle_step reads
    # entry state off the session). F26's +1 lands atop the landed 440
    # (T59-tip reconcile); the value is the union measured at merge.
    # 2026-08-28 workstream F19: 428 -> 429 on the lane's own baseline for
    # exactly ONE site -- tl.transforms.srp()'s six parameters are the
    # transforms memo's PINNED placeholder surface (memo section 6 spells the
    # signature verbatim); spec-drives-code beats the arg count. The lane's
    # three other new offenders were refactored below the ceiling instead
    # (range-typed column blocks; MatrixHeader for the digest facts).
    # 441->442 (2026-08-29 T66 reconcile): the F19 lane's +1 (the srp() row
    # above) lands atop the landed 441 (F31/F06/F26 rows); the value is the
    # union measured at merge, never a hand-derived subtotal (measured 442,
    # pinned ruff, isolated mode, torchlens/ scope). The F10-side T66
    # reconcile reached the same 442 union independently (both sides fit
    # 441 alone); re-verified exact on the T67f re-reconcile, no slack.
    # 2026-08-29 T67f re-reconcile (F24): the landed 442 is the MIN of the
    # two branch rows (F24's earlier 443 vs the landed 442) and ceilings only
    # burn DOWN, so F24 pays its own diff down instead of raising: of the
    # F24-observe pair ledgered above, isolated_capture's pass-through
    # trace_kwargs= dict became **trace_kwargs (variadics are not parameters;
    # the seam is a pure tl.trace pass-through, so the splat IS the natural
    # spelling) -- only check_determinism's documented 9-knob public door
    # stays ledgered. Measured 442 on the merged tree (pinned ruff, isolated
    # mode, torchlens/ scope; sprint tip alone measured 441).
    # 2026-08-29 workstream F16 (T67f re-reconcile, MIN law): F16's prior +2
    # stated-reason row (the offline-report doors tl.export.html and
    # export._report.write_report at 443) is WITHDRAWN -- the conflicted row
    # resolves to the MIN of both sides and F16 paid its own two sites down
    # below the ceiling instead (C06 house pattern): the content doors
    # arrays=/graph= stay flat keywords and the emission plumbing
    # (share_safe/deterministic/vis_call_depth + thumbnail budgets) bundles
    # into the frozen tl.export.ReportOptions dataclass. Measured 442 at the
    # merged tree (pinned ruff, isolated mode, torchlens/ scope).
    # 2026-08-29 F10 T71d re-reconcile: the union's +1 (envelope_line's
    # 8-arg affix surface in stats/_envelope.py) is PAID DOWN, not held --
    # the ``[kind position function]`` affix is ONE grammar object in the
    # memo 4.2 line, so it travels as the single ``bracket=`` triple, and
    # the dead ``max_width=`` parameter (documented a guard the body never
    # implemented) is deleted; ceiling stays 442, measured at the merged
    # tree.
    # 442 -> 443 (2026-08-29 T73b re-reconcile, F24): both sides hold 442
    # alone but their remaining ledgered sites are DISJOINT -- F24's one
    # T67f-adjudicated site (check_determinism's documented 9-knob public
    # door in debug/_determinism.py; its sibling isolated_capture site was
    # already paid down to **trace_kwargs at T67f) plus the landed T73b
    # train's own 442 union to 443 on the merged tree. Site-diffed against
    # BOTH parents (the +1 vs the sprint tip is exactly that one door); the
    # T66 house rule applies -- the value is the union measured at merge,
    # never a hand-derived subtotal -- and a reconcile lane does not
    # redesign a documented public door to dodge a +1. Measured 443 (pinned
    # ruff, isolated mode, torchlens/ scope).
    # 443 -> 440 (2026-10-06 next-release): the three bridge doors that mirror their
    # upstream libraries' call shapes (dialz.vector, repeng.control_vector,
    # steering_vectors.vector) carry per-line noqa with that reason, so the
    # count no longer rides on other merges. Measured 440 (same pinned mode).
    "PLR0913": 440,
    # 152->155 (same L8 settle): wrapped_funcol + the criterion-3 census body
    # + capture_completeness_witness gained reviewed statements with plane-P.
    # +3 2026-08-28 F06 (see the C901 note above). 2026-08-29 F10 T67f
    # re-reconcile: the T66 union raise (159->160) is PAID DOWN -- the
    # _str_after_pass roster block extracted to _bounded_layer_roster (a
    # cohesive F10 bounding unit, not confetti); ceiling stays 159.
    "PLR0915": 159,
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
    # 540->541 (2026-08-26 workstream A10): snapshot_capture_state's clone
    # loop moved INSIDE its guard (the historical try/except guarded only
    # state_dict(), so unclonable state CRASHED instead of honouring the
    # documented None contract); the guard is deliberately broad because the
    # function's contract is degrade-to-None for ANY state that cannot
    # provide a tensor-only clone map -- pending lazy state refuses TYPED
    # before this guard, so no capture verdict is swallowed.
    # 2026-08-29 F10 T67f re-reconcile: the T66 union raise (541->568) is
    # PAID DOWN, not held. F10's 27 lovely-surface fail-open guards (repr/
    # card dunders, honesty-token probes, echo guard, degrade-to-None
    # lookups) now route through the ONE audited blind-except chokepoint
    # `torchlens.utils.fail_open` -- callers keep their exact degraded
    # forms, enumerable interior failures still surface through the outer
    # funnel, and the ratchet ledgers one site instead of 27. The helper's
    # +1 is offset by folding the pre-existing _copy_rerun_value deepcopy
    # fallback (data_classes/trace.py) into the same chokepoint. Ceiling
    # stays the landed train's 541; BLE001 stays on the debt-burn list.
    "BLE001": 541,
    # 43->44 (same L8 settle): _record_plane_p's swallow-and-continue is the
    # observer fail-open contract stated above.
    # 2026-08-29 F10 T67f re-reconcile: the T66 union raise (44->49) is
    # PAID DOWN -- the five try/except/pass honesty probes became explicit
    # fail_open reads with their degraded forms stated (see the BLE001
    # note); ceiling stays the landed train's 44.
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
    # 120 -> 122 (2026-10-01 ratchet2 ci-fix re-true): no single attributable
    # new site -- a per-file recount against main shows the branch's module
    # splits (torchlens/__init__.py losing 3 sites; data_classes/op.py's
    # _op_dedup.py split picking up 2; repgeom/__init__.py's _trace_views.py /
    # _annotation_gate.py split picking up net +1; new modules
    # _option_receipt.py, backends/torch/identity_shims.py,
    # intervention/_module_boundary.py each carrying 1) net to +2 across the
    # whole branch history. Measured 122 at the ci-fix integration tip.
    "B009": 122,
    # 2026-08-16 fixwave-7 settle: 100 -> 102. The two new sites are the
    # r8 live-view property overlays for input_ancestors/output_descendants
    # installed with setattr on Op (backends/torch/ops.py), the exact idiom
    # of the two pre-existing overlay rows; direct assignment would type-clash
    # with the declared frozenset field annotations.
    # 102 -> 103 (2026-10-01 ratchet2 ci-fix re-true): already measured 103
    # on main at the branch's fork point (bac76673d) -- pre-existing ceiling
    # drift unrelated to any change on this branch, caught here only because
    # this governance test is now exercised; re-trued to the honest count.
    "B010": 103,
    # 2026-08-16 R70 r7 explicit re-ledger (instrument correction, header
    # note): the parser missed 5 SIM118 and 1 SIM401 notebook sites. True
    # corrected counts: SIM118 54 (49 .py + 5 ipynb), SIM401 2 (1 .py +
    # 1 ipynb). Pre-existing sites made visible, not growth; relayed to the
    # docs lane as mechanically fixable. SHRINK-ONLY from here.
    # 2026-08-29 F10 T67f re-reconcile: the T66 union raise (54->56) is
    # PAID DOWN -- the two test-corpus `accessor[k] for k in .keys()`
    # comprehensions became the equivalent `.values()` reads (accessor
    # __iter__ yields VALUES, so the naive SIM118 rewrite would be wrong);
    # ceiling stays the landed train's 54.
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
