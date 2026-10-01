"""File-size ratchet: god files may shrink, never grow unnoticed.

R43 (4th pass): 42 files in ``torchlens/`` exceeded 2000 lines, the top five
all GROWING through fix waves (+996 lines on ``data_classes/op.py`` in ~200
commits), and no size ceiling existed anywhere — no ruff rule, no test, no
hook. This ratchet freezes the frontier:

- an UNLEDGERED module may not exceed ``_NEW_FILE_LINE_CAP`` (2000 — the
  repo's own god-file threshold; the style rule remains 800, enforced by
  review, not this backstop);
- a LEDGERED module may not exceed its frozen ceiling (current size at
  ratchet time, rounded up to the next 50 for mid-wave slack). Growing a
  ledgered file is a CONSCIOUS act: raise its ceiling in the same change,
  with a reason, and expect review pushback — the intended direction is DOWN
  (split along the seams catalogued in the R43 findings);
- a ledgered module that drops to/below the cap must LEAVE the ledger
  (two-way staleness, so the ledger cannot rot into permanent exemptions).

GENERATED modules (self-declared via a ``GENERATED`` marker on the first
docstring line, same convention as the ruff extend-exclude lockstep) are
exempt: their generator is the authority for their size.

r7 R43-F1 (opus MED): ``tests/`` joins the ratchet with its own ledger — it
had grown +5,711 lines in 112 commits with no gate, and the repo's largest
file was the tripwire corpus itself (``test_validation.py``, 8,477 lines,
1.6x the worst package god file; its collection alone costs 6.35s, paid by
every one of the 223 mutants in a campaign).

r7 R43-F2 (opus LOW-MED): ceilings carry a RE-KEY OBLIGATION — a row sitting
more than ``_MAX_LEDGER_SLACK`` under its ceiling must be re-keyed down to
the next 50-line step, so a successful split can never rot into permanent
regrowth budget (headroom had doubled 1,090 -> 2,162 before the fixwave-5
re-key practice; this makes the practice structural).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_ROOT = _PROJECT_ROOT / "torchlens"

#: Hard line cap for any module not in the ledger below.
_NEW_FILE_LINE_CAP = 2000

#: Frozen ceilings for the god-file frontier (R43 census, 2026-08-15, rounded
#: up to the next 50). SHRINK-ONLY DOCTRINE: lower a ceiling freely; raising
#: one requires a stated reason in the same change. When a file drops to
#: <= 2000 lines, DELETE its row (the staleness check below enforces this).
#: 2026-08-15 fixwave-3 settle: nine ceilings re-rounded up (validation/core,
#: utils/rng, _io/bundle, user_funcs, backends/torch/{backward,model_prep},
#: visualization/auto_collapse, _capture_state_helpers, capture/trace) — the
#: census was taken per-lane while 12 fix lanes merged in parallel, so each
#: lane's fix growth landed after its territory's ceiling was frozen.
#: 2026-08-15 fixwave-5 settle: 16 ledgered ceilings consciously raised to the
#: next 50-line step above their post-wave sizes -- the growth is the wave's
#: reviewed defensive fixes (typed refusals, R56 teardown guards, provenance
#: sentinels), not drift. tensor_utils.py instead SHRANK below the unledgered
#: cap by splitting the r37 INV-2 alias engine into utils/alias_footprint.py.
#: 2026-08-16 fixwave-6 integration settle: 10 ceilings re-stepped to the next
#: 50 above the union-tree measurement -- this ledger was frozen on the
#: fix/infra-r7 lane while the other fixwave-6 lanes (capture/iomerged/
#: valid-conc/vizgraph) merged their reviewed fix growth to main in parallel;
#: the raise reconciles the branch-frozen census with the merged tree.
#: 2026-08-16 fixwave-7 settle: six ceilings re-stepped to the next 50 above
#: the merged-tree measurement (bundle 4350->4450, scrub 2350->2400,
#: loop_grouping_adapter 2500->2600, user_funcs 3850->3900, exemptions
#: 2950->3000, auto_collapse 2400->2450) -- reviewed fixwave-7 growth
#: (buffer-value channel gating, R29 lazy pair generation, admission
#: ordering, collapse-ceiling honesty) landing in already-ledgered files.
#: 2026-08-16 FEATURE MEGASPRINT wave-0 settle: seven ceilings re-stepped to the
#: next 50 above the merged-tree measurement. UNLIKE the fixwave raises above,
#: this growth is NEW FEATURE MASS by design (L1 grouping, L2 episode capture,
#: L3 aten layer, L4 S1 contract, L5 encoding channel, L7a structure-only, L8
#: census), landing in already-ledgered files. The METAPLAN defers the DEBLOAT
#: pass to the post-features mega-hardening round, so this is a conscious raise
#: with a DEBT RECORD, not an accepted new normal. PRE-SPRINT BASELINE at
#: kickoff 75439a67 (the debloat pass's target to return to or beat):
#:   _io/bundle.py 4450 | backends/torch/backward.py 3800
#:   bundle/__init__.py 2450 | capture/trace.py 2200
#:   data_classes/trace.py 3750 | options.py 2425 | user_funcs.py 3900
#:   _runnable_state.py 2700 (L4 D18 mode-aware projector + snapshot-restore)
#:   visualization/_render_edges.py 2350 (portable rolled-edge label placement)
#: 2026-08-16 L6 merge settle (same debt record): three more ceilings
#: re-stepped for reviewed L6 selection-algebra / edge-substitution mass
#: (edge boundary verdict in validation/core, tier-(ii) edge stores on Op,
#: the v7 save boundary in _io/bundle). PRE-SPRINT BASELINE at 75439a67:
#:   validation/core.py 5300 | data_classes/op.py 5200
#:   (_io/bundle.py already listed above at 4450; L6 re-steps 4500 -> 4550)
#: 2026-08-17 gov-sweep settle: the L6 edge-boundary check family moved out
#: of validation/core into validation/_edge_boundary.py; core re-keyed
#: 5450 -> 5350 (measured 5325).
#: 2026-08-17 L9 merge settle (same debt record): two ceilings re-stepped for
#: reviewed L9 backward-residuals mass -- per-fire timing (keyed-LIFO prehook
#: + pairing), checkpoint invocation tokens (classifier + witness + D1-D6),
#: and the journal/scavenge/finalize implicit-close split all land in
#: backends/torch/backward.py (3900 -> 4450); the two DROP-gated Trace fields
#: + init/load-fill land in data_classes/trace.py (3850 -> 3900).
#: PRE-SPRINT BASELINES unchanged (backward.py 3800, trace.py 3750 at
#: 75439a67); the post-features debloat pass keeps both as targets.
#: 2026-08-17 L8 C2-recording merge settle (same debt record): three ceilings
#: re-stepped for reviewed merge-ranks C2 recording mass -- the plane-W
#: shard-local erasure-prevention chokepoint in _io/bundle.py (4550 -> 4600),
#: the plane-P record branch + rebinds in backends/torch/completeness_witness
#: (2050 -> 2100), and the funcol group-resolution / wait-interposition
#: capability probes in utils/_torch_compat.py (3450 -> 3550). PRE-SPRINT
#: BASELINES unchanged (bundle 4450, _torch_compat 3450-eve, witness 2050-eve
#: at 75439a67); the debloat pass keeps all three as targets.
#: 2026-08-26 P03 fix cycle: options.py (2371, over its 2300 ceiling after the
#: compo option-receipt landed) SPLIT along its natural seams instead of a
#: raise -- the value validators moved to _options_validation.py and the
#: explicitness readers + option receipt to _option_receipt.py; at 1950 lines
#: it drops under the unledgered cap and its row is DELETED per the two-way
#: staleness rule.
_GOD_FILE_CEILINGS: dict[str, int] = {
    "torchlens/validation/core.py": 5350,
    # 5250 -> 5100 (2026-08-27 C07 fix cycle): the user-transform apply +
    # validation helpers split to _op_transforms.py under R43 (the v9
    # injection_provenance rows nudged op.py over); re-keyed down to the
    # next 50-line step above the measured 5069.
    # 5100 -> 5000 (2026-08-29 F10 reconcile): the F10 lovely-surfaces lane
    # deleted op.py's legacy string helpers; re-keyed down to the next
    # 50-line step above the merged measurement (4970). Ceilings follow
    # files down (R43-F2).
    # 5000 -> 4800 (2026-08-29 F20 re-reconcile): the F20 saved-activation
    # dedup split to _op_dedup.py (T77) lands on the merged tree alongside
    # the F10 deletions; re-keyed down to the next 50-line step above the
    # merged measurement (4794).
    "torchlens/data_classes/op.py": 4800,
    "torchlens/_io/runnable.py": 5000,
    "torchlens/utils/rng.py": 4950,
    # 4600 -> 4350 (2026-08-27 C05 fix cycle): the segment descriptor/label
    # family split to _segment_descriptors.py under R43; re-keyed down to the
    # next 50-line step above the post-split measurement (4326).
    "torchlens/visualization/collapse_optimizer.py": 4250,
    # 4400 -> 4403 (2026-08-27 C01 item 5): the _selective_save relocation
    # re-sorted one import into a 4-line parenthesized block (+3 mechanical
    # lines, zero behavior); the god file itself did not grow.
    "torchlens/backends/jax/backend.py": 4403,
    # 4600 -> 4650: the L8 C2-recording settle above re-stepped bundle to 4600
    # but the merged file MEASURES 4603 -- the settle's own re-step was three
    # lines short, red on main since the merge. Reconciled to the next 50-line
    # step for the same reviewed mass (no new growth licensed; PRE-SPRINT
    # BASELINE 4450 unchanged, debloat target unchanged).
    # 4660 -> 4650 (2026-08-29 F01-AMENDED reconcile): the F01 injections
    # stage-1 save-entry refusal slot (+7: comment, lazy import, call) now
    # rides the tip's debloated bundle (union measures 4600) -- the lane's
    # 4660 raise is unnecessary and the ceiling burns back down.
    "torchlens/_io/bundle.py": 4650,
    "torchlens/_io/runnable_load.py": 3850,
    # 4200 -> 4050: the wave-0 governance sweep extracted the structure-only
    # Layer-0 entry contract to capture/_structure_only_entry.py; re-keyed
    # down to the next 50-line step above the post-split measurement.
    # 4050 -> 4055 (2026-08-19 async-disk lane, measured at merge): tl.to_disk
    # gains the documented async_writes/max_pending_bytes knobs. The lane settled
    # its own ceiling to 4050 BEFORE rebasing; main had independently grown the
    # same file, so 4050 was a pre-rebase subtotal and the merged truth is their
    # union. Stepped EXACT (no 50-line slack) so the next raise is also conscious.
    # 4055 -> 3800 (2026-08-26 shim removal): the flat-kwarg warning ladders,
    # moved-name wrappers, and paper-era shims came out of trace()'s entry;
    # re-keyed down to the next 50-line step above the post-deletion measure.
    # 3800 -> 4200: 2026-08-26 A06 defect-lane settle (reviewed fix mass, next
    # 50-line step above the 4150 measurement) -- stop_after halt-engine wiring
    # + never-fired provenance split, save_grads predicate routing, the
    # module_filter zero-save disclosure, the cache-key inversion (curated /
    # neutral ledgers + sweep), chunk-path session-knob forwarding, the
    # batchnorm train-stats warn-once, and the failed-capture preparation
    # release. No new growth licensed; the Phase-2 move-class lane keeps
    # user_funcs as a split target (trace-entry resolution vs cache vs
    # chunking are its natural seams).
    # 4200 -> 4230 (2026-08-28 F33 weightsfree): the W2 admission threading
    # (admit_meta derivation + pending-admission arm around the capture
    # driver) and the settlement seam (wrap-generation stamp + envelope
    # write) land at the trace entry by design; conscious raise, split
    # target unchanged.
    # 4230 -> 4235: C07X adds the unconditional Trace.root_entry_point
    # identity-fact write (foldA D10) at the one-door capture site -- a
    # 2-line addition with no extractable seam; a split here would be
    # confetti (R43: never split code merely to satisfy the number).
    # 2026-08-29 F01-AMENDED fix cycle (T76 bounce): the lane's 4235 -> 4265
    # raise was reverted by moving the funnel/arming bodies to
    # intervention/model_door.py and intervention/injection.py; the F42
    # reconcile drops the lane-side 4235 duplicate row -- the union tree
    # carries F41's reviewed mass under the 4290 row below.
    # raise for the F01/OP2 model-door funnel + log_injections threading is
    # REVERTED (ceilings never bump; C06/F09/F21 precedent) -- the funnel and
    # arming bodies moved to intervention/model_door.py and
    # intervention/injection.py, leaving one-line door/arm spellings plus the
    # coverage-pinned CaptureOptions threading rows at the trace entry.
    # 4235 -> 4250 (2026-08-29 T73b re-reconcile, F24): merge union -- each
    # side fits 4235 alone; the F24-observe entries and the landed T73b
    # train mass union to 4238 on the merged tree. Next 50-line step above
    # the merged measurement, never a hand-derived subtotal.
    # 4235 -> 4290 (2026-08-29 F41 bound-method roots): the ruled root
    # contract lands at the one-door trace entry by design -- the
    # bound-method wrapper swap + refusal teach, the owner-identity /
    # bound_method root-fact derivation, the input-ladder module-root gate,
    # and the tl_authored_root marker write; the wrapper CLASS itself lives
    # in backends/torch/bound_root.py. Conscious raise; split target
    # (trace-entry resolution vs cache vs chunking) unchanged.
    # 4250/4290 -> 4290 (2026-08-29 T80 re-reconcile, F24): duplicate-key
    # collapse -- the landed F41 root-contract mass unions with the
    # F24-observe entries to 4284 on the merged tree; the landed 4290
    # ceiling already clears the union and stays.
    # 4235/4290 -> 4290 (2026-08-30 T82d re-reconcile, F44): duplicate-key
    # collapse -- the F44 model-door/log_injections rows union with the
    # landed F41+T82d mass to exactly 4290 on the merged tree (the landed
    # *_value locals finish F44's ratchet-paydown direct-read style); the
    # landed 4290 ceiling holds and stays.
    # 4235/4290 -> 4290 (2026-08-30 F43 re-reconcile): duplicate-key
    # collapse again -- the T82d landed track_device_memory threading unions
    # with the F43-side log_injections threading to exactly 4290; the merge
    # keeps the T76 inlined capture_options.X spellings (the landed value-
    # extraction locals were dead after the paydown) so the ceiling holds
    # without a raise.
    # 4290 -> 4350 (2026-08-29 F28 re-reconcile): merge union -- the F41
    # bound-method-root mass (4274) fits 4290 alone, and the F28 echo facade
    # residual (+40: the echo= entry param, docstring, and the four thin
    # snoop._entry seam calls; the lane already paid its extraction to
    # torchlens/snoop/_entry.py at its own gate debt commit) rode the F28
    # branch; the union measures 4314. Next 50-line step, conscious raise
    # with both contributions named, not silent regrowth; split target
    # (trace-entry resolution vs cache vs chunking) unchanged.
    # 4350/4290 (2026-08-30 T82d re-reconcile, F28): duplicate-key collapse --
    # the landed T82d train mass (F40c+F22, 4290 alone) unions with the F28
    # echo residual to 4328 on the merged tree; the F28-side 4350 ceiling
    # already clears the union and stays.
    # 4290/4350 -> 4350 (2026-08-30 T85 re-reconcile, F44): duplicate-key
    # collapse -- the landed T85 mass (F28 echo + F01-AMENDED, 4330 alone)
    # already carries the F44 log_injections rows (F01-AMENDED landed the
    # injections law), so the merged tree measures exactly 4330; the landed
    # 4350 ceiling clears it and stays.
    # 4290/4350 (2026-08-30 F43 re-reconcile vs T85): duplicate-key collapse --
    # the landed T85 mass (F01-AMENDED log_injections + F28 echo, 4328 alone)
    # unions with the F43-side threading to 4330 on the merged tree; the
    # landed 4350 ceiling already clears the union and stays.
    "torchlens/user_funcs.py": 4350,
    "torchlens/backends/torch/backward.py": 4450,
    # 3800 -> 3850: L1 adds the grouping knob mirror + grouping_policy stamp
    # settlement (~25 lines) on top of the re-stepped feature-sprint baseline.
    # 3850 -> 3900: L9 adds the two DROP-gated backward-residuals fields
    # (timing provenance, checkpoint witness) + init/load-fill/registration.
    # 3900 -> 3925 (F24 observe): session-time FieldPolicy.DROP rows +
    # load-fill defaults for the observe stores (saved-band decomposition,
    # device-memory samples, nonfinite-prefix facts); the machinery itself
    # lives in torchlens/observe and torchlens/capture/_nonfinite_prefix.py.
    # 3900 -> 3940 (2026-08-28 F33 weightsfree): Trace.check_plan (the D14
    # audit-only plan-check verb) is a Trace method by contract (frozen root
    # budget: no new tl.* name); conscious raise.
    # 3925/3940 -> 3950 (2026-08-29 F24 reconcile): both lanes' rows coexist
    # on the merged tree; re-keyed to the next 50-line step above the merged
    # measurement (3927), never a hand-derived subtotal.
    # 3940 -> 3950 (2026-08-29 F10 re-reconcile): merge union -- F10's
    # lovely Trace surface (3935) and the T71d train (3923) each fit 3940
    # alone; the union measures 3948. Next 50-line step, conscious raise
    # with both contributions named, not silent regrowth.
    # 3950 -> 3955 (2026-08-29 F42 reconcile): the union of two green
    # parents (the T80 tip at its exact 3950 ceiling + the F01-AMENDED lane
    # edits) measures 3951 -- same reviewed mass, no new growth licensed.
    # 3950 -> 4000 (2026-08-29 T73b re-reconcile, F24): both sides' rows
    # coexist and each reached 3950 independently, but the F24-observe DROP
    # rows (3927 alone) and the F10 lovely surface + T71d train (3948 alone)
    # UNION to 3965 on the merged tree -- next 50-line step above the merged
    # measurement, never a hand-derived subtotal.
    # 2026-08-30 F42 re-reconcile: the T82d landed rows (3965-measured union)
    # + the F42 coupling/F01-AMENDED edits (3951 alone) UNION to 3968 on the
    # merged tree -- the landed 4000 ceiling already covers it; no new step.
    "torchlens/data_classes/trace.py": 4000,
    # 3550 -> 3800: fix/private-probe-routing moved the last 9 stray private
    # torch touches (funcol module/ACT/wait-redispatch, checkpoint hook class,
    # engine queue_callback) behind named HAS_* families IN this file -- the
    # LOCKED CLAUDE.md rule pins the chokepoint to this exact module, so the
    # routing mass lands here by design (measured 3756). PRE-SPRINT BASELINE
    # unchanged (3450-eve at 75439a67); the debloat pass keeps it as target.
    # 3800 -> 3850 (2026-08-28 F04 one-backward reads): the GradientEdge /
    # Node-prehook capability probes (HAS_GRADIENT_EDGE, HAS_NODE_PREHOOK)
    # land at the LOCKED chokepoint by design (measured 3814). PRE-SPRINT
    # BASELINE unchanged; the debloat pass keeps 3450-eve as target.
    # 3850 -> 3990 (2026-08-28 F27): the Kineto in-memory event contract, the
    # scope probe, and the memory-profile parity accessor land at the ONE
    # sanctioned private-probe chokepoint BY LAW (the private-probe gate
    # forbids these touches anywhere else), so the chokepoint grows exactly
    # when the probe inventory does. Debloat target unchanged: 3450.
    # 3990 -> 4050 (2026-08-29 F27 reconcile): the merged tree carries BOTH
    # the F04 GradientEdge/Node-prehook probes and the F27 Kineto/scope/
    # memory-parity probes at the one sanctioned chokepoint (measured 4001);
    # next 50-line step. Debloat target unchanged: 3450.
    "torchlens/utils/_torch_compat.py": 4050,
    # 3400 -> 3300 (2026-08-26 shim removal): the crawler-era no-op stubs and
    # patch_policy/patch_modules warn kwargs left; next 50-line step down.
    # 3300 -> 3320 (F24 observe): the device-memory bracket at the one
    # wrapped-call site (before-read, OOM attempted-call row, settle); the
    # sampler/provider live in torchlens/observe/_device_memory.py.
    # 3300 -> 3330 (2026-08-28 F33 weightsfree): the W1-CTX slot consult in
    # the factory-injection helper + the W1-BUF-2 snapshot-carrying-call
    # fallback + the meta-safe alias-key reads; conscious raise, the 50-line
    # step-down target resumes after the sprint.
    # 3320/3330 -> 3350 (2026-08-29 F24 reconcile): both lanes' rows coexist
    # on the merged tree; re-keyed to the next 50-line step above the merged
    # measurement (3337). The post-sprint step-down target is unchanged.
    # 3330 -> 3300 (2026-08-29 F27 reconcile): F27's marker split moved the
    # _op_markers/_gradfn_markers mass out of wrappers (merged measurement
    # 3284); re-keyed down to the min of the merge parents.
    # 3350/3300 -> 3350 (2026-08-29 T80 re-reconcile, F24): duplicate-key
    # collapse -- the landed F27 marker split (3284 on the T80 tree) unions
    # with the F24 device-memory bracket to 3302 on the merged tree; next
    # 50-line step above the merged measurement, never a hand-derived
    # subtotal. The post-sprint step-down target is unchanged.
    "torchlens/backends/torch/wrappers.py": 3350,
    # 3300 -> 3301 (2026-08-27 C01 item 5): same relocation import re-sort (+1).
    "torchlens/backends/tinygrad/backend.py": 3301,
    "torchlens/backends/mlx/backend.py": 3250,
    # A07 (2026-08-26): +9 lines -- the step-1 contract gains the
    # flops_forward/flops_backward boundary-reset writes and their pinned-pair
    # carrier rows (alias rows own no compute); conscious raise, not growth debt.
    # F20 (2026-08-28): +13 for the reviewed step-11 contract diff (the D-17
    # byte model's declared payload reads + the (1,11) pinned-pair carriers).
    # This file is the postprocess contract REGISTRY: it grows exactly when a
    # reviewed contract diff lands, which is its design, not god-file rot.
    # F20 T68c reconcile (2026-08-29): +30 to the measured 3301 -- the tlspec
    # v9 entry-dark columns join step 18's hand-derived read set + probe rows
    # (episode_step / injection_provenance / tl_authored_root), and step 11
    # gains the out_ref lazy-materialization probe surfaced by the F20 refresh
    # on lookback axes. Reviewed contract diffs; re-step to the next 50.
    # 3350/3258 (2026-08-30 T85 re-reconcile, F20): duplicate-key collapse --
    # the landed T85 train (3258 alone) unions with F20's reviewed contract
    # additions to 3301 on the merged tree; the F20-side 3350 ceiling already
    # clears the union and stays.
    "torchlens/postprocess/_contracts.py": 3350,
    # 3200 -> 3250 (2026-08-29 F28 re-reconcile): merge union -- the F41
    # owner-identity source-metadata derivation (3184) fits 3200 alone, and
    # the F28 echo module-seam emits (duck-typed _echo_session reads at the
    # module enter/exit frames, 3200 exact on the F28 branch) rode the F28
    # side; the union measures 3212. Next 50-line step, conscious raise with
    # both contributions named, not silent regrowth.
    # 3250/3200 (2026-08-30 T82d re-reconcile, F28): duplicate-key collapse --
    # the landed T82d train (3200 alone) unions with the F28 echo module-seam
    # emits to 3212 on the merged tree; the F28-side 3250 ceiling already
    # clears the union and stays.
    "torchlens/backends/torch/model_prep.py": 3250,
    # 2950 -> 2800 (2026-08-29 F11 T74b fix): the call-tree display + call-scope
    # resolution helper family split to data_classes/_call_tree.py after the
    # merged tree crossed the frozen 2950 (F10's Module lovely seam +10 atop
    # 2948); ceilings follow files down (R43-F2). Next 50-line step above the
    # post-split measurement (2750; ~2760 with the F10 union). Supersedes the
    # F10-side 2950 -> 3000 raise (dropped at the T74b reconcile merge) --
    # the bounce froze this ceiling, so the split is the fix, not the raise.
    "torchlens/data_classes/module.py": 2800,
    # 2100 -> 2150 (2026-08-29 F11 T74b fix): +9 lines documenting the
    # `fingerprints` parameter on the three run-fold assemblers (the D417
    # lint-ratchet fix for this same bounce); doc mass, not code growth.
    "torchlens/visualization/auto_collapse.py": 2150,
    "torchlens/validation/exemptions.py": 3000,
    "torchlens/backends/paddle/backend.py": 2700,
    "torchlens/_runnable_state.py": 2850,
    "torchlens/capture/arg_positions.py": 2650,
    "torchlens/backends/jax/jaxpr.py": 2550,
    "torchlens/_capture_state_helpers.py": 2350,
    # 2500 -> 2550 (2026-08-29 F10 re-reconcile): merge union -- the F03
    # split base measured 2464 and the F10 lovely bundle card (outcome
    # distribution repr + bounded __str__) adds +44; the union measures
    # 2508. Next 50-line step, conscious raise with both contributions
    # named, not silent regrowth.
    "torchlens/bundle/__init__.py": 2550,
    # Re-keyed 2400 -> 2100 at the F16/T60 fix cycle (2026-08-29): the file
    # crossed its ceiling (2406), so OpAccessor/LayerAccessor split to
    # _layer_accessors.py (the _layer_spec.py precedent); layer.py re-exports
    # both names. 2097 measured, next 50-line step.
    # 2100 -> 2150 (2026-08-29 F10 re-reconcile): merge union -- the F16
    # accessor-split base measured 2097 and the F10 lovely Layer card
    # (__str__/__repr__/_detached_from_trace; the accessor delta re-landed
    # in _layer_accessors.py) adds +36; the union measures 2133. Next
    # 50-line step, conscious raise, not silent regrowth.
    "torchlens/data_classes/layer.py": 2150,
    # Re-keyed 2750 -> 2500 at the F03 T69 fix microlane (2026-08-29): the
    # module-level delta/compare helper seam (delta_map/norm_delta/
    # output_delta/compare/aligned_pairs/show_diff + shared metric
    # primitives) split out to bundle/_deltas.py; measured 2464.
    # 2350 -> 2400: C02 safety tranche lands the OpAccessor basis fix
    # (bug 27) with its coherent get/repr in-place -- the accessor lives
    # with its Layer owner; +11 measured lines, next 50-line step.
    "torchlens/postprocess/loop_grouping_adapter.py": 2600,
    "torchlens/visualization/_render_leaf.py": 2400,
    # Re-keyed down 2450 -> 2350 (r7 R43-F2 slack rule) after the F13 legend
    # rewrite deleted the six-node emitter from this module.
    "torchlens/visualization/_render_edges.py": 2350,
    # Re-keyed 2400 -> 2300 at the C03 fix cycle (2026-08-27): the site-key-first
    # compat checker (check_spec_compat + SpecCompat/TargetManifestDiff and its
    # private helpers) moved to intervention/spec_compat.py, decomposed under
    # the complexity ratchet; save.py re-exports the public names. Ceilings
    # follow files down (R43-F2); save.py stays on the debloat-pass list.
    "torchlens/intervention/save.py": 2300,
    # Raised 2400 -> 2425 at the facet-cache persistence fix (2026-08-17): the
    # +22 lines are the TEACHING half of the completeness refusal -- for an
    # undeclared `_<name>_cache` cell backed by a public property it now names
    # the accessor that populated it and states the DROP remedy, instead of
    # naming an internal field the user never touched. The refusal itself was
    # not weakened; only its message got smarter. Reviewed raise with a stated
    # reason (the ratchet working as intended), NOT silent god-file regrowth --
    # and scrub.py remains on the debloat-pass list as a genuine god file.
    # Raised 2425 -> 2445 at the draw()-poisons-save fix (2026-08-17): the
    # +20 lines enroll `_last_encoding_state` in the Trace runtime-only set
    # and extend the same teaching refusal to LEDGERED-but-undeclared Trace
    # transients (it now quotes the TRACE_EXTERNAL_WRITE_EXEMPTIONS row and
    # states that a ledger row is documentation, not a scrub policy). Same
    # reviewed-raise class as the row above; scrub.py stays on the
    # debloat-pass list.
    # Raised 2445 -> 2500 at the A08 persistence-honesty lane (2026-08-26): the
    # +30 lines are the W3 weightsfree fix -- the structure-only buffer-payload
    # strip (BN running stats are training-derived state a weights-free
    # artifact must not carry; every structure-only save of a buffer-holding
    # model was save-then-cannot-load at the M-C2 gate). Reviewed raise with a
    # stated reason; scrub.py stays on the debloat-pass list.
    # 2026-08-26 A09 fix cycle: source-embedding privacy policy family split
    # out to _io/_source_privacy.py. Re-keyed ONCE at the T12 rebase for the
    # final shape carrying both the A08 W3 raise and the A09 split (measured
    # 2069; next 50-step). Shrink freely, raise consciously.
    "torchlens/_io/scrub.py": 2100,
    "torchlens/debug/_infer_input_shape.py": 2250,
    "torchlens/postprocess/ast_branches.py": 2250,
    "torchlens/visualization/_render_nodes.py": 2150,
    "torchlens/_io/_safe_unpickle.py": 2100,
    "torchlens/visualization/_render_flow.py": 2100,
    # Raised 2250 -> 2300 at the A08 persistence-honesty lane (2026-08-26): the
    # +16 lines are the streamed-bundle settlement seam (WT1 A-IV item 18) --
    # the one publish hook that lands the settled capture-outcome attestation
    # in the streamed artifact right after settle_completed/settle_halted.
    # Reviewed raise with a stated reason; trace.py stays on the debloat list.
    # Raised 2300 -> 2350 at the A06 pre-rebase reconcile (2026-08-27): A08's
    # settlement seam (+16, above) and A06's save_grads bare-callable
    # strict-bool retention helper each independently stepped 2250 -> 2300 on
    # their own bases; combined they measure 2315, so the ledger re-steps to
    # the next 50-line step above the merged measurement. No new growth
    # licensed; debloat target unchanged.
    # Raised 2350 -> 2400 at the F20 T68c reconcile (2026-08-29): the lane
    # unions measure 2354 (each parent grew inside its own slack; the F20
    # parent already carried 2354 from the T67f union). Re-step to the next
    # 50-line step; no new growth licensed, debloat target unchanged.
    "torchlens/capture/trace.py": 2400,
    "torchlens/backends/torch/completeness_witness.py": 2100,
}


#: Maximum slack a ledger row may hold before it must be re-keyed down
#: (r7 R43-F2). Generous enough for one wave of reviewed defensive fixes,
#: small enough that a split's headroom cannot silently become regrowth
#: budget.
_MAX_LEDGER_SLACK = 100

#: Frozen ceilings for the tests/ frontier (r7 R43-F1 census, 2026-08-16,
#: next 50-line step above measurement). Same doctrine as the package
#: ledger: shrink freely, raise consciously with a reason, leave at <= 2000.
#: 2026-08-16 fixwave-6 integration settle: test_backward and
#: test_global_state_inventory re-stepped -- the census ran on the infra lane
#: while the capture lane's reviewed tripwire growth for those files merged
#: to main in parallel.
#: 2026-08-16 fixwave-7 settle: test_validation re-stepped 8500->8600 for
#: the wave's reviewed invariant tripwires; test_merged_engine instead
#: SPLIT (the r8 adversarial classes moved to
#: test_merged_engine_hardening.py) and stays under the unledgered cap.
#: 2026-08-17 FEATURE MEGASPRINT wave-0 settle debt record: the L3 telemetry
#: weak-launch-relation lifecycle row lands after census-settle growth;
#: test_global_state_inventory re-steps 2450->2500 pending the debloat pass.
#: 2026-08-19 post-tour sprint: test_global_state_inventory re-steps 2500->2509
#: for TWO lifecycle rows landed by two different lanes -- the belt sweep
#: pre-filter id set and the MCP bridge's bounded reload cache -- each with a
#: condensed rationale. Stepped EXACT (no headroom) on purpose.
#: SPLIT THIS FILE. It needed THREE conscious raises inside a SINGLE session,
#: which is the ratchet saying the file is the problem, not the rows. The natural
#: seam: the lifecycle-class frozensets are pure data and carry most of the line
#: count, while the census machinery and the assertions are what must be read
#: together. Every new global in the package lands here, so the growth is
#: structural and will recur until the data moves out.
#: 2026-08-26 megasprint fix cycle: SPLIT EXECUTED at exactly that seam, after
#: the _GOVERNED_LOAD_DEPTH row landed 4 lines over the exact-stepped ceiling.
#: The lifecycle-class frozensets, _WEAKLY_HELD, and the _LIFECYCLE_CLASSES
#: tuple moved to tests/_global_state_rows.py (pure data, ~690 lines); the
#: census machinery + assertions stay in test_global_state_inventory.py, which
#: drops to ~1835 lines -- under the unledgered cap, so its row is DELETED per
#: the two-way staleness rule. Future inventory rows land in the rows module,
#: which holds ~1300 lines of headroom before the cap.
#: 2026-08-26 A09 fix cycle (reconciled at the T12 rebase): the child-process
#: / fork guard test family ALSO split out of the inventory module, to
#: tests/test_global_state_child_process_guard.py -- both splits coexist, the
#: inventory module drops to ~1538 lines, and its row stays DELETED.
#: 2026-08-26 shim removal: test_validation 8600->8700, test_real_world_models
#: 4850->5150, test_toy_models 4050->4500 -- the flat->grouped codemod spells
#: every former one-line flat-kwarg trace() call as a wrapped grouped-options
#: call, so the growth is mechanical spelling verbosity, not new test content.
#: Next 50-line step above the post-codemod measure; the debloat pass owns
#: shrinking these back via helper extraction.
_TEST_FILE_CEILINGS: dict[str, int] = {
    "tests/test_validation.py": 8700,
    "tests/example_models.py": 5500,
    "tests/test_real_world_models.py": 5150,
    "tests/test_toy_models.py": 4500,
    "tests/test_auto_collapse_metrics.py": 3200,
    "tests/test_backward.py": 2550,
    "tests/validation_goldens/test_validation_exemption_hardening.py": 2400,
    "tests/test_conditional_branches.py": 2150,
    "tests/test_tlspec_runnable_r41_crossthread_witness.py": 2050,
}


def _is_generated_module(path: Path) -> bool:
    """Return whether a module self-declares as GENERATED (generator authority)."""

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return stripped.startswith(('"""', "'''", 'r"""')) and "GENERATED" in stripped
    return False


def _line_counts(root: Path = _PACKAGE_ROOT) -> dict[str, int]:
    """Return line counts for every non-generated module under ``root``."""

    counts: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        if _is_generated_module(path):
            continue
        relative = path.relative_to(_PROJECT_ROOT).as_posix()
        with path.open(encoding="utf-8") as handle:
            counts[relative] = sum(1 for _ in handle)
    return counts


def _ratchet_violations(counts: dict[str, int], ceilings: dict[str, int]) -> list[str]:
    """Return ratchet violations for a census (pure, red-capability-testable)."""

    violations = []
    for relative, lines in sorted(counts.items()):
        ceiling = ceilings.get(relative)
        if ceiling is None:
            if lines > _NEW_FILE_LINE_CAP:
                violations.append(
                    f"{relative}: {lines} lines exceeds the {_NEW_FILE_LINE_CAP}-line cap "
                    "for unledgered modules — split it; do not add a ledger row for new growth"
                )
        elif lines > ceiling:
            violations.append(
                f"{relative}: {lines} lines exceeds its frozen ceiling {ceiling} — split it "
                "(preferred) or consciously raise the ceiling with a stated reason"
            )
    return violations


def _stale_ledger_rows(counts: dict[str, int], ceilings: dict[str, int]) -> list[str]:
    """Return ledger rows that no longer describe a god file (two-way staleness)."""

    stale = []
    for relative, ceiling in sorted(ceilings.items()):
        lines = counts.get(relative)
        if lines is None:
            stale.append(f"{relative}: ledgered but no longer exists (ceiling {ceiling})")
        elif lines <= _NEW_FILE_LINE_CAP:
            stale.append(
                f"{relative}: {lines} lines is at/below the {_NEW_FILE_LINE_CAP} cap — "
                "delete its ledger row so the exemption cannot rot"
            )
    return stale


def _overslack_rows(
    counts: dict[str, int], ceilings: dict[str, int], max_slack: int = _MAX_LEDGER_SLACK
) -> list[str]:
    """Return ledger rows whose ceiling sits too far above the measurement.

    r7 R43-F2: a shrink-only ratchet converts every successful split into
    permanent regrowth budget unless the ceiling follows the file down.
    """

    overslack = []
    for relative, ceiling in sorted(ceilings.items()):
        lines = counts.get(relative)
        if lines is None or lines <= _NEW_FILE_LINE_CAP:
            continue  # the staleness check owns these
        if ceiling - lines > max_slack:
            step = lines + (50 - lines % 50) % 50 or lines
            overslack.append(
                f"{relative}: ceiling {ceiling} sits {ceiling - lines} lines above the "
                f"measured {lines} — re-key the row down to the next 50-line step ({step})"
            )
    return overslack


def test_no_package_module_exceeds_its_size_ceiling() -> None:
    """Every torchlens module respects the cap or its frozen ledger ceiling."""

    violations = _ratchet_violations(_line_counts(), _GOD_FILE_CEILINGS)
    assert not violations, (
        "file-size ratchet violations (R43 — god files may shrink, never grow "
        "unnoticed):\n  " + "\n  ".join(violations)
    )


def test_no_test_module_exceeds_its_size_ceiling() -> None:
    """Every tests/ module respects the cap or its frozen ledger ceiling."""

    violations = _ratchet_violations(_line_counts(_PROJECT_ROOT / "tests"), _TEST_FILE_CEILINGS)
    assert not violations, (
        "tests/ file-size ratchet violations (r7 R43-F1 — the tripwire corpus "
        "is not exempt from its own doctrine):\n  " + "\n  ".join(violations)
    )


def test_god_file_ledger_has_no_stale_rows() -> None:
    """A ledger row must leave when its file shrinks below the cap or vanishes."""

    stale = _stale_ledger_rows(_line_counts(), _GOD_FILE_CEILINGS)
    stale += _stale_ledger_rows(_line_counts(_PROJECT_ROOT / "tests"), _TEST_FILE_CEILINGS)
    assert not stale, "stale god-file ledger rows:\n  " + "\n  ".join(stale)


def test_ledger_rows_carry_no_excess_slack() -> None:
    """Ceilings follow files down: no row may hoard more than the slack budget."""

    overslack = _overslack_rows(_line_counts(), _GOD_FILE_CEILINGS)
    overslack += _overslack_rows(_line_counts(_PROJECT_ROOT / "tests"), _TEST_FILE_CEILINGS)
    assert not overslack, (
        "ledger rows holding excess regrowth budget (r7 R43-F2):\n  " + "\n  ".join(overslack)
    )


def test_file_size_ratchet_is_red_capable() -> None:
    """Planted censuses trip each violation class (red-capability self-test)."""

    ceilings = {"torchlens/ledgered.py": 2500}
    grown_ledgered = _ratchet_violations({"torchlens/ledgered.py": 2501}, ceilings)
    assert len(grown_ledgered) == 1 and "frozen ceiling" in grown_ledgered[0]
    new_god = _ratchet_violations({"torchlens/new.py": 2001}, ceilings={})
    assert len(new_god) == 1 and "unledgered" in new_god[0]
    assert _ratchet_violations({"torchlens/ok.py": 2000}, ceilings={}) == []
    stale = _stale_ledger_rows({"torchlens/ledgered.py": 1999}, ceilings)
    assert len(stale) == 1 and "delete its ledger row" in stale[0]
    missing = _stale_ledger_rows({}, ceilings)
    assert len(missing) == 1 and "no longer exists" in missing[0]
    overslack = _overslack_rows({"torchlens/ledgered.py": 2400}, {"torchlens/ledgered.py": 2500})
    assert overslack == []  # exactly at the slack budget
    overslack = _overslack_rows({"torchlens/ledgered.py": 2380}, {"torchlens/ledgered.py": 2500})
    assert len(overslack) == 1 and "re-key" in overslack[0] and "2400" in overslack[0]
    # A shrunken-below-cap file is the staleness check's finding, never slack.
    assert _overslack_rows({"torchlens/ledgered.py": 1500}, {"torchlens/ledgered.py": 2500}) == []
