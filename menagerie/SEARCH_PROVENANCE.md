# Menagerie Model-Discovery Provenance & Update Protocol

This file is the **durable, human-readable record of what the menagerie's model-discovery has covered,
when, and to what cutoff** -- plus the **protocol for running and DOCUMENTING future update sweeps** so the
registry can be extended smoothly over time without re-treading covered ground or losing provenance.

Companion docs: `DISCOVER_MODELS.md` (the reusable adversarial *hunt* prompt -- HOW to find missed families),
`METHODOLOGY.md` + `UPDATE_RECIPE.md` + `HARVEST_SOURCES.md` (HOW to build/add a found model). This file is
the *WHAT-has-been-searched and WHEN* ledger those two flank.

---

## PART 1 -- DISCOVERY SWEEP RECORD

### Sweep 001 -- "Definitive Sweep" (2026-06 -> 2026-07-04)

- **Search window:** multi-day, ~2026-06 through **2026-07-04** (completion date).
- **Frontier cutoff:** **2026-07-04** (arXiv). Anything published after this date is, by definition, NOT covered
  and is the primary target of the next sweep.
- **Goal:** enumerate EVERY neural-network architecture *family* ever described (the finite historical space),
  to the standard of certainty -- not "enough," but "cannot name one more."
- **Method (summary):** multi-axis adversarial discovery -- a coordinator fanning out cross-lab sub-hunters
  (Anthropic + OpenAI) across independent axes; native-language + non-English queries; family-not-variant
  discipline (a known arch applied to a new domain/backbone/loss = a VARIANT, rejected); systematic
  venue x year enumeration for deep tails; 5 full cross-lab verification gauntlets alternating with
  harvest/drain waves; then a 10-round cross-lab **adversarial finale** run to **2 consecutive independent
  EXHAUSTED verdicts**.
- **Dedup baseline:** every candidate deduped 3 ways -- against (i) the implemented catalog + classics,
  (ii) the queued-but-unbuilt registry, (iii) the sweep's own finds. Grep-verified aliases (short names /
  acronyms) + zoo-prefix aware.

#### Coverage / exhaustion status (as of the 2026-07-04 cutoff)

| Axis | Status | Notes |
|---|---|---|
| modality | LITERAL ZERO | incl. exotic sub-modalities (Compton/fUS/photoacoustic/terahertz/event/tactile...) |
| region | LITERAL ZERO | native-script queries executed: Chinese/Russian/Korean/Japanese/Iranian/Indian/LatAm/African |
| discipline | LITERAL ZERO | physics/chem/bio/neuro/med/geo/materials/astro/finance NN families swept |
| lineage | LITERAL ZERO | per-lab/researcher/company families (converged 14->16->7->1->0) |
| era | LITERAL ZERO | deep-history founders (MENACE/Rochester/Pandemonium/Adaline...) + old-framework tails |
| venue | LITERAL ZERO | conferences/workshops/challenges/patents/theses/textbooks; defensive-pubs = NOT-a-source |
| framework | LITERAL ZERO | Caffe/Theano/Chainer/MXNet/CNTK/Paddle/Jittor/MindSpore + speech toolkits (ICASSP/Interspeech) |
| **mechanism / recent-arXiv** | **IRREDUCIBLE -- FROZEN at 2026-07-04** | live-growing space; provably cannot reach literal zero (see below) |

**The mechanism frontier is FROZEN, not exhausted.** An 8-slice partitioned exhaustive enumeration of the
2024-2026 primitive frontier showed yield persists at the *leading edge* (co-temporal 2026-06/07 papers) --
i.e. the live research frontier invents new primitive TYPES faster than exhaustive enumeration saturates
existing ones. It is captured to the leading edge as of the cutoff and re-swept on cadence (below), NOT gated
to zero.

**Blind-spot CLASSES the finale caught (now covered -- do not re-treat as gaps):** codec-INTERNAL neural
coding tools; point-cloud/mesh/VCM/light-field media-coding; real-time render-loop graphics internals;
computational-litho OPC/ILT; game-theoretic imperfect-info agents; matrix-manifold / Riemannian DL;
non-additive-measure / belief-function (Choquet/evidential) nets; differential-operator + integral-equation
nets; fuzzy AND/OR / flip-flop / possibilistic. These were genuinely-missed and are now folded.

#### Results
- **12,062** net-new distinct candidate families discovered (clean, deduped, adversarially audited).
- Corpus after reconciliation: **~28,454 unique named architecture families** identified total
  (~12,388 already built into the catalog/classics + ~16,066 net-new queued to build).
- **Verification:** 2 consecutive independent cross-lab adversarial EXHAUSTED verdicts on the finite space.
- **Integrity note:** a dedup zoo-prefix bug was found+fixed mid-sweep (18 already-catalogued models had
  leaked in as "new"; 7 genuine acronym-collisions kept).

#### Internal detail (private, gitignored -- NOT in this public tree)
Full blow-by-blow lives under `.research/menagerie-redesign/fable-discovery/`:
`FABLE_FINAL_REPORT.md` (the sweep report), `METHODS_LOG.md` (every method + yield), `SWEEP_STATE.md`
(round log). Belt-and-suspenders backup: `~/menagerie-fable-backup-DISCOVERY-COMPLETE.tgz`. These are the
authoritative "what queries were run" record for anyone wanting to squeeze the last drops from OLD territory.

---

## PART 2 -- UPDATE-SWEEP PROTOCOL (how to run + DOCUMENT future sweeps)

The registry is a **dated snapshot**. Keeping it current is a periodic, well-bounded operation -- most of
the finite historical space is already covered, so future sweeps are mostly about the moving frontier plus
occasional deeper drains.

### When to run an update sweep
1. **The moving frontier (primary):** after each major conference cycle (NeurIPS/ICML/ICLR/CVPR/ACL/...),
   or on a fixed cadence (recommend quarterly). This targets architectures published *after the last
   cutoff*.
2. **A more capable model is available:** a smarter auditor finds more -- re-running even "exhausted" axes
   at higher capability is worthwhile (this is why `DISCOVER_MODELS.md` is written to be re-dispatched).
3. **Squeezing OLD territory:** the finale named exotic veins that are bounded but re-drainable at higher
   effort (possibilistic/credal/interval aggregation, integral-equation nets deeper, p-adic/quiver exotic
   singletons, etc.) -- see the internal `METHODS_LOG.md` "remaining nameable untried" notes.

### What to target
- **New-since-cutoff:** advance the frontier from the *previous* sweep's cutoff to today. The mechanism/
  recent-arXiv axis is the one that keeps yielding -- walk it by month x sub-mechanism (attention/SSM/MoE/
  positional/memory/token-mixing/generative/exotic-substrate).
- **The finite axes** (modality/region/discipline/lineage/era/venue/framework) are at literal zero as of the
  last cutoff -- only re-sweep them (a) at higher model capability, or (b) for genuinely new venues/regions.
- Honor **family-not-variant** + **use-real-source** (never approximate) throughout.

### How to run
1. Dispatch `DISCOVER_MODELS.md` (cross-lab adversarial sub-hunters).
2. **Rebuild the dedup baseline FIRST** (critical): regenerate the known-name set from the CURRENT
   `master_catalog.jsonl` name+variant + the live classics registry keys + the queued build files, lowercased-
   unique. New finds dedup against THIS (3 stages: built / queued / this-sweep). Grep-verify short-name/acronym
   aliases; the deduper is zoo-prefix aware -- keep it so.
3. Run to genuine dryness (finite axes) + capture-to-cutoff (mechanism frontier). For a "definitive" refresh,
   end with the cross-lab adversarial finale (2 consecutive EXHAUSTED).
4. Fold genuine net-new families into the registry via `METHODOLOGY.md` + `UPDATE_RECIPE.md` (source ladder:
   real library class -> vendor real repo -> faithful port -> faithful reimpl-from-detailed-description ->
   skip only if NO code AND no description). Preserve every alias for findability (build once per
   `graph_shape_hash`; expose all names).
5. Record machine-readable provenance: `menagerie.provenance.record_sweep(...)` + set `added_wave` on new
   rows so `python -m menagerie.status --provenance` joins models back to the sweep.

### How to DOCUMENT the sweep (append a record here -- THIS is the durable log)
After each update sweep, **append a new "Sweep NNN" block to PART 1 above**, using this template. This is what
lets the next maintainer see what's covered and pick up cleanly.

```
### Sweep NNN -- "<name>" (<start> -> <completion date>)
- Search window / Frontier cutoff (advance it): <prev-cutoff> -> <new-cutoff>
- Goal / scope: <what this sweep targeted -- frontier-only? full refresh? a named residual vein?>
- Method: <coordinator + labs + axes; finale y/n>
- Dedup baseline rebuilt from: master_catalog (<N> rows) + classics (<N>) + queue (<N>)
- Coverage/exhaustion delta: <which axes re-confirmed zero; mechanism cutoff bumped to X; any residual left>
- Results: <raw / deduped / net-new>; verification (<EXHAUSTED verdicts / dry passes>)
- Provenance: record_sweep id=<...>, added_wave=<...>
- Internal detail: <path to that sweep's METHODS_LOG / report>
```

### Invariants (do not drift)
- **Cutoff discipline:** every sweep advances a *dated* frontier cutoff; the mechanism axis is never claimed
  "exhausted," only "captured through <date>."
- **Dedup-baseline-first:** always rebuild the 3-stage known-name baseline before a sweep, or you re-find the
  registry.
- **Family-not-variant + real-source:** the two rules that keep the corpus honest and non-slop.
- **All names findable:** aliases are first-class; true distinctness is `graph_shape_hash`, reported alongside
  the named-entry count.
