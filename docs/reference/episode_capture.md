# Episode capture (`capture_kind=episode`)

An EPISODE is a multi-step generation run — a loop that calls one model N
times, feeding each step from the last — captured as ONE wrapped session
product. Declare it on the ordinary capture entry:

```python
import torch
import torchlens as tl
from torchlens.options import EpisodeSpec

class GreedyRunner(torch.nn.Module):
    def __init__(self, model, n_steps):
        super().__init__()
        self.model = model          # the stepped model
        self.n_steps = n_steps

    def forward(self, ids):
        tokens = []
        current = ids
        for _ in range(self.n_steps):
            next_token = self.model(current).argmax(-1, keepdim=True)
            tokens.append(next_token)
            current = torch.cat([current, next_token], dim=1)
        return torch.cat(tokens, dim=1)   # one integer tensor of emitted tokens

log = tl.trace(
    GreedyRunner(model, 20), prompt_ids,
    episode=EpisodeSpec(stepped_module=model, n_steps=20),
)
ledger_payload = log.annotations["episode"]     # header + per-step rows
```

Every spelling on this surface is DOCUMENTED-UNSTABLE (no deprecation shim
owed) pending the rolling naming session; the semantics below are the
ratified contract.

## Cost: a diagnostic-tier product, for tens of steps

The wrapped episode tier is the verification oracle / deep-dive product. Its
cost is SUPERLINEAR in step count — measured on gpt2-124M (CPU, 8-token
prompt): N=20 costs 79 s (103x native) with a 146 MB artifact and 1.9 GB peak
RSS; N=100 costs 657 s (323x native), 947 MB, 5.4 GB peak RSS. Do not plan
wrapped episode captures for hundreds of steps; the guarded-fast tier
(`trace.run(inputs=..., fast=True)`) is the engine for episode-scale
re-runs, and it must reproduce the wrapped tier's tokens bit-exactly (a
pinned cross-tier identity test guards this). The guarded-fast tier is NOT
reachable from the default capture above: `fast=True` collects functional
operations only when the capture requested them explicitly, so an episode
capture meant for fast re-runs must pass a functional save predicate --
`tl.trace(runner, ids, episode=..., save=tl.func("cat"))` (any `tl.func(...)`
selection covering the loop's functional ops) -- or the run refuses typed
(`run_capability_unavailable`, teaching the `save=` remedy). The default
capture re-runs through `trace.run(inputs=...)` (the full transactional
tier) instead. Either re-run is a FRESH execution: the product drops the
episode ledger to a travel note and clears the per-op `episode_step` stamps
(the travel policy below), and the product saves and loads as a plain
capture.

**The declared-step cost ceiling.** A declaration beyond
`EPISODE_DECLARED_STEP_CEILING` (100 steps, counting `n_steps` or the
`forced_tokens` length) refuses typed (`episode_step_ceiling_exceeded`) at
declaration time, before the forward runs.
`EpisodeSpec(acknowledge_step_cost=True)` is the explicit override: a
hundreds-of-steps wrapped capture is a decision, never an accident.

**The declaration's own cost, priced.** Declaring `episode=` changes NOTHING
about the executed model call or the recorded graph (the four-clause
invariant below), but it is not literally free: settlement derives the token
column from the already-computed root output, costing exactly one
`aten.select.int` plus one `aten.view.default` dispatcher read per declared
step — visible to any user-side dispatch-mode counter or profiler. On CUDA
the per-step `.tolist()` tail is one host sync per step (a named cluster
measurement, not a claim). Wall time is NOT an oracle for the declaration
(measured signs disagree across machines); the permanent regression is the
four-clause invariant: identical ordered recorded op labels, root output
bytes, post-run RNG state, and stepped-module call count between
episode-declared and undeclared runs of the same seeded fixture.

## The declaration

`EpisodeSpec(stepped_module=...)` declares the STEPPED MODULE — the
`nn.Module` whose successive top-level calls define step boundaries. It must
be a PROPER submodule of the traced episode root. On a BOUND-METHOD root
(F41, the ruled root contract: `tl.trace(model.generate, ids,
episode=EpisodeSpec(n_steps=N))` — the owner resolves via
`method.__self__` and registers as a submodule of the TL-authored wrapper
root) `stepped_module` DEFAULTS to the owner; on a module root it is
required, and `None` refuses typed (`episode_declaration_invalid`) — wrap
the loop in a module or trace the bound generation method directly. Step
tallying is FLAT: each
top-level call of the stepped module is one step, regardless of loop nesting
inside the root's `forward`. Call 1 is the prefill (ledger row 0,
`role="prefill"`); later calls are decode steps.

Episode grouping keys on successive top-level calls of the declared stepped
module, never on pass-interior graph equality — so a ragged interior (a
routed mixture-of-experts model whose expert runs at only some steps) cannot
change the step count — and pass indices on the trace's layers are per-site
OCCURRENCE counters (pass `k` of a layer is the k-th time THAT layer ran),
not episode steps: a layer absent from a step has no pass there, so pass `k`
need not lie at episode step `k`.

Declared episode-carried state beyond the built-in scope (token prefix, KV
cache, RNG streams) is preflighted at DECLARATION time, unconditionally: any
item without snapshot/restore support refuses typed
(`episode_state_unsnapshotable`) before the forward runs.

Refused combinations (typed, `episode_declaration_invalid`): chunked
forwards, `cache=True`, value-free save policies under an evidence-deriving
kind (`tokens`/`digest` evidence derives from the retained root output;
`step_output_kind="none"` declares a status-only ledger and ADMITS
value-free saves — the kind-conditional save rule), a structure-only marker
(structure-only episodes are out of scope per the ratified
marker-combination table), and non-torch backends.

## The declared evidence derivation (foldA D8)

Per-step evidence is DECLARATION-DRIVEN, never root-output-shape guessing:

- `step_output_kind="tokens"` (default) reads integer per-step token ids
  from the declared source; a float/complex source refuses with a teaching
  message naming the two declarations below.
- `step_output_kind="digest"` reads per-step CONTENT DIGESTS of the declared
  source (`sha256:`-prefixed, over dtype + shape + raw bytes of the per-step
  slice) — any dtype, so float-emitting stepped roots (hidden states,
  logits) are first-class.
- `step_output_kind="none"` derives no evidence at all: a status-only
  ledger, the honest shape for roots whose output has no per-step structure
  (a diffusion model returning one final image).

`step_output_from` declares WHICH root-output slot the evidence derives
from, as a dot-separated container path (dict keys and
namedtuple/`ModelOutput` field names by name, tuple positions by index —
`"sequences"`, `"0"`, `"0.logits"`). Undeclared, the root must return a
single output tensor; multi-output roots refuse with the available slots
listed. The resolved source disclosure is written to the ledger header
(`"output"` for the single-output default).

Per-step positions are TAIL-ALIGNED along the declared `step_axis` by
default: the LAST `n_steps` positions are the per-step emissions (one
appended position per step). A root returning what real `generate()` returns
— prompt+completion — keeps its prompt prefix out of the evidence column, an
emitted-only root is the equal-size special case of the same rule, and a
source carrying FEWER positions than steps ran refuses typed. No cache
arithmetic is involved anywhere: the KV-cache last-token feed and the
append-all feed derive identical evidence columns from identical episodes.
The positions actually read are DISCLOSED in the header's
`step_output_positions` (one step-axis index per row, bound by the capture
digest; `None` when no column derived), so any reader re-derives the column
from the retained root output without the live measurement; grammar-v2
artifacts written before the disclosure lack the slot and read as
tail-aligned.

**The tail-alignment license (token-shaped episodes).** Tail alignment
presumes ONE appended position per step. On a token-shaped episode
(`step_output_kind="tokens"`, or any kind whose step-0 entry was a
token-id-shaped integer tensor) the derivation is licensed only when the
source width is exactly `n_steps` (emitted-only root) or exactly the
MEASURED step-0 entry width plus `n_steps` (prompt+completion); any other
width -- multi-token decoding (speculative/Medusa/MTP), draft+verify loops, a
stepped module called more than once per iteration -- refuses typed
(`episode_declaration_invalid`) with the licensed widths named, instead of
silently misattributing tokens to steps and then grading FALSE exogenous
breaks against the misattributed column. The ONE further admission is the
declared-crossing chain arm (W051 FIX2): when the root returns the FULL fed
id sequence -- every step's MEASURED entry is a prefix of it, row for row --
each step's emission is the position right after its own entry, and
positions a tool call injected between steps are admitted exactly when the
receiving step is declared in `EpisodeSpec(crossings=(k, ...))`; the column
derives at the measured chain positions (disclosed in
`step_output_positions`) and the join into the crossing grades `declared`.
An UNDECLARED surplus stays refused, with the crossings remedy named:
measured alone, two extra positions in a step's entry are indistinguishable
from multi-token decoding by the stepped module, so the declaration -- never
a root-shape guess -- decides. A `digest` column over a float
carried state (fixed-point / diffusion loops, where no token width exists)
is a disclosed positional slice convention keyed by the header's
`step_output_from` / `step_axis` -- recomputable by any reader, consumed by
no join re-grade -- and keeps the `>= n_steps` admission.

The header's `capture_digest` binds the ledger to ITS product: minted at
settlement (`mint_capture_digest`, schema tag `episode_capture_digest_v2`)
as hex SHA-256 over canonical JSON of the product's recomputable identity
facts — ordered recorded op labels, stepped-module address and call count,
and the managed `entry_seed` — PLUS every persisted ledger fact (the header
declaration and disclosure fields, the `step_join` envelope, the
`intervention_digest`, every row's status, coord, `step_output`, and witness
slots; the random `episode_id` is excluded so identical re-runs mint
identical digests). A consumer recomputes it from the product and its
persisted ledger and compares: any post-mint rewrite of ledger content
reads unbound (`episode_coupling_unbound`). The digest is an INTEGRITY
binding, not a signature — a forger who re-mints over swapped content is
caught by the evidence RE-DERIVATION cross-check: attestation and load
re-derive the evidence column from the product's retained root output and
compare it to the rows (`CouplingAttestation.evidence_rederived` is `True`
when it matched, `None` when nothing could be compared — `kind="none"` or an
unretained payload — and a mismatch refuses/quarantines). The historical
v1 form hashed identity alone, so two equal-length prompts minted identical
digests and a rewritten ledger still attested bound; a persisted v1 digest
now reads as PRE-BINDING (`episode_coupling_unmintable`, re-capture to
bind), never as foreign. The attested-coupling work (F42) consumes the
binding; the travel policy below stays the negative guarantee it
complements.

## Attested coupling (`episode=` x `intervene=`)

The combination runs COUPLED (F42; the foldA D5 flip — the historical
pre-execution refusal retired when the verdict's evidence bar was met, and
every spelling here is DOCUMENTED-UNSTABLE pending the naming session):

- **Product binding.** The ledger's `capture_digest` binds it to the exact
  product (above); `trace.episode_coupling` recomputes and compares — the
  positive "this ledger belongs to this product" claim. A mismatch refuses
  typed (`episode_coupling_unbound`); a pre-binding artifact refuses
  `episode_coupling_unmintable`.
- **Step-qualified selectors.** `torchlens.intervention.at_step(*steps)`
  matches ops recorded inside the named 0-based episode steps — live (the
  armed join session's step position) and post hoc (the persisted
  `Op.episode_step` stamps, written at settlement on every episode
  product). It composes with any selector and with pass-qualified labels
  (the step-x-pass cross-product). Evaluating it where no steps exist
  refuses typed (`episode_step_selector_without_episode`,
  `episode_step_unstamped`, `episode_step_selector_invalid`).
- **Perturbed fidelity and fire counts.** A coupled capture arms a
  fire-attribution session: every live `FireRecord` is attributed to its
  step (or the outside-step bucket for root-loop fires before/between/after
  steps — disclosed in the digest, never guessed into a row). Settlement
  writes the reserved C07X slots: per-row `fire_count` (a measured `0` on
  zero-fire started rows — zero and multiple fires are first-class facts;
  `None` only on rows that never started), the header
  `intervention_digest` (hex SHA-256 over the ordered fire facts and the
  armed rule identity, schema `episode_intervention_digest_v1`;
  deterministic, so identical re-runs mint identical digests — compare
  digests, never output equality alone), and
  `fidelity_basis="perturbed"` whenever a fired edit replaced a value
  (outranks the `forced`/escalation bases; `token_feed` still discloses a
  forced feed separately).
- **Per-segment facts.** `trace.episode_coupling.segments` (also
  `torchlens.capture._episode_coupling.coupling_segments`) derives
  contiguous step runs that never span a measured break join
  (`exogenous`/`declared` grades): per-segment step ranges and fire-count
  totals, with unchecked joins and unmeasured envelopes disclosed. A claim
  never spans a break.
- **Replay derives a fresh ledger or refuses.** `run()` on a coupled
  product refuses typed on BOTH engines
  (`episode_coupled_replay_underivable`): no provider re-arms the
  capture-time intervention, so no engine can derive a fresh ledger;
  re-capturing with `episode=` x `intervene=` is the fresh-ledger arm. A
  `do()` edit on a product carrying episode evidence quarantines that
  evidence on the edited product
  (`episode_evidence_dropped_perturbed_replay` note grammar) — a perturbed
  replay is a different execution, and the ledger never rides it silently.

## Discoverability

`trace.capture_kind` reads `"episode"` on episode-declared products and
`"plain"` otherwise (including fresh re-execution products whose evidence
the travel policy dropped, below). `trace.episode` is the public step
access: the parsed ledger with `header`, ordered per-step `rows`,
`steps_completed`, and `truncated_at_step` — `None` for plain captures and
quarantined/dropped payloads. Both spellings are DOCUMENTED-UNSTABLE pending
the naming session. The declaration class is `tl.options.EpisodeSpec` (the
root `tl.EpisodeSpec` spelling awaits the surface-table amendment).

## Settlement refusals keep the product

A settlement refusal (`episode_ledger_incoherent`, the declaration-mismatch
derivation refusals) fires AFTER the capture ran and postprocessed, so the exception
carries the settled product on `exc.partial_log` — recover it with
`tl.partial.from_failed_capture(exc)` instead of paying the forward again.

## Evidence never rides a fresh execution (the travel policy)

The per-step ledger is evidence about ONE executed forward. A fresh
re-execution product — `trace.run(inputs=...)` on any provider, including
guarded-fast — never carries the original capture's episode ledger: the
annotations travel policy (`torchlens/capture/_annotations_travel.py`)
replaces it with an inert in-key note
(`episode_evidence_dropped_fresh_execution`) at the provider settlement
finalizer, and capture-time observer values (`logged_values`) are removed
the same way. A product reporting `path_faithfulness=VERIFIED` therefore
never presents another execution's step evidence. Plain `trace.fork()` is
not a fresh execution and carries the ledger until an engine re-executes.
Re-capture with `episode=` to derive step evidence for new inputs.

## Cross-step continuity: the measured `step_join` (F40c)

The header's `token_feed` reflects the user's DECLARATION of how step inputs
were fed. The step JOIN — that step k+1's input actually CONTINUED step k's
output — is MEASURED (F40c, the trace-verb verdict's mid-call oracle
break contract): live boundary hooks on the stepped module snapshot each
step's entry/exit under `pause_logging` (invisible to the recorded graph;
the four-clause declaration invariant holds), and settlement re-grades every
join exactly, persisting the `episode_step_join_v1` envelope in the header's
reserved `step_join` slot. The check grades the join; it never claims to
know whether the cause was a tool, a human, a file, a network, or a process.

Per-row grades (closed vocabulary; the join INTO step k, grades[0] is null):

- `continuous` — direct continuation: token entries align as a suffix of
  the prior entry plus the evidence-column emission (append, last-token
  KV-cache, and sliding-window feeds are all one rule), or the entry is
  byte-identical to the prior step's output (`digest_direct` basis).
- `forced` — a teacher-forced join (`EpisodeSpec(forced_tokens=...)`): the
  entry continues the prior entry plus the DECLARED forced token for that
  step (`forced_tokens` basis). Graded against the declaration, never the
  evidence column — a forced root may return the model's predictions, which
  are not what was fed. Not a break: the feed is declared and chain-shaped
  by construction; it is also never read as a free `continuous` join. An
  entry that does not continue with the declared token grades `exogenous`
  (the declaration does not match the executed feed).
- `transformed` — the entry derives from the prior step's output through
  ops observed inside the capture (the pass-qualified graph witness; the
  scheduler/diffusion shape).
- `declared` — the entry crosses a boundary declared in
  `EpisodeSpec(crossings=(k, ...))` (the tool-call shape): disclosed, never
  silently continuous — the run is chain-shaped by declaration.
- `exogenous` — a MEASURED undeclared break: content entered the loop from
  outside the capture (the envelope counts the exogenous positions on the
  token basis).
- `unchecked` — the join was not measurable (no tensor entry, snapshot
  failure, oversize entry, a declared `step_input_from` that names no tensor
  argument); the reason is recorded, never a guess.

**Which argument is the entry.** The join measures the stepped call's
CARRIED INPUT, chosen by a disclosed preference rule and persisted per step
in the envelope's `entry_basis` slot (`"positional[i]"` / `"kwarg[name]"`):
an explicit `step_input_from` declaration (positional index or keyword name)
is honored exactly; otherwise a tensor under a preferred keyword
(`input_ids`, `decoder_input_ids`, `inputs_embeds`, `sample`,
`hidden_states`, `x`) wins, then the first non-auxiliary tensor argument
(masks, `position_ids`, `cache_position`, `timestep`/`t`, `labels` are
auxiliary; a 0-d/one-element float tensor beside a wider tensor is read as a
timestep and passed over), then any tensor. The historical rule was "first
tensor argument" with no disclosure, so a kwargs-first masked LM graded its
attention MASK (false exogenous positions on a true continuation) and a
timestep-first denoiser graded its TIMESTEP. Envelopes written before this
slot existed load with the argument undisclosed (`entry_basis` absent).

Three arms (both FORK-4 arms are BUILT; the ruling selects the default's
shape):

- DEFAULT (`feed="open"`, `on_feed_break="disclose"`): the one Trace
  returns with the break marked (`break_step`, plus `live_break_step` — the
  live check detects an undeclared break at the NEXT step entry, the first
  observable point: one-step detection latency). Episode-dependent claims
  across a measured break refuse typed — step-series reads
  (`EpisodeLedger.step_output_series()`), whole-episode replay (`run()` /
  `run(fast=True)`), the blessing fold (`derive_episode_status`), and
  escalation (`episode_feed_break_exogenous` /
  `episode_join_declared_crossing`). Ops, values, graph, audit, and
  per-segment reads are unaffected; the root call may truthfully be
  COMPLETE — the closed-episode claim is what breaks.
- FORK-4 arm (b) (`on_feed_break="refuse"`): the settlement write raises
  `episode_feed_break_exogenous` typed, carrying the settled product (with
  its measured envelope) on `exc.partial_log` — recoverable partial
  evidence, nothing discarded.
- STRICT (`feed="closed"`): the feed is declared closed (direct
  continuation). An UNDECLARED crossing halts capture at the next step
  entry (`episode_feed_closed_violation`; rows 0..k-1 complete, row k
  interrupted, later rows absent, all on `exc.partial_log`) — or settles
  typed over the completed capture in the residual case where the break is
  only measurable against the settlement evidence column (a single-position
  injection is live-indistinguishable from an emission). A DECLARED
  crossing under strict mode stops BEFORE entering the crossing step
  (`episode_declared_crossing_stop`).

The claim stays PER-ARTIFACT: a ledger without the envelope (a
pre-measurement artifact, or a failed live measurement) reads
`step_join=unmeasured` on every surface, its series claims refuse
`episode_join_unmeasured`, and `tl.report.explain` prints the disclosure
line — an unmeasured join can never read as a measured one, whatever the
build measures. Loads validate the envelope FAIL-CLOSED against the rows
(schema, closed grades, prefill null, break-step coherence); any violation
quarantines `episode_ledger_incoherent`.

## The ledger

The per-step status ledger lands ON the product at
`trace.annotations["episode"]` after settlement, in the GRAMMAR v2 shape
(the C07X coordinated tlspec-v9 amendment; `episode_ledger_version=2`,
family-local — future grammar changes bump the family version, never the
tlspec version): a header (`episode_id`, stepped-module address, the
managed-RNG `entry_seed`, token feed, provenance tier, escalation
disclosures, the declared `step_output_kind` — `tokens` (default) /
`digest` / `none` — with the `step_output_from` source disclosure, the
declared `step_axis`, and the minted `capture_digest` binding (all written
by the declared derivation above), plus the measured `step_join` envelope
(F40c, the section above) and the `intervention_digest` coupling slot
written on intervened captures (F42, the attested-coupling section
above)) and one row per step
(`episode_step`, `role`, `status`, coordinates, the generic `step_output`
under the declared kind, the carried-state witness slots
`entry_state_digest` / `exit_state_digest`, and the `fire_count` coupling
slot — measured on coupled captures, `None` on uncoupled ones). Row status
is a closed vocabulary:

- `complete` — step k's forward returned (row-scoped truth, never a claim
  about the product; product truth is the settled `CaptureOutcome`).
- `interrupted` — the settled outcome's frontier lies INSIDE step k (a halt
  or failure mid-step; the frontier disclosure rides the row).
- `absent` — the step never started.

Rows obey the MONOTONE PREFIX LAW (complete prefix, at most one interrupted
row, absent tail) and are write-once at settlement. The ledger is a
DISCLOSURE, never a settlement authority: the episode product settles through
the one existing `CaptureOutcome` authority with its vocabulary unchanged,
and every capability gate (N1–N5) applies verbatim. Truncation stays a
run/report term: an episode halted at a step boundary settles HALTED with a
clean prefix and the truncation disclosed on the ledger
(`steps_completed`, `truncated_at_step`) — nothing rounds up.

THE `cache_len` FIELD IS DELETED (grammar v2). It was a derived arithmetic
guess (prompt length + step) that is wrong on real KV-cache feeds and never
a measured cache fact. Its replacement is the MEASURED GENERIC carried-state
witness (the SV-6 semantic ruling): `entry_state_digest` /
`exit_state_digest` are channel-keyed digests of the carried state at the
step's boundaries, `None` means NOT MEASURED — the only value any shipped
writer emits — and no field may imply an unmeasured cache fact. The
F-WITNESS work designs the full witness schema after the F20 retention seam
and writes real digests; until then absence is the honest disclosure.

Loads validate fail-closed: an episode ledger on a capture without the
declaration refuses typed (`episode_ledger_without_declaration`); a
structure-only ledger carrying step output refuses typed
(`episode_ledger_payload_in_structure_only`); any other geometry violation
QUARANTINES the payload with one warning (`episode_ledger_incoherent`) — the
rows stop being claims and the outcome derivation treats the ledger
fail-closed. The family version field is CONSUMED at load: any
`episode_ledger_version` other than 2 — including the version-less
pre-amendment grammar v1 payload (`cache_len`/`tokens` rows) — quarantines
with the grammar named, and never normalizes (foldA D9: the old-payload row
is a quarantine record, never a normalization ladder). Lane F40b, authorized
by the C07 owner, activated the generalized `step_output_from`/
`step_output_kind` derivation on this grammar and self-authored its
MIGRATIONS row (the D3 authorization line lives in `MIGRATIONS.md`).

## Persistence

`annotations["episode"]` persists plainly as of the tlspec v8 coordinated
bump, as does the Bundle `member_relations` key (below). Loads validate
fail-closed: an undeclared ledger refuses `episode_ledger_without_declaration`
and geometry violations quarantine `episode_ledger_incoherent`. Pre-v8
artifacts never carry the key (the v7 scrub dropped it; the live trace kept
its session-time ledger).

**Loads anchor the ledger to the product.** A grammatically valid ledger is
not enough: the presence of the key was once the ONLY anchor, so a ledger
copied from a real episode artifact into any artifact's
`annotations["episode"]` loaded with zero warnings and made the product an
"episode". Every load now checks, in order, that the header's
`stepped_module` recorded at least one call on this product (from the
restored op records' module entries), that the started-row count equals
that call count and every row's `member_call_index` names a recorded call,
that the persisted `capture_digest` equals the digest recomputed from the
product and the ledger's own content (a legacy v1 identity digest is
admitted at load and reads pre-binding at attestation), and that the
evidence column re-derives from the retained root output (skipped, never
failed, when the payload is not retained). Inside a `.tlspec` load the
payload blobs attach only after the metadata restores, so the bundle loader
runs that fourth anchor a second time once the payloads are attached: a
ledger re-minted over another execution of the same program is refuted by
the LOAD itself, not on the first `trace.episode_coupling` read (which
still re-derives against the attached payload). Any failure quarantines
`episode_ledger_incoherent` with the failed anchor named in the note's
`detail`; the product still loads, its rows stop being claims, and
`Trace.capture_kind` keeps reading `episode` (a quarantined ledger is a
declaration in doubt, not an absent one — the per-op `episode_step` stamps
of a corrupted genuine artifact must stay loadable).

## Teacher forcing (disclosed, non-verifying)

`EpisodeSpec(forced_tokens=...)` declares a teacher-forced feed: the driver
feeds the declared tokens instead of the model's emissions. The ledger header
records `token_feed="forced"` and `fidelity_basis="forced"` — an explicitly
NON-VERIFYING disclosed mode; token-fidelity obligations never verify a
forced episode. Recompute-and-compare remains the verifying default
everywhere else. The measured `step_join` of a forced episode grades each
join against the DECLARED token (`forced` grade, `forced_tokens` basis), so
a root that returns the model's predictions instead of the fed tokens is not
misread as an exogenous break; step-series reads, replay, and the fold stay
open across forced joins.

## The managed RNG recipe

Episode captures use the managed seeding discipline: the capture's effective
`random_seed` (drawn or passed, exactly as for any capture) is recorded as
the ledger `entry_seed`. Re-running the same declaration with the same seed
reproduces a sampled episode bit-identically; pass
`capture=CaptureOptions(random_seed=...)` explicitly for cross-run
reproduction.

## Escalation (session semantics)

A failed or suspect cheap-tier episode escalates by re-running the WHOLE
episode wrapped, disclosed — never a partial product. Build the declaration
with `torchlens.capture._episode_ledger.escalation_spec(producer,
stepped_module=..., reason=...)`: it derives `escalated_from` (a digest over
the producer's persisted outcome payload + ledger rows), the closed-vocabulary
`reason` (`step_failed` / `divergence` / `requested`), and the producer's
token column, so the escalated capture discharges the fidelity obligation at
write time: prefix-equal token columns record `fidelity_basis="tokens"`; a
mismatch records `"diverged"` — the escalated product is still a valid
capture of what it ran, it just is not an escalation of the original episode,
and says so. Fidelity is a disclosure, never a settlement input.

## The floor: per-step Bundles + the derived fold

The declared minimum episode product (invocable by the sprint orchestrator on
size evidence only) is N per-step captures composed in a `Bundle` with S6
`episode_member` relation rows, under OBSERVER DISCIPLINE: the driver's
native forward owns the live carried state; each capture observes an isolated
copy; a capture product is never the source of carried episode state.
`Bundle.derive_episode_status(episode_id, ledger=...)` folds member outcomes
into a DERIVED episode status (`episode_complete` /
`episode_halted_at_step` / `episode_aborted_at_step` /
`episode_failed_at_step` / `episode_unknown`) — a total, fail-closed
derivation, never a Bundle-level settlement. Fold arms that consume
ledger-only declaration facts (the declared step count; a driver-declared
boundary halt) degrade to `episode_unknown` without a re-supplied ledger —
disclosed truth loss until the schema bump persists the ledger. The episode's
provenance tier is the MINIMUM of its members' tiers, never the maximum.

Bundle relation rows never require alignment, never assert structural
comparability, and never dangle: mutators either cascade explicitly
(`cascade_relations=True`) or refuse typed. See
`docs/reference/error_refusal_contract.md` for the full episode/bundle
refusal code family.

## Relation grammar v2 (the C07X amendment)

As of the C07X coordinated tlspec-v9 amendment, relation-kind param keys
split REQUIRED/OPTIONAL per kind (both sets closed; an undeclared key still
refuses). `successor_of` — whose direction is PINNED: `from` is always the
LATER member (the successor), `to` the earlier member it succeeds — admits
three optional params: the `evidence` envelope (`{schema, items[],
facts_digest}`, items graded over the closed
`verified`/`consistent`/`disclosed`/`unchecked`/`divergent` vocabulary with
a contracted unchecked-reason menu and a mandatory nullable `basis`; one
1 MiB canonical-JSON per-row budget), `carry_mode`, and `state_source`.
Loading is preserve-and-disclose: rows of unknown NAMESPACED kinds load
opaque (`OpaqueRelationRow` — preserved verbatim on re-save, never
executed; `tl.load(unknown_relations="refuse")` restores strictness; bare
unknown kinds always refuse), unknown evidence schema ids load opaque
inside a valid envelope, and unknown namespaced top-level `bundle.json`
sections preserve through `Bundle.preserved_sections` while bare-unknown
sections refuse typed. Version-axis rows still ORDER members only — no
relation row ever licenses a cross-member parameter value read.
