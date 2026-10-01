# Capture Outcomes — the settled terminal truth of every capture

Every TorchLens capture product carries ONE typed, settled answer to "how did
this capture end": the `CaptureOutcome` record, written by the single
settlement authority in `torchlens/capture/outcome.py`. Consumers branch on
the record or on the stable refusal codes below — never on exception text.

## The status vocabulary

| status | meaning | `partial` |
|---|---|---|
| `COMPLETE` | settled complete; attested by the settlement stamp | `False` |
| `HALTED` | stopped by `halt=` or imperative `tl.fastlog.halt()`; an honest prefix | `True` |
| `ABORTED_NONFINITE` | stopped by the `raise_on_nan` policy | `True` |
| `FAILED` | the capture raised; phase + diagnostic origin attribution | `True` |
| `UNATTESTED` | structurally finished, but no settlement evidence (legacy artifacts only) | `None` |
| `UNKNOWN` | unprovable terminal state; the most restrictive row | `None` |

`phase` is FAILED-only, not FAILED-required: a `FAILED` outcome normally
carries a `phase`, but two writers legitimately emit `FAILED` with
`phase=None` — the fastlog recorder fallback stamp (when the runtime trace's
settled record is unavailable or non-failed) and the legacy `partial_error`
derivation. When present, `phase` is one of (`FORWARD` — the model forward raised;
`FINALIZE` — the forward completed but finalize/output-extraction/cleanup
raised; `POSTPROCESS` — the postprocess pipeline or the core freeze raised;
`TEARDOWN` — teardown failed AFTER settlement, demoting the outcome) and a
diagnostic-only `origin` (`user_op` / `torchlens` / `interrupt` / `unknown`)
that never steers a capability gate. `phase` is FAILED-*only*, not
FAILED-*required*: two writers legitimately emit `FAILED` with
`phase=None` — the recorder fallback stamp (when the scratch trace's settled
record is unavailable or non-failed) and the legacy `partial_error`
derivation on deserialized recordings. The coherence matrix permits the
absent phase; no gate consults it.

## Where to read it

- `Trace.outcome` — the sidecar (settle stamp on live products; adopted or
  lattice-derived record on loaded artifacts; `None` only on a live trace
  that has not settled yet).
- `Recording.outcome` — stamped by the recorder settlement adapter; derives
  conservatively from the construction status for legacy/recovered
  recordings.
- `PartialTrace.outcome` — forwards the inner trace's sidecar.
- `Bundle.outcome` — the DERIVED worst-of-members fold (declared container
  authority, foldB D9: instance three of the two-answers disease after
  PartialTrace and Recording). The status is the most severe member status
  under the fold's total severity order (`complete < unattested < halted <
  aborted_nonfinite < failed < unknown` — capability-table restrictiveness,
  ties broken by epistemic weakness), so COMPLETE is never blessed above the
  weakest member; a member with no settled outcome folds as UNKNOWN
  fail-closed, `derived=True` always, and the settlement note names the
  driving member. The fold is a DISCLOSURE reported beside results, never a
  settlement and never a per-member gate: capability gating applies at the
  member whose facts a read cites (foldB D8). Both public save doors agree
  by construction — `tl.save(bundle, ...)` delegates to the container door
  (`Bundle.save`), whose member writes run the N1 gate per member; per-trace
  payload options refuse typed (`bundle_save_option_unsupported`).
- `Trace.fork().outcome` — a DERIVED record settled by `stamp_forked`, never
  the parent's attestation by identity: a fork is the sanctioned mutation
  surface, so it settles through the structural lattice (UNATTESTED for a
  complete parent, HALTED-derived for a halted one), keeps the parent's
  committed-op count, and records `forked_from=<parent status>` in the
  settlement note. Capability parity holds (every UNATTESTED cell is
  allow/allow-scoped), so nothing a fork could do before is newly refused.
- `RunOutcome.capture_outcome` — the session-side copy; `RunOutcome.state`
  remains the first-transition log and is never revised (demotion replaces
  only the record).

## Persistence and the load derivation

`_capture_outcome` is a portable schema field (tlspec v7), persisted as a
string-only payload and parsed at load against the closed vocabularies plus a
status-specific coherence matrix (an attestation contradicting the artifact's
structural `halted` / `_tracing_finished` evidence degrades to `UNKNOWN` with
one warning — fail-closed, never a load crash). Attestation is self-declared
by the artifact: a coherent reauthoring is out of scope, the same documented
threat boundary as the runnable contract's section 11.

Artifacts WITHOUT an attestation derive from structural evidence alone, by
truthiness (`False` / `None` / absent are one class):

```
halted and finished      -> HALTED      (derived)
halted and not finished  -> UNKNOWN     (the halted postprocess never completed)
finished and not halted  -> UNATTESTED  (genuine-complete and tail-failure are indistinguishable)
otherwise                -> UNKNOWN
```

No legacy Trace without attestation ever derives `COMPLETE`. Finalized
`Recording` bundles keep their construction-status derivation
(`complete`/`halted` at finalize time is a construction proof); `recover()`
products are `UNKNOWN` (or reconstructed `HALTED` where the halt markers
survived) with `recovered=True`. Derived outcomes never upgrade across
re-saves: `UNATTESTED` re-saves stay `UNATTESTED`.

## The capability table (gates N1–N5)

Enforced through the ONE chokepoint
(`require_capture_capability(trace, capability)`); refusals raise
`tl.errors.CaptureOutcomeError` with a stable `fields["code"]`:

| capability | COMPLETE | HALTED | ABORTED / FAILED | UNATTESTED | UNKNOWN |
|---|---|---|---|---|---|
| validation entry (N2) | allow | allow (halted output block skipped, as always) | refuse N2 | **allow** — the tripwire stays armed | refuse N2 |
| `tl.save` analysis (N1) | allow | allow | refuse N1 | allow (no upgrade) | refuse N1 |
| `tl.save(level="runnable")` (N4) | allow | refuse N4 (runnable vocabulary: `halted_capture_not_runnable`) | refuse N1 | allow-scoped (existing preflights gate) | refuse N1 |
| live replay: refresh / `save_new_outs` / push / replay / rerun / live `run()` / `fast=True` (N5/N3) | allow | refuse N5 | refuse N3 | allow | refuse N3 |
| loaded-sparse `run()` | allow | **allow** — executes exactly the recorded prefix DAG | refuse N3 | allow | refuse N3 |
| `log_backward` / `backward` / `recording_backward` (N3) | allow | allow — the autograd graph IS the captured prefix | refuse N3 | allow | refuse N3 |
| draw / `report.explain` / inspection / fork / load | allow | allow | via `PartialTrace` / pickle inspection, as today | allow | allow (analysis-only, coreless staging) |

Row rationales, stated once:

- **UNATTESTED** permissions equal the historical de-facto behavior for
  loaded finished traces — nothing that worked yesterday is newly refused —
  while the CLAIMS column is empty: no settled-complete assertion, no
  no-swallow guarantee, `partial=None`. Validation entry deliberately stays
  OPEN because validation is the tripwire that catches a freeze-corrupted
  legacy representation.
- **UNKNOWN** refusals replace what was previously an ungated pass-through of
  provably-unfinished artifacts (`tl.save` of failed partials) or an
  undefined crash on unbuilt structures. The nearest legitimate workflows
  (partial inspection, `draw`, `tl.partial.from_failed_capture`) all remain
  open.
- **N5** converted two broken behaviors into typed refusals: halted refresh
  crashed with a misleading "computational graph changed" error, and halted
  `run(inputs=..., fast=True)` SILENTLY SUCCEEDED against the full native
  forward (the false-blessing hole). The refusal names the re-arm follow-on:
  re-capture without `halt=`, or use loaded-sparse `run()`.
- **N4** is a crash→typed conversion: no legacy workflow successfully saved a
  halted runnable (the producer preflight already failed).
- **`Recording.log_backward`** refuses a failed partial with the table's N3
  code (`RecorderStateError`, `fields["code"] == "N3"`) and additionally
  refuses HALTED with `fields["code"] == "recording_backward_halted"` — a
  deliberate Recording-scoped strictness beyond the table's HALTED-allow
  cell, which describes Trace-side backward (a halted Trace holds the prefix
  autograd graph; a halted Recording is a sparse event product whose frontier
  pass retained no complete output to root the backward walk, the same
  boundary that refuses halted-no-payload `to_trace()`).

## `raise_on_nan`'s honest scope

The tripwire checks RAW op-boundary tensors only: a NaN introduced by
`activation_transform` never aborts (COMPLETE), a raw NaN aborts even when
the transform would have sanitized it (ABORTED_NONFINITE), empty tensors are
skipped, and uncheckable dtypes warn once per capture.

## The queryable nonfinite record

*(Every spelling here is DOCUMENTED-UNSTABLE pending naming-session
ratification; no deprecation shim owed on rename.)*

`raise_on_nan` is a stop-and-throw; the queryable form is separate and never
changes control flow. `Trace.nonfinite_ops` returns the pass-qualified labels
of ops whose output held at least one NaN or Inf (each a valid `trace[label]`
key), and `Trace.nonfinite_coverage` returns the frozen evidence disclosure:
`basis`, `checked`, `nonfinite`, `unchecked` (dtypes with no runnable
finiteness kernel), `unexamined`, and `unmapped`. An empty answer is only as
strong as its coverage — the programmatic twin of the prose coverage-gap note
in `first_nonfinite()`.

Two evidence bases:

- **`"saved_payloads"`** (the default): the record derives from the memoized
  saved-payload scan already backing `print(trace)`. Zero capture-time cost;
  the one `isfinite` pass over retained payloads is paid at first query and
  memoized (measured 0.02–0.27 s on resnet18/transformer/gpt2-class models,
  then ~2 ms memoized). On a selective-save capture this basis is honestly
  blind to unsaved ops (`unexamined` says how many).
- **`"capture"`**: `CaptureOptions(track_nonfinite=True)` opts into a per-op
  `torch.isfinite(out).all()` at op commit, covering every committed op —
  saved or not. fp8 widens exactly first; empty tensors are exactly finite;
  uncheckable dtypes land in `unchecked`. CPU flags are read immediately;
  CUDA/MPS flags are deferred as 0-dim bool tensors and drained in ONE batch
  at the capture finalize seam, so the forward is never synchronized per op.

Measured cost of `track_nonfinite=True` relative to a default capture
(min-of-reps, 2026-08-19): CPU (devbox) resnet18-b8 +5.8%, transformer-enc
+14.5%, gpt2 +3.4%; H200 (cu128) resnet18-b8 +4.9%, transformer-enc +6.7%,
gpt2 +4.3%, with an eager per-op read variant costing +6.9%/+9.3%/+7.3% —
the deferral is why a device capture never pays a per-op synchronization.
These numbers are why the knob defaults to OFF everywhere: on default
exhaustive-save captures the free `"saved_payloads"` basis already covers
every op, so the per-op check would buy nothing; on selective-save captures
it is the only way to get verdicts for unsaved ops and is priced above.

The knob is session-time (`FieldPolicy.DROP`, like
`measure_python_peak_memory`): load restores the default, and a loaded trace
serves the `"saved_payloads"` basis over its archived payloads.
`structure_only=True` refuses the combination
(`structure_only_option_conflict`) — there are no values to check. The
capture-basis record covers committed ops only: synthetic input/output mirror
nodes count as `unexamined`, and events whose op was removed by
postprocessing (orphans) count as `unmapped`. Verdicts reflect the tensors
ops actually produced — an intervened op's verdict is of the replaced value
downstream consumers saw.

## The stop-request latch (F6)

`evaluate_halt`, `raise_nonfinite`, and imperative `tl.fastlog.halt()` all
latch a stop request on the active trace before raising. If the forward then
returns "normally" — user code swallowed the control signal in a broad
`except:` — the boundary checkpoint raises
`tl.errors.StopSignalSwallowedError` and the capture settles FAILED, never
COMPLETE. The halt-capable preview backends (paddle's shipped `halt=`, MLX's
halt selector) hold the same contract: their `HaltSignal` raise sites latch
the request on the trace, and `stamp_backend_finalized` — the one preview
settlement stamp — consumes the latch, refusing a latched-but-unhalted
capture with the same typed error and a FAILED settlement instead of the
COMPLETE arm. A swallowed nonfinite abort is FAILED, never a clean
ABORTED_NONFINITE (the abort did not actually stop the forward). Imperative
`halt()` outside any capture propagates to the caller exactly as before.

## Settlement totality and productless escapes

Every product constructor settles through the authority: the torch
orchestrator's arms (success, halted partial-return, halted re-raise,
failure, interrupt), `Recording.to_trace()` (`cooked_from=recording`
provenance note; a halted cooked trace is HALTED and hits N4/N5), each
recorder pass and the recorder exit, and every preview backend's capture
entry (`stamp_backend_finalized` as the LAST act after all tail work). A
failure BEFORE any product is reachable — configuration errors, backend
constructor failures, preview-backend tail failures between the structural
`_tracing_finished` write and the stamp — is a productless escape: the
exception propagates and no stamp exists, so a hypothetical pickle of the
escaped object derives UNATTESTED, never COMPLETE. Post-settlement teardown
failures demote the settled outcome to FAILED/TEARDOWN in both homes
(`settlement_note` discloses `demoted_from=...`); the exception preempts the
return, so no product escapes carrying the undemoted claim.

`_tracing_finished` remains a STRUCTURAL finished-ness marker, never a
settlement point; the writer inventory is pinned by test and every write
sits inside a region whose only success exit reaches an authority stamp.

## Disclosed conversions and residuals

- Halted analysis `tl.save` was BROKEN before this design (the halted
  finalizer leaked `_output_attribution_input_tensors`, refusing every
  export); it now works, per the capability table.
- Postprocess-tail failures (steps 18–20, the core freeze) used to be MASKED
  by an `AttributeError: _raw_graph_ws` cleanup double-fault; F5's guarded
  cleanups let the original exception propagate, and the settlement stamp in
  `finally` survives any double-fault.
- A refresh now re-arms `raise_on_nan` (F2); it silently disarmed before.
- F3b disclosure: the halted path skips `finalize_forward_session` (and with
  it buffer-write reconciliation). Characterization shows byte-identical
  buffer topology, buffer-write event counts, and model buffer state versus
  the complete capture through the halt frontier, so this remains a
  documented skip, not a behavior change.
- `TerminalState` / `RunOutcome.state` is the first-transition log; it and
  the outcome may legitimately disagree only in the demotion direction,
  disclosed by `settlement_note`.
- Preview cleanup asymmetry vs the torch orchestrator: MLX and Paddle run
  `cleanup_model_session` in a `finally` positioned BEFORE their settlement
  stamp (settlement is the last act, per the path-20 contract), so a
  cleanup/teardown failure there is a PRODUCTLESS escape — the exception
  propagates with no settled record at all, and a hypothetical pickle of the
  escaped object derives UNATTESTED. The torch orchestrator instead settles
  FAILED in its `finally` (and demotes on post-settlement teardown failure),
  so the same failure class on torch yields a settled FAILED product with
  `exc.partial_log`. Net semantics are equivalent on the never-COMPLETE
  guarantee (path 7 minus the demoted record); the missing settled record on
  previews is the disclosed residual.

## Public surface added by this design

`Trace.outcome`, `Recording.outcome`, `PartialTrace.outcome`;
`tl.types.{CaptureOutcome, CaptureStatus, CapturePhase, FailureOrigin}`;
`tl.errors.{CaptureOutcomeError, StopSignalSwallowedError,
PartialCaptureLookupError}`; `RunnableErrorCode.HALTED_CAPTURE_NOT_RUNNABLE`;
two additive `HaltSignal.__init__` kwargs (`boundary_kind`,
`boundary_label`); the stored `Recording._outcome` field; the persisted
`_capture_outcome` Trace field (tlspec v7). Zero renames; nothing enters
top-level `__all__`.
