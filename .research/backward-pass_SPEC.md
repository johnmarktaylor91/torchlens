# SPEC: Backward-pass overhaul — full forward parity (v3, SELF-CONTAINED)

Status: FINAL v3.1 (2026-06-11), BUILD-READY. Adversarial review CONVERGED: round 1
REVISE (3B/7M/12N), round 2 REVISE (1B/5M/6N), round 3 SATISFIED (0B/2M/5N — all
round-3 patches folded in verbatim, tagged [R3-*]). Fully self-contained: no content
incorporated by reference to prior versions.
Branch: capture-unification. Owner: JMT. §1 decisions LOCKED (JMT 2026-06-11);
BLOCKER-DESIGN escalation only.

North star: all power-user backward features; explicit accessible metadata; no holes;
maximal UI clarity matching the rest of the package; MLX/JAX reach as much torch
parity as the frameworks permit.

## 1. LOCKED design decisions

1. **Passes = backward invocations.** Loops never enter backward semantics, naming, or
   grouping.
2. **Per-object GradFn records** — one per distinct autograd node object; the current
   per-Layer merging of recurrent grad_fns is removed.
3. **Backward-native labels** `{type}_back_{type_index}_{step_index}` (type stem from
   the autograd class name normalized: AddmmBackward0 -> addmm; non-Backward classes
   keep their stem: AccumulateGrad -> accumulategrad). Numbering in **discovery (BFS
   walk) order**, uniform for paired AND intervening nodes. Forward correspondence is
   never in the label (metadata + accessor bridge only). `_back` is the direction
   token.
4. **Colon = local dense counter everywhere** (`tanh_back_1_4:2` = that node's second
   firing; a :2 implies a :1). Global alignment stored
   (`GradFnCall.backward_pass_index`) and surfaced via `trace.backward_passes` and
   named lookups. **Index bases:** positional bracket access on accessors is 0-based
   (matches existing OpAccessor/GradFnCallAccessor); colon labels and pass NUMBERS
   are 1-based (matches forward `"attn:2"`). Pass-number lookup is always named:
   `backward_passes.for_pass(2)`, `grad_fn.calls.for_pass(2)`, `op.grad_for(bwd=2)`;
   `backward_passes[-1]` / `last_backward_pass` sugar. Error messages enumerate
   available passes ("participated in passes [2]").
5. **Event-sidecar architecture ("two clocks").** Backward capture is event-sourced
   into `CapturedRun.backward_events` (append-only, separate from the sealed forward
   stream). Trace records (GradFn/GradFnCall/BackwardPass) and Recording records
   (GradientRecord) are two projections of this one stream; the parallel fastlog
   backward implementation is unified onto it.
6. **Per-pass-first payloads.** `op.grads` = local-dense accessor over the op's saved
   per-pass gradient records; plain `op.grad` returns the tensor when exactly one is
   saved, raises the same actionable error class as unsaved activations when zero,
   and raises loudly with guidance ("use op.grads[...] / op.grad_for(bwd=k)") when
   several. Default retention = all passes when grad saving is on. All gradient
   facts flow through the diary; Op-side and node-side payloads share blob refs when
   identical. `param.grad` stays the LIVE accumulated tensor (torch parity);
   per-pass increments via AccumulateGrad fire events (`param.grads`), populated
   only by accumulating triggers (autograd.grad does not fire AccumulateGrad —
   probe-verified; documented).
7. **Unified save surface, clean break (NO aliases).** New surface: `save_grads=`
   (bool or predicate; trace-time standing policy + per-call override on triggers),
   `storage=` vocabulary for gradient streaming/memory policy, existing
   `grad_transform`/`save_raw_gradients` pair unchanged. REMOVED OUTRIGHT (no
   shims): `save_gradients`, `gradients_to_save`, fastlog `keep_grad`,
   `save_grads_to`, `keep_grads_in_memory` (absorbed by `storage=`: in-memory
   default, `tl.to_disk(...)` for streaming; fastlog disk-only constraint parity via
   the existing InvalidStorageError behavior). Removal timing per §13 phasing.
8. **Triggers:** `trace.backward/log_backward`, `recording_backward`, wrapped
   `torch.autograd.grad` AND `torch.autograd.backward`. Higher-order at FULL scope
   (arbitrary order, inductive).
9. **Module containment by correspondence + inference** with
   `module_membership_source: "paired" | "inferred" | None`.
10. **Intervention parity:** live backward hooks ride the new emission path; one-call
    `intervene=tl.when(<backward selector>, <grad helper>)` at trace();
    **differentiable replay in-sprint**; backward-only replay DEFERRED with filed
    rationale (the backward computation is defined by the live autograd graph +
    saved_tensors closures, not TL records; live-graph case is already "another
    backward pass"; reconstruction = a TL-side autograd engine; marginal capability
    over {multi-pass + live intervention + differentiable replay} is one saved
    forward per reloaded session).
11. **Validation:** param-grad parity (keep); per-layer oracle = SEPARATE stock run
    with minimal tensor hooks; gradient-flow consistency invariants; structural
    invariants. LOCKED tripwire: never weaken/exempt to pass.
12. **Visualization:** backward rolled = pass-rolled; unrolled = per-GradFnCall with
    per-pass clusters; `bwd=k` filter; NO loop-grouping (grouped node has no name —
    future-maybe with that blocker recorded); combined fwd+bwd stays unrolled +
    filter; the compact answer is the gradient-arrow overlay on forward rolled,
    promoted, with pass-range annotations whenever a gradient edge is absent for
    some passes; module clusters via §6; higher-order nodes visually distinguishable
    (order badge/tint, styling decided in visual phase); accumulation-edge marking
    decided in visual phase (candidates: state-edge dash reuse vs gradient-colored
    distinct-arrowhead edge with `accum` label).
13. **Backend capability tiers** (§9) with MLX/JAX parity roadmap; zero torch leakage
    into core event types.
14. **BackwardPass record** = first-class peer of Layer for the backward axis
    (fields §4).
15. **Memory accounting:** keep `total_gradient_memory` (op-side); add
    `total_backward_memory` over UNIQUE payload refs.
16. **1-based pass numbering** (0-based positional brackets per §1.4); the global
    0-based indexing sweep stays a naming-sprint item.
17. **Events are RUNTIME-ONLY** (forward parity — verified: forward op_events are not
    persisted and are emptied post-materialize). Persisted: projected records +
    payload blobs. All user surfaces read projected records, never the live diary.

## 2. Hook architecture & capture lifecycle

**Tensor hooks (core-event source): installed at OP-RECORD TIME during forward,
persistent.** A grad hook is registered on every graph-connected op output AT THE
MOMENT ITS OP IS RECORDED (not a post-forward sweep): tensor hooks bind to the
tensor's grad_fn at registration time, so op-record-time installation yields correct
per-op gradients under in-place rebasing (`relu_` chains) where a late sweep
misbinds every label in the chain to the final rebased node (probe-verified)
[R2-positive]. Installation is independent of activation-save policy — the tensor is
live at registration regardless (this also fixes the current crash:
`_ensure_layer_grad_hooks` dies via the loud Op accessor on unsaved outs,
backward.py:893 — reconciliation item). The hook closure holds the op label and a
**weakref to the trace** [R2-6] (strong trace refs in persistent hooks would let any
user-held output tensor pin the whole trace after `del trace` — the GC-8 lesson,
tensor_tracking.py:46-71). Payload retention is decided AT FIRE TIME against the
then-active `save_grads` policy — which is what makes per-call policy widening on
any later trigger correct by construction. Forward passes under `torch.no_grad` or
on tensors with no grad_fn simply have no hooks (nothing to observe — documented).
`backward_ready=False` semantics are unchanged by this spec (out of scope, naming
sprint): when grad_fns are absent there is nothing to hook; when present, hooks
install.

**Node hooks (enrichment source): registered per trigger, removed at bracket end.**
The discovery walk at trigger start hooks not-yet-hooked reachable nodes; their
RemovableHandles are retained for the bracket's duration (dropping handles removes
hooks — probe-verified) [R2-guess-9]. Higher-order re-walk per §5.

**Plain `loss.backward()` IS a trigger [R2-1].** `Tensor.backward` routes through
`torch.autograd.backward` via module-attribute lookup (probe-verified), so once §1.8's
wrapper is installed, a plain backward on a registry-known graph (§5) resolves the
owning trace and runs as a full `"autograd_backward"` trigger — with node-graph
enrichment. The documented training-tutorial flow thereby gets BETTER capture than
today, with zero user change. **Reentrancy guard:** one engine invocation = one
bracket — `trace.backward()` calling `loss.backward()` calling the wrapped
`autograd.backward`, or `recording_backward`'s patched `Tensor.backward` calling
through, must not double-trigger; an active-bracket flag on the trace makes inner
wrapper hits pass through and record `trigger` from the OUTERMOST entry
(`"backward"` for trace.backward; `"recording_backward"`; `"autograd_grad"` for
autograd.grad inside recording_backward — outer context noted in metadata).

**Implicit bracket = fallback tripwire + T0 projection path.** If tensor hooks fire
with NO bracket open (engine entered by a path bypassing both wrapped entry points:
custom C++ extensions, pre-wrap `from torch.autograd import backward` references),
the first orphan emission opens an implicit bracket (`trigger="implicit"`,
`implicit=True`, next pass_index), closed lazily at the next synchronization point
(next trigger start, backward-state access via any accessor/validation/save, or
cleanup). Consecutive bypass backwards merge into one implicit pass (boundaries are
undetectable from tensor hooks alone — documented). A warning explains and points to
TL triggers; warning scope: ONCE PER TRACE [R2-8]. Implicit passes carry
`duration=None, peak_memory=None` (unobservable retroactively; dataframe gates
exempt these) [R2-7]. The implicit path doubles as the T0-shaped projection route
that MLX/JAX backends will use (§9) — build once.

**Thread-safety:** autograd fires hooks on per-device worker threads; emission is
append-only with a GIL-safe monotonic `seq`; projections order by
(pass_index, timestamp, seq); nothing may depend on raw intra-bracket append order.

**Error path:** BackwardPassEnd emitted in a finally with `status="error"` when the
engine raises (retain_graph misuse etc.); **the reentrancy active-bracket flag is
cleared in the SAME finally** [R3-3] — a mid-backward exception must not leave the
flag set (which would silently kill all subsequent capture on that trace); the
MidBackwardException test asserts a SUBSEQUENT pass captures normally. The implicit
bracket's lazy-close chokepoint is the projection/materialization entry (single
place every accessor/validation/save path already flows through) [R3-gp1].

## 3. Event schema (sidecar)

CORE (backend-neutral):
- `BackwardPassStart {pass_index, trigger ("backward"|"autograd_grad"|
  "autograd_backward"|"recording_backward"|"implicit"|"replay"), implicit: bool,
  outer_context: str|None, call_context_ref, root_meta: list (shape/dtype per
  differentiated output), root_grad_arguments (gradient=/grad_outputs= shapes),
  inputs_subset (resolved op labels / param addresses; raw-tensor fallback noted in
  metadata), order, origin_backward_pass, save_grads_policy_repr,
  engine_flags: dict|None (create_graph/retain_graph etc. — nullable,
  backend-interpreted; kept out of typed core fields for purity), timestamp}`
- `OpGradObserved {op_label, pass_index, payload_ref|None, shape, dtype, memory,
  timestamp, seq}`
- `BackwardPassEnd {pass_index, duration, peak_memory, status,
  order_attribution_coverage}`

TORCH ENRICHMENT:
- `GradFnDiscovered {object_id, class_name, class_qualname, is_custom,
  op_label|None, param_ref|None, created_in_pass|None, creator_object_id|None,
  source introspection fields, topology: next_grad_fn_ids}`
- `GradFnFired {object_id, pass_index, grad_input_refs, grad_output_refs (EFFECTIVE
  post-intervention values), intervention_fire_ref|None, timestamp, seq}` —
  single-timestamp timing (real durations via prehooks = filed follow-up).

**Autograd object lifetime (mandatory):** the trace strong-refs every discovered
node object until `cleanup()`/`del` (id reuse is rampant otherwise — probes: 2
unique ids across 50 iterations; walks that drop wrappers silently lose nodes to
seen-set collisions). **Fire-time additions [R2-3]: the post-hook appends the
terminal grad_fn WRAPPER OBJECTS (all grad_inputs' grad_fns — plural) to the
strong-ref set at fire time; every recorded id derives from a retained wrapper.**
Recording path: refs owned by the Recording, same release point. Memory growth +
create_graph reference-cycle caution documented in docs/backward.md.

## 4. Records & projections

**GradFn** (per-object): existing introspection surface (class/source fields)
preserved; plus `order: int|None`, `origin_backward_pass`(=created_in_pass),
`creator_object_id`, `differentiates` (label of the creator node, derived),
`modules`/`module_address` + `module_membership_source` (§6); `calls` local-dense
accessor (0-based positional + `for_pass(k)` + `"label:j"` 1-based). The two legacy
type_index schemes collapse to backward-native counting.
**GradFnCall**: `ordinal` (local 1-based, = colon value), `backward_pass_index`
(global), effective payload refs, `intervention_fire_ref`, `is_saved`, timestamp.
**BackwardPass**: `pass_index, trigger, implicit, outer_context, call_context,
root_grad_fn_ids, root_meta, root_grad_arguments, inputs_subset, order
(ROOT-based, §5), origin_backward_pass, engine_flags, save_grads_policy, duration,
peak_memory, status, order_attribution_coverage, grad_fn_calls (ordered)`.
**Op**: `grads` accessor, `grad` (per §1.6), `grad_for(bwd=k)`; existing grad
metadata fields preserved; Op.grad becomes a derived property over per-pass records
(dataclass-field -> property migration, named).
**Param**: `grad` live; `grads` per-pass increments (accumulating triggers only).
**Recording/GradientRecord**: re-implemented as a projection of the same stream;
`GradRecordContext` extended with `pass_index`, `order`.

**Loud-accessor representation rule:** dataframes and serialization NEVER raise.
to_pandas: trace-level dataframes for grad_fns / grad_fn_calls / backward_passes
(FIELD_ORDER-derived column gates, completeness-tested); op-level `grad` column =
value when unambiguous else None, plus `num_saved_grads`. Portable state stores
per-pass records; derived properties excluded. Same fix applied to
`GradFn.backward_duration` (crashes to_pandas TODAY on multi-call GradFns): None
when ambiguous + `total_backward_duration`.

## 5. Triggers, registry, higher-order

**Trace-discovery registry:** maps known grad_fn object ids -> owning trace.
**Population is INCREMENTAL at op-record time** (reconciled at forward finalization
to exactly the LAST live run's ids; superseded first-pass ids dropped) [R3-1]. A
mid-forward engine invocation (autograd.grad inside model.forward — gradient
penalty, inner-loop meta-learning) therefore resolves as a real `autograd_grad`
trigger, never the implicit fallback; forward finalization is also a lazy-close
sync point for any open bracket. Entries removed at trace cleanup.
**Pin-invariant [R3-2]:** every registry key is the id of a wrapper object the
trace strong-refs (per-op grad_fn_handle and/or the §3 fire-time set); never
register an unpinned id — dead-graph id reuse would false-positive-trigger on
unrelated graphs (probe: 1 unique id across 5 dead graphs unpinned; 5/5 unique
pinned). Corruption test included.
The wrapped engine entries inspect the passed roots' graphs for known ids.
**Performance contract [R2-4]:** (a) empty-registry zero-cost bail; (b) early exit
on first known id (note: the no-match walk cost scales with the user's CURRENT
graph size — early exit never helps the no-match case; documented); (c) overhead of
the live-trace-during-training case documented, with `trace.disarm_triggers()`
provided to detach a kept-for-analysis trace from engine interception (also runs at
cleanup). **Disarm semantics [R3-5]:** disarm also no-ops tensor-hook emission for
that trace (a disarmed trace captures nothing; no implicit-bracket surprise). An
optional once-per-trace perf HINT after N consecutive no-match registry walks
points at disarm_triggers(). Auto-disarm is explicitly REJECTED (silent capture
loss). **Multi-trace roots [R2-12]:** a backward whose root graph matches ids from
MULTIPLE live traces opens a bracket on EACH matched trace (tensor-hook emissions
route per owning trace via their closures); a second trace's bracket opened from
inside another trace's `trace.backward()` records `trigger="autograd_backward"`
with `outer_context` noting the foreign trigger [R3-gp3].
**Two-pass capture parity [R2-2]:** the fast pass currently never refreshes
`grad_fn_object_id`/`grad_fn_handle`, so the returned trace's live graph is invisible
to pairing AND the registry (probe: 0/11 paired; zero recorded ids present in the
walked graph). REQUIRED FIX, hoisted to P1/P2: the fast pass refreshes per-op
`grad_fn_object_id` + `grad_fn_handle` at counter-aligned op-record time (it already
touches every out tensor), **and re-runs op-record-time TENSOR-HOOK installation on
the live run's tensors (first-pass hooks die with the dead graph) [R3-4]**; the
registry is populated from the LAST live run's ids. The TwoPassPairingParity test
locks BOTH pairing parity AND op.grads capture parity between full-save and
`layers_to_save` modes.

**Higher-order (FULL scope), concrete algorithm:** mixed-order passes are the norm
(order-2 graphs REUSE order-1 nodes — TanhBackward0 refires alongside
TanhBackwardBackward0; probe-verified). Node order is CREATOR-based:
- When node X fires under create_graph, nodes recorded by X's execution carry
  `creator_object_id = X` and order = order(X) + 1. Mechanism: in X's post-hook,
  capture the terminal new nodes via each `grad_input.grad_fn` (plural,
  wrapper-retained per §3 [R2-3]); the post-pass re-walk propagates creator
  attribution through the newly-created subgraph (reachability from terminals,
  stopping at previously-known nodes). Forward-created nodes: order 1, creator None.
- **Within-pass contention [R2-11]:** a new node reachable from multiple firings'
  subgraphs takes the attribution of its FIRST re-walk discovery (deterministic);
  if candidates differ in order, take the MINIMUM creator order; genuinely ambiguous
  -> order=None (never nondeterministic, never guessed).
- Re-walk: synchronous at bracket end. Roots: autograd.grad -> the returned grads'
  grad_fns; backward(create_graph=True) -> leaf `.grad` tensors' grad_fns (params +
  requires_grad inputs; probe: param.grad.grad_fn == CopyBackwards; leaves whose
  `.grad is None` — sparse participation — are skipped [R3-gp4]).
- `BackwardPass.order` is ROOT-based: 1 + (order of the pass that created the root
  graph; 0 if forward-created). Node order within a pass legitimately varies.
- **Invariant scoping:** the order-chain invariant (order = creator order + 1;
  origin pass exists; forward-created => order 1) applies to nodes WITH resolved
  attribution; `order_attribution_coverage` reported per pass; corruption tests
  prove the invariant trips on resolved chains; unresolved nodes carry order=None.
- Tests: orders 1-3 explicit + induction stress (order N loop) + mixed-order pass
  assertions.

## 6. Module containment

- Paired nodes: the op's containment (`source="paired"`); AccumulateGrad: the
  param's module home (`source="paired"`).
- **Post-forward carve-out [R2-10], operationalized:** unpaired,
  non-creator-attributed nodes encountered on the BFS from the root BEFORE the
  first op-anchored node are post-forward (loss-construction chains: sum -> mean ->
  scale) -> modules=None, source=None.
- Intervening in-forward nodes: creation attribution — owner = the
  later-in-forward-time op neighbor ("the op without which this node would not
  exist"); same-module shortcut when sandwiched between same-module ops. **Tie
  rule:** among equal-adjacency candidates prefer the consumer (later step_index);
  then smaller step_index; then lexicographic label. **Propagation:** BFS from
  op-anchored nodes through intervening-only chains in discovery order; each
  unresolved node takes attribution from its first-resolved neighbor per the tie
  rule. `source="inferred"`.
- Pairing works in ALL capture modes (two-pass fix per §5).

## 7. Save surface, predicates, intervention

1. `save_grads=` on trace() (standing) and on every trigger (per-call override —
   correct by construction per §2 fire-time policy). Bool sugar: True ≡ everything,
   False/None ≡ off (default off). Predicates: the forward language verbatim
   (`tl.func`, `tl.in_module`, compositions) evaluated against GradRecordContext,
   plus grad atoms: `tl.grad_input()`/`tl.grad_output()` kind selectors, existing
   `tl.grad_fn`/`tl.intervening`/`tl.grad_fn_label` selectors, and
   `tl.in_backward_pass(k)` (explicit spelling; terse aliases rejected).
2. `storage=`: in-memory default; `tl.to_disk(...)` streams gradient payloads
   (fastlog disk-only constraint parity; InvalidStorageError behavior preserved).
3. `intervene=` accepts backward when()-clauses at trace(); helpers unchanged
   (`tl.grad_zero/scale/clip/noise/clamp`, custom via bwd_hook with existing
   portability classes).
4. Per-bracket `save_grads_policy` snapshot makes retention auditable on every pass.
5. Clean-break removals per §1.7; timing per §13.

## 8. Differentiable replay

- **Spelling/semantics:** `trace.replay(..., differentiable=True)` RETURNS A NEW
  Trace over its own CapturedRun + own sidecar (fork machinery underneath; Bundle
  relates runs). Non-differentiable legacy replay keeps in-place semantics,
  unchanged.
- **Frontier attachment:** saved tensors enter as FRESH LEAVES
  (`detach().requires_grad_()`) at the replay cone frontier — gradients flow within
  the replayed computation only; no backprop into run-A buffers; no cross-run
  leakage. Frontier leaves queryable (`replay_frontier` metadata).
- **Pairing/hooks:** differentiable replay is a CAPTURE (runs through the wrapper
  machinery): fresh tensor hooks, fresh ids, fresh pairing by construction.
- Memory + detach_saved_activations interplay documented. Acceptance test:
  attribution patching end-to-end (patch activation from run A into a replay of run
  B, backward, read op.grads) verified against a hand-built torch reference.

## 9. Backend capability tiers & parity roadmap

Tiers (each backend declares its ceiling):
- **Derived-gradient preview (non-backward)**: backend-specific functional AD may expose
  leaf gradients outside the backward event model. JAX M1 uses this route through
  `trace.derived_grads`; it does not populate `op.grads`, `save_grads` predicates,
  `backward_passes`, or backward validation flow checks.
- **T0 — pass brackets + leaf gradients**: BackwardPassStart/End + OpGradObserved
  for params/inputs. This remains a true-backward event tier, not the JAX M1 preview.
- **T1 — per-op intermediate gradients**: OpGradObserved for arbitrary ops. JAX T1 is
  research and requires an adjoint design, cost model, and primitive/custom-rule coverage
  table before it can become committed roadmap work.
- **T2 — node-graph enrichment**: GradFnDiscovered/Fired, per-object records, live
  backward intervention. Torch-only (functional AD has no hookable graph);
  permanently out of reach there unless the frameworks grow one. Surfaces degrade
  explicitly (trace.grad_fns empty + tiered capability-flag errors).

| capability | torch | mlx (this sprint) | jax |
|---|---|---|---|
| derived-gradient preview (non-backward) | n/a | research | JAX M1 via `trace.derived_grads` |
| T0 brackets + leaf grads | yes | NotImplementedError + flag | research |
| T1 per-op grads | yes | no | research |
| T2 node graph / live bwd intervention | yes | no | no |

Requirements: (a) zero torch leakage into core event types (object ids and grad_fn
concepts live only in enrichment); (b) `backward_capabilities` declared per backend
with tiered actionable errors; (c) follow-up filed with the MLX T0/T1 implementation
sketch. The implicit-bracket projection path (§2) is the same T0 path true-backward
functional backends would use if later funded.

Amendment note (J4, autonomous overnight mandate): the JAX row was changed to
"derived-gradient preview (non-backward); T1: research" and committed-roadmap language
claiming an implementation path for JAX T1 was scrubbed. JMT should review this delegated decision.

## 10. Validation

(i) Param parity: keep as-is. (ii) Per-layer oracle, mechanism (b): a SEPARATE
state-restored stock run with minimal tensor hooks (no TL machinery; clone at hook;
no retain_grad; zero_grad + RNG restore) comparing per-layer dL/d(output)
pass-for-pass with dtype-aware tolerances. The four AD-32 hazards (retain-grad-on-
clone, same-run coupling, version counters, accumulation bleed) become regression
constraints; build the oracle FRESH — do not resurrect previously-deleted oracle
helpers from git history [R3-gp5]. (iii) Flow-consistency: scoped to
NON-intervened quantities (events carry effective values + intervention provenance;
intervened quantities validated against intervention fire records — no tolerance
hacks); shared-ref path true by construction; independently-captured comparisons
exact. (iv) Structural invariants: bracket integrity (dense pass_index; every fire
in exactly one bracket, implicit included), local-dense ordinals, per-object
uniqueness (one record per retained object), order-chain on resolved subset +
coverage metric, label grammar, containment flags, creator/differentiates targets
exist. All additive; a corruption test per check proves each trips.

## 11. docs/backward.md (new page, required)

(1) Mental model — two clocks, diary, projections; (2) capturing gradients
(save_grads/storage/transforms; plain-backward-is-a-trigger; implicit-pass fallback
semantics); (3) the records (Op.grads, GradFn, GradFnCall, BackwardPass) with the
colon-semantics and index-base side-by-side example; (4) multiple backwards &
accumulation (param vs op; per-pass increments; trigger-dependence of param.grads);
(5) higher-order (mixed-order passes explained; order=None honesty); (6)
intervention — live + differentiable replay with the attribution-patching worked
example; frontier semantics; (7) module containment (paired vs inferred vs None);
(8) validation; (9) limitations & costs (MLX tiers, backward-only replay rationale,
impure-closure replay policy, strong-ref memory growth + create_graph cycles,
registry overhead + disarm_triggers, implicit-pass merge semantics).

## 12. Test plan (zoo + matrix; models land with their phase)

Zoo: MultiBackwardSameLoss (accumulation), TwoLossHeads (sparse participation),
InputsSubsetGrad (autograd.grad + inputs=), RetainGraphReBackward,
CreateGraphSecondOrder, ThirdOrderChain, OrderInduction(N), MixedOrderPass (order-1
refire + order-2 creation in one bracket), RecurrentLoopBackward (per-object labels,
reverse-order indices), InterveningNodeModel (inference + boundary tie case),
PostForwardLossChainModel (carve-out), AccumulateGradParamModel (param.grads),
InplaceChainModel (op-record-time hook binding under in-place rebasing [R2]),
PlainBackwardTriggerModel (plain loss.backward() = full autograd_backward trigger;
per-backward passes; tutorial parity), ImplicitFallbackModel (engine entry bypassing
wrapped entry points -> implicit bracket + merge + once-per-trace warning;
constructible via direct engine invocation
`torch.autograd.variable.Variable._execution_engine.run_backward(...)` or a
pre-wrap saved reference to the original autograd.backward [R3-6]),
GradPenaltyMidForwardModel (autograd.grad(create_graph=True) INSIDE forward —
WGAN-GP/MAML pattern; asserts a real autograd_grad pass during capture and correct
attribution of reused higher-order nodes in the subsequent outer backward [R3-1]),
CheckpointedSegmentModel (torch.utils.checkpoint, BOTH use_reentrant modes:
no-crash, graceful coverage degradation, reentrancy guard passthrough on the
recomputed graph — probe-verified behavior pinned; limitations documented in
§11(9)) [R3-7],
AutogradBackwardFunctionForm, AutogradGradInsideRecordingBackward (precedence +
reentrancy), MultiTraceRootsModel (one loss over two live traces -> a pass on each),
MultiForwardRecordingLoop (id-reuse stress under strong-ref policy),
TwoPassPairingParity (layers_to_save pairing == full-save pairing),
InterventionFlowConsistency, MidBackwardException (finally/status=error),
NonScalarRootBackward (gradient= capture), DifferentiableReplayPatching
(acceptance), OracleHazardRegressions (in-place model, version-counter-sensitive
model, dropout/RNG model), MLXBackwardRaises (tiered message), RecordingPathParity
(Recording projection facts == Trace projection facts), RehydratedAccessors (tlspec
round-trip then accessor behavior).
Matrix per model: records exact (labels, ordinals, pass indices, order + coverage,
containment + source); op.grad loud/quiet/zero behavior + error text; named lookups;
predicates (save_grads incl. in_backward_pass; intervene backward); retention policy
+ per-call override + bracket policy snapshot; to_pandas rows (never-raise rule);
tlspec round-trip; validation green + corruption tests trip; viz smoke per mode
(rolled/unrolled+clusters/bwd filter/combined/overlay annotations) with DOT-level
assertions + one backward demo added to the label-geometry gate.

## 13. Implementation phases

P1  Sidecar schema + emitters: op-record-time tensor hooks (weakref closures) +
    per-trigger node hooks emitting events + reentrancy guard + implicit-bracket
    fallback + strong-ref registry (incremental population + pin-invariant) &
    fire-time wrapper retention + the two-pass grad_fn id/handle/hook refresh
    [R2-2/R3-4 hoisted] + reconciliation grep inventory. `trace.backward`/
    `log_backward` switch onto the new emitters IN THIS PHASE [R3-gp2].
P2  Projections: GradFn/GradFnCall/BackwardPass materialization; labels/numbering;
    accessors + index bases; kill per-Layer merging; structural invariants.
P3  Payloads + policy: OpGradObserved replaces direct Op.grad writes; op.grads /
    param.grads; retention + save_grads + predicates; TRACE-SIDE clean-break
    removals (save_gradients, gradients_to_save, save_grads_to,
    keep_grads_in_memory) — NOTE the two internal call sites that must be updated
    in the same phase to keep the suite green: fastlog/_recorder.py:361
    (gradients_to_save=[] plumb) and postprocess/_materialize.py:380
    ("save_gradients" mapping) [R2-9]; shared-ref dedup; memory aggregates;
    loud-accessor representation (to_pandas/tlspec) + backward_duration fix.
P4  Triggers: autograd.grad + autograd.backward wrappers + registry with
    performance contract + disarm_triggers; higher-order full (re-walk, creator
    attribution + contention rule, order computation + coverage); order tests.
P5  Module containment (tie rule, propagation, post-forward carve-out, pairing
    parity test).
P6  Differentiable replay (new-run semantics, frontier leaves, acceptance test).
P7  Validation: oracle + flow-consistency (intervention-aware) + corruption tests.
P8  Recording/fastlog unification onto the sidecar + FASTLOG-SIDE removals
    (keep_grad) + storage= mapping + GradRecordContext extension; delete the
    parallel implementation.
P9  Visualization (per §1.12; accumulation-edge styling decided in this phase's
    visual review).
P10 tlspec + JSON schema (backward/grad fields added) + to_pandas dataframes +
    summary()/report.explain backward sections + docs/backward.md + glossary sync
    list + tracker filings (backward-only-replay rationale, MLX T0/T1 sketch,
    prehook-timing follow-up, implicit-boundary-detection follow-up) + old-bundle
    load policy (legacy class remaps load; removed fields dropped with manifest
    note).

Per-phase gates: ruff + mypy clean on touched files; smoke green before each phase
commit; the new test file green for that phase's models; full not-slow green at
P10; tripwire grep-proof (no invariant weakened); forward parity locks
(byte-identical forward stream) untouched throughout. The P1 reconciliation
inventory is re-checked at every phase boundary; seed list: validation backward
suite; GradFn/GradFnCall unit tests; fastlog gradient tests; tlspec fixtures; tests
asserting old labels/mirrored indices/merged recurrent GradFns;
intervention/_super/super_logs.py SuperGradFn label-keyed alignment; bundle
grad_fns/grad_fn_calls accessors; glossary conformance tests
(test_glossary_conformance_b2b/b2c); user_funcs.py MLX save_gradients error path;
internal fields Op.save_gradients / Trace.save_gradients / has_gradients /
_grad_layer_nums_to_save / _saved_grads_set; notebooks/training_tutorial.ipynb
(plain-backward flow — now a trigger, retest); the selective-save log_backward
crash; the GradFn.to_pandas multi-call crash; ir/backward.py BackwardSidecar
scaffold (replaced; no dead scaffolding remains).

## 14. Out of scope (filed, not built)

Backward-only replay (rationale filed); MLX/JAX backward implementations (schema +
capability flags + sketch only); Tier-3 training-time gradient routing; global
0-based indexing sweep; loop-grouping for backward views (blocker: grouped node has
no name); `backward_ready` rename (naming sprint); real per-fire durations via
prehooks (follow-up); implicit-bracket boundary detection improvements (follow-up).
