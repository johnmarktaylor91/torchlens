# Trace core design — the converged god-object columnar re-plumbing

Status: CONVERGED (tri-lab, round 2), now HISTORICAL. Basis tree `781a7559`. Every
schema statement below is as-of that pre-v8 tree: the tlspec v8 coordinated bump has
since shipped, so "no `TLSPEC_VERSION` bump" and "rows remain `RUNTIME`/DROP-gated"
sentences describe the sprint's scope at the time, not the current tree — the runtime
authority is `torchlens/_io/__init__.py` and the per-class `FIELD_POLICY` tables.
This document is the
architecture of record for the columnar re-plumbing sprint: it synthesizes the three
independent round-1 designs and the three round-2 reconciles into ONE plan. Where the
round-2 documents state a 2-1 HOLD, the majority position is the plan and the minority's
reopening conditions are recorded verbatim. User-visible choices are reserved as JMT
forks, never resolved here.

Sources (private sprint records, not in-repo): `godobject-r2-fable.md` (merged M0-M13
skeleton), `godobject-r2-opus.md` (the adjudication table; its verdicts are the
convergence record), `godobject-r2-codex.md` (sol's reconcile; PACK-boundary evidence,
column backing, edge-occurrence table, oracle matrix), plus the three round-1 designs
and the runtime census probe (`godobject.md`).

## 1. Final form, in one paragraph

TorchLens has one declared schema and two authorities. The schema is data: for each
record kind a binding table declares every field once — name, order, portable policy,
fork policy, default, annotation, codec, mutability, storage kind — and the column
layout, the facade's annotated `@property` descriptors, the save/pickle state adapters,
the dataframe row, and the `.pyi` are generated from it and checked in; the hand-written
`FIELD_ORDER` literals remain the declared schema those tables must agree with, asserted
in both directions. The two authorities are the **journal** and the **store**. Capture
appends frozen, sequence-numbered events to the journal — authority for *what executed*;
non-portable, droppable. Postprocess projects the journal at materialize time into the
store's mutable builder and transforms builder-backed rows through the 20 ordered steps
with declared column read/write contracts; the result is the **store** (`TraceCore`:
numpy-backed typed columns, per-trace intern pools, one canonical edge-occurrence table
with CSR indexes, pooled ancestor bitsets, group tables, an identity-preserving payload
arena, sparse versioned mutation overlays) — authority for *the finished trace*, and the
only thing saved. Forward topology freezes at step 17, the payload plane settles through
step 20, the core freezes there, and backward passes append atomic epochs. `Trace`,
`Op`, `Layer`, `Module` and siblings own no data: they are two-word row facades
presenting the same flat, navigable, IDE-inspectable API — information about an op still
hangs off the op, it just is not *stored* there. `Trace` is a presenter over
lifecycle-owned components (capture config, core, witness diagnostics, source metadata,
totals, runnable state), so no class is both the storage and the junction, and
`TraceBuildState` is gone — replaced by named per-phase workspaces each owned by exactly
one postprocess phase.

## 2. Canonical census (runtime probe, venv torch 2.13.0, tree `781a7559`)

| class | FIELD_ORDER | storage | @property | defs |
|---|---:|---|---:|---:|
| Trace | 220 | dict | 126 | 133 |
| Op | 183 | slots(185) | 72 | 37 |
| Layer | 98 | dict | 77 | 16 |
| Module | 87 | dict | 75 | 18 |
| ModuleCall | 54 | dict | 45 | 13 |
| Param | 31 | dict | 19 | 13 |
| Buffer | 20 | dict | 26 | 7 |
| GradFn | 52 | dict | 18 | 8 |
| GradFnCall | 12 | dict | 6 | 7 |
| BackwardPass | 18 | dict | 3 | 6 |
| FuncCallLocation | 14 | dict | 7 | 10 |

789 declared fields across the eleven policy-bearing public records. Op container
defaults: 44 entries (22 list / 12 dict / 5 set / 5 tuple), ~39 of them mutable,
~3.0 KB/op of fresh containers at construction. Measured baselines replacing folklore:
shallow Op = 1512 B/op; construction 126-206 us/op; deep probe ~42 KB/layer on a
19-layer CNN including per-trace fixed overhead. The transient 183-key `fields_dict`
at the single Op construction site costs ~9-10 KB/op at peak. All numeric targets are
stated against these measurements, re-verified at M0; prose counts are never copied
forward without a re-probe.

## 3. The agreed architecture

### 3.1 Two authorities, one-way projection (unanimous)

- The **journal** (`_capture_events`) is authoritative for chronological backend
  observations: what executed. It is `FieldPolicy.DROP` — non-portable, absent on loaded
  traces, droppable via `forget_event_stream`. It stays record-shaped (frozen+slots
  events) and stays a sibling sidecar owned by the Trace/captured run, NOT a child of
  `TraceCore` (lifecycles differ). It MAY intern labels into the core's pools (opt-in,
  measured); documented consequence: dropping the journal does not compact pool entries
  it uniquely referenced.
- The **semantic store** (`TraceCore`) is authoritative for the finished trace —
  including synthesized, pruned, relabeled, grouped, and aggregated facts the journal
  never contained — and is the only portable representation.
- No-duplication rule, enforced by test: finished-trace code paths never read the
  retained journal outside a backward/run-refresh/validation-provenance whitelist.

### 3.2 Store shape (unanimous)

One `TraceCore` per trace under `torchlens/_trace_core/` (private namespace). Per-kind
typed tables with narrow typed APIs — each domain owns its table/schema/builder/query
interface; `TraceCore` supplies lifetime, IDs, transactions, and cross-domain
coordination only, so it never becomes a generic dict-driven GodStore. Components:

- **Columns**: NumPy-backed chunked builders; bit-packed bools/validity bitmaps,
  uint8/16 enums, int32 row ids, int64 sizes, float64 times, `float.hex`-stable floats.
  Never object arrays as a default (an object array reintroduces the per-cell Python
  object being deleted). PyArrow stays optional and out of core. Builders may chunk;
  freeze allocates each final array once.
- **Pools**: per-trace intern pools, never process-global. `_POOLED_CLASSES` /
  `_pool_key` absorbed verbatim including the exact-type injectivity proof (`True`, `1`,
  `1.0`, `Bytes(1)` never collapse). Payload tensors, callables, live handles, and
  mutable user objects never enter value pools. Journal and semantic planes share pools
  so a raw label is stored once per run. `_compact_op_metadata` stops being a post-pass:
  pooling becomes the builder's write path.
- **Ancestor closures**: the column contract is a pool-id into an interned
  ancestor-closure pool (one entry per distinct closure over a trace-local label table).
  Encoding is an internal pool choice: dense bitset (the shipped `_AncestorBitset`
  design, generalized to all four ancestor sets) by default, sparse frozen row-id vector
  when the internal-source universe would make a dense bitset wasteful (M0 stress
  synthetic guards this). Closures rematerialize as real `set` on the facade; nothing
  new crosses serialization.
- **Relations**: dense foreign-key columns for single-target relations (op->layer,
  op->func_call, modulecall->module, gradfn->op, gradfncall->gradfn/backwardpass) plus
  ONE canonical edge-occurrence table — (source, target, use kind, argument
  position/path, conditional role, stable seq) — with parent/child CSR indexes holding
  edge ids for everything variable-cardinality. Parallel edges and per-occurrence
  metadata can never collapse. `edge_use_kind`/`is_control_edge_use`/`is_value_edge_use`
  (`ir/events.py`) supply the vocabulary.
- **Groups**: sol's taxonomy — `FunctionCallGroup` (per `func_call_id`),
  `EquivalenceGroup`, `RecurrenceGroup` (ordered), `ParamAliasGroup`, conditional
  source/span groups, Layer-as-aggregate — in opus's representation: group data lives in
  group column-blocks with a `group_id` int column on members; never a shared Python
  record per group. Op rows keep only output-specific facts. A direct write to one Op of
  a shared call group creates a PER-OP overlay, never mutates the shared group row.
  `_copy_shared_fields_for_output` dies here.
  M7 as-landed disposition: Equivalence/Recurrence ship as `MembershipGroups`/`GroupRef`
  live views (`groups.py`); FunctionCall call-facts and the ParamAlias value block ship
  as shared-fact column families (`fact_blocks.py`, `_FACT` member cells, per-row
  hydration on the facade); conditional source/span groups need NO separate group table
  — every per-op conditional relation container is already an interned immutable view
  (M6, equal views share ONE object through the intern pool) and the dense-id primary
  conditional records live on the Trace side (M10 decomposition territory), so a
  conditional group block would duplicate existing sharing without deleting any copy.
- **Payload arena**: preserves IDENTITY, not equality. Same tensor object = same handle;
  equal-but-distinct stays distinct; no value dedup; lazy blob refs stay lazy; in-place
  mutation stays visible through every aliasing handle; replacement writes allocate
  fork-local handles; fork sharing follows today's field policy exactly.
  `save_budget`'s alias-aware accounting reads the arena table instead of re-deriving.
- **Overlays**: sparse versioned per-row overlays for all public writes; transactions
  checkpoint overlays + epochs and roll back atomically.
- **Facade cache**: identity-preserving `(kind, row_id) -> facade` cache
  ("strong-then-weak", section 3.5).

### 3.3 Ingress, mutability planes, and freeze (r2 majority: fable + opus, 2-1 over sol)

Ingress is at step 0: the single Op construction site (`postprocess/_materialize.py`,
the only `Op(fields_dict)` site tree-wide) switches to journal->column-builder
projection. The ~9-10 KB/op transient `fields_dict` and the ~3.0 KB/op of 39 fresh
mutable containers die at that seam. Postprocess steps 1-16.5 mutate builder-backed
facades: scalars are columnar immediately; relation fields remain real mutable staging
containers until their family freezes into the edge-occurrence table. During migration
the builder exposes internal row facades so unmigrated passes keep working and migrate
family by family. No second retained representation survives the sprint.

Freeze is PER-PLANE, not one switch:

- Forward **topology** freezes at step 17 (`_set_tracing_finished`).
- The **payload/runtime plane** legitimately mutates through steps 18-20 (streamed
  bundle finalize, out-eviction, param-ref release); the whole core freezes after step
  20, exactly where `_compact_ancestor_sets` runs today — the freeze step inherits the
  compaction passes' role and the standalone passes are deleted.
- `log_backward` appends an atomic backward **EPOCH** (generation bump), preserving the
  exact lazy watermark/revision invalidation semantics.
- Direct writes and interventions land in the sparse versioned **overlay**.
- Derived caches stay disposable.

Sol's minority position (retain the mutable object graph through postprocess, PACK after
step 20) is not the plan; its gating conditions are adopted verbatim as the M5 seam's
own acceptance evidence: "phase read/write contracts cover every Step 0-20 access; a
row-protocol prototype passes the complete surface/alias/artifact oracle for Op, partial
capture, streamed payload eviction, refresh, and repeated backward; peak memory improves
materially; and no second mutable representation survives." Sol's PACK-boundary facts
stand and are honored: step 17 is a logical transition, not the final physical mutation
point; error/partial paths must produce facade-backed results immediately before public
exposure; partial cores carry `topology_complete=False` and never pretend full
postprocess ran.

Pack-on-expose dissolves by construction (facades exist from step 0, so `exc.partial_log`
and `return_diverged` traces are facade-backed automatically), but the escape-path
inventory survives: every place a Trace/Op leaves `torchlens.*` mid-postprocess
(observers, `tap()`, `span()`, validation internals, bridge adapters) is
inventoried at M0 and covered by oracle scenarios.

"C10" is split (opus's adjudication): "no transient fat-Op graph" is IN (the M5 seam);
"capture writes semantic columns with no journal at all" is OUT permanently — the
journal is a separate authority; merge law, partial capture, and replay depend on it.

### 3.4 Schema: one declared table, generated artifacts (converged keystone)

The existing `RecordFieldPolicy` declaration is extended into the canonical per-kind
binding table. Axes stay typed and separate: persistence/fork/default metadata remains
`RecordFieldPolicy`; a nested `StorageBinding` supplies `StorageKind`
(SCALAR/INTERNED/BITSET/EDGE/GROUP/PAYLOAD/RUNTIME/COMPUTED), annotation, codec,
serializer, null policy, and mutability/view policy — so `FieldPolicy.DROP` never
implies a physical layout.

The current derivation runs backwards (the policy table is built from the FIELD_ORDER
literal at 10 sites). It is inverted with a byte-provable staged transition: the
checked-in `constants.py` literals stay hand-written and authoritative as the declared
schema, and `field_order_from_policy(cls.FIELD_POLICY) == LITERAL` is asserted for all
12 classes, in both directions, before anything is generated. The twelfth class is the
DROP-gated `primitive_op` row: its installed wave-0 bindings remain `RUNTIME`, while an
isolated prerelease audit derives the intended `INTERNED`/`EDGE`/`SCALAR` bindings without
activating a v7 writer.

Generated artifacts, all deterministic, CHECKED IN, and reviewed as source with a
regenerate-and-diff CI test (never runtime codegen, never a bare `__getattr__` field
surface): column layout; annotated `@property` facade descriptors; the explicit semantic
state protocol (`__tl_state_items__`/`__tl_state_restore__`) closing the CRITICAL
state-adapter blindness for all 12 classes at once (13 production consumers); the
`to_pandas` row; the `.pyi` stub (dropped only if the DX oracle shows it conflicts with
inherited annotated properties). Lockstep tests: every FIELD_ORDER entry has exactly one
binding; every portable field a serializer; every stored column declared or explicitly
private; computed collapse metadata stays out of serialization.

### 3.5 Facades, identity, and mutation semantics

- `Op.__slots__ = ("_core", "_row")` — strict, because Op's slots reject unknown names
  today. Dict-backed classes (`Trace`, `Layer`, `Module`, `ModuleCall`, `Param`,
  `Buffer`, `GradFn`, `FuncCallLocation`, ...) keep arbitrary user attributes via
  `__slots__` + `"__dict__"` (lazy instance dict; laziness verified by measurement) —
  slotting them without it is an observable break. Declared fields are real descriptors,
  never stored in extras.
- **Facade identity cache — strong-then-weak.** The observable contract is
  `trace["x"] is trace["x"]` while any reference is live; mutations live in the core, so
  identity is a UX nicety, not a correctness carrier. Sol's lifetime point is honored by
  sequencing: the first production cutover (M5) lands a STRONG per-core cache
  (byte-identical lifetime behavior — an Op stays alive while its Trace lives, exactly as
  today), then the cache flips to weak-valued within the sprint once the M0 lifetime
  inventory and the aliases-v1 oracle prove no observable behavior depends on facades
  surviving with zero user references. Weak-valued is the end state (r2 majority:
  fable + opus): a strong table makes "near-zero retained objects per uninspected row"
  unreachable — one full iteration of a 1M-op trace would pin 1M facades. Deciding
  evidence against the flip: any not-slow case comparing identity of a facade re-fetched
  after all refs were dropped.
- **No hard freeze, ever.** Physical base immutable; the public facade is not.
  `Op.__setattr__`'s existing contract (`DirectActivationWriteWarning` +
  `TraceState.DIRECT_WRITE_DIRTY`) is public behavior under the LOCKED tripwire;
  hard-freezing would turn a warning into an exception — a behavior break dressed as
  safety. All public writes land in the sparse versioned overlay AND stamp the dirty
  state byte-identically; `_internal_set` remains the unstamped path. Sanctioned
  writers: backward epoch append; sparse `annotations` sidecar; intervention/
  direct-write overlay; the step 18-20 payload plane.
- **Relation accessors — immutable views (JMT-DECIDED 2026-08-12, supersedes the
  lazy-hydration default for relation fields).** The label-sequence and label-set
  relation accessors (`parents`, `children`, `modules`, `module_call_stack`,
  `input_to_module_calls`, `output_of_modules`, `output_of_module_calls`,
  `internal_source_parents`, `parent_params`, the conditional child/stack lists,
  `input_ancestors`, `output_descendants`, `root_ancestors`,
  `internal_source_ancestors`) return IMMUTABLE views — `tuple` for sequences,
  `frozenset` for sets — once a trace is finished. This is an authorized public-type
  break: in-place mutation (`op.children.append(...)`) raises instead of sticking, and
  equal immutable views may be shared across records. During postprocess the staging
  containers remain real mutable builtins; the conversion happens at the per-family
  freeze. Dict-shaped relation metadata (`parent_arg_positions`,
  `out_versions_by_child`, `module_entry_arg_keys`, `conditional_elif_children`,
  `conditional_arm_children`, `parent_param_ops`) keeps its mutable dict type (a
  mappingproxy would break `isinstance(..., dict)` consumers and pickling; revisit
  only with a separate authorization).
- **Non-relation public mutable containers — lazy hydration.** Internal consumers read
  compact storage; the first PUBLIC access materializes the exact builtin into that
  row's overlay, which becomes authoritative for that row; uninspected rows stay
  compact.
- **The two group-membership fields** (`equivalent_ops`, `recurrent_ops`): JMT-DECIDED
  2026-08-12 — LIVE group-membership views (immutable, backed by the shared group row;
  reads reflect group state, caller mutation impossible), replacing the historical
  fresh-mutable-copy-per-read barrier. Lands in M7 with the group tables.
- **`Op.copy()`**: internal output-node synthesis is a builder row append seeded from
  the source row; public `.copy()` is a detached one-row core honoring today's EXACT
  selective share/deep policy (deep metadata; share func/handles/source_trace/RNG
  snapshots/saved args/params/payloads per the documented policy). Copies never share
  overlays or group-membership mutation state. The subclass contract is JMT-FORK-6.
- **`Trace.fork()`**: core-level COW — share frozen base + pools; new facade cache, new
  overlay, detached backward epochs, independent intervention/history state, payload
  handles per today's fork policy. Transactions checkpoint overlay + epochs; rollback
  discards atomically. The forkcopier (`_trace_intervention.py`) is DELETED in the same
  wave that proves fork/rollback/payload-alias/parent-run/GC parity — a strict
  improvement (-331 lines, -3 CPython private APIs, faster fork), not a give-back.

### 3.6 Persistence (unanimous)

- FIELD_ORDER contents and order do not change this sprint. The only permitted edits are
  C2-precedent PRIVATE-family collapses, each a lockstep declared-schema table diff in
  its own commit (r2 majority; default is zero changes). Public names never leave
  FIELD_ORDER — storage moves, the schema entry stays. Nothing computed enters it.
- Zero artifact format change. No `TLSPEC_VERSION` bump: the manifest vocabulary, field
  order, blob families, and runnable contract are unchanged, so there is nothing for a
  bump to version. New save reads columns/overlays directly and must NOT materialize N
  facade dicts. One-way legacy decoder: old object-shaped artifacts (and old goldens, in
  every wave gate) load into builders. No dual runtime store, no user-selectable storage
  mode; the migration-only shadow path is test-only and deleted inside its wave.
- Columns hold only already-allowlisted types (`_safe_unpickle`'s allowlist is an
  invisible freeze on value types); any unavoidable new class needs the allowlist entry
  AND a round-trip test in the same commit — the plan requires none.
- Artifact identity is semantic: canonical scrubbed state, deterministic manifest bytes,
  named blob/member hashes — never raw ZIP bytes. Plain pickle loads old artifacts and
  round-trips to the same surface/alias behavior; where current deterministic member
  bytes are stable they remain a stronger regression check.

### 3.6b As-landed single-truth boundary (fix round, 2026-08-12)

What the store IS production truth for today, versus which planes exist as
tested substrate awaiting their named follow-on wave — the record must never
read as if the full section-2 target had landed:

PRODUCTION-WIRED (single truth; every live write path routes through it, and
the pickle/state stream serializes FROM it — a `__dict__` shadow of a
declared field can neither shadow the live read nor the serialized value):
`OpRowStore` for every captured `Op` (row-major under 512 rows, columnar
transpose above — object fields stay object columns by design); the M8 kind
tables (`Param`/`Buffer`/`FuncCallLocation`/`ModuleCall`/`Module`); the M9
atomic backward epochs; the M6/M7 relation planes (edge-occurrence CSR,
interned views, group tables, shared-fact blocks); `TraceCore.label_rows`;
the M10 step write-audit (assignment + in-place content).

SUBSTRATE-ONLY (constructed, unit-proven in `test_trace_core_substrate.py`,
NOT yet production-wired — each names its wave): the identity-preserving
payload arena (saved tensors currently live in cells; wiring is part of the
payload-plane slice), the generic `TraceCore.facade()`/`set_facade_factory()`
weak facade cache (facades are currently built eagerly at materialize and
retained strongly by the trace containers; on-demand facade materialization
is the memory-target slice, section 3.7 numbers), the numpy-typed column
packing for scalar planes beyond the transpose, and the `Trace` PHYSICAL
component decomposition (declared ownership map only — `Trace` fields still
live in its `__dict__`).

Memory disposition (fix round, 2026-08-12, measured marginal census on a
602-op Linear/ReLU stack, warmed process): ~112 objects/op and ~30 KB
shallow bytes/op retained. Ownership: the retained capture-event stream is
25.4 obj/op / 3.4 KB/op (a capability decision — it powers post-hoc
backward and refresh); eager facade records are 5.5/op; the remaining ~81
obj/op are per-op builtin containers (≈11 empty lists, ≈16 singleton
lists, dicts of relation/arg metadata) nested in stored cells. The ≤1 KiB /
<10 obj-per-UNINSPECTED-row success metric is therefore owned by the
unlanded slices above (on-demand facades + typed column packing + event
droppability policy), not reachable by spot fixes; no current number should
be read as that metric being met.

Container-cell pooling (grind r1 M14 slice, 2026-08-13): the freeze-seam
compaction now also replaces exact builtin mutable-container CELLS whose
full content is provably immutable AND either empty or repeated 3+ times
with one shared `PooledCell` per distinct content (`op_store.PooledCell`;
allowlisted op fields + every kind-table field; in-store alias census
guard; facade descriptors hydrate a fresh exact-type container on first
read and cache it back — the `_FACT` semantics). Measured on the same
marginal census (model tail included): 602-op Linear/ReLU stack
118.5 → 102.4 obj/op (15.4 → 14.1 KB/op); 244-op CNN 111.7 → 99.4 obj/op.
The remaining core-store tail is eager facades/accessors and per-record
instance dicts — still owned by the on-demand facade slice.

Singleton-label compaction (grind r2 M14 slice, 2026-08-13): per-record
one-element label lists in the kind tables (param
`all_addresses`/`all_module_addresses`/`co_parent_params`, module
`all_addresses`/`call_labels`, module_call `all_addresses`) have distinct
content per row, so the >=3 pooling threshold never reaches them. The same
freeze-seam pass now stores the bare str element and registers it in the
store's `_compacted_singletons` registry; decode is gated on OBJECT
IDENTITY (`registry[key] is cell_value`), so any later user write
self-invalidates the entry with no write-path hooks. Reads hydrate a fresh
one-element list with cache-back (`_FACT` semantics), `items()`/pickle,
fork views, `detach_record`, and the detached fork fallback all decode, and
the same in-store alias census guards aliased lists. Disclosed sliver: an
out-of-contract user write of the EXACT registered str object back into its
own compacted cell reads as the hydrated list rather than the raw str.
Census: linear602 102.5 → 99.1 obj/op (core store 45.5 → 42.1). The grind
r2 AST-cache seal also removed the transformer census elephant
(`postprocess/ast_branches.py` hot/cold split: 563.9 → 131.9 obj/op).

### 3.7 Trace decomposition (THE deliverable) and TraceBuildState

M10 as-landed disposition (2026-08-12): `TraceBuildState` is dissolved — its
transient fields live in three named per-phase workspaces
(`ir/workspaces.py`: `RawGraphWorkspace` / `ModuleCaptureWorkspace` /
`WrapperRuntimeWorkspace`), each owned by exactly one phase family, dropped at
the transient-cleanup seam, with the backend `finalize_forward_session`
protocol taking the raw-graph workspace as its ownership token; the three
dead fields (`grad_fn_strong_refs`, `output_container_specs*`) were deleted
with the dissolution. The 26 step contracts (design-ppdag-v3, 2026-08-12) declare ENFORCED
op-store column write sets AND read sets plus placeholder probes, row
effects, and trace-state tokens (`postprocess/_contracts.py`); the step
order is DERIVED from them — rank-keyed Kahn over edges oriented by the
frozen `LEGACY_STEP_RANK`, semantically pinned by the reason-bearing
`PINNED_ORDER_PAIRS` corpus (import checks R1/R2/K1, test-side K2). The
env-gated zero-cost-when-off audit catches assignment/deletion (write
interception), IN-PLACE container mutation (order-canonical content
fingerprints of mutable-container cells diffed per step window, existing
rows only — row creation is a step's produces contract, whole-row REMOVAL
is checked against the `row_effects` `deletes` sanction, and released-row
cell accesses are row-lifecycle events), and — under
`TORCHLENS_POSTPROCESS_READ_AUDIT=enforce` — undeclared column READS
(recorded and enforced over the axes matrix in
`tests/support/postprocess_axes.py`; `Op.copy()`'s whole-schema clone loop
is a tagged row-clone access kind, not per-column reads). Disclosed audit
residuals: mutables nested in non-builtin custom objects, kind-table
(non-op-store) cells, and the sealed/fork-view silent skip of the class
swap (benign in-pipeline: sealing happens after step 20).
The Trace PHYSICAL component decomposition (components as storage owners
behind 220 property forwards) is the named remaining slice: the declared
per-field ownership map (`_trace_components.py`) partitions every field
(live counts move with re-keys — recompute from `TRACE_FIELD_OWNERSHIP` in
`_trace_components.py`; at 2026-08-16: header 16 / capture_config 50 /
witness 32 / source_metadata 22 / totals 35 / session 61 / runnable 1 /
graph 99, total 316 ownership rows over the 220 forwarded fields — the
graph plane is absorbed by the core, not built as an object), the lockstep ratchet enforces the map, and
the physical move is mechanical against it.

`Trace` (220 fields) decomposes into a presenter over lifecycle-owned components:
`TraceHeader`, `CaptureConfigSnapshot`, the owned `TraceCore` graph,
`WitnessDiagnostics`, `SourceMetadata`, `Totals`, and the existing
`RunnableTraceState` (sizes per plane: the ownership map above is the
authority; the historical approximate counts here drifted twice). All 220 FIELD_ORDER entries keep emitting the same semantic
mapping through component-backed properties. Label/address/lookup maps move to core
indexes; accessors become facade factories. Mixins stay as presentation organization;
internal algorithms move to narrow component/core protocols. `Module` — the second god
object — gets its own responsibility decomposition into narrow query/presentation
helpers without moving public methods. `TraceBuildState` actually goes away: its 20
transient fields dissolve into named per-phase workspaces, each owned by exactly one
postprocess phase and documented; the 26 `POSTPROCESS_STEP_CONTRACTS` (25 plus step "0") become enforced
declared column read/write sets per step. This work runs as a parallel lane from the
compiler wave on and must never be the squeezed phase — a columnar store can still hide
a god junction behind a smaller object.

## 4. Scope boundary

IN: the canonical schema + compiler; the semantic core with all ten repeatable kinds;
relations, groups, ancestry, payload arena; facade/overlay semantics; COW fork +
forkcopier deletion; Trace + Module decomposition; direct semantic I/O with legacy
decoder; deletion of superseded storage/copy/compaction code; docs lockstep; de-bloat.

OUT (documented handoffs, never silent):
- **Journal columnarization** (2-1 HOLD, fable+opus over sol). Reopening condition,
  verbatim: "Reopens only if W0 shows retained journal bytes on a 100K-op capture >2x
  the semantic store AND lane-columnar beats incremental drain on the same measurement."
- **Streaming/bounded-drain lane** (`topology_complete=False` incremental journal
  drain): a separate ship (r2 majority: opus + sol), parallelizable any time after the
  substrate wave; additive capability, never leaves a half-migrated representation. If
  sprint capacity remains after M13 it may be picked up; otherwise it is the first named
  handoff.
- **`Recording`/`RecordingState`/`ActivationRecord`/`GradientRecord` tables**: same
  anti-pattern, sparse product (K rows, not 10^4-10^6); RAM win negligible, surface risk
  real. Mechanical-later handoff — it reuses the same pools/arena, so later
  decomposition is mechanical. Requires its own memory census and public oracle.
- **`RecordContext` restructure**: C1 just froze that contract as the predicate
  interpreter's input schema. M0 measures sol's retention allegation (one context per
  event; `recent_events`/`recent_ops` duplicate windows); if retention is O(events), a
  targeted retention fix that does NOT touch the frozen contract lands mid-sprint; the
  cursor/range restructure waits for the C1 contract to soak.
- **No-journal capture writer** (capture writes semantic columns directly): OUT
  permanently, per section 3.3.
- New physical columnar `.tlspec` (JMT-FORK-3); Arrow core dependency; global interning;
  `torchlens/projection/` package (the 26 declared steps' order is load-bearing; the substance —
  declared per-step read/write sets — lands without the relocation); UI regrouping
  implementation (plan-only, JMT-FORK-2); `intervention_ready` rename (separate
  authorization).

Any capacity cut happens at a completed wave boundary, named plainly — never mid-fan-out,
never a half-migrated family (where a false-VERIFIED hole would hide).

M11 dependency analysis (2026-08-12, named cut): the COW `Trace.fork()` +
forkcopier deletion + strong->weak facade flip are BLOCKED on the Trace
label-map relocation (the M10 physical-decomposition slice). Concretely:
(a) a COW fork's writes must land in a FORK-LOCAL overlay, but the M5 facade
binds `op._core` to the shared `OpRowStore` whose overlay is store-owned —
COW needs a per-trace store VIEW layering a fork overlay over the shared
base; (b) the fork's records must be FRESH facades over that view, but every
trace-side lookup container (`layer_dict_all_keys`, `layer_logs`, accessor
dicts) holds OBJECT references, so fork facades require the lookup maps to
resolve through `TraceCore.label_rows`-style indexes first; (c) the
`TraceCore._facades` identity cache has zero production consumers today —
identity is carried by those same object-holding maps — so the strong->weak
flip is vacuous until the maps relocate. Order of the mechanical-later
handoff: label maps -> core indexes (accessors become facade factories),
then per-trace store views + COW fork + transactions, then forkcopier
deletion with the fork/rollback/payload-alias/parent-run/GC parity proof,
then the weak-valued flip against the aliases-v1 lifetime rows.

M11 as-landed disposition (2026-08-12): dependency (b) resolved WITHOUT the
full trace-side map relocation — the maps' object entries already carry
their row ids, so the fork MATERIALIZES fresh facades by row (two-word
shells over `OpStoreView`s; a record-identity translator remaps every
record reference reached through cells, hydrated fact blocks, and the
policy-driven trace-field pass), which is exactly "resolving through the
core index" with the parent object as carrier. Public map types are
unchanged (`FieldPolicy.KEEP`, surface-v1-pinned); the maps->indexes
relocation proper stays with the named M10 physical-decomposition slice
(the 220-property component move) and no longer blocks anything.
`OpStoreView` supplies dependency (a): sealed-base sharing, per-fork
overlay, base-overlay/row snapshot at fork, copy-on-first-read isolation
for exact builtin containers (tensor/callable identity preserved — the
payload-sharing contract), GroupRef translation to per-fork cloned group
tables, and cell-stored accessor rebuilds. The fork core is the facade
cache's first production consumer, discharging (c): the cache flipped
weak-valued with ONE documented carve-out — `Op` deliberately refuses
weak references (aliases-v1 row 1b pins the refusal), so non-weakref-able
facades fall to a strong side table; op lifetime stays carried by the
trace-side lookup containers either way. The forkcopier (typed deepcopy
engine + per-field object-graph copier) is DELETED; the standalone
compaction passes folded into the freeze seam
(`data_classes/_compaction.py`, invoked next to `_compact_ancestor_sets`
at the core freeze); `TraceCore.transaction()` checkpoints core overlay +
store overlays/sealed rows + backward epochs atomically. Named remainders,
honestly out of the wave: the direct cell-streaming serialization writer
(save/pickle already flow through the explicit per-class semantic state
protocol; bypassing the descriptor layer is a perf increment gated on its
own byte-identity experiment) and the generic state-walker fallback, which
stays for arbitrary nested scrub/rehydrate values — every core record class
now provably defines both explicit protocol hooks (tripwire in
`test_state_adapter.py`), so the generic branch never fires for records.
COW isolation holds in BOTH directions at fork time: the fork builder runs
`OpStoreView.isolate_mutable_cells()` after the record translator installs,
eagerly copying the minimal leak-closure set — exact builtin mutable
containers plus tuples/frozensets transitively nesting one — into the fork
overlay, so a parent's in-place container mutation after the fork is never
visible to the fork, matching the deepcopy fork's snapshot semantics. The
sweep is SPARSE (F4 closure review): it visits only the base store's cached
mutable-cell index (`_build_mutable_key_index`, built at the store's first
fork, kept valid by overlay-routing on columnar stores and by the sealed
row-major class swap `_SealedRowMajorOpRowStore`) plus the view's
base-overlay snapshot; record facades, `GroupRef` cells, and interned
immutable views stay lazy on first read (translation is not mutation).
The fork->parent isolation direction was already pinned and holds
unconditionally. The one shared residual is unchanged: mutables nested in
NON-builtin custom objects are shared by identity, as the shallow fork path
always accepted.
GC parity is preserved exactly (parent Trace collectable while a fork
lives): the MODULE kind table is deliberately NOT viewed — its cells embed
accessor objects whose ``ModuleCall`` members hold a strong trace
reference, so a view would keep the parent reachable — fork modules
duplicate as detached rows with translated cells (they are few); and the
record translator uses weakref guards for id-recycling protection instead
of pinning parent records (only non-weakref-able ``Op`` facades, which
hold no strong trace reference, are pinned).

## 5. JMT forks (reserved decisions; the plan assumes every default)

1. **JMT-FORK-1 — DECIDED 2026-08-12: immutable views.** Relation accessors
   (`child_ops` family: the label-sequence/label-set relation fields) return immutable
   views (`tuple`/`frozenset`), and `equivalent_ops`/`recurrent_ops` become LIVE
   group-membership views (`frozenset`/`tuple` backed by the shared group row): O(1)
   reads vs the old O(N)-per-read copies (O(N^2) over loops). PUBLIC TYPE CHANGE,
   authorized; aliases-v1 rows updated in the same waves (M6 relations / M7 groups).
2. **JMT-FORK-2 — external UI regrouping** (`op.timing.*` etc.): already decided
   PLAN-ONLY by JMT; the plan document ships in M13 (everyday fields top-level;
   specialist facts under timing/autograd/geometry/provenance/storage/control_flow/
   module_context). No implementation.
3. **JMT-FORK-3 — native columnar `.tlspec` physical format**, separately versioned;
   the follow-on where a `TLSPEC_VERSION` bump becomes meaningful. Out of this sprint.
4. **JMT-FORK-4 — public immutability of finished records** (raise instead of warn).
   Default: keep `DirectActivationWriteWarning` + `DIRECT_WRITE_DIRTY`.
5. **JMT-FORK-5 — `layer_list` + sibling list fields as lazy sequence views.**
   Default: materialize a real list on public access.
6. **JMT-FORK-6 — `Op`/`Layer`/`Trace` subclassing contract** — currently unstated;
   support or refuse explicitly. Needs a decision either way.
7. **JMT-FORK-7 — arbitrary user attributes on dict-backed records.**
   Default: preserved via `__slots__` + `"__dict__"`.

## 6. The P0 oracle (merged; union of all three)

`tests/godobject_oracle/`, extending the existing 31-case `tests/capture_oracle/`
machinery — not a disconnected golden system. Scenario matrix: exhaustive / predicate /
dry-run / record-to-trace; conditional; recurrent; in-place; multi-output; pre-hook
mutation; backward and repeated backward; intervention; direct write; fork + rollback;
disk/lazy payloads; halt + failed partial; module aliases; buffers; param aliases;
loaded analysis and runnable traces. Three deterministic byte streams per case:

- **`surface-v1.json`** — per record, enumerated from FIELD_ORDER in declared order,
  NEVER `__dict__`/slots (else the oracle goes vacuously green when a field becomes a
  property): present/missing, normalized value + exact public type, container kind and
  order, tensor shape/stride/dtype/device/`requires_grad`/payload digest, stable
  callable/source tokens with no raw addresses, `repr`/`str`/summary/dataframe rows,
  accessor keys + iteration order + lookup aliases, `dir()` + annotations + descriptor
  kind per name, invalid-access exception class/args/message. Floats via `float.hex()`;
  sets sorted by canonical bytes; dicts as ordered pairs; only documented volatile
  fields masked, each into a named bucket proven against a baseline-vs-baseline run.
  The documented volatile surface is `torchlens.constants.ARTIFACT_VOLATILE_METADATA_FIELDS`
  (metadata.pkl: `random_seed`, capture start/end times, `_phase_timings`, the four
  duration fields, `forward_peak_memory`) plus `ARTIFACT_VOLATILE_MANIFEST_FIELDS`
  (manifest.json: `created_at`, `rng_state_digests`) — empirically derived from a
  same-seed two-process control; masking anything else is a contract widening that
  needs review.
- **`aliases-v1.json`** — the identity/mutation-effect matrix (the load-bearing piece):
  repeated-lookup identity; weakref lifetime while the Trace lives; multi-output shared
  vs output-specific facts; payload object/storage aliases and equal-but-distinct
  tensors; group-default isolation after a per-Op write; every mutable container
  append/delete/assign behavior; `Op.copy()` selective depth field by field; fork/parent
  isolation, rollback, detached backward epoch; direct-write warning count and
  dirty-state transition; source-Trace collection after references drop.
- **`artifacts-v1.json`** — FIELD_ORDER/RecordSchema digests; exact semantic
  `state_items` key/value stream under both storage shapes; deterministic `.tlspec`
  manifest bytes + named member hashes (never raw ZIP bytes); save/load and plain-pickle
  round-trip digests; old-golden artifact loads in EVERY wave gate; cleanup and
  event-stream-forgotten behavior; loaded analysis + runnable trace behavior.

Non-negotiables: baseline-vs-baseline control runs FIRST (an all-empty diff is vacuous);
NO DOT in the cross-process streams — visualization identity is a SEPARATE oracle
(forward/rolled DOT hashes cross-process; backward/combined in-process only, because
grad_fn names embed `id()`s, with its own two-process baseline-vs-baseline control run
first); perf measurements stay outside identity bytes; a real PyCharm + VS Code
expansion/completion session on Trace/Op/Layer/Module/GradFn at M0, the Op cutover, and
final acceptance.

## 7. Wave plan (merged M0-M13)

| wave | content | key gate beyond the standard block |
|---|---|---|
| **M0** | Merged oracle (section 6) over the scenario matrix; refreshed `op_slots_baseline` + deep-reachable-memory scale baseline (1K/10K/100K/1M synthetic + recurrent + branched; capture peak vs finalized steady state; facade-free vs fully-iterated); direct-mutation + alias + escape-path inventory (test-only tripwire descriptors across not-slow — this list IS the overlay-policy and sanctioned-writer spec); real-IDE session; the A2 (journal-bytes) and A3 (RecordContext retention) measurement probes; ancestor-bitset stress synthetic. No production change | baseline-vs-baseline control green |
| **M1** | **Keystone**: extend `RecordFieldPolicy` with `StorageBinding` axes; invert the derivation; assert generated == checked-in literal bidirectionally, all 11 classes | byte-identical, equality-proven |
| **M2** | **Compiler**: generate `__tl_state_items__`/`__tl_state_restore__` (closes the CRITICAL state-adapter hole, 13 consumers, BEFORE any field moves), checked-in annotated descriptors, `to_pandas` row, `.pyi`; regenerate-and-diff CI. Still object-backed | byte-identical |
| **M3** (parallel lanes) | (a) slot dict-backed classes from the table with `__slots__`+`"__dict__"` (laziness measured); (b) `FunctionCallRef`/`ArgTemplateRef` identity-assumption grep (precondition for M7) | byte-identical |
| **M4** | `_trace_core/` substrate, zero consumers: ids, columns, pools, edge-occurrence + CSR, ancestor-closure pool, groups, overlays, payload arena, facade cache. Executable prototypes immediately: `Op.copy` policy, COW fork + rollback, payload identity, parallel-edge order, partial capture, mutable-container hydration, GC lifetime | standalone unit suite |
| **M5** | **The seam**: `_materialize.py` switches to builder ingress; Op facades authoritative (strong cache); scalars columnar, relations staged mutable; per-plane freeze lands; pools absorb `_compact_op_metadata`, bitsets absorb `_compact_ancestor_sets`; kills the `fields_dict` + 39-container transient; partial/no-op/error paths facade-backed. Shadow dual-write parity oracle runs one full CI cycle INSIDE this wave, then is deleted | dual-path parity, deletion proof, scale benchmark, classics spot-check |
| **M6** | Relations family-by-family: parents/children + arg positions -> module membership/stacks -> conditionals -> param uses/aliases; each family compared old-vs-new before its legacy container dies; ancestor closures wired; relation accessors become IMMUTABLE views (JMT-FORK-1 decided) | aliases-v1 green per family |
| **M7** | Groups: FunctionCall/Equivalence/Recurrence/ParamAlias/conditional group blocks + group_id columns; delete `_copy_shared_fields_for_output`; journal-side shared `FunctionCallRef` per call; `equivalent_ops`/`recurrent_ops` become LIVE group-membership views (JMT-FORK-1 decided) | byte-identical |
| **M8** | Layer as aggregate facade over layer->op relations (kills the ~78-field per-pass copy); Module/ModuleCall/Param/Buffer/FuncCallLocation tables + facades; live-handle/lazy-grad/version/release semantics preserved; Module responsibility decomposition; Trace label maps -> core indexes | per-class oracle |
| **M9** | **Backward, last**: GradFn/GradFnCall/BackwardPass tables; atomic backward EPOCHS preserving the exact watermark/revision invalidation; projection + validation consumers migrate in ONE commit; no mixed object/core backward state survives | repeated-backward oracle cases |
| **M10** (parallel from M2) | **Trace decomposition — the deliverable**: 220 fields -> header + owned components; `TraceBuildState` -> named per-phase workspaces; POSTPROCESS_STEP_CONTRACTS become enforced declared read/write sets; C2-style private-family collapses only as lockstep table diffs if needed | <=~60 fields per component |
| **M11** | Cutover + deletions: COW `Trace.fork()` + transactions; exact public `Op.copy()`; direct semantic serialization (byte-identical artifacts; old-golden loads); facade cache flips strong -> weak-valued (per the M0 lifetime evidence); DELETE forkcopier, standalone compaction passes, legacy writers, the generic core-record state walker | artifacts-v1 + old-golden matrix |
| **M12** | Reserved for the streaming lane ONLY if capacity remains (see section 4; the r2 majority ships it separately). Cut-first, by name | additive only |
| **M13** | De-bloat with LOC delta reported first-class; docs lockstep (glossary -> vault, CLAUDE.md, AGENTS.md, notebooks/audit, examples); JMT-FORK-2 UI-regrouping plan doc; full menagerie validation; final real-IDE gate | grep-clean of retired names |

Sequencing rules baked in: keystone/compiler enablers before anything structural; the
ONE seam riskiest-first-at-narrowest (Op has one construction site and the largest
payoff); fan-out by construction-site count; GradFn LAST (the lazy watermark/revision
projection is the most delicate, and its 3 construction sites migrate with their
validation consumers atomically); Trace decomposition parallel from M2, never queued;
each family switches AND deletes its legacy container within its own wave — per-wave
shadow parity oracles are migration tooling and never survive the sprint.

## 8. Gates and success metrics

Per-wave standard block (every commit; `CUDA_VISIBLE_DEVICES=""`,
`PYTHONDONTWRITEBYTECODE=1`, fresh `TORCHLENS_CACHE_DIR`):
`ruff check . --fix` && `mypy torchlens/`; smoke tier; `tests/godobject_oracle/`;
`tests/test_gc.py`; `tests/capture_oracle/`; `benchmarks/op_slots_baseline.py`
(bytes/op + objects/op improve MONOTONICALLY after the M5 cutover).

Entity/serialization boundaries add the field-contract battery
(`test_field_order_contract`, `test_record_field_policy`, `test_schema_audit`,
`test_trace_field_invariant`, `test_field_lifecycle_matrix`, `test_io_scrub_policy`,
`test_to_pandas_field_coverage`, `test_internals`, `test_quantities_memory`,
`test_state_adapter`, `test_io_pickle`, `test_io_plain_pickle`, `test_io_bundle`,
`test_capture_events_parity`, torch parity gates). Wave boundaries: full `-m "not
slow"`. Pre-existing reds bisect against `781a7559` first.

Scale gates: fixed ordinary/recurrent/branched/multi-output models at 1K + 10K ops per
wave; 100K at M5/M9/final; 1M final only. Capture peak AND finalized steady state
reported separately; facade-free AND fully-materialized modes separately; payload bytes
separate from structural metadata. Perf claims by `time.process_time` CPU A/B only,
never cProfile.

Menagerie: deterministic stratified catalog slice after M5/M8/M9/M11; all classics after
the facade cutover; one forced full catalog validation before merge — zero new failure /
timeout / unverifiable / topology-digest / metadata-invariant regressions vs the ledger.

Success metrics (targets set at M0 from measured baselines, not folklore): <=1 KiB
deep-retained structural metadata per uninspected op row (target: hundreds of bytes);
<10 retained structural Python objects per uninspected row (target ~0 beyond shared
tables); hand-written schema declarations -> 0; <=~60 fields per owned Trace component;
net LOC DOWN, reported first-class; no new module >~1500 lines handwritten (cohesive
generated declarations excepted, split per entity — no generated 5K-line god file).

**Tripwire, LOCKED and unanimous:** a validation/invariant/oracle failure during this
migration is a bug in the migration. No tolerance widens, no check is exempted, no
honesty verdict weakens to make a wave green. The direct-write warning semantics are
preserved byte-identically through the overlay path.
