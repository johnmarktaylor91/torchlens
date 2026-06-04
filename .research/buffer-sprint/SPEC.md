# Buffer Sprint — Spec (v1, 2026-06-04)

Authoritative design for the Buffer refactor. Derived from `.project-context/buffer_refactor_proposal.md`
(Option B) **as revised by the 2026-06-04 design conversation with JMT**, which changed
several core decisions. Where this spec and the old proposal disagree, THIS SPEC WINS.

## Goal

Promote `Buffer` from a `Buffer(Op)` subclass (which mixes graph-node identity with
PyTorch-state identity on one record) to a **first-class persistent entity** (sibling of
`Module`/`Param`), and model buffer graph events as **one node per version** (NOT the old
proposal's buffer-source/buffer-sink two-op split). Every common buffer question should be
answerable in ONE hop from whatever record the user holds.

## The model — two pieces

1. **Persistent `Buffer` entity** — one per buffer **address** (e.g. `bn.running_mean`).
   NOT an Op. Holds all info about that buffer: identity, ownership, tensor properties,
   full version/overwrite history, usage. This is the noun.
2. **Buffer graph nodes = plain `Op`s** (`is_buffer=True`), **one node per version**, living
   in the **main `parents`/`children` graph**, each linking back to its entity via `.buffer`.

## Locked decisions (from the 2026-06-04 design conversation)

- **Node = plain `Op` + `is_buffer` flag.** NO `BufferOp`/`BufferVersion` subclass. Every
  other op-role (`is_input`/`is_output`/`is_internal_source`/`is_internal_sink`/
  `is_terminal_bool`) is a flag, not a subclass — buffer matches that pattern. Field clutter
  on non-buffer ops is handled by a **role-gated repr** (show buffer fields only when
  `is_buffer`), not a subclass. Retire the `Buffer(Op)` subclass; the name `Buffer` becomes
  the entity.
- **One node per version.** A buffer node IS op-like: a sourceless read (initial value) is a
  pure **source node like `torch.ones`**; a written version is an **identity passthrough**
  (`func = identity`, one tensor in -> out). This is ALREADY how `_fix_buffer_layers` (Step 6)
  treats buffers.
- **Edges are direction-only (no edge typing).** **parent = the write** (op that produced
  this version), **children = the reads**. "A child read from it; a parent wrote to it."
- **Node exists only if the buffer is read** (like `torch.ones` — only a node if it enters
  the graph). A never-read buffer gets no node.
- **A read-only buffer (one version) read N times in a loop = ONE node with N children.**
  The reads are out-edges; the N-ness lives in the consuming ops, not the buffer.
- **Each overwrite = a new version-node.** Buffers STAY in loop detection: grouping follows
  op rules (recurrence -> Layers). A buffer overwritten each loop iteration -> version-nodes
  group into a buffer-Layer (rolled view = 1 node, unrolled = N). A buffer rewritten in TWO
  loops -> TWO buffer-Layers, one persistent `Buffer`.
- **Dual labels per node** (exactly like an op inside a reused module, which has both an
  op-label and a module-call label):
  - **op-label**: `buffer_<type_index>:<pass>` — per-Layer pass (op rule, resets per Layer).
  - **address-label**: `<address>:<version>` — entity-GLOBAL version (Module rule;
    `ModuleCall`-parallel; runs continuously across the buffer's Layers).
  - Worked example — a buffer in two loops, 3 passes each, overwritten per pass:
    ```
    op-label      address-label
    buffer_1:1    bn.running_mean:1   (= initial value)
    buffer_1:2    bn.running_mean:2
    buffer_1:3    bn.running_mean:3
    buffer_2:1    bn.running_mean:4
    buffer_2:2    bn.running_mean:5
    buffer_2:3    bn.running_mean:6
    ```
  - This mirrors `ModuleCall`: a module called across two loops is `relu:1,:2,:3` then
    `relu:4,:5,:6` (global call index, non-consecutive within the 2nd Layer). Same here.
- **`:1` = the initial value** (1-based; first state to enter the graph). Overwrites are
  `:2, :3, …`. No deviation from existing 1-based pass indexing.
- **`[address]:N` = the flat global version** (lookup sugar; equals `buffer.versions[N-1]`).
  Returns the version-node **`Op`** (not a wrapper — the version IS an Op). Justified
  asymmetry with `Module`->`ModuleCall`: a module CALL is a composite spanning many ops (needs
  a container class); a buffer VERSION is atomic (one node) -> the Op is the record.
- **Address is foregrounded everywhere** — str, repr, visual, summaries. The address-label
  is in the display by construction, so it's always obvious that `running_mean:1` and
  `running_mean:4` are the same buffer even across loops.
- **Save:** `layers_to_save="all"` INCLUDES buffers. Since each version is a node, saving the
  version-nodes gives the full overwrite history for free. **No separate `save_buffer_history`
  flag.** Selective saving applies the normal rules to buffer nodes.
- **DROP from the old proposal:** the buffer-source/buffer-sink two-op split; the parallel
  state-flow-edge layer (§B) — the one-node-in-main-graph model subsumes both; `*_str`
  companion fields (use `tl.Bytes` per v7 unit-type family).

## Persistent `Buffer` entity — API (Param/Module-parallel)

Sibling of `Module`/`Param`. NOT an Op. Reference-form per glossary Principles 1-4 (label
storage + `@property` resolvers for common cross-class refs).

**Identity:**
- `address`: primary dotted buffer address. `name` (`@property`): last segment.
- `all_addresses`, `has_multiple_addresses`: shared-tensor aliases.
- `ordinal_index`: 0-based position in `trace.buffers`.
- `trace`: runtime back-pointer (weakref; not portable).

**Tensor properties:**
- `shape`, `dtype`, `memory` (`tl.Bytes`).
- `initial_value`, `final_value` (saved tensors when captured), `current_value`
  (runtime-only live ref when still attached).

**Status:**
- `is_static` (`@property`): never overwritten during the trace. `is_overwritten`: `not is_static`.
- `has_grad`: usually False (buffers default `requires_grad=False`); kept for Param symmetry.

**Module ownership (Param-parallel):**
- `module_address` (stored) + `module` (`@property` resolver). `module_name`, `module_cls`
  (`@property`). `all_module_addresses` (shared buffers; no plural `modules` resolver per
  Principle 4).

**Lifecycle / versions (the NEW affordance):**
- `num_overwrites`, `num_versions`.
- `versions`: accessor -> the version-node `Op`s, **flat, execution-ordered** (0-based int
  index; `versions[i]` = the (i+1)-th version). `versions[N-1]` == `trace["address:N"]`.
- `layers`: the buffer-`Layer`(s) this buffer spans (>1 only in the rare multi-loop case).
  This is the field that says "I'm spread across N positions."
- `reads`: read events (consuming ops, or the read out-edges). `writes`: the version-producing
  ops (the in-edges / mutators).
- `value_at(op_label)`: buffer value as SEEN by that op (before it ran). `value_after(op_label)`:
  value after (matters for in-place / write ops; collapses to the same for pure reads).

**Usage (Param-parallel):**
- `num_uses`, `used_by_layers`, `co_used_buffers`.

**Methods:** `release_buffer_ref()` (Param-parallel), `to_pandas()/to_csv()/to_parquet()/to_json()`.

## Buffer node fields (on `Op`, gated by `is_buffer`)

- `is_buffer` (flag), `buffer_address` (str), `buffer` (`@property` -> entity via
  `trace.buffers[buffer_address]`).
- `buffer_version` (int, global 1-based — the address-label colon).
- `is_buffer_initial` (`@property`): `buffer_version == 1` and no writer parent.
- `is_in_place` (`@property`): the node both reads and writes the same buffer in one call.
  NOTE: in the one-node model an in-place MUTATION (`running_mean.mul_(0.9)`) is captured as
  the COMPUTE op (`mul_`) reading version N (in-edge) and writing version N+1 (out-edge to the
  new buffer node); the *buffer nodes* are states, the *mutator op* carries the duality on its
  edges. `is_in_place` flags that mutator. VERIFY this matches capture (see Risks).
- Written versions: `func = identity` (passthrough). Initial versions: source node, no parent.
- Standard Op fields (`out` = the buffer tensor at that version, `parents`, `children`,
  `shape`, `dtype`, `activation_memory`, label fields) all apply.

## Accessors / filtering on `Trace`

- `trace.buffers` -> `BufferAccessor` returning **persistent `Buffer`** entities, keyed by
  address. `trace.buffers["bn.running_mean"]` -> ONE `Buffer` (unambiguous).
- `trace["bn.running_mean:4"]` -> the version-node `Op` (sugar over `buffer.versions[3]`).
- `trace["buffer_2:1"]` -> the same node via op-label.
- `trace.compute_ops` continues to EXCLUDE `is_buffer` ops.
- `Module.buffers` -> persistent `Buffer`s owned by the module (Param-parallel; RESOLVES the
  long-deferred `Module.buffer_layers` item — that field is dropped).
- `Trace.num_buffers` etc. as natural counts.

## Loop detection

Buffers participate (grouping = op rules):
- Read-only buffer in a loop -> ONE node, many children (consumers carry the passes).
- Overwritten-each-iteration buffer -> version-nodes group into a buffer-`Layer`; passes =
  versions-within-that-position.
- Multi-loop overwrite -> multiple buffer-`Layer`s, global version continuous; one entity.
- RISK to verify: version-node chains must NOT create spurious cycles (versions are
  time-ordered -> the chain is acyclic for non-recurrent buffers; recurrent buffers group
  exactly like recurrent ops). Test explicitly.

## What's already built (sprint is largely ADDITIVE)

`postprocess/control_flow.py::_fix_buffer_layers` (Step 6) already: connects each buffer to
its writer via `buffer_source` (= the in-edge/write), makes a written buffer `func=identity`
(passthrough), creates a node only when the buffer is read, dedups same-value appearances,
sequences `buffer_pass` per address. So the write-detection, one-node passthrough, and
main-graph placement EXIST. The sprint ADDS: the persistent `Buffer` entity + its API, the
dual-label (address-global-version), `Buffer`-as-non-Op restructure, accessors, and the
clean version semantics — it does NOT build capture from scratch.

## Migration (CLEAN BREAK — no shims, per JMT's standing directive)

- Remove the `Buffer(Op)` subclass; `Buffer` name -> persistent entity; nodes -> plain
  `Op` + `is_buffer`.
- Update `constants.py` FIELD_ORDER: new `Buffer` entity field set; new Op buffer fields;
  drop dropped fields.
- DOCS-LOCKSTEP (mandatory, same change): the glossary (canonical:
  `.research/glossary_v9_working.md` + vault), `CLAUDE.md`/`AGENTS.md`, `examples/`, audit
  notebooks. `grep` old `Buffer(Op)` / `BufferLog` names clean across `torchlens/`, `tests/`,
  `examples/`, `notebooks/`, glossary.
- Add a **"How buffers work"** explainer (glossary section + a worked example in
  docs/notebook) with the entity/version/Layer picture and the dual-label table above. NAME
  the rolled-view multi-axis case even though its full answer is deferred.

## Deferred (todo'd, NOT in this sprint)

- Rolled-view loop-vs-module collision (general, not buffer-specific).
- Broader Op-subclassing refactor (for completeness only).
- Loop-indexing / recurrence-semantics revisit (leaner alternative).

## Validation & tests (programmatic; the real gate)

New `tests/.../test_buffer_model.py` (or nearest convention). Cover:
- BatchNorm in train (read + overwrite running stats) — versions, value_at/after.
- Static buffer (read-only) — one node, many children, `is_static`.
- Recurrent buffer overwritten in a loop — buffer-Layer, passes = versions, rolled/unrolled.
- Multi-loop buffer — TWO Layers, ONE entity, global version 1..N continuous, `buffer.layers`.
- In-place op (`x.mul_(0.9)`) — `is_in_place`, reads vN writes vN+1.
- Dual labels resolve: `trace["addr:N"]` == `buffer.versions[N-1]` == `trace["buffer_k:p"]`.
- `value_at` / `value_before` / `value_after` correctness incl. the in-place tie-break.
- Save: `layers_to_save="all"` captures buffer version values (history); selective excludes.
- `compute_ops` excludes buffers; `Module.buffers` returns entities.
- Address foregrounded in `repr`/`str`/`summary`; role-gated repr hides buffer fields on
  non-buffer ops.
- VALIDATION INVARIANTS (tripwire): buffer nodes must not break forward replay / metadata
  invariants / loop-detection; NO spurious cycles. Never weaken a check to pass — root-cause.
- `pytest -m smoke` + tier-2 green; `ruff`/`mypy` clean.

## Open implementation risks (verify during build, don't pre-decide)

1. In-place capture (`mul_`/`add_`) producing the read-vN / write-v(N+1) edges cleanly.
2. Loop detection chaining version-nodes without spurious cycles, esp. recurrent overwrite.
3. The address-foregrounded repr/visual + role-gated field display.
4. `value_at`/`value_after` resolution across multi-version / multi-loop.

## Scope estimate

The old proposal said 1-2 weeks. Detection being largely built lowers it, but the
entity + dual-label + restructure + docs-lockstep + tests is still a substantial,
multi-file sprint. Phase it: (1) entity + node fields + restructure; (2) dual-label +
accessors; (3) save/value_at; (4) viz/repr; (5) docs-lockstep + tests.
