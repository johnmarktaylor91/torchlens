# Buffer PLAN v5 — BUILD spec (option-2, both labs validated the approach)

This is the implementation blueprint. The capture-at-the-moment approach (PLAN_v4) was
EMPIRICALLY VALIDATED by both adversarial labs (`PLAN4_REVIEW_{codex,claude}.md`): the
fused replay-validation is non-vacuous and uses the EXISTING replay machinery; the
reassignment hook fires for top-level recurrent loops; hot-path cost is negligible. v5 folds
in every blocking + major fix the reviews found. Replay promise already fixed (P1, 39a5029).
Validation = the gate; tripwire SACRED (no self-feed, no exemption, no weakened check).

## Proven foundations (do not re-litigate; build on these)
- **Fused validation works via existing replay:** `validation/core.py::_execute_func_with_
  restored_state` seeds the buffer from the pre-write node, re-runs the fused kernel, and
  compares — corrupting the captured value FAILS the check (Claude-proved). REUSE this path.
- **Reassignment is catchable** for top-level recurrent loops via class-level `__setattr__`
  patching (Codex+Claude proved 4 events for `for _ in range(4): self.h=tanh(...)`).
- **Fused writes catchable** by post-op value snapshot (value-compare, NOT `_version`).
- Hot-path snapshot cost is negligible (~518us/53 resnet50 BN snapshots).

## The two capture hooks (with the review fixes baked in)

### A. Reassignment hook = SCOPED CLASS patch (NOT instance, NOT __class__-swap)
- Python dunders are type-looked-up, so an instance attribute never fires (review B1). DO
  NOT use `__class__`-swap either (changes `type(self).__name__` mid-forward, shadows user
  `__setattr__` — Claude). INSTEAD: during active capture, patch each PREPARED module's
  CONCRETE CLASS `__setattr__`, gate every event by `self in prepared_weakset`, keep a
  per-class refcount/stack for nested/reentrant traces, restore in the capture `finally`.
  (Codex proved this fires + keeps exact type.)
- **DE-DUP the double-count (Claude B2):** `nn.Module.__setattr__` internally does
  `self._buffers[name]=value`, so a single `self.h=t` would fire the setattr hook AND a
  `_buffers`-dict proxy -> 8 events for 4 writes. FIX: hook the class `__setattr__` ONLY for
  the registered-buffer path; do NOT also proxy `_buffers` for normal setattr. Detect DIRECT
  `self._buffers[name]=t` (which bypasses `__setattr__`) separately and reconcile so it is
  not double-counted with a setattr in the same call.
- Record event `(address, producer = the new tensor's op label, value, kind="reassign")`.

### B. In-place / fused hook = post-op value snapshot in the op wrapper
- For any wrapped op whose tensor args reach a buffer-tagged STORAGE (resolve by storage so
  view/slice/`.data.copy_`/`__setitem__` writes attribute to the buffer — proven caught):
  cheap path = `_version` bump => written; ALWAYS value-snapshot for a known fused-mutator
  list (`batch_norm`, `instance_norm`, ...) since `_version` doesn't bump. If changed (or the
  fused op ran in train/update mode), record `(address, producer=this op, value=post snapshot,
  kind="inplace"|"fused", value_changed: bool)`.
- **Storage-index robustness (Claude major):** key by storage identity but guard against
  `data_ptr` REUSE — validate the buffer object is still the registered one (object identity +
  storage), not just `data_ptr` equality. Overlapping/aliased registered buffers -> attribute
  by the registered-buffer object whose storage range covers the write.
- Fused multi-buffer (BatchNorm = 3): check all buffer args, deterministic order.

## Event semantics (resolve review B4 / Codex 4)
- **Event-history for explicit (reassign/in-place):** each write event = a version;
  `num_overwrites` = count. No dedup by value.
- **Fused/native: emit a write event whenever a known fused mutator runs in train/update
  mode, per mutated buffer arg, carrying `value_changed`.** (Can't prove an idempotent native
  kernel physically wrote, only the state transition — so the event is op-execution-based +
  a flag.) Documented asymmetry; `num_overwrites` still counts these events.

## Version nodes + entity
- Each write event -> identity buffer-version node (plain `Op`+`is_buffer`): parent =
  producer, `out` = recorded value, children = readers of that version; re-link readers to
  read vN; chain v1(initial)->...->vK per address.
- `Buffer` entity (Module/Param-sibling) = projection: address/name, ownership, shape/dtype/
  memory, `initial_value`, `final_value` (end-of-pass snapshot), `versions`, `num_overwrites`,
  `value_at/after`, usage. Accessors: `trace.buffers`, `trace["addr:N"]`, `buffer.versions`,
  `Module.buffers`. `compute_ops` excludes `is_buffer`. Retire `Buffer(Op)`; name -> entity.

## Validation (the gate)
- Explicit/reassign version nodes: value == producer output -> identity replay validates.
- Fused version nodes: validate via `_execute_func_with_restored_state` (seed pre-write,
  re-run kernel, compare post). REQUIRES `save_arg_values=True` (review B3) -> full fused
  version validation is gated on it; in plain capture, fused versions carry the snapshot value
  but are marked not-replay-validated (documented; the STATE TRANSITION still holds). Multi-
  pass fused: the version chain supplies each pass's pre-state (review B3 second half).
- **Silent fallthrough fix (review G/4):** `validation/core.py:~1000` `else: parent_values =
  parent.out` -> convert to a RAISE for buffer-version parents BY SOURCE-EQUALITY, and ensure
  it does NOT fire for valid fused nodes (which legitimately use the restored-state path).

## Carve-outs / majors
- `num_batches_tracked`: it's orphan-pruned today and v4's "it's a real op" premise is FALSE
  (Claude). Decide: either include it as a first-class buffer entity with its `+=1` write
  events (un-prune it), OR exclude integer/counter buffers from the version model with a clear
  rule — but NEVER an entity with `final_value` set and `versions==[]`. Pick include-with-
  events; add an assertion that entity/accessor/`buffer_layers` agree.
- Loop detection + recurrent version nodes: DECIDE grouping (do recurrent version nodes
  collapse like ops in rolled view?) with a test; not deferred-silently.
- `.data = new_tensor` setter: bypasses both hooks. DOCUMENT unsupported + add an
  end-of-capture RECONCILIATION diagnostic (compare each registered buffer's final object/
  storage/value vs the journal; an unjournaled change -> raise/warn loudly). Never silently
  log the post-write value as the initial buffer.
- Intermediate fused write never read then overwritten: genuinely lost, computationally inert
  (irrelevant to model function/replay). Documented.

## Phasing (branch feat/buffer-datamodel; each phase: commit + EVERY stress model still
`validate_forward_pass` True + tier2 green; bisect any regression)
- P5a: storage->buffer index + the two hooks (scoped-class-patch setattr w/ de-dup; op-wrapper
  value-snapshot) recording write events; end-of-capture reconciliation diagnostic. Verify
  events fire for all patterns; replay still True.
- P5b: version-node synthesis from events + reader re-link + the fused restored-state
  validation path + the source-equality fallthrough->raise. Validate all patterns.
- P5c: `Buffer` entity + accessors + targeted fixes (copy_ source edge, alias discovery,
  num_batches_tracked carve-out); retire `Buffer(Op)`; loop-detection grouping decision;
  re-measure & lock dual-label; docs-lockstep (glossary, CLAUDE/AGENTS, examples, notebooks).
- P5d: exhaustive stress tests + ruff+mypy+smoke+tier2.

## Documentation deliverable (REQUIRED — JMT: "this is prob the most niche plumbing in the package")
Write `docs/buffers.md` — a standalone narrative explainer for devs + curious users (match
the tone/structure of `docs/intervention_api.md` / `docs/visibility.md`). Cover, plainly:
- What a registered buffer is, and TorchLens's model: ONE node per buffer version in the graph
  (`producer -> buffer -> reader`), read/write are EDGES not separate nodes; the persistent
  `Buffer` entity (Module/Param-sibling) vs the per-version graph nodes.
- The version model: `initial_value`, each write = a new version, `final_value`,
  `num_overwrites`, `versions`, `value_at/after`; worked example (a recurrent state buffer +
  a BatchNorm running-mean) showing the version chain.
- HOW writes are captured (the three kinds, in user terms): reassignment (`self.h = ...`),
  explicit in-place (`buf.mul_/add_/copy_`, `buf[...] =`, `buf.data.copy_`), and FUSED/native
  (BatchNorm/InstanceNorm running stats — caught by value-snapshot since `_version` doesn't
  bump). One paragraph each, no internal jargon dump but enough that a dev gets WHY.
- How buffers interact with validation/replay (they replay; fused versions validated by
  re-running the kernel under restored state).
- The documented LIMITATIONS, stated honestly: `.data = new_tensor` reassignment is
  unsupported (+ the reconciliation diagnostic that warns); an intermediate fused write that
  is never read and then overwritten can't be displayed (computationally inert — zero effect
  on the model's output or replay, purely an introspection gap); non-registered Python-attr
  "state" is out of scope.
- Accessors cheat-sheet: `trace.buffers`, `trace["addr:N"]`, `buffer.versions`, `Module.buffers`.
Cross-link: add a "Buffers" line to `docs/LIMITATIONS.md` (the residuals) and a pointer from
the README/feature list + the glossary's Buffer entry. Link `docs/buffers.md` from any docs
index/TOC that lists the other `docs/*.md` explainers.

## Stress models (tests MUST cover, all validate True)
recurrent top-level reassignment; BatchNorm1d/2d train (fused 3-buffer); InstanceNorm;
in-place `mul_`/`add_`/`copy_`; multi-overwrite in one call; same buffer in two loops;
view/slice/`.data.copy_`/`__setitem__` buffer writes; `self._buffers[n]=t` direct;
`self.b.data=t` (reconciliation diagnostic fires); shared/overlapping-alias buffers;
`num_batches_tracked`; in-place dual-role; static read-only buffer; double-count check
(N reassignments -> exactly N events).
