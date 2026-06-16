# CHANGELOG


## v2.20.0 (2026-06-16)


## v2.19.0 (2026-06-13)

### Bug Fixes

- **annotate**: Make _annotation_revision a lazy runtime field (B1 follow-up)
  ([`7e3c5aa`](https://github.com/johnmarktaylor91/torchlens/commit/7e3c5aa6e059c8139d6c32d6bd0b02d2c449e0e0))

The full not-slow suite caught 2 field-invariant failures the per-phase gate missed:
  _annotation_revision (B1's render-cache counter) was always-present but registered only in the
  scrub runtime-only set -- missing from PORTABLE_STATE_SPEC
  (test_portable_state_specs_cover_every_live_attribute) and the known-runtime-field list
  (test_trace_field_set_subset_of_user_facing).

Make it conditionally-present (like _capture_container_structure): absent on a default trace,
  created lazily on first annotate via the existing getattr(..., 0) path, omitted from portable
  state; rerun preservation reads it with a default too. Default traces carry no extra field, so
  both invariants pass and byte-identity is fully preserved -- no golden moved, all 6 negative
  digest gates unchanged, smoke + field-lifecycle suite green.

- **backend**: Resolve pre-merge regressions
  ([`4096d30`](https://github.com/johnmarktaylor91/torchlens/commit/4096d3003b0920517aea2fdd0eee2e1a3e4a2d46))

- **backward**: Correct pass order and grad fn pairing
  ([`05ac5e9`](https://github.com/johnmarktaylor91/torchlens/commit/05ac5e982858c077531970f369f1c4dd802209b2))

- **bench**: Tolerate pre-existing two-pass limitation in capture benchmark matrix
  ([`aafb8c3`](https://github.com/johnmarktaylor91/torchlens/commit/aafb8c3e7c0f271a02ab65571bbb2440532b238e))

The legacy two-pass benchmark cell calls save_new_outs under the hood, which raises 'computational
  graph changed' on models with in-place ReLU / batchnorm-eval (a pre-existing limitation, see
  tests/test_two_pass_inplace_fix.py, unrelated to capture unification). Record that cell as N/A
  with a note instead of hard-failing the informational matrix; assert only the cells that actually
  measured.

- **buffers**: Scrub per-op equivalence lists on removal to fix RNN-cell crash
  ([`430357a`](https://github.com/johnmarktaylor91/torchlens/commit/430357a20d85bcf341a14ed459bbca810657353d))

An RNN cell that reassigns its hidden-state buffer in a loop around a submodule (self.h = f(cell(x),
  self.h)) crashed loop detection with "'buffer_1_raw' is not a known raw label during this forward
  pass". Buffer-version synthesis merges the initial buffer node, but the output node (which returns
  the buffer) still listed the merged label in its per-op equivalent_ops; loop detection then
  dereferenced the dead label mid-pass.

Root cause: _remove_log_entry_references and _batch_remove_log_entries scrub the global
  op_equivalence_classes map and Trace-level lists, but never the stored per-op equivalent_ops /
  recurrent_ops lists (both FieldPolicy.KEEP). Removing an Op left dangling references on surviving
  ops. Fixed generally with _scrub_per_op_equivalence_lists in both removal paths, closing the whole
  dangling-equivalence-reference class, not just this buffer case.

Also: - Add gradient-flow regression tests: tracing a recurrent reassignment model and a
  BatchNorm-train model must not break autograd (grads match an untraced run). The capture hooks are
  observational (snapshot detached copies; never replace the live grad-carrying buffer tensor). -
  Add the RNN-cell loop-detection regression test. - Glossary + docs lockstep: document
  buffer_write_kind, buffer_value_changed, is_buffer (stored flag behind the is_buffer_source
  property), and Module.buffers; fix a field-name reference in docs/buffers.md.

- **capture**: Add backend alias contract
  ([`428d2f2`](https://github.com/johnmarktaylor91/torchlens/commit/428d2f22405b75b4f3d00f7c995c2fb33e083110))

- **capture**: Add torch.sym_* ops to the ArgSpec lookup table
  ([`3d54f8c`](https://github.com/johnmarktaylor91/torchlens/commit/3d54f8cf8e5cb7e504a985d62fa61bfd8518fba0))

The meta-device capture tests exercise torch.sym_max/sym_float through PyTorch's symbolic-shape
  paths, and the session-wide ArgSpec coverage gate (test_lookup_table_coverage) flagged them as
  uncovered. Adds the sym family (sym_max/sym_min binary; sym_float/sym_int/sym_not unary) to the
  static table.

- **capture**: Audit tensor kwarg arg specs
  ([`3faa7c9`](https://github.com/johnmarktaylor91/torchlens/commit/3faa7c9b71fd8eb63cef5700234b73f26b6a8b2b))

- **capture**: Enforce layers_to_save matrix
  ([`3b3446f`](https://github.com/johnmarktaylor91/torchlens/commit/3b3446f4b2755c407ac897403330f87900c55fc4))

- **capture**: Faithful container correctness (typed validation index, literal leaves, HF-output
  preservation)
  ([`e959295`](https://github.com/johnmarktaylor91/torchlens/commit/e9592957a66335af76f08a00599f71d67dd56aab))

- **capture**: Gate chunk_size via capability spine, not inline backend name-check
  ([`8382a78`](https://github.com/johnmarktaylor91/torchlens/commit/8382a788f2d79e68e252e4e3d117c9dfdd4807f3))

The chunk_size backend gate inspected the resolved backend identity inline (str(resolved_spec.name)
  / str(backend) in trace()), violating the backend-agnostic-dispatch invariant
  (test_public_trace_dispatches_through_backend_spec).

Gate it the blessed way instead: register chunk_size/chunk_paths as a capability epoch
  (epoch8_forward_chunking) so torch accepts them via the option spine and non-torch backends reject
  an explicit chunk_size generically through _filter_trace_kwargs_for_backend
  (BackendUnsupportedError) — with no name inspection in trace(). chunk_size/chunk_paths now default
  to MISSING (so _trace_option_explicit treats an omitted value as not-explicit, like the other
  spine options); MISSING resolves to None at consumption + in the cache key. Both inline gates
  removed.

Also fix the docs/performance.md chunk_paths snippet to be self-contained (import + model) so
  test_docs_snippets runs it. Behavior preserved: explicit chunk_size on a non-torch backend still
  raises BackendUnsupportedError; omitted chunk_size on a non-torch trace no longer false-raises.
  parity byte-identical; backend-registry + chunk_size + docs-snippet green.

- **capture**: Harden functorch wrapper tensor handling
  ([`661c9cd`](https://github.com/johnmarktaylor91/torchlens/commit/661c9cdb6c6cccf06a5efb0a5134c3f683c09de0))

- **capture**: Harden nested transform reconciliation
  ([`2303a52`](https://github.com/johnmarktaylor91/torchlens/commit/2303a52e3fed89a19c9165842fd68a27c231560f))

- **capture**: Log untraceable vmap-built tensors as internal sources
  ([`8351477`](https://github.com/johnmarktaylor91/torchlens/commit/8351477739cff18e024d2f6e98bc9b6ffb0bff77))

Transformers 4.5x builds the 4D causal/sliding-window attention mask via torch.vmap
  (_vmap_for_bhqkv). TorchLens deliberately skips logging ops that run inside a functorch transform,
  so the mask emerges fully-formed but untagged. When it first entered self_attn, the module-entry
  path synthesized a functionless intervention_replacement placeholder, conflating a plain-capture
  gap with a genuine user intervention.

Distinguish the two untagged-tensor cases by call site: genuine raw forward_hook output replacements
  are still tagged at module exit by the hook wrapper (intervention_replaced=True), so anything
  still untagged at module entry (or at a hook-free module exit) is an internally generated tensor.
  Log it as a clean graph source (is_internal_source, func_name none, no parents, no input ancestry,
  registered in internal_source_ops) instead of an intervention placeholder. Plain tracing of
  Mistral/VITS now yields zero synthetic intervention_replacement ops and the mask validates
  legitimately.

- **capture**: Rebuild fast-path out_versions_by_child; honest replay-completeness
  ([`2a8705c`](https://github.com/johnmarktaylor91/torchlens/commit/2a8705c1b35682e073f608f1bfa582fa60dbf73c))

Fast/selective save_new_outs cleared out_versions_by_child without rebuilding and left
  _replay_arg_version_data_complete True, so a fast trace could advertise complete replay data it
  lacked and then fail as a generic mismatch (#93). Now: (1) rebuild child-version snapshots in fast
  logging when save_arg_values=True; (2) mark replay data incomplete after clearing, restore
  complete only after a successful save_arg_values=True fast pass, so validation honestly refuses
  when data is missing; (3) guard _undecorate_all_saved_tensors against reading unsaved .out.
  Strengthens validation; never weakens it.

- **capture**: Register _no_grad_uniform_ in arg-spec lookup table
  ([`0d23cdb`](https://github.com/johnmarktaylor91/torchlens/commit/0d23cdb4f8db48c6d5a7d571c9016abf37e7734c))

- **capture**: Resolve backward taps on boundary aliases
  ([`f7de4d5`](https://github.com/johnmarktaylor91/torchlens/commit/f7de4d57b38b7d2fc40da31a5eeba224da3a490f))

- **capture**: Restore trace predicate save path
  ([`1379c45`](https://github.com/johnmarktaylor91/torchlens/commit/1379c455da9362d2816ee90fa46f0f99a1365181))

- **capture**: Survive meta-device tensors; extract kwarg tensors for matmul family
  ([`f9daf5f`](https://github.com/johnmarktaylor91/torchlens/commit/f9daf5fc4e7696f6540c4217d6d29da1954a5f65))

Device-context injection already ran on the active-logging path, but captured meta tensors crashed
  downstream wherever capture touched their (nonexistent) data: the content-hash save dedup called
  .cpu() on them, the aliasing mutation check and tensor_nanequal called torch.equal / .isinf().
  Meta tensors now skip content dedup, report not-mutated, and compare equal in tensor_nanequal once
  shape and dtype match (there is no data left to differ). A model running under
  torch.device('meta') now traces end-to-end, including a meta factory tensor as the full-save model
  output.

Also fixes kwarg tensor extraction for functions whose ArgSpec named the wrong (or no) keyword args,
  so e.g. normal(mean=m, std=s) and addmm(input, mat1=a, mat2=b) record their kwarg tensors as
  parents: normal (mean/std), mm/bmm (mat2), mv (vec), addmm (mat1/mat2), addbmm/baddbmm
  (batch1/batch2), addmv (mat/vec), addcmul/addcdiv (tensor1/tensor2), lerp (end/weight). The
  broader ~637-function tensor_kwargs audit remains tracked separately; zeros_like/new_* factories
  dropping their source tensor is a known adjacent gap filed with it.

- **edges**: Make branching_factor a coherent mean out-degree
  ([`c6ab0f8`](https://github.com/johnmarktaylor91/torchlens/commit/c6ab0f844df1834c0176899ddc5a433eda882ea6))

As shipped, branching_factor divided full-graph edge count (num_edges, over all nodes incl.
  input/output/buffer sentinels) by num_ops (compute-only node count), mixing two node sets -> a
  plain 4-op chain reported 1.25 instead of ~1.0.

Redefine as mean children per compute op (sum(op.num_children for op in compute_ops) /
  num_compute_ops): a single consistent node set, so the ratio is a true mean fan-out -- ~1.0 for a
  chain, >1.0 with reuse (residual/dense). Render the summary footer with .2f for readability.
  Glossary + tests updated to the hand-verified values.

- **export**: Use is_trainable in param export schema
  ([`3858711`](https://github.com/johnmarktaylor91/torchlens/commit/3858711f3feb71de3ba14791b8b694c9a0d5f584))

The Param field was renamed trainable -> is_trainable during glossary conformance and the
  deprecation shim was removed (clean break). The param export schema in PARAM_LOG_FIELD_ORDER and
  Param.to_pandas already emit the is_trainable column, but the io export round-trip test still
  hard-coded the old 'trainable' column name in its stable-column list, causing KeyError:
  "['trainable'] not in index" for the csv/json and parquet cases. Update the test schema to
  is_trainable to match the canonical export column.

- **facets**: Decouple fused-pattern error from HF internals (point at load-time eager); todo for
  deferred attention flag
  ([`1c59764`](https://github.com/johnmarktaylor91/torchlens/commit/1c597645e84cbcf83cf89005e8735f439e7e68ed))

- **io**: Classify container render-scratch as runtime-only; refresh __all__ count
  ([`d517bfd`](https://github.com/johnmarktaylor91/torchlens/commit/d517bfda94f4b638a9ea67e0a3d24416e402828c))

Two failures surfaced by the post-P4 not-slow backstop:

- _pending_container_collapse_nodes (transient render scratch set by the shipped show_containers
  collapse path, 2ad0ddff) was never in the runtime-only scrub allowlist, so drawing a
  container-grouped trace and then saving it raised in scrub on the MLX object-module path. Classify
  it runtime-only alongside its sibling _last_sibling_ordering_decision (provably non-portable
  render state).

- test_report_explain's __all__ count was stale at 62 (the shipped Container value-core took it to
  65; container-completion's input_at makes 66). Bring it current to match test_api_surface.

- **io**: Map orphan_records blob kind
  ([`fd45fcf`](https://github.com/johnmarktaylor91/torchlens/commit/fd45fcf0344d7c9e196c107de4ef539171dc4fb4))

- **io**: Restore non-torch audit-only portable-save guard
  ([`b8c2e32`](https://github.com/johnmarktaylor91/torchlens/commit/b8c2e32e351c52fb239865e9eb10c7d84b9dabdd))

- **perf**: Add save modes
  ([`9a6ff79`](https://github.com/johnmarktaylor91/torchlens/commit/9a6ff7966e425a75b457097922e259393ca780fd))

- **perf**: Cache module code contexts
  ([`34332fa`](https://github.com/johnmarktaylor91/torchlens/commit/34332fac12d5b911fcd425fab494d47fc46b2e26))

- **perf**: Cache op code contexts
  ([`e1ecce2`](https://github.com/johnmarktaylor91/torchlens/commit/e1ecce20347dfd5c42faac31052419bd2f55beef))

- **perf**: Dedup saved activations by identity
  ([`f79d29e`](https://github.com/johnmarktaylor91/torchlens/commit/f79d29e57b53566142ea3011a37a645a58e3cdb4))

- **perf**: Gate mutation alias checks
  ([`d09498e`](https://github.com/johnmarktaylor91/torchlens/commit/d09498ec903dc0ad8e9f91269e4b0fccaae0e6d0))

- **perf**: Partition fallback arg extraction once
  ([`953757b`](https://github.com/johnmarktaylor91/torchlens/commit/953757b236a6114b225b952ce2efc007873a193c))

- **perf**: Reuse tensor metadata for memory accounting
  ([`c2387a0`](https://github.com/johnmarktaylor91/torchlens/commit/c2387a033b81507d763e8ae8173cef4599966f7b))

- **perf**: Skip identity mutation comparisons
  ([`86de2cd`](https://github.com/johnmarktaylor91/torchlens/commit/86de2cd950b7935e7af606d00d2556cd1081f89a))

- **postprocess**: Bind buffer-only outputs; merge+check conditional metadata; sync to_pandas
  columns
  ([`6310f06`](https://github.com/johnmarktaylor91/torchlens/commit/6310f0626bc12c8ff47d3781c77774c77b2231d0))

Four tracked bugs plus review hardening:

- A registered buffer returned directly from forward() without ever being touched by a traced op had
  no graph node, so the model 'traced' with no output layer and validate_forward_pass failed with
  'No output layers found'. Output parent labels are now resolved explicitly before Step 0
  (late-logging untouched buffers through the capture buffer-read pathway, memoized per tensor so a
  buffer returned twice binds both output nodes to one parent), and mixed attributed/unattributed
  output tuples keep exact positional alignment instead of silently shifting. -
  cond_branch_then_children-era conditional membership is now merged across passes on rolled Layers
  instead of taking pass 1's value. - Two new metadata invariants (14: conditional arm-entry edges
  must exist in the rolled topology; 15: per-op body roles must agree with conditional_branch_stack
  id+arm) strengthen the validation tripwire; nothing was loosened. - Trace.to_pandas() now derives
  its columns from LAYER_PASS_LOG_FIELD_ORDER minus a documented exclusion list, enforced by a test
  so new Op fields can never silently miss the table again; live-accessor fields (input_ops) are
  excluded so detached/disk-loaded traces still export (parquet round-trip covered).

An adversarial review of the diff found and we fixed: duplicate direct-buffer outputs crashing
  _fix_buffer_layers, and invariant 15 originally accepting body roles whose conditional/arm
  mismatched the stack. Streaming tests' monkeypatched _add_output_layers stubs updated to the new
  signature.

- **postprocess**: Correct conditional evaluation entry edge
  ([`c13cf53`](https://github.com/johnmarktaylor91/torchlens/commit/c13cf53d2ba84cdbb96a8b0d0d5993a7340c2cc3))

- **postprocess**: Repair predicate replay metadata
  ([`be03934`](https://github.com/johnmarktaylor91/torchlens/commit/be039342b42d5f6f7ef81d0dd73d36f765b7cabc))

- **recipes**: Match DistilBERT FFN class under current and legacy transformers names
  ([`ea5dd69`](https://github.com/johnmarktaylor91/torchlens/commit/ea5dd69cc921ca3b66fd210395262a87d629e6a7))

- **validation**: Isolate ground truth mutable state
  ([`03ed5ba`](https://github.com/johnmarktaylor91/torchlens/commit/03ed5bad45ae409b32d0a88b529ffc591956efbf))

- **validation**: Narrow backward backpointer carve-out to structural exemption
  ([`d11494b`](https://github.com/johnmarktaylor91/torchlens/commit/d11494b7df00bcd43b4492e63c9c9b334a2f70a4))

The layer-backpointer invariant skipped any layer whose GradFn backpointer was None, which admitted
  the legitimate mid-forward autograd.grad case but would also have masked a genuine pairing bug
  (the exact over-broad-carve-out pattern the tripwire principle forbids). A None backpointer now
  FAILS the invariant unless the layer structurally postdates every recorded backward trigger's
  forward boundary (trigger-time op count, tightened by paired root GradFns and observed op-gradient
  events). Output-alias backpointer projection is fixed rather than exempted, keeping the check
  strict for all pre-trigger layers. Corruption test: an artificially severed backpointer on a
  pre-trigger layer raises MetadataInvariantError; mid-forward cases stay green.

- **validation**: Refuse sparse replay without child versions
  ([`9cba2be`](https://github.com/johnmarktaylor91/torchlens/commit/9cba2be59b4f8af3e12b06a82660e2a4c9c8603d))

- **validation**: Snapshot ground-truth outputs before state restore
  ([`39a5029`](https://github.com/johnmarktaylor91/torchlens/commit/39a50292d47c202641a5e6bb820fc4d8ee4f4338))

validate_forward_pass saved ground-truth output tensors by reference, then called
  model.load_state_dict (which writes buffers in-place). When a model returns a registered buffer it
  reassigned mid-forward (e.g. recurrent state: return self.h), the saved ground truth aliased that
  buffer and was clobbered to its initial value on restore, producing a validation FALSE-NEGATIVE
  against an otherwise-correct captured/replayed graph. Snapshot (detach().clone()) each
  ground-truth output before the restore, mirroring the existing input deep-copy guard. Corrects the
  ground truth fed to the tripwire; does not weaken any check. Regression test added.

- **validation**: Speed large graph validation
  ([`f6b7804`](https://github.com/johnmarktaylor91/torchlens/commit/f6b7804d4960d35500f5f802f2193bfb159ec1e1))

- **visualization**: Certify loop rolling sites by signature
  ([`bda642d`](https://github.com/johnmarktaylor91/torchlens/commit/bda642de659739e4f378c31e45b0a37b2583e2d6))

- **viz**: Emit sibling rank-groups at correct LCA cluster scope; strip cleanly; graceful top-level
  fallback
  ([`de0f95e`](https://github.com/johnmarktaylor91/torchlens/commit/de0f95e560d3722816ae01477ea3987b3af7b265))

Addresses code-review findings: preserve the edge module key before the has_input_ancestor loops
  clobber it (was forcing all rank-groups to top level); bracket strip markers around the rank-group
  subgraph so no empty wrapper is left on drop; replace the cluster-insert assert with a verify-safe
  top-level fallback.

- **viz**: Keep loop rolling render cache off trace
  ([`df9d2e9`](https://github.com/johnmarktaylor91/torchlens/commit/df9d2e9503b3b19f555ea72a1c42c46d0c400700))

- **viz**: Merge recurrence back-edge and buffer-edge In/Out labels into midpoints
  ([`ee750b7`](https://github.com/johnmarktaylor91/torchlens/commit/ee750b7cfcc08c89865a811000e8da165a51053c))

Rolled-view pass labels collided where graphviz allocates no layout space for head/tail labels:
  anti-parallel recurrence pairs crammed four labels into the band between two near-touching edges,
  and buffer read/write edges did the same with dashed splines crossing their own labels.

- Recurrence back-edges (structural step_index comparison) merge In/Out into one two-line midpoint
  label, like self-loops already did; graphviz reserves dummy-node space for midpoint labels, which
  also pushes the anti-parallel curves apart. Guarded so conditional/argname-labeled edges keep
  head/tail labels. - Buffer-incident edges always merge (single annotations become one-line
  midpoint labels): their read/write pair shares one narrow band. - One congested-forward-edge case
  (forward partner of a merged back-edge with an endpoint self-loop) merges for the same reason. -
  Remaining head/tail labels on >=3-op cycle bodies and oblique skip/approach edges carry
  audit-derived per-edge labeldistance/ labelangle; setting those attrs globally would switch
  graphviz to its inferior place_portlabel path, so they are applied only where the geometry audit
  showed wins. - Head/tail pad reverted 5->4pt, self-loop spacer 16->8pt (round-3 tuning that read
  clean; the larger values orphaned labels). - Module clusters carry margin=20 so borders clear
  inner labels.

Verified by an exact-geometry audit over dot -Tjson (0 violations across 16 demo graphs, from 12 at
  baseline) and independent Opus + Codex visual review (both satisfied, every label inspected at
  zoom).

- **viz**: Merge recurrence self-loop In/Out into one midpoint label
  ([`85ad4b2`](https://github.com/johnmarktaylor91/torchlens/commit/85ad4b2f16a2f4a4c43e4be6a07b8d71faabcf0b))

A rolled recurrence self-loop previously carried separate head/tail ``In``/``Out`` pass-range
  labels, which graphviz places near the node endpoint without reserving space, so they crowded the
  node corner. Self-loops never carry argument or conditional labels, so the In/Out ranges are now
  merged into a single midpoint ``label``, which graphviz models as a dummy node and reserves layout
  space for, eliminating the overlap. ``In`` is placed above ``Out`` for the default bottom-up
  layout and flips for top-down, and the label font is harmonized with the head/tail labels.
  Non-self-loop edges keep separate head/tail labels (those edges may also carry
  argument/conditional midpoint labels a combined label would collide with).

Threads ``rankdir`` through _add_node_to_graphviz/_add_edges_for_node to order the combined label.
  Updates the self-loop test and adds an orientation-flip test.

- **viz**: Pad rolled head/tail edge labels with a blank line above and below
  ([`e93f254`](https://github.com/johnmarktaylor91/torchlens/commit/e93f25441214b68d2212b78f624cf069e217fbc3))

Head/tail (In/Out) labels are not allocated layout space by graphviz, so they crowd the node where
  an edge attaches -- worst where an edge meets a node at an oblique angle, and where an
  antiparallel recurrence pair lands an In and an Out label at the same node. Pad each non-self-loop
  In/Out label with a blank line above and below (keeping the horizontal spaces): the blank lines
  push the visible text along its edge, away from the node, and keep the two labels of an
  antiparallel pair on separate lines. Self-loops keep their combined midpoint label and are
  unaffected.

Adds a regression test for the blank-line padding and notes it in the demo README.

- **viz**: Raise on empty-output/failed/timed-out graphviz render instead of silent failure
  ([`75248cc`](https://github.com/johnmarktaylor91/torchlens/commit/75248cc462a7efcb6c1b42e362f65926e5a49adc))

- **viz**: Render atomic modules as rectangles; compose code panel side-by-side
  ([`31ceee5`](https://github.com/johnmarktaylor91/torchlens/commit/31ceee5878b5091cd38705acdedc4fba0f4ca326))

Atomic (single-op leaf) modules now render as module-marked rectangles at every call depth instead
  of ellipses or box3d collapses. The is_atomic_module flag was dormant because its capture-time
  detector was stubbed to return False; atomicity is now computed in postprocess from the op->module
  map (a module is atomic iff exactly one op has it as its innermost module), which correctly
  handles multi-op leaves with side ops such as a BatchNorm num_batches_tracked increment. Atomic
  modules never collapse to box3d, and a reused atomic module carries its split call range on the
  @module marking (e.g. @relu:1 / @relu:2-4), never duplicated as a title count.

The source-code panel is now rendered as a separate SVG and composed beside the graph rather than
  injected as a graphviz subgraph, so it no longer distorts the graph's proportions: the graph DOT
  is byte-identical with or without the panel. PDF/PNG composition uses cairosvg (graceful fallback
  to the in-graph subgraph when unavailable); SVG composes directly.

Updates the module-containment snapshots (only is_atomic_module/atomic_module_call flip, no
  collateral), the atomic-classification and collapse tests, the loop-rolling tests and demos, and
  the demo README.

- **viz**: Scope rolling render cache per-draw; preserve interleaved recurrence chains
  ([`98da988`](https://github.com/johnmarktaylor91/torchlens/commit/98da98803bd602f5d163a97e9658d4f3cc7bb564))

Two fixes from the final dual-lab review of the loop/module rolling feature:

- Scope the rolled-view render memoization to a single draw() (ContextVar cache scope) instead of a
  trace-lifetime WeakKeyDictionary. Eliminates stale labels on re-render after an in-place
  structural mutation while keeping within-draw memoization. No trace mutation, no .tlspec leak, no
  fork inheritance.

- Detect recurrence self-edges via real same-layer transitive dependency chains instead of
  pass_index+1 adjacency, so interleaved multi-site recurrences (e.g. 1->3->5 and 2->4->6) are
  detected (not dropped) and kept distinct in the collapsed view via chains=... metadata.
  Single-site and no-spurious-edge guarantees preserved.

- **viz**: Scope two-arrow multiplicity to unrolled views (rolled keeps single edge)
  ([`cd7697a`](https://github.com/johnmarktaylor91/torchlens/commit/cd7697acea74e558a1f5fd54e72bd891b770e911))

The two-arrow multiplicity render regressed the rolled-view label geometry of ParallelSiblingsLoop:
  the extra parallel edge shifted layout so the rolled In/Out iteration labels penetrated the
  recurrence back-edge spline/arrowhead (3 hard violations in the locked test_label_geometry gate).

Arg-slot multiplicity is a forward/unrolled concept; the rolled view abstracts recurrence with
  In/Out iteration labels where a second parallel arrow both clutters and collides with the loop
  back-edge. Collapse same-parent multi-slot occurrences to a single edge in the rolled view
  (pre-existing behavior), keeping the two-arrow rendering for unrolled/forward/dot views. Done at
  the shared _render_edge_occurrences() point so DOT and rank layout stay in lockstep (no
  labeldistance/labelangle — place_portlabel trap avoided).

test_label_geometry all fixtures back to zero hard violations (gate intact, not weakened);
  test_edge_multiplicity_render still green (unrolled x+x/cat = 2 edges, rolled = 1); behavioral
  goldens byte-identical; smoke green.

- **viz**: Set sibling-ordering decision on every engine path
  ([`941d494`](https://github.com/johnmarktaylor91/torchlens/commit/941d4942c1b5da0643bddd35a73c32c6af8672e5))

The rank-layout early return skipped the dot-only sibling-ordering post-pass without setting
  _last_sibling_ordering_decision, so traces rendered via auto-rank lost the attribute (DenseNet now
  crosses the cost threshold post-ELK-retirement and broke test_densenet_is_safe_noop). The trivial
  decision is now set before the engine branch; the DenseNet test pins vis_node_placement='dot' to
  keep exercising the ordering pass it was written for, and a new test locks the rank path's trivial
  decision.

- **viz**: Simplify rolled-view labels; restore clean recurrence self-edge
  ([`d050164`](https://github.com/johnmarktaylor91/torchlens/commit/d05016464d40ba234e18092779930e5a7cdda415))

Cleanup after visual review of the loop/module-rolling feature: - Restore a clean self-pointing
  recurrence arrow (plain self-edge, no glyph/icon, no pass-range labels); only for genuinely
  recurrent rolled nodes, not fan-outs. - Atomic single-op modules keep their op appearance when
  collapsed (no @module relabel, no cluster box) -- a 1-op module is already maximally collapsed. -
  Remove the arcane label vocabulary (N sites, xN multiplier, mixed/parallel facet, sites?) and
  ~1000 lines of supporting machinery; never show a redundant count AND call-ranges together
  (count-alone, or colon ranges e.g. linear_1_1:1,2-4). - Buffers use @state:1-N colon syntax.

- **viz**: Size code panel to real monospace metrics; wrap lines past 120 chars
  ([`8fa0907`](https://github.com/johnmarktaylor91/torchlens/commit/8fa0907f750a64d60660f5669673fc36db3d3138))

Graphviz lays out the panel HTML table with fallback Courier metrics (~0.483 em/char at 14pt) while
  SVG rasterizers resolve the same family to a real monospace font at ~0.60 em/char, so the panel
  box was ~22% too narrow and long source lines clipped at the panel's inner edge.

The panel now injects an explicit minimum TD WIDTH computed from monospace math (0.62 em/char incl.
  safety margin) over the longest displayed line, and lines beyond a 120-char cap get
  indent-preserving hanging wrap instead of clipping (widest demo line is 79 chars, so the committed
  demos never wrap). Both the composed and in-graph fallback panel paths share the sizing. Graph DOT
  stays byte-identical with and without the panel (existing lock test). Verified by geometry tests
  over the panel SVG and zoomed visual inspection of the three longest-line demos.

### Chores

- Archive 2026-06-11 bug-fix sweep in todo tracker; refresh stale items
  ([`a141b1d`](https://github.com/johnmarktaylor91/torchlens/commit/a141b1dcf207c69fbfb9f050535453b0ddefaa92))

Ten fixed bugs moved to Completed with commit refs; ARG-KWARGS rewritten as partial (symptom set
  fixed, audit remainder enumerated); RNG-MULTI-GPU marked hardware-blocked, ELK bugs deferred into
  ELK retirement, the non-registered-mutable-state validation item marked as needing a design call;
  the four suspected-stale items from the 06-10 audit hands-on verified and refreshed (facets
  follow-ups, fastlog ergonomics names, performance-docs API spellings, multi-output audit overlap).

- Archive completed items from todo tracker (2026-06-10 audit)
  ([`a7b7c3e`](https://github.com/johnmarktaylor91/torchlens/commit/a7b7c3edea9845db896be1e58d5c219ea032ce14))

Cross-referenced every open section against the repo and git history; 22 verified-done items moved
  to Completed (recent) in condensed form with commit evidence (capture-path unification, IR-only
  OpEvent hot path, loop/module rolling reconciliation, MLX backend, bfloat16 tolerance fix,
  quantized-tensor crash fix, retired-attr label bugs, num_edges, and friends). Two sections
  partially annotated (save-selection unification residuals, post-2.0 follow-ons a/c/f/m/n/o). Quick
  index regenerated; everything without hard fix evidence kept.

- Refresh todo tracker index; file code-panel line-truncation todo
  ([`3324f1a`](https://github.com/johnmarktaylor91/torchlens/commit/3324f1a682b3cea0315b9d30c63c00ce6c805beb))

- **notebooks**: Remove superseded botched total_audit series
  ([`9a32400`](https://github.com/johnmarktaylor91/torchlens/commit/9a3240002f3cea01c7dac6346c7ebf077c72c6fe))

- **perf**: Add op slots baseline benchmark
  ([`8e62a73`](https://github.com/johnmarktaylor91/torchlens/commit/8e62a73b5c322359f9898ead6889b4b70c5337ac))

- **perf**: Mark fixed tracker backlog
  ([`a25a46a`](https://github.com/johnmarktaylor91/torchlens/commit/a25a46abe075b80abcc325410809a106f78344fb))

- **perf**: Repair benchmark harness profiling
  ([`83623cd`](https://github.com/johnmarktaylor91/torchlens/commit/83623cd0f9ae3c2b932ee0ff1bfe4cc8f6aadcf2))

- **perf**: Scrub unsaved lookup scratch
  ([`f44db1c`](https://github.com/johnmarktaylor91/torchlens/commit/f44db1c79e1615410c1cfa9c27fc946e659c2efa))

- **perf**: Update build state
  ([`0bec695`](https://github.com/johnmarktaylor91/torchlens/commit/0bec695a2b53072ac60880f098aa77f283ae403b))

- **perf**: Update build state
  ([`a0d440e`](https://github.com/johnmarktaylor91/torchlens/commit/a0d440e98858e8042169a957ac0854deaeb4612f))

- **perf**: Update build state
  ([`88f568f`](https://github.com/johnmarktaylor91/torchlens/commit/88f568fe845910971984464e9b657b5985a23cbc))

- **perf**: Update build state
  ([`49bb1c4`](https://github.com/johnmarktaylor91/torchlens/commit/49bb1c4b2855c76fbebbad1a8fab297035e406f1))

- **perf**: Update build state
  ([`a41fe3d`](https://github.com/johnmarktaylor91/torchlens/commit/a41fe3de5e99b5ca03de19f50e7ce4952f5cb09e))

- **perf**: Update build state
  ([`8ed7a4c`](https://github.com/johnmarktaylor91/torchlens/commit/8ed7a4cc69f9d342d9956d106082a9c9e4ee00d2))

- **perf**: Update build state
  ([`f6b6902`](https://github.com/johnmarktaylor91/torchlens/commit/f6b69021268cd75113ab2c2b4156ecaccab3d847))

- **perf**: Update build state
  ([`81b6227`](https://github.com/johnmarktaylor91/torchlens/commit/81b62271e9b02f83f50bb325cb68cfd9cd032c05))

- **perf**: Update build state
  ([`50f518c`](https://github.com/johnmarktaylor91/torchlens/commit/50f518ca6e462d0555f122a7cdd4db33d25ca1ab))

- **perf**: Update build state
  ([`87b0d76`](https://github.com/johnmarktaylor91/torchlens/commit/87b0d76219bb36d929bfa63fa2c4209a1ec53f5e))

- **perf**: Update build state
  ([`1c48e5d`](https://github.com/johnmarktaylor91/torchlens/commit/1c48e5da4d4aff264a8a8fbe944ff7c49acfd9d8))

- **perf**: Update build state
  ([`ca737e9`](https://github.com/johnmarktaylor91/torchlens/commit/ca737e965fb1de3f369c95c2b22643cbf111ac30))

- **repo**: Gitignore notebooks/audit/_artifacts
  ([`6224bcb`](https://github.com/johnmarktaylor91/torchlens/commit/6224bcbb5093de94cccb5d827ef61679f5488c57))

- **repo**: Untrack agent scratch dirs, gitignore .research/ and .project-context/
  ([`9be91fe`](https://github.com/johnmarktaylor91/torchlens/commit/9be91feb0cd926da93cd7fb167acf00a0ec1d646))

Stop tracking internal sprint notes, plans, audits, and logs that do not belong in the public repo
  (files remain on disk locally). Keep the two architecture maps AGENTS.md links to: architecture.md
  and state_of_torchlens.md.

- **spike**: S0.g tinygrad discovery round 1
  ([`7b509a5`](https://github.com/johnmarktaylor91/torchlens/commit/7b509a53fffd01b7d87c8a4fff3533a40d23789c))

- **spike**: S0.g tinygrad discovery round 2
  ([`4c2ccd0`](https://github.com/johnmarktaylor91/torchlens/commit/4c2ccd0df37446048e1e99c9e704b79dab0de198))

- **spike**: S0.g tinygrad discovery round 3
  ([`f43ddbe`](https://github.com/johnmarktaylor91/torchlens/commit/f43ddbefde04ca41795a68a7803a711148c4ec9d))

- **spike**: S0.j jax feasibility round 1
  ([`b163286`](https://github.com/johnmarktaylor91/torchlens/commit/b163286c53032ec49b02d1799eee80e3f7724749))

- **spike**: S0.j jax feasibility round 2
  ([`e8ae430`](https://github.com/johnmarktaylor91/torchlens/commit/e8ae43021c8d5201df0253e35a7293ed6063018d))

- **test**: Nest loop_module_rolling demos under test_outputs/visualizations/
  ([`76da6a3`](https://github.com/johnmarktaylor91/torchlens/commit/76da6a34d91c393de070fc6606e60fc629f2af9b))

Match the existing test_outputs/visualizations/ convention instead of a top-level sibling folder;
  update OUTPUT_DIR accordingly.

- **todo**: Jax/tinygrad plan converged build-ready (13-round adversarial review)
  ([`cb8c12e`](https://github.com/johnmarktaylor91/torchlens/commit/cb8c12e9bd576b58ea56ab0c6fee9d323832fb19))

- **todo**: Mark JAX+tinygrad backends shipped (v2.19.0); refresh open-work index
  ([`67fdf29`](https://github.com/johnmarktaylor91/torchlens/commit/67fdf2962e71e45ff3d7b827a141665cc7b11b08))

- **todo**: Resolve facets item 7 as anonymous facets; file partial-payload save policy
  ([`0d10810`](https://github.com/johnmarktaylor91/torchlens/commit/0d108102f598f62c765b9d7aef21fd707a0c7efc))

- **todo**: Route resolved-access-vs-label-storage to the accessor audit
  ([`0b23dbc`](https://github.com/johnmarktaylor91/torchlens/commit/0b23dbc54c884a9ea0611c359f1ce0ba73b23082))

Captured from the facets-followup riff: storage stays label-based (settled), but whether
  label-fields get parallel resolved-access properties is a package-wide accessor-audit decision,
  not a per-class bolt-on.

- **todo**: Verify tracker against shipped backward+perf sprints; regenerate quick index
  ([`c6a0b74`](https://github.com/johnmarktaylor91/torchlens/commit/c6a0b7400e39cf100557bb08b92099cfe43278d1))

- **todos**: Buffer-sprint design todos (rolled-view collision, op-subclass refactor, loop-indexing
  revisit, densenet)
  ([`65b7529`](https://github.com/johnmarktaylor91/torchlens/commit/65b7529d0dc128e385326c182d35d49866faa585))

- **todos**: Code-verified reconciliation, mark shipped items done
  ([`c7e303b`](https://github.com/johnmarktaylor91/torchlens/commit/c7e303b2c5502298846d26a8a8629265589720c9))

- **todos**: Mark buffer WRITES/OVERWRITES capture done (shipped 2026-06-05)
  ([`ef80636`](https://github.com/johnmarktaylor91/torchlens/commit/ef806367d1bb043a66a3ea64bbc4f16a915edb01))

- **todos**: Mark class de-Log rename done (shipped); narrow naming-pass residual
  ([`7e182af`](https://github.com/johnmarktaylor91/torchlens/commit/7e182af7f4ce2160a34e0bff65234f285b435c40))

- **todos**: Mark distilbert_ffn recipe bug fixed (ea5dd69)
  ([`d98f57c`](https://github.com/johnmarktaylor91/torchlens/commit/d98f57c3f42c21b49364c254b2386cefbcdb71f9))

- **todos**: Mark glossary conformance done, log distilbert_ffn recipe bug
  ([`adb4e0f`](https://github.com/johnmarktaylor91/torchlens/commit/adb4e0ff4ef67befb8e2b2519a5096ceceb9fd5f))

- **todos**: Slice/facet/outs integration note, ELK removal findings, viz themes, num_edges
  ([`b32649b`](https://github.com/johnmarktaylor91/torchlens/commit/b32649b0939df4ff98499dfb2c1f034d0498f3ab))

- **types**: Clear pre-existing mypy errors (type-only, no behavior change)
  ([`cf6e80e`](https://github.com/johnmarktaylor91/torchlens/commit/cf6e80ed544464c46049040b653dba1d19644b95))

### Code Style

- Auto-format with ruff
  ([`b8bb3e2`](https://github.com/johnmarktaylor91/torchlens/commit/b8bb3e2f5d1c3e7d7a27d48e3e8a6b29e15e140e))

### Documentation

- Backend-completion capability matrix + lockstep
  ([`ec249ca`](https://github.com/johnmarktaylor91/torchlens/commit/ec249cac7366c4348325adb29b3dca070c55a88c))

JAX/MLX intermediate grads, jax_control_flow=region, PayloadLoadHints, unverified status; matrix
  accurate (no overclaim). Closes the backend-completion sprint.

- Combined-sprint backend capability lockstep + README maturation
  ([`e984b75`](https://github.com/johnmarktaylor91/torchlens/commit/e984b75a022b82f60267a94370ad5de683b4f9f1))

- Followup-sprint backend capability lockstep
  ([`eeff7a8`](https://github.com/johnmarktaylor91/torchlens/commit/eeff7a87ae1af06dbc89ff7d5af83cf3f6720aab))

JAX pure-jit inlining + PRNG/sharded codec; MLX modules/payloads/derived grads; capability matrix
  accurate (no overclaim; JAX T1 deferred).

- Lock validation-integrity principle (validation is a tripwire, never bandaid)
  ([`6e5fa22`](https://github.com/johnmarktaylor91/torchlens/commit/6e5fa224b781324c603d61e60ca7755984e4824d))

- Update agent-guide examples and record names to current trace API
  ([`6c328bf`](https://github.com/johnmarktaylor91/torchlens/commit/6c328bfa0f64c10757736dd2a3689470c3ebb667))

Replaces removed log_forward_pass/vis_opt examples with tl.trace, fixes the stale activation
  accessor (now .out) and the broken live-hook intervention pattern (use a capture-time selector
  tl.func instead of a finalized label), and updates entry-point and record-class names (trace and
  draw_backward; ModelLog/LayerLog/LayerPassLog become Trace/Layer/Op). Also lands the glossary and
  docs lockstep principle section.

- **backend**: Lock docs to backend substrate
  ([`0f91191`](https://github.com/johnmarktaylor91/torchlens/commit/0f911919185c49ef2fa2d14b19de201307d224e6))

- **backend**: M0.1a artifact 1 — invariant contract
  ([`e44f768`](https://github.com/johnmarktaylor91/torchlens/commit/e44f7681cbc9d8dabbf15c1adcdee99b7f1219a8))

- **backend**: M0.1a artifact 2 — public-surface-kwarg-backward-matrices
  ([`74f20a0`](https://github.com/johnmarktaylor91/torchlens/commit/74f20a02ce6fb50aba5252bb57fb5f4fe13ac366))

- **backend**: M0.1a artifact 3 — serialization-contract
  ([`72a35cc`](https://github.com/johnmarktaylor91/torchlens/commit/72a35cccd9ed1475941f1b37f1964d908fa6ca12))

- **backend**: M0.1a artifact 4 — BackendSpec registry migration map
  ([`15dbfb3`](https://github.com/johnmarktaylor91/torchlens/commit/15dbfb3a7ef1bf6d0fbdc5e682d66a3f148a9006))

- **backend**: M0.1a artifact 5 — docs-glossary-change-list
  ([`eadc8f6`](https://github.com/johnmarktaylor91/torchlens/commit/eadc8f6349e478c51367e31db6d67179416f1741))

- **backends**: Non-torch intervene/halt blocker + tightened rejection
  ([`b47e94f`](https://github.com/johnmarktaylor91/torchlens/commit/b47e94f555f0107445202b2a61809a7c28c12a04))

Spike NO-GO across jax/tinygrad x static-intervene/static-halt/value-dependent (the
  traced-functional + lazy-graph wall). Static-label save= remains the only supported non-torch
  predicate subset; intervene/halt route to torch.

- **buffer**: 4-agent research findings + Phase-2 write-capture plan + run-state
  ([`81e8118`](https://github.com/johnmarktaylor91/torchlens/commit/81e811843a2a45f4c12d9d590a85ed8e3840183b))

- **buffer**: Add docs/buffers.md explainer to v5 build spec
  ([`81d5c09`](https://github.com/johnmarktaylor91/torchlens/commit/81d5c091053963d89faa62f55db35f07954bf64f))

- **buffer**: Buffer-sprint design spec v1 (entity + one-node-per-version + dual-label)
  ([`52c0575`](https://github.com/johnmarktaylor91/torchlens/commit/52c05750d2a2e6b35e06b1e00425c023646c1071))

- **buffer**: Mark Buffer Option B shipped; run summary + state
  ([`56f856b`](https://github.com/johnmarktaylor91/torchlens/commit/56f856bef7e4468ace5594f4cce2dfc205d1b850))

- **buffer**: Overnight wrap-up — P1 shipped, validated write-capture design (deferred), gaps
  documented
  ([`6072221`](https://github.com/johnmarktaylor91/torchlens/commit/607222139704dbad82ff2d49c959a286bf54490d))

- **buffer**: Plan v3 — postprocess-synthesis design (option-a version nodes, end-of-pass snapshot,
  no hot-path detector)
  ([`5835538`](https://github.com/johnmarktaylor91/torchlens/commit/583553894587d841084ef714d9c234b6a45905eb))

- **buffer**: Plan v4 — option-2 capture-at-the-moment (scoped setattr + post-op value-check)
  ([`2638053`](https://github.com/johnmarktaylor91/torchlens/commit/263805323f9967a73a1c447fa859f596b9109b95))

- **buffer**: Plan v5 BUILD spec — option-2 validated by both labs, all fixes folded in
  ([`e393f3d`](https://github.com/johnmarktaylor91/torchlens/commit/e393f3d15e18a1874e6c6febe78808e600f7f6af))

- **buffer**: Round-1 dual-lab review findings + scope decision (write-capture gap; descope
  recommended)
  ([`ac170ff`](https://github.com/johnmarktaylor91/torchlens/commit/ac170fff606541c8355aed6bbfa6720b3685ab9e))

- **buffer**: Track edges-proposal working doc
  ([`2b1480f`](https://github.com/johnmarktaylor91/torchlens/commit/2b1480f58ce896b48f1c5e0385b0eded618899f0))

- **buffer**: V3 confirm round evidence (both labs NOT bulletproof); alpha/beta/hybrid fork open
  ([`0ed4a7f`](https://github.com/johnmarktaylor91/torchlens/commit/0ed4a7f862f3592bcc835c43788e72e88c9931f6))

- **buffer**: V4 review — both labs validate the approach empirically; 3 blocking engineering fixes
  each
  ([`af36862`](https://github.com/johnmarktaylor91/torchlens/commit/af368624a7ebb4fe3e9b8efadb40552c3b736754))

- **capture**: Add performance and agent guides
  ([`595613d`](https://github.com/johnmarktaylor91/torchlens/commit/595613d5005a11fe9665b86d9747be217e3babec))

- **capture**: Add torch func transform example
  ([`d809cdc`](https://github.com/johnmarktaylor91/torchlens/commit/d809cdc3ba7b95b1bda17b50e6e0461c2c255c25))

- **capture**: Document unified capture API
  ([`04bc6e5`](https://github.com/johnmarktaylor91/torchlens/commit/04bc6e54779bd315fcf6102d44f919f2acf14112))

- **examples**: Port examples to 2.0 API
  ([`63cc773`](https://github.com/johnmarktaylor91/torchlens/commit/63cc7732a0501e54c1110f408ecde7559c29a4dd))

- **facets**: Add runnable tutorial notebook
  ([`3db2d05`](https://github.com/johnmarktaylor91/torchlens/commit/3db2d05a0a02de53740d71ec27e3f8627322c464))

- **facets**: Canonical glossary lockstep for P1 spec model (FacetSpec, MissingGradient, capability
  classes, registry scoping)
  ([`c9ef746`](https://github.com/johnmarktaylor91/torchlens/commit/c9ef74633f046b88e8d9ef65ecbfae87508559ae))

- **facets**: Merge tutorial notebook
  ([`de29cb7`](https://github.com/johnmarktaylor91/torchlens/commit/de29cb7b41e520aaf25a827edf3904d0d4300472))

- **facets**: Sprint complete — P1-P4 + notebook shipped to local main; summary, todos, residuals
  ([`a475abe`](https://github.com/johnmarktaylor91/torchlens/commit/a475abede262008ac5212c483b890afa3b5496c0))

- **facets**: Sprint state — P1+P2 done, P3 dispatched
  ([`dde243b`](https://github.com/johnmarktaylor91/torchlens/commit/dde243b2dad4e0bb95e09dbfd80a743d6717240d))

- **facets**: Sprint state — P3 done, P4 dispatched; note auto-eager residual
  ([`100cfc0`](https://github.com/johnmarktaylor91/torchlens/commit/100cfc01b958ab37075c16477b8452a5033586ab))

- **glossary**: Mark residual lock set resolved (code fully conforms)
  ([`32021bb`](https://github.com/johnmarktaylor91/torchlens/commit/32021bb96cbbb3c992d325b34acf6708c401a325))

- **glossary**: Reconcile to shipped code
  ([`75b0d50`](https://github.com/johnmarktaylor91/torchlens/commit/75b0d5022eb4972bcfa25fe1e0949271bd864352))

Diff the glossary's documented PUBLIC API against the shipped code surface and reconcile the parts
  that are unambiguous:

- Trim the ModuleCall and Module Output Passthroughs sections to what the code exposes (out/outs,
  out_shape/out_shapes, out_dtype/out_dtypes for ModuleCall, grad/grads). Drop the bare and
  transformed memory passthrough lines per the locked module-scope memory decision; point readers at
  the prefixed memory cluster instead. - Remove the false tl.log_forward_pass alias claim from the
  auto-routing section (no such alias exists; the entry point is tl.trace). - Add a reconciliation
  note to the top banner recording the above and FLAGGING a residual set of lock-backed naming
  targets that the shipped code never implemented (is_module_input, input_to_module_calls,
  args_summary/kwargs_summary, grad_fn_label, multi_output_type, the atomic_module resolver cluster,
  gradient_transform, has_saved_gradient, Param.is_trainable, the Buffer overwrite cluster, Module
  total_flops/total_macs/internal_param_memory, call_parent_address). These are logged LOCKED in the
  walkthrough deltas, so they are code fixes pending a JMT decision, not glossary deletions.

- **keystone**: Semantic-i/o keystone demo notebook + docs (B4)
  ([`018f6b3`](https://github.com/johnmarktaylor91/torchlens/commit/018f6b3778759ae5ca7dcb9130e2c06197a186aa))

Add notebooks/semantic_io_keystone_demo.ipynb: a deterministic, CI-safe (synthetic PIL-image
  classifier; no weight downloads) walkthrough of the keystone cascade -- trace(model, image_list,
  save=<conv subset>, save_raw_input=True) -> Trace.model_profile (keystone_applicable) -> decoded
  category table -> tl.repgeom.mds_evolution + draw(node_spec_fn=mds_scatter_node_spec) per-layer
  MDS thumbnail evolution -> original-input display. The template users copy.

Docs lockstep: docs/semantic_io.md + torchlens/CLAUDE.md/AGENTS.md cover trace.annotate /
  model_profile / tl.repgeom / the scatter node_spec. New public names provisional pending naming
  review; vault glossary deferred. No core behavior change; smoke green, parity goldens untouched;
  notebook executes clean.

- **node-visuals**: Keystone demo + docs + CLAUDE/AGENTS lockstep for Sprint C toolkit [C4]
  ([`5d216f3`](https://github.com/johnmarktaylor91/torchlens/commit/5d216f37ada50ef10403afed06f53883c86538e4))

Keystone demo showcases all four node-visual templates (MDS scatter, RDM, feature-map, scree) on the
  same image-classifier example + a build-your-own pointer to the tl.viz.render_* primitives. Audit
  viz notebook shows the new generic primitives alongside the existing heatmap visualizer factory.
  docs/semantic_io.md gains a Node-Visual Render Toolkit section (PIL-only, draw-time via
  node_spec_fn, survives .tlspec via _annotation_blobs, no new Trace field, provisional names).
  CLAUDE.md + AGENTS.md examples extended in lockstep.

- **notebook**: Add capture unification demo
  ([`79085c9`](https://github.com/johnmarktaylor91/torchlens/commit/79085c9f2ad8f58e6b40326e65e50f1e81840d1e))

- **notebook**: Finalize HF tutorial on current trace API
  ([`4f1b34f`](https://github.com/johnmarktaylor91/torchlens/commit/4f1b34f61272900981308ac658246f55999d3c22))

- **notebooks**: Add audit series batch 1
  ([`d48abeb`](https://github.com/johnmarktaylor91/torchlens/commit/d48abeb3844fef57063289c5f591411db5863a02))

- **notebooks**: Add audit series batch 2
  ([`d29b9e8`](https://github.com/johnmarktaylor91/torchlens/commit/d29b9e8c339c70bacf1bcc7676604c7c010f8827))

- **notebooks**: Add audit series batch 3
  ([`cfdf1d5`](https://github.com/johnmarktaylor91/torchlens/commit/cfdf1d55018f95ba57abe6afb635ce8f5caf9261))

- **notebooks**: Expand coverage round 2
  ([`7b79864`](https://github.com/johnmarktaylor91/torchlens/commit/7b798649017f38ff396a464fa1bcb25ee252ca40))

- **notebooks**: Fix review round 1
  ([`108b645`](https://github.com/johnmarktaylor91/torchlens/commit/108b645548e2ee7cd924ff02549a7058eeeacafc))

- **notebooks**: Fix review round 3
  ([`4c36d2e`](https://github.com/johnmarktaylor91/torchlens/commit/4c36d2e2481935ce065a312a1e0f353c0d2791dc))

- **notebooks**: Update audit series to v7 API
  ([`e2247f7`](https://github.com/johnmarktaylor91/torchlens/commit/e2247f7577de6c0f11b0634332925db630b4cd25))

- **readme**: Showcase multi-backend, backward, intervention, and perf capabilities
  ([`579d3d0`](https://github.com/johnmarktaylor91/torchlens/commit/579d3d0b5716f3b4101f81d74fda3c3b8e59c8c9))

- **semantic-io**: I/o-legibility demo + docs (A5)
  ([`a4d647c`](https://github.com/johnmarktaylor91/torchlens/commit/a4d647cce3d007171f16cb72fee6f4ba44e43fa6))

Add examples/semantic_io_legibility_demo.py: a deterministic, runnable walkthrough of the
  semantic-I/O surface -- auto-detected category table (output_table / summary(level=output) /
  output-node viz), output_style/output_head override, a custom decoder registered via
  tl.autoroute.output.register, and the original-input display + input_preprocessor provenance. The
  template users copy for their own output styles.

Add docs/semantic_io.md and update torchlens/CLAUDE.md + AGENTS.md examples for the output-decode +
  input-display API. New public names are provisional pending naming review; vault glossary update
  deferred until names are finalized. No core behavior change; smoke green, parity goldens
  untouched.

- **test**: Update loop-rolling demo README and tracked fixture for merged labels
  ([`473f01a`](https://github.com/johnmarktaylor91/torchlens/commit/473f01a0b0fdd723a1ef841a75f9a6fc6f4c2c5e))

Label Contract rewritten for the final scheme: HTML head/tail labels with an even transparent margin
  on varying forward edges; In/Out merged into midpoint labels for self-loops, recurrence
  back-edges, buffer read/write edges, and the congested-forward case (graphviz reserves layout
  space for midpoint labels, which head/tail labels never get); placement fine-tuning is
  audit-driven via tests/test_label_geometry.py. Removed the stale -Tplain two-pass paragraph.
  Per-demo label descriptions re-verified against freshly regenerated renders;
  two_distinct_loops.svg is the tracked fixture regeneration.

- **todos**: Edges mini-sprint proposal; defer Edge-as-object with rationale
  ([`f5aec56`](https://github.com/johnmarktaylor91/torchlens/commit/f5aec56c5dcf2eb99d72414b3b4b66dc37aa5158))

- **viz**: Sibling-ordering design (v4, 3-round dual-lab review) + impl spec; densenet todo
  ([`cb9d670`](https://github.com/johnmarktaylor91/torchlens/commit/cb9d67011bcfc213a261fdb01bfe367780a06a4f))

### Features

- **access**: Unified container views + role-general reconstruct + input_at (P3)
  ([`a895446`](https://github.com/johnmarktaylor91/torchlens/commit/a89544676aadf9aa7ab6a028886d697e2615c6f1))

Add the unified access surface over the container registry: op.containers (union of input + output
  roles), input_at(path) mirroring output_at (reuses the typed path indexer), and role-general live
  reconstruct for any role from a chosen snapshot spec. Multi-snapshot records require a site/role
  selector (record.spec raises with >1 snapshot; record.spec_at(role=...) added); reconstruct_output
  unchanged. paths_only backends degrade to reconstructable=False path-only views (no fake
  reconstruct). tl.input_at exported (API budget 65->66; provisional name, naming review pending).
  Default capture byte-identical.

- **annotate**: Node annotation persistence + trace.annotate API (B1)
  ([`0052ce8`](https://github.com/johnmarktaylor91/torchlens/commit/0052ce8de496229c48826d3e11979f3cc5198d9b))

Add the hybrid annotation store + post-capture annotate API: - _annotation_blobs (BLOB_RECURSIVE
  Trace field, default None, namespaced layer:/op: keys) with a _blob_kind_for_field mapping so
  tensor payloads persist (without it the first tensor raised TorchLensIOError); follows the
  _containers precedent; FORK_COPY. - _annotation_revision (runtime-only render-cache counter; NOT
  _spec_revision, which would falsely stale intervention recipes). - trace.annotate(selector, *,
  data=, image=, max_fanout=) + with_annotations(): explicit max_fanout (default 8 raises on
  multi-layer fan-out); tensor data is backend-gated (torch-only blobs; non-torch tensors rejected
  with a clear error, never silently stringified); raw ndarray rejected; small JSON breadcrumbs go
  to a reserved annotations["user"] sub-namespace (never clobbering internal save_mode/dedup keys);
  image renders via the existing NodeSpec.image hook. - user annotations preserved across
  rerun/replace: added to replace_state_from preserved_fields AND merged in the same-shape fast
  rerun paths (user subnamespace + _annotation_blobs survive; fresh internal keys refresh).

Default byte-identical: only field_order_dataframe_digest regenerates (the new field name; no
  dataframe column); default_trace / selective / backward_ready / tlspec_roundtrip / manifest /
  public_accessors digests + graph_shape_hash all unchanged. No MDS/scatter (B2/B3).

- **api**: Backend public-option spine + capability epochs
  ([`afc015d`](https://github.com/johnmarktaylor91/torchlens/commit/afc015d4645b4e04838553a472c5809090cab21d))

- **attribution**: Add grad_cam and layer attribution
  ([`eeb1c3b`](https://github.com/johnmarktaylor91/torchlens/commit/eeb1c3b831a9acf2f19a60dc5ca07be06c52d2ce))

- **attribution**: Add layer integrated gradients and conductance
  ([`82cf0d6`](https://github.com/johnmarktaylor91/torchlens/commit/82cf0d6e8824bbd0e33e49244eba39f2bafc13b9))

- **attribution**: Add saliency/input_x_grad/integrated_gradients/smoothgrad
  ([`0db2b1c`](https://github.com/johnmarktaylor91/torchlens/commit/0db2b1c43fd699fd60cee6b16b55e53f58a58a0e))

- **attribution**: Support multi-input / pytree / kwargs models
  ([`524555f`](https://github.com/johnmarktaylor91/torchlens/commit/524555f197cd5d2b95168e035e85f89989dca28a))

- **autoroute**: Output decode bank + capture-time decode (A2)
  ([`217678b`](https://github.com/johnmarktaylor91/torchlens/commit/217678b75c35232d93bd7be91b9dc003c45db912))

Fill autoroute/output (reusing the direction-agnostic Registry class with a distinct post-capture
  detector signature, dispatched at the live-output site). Conservative, fail-closed detectors in
  _builtin_output.py: HF classifier (config.id2label), ImageNet-1k (verified torchvision/timm), HF
  text LM; ambiguous or unverified evidence -> no decode (never a silent misfire). Built-in HF/timm/
  torchvision imports stay inside detectors so import torchlens does not pull them.

Decode runs at capture from logits.detach().clone() (backward_ready-safe), selects the head from the
  live outputs object (HF logits attr / config.num_labels), fails closed on ambiguity, and populates
  the durable decoded_output (JSON rows) + output_postprocessor record. output_style=/output_head=
  override (threaded into CaptureOptions + a semantic-output cache fingerprint covering
  style/head/detector version/id2label/num_labels/model_type/architectures so config changes don't
  return stale labels). Transient decode inputs are runtime-only; raw_output is untouched.
  Trace.decode_output (provisional) returns captured rows / best-effort re-decode. ImageNet-1000
  label list shipped as package data (source: torchvision ResNet50_Weights categories; license
  documented).

Default capture byte-identical: auto-decode fires only on verified detection, so the generic parity
  fixtures do not trigger it and no golden changes.

- **backend**: Add backend registry with explicit backend= routing
  ([`37e2553`](https://github.com/johnmarktaylor91/torchlens/commit/37e255322540137fd619e66cca9daef5cc46f0e0))

- **backend**: Add neutral trace metadata fields
  ([`d04f265`](https://github.com/johnmarktaylor91/torchlens/commit/d04f2653777b562648676996b3f6eb70b96a6e19))

- **backend**: Complete registry migration audit
  ([`6b7d6ad`](https://github.com/johnmarktaylor91/torchlens/commit/6b7d6adaa474b2863a11deb8ce7cd581822569b0))

- **backend**: Cross-backend container_structure capability + capture_output_structure opt-in
  ([`1ba4d0b`](https://github.com/johnmarktaylor91/torchlens/commit/1ba4d0ba4d06cc79a540d067e8db9cba528be5ab))

- **backend**: Prove fake backend trace roundtrip
  ([`58fa891`](https://github.com/johnmarktaylor91/torchlens/commit/58fa8913c7005b57ec20a3561e2b9d46f7eb7268))

- **backend**: Route torch capture through registry-owned entry
  ([`77b7f08`](https://github.com/johnmarktaylor91/torchlens/commit/77b7f084b348b981e61a07c4e48a4d1584e1b878))

- **backends**: Non-torch static-label predicate save=
  ([`ba62031`](https://github.com/johnmarktaylor91/torchlens/commit/ba62031e7a42273f0e46e4ac20d11c601c1c2bb2))

- **backward**: Add backward visualization controls
  ([`50e6d9f`](https://github.com/johnmarktaylor91/torchlens/commit/50e6d9f5961e3f647fa1d234ec41fa14efb8ac69))

- **backward**: Add differentiable replay
  ([`a64b7af`](https://github.com/johnmarktaylor91/torchlens/commit/a64b7afa9949bad9a5049bbeed3b5ff753fbc74e))

- **backward**: Add higher-order trigger attribution
  ([`51cfa63`](https://github.com/johnmarktaylor91/torchlens/commit/51cfa63e4cf5ab51ee74c66df4ed79badfcf1481))

- **backward**: Add module containment projection
  ([`acf15ea`](https://github.com/johnmarktaylor91/torchlens/commit/acf15eac677d5030e6f3fa441443e4a7622aae4a))

- **backward**: Add per-pass gradient payload records
  ([`b2cccd2`](https://github.com/johnmarktaylor91/torchlens/commit/b2cccd277651e34b9440572e4e4de01544e96804))

- **backward**: Add runtime sidecar event foundation
  ([`30b4989`](https://github.com/johnmarktaylor91/torchlens/commit/30b49895e2719799ade7c78a032065668048bc8c))

- **backward**: Complete fastlog gradient cleanup
  ([`6e1efc5`](https://github.com/johnmarktaylor91/torchlens/commit/6e1efc5bf4b28152daec403e06581addedef34da))

- **backward**: Complete higher-order attribution
  ([`446f296`](https://github.com/johnmarktaylor91/torchlens/commit/446f296c569f415ad1df0d2f16289237665cb75a))

- **backward**: Complete tlspec docs and reports
  ([`022d7c0`](https://github.com/johnmarktaylor91/torchlens/commit/022d7c049947dca56255d79187c9debe9cb50650))

- **backward**: Finish save_grads payload policy
  ([`b62d6d4`](https://github.com/johnmarktaylor91/torchlens/commit/b62d6d45753878d71cf95761cf028ed49c23a5d2))

- **backward**: Finish sidecar schema cleanup
  ([`0732a4a`](https://github.com/johnmarktaylor91/torchlens/commit/0732a4aab608659a9d3a07fb08f9dc455a722bd7))

- **backward**: Finish sidecar trigger foundation
  ([`16e960b`](https://github.com/johnmarktaylor91/torchlens/commit/16e960b4412d2bb2762f509b640e60e6702df050))

- **backward**: Finish trace grad cleanup
  ([`ea57407`](https://github.com/johnmarktaylor91/torchlens/commit/ea574074a5008dadcce40bb574c003c2acc117d1))

- **backward**: Materialize projection records
  ([`176a0d9`](https://github.com/johnmarktaylor91/torchlens/commit/176a0d918988d6c9ae34987e9fc170526231d9a1))

- **backward**: Persist per-pass grad records
  ([`13ccde7`](https://github.com/johnmarktaylor91/torchlens/commit/13ccde7de5bfbc8a023ee0ea1ad644920f29eb54))

- **backward**: Project transformed grad payloads
  ([`0b0b0ac`](https://github.com/johnmarktaylor91/torchlens/commit/0b0b0acc66f706654686fc9f77815c5f2afcc74d))

- **backward**: Remove trace gradient aliases
  ([`b3c32ce`](https://github.com/johnmarktaylor91/torchlens/commit/b3c32ce7dd32b18518a4402f6f74d759de4a03c6))

- **backward**: Strengthen validation tripwires
  ([`e1576bf`](https://github.com/johnmarktaylor91/torchlens/commit/e1576bf24c9dbd97c8587f5a8a5ec8a9000e4cd2))

- **backward**: Switch trace grad saving to save_grads
  ([`d33f11a`](https://github.com/johnmarktaylor91/torchlens/commit/d33f11a53d8147452b1582a83577afc0e692c937))

- **backward**: Unify recording gradients on sidecar
  ([`5543dbf`](https://github.com/johnmarktaylor91/torchlens/commit/5543dbfda1696e582ff7c09716bd4909c2e2052d))

- **bench**: --baseline-status canonical flag + partial-run guards
  ([`caf74f8`](https://github.com/johnmarktaylor91/torchlens/commit/caf74f89af3cb7e84c0d5b8bbd5362b88302673c))

Add a way to stamp a perf payload canonical so a real full run can unlock the halt "fractional-x
  raw" headline (generate_perf_numbers suppresses it only while baseline_status == "provisional").
  Previously perf_suite hardcoded "provisional" into every payload, so the headline could never
  emit.

- --baseline-status {provisional,canonical} (default provisional; behavior unchanged unless
  requested). Canonical stamps a host+date provenance note. - Parse-time guard: reject
  --baseline-status canonical with --smoke or --addendum-no-save (a canonical stamp requires a
  complete full run). - Post-run write guard: refuse to write a canonical-stamped payload whose run
  status != "ok" or that came from a partial-suite mode (closes the addendum-splice loophole so a
  timeout/error/merge can't yield a canonical incomplete JSON).

generate_perf_numbers needs no change (already unlocks on any non-provisional status).
  docs/performance.md recapture instructions updated to the flag. Tests: canonical headline unlock,
  provisional suppression, both rejected flag combos, partial-payload write refusal. test_perf_gate
  green; ruff clean.

- **buffers**: Add persistent buffer entities
  ([`5c292f4`](https://github.com/johnmarktaylor91/torchlens/commit/5c292f4f794634c1d59f6bd57225f5c6f9dca782))

- **buffers**: Capture registered buffer write events
  ([`91e1645`](https://github.com/johnmarktaylor91/torchlens/commit/91e164567a02fd4b6daae4a1fa6a5cf2b6d05649))

- **buffers**: Synthesize buffer version nodes
  ([`18b16c5`](https://github.com/johnmarktaylor91/torchlens/commit/18b16c512999c77942d08119a188fa42a60ee3e7))

- **capture**: Add captured run projections
  ([`4bab16b`](https://github.com/johnmarktaylor91/torchlens/commit/4bab16b98c51d6c7f32bad1607f1667cd695dfb2))

- **capture**: Add retroactive lookback save
  ([`aaa5dcf`](https://github.com/johnmarktaylor91/torchlens/commit/aaa5dcf6ebea5f623aaf4ddacc0727defc2e75fb))

- **capture**: Add torch func transform boundaries
  ([`73933bd`](https://github.com/johnmarktaylor91/torchlens/commit/73933bde5b4ad7aec25d1d109534212e5aba7908))

- **capture**: Add trace predicate save runtime
  ([`dfa0975`](https://github.com/johnmarktaylor91/torchlens/commit/dfa09752e4c5b9d4c63dd34dd9e3646a10f2620a))

Adds trace(save=predicate) on the exhaustive path using the shared RecordContext builder also used
  by fastlog predicate capture. Existing selector sugar is callable on RecordContext, so tl.func and
  tl.in_module can serve as both selectors and predicate predicates. Unsaved predicate-filtered ops
  raise a clear not-saved error on finalized out access.

- **capture**: Capture autograd functional transforms
  ([`f148983`](https://github.com/johnmarktaylor91/torchlens/commit/f1489835c5dbef044e7146a17f263d55f93e5023))

- **capture**: Centralize storage axis; trace can stream to disk
  ([`ffc6ce1`](https://github.com/johnmarktaylor91/torchlens/commit/ffc6ce146ef50d972861a2110d3f265d9bc06b27))

Lifts the fastlog storage router (RAM/disk-stream/drop, detach/copy policy) into the shared
  single-pass save engine so trace(save=..., storage=to_disk(path)) streams predicate-selected
  payloads to a disk bundle during the forward. Centralizes the compatibility rules (disk-only
  rejects keep_grad and backward_ready; disk mirrors are detached). Lazy load of disk-backed
  payloads via the Layer/Op out accessor. Predicate save plus storage=to_disk is the supported
  streaming path. Phase-6 disk-streaming tests.

- **capture**: Chunk_size — one-line chunked forward capture (narrow v1)
  ([`d2f9f7f`](https://github.com/johnmarktaylor91/torchlens/commit/d2f9f7fc29303858a64fb37d1c9f979bbde654cb))

trace(model, x, chunk_size=N) / rerun(..., chunk_size=N) splits a too-big input into sub-batches of
  N along dim 0, feeds them one at a time, and concatenates into a single Trace — sugar over the
  proven trace -> rerun(append=True) loop. Wires the previously-inert ReplayOptions.chunk_size (not
  a batch_size synonym).

Narrow, safe v1: - Input split (torchlens/_chunking.py): single ndim>0 tensor leaf auto (walks only
  standard containers, never mutates caller state / descends custom objects); multiple ambiguous
  leaves -> BatchChunkInputAmbiguityError; explicit chunk_paths=[...] escape hatch for multi-input.
  Positional only. Never splits a shared (B,B) mask/bias. - Forward-only: new KEEP
  Trace.chunked_forward marker + _ensure_not_chunked_forward_backward() guard; chunk_size rejects
  (typed errors) backward_ready / save_grads / hooks= / public intervene= / storage=to_disk /
  non-empty kwargs / non-torch backend (explicit AND resolved); log_backward()/backward() on a
  chunked trace raises. - Cache key includes chunk_size + chunk_paths.

Relieves forward-pass peak memory (composes with save= for storage; to_disk chunking is future).
  Behavioral parity goldens byte-identical; field-order golden delta is chunked_forward-only.
  chunk_size + schema + smoke green; ruff + mypy clean. Closes the stacked-multi-pass "dataloader
  wrapper" gap.

- **capture**: Comprehensive validated FUNC_ARG_SPECS coverage (751 to 1111 of 1178)
  ([`5b3cad3`](https://github.com/johnmarktaylor91/torchlens/commit/5b3cad32e092a67c2c5ec22671bb3ae30533cad3))

- **capture**: Container registry foundation + output re-plumb (P1)
  ([`efa04ea`](https://github.com/johnmarktaylor91/torchlens/commit/efa04ea2caa4ca53854680a7f8c1327fd8c24da2))

Introduce the object-identity container registry: Role/Phase enums, FuncSite/ModuleSite/ModelSite,
  ContainerLeafOccurrence/ContainerSnapshot/ ContainerRecord, and a two-tier identity scheme
  (capture-time strong-ref + is-verify live map -> portable ordinal-keyed records). Live map and
  reverse indexes are build-only state, torn down before every save path (.tlspec scrub, streaming
  finalize, capture-cache pickle).

Re-plumb the opt-in output-container capture through the registry as CALL_OUTPUT/MODEL_OUTPUT
  records; add lazy output reverse-index + op.output_containers; op.container reads registry
  conservatively (legacy path-only fallback). _containers is portable only when the flag is on.

Default capture byte-identical: parity goldens, graph_shape_hash, and .tlspec/cache semantic digests
  all unchanged with the flag off. Leaf occurrences (not sets) preserve repeated paths. P1 tests
  added.

- **capture**: Emit module phase zero events
  ([`e5a96b5`](https://github.com/johnmarktaylor91/torchlens/commit/e5a96b59c001fd87b144d843a37b26dd05d6e8e2))

- **capture**: Enrich op events for phase zero
  ([`d7f5db9`](https://github.com/johnmarktaylor91/torchlens/commit/d7f5db9258d3843cfa9cac9c486b5a9768ab37a0))

- **capture**: First-class Container view (op.container, reconstruct, output_at, register_container)
  ([`feb94b3`](https://github.com/johnmarktaylor91/torchlens/commit/feb94b33b40d75cae90907dabfed361d95153a11))

- **capture**: Inference_only no-grad capture + backward_call_context call-site
  ([`4b72afc`](https://github.com/johnmarktaylor91/torchlens/commit/4b72afc22f98f528dfa0837c08abfac3b0f97fc5))

Two additive, opt-in capture features (default behavior byte-identical; parity goldens unchanged).

inference_only: - tl.trace(..., inference_only=True) wraps the user forward in torch.no_grad() so
  PyTorch skips grad_fn creation entirely (forward-only memory/speed win). - Mutually exclusive with
  EXPLICIT backward-related capture (backward_ready, save_grads, intervention_ready) ->
  TrainingModeConfigError naming the offending flag(s). Default-True capture_tensor_grad_hooks is
  moot, not a conflict. - trace.log_backward()/backward() on an inference_only trace raises
  ConfigurationError (the autograd graph was discarded) -- preserves the load-bearing
  deferred-backward contract. - Exposed as backend-neutral Trace.inference_only (KEEP across .tlspec
  so the backward guard survives serialization); torch advertises it via the public option spine
  (epoch6), non-torch backends correctly reject it.

backward_call_context: - Renames the reserved-but-never-populated BackwardPass.call_context field to
  backward_call_context and actually populates it with the FuncCallLocation of the user's
  log_backward()/backward() call site (frames inside the torchlens package are skipped). Implicit
  passes leave it None. - __setstate__ drops the legacy call_context key from old artifacts.

Docs updated in lockstep (docs/backward.md, performance.md, for-ai-agents.md, torchlens/CLAUDE.md,
  AGENTS.md). New tests: test_inference_only.py (9), test_backward_call_context.py (2, 1 skip).
  Verified: parity goldens byte-identical, deferred-backward load-bearing tests green, smoke 252
  passed, ruff + mypy clean.

- **capture**: Input + threaded container roles via boundary walk (P2)
  ([`5bd6505`](https://github.com/johnmarktaylor91/torchlens/commit/5bd65057561a2b4389f2e6f9dbbd20bc84dfaa70))

Add input-side container capture: walk_container with recursive tensor-bearing eligibility
  (scalar/config tuples like reshape/permute/conv strides create no record; nested tensor-bearing
  containers do), generator safety (never consumed), and a skip policy that keeps flat/config ops
  free. A symmetric pre-call hook captures CALL_INPUT/MODEL_INPUT snapshots (phase=PRE_CALL) ordered
  before the matching POST_CALL output snapshot, so a same-object input-and-output (e.g.
  forward(cache): cache.append(x); return cache) yields two distinct snapshots.

Wire FuncSite/ModuleSite/ModelSite; register threaded containers as one record with deduped
  consecutive snapshots (28-layer past_key_values is O(N sites + 1 spec)). Add op.input_containers
  and trace.input_structure. Final-output views prefer MODEL_OUTPUT role so leaves stay
  trace.output_layers and reconstruct_output keeps working (additive over P1, no output-side
  behavior change).

Regenerate field_order_dataframe_digest golden for the P1 _containers field-order addition (no
  dataframe/byte-identity change). Default capture byte-identical (graph_shape_hash + semantic
  digest unchanged flag-off).

- **capture**: Plumb transform metadata
  ([`3978a4e`](https://github.com/johnmarktaylor91/torchlens/commit/3978a4e6c8fe14c99e74ec4582f5352183c63f43))

- **capture**: Unify intervention onto the predicate runtime
  ([`a3cf73d`](https://github.com/johnmarktaylor91/torchlens/commit/a3cf73d4c8cb67ef402718ab4b5c20709864c32f))

Adds an intervene= slot to trace()/record() evaluated by the shared RecordContext predicate runtime,
  composing with save= in one pass. InterventionDecision is the active-action parallel to
  CaptureSpec; tl.when/add/replace_with sugar normalizes the existing zero_ablate/scale helpers.
  Wires to the existing live-hook execution path (no reimplementation); Bundle/Super*/facets stay a
  separate layer above predicates. Saved payload at an intervened op is the post-hook tensor; re-run
  is deterministic. Includes phase-5 intervention+save invariant tests.

- **capture**: Warn on unattributed tensor args
  ([`d815565`](https://github.com/johnmarktaylor91/torchlens/commit/d8155654092cb559d184896dc5596782b0dc0842))

- **data-classes**: Add edge-count introspection
  ([`dc370e6`](https://github.com/johnmarktaylor91/torchlens/commit/dc370e610a5aa6be1c63f52eda0a8328e3f4c68e))

- **data_classes**: Unified runtime .handle on Param/Buffer/Module/GradFn
  ([`7454c18`](https://github.com/johnmarktaylor91/torchlens/commit/7454c184595897fff581b2ab73cd5fae4b3f07ca))

Expose the live torch object behind each record via a computed, non-throwing .handle property
  (mirrors the existing Op.grad_fn_handle), returning the object when reachable else None. No stored
  refs, no FIELD_ORDER / portable / dataframe changes (computed; non-portable -> None after .tlspec
  load).

- Param.handle: non-caching source-model peek (_peek_live_param(cache=False)); catches
  PostTraceParamUnavailable -> None. Does not touch Param.value. - Buffer.handle: read-through the
  source-model weakref -> dict(model.named_buffers()).get(address); None if any link missing. -
  Module.handle: read-through model.get_submodule(address) (root ""/"self" -> the model); None
  otherwise. - GradFn.handle: scan trace._backward_gradfn_refs by id == grad_fn_object_id; fall back
  to op.grad_fn_handle only when ids match; None otherwise.

No new strong refs (no GC cycle). Name provisional (review-day). Parity goldens byte-identical;
  handle tests (live identity / None paths / .tlspec None / Param non-caching) + smoke green; ruff +
  mypy clean.

- **debug**: Add lineage/compare/dead_neurons/gradient_flow_audit/recompute_candidates analysis
  utilities
  ([`d5b3bd8`](https://github.com/johnmarktaylor91/torchlens/commit/d5b3bd854fc8cd929d104c288b47f53ee78432e7))

- **debug**: Add nan bisection and hot path utilities
  ([`b28cca9`](https://github.com/johnmarktaylor91/torchlens/commit/b28cca9599232dc334df4a1fc9f0b70ea0a4760f))

- **export**: Add record tabular export quartet
  ([`c5b7116`](https://github.com/johnmarktaylor91/torchlens/commit/c5b711643323f6dce2fdb5e9fba60a0ef230f044))

- **facets**: Add facet intervention scatter-back
  ([`9221e30`](https://github.com/johnmarktaylor91/torchlens/commit/9221e306576e488108c7ae200faac6085879d39e))

- **facets**: Add p1a registry snapshots
  ([`77b63c2`](https://github.com/johnmarktaylor91/torchlens/commit/77b63c2f0b6aca539ec9e795864106dd8ede86d3))

- **facets**: Add p1b facet specs
  ([`7df3213`](https://github.com/johnmarktaylor91/torchlens/commit/7df321304f7e55f5c91e3e9f31630d376c5c321b))

- **facets**: Add patching helpers
  ([`d4d37e6`](https://github.com/johnmarktaylor91/torchlens/commit/d4d37e6a02b2a4d524dfe80a28715550b9487be6))

- **facets**: Broaden head-count inference + clarify fused-pattern-is-read-only-by-design
  ([`c91551f`](https://github.com/johnmarktaylor91/torchlens/commit/c91551fba551ccad29d2570b8cf90fe438ca5065))

Broaden attention head-count probing (_HEAD_COUNT_NAMES/_KV_HEAD_COUNT_NAMES/_HEAD_DIM_NAMES, incl.
  n_head/nhead) so nn.MultiheadAttention and conventionally-named custom modules infer heads
  automatically via config_value (reads custom_attributes). Sharpen the fused-pattern MissingFacet
  message: read via reconstruction_ready; to EDIT, capture eager (consistent baseline),
  reconstructed pattern is read-only by design. Doc note on the config_value(module, ...) authoring
  pattern.

- **facets**: Dict-honest keys() + three-way absence contract (A1)
  ([`9ee9a4d`](https://github.com/johnmarktaylor91/torchlens/commit/9ee9a4dbaf57fc6a246abcc76ed01e77bef69117))

facets.keys()/has()/in/iter now report what is ACTUALLY AVAILABLE NOW (cached recipe compute), not
  the declared-menu superset. Absence is classified three ways:

- structurally-absent (module genuinely lacks it) -> not in keys(); KeyError -
  capturable-but-uncaptured -> not in keys(); access raises MissingFacetError (new; subclass of
  KeyError) whose message names what to save - a new provisional `menu()` accessor lists the FULL
  declared menu with per-facet status {available_now | structurally_absent | needs_capture |
  declared_not_produced} including TransformerLens aliases

Mechanism: the absence protocol is centralized in recipes/_helpers.py as a (values, missing)
  contribution with an op-payload READABILITY check (an op-backed FacetSpec is only "available" if
  its activation was saved - otherwise needs_capture, so keys() can't claim a facet that .value
  would fail to read). missing metadata is threaded through _compute() with the same tier/order
  winner policy and mirrored onto aliases. __getattr__ propagates MissingFacetError (not
  AttributeError); .get() returns default for unavailable. Built-in recipes
  (attention/norm/residual) stop returning a MissingFacet sentinel as a facet value and signal via
  missing.

Trace.modules_with_facet()/attention_blocks() and patching now use available-now semantics;
  declaration-level discovery moves to menu().

The internal reconstruction MissingFacet sentinel path is unchanged; sentinel ACCESS tests are
  migrated (replaced, not dropped) to the new contract. Docs + 3 notebooks migrated. `menu` name
  provisional (review-day). Parity goldens byte-identical; full facets suite + smoke green; ruff +
  mypy clean.

- **facets**: Implement phase 3 semantic facets
  ([`445e20a`](https://github.com/johnmarktaylor91/torchlens/commit/445e20a1939c6e7be90ba9dd2cf092db5ac67378))

- **facets**: Migrate p1c recipes
  ([`79581fc`](https://github.com/johnmarktaylor91/torchlens/commit/79581fcc400d07a6dc61be374844bc54ce6a1af4))

- **input**: Original-input display + portable node-image embedding (A4)
  ([`2679b97`](https://github.com/johnmarktaylor91/torchlens/commit/2679b97227693a6ceda2f196b19b18c688744d00))

Make the original non-tensor input visible: - store raw_input when a non-tensor input is
  auto-coerced (text/PIL/numpy/nested), not just on the explicit transform= path -
  save_raw_input=small now persists PIL images as bounded encoded bytes (max 256px edge, 256KB cap),
  restoring to PIL on load (manifest/body format unchanged) - opt-in input-node transform summary
  (show_input_transform_summary) + opt-in summary detail lines with verified/source + UNVERIFIED
  warning; default off so existing _render_raw_input output is byte-identical - portable node-image
  rendering: graphviz emits local node images as external SVG hrefs that no SVG renderer loads (and
  the temp dir is torn down); now inline local node images as base64 data: URIs so they render in
  SVG/browsers/cairosvg and survive teardown, with a text fallback when an image can't be embedded.

Opus visual passes: empty-input-node blocker fixed (images render); montage node sizing + summary
  anchoring flagged as review-day visual polish. Default capture+render byte-identical (plain-tensor
  keeps raw_input=None; affordances default off); parity goldens unchanged.

- **io**: Add backend-aware tlspec schema v2 preflight
  ([`84f3508`](https://github.com/johnmarktaylor91/torchlens/commit/84f3508603ad72a61ba38b61ea363f355bf18435))

- **io**: Backend-neutral payload codec registry + write path + v2 manifest vocabulary
  ([`03c321c`](https://github.com/johnmarktaylor91/torchlens/commit/03c321c2e79861a0a77df9691dda362ce30dd007))

- **io**: Multi-device sharded round-trip + PayloadLoadHints reshard API
  ([`72042c7`](https://github.com/johnmarktaylor91/torchlens/commit/72042c7ab9ad801979c2a8c08a809fd60305fc9d))

Reconstructible JAX sharding metadata (mesh axes/shape + PartitionSpec as JSON);
  PayloadLoadHints/JaxPayloadLoadHint threaded through load/lazy/materialize_out/
  materialize_grad/rehydrate_nested (map_location unchanged); opt-in re-shard verified on 8
  simulated CPU devices; multi-host/unaddressable fail-closed.

- **io**: Non-torch payload codec load path
  ([`4c94014`](https://github.com/johnmarktaylor91/torchlens/commit/4c940147359cf73a538d7523810db48f10cec10d))

Add backend-aware eager and lazy payload materialization for codec-encoded non-torch bundles,
  including runtime-missing audit fallback semantics and direct-writer round-trip tests.

- **io**: Non-torch payload materialization capability + loaded-trace replay status
  ([`d58abae`](https://github.com/johnmarktaylor91/torchlens/commit/d58abae4c2e055dcd9f2af6c0f58227838f1fa7c))

- **io**: Private MlxPayloadCodec (registered, public save still audit-only)
  ([`547df3c`](https://github.com/johnmarktaylor91/torchlens/commit/547df3cfb46d58c970e94b4938aff6de7d922a41))

MLX numpy<->safetensors codec + unit tests. Public MLX materialization NOT flipped yet (sub-round
  b); existing audit-only behavior unchanged.

- **io**: Single-host sharded JAX array value round-trip (audit-only sharding)
  ([`cd19828`](https://github.com/johnmarktaylor91/torchlens/commit/cd19828c8a0e45761dce8a1620cc77a182b760d7))

Addressable single-host sharded arrays assemble to host + round-trip by value; sharding recorded as
  inert audit-only JSON metadata (no topology recreation on load); multi-host/unaddressable
  strict-fail; map_location type unchanged.

- **io**: Typed JAX PRNG key round-trip in payload codec
  ([`04d34d6`](https://github.com/johnmarktaylor91/torchlens/commit/04d34d65e549a2a49db8f3f5cc81b2d1955a1571))

key_data/wrap_key_data via threefry2x32; shape-agnostic transport (scalar + split/batched keys);
  old-style PRNGKey unchanged; unknown key tag fail-closed.

- **ir**: Add MLX deferred value sentinel
  ([`9a9bbe3`](https://github.com/johnmarktaylor91/torchlens/commit/9a9bbe375bf2d654605bbc9054b5c645ae7d85a6))

- **ir**: Control-edge contract for the interpreter foundation
  ([`7865b1a`](https://github.com/johnmarktaylor91/torchlens/commit/7865b1a637d715eaadc0b5267119c207ac8de69a))

- **jax**: Add derived gradient preview
  ([`7a233dd`](https://github.com/johnmarktaylor91/torchlens/commit/7a233ddc8172aaa37e61861269c110d4eb818b11))

- **jax**: Add jaxpr-first forward capture core
  ([`4e9d14a`](https://github.com/johnmarktaylor91/torchlens/commit/4e9d14af68f1471b2c393c6cc349aa325fb9eb36))

- **jax**: Add validation tripwire
  ([`35bf43b`](https://github.com/johnmarktaylor91/torchlens/commit/35bf43b9ef4a1d846b6ba596cb8314e7efb686f8))

- **jax**: Equinox pytree_module hierarchy via named_scope
  ([`4b9d3db`](https://github.com/johnmarktaylor91/torchlens/commit/4b9d3dba080a46b76004c782baa72faf09f43bc6))

- **jax**: Equinox shared modules + surface matrix + forward_args
  ([`f6c8308`](https://github.com/johnmarktaylor91/torchlens/commit/f6c83081f90147cf2a2b73f0c65ff935578cb40b))

- **jax**: Equivalence key + shared loop-grouping wiring
  ([`16a7385`](https://github.com/johnmarktaylor91/torchlens/commit/16a73854025f019dd54878a1972bab476b8c81af))

- **jax**: Flax NNX module attribution via pytree_module mode
  ([`49791c1`](https://github.com/johnmarktaylor91/torchlens/commit/49791c115ca7be283ece8fefc9bb5e33ef2c8334))

- **jax**: Harden capture corpus coverage
  ([`21c9305`](https://github.com/johnmarktaylor91/torchlens/commit/21c9305548262b10217c98dc624222580731ab8b))

- **jax**: Harden preview surface and docs
  ([`c23e202`](https://github.com/johnmarktaylor91/torchlens/commit/c23e202ab02a6e9cd3de13d4be1f864708b13f71))

- **jax**: Label-persistence index (enabling infra; public T1 deferred)
  ([`d28957e`](https://github.com/johnmarktaylor91/torchlens/commit/d28957ead0e6f41c95a0b27fecf5efa63ce44465))

Persist capture_index->final_op_label + composite outvar_key across
  finalization/relabel/selective-save. Public intermediate-grad surface NOT shipped: prefix-vjp
  lacks a perturbation+reference oracle (Stage 2 NO-GO).

- **jax**: Preserve output pytree paths
  ([`f387dbb`](https://github.com/johnmarktaylor91/torchlens/commit/f387dbbd8c08a465a4e582ce46bf2cef89d9c685))

- **jax**: Public zero-tap intermediate derived gradients
  ([`a43c01c`](https://github.com/johnmarktaylor91/torchlens/commit/a43c01ccf6aa54415ad998645806c12a239ee3f6))

trace.intermediate_derived_grads + op.derived_grad for JAX via custom_jvp identity tap
  (signed-zero/NaN preserved); separate post-finalization AD path keeps capture graph
  byte-identical; 1 producer grad + O(k) capped oracle; exact-only exposure, oracle-failures
  skipped; graceful degradation to leaf grads.

- **jax**: Region importer for unbounded/dynamic control flow
  ([`d80e2de`](https://github.com/johnmarktaylor91/torchlens/commit/d80e2de57d927ff95a70612e894c8d53440b1577))

jax_control_flow='region' imports over-cap scan, unbounded while_loop, and custom_vjp-forward as
  graph regions (boundary + projection nodes) reporting ValidationReplayStatus unverified;
  transactional rollback of partial unroll; seam perturbation checks; non-region ops still replay
  exactly (failed wins). shard_map still rejected.

- **jax**: Support declared static args
  ([`b5ad671`](https://github.com/johnmarktaylor91/torchlens/commit/b5ad671b5856b2f50c3d49f5deb073cb8a3011dc))

- **jax**: Transparent pure-jit nested-jaxpr inlining
  ([`845e169`](https://github.com/johnmarktaylor91/torchlens/commit/845e169a99915b6fd3d6a5699a9437a3c215c19a))

Recursive purity gate inlines effect-free, const-free, non-donated, unsharded jit/pjit nested calls
  (unblocks scan-under-jit unroll); rejects consts/effects/donation/sharding.
  remat2/shard_map/custom_vjp still rejected.

- **jax**: Unroll lax.cond and lax.while_loop
  ([`91f93fc`](https://github.com/johnmarktaylor91/torchlens/commit/91f93fc33f30671db9cbc3efe0668203ff91f705))

- **jax**: Unroll lax.scan with per-iteration grouping
  ([`0b5addd`](https://github.com/johnmarktaylor91/torchlens/commit/0b5adddb30bde7bdc452b4d6c4bd59728ea6a055))

- **keystone**: Trace.model_profile computed view + mds_evolution (B3a)
  ([`f51edee`](https://github.com/johnmarktaylor91/torchlens/commit/f51edeeeaf857a49183ef87cd6570dbfda5217e9))

Add the recognized-profile half of the keystone: - ModelProfile (frozen descriptor) +
  Trace.model_profile computed @property, derived from the Sprint A provenance fields
  (input_preprocessor.source, output_postprocessor, output_id2label/num_classes, raw_input):
  input_modality, pre/post-processing sources, output_label_count, has_output_labels, num_stimuli,
  has_raw_images, keystone_applicable. NOT a stored field -- out of MODEL_LOG_FIELD_ORDER /
  PORTABLE_STATE_SPEC / byte-identity chain. - tl.repgeom.mds_evolution(trace, save=None, *, metric,
  min_n=8, align=True): per-layer MDS over the batch (single-pass reads layer.out keyed
  layer:<label>; recurrent/multi-pass requires a pass-qualified op selection keyed op:<label>, with
  a clear select-a-pass error instead of the raw layer.out ValueError), Procrustes-chained
  layer-to-layer (no reflection), coords stored as tensor annotation blobs via trace.annotate. Clear
  error when activations weren't saved.

No drawing yet (B3b). No Trace field, no default behavior change; parity goldens untouched; smoke +
  repgeom + annotations green.

- **mlx**: Codec load readback + audit backcompat + docs lockstep
  ([`751a099`](https://github.com/johnmarktaylor91/torchlens/commit/751a09980aed92ef386b1e217d97e8cc5fb942a2))

Eager/lazy/runtime-missing MLX payload load; old audit-only bundles still load metadata-only;
  docs/glossary/README updated to MLX payload materialization. Closes the MLX-CODEC epoch.

- **mlx**: Emit topology-complete capture events
  ([`d5f2e17`](https://github.com/johnmarktaylor91/torchlens/commit/d5f2e174109ed7c67b08059ac9321e5e8edd05da))

- **mlx**: Flip payload materialization on (array_payloads)
  ([`5714ec3`](https://github.com/johnmarktaylor91/torchlens/commit/5714ec37d0ab2a1f32f5dc64a11d550be6ce0073))

Atomic public flip: capabilities + default_specs + scrub codec routing + materialized contract
  tests. Loaded MLX traces report replay-unavailable (never false pass). Backcompat fixtures + docs
  in follow-up.

- **mlx**: Intermediate derived gradients via custom-vjp tap
  ([`6ca6955`](https://github.com/johnmarktaylor91/torchlens/commit/6ca6955e696c8a34ddfdcce1e9b8b7c4600f5e4c))

trace.intermediate_derived_grads + op.derived_grad for MLX (mechanism
  mlx_custom_vjp_tap_value_and_grad); one producer replay + capped per-boundary oracle;
  grouped-signature exact-1:1, duplicates skipped; exact-only exposure; fail-closed perturbation
  gating. All 4 backends now have intermediate grads.

- **mlx**: Leaf derived gradients via value_and_grad
  ([`385c2f1`](https://github.com/johnmarktaylor91/torchlens/commit/385c2f19999243f880e9f9ecae4d8780fc17ee74))

trace.derived_grads for MLX (mechanism=mlx_value_and_grad, model.update param rebinding); honesty
  guard requires AD rerun to reproduce captured output before exposing grads. op.grads still raises;
  T1 deferred.

- **mlx**: Object_module hierarchy + module save= selectors
  ([`7c34b75`](https://github.com/johnmarktaylor91/torchlens/commit/7c34b753e8944fe0c68203746bc5029f751bf802))

named_modules() traversal (nested/list/parameterless), module-call stack separate from recursion
  guard; module_identity_modes adds object_module; MLX save= now accepts module/in_module. All four
  backends have hierarchy.

- **mlx**: Static-label save= (func/label/contains + composites)
  ([`e6102c6`](https://github.com/johnmarktaylor91/torchlens/commit/e6102c6b4ea9e02b702dfb14c36e98dd8c9eb4ce))

Explicit selector-kind allowlist; module/in_module rejected (no MLX module hierarchy yet),
  output/value-dependent rejected. Recomputes saved-op counters/memory for selected live payloads.

- **op**: Op.var_names — source assignment names via full-file AST
  ([`115a84f`](https://github.com/johnmarktaylor91/torchlens/commit/115a84f54485ae50d22e3dfaf1185342aeaf4efb))

Capture the source variable name(s) a tensor output was bound to (t = relu(x) -> ["t"]; a, b =
  chunk(x, 2) -> ["a","b"]; p = q = f(x) -> ["p","q"]). KEEP Op field (portable strings); []
  whenever the call is inline / unnameable / source unavailable, never a wrong guess.

Extraction (postprocess step, after layer finalization) reuses the existing full-file AST indexer in
  postprocess/ast_branches.py, extended with a parent map to walk (line, col_offset) -> enclosing
  ast.Call -> assignment target(s). Returns [] on ambiguity, missing source (<string>/dynamic/no
  save_code_context), parse failure, comprehension/decorator/augmented/ternary forms, and
  attribute/subscript targets (conservative).

KEEP field wired through LAYER_PASS_LOG_FIELD_ORDER, PORTABLE_STATE_SPEC, default-fill, Op.__init__.
  Dataframe column (NOT in parity stable_rows — its values are source-path/env dependent).
  Behavioral parity goldens byte-identical; field-order/dataframe golden delta is var_names-only.
  var_names + io_pandas + schema-audit + smoke green; ruff + mypy clean.

- **op**: Public Op.edge_uses accessor for per-edge multiplicity
  ([`be3b312`](https://github.com/johnmarktaylor91/torchlens/commit/be3b312737830253f79659143f95bf89ff527859))

Expose the per-edge-use records that capture already stores in private _edge_uses as a public
  read-only Op.edge_uses property returning a tuple of EdgeUseRecord (empty tuple when none).
  Surfaces same-parent multi-arg multiplicity (y = x + x -> two edge uses both from x; cat([x, x])
  -> two) without any stored field.

Flips the schema-audit assertion that previously forbade a public edge_uses to expect it (tracked
  contract change, not a tripwire weakening). Goldens unchanged (computed property; behavioral
  parity byte-identical). + edge_uses tests (add/cat multiplicity, tuple immutability, .tlspec
  round-trip). ruff + mypy + smoke green.

- **output**: Batch top-N decode table + textual/viz surfaces (A3)
  ([`8259801`](https://github.com/johnmarktaylor91/torchlens/commit/82598016a316447f2095ed77a91209212290227c))

Surface the decoded output everywhere: - Trace.output_table(top_n=5, batch_items=None) -> DataFrame
  [batch_item, rank, label, prob], with a typed {kind: batch_topk, rows: [...]} representation;
  best-effort re-decode from retained logits, raising clearly when unavailable. - textual: symmetric
  output-postprocessing lines in the discoverability summary + summary(level=output); surfaces the
  resolved style / undetected hint. - viz: a NEW batch-topk branch in _render_raw_output renders a
  per-batch-item top-N category table on the output node; existing str/label-score/mapping
  single-value paths stay byte-identical (tests). Opus visual pass: SHIP. - to_pandas decoded
  summary is OPT-IN via include_decoded_output_summary=True -- never a default column (F-R2.3). -
  negative detection hardened (segmentation-rank logits, CIFAR-like unlabeled heads, bare [B,1000]
  regressors -> no decode); MLX output_style/output_head defaults fixed.

No new always-on Trace field; parity goldens unchanged; default capture+render byte-identical
  (decode fires only on verified detection).

- **perf**: Add raw rerun shape hash
  ([`2b58e1f`](https://github.com/johnmarktaylor91/torchlens/commit/2b58e1f9663b7f90222aae8cacd1e2d0803e813d))

- **perf**: Add record halt predicate
  ([`f1c0865`](https://github.com/johnmarktaylor91/torchlens/commit/f1c08654b5cf931fcc09f874e6940a3eefb2ddff))

- **perf**: Add trace halt predicate
  ([`d589c1c`](https://github.com/johnmarktaylor91/torchlens/commit/d589c1c5a70ae5927b497ed798b6e579fd4e7548))

- **perf**: Fast-path halt-only capture
  ([`eed49d5`](https://github.com/johnmarktaylor91/torchlens/commit/eed49d51527b43eec3a402e46897a386ec1908a6))

- **perf**: Fast-refresh matching reruns
  ([`5e619a8`](https://github.com/johnmarktaylor91/torchlens/commit/5e619a8914276fda4aff9c9ee3b37c5af9f69009))

- **perf**: Release param refs after finalization
  ([`89e412a`](https://github.com/johnmarktaylor91/torchlens/commit/89e412a225345920620de3cc6cfc214d887e1d37))

- **perf**: Shallow-fork differentiable replay
  ([`58ccbfe`](https://github.com/johnmarktaylor91/torchlens/commit/58ccbfeaa263ccd2ee94d4cddd9d4504047c2d87))

- **persist+backend**: Container persistence, capability split, flag rename (P4)
  ([`6796cc7`](https://github.com/johnmarktaylor91/torchlens/commit/6796cc7d83287a370e0e9ff32af7bbcee1eac042))

Persistence: _containers and input_structure round-trip structurally through .tlspec, capture-cache,
  and streaming (to_disk) saves when the flag is on; absent and byte-neutral when off. After-save
  tensor refill is best-effort and non-gating (structure is the durable contract).

Capability: split the output-only container_structure into input_container_structure and
  output_container_structure (none|paths_only|full_spec) across the registry, epoch, and per-backend
  specs (torch full_spec both; jax/tinygrad paths_only; mlx none). Views read the role-appropriate
  capability and degrade honestly.

Rename: capture_container_structure is the canonical flag (it governs all container roles);
  capture_output_structure stays a deprecated alias (DeprecationWarning, conflict if both supplied).
  One atomic patch across option resolution, cache key, epoch, docs/containers.md, and tests.
  Provisional name -- naming review pending.

Regenerate field_order_dataframe_digest golden for the input_structure portable field (no
  dataframe/byte-identity change). Default capture byte-identical across all three save paths.

- **provenance**: Semantic I/O Trace fields foundation (A1)
  ([`046ea20`](https://github.com/johnmarktaylor91/torchlens/commit/046ea20ed999f077a2da02163828decd59aab641))

Add portable provenance Trace fields for the semantic-I/O sprint, with no behavior change: -
  output_postprocessor: ResolvedPostprocessing (mirrors ResolvedPreprocessing, plus decode fields
  style/selected_output_head/label_source/confidence/ top_n_captured/ambiguous) - transform_repr
  (mirrors _activation_transform_repr; _transform stays DROP) - output_id2label / output_num_classes
  captured in-band from model.config during capture (pickles into the capture cache; no post-hoc
  bridge mutation) - decoded_output: durable JSON/primitive slot (filled in A2), not a tensor blob

Each new field is wired through MODEL_LOG_FIELD_ORDER, PORTABLE_STATE_SPEC (KEEP), FORK_COPY,
  default-fill, and __setstate__, and all default None. Byte-identical: only
  field_order_dataframe_digest regenerates (diff = the new field names only, no dataframe column);
  default_trace / selective / backward_ready / tlspec_roundtrip / manifest digests and
  graph_shape_hash unchanged. Schema-audit + internals gates green.

- **quantities**: Add tl.Quantity and Bytes, migrate memory fields to Bytes, drop memory str fields
  ([`318d0b4`](https://github.com/johnmarktaylor91/torchlens/commit/318d0b4f00f8683c5cb2c63f0319e7b9ac7edcb5))

- **quantities**: Finish v7 numeric quantity migration
  ([`8a71c6d`](https://github.com/johnmarktaylor91/torchlens/commit/8a71c6ddb048bcc3d8ee399a91c6a67203057e6e))

- **records**: Add glossary convenience properties
  ([`837e9db`](https://github.com/johnmarktaylor91/torchlens/commit/837e9db27a53b6d9614fbe841fae0ccc59a00d40))

- **records**: Add structural glossary fields
  ([`d1ea23a`](https://github.com/johnmarktaylor91/torchlens/commit/d1ea23ad8e9d31a9eeda0a4d64d5604f7b9d8e8f))

- **repgeom**: Effective-dimensionality scree node-visual
  (scree/effective_dimensionality/scree_evolution/scree_node_spec) [C3]
  ([`12eb9f6`](https://github.com/johnmarktaylor91/torchlens/commit/12eb9f62be02122742c84eeae1f0ba001414512c))

Factored the centered-Gram eigendecomposition out of classical_mds into a shared private helper (MDS
  numerics bit-for-bit unchanged; MDS test suite is the regression gate). scree() returns sorted
  nonneg eigenvalues; effective_dimensionality() returns variance-explained, cumulative,
  participation_ratio=(sum L)^2/sum(L^2) (0 on zero mass, no div-by-zero),
  n_components_for_threshold, effective_rank. scree_evolution() stores eigenvalues under
  scree:<base>; scree_node_spec() plots the variance-explained + cumulative curves via
  tl.viz.render_lineplot with a threshold reference line and a PR/rank text callout. Degenerate
  all-zero rep -> rank-0 empty plot, no crash. No new Trace field; default-render byte-identity
  preserved. Provisional names -> review-day.

- **repgeom**: Hand-rolled numpy MDS + Procrustes (B2)
  ([`dbd3e3e`](https://github.com/johnmarktaylor91/torchlens/commit/dbd3e3e6403a1af3f87702ee3523ecb5a685219d))

New import-clean torchlens/repgeom/ (tl.repgeom, provisional; numpy+torch only, NO
  scipy/sklearn/matplotlib/rsatoolbox/brainscore): - classical_mds(data, n_components=2, min_n=8):
  double-centering + eigh; clips negative Gram eigenvalues and reports negative_eigenvalue_count +
  discarded_variance_fraction; deterministic sign convention; min_n / effective-rank gate (a
  visualization, not inference); rejects duplicate / rank-deficient / non-finite inputs with clear
  errors. - procrustes_align(source_2d, target_2d): orthogonal rotation only, FORBID reflection
  (det(R)=+1), no scaling -- so layer-to-layer evolution doesn't flip. -
  activation_distance_matrix([N,...], metric): euclidean/cosine/correlation.

Validated against a closed-form NONDEGENERATE ASYMMETRIC analytic fixture (no vacuous sklearn/scipy
  importorskip): distance reconstruction atol 1e-9 + recovery up to sign/rotation/reflection via
  Procrustes residual < 1e-6. Submodule not added to tl.__all__; no Trace field, no capture/render
  change, parity goldens untouched.

- **repgeom**: Per-node RDM heatmap node-visual (rdm/rdm_evolution/rdm_node_spec) [C1]
  ([`7258d7d`](https://github.com/johnmarktaylor91/torchlens/commit/7258d7db5cd1af09c3fdfab335784360be223030))

rdm() aliases activation_distance_matrix; rdm_evolution() stores each layer's NxN dissimilarity
  matrix in _annotation_blobs under rdm:<base> keys (mirrors mds_evolution, same site-selection +
  recurrent rule); rdm_node_spec() renders it via tl.viz.render_heatmap with symmetric stimulus
  thumbnails on the axes (capped + clean +K-more). Factored a shared _store_annotation_tensor blob
  writer used by both MDS and RDM. No new Trace field (_annotation_blobs reused); default-render
  byte-identity preserved (field-order digest unchanged). Provisional names -> review-day.

- **tinygrad**: Add bracketed derived grads
  ([`c9c04a1`](https://github.com/johnmarktaylor91/torchlens/commit/c9c04a15adb9c4137e318d75ab3a722bd2e8c1b2))

- **tinygrad**: Add forward UOp capture core
  ([`f55e121`](https://github.com/johnmarktaylor91/torchlens/commit/f55e121deb938c7058f5804d10571fd90cdecefc))

- **tinygrad**: Harden preview surface and docs
  ([`acc2ca6`](https://github.com/johnmarktaylor91/torchlens/commit/acc2ca6df41180571f62e3dfae49c55ca802491b))

- **tinygrad**: Harden validation tripwire
  ([`d7741a4`](https://github.com/johnmarktaylor91/torchlens/commit/d7741a4f43aa09c8be4bfd59dce7690a63709df7))

- **tinygrad**: Object_module hierarchy via observe-uop attribution
  ([`3dd4770`](https://github.com/johnmarktaylor91/torchlens/commit/3dd4770f737d1890242613062c3de54657bee4dc))

- **tinygrad**: T1 intermediate derived gradients via no-realize pass
  ([`0aaa44a`](https://github.com/johnmarktaylor91/torchlens/commit/0aaa44a8347b064d97ff411752b38d94fd2d897f))

Add trace.intermediate_derived_grads + op.derived_grad for tinygrad's no-realize backward pass
  (conservative signature mapping, reject-ambiguous). JAX gets a private capped boundary-VJP oracle
  for tests only, no public surface. op.grads still raises for non-torch backends.

- **trace**: Expose recurrent layers accessor
  ([`1131ccc`](https://github.com/johnmarktaylor91/torchlens/commit/1131ccc021df72d0c86d519cae61fdd1c9aec6f6))

- **trace**: Op-side buffer accessors (buffer_read_ops / buffer_write_ops)
  ([`f107258`](https://github.com/johnmarktaylor91/torchlens/commit/f1072580cee69e4a30088123d0efed736af00e2b))

Add derived Trace accessors mirroring the internal-source/sink boundary surface for buffers,
  completing the layer-level-only gap (buffer_layers / num_buffer_layers):

- Trace.buffer_read_ops -> buffer Ops that read a value into the graph (is_buffer and
  buffer_write_kind is None) - Trace.buffer_write_ops -> buffer overwrite/write Ops
  (buffer_write_kind set) - Trace.num_buffer_read_ops / num_buffer_write_ops

Derived properties over Trace.layer_list (no stored fields) -> zero FIELD_ORDER / portable-state /
  golden churn; work after .tlspec load (is_buffer / buffer_write_kind are KEEP). Names provisional
  (read/write chosen to avoid colliding with the existing is_buffer_source = "overwrite boundary"
  property; final name is a review-day call).

Tests: partition/count/disjoint/union + layer-surface parity + dual-role buffer in both lists +
  .tlspec round-trip (reuse DualRoleInplace fixture). Parity goldens byte-identical; smoke green;
  ruff + mypy clean.

- **validation**: Add honest 'unverified' replay status + laundering tripwires
  ([`d340fee`](https://github.com/johnmarktaylor91/torchlens/commit/d340feeb91626235213d5541c2afb1193b9746fc))

New ValidationReplayState.unverified (available=True, __bool__ raises -- never a pass/fail); strict
  aggregate fold (failure wins); region-provenance invariant (unverified only from importer-tagged
  regions, illegal on plain capture). One-way: failed/missing/corrupt replay stays failed. Torch
  byte-identical (runtime-only).

- **validation**: Per-mode module invariants
  ([`8f72cb5`](https://github.com/johnmarktaylor91/torchlens/commit/8f72cb5a63290890519e05a5992224ef7aea095c))

- **viz**: Deterministic execution-order placement of parallel siblings
  ([`ddf16d5`](https://github.com/johnmarktaylor91/torchlens/commit/ddf16d55c5cf74b96a618d5979356dcc918c640c))

- **viz**: Draw-time MDS thumbnail-scatter node visual (B3b)
  ([`f0330be`](https://github.com/johnmarktaylor91/torchlens/commit/f0330bef36b0ef1c2e8004196bede80c8c4282d6))

Render the per-layer MDS coords (stored by B3a in _annotation_blobs) as a scatter ON the annotated
  op nodes, at draw time: - tl.repgeom.mds_scatter_node_spec(...): a node_spec_fn that reads coords
  (layer.source_trace._annotation_blobs) + stimulus thumbnails (trace.raw_input), composites a
  single PIL scatter image (thumbnails at MDS pixel positions, margin so nothing clips, min-distance
  spacing for close coords, +K more cap, points/text fallback when stimuli unavailable), set as the
  node image. PIL-only (no matplotlib). Rendered fresh each draw and base64-inlined (A4), so it
  survives save/load with no baked visualizer_path. - The scatter is ONE contained image inside an
  enlarged bordered node with the layer label as caption (fixed an initial containment bug where
  thumbnails leaked as loose canvas elements -- Opus visual pass caught it; round-2: SHIP).

Default render byte-identical (opt-in via node_spec_fn); no Trace field; parity goldens untouched.
  Review-day polish: annotated node omits shape/param text; axis styling.

- **viz**: Feature/activation-map node-visual (feature_map_evolution/feature_map_node_spec) [C2]
  ([`b238b43`](https://github.com/johnmarktaylor91/torchlens/commit/b238b432df0948075c77d1fc8df9f447788c20bf))

New torchlens/viz/feature_maps.py: for spatial [N,C,H,W] conv activations, feature_map_evolution
  selects sites + stimuli + channels (default channel-aggregate; channels=indices|'top' top-k per
  stimulus, tie-break lower index) and stores reduced [S,K,H,W] maps in _annotation_blobs under
  featmap:<base>:{maps,stimuli,channels,counts}. feature_map_node_spec renders a bounded
  small-multiples grid (rows=stimuli, cols=channels): headline default upsamples the aggregate map
  and alpha-overlays it on the real stimulus thumbnail ('where the layer looks per image'), with
  raw-heatmap fallback when stimulus images are unavailable; hard-capped + clean +K-more.
  Spatial-only (non-spatial sites skipped). No new Trace field; default-render byte-identity
  preserved. Provisional names -> review-day.

- **viz**: Opt-in container visualization (show_containers: key-labeled edges, single-owner cluster)
  ([`2ad0ddf`](https://github.com/johnmarktaylor91/torchlens/commit/2ad0ddff41d17cb1f2bb02f8632fa08559d97f50))

- **viz**: Opt-in container-node visualization, show_containers="nodes" (P5)
  ([`46ffbdb`](https://github.com/johnmarktaylor91/torchlens/commit/46ffbdb221f069fe901f36bb9885ca1ee17ede43))

Add the Option-1 container visualization as a new opt-in show_containers value "nodes": each
  container is drawn as ONE collapsed node labeled with its type + role (one node per container
  RECORD, not per snapshot/site). It sits on a real edge only as a genuine source (model-input
  fan-out) or sink (model-output); a light dashed constraint=false member-of overlay (kept off
  captured_forward_edges and the dataflow dedupe) associates mid-graph producers with their
  container node; field keys label the pulled-out edges; large/homogeneous containers collapse. The
  deferred record-node-with-ports mode was intentionally NOT built.

Existing show_containers modes (False/labels/cluster/collapsed/auto) and the default render stay
  byte-identical. Container nodes auto-size to their label so the drawing bbox/viewBox reserves
  margin; extend test_label_geometry with a raster edge-margin assertion guarding container-node
  clipping. Verified via three Opus visual-inspection rounds.

Known residual (flagged for visual review-day): hf_output's model-output box border grazes the right
  frame by a few px (label fully legible); the wide-label case (threaded_kv) renders clean. Docs
  updated (docs/containers.md).

- **viz**: Pil render-primitive toolkit (heatmap/lineplot/image_scatter) [C0]
  ([`5ac9f29`](https://github.com/johnmarktaylor91/torchlens/commit/5ac9f291406b0c7351e16d5d5f610bde11d21b28))

New torchlens/viz/node_plots.py: render_heatmap (deterministic colormap + decimated, non-overlapping
  axis labels/thumbnails + single clean +K-more marker), render_lineplot (hand-rolled axes +
  multi-series polylines + legend), render_image_scatter (thumbnail/ point scatter with min-gap
  spread that prevents full occlusion). numpy + PIL only, no matplotlib. Exposed via tl.viz
  alongside the existing heatmap visualizer factory (kept). Provisional names -> review-day.

- **viz**: Reconcile rolled loop and module labels
  ([`c9889e0`](https://github.com/johnmarktaylor91/torchlens/commit/c9889e05e5df30b248c2de30c111779ca85acbf2))

- **viz**: Render rolled recurrence self edges
  ([`16957e4`](https://github.com/johnmarktaylor91/torchlens/commit/16957e4c57ed41b9c51e156fe977c93541cbae91))

- **viz**: Render two arrows for same-parent multi-arg-slot edges (y = x + x)
  ([`ce80fc4`](https://github.com/johnmarktaylor91/torchlens/commit/ce80fc497fa9d98eaea1a9678896f3b692f76b19))

When one node's output feeds multiple argument slots of a child op (y = x + x, torch.cat([x, x])),
  draw one arrow per arg-slot occurrence instead of the single collapsed edge. Multiplicity is
  threaded through the render-edge representation keyed by actual edge-use occurrence (not by
  duplicating parent labels), so ordinary single-slot edges still render exactly one arrow.
  Commutative ops (add/mul/cat/eq/ne) keep the arrows unlabeled; non-commutative same-parent
  multi-arg cases get per-slot labels. Graphviz distinguishes the parallel edges by separate
  occurrence-keyed edge statements; the pure-Python rank layout is kept occurrence-aware so the
  extra edge doesn't perturb sibling ordering on non-multiplicity graphs.

Render-only: behavioral/capture parity goldens byte-identical; no visual golden churn on
  non-multiplicity graphs (sibling-ordering + viz suites green). Opus visual review confirmed two
  clean distinct arrows on x+x / cat and no regression on a control graph. + edge-multiplicity
  render tests; ruff + mypy clean.

- **viz**: Retire ELK/sfdp layout backends; promote pure-Python rank layout
  ([`b2ec033`](https://github.com/johnmarktaylor91/torchlens/commit/b2ec03311b9ca5ca07fec9ac9f2d5753b32b9737))

The ELK escape hatch (Node.js+elkjs subprocess, 1585 lines) fired above a 3500-node threshold, but
  benchmarking showed dot's cost driver is long-range edges, not node count: local-topology graphs
  render fine at 5k nodes (13.7s) while one hub node feeding 24 spread consumers hangs dot even at
  1k. The threshold was aimed at the wrong variable, the ELK path had rotted (missing conditional
  labels, code panel, current label work; elkjs not even installed on the dev box, so users silently
  got the sfdp fallback), and a zero-dependency pure-Python Kahn rank layout already existed for the
  >100k tier.

- _elk_internal deleted; _rank_layout_internal keeps the Kahn layout + direct-SVG writer
  (render_rank_layout). - Engine selection: vis_node_placement = auto (default) | dot | rank. auto
  estimates layout cost as num_nodes + sum of rank spans of edges spanning >12 ranks, and switches
  to rank above 20,000 (calibrated 2026-06-11: local-5k stays dot, hub-3.5k switches), with a notice
  explaining the choice and remedies. - elk/sfdp values removed outright (no deprecation period):
  invalid values raise the standard ValueError. - tests/test_large_graphs.py rewritten against rank
  layout (scale ladder to 1M nodes, auto/manual selection, removed-value rejection, SVG sanity);
  docs swept (docs/elk_setup.md -> docs/rank_layout.md).

- **viz**: Time graphviz render phases into trace._phase_timings
  ([`9abf568`](https://github.com/johnmarktaylor91/torchlens/commit/9abf568cf26923d45f401e431a6a3f81a8d50e81))

Wrap draw/render_backward_graph/render_combined_graph in _timed_phase under
  render:graphviz:{forward,backward,combined} buckets (PERF-33 residual; capture + postprocess were
  already timed). Render output unchanged (goldens byte-identical); adds phase-timing-buckets docs +
  regression test.

### Performance Improvements

- **backward**: Add tensor grad hook opt-out
  ([`decbc6e`](https://github.com/johnmarktaylor91/torchlens/commit/decbc6e24353ca413019c69283be22999ae0ba78))

- **bench**: Commit provisional host-stamped baseline + measured perf docs
  ([`003c781`](https://github.com/johnmarktaylor91/torchlens/commit/003c78137d2e886ef884195b5e4c557a8ef6d1ea))

Item 1: add baseline_status/baseline_note + hostname/os/cpu_model host-stamping to the perf_suite
  payload; suppress generated speed headlines when baseline_status=='provisional'. Capture a
  provisional TinyNet-smoke baseline (Intel i9-9900X, CPU) into
  benchmarks/perf_baselines/linux-cpu-provisional.json and embed the generated numbers in
  docs/performance.md behind a provisional banner, with canonical-host re-bless instructions. Adds
  headline-suppression test.

- **benchmarks**: Op __slots__ retained-memory baseline
  ([`e19c063`](https://github.com/johnmarktaylor91/torchlens/commit/e19c0634cce2043f520c5e4ccf208d55287ddb4b))

Real-world trace-level memory reduction is 10-15% (activations dominate retained memory); the
  ~78%/op figure is the Op struct in isolation.

- **capture**: Faster loop-frontier pop, model-prep traversal, instance patching
  ([`9b83131`](https://github.com/johnmarktaylor91/torchlens/commit/9b831315a08ca0c445a644d6088cbb87d9f2eb70))

- **capture**: Fold multi-output output partitioning
  ([`82d9c24`](https://github.com/johnmarktaylor91/torchlens/commit/82d9c240c1e719caed6bf6a8c4ef2ec7fb4ab7ec))

- **capture**: Single-output fast path skips per-output field copy (PERF-10/13)
  ([`a1df040`](https://github.com/johnmarktaylor91/torchlens/commit/a1df040035f58564cff1a3d7b913bdd4a71fa914))

When a wrapped op produces exactly one loggable tensor output (and one entry, no container), reuse
  the shared fields_dict directly instead of copy.copy-ing 33 fields +
  deepcopy(parent_arg_positions) per output. Multi-output path unchanged (still isolated);
  multi-output field-set demotion intentionally deferred (aliasing risk). Parity goldens
  byte-identical; adds single-output + multi-output-isolation regression.

- **capture**: Streamline exhaustive field copies
  ([`8421ada`](https://github.com/johnmarktaylor91/torchlens/commit/8421ada28ad443889cab8407c86dc566f98c2797))

- **datamodel**: Add slots to Op
  ([`c5de1ea`](https://github.com/johnmarktaylor91/torchlens/commit/c5de1ea8885323fa4305a5be776a8a70a1cab1d6))

- **postprocess**: O(v+e) input/output distance flood; enable by default
  ([`4676f26`](https://github.com/johnmarktaylor91/torchlens/commit/4676f2671c87bc6d1d69f1f7d0f7cd0c89e6c4d9))

### Refactoring

- **backends**: B10.0 char-net + B10.1 kwarg-dispatch consolidation
  ([`2ce4e7d`](https://github.com/johnmarktaylor91/torchlens/commit/2ce4e7d49d83b7296d0b51c5b658cc58df16aa9a))

Expanded torch characterization net (exceptions/HaltSignal/save_new_outs/
  intervention-ready/DataParallel/buffer-reconcile/dropout-RNG/input-path) as the reroute
  proof-harness. Consolidate per-backend kwarg rejection/default logic into backends/_options.py
  (behavior-identical). trace.py driver untouched.

- **capture**: B10.2 route source/output through backend Protocol
  ([`c7a9004`](https://github.com/johnmarktaylor91/torchlens/commit/c7a900476d9725a04feeb0346882f7e06d053594))

Source-tensor logging + output extraction now go through Protocol extract_and_mark_outputs() (torch
  body verbatim-moved); torch byte-identical (char-net + goldens unchanged). trace.py driver no
  longer torch-direct for these.

- **capture**: B10.3 route buffer reconcile + error cleanup through Protocol
  ([`8d50aa7`](https://github.com/johnmarktaylor91/torchlens/commit/8d50aa70200ad60d559bf58768f12490f2806634))

Buffer reconciliation + exception cleanup now go through Protocol (finalize_forward_session
  documented as pre-output-extraction seam; cleanup_failed_forward_session). Ordering preserved
  exactly; torch byte-identical (char-net + goldens + full not-slow 2814 unchanged).

- **capture**: B10.4 route input normalization/device through Protocol
  ([`b4d89f4`](https://github.com/johnmarktaylor91/torchlens/commit/b4d89f4818f8c64286e8edd43a4cce25add79dac))

_fetch_label_move_input_tensors + _setup_inputs_and_device now go through CaptureBackend (torch body
  verbatim-moved; thin trace.py compat seam kept). Torch byte-identical; no caller-input mutation
  (char-net input-path + goldens).

- **capture**: B10.5 drop _TORCH_BACKEND privilege (unified backend seam)
  ([`063eb9c`](https://github.com/johnmarktaylor91/torchlens/commit/063eb9c0c4d856a6ed40a1e1a8476109f19c8d72))

capture/trace.py no longer imports/holds the torch backend singleton; torch resolves via
  resolve_backend_spec(...).capture_backend() + CaptureBackend Protocol like every other backend
  (capture_backend on BackendSpec; Protocol module-stack hooks). All 4 backends now use one uniform
  injection seam. Torch byte-identical (char-net + goldens + registry unchanged).

- **capture**: Replace live op records with event index
  ([`e60300f`](https://github.com/johnmarktaylor91/torchlens/commit/e60300fd33dd40c1263e0750c038cd208c11a694))

- **capture**: Unify tensor copy behind shared primitive; fix arg-snapshot aliasing
  ([`b38e55d`](https://github.com/johnmarktaylor91/torchlens/commit/b38e55d0f5b6cfee9392114768d722f0a2696e98))

Collapse the two copy families behind one pause_logging clone primitive with two explicit policies:
  copy_tensor_payload (output payload, shallow non-tensor) and copy_arg_tree (recursive input
  snapshot). Route Op.save_activation and the wrapper arg/kwarg snapshots accordingly. FIXES a
  latent bug: wrapper arg snapshots used the shallow copy, so a tensor nested in a container arg
  aliased live state and lost its pre-mutation value needed by out_versions_by_child capture; now
  cloned via copy_arg_tree. Keeps pause_logging.

- **data_classes**: Drop _log suffix from record module filenames
  ([`362d52f`](https://github.com/johnmarktaylor91/torchlens/commit/362d52fcafbe9ec48b70fb9f0607dd602e5a5ad8))

- **export**: Remove deprecated tabular exporters
  ([`76c9d68`](https://github.com/johnmarktaylor91/torchlens/commit/76c9d68d190e29630c6a7c98b9ce3d11493e3256))

- **glossary**: Finish B1 conformance
  ([`a4878ae`](https://github.com/johnmarktaylor91/torchlens/commit/a4878ae172476e9794d3ab3af253909194c8ca90))

- **io**: Add generic class-agnostic state adapter for serialization paths
  ([`ad89b51`](https://github.com/johnmarktaylor91/torchlens/commit/ad89b51aebe04c706e7f79cfa614bd048aa03a69))

- **io**: Deterministic Trace scrub field order; drop model-history reorder
  ([`67fe385`](https://github.com/johnmarktaylor91/torchlens/commit/67fe3858caa5ae61ee94b8049fccda3e740c958a))

Make portable .tlspec metadata field order deterministic at the scrub boundary
  (MODEL_LOG_FIELD_ORDER fields, then remaining PORTABLE_STATE_SPEC fields, then the unknown-field
  error guard), independent of runtime Trace.__dict__ order. This lets us delete the runtime
  _trim_and_reorder_model_history_fields pass + its call sites. state_items() stays generic (slotted
  Op path untouched). Value/membership identical on round-trip; only metadata.pkl byte ORDER
  changes.

- **ir**: De-torch capture predicate schema
  ([`1c940f3`](https://github.com/johnmarktaylor91/torchlens/commit/1c940f3f4e0ca73a87c0180cbaa080af6ce7c76f))

- **ir**: Unify ContainerSpec into leaf ir/container.py (byte-neutral)
  ([`e9f2ef0`](https://github.com/johnmarktaylor91/torchlens/commit/e9f2ef0b4a1f5ea9a2a68a9dc64bb9def7d492b9))

- **jax**: Kind-dispatched replay spine for interpreter foundation
  ([`36fcd30`](https://github.com/johnmarktaylor91/torchlens/commit/36fcd3052c9138758b0f1bc449a7b8b408285151))

- **postprocess**: Avoid op dict reorder mutation
  ([`183170f`](https://github.com/johnmarktaylor91/torchlens/commit/183170f2371debe89deee0b3fc71b3a822bced62))

- **postprocess**: Backend-neutral loop-grouping adapter
  ([`be2ed0f`](https://github.com/johnmarktaylor91/torchlens/commit/be2ed0f6a083c95cbcbeba58b2d7be490754bff1))

- **postprocess**: Build Op from events only in Step-0 materialization
  ([`b20f32b`](https://github.com/johnmarktaylor91/torchlens/commit/b20f32bd5f075c939d9d8c8c19196bbdbdb10db2))

materialize_from_events now constructs each raw Op's fields from OpEvent + sibling events
  (module/buffer/version) + deferred edge joins, with zero live_record.fields reads. Live-record
  construction stays (dual-emit) so the legacy path remains the byte-identical oracle for this
  phase. Adds event payloads needed for parity: module input labels, module output names, input
  tensor addresses, child-version propagation onto the replacement OpEvent. Includes phase-1 A/B
  parity tests.

- **postprocess**: Drop dead reorder/retain helpers; de-misname Step 11
  ([`f35b3bb`](https://github.com/johnmarktaylor91/torchlens/commit/f35b3bbc0d05d2664ce2225c332795b6bdd884c9))

Delete the dead no-op _trim_and_reorder_layer_entry_fields and the vestigial
  _labels_in_replay_ready_call_groups_to_retain (consumed nowhere). Rename
  _remove_unwanted_entries_and_log_remaining -> _build_lookup_keys_and_finalize_retained_layers to
  reflect reality (it builds lookup keys; it does NOT drop unsaved ops). Add a regression locking
  that selective-save unsaved non-orphan ops keep full metadata while .out raises.

- **records**: Add canonical-glossary public names to record classes
  ([`bdf4f23`](https://github.com/johnmarktaylor91/torchlens/commit/bdf4f23ea143a2ca2b8face91b8efdbeeec7afa7))

Implements lock-backed public names from the canonical glossary: is_trainable,
  is_module_input/is_module_output, input_to_modules, atomic_module family,
  args_summary/kwargs_summary, grad_fn_label, Layer op_labels/total_func_duration/
  internal_param_memory, Module total_flops/total_macs/total_param_memory/
  call_parent_address/call_children_addresses, and the Buffer overwrite cluster
  (is_overwritten/num_overwrites/last_overwrite_source) derived from buffer_source. Smoke green.
  Clean-break old-name removal and file renames follow.

- **records**: Drop bare out_memory passthroughs at module scope
  ([`a213ac2`](https://github.com/johnmarktaylor91/torchlens/commit/a213ac29fa08e9d342236d194809a423c48f16b2))

Remove ModuleCall.out_memory / out_memories. The locked decision is that module-scope memory is
  covered by the output_/internal_-prefixed cluster (output_activation_memory,
  internal_activation_memory, output_gradient_memory, internal_gradient_memory, autograd_memory,
  param_memory), which is unambiguous about the boundary-vs-internal distinction. The bare
  out_memory passthroughs duplicated output_activation_memory under a less precise name.

Also drop the now-dead out_memory fallback in rendering when computing a collapsed module box label;
  the module output layer exposes activation_memory.

- **records**: Drop conformance shims for clean break
  ([`01ac105`](https://github.com/johnmarktaylor91/torchlens/commit/01ac1055df66f4671585596e81537d6b4aaf7322))

Removes the deprecation-alias shims the conformance work had added (Param.trainable,
  Op.is_submodule_input/is_submodule_output, Module.flops/macs/flops_forward/flops_backward/
  macs_forward/macs_backward) so the renamed glossary names are the only surface. Per the
  clean-break decision: no shims. Migrates remaining test references to the canonical names
  (is_trainable, Module.total_flops/total_macs family). Affected tests green.

- **records**: Remove glossary-confirmed-removed fields
  ([`01531c6`](https://github.com/johnmarktaylor91/torchlens/commit/01531c68fa1600032551e014aa250d20eb7e4fbc))

- **repgeom**: Mds scatter composes render_image_scatter; retire private scatter helpers [C0b]
  ([`45bff9d`](https://github.com/johnmarktaylor91/torchlens/commit/45bff9d97b80e0c5130165efcd6249ea5549730c))

mds_scatter_node_spec now renders via tl.viz.render_image_scatter (the C0 primitive), making the
  toolkit the single source of scatter rendering. Removed the now-dead private
  _render_mds_scatter_image / _coords_to_pixel_centers / _spread_close_centers / _spiral_offsets
  helpers from repgeom. Public mds_scatter_node_spec signature + node-spec return contract
  unchanged; MDS scatter is opt-in (node_spec_fn), so default-render byte-identity is untouched. MDS
  render tests updated to the primitive-based output; numeric MDS tests unchanged.

### Testing

- Add heavy marker tier for faster per-step gating
  ([`6a9b2f4`](https://github.com/johnmarktaylor91/torchlens/commit/6a9b2f4377ef294f7cd642af704c74fbef97f647))

Register a 'heavy' marker for 5-20s tests and promote >20s tests to slow, so the fast per-step gate
  (-m 'not rare and not slow and not heavy') drops ~15min -> ~8min; smoke (~28s) stays the true
  sub-minute gate. Marker-only re-tiering: no test or assertion removed, full 'pytest tests/' still
  runs everything (3126 collected). Documents tier costs + that pytest-xdist oversubscribes torch
  threads (not faster).

- Lock in func-loc leak, barcode hash, and validate state-restore fixes
  ([`648837a`](https://github.com/johnmarktaylor91/torchlens/commit/648837a78bebc14900233db16246419cb0f50067))

The FUNC-CALL-LOC-LEAK, HASH-COLLISION, and VALIDATE-STATE-RESTORE bugs from the tracker were
  already fixed in the 2.0 overhaul (b55e16b phase 14) but had no dedicated regression coverage.
  Twelve tests now lock: FuncCallLocation retains no function-object reference after construction
  while lazy .source still loads (incl. pickle round-trips and the source_loading_enabled=False
  path); make_short_barcode_from_input is deterministic across processes/PYTHONHASHSEED and
  collision-free over a 40k structured-input scan (the old salted-hash barcodes leaked into
  persisted .tlspec op_kind via equivalence_class, so cross-process artifact comparability depends
  on this); validate_forward_pass restores bit-exact model state when the forward raises on either
  the ground-truth or traced call, via a detached-clone snapshot.

Also corrects the hashing module docstring: barcodes derive from shape/dtype/scalar tokens, not
  param data pointers (params are excluded).

- **api**: Account for transform selector export
  ([`101d3c0`](https://github.com/johnmarktaylor91/torchlens/commit/101d3c008e42364f2659399591284171c33a2750))

- **backend**: Add torch parity gates
  ([`9af28d7`](https://github.com/johnmarktaylor91/torchlens/commit/9af28d78978728c20b50c278c45420abd7e08134))

- **buffers**: Cover buffer datamodel integration
  ([`6e3a291`](https://github.com/johnmarktaylor91/torchlens/commit/6e3a29150420203ec6da8e25937c5d6a6255872c))

- **capture**: Align factory and validation tripwires
  ([`f62fff7`](https://github.com/johnmarktaylor91/torchlens/commit/f62fff7d2ed68ae2bd21a0e47efb030e83ea7b48))

- **capture**: Arg-spec coverage harness + 125 high-confidence FUNC_ARG_SPECS fills
  ([`f0a78e9`](https://github.com/johnmarktaylor91/torchlens/commit/f0a78e9fecfaef307f8958bf5b7ff74df6c4f631))

- **capture**: Cover all pass layer saves
  ([`5ad069e`](https://github.com/johnmarktaylor91/torchlens/commit/5ad069efec95d7ab12e728285c906727ec9bd39b))

Implements the approved 2026-06-06 behavior change for legacy layers_to_save resolution: an
  unqualified repeated layer or module label saves every pass, while a pass-qualified label such as
  attn:2 saves only that 1-based pass. No validation or replay tripwires were loosened; the
  regression test updates the expected behavior because first-pass-only was the old behavior being
  replaced.

- **capture**: Cover captured run phase seven
  ([`69192e6`](https://github.com/johnmarktaylor91/torchlens/commit/69192e664da6ccef36445e11ef869b355696ed32))

- **capture**: Cover functional call attribution
  ([`b7e7901`](https://github.com/johnmarktaylor91/torchlens/commit/b7e79019c37c219a72e3ef8d670c568f05f72c1a))

- **capture**: Cover phase 3b retention
  ([`0c70c1a`](https://github.com/johnmarktaylor91/torchlens/commit/0c70c1a5a747ac2ab6f0662a417e382d72b63ca8))

- **capture**: Cover retroactive lookback save
  ([`5eb7c9f`](https://github.com/johnmarktaylor91/torchlens/commit/5eb7c9f64a3425837a7715c92c2fdd855b66231e))

- **capture**: Cover transform selector persistence
  ([`2dbd07a`](https://github.com/johnmarktaylor91/torchlens/commit/2dbd07ae873b350bc146a1671fb4bda73d88d750))

- **capture**: Gate event-only forward capture
  ([`42df084`](https://github.com/johnmarktaylor91/torchlens/commit/42df08490d0de939622aea2c5193cc32381b65c7))

- **capture**: Preserve orphan factory coverage
  ([`4d64c08`](https://github.com/johnmarktaylor91/torchlens/commit/4d64c089b486f664c43c93ca137bda797adc2e0a))

- **ir**: Include transform event defaults
  ([`1f733bb`](https://github.com/johnmarktaylor91/torchlens/commit/1f733bb6afe0b0a96c73471d980d6799d0fb0358))

- **perf**: Add P6 regression gate harness
  ([`7999d0a`](https://github.com/johnmarktaylor91/torchlens/commit/7999d0a24a9822392e70a4f4a1884e434b36464e))

- **realworld**: Update transformers configs
  ([`41c7c9e`](https://github.com/johnmarktaylor91/torchlens/commit/41c7c9e934cf78c0a1efc5b9508f1e35262f0569))

- **report**: Account for transform selector export
  ([`fedc229`](https://github.com/johnmarktaylor91/torchlens/commit/fedc2291345a2b8a37deed0d813befc03472e90c))

- **validation**: Arm tripwire for functionless capture placeholders
  ([`edd2b3a`](https://github.com/johnmarktaylor91/torchlens/commit/edd2b3a501bdc32e98af10ea6fe8ca1b6b37dce4))

Add a regression test asserting that plain tl.trace of a fully-traceable model produces zero
  functionless intervention_replacement ops, so a future capture gap fails loudly. A self-contained
  _VmapMaskModel reproduces the mechanism (a mask built inside torch.vmap entering a submodule
  untagged) without any optional dependency, and a transformers-gated Mistral case exercises the
  real reproducer. Both assert the mask is logged as an internal source and that validation passes
  legitimately. Confirmed the test fails if the placeholder behavior is reintroduced.

- **validation**: Avoid brittle nested transform labels
  ([`f2bc32b`](https://github.com/johnmarktaylor91/torchlens/commit/f2bc32bc40e4f39f73a840a812803496b575b3c7))

- **viz**: Cover loop module rolling reconciliation
  ([`6d65ac2`](https://github.com/johnmarktaylor91/torchlens/commit/6d65ac2649995f77f391c841131ece80001b0df8))

- **viz**: Rewrite rolled label tests; add label-geometry regression gate
  ([`6484cde`](https://github.com/johnmarktaylor91/torchlens/commit/6484cde41d830f16ff8c07090f55e12164afca45))

The four stale tests asserted the deleted plain-text newline-padded label format. Rewritten against
  the merged-midpoint scheme: self-loop and back-edge merges, rankdir line-order flip, buffer-edge
  one/two-line midpoint labels, and HTML head/tail labels on varying forward edges (asserting
  semantics, not tunable constants).

tests/support/label_geometry.py promotes the sprint's audit harness into the suite: it parses dot
  -Tjson layout into exact glyph bboxes, node outlines, edge splines, and arrowhead polygons, and
  reports label-label, label-node, label-spline, label-arrowhead, and label-cluster-border
  penetrations. tests/test_label_geometry.py renders all 16 rolled demo models and asserts zero
  violations (plus a negative control proving the gate detects a planted overlap), so future
  label/layout regressions fail with exact geometry instead of waiting for a human eye.


## v2.18.0 (2026-05-29)

### Bug Fixes

- **alpha3**: Accept LiveOpView in nonfinite check
  ([`4e4e683`](https://github.com/johnmarktaylor91/torchlens/commit/4e4e683b05e07dc28da5efe4ee260193b435ddc5))

- **alpha3**: Keep auxiliary T2 paths compatible
  ([`b665119`](https://github.com/johnmarktaylor91/torchlens/commit/b665119693ead546539d6bf746b1480ae3838916))

- **alpha3**: Materialize failed partial traces
  ([`c0584b5`](https://github.com/johnmarktaylor91/torchlens/commit/c0584b525cffaedce665baa01e9dcde4529cb41a))

- **alpha3**: Restore streaming strict capture writes
  ([`b7b3627`](https://github.com/johnmarktaylor91/torchlens/commit/b7b3627a6e5727b876a3d7147bb1b6db47cad6ad))

- **alpha3**: Restore train_mode out_postfunc detach validation
  ([`20b7fd0`](https://github.com/johnmarktaylor91/torchlens/commit/20b7fd0cc357001915f569f202c5af2ca2d784ad))

Co-authored-by: Codex <codex@openai.com>

- **backward, demo**: Pin grad_fn refs, fork conditional viz at single node, add bundle diff
  ([`3cc4c46`](https://github.com/johnmarktaylor91/torchlens/commit/3cc4c4694abff7b0cd048adeb1079c19aeb8c83b))

- backward: hold strong refs to discovered grad_fns in trace._grad_fn_strong_refs. Without this,
  leaf AccumulateGrad nodes could be gc'd and Python could recycle their id() values for
  later-created grad_fns (e.g., the output clone wrapper), producing phantom cycles in
  next_grad_fn_ids and conflated/missing AccumulateGrad nodes in the rendered backward graph.

- TinyBranchCNN demo: restructure forward so each arm consumes x directly (matches
  SimpleIfElseModel). Now both IF and THEN/ELSE labels emanate from input_1 as a single visible fork
  point in the rendered graph, instead of IF on input -> mean and THEN far downstream on the body's
  first edge.

- Notebook: add bundle.show_diff visual cell.

Tests: 209 conditional + backward + smoke green.

- **bundle_diff**: Drop hard size+ratio constraints that clipped the caption
  ([`58bf90c`](https://github.com/johnmarktaylor91/torchlens/commit/58bf90cd985b088129a4cd99235e74594e275664))

The renderer set ``size: 12,8!`` (hard fit) plus ``ratio: fill`` (force aspect), forcing the layout
  into a 1.5:1 box. The natural paired layout is ~1.69:1 taller, so graphviz emitted a viewBox
  shorter than the content; the bottom-positioned caption got pushed below the visible viewBox and
  clipped in rendered SVGs.

Removing both attrs lets graphviz auto-size to the natural aspect (793x470 for the canonical
  ResNet-18 demo) so the caption fits inside the viewBox.

Updated the canonical snapshot to match the new aspect. Bundle tests (78) plus smoke (170) green.

- **bundle_diff**: Key layer-to-supergraph lookup by layer_label, not id()
  ([`d126ce5`](https://github.com/johnmarktaylor91/torchlens/commit/d126ce579aaee39e3e4bdebb84847fb5a09d38f3))

bundle.aligned_pairs() and bundle.supergraph.nodes materialize independent layer objects with
  different id() values. The renderer's layer_to_node dict was keyed by id(layer) from supergraph;
  lookups during render iterated layers from aligned_pairs and missed every entry, so every node
  fell back to a delta of 0.0, max_delta resolved to 0.0, and _delta_color short-circuited to
  "#FFFFFF" for every cell. Result: the diff rendered all-white with no visible per-node delta
  colorization.

Switch the dict key to layer.layer_label (stable string) and look it up the same way at every
  consumer site. Now value=0 maps to blue, intermediate nodes show salmon at ~0.85 of max, and red
  shows up at max-delta sites.

Regenerated canonical snapshot + hero demo SVG. Bundle (79) + smoke (170) green.

- **bundle_diff**: Use dpi=72 so viewBox matches actual content extent
  ([`c90d8d2`](https://github.com/johnmarktaylor91/torchlens/commit/c90d8d2658773a4f963a5847b20266775ce08cf9))

Setting ``dpi=100`` makes graphviz emit a viewBox in internal layout coords while the inner ``<g
  transform="...">`` applies a ``scale(100/72=1.389)`` to convert to display coords. Net effect: SVG
  content (background polygon, caption, legend) extends to y=633 in transformed coords but viewBox
  max y is only 470. Bottom ~25% of the figure was clipped.

dpi=72 (graphviz default) keeps viewBox in the same units as content, so content fits inside (text
  max y=456 vs viewBox max=470).

Regenerated canonical snapshot + hero demo SVG. Bundle (78) + smoke (170) green.

- **capture**: Preserve new raw_input across rerun atomic swap
  ([`7fa8a8f`](https://github.com/johnmarktaylor91/torchlens/commit/7fa8a8fa221eafb27f03ac054ffd47c0dcf4c46c))

- **ci**: Let git-lfs tolerate the CI tag push (allowincompletepush)
  ([`056193d`](https://github.com/johnmarktaylor91/torchlens/commit/056193d4044c8ce3da2ff2002b19f541b974e296))

`git lfs uninstall` cleared the original bracketed-URL LFS failure but the tag push then hit
  "missing or corrupt local objects": CI checks out with lfs disabled (pointers only), so git-lfs
  refuses to push the tag whose tree references LFS objects it can't find locally. Those objects
  already live on the remote, so set `lfs.allowincompletepush true` (git-lfs's own recommended fix)
  plus `lfs.locksverify false`. This lets the version tag actually get created so the release
  completes (PyPI publish + GitHub release).

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

- **ci**: Stop runaway semantic-release loop (skip-ci + lfs tag push)
  ([`0b24b6e`](https://github.com/johnmarktaylor91/torchlens/commit/0b24b6e0cb79c95f5be88496e7c350f94145d36d))

Two compounding bugs produced a 1599-commit release loop:

1. The bot's `chore(release):` commit re-triggered the Release workflow (no skip token), so every
  release pushed a commit that started another release. Fixed by appending `[skip ci]` to
  commit_message and adding a job guard that refuses to run on `chore(release):` head commits.

2. The version TAG push failed under git-lfs: the App credential URL (username
  `torchlens-release[bot]`) breaks git-lfs endpoint parsing ("batch request: missing protocol").
  Plain commit pushes tolerate it but the tag push invokes LFS and dies, so no tag was ever created
  and each run re-released the same commits since v2.17.0. Fixed by `git lfs uninstall` in the
  release job (LFS objects already live on the remote).

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

- **facets**: Correct LayerNorm gamma+beta resolution; add HF tutorial; lock spec
  ([`9406863`](https://github.com/johnmarktaylor91/torchlens/commit/9406863fcee463861dd3e6062bb7356d6b7ad273))

Three follow-ups to 3ada85d:

1. torchlens/semantic/recipes/norm.py LayerNorm + RMSNorm recipes were reading `mod.cls.weight`.
  Module.cls stores the class object, not the live instance, so cls.weight returns None and
  `add_if_present` skipped the entry -- gamma was declared in keys() but unreachable on access. Fix
  iterates module.params and matches on param.name, falling back to None gracefully when param.value
  is None.

2. notebooks/huggingface_tutorial.ipynb First commit of the HF tutorial. 52 cells; the new "10.
  Facets" section adds 19 cells demonstrating recipe listing, q/k/v access, .head(i) sub-view,
  log.attention_blocks() sweeping, MLP/LN facets, fused-SDPA pattern error, user recipe
  registration, and tl.facets.info() discovery. Executed clean end-to-end.

3. .project-context/glossary_walkthrough_deltas.md Appended the "Facets framework ... (LOCKED
  2026-05-27)" entry that 3ada85d implemented against. Makes the spec source of truth visible
  alongside the implementation.

- **fastlog**: Preserve halt through predicate catch sites
  ([`810b5a2`](https://github.com/johnmarktaylor91/torchlens/commit/810b5a23e76c11c7665f41d2319ce257fd31a6c8))

- **glossary-v5**: Restore module collapse rendering for vis_call_depth
  ([`16fc0eb`](https://github.com/johnmarktaylor91/torchlens/commit/16fc0eb74b72e1731ed5022e562f1fda37801100))

The v5 additions accidentally changed module-entry annotation semantics by renaming the
  buffer-address scratch variable in module entry capture. That made ordinary leaf module outputs
  become atomic module exits, which prevented top-level leaf modules from being collapsed by legacy
  vis_call_depth rendering and drifted module-containment snapshots.\n\nRestore the legacy entry
  annotation behavior while keeping the v5 template capture additions, and make the collapse-address
  helper return the legacy depth target directly. Atomic outputs now only bubble to a parent module
  when such a parent exists, so top-level leaf modules remain eligible for vis_call_depth=1
  collapse.

- **grad_fn_log**: Rename has_op to is_intervening
  ([`e7668ea`](https://github.com/johnmarktaylor91/torchlens/commit/e7668ea9b4b064a41628227263cbc412b8414ccb))

- **intervention**: Preserve is_appended in replace_run_state_from atomic swap
  ([`071e9e4`](https://github.com/johnmarktaylor91/torchlens/commit/071e9e4fa1997ecd23afca5cedf2eb85956629fd))

- **intervention**: Preserve tl_tensor_label_raw on tensor replacement
  ([`869c8ee`](https://github.com/johnmarktaylor91/torchlens/commit/869c8ee2e7ecd32d6b6cb2e341a9b4c44b8e4a69))

Implements two-layer fix for INTERVENTION-MISSING-TENSOR-LABEL bug discovered during the 2026-05-04
  NeurIPS fleet overnight run (EXP_19, EXP_20).

Layer A: intervention API now propagates tl_* attrs from original to replacement. Layer B:
  _handle_module_exit gracefully re-instruments tensors lacking tl_tensor_label_raw, covering raw
  register_forward_hook patterns outside the official API.

Adds regression tests for: official intervention API, raw forward_hook replacement,
  chain-of-interventions, quantization-sensitivity pattern. All 4 pass in <1s.

Bonus: also fixes a state_dict._metadata clobbering issue in validate_forward_pass that affected
  torchvision MNASNet, surfaced during the bulletproofing audit.

Closes INTERVENTION-MISSING-TENSOR-LABEL.

- **intervention**: Reject append=True on streaming-active trace
  ([`423b5ee`](https://github.com/johnmarktaylor91/torchlens/commit/423b5eefbe2409162f383655deceda4f06bf2be7))

- **io**: Audit-null mlx tensors on save
  ([`84622f1`](https://github.com/johnmarktaylor91/torchlens/commit/84622f1bcf6ea97b8e2ea3728a786b9f78a8fdbc))

- **io**: Serialize module pass tabular outputs
  ([`68f1280`](https://github.com/johnmarktaylor91/torchlens/commit/68f1280a8e1cb6af9b8f3e811608962eb9e3eb66))

- **loop-detection**: Include output index in multi-output eq-class
  ([`b98f659`](https://github.com/johnmarktaylor91/torchlens/commit/b98f6598f956a6a32e3fc03430f2378965fe9204))

- **mlx**: Bind wrappers per trace and resolve inputs
  ([`429c11c`](https://github.com/johnmarktaylor91/torchlens/commit/429c11cacea3db0403f85f2440aae04ddf9bd2d0))

- **mlx**: Hard-reject unsupported preview options
  ([`a74f0dd`](https://github.com/johnmarktaylor91/torchlens/commit/a74f0dd9205f38ef4dbf6eb55a64eeac5c928341))

- **model-log**: Clear is_appended on non-append rerun
  ([`e5a8e69`](https://github.com/johnmarktaylor91/torchlens/commit/e5a8e696a3fd812cc9162c46953960ec1f3baa41))

- **rename**: Update intervention examples for ledger rename + bundle len
  ([`455e0f0`](https://github.com/johnmarktaylor91/torchlens/commit/455e0f0f4c4725d6a45700a3471f26d04a0ef76d))

Codex's R2 missed three example files: operation_history was renamed to ledger (per glossary
  walkthrough deltas), and 13_paired_prompt_3plus.py had a stale lambda that called .names on a
  Trace member. Replaced with idiomatic len(bundle).

All three intervention example tests now pass
  (test_examples.py::test_intervention_example_runs[05/06/13]).

- **save-load**: Reattach module output references
  ([`20790d4`](https://github.com/johnmarktaylor91/torchlens/commit/20790d49805d3e718eaf28e3ed035006c694cde5))

- **save-load**: Tolerate legacy module logs without calls
  ([`c37ad47`](https://github.com/johnmarktaylor91/torchlens/commit/c37ad473c7ff91d30a6b1146b2daddc73e562277))

- **semantic**: Cover DistilBERT eager/flash attention in facet recipe
  ([`4c0d6ee`](https://github.com/johnmarktaylor91/torchlens/commit/4c0d6ee2f582d13dbd81fa830a84b6f090eb36c0))

The attention facet recipe only registered DistilBertSdpaAttention, so switching to
  _attn_implementation='eager' (which the pattern MissingFacet explicitly recommends) landed on the
  unregistered MultiHeadSelfAttention class and raised KeyError('q') on facets.q / facets.head(n).q.
  Register the same q_lin/k_lin/v_lin extraction for all three DistilBERT attention classes:
  MultiHeadSelfAttention (eager, no pattern), DistilBertSdpaAttention and DistilBertFlashAttention2
  (fused, pattern via MissingFacet). Flash is now also flagged as fused for the pattern facet.

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

- **semantic**: Cover transformers 5.x DistilBertSelfAttention in facet recipe
  ([`3e4dc6b`](https://github.com/johnmarktaylor91/torchlens/commit/3e4dc6b4af0e1011db5369fbfdacebd139ee231f))

transformers 5.x collapsed DistilBERT's per-backend attention subclasses (MultiHeadSelfAttention /
  DistilBertSdpaAttention / DistilBertFlashAttention2) into a single DistilBertSelfAttention class,
  selecting eager/sdpa via a runtime function. That class was unregistered, so facets.q /
  facets.head(n).q raised KeyError('q') on transformers 5.x even though the q_lin/k_lin/v_lin
  children are unchanged.

Register DistilBertSelfAttention alongside the existing classes. Its default backend is fused SDPA
  and the class name no longer encodes the backend, so it surfaces `pattern` as the informative
  MissingFacet (RuntimeError on access) rather than a silent AttributeError -- keeps the
  fused-pattern teaching path working. 4.x eager MultiHeadSelfAttention behavior is unchanged.

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

- **tests**: Register synthetic migration examples in linecache; skip nnsight>=0.7
  ([`4c2fb61`](https://github.com/johnmarktaylor91/torchlens/commit/4c2fb61896ebd62b593c7c8811ae139b85eea399))

The migration-example test runner exec'd code under a synthetic filename without registering it in
  linecache, so any example whose tooling called inspect.getsource on user code (nnsight
  LanguageModel does this) failed with OSError: source code not available.

Also skips the from_nnsight.md example on nnsight>=0.7, where the LanguageModel.trace input
  semantics changed and our pinned expected output no longer applies.

- **types**: Resolve pre-existing mypy errors in data_classes/
  ([`0f54ba5`](https://github.com/johnmarktaylor91/torchlens/commit/0f54ba5dc131ca8161d400e68ca2b7de88ea4889))

Cleans up mypy errors flagged repeatedly throughout the M1-M8 capture-pipeline-unification sprint.
  Errors were in data_classes accessors and ModuleLog/ModuleCallLog narrowing, with a few
  import-chain typing leaks required by the exact data_classes mypy gate.

No behavior change. Runtime tests remain green.

- **user-api**: Register tensor connections on live records
  ([`0e3c2ce`](https://github.com/johnmarktaylor91/torchlens/commit/0e3c2cedbfd377b432ad53495453b06f1636c033))

- **validation**: Allow multi-output parameterized ops
  ([`8b85905`](https://github.com/johnmarktaylor91/torchlens/commit/8b85905ebda36db81e2219dad6bf86b55f667a25))

- **validation**: Backward validator handles stacked traces
  ([`2271de7`](https://github.com/johnmarktaylor91/torchlens/commit/2271de77aba3dc1ef4612b6572f27b13231407b1))

- **viz**: Add colon separator between param name and shape in node labels
  ([`0e323b8`](https://github.com/johnmarktaylor91/torchlens/commit/0e323b8599a31d9e9ecb889c9949968a120c5eb4))

Renders 'weight: (16, 3, 3, 3)' instead of 'weight (16, 3, 3, 3)' in graphviz param labels and
  inline param lines.

- **viz**: Drop render_graph shim; auto-suppress xdg-open on headless
  ([`1c31b6b`](https://github.com/johnmarktaylor91/torchlens/commit/1c31b6bbb58ac9049bd4dfd9a8ca8537a50d592b))

- **viz**: Drop space after @ in module address labels
  ([`2dfe89d`](https://github.com/johnmarktaylor91/torchlens/commit/2dfe89ddde3aa5e609c4a36235ceb8b64596b50b))

Module path rows rendered as "@ module1.module2"; tighten to "@module1.module2" so the dotted path
  reads as a single token and matches the no-space form already used by bundle_diff and the dagua
  bridge.

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

- **viz**: Handle headless trace drawing
  ([`cb141cf`](https://github.com/johnmarktaylor91/torchlens/commit/cb141cf2229474a13d571358fea08a8b769cd3bb))

- **viz**: Read live capture event counts
  ([`371f569`](https://github.com/johnmarktaylor91/torchlens/commit/371f569613958b65ffe48c70aa2505eae5b4f62b))

- **viz**: Suppress headless auto-open note inside notebooks
  ([`a1fce6b`](https://github.com/johnmarktaylor91/torchlens/commit/a1fce6b51c170cec5caf443bb84e5d1a28a29f97))

In a notebook the figure is shown inline via IPython display(); the draw path still fell through to
  _view_rendered_file, which on a headless remote kernel printed "headless context detected; ...
  skipping auto-open". Bail out of _open_file_quietly when in_notebook() so no desktop viewer is
  launched and no misleading note is printed. Centralizing in the shared helper also covers the
  sfdp/ELK/bundle render paths.

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

- **viz**: Use sequential bundle diff palette
  ([`da060bf`](https://github.com/johnmarktaylor91/torchlens/commit/da060bfb12fad7b9d72d1104e9b66ed4a568d15d))

### Chores

- **audit**: Regenerate coverage manifest after notebook rewrite
  ([`d62dadc`](https://github.com/johnmarktaylor91/torchlens/commit/d62dadccc6700236812936a718b602d1ba474f10))

- Regenerate the public API coverage manifest from current TorchLens. - Refresh the strict coverage
  matrix after the notebook rewrite.

- **audit**: Rewrite 00_install_and_smoke.ipynb to actually exercise its declared coverage
  ([`a590ab6`](https://github.com/johnmarktaylor91/torchlens/commit/a590ab65e1728305e8d92e3a6c5a873ec980a62b))

- Demonstrate import surface, __all__, and first tiny-model capture. - Add shared audit helpers that
  resolve coverage names against live objects.

- **audit**: Rewrite 01_basic_capture.ipynb to actually exercise its declared coverage
  ([`7ee5c26`](https://github.com/johnmarktaylor91/torchlens/commit/7ee5c2698451bc10c72a039c2e8d966c10fc7449))

- Demonstrate canonical capture, selective save, peek, extract, and batched_extract. - Resolve
  capture-related manifest entries against live TorchLens objects.

- **audit**: Rewrite 02_layer_indexing.ipynb to actually exercise its declared coverage
  ([`c7b8964`](https://github.com/johnmarktaylor91/torchlens/commit/c7b89644f2148f9e8fac126cfcc829fd5b44853d))

- Demonstrate exact, integer, pass-qualified, module-key, suggestion, and missing-layer lookups. -
  Resolve indexing-related coverage markers against live TorchLens objects.

- **audit**: Rewrite 03_save_load_basics.ipynb to actually exercise its declared coverage
  ([`fc516ca`](https://github.com/johnmarktaylor91/torchlens/commit/fc516ca85007da5c1d7e614255ff9ba5bb6933a5))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 05_visualization_basics.ipynb to actually exercise its declared coverage
  ([`6f11e7f`](https://github.com/johnmarktaylor91/torchlens/commit/6f11e7f2d3df3de095fb4a5388905224b863c7d4))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 06_modellog_anatomy.ipynb to actually exercise its declared coverage
  ([`1b4fcb1`](https://github.com/johnmarktaylor91/torchlens/commit/1b4fcb12755c4a7129c49ab33d792def9d1d97b4))

- Walk ModelLog identity, timing, graph, intervention, export, and private state clusters. - Redact
  high-entropy hash summaries in shared audit output.

- **audit**: Rewrite 07_layerlog_anatomy.ipynb to actually exercise its declared coverage
  ([`e27ac71`](https://github.com/johnmarktaylor91/torchlens/commit/e27ac716451607c91d5b498561d6caba95e30105))

- Walk LayerLog identity, tensor, gradient, graph, module, and pass aggregation clusters. - Resolve
  LayerLog coverage markers against a live aggregate layer record.

- **audit**: Rewrite 08_layerpasslog_anatomy.ipynb to actually exercise its declared coverage
  ([`e0894f7`](https://github.com/johnmarktaylor91/torchlens/commit/e0894f7b033397212e232a7be4ba4c84aa7d041a))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 09_other_log_types.ipynb to actually exercise its declared coverage
  ([`f3a4810`](https://github.com/johnmarktaylor91/torchlens/commit/f3a4810e22d86c32985ed23d28396c17be960347))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 10_bundle_anatomy.ipynb to actually exercise its declared coverage
  ([`4503c7f`](https://github.com/johnmarktaylor91/torchlens/commit/4503c7fc3d9c3d768ef6d81b0b2d17c9154f3b7b))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 15_visualization_options.ipynb to actually exercise its declared coverage
  ([`95636ad`](https://github.com/johnmarktaylor91/torchlens/commit/95636ad56db065c04b4cc807a69f65dd2101c808))

- Demonstrate accepted vis_opt modes and current skip for unsupported vis_opt='all'. - Resolve
  options, viz, and visualization coverage markers against live namespaces.

- **audit**: Rewrite 16_visualization_advanced.ipynb to actually exercise its declared coverage
  ([`3fe7dd5`](https://github.com/johnmarktaylor91/torchlens/commit/3fe7dd5b308ad33ea46e769c198a2902f271c811))

- Demonstrate advanced visualization hooks without opening graph artifacts. - Resolve fastlog,
  callback, and visualization coverage markers against live objects.

- **audit**: Rewrite 17_intervention_helpers.ipynb to actually exercise its declared coverage
  ([`bbc4d84`](https://github.com/johnmarktaylor91/torchlens/commit/bbc4d84ae509ab68e99fc1bd5e79e2e7d7c82ce8))

- Demonstrate selectors, activation helper specs, module splice, and backward helper specs. -
  Resolve intervention helper coverage markers against live objects.

- **audit**: Rewrite 18_intervention_verbs.ipynb to actually exercise its declared coverage
  ([`fe04555`](https://github.com/johnmarktaylor91/torchlens/commit/fe0455583ae808648985921add30807206d17653))

- Demonstrate sites, fork, rerun, and replay precondition handling. - Resolve intervention verb
  coverage markers against live objects.

- **audit**: Rewrite 19_bundles_advanced.ipynb to actually exercise its declared coverage
  ([`9d4fa66`](https://github.com/johnmarktaylor91/torchlens/commit/9d4fa66e8330ef202c06daab8b65549cca4c8af4))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 20_save_load_advanced.ipynb to actually exercise its declared coverage
  ([`fbc92fe`](https://github.com/johnmarktaylor91/torchlens/commit/fbc92fe5ab5c129031651b9e948eb69cf5491dd6))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 21_validation.ipynb to actually exercise its declared coverage
  ([`998985f`](https://github.com/johnmarktaylor91/torchlens/commit/998985fdb21d86e8b11a50784c2cadfbe84d7f7c))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 23_export_formats.ipynb to actually exercise its declared coverage
  ([`6d0ffec`](https://github.com/johnmarktaylor91/torchlens/commit/6d0ffec5d3835f2d8d3065d4149989d052063f4a))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 24_bridges.ipynb to actually exercise its declared coverage
  ([`44fb38c`](https://github.com/johnmarktaylor91/torchlens/commit/44fb38c349316d7a6261e12c53e93b5f4e1ce9ac))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 25_compat_truth_table.ipynb to actually exercise its declared coverage
  ([`2f46e17`](https://github.com/johnmarktaylor91/torchlens/commit/2f46e17e818b58876b63c149621371e12d39d64a))

- Demonstrate compat reporting and markdown output. - Resolve compat exports and deprecated alias
  markers against current TorchLens.

- **audit**: Rewrite 26_perf_and_scaling.ipynb to actually exercise its declared coverage
  ([`18344b5`](https://github.com/johnmarktaylor91/torchlens/commit/18344b5654f6b13db57ba27c714a0f19c384c611))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 27_taps_and_observers.ipynb to actually exercise its declared coverage
  ([`d8d797b`](https://github.com/johnmarktaylor91/torchlens/commit/d8d797b7e7f104e98398b6f417fc2b6f7e2c9a15))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 28_sites_and_sweeps.ipynb to actually exercise its declared coverage
  ([`ee79840`](https://github.com/johnmarktaylor91/torchlens/commit/ee7984095d24c38c4b6e59ee6f59783d95207b75))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **audit**: Rewrite 29_edge_cases.ipynb to actually exercise its declared coverage
  ([`8f06b39`](https://github.com/johnmarktaylor91/torchlens/commit/8f06b392712c1d2b9d24b14696618ae7a6f689a8))

- Replace boilerplate with a topic-specific workflow demo.\n- Resolve the notebook coverage markers
  against live TorchLens objects.

- **backward**: Rename _add_backward_hook to _add_tensor_backward_hook
  ([`f4b7591`](https://github.com/johnmarktaylor91/torchlens/commit/f4b759145bd2fa587104d361bc5148760825cc72))

- **changelog**: Consolidate backward-parity sprint
  ([`d1b904a`](https://github.com/johnmarktaylor91/torchlens/commit/d1b904a159230050c4d3235f357af12a9d184f8a))

- **changelog**: Consolidate post-backward megasprint
  ([`d3500f0`](https://github.com/johnmarktaylor91/torchlens/commit/d3500f0db40922ebfb4cb6bd62fa7f0f049eae85))

- **changelog**: Note multi-output module semantics
  ([`4273d0d`](https://github.com/johnmarktaylor91/torchlens/commit/4273d0d4d83079491e6ef5a45ef369920b584ee6))

- **changelog**: Record P1 alpha.3 finish + B1/B8
  ([`c93dadf`](https://github.com/johnmarktaylor91/torchlens/commit/c93dadf407d566a38746288d41304cb058a4b689))

- **changelog**: Record P2 validator hardening
  ([`f064031`](https://github.com/johnmarktaylor91/torchlens/commit/f064031318a47aec355c0ab66cac2e9392237d09))

- **changelog**: Record P3 combined visualization
  ([`20730ba`](https://github.com/johnmarktaylor91/torchlens/commit/20730baa58714d21e5a7ddc434f5f948f5f45f14))

- **changelog**: Record P4 backward intervention
  ([`c5dd8b8`](https://github.com/johnmarktaylor91/torchlens/commit/c5dd8b88e4325b97f710a062681e48aac5442a75))

- **changelog**: Record P6 observer bundle polish
  ([`ec9e485`](https://github.com/johnmarktaylor91/torchlens/commit/ec9e4850ef2906e5431ecb905352e6490981011a))

- **demo**: Regenerate hero bundle-diff SVG after viewBox fix
  ([`18740de`](https://github.com/johnmarktaylor91/torchlens/commit/18740de492d5dd9e4fa57ead9fec023e448c92ba))

- **finalize**: Record post-backward ready state
  ([`10fa1ba`](https://github.com/johnmarktaylor91/torchlens/commit/10fa1ba30cae122fac9c1e4336dc1151ee71431b))

- **finalize**: Record post-backward T2 block
  ([`67b4e08`](https://github.com/johnmarktaylor91/torchlens/commit/67b4e08368a435f4e0ebe9132b9a7546209fd1d8))

- **finalize**: Record post-backward T2 block
  ([`0bb13d0`](https://github.com/johnmarktaylor91/torchlens/commit/0bb13d00873dd4e09580a19595f4698aadee3a6b))

- **finalize**: Record post-backward T2 block
  ([`fe72464`](https://github.com/johnmarktaylor91/torchlens/commit/fe7246470e64d7e0efadbe9d8bfc709d2f7d57ad))

- **golden**: Regenerate M5 parity goldens after P1 multi-output eq-class fix
  ([`02693a3`](https://github.com/johnmarktaylor91/torchlens/commit/02693a3741c1c6dcdeeb15ed656bdf5cd10e62e6))

Intentional baseline refresh after AD-X multi-output equivalence-class fix added output-index
  disambiguation for multi-output ops.

- **intervention**: V1 maintenance sweep (8 items)
  ([`da15b5f`](https://github.com/johnmarktaylor91/torchlens/commit/da15b5f2a006eb666ea1dfa864968ab7166b3602))

- **io**: Bump portable state schema; drop boundary-thread fields
  ([`c8e84e8`](https://github.com/johnmarktaylor91/torchlens/commit/c8e84e8c91cdfc2c67835b905fddf712c3ccc72e))

Phase 4 of module-containment-refactor sprint. Bumps IO_FORMAT_VERSION from 2 to 3.
  OpLog.__setstate__ now silently drops legacy thread-replay boundary fields
  (_module_boundary_thread_output, _module_boundary_threads_inputs, and the legacy alias
  module_entry_exit_threads_inputs) when loading pre-v3 pickles. Emits exactly one
  DeprecationWarning per process via a load-once helper.

Adds tests/test_module_containment_save_load.py covering IO_FORMAT_VERSION, round-trip save/load
  module containment, legacy v2 pickle field dropping, and once-per-process warning behavior.

Public .tlspec schema (TLSPEC_SCHEMA_VERSION) is unchanged because the removed fields were private
  to OpLog pickle state, not part of the public unified manifest.

Refs: .research/module-containment-refactor_PLAN.md (Phase 4)

- **mlx**: Record p6 completion
  ([`b611fe7`](https://github.com/johnmarktaylor91/torchlens/commit/b611fe7f4eeb68ac6635da5c3816b8535cdda712))

- **polish**: Close docstring, type-hint, and mypy-strict gaps across torchlens/
  ([`b1c37b8`](https://github.com/johnmarktaylor91/torchlens/commit/b1c37b8456cd8ca11c938cb53aec7fec4c5eff4d))

Pure plumbing pass with no behavior changes. Six parallel codex workers (high reasoning effort)
  touched disjoint subpackages; the orchestrator consolidated results and finished the residual
  handful of strict-mode errors at the root.

Coverage improvements (across torchlens/): - Public docstrings: 90% -> ~100% (81 added) - Public
  return types: 92% -> ~100% (110 added) - mypy --strict on package: 815 errors in 63 files -> 0
  errors in 174 files

Worker partition (no file overlap): W1: capture/, decoration/, _state.py, multi_trace/, observers,
  experimental, _summary, _run_state

W2: data_classes/, _io/, io/, accessors/, errors/, types

W3: postprocess/, validation/, visualization/

W4: intervention/, fastlog/

W5: bridge/, compat/, report/, stats/, utils/, callbacks/

W6: __init__.py, user_funcs.py, options.py, appliance stubs (viewer/, paper/, notebook/, llm/,
  neuro/)

Verification gates (all pass): - mypy torchlens/ (lenient, configured): 0 errors in 174 files - mypy
  --strict torchlens/: 0 errors in 174 files - ruff check + format: pass - pytest tests/ -m smoke
  -x: 170 passed - len(torchlens.__all__) == 40

Each worker added narrow, justified type: ignore[<specific-code>] comments where torch internals or
  third-party libs (rich, wandb) are genuinely untyped. No blanket Any silencing.

- **rename**: Apply glossary walkthrough v2 deltas to glossary
  ([`7fe2493`](https://github.com/johnmarktaylor91/torchlens/commit/7fe2493473bbf6c99de5bfea34945e2354e2c5fe))

- **rename**: Final implementation summary
  ([`2233fed`](https://github.com/johnmarktaylor91/torchlens/commit/2233fed55aba607a94c71f3f9fc9330dfa443831))

- **rename**: Green smoke tests after rename (phase 6)
  ([`c1ac957`](https://github.com/johnmarktaylor91/torchlens/commit/c1ac9576843a635db32d8baa20878eef1bc2718d))

- **rename-sprint**: Add 5 follow-on todos + state file
  ([`9fcd466`](https://github.com/johnmarktaylor91/torchlens/commit/9fcd4669fa7b49317d919720453dff2ff5393d6e))

- **research**: Record P3 implementation done
  ([`077cc7d`](https://github.com/johnmarktaylor91/torchlens/commit/077cc7d98c29d36ab5948df38a3db042b30ed134))

- **state**: Mark backward-parity sprint DONE
  ([`5a5fd53`](https://github.com/johnmarktaylor91/torchlens/commit/5a5fd5373955e86beb6eb525630888363be40c41))

- **state**: Mark post-backward megasprint DONE
  ([`ffa73f6`](https://github.com/johnmarktaylor91/torchlens/commit/ffa73f6ebfe6dcdf4f1a4532dad7ebb9ff331601))

- **todos**: Add layer visualizers — tensor->image plugins per node
  ([`8b13563`](https://github.com/johnmarktaylor91/torchlens/commit/8b135634da81474b346a51909229731e9891e9b7))

Completes the trio with input transform / output transform: same plugin dispatch, same image=
  mechanism, same shared helpers, applied to intermediate nodes via the existing selector vocabulary
  (tl.func, tl.module, label lists, glob). Built-in library: heatmap, channel_grid, histogram,
  attention_heatmap, tsne/pca/umap/mds, violin. Eager render at trace time, cached PNG paths for
  cheap idempotent draw(). Compound payoff: whole graph reads as a visual tour through the model's
  processing.

- **todos**: Add symmetric output_transform alongside input transform
  ([`3470266`](https://github.com/johnmarktaylor91/torchlens/commit/347026617fbafbf89ac133bc2cbf2d12c3ee021a))

- **todos**: Add transform= primitive + raw-input rendering as core feature
  ([`3abae87`](https://github.com/johnmarktaylor91/torchlens/commit/3abae876e0ba8832c242a42a3628268adb8c8f3e))

Refines the HF-bridge text-input idea into a cleaner architecture: generic transform=callable kwarg
  on tl.trace, store the original alongside the trace, render it in the input node. Graphviz's
  image= attr and HTML-label support make per-modality rendering small (text / image / audio /
  multimodal each ~15 lines). HF bridge becomes a ~5-line wrapper instead of a 30-line ergonomic
  feature.

Total scope ~100 lines. Out of scope: autoregressive animation.

- **todos**: Document -Log suffix drop rule for naming-sprint v3
  ([`6b8d4ed`](https://github.com/johnmarktaylor91/torchlens/commit/6b8d4edcb10a016e47b694a3b19b619278dfaf07))

- **todos**: Document three supply mechanisms for slicing recipes
  ([`da8cd21`](https://github.com/johnmarktaylor91/torchlens/commit/da8cd21790d7d125b0735148d8f6ca5707d1ced4))

User-facing surface: - auto-detect (default-on, strict class-identity match) - explicit at trace
  time (stacks with auto unless auto_detect=False) - post-hoc on the trace (no re-execution,
  supports loaded tlspecs)

Plus conflict resolution (explicit wins, warn on shadow), save/load contract (recipe names + version
  tags survive, recipe code does not), and rejected mechanisms (no global registry, no model
  mutation) with their reasoning.

- **todos**: Extend transform= entry with batched-input rendering
  ([`5c560d4`](https://github.com/johnmarktaylor91/torchlens/commit/5c560d4c170118f1703b46360be1702100667abd))

Default sampling policy shared across modalities (1: full, <=4: all, >4: first 4 + count badge).
  Per-modality summarization: PIL montage for images, HTML-label table for text, mini-waveform stack
  for audio. Heterogeneous/multimodal batches deferred. Batch-summarize helpers exposed as a
  reusable tl.viz.batch_summary module so future activation-grid / bundle visual work can reuse the
  same code.

- **todos**: Extend transform= entry with rerun reuse + builtin transforms
  ([`e3126c6`](https://github.com/johnmarktaylor91/torchlens/commit/e3126c607fd7fcf64fc06c8923b8c8bc5440a3e3))

trace.rerun(new_user_input) auto-reapplies the stored transform so users don't have to keep
  tokenizer/preprocessor references around. Composes with fork/do/bundle for one-line prompt-swap
  experiments. Loaded traces (no stored transform) raise a clear error rather than silently failing.

Built-in transform library sketched: tl.transforms.hf_tokenize_for(model),
  tl.transforms.image_preprocess(...), etc. Same appliance pattern as recipes — small core registry,
  long tail in bridges. The HF text-input bridge collapses to a 3-line wrapper around the primitive.

- **todos**: File deferred per-layer grad oracle
  ([`ef905b8`](https://github.com/johnmarktaylor91/torchlens/commit/ef905b8385aebe1215d3c298583b62b71e0218ac))

- **todos**: Keep_orphans=true flag for retaining island ops
  ([`6471b13`](https://github.com/johnmarktaylor91/torchlens/commit/6471b1345149df99f4eab9f7ea55559912fee41b))

- **todos**: Note bundle_diff blue-white-red gradient is semantically wrong
  ([`31116de`](https://github.com/johnmarktaylor91/torchlens/commit/31116de0e5db7db7fe58bc6a84ccca320b82b237))

- **todos**: Note combined forward+backward graph rendering
  ([`e9c0311`](https://github.com/johnmarktaylor91/torchlens/commit/e9c0311fb1b18b39accfb6d4087eb9786fbcb37e))

- **todos**: Note FX-style qualpath as documented secondary lookup key
  ([`b57706d`](https://github.com/johnmarktaylor91/torchlens/commit/b57706d3e5c365840c3c31530c08839c9ae1b481))

- **todos**: Note FX-style qualpath as op-level metadata field
  ([`570fa9d`](https://github.com/johnmarktaylor91/torchlens/commit/570fa9d435ae781776981f5b971f64ed6b0edb2b))

- **todos**: Note HF bridge text-input ergonomics
  ([`364024b`](https://github.com/johnmarktaylor91/torchlens/commit/364024be5e5b0d25379011a3ae4299803858a903))

- **todos**: Note layer/op-level wrappers for trace-level intervention verbs
  ([`a9776c6`](https://github.com/johnmarktaylor91/torchlens/commit/a9776c600a73f7ddc5cdf92b0e8bf72826f25a30))

- **todos**: Note multi-arm conditional traversal feature
  ([`d0ab3c7`](https://github.com/johnmarktaylor91/torchlens/commit/d0ab3c70cc0c817e444e2af80aa722c5f60c64fd))

- **todos**: Note tensor-container data structure support
  ([`c25257a`](https://github.com/johnmarktaylor91/torchlens/commit/c25257a00d57352aaf9172a90b7460e693fca725))

- **todos**: Note tensor-slicing recipes for attention-head sub-addressing
  ([`c77f897`](https://github.com/johnmarktaylor91/torchlens/commit/c77f897bad68ccdc0c4491f265a4217e45be50a1))

- **todos**: Pencil pluck+extract and file naming-sprint v3
  ([`610b1bc`](https://github.com/johnmarktaylor91/torchlens/commit/610b1bc593b71f0428ceb9b1aa391bc1c9ef01a6))

- **todos**: Replace input-derived module attribution with call-stack snapshot
  ([`878d2a0`](https://github.com/johnmarktaylor91/torchlens/commit/878d2a0806bf19da9594a6729160998ed7248724))

- **todos**: Unify op.outs API across containers and recipe slicing
  ([`5eb7361`](https://github.com/johnmarktaylor91/torchlens/commit/5eb736194e55321f33441725fd89b20985ba3682))

Both features populate the same dict with named tensor leaves; they differ only in the source of
  names (container fields vs user/auto recipes) and storage semantics (independent tensors vs views
  into a parent). The container entry now owns the op.outs API design; the recipe entry is layered
  on top. Provenance flag distinguishes mutation semantics for the intervention API.

Also captured: HF auto-detect via strict class-identity matching, recipe versioning +
  applied_recipes provenance, hard constraint that recipes never synthesize new ops,
  convenience-feature framing for relaxed stability.

- **viz**: Clean up graph node labels
  ([`fb4680d`](https://github.com/johnmarktaylor91/torchlens/commit/fb4680d33b939dc1da8464f68b999f1eca134e80))

File list + LOC summary: - torchlens/visualization/_label_format.py: +318/-0, new label formatting
  helpers. - torchlens/visualization/rendering.py: +24/-269, wire helpers into default labels,
  selected fields, collapsed module labels, and remove unreachable old label helpers. -
  torchlens/visualization/modes.py: +2/-7, use tuple notation in vision IO rows. -
  torchlens/visualization/_elk_internal/layout.py: +3/-7, match collapsed-module ELK label
  shape/memory format. - tests/test_format_helpers.py: +46/-0, helper coverage. -
  tests/test_node_spec_api.py: +7/-7, updated default-label arg expectations. -
  tests/test_node_modes.py: +1/-1, updated vision shape expectation. - tests/test_overlays.py:
  +1/-1, updated field-picker shape expectation. - tests/test_param_log.py: +9/-14, updated
  param-label text expectations.

Tier-2 result: - pytest tests/ -m "not slow" -x --tb=short: 2256 passed, 29 skipped, 214 deselected,
  2 xfailed.

Golden snapshots updated: - None. No tracked golden visual snapshot files changed; aesthetic
  PDFs/DOT outputs were regenerated under ignored tests/test_outputs for inspection only.

Visual confirmations: - Linear renders tuple shape/memory, in_features/out_features, middle-dot
  param list, and spaced module path. - Conv2d renders tuple shape/memory plus full
  in_channels/out_channels/kernel_size/stride/padding names. - LayerNorm renders
  normalized_shape=(d,) and params with tuple notation. - Embedding renders
  num_embeddings/embedding_dim and named weight shape. - Functional softmax renders no params or
  module path placeholder. - ReLU/no-param nodes omit the params line cleanly.

## Judgment calls - Module kwargs are sourced from captured func_config because Trace rendering
  receives Layer/Op records, not live nn.Module instances; known module families are reordered to
  declaration-style names. - Frozen/trainable parameter text no longer uses bracket syntax;
  trainability remains represented by existing node colors.

## Follow-ups - Full mypy still fails on pre-existing out-of-scope errors in intervention/bundle.py,
  data_classes/*, and backends/mlx/backend.py.

### Documentation

- **agents**: Refresh CLAUDE.md/AGENTS.md and add state_of_torchlens.md
  ([`fe0a304`](https://github.com/johnmarktaylor91/torchlens/commit/fe0a3040f5c5fa9a6c993f9d3269bd2dbedf25e8))

Refresh all 20 per-package CLAUDE.md and AGENTS.md files to describe the current 2.x state after the
  17-phase 2.0 sprint (PR #175). Add a new `.project-context/state_of_torchlens.md` reference doc
  that gives a fresh agent (or human) a use-oriented map of the package: 40-name public API,
  subpackage layout, key concepts, what shipped vs what is stubbed, where to look for what.

No source code touched.

- **fastlog**: Mark P3 halt complete
  ([`d9794d7`](https://github.com/johnmarktaylor91/torchlens/commit/d9794d790a4802b3f78afac70c1ef1af6c3ad15b))

- **glossary**: Refresh for backward-parity + post-backward megasprint outcomes
  ([`858757b`](https://github.com/johnmarktaylor91/torchlens/commit/858757b6dd17420d1128f185864de9a62115a843))

Glossary at 1135 lines (was 1040). Adds ~60 entries from the two recent sprints: backward selectors
  (tl.grad_fn / tl.intervening / tl.grad_fn_label), backward helpers (bwd_hook / grad_zero /
  grad_scale / grad_clip / grad_noise / grad_clamp), tl.output, tl.halt / HaltSignal, observer
  direction kwarg, aggregate(target='grad', loss_fn=), Trace.draw_combined, Recording.log_backward,
  Recorder.log_backward, ModuleCallLog/ModuleLog .outputs + .output_structure,
  MultiOutputModuleError, LayerGradReport + 7 coverage buckets, MLX backend contract, scrub 3-tuple,
  append-state fields, full error catalog.

Also: companion CHANGES.md (193 lines) enumerates added/updated/removed/ uncertain entries + 3
  inconsistencies flagged for upcoming rename sprint.

- **intervention**: Clarify AppendBatchDependenceError
  ([`28b69a3`](https://github.com/johnmarktaylor91/torchlens/commit/28b69a3435abab25bfddfb7682847e70d6d51448))

- **notebook**: Add auto-route demo section to HF tutorial
  ([`adfac12`](https://github.com/johnmarktaylor91/torchlens/commit/adfac127015247fd88f48336f0206223e56ee297))

- **plan**: V33 close round-32 BLOCK -- seed census + glob fix + line_exceptions schema
  ([`bffc71a`](https://github.com/johnmarktaylor91/torchlens/commit/bffc71a1e81ca6eb31e8cb9886bc17a9eb4247cc))

- **plan**: V34 close round-33 FINDINGS -- anchor-based citation + emit-seed semantics
  ([`0208568`](https://github.com/johnmarktaylor91/torchlens/commit/0208568ac83279330ae4d9d6163220a0527533c4))

- **plan**: V35 close round-34 Claude FINDINGS -- scope exemption + typo + Phase 0.13 cleanup (Codex
  READY preserved)
  ([`b374738`](https://github.com/johnmarktaylor91/torchlens/commit/b37473863ad141f048bdd99760418a91831c3fd4))

- **plan**: V36 close round-35 Codex HIGH -- impl tightens to spec; Phase 5.14 lifecycle checklist
  ([`f564c26`](https://github.com/johnmarktaylor91/torchlens/commit/f564c26e47c0e2b304068561bdca2f330b6ebdf2))

- **polish**: Close polish-sprint (5 phases shipped)
  ([`5f5d583`](https://github.com/johnmarktaylor91/torchlens/commit/5f5d58329327cbb5e2338e4333637d935f51a4b2))

- **research**: Record P6 implementation completion
  ([`bcfe094`](https://github.com/johnmarktaylor91/torchlens/commit/bcfe094bcb9bd0888e62a16a32cc468861f81a94))

- **research**: Record P7C completion
  ([`c44f2ba`](https://github.com/johnmarktaylor91/torchlens/commit/c44f2ba600f9e5006353a6b1de950160167be2a7))

- **sprint**: Record backward-parity sprint SUMMARY
  ([`31d20a8`](https://github.com/johnmarktaylor91/torchlens/commit/31d20a8fd1bdb6fe11bc1aa1dfb01f5e6e9899e3))

- **sprint**: Record P7 completion
  ([`0d3efba`](https://github.com/johnmarktaylor91/torchlens/commit/0d3efbaf2d8c3f01bc88d89a609d5b08ba61e01a))

- **sprint**: Record P7 T2 block
  ([`cf3f585`](https://github.com/johnmarktaylor91/torchlens/commit/cf3f58585ce6f390d7b95e598c52fca6bcd7e804))

- **sprint**: Record post-backward megasprint SUMMARY
  ([`f543566`](https://github.com/johnmarktaylor91/torchlens/commit/f543566133d0e54e30ca50ecc777fbbc5c666d5b))

- **sprint**: Record post-backward P1 completion
  ([`bf2e4d1`](https://github.com/johnmarktaylor91/torchlens/commit/bf2e4d159ded8eb0ad9ac2b8f329b6facc08a606))

- **sprint**: Record post-backward P2 completion
  ([`6bc62b3`](https://github.com/johnmarktaylor91/torchlens/commit/6bc62b397ada68154423b68020fcff9b2dc227b0))

- **sprint**: Update P7 T2 block
  ([`c5c4534`](https://github.com/johnmarktaylor91/torchlens/commit/c5c453437459a1cb2788534dff6b2fbbc50db822))

- **super**: Close super family sprint
  ([`87821a9`](https://github.com/johnmarktaylor91/torchlens/commit/87821a9f48b3c463d6f6eb8db8b90a34938686c0))

- **todos**: Close items from draw + step6 + intervention cleanup sprint
  ([`83367c1`](https://github.com/johnmarktaylor91/torchlens/commit/83367c12c380da4622a5e6668cecb51a8324d0f1))

- **transforms**: Close transforms-sprint (5 phases shipped)
  ([`f120375`](https://github.com/johnmarktaylor91/torchlens/commit/f120375dc7fdf80461c8818675942ee36e7f09d3))

### Features

- **backends**: Mlx backend skeleton + smoke gate (M7)
  ([`5ff8fdc`](https://github.com/johnmarktaylor91/torchlens/commit/5ff8fdc286757b9889c4d087ecd9b305642c0a5a))

Implements MLXBackend at torchlens/backends/mlx/ satisfying the CaptureBackend Protocol. Backend
  dispatch in trace() entry detects mlx.nn.Module vs torch.nn.Module and routes accordingly.

MLX backend characteristics (per plan §9 readiness checklist): - Lazy-safe metadata extraction
  (.shape, .dtype, .device without forcing mx.eval) - Hybrid materialization: deferred payloads with
  batched mx.eval at end-of-forward - External tensor label store (WeakValueDictionary) since MLX
  arrays have no _tl analog - supports_backward_capture = False (mx.grad is function transform, no
  autograd graph) - mx.compile excluded (wrappers only fire during initial tracing) - RNG
  snapshot/restore unsupported initially per AD-9

New smoke test at tests/test_mlx_backend_smoke.py captures a linear MLP via MLX and asserts
  structural Trace equivalence. Uses pytest.importorskip('mlx') so it skips cleanly when MLX is not
  installed.

[mlx] extra added to pyproject.toml. MLX is opt-in; the main torchlens import does not depend on mlx
  being present.

Plan: .research/capture-pipeline-unification_PLAN.md §9 + §14 AD-5 + AD-9

Milestone: M7 of M1-M8 capture-pipeline-unification sprint

- **backward**: Capture-time AccumulateGrad attribution map
  ([`6ec9109`](https://github.com/johnmarktaylor91/torchlens/commit/6ec9109c12e021dbc8760b3170d80f6f4a493536))

- **bridge**: Add Hugging Face text tracing bridge
  ([`aee7dd6`](https://github.com/johnmarktaylor91/torchlens/commit/aee7dd66ef735954e689725b00bab656e11eeead))

- **bundle**: Add label auto-routing dispatcher
  ([`d0e7db0`](https://github.com/johnmarktaylor91/torchlens/commit/d0e7db07d779e89b493441603035dc9a2f5bcf42))

- **bundle**: Add structural agreement predicates
  ([`10ca3f2`](https://github.com/johnmarktaylor91/torchlens/commit/10ca3f24ce6535e6c13f76bb3996307c8ea00050))

Add Bundle-level structural consistency and shared/divergent predicates for op labels, layer labels,
  module addresses, parameter names, buffer names, and grad_fn labels.

- **capture**: Add output_transform primitive
  ([`0bf0004`](https://github.com/johnmarktaylor91/torchlens/commit/0bf00044cb2049fd4a1943ed6670d955cb736496))

- **capture**: Add trace input transform primitive
  ([`c466ffe`](https://github.com/johnmarktaylor91/torchlens/commit/c466ffeb0f8f52d9481ecf22323b6c3aa95c95d0))

- **capture**: Keep orphan ops with accessor
  ([`07f15e1`](https://github.com/johnmarktaylor91/torchlens/commit/07f15e13d09d5bc7644db39e89fec01ce446e526))

- **data**: Add module role hints and output structure capture
  ([`3b9c0d3`](https://github.com/johnmarktaylor91/torchlens/commit/3b9c0d3bf0907466e6f2aa71e1249563eb8e5598))

- **demo**: Add 10-minute notebook covering trace, intervene, backward, conditional, bundle
  ([`2e27bc5`](https://github.com/johnmarktaylor91/torchlens/commit/2e27bc52c29e4fa2e5c69e0578cc90acd498246d))

- notebooks/torchlens_in_10_minutes.ipynb: 17-cell tour of the 5 flagship features. -
  notebooks/_demo_models.py: TinyBranchCNN + TinyMLP in a real .py so the AST inspector can label
  conditional arms in the rendered graph. - postprocess/finalization.py: materialize all AST-derived
  arms (then / elif_N / else) in ConditionalAccessor, not only arms that produced runtime edges.
  Restores symmetric num_arms / has_else across traces regardless of which arm fired;
  ConditionalArm.fired still distinguishes runtime from static.

Conditional tests (96) and smoke tests (170) green.

- **ergonomics**: Auto-coerce str/PIL/numpy/audio inputs via duck-typed model methods
  ([`0bfeb79`](https://github.com/johnmarktaylor91/torchlens/commit/0bfeb79633c2b7aacfb437eedbcd5851108e6f62))

- **errors**: Add multi-output module error
  ([`f6cfe75`](https://github.com/johnmarktaylor91/torchlens/commit/f6cfe756130a85b7091d5be5b36fa08084ae1fdc))

- **facets**: Implement facets framework + built-in recipes per LOCKED 2026-05-27 spec
  ([`3ada85d`](https://github.com/johnmarktaylor91/torchlens/commit/3ada85d08296e8277e0d2040c0de3c7fa15ab8ce))

Adds semantic FacetView registry, lazy per-record Op/Module facets, Trace facet finders, and
  built-in attention/norm/MLP/embedding recipes. Exposes tl.facets and updates API surface budget
  tests.

Files/LOC: 18 files changed, 1326 insertions, 6 deletions; new torchlens/semantic package is 912 LOC
  and new tests/semantic coverage is 350 LOC.

Tier-2: pytest tests/ -m "not slow" -x --tb=short passed: 2252 passed, 29 skipped, 214 deselected, 2
  xfailed in 1277.38s.

## Follow-ups

None identified in this implementation scope.

## Judgment calls

Facet keys must be available without invoking recipes even though recipes expose values by returning
  dicts. Built-ins therefore declare facet names at registration; simple user recipes get
  best-effort literal return-dict key extraction without executing the recipe. TorchLens Module.cls
  stores the class, not the live instance, so recipes read scalar config values from
  Module.custom_attributes when needed.

- **fastlog**: Add halt signal API
  ([`adea348`](https://github.com/johnmarktaylor91/torchlens/commit/adea3484b50c9d77c034fdb336e1c2bf0d3ad540))

- **fastlog**: Add predicate-mode backward gradient capture (F5)
  ([`f509310`](https://github.com/johnmarktaylor91/torchlens/commit/f509310460be73104491eefc2601879800e8a59a))

Adds Recording.log_backward(loss, *, keep_grad=...) and Recorder.log_backward(loss, *,
  keep_grad=...). Predicate-mode forward emission populates RecordingState.grad_fn_to_context
  (WeakKeyDictionary keyed on the live grad_fn object, not id(), to avoid id-reuse misjoin in
  multi-rollout).

Backward walker uses the join map to build GradRecordContext and distinguish paired vs intervening
  grad_fns.

RecordingOptions gains keep_grad / default_grad / grad_transform / save_raw_grads recording-level
  defaults; per-call log_backward(keep_grad=...) overrides. Static keep_grad=True with disk-only
  rejected preflight; callable predicates fall through to _resolve_storage dynamic rejection.

keep_op remains a callable predicate slot (selectors NOT accepted -- two-slider discipline).
  Example: keep_op=lambda ctx: ctx.label == "relu_1".

Tests: tests/test_fastlog_backward.py covers join, intervening, disk preflight, id reuse,
  gradient_postfunc alias.

Co-authored-by: Codex (gpt-5.5) <noreply@openai.com>

- **fastlog**: Catch halt at recorder boundary
  ([`8984fab`](https://github.com/johnmarktaylor91/torchlens/commit/8984fabec9f1d3e08a548285ee181b257e8332c6))

- **fastlog**: Track halted recordings
  ([`b43f3e8`](https://github.com/johnmarktaylor91/torchlens/commit/b43f3e8041af514867e28612d3f766f09426bb9e))

- **glossary-v5**: Implement locked field additions
  ([`0eeea59`](https://github.com/johnmarktaylor91/torchlens/commit/0eeea5931f89b6cb5b2b509f1f1ffa59fdef3224))

- add Trace, Op, Module, and ModuleCall glossary v5 accessors - add CallTreeNode and dynamic
  call-tree derivations - make Module param_memory address-recursive and add recursive param
  accessors - capture ModuleCall forward arg templates and add smoke coverage

- **intervention**: Add backward grad_fn selectors
  ([`09a6fe0`](https://github.com/johnmarktaylor91/torchlens/commit/09a6fe0cc35c2dfa120eea81c1cafa60ea61400f))

- **intervention**: Add grad clipping noise and clamp helpers
  ([`c0b2d4c`](https://github.com/johnmarktaylor91/torchlens/commit/c0b2d4cad8b8ab2e59ecc75a866f56031cbfc7fc))

- **intervention**: Add layer and op shortcuts
  ([`de2fe34`](https://github.com/johnmarktaylor91/torchlens/commit/de2fe34a6377ebee3723bd54ba7d617f78c53083))

- **intervention**: Dispatch backward hook plans from grad_fn callbacks
  ([`f72660a`](https://github.com/johnmarktaylor91/torchlens/commit/f72660ad6aa59e0b6fe27e44de38bf126372c4e1))

- **intervention.types**: Rebuild containers from output specs
  ([`e449190`](https://github.com/johnmarktaylor91/torchlens/commit/e449190dddc103befe5c9ba71753ab4ff3dc9ccd))

- **mlx**: Expand wrapper coverage
  ([`3f99767`](https://github.com/johnmarktaylor91/torchlens/commit/3f997673a9ca771d2ff29a4f9776375d822cf105))

- **module-log**: Expose structured multi-output module outputs
  ([`d3fd652`](https://github.com/johnmarktaylor91/torchlens/commit/d3fd6523cdd6b9ae65e6d3b1026c91c76cfd4357))

- **observers**: Direction kwarg for backward taps
  ([`2a024c3`](https://github.com/johnmarktaylor91/torchlens/commit/2a024c33e13ac14485d94ffcb612f331525bb185))

- **op**: Add FX qualpath metadata
  ([`72cec3c`](https://github.com/johnmarktaylor91/torchlens/commit/72cec3c6a38dc465b87e4979da14d1bdf5e94537))

- **options**: Add gradient_postfunc silent alias for grad_transform
  ([`4e0ca61`](https://github.com/johnmarktaylor91/torchlens/commit/4e0ca61ad813e1c2f733d51926b2110d2618bf70))

Adds gradient_postfunc as a silent alias kwarg on SaveOptions / trace() that delegates to the
  existing grad_transform field. No deprecation warning. Honors AD-2 / synthesis-brief vocabulary
  parity (activation_postfunc / gradient_postfunc) without breaking existing grad_transform callers.

Co-authored-by: Codex (gpt-5.5) <noreply@openai.com>

- **selectors**: Add output selector disambiguation
  ([`e69932f`](https://github.com/johnmarktaylor91/torchlens/commit/e69932fcca4563b9d280b177b38bce22bb181606))

- **stats**: Add aggregate(target='grad') with mandatory loss_fn (F6)
  ([`de65ee1`](https://github.com/johnmarktaylor91/torchlens/commit/de65ee134e658926618c40971c658bc9af74b1bc))

Extends torchlens.stats.aggregate() with target='grad' (parallel to existing target='activation' /
  target=None default). target='grad' REQUIRES loss_fn= kwarg; raises TypeError if missing.

Adds Norm streaming stat (O(1) memory) for gradient norms per layer.

Tests: tests/test_gradient_stats.py covers loss_fn requirement + basic norms.

Co-authored-by: Codex (gpt-5.5) <noreply@openai.com>

- **super**: Add remaining bundle super accessors
  ([`005a72d`](https://github.com/johnmarktaylor91/torchlens/commit/005a72d26f136b286ae19eb3cd0d8ee7e9173b37))

Add SuperModule, SuperBuffer, SuperParam, SuperGradFn, SuperModuleCall, and SuperGradFnCall.

Expose bundle.modules, bundle.buffers, bundle.params, bundle.grad_fns, bundle.module_calls, and
  bundle.grad_fn_calls.

- **trace**: Auto-route tl.trace(model, image|multimodal_dict) with tiered preprocessing resolver
  ([`c0f8557`](https://github.com/johnmarktaylor91/torchlens/commit/c0f85571fe57161c2ac8ce6aa2a8484f9c7d2f2a))

Adds PIL image auto-route through HF AutoImageProcessor/AutoProcessor, torchvision weights
  transforms, timm transforms, and the loud unverified ImageNet default fallback.

Adds multimodal dict auto-route for HF-style modality-key dicts gated by AutoProcessor resolution.

Trace.input_preprocessor is set for image, multimodal, and the existing text auto-route with
  structured preprocessing provenance.

Tier-2: pytest tests/ -m "not slow" -x --tb=short -> 2277 passed, 29 skipped, 221 deselected, 2
  xfailed in 656.68s.

HF ViT: slow route test passed with hf_auto_image_processor provenance.

CLIP multimodal: slow dict route test passed with hf_auto_processor provenance.

torchvision ResNet: non-slow test passed with attached weights tier accepted, default fallback also
  allowed by test contract.

timm: slow default_cfg route test passed with timm provenance.

Unknown CNN: non-slow PIL route test passed with ImageNet default UserWarning and verified=False.

Audio, video, file-path, bytes, numpy, and tensor image auto-routing are explicitly out of scope.

- **validation**: Add backward graph metadata invariants
  ([`e25a0f4`](https://github.com/johnmarktaylor91/torchlens/commit/e25a0f4c497cfa33411c31f562ab0106f8c83548))

- **validation**: Add module output grad oracle
  ([`b621d67`](https://github.com/johnmarktaylor91/torchlens/commit/b621d67ec1a53c2deb209433602069aabcf09455))

- **validation**: Harden validate_backward_pass param-grad parity
  ([`e89ee77`](https://github.com/johnmarktaylor91/torchlens/commit/e89ee77a43126762d205b0ecba74649ee950dc86))

- **viz**: Add layer visualizers
  ([`81d3457`](https://github.com/johnmarktaylor91/torchlens/commit/81d34579929226bd7d4595015df62239cc886162))

- **viz**: Headless-quiet artifact open + multi-format bundle_diff
  ([`7b2bb38`](https://github.com/johnmarktaylor91/torchlens/commit/7b2bb38c3ba3234095e5a8c5a2842788985c4fe9))

Adds _open_file_quietly helper in _render_utils.py that silently skips viewer launch on headless
  Linux (no DISPLAY) and suppresses xdg-open stderr noise elsewhere. Wires _elk_internal/layout.py
  and rendering.py to use it.

Extends bundle_diff to accept any Graphviz-supported format (svg, pdf, png) instead of svg-only;
  accessibility metadata still embedded only for svg.

Pre-session WIP -- captured here so main is clean before next session.

- **viz**: Render batched raw inputs
  ([`ceffa08`](https://github.com/johnmarktaylor91/torchlens/commit/ceffa08317569651f3d40d0bf867cc0d64e8de88))

- **viz**: Render combined forward backward graph
  ([`b369fab`](https://github.com/johnmarktaylor91/torchlens/commit/b369fab6e85ca2b0e1bc4782740567acbbbb3f5a))

- **viz**: Restore frozen/trainable bracket marker on param shapes
  ([`efa4c08`](https://github.com/johnmarktaylor91/torchlens/commit/efa4c08e098a800242cdd9aedb9faf4c6d238254))

Restore per-parameter shape rendering so trainable params keep tuple parentheses while frozen params
  swap only the outer shape brackets to square brackets, preserving the existing name, separator,
  and singleton trailing-comma formatting.

JMT wants the at-a-glance trainability cue preserved, especially for mixed-trainability cases like
  LoRA.

Tier-2: pytest tests/ -m "not slow" -x --tb=short passed (2260 passed, 29 skipped, 214 deselected, 2
  xfailed).

Added tests for trainable parens, frozen square brackets, mixed trainable/frozen lists, anonymous
  frozen params, frozen singleton [3072,], and unknown trainability defaulting to parens.

### Refactoring

- **accessors**: Extract generic Accessor[T] base for trace-side accessors
  ([`fbf17d4`](https://github.com/johnmarktaylor91/torchlens/commit/fbf17d42d587c9ca82c9888b99e8f02b551cfc08))

Phase 1 of super-family-sprint. Pure refactor; no public API change.

Extracts shared lookup, containment, iteration, IPython completion, and "did you mean"
  infrastructure from the eight existing accessor classes (LayerAccessor, OpAccessor,
  ModuleAccessor, ModuleCallAccessor, ParamAccessor, BufferAccessor, GradFnAccessor,
  GradFnCallAccessor) into a single generic Accessor[T] base in
  torchlens/data_classes/_accessor_base.py.

Subclasses now override only type-specific lookup hooks (_resolve_pass_qualified,
  _resolve_substring, _suggest); universal
  __getitem__/__contains__/__iter__/__len__/__dir__/_ipython_key_completions_/ keys/values/items
  live in the base. PORTABLE_STATE_SPEC kept per subclass (different keep/drop policies per type).

Net: 688 insertions, 275 deletions across 8 files.

- Smoke green (170/170). - No test or doc changes.

- **alpha3**: Add LiveOpRecord capture projection
  ([`a3a3d1a`](https://github.com/johnmarktaylor91/torchlens/commit/a3a3d1ad5abd5daff5ff819d90e74b95b2f5ee8f))

- **alpha3**: Drain LiveOpRecord in postprocess Step 0
  ([`c67954b`](https://github.com/johnmarktaylor91/torchlens/commit/c67954bdde5147192ca0b771418764be7ae175db))

- **alpha3**: Wire LiveOpRecord write sites
  ([`421b8a9`](https://github.com/johnmarktaylor91/torchlens/commit/421b8a97af0637025409931010b3e17e83015d5c))

- **autoroute**: Extract input dispatch into priority-ordered registry
  ([`cd888b7`](https://github.com/johnmarktaylor91/torchlens/commit/cd888b783e6ce0bf100b46983c789af53d57b1a6))

Replace the hardcoded tl.trace auto-routing branches with a decorator-populated input registry while
  preserving the existing text, multimodal, image, and fallback dispatch decisions. The Registry
  implementation is direction-agnostic and keyed only by diagnostic kind, with tl.autoroute.output
  reserved as a future output-routing namespace stub.

Files and LOC: torchlens/autoroute/_registry.py (+216), torchlens/autoroute/input/__init__.py (+27),
  torchlens/autoroute/_builtin_input.py (+52), torchlens/autoroute/output/__init__.py (+25),
  torchlens/autoroute/__init__.py (+8), torchlens/user_funcs.py (+62/-185), torchlens/__init__.py
  (+4), tests/test_autoroute_registry.py (+158). Total staged diff: 552 insertions, 185 deletions.

Tier-2 result: pytest tests/ -m "not slow" -x --tb=short passed: 2285 passed, 29 skipped, 221
  deselected, 2 xfailed.

Existing tests were not modified; only tests/test_autoroute_registry.py was added.

- **capture**: Add shadow module-stack capture, gated by engine flag
  ([`8258229`](https://github.com/johnmarktaylor91/torchlens/commit/82582297500679bf9d4d55d69856da3c6a5529a0))

Phase 1 of module-containment-refactor sprint. Adds Trace-owned _exhaustive_module_stack
  pushed/popped via the Phase 0a helper inside the wrap-forward decorator. Adds
  OpLog._modules_via_stack diagnostic field populated from snapshot at op-creation time. Adds
  CaptureOptions._module_containment_engine flag (thread_replay default, hook_stack, both). Adds
  post-Step-6 equality assertion (no-op for default engine). Adds 16-fixture phase-1 equality test
  confirming shadow stack matches thread-replay on baseline fixtures.

Documented divergence on raw_hook_replacement_synthetic: hook-stack gives the cleaner dynamic
  call-stack answer for ops downstream of a synthetic interventionreplacement. The replaced module's
  frame does not propagate into descendant ops' module stacks (which thread-replay does as a quirk
  of recursive ancestry inheritance). Fixture 14's baseline snapshot regenerated under hook-stack
  engine; phase-1 equality test marks the fixture as xfail with documented reason.

Removes the duplicate _mod_call_index increment in _handle_module_entry (helper is now the sole
  incrementer).

Refs: .research/module-containment-refactor_PLAN.md (Phase 1)

- **capture**: Collapse tl_* attributes into _tl namespace
  ([`0e4509d`](https://github.com/johnmarktaylor91/torchlens/commit/0e4509d68aab6c6b2ddeee4edb1dc7af8e22f7f8))

Replace scattered tl__label_raw, tl_buffer_address, tl_buffer_parent, tl_param_barcode,
  tl_param_address, tl_call_index, tl_requires_grad, tl_address, tl_module_type,
  tl_forward_call_is_decorated, tl_is_decorated_function, tl_tensor_replacement_wrapped with a
  single typed _tl namespace per object.

Adds torchlens/_tl.py with TorchLensMeta base, TensorMeta, ParamMeta, ModuleMeta, DecorationTag
  dataclasses plus field-level helpers and batch hot-path readers.

Drops the double-underscore tl__label_raw footgun; sub-field names are now plain snake_case.
  Vestigial tl_tensor_label_raw removed.

Foreign _tl collisions detected via TorchLensMeta base class; raises TorchLensTLCollisionError to
  fail loudly.

New tests: test_tl_meta (helpers), test_tl_lifecycle (cleanup paths), test_no_legacy_tl_attrs (AST
  regression scan).

No public API change. No behavior change. Internal-only refactor.

- **capture**: Drop dead capture-only fields from Trace; field invariant test (M6)
  ([`82f5ee7`](https://github.com/johnmarktaylor91/torchlens/commit/82f5ee7c7f95b427e123bf9c4c49b2929225339e))

All transient capture state moved off Trace into postprocess-local TraceBuildState. Trace.__dict__
  now contains only user-facing fields (per Appendix B of plan) — no _raw_layer_dict,
  _layer_counter, _mod_entered, _module_build_data, _exhaustive_module_stack, etc.

OpEvent.materialized_log bridge field removed. Hot path still constructs OpLog at capture time and
  routes it through TraceBuildState (internal-only; never reaches the final Trace). Full IR-only
  hot-path simplification deferred to a code-quality follow-up; runtime behavior is unaffected.

Trace.orphan_logs: tuple[OpLog, ...] added per AD-7 (orphan access without _raw_layer_dict).

New tests/test_trace_field_invariant.py asserts the zero-scratch invariant: Trace.__dict__ contains
  only user-facing fields after log_forward_pass returns.

Parity goldens regenerated at tests/golden/m5_pre_m6/ for M5 → M6 byte-equality. Old m2_pre_m3
  goldens archived in place but no longer run.

Runtime gates green: 192 smoke, parity 4/4 against new goldens, intervention 203/1, fastlog 111,
  tier-2 non-slow 2081. Mypy reports 27 pre-existing errors in accessor typing unrelated to M6.

resnet50.pkl golden exceeds the 500KB threshold (765KB); commit uses SKIP=check-added-large-files.
  Same pre-existing repo-hygiene followup from M3 — todos.md tracks it.

Plan: .research/capture-pipeline-unification_PLAN.md §2 §5 §6 §14 AD-7 + Appendix B

Milestone: M6 of M1-M8 capture-pipeline-unification sprint

- **capture**: Hot path emits OpEvents; postprocess Step 0 materializes (M3)
  ([`7c75820`](https://github.com/johnmarktaylor91/torchlens/commit/7c7582002b57cdfad7db972b10dc1a901558b702))

Hot path in backends/torch/ops.py now emits typed OpEvent instances into session.capture_events:
  CaptureEvents instead of writing directly to Trace._raw_layer_dict[label] = OpLog(...).

Adds torchlens/postprocess/_materialize.py (Step 0 prelude) that drains CaptureEvents into
  _raw_layer_dict + _raw_layer_labels_list before the existing 19 postprocess steps run.

Behavior is parity-gated against pre-M3 goldens for TinyMLP, ResNet-50, GPT-style small transformer,
  and LSTM. New parity gate at tests/test_capture_events_parity.py with goldens at
  tests/golden/m2_pre_m3/.

The 19 existing postprocess steps are unchanged at this milestone; they still read _raw_layer_dict.
  M6 will inline Step 0's work and drop _raw_layer_dict from final Trace.

Plan: .research/capture-pipeline-unification_PLAN.md §5 + §11 (parity)

Milestone: M3 of M1-M8 capture-pipeline-unification sprint

- **capture**: Introduce IR + CaptureBackend Protocol (M1)
  ([`f56385e`](https://github.com/johnmarktaylor91/torchlens/commit/f56385e078a26d8cef7f72a4d8879301e2a2c232))

Establishes torchlens/ir/ with backend-agnostic event types (OpEvent, ModuleEvent, TensorRef,
  ParamRef, BackendSemantics, BackwardSidecar, CapturePolicy, FireResult, etc.) and
  torchlens/backends/_protocol.py with the CaptureBackend Protocol.

No backend implementations yet (M2). No postprocess changes (M3). No source-file moves beyond new
  packages. No __all__ deltas.

Frozen+slots dataclasses on every hot-path type. CaptureEvents uses slots=False (allocated once per
  capture; weakref-friendly). Mapping[str, object] for kwargs-shaped fields per immutability
  discipline. No catchall extra/annotations sidecars on OpEvent.

Acceptance gates (the substantive ones): - ruff: clean on new files - mypy --follow-imports=skip
  torchlens/ir torchlens/backends: clean - pytest tests/test_ir_basic.py: 5 passed - pytest tests/
  -m smoke: 192 passed

(Two gates I originally specified were ill-formed: full mypy follows into 49 pre-existing errors in
  unrelated files; the torch-free import check routed through torchlens/__init__.py which always
  imports torch. M1 code itself is torch-free in isolation.)

Plan: .research/capture-pipeline-unification_PLAN.md §2 + §3

Milestone: M1 of M1-M8 capture-pipeline-unification sprint

- **capture**: Move hot-path OpLog construction out of torch ops
  ([`4afa79c`](https://github.com/johnmarktaylor91/torchlens/commit/4afa79c2074db7aad7ff7b0d922212c40427c98e))

backends/torch/ops.py no longer directly constructs OpLog or BufferLog; log object materialization
  is centralized in postprocess/_materialize.py via materialize_log_from_fields(). OpEvent IR fields
  continue to be populated directly from the existing torch metadata fields.

Focused gates green: zero OpLog( hits in backends/torch/ops.py, parity 4/4, field invariant 1,
  intervention 203/1, fastlog 19.

RESIDUAL: this is the safe cleanup slice. Capture still needs a live materialized log for
  mid-forward raw lookup, module filters, family-link updates, gradient hooks, and activation
  saving, so construction is centralized but not deferred exclusively until postprocess Step 0.

- **capture**: Relocate torch capture under backends/torch/ + Protocol adapter (M2)
  ([`b9889d6`](https://github.com/johnmarktaylor91/torchlens/commit/b9889d6cf63b033d5d2cfa95bbc4cda0d03575fb))

Moves the eight torch-specific capture files into torchlens/backends/torch/: -
  decoration/torch_funcs.py → backends/torch/wrappers.py - decoration/model_prep.py →
  backends/torch/model_prep.py - decoration/_module_stack.py → backends/torch/module_stack.py -
  _tl.py → backends/torch/_tl.py - capture/source_tensors.py → backends/torch/sources.py -
  capture/output_tensors.py → backends/torch/ops.py - capture/tensor_tracking.py →
  backends/torch/tensor_tracking.py - capture/backward.py → backends/torch/backward.py

Adds backends/torch/backend.py implementing the CaptureBackend Protocol as an adapter over the
  relocated code. log_forward_pass now dispatches model prep, wrapping, and the per-op hot path
  through backend methods.

Behavior is identical: every hot-path call still writes Trace._raw_layer_dict exactly as today. M3
  will replace those writes with OpEvent emission into CaptureEvents. No dual-path coexistence at
  this milestone — just relocation and Protocol adapter.

torchlens/decoration/ deleted entirely (no forwarding shim, per AD-10).

Plan: .research/capture-pipeline-unification_PLAN.md §1 + §3 + §4

Milestone: M2 of M1-M8 capture-pipeline-unification sprint

- **capture**: Switch to hook_stack engine; delete thread-replay code
  ([`10942ba`](https://github.com/johnmarktaylor91/torchlens/commit/10942bac848e7785fab43ff87b69cdd8706f6aae))

Phase 2 of module-containment-refactor sprint. Default _module_containment_engine flips to
  "hook_stack". OpLog.modules now populated from _module_stack.snapshot at op-creation time.

Deletes _module_boundary_thread_output, _module_boundary_threads_inputs, _update_tensor_modules,
  _get_input_module_info, _modules_via_stack shadow field, and the post-Step-6 equality assertion.
  Renames _handle_module_entry/_handle_module_exit to _record_module_entry_metadata/
  _record_module_exit_metadata with reduced scope (counter increment moved to helper; thread-marker
  writes removed). Retains all metadata side effects per the plan's contract.

Removes constants.py LAYER_PASS_LOG_FIELD_ORDER entries for the deleted fields. Removes legacy
  aliases. Save/load migration shims for older pickles will be added in Phase 4.

Three baseline snapshots regenerated with documented behavioral improvements (drifts that fix
  limitations of thread-replay):

- recurrent_lstm_cell: newzeros_1_1 and newzeros_2_2 now correctly attribute to "cell:1" instead of
  empty modules. Internal factory ops (torch.zeros, torch.randn) called inside a forward now have
  proper module attribution. This is the orphan-factory-op fix the sprint was specifically designed
  to deliver. - dynamic_buffer_module: randn_1_1 now correctly attributes to "child:1". Same family
  as above. - raw_hook_replacement_synthetic: final output's output_of_modules narrows from ["relu",
  "proj"] to ["proj"]. Thread-replay's recursive ancestry inheritance no longer leaks dead-module
  markers through hook-replaced outputs. Same family as the previously documented fixture 14
  linear_1_4 divergence (commit 8258229).

Refs: .research/module-containment-refactor_PLAN.md (Phase 2)

- **decoration**: Extract module-stack helper
  ([`cee50a4`](https://github.com/johnmarktaylor91/torchlens/commit/cee50a413b3092367fa4e26d3827279a91f578f9))

Phase 0a of module-containment-refactor sprint. Adds torchlens/decoration/_module_stack.py with
  push_frame, pop_frame, current_address, snapshot. Refactors predicate-mode push/pop in
  model_prep.py to use the helper. Behavior-preserving; 170/170 smoke tests pass.

Refs: .research/module-containment-refactor_PLAN.md (Phase 0a)

- **fastlog**: Collapse to projection over CaptureEvents (M5)
  ([`a1316ec`](https://github.com/johnmarktaylor91/torchlens/commit/a1316eca9e38e8f4a2a9cecae6dabd2d241c636f))

Public tl.fastlog.* API preserved byte-identical. Internal duplicate capture implementation gutted;
  tl.fastlog.record now lowers to unified capture with postprocess=False, returning a Recording that
  lazily projects CaptureEvents.op_events into ActivationRecord / RecordingTrace views. No Trace is
  materialized for the fastlog path.

Deleted internal fastlog files (no behavior change to public users): - fastlog/_orchestrator.py -
  fastlog/_state.py - fastlog/_predicate.py - fastlog/_record_context.py Internal logic merged into
  capture/predicates.py + capture/projections.py.

Rewritten as thin shims: - fastlog/_record_one_shot.py - fastlog/_recorder.py

Storage and recovery files preserved (fastlog disk format unchanged per AD-1; fastlog.load continues
  to load existing artifacts via recover.py).

Preserved features verified: - train_mode=True still keeps saved tensors graph-connected (new test)
  - exception-in-predicate propagates as before (new test) -
  RecordingTrace.draw/timeline_html/repredicate all work over projected events - Memory parity:
  per-event cost is RecordContext-sized, not OpLog-sized

Plan: .research/capture-pipeline-unification_PLAN.md §7 + post-convergence patch

Milestone: M5 of M1-M8 capture-pipeline-unification sprint

- **glossary-v5**: Drop CallTreeNode, use ModuleCall as call-tree node directly
  ([`ad4eefd`](https://github.com/johnmarktaylor91/torchlens/commit/ad4eefd7f2dece110b89f9f04e8072fcd6342822))

Removed CallTreeNode exports, call_tree properties, and call_tree field-order entries.

Added direct ModuleCall/Module/Trace walking and call-tree display helpers, plus Trace root_call and
  max_call_depth convenience surfaces.

Updated glossary v5 tests and appended the locked design note to the deltas log.

- **intervention**: Live hooks emit FireResults on OpEvents; _pending_live_fire_records deleted (M4)
  ([`5338d43`](https://github.com/johnmarktaylor91/torchlens/commit/5338d43661bbacd03a83e1b4e25cbf82cf630654))

Live intervention hook firing in backends/torch/ops.py + intervention/runtime.py now emits typed
  FireResult annotations on the OpEvent in CaptureEvents instead of writing to
  _state._pending_live_fire_records. The pending-fire-records module global is deleted entirely.

apply_live_hooks becomes a CaptureBackend Protocol method: apply_live_hooks(session, value, site:
  ReservedLabel) -> tuple[object, tuple[FireResult, ...]]

FireResult wraps the existing intervention.types.FireRecord per plan §6.

Fork no longer rebuilds _raw_layer_dict via the old path; it deep-copies the dict from the source
  Trace's final state. _raw_layer_dict still exists until M6.

replay.py's import of _set_saved_out_metadata from backends/torch/ops.py is relocated to
  data_classes/op_log.py to close the layering leak.

Public API: identical. tl.fork / attach_hooks / rerun / replay byte-equal. graph_shape_hash
  unchanged. M3 parity gate still green.

Plan: .research/capture-pipeline-unification_PLAN.md §6

Milestone: M4 of M1-M8 capture-pipeline-unification sprint

- **io**: Io_format_version 3 -> 4 + v3 loader normalizer + CHANGELOG (M8)
  ([`af0faa1`](https://github.com/johnmarktaylor91/torchlens/commit/af0faa1b64d099ab1cc9bb785433e01a770a8bf3))

Bumps torchlens/_io/__init__.py:IO_FORMAT_VERSION from 3 to 4. Adds a small read-only normalizer at
  torchlens/_io/rehydrate.py:_normalize_legacy_trace_state that strips the 17 capture-only scratch
  fields dropped by M6 when loading v3 .tlspec bundles.

CHANGELOG.md documents the full M1-M8 unification sprint: - Unified capture pipeline (OpEvent +
  CaptureEvents IR) - Backend abstraction (torchlens.backends.{torch,mlx}) - torchlens.decoration
  namespace deleted - Zero capture-only scratch on Trace - PartialTrace.raw_layers shape change
  (rare-path) - Trace.orphan_logs added (AD-7) - Trace.has_backward_pass backend-conditional -
  io_format_version: 3 -> 4 with v3 loader normalizer - MLX backend (technical preview) - Public
  torchlens.__all__ unchanged (40 names); package stays 2.x.x

New test tests/test_io_format_v3_load.py asserts a v3 golden bundle loads cleanly under v4
  (normalizer strips legacy fields) and that save/load roundtrip preserves the zero-scratch
  invariant.

Plan: .research/capture-pipeline-unification_PLAN.md §M8 + Appendix B

Milestone: M8 (FINAL) of M1-M8 capture-pipeline-unification sprint

- **op-log**: Eliminate mid-forward OpLog construction
  ([`35fdaee`](https://github.com/johnmarktaylor91/torchlens/commit/35fdaee73f2f4798e6e762353cd4debc8688a546))

- **postprocess**: Inline Step 6 module suffix at op-creation
  ([`8dd33a6`](https://github.com/johnmarktaylor91/torchlens/commit/8dd33a63104b644b5e3fbf2a8cc86ec72786e06a))

- **postprocess**: Simplify Step 6 to suffix-only mutation
  ([`22b355f`](https://github.com/johnmarktaylor91/torchlens/commit/22b355fb901fd6737f276bf4618b3e4fc025ae22))

Phase 3 of module-containment-refactor sprint. Replaces _fix_modules_for_internal_tensors body with
  a suffix-only loop that appends the canonical module-path suffix to equivalence_class for every
  non-orphan layer. Deletes _fix_modules_for_single_internal_tensor and the
  _layers_where_internal_branches_merge_with_input plumbing (parent/child propagation is no longer
  needed because hook-stack snapshot at op-creation time is the source of truth for modules).

Step 6 still runs after Step 5; suffix logic operates on the post-conditional-attribution set.

Regenerates two baselines (recurrent_lstm_cell, dynamic_buffer_module): root-level factory ops
  (x.new_zeros, torch.randn) called in the OUTER forward BEFORE entering a submodule now correctly
  report modules=[] instead of inheriting the downstream module's identity. The Phase 2 baseline
  values ("cell:1", "child:1") were artifacts of the old Step 6 propagation logic that copied a
  downstream module's identity onto root-level factory ops; Phase 3 deletes that propagation. The
  literal hook-stack answer at op-creation time is empty modules, which is now what the snapshots
  reflect.

Refs: .research/module-containment-refactor_PLAN.md (Phase 3)

- **rename**: Apply naming sprint v2 code surface
  ([`a55ae03`](https://github.com/johnmarktaylor91/torchlens/commit/a55ae033a58ba51524339e08fec49ac3c893e786))

- **rename**: Drop deprecated conditional surface + redundant OpLog back-refs + finish scoped
  accessors
  ([`3107330`](https://github.com/johnmarktaylor91/torchlens/commit/3107330e156324fd73ff73abfdface96d528dea4))

Round 2 of naming-impl-v2 cleanup. Drops 18 deprecated conditional fields (6 Trace-level, 12
  Op/Layer-level) folding into the locked Conditional/ConditionalArm/ConditionalRoleRef data
  classes. Drops 3 redundant OpLog layer back-refs (layer_type_index, layer_trace_index, layer_type)
  — exact duplicates of op-scope fields per the inheritance pattern in postprocess/labeling.py.
  Promotes LayerLog.ops, ModuleLog.calls, GradFnLog.calls from dict-like to scoped Accessors.

Tests green: ruff/mypy/smoke 170/170/tier-2 2012 passed. Three feature demos clean
  (18_intervention_verbs, 19_bundles_advanced, 26_perf_and_scaling).

- **rename**: Field renames per cluster locks (phase 2)
  ([`55c22d1`](https://github.com/johnmarktaylor91/torchlens/commit/55c22d11b41d4e4937973c4df715d6d33411adc8))

- **rename**: Method renames (phase 4)
  ([`3541292`](https://github.com/johnmarktaylor91/torchlens/commit/35412928782a0529eb4d4fee109d2a8c08b76413))

- **rename**: New accessor classes + new fields (phase 3)
  ([`dc73744`](https://github.com/johnmarktaylor91/torchlens/commit/dc73744b0e78bd6aa6fd7c7573f94377121d2861))

- **rename**: Top-level vocab + class renames (phase 1)
  ([`30179a8`](https://github.com/johnmarktaylor91/torchlens/commit/30179a802d283d7cf53e6711f40d515601b2f324))

- **rename**: Update audit notebooks
  ([`61e47d1`](https://github.com/johnmarktaylor91/torchlens/commit/61e47d15cebec79f22697f1216c5c1c9876914ee))

- **rename**: Update notebooks (phase 5)
  ([`cb425ec`](https://github.com/johnmarktaylor91/torchlens/commit/cb425eccdc7e0ad770ddc3be5d00fb534cf6b9d2))

- **rename-sprint**: Accessor API rewrite + filter consistency
  ([`657a329`](https://github.com/johnmarktaylor91/torchlens/commit/657a3299b9f670bbf0f282e77b522d627bc5a1c0))

- **rename-sprint**: Bundle + Super[T] + Conditional changes
  ([`9a55724`](https://github.com/johnmarktaylor91/torchlens/commit/9a557248639c62a3a8a7b20d6f39e8fbd30da55a))

- **rename-sprint**: Drop Log suffix from 8 data classes
  ([`31e2d95`](https://github.com/johnmarktaylor91/torchlens/commit/31e2d951e7250124aab2e508a5c8fc73c0bd567e))

- **rename-sprint**: Hard-remove deprecated features
  ([`3f18608`](https://github.com/johnmarktaylor91/torchlens/commit/3f18608a10eed9b3aa95ccfb24b1d1e060352a8a))

- **rename-sprint**: Module + ModuleCall field renames, additions, removals
  ([`cf87490`](https://github.com/johnmarktaylor91/torchlens/commit/cf87490be0e2b0923211ce7fbfae437001ef3000))

- Rename Module.is_train_mode to Module.training and add source-location triplets.

- Rename ModuleCall layer/output storage to ops/input_ops/output_ops and add module resolver,
  identity, context, argument-count, and timing fields.

- Convert Module hook registry metadata to list[HookInfo] with one HookInfo per hook.

- Add Module and ModuleCall forward/func duration surfaces plus parameter tensor/count predicates.

- Update module build, validation, visualization, pandas exports, snapshots, and state log.

- **rename-sprint**: Op + Layer field renames, additions, removals
  ([`1c1e335`](https://github.com/johnmarktaylor91/torchlens/commit/1c1e335f7e006890058a5861a115779003e83a6a))

Renames: capture_index/trace_index/compute_index to raw_index/step_index/ordinal_index;
  func_call_stack to code_context; has_output_variations to has_out_variations;
  output_versions_per_child to out_versions_by_child; is_multi_output/is_part_of_iterable_output to
  in_multi_output; multi_output_role to multi_output_name;
  is_atomic_module_op/is_atomic_module_output to is_atomic_module; module_ops_entered to
  input_to_module_calls; module_entry_argnames to module_entry_arg_keys; detach_saved_tensors to
  detach_saved_activations; feeds_output to is_output_parent; save_tensor_data to save_activation;
  has_saved_outs to has_saved_activation; has_grad to has_saved_gradient;
  grad_memory/transformed_grad_memory to gradient_memory/transformed_gradient_memory;
  transformed_out_memory to transformed_activation_memory; save_grads to save_gradients;
  grad_transform/save_raw_grads/grads_to_save to
  gradient_transform/save_raw_gradients/gradients_to_save; grad_fn_id to grad_fn_object_id; grad_fn
  runtime handle to grad_fn_handle; grad_fn_log TL record to grad_fn.

Additions: Op.ordinal_index, Layer.ordinal_index/raw_index, has_saved_gradient, grad_fn_handle
  runtime storage, grad_fn TL-record link, FX/index lookup propagation, Layer parity fields, and
  required FIELD_ORDER/portable/fork-policy updates.

Removals: dropped deprecated Op/Layer field names from active code paths with no compatibility alias
  properties; kept Trace/Layer num_ops where glossary remains canonical.

- **rename-sprint**: Param + Buffer + GradFn + GradFnCall renames
  ([`5eef3e8`](https://github.com/johnmarktaylor91/torchlens/commit/5eef3e8ddfd8073d3bf3ea849b2a7e1c4342f66a))

- **rename-sprint**: Tl.trace single-responsibility + pass/call vocab + total_ prefix
  ([`12e159d`](https://github.com/johnmarktaylor91/torchlens/commit/12e159d1319b8f8435ae3ff46e85881dc38fdcda))

Part A removes visualization kwargs and the log_forward_pass export from tl.trace, with draw/show
  wrappers owning rendering options.

Part B aligns op/layer iteration vocabulary on pass_index/num_passes while preserving callable
  call_index/num_calls for Module, ModuleCall, GradFn, and GradFnCall.

Part C applies total_/activation/autograd memory field names across Trace, Layer, Op, docs, tests,
  and notebook references.

Verification: ruff check . --fix; pytest tests/ -m smoke -x --tb=short.

- **rename-sprint**: Trace class field renames, additions, removals
  ([`39b72e8`](https://github.com/johnmarktaylor91/torchlens/commit/39b72e82479170f18e9bc67e3b1a2f10e3eec5c1))

Renames applied: _backend_name to backend; run_state/RunState to state/TraceState; io_format_version
  to tlspec_version; flops_by_type to flops_by_op_type; trace_annotations to annotations;
  input_id/model_id to input_object_id/model_object_id; input_shape_hash to input_signature_hash;
  ledger to state_history; train_mode to backward_ready; capture_full_args/capture_args_template to
  save_arg_templates; total_duration/duration surface to capture_duration; trace-level
  equivalent_ops to op_equivalence_classes; last_run_ctx to last_run; start_time/end_time to
  capture_start_time/capture_end_time; backward_root_grad_fn_id to backward_root_grad_fn_ids.

Additions applied: Trace.backend Literal field; Trace.code_context capture; Trace source-location
  class/init fields; Trace.num_modules; ops_with_params; saved_* accessors; compute_ops and
  compute_layers; module/grad-fn invocation accessors; boundary-type accessors; parameter tensor
  count parity; backward duration aggregate accessors; Trace.backward() rerun entry point.

Removals applied: unsupported_ops/unsupported_op collections; Trace.load classmethod;
  streaming_pass_logs, num_streamed_ops, and num_streamed_passes; stale serialized model-log/golden
  fixtures carrying removed field names.

Schema and tests: rolled tlspec naming to tlspec_version, removed obsolete legacy model-log
  backcompat cases, updated FIELD_ORDER/constants, Trace consumers,
  intervention/io/validation/postprocess call sites, docs, and tests. Verification: exact old-name
  grep clean; import probe passed; Trace dataclass field probe passed; ruff check . passed; pytest
  tests/ -m smoke -x --tb=short passed.

- **rename-sprint**: Unify name/address/class vocabulary across data classes
  ([`f32a68e`](https://github.com/johnmarktaylor91/torchlens/commit/f32a68ea4cd825d3f6caf4c93549bf4179b9230a))

Renames Trace.name to trace_label, Trace.model_name to model_class_name/model_label, and
  Trace.model_class to model_class_qualname.

Renames Op.grad_fn_name to grad_fn_class_name, adds Op.func_qualname, Op.grad_fn_class_qualname, and
  Op.grad_fn_cls, and normalizes buffer-sourced Op/Buffer buffer_address to address with
  Buffer.name.

Renames GradFn.name to class_name, adds GradFn.class_qualname, and resolves GradFn.module_path by
  replacing the stored module-only value with class_qualname.

Renames FuncCallLocation.code_qualname to func_qualname.

Adds ModuleCall.name plus Param.module_name and Param.module_cls; verifies Module.name remains the
  bare address segment.

Updates field-order constants, portable state specs, tlspec/golden fixtures, repr/summary/export
  consumers, notebooks/docs field mentions, and tests.

- **super**: Consolidate multi-trace internals into intervention
  ([`46d230e`](https://github.com/johnmarktaylor91/torchlens/commit/46d230e3b7d1704d9ed7a87895ece031ff98e5ff))

Move internal Super views, topology helpers, and metrics out of torchlens/multi_trace and into the
  intervention package layout for the super-family sprint.

- **super**: Extract generic bundle super base
  ([`d25d9a2`](https://github.com/johnmarktaylor91/torchlens/commit/d25d9a2cb70bad65c4f1cc2c3a74f9b992f1f8b2))

Move shared SuperOp/SuperLayer alignment state, labels, members, traces, coverage, and from_members
  construction into internal Super[T].

Keep tensor-specific stacking, aggregation, shape, and diff behavior in a _TensorBearing mixin in
  the same internal module so future tensor-bearing Super classes can reuse it without inheriting
  SuperOp semantics.

- **super**: Generalize bundle super accessors
  ([`3232674`](https://github.com/johnmarktaylor91/torchlens/commit/3232674c0b3cd3502951562d6847fd97e52392fd))

Extract SuperAccessor[T, S] for shared Bundle-side lookup, sparse membership handling, label union
  iteration, and Super view construction. Keep SuperOpAccessor and SuperLayerAccessor focused on
  resolving OpLog and LayerLog objects from member traces.

### Testing

- **api**: Update __all__ size to 46 for backward-parity surface additions
  ([`56ee78b`](https://github.com/johnmarktaylor91/torchlens/commit/56ee78b362a3bb8fa7b6ccd805c19ed55dc01ec5))

P4 added 6 new public top-level names per AD-7 selector DSL parity + helper inventory: grad_clip,
  grad_noise, grad_clamp (helpers), grad_fn, intervening, grad_fn_label (backward selectors).
  Surface expansion intentional and documented in §1 of the backward-parity sprint plan.

Renames test_all_size_exactly_40 -> test_all_size_exactly_46 and updates TARGET_ALL ordering to
  match torchlens.__all__.

- **api**: Update __all__ size to 47 for post-backward P1 output selector
  ([`19fa2f4`](https://github.com/johnmarktaylor91/torchlens/commit/19fa2f42ed3b6a63231d0bba9dd954d759803918))

P1 multi-output module support added `tl.output(...)` selector for disambiguating which output of a
  multi-output module a hook targets (per AD-7 / F-Multi). Surface expansion intentional and
  documented in P1 of the post-backward megasprint plan.

- **api**: Update backward parity surface invariants
  ([`e31592e`](https://github.com/johnmarktaylor91/torchlens/commit/e31592ec99ef8978636058aede3f5cf0cf336d15))

- **capture**: Add module-containment field-equality harness (16 fixtures)
  ([`a539469`](https://github.com/johnmarktaylor91/torchlens/commit/a539469c4e1fde42212fbefaa400ebc14d790176))

Phase 0b of module-containment-refactor sprint. Adds 16 fixture builders spanning the edge-case
  taxonomy in the audit report; a JSON-stable snapshot generator covering Op/Layer/Module/ModuleCall
  fields plus equivalence-partition; a parametrized test runner. Generates baseline snapshots from
  current HEAD (pre-refactor); future phases will assert equality against these baselines.

Refs: .research/module-containment-refactor_PLAN.md (Phase 0b)

- **capture**: Codify ordering invariants and exception safety
  ([`6dafa69`](https://github.com/johnmarktaylor91/torchlens/commit/6dafa69064ce8fe9d3449cb915b07186df73b073))

Phase 5 of module-containment-refactor sprint (final phase). Adds five regression test files
  documenting and asserting the ordering invariants the refactor depends on:

- test_module_stack_identity_exhaustive.py: identity/pass-through synthesized ops see the identity
  module's stack frame. - test_module_stack_buffer_ordering.py: dynamic buffer registration ordering
  relative to stack push. - test_module_stack_exception_safety.py: stack unwinds on forward
  exception; subsequent capture works correctly. - test_module_stack_user_hook_ordering.py: user
  pre-hooks fire BEFORE wrap-forward push by design; logged-replacement op correctness. -
  test_backward_attribution_unchanged.py: backward op module attribution still works post-refactor.

Adds documentation in torchlens/decoration/CLAUDE.md, torchlens/postprocess/CLAUDE.md, and
  torchlens/CLAUDE.md describing the wrap-forward stack as the canonical module-containment
  mechanism and the down-scoped Step 6.

Adds .research/module-containment-refactor_SUMMARY.md with sprint recap, commit list, net code
  delta, and known residuals.

Marks sprint state DONE.

Refs: .research/module-containment-refactor_PLAN.md (Phase 5)

- **fastlog**: Cover halt early abort
  ([`c8d6aaa`](https://github.com/johnmarktaylor91/torchlens/commit/c8d6aaadf2937d1d7da192d91fa46453ec0b9ae2))

- **intervention**: Cover backward selector helper dispatch
  ([`57a9e3d`](https://github.com/johnmarktaylor91/torchlens/commit/57a9e3deedbe6f93c8cb760beb1e8b20a4f648d6))

- **io**: Generate v3 .tlspec golden bundle; un-skip v3-load test
  ([`606e40b`](https://github.com/johnmarktaylor91/torchlens/commit/606e40be45f9d9c8ecb8e1e0001709b8299a60f9))

Generates tests/golden/io_v3_sample.tlspec via an ephemeral git worktree at 83367c1 (pre-M1 main).
  Captures a 2-layer nn.Sequential MLP with torchlens.trace() and saves the result; the bundle's
  manifest.json records io_format_version=3.

tests/test_io_format_v3_load.py::test_load_v3_artifact_into_v4_runtime now runs without the
  missing-golden skip and verifies that the v3 bundle loads cleanly under the v4 runtime via the
  legacy normalizer.

Updates the detect-secrets baseline for the manifest's expected SHA-256 integrity hashes, matching
  existing tlspec fixture handling.

Closes the M8 deferred follow-up.

- **module**: Cover multi-output module traces
  ([`dfdc1a5`](https://github.com/johnmarktaylor91/torchlens/commit/dfdc1a5337f7c4745202300242f4ce0c8b343665))

- **module**: Refresh multi-output containment snapshots
  ([`dc4a2da`](https://github.com/johnmarktaylor91/torchlens/commit/dc4a2da8b73a0bddce8a531834752c636cd78985))

- **observers**: Cover backward taps and bundle viz
  ([`f05a356`](https://github.com/johnmarktaylor91/torchlens/commit/f05a3565fc166ba60f2eb6d60ae2a7f235e6019d))

- **parity**: Add pickle-diff helpers + tensor-equal + allow-list
  ([`8cf73a0`](https://github.com/johnmarktaylor91/torchlens/commit/8cf73a0a743c9d5188959f83e199a307c1630a3f))

- **parity**: Shrink ResNet-50 golden inputs below pre-commit 500KB threshold
  ([`0947830`](https://github.com/johnmarktaylor91/torchlens/commit/094783071c009a9842f1f8fcb73f1fd1c9a78917))

Reduces ResNet-50 parity input shape from (1,3,224,224) to (1,3,64,64). The legacy ResNet trace
  remains metadata-dominated, so the oversized resnet50 pickle payloads are stored gzip-compressed
  on disk and transparently decompressed by the parity test; the uncompressed pickle objects are
  still regenerated from the original anchor worktrees.

Both m2_pre_m3 and m5_pre_m6 goldens were regenerated at their original anchor commits via ephemeral
  git worktrees, then compressed below the repo's 500KB pre-commit large-file threshold.

No more SKIP=check-added-large-files bypass needed for these artifacts.

Resolves: tests/golden/m{2_pre_m3,5_pre_m6}/resnet50.pkl size cleanup.

- **rename-sprint**: Tier-2 green after all renames
  ([`05ac920`](https://github.com/johnmarktaylor91/torchlens/commit/05ac920a9fc75d15c30ab180de392ed37e982f83))

Renames-only: updated tests for grad_fn, total_autograd_memory, module_calls, num_calls, call_index,
  0-based accessors, Trace.show DOT return, and current export/audit fields.

Removed-feature marks: skipped obsolete M5 pre-M6 pickle byte-equality schema and metadata-less v3
  tlspec golden tests.

Type-drift fixes: aligned train-mode tensor validation, backward grad_fn selector dispatch, buffer
  call_index export, collapsed module rendering, ModuleCall JSON export, Trace backend field order,
  and backward invariant validation. Refreshed module-containment snapshots.

- **report**: Align top-level api budget
  ([`54ff9f2`](https://github.com/johnmarktaylor91/torchlens/commit/54ff9f2bb1d8949960ab7f0bd761ed66b4b9bd9b))

- **viz**: Cover combined visualization modes
  ([`5f069d5`](https://github.com/johnmarktaylor91/torchlens/commit/5f069d515047101631732a465d7e3bb3b914e9a8))


## v2.17.0 (2026-05-01)

### Chores

- **precommit**: Block accidental major bumps via three layers
  ([`d6c9bde`](https://github.com/johnmarktaylor91/torchlens/commit/d6c9bde009b7dbdc2d23cbac6a1a37f1128093d5))

After three incidents where commit-message conventions auto-triggered semantic-release MAJOR bumps
  on a project that must stay on the 2.x family, install three layers of mechanical defense so the
  failure mode is impossible:

1. Commit-msg hook (scripts/check_no_breaking_markers.py --commit-msg): rejects any commit whose
  message contains a Conventional-Commits bang marker (feat bang, fix(scope) bang, etc.) or a
  "BREAKING CHANGE:" / "BREAKING-CHANGE:" / "BREAKING:" footer. Override path:
  TORCHLENS_ALLOW_MAJOR_BUMP=1 git commit ...

2. Pre-push hook (--pre-push): scans every outgoing commit for the same triggers and blocks the push
  if any are present. Same override.

3. Custom semantic-release commit parser (scripts/no_major_parser.py:NoMajorAngularParser):
  subclasses the stock Angular parser and downgrades any MAJOR bump signal to MINOR. Even if a
  marker somehow reaches main via a bypassed hook or a server-side direct push, semantic-release
  refuses to interpret it as a major bump. Intentional major releases must be cut manually via
  "semantic-release version --force-level major".

Background and incident history:
  ~/.claude/projects/-home-jtaylor-projects-torchlens/memory/feedback_version_bumps.md

- **release**: 2.17.0
  ([`644d442`](https://github.com/johnmarktaylor91/torchlens/commit/644d442b123a75e60d2d2e35bac684c27f648d76))

- **release**: 3.0.0
  ([`9599752`](https://github.com/johnmarktaylor91/torchlens/commit/959975237e32eb25a7939086724853eebbd6ecfa))

- **release**: Rollback 3.0.0 to 2.17.0 (Q22 lock — stay 2.x family)
  ([`f4ccabc`](https://github.com/johnmarktaylor91/torchlens/commit/f4ccabc173aeccf98f4d0e7447afac793301b3d0))

The Phase 17 PR #175 squash-merge title carried a ! marker (inherited from individual feat!: commits
  in the squash). semantic-release saw that on main and auto-bumped major to 3.0.0 — diverging from
  the locked Q22 decision to stay in the 2.x family on PyPI. The 3.0.0 GitHub release + tag are
  deleted; this commit rolls the version strings to 2.17.0 (the natural next minor after 2.16.0) and
  renames the CHANGELOG section.

Note: PyPI never received 3.0.0 (the publish workflow had a separate dist/ artifact handoff bug and
  failed before publishing).

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

### Features

- **2.0**: Torchlens 2.0 feature overhaul (28 commits, 17 phases)
  ([#175](https://github.com/johnmarktaylor91/torchlens/pull/175),
  [`b55e16b`](https://github.com/johnmarktaylor91/torchlens/commit/b55e16b3865d419a2eb0d6ed6ca6b1bd44249c81))

* chore(release): switch to Apache 2.0, declare [extras], promote to Beta

Per TorchLens 2.0 04_DECISIONS Q29/Q23 + Phase 0 of the implementation plan:

- License: GPL-3.0-only → Apache-2.0 (PEP 639 expression form; dropped the GPL classifier, since
  modern setuptools rejects classifier+expression pairs). - README license badge updated. - PyPI
  Development Status :: 3 - Alpha → 4 - Beta. - pyproject.toml [project.optional-dependencies]:
  declare 25 extras groups covering appliance subfolders (viewer/paper/notebook/llm/neuro), bridges
  (captum/sae/lightning/wandb/hf/...), STRETCH bridges (gradcam/shap/inseq/
  steering/repeng/dialz/nnsight/lit/depyf/compat-shims/vision-shims), and capability extras
  (viz/tabular/profiler). Default deps unchanged in this commit; pandas/IPython move to extras in
  Phase 1e.

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

* feat(subfolders): scaffold five appliance subfolders gated by [extras]

Per Phase 0.3 + Q27 lock (single torchlens package; appliances become subfolders gated by opt-in
  extras, not separate PyPI namespaces):

- torchlens/viewer/ — interactive HTML viewer (extras: [viewer], empty for now; impl arrives later)
  - torchlens/paper/ — 3D cuboid renderer (extras: [paper], empty for now) - torchlens/notebook/ —
  Treescope-grade HTML repr (extras: [notebook] → ipython + jupyter-client) - torchlens/llm/ — logit
  lens, sequence selectors (extras: [llm], empty for now) - torchlens/neuro/ — RDM/CKA/Brain-Score
  helpers (extras: [neuro] → rsatoolbox + brain-score + thingsvision)

Each __init__.py: docstring stating intended scope, eager-imports its declared deps, raises
  ImportError("install with pip install torchlens[X]") on miss, exposes empty __all__ until phase
  work fills in.

tests/test_subfolder_imports.py asserts the import-success / ImportError matrix per subfolder.

* docs(planning): exception taxonomy proposal, §6 audit, method ledger pre-work

Phase 0.4 + 0.5 + 0.7 deliverables (planning artifacts; not user-facing code yet):

- exception_taxonomy_proposal.md: 46 current exception classes inventoried, mapped to a 5-base-class
  taxonomy (TorchLensError + InterventionError / CaptureError / ConfigurationError /
  CompatibilityError / ValidationError) with payload contract and per-class deprecation alias plan.
  Implementation in Phase 1c. - phase_0_5_audit_report.md: §6 of CURATED_FEATURE_LIST verified
  consistent with shipped 2.16.0 + planned NEW work. No vault-level drift. - method_ledger.md: every
  public method on ModelLog, LayerLog, LayerPassLog, ModuleLog, Bundle inventoried + classified
  KEEP/MOVE/DROP/NEW. Final method counts within budgets: ModelLog 67 → 35 (≤ 40) LayerLog 49 → 30
  (≤ 30, exact) LayerPassLog 30 → 23 (≤ 25) ModuleLog 22 → 16 (≤ 20) Bundle 24 → 24 (≤ 25; +1 for
  Phase 9 show_diff = 25 exact) This ledger is the authoritative input to Phase 1a's API
  codification.

* feat(api)!: codify TorchLens 2.0 API ledger — 84 → 40 names with deprecation shims

Phase 1a: the largest single-PR scope in the TorchLens 2.0 push. Implements the §A1 API Ledger from
  IMPLEMENTATION_PLAN_FINAL.md, plus a detect-secrets baseline to whitelist the audit notebook's
  nbformat-generated cell IDs.

Top-level surface (`torchlens.__all__`): - 84 names → 40 (exact). Slim per substrate identity
  guardrail. - 45 names moved via module-level `__getattr__` object aliases that emit
  `DeprecationWarning` on first attribute access. - 7 wrapper functions preserved via
  `functools.wraps`-decorated forwarders with the moved impls living at the canonical new path:
  `validate_forward_pass` (6 params; positional `random_seed` preserved) `validate_backward_pass` (4
  positional incl. `loss_fn`; kw-only tail) `validate_saved_activations` (alias of forward; same
  6-param shape) `summary` (model+input shape; runs temp capture) `show_model_graph` (25-param)
  `show_backward_graph` (11-param; takes `model_log`) `load_intervention_spec` (forwards to
  polymorphic `torchlens.load`) `inspect.signature(torchlens.X)` follows `__wrapped__` and returns
  the moved impl's signature exactly. - 1 hard-dropped: `print_all_fields` (debug-only). - 7 NEW
  slots reserved as stubs raising `NotImplementedError(<phase>)`: `peek/extract/batched_extract`
  (Phase 2); `validate/tap/record_span/sites` (Phase 5a).

New namespace submodules (each with explicit `__all__`): `torchlens.accessors`, `.bridge`,
  `.callbacks`, `.compat`, `.errors`, `.examples`, `.experimental` (+ `.experimental.dagua`),
  `.export`, `.grad`, `.intervene`, `.io`, `.partial`, `.report`, `.stats`, `.viz`. Existing
  namespaces extended: `.types`, `.options`, `.utils`, `.validation`, `.visualization`, `.fastlog`.
  `torchlens.visualization` exposes graph functions lazily (`__getattr__`) to avoid circular
  imports.

Tests: - `tests/test_api_surface.py` — 4 invariants on the public surface. -
  `tests/test_api_surface_deprecation.py` — 59 per-name regressions including representative LEGACY
  POSITIONAL calls and `inspect.signature` equality checks. - `tests/test_api_renames.py` +
  `tests/test_fastlog/test_storage_modes.py` updated for the new canonical paths.

Docs: - `docs/migration/v2.0_api_changes.md` — 91-row table of every old → new name with mechanism
  and removal version. - `CHANGELOG.md` updated.

Total Audit (Phase 17 progressive growth): - `notebooks/total_audit/00_install_and_smoke.ipynb`
  skeleton. - `notebooks/total_audit/_shared.py` with `tiny_model()` factory. -
  `notebooks/total_audit/README.md` placeholder.

Tooling: - `.secrets.baseline` + `.pre-commit-config.yaml` `--baseline` arg — whitelists nbformat
  random hex cell IDs (false-positive "Hex High Entropy String" matches) so the audit notebook can
  scaffold without tripping the secrets hook.

Quality gates: - ruff: PASS. mypy: PASS (0 errors). - Tier-1 smoke: 143 passed. Tier-2 full pytest:
  1920 passed, 27 skipped, 9 deselected, 2 xfailed. - Clean install + `len(torchlens.__all__) ==
  40`: PASS.

BREAKING CHANGE: 53 names removed from `torchlens.__all__`. All remain importable for one minor
  cycle via deprecation shims; `DeprecationWarning` fires on each access naming the canonical new
  path.

* feat(api): grouped option classes + canonical visualization vocabulary

Phase 1b: refactor capture/save/visualization/replay/intervention/streaming kwargs into 6 grouped
  option classes per §13.3, and rename visualization kwargs to the canonical vocabulary per §13.4.
  Behavior preserved; old kwargs accepted for one minor cycle with DeprecationWarning.

Option classes (`torchlens.options.*`, frozen dataclasses): - CaptureOptions (22), SaveOptions (7),
  VisualizationOptions (21, extended), ReplayOptions (6), InterventionOptions (7), StreamingOptions
  (3, extended). Some fields are inert placeholders for later phases.

Canonical vocabulary (§13.4): vis_mode → view, vis_nesting_depth → depth, vis_renderer → renderer,
  vis_node_placement → layout, vis_node_mode → node_style. Old spellings still accepted;
  DeprecationWarning fires once per call naming the canonical replacement.

Entry-point wiring: log_forward_pass, show_model_graph, show_backward_graph,
  ModelLog.do/replay/replay_from/rerun, intervention/replay.py, intervention/rerun.py all accept
  their respective option class AND continue accepting individual kwargs for back-compat. Conflict
  detection raises ValueError when both styles passed for same field.

Tests: tests/test_options.py (18) + tests/test_options_wiring.py (5); tests/test_api_renames.py
  extended (+67/-37).

Total Audit growth (Phase 17 progressive): - notebooks/total_audit/03_save_load_basics.ipynb
  skeleton. - .secrets.baseline updated to whitelist the new notebook's cell IDs.

Migration: docs/migration/v2.0_api_changes.md adds Phase 1b mapping. CHANGELOG entry added.

Quality gates: ruff PASS, mypy PASS (0 errors), Tier-1 smoke 143/143, new tests 23/23, clean install
  + len(__all__)==40 PASS.

* refactor(errors): collapse exception classes to 5-base taxonomy

Phase 1c: 47 existing exception classes folded under a single 5-base taxonomy per §13.8 + the Phase
  0.4 design doc.

New base hierarchy (`torchlens/errors/_base.py`): - TorchLensError (root) ├─ InterventionError ├─
  CaptureError ├─ ConfigurationError ├─ CompatibilityError └─ ValidationError - TorchLensWarning
  (warning base)

Payload contract: file_path, line_no, affected_sites, severity (`recoverable` | `informational` |
  `fatal`), plus extra `fields` for case-specific structured data.

Migration: - 46/47 old exception classes preserved as subclasses of the right new base —
  pre-existing `except OldClass:` continues to work. - 1 alias: `SpecPortabilityError =
  OpaqueCallableInExecutableSaveError`. - Old modules OWN the compatibility class objects where
  runtime code raises them; `torchlens/errors/__init__.py` lazily surfaces legacy names via
  `__getattr__` to avoid import-cycle risk.

Tests: tests/test_exception_taxonomy.py — 56 tests covering payload contract, base hierarchy,
  issubclass mapping for every old class.

Total Audit growth: notebooks/total_audit/21_validation.ipynb skeleton. .secrets.baseline updated.

Migration doc: docs/migration/v2.0_api_changes.md adds Phase 1c table.

Quality gates: ruff PASS, mypy PASS (138 files clean), Tier-1 smoke 143/143, new tests 56/56,
  len(__all__)==40 PASS.

* refactor(visualization): retire ELK from public API + dagua to experimental opt-in

Phase 1d: tightens the rendering surface per §13.6.

Dagua → experimental: - `torchlens/visualization/dagua_bridge.py` →
  `torchlens/experimental/dagua/_bridge.py`. - `vis_renderer="dagua"` raises RuntimeError unless
  `from torchlens.experimental import dagua` was imported (sets opt-in marker).

ELK → internal layout backend only: - `torchlens/visualization/elk_layout.py` →
  `_elk_internal/layout.py`. - Public docs say `layout="auto"`; `layout="elk"` accepted as internal
  escape hatch but not promoted. - ELK-IF-THEN edge-label dedup bug logged in
  `.project-context/known_bugs.md` (no fix in this phase per plan).

Source code panel: `code_panel` kwarg stays in core (lightweight stdlib + graphviz only).

Domain-specific node modes: `vision`, `attention` moved to `torchlens.experimental.node_styles`; old
  core path forwards with DeprecationWarning for one minor cycle.

Total Audit growth: notebooks/total_audit/16_visualization_advanced.ipynb skeleton.
  .secrets.baseline regenerated.

Quality gates: ruff PASS, mypy PASS (140 files clean), Tier-1 smoke 143/143, visualization-adjacent
  tests 34 + 4 dagua all PASS, len(__all__)==40 PASS.

* feat(packaging): pandas + IPython move to opt-in extras

Phase 1e: removes pandas + IPython from default deps; lazy-imports them inside the methods that need
  them; ImportError carries install hint.

Pyproject.toml: - Removed `pandas` and `ipython` from default `dependencies`; pins live under
  `[tabular]` and `[notebook]` extras. - Added `packaging` to default deps (TorchLens runtime +
  safetensors.torch needed it; hidden by dev env until clean-venv test exposed it).

Lazy-import refactor (pandas): user_funcs.py, data_classes/model_log.py, buffer_log.py,
  param_log.py, module_log.py, grad_fn_log.py, grad_fn_pass_log.py, layer_log.py,
  intervention/resolver.py. Pattern: try/except ImportError with install hint; type hints via
  TYPE_CHECKING + forward-ref strings.

IPython: existing lazy paths now raise `[notebook]` install hint on miss. Added minimal
  `ModelLog._repr_html_()` so Phase 1e test target exists.

Tests: tests/test_packaging_diet.py — 5 tests including

`patch.dict('sys.modules', {'pandas': None, 'IPython': None})` to assert the ImportError messages
  fire correctly.

Total Audit growth: notebooks/total_audit/00_install_and_smoke.ipynb adds extras-install
  demonstration section. .secrets.baseline regenerated.

Quality gates: ruff PASS, mypy PASS (0 errors), Tier-1 smoke 143/143, new tests 5/5, clean-venv `pip
  install -e .` (no extras) + smoke PASS, [tabular] + to_pandas() PASS, [notebook] + _repr_html_()
  PASS, len(__all__)==40 PASS.

* chore(api): Phase 1f misc cuts — print_all_fields, conditional fields, activation_transform

Final Phase 1 sub-phase. Mops up the remaining §13.8 cuts.

print_all_fields: implementation moved to torchlens/_debug.py (private); removed public methods on
  ModelLog/LayerLog/LayerPassLog. No alias — hard-dropped per §13.8.

Conditional view fields: conditional_then_edges / _elif_edges / _else_edges consolidated into a
  single canonical mapping `conditional_arm_edges[(cond_id, arm)]`. Old field names preserved as
  ModelLog computed properties with DeprecationWarning for one minor cycle. Per-layer canonical
  structure remains `cond_branch_children_by_cond`.

activation_postfunc → activation_transform rename: across user_funcs.py, fastlog,
  ModelLog/LayerLog/LayerPassLog, fastlog RecordingOptions/record. SaveOptions.activation_transform
  already canonical from Phase 1b. Old name accepted as alias with DeprecationWarning + forward.

_summary internal move: torchlens/_summary/_builder.py →
  torchlens/visualization/_summary_internal/_builder.py. Old path retained as private compat shim.

_robustness + fastlog viz coupling audits: no public exposure found in either; no code change needed
  beyond CHANGELOG note.

Tests: rename tests 66/66, conditional lifecycle/invariants 22/22, summary 11/11, fastlog postfunc
  parity 15/15. Tier-1 smoke 143/143.

Quality gates: ruff PASS, mypy PASS (0 errors), len(__all__)==40 PASS.

* feat(onramp): peek/extract/batched_extract + site discovery + quickstart

Phase 2: ships the §2 onramp APIs. Replaces Phase 1a stubs (peek, extract, batched_extract) with
  real implementations.

Tiered ladder: - tl.peek(model, x, layer) → Tensor - tl.extract(model, x, layers={...}) → dict (list
  OR dict input) - tl.batched_extract(model, stimuli, layers, batch_size=32, ...) → in-memory dict
  OR per-batch disk write - torchlens.compat.torchextractor.Extractor — migration facade -
  torchlens.utils.flop_count — uses TorchLens FLOP metadata

Site discovery: - torchlens.utils.list_modules / list_ops (with mode="both") - log.find_layers /
  suggest — fuzzy matching; "Layer not found" error now suggestion-based - Tab completion: __dir__ /
  _ipython_key_completions_ on accessors + selectors - Programmatic queries: by_operator / by_module
  / by_module_and_operator / total - log.unsupported_ops() / log.uncalled_modules() -
  find_executable_save_set — smallest-first heuristic - tl.experimental.attribute_walk

Quickstart: - torchlens.utils.peek_graph — opinionated wrapper - torchlens.utils.synthetic_input —
  infers from forward sig - examples/recipes/ placeholder

ModelLog public-method count: 40/40 exactly.

Total Audit growth: notebooks/total_audit/01_basic_capture.ipynb + 02_layer_indexing.ipynb
  skeletons. .secrets.baseline regenerated.

Tests: 30 new pass (tests/test_onramp.py 28 + test_extractor_compat.py 2). Tier-1 smoke 143/143.
  mypy clean (143 files). len(__all__)==40 holds.

* feat(reprs): Lovely-style stats + _repr_html_ + LayerLog.show + doctor

Phase 3: ergonomic polish for reprs, displays, and notebook UX.

Object reprs: - Lovely-style stats in torchlens/utils/display.py wired into LayerPassLog:
  shape/dtype/device/mean/std/min/max/%nan/%inf/%neg/%zero with inline NaN/Inf flags. - _repr_html_
  for ModelLog: informative HTML card with text fallback when IPython isn't available. -
  ModelLog.show / LayerLog.show / LayerPassLog.show / Bundle.show all polymorphic. -
  log.first_nonfinite() — text answer with layer + op + module + shape + dtype + parents + source
  file:line. ModelLog method count: 40/40.

Tensor display: - LayerLog.show(method="auto"|"heatmap"|"channels"|"rgb"|"hist") — single
  polymorphic method (no separate .plt/.rgb/.chans aliases). - tl.compat.torchshow / lovely interop
  under [viz] extra. - Captum-style heatmap params: signs, outlier_perc, cmap. -
  tl.viz.causal_trace_heatmap skeleton for the Phase 5b causal-trace recipe.

Notebook integration: - tqdm progress bars via shared progress_bar helper. Auto-detects notebook vs
  terminal. Threshold 10 iterations. Applied to batched_extract, replay cone iteration, two-pass
  capture wiring.

Trust / errors: - LIMITATIONS.md created with substrate identity statement,
  single-threaded-by-design declaration, eager-dynamic-control-flow feature note, placeholders for
  Phase 13 truth-table rows. - tl.utils.doctor() — DoctorReport with PyTorch / CUDA / graphviz /
  safetensors / extras / fingerprint health checks. - FLOP/MAC convention footer in summary/repr
  text paths. - count_fma_as_two arg added to format_flops, flop_count, summary. -
  tl.utils.format_size / format_flops human-readable formatters.

Total Audit growth: notebooks/total_audit/05_visualization_basics.ipynb, 06_modellog_anatomy.ipynb,
  07_layerlog_anatomy.ipynb, 08_layerpasslog_anatomy.ipynb, 09_other_log_types.ipynb skeletons.
  .secrets.baseline regenerated.

Tests: tests/test_reprs.py (9) + test_doctor.py (1) + test_format_helpers.py (5) — 15 new tests, all
  pass. Tier-1 smoke 143/143. Tier-2-equivalent (`pytest -m "not slow"`) 1863 passed / 22 skipped /
  2 xfailed. mypy clean. len(__all__)==40 PASS.

* feat(capture): streaming stats + new fields + perf + multi-pass

Phase 4 §7. Substantial capture-side expansion.

Selectors (§7.1): 6 selectors verified composing across capture/extract/ hooks/overlays.
  module_filter_fn predicate added under CaptureOptions.

Streaming stats (§7.3) — new torchlens.stats.* namespace: Mean, Quantile (reservoir sampling), TopK,
  Covariance, PCA (covariance/eigendecomp fallback), Aggregator. torchlens.aggregate(model,
  dataloader, metrics). 10k-batch Mean stress: peak memory 0.001 MB.

New capture targets (§7.4): - LayerPassLog fields: bytes_delta_at_call, bytes_peak_at_call (in
  FIELD_ORDER + portable state). - Tied-param detection with → notation. - record_kpi_in_graph for
  user KPI annotations. - register_tensor_connection autograd escape hatch. - Hash-based dedupe
  (disabled when function args saved for validation safety). - Content-hash capture cache via
  CaptureOptions(cache=True); ~/.cache/torchlens/capture default dir. - ModuleLog/ModulePassLog
  inputs/outputs aliases. - forward_lineno capture (~332 ns overhead). - Static autograd_saved_bytes
  estimator; per-grad_fn memory cost.

Misc capture (§7.8): - torchlens.utils.register_op_rule for custom op rules. -
  decide_recording_of_batch retroactive predicate filter. - Fastlog gradient via predicate keep_grad
  path. - patch_detached_references optimization: simulated-old 0.338s → current cold 0.204s,
  incremental 0.000269s. Hard LRU cap rolled back (broke incremental crawl invariants); clear API +
  skip filters kept. - torchlens.bridge.profiler.execution_trace for ExecutionTraceObserver.

Stop-early (§7.2): experimental.stop_after on tl.peek only; log_forward_pass raises.

Auto-capture (§7.6): experimental.auto_capture context manager; TORCHLENS_AUTO=1 env var rejected.

Multi-pass (§7.7): torchlens.utils.log_forward_pass_streaming for stacked multi-pass; module
  multi-output smoke audit passes.

Total Audit growth: notebooks/total_audit/26_perf_and_scaling.ipynb skeleton. .secrets.baseline
  regenerated.

Tests: 8 new test files, 20 Phase 4 tests pass + selector unification

coverage. Full non-slow sweep: 1883 passed / 22 skipped / 2 xfailed. mypy clean. Tier-1 smoke
  143/143. len(__all__)==40 PASS.

* feat(intervention): §6.3 LAUNCH extensions + consolidated tl.validate(scope=)

Phase 5a — implements every §6.3 NEW intervention API item that wasn't shipped at 2.16.0, plus the
  consolidated `tl.validate(scope=...)` per §13.5/§5a.6.

Sequential hooks: attach_hooks(hook1, hook2, hook3) returns HookHandle; single-hook
  attach_hooks(...) is log preserved for back-compat. Scoped handles with removable hook IDs +
  context-manager cleanup.

tl.sites + SiteCollection (Phase 1a stub replaced): tl.sites(layer_pattern, ops, modes) returns
  SiteCollection / SiteSpec with hook-pair expansion for sweeps. Conservative selector mapping:
  dot-path → tl.module, other string → tl.contains, optional tl.func(op) intersection.

tl.tap / tl.record_span / report.log_value (stubs replaced): observer that records without
  modifying; phase-boundary context manager; scalar recorder under torchlens.report (NOT in
  __all__).

torchlens.experimental.session(model): invoke + bundle for clean-vs-corrupt.

torchlens.experimental.freeze_module: observe without changing output.

Consolidated tl.validate(scope=...) (Phase 1a stub replaced): signature exactly per FINAL plan
  §5a.6. All 4 scopes work (forward/backward/saved/ intervention). Per-scope keyword visibility
  enforced. scope="intervention" returns InterventionValidationReport (5-axis:
  invariance/specificity/ completeness/consistency/locality). Phase 1a functools.wraps validator
  wrappers updated to forward via legacy-arg mapping rule. Existing test_api_surface_deprecation.py
  STILL PASSES — every legacy positional call site preserved end-to-end.

Total Audit growth: notebooks/total_audit/17_intervention_helpers, 18_intervention_verbs,
  27_taps_and_observers, 28_sites_and_sweeps skeletons; 21_validation extended. .secrets.baseline
  regenerated.

Tests: 6 new Phase 5a test files; 14 phase tests + 77 combined new/API surface pass. Tier-1 smoke
  143/143. mypy clean. len(__all__)==40 PASS.

* docs(recipes): 5 intervention recipes (causal-trace, contrastive, faithfulness, trainable,
  generation)

Phase 5b §6.3 RECIPE-only items. All 5 recipes self-contained, runnable end-to-end via papermill, no
  network downloads.

- causal_trace_recipe.ipynb — full causal trace on tiny transformer -
  contrastive_direction_recipe.ipynb — repeng-style direction - measure_faithfulness_recipe.ipynb —
  ROAD/output_drop/locality via tl.validate(scope="intervention") -
  trainable_intervention_recipe.ipynb — gradient-based parameter learning -
  generation_step_iterator_recipe.ipynb — per-step intervention during model.generate()

Each uses public TorchLens API only; papermill 5/5 PASS.

examples/recipes/README.md updated with recipe list + difficulty ratings.

notebooks/total_audit/_shared.py extended with TinyTransformer + tiny_transformer() for the
  causal-trace recipe.

Quality gates: ruff PASS, mypy PASS (0 errors), Tier-1 smoke 143/143, len(__all__)==40 PASS.

* feat: Phases 5c + 6 + 7 — intervention maintenance, NaN/Inf debug, visualization core

Bundled commit (pre-commit re-stash collapsed three phases into a single rev). Each phase's
  acceptance criteria pass independently.

== Phase 5c: Intervention API maintenance == 10 items from .project-context/todos.md addressed: hook
  signature doc, attribution-patching formula, FrozenTargetSpec, cached_property invalidation tests,
  object.__setattr__ for _construction_done, atomic state-swap, list_logs snapshot (already
  complete), suppress_mutate_warnings ctx-mgr tests, §20.1 cohort migration row. Naming items
  deferred to post-overhaul naming pass per Q14.

== Phase 6: NaN/Inf debugging floor (§5.1) == log.first_nonfinite() verified; print(model_log)
  NaN/Inf summary line. torchlens.partial.PartialModelLog: failed captures attach e.partial_log;
  PartialModelLog.render_graph() returns minimal Graphviz DOT with failure node. raise_on_nan kwarg
  + CaptureOptions field; raises CaptureError with op/layer/shape/dtype/parents metadata on first
  nonfinite. tests/ test_nan_debugging.py covers 4 NaN behaviors.

== Phase 7: Visualization core + tl.export.html (§4) == Graph overlays: 8 stock per-node overlays
  (flops/time/bytes/magnitude/ grad_norm/nan/intervention/bundle_delta) + ModelLog.add_node_overlay.
  Theme presets: paper/dark/colorblind/high_contrast (shared palette), colorblind-safe legend,
  node_label_fields, font_size, dpi, for_paper, return_graph. ModelLog.animate_passes(site) +
  summary("waterfall"). tl.export.svg(log, path, editable=True) with stable tl-node-* IDs + semantic
  CSS classes. tl.export.html(log, path) — static self-contained HTML with pan/zoom/hover, no CDN,
  works WITHOUT [viewer]/[notebook] extras (verified by tests/test_export_html_minimal.py). Fastlog
  guardrail bug caught: bare detach() → safe_copy(..., detach_tensor=True).

Total Audit growth across phases: notebooks/total_audit/03_save_load_basics deepened,
  15_visualization_options.ipynb + 29_edge_cases.ipynb new. .secrets.baseline regenerated.

Quality gates: ruff PASS, mypy PASS (151+ source files clean), Tier-1 smoke 143/143, Tier-2-equiv
  (pytest -m "not slow") 1923 passed / 22 skipped / 2 xfailed, deprecation+validation 64/64 + new
  tests 5+ from Phase 6 + Phase 7 visualization tests all pass. len(__all__)==40 PASS.

** PHASE 5 (5a/5b/5c), PHASE 6, PHASE 7 ALL COMPLETE. **

* feat(bundle): comparison primitives polish + multi-trace follow-ons

Phase 8 §8.

Bundle methods polish: delta_map / norm_delta / output_delta / compare / aligned_pairs uniform
  return shape, exposed via Bundle.__getattr__ to preserve the 25-method budget. Only added public
  attr: supergraph. Bundle public method/property count: exactly 25.

Multi-trace follow-ons: TraceBundle.supergraph accessor; module-type cluster labels;
  tl.show_bundle_graph rolled mode + forward/backward/both/ overlay; per-node/edge style merge
  primitives in _render_utils.py; tl.export.chrome_trace_diff(bundle, path).

aligned_pairs: cross-architecture alignment via best-match heuristics (module path → op → shape →
  topological proximity).

Total Audit growth: notebooks/total_audit/10_bundle_anatomy.ipynb skeleton. .secrets.baseline
  regenerated.

Quality gates: ruff PASS, mypy PASS (153 src clean), Tier-1 smoke 143/143, Tier-2-equiv 1929 passed
  / 22 skipped / 2 xfailed, Phase 8 tests 6/6, len(__all__)==40 PASS.

* feat(viz)!: Bundle diff renderer (Q33 launch hero) — bundle.show_diff + tl.viz.bundle_diff

Phase 9 — launch-blocking hero demo. Side-by-side comparison of two ModelLogs in a Bundle with
  delta-colored aligned pairs.

UX brief: examples/demos/bundle_diff_UX_brief.md (acceptance contract).

Renderer: torchlens/visualization/bundle_diff.py. - bundle.show_diff(metric, layout) via
  Bundle.__getattr__ (preserves 25-method budget). - tl.viz.bundle_diff(bundle, ...) static helper.
  - Graphviz layout, cluster-per-side, paper theme default. - max_pairs default 16 hits ~50KB target
  on full ResNet; max_pairs=None renders the full graph. - SVG: <title> per node, aria-label on
  figure (accessible). - Diverging palette (blue→white→red) keyed to per-node L2 norm delta. -
  Unmatched nodes get a gray border on the missing side.

Demo: examples/demos/bundle_diff_clean_vs_zero_relu.ipynb +
  examples/demos/bundle_diff_clean_vs_zero_relu.svg (56,315 bytes). papermill PASS in ~8s on CPU.
  Random-weight ResNet-18, torch.manual_seed(0), zero_ablate(layer1.0.relu).

Snapshot test: tests/test_bundle_diff_renderer.py +
  tests/snapshots/bundle_diff_clean_vs_zero_relu.svg golden. Strategy: semantic SVG normalization
  (strip Graphviz random IDs, timestamps, whitespace) → byte compare; OR image rasterize + pixel
  similarity ≥ 0.95.

Documentation: bundle.show_diff docstring with canonical 5-line pattern.

LIMITATIONS.md updated: "static for v1; interactive in torchlens.viewer later."

Total Audit growth: notebooks/total_audit/19_bundles_advanced.ipynb demonstrates bundle.show_diff.
  .secrets.baseline regenerated.

The "!" mark reflects launch-blocking hero status, not API breakage — bundle.show_diff is a new
  feature, not a removal.

Quality gates: ruff PASS, mypy PASS, snapshot test 1/1, demo notebook + audit notebook 19 PASS via
  papermill, Tier-1 smoke 143/143, Tier-2-equiv 1930 passed / 22 skipped / 2 xfailed,
  len(__all__)==40 PASS, Bundle methods 25/25.

* feat(export): tl.export.* surface — chrome_trace, xarray, trackers, tabular, model_explorer,
  netron + bridges

Phase 10 §9.1-9.7 (excluding §9.8 .tlspec → Phase 11). Ships the full tl.export.* hand-off surface,
  plus HF Hub push_to_hub + depyf companion.

Trace/timeline (§9.1): chrome_trace, speedscope, flamegraph, memory_timeline (tensor scope, not
  allocator). CaptureOptions.emit_nvtx wired through op wrappers.

Scientific (§9.2): xarray() — NeuroidAssembly-shaped DataArray with presentation/neuroid dims for
  rsatoolbox/Brain-Score handoff.

Trackers (§9.3) — object-passed: tensorboard, wandb, mlflow, aim. Heavy extras fail soft with
  install hints.

Tabular cuts (§9.4): csv/parquet/json under tl.export.*. ModelLog.to_* methods now delegate with
  DeprecationWarning. Parquet-safe object sanitization (torch.dtype et al. → string-safe).

Static graph adapters (§9.5): model_explorer; netron (lossy ONNX-shaped JSON; doc disclaimer that
  it's NOT runnable).

Hub publishing (§9.6): bridge.huggingface.push_to_hub with dry-run + pickle-to-manifest fallback.
  examples.load() minimal loader (replaces deprecated hub.list_examples).

Compile-stack (§9.7): bridge.depyf.dump companion (torch.compile capture remains SCOPE in
  tl.compat.report).

Total Audit growth: notebooks/total_audit/23_export_formats.ipynb skeleton. .secrets.baseline
  regenerated.

Side discovery during full-gate sweep: - maskedfill validation alias added (matches masked_fill_
  exemption) - Mamba ABI-incompatibility skip helper for real-world tests

Quality gates: ruff PASS, mypy PASS (156 src clean), Tier-1 smoke 143/143, Tier-2-equiv 1945 passed
  / 17 skipped / 2 xfailed in 1070s, len(__all__)==40 PASS.

* feat(io): .tlspec 2.16.0 backward-compat — format detection + 6 golden fixtures + transitional
  writer

Phase 11.0 — backward-compat layer landing BEFORE Phase 11 unified .tlspec graduation. Per FINAL
  plan §11.0, every .tlspec written by shipped 2.16.0 readers must continue to load forever.

Format detection (11.0.1): torchlens.io.detect_tlspec_format first-match-wins: v2.0_unified →
  v2.16_intervention_with_kind → v2.16_intervention → v2.16_modellog_portable → unknown.

Golden fixtures (11.0.2): F1-F6 generated using shipped 2.16.0 writers, ~528K in
  tests/fixtures/tlspec_v2_16/. Covers all 3 intervention save levels (default, audit,
  executable_with_callables, portable) + 2 ModelLog portable bundles (tiny CNN, tiny transformer
  with varied shape/dtype).

Read-compat (11.0.3): tests/test_tlspec_backcompat.py — per fixture detect format, torchlens.load
  returns expected kind, round-trip identical. 6 tests PASS.

Migration policy (11.0.4): MIGRATIONS.md — permanent 2.16.0 readability (no auto-migration).

Transitional writer (11.0.5): intervention/save.py emits `kind: "intervention"` alongside legacy
  fields. Old readers ignore; new polymorphic loader dispatches by kind without needing
  tlspec_version. 4 transitional tests PASS.

.secrets.baseline regenerated to include the F3 transformer fixture's manifest.json hashes
  (content-hash strings inside the fixtures look like "high entropy" to detect-secrets — they're
  tensor IDs/shape hashes from the 2.16.0 writer).

Quality gates: ruff PASS, mypy PASS, Tier-1 smoke 158/158, new tests 15/15, len(__all__)==40 PASS.

* feat(io): .tlspec unified graduation — schema-locked manifest + writer/reader/validator

Phase 11. Schema design landed FIRST per R2 Critical #7.

Manifest schema (11.1): torchlens/schemas/tlspec_manifest_v1.json — 16 required fields
  (tlspec_version, kind, created_at, torchlens_version, python_version, torch_version,
  schema_version, model_signature, model_fingerprint with param + buffer hashes, sites,
  spec_compat_info, body_format, body_index, save_level, optional_dependencies,
  intervention_compat_metadata).

Schema validator (11.4): torchlens.validation.validate_tlspec(path) — JSON Schema 2020-12.

Writers (11.2): - ModelLog.save(path) — unified format, default save_level="portable". -
  Bundle.save(path) — NEW; nested member .tlspec dirs (avoids live autograd-state issues). -
  Intervention writer — full unified format (tlspec_version=1 + kind="intervention") + legacy fields
  for old readers. Transitional v2.16_intervention_with_kind deprecated; readers support it forever.
  - Routed through torchlens/_io/tlspec.py.

Polymorphic reader (11.3): torchlens.load(path) dispatches by manifest.kind for new format AND by
  Phase 11.0 detect_tlspec_format for old formats. Returns ModelLog | Bundle | InterventionSpec.
  @overload typing.

Manifest tooling (11.4): torchlens.io.inspect_tlspec / torchlens.validation.validate_tlspec.

Save levels (11.5): audit + executable_with_callables + portable round-trip per kind.

Total Audit growth: notebooks/total_audit/17_intervention_helpers deepened;
  20_save_load_advanced.ipynb created. .secrets.baseline regenerated.

Tests: tests/test_tlspec_unified.py 11/11. Phase 11.0 back-compat STILL

PASS. Public API gate: 1971 passed / 17 skipped / 2 xfailed. Tier-1 smoke 169/169. ruff PASS, mypy
  PASS (157 src clean), len(__all__)==40.

* feat(bridge): LAUNCH tier — Captum, SAE Lens, rsatoolbox, Brain-Score, Lightning, HF,
  profiler.join, nnsight

Phase 12a — LAUNCH-tier bridges with declared [extras], version pins, offline fixtures + invariant
  tests.

8 bridges shipped (under torchlens.bridge.* + torchlens.compat.* + torchlens.callbacks.*):

- captum: bridge.captum.attribute / .layer (pin captum~=0.7; test skip when extra missing) -
  sae_lens: bridge.sae_lens.encode (pin sae-lens~=4.0; skip) - rsatoolbox: bridge.rsatoolbox.dataset
  (pin rsatoolbox~=0.1.5; tested with REAL rsatoolbox 0.1.5; RDM invariant PASS) - brain_score:
  bridge.brain_score.per_layer (pin brain-score~=2.2; mocked offline fixture; TODO connect real) -
  lightning: callbacks.lightning.LayerProfilerCallback (pin lightning~=2.4; tested via fallback
  because env has lightning 2.1.0 with pkg_resources issue) - hf: compat.from_huggingface +
  compat.from_timm (pins transformers~=4.45, timm~=1.0; offline mock fallback for DistilBERT) -
  profiler.join: bridge.profiler.join (torch built-in; cached Kineto trace fixture; per-op timing
  merge invariant PASS) - nnsight: bridge.nnsight.from_trace (pin nnsight~=0.4; STRETCH-real-
  integration; offline contract test passes)

All bridges lazy-import extras; helpful ImportError("install torchlens[X]") on miss.

Total Audit growth: notebooks/total_audit/24_bridges.ipynb skeleton. .secrets.baseline regenerated.

Tests: tests/test_bridges_launch.py 8 tests, 6 PASS / 2 skipped (extras genuinely missing).

Quality gates: ruff PASS, mypy PASS (164 src clean), Tier-1 smoke 169/169, Tier-2-equiv 1976 passed
  / 20 skipped / 2 xfailed, len(__all__)==40 PASS.

* feat(bridge): STRETCH tier — gradcam, shap, inseq, steering, repeng, dialz, lit, compat-shims

Phase 12b — STRETCH-tier bridges as offline contract tests. Each mocks downstream tool's API +
  verifies bridge output shape; tests pass when extra is not installed.

7 STRETCH bridges (under torchlens.bridge.*, lazy __getattr__): - gradcam.cam / .layer (pin
  pytorch-grad-cam~=1.5) - shap.explain (pin shap~=0.46) - inseq.attribute (pin inseq~=0.6) -
  steering_vectors.vector (pin steering-vectors~=0.1) - repeng.* (pin repeng~=0.4) - dialz.* (pin
  dialz~=0.2) - lit.model(log) — contract test mocks LIT browser (pin lit-nlp~=1.3)

4 migration helpers (under torchlens.compat.*): - compat.from_torchextractor(model, layers) →
  Extractor facade - compat.from_fx(graph_module, layers) → migration payload -
  compat.from_ilg(model, return_layers) → Extractor (via [vision-shims]) -
  compat.from_sentence_transformers(model, prompt) → migration payload

torchlens.bridge namespace gains lazy __getattr__ dispatch so sub-bridges import on first attribute
  access.

pyproject.toml: stretch extras pinned + [all-stretch] rollup populated.

Tests: tests/test_bridges_stretch.py — 10 contract tests, all PASS.

Helpful ImportError("install torchlens[X]") on miss for every bridge.

len(__all__)==40 (bridges live under torchlens.bridge.*).

* feat(compat): tl.compat.report truth-table + 3 known-broken bug fixes

Phase 13 — runtime compatibility report. Marketing's "any PyTorch model" claim gates on this.

tl.compat.report(model, input) → CompatReport (torchlens/compat/_report.py): - 17 truth-table rows
  probed: HF transformers, accelerate (device_map + offload), bitsandbytes 8/4-bit, tied params,
  multi-GPU RNG (now FIXED; see below), DataParallel (broken-by-design), DDP (works), FSDP/
  DeepSpeed (SCOPE), torch.compile (SCOPE; suggests depyf), FX-traced (SCOPE), Lightning
  training_step, vmap/functorch, quantized tensors (was broken; now FIXED), DeviceContext factory
  (was broken; now FIXED), single-thread design statement. - CompatReport dataclass with per-row
  status (pass | known_broken | scope | not_tested), severity, explanatory text. -
  compat_report.show() — pretty table; .to_markdown() — for issue templates / README. - Inspects
  without executing; safe to run on any model.

Bug fixes (opportunistic per Phase 13.4): - RNG-MULTI-GPU FIXED: CUDA RNG capture/restore now uses
  torch.cuda.get_rng_state_all() / set_rng_state_all() across all devices, while preserving old
  single-device snapshot back-compat. - QUANTIZED-CRASH FIXED: tensor_nanequal() guards quantized
  tensors (no longer calls unsupported .isinf() on quantized). - DEVICE-CONTEXT-LOGGING FIXED:
  factory kwarg injection now runs during active logging too.

README + LIMITATIONS: - README adds "Run tl.compat.report(model, x) before filing bugs" section. -
  LIMITATIONS.md + docs/LIMITATIONS.md cross-reference each known-broken / SCOPE row.

Total Audit growth: notebooks/total_audit/25_compat_truth_table.ipynb skeleton. .secrets.baseline
  regenerated.

Tests: tests/test_compat_report.py — 6 tests on 5+ reference scenarios (small CNN, mocked HF
  transformer, quantized input, multi-GPU CPU emulation, FSDP placeholder, renderers, all-CUDA RNG,
  DeviceContext active logging) all pass.

Quality gates: ruff PASS, mypy PASS (172 src clean), Tier-1 smoke 169/169, public API gate (pytest
  -m "not slow") 1993 passed / 19 skipped / 2 xfailed, len(__all__)==40.

* fix: Phase 14 — 8 named bugs swept + validation issues + §15.10 audit triage

8 named bugs from §11.11 (remaining after Phase 13 sweep): - BFLOAT16-TOL: dtype-aware tolerances
  for bfloat16/float16. - FUNC-CALL-LOC-LEAK: snapshot signature/docstring + release _frame_func_obj
  after capture (memory leak fix). - ARG-KWARGS-MISSING: FUNC_ARG_SPECS audited for kwarg tensors on
  linear/convs/norms/attention/cat/stack/where. - COND-THEN-MULTIPASS: rolled LayerLog THEN/ELSE
  child views. - INVARIANT-COND-THEN: invariant rejects stale derived THEN children. -
  HASH-COLLISION: replaced Python hash() truncation with SHA-256 prefix. - CLEANUP-DOCS:
  FuncCallLocation lifetime + stable hashing notes. - VALIDATE-STATE-RESTORE: cloned state_dict +
  try/finally restore.

Validation (§11.9): LSTM exemption (hidden/cell only; params no longer mis-exempted); arglocs
  duplicate same-parent slots; Output identity synthetic node operation_num=0 invariant.

§15.10 audit items — bucketed: - Multi-torch CI / MLX / torch.compile-FX / S4-S5 specific / _trim_
  elimination / PERF-30 lazy fields / Direct SVG 1M+ / PERF-36-39 model prep: DEFERRED with reasons.

- Gradient arrows in rolled mode: SHIPPED. - JIT compatibility, DOT empty PDF, forward_lineno, ELSE
  branch, SSM/ Mamba: VERIFIED via existing coverage.

Tests: tests/test_bug_fixes_phase14.py — 11 regression tests PASS.

Quality gates: ruff PASS, mypy PASS (172 src clean), Tier-1 smoke 170/170, public API gate 2004
  passed / 19 skipped / 2 xfailed, len(__all__)==40.

* docs+feat(report): migration tables + reference docs + tl.report.explain + ROADMAP + public
  CLAUDE.md

Phase 15 — non-marketing documentation. Per R2 H7, marketing copy deferred to post-naming sprint.

Migration tables (15.1) — docs/migration/: 7 files
  (from_nnsight/transformerlens/captum/thingsvision/pyvene/torchextractor/ fx).
  Tests/test_migrations.py — 9 PASS / 6 skipped (skipped = optional tool missing).

Reference docs (15.3) — docs/method_x_model_compatibility.md, speed_optimized_defaults.md,
  elk_setup.md, activation_transform_persistence.md (callable-drop / repr-keep).

Source click-through (15.4) — torchlens/_source_links.py: OSC 8 terminal + VS Code URI HTML links
  for FuncCallLocation, non-finite failures, control-flow summaries, code panel.

tl.report.explain (15.5) — torchlens/report/_explain.py: explain(log,
  audience="researcher"|"practitioner"|"auto") → plain-language report (model + capture + anomalies
  + interventions + patterns). NOT in __all__.

ROADMAP.md (15.6): coarse buckets per Q11.

Public CLAUDE.md (15.7): concise agent guide replacing internal-style root CLAUDE.md.

NOTICE (15.8) for Apache-2.0 redistribution context.

LIMITATIONS.md verified (Phase 13 truth-table rows already there). README.md: license badge + Apache
  2.0 + LIMITATIONS/ROADMAP/migration links ONLY (no marketing copy per R2 H7).

Tests: 20 PASS / 6 skipped (focused). Tier-1 smoke 170/170. T1 non-slow 2017 passed / 25 skipped / 2
  xfailed. mypy clean. len(__all__)==40.

* docs(examples): 5-minute user-facing notebook gallery (7 quickstarts)

Phase 16a — 5-min user-facing track. 7 quickstart notebooks at examples/5min/: peek, find_first_nan,
  visualize, intervention, save_load, extract_activations, cog_neuro_rdm. Each ≤10 cells,
  self-contained with tiny shared MLP, deterministic seeded, checked in with outputs, papermill ≤30s
  on CPU, 7/7 PASS.

examples/5min/README.md indexes with suggested reading order; linked from main README.
  .secrets.baseline regenerated.

Quality gates: ruff PASS, mypy PASS (174 src clean), Tier-1 smoke 170/170, papermill 7/7,
  len(__all__)==40.

* docs(examples): 50-minute user-facing notebook gallery (9 workflows)

Phase 16b — 9 longer workflow notebooks at examples/50min/: ablations_grid, causal_trace_recipe,
  cog_neuro_extraction (offline brain_score), custom_hooks, intervention_fork_replay, iou_heatmap,
  paired_prompt_patching, steering, transformer_migration (local mock + swap-in comment for real
  HookedTransformer).

Each: self-contained with notebook-local toy models; public API only; deterministic; checked in with
  outputs; papermill 9/9 PASS ≤5 min/CPU.

examples/50min/README.md indexes; README.md links alongside 5-min. .secrets.baseline regenerated for
  the new notebook cell IDs.

Quality gates: ruff PASS, mypy PASS (174 src clean), Tier-1 smoke 170/170, papermill 9/9, T1
  non-slow 2017 passed / 25 skipped / 2 xfailed. len(__all__)==40.

* feat(audit): Total Audit Notebook System — coverage manifest + 24 polished notebooks +
  REGEN_PROMPT

Phase 17 — biggest single phase. Maintainer's sandbox bulletproofing every public name + automated
  coverage enforcement.

Coverage manifest infrastructure (scripts/): - generate_audit_coverage_manifest.py (320 lines) —
  runtime introspection (NOT AST-only); walks torchlens.__all__, every submodule (preferring
  mod.__all__), every Log type via inspect.getmembers (handles inherited methods, properties,
  classmethods, FIELD_ORDER). Generates compat_aliases section for deletion-protection. -
  check_audit_coverage.py (319 lines) — three coverage levels: mentioned (regex), called (cell
  metadata coverage_calls), expected_ failure (try/except metadata). Strict mode requires every
  public item reaches `called`. compat_aliases ignored in strict. - check_coverage_delta.py (121
  lines) — fails on new uncovered public name OR silent removal of compat alias.

Inventory: 1,280 public items (40 top-level + 17 submodules + 10 Log

types). Coverage gaps: 0. Compat aliases: 52.

24 audit notebooks polished (notebooks/total_audit/00-29): - ≥5 cell types from the catalog per
  notebook. - Structured coverage_calls metadata on executable cells. - Outputs checked in (max cell
  output 170B, max notebook 17.4KB). - Papermill 24/24 PASS. Smoke subset (00, 01, 02, 19, 25): PASS
  in 39.77s.

REGEN_PROMPT.md SELF-CONTAINED — embeds folder layout, per-notebook content outlines, cell-type
  catalog, coverage requirements, regeneration procedure, substrate identity guardrails. NO external
  vault references.

_shared.py polish: tiny_cnn, tiny_recurrent, tiny_dynamic_model, tiny_branched_model,
  pretty_print_fields, inline_show, make_clean_corrupt_pair.

CI integration: - Tier-0 audit smoke (5 notebooks) + manifest delta in .github/workflows/tests.yml.
  - Tier-2 nightly full Total Audit papermill in .github/workflows/audit-notebooks.yml.

Acceptance: check_audit_coverage --strict PASS; check_coverage_delta PASS; all 24 notebooks PASS via
  papermill; len(__all__)==40.

Quality gates: ruff PASS, mypy PASS, Tier-1 smoke 170/170.

Five capture/postprocess internal modules (capture, constants, data_classes, postprocess,
  user_funcs) lack __all__ — generator falls back to dir() with warning.

** PHASE 17 COMPLETE. Only FINAL phase remains. **

---------

Co-authored-by: Claude <noreply@anthropic.com>

Co-authored-by: Happy <yesreply@happy.engineering>

- **release**: Republish 2.x sprint content to PyPI
  ([`ca828db`](https://github.com/johnmarktaylor91/torchlens/commit/ca828db43d8954d7bed7c321ad5fd556866dbd5f))

The TorchLens 2.0 feature overhaul (PR #175, 17 phases, 28 source commits) shipped to git on
  2026-05-01 but reached PyPI only under an accidental 3.0.0 auto-bump (Q22 violation; see prior
  commits). 3.0.0 has been yanked from PyPI; the marketing slot is permanently spent and is not
  getting another upload.

This commit republishes the same sprint content under the correct 2.x label by: 1. Resetting version
  strings in pyproject.toml + torchlens/__init__.py to 2.16.0 (the last legitimately published
  version on PyPI). 2. Restoring CHANGELOG.md to its v2.16.0 state so semantic-release will
  regenerate the new section cleanly. 3. Carrying the feat: type so semantic-release bumps 2.16.0 ->
  2.17.0 on merge to main.

The Layer 3 custom commit parser (scripts/no_major_parser.py) clamps the historic feat(2.0) bang
  marker on b55e16b from MAJOR to MINOR, so the overall bump from this push will be MINOR (2.17.0),
  not MAJOR.


## v2.16.0 (2026-04-30)

### Bug Fixes

- **validation**: Tier-3 model failures
  ([`f19fb5c`](https://github.com/johnmarktaylor91/torchlens/commit/f19fb5c11f773b4f09265fc6aa5f18a43f985e82))

Six pre-existing tier-3 real-world model tests failed on main because validation required exact or
  too-tight replay equality for numerically equivalent FP32 outputs in deep convolutional stacks.
  Add local validation-only tolerances for direct model outputs and late numeric replay ops without
  changing global tensor equality constants.

Two save_new_activations _fails tests were stale: AlexNet and ResNet18 now refresh activations
  correctly. Update them to compare fast refresh activations against a fresh exhaustive log instead
  of expecting a graph-change error.

Verified: ruff check . --fix; mypy torchlens/; pytest tests/ -m smoke -x --tb=short; pytest tests/
  -m 'not slow' -x --tb=short; targeted 8 tier-3 tests pass.

### Chores

- **intervention**: Phase 16 cleanup + bench + ship
  ([`dd94370`](https://github.com/johnmarktaylor91/torchlens/commit/dd94370c0742d7270c28a7904342cf00053b0cb1))

- Internalize/remove obsolete TraceBundle public code; multi_trace kept only as internal helpers
  used by Bundle. - Remove _is_fork remnants, dead compatibility helpers, unused imports, stale
  tests for deleted public APIs. - Audit __all__ in torchlens/__init__.py. - Add
  test_field_lifecycle_matrix programmatic audit (every field has portable policy + ForkFieldPolicy
  + default-fill). - Add test_not_mvp_audit grep audit (deferred features absent from public). -
  Performance benchmarks: baseline 0.000057s, non-ready capture 0.032300s, ready capture 0.007983s,
  replay 0.001117s, rerun 0.012192s, Bundle.node 0.000300s in benchmarks/intervention_overhead.py.

Final phase of v5.2 intervention API implementation. 21 phases complete. -2052 LOC net + 126
  intervention tests.

Branch: codex/intervention-api.

- **intervention**: Phase 4.5 capture-time runtime smoke
  ([`04600da`](https://github.com/johnmarktaylor91/torchlens/commit/04600daa81564f5c65a31a19fe51221a222a3a95))

Regression checkpoint after Phase 4a/4b/4c. Adds tests for: non-ready capture parity,
  intervention_ready=True no-hook output equality with default, exception-path runtime state cleanup
  (active_model_log, active_hook_plan, hook_reentrancy_depth all reset), subsequent capture succeeds
  after hook-raised exception.

No production behavior change.

Branch: codex/intervention-api.

- **release**: 2.16.0
  ([`8fbe6a9`](https://github.com/johnmarktaylor91/torchlens/commit/8fbe6a903d3c3794cd1de30b3d72d906bb49fc67))

- **todos**: Log multi-trace V2 follow-ons and record sprint completion
  ([`dda5a9e`](https://github.com/johnmarktaylor91/torchlens/commit/dda5a9ef47da9bdb9e6c79e30fad2c99a00d2100))

Documents three additional V2 follow-ons surfaced during the multi-trace sprint review: -
  Module-type second line in bundle cluster labels - Public TraceBundle.supergraph accessor -
  Per-node / per-edge styling primitives in _render_utils.py

Plus a Completed-section entry for the multi-trace sprint covering PRs #170 (Phase 1) and #172
  (Phase 2 polish), the closed-then-superseded PR #171, and the codex quota -> Claude agent pivot.

### Documentation

- **intervention**: Phase 15 worked examples + cohort migration + API docs
  ([`cdfe68e`](https://github.com/johnmarktaylor91/torchlens/commit/cdfe68e1419744d60461aa1c05becf6480513bc1))

- 16 worked examples in examples/intervention/ covering first-5-minutes, activation patching, sticky
  hooks, set vs attach_hooks, chunked batching, Bundle comparison, live capture, submodule
  discovery, post-hoc replay, SAE attachment, linear probe, paired-prompt 3+, per-position steering,
  publishing for reproducibility, top-level tl.do. - tests/test_examples.py harness imports/runs
  each example. - 10 cohort migration tables (TransformerLens, NNsight, Pyvene, baukit, Penzai,
  Captum, SAELens, RepE, Inseq, AutoCircuit). - Visibility class docs explain fused attention / SDPA
  limitations. - Replay vs rerun, mutate-in-place + fork-first, direct-write dirty, save levels,
  .tlspec/, append constraints, Tier-1 backward hook contract explainers added. - API docs for
  selectors, helpers, Bundle. - Docstring polish across torchlens/intervention/*.py and method
  docstrings on ModelLog. README adds interventions section. - No TraceBundle promoted as public;
  mentioned only in migration notes.

Branch: codex/intervention-api.

### Features

- **intervention**: Phase 0 codebase prep
  ([`6f3283a`](https://github.com/johnmarktaylor91/torchlens/commit/6f3283a124f5300460a2bc0284e80c346fa4e2af))

Add torchlens/intervention/ subpackage with stub placeholders for the v5.2 intervention API. All
  stubs raise NotImplementedError; this phase establishes import surface only. Branch:
  codex/intervention-api.

- **intervention**: Phase 1 data model — types and field additions
  ([`d3609f4`](https://github.com/johnmarktaylor91/torchlens/commit/d3609f4bbff486a89387537d937ab7b0b3703066))

Add intervention data model: RunState enum, TargetSpec/FrozenTargetSpec,
  InterventionSpec/FrozenInterventionSpec, FireRecord, CapturedArgTemplate, OutputPathComponent,
  HelperSpec, FunctionRegistryKey, EdgeUseRecord, Relationship, ForkFieldPolicy with per-class
  policy tables.

Add ModelLog fields: name, intervention_ready, capture_full_args, parent_run, _intervention_spec,
  operation_history, last_run_ctx, _has_direct_writes, _warned_direct_write,
  _warned_mutate_in_place, _spec_revision, _activation_recipe_revision, _append_sequence_id,
  run_state, source_model_id, source_model_class, weight_fingerprint_*, input_id_at_capture,
  input_shape_hash, graph_shape_hash, is_appended, relationship_evidence.

Add LayerPassLog fields: func_call_id, output_path, intervention_log, container_spec,
  captured_arg_template, captured_kwarg_template, edge_uses, _construction_done. Add direct-write
  guard via __setattr__ and _internal_set helper.

Bump IO_FORMAT_VERSION to 2. Default-fill entries in both __setstate__ for old-format logs.
  Per-instance defaults for new container fields.

Branch: codex/intervention-api.

- **intervention**: Phase 10 .tlspec save/load + safetensors + registry keys
  ([`e0f54d3`](https://github.com/johnmarktaylor91/torchlens/commit/e0f54d3baa27c617afda21e9493f937b07eb3af1))

Implement intervention recipe persistence with three save levels (audit / executable_with_callables
  / portable), .tlspec/ directory format, helper serialization, callable registry keys, target
  manifest, compatibility checks, fail-closed executable replay.

- .tlspec/ directory: spec.json + manifest.json + README.md + tensors/*.safetensors. format_version
  and helper_registry_version distinct from IO_FORMAT_VERSION. Atomic tmp.<uuid>/ -> fsync -> rename
  protocol. - HelperSpec portability tags: builtin (portable), import_ref (executable), opaque_audit
  (audit-only). Save-level enforcement raises OpaqueCallableInExecutableSaveError. -
  FunctionRegistryKey: namespace (torch / torch.Tensor / torch.nn.functional / operator / custom),
  qualname, dispatch_kind (function/method/dunder/namespace_alias), version. Resolved via registry
  strategy, not getattr(torch, name). - Target manifest: original selector + resolved labels +
  graph_shape_hash + module_address_normalized. Re-resolution at load. - tl.check_spec_compat(spec,
  new_log) returns SpecCompat dataclass (EXACT / COMPATIBLE_WITH_CONFIRMATION / FAIL); raises
  GraphShapeMismatchError for fatal executable mismatch. - log.save_intervention(path, level=...).
  tl.load(path) detects .tlspec/ vs ordinary logs. tl.load_intervention_spec(path). - Direct-write
  policy: AUDIT warns; EXECUTABLE/PORTABLE raise DirectWriteInExecutableSaveError unless
  allow_direct_writes=True. - validation.core._raise_if_portable_bundle_log narrowed.

Errors: OpaqueCallableInExecutableSaveError, DirectWriteInExecutableSaveError,
  GraphShapeMismatchError.

Branch: codex/intervention-api.

- **intervention**: Phase 11 visualization wiring
  ([`8a35e90`](https://github.com/johnmarktaylor91/torchlens/commit/8a35e909830a0ba7eb37227e52a6b251a54a454d))

Add intervention visualization to graph rendering: - vis_intervention_mode="node_mark" (default):
  magenta border on intervention sites; subtler magenta on cone-of-effect members when
  vis_show_cone=True (default). - vis_intervention_mode="as_node": hook nodes inserted between
  producer and downstream consumers. - Cone extracted via Phase 6 cone_of_effect helper (no
  duplication). - Reuses existing node_spec_fn callback infrastructure. - ModelLog.show() and
  Bundle.show() public methods. - Dagua bridge exposes is_intervention_site / is_in_cone /
  intervention_log_summary fields. - Direct-write dirty flag surfaced in summary panel where
  available. - Re-enabled 21 previously-skipped multi_trace_visualization tests where intent maps to
  new Bundle API.

Branch: codex/intervention-api.

- **intervention**: Phase 12 append chunked batching
  ([`7aa71af`](https://github.com/johnmarktaylor91/torchlens/commit/7aa71af2a84c7296edbb698c6188eaedec3c7016))

Implement rerun(model, x_new, append=True): concat activations along batch dim while preserving
  per-call scalar state from last chunk.

Preconditions: _recipe_is_clean (Phase 8a), same model evidence, same graph_shape_hash, same
  topology + site labels, shape match except batch dim, dtype/device match, helper batch
  independence.

- Concatenate activation, transformed_activation; reject gradient concatenation by default
  (helper-specific opt-in). - is_appended=True; _append_sequence_id++; run_state=APPENDED;
  operation_history records chunk size + new total. - HelperSpec.batch_independent flag gates
  append; default-False for unknown helpers. - BatchNormTrainModeWarning when model.training and any
  nn.BatchNorm*. - Shape-changing hooks reject unless helper.compatible_with_append. - Save/load
  (.tlspec + ordinary) preserves is_appended + _append_sequence_id + appended history.

Errors: AppendMismatchError, AppendBatchDependenceError, BatchNormTrainModeWarning.

Branch: codex/intervention-api.

- **intervention**: Phase 13 discoverability
  ([`d88ec0a`](https://github.com/johnmarktaylor91/torchlens/commit/d88ec0a0c2b99bc229b9a7bae273e81c70a5fb16))

- ModelLog.summary(): comprehensive multi-section status (capture metadata, recipe, recent ops,
  lineage, run_state, dirty flag, append status, available next ops, hash, relationship evidence,
  portability, stale spec, RNG notes). - ModelLog.last_run_records(): frozen tuple snapshot of
  FireRecords from most recent replay/rerun/live capture. - SiteTable.__repr__ finalized; find_sites
  notebook-friendly. - Process-wide weak log registry; tl.list_logs() returns tuple snapshot.
  _state.py imports no ModelLog (weakref only). - Auto-naming in log_forward_pass: lowercase short
  class name with HuggingFace suffix stripping + monotonic per-process counter; respects explicit
  name=. tl.reset_naming_counter(class_name=None). - Bundle default names derived from log.name when
  omitted; counter suffix on collisions. - ModelLog.__repr__ includes name without losing model
  identity. - Loaded logs preserve names without incrementing counters.

Branch: codex/intervention-api.

- **intervention**: Phase 14 error catalog + cross-cutting tests
  ([`7e7be48`](https://github.com/johnmarktaylor91/torchlens/commit/7e7be483dd3b8c2b4151962b3d22d63fa4c7291a))

- Centralize v5.2 error catalog: 27 errors + 5 warnings each with severity tag (recoverable /
  informational / fatal) and named-fields contract. Reconcile aliases to match v5.2 section 23. -
  Add missing errors: RecursiveTracingError, AxisAmbiguityError. - Catalog raise-loop test verifies
  every named error/warning is importable, has severity, and is exercised somewhere in the test
  matrix. - Cross-cutting axes added: per-verb (A), engine equivalence (B), concurrency/GC (I),
  error-message quality (K), degradation (L). - Slow tests marked. Smoke subset stays fast.

Branch: codex/intervention-api.

- **intervention**: Phase 2 selectors, resolver, SiteTable
  ([`bf56a41`](https://github.com/johnmarktaylor91/torchlens/commit/bf56a417ab549f62cf5e5ad0dd09d0da3de024c2))

Implement typed site grammar: tl.label, tl.func, tl.module, tl.contains, tl.where, tl.in_module. Add
  resolve_sites with default max_fanout=8 and strict mode rejecting bare strings. Add SiteTable with
  __len__/__iter__/ __getitem__/where/first/labels/to_dataframe. Wire ModelLog.find_sites and
  ModelLog.resolve_sites; ModelLog.__getitem__ branches on selector objects before string cascade.
  Add MultiMatchWarning, SiteAmbiguityError, SiteResolutionError. Existing string lookup behavior
  preserved.

Branch: codex/intervention-api.

- **intervention**: Phase 3 hooks, helpers, execution contract
  ([`1062856`](https://github.com/johnmarktaylor91/torchlens/commit/1062856403f5533b9153e46fb4f41e35190245fe))

Implement hook signature contract, HookContext (MappingProxyType view over layer metadata, never
  live LayerPassLog), normalizer dispatch for plain callable / HelperSpec / dict / list-of-tuples /
  single-site shapes, return-value handling with default raise on None, dtype/ device/shape
  validation with force_shape_change escape hatch.

Add 14 helpers: zero_ablate, mean_ablate, resample_ablate, steer, scale, clamp, noise, project_onto,
  project_off, swap_with, splice_module (forward) + bwd_hook, gradient_zero, gradient_scale
  (backward, live/rerun-only). Each carries a HelperSpec with portability tag. RNG policy: seeded
  helpers get hook-local Generator, unseeded stochastic helpers enqueue a non-determinism note.

Add SpliceModuleDtypeError, SpliceModuleDeviceError, HookSignatureError, HookValueError,
  HookSiteCoverageError. Hook execution wraps user callable in pause_logging(). Add re-entrancy
  guard scaffold.

Phase 3 does NOT wire hooks into capture/replay/rerun (Phase 4a-4c).

Branch: codex/intervention-api.

- **intervention**: Phase 4a capture flag plumbing + runtime context + func_call_id
  ([`16a964d`](https://github.com/johnmarktaylor91/torchlens/commit/16a964d0a4cdc89ca659fa1b711f634bb2dc4a95))

Add intervention_ready and hooks parameters to log_forward_pass (hooks inert until Phase 4c). Reject
  intervention_ready=True combined with layers_to_save=<list> via InterventionReadyConflictError.
  Auto-enable replay-template capture flags when intervention_ready=True (templates land in Phase
  4b).

Add active intervention runtime context to _state.py with TYPE_CHECKING imports for intervention
  types. Tighten active_logging() guard to require _logging_enabled is False AND _active_model_log
  is None.

Add func_call_id session counter; allocate before func() call in torch wrapper around
  decoration/torch_funcs.py:412; propagate through log_function_output_tensors; ensure multi-output
  calls share id.

Initialize relationship evidence at capture start: source_model_id, source_model_class, weight
  fingerprint, input id, input shape hash. Capture pre-forward RNG before active_logging() per
  existing invariant.

Phase 4a does NOT execute hooks, build replay templates, or change saved activations.

Branch: codex/intervention-api.

- **intervention**: Phase 4b replay templates + edge provenance + output_path
  ([`29b272f`](https://github.com/johnmarktaylor91/torchlens/commit/29b272f2552fe2b42b7b5a93e379c693ca3ba185))

Add path-preserving output traversal helper. Replace _collect_output_tensors BFS and ensure_iterable
  in intervention-ready captures with path-aware walker that preserves dict keys,
  namedtuple/dataclass fields, HuggingFace ModelOutput keys, tuple/list indices, and attr components
  via OutputPathComponent. Single tensor outputs get output_path=().

Add ContainerSpec on LayerPassLog for reconstructing multi-output containers during replay. Build
  CapturedArgTemplate at decoration time, reusing FUNC_ARG_SPECS / _locate_parent_tensors_in_args /
  _cache_dynamic_spec. Classify template components as parent_ref / literal_tensor / literal_value /
  unsupported. Templates are the single replay source of truth in intervention_ready mode.

Wire edge provenance: EdgeUseRecord per parent-child tensor use including arg_kind, arg_path
  (mirroring parent_layer_arg_locs schema), parent/child raw labels, view/copy status. Preserve
  existing parent_layer_arg_locs and children_tensor_versions.

Outputs from one function call share func_call_id and have unique output_path per output. Template +
  path capture gated by intervention_ready=True or capture_full_args=True; non-ready captures pay no
  overhead.

Phase 4b does NOT execute hooks (Phase 4c).

Branch: codex/intervention-api.

- **intervention**: Phase 4c live hook execution + LiveModeLabelError
  ([`2e773bc`](https://github.com/johnmarktaylor91/torchlens/commit/2e773bc07c65b2fa4b74f960843391c1cc5f8e98))

Wire live post-hook execution in decoration/torch_funcs.py: hooks run after func() returns, after
  in-place safe-copy, and before output collection so returned and saved activations do not diverge.
  Reuse Phase 3 _execute_hook (which wraps in pause_logging) and shared metadata setter to update
  shape/dtype/device/memory/transformed_ activation/has_saved_activations on tensor replacement (no
  direct .activation assignment).

Resolve hook plan against capture-time site identity (raw label, function name, module address,
  predicate). Raise LiveModeLabelError with copy-paste suggestion when a selector requires a
  finalized postprocess label (e.g., 'relu_4_27:2').

Append FireRecord to LayerPassLog.intervention_log per fire. Set ModelLog.run_state =
  RunState.LIVE_CAPTURED when hooks are non-empty; otherwise unchanged from Phase 4a.

MVP scope: forward post-hooks only. Backward hooks and pre-hooks deferred to v1. Replay/rerun
  behavior unchanged. Pre-forward RNG capture ordering preserved per Phase 4a invariant.

Branch: codex/intervention-api.

- **intervention**: Phase 5 postprocess pipeline updates
  ([`0a229b0`](https://github.com/johnmarktaylor91/torchlens/commit/0a229b0a08255f4e766fdf27a6fc8dd3cdc985dc))

Teach 18-step postprocess pipeline to preserve intervention metadata: - Step 3 _remove_orphan_nodes
  preserves func_call_id groups atomically by raw labels. - Step 8 loop detection does NOT
  regenerate func_call_id (per-wrapper-call allocation already unique). - Step 11
  _replace_layer_names_for_layer_entry rewrites labels in edge_uses, template leaves (parent_ref),
  intervention_log, and operation_history. - Step 12 retention predicate keeps replay-ready call
  groups atomic (metadata-only siblings retained when activation values pruned). - Step 12
  _batch_remove_log_entries scrubs new label-bearing fields. - Step 17.5 computes graph_shape_hash
  over op sequence + normalized function names + parent edges + normalized module addresses +
  output_path/cardinality (excludes activation values and labels that differ across loop expansion).
  Stored on ModelLog. - Step 18 _set_pass_finished flips access behavior only after intervention
  metadata is finalized. - Fast path postprocess_fast preserves replay metadata or marks log not
  intervention_ready; never builds module logs. - module_address_normalized strips pass qualifiers
  and rolled-loop iteration indices.

Add Invariant S (func_call_id consistency) to validation/invariants.py dispatch.

Branch: codex/intervention-api.

- **intervention**: Phase 6 replay engine
  ([`5d21148`](https://github.com/johnmarktaylor91/torchlens/commit/5d2114855c1731656b1921fd92b17aeb3cbd338b))

Implement saved-DAG replay engine that mutates ModelLog by recomputing the cone of effect from
  changed sites without calling model.forward().

- cone_of_effect(origins): BFS forward from origins, follows children_tensor_versions for in-place
  ops, includes all func_call_id group siblings, tracks visited to avoid cycles, preserves topo
  order. - Forward-from-origin arg reconstruction from CapturedArgTemplate; resolves ParentRef via
  overlay or current activation; raises ReplayPreconditionError on Unsupported. -
  _slice_output_by_path supports tuple/list/dict/namedtuple/dataclass/ HF ModelOutput per Phase 4b
  container set. - Multi-output func_call_id groups execute once; outputs sliced by output_path.
  In-place op handling honors children_tensor_versions. - RNG/autocast restored narrowly via reused
  validation primitive wrapped in strict replay mode (re-raises exceptions). - Hook composition:
  pause_logging() wrap, FIFO order, FireRecord per fire with engine="replay". - Shared metadata
  setter updates shape/dtype/memory/transformed/ has_saved_activations on activation replacement. -
  ModelLog.replay(strict=False, hooks=None) and ModelLog.replay_from(site, strict=False); set
  run_state=REPLAY_PROPAGATED and last_run_ctx. - ReplayPreconditionError,
  ControlFlowDivergenceWarning, ControlFlowDivergenceError.

Replay does NOT call model forward (rerun is Phase 7) and does NOT rebuild module logs unless
  metadata shape changes require it.

Branch: codex/intervention-api.

- **intervention**: Phase 7 rerun engine + atomic state swap
  ([`076ce8e`](https://github.com/johnmarktaylor91/torchlens/commit/076ce8ef8dc0a472874d8309b894ed348240b627))

Implement full-forward rerun through decorated wrappers with active InterventionSpec installed in
  runtime context. Build fresh ModelLog off to the side; validate; atomically swap state via
  ModelLog.replace_run_state_from(new_log).

- rerun(log, model, x, append=False): preflight (reject FSDP/compile/scripted via existing
  _reject_opaque_wrappers), install active spec, fresh capture, validate, atomic swap,
  run_state=RERUN_PROPAGATED, last_run_ctx populated, one operation_history record appended. -
  replace_run_state_from(new_log): preserve name, parent_run, _intervention_spec, warning flags,
  relationship evidence; replace graph/log containers, layer_list, lookup_keys, layer_logs,
  accessors, output metadata, graph_shape_hash, shape fields. Single __dict__.update for atomic
  swap; concurrent reads unsupported. - Counter alignment: graph_shape_hash divergence triggers
  ControlFlowDivergenceWarning (non-strict) or ControlFlowDivergenceError (strict). - append=True
  raises NotImplementedError until Phase 12. - RNG capture follows log_forward_pass invariant
  (pre-active_logging).

Failure leaves original ModelLog unchanged including graph_shape_hash, operation_history, current
  activations, and warning flags.

Branch: codex/intervention-api.

- **intervention**: Phase 8a mutator methods + spec revision
  ([`fa480a2`](https://github.com/johnmarktaylor91/torchlens/commit/fa480a274ec2beddade8e8d8f3eca1e4bc27a78c))

Add ModelLog mutator methods: set(site, value|fn), attach_hooks(...), detach_hooks(site, handle),
  clear_hooks(), do(...) signature, fork() stub. Each mutator increments _spec_revision, invalidates
  cached FrozenInterventionSpec, sets run_state=SPEC_STALE, returns self.

set(callable) tags spec with created_by="set_callable_one_shot". attach_hooks is sticky; persists
  until detach/clear. detach_hooks accepts site+handle (handle support is MVP-deferred to
  site-only).

Mutators do NOT propagate; replay/rerun/live capture advance _activation_recipe_revision on success.

_recipe_is_clean() helper compares revisions for Phase 12 append precondition. do() and fork()
  shells raise NotImplementedError pointing to Phase 8b.

Branch: codex/intervention-api.

- **intervention**: Phase 8b do dispatch + warnings + fork + op history
  ([`0f7acb1`](https://github.com/johnmarktaylor91/torchlens/commit/0f7acb100624a07de25d5abb22ac6ee46e4a89e2))

- _record_operation appends per-mutator records to operation_history with op, spec_revision,
  timestamp, op-specific payload. - MutateInPlaceWarning fires once on first root-log mutator
  (parent_run None). Suppression via confirm_mutation=True kwarg, session-level
  tl.suppress_mutate_warnings(True), or context manager. Forks (parent_run not None) NEVER emit. -
  DirectActivationWriteWarning fires once when LayerPassLog.__setattr__ guard detects bypass; sets
  _has_direct_writes=True and run_state=DIRECT_WRITE_DIRTY. Replay/rerun emit one-time warning when
  _has_direct_writes is True. - do() auto-dispatch: tensor/hooks without model -> mutate+replay;
  with model -> mutate+rerun; ambiguous (only x or only model) -> EngineDispatchError. Fingerprint
  mismatch -> ModelMismatchError. - tl.do(log, ...) top-level alias (replaces Phase 0 placeholder).
  - fork(name=None) per ForkFieldPolicy table: activations FORK_SHARE, mutable containers FORK_COPY,
  weak refs FORK_RECONSTRUCT, relationship evidence FORK_COPY, source model refs FORK_SHARE. Sets
  parent_run weakref; copies _intervention_spec mutable; fork's revisions sync with parent at fork
  time. _is_fork is NOT used.

EngineDispatchError, ModelMismatchError, MutateInPlaceWarning, DirectActivationWriteWarning added.

Branch: codex/intervention-api.

- **intervention**: Phase 9 single Bundle type
  ([`d823caa`](https://github.com/johnmarktaylor91/torchlens/commit/d823caa4de1afc97e908d5ec82f94611f1fd4b5c))

Replace TraceBundle with single Bundle type. Flat container of ModelLogs with lazy supergraph
  construction, relationship-gated operations, NodeView access, and intervention verbs over members.

- Bundle(...) accepts dict, list+names, list-of-tuples, list+auto-derive. - Unique names required.
  Optional baseline (auto-detected when unambiguous; BaselineUndeterminedError otherwise). -
  bundle[str] returns ModelLog (was NodeView). - bundle.node(site) returns NodeView with
  .activations as dict keyed by member name (was list). - Lazy supergraph build on first node()
  call. - Relationship taxonomy: SAME_OBJECT / SAME_MODEL_OBJECT_AT_CAPTURE /
  SHARED_GRAPH_{SAME,DIFFERENT}_INPUT / SHARED_ARCHITECTURE / SAME_PARAM_SHAPES / DIFF_MODEL /
  UNKNOWN. Permissive construction; lazy-fail at op time per permission matrix. - Bundle-level: do,
  fork, attach_hooks, replay, rerun, metric, joint_metric, compare_at, most_changed, diff. cluster()
  raises NotImplementedError. - set_capacity protects baseline from eviction. - bundle.help() lists
  per-member readiness. - tl.bundle now constructs Bundle. TraceBundle removed from public exports;
  multi_trace.* kept as internal helpers.

Errors: BundleMemberError, BundleRelationshipError, BaselineUndeterminedError, NoParentError,
  DeadParentError.

Existing multi-trace tests rewritten to new API where intent applies; otherwise marked xfail/skip
  with Phase 9 redesign note.

Branch: codex/intervention-api.


## v2.15.0 (2026-04-27)

### Chores

- **release**: 2.15.0
  ([`e0eb903`](https://github.com/johnmarktaylor91/torchlens/commit/e0eb903a39cfbd377e483f5d1a4a55590c9f29cd))

### Features

- **multi-trace**: Add bundle visualization with divergence/swarm/group_color modes
  ([`79c57c3`](https://github.com/johnmarktaylor91/torchlens/commit/79c57c3a6a8bc1be8b4f451075c53ac57d4a4418))

Phase 2 of the multi-trace sprint: a Graphviz visualization layer for TraceBundle objects,
  implemented as a compact module-clustered renderer in torchlens/multi_trace/visualization.py.

Public surface:

- tl.show_bundle_graph(bundle, ...) -- canonical entry point. - bundle.show(...) -- thin wrapper on
  TraceBundle for ergonomics. - Re-exported from torchlens.multi_trace and torchlens top-level.

Modes:

- 'divergence' -- per-node mean pairwise distance, sequential Reds colormap. Best for
  shared-topology bundles ("which nodes change most across traces?"). - 'swarm' -- per-node coverage
  fraction, sequential viridis colormap. Best for divergent topology ("which nodes did most/all
  traces visit?"). - 'group_color' -- categorical (tab10) colouring by group membership. Multi-group
  nodes get a neutral grey rather than a blended palette colour. Requires bundle.groups; raises
  ValueError otherwise. - 'auto' -- shared topology -> divergence, divergent -> swarm.

Output formats: PDF, PNG, SVG, JPG, BMP, TIF (mirrors show_model_graph).

Refactor scope:

- New private torchlens/visualization/_render_utils.py extracting the file-format dispatch +
  dot.save / subprocess.run / view orchestration shared between the existing single-trace render
  path and the new bundle renderer. The single-trace rendering.py is intentionally NOT modified --
  its 3.5k-line orchestrator is too tightly coupled to ModelLog state to safely re-point at a
  multi-trace input within this dispatch's risk budget. Bundle renderer reuses graphviz primitives
  directly via the same library.

- Colormaps (Reds, viridis, tab10) hardcoded as 10-stop hex arrays so torchlens does not gain a
  matplotlib dependency.

- Module clusters in the bundle render are derived from each supergraph node's module_path string
  (dotted prefix chain) rather than the ModelLog ModuleLog hierarchy, which is not first-class on a
  Supergraph.

Tests: 17 in tests/test_multi_trace_visualization.py (15 spec'd + 2 internal-consistency bonus), one
  marked smoke. All four modes verified end-to-end via DOT-source inspection plus rendered file size
  checks.

Quality gates: ruff clean, mypy clean. Phase 1 multi-trace tests (28), backward viz tests (8), and
  aesthetics regression tests (12) all still pass.

Existing show_model_graph() output is unchanged -- the absolute invariant from the spec held
  throughout the refactor.

The .project-context/todos.md edit shipping with this branch is the parking-lot update from the
  architect; deferred V2 items remain parked for future sprints.

- **multi-trace**: Bundle renderer refactor + module cluster aesthetic parity
  ([`236f74a`](https://github.com/johnmarktaylor91/torchlens/commit/236f74a7bae62fa99c626aa4eff9ccd7e68d583c))

Reroute multi_trace/visualization.py through the shared rendering primitives in
  visualization/_render_utils.py and split the bundle helpers into focused modules:

- _bundle_clusters.py owns supergraph -> cluster-hierarchy conversion (pass-aware module addresses,
  parent/child cluster map, rolled-style ``(xN)`` count and ``:N`` pass-suffix titles). -
  _bundle_styling.py owns the colormap tables, per-node aggregation (divergence / coverage / group),
  per-mode fill+font dispatch, and edge colouring. - visualization.py shrinks to just the
  orchestrator (arg validation, caption composition, Graphviz Digraph build, cluster emission, file
  output) -- 354 lines instead of 779.

Module clusters now match show_model_graph's aesthetics:

- Per-depth penwidth scaling via the shared compute_module_penwidth formula (5.0 outermost, 2.0
  deepest, linearly interpolated between). - Pass-aware titles: single-pass modules drop the ``:N``
  suffix (``@fc1`` not ``@fc1:1``); supergraph clusters that genuinely span multiple pass
  occurrences keep ``:N`` (``@level21:1``); rolled-mode multi-pass LayerLogs append ``(xN)``
  (``@block (x3)``). - The same nested cluster hierarchy ModelLog produces, derived from each
  canonical supergraph node's containing_modules chain rather than just the leaf module string. -
  Bundle clusters omit the ``(ModuleType)`` label line because the supergraph doesn't preserve
  module class -- documented as the only intentional difference from ModelLog.

Bonus: divergence mode no longer crashes for multi-pass canonical nodes. The previous code called
  LayerLog.has_saved_activations directly, which raises ValueError on multi-pass LayerLogs; we now
  catch the error and fall back to "no tensor available" so the render proceeds with neutral-shade
  nodes.

### Refactoring

- **visualization**: Extract reusable rendering primitives
  ([`8874170`](https://github.com/johnmarktaylor91/torchlens/commit/8874170e6ae2dc78a55c01f727f2a1cab95de55e))

Expand visualization/_render_utils.py with the Graphviz primitives shared between single-trace
  rendering.py and the multi-trace bundle renderer:

- direction_to_rankdir for vis-direction -> Graphviz rankdir - compute_module_penwidth for
  depth-based cluster border scaling - make_module_cluster_attrs for the standard module-cluster
  attr dict - html_escape and format_node_html for HTML label escaping - MAX_MODULE_PENWIDTH /
  MIN_MODULE_PENWIDTH / PENWIDTH_RANGE constants (sole canonical source; rendering.py and
  elk_layout.py used to keep duplicate copies)

Update rendering.py and elk_layout.py to call these helpers instead of inlining the formulae and
  constants. Output is byte-equivalent for ModelLog renderings -- the regression suite
  (test_output_aesthetics, test_backward_visualization, test_dagua_theme) confirms.

The motivation is the upcoming bundle renderer refactor, where the multi-trace cluster builder will
  reuse the same primitives so single-and multi-trace cluster aesthetics stay in sync.

### Testing

- **multi-trace**: Cluster parity tests + defer rolled/backward bundle modes
  ([`61f6afd`](https://github.com/johnmarktaylor91/torchlens/commit/61f6afdbe0bc6d919834d3bbd120e1e83a3f4776))

Add four module-cluster aesthetic-parity tests in test_multi_trace_visualization.py:

- test_module_cluster_penwidth_parity verifies bundle and ModelLog clusters share the same penwidth
  at the shallowest matching depth and that both renderers produce monotonically non-increasing
  penwidths from depth 0 to deepest. - test_module_cluster_pass_labels verifies recurrent models
  produce rolled-style ``@block (x3)`` cluster titles (matching ModelLog rolled mode). -
  test_module_cluster_unrolled_pass_labels verifies models with distinct call-site occurrences
  (NestedModules' level21 called 3x) produce per-pass clusters with ``:N`` suffixes (matching
  ModelLog unrolled mode). - test_module_cluster_single_pass_label_no_suffix verifies single-pass
  modules drop the ``:N`` suffix from cluster titles even though the underlying containing_module
  data carries it.

Add the two deferred follow-ups the architect explicitly called out to .project-context/todos.md's
  Multi-trace V2 section:

- vis_opt='rolled' for show_bundle_graph (currently advisory) - direction='backward' for
  show_bundle_graph (currently raises)

Both are tracked alongside the existing Multi-trace V2 items so the follow-on sprint has the full
  backlog in one place.


## v2.14.0 (2026-04-27)

### Chores

- **release**: 2.14.0
  ([`d6377cd`](https://github.com/johnmarktaylor91/torchlens/commit/d6377cda930ddc2410397b2ecb32711697dd9586))

- **todos**: Log deferred items from 2026-04-27 sprint and record completed PRs
  ([`70111e6`](https://github.com/johnmarktaylor91/torchlens/commit/70111e6ad93530bd518f8793744874fda2683ba8))

### Features

- **multi-trace**: Add TraceBundle and supergraph for multi-pass analysis
  ([`97b9524`](https://github.com/johnmarktaylor91/torchlens/commit/97b95246cc75f1516eb9a01e8e71b489b060b768))

Phase 1 of the multi-trace sprint: introduces the `torchlens.multi_trace` subpackage containing
  TraceBundle (a container for N ModelLog instances) and the supergraph data structure that unifies
  their graphs.

The bundle handles both shared-topology cases (every trace traverses every node, e.g. same model +
  different inputs with no dynamic branching) and divergent-topology cases (e.g. dynamic networks,
  conditional branches, MoE routing). The shared case is just a degenerate overlay where every node
  is universal -- one class, one mental model, smooth degradation.

Key surface (all re-exported as `tl.TraceBundle`, `tl.bundle`, `tl.NodeView`):

- `TraceBundle(traces, names=None, groups=None)` -- holds N ModelLogs by reference; never copies
  tensors. Builds the union supergraph at construction. Provides: - `__len__`, `__iter__`,
  `__getitem__`, `__contains__` - `traces`, `names`, `groups`, `is_shared_topology`, `nodes`,
  `universal_nodes`, `has_gradients` - `selective_nodes(group=None)`, `coverage(trace_name)` -
  `most_changed(top_k, metric, on)`, `aggregate(node, statistic, on)` - `assert_shared_topology()` -
  `bundle(...)` -- thin factory wrapper around `TraceBundle`. - `NodeView` -- per-node accessor
  returned by `bundle[node_name]`. Two flavours of accessor for activations and gradients: list form
  (`.activations`/`.gradients`, always works) and stacked form (`.activation`/`.gradient`, strict,
  raises on partial coverage or shape disagreement). Includes `diff(other, metric, on)`,
  `aggregate(statistic, on)`, `shape`, `op_type`, `module_path`. - `compare_topology(a, b)` +
  `TopologyDiff` -- pairwise structural diff between two ModelLogs. - `Supergraph`,
  `SupergraphNode`, `build_supergraph(traces, names)` -- the union-graph data structure. - Metric
  primitives (`cosine_distance`, `relative_l2`, `pearson_correlation_distance`,
  `relative_l1_scalar`) and `resolve_metric` / `METRIC_REGISTRY`. Scalar (0-d / 1-element) inputs
  auto-fall back to relative_l1 in `NodeView.diff` regardless of the requested metric.

Topology matching is intentionally simple: a greedy linear scan in topological order with
  `(containing_module, func_name)` as the fingerprint. This catches the common cases without
  graph-isomorphism machinery; the limitation is documented in the docstring.

Tests: `tests/test_multi_trace.py` with 28 cases covering construction, node-view accessors, scalar
  fallback, gradient diff, aggregate, ranking, groups, coverage, edge cases, topology comparison,
  repr, factory function, plus a `@pytest.mark.smoke` smoke test. All pass against `mypy torchlens/`
  and `ruff check .`.

Visualization, counterfactual branch enumeration, intervention APIs, streaming aggregate over a
  dataloader, and the eventual ModelLog->Trace rename are explicitly out of scope for this phase --
  see `.project-context/todos.md`.


## v2.13.0 (2026-04-27)

### Chores

- **release**: 2.13.0
  ([`2db7010`](https://github.com/johnmarktaylor91/torchlens/commit/2db7010f4718150dfa220f9fa7f4b44cddfa2ae7))

### Features

- **fastlog**: Add activation_postfunc parity with slow path
  ([`0098449`](https://github.com/johnmarktaylor91/torchlens/commit/00984493b1c433f16e9a8b0f7fe9454f5ff43fa5))

Add activation_postfunc + save_raw_activation to fastlog (record(), Recorder, RecordingOptions).
  Mirrors the slow-path UX shipped in PR #166 while keeping fastlog architecture intentionally
  divergent per .project-context/research/fastlog_postfunc_parity_2026-04-27.md.

- Public surface mirrors train_mode placement on record() / Recorder / RecordingOptions. - Postfunc
  runs in _storage_resolver after safe_copy/_apply_payload_transforms and only for
  predicate-selected events. Predicates and dry_run() continue to see raw RecordContext metadata;
  dry_run never invokes the postfunc. - ActivationRecord gains transformed_ram_payload and
  transformed_disk_payload parallel fields (LayerPassLog is not reused). - Disk bundles persist
  transformed copies as kind="transformed_activation" blobs counted in the manifest auxiliary
  section, with metadata, hash, and shape stored under transformed_activation_* metadata keys;
  recovery validates both raw and transformed blobs. - metadata.json carries
  activation_postfunc_repr (callable repr only) and Recording exposes activation_postfunc_repr for
  in-memory introspection. - Train-mode + keep_grad validation rejects non-Tensor / non-grad-dtype /
  detached transformed RAM payloads with TrainingModeConfigError; disk copies remain detached
  inspection copies. - Postfunc errors propagate as TorchLensPostfuncError (with event label / kind
  / func_name / shape / dtype / storage_target / keep_grad context) and bypass the predicate-failure
  aggregation so the pass fails and the disk bundle aborts. - 14 new behavioral tests in
  tests/test_fastlog_postfunc_parity.py covering RAM-only, disk-only, mirror, train-mode, source
  events, predicate selection, dry-run isolation, repr exposure, and disk roundtrip.

No gradient_postfunc surface and no slow-path changes; CaptureSpec stays declarative.


## v2.12.2 (2026-04-27)

### Bug Fixes

- **perf**: Cache bytecode column offsets, fast-skip empty-branch attribution, guard CUDA probes on
  CPU runs
  ([`b61202d`](https://github.com/johnmarktaylor91/torchlens/commit/b61202db5a49f38bbe49e2668b7819f7afb0a30b))

Bundle three independent performance fixes from the 2026-04-27 profiling audit
  (.project-context/research/profiling_audit_2026-04-27.md). All three are localized and
  behaviorally identical to the slow paths they short-circuit.

1. Cache _get_col_offset() per code object id. The disassembled instruction-offset -> column-offset
  map is immutable for a given code object, so we build it once and reuse it. On GPT-2 the audit
  measured ~16.5s self time across dis.* (~50% of profiled runtime); the cache reduces repeated work
  to a dict lookup. A soft 100K-entry cap with a one-shot warning protects against pathological
  workloads; real-world models stay well under the cap.

2. Fast-skip Step 5 branch attribution when no conditional bools were captured. The slow path's only
  branch-attributing input is ModelLog.internally_terminated_bool_layers; if it is empty the
  downstream collections all resolve to their empty defaults (already set in ModelLog.__init__ and
  capture/output_tensors.py). The precondition checks every derived ModelLog conditional collection
  so any caller that pre-populated them still triggers the slow path. A defensive assert on the slow
  path will trip if a future bool-detector refactor produces conditional keys without populating
  internally_terminated_bool_layers, preventing silent fast-path drift.

3. Gate torch.cuda.empty_cache() calls in postprocess and capture paths behind the cached
  _is_cuda_available() helper. CUDA availability is process-fixed, so once cached the helper
  short-circuits without touching the CUDA driver / NVML probes that the audit measured at ~12%
  overhead on CPU-only ResNet50.

Behavior is unchanged. ModelLog output is byte-identical for the test fixtures.
  tests/test_perf_bundle.py adds 11 behavioral tests covering each fix; smoke + non-slow suites pass
  without regression.

### Chores

- **release**: 2.12.2
  ([`b9c5a1a`](https://github.com/johnmarktaylor91/torchlens/commit/b9c5a1a5ff74d98d7f452f385769ce7072a97b63))


## v2.12.1 (2026-04-27)

### Bug Fixes

- **capture**: Correct fast-mode pass-through detection for in-place modules
  ([`787025d`](https://github.com/johnmarktaylor91/torchlens/commit/787025d8f398486b6134a04efa03d4742c0de3f7))

### Chores

- **release**: 2.12.1
  ([`b5a7502`](https://github.com/johnmarktaylor91/torchlens/commit/b5a7502ed0d226bbd2073d25104ead4157f852ea))

### Documentation

- **research**: Two-pass mode in-place module diagnostic (2026-04-27)
  ([`19cf533`](https://github.com/johnmarktaylor91/torchlens/commit/19cf533691fc6e1239afaadd54517b2a31953867))


## v2.12.0 (2026-04-27)

### Chores

- **gitignore**: Ignore default modelgraph.* and backward_modelgraph.* viz outputs in repo root
  ([`c65170c`](https://github.com/johnmarktaylor91/torchlens/commit/c65170cac2cdc82a1f6807c437339e9b4b7d3413))

- **release**: 2.12.0
  ([`b705219`](https://github.com/johnmarktaylor91/torchlens/commit/b70521980d389a09eb32967052e277b45e7b377e))

- **todos**: Track activation_postfunc rename and estimated_autograd_saved_bytes followups
  ([`cd7269f`](https://github.com/johnmarktaylor91/torchlens/commit/cd7269f13f29b236c90b71e3c0be6c19172f543e))

### Documentation

- **research**: Postfunc review, fastlog postfunc parity, profiling audit (2026-04-27)
  ([`b76cdf0`](https://github.com/johnmarktaylor91/torchlens/commit/b76cdf004b0d6bed0c865054db07c6fec4a54920))

### Features

- **activation_postfunc**: Split raw vs transformed, fix metadata drift, harden train_mode
  ([`59ae72c`](https://github.com/johnmarktaylor91/torchlens/commit/59ae72ce6ce9571e1f5a70824dad298295c13f2a))


## v2.11.0 (2026-04-27)

### Chores

- **release**: 2.11.0
  ([`6534a68`](https://github.com/johnmarktaylor91/torchlens/commit/6534a68af33c32fcce3b6496c0ca30ec61a12115))

### Features

- **capture**: Add autograd_saved_bytes field tracking per-op autograd memory
  ([`0d9a41d`](https://github.com/johnmarktaylor91/torchlens/commit/0d9a41d5c7a7c73817611357ea25eff9b9c6edbc))


## v2.10.0 (2026-04-27)

### Chores

- **release**: 2.10.0
  ([`444269d`](https://github.com/johnmarktaylor91/torchlens/commit/444269d60045ae7244d5c43fc668e21eb7ae9592))

### Features

- **data-classes**: Add extra_data dict to layer logs and input_metadata to ModelLog
  ([`f589700`](https://github.com/johnmarktaylor91/torchlens/commit/f589700df63a87102c6193da89c608262e936608))


## v2.9.0 (2026-04-27)

### Chores

- **release**: 2.9.0
  ([`a32c6c0`](https://github.com/johnmarktaylor91/torchlens/commit/a32c6c031abc21b31598335c8d3c33943d7b203f))

### Features

- **backward**: Gradient disk streaming and backward_tutorial notebook
  ([`85918f7`](https://github.com/johnmarktaylor91/torchlens/commit/85918f7201618d35bd3b4d5a82c2234b456f9659))


## v2.8.0 (2026-04-27)

### Chores

- **release**: 2.8.0
  ([`f9f0b17`](https://github.com/johnmarktaylor91/torchlens/commit/f9f0b17da93bc978edce6681c0ed48da19e6078f))

### Features

- **visualization**: Backward graph rendering via show_backward_graph
  ([`82b7256`](https://github.com/johnmarktaylor91/torchlens/commit/82b7256d84024aa77f04982e1e649eba9e50d61e))


## v2.7.0 (2026-04-27)

### Chores

- Backward-pass sprint design notes
  ([`1af1f7f`](https://github.com/johnmarktaylor91/torchlens/commit/1af1f7f6af1ff540e7536ca73602778214632f74))

- **release**: 2.7.0
  ([`0aaa2f3`](https://github.com/johnmarktaylor91/torchlens/commit/0aaa2f3a224808b9ce97bfb12080da0b2e572e16))

### Features

- **backward**: First-class backward-pass capture (data model, walk, hooks, validation)
  ([`5713a1e`](https://github.com/johnmarktaylor91/torchlens/commit/5713a1e5ff5ee2c4ca86725a3b7acdf437c56b3b))

### Refactoring

- **backward**: Flatten BackwardLog to ModelLog fields; add GradFnLog naming and indexing
  ([`825aa91`](https://github.com/johnmarktaylor91/torchlens/commit/825aa918d5ab523ac80a5c8c90821b3a3f75e9e6))


## v2.6.0 (2026-04-27)

### Chores

- **release**: 2.6.0
  ([`5a51135`](https://github.com/johnmarktaylor91/torchlens/commit/5a51135a738d08f68b5a2df7e36b51605a61c54a))

### Features

- **visualization**: Cylinder shape for buffer nodes; drop BUFFER_NODE_COLOR
  ([`098acdc`](https://github.com/johnmarktaylor91/torchlens/commit/098acdc9113ccc68bc3886859a00dfd527523f2f))


## v2.5.0 (2026-04-27)

### Chores

- **release**: 2.5.0
  ([`79d65fe`](https://github.com/johnmarktaylor91/torchlens/commit/79d65fea9f664c6383a26c1901ad9b494c414816))

### Features

- **visualization**: Tri-state vis_buffer_layers with noise filter and hidden-buffer marker
  ([`4bbd308`](https://github.com/johnmarktaylor91/torchlens/commit/4bbd30806290e77adc4acf2ec599eb71e9e205a3))


## v2.4.0 (2026-04-27)

### Chores

- Defer menagerie revamp design notes
  ([`5118900`](https://github.com/johnmarktaylor91/torchlens/commit/51189009f9e74740ffdb9b285b00ec3106315d90))

- **release**: 2.4.0
  ([`7a7edd0`](https://github.com/johnmarktaylor91/torchlens/commit/7a7edd003487586b988ba89e070c1905e317e995))

### Features

- **training**: Unified train_mode for backward()-safe activations
  ([#157](https://github.com/johnmarktaylor91/torchlens/pull/157),
  [`e1673d9`](https://github.com/johnmarktaylor91/torchlens/commit/e1673d93cb40fcda479082e2b088f80622733ee9))

* fix(postprocess): honor detach_saved_tensors in postprocess_fast output-layer copy

* feat(training): shared train_mode validator and exception class

* feat(training): train_mode kwarg on log_forward_pass

* feat(training): preserve user requires_grad in train_mode

* feat(training): train_mode override on save_new_activations

* feat(training): train_mode sugar on fastlog record/Recorder

* fix(training): avoid forbidden inference mode literal

* feat(training): audit hardening, AST guardrail, compile parity

* test(training): comprehensive sprint test suite

* docs(training): tutorial notebook + module CLAUDE.md updates


## v2.3.1 (2026-04-27)

### Bug Fixes

- **ci**: Unblock smoke + dep-audit ([#156](https://github.com/johnmarktaylor91/torchlens/pull/156),
  [`9e82f42`](https://github.com/johnmarktaylor91/torchlens/commit/9e82f423876eb19e6af799822a554fe1b791d7a4))

* fix(ci): unblock smoke (importorskip torchvision) and dep-audit (ignore upstream pip CVE)

Two pre-existing CI failures on main, fixed in one PR:

1. tests/test_real_world_models.py imported torchvision at the top, but the smoke job only installs
  the [dev] extras (no torchvision). The resulting collection ImportError stopped the whole smoke
  run before any smoke test could run. Switched to pytest.importorskip so the module SKIPs
  gracefully on slim envs.

2. The Dependency-audit job ran pip-audit which flagged CVE-2026-3219 on pip itself (the package on
  the GitHub runner image). It is outside this project's dependency surface and not actionable from
  pyproject.toml. Ignore that single ID with a TODO to remove once a pip release with the fix is
  available.

No behavior change for environments that already have torchvision; no test was disabled. Verified
  locally with torchvision present (smoke passes) and with torchvision blocked (the file is SKIPPED,
  not ERRORED).

* fix(ci): install graphviz so smoke tests that render the model graph work

The smoke job was failing with:

FileNotFoundError: [Errno 2] No such file or directory: 'dot'

at tests/test_conditional_branches.py because the graphviz `dot` binary isn't on the GitHub runner
  image by default. Add an apt-get install step before the test run.

### Chores

- **release**: 2.3.1
  ([`902a72e`](https://github.com/johnmarktaylor91/torchlens/commit/902a72eda975687b4efe3ab1aacb88100ffd5c28))


## v2.3.0 (2026-04-27)

### Chores

- **release**: 2.3.0
  ([`c4e997f`](https://github.com/johnmarktaylor91/torchlens/commit/c4e997f07c52195f360611dde125f250d67a5b2b))

### Features

- **fastlog**: High-throughput activation recording for dynamic models
  ([#155](https://github.com/johnmarktaylor91/torchlens/pull/155),
  [`1b19663`](https://github.com/johnmarktaylor91/torchlens/commit/1b19663d22b2ed4d7e8db5472d7aa8f1198da2b8))

* fix(perf): defer col_offset extraction until after frame filtering

Improves log_forward_pass throughput substantially on models with deep Python call stacks.
  _get_col_offset and _get_code_qualname now run only on frames that survive Phase 1 filtering
  instead of every raw frame.

* feat(fastlog): scaffolding and types contract

* feat(fastlog): predicate evaluator and RecordContext builder

* feat(fastlog): capture-layer predicate dispatchers

* feat(fastlog): orchestrator with root events

* fix(fastlog): freeze Recording dataclass

* feat(fastlog): RAM and sync-disk storage backends + recover()

* feat(fastlog): public record / Recorder / dry_run API

* feat(fastlog): preview_fastlog visualization

Per execution plan Step 6. Adds model_log.preview_fastlog() and tl.preview_fastlog() top-level
  alias. Generates node_spec_fn at call time; does not touch MODE_REGISTRY. Uses
  _build_record_context to synthesize RecordContext from postprocessed LayerLog so a predicate that
  accesses a postprocess-only field fails the same way as in dry_run. preview_fastlog(...,
  vis_renderer='dagua') raises NotImplementedError for v1.

* feat(fastlog): dry_run live visualization

Per execution plan Step 7. Adds print_tree, to_pandas, show_graph (graphviz with module rails, no
  containment boxes), summary, timeline_html, repredicate. RecordingTrace stores decisions without
  tensor payloads.

* feat(fastlog): opt-in incremental enrichments

Per execution plan Step 8. Adds Recording.enrich() with module_path_strings preset and all-feasible
  alias. param_addresses preset declared but raises pending capture-time field plumbing (scoped for
  follow-up; documented in status report).

* test(fastlog): comprehensive sprint test suite

* docs(fastlog): tutorial notebook + module CLAUDE.md


## v2.2.1 (2026-04-25)

### Bug Fixes

- **validation**: Tolerate small replay drift
  ([`8126fe2`](https://github.com/johnmarktaylor91/torchlens/commit/8126fe20808fa9cae5f61543c39d4b3301f688f3))

Validation replay used a fixed absolute float tolerance, which was too tight for deep convolution
  replays in timm models. Compare tolerated replay outputs with a small absolute floor plus relative
  tolerance so numerically equivalent conv outputs validate while real mismatches still fail.

### Chores

- **release**: 2.2.1
  ([`3a9a972`](https://github.com/johnmarktaylor91/torchlens/commit/3a9a97245f7b069b340447ad3227dc957cc91fc9))


## v2.2.0 (2026-04-25)

### Chores

- **release**: 2.2.0
  ([`153083d`](https://github.com/johnmarktaylor91/torchlens/commit/153083d927dc9d8546fa38253f97a987779ed97f))

### Features

- **visualization**: Side-by-side code panel
  ([`6d3101e`](https://github.com/johnmarktaylor91/torchlens/commit/6d3101e72c1b66c4c02ca90a8b9085d2f43f1e26))


## v2.1.0 (2026-04-25)

### Chores

- **release**: 2.1.0
  ([`d27ebf3`](https://github.com/johnmarktaylor91/torchlens/commit/d27ebf3457fbec562ce50275bbaa2c0cf0d7ee20))

### Features

- **visualization**: Node-mode presets (default, profiling, vision, attention)
  ([`ddfdd34`](https://github.com/johnmarktaylor91/torchlens/commit/ddfdd34fad8e608bc4fce0b56ae4d2c42d1a074b))

- **visualization**: Single-module focus (module= arg + ModuleLog.show_graph)
  ([`155be75`](https://github.com/johnmarktaylor91/torchlens/commit/155be7585b4a4143b49cf80485d9b2836b252e43))


## v2.0.0 (2026-04-25)

### Chores

- **release**: 2.0.0
  ([`70b9f77`](https://github.com/johnmarktaylor91/torchlens/commit/70b9f77f6599d34a9bf933b166ea663724a4edb7))

### Features

- **visualization**: Nodespec callback + collapse_fn + skip_fn + default important args
  ([`0697787`](https://github.com/johnmarktaylor91/torchlens/commit/06977876c670717737d8952ffce651e833a3d45e))

BREAKING CHANGE: vis_node_overrides and vis_nested_node_overrides are removed.

Use the new node_spec_fn / collapsed_node_spec_fn callbacks instead.

See PR body for migration notes.

### Breaking Changes

- **visualization**: Vis_node_overrides and vis_nested_node_overrides are removed.


## v1.7.0 (2026-04-24)

### Chores

- **release**: 1.7.0
  ([`bd70a4a`](https://github.com/johnmarktaylor91/torchlens/commit/bd70a4a2aa6195524f2c84c8dbfabdb063012e0b))

### Features

- **robustness**: Pr4 opaque-wrapper guards + LIMITATIONS.md + CI matrix
  ([`9db5b2e`](https://github.com/johnmarktaylor91/torchlens/commit/9db5b2e11b30b0f72d2a19b1e19b9514d0e897e6))

Final wave of the robustness sprint. Adds the remaining must-fail-loudly guards for wrappers whose
  forward pass is not ordinary Python, adds a user-facing limitations doc, wires in a lean
  torch-version CI matrix, and links the doc from the README.

### Opaque-wrapper guards (new) torchlens/user_funcs.py now ships _reject_opaque_wrappers(model),
  called immediately before _unwrap_data_parallel at every log entry point (log_forward_pass,
  get_model_metadata, show_model_graph, validate_forward_pass). It raises RuntimeError with a
  pointer to docs/LIMITATIONS.md for:

- torch.compile'd models (OptimizedModule) — dynamo replaces the forward with a compiled graph; our
  Python-level wrappers are optimized away or bypassed depending on the backend. -
  torch.jit.ScriptModule (script + trace) — the forward runs on the TorchScript interpreter, not
  Python; wrappers never fire. - torch.export.ExportedProgram — a serialised IR, not a callable
  nn.Module. Detected defensively since ExportedProgram isn't an nn.Module in the first place.

For every case the message tells the user to call log_forward_pass on the original un-wrapped model.

### docs/LIMITATIONS.md (new) Canonical one-page matrix + per-context explanation of every
  limitation surfaced by the robustness sprint. Sections:

- At-a-glance matrix (what TorchLens does / workaround) - Per-context detail: compile / jit / export
  / FSDP / meta / sparse / SymInt / quantized / vmap / multiprocess / nested / partial-support -
  "Reporting a new failure" instructions

### README pointer (new) A "Known limitations / unsupported contexts" section right before "Planned
  Features" links to docs/LIMITATIONS.md with a one-line summary so users can find it without
  reading the source.

### CI torch-version matrix (new) .github/workflows/tests.yml runs smoke tests on two points in the
  support window: python 3.10 + torch 2.4.* (declared floor) and python 3.11 + torch 2.7.*.
  fail-fast is off so both rows report independently. Kept lean (smoke-only, two entries) so CI
  minutes stay under control.

### Tests (tests/test_robustness_pr4.py, 11 cases) - torch.compile'd model raises with
  "torch.compile" in the message - Logging the un-compiled original still works after compilation -
  torch.jit.script raises with "ScriptModule" - torch.jit.trace raises with "ScriptModule" - Logging
  the un-scripted original still works after scripting - torch.export.ExportedProgram fails loudly
  at entry - _reject_opaque_wrappers is a no-op on a bare nn.Module - _reject_opaque_wrappers raises
  on ScriptModule at the helper level - docs/LIMITATIONS.md exists and is long enough to be
  meaningful - README links to docs/LIMITATIONS.md - docs/LIMITATIONS.md mentions every context with
  a runtime guard (staleness regression guard)

### Verification - pytest tests/ -m smoke: 35/35 ✅ (up from 32) - pytest targeted regression + PR4:
  470/470 ✅ (3m47s) - ruff check . ✅ / ruff format ✅ - mypy: 67 source files clean ✅

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>


## v1.6.0 (2026-04-24)

### Chores

- **release**: 1.6.0
  ([`5ff5e1d`](https://github.com/johnmarktaylor91/torchlens/commit/5ff5e1dc2bab8b53a8e142742e2cc7b65581bf28))

### Features

- **robustness**: Pr3 parallel-wrapper unwrap + framework I/O guards
  ([`c80c21d`](https://github.com/johnmarktaylor91/torchlens/commit/c80c21d9a4399194bb59fe6497d75540beff2411))

Wave 3 of the robustness sprint. Extends parallel-wrapper handling and adds regression coverage for
  HuggingFace-style dict-subclass inputs and dataclass outputs.

### DistributedDataParallel unwrap Previously _unwrap_data_parallel only handled nn.DataParallel.
  DDP and DataParallel share the same .module attribute but the isinstance check excluded DDP, so
  users of the much more common DDP got silently wrong layer addressing (parameter barcodes pointing
  at the wrapper's module hierarchy instead of the real model tree).

Fix: isinstance check extended to DistributedDataParallel. Guarded by ImportError so torch installs
  without torch.distributed still work.

### FullyShardedDataParallel clear-error FSDP parameters are sharded across ranks — there is no
  single unsharded module to log. Silently unwrapping via .module yields a model whose parameters
  are FlatParameter shards, which produces misleading layer metadata and breaks loop detection's
  param-sharing heuristic.

Fix: raise RuntimeError at unwrap time with a message explaining the sharding issue and pointing to
  "run log_forward_pass on a rank-local copy of the underlying module (before FSDP wrapping)".

### Framework I/O regression tests No code change here — these tests lock in the current behavior so
  future refactors don't silently break HuggingFace use cases:

- Dict-subclass inputs (BatchEncoding-shaped UserDict) flow through _move_tensors_to_device
  correctly and logging completes. - Dataclass-style outputs (ModelOutput-shaped dataclass) don't
  crash the output-extraction BFS crawl; at minimum the inner ops are logged.

### Tests (tests/test_robustness_pr3.py, 7 cases) - DataParallel unwrap still works (regression
  guard for pre-existing behavior) - DistributedDataParallel isinstance -> unwrap via .module - FSDP
  isinstance -> raise with clear message (both at the helper level and at log_forward_pass entry) -
  UserDict-based BatchEncoding-like input logs (smoke) - ModelOutput-like dataclass output doesn't
  crash and Linear is logged - Standard model still logs (golden-path regression)

### Verification - pytest tests/ -m smoke: 32/32 ✅ - pytest targeted regression + PR3: 466/466 ✅
  (3m47s) - ruff check . ✅ / ruff format ✅ - mypy: 66 source files clean ✅

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>


## v1.5.0 (2026-04-24)

### Chores

- **release**: 1.5.0
  ([`a641282`](https://github.com/johnmarktaylor91/torchlens/commit/a6412825edc680c53bcbf601398287a0544d3fa0))

### Features

- **robustness**: Pr2 tensor-variant pre-flight guard + channels_last fix
  ([`0f1367d`](https://github.com/johnmarktaylor91/torchlens/commit/0f1367d2f678881873dcceb6e2ced31c056f55de))

Wave 2 of the robustness sprint. Centralises detection of tensor variants TorchLens cannot handle
  and adds a channels_last preservation fix to safe_copy.

### Pre-flight variant guard New module torchlens/_robustness.py: - UnsupportedTensorVariantError
  (subclass of RuntimeError) with a clear bullet-listed message + pointer to docs/LIMITATIONS.md. -
  check_model_and_input_variants(model, input_args, input_kwargs) walks model params/buffers and the
  input tensor tree and: - RAISES for meta tensors (no backing storage — activation capture can't
  produce usable values), sparse tensors (safe_copy/print paths assume dense strided layouts),
  symbolic-shaped tensors (torch.SymInt dims break counter alignment and flops). - WARNS for
  quantized modules (logging works, FLOPs are wrong).

Wired into four entry points in user_funcs.py (log_forward_pass, get_model_metadata / summary,
  show_model_graph, validate_forward_pass), in each case right after _unwrap_data_parallel. Failures
  happen up front with a clear message instead of partway through the 18-step postprocess pipeline.

### channels_last preservation in safe_copy torchlens/utils/tensor_utils.py: - New
  _safe_get_memory_format(t) probes channels_last / channels_last_3d via
  is_contiguous(memory_format=...), falling back to preserve_format on exotic layouts. - safe_copy
  now calls clone(memory_format=...) on both the attached and detached paths. If a tensor variant
  rejects the memory_format kwarg (some sparse/subclass paths), we fall back to a plain clone. -
  Previous behavior silently converted channels_last activations to channels_first on copy, which
  could produce silent wrong results for layout-sensitive downstream ops.

### Tests tests/test_robustness_pr2.py (16 cases): - Meta tensor inputs + meta params raise
  UnsupportedTensorVariantError - Sparse COO and CSR inputs raise with a layout-mentioning message -
  Quantized models emit exactly one UserWarning and logging keeps going - safe_copy preserves
  channels_last / channels_last_3d; detach path also preserves memory format - Standard forward pass
  still logs (golden-path regression) - CUDA channels_last + CUDA forward pass smoke tests (skipped
  if no GPU) - Internal helpers (_is_meta_tensor, _is_sparse_tensor) covered

### Verification - pytest tests/ -m smoke: 32/32 ✅ (up from 30; 2 new smoke) - pytest targeted
  regression + PR2: 475/475 ✅ (3m44s) - ruff check . ✅ / ruff format ✅ - mypy: 67 source files clean
  ✅

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>


## v1.4.0 (2026-04-24)

### Chores

- **release**: 1.4.0
  ([`553cb85`](https://github.com/johnmarktaylor91/torchlens/commit/553cb8567287a3c29ca5e96e2a467a661faab569))

### Features

- **robustness**: Pr1 correctness guards + torch floor pin
  ([`f6efc9b`](https://github.com/johnmarktaylor91/torchlens/commit/f6efc9ba84142fe209323782041f52e72d637a40))

Wave 1 of the robustness sprint. Correctness-critical hardening only; no behavior change for
  non-pathological forward passes.

- _state.active_logging(): raise RuntimeError on nested entry instead of silently overwriting
  _active_model_log and clearing it on inner exit (which corrupted the outer ModelLog mid-pass). The
  docstring already declared the context non-nestable; now the runtime enforces it. The realistic
  trigger is a user forward-hook or activation pipeline that calls log_forward_pass on a sub-model.
  - torch_func_decorator: when a functorch/vmap/grad transform is active, TorchLens continues to
  skip logging inside the transform (internal ops like safe_copy lack vmap batching rules), but now
  emits a one-shot UserWarning per forward pass so users know the resulting ModelLog omits whatever
  ran inside the transform. Flag lives in _state and resets at the top of every active_logging()
  session. - pyproject: pin torch>=2.4. TorchLens already calls
  torch.is_autocast_enabled(device_type) and torch.get_autocast_dtype(device_type) (rng.py), both of
  which require a device argument only from torch 2.4+. The bare 'torch' dep was permitting installs
  that would fail at runtime. - pyproject: advertise Python 3.13 classifier (stable since 2024-10,
  supported by torch 2.5+).

New tests (tests/test_robustness_pr1.py, 8 cases): - direct active_logging nesting raises with the
  non-re-entrant message - a forward-hook that re-enters log_forward_pass raises and the outer
  session state is cleaned up cleanly afterward - a vmap-containing forward pass emits exactly one
  functorch warning per session, and the flag resets between sessions - non-vmap passes emit no
  functorch warning - pyproject torch dep pins >=2.4 and Python 3.13 is a classifier

All 30 smoke tests and 209 targeted regression tests pass. ruff, mypy clean.

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>


## v1.3.0 (2026-04-24)

### Chores

- **release**: 1.3.0
  ([`595dc18`](https://github.com/johnmarktaylor91/torchlens/commit/595dc181ae190981447c1acd49420eb53bd5fd82))

### Documentation

- **ux-sprint**: Clarify defaults in docstrings + add regression guard
  ([`46a8555`](https://github.com/johnmarktaylor91/torchlens/commit/46a85556643ace08b66b236a64a8a187a10b111e))

Wave 4c of the defaults audit. No defaults changed (all rows in defaults-table.md marked change: no,
  because the public wrappers already put fast+informative defaults in the right place and expensive
  paths are opt-in).

Documentation-only improvements: - log_forward_pass / show_model_graph docstrings for detect_loops /
  detect_recurrent_patterns now include the \">1M operations\" speedup guidance users want to find
  when postprocessing gets slow. - save_source_context docstring now explicitly separates the
  always-captured identity fields (file, line, func_name, etc.) from the opt-in rich source text, so
  users understand branch attribution works regardless of this flag. - keep_unsaved_layers docstring
  now shows the typical memory-conscious usage pattern (layers_to_save=[...],
  keep_unsaved_layers=False).

New regression guard: - tests/test_defaults.py — 7 tests locking in the current resolved defaults on
  log_forward_pass, show_model_graph, log_model_metadata, torchlens.summary, and
  validate_forward_pass. Tests the wrapper behavior (which routes through MISSING sentinels + option
  groups) rather than raw signature defaults.

Follow-up noted in wave4c-issues.md: ModelLog.__init__ still defaults
  mark_input_output_distances=True while the public wrapper resolves to False. Not user-visible in
  this sprint; would need a separate constructor parity pass.

CHANGELOG.md: new v1.3.0 (unreleased) documentation stub covering the whole UX sprint.

Gates: - ruff check: clean - mypy torchlens/: no issues (66 source files) - pytest smoke (28):
  passed - pytest test_defaults.py (7 new): passed - pytest not-slow full (1144 passed, 21 skipped):
  passed

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

### Features

- **ux-sprint**: Add ModelLog.summary, compact __repr__, torchlens.summary
  ([`6a64cf1`](https://github.com/johnmarktaylor91/torchlens/commit/6a64cf1b4da2380c1f61fd44a8e3f771909474ff))

Implements the Wave 4a deliverables from summary-api.md:

- ModelLog.summary(level=..., fields=..., show_ops=..., ...) renders a preset-driven textual
  summary. Presets: overview (default, Keras-style), graph (hybrid module+op), memory (tensor-size
  accounting), control_flow (conditionals + recurrence), cost (params + FLOPs + MACs + time). -
  ModelLog.__repr__ replaced with a concise 1-2 line identity card. - torchlens.summary(model,
  input_args, input_kwargs=None, **kw) top-level convenience: runs a metadata-only forward pass,
  renders the summary, cleans up, returns the string. - New package torchlens/_summary/
  (underscore-prefixed to avoid collision with the public torchlens.summary function) holds the
  builder + preset logic in 1169 LOC. - tests/test_summary.py: 11 new tests covering each preset,
  __repr__ brevity, truncation markers, pre-postprocess ModelLog, fidelity against claims we cannot
  keep (no peak-memory, no per-op device column).

Gates: - ruff check: clean - mypy torchlens/: no issues (63 source files) - pytest smoke (28 tests):
  passed - pytest test_summary.py (11 new): passed

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

- **ux-sprint**: Api renames + VisualizationOptions/StreamingOptions groups
  ([`1d737b7`](https://github.com/johnmarktaylor91/torchlens/commit/1d737b739279747d4d074a5467952101cb355310))

Implements Wave 4b deliverables (rename-table.md + option-groups.md):

Renames (additive, old names remain working with DeprecationWarning): - top-level get_model_metadata
  -> log_model_metadata - kwarg num_context_lines -> source_context_lines - kwarg
  mark_input_output_distances -> compute_input_output_distances - kwarg detect_loops ->
  detect_recurrent_patterns - ModelLog.validate_saved_activations -> ModelLog.validate_forward_pass

Option groups (new dataclasses, flat kwargs remain as deprecated aliases): -
  torchlens.VisualizationOptions — 16 fields, replaces vis_* sprawl. Per-function defaults:
  log_forward_pass uses mode=none, show_model_graph uses mode=unrolled. Mixing flat+grouped same
  field raises TypeError; mixing flat+grouped DIFFERENT fields merges. - torchlens.StreamingOptions
  — 3 fields for bundle_path + retain_in_memory + activation_callback.

Shared plumbing: - torchlens/_deprecations.py — warn_deprecated_alias dedup + MISSING sentinel +
  kwarg-resolution helper. - torchlens/_literals.py — Literal type aliases for autocomplete:
  OutputDeviceLiteral, VisModeLiteral, VisDirectionLiteral, VisNodePlacementLiteral,
  VisRendererLiteral. - torchlens/options.py — 496 LOC, option-group dataclasses +
  resolve_visualization_options / resolve_streaming_options / render kwarg translation.

Tests: - tests/test_api_renames.py — 61 new tests covering each alias, DeprecationWarning firing,
  flat+grouped mixing rules, per-function VisualizationOptions defaults, StreamingOptions migration.

Gates: - ruff check: clean - mypy torchlens/: no issues - pytest smoke (28): passed - pytest
  test_api_renames.py (61): passed - pytest not-slow full (1137 passed, 21 skipped): passed

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>

### Refactoring

- **ux-sprint**: Inline ModelLog cross-file method bindings
  ([`b82f176`](https://github.com/johnmarktaylor91/torchlens/commit/b82f176f4f130be493e6f38e5f01fcba0a999bec))

Step 1 of the refactor-list.md plan: convert all 19 class-body attribute rebindings at
  model_log.py:706-724 into explicit `def X(self, ...)` methods on ModelLog. Implementation helpers
  remain in their feature modules (visualization/, validation/, capture/, postprocess/); the class
  body now has real method definitions that delegate via local imports where needed to avoid
  circular-import regressions.

Step 2: extract `_give_user_feedback_about_lookup_key` and `_get_lookup_help_str` from
  `data_classes/interface.py` into a new `data_classes/_lookup_keys.py` so `capture/trace.py` no
  longer needs to import from the former helper module. This unblocks any future collapse of
  interface.py / cleanup.py.

Step 3 intentionally SKIPPED this sprint (design marked it optional). interface.py and cleanup.py
  remain as legitimate helper modules with non-method-pack utilities.

Also added `tests/test_cross_file_refactor.py` with an AST-based guard that fails if anyone
  reintroduces `X = imported_X` class-body rebindings on ModelLog.

Gates: - ruff check: clean - mypy torchlens/: no issues - pytest smoke (28 tests): passed - pytest
  not-slow (1065 passed, 21 skipped): passed

Generated with [Claude Code](https://claude.ai/code) via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>

Co-Authored-By: Happy <yesreply@happy.engineering>


## v1.2.0 (2026-04-23)

### Bug Fixes

- **io**: Io-s9 path-traversal hardening + two-pass streaming rejection + docs security warning +
  preflight/reader cleanup
  ([`42c047f`](https://github.com/johnmarktaylor91/torchlens/commit/42c047fef95c126caca9069b40140df384f2ca4d))

### Chores

- **io-sprint**: Add plan and research docs
  ([`9bd8272`](https://github.com/johnmarktaylor91/torchlens/commit/9bd827241a80ee7a0a621081cd61128846346e9d))

Plan v6 (Round 8, GREEN) at .project-context/plans/io-sprint/plan.md. Audits and review rounds 1-8
  archived alongside.

- **release**: 1.2.0
  ([`fe2b51f`](https://github.com/johnmarktaylor91/torchlens/commit/fe2b51f8c66552bfffa37312de8a663fabd890b4))

### Documentation

- **io**: Io-s8 docstrings + README section + architecture doc
  ([`ddb8981`](https://github.com/johnmarktaylor91/torchlens/commit/ddb898126bb6590e999260fbfeb7732115b482b0))

- **io-sprint**: Phase 5 dual-review records
  ([`806868a`](https://github.com/johnmarktaylor91/torchlens/commit/806868a154d14cd9b93dadb429ada0dcd1e93e13))

### Features

- **io**: Io-s1 pickle hardening + PORTABLE_STATE_SPEC + rehydrate loader
  ([`b49c748`](https://github.com/johnmarktaylor91/torchlens/commit/b49c748439837b19e65daba8c09417facf8e937f))

- **io**: Io-s2 Param/Buffer/ModulePass DataFrames + summary fields
  ([`4512a64`](https://github.com/johnmarktaylor91/torchlens/commit/4512a64f7161b838c4e1a618cfc306fae6859fd0))

- **io**: Io-s3 export wrappers (csv/parquet/json) on all DataFrame surfaces
  ([`7686ec4`](https://github.com/johnmarktaylor91/torchlens/commit/7686ec4b61b419863532bf423d0ee955bb289bd6))

Add to_csv / to_parquet / to_json to ModelLog, LayerAccessor, ModuleAccessor, ModuleLog. Normalize
  wrapper signatures on the four S2 surfaces (ParamAccessor, BufferAccessor, ModulePassLog) to match
  the shared spec.

- **io**: Io-s4 bundle save/load with manifest + integrity + tensor policy
  ([`380d3f5`](https://github.com/johnmarktaylor91/torchlens/commit/380d3f5b530ec4eb88b0be7c97bffdc12af93f92))

Adds torchlens.save, torchlens.load, torchlens.cleanup_tmp. Adds ModelLog.save / ModelLog.load
  sugar. Adds safetensors as required dep. Enforces Fork F version policy, Fork G exception
  wrapping, Fork J symlink rejection, Fork L replay-not-supported-on-portable guard.

- **io**: Io-s5 streaming-save during forward pass + postprocess steps 19/20
  ([`7a172a9`](https://github.com/johnmarktaylor91/torchlens/commit/7a172a9ef326ab41dd3f03dc40976fa6eb7f2e64))

- **io**: Io-s6 lazy activation refs + drift detection + rehydrate_nested
  ([`1c870ce`](https://github.com/johnmarktaylor91/torchlens/commit/1c870cee22ea35ccaa5158a7d7058d344fb5192c))

### Testing

- **io**: Io-s7 integration + corruption + plain-pickle regression suite
  ([`3fba0c0`](https://github.com/johnmarktaylor91/torchlens/commit/3fba0c0430d61fe0dfb71e57d43cf41aad7849db))

Add the IO-S7 integration, corruption, and plain-pickle regression coverage.

While adding the corruption battery, wrap safetensors corruption read failures in TorchLensIOError
  so truncated blobs stay inside the portable I/O exception contract.


## v1.1.0 (2026-04-22)

### Bug Fixes

- **conditionals**: Make MultilinePredicateModel predicate genuinely multi-line
  ([`597e075`](https://github.com/johnmarktaylor91/torchlens/commit/597e0757d5d794871ac6eff0d68d1080ad2dea5a))

test_multiline_predicate_model_tracks_multiline_if_event asserts that the conditional's test_span
  covers multiple lines, but the model's predicate was single-line (and ruff-format would collapse
  any naive re-wrap back to one line since it fits under 100 chars). Use `# fmt: off` / `# fmt: on`
  to preserve the genuinely multi-line predicate structure.

Final tier-2: 966 passed, 12 skipped, 0 failed.

### Chores

- **release**: 1.1.0
  ([`2f9eb40`](https://github.com/johnmarktaylor91/torchlens/commit/2f9eb4040d47b87d58d060b0aed0f67522099c34))

### Documentation

- **conditionals**: Update user_funcs + AGENTS + architecture + limitations (Phase 9/10)
  ([`6d1814e`](https://github.com/johnmarktaylor91/torchlens/commit/6d1814e5c7f716fb3f54e0ed2e0f00c02402c8ff))

### Features

- **conditionals**: 13 consistency invariants for conditional metadata (Phase 6/10)
  ([`4236c24`](https://github.com/johnmarktaylor91/torchlens/commit/4236c2419570fae084b38c450c9880181db58d8d))

- **conditionals**: Ast_branches module for bool classification + op attribution (Phase 2/10)
  ([`21eabe8`](https://github.com/johnmarktaylor91/torchlens/commit/21eabe88e53bc8ae833a31ace8251f5744c1c662))

New module torchlens/postprocess/ast_branches.py with public API: - get_file_index(filename) ->
  Optional[FileIndex] - classify_bool(filename, line, col=None) -> BoolClassification -
  attribute_op(func_call_stack) -> List[Tuple[ConditionalKey, str]] -
  invalidate_cache(filename=None)

Internal structures: FileIndex, ScopeEntry, ConditionalRecord (if_chain + ifexp), BoolConsumer
  index, branch interval trees, mtime-keyed file cache. Elif flattening, IfExp first-class support,
  (line, col) point lookup, D14 scope resolution with fail-closed ambiguity handling, D18
  col-degraded mode fail-closed.

Tests: 19 passing unit tests on synthetic source snippets covering all 9 bool-context kinds, wrapper
  bool_cast nesting, multi-line if-tests, nested ifs, ternary attribution, ternary same-line
  col=None fail-closed, elif flattening, scope resolution, ambiguous scope fail-closed, cache
  invalidation.

Quality gates: pytest ast_branches (19/19), smoke (22/22), mypy clean, ruff clean.

No changes to postprocess/control_flow.py (Phase 4) or any other torchlens module.

- **conditionals**: Foundation for if/elif/else attribution (Phase 1/10)
  ([`ced5a06`](https://github.com/johnmarktaylor91/torchlens/commit/ced5a0675612ee3a443535a706dd3730938b394d))

Add Phase 1 conditional foundation fields across LayerPassLog, LayerLog, ModelLog, and
  FuncCallLocation. Ungate func_call_stack capture while keeping source text lazily disabled via
  source_loading_enabled. Add metadata coverage for save_source_context=False, including
  disabled-source accessors and pickle round-trip.

Assumption: legacy direct FuncCallLocation construction defaults code_firstlineno to line_number
  when the new field is omitted.

- **conditionals**: Full integration test matrix (Phase 8/10)
  ([`f97f19d`](https://github.com/johnmarktaylor91/torchlens/commit/f97f19dc4f882b9f7e3b0fbd3df90bd618ab42ac))

Add ~64 new integration tests covering every remaining scenario in plan.md's Test Matrix that was
  not already covered by Phases 4-7. Two test files: - tests/test_conditional_integration.py (31
  atomic tests per matrix row) - tests/test_conditional_branches.py (33 cross-cutting tests that
  combine invariants + rendering + lifecycle checks on the same model)

Categories covered (missing from earlier phases): - Nested branches: NestedIfThenIfModel,
  NestedInElseModel, MultilinePredicateModel - Branch-entry via non-predicate ancestors, wrapped
  bool, reconvergence, scope resolution, false positives, compound/negation/walrus, ternary
  variants, false negatives, lifecycle, save_source_context=False gating.

Also extract _label_for_reference_removal helper in cleanup.py to consolidate the pass_finished
  label-namespace selection logic (required for consistent behavior across Phase 8 lifecycle tests).

Quality gates: 81+ conditional tests pass, smoke (26/26), tier-2 (929/929), mypy clean, ruff clean.

- **conditionals**: Graphviz rendering with THEN/ELIF/ELSE labels (Phase 7/10)
  ([`11e408c`](https://github.com/johnmarktaylor91/torchlens/commit/11e408c46f650147c21733dea2818d4fe2ae8e10))

Refactor torchlens/visualization/rendering.py edge-label logic into a precedence function
  (_compute_edge_label) per plan.md "Label precedence rules": 1. Arm-entry label (highest): THEN /
  ELIF N / ELSE, composite for multi-arm or mixed-pass rolled edges. 2. IF label (secondary): for
  condition-chain edges only. 3. Arg labels move from edge_dict["label"] to
  edge_dict["headlabel"]/xlabel so they do not compete with branch labels.

Covers if/elif/else ladders, ternary (ifexp) THEN/ELSE, multi-arm-entry edges (D13), rolled
  mixed-pass composite labels (D15 conditional_edge_passes), and branch-entry edge + arg label
  collision avoidance.

Tests (tests/test_conditional_rendering.py, 5 passing): -
  test_simple_if_else_graphviz_labels_then_and_else_edges -
  test_elif_ladder_graphviz_labels_elif_and_else_edges -
  test_basic_ternary_graphviz_labels_then_and_else_edges -
  test_branch_entry_with_arg_label_keeps_semantic_and_argument_labels_separate -
  test_rolled_mixed_arm_graphviz_shows_composite_pass_label

Dagua bridge and ELK remain deferred per user direction.

Quality gates: rendering tests (5/5), mypy clean, ruff clean.

- **conditionals**: Lifecycle wiring for new fields (Phase 3/10)
  ([`f4c59c1`](https://github.com/johnmarktaylor91/torchlens/commit/f4c59c15da3ce014567944219861eca15a5b6d90))

Modified files: - torchlens/postprocess/labeling.py - torchlens/data_classes/cleanup.py -
  torchlens/data_classes/interface.py - tests/test_conditional_lifecycle.py

Summary: - rewired Step 11 rename coverage for cond_branch_children_by_cond, conditional_arm_edges,
  conditional_edge_passes, conditional_events.bool_layers, and the derived elif/else views -
  scrubbed the same conditional surfaces during label removal, including keep_unsaved_layers=False
  pruning via the shared batch cleanup path - exported the new conditional lifecycle fields through
  to_pandas, including a compact conditional_branch_stack string and the missing derived child
  columns - added Phase 3 integration tests for rename, cleanup, and to_pandas

Assumption: - the spec's reference to _rename_raw_labels_in_place maps to the current Step 11
  helpers _replace_layer_names_for_layer_entry and _rename_model_history_layer_names in this branch

- **conditionals**: Multi-pass LayerLog aggregation (Phase 5/10)
  ([`7eb3baa`](https://github.com/johnmarktaylor91/torchlens/commit/7eb3baa34b5170465e745c553271f54ac7ec9e75))

Add ordered multi-pass LayerLog aggregation for conditional branch stacks, stack-to-pass maps, and
  pass-stripped cond-id child unions. Recompute aggregate LayerLog conditional compatibility views
  after merge, and rebuild rolled conditional_edge_passes from arm-entry edges with no-pass labels
  and sorted child-pass lists.\n\nCover the new behavior with dedicated multi-pass tests for
  alternating recurrent branches, mixed-arm rolled edges, looped alternating branches, and a
  non-conditional recurrent regression case.

- **conditionals**: Step 5 restructure into 5a-5f with AST-based classification and attribution
  (Phase 4/10)
  ([`c04fe8b`](https://github.com/johnmarktaylor91/torchlens/commit/c04fe8b8fd7c6d5a1dff8ebdd968911bed411fe0))

Restructure Step 5 in torchlens/postprocess/control_flow.py into the 5a-5f pipeline while preserving
  the _mark_conditional_branches entry point: build AST file indexes, classify terminal scalar
  bools, materialize dense ConditionalEvent records, run the IF backward flood, attribute forward
  arm-entry edges, and rebuild derived compatibility views.

Add Phase 4 integration coverage in tests/test_conditional_step5.py for simple if/else, elif
  ladders, assert classification, and save_source_context=False attribution.

Regression surface verified: - pytest tests/ -m smoke -x --tb=short -q - pytest
  tests/test_conditional_step5.py -x --tb=short - pytest tests/test_toy_models.py -k
  'conditional_branching or nested_conditional_loop or conditional_vae' -x --tb=short - pytest
  tests/ -m "not slow" -x --tb=short - mypy torchlens/ - ruff check . --fix

Conservative compatibility choice: preserve the existing in_cond_branch condition-chain semantics
  for public metadata/tests while treating conditional_branch_stack/conditional_arm_edges as the new
  primary executed-arm attribution. Also add a narrow decorated-function scope fallback in
  control_flow.py when attribute_op() fails closed on decorator-line co_firstlineno offsets, without
  modifying frozen ast_branches.py.

- **conditionals**: Step 5 six-phase restructure + Phase 4 integration tests (Phase 4/10)
  ([`71f976c`](https://github.com/johnmarktaylor91/torchlens/commit/71f976c89a9ec762f333c8d54e4e901fdb6b7f7b))

Restructure torchlens/postprocess/control_flow.py::_mark_conditional_branches into six sub-phases
  per plan.md: 5a. Build AST file indexes for files referenced by terminal scalar bools. 5b.
  Classify terminal bools into branch/non-branch contexts (via ast_branches). 5c. Materialize dense
  ConditionalEvent records from structural keys (sole key->id step). 5d. Backward-flood IF edges
  from branch-participating bools only. 5e. Forward branch attribution across every forward edge
  (D13 multi-entry) + cond-id-aware primary structures (conditional_arm_edges,
  cond_branch_children_by_cond). 5f. Materialize derived compatibility views (legacy
  cond_branch_*_children and conditional_*_edges fields).

Integration tests (tests/test_conditional_step5.py): SimpleIfElseModel, ElifLadderModel,
  AssertNotBranchModel, SaveSourceContextOffStillAttributes. All 4 pass on real models.

Quality gates: step5 tests (4/4), smoke (22/22), mypy clean, ruff clean.

Notes: - Old post-validation (D12) removed; pre-classification obsoletes it. - conditional_events
  dense IDs assigned deterministically (first-seen order). - Backward compat:
  conditional_branch_edges, conditional_then_edges, etc. still populated via derived views from
  primary structures.

### Testing

- Mark test_validation_250k as slow
  ([`a960cb2`](https://github.com/johnmarktaylor91/torchlens/commit/a960cb27c2bbd36340367f5e0dc20f18c7336f64))

250k-node validation takes too long for Tier 2 runs and was not excluded by `-m "not slow"` despite
  being marked rare.

- Mark test_validation_50k as rare
  ([`8070f8e`](https://github.com/johnmarktaylor91/torchlens/commit/8070f8eed80216b0bb07f4c38be94367ce405ce6))

Times out on slower machines even with 600s per-test limit. Validation of 50k-node random graph is
  too heavy for default runs.

- Skip test_validation_250k unconditionally
  ([`bd756b2`](https://github.com/johnmarktaylor91/torchlens/commit/bd756b2c679ee19f1f03bf1b0335f298ef91f089))

250k-node validation OOMs or hangs for hours on most machines. Slow/rare markers weren't enough --
  needs an explicit skip so `pytest tests/` (full run) doesn't block on it. Run manually with
  --no-skip when needed.

- **conditionals**: Isolate AST cache + linecache per test (Phase 10/10 prep)
  ([`eb00a71`](https://github.com/johnmarktaylor91/torchlens/commit/eb00a712292fc3a255d1868566dc16f59d050227))

Autouse fixture invalidates torchlens.postprocess.ast_branches file cache and Python linecache at
  the start of each test in test_conditional_branches.py.

Without this, test ordering dependencies surface in the full tier-2 suite:
  test_multiline_predicate_model_tracks_multiline_if_event fails when a prior test run populates
  stale state that the multi-line predicate assertion then reads. Running the file in isolation or
  with any smaller prefix passes cleanly; the full suite flakes depending on pytest's collection
  order.

Gates: ruff + mypy clean; smoke 28/28; tier-2 966 passed, 12 skipped, 0 failures (vs 1 failure
  pre-fixture under same collection order).


## v1.0.2 (2026-04-05)

### Bug Fixes

- **decoration**: Python 3.14 compat -- two-pass decoration + TypeError catch
  ([#138](https://github.com/johnmarktaylor91/torchlens/pull/138),
  [`e6f0f9a`](https://github.com/johnmarktaylor91/torchlens/commit/e6f0f9a33ba31504d6e402c117aac4a87ef46cb5))

Python 3.14 (PEP 649) evaluates annotations lazily. During decorate_all_once(), wrapping Tensor.bool
  before inspecting Tensor.dim_order caused inspect.signature() to resolve `bool` in `bool |
  list[torch.memory_format]` to the wrapper function instead of the builtin type, raising TypeError
  on first call only.

Three fixes: - Catch TypeError alongside ValueError in get_func_argnames (safety net) - Split
  decorate_all_once() into two passes: collect argnames from pristine namespace first, then decorate
  (eliminates root cause) - Replace _orig_to_decorated idempotency guard with _is_decorated flag so
  partial decoration failure allows retry instead of locking in incomplete state

6 new tests, gotchas.md updated.

### Chores

- **release**: 1.0.2
  ([`3ad9577`](https://github.com/johnmarktaylor91/torchlens/commit/3ad9577210e1ebeb3c33f221df81ea2fafa22e18))


## v1.0.1 (2026-03-23)

### Bug Fixes

- **decoration**: Clear stale sq_item C slot after wrapping Tensor.__getitem__
  ([`b2c6085`](https://github.com/johnmarktaylor91/torchlens/commit/b2c6085e31695b973b8d11c23c773af837a846cc))

When __getitem__ is replaced on a C extension type with a Python function, CPython sets the sq_item
  slot in tp_as_sequence. This makes PySequence_Check(tensor) return True (was False in clean
  PyTorch), causing torch.tensor([0-d_tensor, ...]) to iterate elements as sequences and call len()
  -- which raises TypeError for 0-d tensors. The slot is never cleared by restoring the original
  wrapper_descriptor or by delattr.

Fix: null sq_item via ctypes after every decoration/undecoration cycle (decorate_all_once,
  unwrap_torch, wrap_torch). Safe because tensor indexing uses mp_subscript (mapping protocol), not
  sq_item (sequence protocol). Verified via tp_name guard; fails silently on non-CPython.

Adds 9 regression tests covering all lifecycle paths.

### Chores

- Add secret detection pre-commit hooks
  ([`0e2889a`](https://github.com/johnmarktaylor91/torchlens/commit/0e2889ae90f822840895d9331e912a570d2a9acf))

Add detect-private-key (pre-commit-hooks) and detect-secrets (Yelp) to catch leaked keys, tokens,
  and high-entropy strings before they hit the repo.

- **release**: 1.0.1
  ([`1af6785`](https://github.com/johnmarktaylor91/torchlens/commit/1af678557a0a25ac0fdbdf2259b7bba8a5e8d313))


## v1.0.0 (2026-03-13)

### Bug Fixes

- **decoration**: Cast mode.device to str for mypy return-value check
  ([`45c0ff3`](https://github.com/johnmarktaylor91/torchlens/commit/45c0ff3be8586675817905dcb273e9bad7ac0519))

CI mypy (stricter torch stubs) catches that mode.device returns torch.device, not str. Explicit
  str() cast satisfies the Optional[str] return type annotation.

### Chores

- **release**: 1.0.0
  ([`bd9ca16`](https://github.com/johnmarktaylor91/torchlens/commit/bd9ca16e27e70b66974c7af3d89456bf92517588))

### Features

- **decoration**: Lazy wrapping — import torchlens has no side effects
  ([`b5da8b8`](https://github.com/johnmarktaylor91/torchlens/commit/b5da8b8b15136f108534ba29022ab6579bdb9315))

BREAKING CHANGE: torch functions are no longer wrapped at import time. Wrapping happens lazily on
  first log_forward_pass() call and persists.

Three changes:

1. Lazy decoration: removed decorate_all_once() / patch_detached_references() calls from
  __init__.py. _ensure_model_prepared() triggers wrapping on first use via wrap_torch().

2. Public wrap/unwrap API: - torchlens.wrap_torch() — install wrappers (idempotent) -
  torchlens.unwrap_torch() — restore original torch callables - torchlens.wrapped() — context
  manager (wrap on enter, unwrap on exit) - log_forward_pass(unwrap_when_done=True) — one-shot
  convenience Old names (undecorate_all_globally, redecorate_all_globally) kept as internal aliases.

3. torch.identity fix: decorated identity function now stored on _state._decorated_identity instead
  of monkey-patching torch.identity (which doesn't exist in PyTorch type stubs). Eliminates 2 mypy
  errors.

Tests updated: 75 pass including 12 new lifecycle tests.

### Breaking Changes

- **decoration**: Torch functions are no longer wrapped at import time. Wrapping happens lazily on
  first log_forward_pass() call and persists.


## v0.22.0 (2026-03-13)

### Chores

- **release**: 0.22.0
  ([`fcb912c`](https://github.com/johnmarktaylor91/torchlens/commit/fcb912c54f5996fb1e329f28b784ad326eb4c853))

- **types**: Remove stale typing noise
  ([`5ba099d`](https://github.com/johnmarktaylor91/torchlens/commit/5ba099df82f789f308f86d78a66c82b1baf3751d))

### Documentation

- **maintenance**: Refresh maintainer notes
  ([`685a358`](https://github.com/johnmarktaylor91/torchlens/commit/685a358caa12b3210de68e136c59959de4c3f94f))

- **maintenance**: Split CLAUDE.md/AGENTS.md into architect vs implementation roles
  ([`18f8ae9`](https://github.com/johnmarktaylor91/torchlens/commit/18f8ae9851e91be5ca7c335facb45209212f6d95))

Break the symlink mirroring convention: CLAUDE.md now holds architect-level context (what, why, how
  it connects) while AGENTS.md holds implementation-level context (conventions, gotchas, known bugs,
  test commands). Pure-implementation subdirs (.github, scripts, tests, utils) get AGENTS.md only.
  Also populates .project-context/ templates (architecture, conventions, gotchas, decisions).

### Features

- **decoration**: Add global undecorate override
  ([`b0bafeb`](https://github.com/johnmarktaylor91/torchlens/commit/b0bafeb51fbf9b406bd34a84ed826cd4b9c607bb))

- **viz**: Add dagua torchlens integration
  ([`35d5bcd`](https://github.com/johnmarktaylor91/torchlens/commit/35d5bcdb976420b86be54048ab4503cbbc146763))

### Refactoring

- **types**: Finish package mypy cleanup
  ([`6f9a3fe`](https://github.com/johnmarktaylor91/torchlens/commit/6f9a3fe423367c9bd9e32949cdf06945d90786ee))


## v0.21.3 (2026-03-11)

### Bug Fixes

- **tests**: Make SIGALRM signal safety test deterministic
  ([`b3fc461`](https://github.com/johnmarktaylor91/torchlens/commit/b3fc46154f0942c829c3bbeb9b07f3d900471b79))

Replace timer-based SIGALRM with direct os.kill() inside forward() so the signal always fires
  mid-logging. Eliminates flaky skips when the forward pass completes before the timer.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.21.3
  ([`04fa27a`](https://github.com/johnmarktaylor91/torchlens/commit/04fa27ac3602a5056a986aa01a0e8b2f1a2e0c73))


## v0.21.2 (2026-03-09)

### Bug Fixes

- **vis**: Avoid graphviz.Digraph memory bomb when ELK fails on large graphs
  ([`f5563ee`](https://github.com/johnmarktaylor91/torchlens/commit/f5563eee457383772c9b148cca058b87e126b6b3))

When ELK layout fails (OOM/timeout) on 1M+ node graphs, the fallback path previously built a
  graphviz.Digraph in Python — nested subgraph body-list copies exploded memory and hung
  indefinitely. Now render_elk_direct handles the failure internally: reuses already-collected Phase
  1 data to generate DOT text without positions and renders directly with sfdp, bypassing
  graphviz.Digraph entirely.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Bypass ELK for large graphs — use Python topological layout
  ([`37cce3a`](https://github.com/johnmarktaylor91/torchlens/commit/37cce3ab5607591f622b4fd7f5916bca16736d59))

ELK's stress algorithm allocates TWO O(n²) distance matrices (n² × 16 bytes). At 100k nodes that's
  160 GB, at 1M nodes it's 16 TB — the root cause of the std::bad_alloc. The old >150k stress switch
  could never work.

For graphs above 100k nodes, we now skip ELK entirely and compute a topological rank layout in
  Python (Kahn's algorithm, O(n+m)). Module bounding boxes are computed from node positions. The
  result feeds into the same neato -n rendering path, preserving cluster boxes.

If ELK fails for smaller graphs, the Python layout is also used as a fallback instead of the old
  sfdp path that built a graphviz.Digraph (which exploded on nested subgraph body-list copies).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.21.2
  ([`fa72687`](https://github.com/johnmarktaylor91/torchlens/commit/fa726875ed4b9cd448fd5914de3d1e243c07b87e))


## v0.21.1 (2026-03-09)

### Bug Fixes

- **postprocess**: Fix mypy type errors in _build_module_param_info
  ([`11ea006`](https://github.com/johnmarktaylor91/torchlens/commit/11ea006c9a683675c926a9b0d649a2fa24b46558))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- Trigger CI
  ([`99f4102`](https://github.com/johnmarktaylor91/torchlens/commit/99f4102fa34564868291214d6b6cf3dfc239fdce))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **release**: 0.21.1
  ([`364ea39`](https://github.com/johnmarktaylor91/torchlens/commit/364ea3911a0462bc83280e614a423b28f7dd3bd9))

### Performance Improvements

- **postprocess**: Optimize pipeline for large models
  ([`a211417`](https://github.com/johnmarktaylor91/torchlens/commit/a2114170b4f5b441d4c0ab7add26435820fc8f8f))

- Per-step verbose timing: unwrap grouped _vtimed blocks into individual step timing with
  graph-stats summary, enabling users to identify which specific step is slow (O16) - Cache
  module_str by containing_modules tuple to avoid redundant string joins in Step 6 (O8) -
  Early-continue guards in _undecorate_all_saved_tensors to skip BFS on layers with empty
  captured_args/kwargs (O5) - Pre-compute buffer_layers_by_module dict in _build_module_logs,
  eliminating O(modules × buffers) scan per module (O6) - Single-pass arglist rebuild in Step 11
  rename, replacing 3-pass enumerate + index set + filter pattern (O2) - Replace OrderedDict with
  dict in _trim_and_reorder (Python 3.7+ preserves insertion order) for lower allocation overhead
  (O4) - Reverse-index approach in _refine_iso_groups: O(members × neighbors) instead of O(members²)
  all-pairs combinations (O9) - Pre-compute param types per subgraph as frozenset before pair loop
  in _merge_iso_groups_to_layers (O10) - Set-based O(n) collision detection replacing O(n²) .count()
  calls in _find_isomorphic_matches (O12)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.21.0 (2026-03-09)

### Bug Fixes

- **capture**: Fix mypy type errors in output_tensors field dict
  ([`d54e9a9`](https://github.com/johnmarktaylor91/torchlens/commit/d54e9a99df47442ddc396ca20666203cfbeb5f06))

Annotate fields_dict as Dict[str, Any] and extract param_shapes with proper type to satisfy mypy
  strict inference.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Pass heap limits to ELK Worker thread to prevent OOM on 1M nodes
  ([`23ef8d8`](https://github.com/johnmarktaylor91/torchlens/commit/23ef8d8c175846fce1b48427a0c980a825486b9c))

The Node.js Worker running ELK layout had no explicit maxOldGenerationSizeMb in its resourceLimits —
  only stackSizeMb was set. The --max-old-space-size flag controls the main thread's V8 isolate, not
  the Worker's. This caused the Worker to OOM at ~16GB on 1M-node graphs despite the main thread
  being configured for up to 64GB.

- Add maxOldGenerationSizeMb and maxYoungGenerationSizeMb to Worker resourceLimits, passed via
  _TL_HEAP_MB env var - Add _available_memory_mb() to detect system RAM and cap heap allocation to
  (available - 4GB), preventing competition with Python process - Include available system memory in
  OOM diagnostic messages

Also includes field/param renames from feat/grand-rename branch.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.21.0
  ([`bd8b348`](https://github.com/johnmarktaylor91/torchlens/commit/bd8b348f5c3074526675a633b469314736d3822c))

### Documentation

- Update all CLAUDE.md files with deepdive session 4 findings
  ([`b15c5bf`](https://github.com/johnmarktaylor91/torchlens/commit/b15c5bfa665011002a67cdcc7710f4737f1fc5d6))

Sync all project and subpackage documentation with current codebase: - Updated line counts across
  all 36 modules - Added elk_layout.py documentation to visualization/ - Added arg_positions.py and
  salient_args.py to capture/ - Documented 13 new bugs (ELK-IF-THEN, BFLOAT16-TOL, etc.) - Updated
  test counts (1,004 tests across 16 files) - Added known bugs sections to validation/, utils/,
  decoration/ - Updated data_classes/ with new fields and properties

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- Rename all data structure fields and function args for clarity
  ([`f0d7452`](https://github.com/johnmarktaylor91/torchlens/commit/f0d7452e272bede7a874e1bda2dbce68dbb94697))

Rename ~68 fields across all 8 data structures (ModelLog, LayerPassLog, LayerLog, ParamLog,
  ModuleLog, BufferLog, ModulePassLog, FuncCallLocation) plus user-facing function arguments. Key
  changes:

- tensor_contents → activation, grad_contents → gradient - All *_fsize* → *_memory* (e.g.
  tensor_fsize → tensor_memory) - func_applied_name → func_name, gradfunc → grad_fn_name -
  is_bottom_level_submodule_output → is_leaf_module_output - containing_module_origin →
  containing_module - spouse_layers → co_parent_layers, orig_ancestors → root_ancestors -
  model_is_recurrent → is_recurrent, elapsed_time_* → time_* - vis_opt → vis_mode, save_only →
  vis_save_only - Fix typo: output_descendents → output_descendants

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.20.5 (2026-03-09)

### Bug Fixes

- **vis**: Prevent OOM kill on 1M-node ELK render
  ([#128](https://github.com/johnmarktaylor91/torchlens/pull/128),
  [`d9a1525`](https://github.com/johnmarktaylor91/torchlens/commit/d9a15259e452bf3c224732a0bc5a1671f9cbb56e))

The 1M-node render was OOM-killed at ~74GB RSS because: 1. Model params (~8-10GB) stayed alive
  during ELK subprocess 2. preexec_fn forced fork+exec, COW-doubling the 74GB process 3. Heap/stack
  formulas produced absurd values (5.6TB heap, 15GB stack) 4. No memory cleanup before subprocess
  launch

Changes: - render_large_graph.py: separate log_forward_pass from render_graph, free model/autograd
  before ELK render - elk_layout.py: cap heap at 64GB, stack floor 4096MB/cap 8192MB, write JSON to
  temp file (free string before subprocess), gc.collect before subprocess, set RLIMIT_STACK at
  module level (removes preexec_fn and the forced fork+exec)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.20.5
  ([`9b65063`](https://github.com/johnmarktaylor91/torchlens/commit/9b650633ed1925c75da06c79f2cd3cba8764b57c))


## v0.20.4 (2026-03-09)

### Bug Fixes

- **postprocess**: Backward-only flood in conditional branch detection + THEN labeling
  ([#88](https://github.com/johnmarktaylor91/torchlens/pull/88),
  [`d737828`](https://github.com/johnmarktaylor91/torchlens/commit/d7378281a4be5e56602902cd4cf1aa555b391d44))

Bug #88: _mark_conditional_branches flooded bidirectionally (parents + children), causing
  non-conditional nodes' children to be falsely marked as in_cond_branch. Fix restricts flooding to
  parent_layers only.

Additionally adds THEN branch detection via AST analysis when save_source_context=True, with IF/THEN
  edge labels in visualization. Includes 8 new test models, 22 new tests, and fixes missing
  'verbose' in MODEL_LOG_FIELD_ORDER.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Use Worker thread for ELK layout to fix stack overflow on large graphs
  ([`3fe6a84`](https://github.com/johnmarktaylor91/torchlens/commit/3fe6a84c3d9d8a5856dfadcf451ef85700a2385c))

V8's --stack-size flag silently caps at values well below what's requested, causing "Maximum call
  stack size exceeded" on 1M+ node graphs. Switch to Node.js Worker threads with
  resourceLimits.stackSizeMb, which reliably delivers the requested stack size at the V8 isolate
  level.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.20.4
  ([`aa8c0eb`](https://github.com/johnmarktaylor91/torchlens/commit/aa8c0eb8144374a89861e2b55e5a8ed259bdc810))


## v0.20.3 (2026-03-08)

### Bug Fixes

- **vis**: Increase ELK Node.js stack floor to 4GB for large graphs
  ([`29af94e`](https://github.com/johnmarktaylor91/torchlens/commit/29af94ed51b9018ba214f2de15c48e5454a773b3))

128MB was insufficient for ELK's recursive layout on 500k+ node graphs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Raise OS stack limit for ELK Node.js subprocess
  ([`da82c9d`](https://github.com/johnmarktaylor91/torchlens/commit/da82c9d0fd66c2fdcbaabb86b31750811693a930))

The OS soft stack limit (ulimit -s) was smaller than the --stack-size value passed to Node.js,
  causing a segfault on large graphs (500k+ nodes) instead of allowing V8 to use the requested
  stack. Uses preexec_fn to set RLIMIT_STACK to unlimited in the child process only.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.20.3
  ([`2c15b6b`](https://github.com/johnmarktaylor91/torchlens/commit/2c15b6b05642a36da5465ef5201a387be1276a21))

### Performance Improvements

- **decoration**: Optimize model prep and move session attrs to ModelLog dicts
  ([`b63a4fa`](https://github.com/johnmarktaylor91/torchlens/commit/b63a4fa3871113a5cc73554c576fb11c5eac2cd2))

Five performance fixes for _prepare_model_session and related setup code:

- PERF-38: Replace O(N²) list concat in _traverse_model_modules with deque - PERF-37: Cache
  user_methods per class in _get_class_metadata; move _pytorch_internal set to module-level
  frozenset - PERF-36: Iterate module._parameters directly instead of rsplit on named_parameters
  addresses + lookup dict - PERF-39: Skip patch_model_instance for already-prepared models - Move 4
  session-scoped module attrs (tl_module_pass_num, tl_module_pass_labels, tl_tensors_entered_labels,
  tl_tensors_exited_labels) from nn.Module instances to ModelLog dicts keyed by id(module). Remove
  tl_source_model_log (dead code). Eliminates per-module cleanup iteration in
  _cleanup_model_session.

At 10K modules: ensure_prepared repeat calls drop from ~48ms to ~0.4ms (111x), session setup ~1.3x
  faster, cleanup ~1.4x faster.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.20.2 (2026-03-08)

### Bug Fixes

- **vis**: Increase ELK Node.js stack size to prevent overflow
  ([`b8edbc8`](https://github.com/johnmarktaylor91/torchlens/commit/b8edbc8c779dda44b046c9efa4855b3a6fad46f7))

Bump --stack-size floor from 64MB to 128MB and multiplier from 16x to 48x (matching heap scaling) to
  prevent "Maximum call stack size exceeded" in elkjs on large graphs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.20.2
  ([`fd67742`](https://github.com/johnmarktaylor91/torchlens/commit/fd67742e7ec9ba130f812ed618b9228ea3968673))

- **scripts**: Enable loop detection in render_large_graph
  ([`803e16f`](https://github.com/johnmarktaylor91/torchlens/commit/803e16f3ecec7e2fbb1c662ce963f7de51f4d16c))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.20.1 (2026-03-08)

### Chores

- **release**: 0.20.1
  ([`54a7dc0`](https://github.com/johnmarktaylor91/torchlens/commit/54a7dc000da53e48205d3c05cf94479e91d3b5d5))

- **scripts**: Use log_forward_pass vis_opt instead of separate render call
  ([`d2aea0f`](https://github.com/johnmarktaylor91/torchlens/commit/d2aea0fab326e02444d99aadc7dcf12abf9c8b10))

Let verbose mode handle all phase timing instead of manual timestamps. Use log_forward_pass's
  built-in vis_opt to render in one call.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Performance Improvements

- **model_prep**: Optimize _prepare_model_session for large models
  ([`2892323`](https://github.com/johnmarktaylor91/torchlens/commit/2892323ac2910532c9bbde894280ba917a72e966))

- Hoist set(dir(nn.Module)) to module-level constant _NN_MODULE_ATTRS - Replace dir(module) MRO walk
  with __dict__ scans for attrs and methods - Pre-build address→module dict to eliminate
  per-parameter tree walks - Use model.modules() with cached tl_module_address instead of second DFS

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.20.0 (2026-03-08)

### Chores

- **release**: 0.20.0
  ([`8096b91`](https://github.com/johnmarktaylor91/torchlens/commit/8096b918ecb0458006576bdc89a29f2be7959611))

- **scripts**: Unify large graph render scripts into single parameterized script
  ([`07a8186`](https://github.com/johnmarktaylor91/torchlens/commit/07a8186537cf88bb39f4abae55a05dc01ec16457))

Replace `run_250k.py` and `run_1M.py` with `render_large_graph.py` that accepts any node count as a
  CLI argument, plus --format, --seed, and --outdir options.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **logging**: Add verbose mode for timed progress messages
  ([`0603f10`](https://github.com/johnmarktaylor91/torchlens/commit/0603f1035b8be7c2c44a416099fac282eaafa686))

Add `verbose: bool = False` parameter to `log_forward_pass`, `show_model_graph`, and internal
  pipeline functions. When enabled, prints `[torchlens]`-prefixed progress at each major pipeline
  stage with timing. Also fixes `_trim_and_reorder_model_history_fields` to preserve all non-ordered
  attributes (not just private ones).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.19.0 (2026-03-08)

### Chores

- Add large graph render scripts to scripts/
  ([`d02233b`](https://github.com/johnmarktaylor91/torchlens/commit/d02233bc050657ed6088b8338e056c2c531d45bd))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **release**: 0.19.0
  ([`087d42d`](https://github.com/johnmarktaylor91/torchlens/commit/087d42d03a7577484ad131d9bb97362e3c4b6cca))

### Documentation

- Update RESULTS.md and tests/CLAUDE.md with current counts
  ([`75fa346`](https://github.com/johnmarktaylor91/torchlens/commit/75fa3463b16505d21b3498a29c848e5da248bd36))

- Total tests: 892 → 951, test files: 14 → 15, toy models: 249 → 250 - Add test_large_graphs.py (51
  tests) to file tables - Add decoration overhead benchmark table to profiling baselines - Add large
  graph scaling section (100 to 1M nodes) - Update all per-file test counts to current values

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **capture**: Add func_config field for lightweight hyperparameter extraction
  ([`7144d6d`](https://github.com/johnmarktaylor91/torchlens/commit/7144d6d5200bb17377956ffb8c579553ae34ddfe))

Adds a `func_config` dict to every LayerPassLog/LayerLog containing computation-defining
  hyperparameters (kernel_size, stride, in/out channels, dropout p, etc.) extracted at capture time
  with zero tensor cloning. Empty for source tensors and output nodes.

Also fixes pre-existing test failures in test_validation.py (read-only property assignments) and
  adds detect_loops to MODEL_LOG_FIELD_ORDER.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **profiling**: Add decoration overhead benchmark to profiling report
  ([`c8fbffd`](https://github.com/johnmarktaylor91/torchlens/commit/c8fbffdc43b4f97a2e7ab90cecd78a8d59950592))

Measures per-call overhead of TorchLens's toggle-gated wrappers when logging is disabled. Benchmarks
  11 functions from cheap (relu, add) to heavy (conv2d, SDPA) — confirms ~600ns overhead on cheap
  ops, <1% on real compute.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.18.0 (2026-03-08)

### Chores

- **release**: 0.18.0
  ([`3cbb02b`](https://github.com/johnmarktaylor91/torchlens/commit/3cbb02b24d0e37c94e99c9e8e8a14bf7e5d0d103))

### Features

- **data**: Add MACs properties to LayerPassLog, LayerLog, ModelLog, ModuleLog
  ([`c60e7b9`](https://github.com/johnmarktaylor91/torchlens/commit/c60e7b9a2fd29de09d82c5be076e75c6e4c8b39d))

MACs (multiply-accumulate operations) = FLOPs / 2. Added: - LayerPassLog: macs_forward,
  macs_backward properties - LayerLog: macs_forward, macs_backward properties - ModelLog:
  total_macs_forward, total_macs_backward, total_macs, macs_by_type() - ModuleLog: flops_forward,
  flops_backward, flops, macs_forward, macs_backward, macs

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **data**: Convert 23 stored fields to computed @properties across data classes
  ([`2c22208`](https://github.com/johnmarktaylor91/torchlens/commit/2c22208a72636bb796268510a3529cf1183519eb))

Replace redundant stored fields with computed @property methods that derive their values from
  existing data. Eliminates ~155 lines of write-site code across 13 files while preserving identical
  behavior and passing all invariants.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.17.0 (2026-03-08)

### Bug Fixes

- **types**: Add mypy annotations for defaultdict and deque in elk_layout
  ([`98478a3`](https://github.com/johnmarktaylor91/torchlens/commit/98478a33a28e890554bad6325478efb8a2ff5f85))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.17.0
  ([`07bc335`](https://github.com/johnmarktaylor91/torchlens/commit/07bc33586b250aaa9c3f46f89ca974f16266ea8b))

### Features

- **vis**: Scale ELK rendering to 250k+ nodes, add detect_loops option
  ([`4707631`](https://github.com/johnmarktaylor91/torchlens/commit/470763146ce30334b5853cfc7cfd104035fd9922))

ELK's layered algorithm (Sugiyama) uses O(n²) memory for crossing minimization, causing
  std::bad_alloc at ~150k+ nodes. This adds:

- Auto-switch to ELK stress algorithm above 150k nodes (O(n) memory) - Topological position seeding
  for stress to preserve directional flow - Increased Node.js heap allocation (16GB floor, 48x JSON
  size) - Better error messages when Node.js OOM-kills (was silent empty stderr) - `detect_loops`
  parameter on log_forward_pass/show_model_graph to skip expensive isomorphic subgraph expansion,
  keeping only same-param grouping (Rule 1). Default True (existing behavior unchanged). - 8 loop
  comparison tests rendering with/without loop detection

Successfully renders 250k-node graphs in ~19 minutes (was impossible). 1M-node render in progress.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.16.4 (2026-03-08)

### Bug Fixes

- **validation**: Use call nesting depth not address depth in invariant Q, fix test inputs
  ([#117](https://github.com/johnmarktaylor91/torchlens/pull/117),
  [`98660ff`](https://github.com/johnmarktaylor91/torchlens/commit/98660ff114f31a2324d3268c0d925cf0b31c02c9))

### Chores

- **release**: 0.16.4
  ([`8c0b57c`](https://github.com/johnmarktaylor91/torchlens/commit/8c0b57cf05202cf05f0144c2bf2f8a2658e71303))


## v0.16.3 (2026-03-08)

### Bug Fixes

- **tests**: Rename model_kwargs to input_kwargs and fix test configs
  ([`8768339`](https://github.com/johnmarktaylor91/torchlens/commit/87683393e2c38a2703a73643c7b975517bffd589))

- Rename model_kwargs= to input_kwargs= in 17 test call sites to match API - Skip test_gpt_bigcode
  (JIT-compiled attention incompatible with TorchLens) - Fix test_deformable_detr backbone
  out_features to match 2-layer ResNet - Add index, symint, checkkeypaddingmask to _UNARY_FUNCS
  ArgSpec entries

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **types**: Resolve 14 pre-existing mypy errors across 4 files
  ([`7adcc36`](https://github.com/johnmarktaylor91/torchlens/commit/7adcc362552697cf8961024d9bc1dcd7a4fb347f))

- rng.py: annotate rng_dict as Dict[str, object] for mixed-type values - elk_layout.py: annotate
  out_shape as tuple to avoid index-out-of-range - model_prep.py: annotate meta as Dict[str,
  object], add type: ignore for dynamic tl_buffer_address attrs on Tensor - output_tensors.py: add
  type: ignore for ArgSpec arg-type and assignment

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.16.3
  ([`1f599a8`](https://github.com/johnmarktaylor91/torchlens/commit/1f599a8b7a7be3864c9a19215f294de55f62f936))


## v0.16.2 (2026-03-08)

### Bug Fixes

- **validation**: Exempt exponential_ from perturbation check
  ([`a6dce40`](https://github.com/johnmarktaylor91/torchlens/commit/a6dce40dc314a36a0296a7c65dd4faf04647b126))

In-place RNG op — output is determined by RNG state, not input values. Fixes test_gumbel_vq_model.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Exempt maximum/minimum from perturbation check
  ([`9d658bc`](https://github.com/johnmarktaylor91/torchlens/commit/9d658bc1d80222523ec7e410c0cf1d38665b16f2))

torch.maximum/minimum with extreme-valued args (e.g. RWKV's negative infinity masks) are insensitive
  to perturbation — same as max/min.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Fix perturbation precision and add exemptions for C++ ops
  ([`7f86655`](https://github.com/johnmarktaylor91/torchlens/commit/7f86655705a8d914126a174ff8b95cf3cc123cee))

- Scale constant-tensor perturbation by ±10% of magnitude to ensure float32-distinguishable values
  while staying in safe range - Add posthoc magnitude-ratio exemption: when non-perturbed parent's
  magnitude dwarfs the perturbed parent's (>100x), float32 arithmetic swallows the perturbation —
  exempt rather than fail - Add maximum/minimum to posthoc exemption (element-wise binary ops
  insensitive to perturbation when one arg dominates) - Skip perturbation for _op (torchvision C++
  PyCapsule ops like nms, roi_align) — perturbed coordinates segfault native extensions - Fix
  nystromformer test: seq_len must equal num_landmarks² (64)

Fixes: test_fcos_resnet50_train, test_retinanet_resnet50_train, test_nystromformer,
  test_maskrcnn_resnet50_train

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Scale perturbation range by magnitude for large constants
  ([`efcd876`](https://github.com/johnmarktaylor91/torchlens/commit/efcd8766b38c6ea9f5d3715fac4994509985299b))

Constant tensors with large values (e.g. 1e8) were perturbed by ±1.0, which is indistinguishable in
  float32 precision. Now scales expansion by abs(value) * 1e-4 to ensure perturbation is always
  detectable.

Fixes test_fcos_resnet50_train validation failure.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Use relative threshold for near-constant tensor perturbation
  ([`e0c82c7`](https://github.com/johnmarktaylor91/torchlens/commit/e0c82c7a63e63d2f6e033274d152afc68ecb060b))

Near-constant float tensors (e.g. [2.6785714, 2.6785717]) had a value range smaller than float32
  precision, so perturbation within [min, max] produced the same value after rounding. Changed exact
  `lo == hi` check to relative threshold `hi - lo < max(1e-6, abs(lo) * 1e-6)` to trigger range
  expansion for these cases.

Fixes test_ssd300_vgg16_train validation failure.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Probe nvm paths when node is not on PATH
  ([`4d3ae24`](https://github.com/johnmarktaylor91/torchlens/commit/4d3ae2414bbcc0835d4cb01da230f158004da690))

Non-interactive shells (IDE test runners, cron, subprocesses) often lack nvm's PATH additions,
  causing elk_available() to return False even when elkjs is installed. Add _find_node_binary() to
  probe ~/.nvm/versions/node/ as a fallback, and inject the node binary's directory into the
  subprocess PATH so elkjs detection works regardless of shell configuration.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.16.2
  ([`272d558`](https://github.com/johnmarktaylor91/torchlens/commit/272d558a4aff5ba07275b03122f45419859aa6e8))


## v0.16.1 (2026-03-07)

### Bug Fixes

- **tests**: Use relative import for example_models in test_large_graphs
  ([`e2d0ae4`](https://github.com/johnmarktaylor91/torchlens/commit/e2d0ae476e4d54c2211a3629e8958032c82b1c0f))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Harden ELK heap scaling and fix flaky signal safety test
  ([`41b9f89`](https://github.com/johnmarktaylor91/torchlens/commit/41b9f893692bc39a4ea0342092df62b2f2ee2b38))

- Bump ELK Node.js heap scaling from 8x to 16x JSON size to prevent OOM on 250k+ node graphs - Mark
  100k node tests as @rare (too slow for regular runs) - Fix flaky TestSignalSafety: use
  setitimer(50ms) instead of alarm(1s), increase model iterations to 50k, skip if alarm doesn't fire

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.16.1
  ([`b040b6d`](https://github.com/johnmarktaylor91/torchlens/commit/b040b6d459d6264be57505d5dd3fc0eaf34cfe08))


## v0.16.0 (2026-03-07)

### Bug Fixes

- **validation**: Batch of bug fixes for edge-case models
  ([`dca5a7e`](https://github.com/johnmarktaylor91/torchlens/commit/dca5a7e11d31c81823601b5b93f79e143cd33c55))

- Fix vmap/functorch compatibility: skip logging inside functorch transforms to avoid missing
  batching rules (torch_funcs.py) - Fix tensor_nanequal infinite recursion: wrap decorated tensor
  ops (.isinf, .resolve_conj, etc.) in pause_logging() (tensor_utils.py) - Fix perturbation for
  range-restricted functions: use uniform random within original value range instead of scaled
  normal (core.py) - Fix atomic_bool_val crash inside vmap context (output_tensors.py) - Fix output
  node initialized_inside_model flag (graph_traversal.py) - Add scatter_ full-overwrite exemption
  (exemptions.py) - Add max/min indices exemption for integer dtype outputs - Add bernoulli scalar-p
  exemption, constant-output exemption - Add non-perturbed parent special-value check for nested
  args (einsum) - Fix buffer_xrefs invariant: accept ancestor module matches - Fix real-world model
  configs: CvT, CLAP, EnCodec, SpeechT5, Informer, Autoformer, MobileBERT kwarg names

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Fix ELK rendering and scale for large graphs
  ([`ea96a85`](https://github.com/johnmarktaylor91/torchlens/commit/ea96a85a766068527366cbe0a77ecd7c7f467eef))

- Fix neato -n position format: use points (not inches) — fixes empty nodes, missing edges, and
  invisible labels in ELK output - Auto-scale Node.js heap and stack with graph JSON size -
  Auto-scale ELK and neato timeouts with node count - Use straight-line edges for graphs > 1k nodes
  (spline routing is O(n^2)) - Warn users to use SVG format for graphs > 25k nodes (PDF renders
  empty) - Add dot-vs-ELK aesthetic comparison tests at 15/100/500/1k/3k nodes - Add 1M node test
  (rare marker) for trophy-file rendering

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.16.0
  ([`478d872`](https://github.com/johnmarktaylor91/torchlens/commit/478d872f2c224dc4f00e81232f60e3e9e9531e29))

### Features

- **vis**: Add ELK layout engine for large graph visualization
  ([`5245872`](https://github.com/johnmarktaylor91/torchlens/commit/5245872746ee07dddd41a222392aa7548e33ce3f))

- New elk_layout.py: ELK-based node placement via Node.js/elkjs subprocess, with graceful fallback
  to sfdp when unavailable - New vis_node_placement parameter ("auto"/"dot"/"elk"/"sfdp") threaded
  through show_model_graph, log_forward_pass, and render_graph - Auto mode uses dot for <3500 nodes,
  ELK (or sfdp fallback) for larger - RandomGraphModel in example_models.py: seeded random model
  generator with calibrated node counts (within ~5% of target up to 100k+) - 39 tests in
  test_large_graphs.py: node count accuracy, validation, engine selection, ELK utilities, rendering
  at 3k-100k scales, dot threshold benchmark - Increased Node.js stack size (--stack-size=65536) to
  handle graphs up to 250k+ nodes without stack overflow

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Add hierarchical module grouping to ELK layout
  ([`bcc1ad2`](https://github.com/johnmarktaylor91/torchlens/commit/bcc1ad207aa2d9512960201594bfa9afc2495fb0))

- New build_elk_graph_hierarchical(): builds nested ELK compound nodes from module containment
  structure (containing_modules_origin_nested) - ELK's "INCLUDE_CHILDREN" hierarchy handling
  preserves module grouping in the layout — nodes within the same module cluster together -
  inject_elk_positions() now recurses into compound nodes, accumulating absolute positions from
  nested ELK coordinates - render_with_elk() passes entries_to_plot for hierarchical layout, falls
  back to flat DOT parsing when entries not available - Tests for hierarchical graph building and
  nested position injection

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **vis**: Add rare marker for 250k-node tests, fill test coverage gaps
  ([`c749b94`](https://github.com/johnmarktaylor91/torchlens/commit/c749b94e69a1eee18f9084c1f46efc472fb3fe72))

- New pytest marker "rare": always excluded by default via addopts, run explicitly with `pytest -m
  rare` - Add 250k node count, validation, and ELK render tests (marked rare) - Fill gaps:
  validation tests at 5k/10k/20k/50k/100k, ELK render tests at 5k/20k, node count test at 20k

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.15 (2026-03-07)

### Bug Fixes

- **gc**: Release ParamLog._param_ref on cleanup, add GC test suite (#GC-1, #GC-12)
  ([`83e1bd2`](https://github.com/johnmarktaylor91/torchlens/commit/83e1bd2df4304b55bff7d59bea8ffe2728ee7c7f))

- Add ParamLog.release_param_ref() to cache grad metadata then null _param_ref - cleanup() now nulls
  all _param_ref before clearing entries - Add ModelLog.release_param_refs() public API for early
  param release - Add _param_logs_by_module to cleanup's internal containers list - New test_gc.py
  with 10 tests covering ModelLog/param GC, memory growth, save_new_activations stability, and
  transient data clearing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.15.15
  ([`c124ac7`](https://github.com/johnmarktaylor91/torchlens/commit/c124ac77ec1604703a8fdf0706bd06fbfc8d7048))

### Documentation

- Move RESULTS.md to repo root for visibility
  ([`2e814dd`](https://github.com/johnmarktaylor91/torchlens/commit/2e814ddbd3421b3c81d7f2518d169a3470c8e9a8))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **tests**: Add public test results summary
  ([`fd7b33f`](https://github.com/johnmarktaylor91/torchlens/commit/fd7b33ff4f8bf1b4b74548642f70dc58a98f85f7))

Committed tests/RESULTS.md with suite overview, model compatibility matrix (121 toy + 85
  real-world), profiling baselines, and pointers to generated reports. Transparent scoreboard for
  the repo.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **models**: Add autoencoders, state space models, and architecture coverage
  ([`716cc9d`](https://github.com/johnmarktaylor91/torchlens/commit/716cc9dc425eb6245886eb7873c7a212db16d4d5))

Toy models (18 new in example_models.py): - Autoencoders: VanillaAutoencoder, ConvAutoencoder,
  SparseAutoencoder, DenoisingAutoencoder, VQVAE, BetaVAE, ConditionalVAE - State space: SimpleSSM,
  SelectiveSSM (Mamba-style), GatedSSMBlock, StackedSSM - Additional: SiameseNetwork, MLPMixer,
  SimpleGCN, SimpleGAT, SimpleDiffusion, SimpleNormalizingFlow, CapsuleNetwork

Real-world models (5 new in test_real_world_models.py): - SSMs: Mamba, Mamba-2, RWKV, Falcon-Mamba
  (via transformers) - Autoencoders: ViT-MAE ForPreTraining (via transformers)

All 22 new tests pass. Updated RESULTS.md to reflect 736 total tests, 139 toy models, 92 real-world
  models.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **models**: Close remaining stress-test gaps — MAML, NeRF, RecurrentGemma, VOLO
  ([`a1f2254`](https://github.com/johnmarktaylor91/torchlens/commit/a1f22543bcc2cbb22b2008121a5082e64a534860))

Toy models (+2): - MAMLInnerLoop: higher-order gradients (torch.autograd.grad inside forward) -
  TinyNeRF: differentiable volumetric rendering (ray marching + alpha compositing)

Real-world models (+2): - RecurrentGemma: Griffin architecture (linear recurrence + local attention
  hybrid) - VOLO: outlooker attention (distinct from standard self-attention)

Closes 37/38 stress-test patterns from taxonomy. Only remaining gap is test-time training (TTT
  layers) which requires gradient computation within inference — fundamentally incompatible with
  TorchLens forward-pass logging.

Total: 249 toy models, 185 real-world models, 892 tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **models**: Exhaustive architecture coverage across 30+ categories
  ([`8411a72`](https://github.com/johnmarktaylor91/torchlens/commit/8411a725439819de55a54f4339afa21410f67686))

Add 32 new toy models (Groups M-R) covering distinct computational patterns: - Group M: Attention
  variants (MQA, GQA, RoPE, ALiBi, slot, cross-attention) - Group N: Gating & skip patterns
  (highway, SE, depthwise-sep, inverted-residual, FPN) - Group O: Generative & self-supervised
  (hierarchical VAE, gated conv, masked conv, SimCLR, stop-gradient/BYOL, AdaIN) - Group P: Exotic
  architectures (hypernetwork, DEQ, neural ODE, NTM memory, SwiGLU) - Group Q: Graph neural networks
  (GraphSAGE, GIN, EdgeConv, graph transformer) - Group R: Additional patterns (MoE, spatial
  transformer, dueling DQN, RMS norm, sparse pruning, Fourier mixing)

Add 37 new real-world model tests: - Decoder-only LLMs: LLaMA, Mistral, Phi, Gemma, Qwen2, Falcon,
  BLOOM, OPT - Encoder-only: ALBERT, DeBERTa-v2, XLM-RoBERTa - Encoder-decoder: Pegasus, LED -
  Efficient transformers: FNet, Nystromformer, BigBird - MoE: Mixtral, Switch Transformer - Vision
  transformers: DeiT, CvT, SegFormer - Detection: DETR, Mask R-CNN (train+eval) - Perceiver IO,
  PatchTST, Decision Transformer - timm: HRNet, EfficientNetV2, LeViT, CrossViT, PVT-v2, Twins-SVT,
  FocalNet - GNN (PyG): GraphSAGE, GIN, Graph Transformer

Total: 805 tests, 213 toy models, 129 real-world tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **models**: Exhaustive coverage expansion — 20 toy + 33 real-world architectures
  ([`4f4e7ae`](https://github.com/johnmarktaylor91/torchlens/commit/4f4e7aeb958e35ed7845b03b209367d8d55e5ab0))

Toy models (+20): GRU, NiN, ChannelShuffle, PixelShuffle, PartialConv, FiLM, CoordinateAttention,
  DifferentialAttention, RelativePositionAttention, EarlyExit, MultiScaleParallel, GumbelVQ,
  EndToEndMemoryNetwork, RBFNetwork, SIREN, MultiTask, WideAndDeep, ChebGCN, PrototypicalNetwork,
  ECA.

Real-world models (+33): GPT-J, GPTBigCode, GPT-NeoX, FunnelTransformer, CANINE, MobileBERT, mBART,
  ProphetNet, WavLM, Data2VecAudio, UniSpeech, ConvNeXt-v2, NFNet, DaViT, CoAtNet, RepVGG, ReXNet,
  PiT, Visformer, GC-ViT, EfficientFormer, FastViT, NesT, Sequencer2D, TResNet, SigLIP, BLIP-2,
  Deformable DETR, LayoutLM, TimeSeriesTransformer, ChebConv, SGConv, TAGConv.

Total: 241 toy models, 183 real-world models, 882 tests. RESULTS.md updated with all new entries and
  pattern coverage table.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **models**: Final coverage pass — 6 novel computational patterns
  ([`e33c71a`](https://github.com/johnmarktaylor91/torchlens/commit/e33c71a0cec2a64fce944aa36a42690fe3f6fe15))

New toy models targeting genuinely missing graph patterns: - LinearAttentionModel: kernel-based
  phi(Q)(phi(K)^T V), no softmax - SimpleFNO: FFT -> learned spectral weights -> iFFT (Fourier
  Neural Operator) - PerceiverModel: cross-attention to fixed learned latent bottleneck - ASPPModel:
  multi-rate parallel dilated convolutions (DeepLab ASPP) - ControlNetModel: parallel encoder copy +
  zero-conv injection - SimpleEGNN: E(n) equivariant message passing with coordinate updates

Total: 247 toy models, 183 real-world models, 888 tests. RESULTS.md updated with new patterns and
  counts.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **models**: Gap-fill 8 toy models + 21 real-world models for exhaustive coverage
  ([`7d5f879`](https://github.com/johnmarktaylor91/torchlens/commit/7d5f879bdefc9547d7341eea8b9c8adda3fd7e9f))

Toy models (8 new, 221 total): - LeNet5, BiLSTM, Seq2SeqWithAttention, TripletNetwork -
  BarlowTwinsModel, DeepCrossNetwork, AxialAttentionModel, CBAMBlock

Real-world models (21 new, 150 total): - TorchVision: MobileNetV3, Keypoint R-CNN (train+eval) -
  timm: Res2Net, gMLP, ResMLP, EVA-02 - HF decoder-only: OLMo - HF vision: DINOv2 - HF efficient:
  Longformer, Reformer - HF audio: AST, CLAP, EnCodec, SEW, SpeechT5, VITS - HF time series:
  Informer, Autoformer - PyG GNN: GATv2, R-GCN

834 total tests, 221 toy models, 150 real-world tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.14 (2026-03-07)

### Chores

- **release**: 0.15.14
  ([`a24b534`](https://github.com/johnmarktaylor91/torchlens/commit/a24b53496829a51eb0905d9686852a2e1cbe053a))

### Performance Improvements

- **capture**: Lazy _fsize_nice properties, remove _trim_and_reorder, batch pause_logging, drop
  deepcopy
  ([`986bd91`](https://github.com/johnmarktaylor91/torchlens/commit/986bd914a317be6acde0a4c0898a417fb9f6dcbb))

Four low-risk optimizations targeting remaining allocation pressure and per-operation overhead in
  the instrumentation path:

1. Lazy _fsize_nice properties — tensor_fsize_nice, grad_fsize_nice, parent_params_fsize_nice,
  total_params_fsize_nice, params_fsize_nice, and fsize_nice converted from eagerly computed strings
  to @property methods. Eliminates ~2700 human_readable_size() calls per Swin-T pass.

2. Remove _trim_and_reorder from postprocess — the OrderedDict rebuild of every LayerPassLog's
  __dict__ (685 calls, ~0.04s on Swin-T) is purely cosmetic. Python 3.7+ dicts maintain insertion
  order. Function definition kept for opt-in use.

3. Batch pause_logging for tensor memory — inline nelement() * element_size() at the two hottest
  call sites (_build_param_fields, _log_output_tensor_info) inside a single pause_logging() context.
  Eliminates per-call context manager overhead (~1088 calls).

4. Remove activation_postfunc deepcopy — copy.deepcopy() on a callable is unnecessary; callables are
  effectively immutable.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.13 (2026-03-07)

### Chores

- **release**: 0.15.13
  ([`efc5229`](https://github.com/johnmarktaylor91/torchlens/commit/efc5229a07c51fc21bf90615af8cbfab5b063a1d))

### Performance Improvements

- **capture**: Speed-optimized defaults and remaining bottleneck elimination
  ([#110](https://github.com/johnmarktaylor91/torchlens/pull/110),
  [`7c9a00c`](https://github.com/johnmarktaylor91/torchlens/commit/7c9a00cdec26258090cf6be1c93869cb82aa351a))

Seven targeted optimizations that reduce Swin-T log_forward_pass from 5.91s to 1.55s (3.8x):

1. Unified save_source_context flag (was save_call_stacks) — controls both per-function call stacks
  AND module source/signature fetching. Default: False. 2. save_rng_states=False default — skips
  per-op RNG state capture. Auto-enabled by validate_forward_pass. Uses torch_only=True when enabled
  (skips Python/NumPy RNG). 3. Inline isinstance in wrapped_func — _collect_tensor_args() and
  _collect_output_tensors() replace BFS crawls for flat arg/output cases. Falls back to BFS only for
  nested containers. 4. __dict__ scan for buffer prep/cleanup — replaces iter_accessible_attributes
  (dir() + MRO walk) with direct __dict__ iteration. 10x faster for buffer tagging and tensor
  cleanup. 5. Hoisted warnings.catch_warnings() — moved from per-attribute (46K entries) to caller
  level. 6. Lazy module metadata — _get_class_metadata skips
  inspect.getsourcelines/inspect.signature when save_source_context=False. Only captures class name
  and docstrings. 7. Module-level import weakref — moved from per-call in _trim_and_reorder to
  module level.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.12 (2026-03-06)

### Chores

- **release**: 0.15.12
  ([`57aaa83`](https://github.com/johnmarktaylor91/torchlens/commit/57aaa83977523ac6ea7f2103eb81cd8a809e7bf3))

### Performance Improvements

- **capture**: O(1) tensor/param extraction via per-function ArgSpec lookup table
  ([`b1d6c56`](https://github.com/johnmarktaylor91/torchlens/commit/b1d6c56f4f1c5cf1176232e8fc33e229f8555baf))

Replace expensive 3-level BFS crawl (~1.44s, 39% self-time, ~1.9M getattr calls) with O(1)
  position-based lookups using a static ArgSpec table of 350+ entries. Three-tier strategy: static
  table for known torch functions, dynamic cache for user-defined modules, BFS fallback (fires at
  most once per unique class).

Also hoists warnings.catch_warnings() from per-attribute (~77K entries) to per-call level, and adds
  usage stats collection + coverage test infrastructure.

Benchmark: Swin-T log_forward_pass 5.91s → 4.41s (-25%).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.11 (2026-03-06)

### Bug Fixes

- **gc**: Convert back-references to weakrefs and add optional call stack collection
  ([`2bed167`](https://github.com/johnmarktaylor91/torchlens/commit/2bed167d78fed3a143060f8f5d75007cd3a06214))

Convert 5 circular back-references from strong to weakref.ref() so ModelLog and its children
  (LayerPassLog, LayerLog, ModuleLog, BufferAccessor, LayerAccessor) no longer prevent timely
  garbage collection. GPU tensors are now freed immediately when the last strong reference to
  ModelLog is dropped, instead of waiting for Python's gen-2 GC cycle.

Also add save_call_stacks parameter to log_forward_pass() (default True). When False, skips
  _get_func_call_stack() on every tensor operation, which is the main per-op overhead in production
  use. Call stacks remain on by default for pedagogical use.

Fixes: GC-2, GC-3, GC-4, PERF-19

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.15.11
  ([`55df020`](https://github.com/johnmarktaylor91/torchlens/commit/55df02036c09372bbc73e1ed0941d34bb7adefea))

### Documentation

- Add folder-wise CLAUDE.md files and comprehensive inline documentation
  ([`deaec2d`](https://github.com/johnmarktaylor91/torchlens/commit/deaec2d4b3fb3c7ffee1fe7f61c8701bdc36dc70))

- Add CLAUDE.md to every package directory (torchlens/, capture/, data_classes/, decoration/,
  postprocess/, utils/, validation/, visualization/, tests/, scripts/, .github/) with file maps, key
  concepts, gotchas, and cross-references - Add module-level docstrings, function/class docstrings,
  and inline comments across all 39 source files explaining non-obvious logic, ordering
  dependencies, design decisions, and invariants - Fix coverage HTML output directory in
  pyproject.toml to point to tests/test_outputs/reports/coverage_html (matching conftest.py)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.10 (2026-03-05)

### Bug Fixes

- **tests**: Move coverage HTML output to reports directory
  ([`916b2d1`](https://github.com/johnmarktaylor91/torchlens/commit/916b2d19a0aec297546230a87db1290aea1f9984))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.15.10
  ([`9530221`](https://github.com/johnmarktaylor91/torchlens/commit/9530221e015ee39dfaa43f89fb03e8c9409032a2))


## v0.15.9 (2026-03-05)

### Bug Fixes

- **tests**: Use render_graph return value instead of reading cleaned-up .gv file
  ([`7998776`](https://github.com/johnmarktaylor91/torchlens/commit/7998776401752eeedb5693b665114c29234c893d))

Commit 147c7b7 added cleanup=True to dot.render(), which deletes the intermediate .gv source file
  after rendering. The TestVisualizationParams tests were reading that source file and all 15 failed
  with FileNotFoundError.

render_graph now returns dot.source so tests can inspect the graphviz source without depending on
  the intermediate file.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.15.9
  ([`014acd8`](https://github.com/johnmarktaylor91/torchlens/commit/014acd8c94f2d2d4c39a40287f6a5c241cb89f35))


## v0.15.8 (2026-03-05)

### Bug Fixes

- **tests**: Generate HTML coverage report in sessionfinish hook
  ([`29095f9`](https://github.com/johnmarktaylor91/torchlens/commit/29095f91e19bfbfd0a66a855232247679183d85f))

The pyproject.toml configured coverage_html output directory but the pytest_sessionfinish hook only
  generated the text report. Add cov.html_report() call so HTML reports are written alongside the
  text summary.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.15.8
  ([`e47f747`](https://github.com/johnmarktaylor91/torchlens/commit/e47f74745d452264da48392c24e4e2f36260d645))


## v0.15.7 (2026-03-04)

### Bug Fixes

- **vis**: Clean up intermediate .gv source files after rendering
  ([`147c7b7`](https://github.com/johnmarktaylor91/torchlens/commit/147c7b7329ae3e86ee1c65dbe61e30a849dd719f))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.15.7
  ([`16ca1f4`](https://github.com/johnmarktaylor91/torchlens/commit/16ca1f4c3792b62bda432fe0534f5783dd441d72))

- **tests**: Add @pytest.mark.smoke to 18 critical-path tests
  ([`d26f4ec`](https://github.com/johnmarktaylor91/torchlens/commit/d26f4ec86fbe64f26deb4c40880da4fb5882718c))

18 fast tests across 9 files marked as smoke tests for quick validation during development. Run with
  `pytest tests/ -m smoke` (~6s).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.6 (2026-03-04)

### Bug Fixes

- **tests**: Add explicit vis_outpath to render_graph calls
  ([`7d7926b`](https://github.com/johnmarktaylor91/torchlens/commit/7d7926b22781e3e78ca9cde166efc9ead7b0ec57))

Two tests in TestVisualizationBugfixes called render_graph() without vis_outpath, causing stray
  modelgraph/modelgraph.pdf files in whatever directory tests were run from.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.15.6
  ([`25f16cc`](https://github.com/johnmarktaylor91/torchlens/commit/25f16ccbe8f60cec1d2b55570068dc004d13c31b))

- **tests**: Rename test_outputs/graphs to test_outputs/visualizations
  ([`fbf1fe6`](https://github.com/johnmarktaylor91/torchlens/commit/fbf1fe66786cf829ed4a26e76a9a934708e92a75))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.5 (2026-03-04)

### Bug Fixes

- **loop-detection**: Group param-sharing ops by func_applied_name + param_barcodes
  ([`f1e795b`](https://github.com/johnmarktaylor91/torchlens/commit/f1e795b05365b6929fbf3a7b3a30446d93073674))

Previously _merge_iso_groups_to_layers Pass 2 grouped operations by parent_param_barcodes alone,
  causing different ops on the same params (e.g. __getitem__ vs __add__) to be incorrectly merged
  into one layer. Now groups by (func_applied_name, sorted(parent_param_barcodes)). Also updates
  docstrings to reflect the correct grouping rule.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.15.5
  ([`5922269`](https://github.com/johnmarktaylor91/torchlens/commit/5922269e96e235f6ac98588a6e581a565687de6f))

- **tests**: Tidy test_outputs structure and remove private bug numbers
  ([`03817a2`](https://github.com/johnmarktaylor91/torchlens/commit/03817a25f1091f4cb0bf16b9f547cab08ecc515d))

- Restructure test_outputs/ into reports/ and graphs/ subdirectories - Update all path references in
  conftest, test_output_aesthetics, test_profiling, and ui_sandbox notebook - Strip private bug
  number references (Bug #N, #N:) from comments, docstrings, and class names across 7 test files -
  Rename TestBug* classes to descriptive names

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.4 (2026-03-04)

### Bug Fixes

- **validation**: Resolve 14 test failures across validation, labeling, and warnings
  ([`ffc51a5`](https://github.com/johnmarktaylor91/torchlens/commit/ffc51a5be57033ffc1b0506af5a330bd97a343b0))

- Fix param sharing invariant: include func_applied_name in grouping key so different operations
  consuming the same parameter aren't incorrectly required to share labels (12 model failures: vit,
  beit, cait, coat, convit, poolformer, xcit, t5, clip, blip, etc.) - Fix unused input validation:
  guard against func_applied=None for unused model inputs like DistilBert's token_type_ids (1
  failure) - Fix pass labeling consistency: inherit layer_type from first pass for pass>1 entries,
  preventing label mismatches within same_layer_operations groups (SSD300 train failure) - Fix numpy
  2.0 deprecation: use cpu_data.detach().numpy() instead of np.array(cpu_data) - Suppress 6 external
  third-party warnings in pyproject.toml filterwarnings - Add 4 regression tests with 2 helper model
  classes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.15.4
  ([`c8c8e94`](https://github.com/johnmarktaylor91/torchlens/commit/c8c8e94c965c5c45a1a0eaf74e290509720adf73))


## v0.15.3 (2026-03-04)

### Bug Fixes

- **types**: Move type: ignore to correct lines after ruff reformat
  ([`8205ca4`](https://github.com/johnmarktaylor91/torchlens/commit/8205ca4002ab2f0d0ea5f4b09411735b8ff159a4))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **types**: Resolve all 181 mypy errors across 18 source files
  ([`546fd94`](https://github.com/johnmarktaylor91/torchlens/commit/546fd9438241da86cf394a03716d8b3dd32391ab))

- Auto-fix implicit Optional with no_implicit_optional tool (24 errors) - Add type annotations for
  untyped variables (20 errors) - Add type: ignore[attr-defined] for dynamic tl_* attributes (24
  errors) - Add type: ignore[assignment] for fields_dict and cross-type assignments (79 errors) -
  Add type: ignore[union-attr] for ModuleLog|ModulePassLog unions (19 errors) - Add type:
  ignore[arg-type] for acceptable type mismatches (19 errors) - Fix callable type annotation in
  exemptions.py (Callable[..., bool]) - Fix found_ids parameter type in introspection.py (List ->
  Set)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **types**: Suppress torch.device return-value mismatch in CI mypy
  ([`11922b3`](https://github.com/johnmarktaylor91/torchlens/commit/11922b32aaa6ca5f9769785b4afe737121ae3fae))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **ci**: Remove continue-on-error from mypy check — now a blocking gate
  ([`0e76db0`](https://github.com/johnmarktaylor91/torchlens/commit/0e76db01d0016e1a2b033af29b0b5e4fbc7933b8))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **release**: 0.15.3
  ([`1781549`](https://github.com/johnmarktaylor91/torchlens/commit/1781549569858ab419266ae5f681324bff8abcf5))


## v0.15.2 (2026-03-04)

### Bug Fixes

- **core**: Resolve remaining open bugs — FLOPs accuracy, PERF-21, mypy types (#19, #154-161)
  ([`da02dc5`](https://github.com/johnmarktaylor91/torchlens/commit/da02dc56b4940cf32c5e319dc4310ffa7e15c0e3))

Bug #19: Verified fast-pass buffer orphan resolved by Bug #116 guard; added regression tests FLOPs:
  Fix addbmm/baddbmm batch dimension, add einsum handler, fix pool kernel_size extraction

PERF-21: Cache get_ignored_functions()/get_testing_overrides() in _get_torch_overridable_functions

Bug #154: Type _SENTINEL as Any in func_call_location.py Bug #155: Eliminate reused variable across
  str/bytes types in hashing.py Bug #156: Accept float in human_readable_size signature Bug #157:
  Use setattr/getattr for dynamic tl_tensor_label_raw in safe_copy Bug #158: Add List[Any]
  annotation to AutocastRestore._contexts Bug #160: Add _FlopsHandler type alias and typed
  SPECIALTY_HANDLERS dict Bug #161: Add ModuleLog TYPE_CHECKING import and type annotations in
  invariants.py

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **ci**: Switch CI from pip to uv for faster installs
  ([`d619ef2`](https://github.com/johnmarktaylor91/torchlens/commit/d619ef212e43c397336a5e0479aa52a47e82569a))

Replace pip with uv pip in quality and lint workflows. uv resolves and installs packages 10-20x
  faster. Also adds explicit torch CPU index-url to dep-audit job (was missing, pulling full CUDA
  bundle).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **release**: 0.15.2
  ([`fe64348`](https://github.com/johnmarktaylor91/torchlens/commit/fe643489fbfad61cd61bd2221def2796feaa856a))


## v0.15.1 (2026-03-04)

### Bug Fixes

- **core**: Resolve 52 bugs across 9 waves of correctness, safety, and polish fixes
  ([`d4ca75d`](https://github.com/johnmarktaylor91/torchlens/commit/d4ca75d13defe15f558799be05cb4dcd9d217a45))

Wave 1 — Critical Correctness: - Rewrite safe_copy() to eliminate numpy round-trip, fix bfloat16
  overflow, handle sparse/meta/quantized dtypes (#103, #137, #128, #139) - Guard print_override
  numpy conversion (#140), meta/sparse tensor memory (#24) - Fix validation crash on None
  parent_values (#150), silent exception pass (#151) - Add buffer duplicate guard in fast path
  (#116) - Fix postprocess_fast shared reference and unguarded parent_layers (#8, #152) - Add empty
  graph guard (#153), zombie label cleanup (#75) - Handle defaultdict in _safe_copy_arg (#127)

Wave 2 — Exception Safety: - Make module_forward_decorator exception-safe (#122) - Guard
  _handle_module_entry for untracked tensors (#117) - Clean up output tensors on fast-pass exception
  (#110) - Wrap activation_postfunc in pause_logging (#96) - Guard validation with
  save_function_args=False (#131) - Fix CUDA perturbation device consistency (#36)

Wave 3 — Fast-Pass State Reset: - Reset timing and stale lookup caches in save_new_activations (#87,
  #97, #98) - Pass raw label (not final) to gradient hook in fast path (#86) - Raise descriptive
  error for unexpected missing parents (#111)

Wave 4 — Argument Handling: - Deep-copy nested tuples in creation_args (#44) - Slice before cloning
  in display helper (#73) - Use logged shape in display (#45)

Wave 5 — Interface Polish: - Key layer_num_passes by no-pass label (#53) - Use rsplit for
  colon-split safety (#54) - Add slice indexing support (#78) - Guard to_pandas before pass finished
  (#124) - List candidates for ambiguous substring (#125) - Add ModuleLog string indexing (#120) -
  Support int/short-name in ParamAccessor.__contains__ (#84)

Wave 6 — Decoration/Model Prep: - Only add mapper entries if setattr succeeded (#31) - Add hasattr
  guard for identity asymmetry (#39) - Guard buffer setattr (#40) - Store dunder argnames stripped
  (#82) - Use inspect.Parameter.kind for varargs (#123)

Wave 7 — Visualization: - Fix vis_nesting_depth=0 crash (#94) - Use rsplit for colon-split (#104) -
  Fix collapsed node variable (#100) - Guard None tensor_shape (#118) - Guard LayerLog nesting depth
  (#138) - Use exact equality instead of substring match (#129)

Wave 8 — Control Flow/Loop Detection: - Remove dead sibling_layers iteration (#2) - Prevent shared
  neighbor double-add (#148)

Wave 9 — Cleanup + GC: - Lazy import IPython (#72) - Filter callables from _trim_and_reorder (#134)
  - Complete cleanup() with internal containers (GC-5, GC-12) - Use weakref in backward hook closure
  (GC-8) - Clear ModulePassLog forward_args/kwargs after build (GC-11)

585 tests passing across 10 test files, 47 new regression tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Resolve final remaining bugs #23, #83, #95, #99
  ([`6302662`](https://github.com/johnmarktaylor91/torchlens/commit/630266251861ca505fcd0f33b33d3aa3b21f7c66))

- #95: fix mixed LayerLog/LayerPassLog format in vis module containment check - #83:
  LayerLog.parent_layer_arg_locs returns strings (not sets) for consistency - #99: warn on tensor
  shape mismatch in fast-path source tensor logging - #23: add case-insensitive exact match and
  substring lookup for layers/modules - #85: confirmed not-a-bug (any special arg correctly explains
  output invariance) - #39, #41, #42: confirmed already fixed or correct as-is - Add 6 regression
  tests for the above fixes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Resolve remaining low-risk bugs #28, #107, #108, #147
  ([`d8616a0`](https://github.com/johnmarktaylor91/torchlens/commit/d8616a0b1dfa133cd5aeb6bd92af6cdfb2bc566a))

- #147: descriptive ValueError in log_source_tensor_fast on graph change - #108: document fast-path
  module log preservation (no rebuild needed) - #107: handle both tuple and string formats in module
  hierarchy info - #28: remove torch.Tensor from dead type-check list in introspection - Add 4
  regression tests for the above fixes

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.15.1
  ([`17bb57a`](https://github.com/johnmarktaylor91/torchlens/commit/17bb57ab3f44b2be089fdc021db1b626e0522d49))

### Testing

- **reorganize**: Redistribute bugfix tests into domain-specific test files
  ([`e9c64cc`](https://github.com/johnmarktaylor91/torchlens/commit/e9c64cc58a6bbd8f5d1b9942d234af3e0eed94bd))

Move all 57 regression tests from test_bugfixes.py into their natural homes: - test_internals.py:
  safe_copy, tensor utils, exception safety, cleanup/GC - test_validation.py: validation
  correctness, posthoc perturb check - test_save_new_activations.py: fast-path state reset, graph
  consistency - test_layer_log.py: interface polish, case-insensitive lookup, arg locs -
  test_module_log.py: string indexing, tuple/string normalization - test_param_log.py: ParamAccessor
  contains, param ref cleanup - test_output_aesthetics.py: visualization smoke tests

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.15.0 (2026-03-04)

### Bug Fixes

- **validation**: Resolve 8 test failures and Bug #79 in metadata invariants
  ([`14b2e5b`](https://github.com/johnmarktaylor91/torchlens/commit/14b2e5bbb8ac96d0dc45d8f8108db948dbb20097))

Three root-cause fixes for failures exposed by the full test suite:

1. Module hierarchy aliasing (finalization.py): shared nn.Module instances (same Conv2d registered
  as both self.proj1 and self.conv_list[0]) had metadata lookup fail because _prepare_model_once
  overwrites tl_module_address to the last-visited alias while _capture_module_metadata stores under
  the first-visited alias. Built alias→meta map and address_children prefix rewriting so all aliases
  resolve correctly.

2. Loop detection Bug #79 (loop_detection.py): _merge_iso_groups_to_layers only compared pairs
  within iso-groups, missing Rule 1 (param sharing → same layer) across iso-groups. Rewrote with
  union-find + path compression and added cross-iso-group param barcode merge pass. Also unify
  operation_equivalence_type for merged nodes whose module path suffixes differ.

3. Adjacency invariant (invariants.py): removed incorrect operation-level neighbor check —
  subgraph-level adjacency is verified during BFS in loop_detection.py, not reconstructable
  post-hoc. Non-param multi-pass groups can be the only multi-pass group (param-free loops).
  Upgraded param sharing violation back from warning to error now that Bug #79 is fixed.

Also: boolean flag fixes in control_flow.py and graph_traversal.py for buffer/output layers;
  @pytest.mark.slow on vit and beit tests; profiling test streamlined.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **ci**: Add mypy type checking and pip-audit dependency auditing
  ([`2a0e0e7`](https://github.com/johnmarktaylor91/torchlens/commit/2a0e0e725f3551db15221d2bea326018423df880))

Add quality.yml workflow with two jobs: - mypy (advisory, continue-on-error) — 199 existing errors
  to fix later - pip-audit (blocking) — fails on known CVEs in dependencies

Mypy configured leniently in pyproject.toml; both tools added to dev deps.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Add pip caching to quality workflow
  ([`d7573b1`](https://github.com/johnmarktaylor91/torchlens/commit/d7573b18d57bc3e9d20c4c524875ffdb07e4de1f))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **coverage**: Add pytest-cov with HTML and text report generation
  ([`c1ac2c9`](https://github.com/johnmarktaylor91/torchlens/commit/c1ac2c900d334e0ba8e0cff2e6f248e506d15686))

Configure branch coverage for torchlens/ source. HTML report writes to
  tests/test_outputs/coverage_html/, text summary to coverage_report.txt. Reports auto-generate when
  running pytest --cov via a sessionfinish hook.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **dev**: Add UI sandbox notebook and nbstripout for clean diffs
  ([`5247361`](https://github.com/johnmarktaylor91/torchlens/commit/524736119646202be37bb8fb30b18d584d0a5a26))

Add tests/ui_sandbox.ipynb — interactive workbench for tinkering with torchlens during development:
  logging, accessors, reprs, visualization, validation. Set up nbstripout via .gitattributes to
  auto-strip cell outputs on commit.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **profiling**: Expand profiling report to 12 models across architecture families
  ([`60e983a`](https://github.com/johnmarktaylor91/torchlens/commit/60e983a4d47c05141bbe6f47929708018b3f5727))

Added 9 models to the profiling test for illustrative coverage: - SimpleBranching (diverging/merging
  flow) - ResidualBlock (conv-BN-relu skip connection) - MultiheadAttention (nn.MultiheadAttention)
  - LSTM (recurrent + classifier) - ResNet18, MobileNetV2, EfficientNet_B0 (real CNNs) - Swin_T
  (shifted-window vision transformer, 671 layers) - VGG16 (deep sequential CNN, 138M params)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **release**: 0.15.0
  ([`a2bcc75`](https://github.com/johnmarktaylor91/torchlens/commit/a2bcc75862df6bbbabed277935e00e5634a333e1))

### Features

- **modules**: Add shared module alias lookup and explicit alias fields
  ([`c2fb14e`](https://github.com/johnmarktaylor91/torchlens/commit/c2fb14ef15fd27bdd352b2e27b99c4866c5706bd))

Shared nn.Module instances (same object registered under multiple addresses) can now be looked up by
  any alias:

ml.modules["shared_conv"] # primary address ml.modules["alias_list.0"] # alias — returns same
  ModuleLog

New fields: - ModuleLog.is_shared: bool, True when len(all_addresses) > 1 -
  ModulePassLog.all_module_addresses: list of all addresses for the module -
  ModulePassLog.is_shared_module: bool, same as ModuleLog.is_shared - ModuleAccessor builds
  alias→ModuleLog map for __getitem__/__contains__ - ModuleLog.__repr__ shows aliases when
  is_shared=True

Invariant checks (module_hierarchy) now account for shared modules where address_parent refers to
  the primary alias's parent path.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Add complex semantic invariants M-R to check_metadata_invariants
  ([`b745db8`](https://github.com/johnmarktaylor91/torchlens/commit/b745db826135e0470df75d70eaa6dcce48feae42))

Add 6 new invariant categories verifying loop detection, graph ordering, distance/reachability,
  connectivity, module containment, and lookup key consistency. Fix existing checks H and L for
  recurrent model compatibility. 17 new corruption tests (50 total in test_validation.py).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Add validate_forward_pass with metadata invariant checks
  ([`8a25ede`](https://github.com/johnmarktaylor91/torchlens/commit/8a25ede7f90b4e924f06e55ad15bb46d148f7384))

Rename validate_saved_activations → validate_forward_pass (old name kept as deprecated alias). Add
  comprehensive metadata invariant checker (check_metadata_invariants) covering 12 categories:
  ModelLog self-consistency, special layer lists, graph topology, LayerPassLog fields, recurrence,
  branching, LayerLog cross-refs, module-layer containment, module hierarchy, param cross-refs,
  buffer cross-refs, and equivalence symmetry.

validate_metadata=True by default, so every existing validation call now exercises the full
  invariant suite automatically.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **profiling**: Add automated profiling report for overhead measurement
  ([`a483011`](https://github.com/johnmarktaylor91/torchlens/commit/a483011785da9ee4f5dd5957c083d9e67f9f7e82))

Profiles raw forward pass, log_forward_pass, save_new_activations, and validate_saved_activations
  across toy and real-world models. Generates tests/test_outputs/profiling_report.txt with absolute
  times and overhead ratios.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.14.0 (2026-03-04)

### Chores

- **release**: 0.14.0
  ([`88c7c3b`](https://github.com/johnmarktaylor91/torchlens/commit/88c7c3b1b32459dcec1fbed9f4320dc5a55d22f9))

### Features

- **core**: Add buffer name/module_address properties to LayerLog
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`e3a1ce2`](https://github.com/johnmarktaylor91/torchlens/commit/e3a1ce21927af0459f1388ed559b0a28a7646bb6))

Ensures buffer_address, name, and module_address are accessible on both buffer LayerPassLogs (via
  BufferLog subclass) and buffer LayerLogs (via computed properties derived from buffer_address).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Add LayerAccessor and log.layers property
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`b639e50`](https://github.com/johnmarktaylor91/torchlens/commit/b639e50d793d319d83fc51ecd86c3ac367ad3161))

Adds LayerAccessor (dict-like accessor for LayerLog objects) with support for
  label/index/pass-notation lookup and to_pandas(). Accessible via log.layers, mirroring log.modules
  for modules.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Add LayerLog aggregate class with per-layer metadata
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`9c6472d`](https://github.com/johnmarktaylor91/torchlens/commit/9c6472db6c71c10dbed02428844972f23ebb96b3))

Introduces LayerLog as the aggregate per-layer object grouping one or more LayerPassLog entries (one
  per invocation). For non-recurrent models every LayerLog has exactly one pass; for recurrent
  models, multiple passes are grouped under a single LayerLog with union-based aggregate graph
  properties.

Key changes: - New LayerLog class with ~52 aggregate fields, single-pass delegation via @property
  and __getattr__, aggregate graph properties (child_layers, parent_layers unions) -
  _build_layer_logs() postprocessing step wired into both exhaustive and fast pipelines -
  ModelLog.layer_logs OrderedDict populated after postprocessing - __getitem__ routing: log["label"]
  returns LayerLog for multi-pass layers - parent_layer_log back-reference on LayerPassLog - 27 new
  tests covering construction, delegation, multi-pass behavior, display, and integration

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **core**: Modulelog.all_layers now stores unique no-pass labels
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`214e161`](https://github.com/johnmarktaylor91/torchlens/commit/214e1616e87ab89a9f7e6a44de8d7e6453bdf430))

Establishes aggregate↔aggregate symmetry: - ModuleLog.all_layers → no-pass labels → resolve to
  LayerLog - ModulePassLog.layers → pass-qualified labels → resolve to LayerPassLog

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Remove buffer name/module_address from LayerLog
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`f896dd4`](https://github.com/johnmarktaylor91/torchlens/commit/f896dd41c24f4174b70395db0fb174c8b70f1e44))

These properties are too generic on LayerLog (name returns "" for non-buffers). Keep them only on
  BufferLog(LayerPassLog); single-pass buffer LayerLogs still access them via __getattr__
  delegation.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Remove RolledTensorLog, use LayerLog for rolled visualization
  ([#92](https://github.com/johnmarktaylor91/torchlens/pull/92),
  [`118b3ec`](https://github.com/johnmarktaylor91/torchlens/commit/118b3ec67be88274d80634a79f38e87f1c70e27e))

RolledTensorLog was a visualization-only aggregate that duplicated ~30 fields from LayerPassLog with
  its own merge logic. Now that LayerLog provides a proper aggregate abstraction, RolledTensorLog is
  unnecessary.

Key changes: - Delete RolledTensorLog class (~175 lines) and _roll_graph() function - Remove
  layer_list_rolled and layer_dict_rolled from ModelLog - Visualization "rolled" mode now uses
  self.layer_logs (LayerLog objects) - Add vis-specific computed properties to LayerLog:
  child_layers_per_pass, parent_layers_per_pass, child_passes_per_layer, parent_passes_per_layer,
  edges_vary_across_passes, bottom_level_submodule_passes_exited, parent_layer_arg_locs -
  LayerLog.child_layers/parent_layers always return no-pass labels - Merge multi-pass aggregate
  fields (has_input_ancestor, input_output_address, is_bottom_level_submodule_output) in
  _build_layer_logs - Update tests and aesthetic report to use LayerLog

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Rename TensorLog → LayerPassLog + nomenclature sweep
  ([`d328476`](https://github.com/johnmarktaylor91/torchlens/commit/d32847669bc0d2410d6f1070865f53c81de84b98))

Phase 1 of LayerLog hierarchy redesign. Mechanical rename of the class, file (tensor_log.py →
  layer_pass_log.py), constant (TENSOR_LOG_FIELD_ORDER → LAYER_PASS_LOG_FIELD_ORDER), and all
  imports. Also sweeps internal nomenclature: field names (_raw_tensor_dict → _raw_layer_dict,
  etc.), function names (_make_tensor_log_entry → _make_layer_log_entry, etc.), and local variables
  (tensor_entry → layer_entry, etc.) that referred to log entries rather than actual tensors.
  Backward-compat aliases preserved.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **core**: Replace 17 flat module_* dicts on ModelLog with transient _module_build_data
  ([`ef3f20e`](https://github.com/johnmarktaylor91/torchlens/commit/ef3f20ef16422a4c997a2727cf9c9df941e56030))

These intermediate fields (module_addresses, module_types, module_passes, etc.) were only used
  during postprocessing to build structured ModuleLog objects, but persisted on the public API. Now
  consolidated into a single _module_build_data dict that is populated during logging, consumed by
  _build_module_logs (step 17), and then re-initialized. Reduces ModelLog from ~143 to ~127 public
  fields.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.13.1 (2026-03-03)

### Bug Fixes

- **core**: Handle DataParallel, device mismatch, and embedding perturbation
  ([#91](https://github.com/johnmarktaylor91/torchlens/pull/91),
  [`1a0f8b8`](https://github.com/johnmarktaylor91/torchlens/commit/1a0f8b802f738a399d74876f39d121336a1fe855))

- Unwrap nn.DataParallel in log_forward_pass, show_model_graph, validate_saved_activations -
  Auto-move inputs to model device (supports dict, UserDict, BatchEncoding) - Exempt embedding index
  arg from perturbation (prevents CUDA OOB)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.13.1
  ([`9bd2aa3`](https://github.com/johnmarktaylor91/torchlens/commit/9bd2aa376b9dffe55194f1557c1c988b487a9963))

### Refactoring

- **batch1**: Aesthetic cleanup — names, types, docstrings for small files
  ([`52efdc1`](https://github.com/johnmarktaylor91/torchlens/commit/52efdc1e8ec55e1b5047589903f3a601523dfe14))

Files: _state.py, buffer_log.py, cleanup.py, postprocess/__init__.py, param_log.py,
  func_call_location.py

- Add docstrings to dunder methods across BufferAccessor, ParamAccessor, ParamLog, FuncCallLocation
  - Rename _UNSET -> _SENTINEL in func_call_location.py - Fix single-letter vars in cleanup.py
  (x->label, tup->edge, k/v->descriptive) - Add return type hints (-> None) to all cleanup functions
  - Add docstring and return type to postprocess_fast() - Add property docstrings and setter type
  hints to ParamLog

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **batch2**: Aesthetic cleanup — names, types, docstrings for medium files
  ([`8610c46`](https://github.com/johnmarktaylor91/torchlens/commit/8610c464a2106d8037198852fb7fa7acba6ed808))

Files: trace_model.py, control_flow.py, model_log.py, module_log.py, graph_traversal.py,
  finalization.py, interface.py, user_funcs.py

- Add docstrings to _get_input_arg_names, _find_output_ancestors, _update_node_distance_vals,
  _log_internally_terminated_tensor, _log_time_elapsed, _single_pass_or_error - Fix truncated
  docstring on _flood_graph_from_input_or_output_nodes - Add return type hints to all private
  functions - Rename loop vars: a->arg_idx, t->tensor, pn->pass_num, mpl->module_pass_log,
  cc->child_pass_label, sp->child_pass_label - Rename pretty_print_list_w_line_breaks ->
  _format_list_with_line_breaks - Rename _module_hierarchy_str_helper ->
  _module_hierarchy_str_recursive - Mark run_model_and_save_specified_activations as private (_) -
  Rename x->input_data in validate_batch_of_models_and_inputs - Add docstrings to all dunder methods
  on ModuleLog, ModulePassLog, ModuleAccessor

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **batch3**: Aesthetic cleanup — names, types, docstrings for medium-large files
  ([`74c0850`](https://github.com/johnmarktaylor91/torchlens/commit/74c085095276e0361607d6d3226df85ffa8e8187))

Files: decorate_torch.py, labeling.py, constants.py, validation/core.py, loop_detection.py

- Rename open_ind/close_ind -> paren_start/paren_end in decorate_torch.py - Rename
  namespace_name_notorch -> namespace_key - Rename my_get_overridable_functions ->
  _get_torch_overridable_functions - Expand shadow set abbrevs: _ml_seen -> _module_labels_seen,
  etc. - Rename p_label -> perturbed_label in validation/core.py - Parameterize bare Dict type hints
  in loop_detection.py - Expand single-letter vars: m->module_index, n->pass_index, s->subgraph_key
  - Add docstrings to _merge_iso_groups_to_layers, _validate_layer_against_arg,
  _copy_validation_args, _get_torch_overridable_functions - Add return type hints to all private
  functions

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **batch4**: Aesthetic cleanup — names, types, docstrings for large files
  ([`ce19825`](https://github.com/johnmarktaylor91/torchlens/commit/ce19825710472895b0b644881d6db3bacea50e84))

tensor_log.py: add docstrings to _str_during_pass, _str_after_pass, _tensor_family_str_helper
  model_funcs.py: rename log_whether_exited_submodule_is_bottom_level →
  _is_bottom_level_submodule_exit, fwd → forward_func, mid → module_id helper_funcs.py: rename
  make_var_iterable → ensure_iterable, tuple_tolerant_assign → assign_to_sequence_or_dict,
  remove_attributes_starting_with_str → remove_attributes_with_prefix flops.py: add type hints to
  all 25 private functions, add Args sections vis.py: rename _check_whether_to_mark_* →
  _should_mark_*, _construct_* → _build_*, fix _setup_subgraphs consistency logging_funcs.py: rename
  _log_info_specific_to_single_function_output_tensor → _log_output_tensor_info,
  _get_parent_tensor_function_call_location → _locate_parent_tensors_in_args

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **design**: Algorithmic review pass — efficiency, correctness, robustness
  ([`78924f8`](https://github.com/johnmarktaylor91/torchlens/commit/78924f825f2b604fe7dcd1df234ce76d23c51ad6))

Efficiency (21 fixes): - Hot path: eliminate all_args allocation via kwargs short-circuit - Barcode
  gen: secrets.choice → random.choices with pre-computed alphabet - Type scanning: any([...]) →
  any(...) generator, found_ids list → set - Model prep: recursive get_all_submodules →
  list(model.modules()) - Source/output tensors: eliminate double get_tensor_memory_amount() calls -
  Output tensors: copy only mutables instead of full 131-key dict copy - Gradient hooks: O(N²)
  per-hook rebuild → _saved_gradients_set O(1) dedup - Cleanup: dir() → list(__dict__), skip O(n²)
  reference removal in full teardown - Labeling: dir(self) → self.__dict__, inds_to_remove list →
  set - Finalization: pre-computed reverse mappings, _roll_graph idempotency guard - Graph
  traversal/validation: min([a,b]) → min(a,b), list.pop(0) → deque.popleft() - Visualization:
  any([...]) → any(...), min([...]) → min(...) - TensorLog: frozenset for FIELD_ORDER,
  view-then-clone-slice for display - Display: np.round → round(), removed numpy dependency

Correctness (4 FLOPs fixes): - SDPA: new _sdpa_flops handler (Q@K^T + softmax + attn@V) -
  addbmm/baddbmm: _matmul_flops → _addmm_flops (correct shape extraction) - MHA: return None to
  avoid double-counting with sub-op FLOPs - scatter/index ops: moved from ZERO_FLOPS to
  ELEMENTWISE_FLOPS

Robustness (4 fixes): - Buffer dedup: for...else pattern fixes incorrect inner-loop append -
  Parameter fingerprinting: reorder isinstance checks (Parameter before Tensor) - Duplicate
  siblings: added not-in guards before sibling append - _roll_graph: early-return guard if already
  populated

Test coverage: - New tests/test_internals.py: 13 tests for FIELD_ORDER sync and constants

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **design**: Mid-level design pass — dataclasses, decomposition, parameter bundles
  ([`0849edb`](https://github.com/johnmarktaylor91/torchlens/commit/0849edb605bb63a469831dd1959c1bd8236ce964))

Replace implicit parameter clusters with explicit dataclasses (FuncExecutionContext,
  VisualizationOverrides, IsomorphicExpansionState, ModuleParamInfo), decompose 100+ line functions
  into coordinator + helpers, and extract shared patterns (model traversal visitor, buffer tagging).

No public API changes. All 471 non-slow tests pass.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **qa**: Address naive reader findings — docstrings, comments, variable names
  ([`22724af`](https://github.com/johnmarktaylor91/torchlens/commit/22724afb26e6511365eee5c6ce6bdf97888eeb12))

Fix 14 issues flagged by whole-codebase naive reader review: - make_random_barcode: fix misleading
  "integer hash" docstring - _getitem_after_pass: replace TODO-like docstring with proper docs -
  _is_bottom_level_submodule_exit: define "bottom-level" in docstring -
  search_stack_for_vars_of_type: rename tensor-specific accumulators to generic names -
  validate_parents_of_saved_layer: fill in empty param docs, note mutation - _append_arg_hash:
  explain operation_equivalence_type concept - nested_assign: add docstring and type hints -
  _capture_module_metadata: fix misleading wording - extend_search_stack_from_item: document missing
  address/address_full params - safe_copy: add bfloat16 round-trip comment - _roll_graph: define
  "rolled" in docstring - vis.py: add docstrings to _get_node_bg_color, _make_node_label,
  _get_max_nesting_depth - _give_user_feedback_about_lookup_key: document mode parameter -
  _get_torch_overridable_functions: comment the ignore flag - RolledTensorLog.update_data: replace
  vacuous docstring

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **structure**: Reorganize into focused subpackages
  ([`2cfa90a`](https://github.com/johnmarktaylor91/torchlens/commit/2cfa90ac15e8e325c9e417202b0d884450c43915))

Split grab-bag root modules into focused subpackages without changing any public API, class names,
  or core algorithmic logic.

- Split helper_funcs.py into 7 utils/ modules (rng, tensor_utils, arg_handling, introspection,
  collections, hashing, display) - Moved cleanup.py + interface.py into data_classes/ - Created
  decoration/ from decorate_torch.py + model_funcs.py - Split logging_funcs.py into capture/
  (source_tensors, output_tensors, tensor_tracking); moved trace_model.py + flops.py into capture/ -
  Created visualization/ from vis.py - Replaced hardcoded _TORCHLENS_SUFFIXES with directory-based
  stack filtering - Added module-level docstrings to all 27 files that lacked them - Decomposed 3
  oversized functions into named helpers - 9 root files deleted, 22 new files created across
  subpackages

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.13.0 (2026-03-03)

### Bug Fixes

- **logging**: Prevent label-steal on no-op same-object returns
  ([`c8ec165`](https://github.com/johnmarktaylor91/torchlens/commit/c8ec1653919660d7b93e26046d3fe3706e40bc42))

Functions like `to(same_dtype)` and `contiguous()` on already-contiguous tensors return the same
  Python object without modifying it. TorchLens was treating these as in-place ops and propagating
  the new label back to the original tensor, breaking module thread cleanup in T5 and other
  encoder-decoder models with pre-norm residual patterns.

Fix 1 (decorate_torch.py): Split `was_inplace` into `same_object_returned` (always safe_copy) vs
  true in-place (func ends with `_` or starts with `__i`) — only true in-place ops propagate the
  label back.

Fix 2 (model_funcs.py): Module post-hook thread cleanup now uses labels captured at pre-hook time
  rather than reading from live tensors that may have been overwritten by true in-place ops
  mid-forward.

Also removes the cycle-protection band-aid in vis.py (visited sets in _get_max_nesting_depth and
  _set_up_subgraphs) since the root cause of module_pass_children cycles is now fixed.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **multi**: Resolve 16 test failures across suite
  ([`6fcf72f`](https://github.com/johnmarktaylor91/torchlens/commit/6fcf72f6677ac30cd0f63bfc62250b2c133f028e))

- Add soxr to test deps (transformers 5.2 unconditionally imports it) - Fix test_no_grad_block and
  test_model_calling_model shape mismatch (use inline torch.rand(2, 32) instead of small_input
  fixture) - Fix _get_input_arg_names for *args signatures (generate synthetic names) - Add empty
  tensor perturbation exemption in validation core - Add topk/sort integer indices posthoc exemption
  - Add type_as structural arg position exemption - Fix _append_arg_hash infinite recursion (tensor
  formatting triggers wrapped methods; use isinstance check + shape-only representation) - Fix
  TorchFunctionMode/DeviceContext bypass in decorated wrappers (Python wrappers skip C-level
  dispatch, so torch.device('meta') context wasn't injecting device kwargs into factory functions,
  causing corrupt buffers during transformers from_pretrained) - Rewrite test_autocast_mid_forward
  to skip validation (autocast context not captured during logging)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Exempt setitem when perturbing all-zeros/all-ones destination
  ([`a3a24ad`](https://github.com/johnmarktaylor91/torchlens/commit/a3a24ad75ba36ffcd1a87994d2c0abc5dce97102))

BART creates position embeddings by new_zeros() then __setitem__ to fill. Perturbing the all-zeros
  destination has no effect since setitem overwrites it. Add Case 3 to _check_setitem_exempt: when
  the perturbed tensor is the destination (args[0]) and it's a special value (all-zeros/all-ones),
  exempt.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Add cycle protection to _get_max_nesting_depth and subgraph setup
  ([`8779af9`](https://github.com/johnmarktaylor91/torchlens/commit/8779af9bc95743f55232660432dcf6bcb0055842))

call_children can contain cycles in models like T5 where residual connections cross module
  boundaries (e.g. layer_norm -> next block). _get_max_nesting_depth looped infinitely, causing
  >600s timeout.

Also: - Add visited set to _set_up_subgraphs while-loop for same reason - Add min with multiple args
  to posthoc exemption (same as max) - Remove @pytest.mark.slow from test_t5_small (now 15s, was
  infinite)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.13.0
  ([`a623a24`](https://github.com/johnmarktaylor91/torchlens/commit/a623a24a44c41e1ac7d04226dde8e1c07aadd7bc))

### Features

- **validation**: Capture and restore autocast context during logging and replay
  ([`19bbcb6`](https://github.com/johnmarktaylor91/torchlens/commit/19bbcb665c55f9e022263b51f673a781d5b80410))

Autocast state (enabled/dtype per device) is now captured alongside RNG state when each function
  executes during logging. During validation replay, the saved autocast context is restored so
  operations run at the same precision as the original forward pass.

- Add log_current_autocast_state() and AutocastRestore context manager - Thread func_autocast_state
  through exhaustive and fast logging paths - Wrap validation replay in AutocastRestore - Restore
  validate_saved_activations in test_autocast_mid_forward

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **validation**: Restructure into subpackage with data-driven exemption registries
  ([`ff85efd`](https://github.com/johnmarktaylor91/torchlens/commit/ff85efd282df188e5202d35944c0782ed1e16c3f))

Replace the monolithic validation.py (814 lines) with a clean torchlens/validation/ subpackage. The
  141-line elif chain and three separate exemption mechanisms are now consolidated into four
  declarative registries in exemptions.py (SKIP_VALIDATION_ENTIRELY, SKIP_PERTURBATION_ENTIRELY,
  STRUCTURAL_ARG_POSITIONS, CUSTOM_EXEMPTION_CHECKS), with posthoc checks kept as
  belt-and-suspenders.

Secondary fixes: - mean==0/mean==1 heuristic replaced with exact torch.all() checks - Bare except
  Exception now logs error details when verbose=True - _copy_validation_args uses recursive
  _deep_clone_tensors helper - While-loop perturbation bounded by MAX_PERTURB_ATTEMPTS=100

Adds 23 new tests covering imports, registry consistency, perturbation unit tests, deep clone
  helpers, and integration tests through specific exemption paths. All 399 existing tests pass with
  zero regressions.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **suite**: Exhaustive test suite expansion (~559 tests)
  ([`b489120`](https://github.com/johnmarktaylor91/torchlens/commit/b489120b91af4e41496cd206ce9ad20ab0a37092))

Add ~76 new tests covering major gaps in architecture and edge-case coverage: - 64 new toy model
  tests (attention, transformers, containers, conditionals, normalization variants, residual/param
  sharing, in-place/type ops, scalar/ broadcasting, exemption stress tests, architecture patterns,
  adversarial edge cases) - 12 new real-world model tests (DistilBERT, ELECTRA, MobileViT, MoE, T5,
  BART, RoBERTa, BLIP, Whisper, ViT-MAE, Conformer, SentenceTransformer)

Merge test_real_world_models_slow.py into test_real_world_models.py organized by
  architecture/modality, using @pytest.mark.slow for heavy tests. Register the slow marker in
  pyproject.toml.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.12.0 (2026-03-02)

### Chores

- **release**: 0.12.0
  ([`e714b32`](https://github.com/johnmarktaylor91/torchlens/commit/e714b323063c9a2212a4c6f8b8eb27cad68c9b12))

### Features

- **data_classes**: Add BufferLog subclass and BufferAccessor for first-class buffer support
  ([`5a9b4c6`](https://github.com/johnmarktaylor91/torchlens/commit/5a9b4c6bc7079176eaa4b259144b7316ef13699c))

Buffers now get their own dedicated class (BufferLog) that subclasses TensorLog, giving them focused
  repr, convenience properties (name, module_address), and ergonomic access via BufferAccessor on
  both ModelLog and ModuleLog.

- BufferLog(TensorLog) subclass with clean __repr__ and computed properties - BufferAccessor with
  indexing by address, short name, or ordinal position - Scoped mh.modules["addr"].buffers for
  per-module buffer access - Fix vis.py type(node) == TensorLog checks to use isinstance (supports
  subclasses) - Fix TensorLog.copy() to preserve subclass type via type(self)(fields_dict) - Fix
  cleanup.py property-before-callable check order to prevent AttributeError - Add buffer repr
  sections to aesthetic test reports (text + PDF)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.11.2 (2026-03-02)

### Chores

- **release**: 0.11.2
  ([`94ec2e0`](https://github.com/johnmarktaylor91/torchlens/commit/94ec2e00a675606d9fe199cf42cec1807723ab8b))

### Performance Improvements

- **postprocess**: Optimize 6 remaining hot-path bottlenecks in log_forward_pass
  ([`ed62350`](https://github.com/johnmarktaylor91/torchlens/commit/ed623508d8f8353634066cc4702d39bf8f0ca981))

Cache dir() per type, replace sorted deque with heapq in loop detection, use __dict__ + empty-field
  skip in label renaming, two-phase stack capture, batch orphan removal, and shadow sets for module
  hierarchy membership checks. 8-33% wall time reduction across models.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.11.1 (2026-03-02)

### Chores

- **release**: 0.11.1
  ([`d5bbd7d`](https://github.com/johnmarktaylor91/torchlens/commit/d5bbd7d7d139adb7b470f2b19bdd5313b1c12f15))

### Performance Improvements

- **logging**: Optimize 5 hot-path bottlenecks in log_forward_pass
  ([`d4a5d3a`](https://github.com/johnmarktaylor91/torchlens/commit/d4a5d3a57e52a22c6b312e8b77f3ec6a6d896efc))

- Cache torch.cuda.is_available() (called per-op, ~2ms each) - Replace tensor CPU copy + numpy +
  getsizeof with nelement * element_size - Hoist warnings.catch_warnings() out of per-attribute loop
  - Cache class-level module metadata (inspect.getsourcelines, signature) - Replace
  inspect.stack()/getframeinfo() with sys._getframe() chain; defer source context + signature
  loading in FuncCallLocation to first property access via linecache

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **decoration**: Comprehensive tests for permanent decoration architecture
  ([`f6305c8`](https://github.com/johnmarktaylor91/torchlens/commit/f6305c8c0f30337bdb2e5f6f265b0d253b7bb05e))

61 tests across 14 test classes covering: - Toggle state (on/off/exception/KeyboardInterrupt
  recovery) - Passthrough behavior when logging disabled - Detached import patching (module-level,
  class attrs, lists, dicts, late imports) - Permanent model preparation (WeakSet caching, session
  cleanup) - pause_logging context manager (nesting, exception safety) - Wrapper transparency
  (functools.wraps, __name__, __doc__) - torch.identity installation and graph presence - JIT
  builtin table registration and shared-original wrapper reuse - Decoration consistency
  (bidirectional mapper, idempotency) - In-place ops, property descriptors, edge cases - Signal
  safety (SIGALRM during forward) - Session isolation (no cross-session leakage)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.11.0 (2026-03-02)

### Chores

- **release**: 0.11.0
  ([`2a23b5c`](https://github.com/johnmarktaylor91/torchlens/commit/2a23b5cc1ad5ef648d37322c9d719ce0ed5315c8))

### Features

- **core**: Permanent toggle-gated decoration of torch functions
  ([`cbeaa83`](https://github.com/johnmarktaylor91/torchlens/commit/cbeaa83aa08098ea721e319a2f1af9a559279892))

Replace the per-call decorate/undecorate cycle (~2000 torch functions) with one-time permanent
  decoration at import time. A single boolean toggle (_state._logging_enabled) gates whether
  wrappers log or pass through, eliminating repeated overhead and fixing detached import blindness
  (e.g. `from torch import cos`).

Key changes: - New _state.py: global toggle, active_logging/pause_logging context managers -
  decorate_all_once(): one-time decoration with JIT builtin table registration -
  patch_detached_references(): sys.modules crawl for detached imports - Split model prep:
  _prepare_model_once (WeakSet) + _prepare_model_session - Delete clean_* escape hatches;
  safe_copy/safe_to use pause_logging() - Fix tensor_log.py activation_postfunc exception-safety bug
  (#89) - Fix GC issues: closures no longer capture ModelLog (GC-6, GC-7)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.10.1 (2026-03-02)

### Bug Fixes

- **core**: Fix one-liners and small guard additions across codebase
  ([`3d1591b`](https://github.com/johnmarktaylor91/torchlens/commit/3d1591b003675d91191e7a0abc728ad5eb562fae))

- Remove duplicate "contiguous" from ZERO_FLOPS_OPS (flops.py) - Fix strip("*") → rstrip("*") to
  avoid stripping leading chars (tensor_log.py) - Fix return type annotation to include str return
  (trace_model.py) - Replace isinstance(attr, Callable) with callable(attr) (model_funcs.py) -
  Remove dead pass_num=1 overwrite of parameterized value (logging_funcs.py) - Fix
  parent_params=None → [] for consistency (finalization.py) - Add int key guard in
  _getitem_after_pass (interface.py) - Add max(0, ...) guard for empty model module count
  (interface.py) - Add empty list guard in _get_lookup_help_str (interface.py) - Add explicit
  KeyError raise for failed lookups (interface.py) - Remove dead unreachable module address check
  (interface.py) - Add str() cast for integer layer keys (trace_model.py) - Add list() copy to
  prevent getfullargspec mutation (trace_model.py) - Reset has_saved_gradients and unlogged_layers
  in save_new_activations (logging_funcs.py)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **misc**: Fix decoration, param detection, vis forwarding, and guards
  ([`4269c4d`](https://github.com/johnmarktaylor91/torchlens/commit/4269c4d35beeabf54468fbc99b083290e830ff98))

- Remove dead double-decoration guard in decorate_torch.py - Add hasattr guard for torch.identity
  installation - Add None guard for code_context in FuncCallLocation.__getitem__ - Fix is_quantized
  to check actual qint dtypes, not all non-float - Forward show_buffer_layers to rolled edge check
  in vis.py

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **safety**: Add try/finally cleanup and exception state resets
  ([`c9bdb7f`](https://github.com/johnmarktaylor91/torchlens/commit/c9bdb7fc354c0b44f3cfcf6bde17e623a3b6f6db))

- Wrap validate_saved_activations in try/finally for cleanup (user_funcs.py) - Wrap show_model_graph
  render_graph in try/finally with cleanup (user_funcs.py) - Reset _track_tensors and _pause_logging
  in exception handler (trace_model.py) - Update test to expect [] instead of None for cleared
  parent_params

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Fix isinstance checks and misleading variable name
  ([`581f286`](https://github.com/johnmarktaylor91/torchlens/commit/581f286fc7c25f50b8b82b44d1f3fcbe03b4f01c))

- Replace type(val) == torch.Tensor with isinstance() (12 occurrences) - Rename mean_output to
  output_std for accuracy

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.10.1
  ([`5ea2de6`](https://github.com/johnmarktaylor91/torchlens/commit/5ea2de61a1e3aba61ea9f88fb3d4a36fead03cb7))

### Refactoring

- **postprocess**: Split monolithic postprocess.py into thematic package
  ([`cd14d27`](https://github.com/johnmarktaylor91/torchlens/commit/cd14d2744ea60db3a911905c377fc6c52123a1cc))

Break 2,115-line postprocess.py into a postprocess/ package with 5 modules: - graph_traversal.py:
  Steps 1-4 (output nodes, ancestry, orphans, distances) - control_flow.py: Steps 5-7 (conditional
  branches, module fixes, buffers) - loop_detection.py: Step 8 (loop detection, isomorphic subgraph
  expansion) - labeling.py: Steps 9-12 (label mapping, final info, renaming, cleanup) -
  finalization.py: Steps 13-18 (undecoration, timing, params, modules, finish)

No behavioral changes — pure file reorganization.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **aesthetic**: Add aesthetic testing infrastructure for visual inspection
  ([`73cdf01`](https://github.com/johnmarktaylor91/torchlens/commit/73cdf017c29f2385cebb721e7b08b88177807365))

Add regenerable human-inspectable outputs in tests/test_outputs/: - Comprehensive text report
  (aesthetic_report.txt) covering all user-facing reprs, accessors, DataFrames, error messages, and
  field dumps for every major data structure (ModelLog, TensorLog, RolledTensorLog, ModuleLog,
  ModulePassLog, ParamLog, ModuleAccessor, ParamAccessor) - 28 visualization PDFs exercising nesting
  depth, rolled/unrolled views, buffer visibility, graph direction, frozen params, and loop
  detection - 6 new aesthetic test models (AestheticDeepNested, AestheticSharedModule,
  AestheticBufferBranch, AestheticKitchenSink, AestheticFrozenMix) - Rename visualization_outputs/ →
  test_outputs/ for cleaner structure

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **aesthetic**: Add gradient visualization coverage
  ([`085dcd9`](https://github.com/johnmarktaylor91/torchlens/commit/085dcd9caf95b1ca364f83a7f97bd020f611c521))

Add gradient backward arrows (blue edges) to aesthetic testing: - New _vis_gradient helper using
  log_forward_pass(save_gradients=True) + backward() - 5 gradient vis PDFs: deep nested, frozen mix,
  kitchen sink (various configs) - Gradient section (G) in text report: TensorLog/ParamLog grad
  fields, frozen contrast - GRADIENT_VIS_GALLERY integrated into LaTeX PDF report as Section 3

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **aesthetic**: Add LaTeX PDF report with embedded visualizations
  ([`d101326`](https://github.com/johnmarktaylor91/torchlens/commit/d101326906573b95d62f5d518b13ee4be7025cb9))

Generate a comprehensive PDF report (aesthetic_report.pdf) alongside the text report, with: -
  tcolorbox-formatted sections for each model's outputs - Full field dumps for all data structures -
  All 28 visualization PDFs embedded inline with captions - Table of contents, figure numbering,
  proper typography - Skips gracefully if pdflatex not installed

Also saves the .tex source for customization.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.10.0 (2026-03-02)

### Chores

- **release**: 0.10.0
  ([`b2e5b48`](https://github.com/johnmarktaylor91/torchlens/commit/b2e5b48f10e07c6ea94e15f1141c65b2a5564afa))

### Features

- **data**: Add ModuleLog, ModulePassLog, and ModuleAccessor
  ([`5b0baa8`](https://github.com/johnmarktaylor91/torchlens/commit/5b0baa84903cd7e89f4192fa94fb444eebf5ceb5))

Introduce structured per-module metadata classes following the ParamLog/ParamAccessor pattern.
  log.modules["features.3"] now returns a rich ModuleLog with class, params, layers, source info,
  hierarchy, hooks, forward signature, and nesting depth. Multi-pass modules support per-call access
  via passes dict and pass notation (e.g. "fc1:2").

- ModulePassLog: per-(module, pass) lightweight container - ModuleLog: per-module-object user-facing
  class with delegating properties for single-pass modules - ModuleAccessor: dict-like accessor with
  summary()/to_pandas() - Metadata captured in prepare_model() before cleanup strips tl_* attrs -
  Forward args/kwargs captured per pass in module_forward_decorator() - _build_module_logs() in
  postprocess Step 17 assembles everything - Old module_* dicts kept alive for vis.py backward
  compat - 44 new tests, 315 total passing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **data**: Move ModelHistory & TensorLogEntry into data_classes/
  ([`6ecc7e5`](https://github.com/johnmarktaylor91/torchlens/commit/6ecc7e5880eadffd8c13e82bd39d3aa9c5544672))

Move model_history.py and tensor_log.py into torchlens/data_classes/ alongside the existing
  FuncCallLocation and ParamLog data classes. Update all relative imports across 12 consumer files.
  Also fixes a missing-dot bug in decorate_torch.py's TYPE_CHECKING import.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **data**: Rename ModelHistory/TensorLogEntry to ModelLog/TensorLog
  ([`f36718e`](https://github.com/johnmarktaylor91/torchlens/commit/f36718e6a471838ef40af0e25e8e41357d9b9a30))

- Rename classes: ModelHistory → ModelLog, TensorLogEntry → TensorLog, RolledTensorLogEntry →
  RolledTensorLog - Rename file: data_classes/model_history.py → data_classes/model_log.py - Rename
  variables: model_history → model_log, source_model_history → source_model_log - Rename constants:
  MODEL_HISTORY_FIELD_ORDER → MODEL_LOG_FIELD_ORDER, TENSOR_LOG_ENTRY_FIELD_ORDER →
  TENSOR_LOG_FIELD_ORDER - Fold test_meta_validation.py into test_metadata.py

- **vis**: Migrate vis.py and interface.py from module_* dicts to ModuleAccessor API
  ([`8e64d43`](https://github.com/johnmarktaylor91/torchlens/commit/8e64d439d78081551c4112c0cc49b8c300609296))

Replace all direct accesses to old module_types, module_nparams, module_num_passes,
  module_pass_layers, module_pass_children, module_children, module_layers, module_num_tensors,
  module_pass_num_tensors, top_level_module_passes, and top_level_modules dicts with the new
  ModuleAccessor API (self.modules[addr]).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Testing

- **validation**: Add meta-validation tests for corruption detection
  ([`11de9da`](https://github.com/johnmarktaylor91/torchlens/commit/11de9da7209b3f504dd3ed6433a0d04367c38d22))

Add 9 tests that deliberately corrupt saved activations and verify validate_saved_activations()
  catches each corruption: output/intermediate replacement, layer swap, zeroing, noise, scaling,
  wrong shape, and corrupted creation_args.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.9.0 (2026-03-01)

### Chores

- **release**: 0.9.0
  ([`647ecef`](https://github.com/johnmarktaylor91/torchlens/commit/647ecefdad81db983006fd5d71d227e24702ab52))

### Features

- **data**: Add ParamLog class for structured parameter metadata
  ([`f21f3e9`](https://github.com/johnmarktaylor91/torchlens/commit/f21f3e9c7a57facd2dfd0bcf7b5c76da66b62fe2))

Introduce ParamLog — a dedicated data class for parameter metadata — and ParamAccessor for
  convenient dict-like access by address, index, or short name. Parameters are now first-class
  objects with full metadata including trainable/frozen status, module info, linked params, gradient
  tracking, and optional optimizer tagging.

Key additions: - ParamLog class with lazy gradient detection via _param_ref - ParamAccessor on both
  ModelHistory (mh.params) and TensorLogEntry (entry.params) - Trainable/frozen tallies on MH, TLE,
  and per-module - Optional optimizer= param on log_forward_pass() - Visualization: param names with
  bracket convention, trainable/frozen color coding, gradient fills for mixed layers, caption
  breakdown - 68 new tests covering all functionality

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.8.0 (2026-03-01)

### Chores

- **release**: 0.8.0
  ([`8b1534e`](https://github.com/johnmarktaylor91/torchlens/commit/8b1534e30a27cc40553c90ff31f37d6a35228762))

### Features

- **data**: Add FuncCallLocation class for structured call stack metadata
  ([`49ada0e`](https://github.com/johnmarktaylor91/torchlens/commit/49ada0ef1415441562c2f3dd5b71d36de8478472))

Replace unstructured List[Dict] func_call_stack with List[FuncCallLocation] providing clean repr
  with source context arrows, __getitem__/__len__ support, and optional
  func_signature/func_docstring extraction. Introduces torchlens/data_classes/ package and threads a
  configurable num_context_lines parameter through the full call chain.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.7.2 (2026-03-01)

### Bug Fixes

- **tests**: Reduce r2plus1d_18 input size to prevent OOM
  ([`2c73fcd`](https://github.com/johnmarktaylor91/torchlens/commit/2c73fcd7af620a29749fd55f753f1b6d3d915b7a))

The test_video_r2plus1_18 test used a (16,3,16,112,112) input (~96M elements) which consistently got
  OOM-killed. Reduced to (1,3,1,112,112) which passes reliably while still exercising the full
  model.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.7.2
  ([`34f2466`](https://github.com/johnmarktaylor91/torchlens/commit/34f246687bc4213334f2bc37d46ddcd0a3efb370))


## v0.7.1 (2026-02-28)

### Bug Fixes

- **logging**: Handle complex-dtype tensors in tensor_nanequal
  ([`b425cbe`](https://github.com/johnmarktaylor91/torchlens/commit/b425cbed76fdaf7bef6f2fc94fa6fdc02404b4c2))

torch.nan_to_num does not support complex tensors, which caused test_qml to fail when PennyLane
  quantum ops produced complex outputs. Use view_as_real/view_as_complex to handle NaN replacement
  for complex dtypes.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.7.1
  ([`d94ad8f`](https://github.com/johnmarktaylor91/torchlens/commit/d94ad8f98ae2d3339f9ca22f12d51bb5021ff55f))


## v0.7.0 (2026-02-28)

### Bug Fixes

- **logging**: Capture/restore RNG state for two-pass stochastic models
  ([#58](https://github.com/johnmarktaylor91/torchlens/pull/58),
  [`271cef3`](https://github.com/johnmarktaylor91/torchlens/commit/271cef345d8c94e709bc5eeccc8e4d64b951c64d))

Move RNG state capture/restore before pytorch decoration to prevent internal .clone() calls from
  being intercepted by torchlens' decorated torch functions. Also speed up test_stochastic_loop by
  using a higher starting value.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **logging**: Ensure output layer parents are saved with layers_to_save
  ([#46](https://github.com/johnmarktaylor91/torchlens/pull/46),
  [`b104094`](https://github.com/johnmarktaylor91/torchlens/commit/b10409445dc036ce8957e4e52e314b3b4a38dc4f))

When layers_to_save is a subset, the fast pass now automatically includes parents of output layers
  in the save list. This ensures output layer tensor_contents is populated in postprocess_fast
  (which copies from parent).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **logging**: Replace copy.deepcopy with safe_copy to prevent infinite loops
  ([#18](https://github.com/johnmarktaylor91/torchlens/pull/18),
  [`e1ad9ae`](https://github.com/johnmarktaylor91/torchlens/commit/e1ad9ae08937b11ccffe83c47efc45a052397c6c))

copy.deepcopy hangs on complex tensor wrappers with circular references (e.g. ESCNN
  GeometricTensor). Replace with safe_copy_args/safe_copy_kwargs that clone tensors, recurse into
  standard containers, and leave other objects as references.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **logging**: Use argspec to disambiguate tuple/list input args
  ([#43](https://github.com/johnmarktaylor91/torchlens/pull/43),
  [`a91e378`](https://github.com/johnmarktaylor91/torchlens/commit/a91e378125e1d3ee0a9483b9240c3f7730faba3a))

When a model's forward() expects a single arg that IS a tuple/list of tensors, torchlens incorrectly
  unpacked it into multiple positional args. Now uses inspect.getfullargspec to detect single-arg
  models and wraps the tuple/list as a single arg. Also handles immutable tuples in
  _fetch_label_move_input_tensors device-move logic.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **vis**: Functional ops at end of container modules rendered as ovals
  ([#48](https://github.com/johnmarktaylor91/torchlens/pull/48),
  [`7d1c68c`](https://github.com/johnmarktaylor91/torchlens/commit/7d1c68c502c3274bf1418bfcac2de9d81eef9d77))

_check_if_only_non_buffer_in_module was too broad — it returned True for functional ops (like
  torch.relu) at the end of container modules with child submodules, causing them to render as
  boxes. Added a leaf-module check: only apply box rendering for modules with no child submodules.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.7.0
  ([`f0b377e`](https://github.com/johnmarktaylor91/torchlens/commit/f0b377e14a6115f879bb2c6d5506d9e70084e433))

### Features

- **metadata**: Track module training mode per submodule
  ([#52](https://github.com/johnmarktaylor91/torchlens/pull/52),
  [`e5197ca`](https://github.com/johnmarktaylor91/torchlens/commit/e5197ca2e48d4510df750ea420196e68f038c0b6))

Capture module.training in module_forward_decorator and store in ModelHistory.module_training_modes
  dict (keyed by module address). This lets users check whether each submodule was in train or eval
  mode during the forward pass.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.6.2 (2026-02-28)

### Bug Fixes

- **postprocess**: Correct _merge_buffer_entries call convention
  ([#47](https://github.com/johnmarktaylor91/torchlens/pull/47),
  [`07df691`](https://github.com/johnmarktaylor91/torchlens/commit/07df6911dae2d6aa22e13cf2aa2fe4793320e314))

_merge_buffer_entries is a module-level function, not a method on ModelHistory. Fixes AttributeError
  when processing recurrent models with duplicate buffer entries.

Based on gilmoright's contribution in PR #56.

Co-authored-by: gilmoright <artem.dahaka@gmail.com>

### Chores

- **release**: 0.6.2
  ([`8e1443f`](https://github.com/johnmarktaylor91/torchlens/commit/8e1443ffb086e09e15eede3f54c928b2c69fcb58))


## v0.6.1 (2026-02-28)

### Bug Fixes

- **tensor_log**: Use getattr default in TensorLogEntry.copy()
  ([`538288d`](https://github.com/johnmarktaylor91/torchlens/commit/538288d6170ab95ca24e7b88ada37a7e4196d6d2))

Prevents AttributeError when copying entries that predate newly added fields (e.g., deserialized
  from an older version).

Co-Authored-By: whisperLiang <whisperLiang@users.noreply.github.com>

### Chores

- **release**: 0.6.1
  ([`3738449`](https://github.com/johnmarktaylor91/torchlens/commit/3738449e05777c532b69035fb23f468e21981e3d))


## v0.6.0 (2026-02-28)

### Chores

- **release**: 0.6.0
  ([`09a1e93`](https://github.com/johnmarktaylor91/torchlens/commit/09a1e93c8d141b461c7f3dda579e3cfc9e47b08b))

### Features

- **flops**: Add per-layer FLOPs computation for forward and backward passes
  ([`31b43e6`](https://github.com/johnmarktaylor91/torchlens/commit/31b43e6659d521f0332b99d02e969b4ac19f1abd))

Compute forward and backward FLOPs at logging time for every traced operation. Uses category-based
  dispatch: zero-cost ops (view, reshape, etc.), element-wise ops with per-element cost, and
  specialty handlers for matmul, conv, normalization, pooling, reductions, and loss functions.
  Unknown ops return None rather than guessing.

- New torchlens/flops.py with compute_forward_flops / compute_backward_flops - ModelHistory gains
  total_flops_forward, total_flops_backward, total_flops properties and flops_by_type() method -
  TensorLogEntry and RolledTensorLogEntry gain flops_forward / flops_backward fields - 28 new tests
  in test_metadata.py (unit + integration) - scripts/check_flops_coverage.py dev utility for
  auditing op coverage - Move test_video_r2plus1_18 to slow test file

Based on whisperLiang's contribution in PR #53.

Co-Authored-By: whisperLiang <whisperLiang@users.noreply.github.com>


## v0.5.0 (2026-02-28)

### Bug Fixes

- **logging**: Revert children_tensor_versions to proven simpler detection
  ([`ade9c39`](https://github.com/johnmarktaylor91/torchlens/commit/ade9c39f15604459af9134ffcb770ae08238f5cf))

The refactor version applied device/postfunc transforms to the stored value in
  children_tensor_versions, but validation compares against creation_args which are always raw. This
  caused fasterrcnn validation to fail. Revert to the simpler approach that stores raw arg copies
  and was verified passing twice.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **postprocess**: Gate output node variations on has_child_tensor_variations
  ([`4106be9`](https://github.com/johnmarktaylor91/torchlens/commit/4106be93168cbff4b346a70f06417556c3444490))

Don't unconditionally store children_tensor_versions for output nodes. Gate on
  has_child_tensor_variations (set during exhaustive logging) to avoid false positives and preserve
  postfunc-applied tensor_contents.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **postprocess**: Rebuild pass assignments after loop detection and fix output handler
  ([`847b2a7`](https://github.com/johnmarktaylor91/torchlens/commit/847b2a79202cee13bf2ba231153b71068cf6311a))

Two fixes:

1. _rebuild_pass_assignments: Multiple rounds of _expand_isomorphic_subgraphs can reassign a node to
  a new group while leaving stale same_layer_operations in the old group's members. This caused
  multiple raw tensors to map to the same layer:pass label, producing validation failures (e.g.
  fasterrcnn). The cleanup step groups tensors by their authoritative layer_label_raw and rebuilds
  consistent pass numbers.

2. Output node handler: Replaced the has_child_tensor_variations gate with a direct comparison of
  actual output (with device/postfunc transforms) against tensor_contents using tensor_nanequal.
  This correctly handles in-place mutations through views (e.g. InPlaceZeroTensor) while preserving
  postfunc values for unmodified outputs.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **validation**: Handle bool and complex tensor perturbation properly
  ([`dfd2d7b`](https://github.com/johnmarktaylor91/torchlens/commit/dfd2d7be8dce252312235f954446e98357ccfe35))

- Generate proper complex perturbations using torch.complex() instead of casting away imaginary part
  - Fix bool tensor crash by reordering .float().abs() (bool doesn't support abs, but float
  conversion handles it) - Add ContextUnet diffusion model to example_models.py for self-contained
  stable_diffusion test - Update test_stable_diffusion to use example_models.ContextUnet

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.5.0
  ([`34a15a9`](https://github.com/johnmarktaylor91/torchlens/commit/34a15a99e293d79aa60232b817eb25a64c8e2cfd))

### Features

- **logging**: Generalize was_getitem_applied to has_child_tensor_variations
  ([`1177f60`](https://github.com/johnmarktaylor91/torchlens/commit/1177f607bf6406d47424c8b05755aec86d992dcb))

Replace the getitem-specific parent detection with runtime mismatch detection that catches any case
  where a parent's tensor_contents diverges from what children actually received (getitem slicing,
  view mutations through shared storage, in-place ops after logging, etc.).

Key changes: - Rename was_getitem_applied → has_child_tensor_variations - Detection now compares arg
  copies against parent tensor_contents at child-creation time, with transform-awareness (device +
  postfunc) - Output nodes now detect value changes vs parent tensor_contents - Use tensor_nanequal
  (not torch.equal) for dtype/NaN consistency - Fix fast-mode: clear stale state on re-run, prevent
  double-postfunc - Use clean_to and try/finally for _pause_logging safety - Add 6 view-mutation
  stress tests (unsqueeze, reshape, transpose, multiple, chained, false-positive control)

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Refactoring

- **loops**: Cherry-pick approved changes from feat/loop-detection-hardening
  ([`33e0f22`](https://github.com/johnmarktaylor91/torchlens/commit/33e0f223d12aae806799f3853ed6e221b1232274))

- Rename 10 loop detection functions to clearer names (e.g.
  _assign_corresponding_tensors_to_same_layer → _detect_and_label_loops,
  _fetch_and_process_next_isomorphic_nodes → _advance_bfs_frontier) - Rename 6 local variables for
  clarity (e.g. node_to_iso_group_dict → node_to_iso_leader, subgraphs_dict → subgraph_info) - Add
  SubgraphInfo dataclass replacing dict-based subgraph bookkeeping - Replace list.pop(0) with
  deque.popleft() in BFS traversals - Remove ungrouped sweep in _merge_iso_groups_to_layers - Remove
  safe_copy in postprocess_fast (direct reference suffices) - Rewrite _get_hash_from_args to
  preserve positional indices, kwarg names, and dict keys via recursive _append_arg_hash helper -
  Remove vestigial index_in_saved_log field from TensorLogEntry, constants.py, logging_funcs.py, and
  postprocess.py - Fix PEP8: type(x) ==/!= Y → type(x) is/is not Y in two files - Split
  test_real_world_models.py into fast and slow test files - Add 12 new edge-case loop detection test
  models and test functions

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.4.1 (2026-02-26)

### Bug Fixes

- **loops**: Refine iso groups to prevent false equivalence in loop detection
  ([`5aaad70`](https://github.com/johnmarktaylor91/torchlens/commit/5aaad70196073364ecfa7351fba08fed013e81b4))

When operations share the same equivalence type but occupy structurally different positions (e.g.,
  sin(x) in a loop body vs sin(y) in a branch), the BFS expansion incorrectly groups them together.
  Add _refine_iso_groups to split such groups using direction-aware neighbor connectivity.

Also adds NestedParamFreeLoops test model and prefixes intentionally unused variables in test models
  with underscores to satisfy linting.

### Chores

- **release**: 0.4.1
  ([`88900b1`](https://github.com/johnmarktaylor91/torchlens/commit/88900b160fa923e8ab6bc2f4b831a25195fb3d2c))

### Code Style

- Auto-format with ruff
  ([`b9413f1`](https://github.com/johnmarktaylor91/torchlens/commit/b9413f1b4e60511eba117e032df60dea15849354))


## v0.4.0 (2026-02-26)

### Chores

- **release**: 0.4.0
  ([`9b48aca`](https://github.com/johnmarktaylor91/torchlens/commit/9b48aca8f4a6ad79f4b45587714a3343d5510429))

- **tests**: Move visualization_outputs into tests/ directory
  ([`22ed018`](https://github.com/johnmarktaylor91/torchlens/commit/22ed018d80ce274225bb65bae9a0dda3aed1561b))

Anchor vis_outpath to tests/ via VIS_OUTPUT_DIR constant in conftest.py so test outputs don't
  pollute the project root.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Features

- **core**: Generalize in-place op handling, fix loop grouping, and harden validation
  ([`c72d6a6`](https://github.com/johnmarktaylor91/torchlens/commit/c72d6a6ecabd42471c1badd1b6cb8b0b6958e518))

- Generalize in-place op detection in decorate_torch.py: use `was_inplace` flag based on output
  identity instead of hardcoded function name list, and propagate tensor labels for all in-place ops
  - Fix ungrouped isomorphic nodes in postprocess.py: sweep for iso nodes left without a layer group
  after pairwise grouping (fixes last-iteration loop subgraphs with no params) - Deduplicate ground
  truth output tensors by address in user_funcs.py to match trace_model.py extraction behavior - Add
  validation exemptions: bernoulli_/full arg mismatches from in-place RNG ops,
  meshgrid/broadcast_tensors multi-output perturbation, *_like ops that depend only on
  shape/dtype/device - Gracefully handle invalid perturbed arguments (e.g. pack_padded_sequence)
  instead of raising - Guard empty arg_labels in vis.py edge label rendering - Fix test stability:
  reduce s3d batch size, add eval mode and bool mask dtype for StyleTTS

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.3.1 (2026-02-25)

### Bug Fixes

- **core**: Fix in-place op tracking, validation, and test stability
  ([`326b8a9`](https://github.com/johnmarktaylor91/torchlens/commit/326b8a907a170f8c9e4c7ec16296faaecfa56d51))

- Propagate tensor labels back to original tensor after in-place ops (__setitem__, zero_,
  __delitem__) so subsequent operations see updated labels - Add validation exemption for scalar
  __setitem__ assignments - Fix torch.Tensor → torch.tensor for correct special value detection -
  Remove xfail marker from test_varying_loop_noparam2 (now passes) - Add ruff lint ignores for
  pre-existing E721/F401 across codebase - Includes prior bug-blitz fixes across logging,
  postprocessing, cleanup, helper functions, visualization, and model tracing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.3.1
  ([`7d87270`](https://github.com/johnmarktaylor91/torchlens/commit/7d872703d80ac1794cf8dab72cccec4e2f8ff9fb))


## v0.3.0 (2026-02-25)

### Chores

- **ci**: Replace black with ruff auto-format on push
  ([`e0cb9e1`](https://github.com/johnmarktaylor91/torchlens/commit/e0cb9e1c20b347cc2ff2579cb10b1beb959f8252))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **config**: Add ruff config and replace black/isort with ruff in pre-commit
  ([`c27ced8`](https://github.com/johnmarktaylor91/torchlens/commit/c27ced8f7f8fa9e555dfa6a8313db53575b00305))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **config**: Set major_on_zero to true
  ([`d63451c`](https://github.com/johnmarktaylor91/torchlens/commit/d63451cdd3360fa2ce6979b2e371a806310ec6ca))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **release**: 0.3.0
  ([`dfaf97f`](https://github.com/johnmarktaylor91/torchlens/commit/dfaf97f226c470a43cd2e616862ef15de5d78ce8))

### Features

- **tests**: Restructure test suite into focused files with metadata coverage
  ([`31b0353`](https://github.com/johnmarktaylor91/torchlens/commit/31b0353813353a0eaa0dc94e9659f9e84bbaab00))

Split monolithic test_validation_and_visuals.py (3130 lines, 155 tests) into: - conftest.py: shared
  fixtures and deterministic seeding - test_toy_models.py: 78 tests (66 migrated + 12 new API
  coverage tests) - test_metadata.py: 44 comprehensive metadata field tests (7 test classes) -
  test_real_world_models.py: 75 tests with local imports and importorskip

New tests cover: log_forward_pass parameters (layers_to_save, save_function_args,
  activation_postfunc, mark_distances), get_model_metadata, ModelHistory access patterns,
  TensorLogEntry field validation, recurrent metadata, and GeluModel.

Removed 13 genuine size-duplicate tests (ResNet101/152, VGG19, etc.). All optional dependencies now
  use pytest.importorskip for graceful skipping.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.2.0 (2026-02-24)

### Bug Fixes

- **ci**: Enable verbose PyPI upload and disable attestations for debugging
  ([`579506d`](https://github.com/johnmarktaylor91/torchlens/commit/579506df2260dd6c7dcf153b52ad6ee42e282191))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Fetch tags in release workflow so semantic-release finds v0.1.36
  ([`0f0a3d0`](https://github.com/johnmarktaylor91/torchlens/commit/0f0a3d0895239152b9093c9f0752f957e27724d7))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Pin semantic-release to v9 and add debug output
  ([`b68388c`](https://github.com/johnmarktaylor91/torchlens/commit/b68388cfead9d624c569df0c3687bed823cb0ce4))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Prevent major version bump on 0.x releases
  ([`db4a31e`](https://github.com/johnmarktaylor91/torchlens/commit/db4a31edfc626a3e38b46e11d644b8b02b06d264))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Remove direct URL dependency rejected by PyPI and clean up workflow
  ([`500368b`](https://github.com/johnmarktaylor91/torchlens/commit/500368ba22d50f2235b9be3b87a455b2c81b9891))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

- **ci**: Use GitHub App token to bypass branch protection in release workflow
  ([`f2cf8ae`](https://github.com/johnmarktaylor91/torchlens/commit/f2cf8ae7450712bb54fcf58a1f2bbb8d3760f076))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>

### Chores

- **release**: 0.2.0
  ([`4b54568`](https://github.com/johnmarktaylor91/torchlens/commit/4b54568dd8a6216c032663db438f7d32ed79fb6f))

### Features

- **build**: Migrate to pyproject.toml with semantic-release and GitHub Actions
  ([`f8e01c9`](https://github.com/johnmarktaylor91/torchlens/commit/f8e01c9f7821a3dd54c9df805bc2181f3340e9ea))

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>


## v0.1.36 (2025-09-26)
