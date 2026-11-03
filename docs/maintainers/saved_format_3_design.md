# TorchLens 3.0 saved format: the design

The design behind `torchlens/_io/format3/`: what the 3.0 artifact contains, how it is read and
written, what leaves the code base with the pickle carrier, and what it is expected to cost.
The constants are `torchlens/_io/format3/contract.py`; the prose contract is
`saved_format_3.md` beside this page, and this page says where it extends that contract.
Maintainer decisions it implements (31 Oct and 1 Nov 2026): drop pickle for typed JSON plus
safetensors, with unencodable user objects stringified under a disclosure; save the
intervention recipe with the trace and refuse only callables (io.1); persist the derived
columns at the portable level (io.2); key phase timings by pipeline step name (io.5); add
`Op.is_constant` (backends.2); keep the save-time reload check and the checked load (late.21).

## 0. Summary

Pickle carries exactly one thing today: `metadata.pkl`, the scrubbed Trace state dict with
record objects as pickle globals (plus a legacy `kind=bundle` branch in `bundle_container.py`).
Everything else in a tlspec 11 artifact is already typed JSON with its own decoder: the
manifest, the sparse runnable descriptor, `bundle.json`, the intervention `spec.json`, merged
artifacts. The 3.0 format therefore replaces one carrier and the machinery that existed only
because that carrier was pickle: the canonical pickler, the restricted unpickler and its
opcode prescan, the dry-run reload, the pickle-global rename table, the persisted anchors, the
state-key shadow filter, the pre-3 module-path probe, and the forgery validators that re-derive
a grammar the decoder can enforce from a schema.

What replaces it: per-record JSON tables (`records/<record>.json`, columns plus rows) whose
column set is the bindings' `persisted_field_names`, a closed typed-value vocabulary
(`$tl`-tagged objects) for everything JSON cannot carry, one schema-directed decoder that
builds records without `eval`, `getattr`-by-name or import-by-name (torch dtypes come from a
table), chunked safetensors files for every tensor payload (one key per tensor, opaque keys,
chunks capped so streaming saves append by opening a new chunk), and a disclosure object for
any user value the encoder cannot express. The runnable component keeps its frozen descriptor
spellings and moves carrier only. The intervention recipe is saved beside the trace (io.1) with
a callable placeholder that refuses rerun and nothing else.

Expected effect on the perf-profile rows: save 40 to 55 percent faster, load 35 to 50 percent
faster (section 12). About 16,450 lines of `torchlens/_io/` leave or shrink to a door, about
4,050 arrive (section 10); the layering ceilings fall and cannot rise (section 13).

## 1. The layer

One package, `torchlens/_io/format3/`. Nine modules plus `__init__`, every one under the
800-line file cap (`scripts/checks/file_cap.py`, no new allowlist row). Nine is the smallest count that keeps each under the cap and gives the three
build streams disjoint write territories (writer: `writer.py`, `payloads.py`, `values.py`,
`schema.py`; reader: `decode.py`, `reader.py`, `validate.py`; runnable: `runnable.py`).

| Module | Holds | Budget (lines) | Stream |
| --- | --- | ---: | --- |
| `contract.py` | constants (exists, contract stream): envelope names, tags, codecs, refusal table | 250 | contract (landed) |
| `schema.py` | per-record schema derived at import from `STORAGE_BINDINGS` (`_data_classes/_schema_bindings.py`), field annotations and the writer contract's `component_specs`; nested-family sub-schemas (section 9); schema digest | 600 | writer |
| `values.py` | the typed-value codec: Python value to tagged JSON and back, both directions in one module so tags stay symmetric; dtype table; disclosure builder | 450 | writer |
| `decode.py` | the decoder half of the value codec: tagged JSON back to Python values through the closed vocabulary, field and name checks per tag, `record` construction; split from `values.py` because the encoder and its tables fill that module's budget | 400 | reader |
| `_families.py` | the generic applier of the family grammar nodes in `schema.FAMILIES` (section 9): node shapes, closed vocabularies and within-cell constraints checked on each decoded cell before record construction; split from `decode.py` for its budget | 400 | reader |
| `payloads.py` | safetensors chunk writer and reader with opaque keys; `LazyActivationRef` rebuilt on (chunk, key) | 450 | writer |
| `writer.py` | `write_trace`, `write_bundle`, `write_intervention_recipe`: include-flag resolution (reuses `save_policy.py`), privacy and barcode policies carried over from `scrub.py`, record-table emission, manifest, reload check, atomic publish | 750 | writer |
| `recipe.py` | the intervention recipe (section 8): rule and value encoding with callable placeholders at rule level and in edit arguments, recipe tensors on the shared key counter, typed rebinding at load; `write_intervention_recipe` and the trace writer call it | 700 | writer |
| `reader.py` | `read_manifest`, `validate_artifact`, `load_trace`, `load_bundle`, `load_intervention_recipe`: manifest first, digests, record tables through the decoder, record construction, accessor rebuild, grouping settlement | 750 | reader |
| `route.py` | the `bundle.load` switch into the 3.0 format: the root test and the door dispatch, kept outside `bundle.py` and its import cycle | 100 | reader |
| `validate.py` | cross-record invariants after decode (section 9): site keys, structure-only coherence, checkpoint windows, edge carriers, graph structure, kernel relations, identity facts, injection provenance, evidence facts | 600 | reader |
| `runnable.py` | thin adapter: descriptor JSON and state chunks under `runnable/`; delegates to `_io/runnable.py` (producer) and `_io/runnable_load.py` (loader), which keep their frozen contracts | 200 | runnable |
| `runnable_families.py` | Fixed runnable family names, shared chunk writer, checked tensor references |
| `container.py` | Bundle members, manifest bindings and relations |

Public doors do not move: `tl.load`, `Trace.save`, `Bundle.save`, `tl.io`
(`torchlens/io/__init__.py`), `Trace.save_intervention`. `_io/bundle.py` becomes the dispatch
behind those doors (section 10 says what of it survives). `_io/format3` imports only
`_data_classes`, `_io.format_errors`, `_io._json`, `_io.save_policy`, `_io.paths`,
`_io._durability`, `_io._source_privacy`, `_io.payload_codec` and `utils/_torch_compat`;
never the `torchlens` top package, `_runnable` (the runnable adapter imports it inside the
two functions that need it, as `_io/runnable.py` does today) or any backend package.

Rejected: (a) rewriting in place inside `bundle.py`/`scrub.py`/`rehydrate.py`: the pickle
assumptions are load-bearing in all three (object graph walk, setstate hooks, anchors) and the
in-place edit would keep every pickle test alive during the build; (b) one `format3.py`
module: over the cap by a factor of four; (c) four modules: `writer.py` and `reader.py` would
each pass 1,500 lines once the scrub policies and the invariants move in.

## 2. Envelope

Directory artifact (the `.tlspec` suffix stays a convention, not a requirement). Names are the
contract's constants and the contract page's section (a), adopted as written.

```
<name>.tlspec/
  manifest.json          required; read first; nothing else is opened before it validates
  records/<record>.json  one file per record kind present (twelve kinds, RECORD_INDEX)
  blobs/<codec>-<nnnn>.safetensors   tensor payloads, chunked (section 6)
  annotations.json       optional (contract page (a))
  intervention.json      io.1: present when the trace was captured with an intervention
  runnable/descriptor.json           runnable level: the sparse descriptor (content frozen)
  runnable/state_dict-<nnnn>.safetensors, runnable/buffers.safetensors
  members/<index>-<name>/...         kind=bundle: one trace artifact per member; relations.json
```

Manifest: the contract's field list (`format`, `format_version` 3, `tlspec_version`, `kind`,
writer identity, `backend`, `backend_runtime`, `model_signature`, `model_fingerprint`,
`save_level`, `payload_policy`, `contract_digest`, `codecs`, `records` as
`{record: {path, rows, sha256, bytes}}`, `files` as `{relpath: {sha256, bytes}}`, optional
`runnable` and `intervention` objects). This design adds one optional object:

| Key | Value |
| --- | --- |
| `disclosures` | `{count, first: [up to 16 "at" paths]}` of `unencodable` values written (section 5); absent when the count is zero |

The `files` digests are the single integrity root (they replace `bundle_metadata.py`'s
metadata digest and `_artifact_anchors.py`'s manifest-vs-metadata anchors); a listed file whose
digest or length disagrees refuses `bundle_metadata_integrity_refused` before parse; unlisted
files are never read. Blob verification is eager by default (as today) and deferred only for
`lazy=True` reads of tensors that are never materialized. The manifest has no self-digest; the
save-time reload check (section 9) is its check.

Rejected: a single `artifact.json` holding manifest plus records (loses per-record digests and
lazy column skipping; 100 MB JSON parses are slow and un-streamable); YAML or TOML (no gain,
new decoder surface); zip container (atomic rename of a directory with fsync is already what
`_durability` does; a zip would make streaming appends a rewrite); a `schema_digest` beside
`contract_digest` (redundant: the writer contract already digests the field policies and
component specs the schemas derive from).

## 3. Record tables

The contract page (b), adopted: `records/trace.json` is one object
`{"record": "trace", "tlspec_version": 11, "fields": {key: value}}`; every other kind is
columnar:

```json
{"record": "op", "tlspec_version": 11, "rows": 1200,
 "columns": {"label": [...], "func_name": [...], ...}}
```

- The column set is `persisted_field_names(record)` from `_io/field_registry.py` (the
  bindings), filtered by level: a `required` key is a column at every level; an `optional` key
  with a payload flag is a column iff that flag is on in `payload_policy`; a forbidden column
  refuses (`record_schema_invalid`); absent optional columns are absent, never
  null-filled. The reader refuses a column it does not know (`unknown_persisted_field`).
  Within the 3 major the one forward-compatibility rule is: a key the schema marks as added
  after the file's `tlspec_version` may be absent (this is how the writer stream's tlspec 12
  columns read tlspec 11 files during the build).
- Every column array has length `rows`; rows are in live container order (ops by execution
  order, layers by label order, modules by address order). `trace` has one row (the object
  form); record kinds with no instances have no file and no `records` entry.
- Cross-record references are the label and address strings the live records hold today
  (`parent_ops`, `module_address`, site keys); no row ids are introduced. EDGE and BITSET
  columns are therefore `id_array`: arrays of those strings.
- Each cell is encoded by `StorageKind` (`JSON_TYPE_BY_STORAGE_KIND`): SCALAR, INTERNED,
  GROUP and COMPUTED as `value` (section 4; INTERNED writes the string, GROUP writes the member
  list the accessor exposes, the store layout is physical); EDGE and BITSET as `id_array`;
  PAYLOAD as `tensor_ref` (a tensor reference or a container of them); RUNTIME never written.
  Fields whose policy is `DROP` are not columns.
- Nested record objects held by a field (for example `dict[int, BackwardPass]` on `Trace`)
  are the referencing strings or indices the live field holds, pointing into that record's
  table, never inline copies. Nested value objects (the writer contract's `component_specs`
  classes that are not records: `Duration`, `HookInfo`, `ModuleInputSnapshot`,
  `PreHookEffect`, `ContainerSpec`, `KernelLaunch`, `Edit`, `FireRecord`,
  `_AtenExecutionContext` and the rest of the 50) are encoded inline as `record`-tagged
  objects (section 4).

Rejected: array of objects per row (174 Op keys repeated per row triples the file and the
parse time); row-major arrays with a columns header (this design's first draft; the contract's
columnar form makes the column-set check a key-set comparison on the file and lets a lazy
reader skip payload columns, and the decoder zips columns at the same cost); separate file per
column (thousands of files).

## 4. Typed values: the closed vocabulary

Plain JSON carries `null`, `bool`, finite `number` (ints within 2^53), `string`, `array`
(Python `list`), and `object` with string keys (Python `dict` whose keys are all `str` and
none equal to the marker). Everything else is an object with the marker key `$tl` whose value
is a tag from `TYPED_VALUE_TAGS` (contract page (d), adopted). The tag set is closed; a tag the
decoder does not know refuses `record_schema_invalid` naming
the record, row and column.

| Tag | Payload fields | Decodes to |
| --- | --- | --- |
| `tensor` | `TENSOR_REFERENCE_FIELDS` (`blob`, `key`, `dtype`, `shape`, `device`, `requires_grad`, `codec`, `complex`) | a tensor (eager) or `LazyActivationRef` |
| `tuple` | `items` | `tuple` |
| `set`, `frozenset` | `items` (canonically sorted by the encoder) | `set`, `frozenset` |
| `dict` | `items`: array of `[key, value]` pairs (used when any key is not a `str`, or a key equals `$tl`) | `dict` |
| `bytes` | `base64`, `bytes` (length; up to 1 MiB) | `bytes` |
| `float` | `v`: `nan`, `inf`, `-inf` | `float` |
| `int` | `v`: decimal, used beyond 53 bits | `int` |
| `complex` | `re`, `im` | `complex` |
| `dtype` | `name` (torch dtype name without prefix; numpy dtype string for `numpy_safetensors_v1`) from the dtype table in `values.py` | `torch.dtype` or `numpy.dtype` |
| `device` | `name` | `torch.device` |
| `type` | `name` (dotted) | the dotted name string; the reader imports nothing. A field that needs the class object resolves it through the `record` table (none is known today; a builder that finds one files it) |
| `enum` | `type` (TorchLens-owned enum class from the closed table), `name` (member name) | the enum member |
| `record` | `type` (a class name from the closed allowlist), `fields` (object, encoded recursively by the class's sub-schema) | the TorchLens value object (`OpRef`, `FireRecord`, `Edit`, audit rows, `Duration`, `KernelLaunch`, ...) |
| `ref` | `record` (record kind), `row` (index into that table) | the record object in that row; written where a live field holds a record object (op lists, accessors, call-location lists) |
| `unencodable` | `UNENCODABLE_FIELDS`: `type`, `repr` (at most 256 chars), `repr_length`, `at` | the disclosure object itself (section 5) |
| `callable_placeholder` | `CALLABLE_PLACEHOLDER_FIELDS`: `position`, `qualname`, `repr_length` | the placeholder object (section 8) |

The `record` allowlist: the writer contract's `component_specs` table (50 classes at v11,
`field_registry.component_spec_tables()`) plus the value classes that declare no state spec of
their own (unit quantities such as `Bytes` and `Duration`, `DtypeRef`, `DeviceRef`,
`EdgeUseRecord`, `SourceText`, ...), listed beside the codec in `format3/values.py`; record
classes are excluded (they are tables, reached by `ref`) and enums use `enum`. The component
table alone is not enough: the writer found those value classes in persisted columns of a plain
CNN trace. The unpickler is deleted by this design (section 10). Filed as an amendment
(section 15).

The `ref` tag: a live field that holds a record object rather than its label (the trace's op
list and lookup dicts, `Layer.ops`, `Module.params`, `Op.code_context`, `GradFn.calls`,
`BackwardPass.grad_fn_calls`) writes `{"$tl": "ref", "record": kind, "row": index}`. The
schema cannot type these cells from annotations (most are unannotated), so the reference is
tagged; the index points into the record table, whose rows are in live container order.

Rules the decoder enforces and the encoder guarantees:

- No `eval`, no `pickle`, no `importlib`, no `getattr` on a module by a name from the file.
  `dtype`, `enum` and `record` resolve through tables built at import time from the component
  class list and `torch`'s dtype attributes (`utils/_torch_compat.py` owns the probe). A name
  outside a table refuses; it is never looked up.
- Encoding is canonical: object keys sorted, sets sorted by the encoder's total order (type
  name, then repr), floats via `repr`, no trailing whitespace, UTF-8, `ensure_ascii=False`.
  Two saves of the same trace produce byte-identical record files (the property
  `tests/test_metadata_canonical_bytes.py` holds for the pickle today).
- Bounds: `_io/_json.py` keeps its limits (depth 200, 512 MiB, 64 M nodes) and both the
  record decoder and the manifest reader go through it.
- The decoder is one function over one schema node; it never dispatches on the Python type
  of what it finds beyond the JSON kinds. The schema says what a cell may hold (section 9);
  a value of another kind refuses.

## 5. Unencodable values: stringified with a disclosure (format.1)

A value the encoder meets that is not in section 4's vocabulary (a user object stored in a
module attribute, a custom metadata value, a third-party container) is written as
`{"$tl": "unencodable", "type": "<module>.<qualname>", "repr": "<repr, cut at 256>",
"repr_length": <int>, "at": "<record>[<row>].<column><path>"}`. The encoder never calls
anything on the value but `type()` and `repr()`; a `repr` that raises yields
`"<unrepresentable>"`. A `torch.nn.Module` or tensor subclass found where a plain value was
expected goes the same way (tensors in PAYLOAD and tensor-typed columns go to blobs; a tensor
inside a user mapping in a `value` column is a disclosure too, per contract page (c)).

Disclosure: the manifest `disclosures.count` and the first 16 `at` paths; `tl.load` attaches
an advisory `unencodable_values_stringified` (ruled, section 14) to the loaded
trace's capture advisories, so a user who saved a trace with custom attributes learns on load
that they came back as text. The decoder returns the disclosure object (a small frozen
dataclass in `decode.py` with the four fields and a `__repr__` that shows them; its public
name is `tl.types.UnencodableValue`, ruled) in place of
the value, so `op.module_attributes["x"]` is a visible stand-in and never silently `None`.

Precedent: `scrub._stringify_value` and the manifest keys `custom_attributes_disclosure` and
`buffer_values_disclosure`; those two keys fold into `disclosures`.

Rejected: refusing the save (format.1 says stringify); dropping silently (hides data loss);
pickling the odd value into a side blob (the ruling removes pickle entirely).

## 6. Tensor payloads

- Codecs: `torch_safetensors_v1` and `numpy_safetensors_v1` (`PAYLOAD_CODEC_IDS`). The
  backend transport codecs in `_io/payload_codec.py` (jax, tinygrad, mlx, paddle, tf) stay:
  they produce numpy arrays that `numpy_safetensors_v1` carries, plus their hint metadata,
  which rides in the tensor reference's `codec` object as today.
- Reference object, structure-only form, dtype and device spellings, the
  `metadata_value_unportable` refusal for a tensor the codec cannot hold, complex as the real
  view with `complex: true`, and no inline tensors (threshold 0): contract page (c), adopted. A
  tensor met in a metadata position (a user mapping in a `value` column) becomes an
  `unencodable` disclosure, never a refusal and never a blob.
- Chunked packed files: `blobs/<codec>-<nnnn>.safetensors`. Every tensor is one key `t<k>`
  (opaque, monotone per artifact) in exactly one chunk. A chunk is closed when it would pass
  `PAYLOAD_CHUNK_BYTES` (256 MiB, the contract's cap, a module constant in `payloads.py`,
  overridable only through the private `TORCHLENS_PAYLOAD_CHUNK_BYTES` environment variable
  for tests). A reference is `{"blob": "blobs/torch_safetensors_v1-0000.safetensors",
  "key": "t17", ...}`.
- Where this departs from contract page (c), and why (filed as an amendment, section 15): the
  contract packs by field family (`out`, `grad`, `transformed_out`, ..., `annotation`,
  `recipe`) under keys `<record>.<row>.<field>[.<path>]`. Row-addressed keys cannot be written
  by the streaming writer: during capture the row order is not known (orphan removal,
  relabeling and container adoption renumber ops at finalize), and safetensors keys cannot be
  renamed without rewriting the file. `relabel_blob` exists in `streaming.py` for exactly this.
  Opaque keys make a relabel a record-cell edit and nothing else. Family packing buys nothing
  safetensors does not already give: `safe_open` reads one tensor by offset whatever else is
  in the file, and the per-file digest stays bounded by the chunk cap. Chunking by codec keeps
  one chunk series per dtype universe.
- Why chunks at all: one file per tensor costs an fsync per tensor (1,208 fsyncs, 1.7 s on
  chain_600; 415, 0.6 s on nsrt_cnn) and a directory with thousands of entries; one file per
  artifact cannot be appended to (safetensors is header-first) so streaming saves would have to
  rewrite it at finalize. Chunks give streaming an append (open the next chunk).
- Views and shared storage: safetensors has no aliasing. Each tensor reference is written
  contiguous; two record cells that referred to the same storage live get two keys. The
  `tensor_policy.py` rules (what is saved at which level) are unchanged.
- Lazy loading: `LazyActivationRef` (`_io/lazy.py`) keeps its public surface (`blob_path`,
  `materialize`, identity check) and holds (artifact root, chunk file, key, dtype, shape,
  device). `materialize` opens the chunk with `safetensors.safe_open` and reads one key; the
  file identity check (`_file_identity`) moves to the chunk. The eviction and `to_disk` paths
  in `_io/streaming.py` and `_memory/` that today write one blob file per evicted tensor write
  to the open chunk through `payloads.ChunkWriter.submit` instead.

Rejected: npz or raw `.bin` with offsets (safetensors is already a dependency and already the
blob format); base64 tensors in JSON (slow, 33 percent larger, defeats lazy); inline small
tensors in value columns (this design's first draft; the contract's reason wins: one encoder,
one validator, one digest path, and packing removes the per-file cost inlining avoided).

## 7. Runnable component

What is frozen and does not change: the sparse descriptor
`sparse_recorded_taken_path_v2`, the call recipe
`non_tensor_args_tensor_slots_context_and_obligations_v3`, callable-ref 1, state binding
`module_path_role_v1`, the initializer policy version, `state_dict_v1` and
`runnable_nonpersistent_buffer_v1` payload layer schemas, and the loader's allowlisted
callable-ref resolution in `_io/runnable_load.py` (`_getattr_allowlisted`). That resolution is
replay machinery (which torch function to call), not record decoding; it stays the only
name-to-object step in the load path and keeps its allowlist.

What changes (carrier only; names per contract page (a) and the runnable hand-off note):

- The descriptor JSON moves from `manifest.json["run"]` to `runnable/descriptor.json`
  (listed in `files` with its digest). `manifest.json` keeps `kind`, `save_level: runnable`
  and the `runnable` object (component versions and paths).
- `state_dict_v1` tensors go to `runnable/state_dict-<nnnn>.safetensors` chunks (the same
  chunk writer, opaque keys) and nonpersistent buffers to `runnable/buffers.safetensors`.
  Their references inside the descriptor are `{blob, key}` like every other tensor reference;
  the per-tensor blob files go.
- The module accessor state (`_io_module_accessor_state`) and `sites` that the runnable
  reader needs from the Trace are record columns (`trace` and `module` tables), read through
  the one decoder; `attach_sparse_run_readiness` and `runnable_coherence.py` take the decoded
  trace as they take the rehydrated one today.
- `executable_with_callables` keeps its semantics: the manifest flag plus the descriptor's
  callable refs; nothing is pickled, so the level no longer carries a pickle risk and the
  `custom_callable_*` refusals stay as the gate on resolution.

Rejected: folding the descriptor into the `trace` record (it is a separate, frozen contract
with its own version line and its own loader; a column would couple the two version lines).

## 8. Intervention recipe (io.1)

- `intervention.json` is written beside the records when the trace was captured with an
  intervention: the contract page (e) grammar: `format = torchlens.intervention_recipe`, `format_version = 3`,
  `helper_registry_version`, `save_level`, `executable`, `spec_derived`, `append_state`,
  `rules[]`, `helpers[]`, `function_registry_keys[]`; a rule is `{index, direction, selector,
  edit, metadata}` with the selector as `_target_spec_to_json` emits it. The selector, target,
  sampling and edge-address JSON are the existing `intervention/_spec_codec.py` encoders;
  nothing new is invented for them. Tensor arguments go to `blobs/` through the chunk writer.
- Built-in edits (`Edit` with `portability == "builtin"`) are encoded field by field (`name`,
  `args`, `kwargs`, `metadata`, `signal`, `is_batch_independent`,
  `is_compatible_with_append`, `selection_recipe`) through the typed-value codec; `factory`
  stays `DROP`.
- A callable edit has `portability: callable_placeholder` and carries
  `{"$tl": "callable_placeholder", "position": <rule index>, "qualname": "<module>.<qualname>",
  "repr_length": <int>}` (contract page (e)); a callable inside a built-in edit's `args`/`kwargs`
  is replaced in place by the same object with `position` as `"<rule index>.<arg path>"`.
  `executable` is false in the file and the manifest. The recipe still loads; a rerun that would
  need the callable refuses `run_intervention_spec_not_persisted` (code kept for exactly this
  case; row text rewritten by the writer stream to say "a callable edit was replaced by a
  placeholder at <position>"). Everything else about a loaded intervened trace reruns.
- `Trace._intervention_spec` changes policy from `DROP` to a non-column carrier: the writer
  reads it and emits `intervention.json`; the reader sets it from the file. The guard in
  `intervention/_intervened_rerun_guard.py` narrows from "loaded intervened traces never
  rerun" to "loaded recipes with a placeholder never rerun".
- `Trace.save_intervention` stays the separate-recipe door: it writes the same
  `intervention.json` content as a one-file artifact with `kind: intervention`
  (`manifest.json` plus `intervention.json`), replacing today's bare `spec.json`.
- The behaviour net asserts the persisted recipe: `tests/behaviour/_fill_intervention.py` (the
  matrix check) and `tests/behaviour/test_f7_intervention_persistence.py` (one cell per promise).

## 9. Validation: schema-directed decode plus cross-record invariants (late.21)

The 36 functions of `forgery_validation.py`, the 12 of `_load_validators.py` and the 9 of
`_forgery_identity_facts.py` split in two:

Grammar (shape, closed vocabularies, required keys, scalar kinds) becomes declarative
sub-schemas in `schema.py`, applied by the one decoder as it builds each cell:
`_validate_distributed_scope`, `_validate_timing_provenance` (keys from
`PIPELINE` step names, io.5), `_validate_checkpoint_witness`, `_validate_checkpoint_verdict`,
`_validate_checkpoint_token_record`, `_validate_intervention_audit` and `_validate_audit_row`
with `_validate_act_audit_row`, `_validate_edge_audit_row` and `_validate_param_audit_row`,
`_validate_patch_source`, `_validate_param_occurrence`, `_validate_audit_event`,
`_validate_event_identity`, `_validate_event_payload`, `_validate_repr_tuple`,
`_validate_selection_recipe`, `_validate_kernel_telemetry` and `_validate_kernel_launch`, `_validate_sidecar_namespace`,
`_validate_capture_advisories`, `_validate_source_snapshots`, `_validate_evidence_claims`,
`_validate_annotation_families`, `_validate_op_annotations`,
`_validate_executed_flops_records`, `_validate_peak_memory_reason`,
`_validate_episode_step_stamps`. Each becomes a schema node (a small dict literal: `kind`,
`keys`, `vocab`, `items`) in a `FAMILIES` table; the decoder refuses with the family's existing
code (`artifact_<family>_invalid`) and the record, row and column path.

Cross-record (needs more than one cell) stays procedural in `validate.py`, run after all
tables are decoded and before accessors are rebuilt: `_validate_site_keys` (site keys name
existing ops), `_validate_checkpoint_windows` (ordering across ops), `_validate_audit_sites`,
`_validate_audit_edges`, `_validate_audit_params`, `_validate_event_chain`,
`_validate_structure_only_coherence`, `_validate_kernel_relations`,
`_validate_graph_structure`, `_validate_edge_carriers` and `_validate_edge_entry` (targets
exist), `_validate_injection_provenance` and `check_injection_provenance_row`,
`_validate_structure_evidence` and `_validate_evidence_facts`, `_validate_root_entry_point`,
`_validate_tl_authored_root`, `_validate_identity_facts`. Codes unchanged.

Behaviours kept (late.21): the writer's reload check (after writing the record tables and
before the atomic rename, decode them from the written bytes with the same reader and compare
the schema digest and row counts; this replaces `preflight_metadata_portability`'s dry-run
unpickle and costs one JSON parse instead of a second pickle round trip), and the reader's
checked load (digests, schema, invariants) as the default of `tl.load`; `lazy=True` defers
blob reads only, never checks.

Validation integrity (LOCKED): no check listed above is dropped; each is either a schema node
or a function in `validate.py`, and `tests/test_format3_design_pins.py` (section 11) pins
that every name in these two lists appears in this page until the builders replace it with
the per-family tests.

## 10. Deletion list (line counts at base 2169a3239)

Delete outright once the reader stream's `load_trace` reads the golden set:

| Module | Lines | Why it existed |
| --- | ---: | --- |
| `torchlens/_io/_safe_unpickle.py` | 1983 | restricted unpickler, allowlist, opcode prescan, `_RenameAwareUnpickler`, `_RENAMED_PICKLE_GLOBALS` |
| `torchlens/_io/rehydrate.py` | 1510 | `Trace.__new__` plus `__setstate__` walk over pickled state |
| `torchlens/_io/_portability_preflight.py` | gone | dry-run unpickle before publish |
| `torchlens/_io/state_keys.py` | 307 | shadow filter for pickled setstate key sets |
| `torchlens/_io/_inline_tensor.py` | 275 | pickle reduce of small tensors |
| `torchlens/_io/_pre3_module_paths.py` | gone | pre-3 module-path probe in pickle globals (the 2.x refusal stays, section 14, by a cheaper test) |
| `torchlens/_io/_canonical_pickle.py` | 103 | deterministic pickler |
| `torchlens/_io/_artifact_anchors.py` | 91 | manifest-vs-metadata anchors (manifest `files` digests replace them) |
| `torchlens/_io/_rehydrate_lazy_names.py` | 53 | lazy name table for rehydrate |
| `torchlens/_io/bundle_metadata.py` | 314 | metadata digest and gated member relations over pickled sections |
| `torchlens/_io/bundle_container.py` | 329 | `bundle.json` and the legacy `kind=bundle metadata.pkl` branch |
| `torchlens/_io/forgery_validation.py` | 1996 | rewritten as schema nodes plus `validate.py` (section 9) |
| `torchlens/_io/_load_validators.py` | 494 | same |
| `torchlens/_io/_forgery_identity_facts.py` | 265 | same |
| `torchlens/_io/scrub.py` | 1945 | object-graph scrub to a pickleable state dict; the privacy, barcode-remap (`equivalence_class`, 389-445) and structure-only policies move to `writer.py` as functions over record cells |
| `torchlens/_io/accessor_rebuild.py` | 127 | folds into `reader.py` (same two functions) |
| Subtotal | 10,307 | |

Shrink to a door or a table:

| Module | Lines now | After | What stays |
| --- | ---: | ---: | --- |
| `torchlens/_io/bundle.py` | 4352 | about 400 | the public door dispatch (`save`, `load`, level and path checks, symlink refusal); the runnable branch calls `format3.runnable` |
| `torchlens/_io/manifest.py` | 1112 | about 250 | `TlSpecManifest` as the typed view of `manifest.json`; the v1/v2 manifest schema ids retire with the pickle manifest (the 3.0 manifest is the contract's) |
| `torchlens/_io/format_contract.py` | 539 | about 300 | `TLSPEC_VERSION`, `MIN_TORCHLENS_VERSION_TEXT`, `read_tlspec_version` for in-process state; `MIN_TLSPEC_VERSION` and `enforce_version_policy` go: the floor is `contract.RECORD_CONTRACT_FLOOR` and the one 2.x refusal lives in `reader.read_manifest` |
| `torchlens/_io/state_contract.py` | 398 | about 200 | `governed_artifact_load`, `enforce_known_state` for in-process `__setstate__`; the unknown-key remedy text |
| `torchlens/_io/streaming.py` | 998 | about 600 | `BundleStreamWriter` on `payloads.ChunkWriter`; staged metadata becomes staged record tables |
| `torchlens/_io/lazy.py` | 256 | about 200 | `LazyActivationRef` on (chunk, key) |
| `torchlens/_io/tlspec.py` | 1044 | about 600 | public inspect helpers; `detect_tlspec_format` returns `v3` or `pre_3` (the four `v2.*` labels go) |
| Subtotal | 8,699 | about 2,550 | removes about 6,149 |

Unchanged in role (not touched by the format streams beyond import lines): `_io/runnable.py`
4985 and `_io/runnable_load.py` 3793 (carrier pointers only), `payload_codec.py` 1935,
`injection_codec.py` 798, `_source_privacy.py` 479, `sidecar.py` 387, `_json.py` 385,
`prerelease.py` 301 (its state key becomes a `trace` column flag), `save_policy.py` 197,
`_save_gates.py` 185, `_small_raw.py` 183, `runnable_coherence.py` 169, `paths.py` 164,
`tensor_policy.py` 163, `_artifact_strings.py` 162, `_durability.py` 156,
`writer_contract.py` 151, `payload_reader.py` 142 (reads chunks), `format_errors.py` 128,
`__init__.py` 120, `_root_manifest.py` 106, `_tlspec_json_helpers.py` 93,
`streaming_async.py` 229, `_payload_codec_paddle.py` 36, `_field_policy.py` 29,
`_torch_symbols.py` 16, `field_registry.py` 275.

Record classes: `__getstate__`/`__setstate__` and the pickle-global status of the twelve
record classes stay for in-process pickling (record design section 6, the record streams own
them); the format never calls them. The `__setstate__` chokepoint call into
`validate_persisted_forgery_surfaces` (`trace.py:3305`) is removed when
`forgery_validation.py` goes; that line is the one record-class edit this design needs and it
lands with the record-class fold.

Net: about 16,450 lines leave `torchlens/_io/` and about 4,050 arrive in `format3/`
(section 1 budgets), a reduction of about 12,400 lines.

## 11. Test and golden migration

Across `tests/`, 166 files reference at least one of the deleted modules or the pickle
carrier; the groups below cover them.

Delete with their subject (pickle-only tests; counts are files):
`test_io_pickle.py` (359 lines), `test_io_plain_pickle.py` (568),
`test_metadata_canonical_bytes.py` (87; its property moves to a record-table bytes test),
`test_w051_io_portability_preflight.py` (214), `test_w051_io_artifact_anchors.py` (414),
`test_r4_metadata_unpickle_rce.py` (419), `test_r18m_safe_unpickle_alloc.py` (230),
`test_r49_unpickler_weights_only_baseline.py` (366), `test_r63_construct_deny.py` (372),
`test_safe_unpickle_stream_bounds.py` (310), `test_tlspec_unpickler_first_party_rce.py`
(399), `test_io_metadata_enum_pickle_by_value.py` (270), `test_state_keyset_contract.py`
(156, the setstate shadow filter), `test_unpickle_c_path.py` (unpickle work; moot),
`test_io_embedded_tensor_route_refused.py` (unpickle work; the route no longer exists in
any form; keep one assertion that no `.pkl` is written under an artifact), and the 39 files
that import `_safe_unpickle` plus the 13 that import `_canonical_pickle` (the inventory names
them; most are refusal and allowlist batteries).

Rewrite against the 3.0 reader (same property, new carrier): the 47 files that name
`metadata.pkl` and the 23 that pin `TLSPEC_VERSION` or `MIN_TLSPEC_VERSION` (8); the forgery
batteries (`test_validation_key_forgery.py`, `test_log_injections_s2_forgery.py` and the 6
others) become record-cell forgery tests: edit a cell in `records/<record>.json`, expect the
family code; `test_io_format_v3_load.py` and `test_load_refuses_saved_by_2x.py` keep their
tables and assert the one `artifact_version_below_floor` error.

Keep as refusal fixtures, never regenerated: `tests/goldens/legacy/` (504 K) and
`tests/fixtures/tlspec_v2_16/F2_modellog_tiny_cnn.tlspec` (44 K).

Regenerate on the pinned environment (python 3.10, torch 2.13.0; byte goldens enforce only
there): `tests/agent_surface_goldens/clean.tlspec` (108 K; becomes `manifest.json`,
`records/`, one chunk), `tests/backend_parity/goldens/tlspec_manifest_projection.json` and
`tlspec_roundtrip_digest.json` (the projection keys change to the 3.0 manifest).

Delete fixtures: `tests/fixtures/legacy_trace_85c8c60f.pkl` (28 K; used by
`test_fw2b_pickle_recipe_globals.py` and `test_runnable_trace_seam.py`, both pickle-recipe
tests of the artifact path; the in-process pickle recipe keeps its own battery in the record
streams); update `tests/fixtures/PROVENANCE`. `tests/godobject_oracle/goldens/state_keysets.json`
keeps its `pickle` key (in-process `__getstate__`, record design section 6).

Design pin test (cheap, unmarked tier): `tests/test_format3_design_pins.py` reads
the deletion table of the design page and asserts each named module is present at base, so a
deletion that happens before the design page is updated fails loudly; the writer and reader
streams flip each row to "gone" as they delete.

## 12. Expected save and load cost

Shares measured by the perf profile (one Linux host, cold, median of 3) at base; save: canonical pickler
3.5 s of 11.6 s (nsrt_cnn, instrumented) and 13.0 of 42 s (chain_600), dry-run reload 2.8 and
9.9, nested BlobRef walk 1.6 and 5.7, per-blob fsync 0.6 (415 files) and 1.7 (1,208 files);
load: restricted unpickler 2.7 of 7.6 and 9.9 of 26, rehydrate walk 3.4 and 12.3.

Replacement costs: the encode walk is one pass over record cells (the scrub walk already does
one; the two fuse), emitting JSON through the C `json` encoder (about 0.3 s per 50 MB); the
reload check is one C parse plus the typed decode (the same decoder the reader runs, about
40 percent of today's rehydrate walk because it builds records from rows by a per-record plan
instead of a generic setstate); blob fsyncs fall to one per chunk.

| Row | save ms now | expected | load ms now | expected |
| --- | ---: | ---: | ---: | ---: |
| sionna_small | 743 | 400 to 480 | 620 | 350 to 420 |
| nsrt_cnn | 5302 | 2500 to 3200 | 3360 | 1800 to 2200 |
| vjepa | 2914 | 1400 to 1750 | 1983 | 1050 to 1300 |
| skvmn | 3249 | 1550 to 1950 | 2219 | 1200 to 1450 |
| crystalformer | 4297 | 2050 to 2600 | 2741 | 1450 to 1800 |
| hodgkin_huxley | 4861 | 2300 to 2900 | 3339 | 1800 to 2200 |
| chain_600 (1200 ops) | 18188 | 8000 to 10500 | 10983 | 5500 to 7000 |

The expected columns apply the measured shares (pickler 30 percent, dry run 24, blob walk 14,
fsync 5 of save; unpickler 36 to 38 and rehydrate 45 of load) and the replacement estimates
above; the ranges are the uncertainty on the encode walk and the typed decode. Acceptance:
the writer and reader streams rerun the perf-profile rows on the same host and record them; a row outside its range is a finding, not a failure.

## 13. Layering

Baselines (`scripts/checks/baselines.json`): `largest_scc` 493, `in_function_internal_imports`
2718. `torchlens/_io/` holds 157 in-function internal imports: `bundle.py` 37,
`runnable.py` 26, `payload_codec.py` 16, `runnable_load.py` 9, `injection_codec.py` 9,
`scrub.py` 8, `_save_gates.py` 7, `bundle_metadata.py` 6, `streaming.py` 4,
`forgery_validation.py` 4, `_safe_unpickle.py` 4, `_forgery_identity_facts.py` 4,
`writer_contract.py` 3, `rehydrate.py` 3, others 17.

Deleted modules remove 29 deferred imports outright (`scrub` 8, `bundle_metadata` 6,
`forgery_validation` 4, `_safe_unpickle` 4, `_forgery_identity_facts` 4, `rehydrate` 3);
`bundle.py`'s 37 fall to about 6 (the door keeps the `_runnable` and `_memory` deferrals the
runnable and streaming paths need). `format3/` modules import at module top from the set in
section 1 and add no deferred import except the two in `format3/runnable.py` that reach
`_runnable` (as `_io/runnable.py` does today). Net: the in-function ceiling falls by at least
55; the SCC does not grow because `format3` imports nothing that imports `_io` back
(`_data_classes` and `utils` are below `_io` in the spine; `scripts/checks/layering.py` and
`arch_spine_layer_lint.py` are the gates). Neither baseline is raised by any format stream.

## 14. Refusal table

Codes reused from `docs/reference/error_refusal_contract.md` (the contract's `REFUSAL_CODES`
map is the binding list; this table says when each fires in the 3.0 reader and writer):

| Situation | Code | Where |
| --- | --- | --- |
| no `manifest.json` | `manifest_missing` | reader, before any other file |
| manifest not UTF-8 JSON / not an object / fails the manifest schema | `manifest_unreadable` / `manifest_not_json_object` / `manifest_schema_invalid` | reader |
| `manifest.json` without `format`, or a `metadata.pkl` or `bundle.json` at the root, or `tlspec_version` below 11 | `artifact_version_below_floor` (`ArtifactTooOldError`), one error, remedy: re-save with TorchLens 2.36.x | reader (replaces `_pre3_module_paths`) |
| `format_version` above 3 or `tlspec_version` above the runtime's | `artifact_version_above_runtime` | reader |
| `kind` not the kind the door expects | `artifact_kind_mismatch` | reader |
| `save_level` unknown | `artifact_save_level_invalid` | reader |
| a `files` digest or byte count disagrees, or a record file is not an object with `columns`/`rows` | `bundle_metadata_integrity_refused` | reader |
| a column not in the schema | `unknown_persisted_field` | decoder |
| a cell of the wrong kind, an unknown `$tl` tag, a tag payload missing a field, a name outside a table | `record_schema_invalid` | decoder |
| a nested family fails its sub-schema | the family's `artifact_<family>_invalid` | decoder |
| a cross-record invariant fails | its existing code (`artifact_structure_only_incoherent`, `artifact_site_key_invalid`, ...) | `validate.py` |
| a referenced chunk or key is missing, or safetensors refuses the file | `bundle_load_failed` | payloads |
| a path component escapes the root or is a symlink | `load_path_symlink_rejected` | reader |
| a tensor dtype safetensors cannot carry | `metadata_value_unportable` | writer |
| include flags contradict the level | `save_payload_level_conflict` | writer (`save_policy`) |
| target exists / rename fails / reload check disagrees | `bundle_save_failed` | writer |
| rerun of a recipe with a `callable_placeholder` | `run_intervention_spec_not_persisted` (row text rewritten) | intervention guard |
| a 3.0 door not yet built | `bundle_save_failed` / `bundle_load_failed` | stubs |

Spellings ruled by the maintainer (all in code; `PROVISIONAL_REFUSALS` is empty):

- Ruled: `record_schema_invalid` is the code for a record cell that fails the typed schema
  (a row of the public refusal contract; the message names record, row and column).
- Ruled: `tl.types.UnencodableValue` is the public stand-in type (type name, repr text, where
  it was).
- Ruled: `unencodable_values_stringified` is the load advisory when `disclosures.count > 0`,
  in the same list as the capture advisories.
- Ruled: `tl.io.detect_tlspec_format` returns `v3`, `pre_3`, or `unknown` for a folder
  that is not a TorchLens save. The four `v2.*` labels are removed.

## 15. Where this design and the contract page differed

Adopted from the contract page (`saved_format_3.md`) after this design's first draft
disagreed: columnar record files and
label-string cross references (section 3); no inline tensors (section 6); `type` decodes to
its dotted name; `record` as the tag for TorchLens value objects; the runnable and bundle
layouts and the recipe `format_version` (sections 2, 7, 8); no `schema_digest`.

Corrections this design made to the contract's first draft, each with the exact change, since
folded into `contract.py` and the contract page so the two documents now agree:

1. Blob keys are opaque (`t<k>`), chunks are per codec (`blobs/<codec>-<nnnn>.safetensors`)
   with the contract's 256 MiB cap; not per field family under `<record>.<row>.<field>` keys.
   Reason: the streaming writer does not know the row at write time and safetensors keys
   cannot be renamed (section 6). Builders: `payloads.py` owns the key counter; the record
   cell carries the key; `relabel_blob` edits the cell only.
2. `RECORD_TYPE_ALLOWLIST_SOURCE` is the writer contract's `component_specs`
   (`torchlens._io.field_registry.component_spec_tables`), not the restricted unpickler module,
   which this design deletes (section 10). Same classes; a digested, surviving source.
3. Manifest gains the optional `disclosures` object (section 2) so `tl.load` can attach the
   `unencodable_values_stringified` advisory and `tl.io.inspect` can show the count without
   decoding every cell.
4. The intervention recipe's rule-level `callable_placeholder` is also allowed inside
   `args`/`kwargs` of a built-in edit (a callable argument), with `position` as
   `<rule index>.<arg path>` (section 8); the contract page (e) first placed it at the edit's
   `portability` only. Both forms refuse rerun with the same code.

Everything else in `contract.py` and the contract page (names, codes, levels, record index, planned
and live-side keys, io.2 settle-at-save, `PHASE_TIMING_KEY_SOURCE`, the 2.x refusal rule,
`record_schema_invalid`) is adopted as written.

## 16. Build order

1. Writer stream: `schema.py`, `values.py`, `payloads.py`, then `writer.py` behind a private
   flag; the reload check uses a minimal decoder in `values.py` until the reader lands.
   Regenerates the bindings page when it adds the io.2 and backends.2 columns (bumps
   `tlspec_version`). The behaviour net's F7 cells assert the persisted recipe.
2. Reader stream: `reader.py`, `validate.py` (moving the two lists of section 9), then
   switches `tl.load` and deletes section 10's modules and tests in the same change set that
   reads the regenerated goldens.
3. Runnable stream: `format3/runnable.py`, carrier pointers in `_io/runnable.py` and
   `runnable_load.py`, the state chunk writer; depends on 1 for `payloads.py`.
4. The names ruling's rename (`backward_pass_index` to `backward_pass_number`) lands with the
   names rename stream; whichever of it and the writer stream lands second rebases the column name.
