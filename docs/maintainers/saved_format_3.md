# The 3.0 saved format

TorchLens 3.0 saves a trace as a directory of typed JSON record files plus safetensors blob
files. Nothing in the directory is pickled, and the reader never executes anything it reads.
This page is the contract the writer, reader and runnable component are built against; the
constants it names live in `torchlens/_io/format3/contract.py`, and
`tests/test_format3_contract.py` pins them to the generated field page
`docs/maintainers/persisted_fields.md`. The package layout and the deletion table are the design
page `saved_format_3_design.md`: `contract.py`, `schema.py`, `values.py`, `payloads.py`,
`writer.py`, `reader.py`, `validate.py`, `runnable.py`, each under 800 lines; grammar validators
are schema nodes, cross-record validators live in `validate.py`. Until the three doors land, the functions in
`torchlens/_io/format3/` refuse typed and name the module that fills each.

What the format replaces: the `metadata.pkl` of the pickle writer (a restricted-unpickler
payload) and its inline-tensor pickle reduction. What it keeps: the record contracts and keys of
`persisted_fields.md`, the save levels and payload flags of `torchlens/_io/save_policy.py`, the
safetensors payload codecs, the writer identity stamp and the writer contract digest.

## 1. Envelope

### Directory layout

```
<name>.tlspec/
  manifest.json                      the only file the reader opens blind
  records/trace.json                 one object: the trace record
  records/op.json                    one columnar table per record kind (twelve files)
  records/layer.json ... records/param.json
  blobs/<codec>-<nnnn>.safetensors   every tensor payload, one file series per codec id
  annotations.json                   the annotation and sidecar namespaces, when present
  intervention.json                  the intervention recipe, when the trace carries one
  runnable/                          the runnable component (section 7), at the runnable and
                                     executable_with_callables levels
```

A bundle is `manifest.json` with `kind: "bundle"`, `members/<index>-<name>/` holding one trace
artifact each, and `relations.json`. A recipe saved on its own by `save_intervention` is
`manifest.json` with `kind: "intervention"`, `intervention.json` and its `blobs/`.

### manifest.json

| field | type | meaning |
| --- | --- | --- |
| `format` | string | always `"torchlens.saved_format"` |
| `format_version` | integer | the format major; `3` for this contract |
| `tlspec_version` | integer | the record contract version the records follow |
| `kind` | string | `trace`, `bundle` or `intervention` |
| `torchlens_version`, `python_version`, `torch_version`, `platform`, `created_at` | string | writer identity (`created_at` is ISO 8601 UTC); required, never part of any digest |
| `backend`, `backend_runtime` | string | the capture backend and its runtime, as the tlspec 11 manifest stamps them |
| `save_level` | string | one of `portable`, `audit`, `runnable`, `executable_with_callables` |
| `payload_policy` | object | the resolved flags `include_outs`, `include_grads`, `include_saved_args`, `include_rng_states`, `include_buffer_values`, `structure_only`, all booleans |
| `contract_digest` | string | the writer contract digest of the record contract the records follow |
| `codecs` | array | the payload codec ids used; a subset of `torch_safetensors_v1`, `numpy_safetensors_v1` |
| `records` | object | per record kind: `{"path": "records/op.json", "rows": N, "sha256": hex, "bytes": N}` |
| `files` | object | every file in the directory except `manifest.json`, by relative path: `{"sha256": hex, "bytes": N}` |
| `model_signature`, `model_fingerprint` | string | as the tlspec 11 manifest stamps them |
| `runnable`, `intervention` | object | present with the component; component schema versions and file paths |
| `disclosures` | object | optional; `{"count": N, "first": [up to 16 field paths]}` when the artifact holds N > 0 unencodable disclosures (section 4), absent when none; when present the loader attaches the load advisory `unencodable_values_stringified` |

### Version floor and refusals

The reader accepts a manifest only when `format` equals `torchlens.saved_format`,
`format_version` equals 3, and `tlspec_version` lies between the reader floor and the runtime's
record contract version (both are constants of `contract.py`; the floor is the first record
contract written in this format, so the 3.0 reader reads nothing older than itself).

- A directory with a `metadata.pkl`, a manifest without `format`, or `format_version` below 3,
  refuses `artifact_version_below_floor` (saved by torchlens 2.x; the remedy names 2.36.x).
- `format_version` above 3, or `tlspec_version` above the runtime's, refuses
  `artifact_version_above_runtime`.
- `tlspec_version` below the floor refuses `artifact_version_below_floor`.

`tl.io.detect_tlspec_format(path)` returns `"v3"` for a directory with a 3.0 manifest, `"pre_3"`
for a 2.x save (a `metadata.pkl` or a manifest without `format`), and `"unknown"` for a directory
that is not a TorchLens save at all.

### Digest relation

Every file other than the manifest is listed in `files` with its SHA-256 and byte length. The
reader verifies each file's digest before parsing it; a mismatch or a truncated file refuses
`bundle_metadata_integrity_refused`. A file present in the directory but absent from `files` is
ignored and never read (a `README.txt` is allowed there). The manifest itself carries no digest
of its own; the save preflight (section 6) reloads the written directory once, so a manifest
that does not describe its directory never publishes.

## 2. Records

### Record files

`records/trace.json` is the single trace record:

```json
{"record": "trace", "tlspec_version": 12, "fields": {"name": "...", "_options_mode": "...", "...": "..."}}
```

Every other record kind is a columnar table, one array per key, every array of length `rows`:

```json
{"record": "op", "tlspec_version": 12, "rows": 3,
 "columns": {"label": ["conv2d_1_1", "relu_1_2", "linear_1_3"], "func_name": ["conv2d", "relu", "linear"], "...": []}}
```

Row order is the live container's order (ops in forward order, modules in registration order,
backward passes in pass order). References between records are the label and address strings the
live records carry today; the format adds no row ids.

### Keys and types

The twelve record kinds, their classes and their key counts are the Records table of
`persisted_fields.md`; `contract.RECORD_INDEX` repeats them and the pin test holds the two equal.
Every persisted key of the page is a key of its record file, spelled as stored: a leading
underscore is part of the key (`_options_mode` is the key of the store behind
`Trace.options.mode`). At the format level no key is private; the public spelling is the
accessor the page names.

A key's JSON encoding follows its storage kind in `torchlens/_data_classes/_schema_bindings.py`
through `contract.JSON_TYPE_BY_STORAGE_KIND`:

| storage kind | JSON encoding | note |
| --- | --- | --- |
| `scalar`, `interned`, `computed`, `group` | `value` | the value itself under the value rules (section 4); interning and grouping are physical layouts, not part of the format |
| `edge`, `bitset` | `id_array` | an array of row references (label strings for edges, op indices for ancestor bitsets) |
| `payload` | `tensor_ref` | a tensor reference (section 3) or a container of them |
| `runtime` | `never` | never written; such a key in a record file refuses `unknown_persisted_field` |

Where the class declares an annotation (`int`, `str | None`, `dict[str, Any]`), the reader
validates the JSON value against it; `Any` admits any value the value rules produce.

### Optional keys per save level

A key the field page marks `required` is a column of every artifact. A key marked `optional`
with a payload flag is a column exactly when that flag is on in `payload_policy`; otherwise the
column is absent, never null-filled. A column the level forbids refuses `record_schema_invalid`
(section 6). A structure-only capture keeps every
payload column but its tensor references carry no blob (section 3).

### Keys and values the format adds

The tlspec 12 bump that ships with the format adds, so that a loaded trace equals the live one:

- `op.is_constant` (boolean, required at every level, fact): set by the shared constants rule for
  closure constants and literal producers, following `is_buffer`; the Paddle and tinygrad
  constant rules set it the same way.
- `op.arg_expressions` (value): the computed argument expressions, written at save time.
- Values settled on the live side before writing, so the loaded value equals the live one:
  `op.equivalence_class`, `op.func_duration`, `op.edge_substitutions`,
  `op.edge_replacement_stamps` (an absent map is written as an empty object, never null), and
  `trace._options_grouping_policy` (the grouping stamp the loader used to settle after loading is
  settled by the writer; the reader verifies it instead of rewriting it).
- `grad_fn_call.backward_pass_number` replaces `backward_pass_index` (a rename landing with the
  names work; whichever lands second rebases).
- `trace._phase_timings` is an object keyed by the step names of
  `torchlens._postprocess._pipeline.PIPELINE` (`add_output_layers`, `map_final_labels`, ...),
  float seconds per step; the former `postprocess:Step N: description` keys are gone. A timing
  bucket outside the pipeline keeps its stage name as a snake_case key; the writer contract pins
  the full key set.

Both op keys are in the tlspec 12 bindings and on the page (the op record has 176 keys), so
`contract.PLANNED_KEYS` is empty. A key a later contract plans goes there first; the pin test
asserts it is absent from the page until the change that adds its binding, regenerates the page
and bumps `contract.RECORD_CONTRACT_VERSION` also removes it from the plan.

## 3. Tensor references

A tensor is written as one entry of a safetensors file under `blobs/` and referenced from the
record as a typed object:

```json
{"$tl": "tensor", "blob": "blobs/torch_safetensors_v1-0001.safetensors", "key": "t17",
 "dtype": "float32", "shape": [1, 16, 28, 28], "device": "cuda:0", "requires_grad": false,
 "codec": "torch_safetensors_v1"}
```

- `blob` is the file's path relative to the artifact root, one series of files per codec id
  (`blobs/<codec>-<nnnn>.safetensors`, 256 MiB cap per file); `key` is an opaque entry name
  `t<k>`, monotone per artifact, and every tensor lives in exactly one file. The record cell
  carries the key, so the writer can stream tensors before finalization renumbers and relabels
  the ops; safetensors reads one entry by offset whatever else the file holds.
- `dtype` is the torch dtype name without its prefix (`float32`, `bfloat16`, `bool`,
  `float8_e4m3fn`). A complex tensor is stored as its real view with `"complex": true`.
- `device` is the device at capture as a string; the reader materializes on CPU unless asked
  otherwise and keeps the recorded device string.
- `codec` is `torch_safetensors_v1` for torch tensors and `numpy_safetensors_v1` for array
  payloads of other backends.
- No tensor is stored inline (threshold zero). One encoder, one validator and one digest path
  cover every tensor; safetensors carries zero-dimensional and zero-size tensors, and the per-entry
  header cost is small because entries are packed into shared files.
- Structure-only: the reference keeps `dtype`, `shape`, `device` and `requires_grad` and has no
  `blob` or `key`. A structure-only artifact that names any blob refuses
  `artifact_structure_only_incoherent`, at write and at read.
- A payload tensor the codec cannot hold (quantized, sparse, nested, a subclass with extra
  state) refuses the save with `metadata_value_unportable`, naming the key; a tensor in a
  metadata position (an annotation value, a small raw input) that the codec cannot hold is
  written as an unencodable disclosure (section 4), never refused.

## 4. Values, and user objects that cannot be encoded

Plain JSON carries `null`, booleans, integers within 53 bits, finite floats, strings, arrays and
string-keyed objects. Everything else is a typed object whose marker key is `$tl` and whose tag
is one of `contract.TYPED_VALUE_TAGS`:

| tag | form | loads as |
| --- | --- | --- |
| `tuple`, `set`, `frozenset` | `{"$tl": "tuple", "items": [...]}` | the container |
| `dict` | `{"$tl": "dict", "items": [[k, v], ...]}` | a mapping with non-string keys, or one that itself uses the marker key |
| `int` | `{"$tl": "int", "v": "<digits>"}` | an integer beyond 53 bits |
| `float` | `{"$tl": "float", "v": "nan" or "inf" or "-inf"}` | the non-finite float |
| `complex` | `{"$tl": "complex", "re": x, "im": y}` | the complex number |
| `bytes` | `{"$tl": "bytes", "base64": "...", "bytes": N}` | bytes up to 1 MiB; larger becomes `unencodable` |
| `dtype`, `device` | `{"$tl": "dtype", "name": "float32"}` | the torch object |
| `type` | `{"$tl": "type", "name": "torch.nn.modules.conv.Conv2d"}` | the dotted name as a string; the reader imports nothing |
| `enum` | `{"$tl": "enum", "type": "...", "name": "..."}` | a TorchLens-owned enum member; a user enum becomes `unencodable` |
| `record` | `{"$tl": "record", "type": "...", "fields": {...}}` | a TorchLens value object (`OpRef`, `FireRecord`, `Edit`, the audit rows, unit quantities such as `Bytes`) whose type is on the record-type allowlist: the writer contract's component classes plus the value classes listed beside the codec in `format3/values.py`; any other type refuses |
| `ref` | `{"$tl": "ref", "record": "op", "row": 17}` | row 17 of `records/op.json`: written where a live field holds a record object (an op list, an accessor, a call-location list) rather than its label |
| `tensor` | section 3 | the tensor |
| `unencodable` | below | a disclosure |
| `callable_placeholder` | section 5 | an audit-only edit |

A user object no rule covers is stringified with a disclosure, never pickled:

```json
{"$tl": "unencodable", "type": "mypkg.config.RunConfig", "repr": "RunConfig(lr=0.001, ...)",
 "repr_length": 412, "at": "op.17.saved_kwargs.cfg"}
```

`repr` keeps at most 256 characters (`contract.UNENCODABLE_REPR_LIMIT`), `repr_length` is the
full length and `at` is the field path. The reader materializes it as an `UnencodableValue`, a
public frozen record with those four fields whose `repr` is the saved text, so a user can test for
it with `isinstance`. A loaded artifact whose manifest carries `disclosures` gets the load
advisory `unencodable_values_stringified`, in the same list as the capture advisories.

## 5. The intervention recipe

A trace saved with a staged intervention spec carries `intervention.json`, and `save_intervention`
writes the same file in its own directory: one grammar, two homes.

```json
{"format": "torchlens.intervention_recipe", "format_version": 3, "helper_registry_version": 4,
 "save_level": "portable", "executable": true, "spec_derived": false, "append_state": {},
 "rules": [
   {"index": 0, "direction": "forward",
    "selector": {"selector_kind": "label", "selector_value": "relu_1_2", "strict": true, "slice_spec": null, "metadata": {}},
    "edit": {"name": "scale", "args": [0.5], "kwargs": {"allow_shape_change": false}, "portability": "builtin",
             "signal": null, "is_batch_independent": true, "is_compatible_with_append": false,
             "metadata": {}, "selection_recipe": null},
    "metadata": {}}
 ],
 "capture_rules": [],
 "helpers": [{"name": "scale", "version": 1}], "function_registry_keys": []}
```

- An edit is the `Edit` identity under the value rules: `name`, `args`, `kwargs`, `portability`,
  `signal`, `is_batch_independent`, `is_compatible_with_append`, `metadata`, `selection_recipe`.
  A tensor argument (a steering vector, a patch source) is a tensor reference into the
  artifact's `blobs/`. Every built-in edit of the registered vocabulary
  (`scale`, `zero_ablate`, `steer`, `patch_from`, `clip_norm`, `set_affine_coordinates`,
  the subspace and stochastic edits, and the rest the helper registry names) saves this way and
  rebuilds through the same registry `tl.load` uses today.
- A rule whose edit holds a user callable saves without it: the edit's `portability` is
  `callable_placeholder` and it carries
  `{"$tl": "callable_placeholder", "position": 2, "qualname": "mymod.f", "repr_length": 61}`
  with the rule index as `position`. A callable found inside a built-in edit's `args` or
  `kwargs` is replaced in place by the same object with `position` set to
  `"<rule index>.<argument path>"` (`"0.kwargs.transform"`). Either way the manifest's
  `intervention.executable` is false and the rule loads as an audit-only edit.
- Such a recipe is written only at the audit level: either `Trace.save(level="portable")`
  or `save_intervention(level="portable")` on a
  recipe holding a user callable, or a value the format cannot carry, refuses at save time with
  `intervention_spec_not_portable`; the remedy is `level="audit"` or built-in edits.
- `capture_rules` is an optional array (absent means empty) preserving original built-in
  capture predicates alongside the canonical staged `rules`. Each entry has exactly `index`
  (its zero-based array position), `selector` (the same closed TargetSpec object as a rule),
  and `edit` (the same built-in Edit identity). Unknown recipe keys, unknown entry keys,
  malformed indices or selectors, and non-built-in capture edits refuse
  `manifest_schema_invalid`. The existing typed value rules apply to every member; unknown
  selector kinds or edit names refuse through the existing selector/helper registry.
  A loaded rerun rebuilds this predicate with the code-owned selector and edit registries,
  then checks that it restages exactly the saved label rules. It never resolves a callable
  by an artifact-supplied import name. A callable predicate refuses
  `intervention_spec_not_portable` at save time.
- A loaded trace whose recipe holds a placeholder refuses every rerun door
  (`run(model, x)` plain, append or chunked, a fork's `run`, `do(engine="rerun")`) with
  `run_intervention_spec_not_persisted`; the message names the rule position and the callable's
  qualified name, and the remedy is to rebuild that rule with its callable and pass it as
  `intervene=` to a fresh capture, or to re-save with built-in edits. The code keeps its spelling
  and narrows to this one case: a recipe of built-in edits is persisted, so a loaded intervened
  trace reruns with its recorded intervention.

## 6. Refusals

One table for the writer and the reader. Codes are rows of
`docs/reference/error_refusal_contract.md`; `contract.REFUSAL_CODES` names them by situation.

| situation | code | raised by |
| --- | --- | --- |
| no `manifest.json` | `manifest_missing` | reader |
| manifest unreadable or over the parse bound | `manifest_unreadable` | reader |
| manifest root not an object | `manifest_not_json_object` | reader |
| manifest violates this contract (missing field, wrong type, `records` or `files` entry wrong) | `manifest_schema_invalid` | reader |
| 2.x artifact, `format_version` below 3, record contract below the floor | `artifact_version_below_floor` | reader |
| `format_version` above 3 or record contract above the runtime | `artifact_version_above_runtime` | reader |
| specialized loader given another kind | `artifact_kind_mismatch` | reader |
| unknown save level | `artifact_save_level_invalid` | writer, reader |
| file digest or length mismatch, truncated record file | `bundle_metadata_integrity_refused` | reader |
| key not in the record contract, or a `runtime` kind key | `unknown_persisted_field` | reader |
| record value violates the contract (type, row length, column the level forbids, typed object with an unknown tag, `record` type off the allowlist) | `record_schema_invalid` | reader |
| blob file or entry named by a reference is missing | `bundle_load_failed` | reader |
| blob in a structure-only artifact, or a structure-only marker over retained payloads | `artifact_structure_only_incoherent` | writer, reader |
| load path is a symlink | `load_path_symlink_rejected` | reader |
| payload tensor the codec cannot hold | `metadata_value_unportable` | writer |
| explicit payload flag contradicts the level | `save_payload_level_conflict` | writer |
| target exists without `overwrite` | `bundle_save_failed` | writer |
| portable save of a recipe holding a user callable or a value the format cannot carry | `intervention_spec_not_portable` | writer |
| rerun of a loaded trace whose recipe holds a callable placeholder | `run_intervention_spec_not_persisted` | rerun doors |
| a door called before its lane filled it | `bundle_save_failed`, `bundle_load_failed` | the stubs |

`record_schema_invalid` and `intervention_spec_not_portable` (ruled 2 Nov 2026) join the error
contract table in the change that adds their raise sites, the reader and the writer respectively.
Until the reader lands, `contract.PROVISIONAL_REFUSALS` maps the record-level code to
`manifest_schema_invalid`, the code a builder raises meanwhile.

Two behaviours stay as they are today and apply to the JSON reader: the save preflight reloads
every save once through the reader before publishing it, and the checked load validates every
record against this contract (known keys, types by storage kind, row lengths, tensor references
resolving to listed blobs) before any object is built.

## 7. Runnable

The runnable component lives under `runnable/`: `descriptor.json` (the descriptor, call recipe,
callable reference and initializer policy at the component versions the writer contract pins),
`state_dict-NNNN.safetensors` and `buffers.safetensors`. A callable reference that cannot be
written follows the callable placeholder rule and makes the artifact audit-only. The `runnable`
manifest object lists the component versions and files.

- `descriptor.json` is canonical JSON (sorted keys, no NaN or infinity) and is byte for byte the
  descriptor the tlspec 11 manifest carried under `run`. Callables appear only as callable registry
  records (a registry key per callable); no callable object is ever written.
- `state_dict-NNNN.safetensors` holds the capture-time state dict, numbered from 0001, each file at
  most 256 MiB (a larger tensor gets a file of its own). `buffers.safetensors` holds the
  non-persistent buffer values. `activations-NNNN.safetensors` holds the capture-selected
  activations when the save archived them. Every tensor is one opaque key `t<k>` in one file, as in
  section 3 (numbered on the artifact's shared counter; a reference names its file and key); tensors that share storage (tied weights, view activations) are written as separate
  copies, and a complex tensor is stored as its real view with `"complex": true`.
- The `runnable` manifest object is closed. Required: `descriptor`, `descriptor_version`,
  `call_recipe_version`, `callable_ref_version`, `initializer_policy_version`,
  `witness_registry_version`. Each present family adds three keys: its files, its schema and its
  tensor references. `state_dict` (list), `state_dict_family`, `state_dict_tensors` (state name to
  tensor reference); `buffers`, `buffers_family`, `buffers_tensors` (buffer name to tensor
  reference); `activations` (list), `activations_family`, `activations_tensors` (the descriptor's
  activation blob id to tensor reference). Every listed file has a `files` row.
- Load refuses `manifest_schema_invalid` for an unknown key, a family missing one of its three keys,
  a version other than the runtime's, a file name outside the fixed names, chunk numbering that is
  not 0001 upward, a listed file without a `files` row, a tensor reference that is not a contract
  tensor reference into its own family's files, or a `runnable/` directory the manifest does not
  list; `bundle_metadata_integrity_refused` for a file whose sha256 or byte length differs, or for
  a reference whose key is absent, repeated or whose dtype or shape disagree, or a stored tensor no
  reference names;
  `bundle_load_failed` for a listed file that is missing; `load_path_symlink_rejected` for a
  symlinked file. The families then bind through the same strict state binders as before, so
  readiness, numeric attestation and path faithfulness prove what they proved in tlspec 11.

A bundle's member list references each member by `name`, `path` (`members/<index>-<name>`, the
name part keeping letters, digits, `.`, `_` and `-` and at most 64 characters) and
`manifest_sha256`, the sha256 of that member's `manifest.json`. A member swapped or edited after
the save refuses `bundle_metadata_integrity_refused`; an unknown reference key or a mistyped hash
refuses `manifest_schema_invalid`. `relations.json` holds the closed relation rows and is listed
in the bundle manifest's `files`.

## 8. Worked examples

A portable save of a tiny CNN (`conv2d`, `relu`, `linear`), excerpts:

```json
// manifest.json
{"format": "torchlens.saved_format", "format_version": 3, "tlspec_version": 12, "kind": "trace",
 "torchlens_version": "3.0.0", "python_version": "3.11.9", "torch_version": "2.13.0",
 "platform": "Linux-x86_64", "created_at": "2026-11-02T09:00:00Z",
 "backend": "torch", "backend_runtime": "eager", "save_level": "portable",
 "payload_policy": {"include_outs": true, "include_grads": true, "include_saved_args": false,
                    "include_rng_states": false, "include_buffer_values": true, "structure_only": false},
 "contract_digest": "<sha256 of the record contract>", "codecs": ["torch_safetensors_v1"],
 "records": {"trace": {"path": "records/trace.json", "rows": 1, "sha256": "...", "bytes": 48211},
             "op": {"path": "records/op.json", "rows": 3, "sha256": "...", "bytes": 19042}, "...": {}},
 "files": {"records/op.json": {"sha256": "...", "bytes": 19042},
           "blobs/out-0001.safetensors": {"sha256": "...", "bytes": 102592}, "...": {}}}
// records/op.json (three columns of 176)
{"record": "op", "tlspec_version": 12, "rows": 3,
 "columns": {"label": ["conv2d_1_1", "relu_1_2", "linear_1_3"],
             "is_constant": [false, false, false],
             "out": [{"$tl": "tensor", "blob": "blobs/torch_safetensors_v1-0001.safetensors", "key": "t0",
                      "dtype": "float32", "shape": [1, 8, 26, 26], "device": "cpu",
                      "requires_grad": true, "codec": "torch_safetensors_v1"}, {"...": {}}, {"...": {}}]}}
```

A runnable save of the same model: `save_level` is `runnable`, `payload_policy` turns every
payload flag off, `records/op.json` has no `out` or `grad` column, and the manifest adds

```json
"runnable": {"descriptor": "runnable/descriptor.json", "descriptor_version": "sparse_recorded_taken_path_v2",
             "call_recipe_version": "non_tensor_args_tensor_slots_context_and_obligations_v3",
             "state_dict": ["runnable/state_dict-0001.safetensors"], "state_dict_family": "state_dict_v1"}
```

An intervened save with one built-in edit and one callable: `intervention.json` has two rules;
rule 0 is the `scale` edit of section 5 and rule 1 is

```json
{"index": 1, "direction": "forward",
 "selector": {"selector_kind": "label", "selector_value": "linear_1_3", "strict": true, "slice_spec": null, "metadata": {}},
 "edit": {"name": "custom", "args": [], "kwargs": {}, "portability": "callable_placeholder",
          "callable": {"$tl": "callable_placeholder", "position": 1, "qualname": "experiments.hooks.clamp_top", "repr_length": 58},
          "signal": null, "is_batch_independent": false, "is_compatible_with_append": false, "metadata": {}, "selection_recipe": null},
 "metadata": {}}
```

with `"executable": false` at the top of the file and in the manifest's `intervention` object.
`records/op.json` carries `is_edited` true and the `edits` fire records for both ops; loading
succeeds, and `run(model, x)` refuses `run_intervention_spec_not_persisted` naming rule 1 and
`experiments.hooks.clamp_top`. This recipe is the audit-level form: the same intervention saved
with `save_intervention(level="portable")` refuses `intervention_spec_not_portable` at save time.

## 9. Loaded equals live

A trace loaded from a 3.0 artifact equals the live trace it was saved from, at every save level:

- Column sets: for each record kind, the loaded record's keys equal the live record's persisted
  keys at that level (the required keys plus the optional keys whose flag is on). The test derives
  the expected set from `persisted_fields.md` and `save_policy` and compares it with the loaded
  record file and with the loaded objects.
- Values, per storage kind: `value` and `id_array` keys compare equal after the round trip
  (floats by exact repr, non-finite floats by kind); `tensor_ref` keys compare by `torch.equal`,
  dtype, shape and the recorded device string, with the loaded tensor on CPU; an `unencodable`
  disclosure compares its `repr` with the first 256 characters of the live value's `repr`.
- The settled values of section 2 compare equal because the writer settles them; the reader
  rewrites nothing after loading.

The proof is `tests/test_format3_loaded_equals_live.py`, written with the writer: a tiny CNN
traced once, saved at each of the four levels, loaded, and compared record by record.

Payload rule clarification (2026-11-02): an unsupported sparse payload refuses
`metadata_value_unportable` regardless of `strict`. Complex64 and complex128 use their real
views in chunks and round-trip exactly; neither is skipped. Raw inputs and transformed
outputs retain numeric arrays and dense tensors through the same payload codecs. Values
without a portable encoding retain the typed `UnencodableValue` disclosure.
