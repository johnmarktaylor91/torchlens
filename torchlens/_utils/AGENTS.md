# utils/ - Shared Utilities

## What This Is
Small shared helpers used across capture, postprocess, validation, and data classes. Keep
this package mostly stateless and free of high-level TorchLens business logic.

## Files

| File | Purpose |
|------|---------|
| `arg_handling.py` | Safe arg copying and input normalization for model calls |
| `collections.py` | Iterable and nested-index helpers |
| `display.py` | Human-readable formatting, verbosity, identity helper |
| `hashing.py` | Barcodes, short hashes, graph-shape hash |
| `introspection.py` | Recursive object search and nested getattr/assign |
| `rng/` | Package: `streams.py` (Python, NumPy, torch, CUDA stream snapshot/restore), `seeding.py`, `restore.py` (the one restore door `RngStreamsRestore`, used by `tl.trace`, replay and `tl.validate`), `autocast.py`; the host-nondeterminism monitor in `monitor.py` with its `registry.py`, `channels.py`, `engine_state.py`, `inventory_types.py`, `call_sites.py`, `monitor_state.py`, and the method groups it calls through class aliases: `inventory_walk.py`, `deep_inventory.py`, `generator_sweep.py`, `numpy_frames.py`, `digest.py` |
| `tensor_utils.py` | `safe_copy`, `safe_to`, `tensor_nanequal`, tensor memory helpers |
| `_sparse_parts.py` | Sparse tensor components per layout (`sparse_component_tensors`) and rebuilding a sparse tensor with new values (`sparse_with_values`); torch-only leaf |
| `alias_footprint.py` | THE absolute-byte three-valued alias/overlap engine (r37 INV-2), split out of `tensor_utils.py` |
| `_subprocess.py` | The ONE bounded-subprocess spawn discipline (group teardown, no orphaned grandchildren); never hand-roll `subprocess.run` for children |
| `_torch_compat.py` | LOCKED chokepoint for every fragile torch-private probe / cross-version signature; `HAS_*` capability flags (see root `AGENTS.md`) |
| `_callable_safety.py` | Security gate deciding which resolved callables are pure forward/tensor ops (untrusted `.tlspec` registry) |
| `_stdlib_denial.py` | The stdlib and builtin module denial set behind `_callable_safety` (`is_denied_stdlib_or_builtin_module`); stdlib-only leaf |
| `_multipass_access.py` | Multi-pass-safe attribute access for aggregate (recurrent) `Layer` objects |
| `_torch_symbols.py` | Single sanctioned spelling for resolving top-level `torch` attributes on the load/decode/exec path |
| `_uninit_alloc.py` | Closed uninitialized-memory value-source name table (the ONE shared predicate block; import it directly) |
| `env_flags.py` | THE closed-vocabulary boolean env-knob parser (`closed_bool_env`); torchlens-owned on/off knobs must parse through it, never exact-`"1"` or raw truthiness (round-7 R47) |
| `__init__.py` | 1,000+ lines of PUBLIC API — `doctor()`, `list_modules`/`list_ops`, `flop_count`, `peek_graph`, `synthetic_input`, `find_executable_save_set`, `trace_streaming`, and the `_LAZY_EXPORTS` `__getattr__`; NOT editable boilerplate |

(Source-link helpers live at top level in `torchlens/_source_links.py`, not in this package.)

## Tensor Operations
- `safe_copy()` clones through clean torch functions and preserves TorchLens raw-label
  metadata when the logging pipeline needs it.
- `safe_to()` moves tensors under `pause_logging()`.
- `tensor_nanequal()` is NaN-aware and complex-aware.
- `get_memory_amount()` deliberately AVOIDS `pause_logging()`: it resolves the unwrapped
  size methods once instead of toggling global logging state per tensor (hot-path perf).
- `MAX_FLOATING_POINT_TOLERANCE` is a legacy fp32-only facade (the float32 row
  of `_DTYPE_FLOAT_TOLERANCES`); validation derives per-dtype tolerances via
  `derive_float_tolerances`/`_tolerances_for_dtype` in `validation/core.py`
  and no longer consumes this symbol.

## RNG and Autocast
- `log_current_rng_states()` and `set_rng_from_saved_states()` (`rng/streams.py`) support deterministic replay.
- `RngStreamsRestore` (`rng/restore.py`) is the one door for leaving the user's global streams as
  found: `tl.trace` registers its `restore` on every settlement path, replay brackets with it, and
  `tl.validate` is decorated with `restores_rng_streams`.
- Autocast state helpers (`rng/autocast.py`) are used by wrapper execution context and validation replay.
- Capture RNG before entering `active_logging()`.
- CUDA RNG state is snapshotted ONLY once the process has initialized CUDA
  (`_snapshot_cuda_rng_states()`): `torch.cuda.get_rng_state_all()` eagerly initializes every
  visible device, which made a pure-CPU capture abort on a visible-but-unusable CUDA stack. A
  failing read degrades to a warning and latches off (`rng.streams._cuda_rng_unusable`, re-armed per capture by `set_random_seed`) for both snapshot
  and restore; real CUDA captures are byte-identical because they have initialized CUDA already.

## Hashing
- `make_random_barcode()` supports barcode nesting detection.
- `make_short_barcode_from_input()` feeds operation equivalence.
- `compute_graph_shape_hash()` is used during postprocess finalization.

## Introspection
- `get_vars_of_type_from_obj()` is a bounded recursive finder for tensors/modules.
- `_ATTR_SKIP_SET` avoids expensive tensor pseudo-properties.
- The CUDA probes live in the leaf module `_cuda_probe.py` (torch only, no TorchLens imports,
  so `rng/streams.py` and the cleanup paths read them without joining the capture import cycle):
  `_is_cuda_available()` caches CUDA availability (`_cuda_available`) to avoid repeated driver
  probes, and treats a raising probe as "no CUDA" (warned once) so a broken accelerator cannot
  abort a CPU capture; `_is_cuda_initialized()` is the uncached, probe-free read of torch's own
  init flag; use it to skip opportunistic CUDA work instead of force-initializing devices.

## Gotchas
- Clean torch function imports must happen before decoration or use originals from `_state`.
- Avoid importing high-level modules here; utility imports should stay low in the dependency graph.
- Validation tolerance and quantized tensor behavior are known edge cases.
- Hash collisions are possible; do not use short hashes as security or persistence identifiers.
