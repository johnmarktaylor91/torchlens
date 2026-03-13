# capture/ — Real-Time Tensor Operation Logging

## What This Does
Captures every tensor operation during a model's forward pass. Handles both exhaustive
mode (full metadata, ~131 fields per operation) and fast mode (reuses prior graph structure,
saves only requested tensor data).

## Files

| File | ~Lines | Purpose |
|------|--------|---------|
| `trace.py` | 500 | Forward-pass orchestration: input normalization, model execution, session setup/cleanup |
| `output_tensors.py` | 898 | Core logging: builds LayerPassLog entries, exhaustive/fast path split, identity detection |
| `source_tensors.py` | 357 | Logs input and buffer tensors as source nodes in the graph |
| `tensor_tracking.py` | 407 | Barcode system, parent-child links, backward hooks, arg hashing |
| `arg_positions.py` | 961 | O(1) tensor extraction via 3-tier lookup: static table (639 entries), dynamic cache, BFS fallback |
| `salient_args.py` | 444 | Extracts significant function args (hyperparameters) for metadata. 27 extractors for 50+ layer types |
| `flops.py` | 1393 | Per-operation FLOPs computation (3-tier: zero/elementwise/specialty, ~290 ops) |

## How It Connects

Decorated wrappers in `decoration/torch_funcs.py` call into this package for every logged
operation. `trace.py` orchestrates the forward pass, `output_tensors.py` builds the
LayerPassLog entries that `postprocess/` later cleans up and labels. `source_tensors.py`
creates the source nodes (inputs, buffers) that form the graph roots.

The exhaustive/fast split is the key architectural boundary: exhaustive captures everything,
fast reuses the prior graph structure and only saves tensor data for requested layers.
Counter alignment between the two paths is maintained via identical increment logic in
both `output_tensors.py` and `source_tensors.py`.

## Key Functions

### trace.py
- `run_and_log_inputs_through_model()` — Core runner. Normalizes inputs, prepares model,
  runs forward pass inside `active_logging()` context, then triggers postprocessing.
- `save_new_activations()` — Re-logs activations for new inputs on already-traced model.
- Critical ordering: RNG capture → active_logging → model forward → cleanup → postprocess

### output_tensors.py
- `log_function_output_tensors_exhaustive()` — Creates 131-field dict per output tensor,
  builds LayerPassLog, tracks parent-child relationships
- `log_function_output_tensors_fast()` — Maps operation counter to existing raw label,
  validates function name matches, updates only tensor data + timing + RNG
- `_output_should_be_logged()` — Tensor logged if unlabeled OR bottom-level function

### tensor_tracking.py
- `_get_operation_equivalence_type()` — Structural fingerprint for loop detection:
  `{func_name}_{arg_hash}[_outindex{i}][_module{origin}]`
