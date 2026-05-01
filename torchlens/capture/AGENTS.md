# capture/ - Implementation Guide

## Label Formats
- Source tensors: `{type}_{num}_raw`, for example `input_0_raw` or `buffer_1_raw`.
- Function outputs: `{type}_{num}_{counter}_raw`, for example `conv2d_1_5_raw`.
- Labels are raw during capture and become final labels in `postprocess/labeling.py`.
- Pass-qualified final labels use `{label}:{pass_num}`.

## arg_positions.py
- Main entry point: `extract_tensors_and_params(args, kwargs, func_name)`.
- Lookup order: `FUNC_ARG_SPECS` static table -> `_DYNAMIC_SPEC_CACHE` -> BFS fallback.
- `ArgSpec` stores tensor arg indexes, tensor kwarg names, param arg indexes, and param kwarg names.
- Keep keyword handling accurate; stale entries can hide graph parents.

## salient_args.py
- Uses `@_register()` per function/layer family.
- `_build_arg_name_map()` maps positional args to names.
- Extractors are failure-safe and return `{}` on unexpected errors.
- Metadata is display-oriented; never let it affect graph correctness.

## flops.py
- Zero-FLOPs ops, elementwise ops, and specialty handlers feed
  `compute_forward_flops()` and `compute_backward_flops()`.
- `register_op_rule()` is the extension point.
- MAC convention is 2 FLOPs.

## output_tensors.py Gotchas
- In-place ops rely on `safe_copy()` stripping `tl_tensor_label_raw` from clones so the
  mutation is logged as a new operation.
- Barcode nesting detection distinguishes bottom-level ops from wrapper-level composites.
- `activation_postfunc` must run under `pause_logging()`.
- Live intervention hooks run in this layer; hook output validation lives under
  `intervention/runtime.py`.
- Fast-path validation must check tensor existence, execution order, and parent alignment.

## source_tensors.py Gotchas
- Input and buffer roots must increment counters consistently with exhaustive and fast paths.
- Buffer duplicate guards prevent repeated buffer source nodes when many ops read one buffer.

## backward.py Gotchas
- Backward capture walks autograd `grad_fn` links and installs hooks after forward capture.
- Gradient streaming reuses `_io` bundle refs and must finalize/evict in postprocess finalization.
- Validation for backward lives in `validation/backward.py`.

## Known Risks
- Dynamic `arg_positions` cache is process-local and is not automatically invalidated across
  torch version changes.
- Keyword tensor coverage should be checked whenever adding static specs.
- Fast capture assumes deterministic graph shape between passes; random/control-flow drift is
  a hard error.
