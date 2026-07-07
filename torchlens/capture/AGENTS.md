# capture/ - Implementation Guide

## Label Formats
- Source tensors: `{type}_{num}_raw`, for example `input_0_raw` or `buffer_1_raw`.
- Function outputs: `{type}_{num}_{counter}_raw`, for example `conv2d_1_5_raw`.
- Labels are raw during capture and become final labels in `postprocess/labeling.py`.
- Pass-qualified final labels use `{label}:{call_index}`.

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

## projections.py Gotchas
- Sparse `Recording` projections must preserve pass indexes, raw labels, and payload refs.
- Full `Trace` projections must preserve backend-neutral event metadata until postprocess finalizes
  labels and graph structure.

## predicates.py / stop.py Gotchas
- `followed_by` must stay correct-or-loud for unsupported predicate shapes.
- Halt and nonfinite directives must return partials only through the explicit StopDirective policy.
- Validation for backward lives in `validation/backward.py`.

## Known Risks
- Dynamic `arg_positions` cache is process-local and is not automatically invalidated across
  torch version changes.
- Keyword tensor coverage should be checked whenever adding static specs.
- Predicate-backed selective capture assumes deterministic graph shape for validation-sensitive
  paths; random/control-flow drift must fail clearly when alignment is required.
- `torch.func` / functorch wrappers log transform boundary ops and run the inner callable
  under paused logging. Preserve the boundary parent edge and transform metadata.
- Unlabeled tensor args are provenance markers, not graph parents. Inputs, params, buffers,
  and module tensor attributes should remain known sources; foreign captured tensors should warn.
