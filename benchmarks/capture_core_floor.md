# Call-time floor counters

`capture_core_floor.py` counts, for one warmed `tl.trace`, every Python and C call made inside the forward (`_capture/run.py::_run_forward`) and splits them into the sections the call-time lanes work on. Each count is divided by the number of function outputs the trace logged (calls of `_log_output_tensor_info`), not by rows. `capture_core_cost.py` records them under each model's `floor` block for the six floor models (chain1000, varied200, loopcell100, resnet18, gpt2_tiny4, mod_relu200), so `--check` and `--compare` see them.

## Counter names

`floor.<model>.<name>`, where `<name>` is one of:

| name | what it counts | marker functions (`SECTION_MARKERS`) |
|---|---|---|
| `arg_scan_per_output` | argument scan, parents, versions, parameters, slot map, source resolution, argument summaries | `_extract_arg_tensors_and_params`, `get_arg_tensors_for_resolution`, `split_non_tensor_args`, `_bind_call_alias_labels`, `_repin_rebound_receiver`, `resolve_or_late_index_buffer_address`, `log_source_tensor`, `log_closure_constant_if_known`, `snapshot_arg_slots`, `parents_from_slots`, `_set_arg_template_fields`, `_build_graph_relationship_fields`, `_build_param_fields`, `witness_from_slots`, `extract_salient_args` |
| `call_site_per_output` | call-site capture | `_set_call_site_fields`, `_get_code_context` |
| `autograd_per_output` | the output walk and autograd saved-tensor statistics | `_partition_output_entries_with_autograd_stats` |
| `output_facts_per_output` | shape, memory, gradient function, in-place and edit facts of each output | `_metadata_shape`, `get_memory_amount_from_metadata`, `recorded_op_grad_fn`, `pop_user_grad_fn`, `pop_same_object_mutation`, `_pop_tensor_live_fire_results`, `_label_version_baseline` |
| `payload_per_output` | the saved payload: copy, save budget, duplicate checks | `_save_activation_fields`, `_stream_activation_fields`, `_save_call_args_fields` |
| `emit_rest_per_output` | the rest of the emit (`log_function_output_tensors`): fields dict, row, commit, equivalence classes | (the emit itself) |
| `wrapper_outside_emit_per_output` | the function wrapper (`wrapped_func`) outside the emit: admission, the operation, label handling, nested wrapper calls | (the wrapper itself) |
| `module_handling_per_output` | TorchLens code in the forward outside the function wrapper: the session forward, module entry and exit records, module output logging | (any `torchlens` code reached from the model's code) |
| `model_code_per_output` | the model's own code and torch's module-call machinery | (the rest) |
| `forward_total_per_output` | the sum of the nine sections above | |
| `label_read_stamp_per_output` | overlay: calls inside `get_tensor_label` and `set_tensor_label` (`_utils/_tensor_label.py`) wherever they run in the forward; also counted in their section | |
| `hits.<section>_per_output` | marker calls that opened each of the five emit sections | |
| `outputs` | function outputs of the counted trace (the divisor) | |

## Rules for lanes

- Attribution: inside the emit, the first marker reached opens its section and keeps every call beneath it (the outermost section wins). A nested wrapper call made by the payload's copy counts as payload.
- A lane that moves a section's work into a new function adds that function's name to the section's markers in `SECTION_MARKERS` in the same change. Names match in any module of the `torchlens` package, so moving a function between modules needs no change.
- `capture_core_cost.py --check` runs `check_markers`: a section whose markers fired at the baseline and fire no more fails, because its counter would read low while the work moved to `emit_rest`.
- Bars: a lane bar names a counter above, for example `floor.chain1000.call_site_per_output` at 35% of its baseline. The baselines are in `perf_baselines/capture_core_bars.json` under `baseline`, measured at commit ff8c7700df48de07b9abd7e42880f4f2fc3aa019 with `--counters` (the C1 to C9 baselines in that file are from the earlier commit named in its `base_commit`).
- Two counts of one commit agree to within a few calls per trace (`python benchmarks/capture_core_floor.py --compare a.json b.json` lists every counter that differs and fails one beyond the bars' 0.5% tolerance). `--counters` runs each model in a fresh interpreter with `PYTHONHASHSEED=0` and address randomization off; what still varies is whether the allocator hands a new tensor a storage address already in the label stamp's alias index (`_register_storage_alias`), which moves 1 to 7 calls in a trace of 1.4 to 2.7 million.

## Wall time

`python benchmarks/capture_core_floor.py --wall --out wall.json` runs the decision measurement in shadow mode (`capture_core_floor_shadow.py`): per model, the whole trace, the forward, the forward less today's emit, the six observation sections timed through today's helpers, and the bare model, in microseconds per function output, minimum of nine traces, one thread. Wall never gates a lane; it is reported on a lane's tip and its base in one job and gates the checkpoint lane.
