# Trace: the object a capture returns

## Size and labels

### trace-facts-counts
How big a capture is: entries, edges, boundary layers and parameter counts.

- covers: `Trace.__len__`, `Trace.num_edges`, `Trace.num_input_layers`, `Trace.num_output_layers`, `Trace.num_buffer_layers`, `Trace.num_params`, `Trace.num_params_trainable`, `Trace.num_params_frozen`, `Trace.has_trainable_params`, `Trace.has_frozen_params`, `Trace.num_uncalled_modules`
- tasks: inspect, summary
- fixture: toy

```python
trace = tl.trace(model, x)
print("len", len(trace))
print("num_edges", trace.num_edges)
print("num_input_layers", trace.num_input_layers)
print("num_output_layers", trace.num_output_layers)
print("num_buffer_layers", trace.num_buffer_layers)
print("num_params", trace.num_params)
print("num_params_trainable", trace.num_params_trainable)
print("num_params_frozen", trace.num_params_frozen)
print("has_trainable_params", trace.has_trainable_params)
print("has_frozen_params", trace.has_frozen_params)
print("num_uncalled_modules", trace.num_uncalled_modules)
# Output:
# len 5
# num_edges 4
# num_input_layers 1
# num_output_layers 1
# num_buffer_layers 0
# num_params 58
# num_params_trainable 58
# num_params_frozen 0
# has_trainable_params True
# has_frozen_params False
# num_uncalled_modules 0
```

### trace-facts-label-lists
The label lists a trace keeps: layers, boundary layers and graph sources and sinks.

- covers: `Trace.layer_labels`, `Trace.input_layers`, `Trace.output_layers`, `Trace.layers_with_params`, `Trace.buffer_layers`, `Trace.internal_source_ops`, `Trace.internal_sink_ops`, `Trace.internally_terminated_bool_ops`
- tasks: inspect
- fixture: toy

```python
trace = tl.trace(model, x)
print("layer_labels", trace.layer_labels)
print("input_layers", trace.input_layers)
print("output_layers", trace.output_layers)
print("layers_with_params", trace.layers_with_params)
print("buffer_layers", trace.buffer_layers)
print("internal_source_ops", trace.internal_source_ops)
print("internal_sink_ops", trace.internal_sink_ops)
print("internally_terminated_bool_ops", trace.internally_terminated_bool_ops)
# Output:
# layer_labels ['input_1', 'linear_1_1', 'relu_1_2', 'linear_2_3', 'output_1']
# input_layers ['input_1']
# output_layers ['output_1']
# layers_with_params {'linear_1_1': ['linear_1_1'], 'linear_2_3': ['linear_2_3']}
# buffer_layers []
# internal_source_ops []
# internal_sink_ops []
# internally_terminated_bool_ops []
```

### trace-facts-identity
What was traced and the trace's own name, state and format version.

- covers: `Trace.name`, `Trace.model_name`, `Trace.state`, `Trace.state_history`, `Trace.tlspec_version`, `Trace.profile_enabled`
- tasks: inspect
- fixture: toy

```python
trace = tl.trace(model, x)
print("name", trace.name)
print("model_name", trace.model_name)
print("state", trace.state)
print("state_history", trace.state_history)
print("tlspec_version", trace.tlspec_version)
print("profile_enabled", trace.profile_enabled)
# Output:
# name toy_...
# model_name Toy
# state TraceState.PRISTINE
# state_history []
# tlspec_version 11
# profile_enabled False
```

### trace-facts-capture-flags
Flags about how the capture ran and stopped.

- covers: `Trace.is_stopped`, `Trace.stop_reason`, `Trace.is_replayable`, `Trace.capture_guard_passes`, `Trace.capture_tensor_grad_hooks`, `Trace.num_context_lines`, `Trace.save_visualizations`, `Trace.batch_render`
- tasks: inspect
- fixture: toy

```python
trace = tl.trace(model, x)
print("is_stopped", trace.is_stopped)
print("stop_reason", trace.stop_reason)
print("is_replayable", trace.is_replayable)
print("capture_guard_passes", len(trace.capture_guard_passes))
print("capture_tensor_grad_hooks", trace.capture_tensor_grad_hooks)
print("num_context_lines", trace.num_context_lines)
print("save_visualizations", trace.save_visualizations)
print("batch_render", trace.batch_render)
# Output:
# is_stopped False
# stop_reason None
# is_replayable False
# capture_guard_passes 1
# capture_tensor_grad_hooks True
# num_context_lines 7
# save_visualizations False
# batch_render auto
```

### trace-facts-empty-by-default
Fields that stay empty on a plain forward: no kept input, transform, renderer or backward peak.

- covers: `Trace.input`, `Trace.transformed_output`, `Trace.backend_runtime_device_summary`, `Trace.backward_peak_memory_reason`, `Trace.layer_visualizers`, `Trace.last_run`
- tasks: inspect
- fixture: toy

```python
trace = tl.trace(model, x)
print("input", trace.input)
print("transformed_output", trace.transformed_output)
print("backend_runtime_device_summary", trace.backend_runtime_device_summary)
print("backward_peak_memory_reason", trace.backward_peak_memory_reason)
print("layer_visualizers", trace.layer_visualizers)
print("last_run", trace.last_run)
# Output:
# input None
# transformed_output None
# backend_runtime_device_summary None
# backward_peak_memory_reason None
# layer_visualizers None
# last_run None
```

### trace-output-labels-decode
A model with `config.id2label` gets its output labels recorded, and decodes by name.

- covers: `Trace.output_id2label`, `Trace.output_num_classes`, `Trace.decode_output`, `Trace.output_table`
- tasks: output, decode
- fixture: toy

```python
from types import SimpleNamespace
model.config = SimpleNamespace(id2label={0: "cat", 1: "dog"})
trace = tl.trace(model, x)
print("output_id2label", trace.output_id2label)
print("output_num_classes", trace.output_num_classes)
print(type(trace.decode_output(k=1)).__name__)
table = trace.output_table(k=1)
print(len(table), list(table.columns))
# Output:
# output_id2label {0: 'cat', 1: 'dog'}
# output_num_classes 2
# dict
# 3 ['batch_item', 'rank', 'label', 'prob']
```

### trace-facts-bookkeeping
Per-trace diagnostics and registries; empty on a clean capture.

- covers: `Trace.annotations`, `Trace.escape_diagnostics`, `Trace.completeness_diagnostics`, `Trace.facet_functions`, `Trace.intervention_audit`
- tasks: inspect
- fixture: toy

```python
trace = tl.trace(model, x)
print("annotations", trace.annotations)
print("escape_diagnostics", trace.escape_diagnostics)
print("completeness_diagnostics", trace.completeness_diagnostics)
print("facet_functions", type(trace.facet_functions).__name__)
print("intervention_audit", trace.intervention_audit)
# Output:
# annotations {}
# escape_diagnostics []
# completeness_diagnostics []
# facet_functions FacetRegistrySnapshot
# intervention_audit []
```

## Looking up records

### trace-getitem-label-or-selector
Index a trace by label for one record, or by a selector for a Selection.

- covers: `Trace.__getitem__`
- tasks: lookup, select
- fixture: toy

```python
trace = tl.trace(model, x)
layer = trace["relu_1_2"]
print(type(layer).__name__, layer.label)
selection = trace[tl.func("relu")]
print(type(selection).__name__, selection.labels)
# Output:
# Layer relu_1_2:1
# Selection ('relu_1_2:1',)
```

### trace-iterate
Iterating a trace walks its entries in execution order.

- covers: `Trace.__iter__`
- tasks: lookup, iterate
- fixture: toy

```python
trace = tl.trace(model, x)
for entry in trace:
    print(type(entry).__name__, entry.label)
# Output:
# Op input_1:1
# Op linear_1_1:1
# Op relu_1_2:1
# Op linear_2_3:1
# Op output_1:1
```

### trace-boundary-op-views
Filtered op views: inputs, outputs, real compute and ops that use parameters.

- covers: `Trace.input_ops`, `Trace.output_ops`, `Trace.compute_ops`, `Trace.ops_with_params`
- tasks: lookup
- fixture: toy

```python
trace = tl.trace(model, x)
print("input_ops", [op.label for op in trace.input_ops])
print("output_ops", [op.label for op in trace.output_ops])
print("compute_ops", [op.label for op in trace.compute_ops])
print("ops_with_params", [op.label for op in trace.ops_with_params])
# Output:
# input_ops ['input_1:1']
# output_ops ['output_1:1']
# compute_ops ['linear_1_1:1', 'relu_1_2:1', 'linear_2_3:1']
# ops_with_params ['linear_1_1:1', 'linear_2_3:1']
```

### trace-saved-views
Views that keep only records with saved activations.

- covers: `Trace.saved_ops`, `Trace.saved_layers`, `Trace.saved_modules`, `Trace.saved_module_calls`
- tasks: lookup, select-save
- fixture: toy

```python
trace = tl.trace(model, x, save=tl.module("fc1"))
print("saved_ops", [op.label for op in trace.saved_ops])
print("saved_layers", len(trace.saved_layers))
print("saved_modules", len(trace.saved_modules))
print("saved_module_calls", len(trace.saved_module_calls))
# Output:
# saved_ops ['linear_1_1:1']
# saved_layers 1
# saved_modules 1
# saved_module_calls 1
```

### trace-params-buffers
Parameter records by address; a model without buffers has an empty buffer view.

- covers: `Trace.params`, `Trace.buffers`
- tasks: lookup, params
- fixture: toy

```python
trace = tl.trace(model, x)
print("params", len(trace.params))
print("fc1.weight", tuple(trace.params["fc1.weight"].shape))
print("buffers", len(trace.buffers))
# Output:
# params 4
# fc1.weight (8, 4)
# buffers 0
```

### trace-module-calls-root
Module calls by call label, the root module and modules never called.

- covers: `Trace.module_calls`, `Trace.root_module`, `Trace.uncalled_modules`
- tasks: lookup, modules
- fixture: toy

```python
trace = tl.trace(model, x)
print("module_calls", len(trace.module_calls))
print("root_module", type(trace.root_module).__name__)
print("uncalled_modules", list(trace.uncalled_modules))
# Output:
# module_calls 3
# root_module Module
# uncalled_modules []
```

### trace-uncalled-modules
A registered module the forward never calls shows up as uncalled.

- covers: `Trace.uncalled_modules`, `Trace.num_uncalled_modules`
- tasks: modules, debug
- fixture: toy

```python
model.spare = torch.nn.Linear(2, 2)
trace = tl.trace(model, x)
print("num_uncalled_modules", trace.num_uncalled_modules)
print("uncalled_modules", [str(m) for m in trace.uncalled_modules])
# Output:
# num_uncalled_modules 1
# uncalled_modules ['spare']
```

## Module call tree

### trace-call-tree-walk
The root of the call tree and a depth-first walk over every module call.

- covers: `Trace.call_tree`, `Trace.walk_calls`
- tasks: modules, call-tree
- fixture: toy

```python
trace = tl.trace(model, x)
print("root", type(trace.call_tree).__name__)
for call in trace.walk_calls():
    print(call.label)
# Output:
# root ModuleCall
# self:1
# fc1:1
# fc2:1
```

### trace-show-call-tree
Print the module call tree as ASCII, one line per module call.

- covers: `Trace.show_call_tree`
- tasks: modules, call-tree
- fixture: toy

```python
trace = tl.trace(model, x)
trace.show_call_tree()
# Output:
# self:1 (forward_duration: ...)
# |-- fc1:1 (forward_duration: ...)
# `-- fc2:1 (forward_duration: ...)
```

### trace-module-collapse-order
Modules ranked by how good a collapse point they are when drawing.

- covers: `Trace.module_collapse_order`
- tasks: draw, modules
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace.module_collapse_order)
# Output:
# [('0', 0.0), ('1', 0.0), ('2', 0.0), ('self', 0.0)]
```

## Summaries and reports

### trace-summary
A text summary of the capture, one row per layer.

- covers: `Trace.summary`
- tasks: summary
- fixture: toy

```python
trace = tl.trace(model, x)
report = trace.summary()
print(type(report).__name__)
print(str(report).splitlines()[0])
# Output:
# SummaryReport
# Toy | input (3, 4) float32
```

### trace-to-pandas
One row per layer pass as a pandas table.

- covers: `Trace.to_pandas`
- tasks: summary, table
- fixture: toy

```python
trace = tl.trace(model, x)
frame = trace.to_pandas()
print(len(frame))
print(list(frame["layer_label"]))
# Output:
# 5
# ['input_1', 'linear_1_1', 'relu_1_2', 'linear_2_3', 'output_1']
```

### trace-to-agent-json
A machine-readable dump of the trace for an agent.

- covers: `Trace.to_agent_json`
- tasks: summary, agent
- fixture: toy

```python
trace = tl.trace(model, x)
dump = trace.to_agent_json(max_ops=3)
print(type(dump).__name__)
print(sorted(dump)[:8])
# Output:
# dict
# ['capture', 'counts', 'guide', 'health', 'inputs', 'layer_labels', 'logged_values', 'memory']
```

### trace-explain
Explain a trace in plain language.

- covers: `Trace.explain`
- tasks: summary, explain
- fixture: toy

```python
trace = tl.trace(model, x)
report = trace.explain()
print(type(report).__name__)
print(str(report).splitlines()[0])
# Output:
# ExplainReport
# TorchLens report
```

### trace-audit
A one-call health report for a finished trace.

- covers: `Trace.audit`
- tasks: health, debug
- fixture: toy

```python
trace = tl.trace(model, x)
report = trace.audit()
print(type(report).__name__)
print(str(report).splitlines()[0])
# Output:
# HealthReport
# HealthReport: 0 findings, 3 checks run, 7 skipped
```

### trace-capabilities
Which report families this trace can answer.

- covers: `Trace.capabilities`
- tasks: summary
- fixture: toy

```python
trace = tl.trace(model, x)
card = trace.capabilities()
print(type(card).__name__)
# Output:
# CapabilityCard
```

### trace-bill-of-materials
Inventory of what this trace actually contains.

- covers: `Trace.bill_of_materials`
- tasks: summary, save
- fixture: toy

```python
trace = tl.trace(model, x, save=tl.func("relu"))
bom = trace.bill_of_materials()
print(sorted(bom))
# Output:
# ['activations', 'annotations', 'backward', 'buffers', 'capture', 'graph', 'logged_values', 'parameters']
```

### trace-narrate
The trace's events in the live-narration grammar.

- covers: `Trace.narrate`
- tasks: summary, narrate
- fixture: toy

```python
trace = tl.trace(model, x)
text = str(trace.narrate(last=3))
print(text)
# Output:
# #3      relu_1_2:1                [3,8] f32 cpu
#   #4      linear_2_3:1              [3,2] f32 cpu           @fc2
# #5      output_1:1                [3,2] f32 cpu           @fc2
```

## Fact families

### trace-fact-families
The grouped fact objects hanging off a trace.

- covers: `Trace.graph`, `Trace.autograd`, `Trace.cost`, `Trace.factcore`, `Trace.health`, `Trace.health_facts`, `Trace.provenance`, `Trace.source`, `Trace.options`, `Trace.outcome`, `Trace.model_profile`
- tasks: inspect, facts
- fixture: toy

```python
trace = tl.trace(model, x)
print("graph", type(trace.graph).__name__)
print("autograd", type(trace.autograd).__name__)
print("cost", type(trace.cost).__name__)
print("factcore", type(trace.factcore).__name__)
print("health", type(trace.health).__name__)
print("health_facts", type(trace.health_facts).__name__)
print("provenance", type(trace.provenance).__name__)
print("source", type(trace.source).__name__)
print("options", type(trace.options).__name__)
print("outcome", type(trace.outcome).__name__)
print("model_profile", type(trace.model_profile).__name__)
# Output:
# graph TraceGraph
# autograd TraceAutograd
# cost TraceCost
# factcore FactCore
# health TraceHealth
# health_facts HealthFacts
# provenance TraceProvenance
# source TraceSource
# options TraceSettings
# outcome CaptureOutcome
# model_profile ModelProfile
```

### trace-compute-coverage
How many rows have known compute counts, and which ops lack them.

- covers: `Trace.compute_coverage`, `Trace.unknown_flop_ops`, `Trace.macs_unknown_split_ops`
- tasks: cost, flops
- fixture: toy

```python
trace = tl.trace(model, x)
print("compute_coverage", trace.compute_coverage)
print("unknown_flop_ops", trace.unknown_flop_ops)
print("macs_unknown_split_ops", trace.macs_unknown_split_ops)
# Output:
# compute_coverage {'known': 3, 'zero_by_rule': 0, 'not_applicable': 2, 'unknown': 0}
# unknown_flop_ops ()
# macs_unknown_split_ops ()
```

### trace-profile
A resource profile table per op or per module.

- covers: `Trace.profile`
- tasks: cost, profile
- fixture: toy

```python
trace = tl.trace(model, x)
profile = trace.profile(level="module", sort_by="param_count")
print(type(profile).__name__)
# Output:
# ProfileReport
```

## Saving and loading

### trace-save-load
Save a trace to a `.tlspec` folder and load it back.

- covers: `Trace.save`, `tl.load`
- tasks: save, load
- fixture: toy

```python
trace = tl.trace(model, x, save=tl.func("relu"))
trace.save("run.tlspec")
loaded = tl.load("run.tlspec")
print(loaded.op_labels == trace.op_labels)
print(torch.equal(loaded.ops["relu_1_2"].out, trace.ops["relu_1_2"].out))
# Output:
# True
# True
```

### trace-readiness
Save a runnable trace, load it, and ask what it can do without the model.

- covers: `Trace.readiness`
- tasks: load
- fixture: toy

```python
options = tl.options.CaptureOptions(replayable=True)
trace = tl.trace(model, x, capture=options, save="all")
trace.save("run.tlspec", level="runnable", include_weights=True)
loaded = tl.load("run.tlspec")
print(str(loaded.readiness).split(",")[0])
# Output:
# ReadinessReport(status=ready
```

### trace-archived-activations
Activations carried in a runnable archive, readable for inspection.

- covers: `Trace.archived_activations`
- tasks: load
- fixture: toy

```python
options = tl.options.CaptureOptions(replayable=True)
trace = tl.trace(model, x, capture=options, save="all")
trace.save("run.tlspec", level="runnable", include_weights=True)
loaded = tl.load("run.tlspec")
print(type(loaded.archived_activations).__name__, len(loaded.archived_activations))
# Output:
# dict 0
```

## Forks and edits

### trace-fork-do
Fork a trace and ablate one op in the fork; the original keeps its values.

- covers: `Trace.fork`, `Trace.do`, `Trace.parent_trace`, `Trace.root_trace`, `Trace.is_pristine`
- tasks: intervene, fork
- fixture: toy

```python
trace = tl.trace(model, x)
fork = trace.fork("ablated")
fork.do(tl.func("relu"), tl.zero_ablate())
print("fork relu zero", bool((fork.ops["relu_1_2"].out == 0).all()))
print("original relu zero", bool((trace.ops["relu_1_2"].out == 0).all()))
print("parent is trace", fork.parent_trace is trace)
print("root is trace", fork.root_trace is trace)
print("pristine", trace.is_pristine, fork.is_pristine)
# Output:
# fork relu zero True
# original relu zero False
# parent is trace True
# root is trace True
# pristine True False
```

### trace-fork-scoped
A `with` block around a fork scopes its edits: on exit they detach, so a rerun is clean.

- covers: `Trace.__enter__`, `Trace.__exit__`
- tasks: intervene, fork
- fixture: toy

```python
trace = tl.trace(model, x)
with trace.fork() as scoped:
    scoped.do(tl.func("relu"), tl.zero_ablate())
    print("inside", bool((scoped.ops["relu_1_2"].out == 0).all()))
scoped.run(model, x)
print("rerun after exit", bool((scoped.ops["relu_1_2"].out == 0).all()))
# Output:
# inside True
# rerun after exit False
```

### trace-recipe-clear-edits
The recipe lists the staged edits; `clear_edits` detaches them.

- covers: `Trace.recipe`, `Trace.clear_edits`
- tasks: intervene
- fixture: toy

```python
fork = tl.trace(model, x).fork()
fork.do(tl.func("relu"), tl.zero_ablate())
print("hooks staged", len(fork.recipe.hook_specs), len(fork.recipe.target_value_specs))
fork.clear_edits(tl.func("relu"))
print("after clear", len(fork.recipe.hook_specs), len(fork.recipe.target_value_specs))
# Output:
# hooks staged 1 0
# after clear 0 0
```

### trace-intervention-audit
Each edit leaves an audit entry naming what it touched.

- covers: `Trace.intervention_audit`
- tasks: intervene, audit
- fixture: toy

```python
trace = tl.trace(model, x)
fork = trace.fork()
fork.do(tl.func("relu"), tl.zero_ablate())
print(len(fork.intervention_audit))
# Output:
# 1
```

### trace-run-rerun
Rerun an edited trace on a new input; `last_run` and `last_run_records` describe the run.

- covers: `Trace.run`, `Trace.last_run`, `Trace.last_run_records`
- tasks: rerun, intervene
- fixture: toy

```python
edit = tl.when(tl.func("relu"), tl.zero_ablate())
trace = tl.trace(model, x, intervene=edit)
trace.run(model, torch.randn(3, 4))
print("engine", trace.last_run["engine"])
print("records", len(trace.last_run_records()))
print("relu zero", bool((trace.ops["relu_1_2"].out == 0).all()))
# Output:
# engine rerun
# records 0
# relu zero True
```

### trace-save-intervention
Save a fork's edit recipe so another session can re-apply it.

- covers: `Trace.save_intervention`
- tasks: intervene, save
- fixture: toy

```python
fork = tl.trace(model, x).fork()
fork.do(tl.func("relu"), tl.zero_ablate())
fork.save_intervention("ablate.tlspec")
import os
print(os.path.isdir("ablate.tlspec"))
# Output:
# True
```

### trace-replace-state-from
Swap in the run-state of another trace of the same model.

- covers: `Trace.replace_state_from`
- tasks: rerun
- fixture: toy

```python
trace = tl.trace(model, x)
fresh = tl.trace(model, x * 2)
trace.replace_state_from(fresh)
print(torch.equal(trace.ops["relu_1_2"].out, fresh.ops["relu_1_2"].out))
# Output:
# True
```

## Sites and regions

### trace-find-sites
A table of intervention sites matching a query; `.first()` is the first match.

- covers: `Trace.find_sites`
- tasks: select, intervene
- fixture: toy

```python
trace = tl.trace(model, x)
sites = trace.find_sites(tl.func("linear"))
print(len(sites))
print(tuple(sites.first().out.shape))
# Output:
# 2
# (3, 8)
```

### trace-sites
Every structural site of the trace as one table.

- covers: `Trace.sites`
- tasks: select
- fixture: toy

```python
trace = tl.trace(model, x)
frame = trace.sites()
print(type(frame).__name__, len(frame))
# Output:
# DataFrame 5
```

### trace-stack-passes
Stack the saved outputs of every pass a selector matches, in execution order.

- covers: `Trace.stack`
- tasks: loops, multi-pass
- fixture: loop

```python
trace = tl.trace(model, x)
stacked = trace.stack(tl.func("tanh"))
print(stacked.labels)
print(tuple(stacked.tensor.shape))
# Output:
# ('tanh_1_2', 'tanh_1_2', 'tanh_1_2')
# (3, 2, 4)
```

### trace-between
The ops that carry influence from a source op to a sink op.

- covers: `Trace.between`
- tasks: subgraph
- fixture: toy

```python
trace = tl.trace(model, x)
view = trace.between("linear_1_1", "linear_2_3")
print(type(view).__name__)
print([op.label for op in view])
# Output:
# TraceSlice
# ['linear_1_1:1', 'relu_1_2:1', 'linear_2_3:1']
```

### trace-subgraph
The sub-DAG view of one region of the trace.

- covers: `Trace.subgraph`
- tasks: subgraph
- fixture: toy

```python
trace = tl.trace(model, x)
view = trace.subgraph(tl.func("relu"))
print(type(view).__name__)
# Output:
# TraceSlice
```

### trace-annotate
Attach your own data to selected ops; each op keeps it under `annotations["user"]`.

- covers: `Trace.annotate`
- tasks: annotate
- fixture: toy

```python
trace = tl.trace(model, x)
trace.annotate(tl.func("relu"), data={"note": "hidden"})
print(trace.ops["relu_1_2"].annotations["user"]["data"])
# Output:
# {'note': 'hidden'}
```

## Checks and non-finite values

### trace-nonfinite
Find where a NaN first appears: the report, the ops that held one, and the verdict.

- covers: `Trace.find_nan`, `Trace.first_nonfinite`, `Trace.nonfinite_ops`, `Trace.nonfinite_verdict`, `Trace.nonfinite_coverage`
- tasks: debug, nan
- fixture: toy

```python
model.fc2.bias.data[0] = float("nan")
trace = tl.trace(model, x)
report = trace.find_nan()
print("find_nan", report.found, report.label)
print("first_nonfinite", trace.first_nonfinite().split(" (")[0])
print("nonfinite_ops", trace.nonfinite_ops)
print("nonfinite_verdict", trace.nonfinite_verdict)
print("nonfinite_coverage", type(trace.nonfinite_coverage).__name__)
# Output:
# find_nan True linear_2_3
# first_nonfinite First non-finite saved out is in layer linear_2_3
# nonfinite_ops ('linear_2_3:1', 'output_1:1')
# nonfinite_verdict found
# nonfinite_coverage NonfiniteCoverage
```

### trace-validate-forward-pass
Replay the saved values against the model's real output, and check metadata invariants.

- covers: `Trace.validate_forward_pass`, `Trace.check_metadata_invariants`, `Trace.validation_replay_status`
- tasks: validate
- fixture: toy

```python
options = tl.options.CaptureOptions(replayable=True)
trace = tl.trace(model, x, capture=options, save="all")
status = trace.validate_forward_pass(model(x))
print("forward", status)
print(status.message)
print("metadata", trace.check_metadata_invariants())
print("validation_replay_status", trace.validation_replay_status)
# Output:
# forward ValidationStatus: unverified, torch, live, 1 replayed, 0 failed
#   available available
#   reason    region_not_per_op_replayable
#   message   Replay validation partially verified this trace: 1 nodes replayed, 0 pure nodes unverified, 0 effect regions unverified, 3 legacy regions unverified, 0 failures.
#   replay    1 replayed, 3 unverified, 0 failed
#   more      .decisions  .unverified_reason_counts  .exempted_reason_counts
# Replay validation partially verified this trace: 1 nodes replayed, 0 pure nodes unverified, 0 effect regions unverified, 3 legacy regions unverified, 0 failures.
# metadata True
# validation_replay_status ValidationStatus: unverified, torch, live, 1 replayed, 0 failed
#   available available
#   reason    region_not_per_op_replayable
#   message   Replay validation partially verified this trace: 1 nodes replayed, 0 pure nodes unverified, 0 effect regions unverified, 3 legacy regions unverified, 0 failures.
#   replay    1 replayed, 3 unverified, 0 failed
#   more      .decisions  .unverified_reason_counts  .exempted_reason_counts
```

## Rebuilding outputs

### trace-reconstruct-output
Rebuild the model's Python output object from the saved values.

- covers: `Trace.reconstruct_output`
- tasks: output
- fixture: toy

```python
trace = tl.trace(model, x)
out = trace.reconstruct_output()
print(torch.equal(out, model(x)))
# Output:
# True
```

### trace-reconstruct-container
Rebuild a container a submodule returned (here a dict) from the captured values.

- covers: `Trace.reconstruct_container`
- tasks: output
- fixture: none

```python
class Split(torch.nn.Module):
    def forward(self, x):
        return {"a": x * 2, "b": -x}
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.split = Split()
    def forward(self, x):
        parts = self.split(x)
        return parts["a"] + parts["b"]
options = tl.options.CaptureOptions(capture_container_structure=True)
trace = tl.trace(Net(), torch.ones(2), capture=options)
rebuilt = trace.reconstruct_container()
print(type(rebuilt).__name__, sorted(rebuilt))
# Output:
# dict ['a', 'b']
```

## Gradients

### trace-backward-saved-grads
Run backward through a trace and read the views of saved gradients.

- covers: `Trace.backward`, `Trace.saved_grad_ops`, `Trace.saved_grad_modules`, `Trace.saved_grad_module_calls`, `Trace.grad_fn_site_summary`
- tasks: gradients, backward
- fixture: toy

```python
options = tl.options.CaptureOptions(save_grads=True)
trace = tl.trace(model, x, capture=options)
trace.backward(trace.output.sum())
print("saved_grad_ops", [op.label for op in trace.saved_grad_ops])
print("saved_grad_modules", len(trace.saved_grad_modules))
print("saved_grad_module_calls", len(trace.saved_grad_module_calls))
print("grad_fn_site_summary", len(trace.grad_fn_site_summary))
# Output:
# saved_grad_ops ['linear_1_1:1', 'relu_1_2:1', 'linear_2_3:1', 'output_1:1']
# saved_grad_modules 3
# saved_grad_module_calls 3
# grad_fn_site_summary 4
```

### trace-capture-backward
Capture a backward you call yourself inside a context manager.

- covers: `Trace.capture_backward`
- tasks: gradients, backward
- fixture: toy

```python
options = tl.options.CaptureOptions(save_grads=True)
trace = tl.trace(model, x, capture=options)
with trace.capture_backward():
    trace.output.sum().backward()
print(len(trace.saved_grad_ops) > 0)
# Output:
# True
```

### trace-armed-edits
A gradient edit with no loss yet comes back armed: listed, waiting for `backward`.

- covers: `Trace.armed_edits`
- tasks: gradients, intervene
- fixture: toy

```python
options = tl.options.CaptureOptions(backward_ready=True, save_grads=True)
trace = tl.trace(model, x, capture=options)
armed = trace.do(tl.grad("relu_1_2"), tl.scale(0.0), model=model, inputs=x)
print(len(armed.armed_edits), armed.armed_edits[0].signal)
print("source untouched", trace.armed_edits)
# Output:
# 1 backward
# source untouched ()
```

## Other capture modes

### trace-aten-ops
Record the ATen operations under each torch call.

- covers: `Trace.has_aten_ops`, `Trace.aten_ops`, `Trace.aten_gaps`
- tasks: aten
- fixture: toy

```python
options = tl.options.CaptureOptions(aten_ops=True)
trace = tl.trace(model, x, capture=options)
print("has_aten_ops", trace.has_aten_ops)
print("aten_ops", len(trace.aten_ops) > 0)
print("aten_gaps", type(trace.aten_gaps).__name__)
# Output:
# has_aten_ops True
# aten_ops True
# aten_gaps RecordTuple
```

### trace-logged-values
Values the forward logs with `tl.observers.log_value` land on the trace.

- covers: `Trace.logged_values`
- tasks: observe
- fixture: none

```python
class Net(torch.nn.Module):
    def forward(self, x):
        tl.observers.log_value("mean", round(float(x.mean()), 3))
        return x * 2
trace = tl.trace(Net(), torch.ones(2, 2))
print(trace.logged_values)
# Output:
# {'mean': 1.0}
```

### trace-injected-ops
With `log_injections=True`, ops a hook adds are recorded apart from the model's own ops.

- covers: `Trace.injected_ops`, `Trace.model_ops`
- tasks: intervene, hooks
- fixture: toy

```python
options = tl.options.CaptureOptions(log_injections=True)
hook = tl.when(tl.func("relu"), lambda out, *, ctx: out * 0.5)
trace = tl.trace(model, x, capture=options, intervene=hook)
print("injected_ops", [op.label for op in trace.injected_ops])
print("model_ops", [op.label for op in trace.model_ops])
# Output:
# injected_ops ['relu_1_2:1/inj_1']
# model_ops ['input_1:1', 'linear_1_1:1', 'relu_1_2:1', 'linear_2_3:1', 'output_1:1']
```

### trace-buffer-ops
A module with a running buffer: reads and writes of that buffer.

- covers: `Trace.buffer_read_ops`, `Trace.buffer_write_ops`
- tasks: buffers
- fixture: none

```python
model = torch.nn.BatchNorm1d(4).train()
trace = tl.trace(model, torch.randn(3, 4))
print("buffer_read_ops", list(trace.buffer_read_ops))
print("buffer_write_ops", list(trace.buffer_write_ops))
# Output:
# buffer_read_ops ['buffer_1:1', 'buffer_3:1', 'buffer_4:1']
# buffer_write_ops ['buffer_2:1', 'buffer_5:1', 'buffer_6:1']
```

### trace-transforms
A `torch.func` transform is recorded as one boundary op.

- covers: `Trace.transforms`
- tasks: transforms
- fixture: none

```python
class Net(torch.nn.Module):
    def forward(self, x):
        return torch.func.vmap(torch.sin)(x)
trace = tl.trace(Net(), torch.randn(3, 2))
print([op.label for op in trace.transforms])
# Output:
# ['vmap_1_1:1']
```

### trace-structure-only
A structure-only capture checks a plan without values, then discharges against a real run.

- covers: `Trace.check_plan`, `Trace.discharge_against`
- tasks: plan, structure
- fixture: toy

```python
options = tl.options.CaptureOptions(structure_only=True)
plan_trace = tl.trace(model, x, capture=options)
print(type(plan_trace.check_plan(["relu_1_2:1"])).__name__)
report = plan_trace.discharge_against(tl.trace(model, x))
print(report.verdict.value)
# Output:
# PlanCheckReport
# corroborated
```

### trace-episode
A plain forward has no episode ledger and no coupling verdict.

- covers: `Trace.episode`, `Trace.episode_coupling`
- tasks: episode
- fixture: toy

```python
trace = tl.trace(model, x)
print("episode", trace.episode)
print("episode_coupling", trace.episode_coupling)
# Output:
# episode None
# episode_coupling None
```

## Geometry and facets

### trace-receptive-fields
Receptive and projective field tables for every layer of a conv net.

- covers: `Trace.receptive_fields`, `Trace.projective_fields`
- tasks: receptive-field
- fixture: cnn

```python
trace = tl.trace(model, x)
fields = trace.receptive_fields(level="layer")
print(type(fields).__name__, fields.level, len(fields.to_pandas()))
reach = trace.projective_fields(level="layer")
print(type(reach).__name__, len(reach.to_pandas()))
# Output:
# ReceptiveFieldProfile layer 5
# ReceptiveFieldProfile 5
```

### trace-facets-attention
Modules with a facet: attention blocks and the modules exposing a residual stream.

- covers: `Trace.attention_blocks`, `Trace.modules_with_facet`
- tasks: lm, facets
- fixture: lm
- requires: transformers

```python
trace = tl.trace(lm, ids)
print([m.address for m in trace.attention_blocks()])
print([m.address for m in trace.modules_with_facet("resid_post")])
# Output:
# ['transformer.h.0.attn', 'transformer.h.1.attn']
# ['transformer.h.0', 'transformer.h.1']
```

### trace-using-appliance
Get a registered appliance bound to a trace.

- covers: `Trace.using`
- tasks: lm, appliances
- fixture: lm
- requires: transformers

```python
trace = tl.trace(lm, ids)
appliance = trace.using("mechinterp")
print(type(appliance).__name__)
# Output:
# MechInterpAppliance
```

## Drawing and housekeeping

### trace-draw
Draw the graph to a file; the call returns a Drawing.

- covers: `Trace.draw`
- tasks: draw
- fixture: toy

```python
trace = tl.trace(model, x)
drawing = trace.draw("graph.svg")
import os
print(type(drawing).__name__, os.path.exists("graph.svg"))
# Output:
# Drawing True
```

### trace-animate
Animate the passes of a looping layer into an HTML file.

- covers: `Trace.animate`
- tasks: draw, loops
- fixture: loop

```python
trace = tl.trace(model, x)
drawing = trace.animate("linear_1_1", path="passes")
print(type(drawing).__name__)
# Output:
# Drawing
```

### trace-show
The notebook-friendly render dispatcher; run as a script it also writes `<Model>.pdf` and prints its path.

- covers: `Trace.show`
- tasks: draw
- fixture: toy

```python
trace = tl.trace(model, x)
shown = trace.show()
print(type(shown).__name__)
# Output:
# ...Drawing
```

### trace-preview-fastlog
Preview which ops a fastlog predicate would keep, as a graph.

- covers: `Trace.preview_fastlog`
- tasks: fastlog
- fixture: toy

```python
trace = tl.trace(model, x)
preview = trace.preview_fastlog(predicate=lambda ctx: ctx.func_name == "relu")
print(type(preview).__name__)
# Output:
# ...Drawing
```

### trace-load-state-dict
Stage new weights onto a loaded runnable trace, then run it without the model object.

- covers: `Trace.load_state_dict`
- tasks: load, rerun
- fixture: toy

```python
options = tl.options.CaptureOptions(replayable=True)
trace = tl.trace(model, x, capture=options, save="all")
trace.save("run.tlspec", level="runnable", include_weights=True)
loaded = tl.load("run.tlspec")
loaded.load_state_dict(model.state_dict())
result = loaded.run(inputs=x)
print(torch.allclose(result.output, model(x)))
# Output:
# True
```

### trace-cleanup
Free a trace's data when you are done with it.

- covers: `Trace.cleanup`
- tasks: memory
- fixture: toy

```python
trace = tl.trace(model, x)
trace.cleanup()
print("cleaned")
# Output:
# cleaned
```
