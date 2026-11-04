# Options: CaptureOptions for tl.trace, VisualizationOptions for drawing

## Gradients and training

### options-save-grads
Save gradients at capture, run the backward, then read an op's gradient.

- covers: `CaptureOptions.save_grads`, `CaptureOptions.capture_tensor_grad_hooks`
- tasks: gradients, backward
- fixture: toy

```python
opts = tl.options.CaptureOptions(save_grads=True)
trace = tl.trace(model, x, capture=opts)
trace.backward(trace[trace.output_layers[0]].out.sum())
print("relu grad", tuple(trace.ops["relu_1_2"].grad.shape))
print("capture_tensor_grad_hooks", opts.capture_tensor_grad_hooks)
# Output:
# relu grad (3, 8)
# capture_tensor_grad_hooks True
```

### options-backward-ready
`backward_ready` keeps the saved graph connected for training; `inference_only` runs under no_grad.

- covers: `CaptureOptions.backward_ready`, `CaptureOptions.inference_only`
- tasks: gradients, training
- fixture: toy

```python
ready = tl.trace(model, x, capture=tl.options.CaptureOptions(backward_ready=True))
frozen = tl.trace(model, x, capture=tl.options.CaptureOptions(inference_only=True))
print("backward_ready", ready.options.backward_ready, ready.ops["linear_1_1"].out.requires_grad)
print("inference_only", frozen.options.inference_only, frozen.ops["linear_1_1"].out.requires_grad)
# Output:
# backward_ready True True
# inference_only True False
```

### options-detach-saved
`detach_saved_activations=True` stores activations cut off from autograd, which saves memory.

- covers: `CaptureOptions.detach_saved_activations`
- tasks: memory, activations
- fixture: toy

```python
kept = tl.trace(model, x)
cut = tl.trace(model, x, capture=tl.options.CaptureOptions(detach_saved_activations=True))
print("default", kept.ops["linear_1_1"].out.requires_grad)
print("detached", cut.ops["linear_1_1"].out.requires_grad)
# Output:
# default True
# detached False
```

### options-grad-transform
Store a summary of each gradient instead of (or beside) the raw gradient.

- covers: `CaptureOptions.grad_transform`, `CaptureOptions.save_raw_grads`
- tasks: gradients, transforms
- fixture: toy

```python
opts = tl.options.CaptureOptions(save_grads=True, grad_transform=lambda g: g.abs().sum(), save_raw_grads=False)
trace = tl.trace(model, x, capture=opts)
trace.backward(trace[trace.output_layers[0]].out.sum())
op = trace.ops["relu_1_2"]
print("transformed", round(op.transformed_grad.item(), 3))
print("raw kept", op.grad is not None)
# Output:
# transformed 4.051
# raw kept False
```

## What gets saved

### options-activation-transform
Store a reduced activation (here the mean) instead of the raw tensor.

- covers: `CaptureOptions.activation_transform`, `CaptureOptions.save_raw_activations`
- tasks: activations, transforms, memory
- fixture: toy

```python
opts = tl.options.CaptureOptions(activation_transform=lambda t: t.mean(), save_raw_activations=False)
trace = tl.trace(model, x, capture=opts)
op = trace.ops["relu_1_2"]
print("transformed", round(op.transformed_out.item(), 3))
print("raw kept", op.out is not None)
# Output:
# transformed 0.271
# raw kept False
```

### options-input-output-transform
Transform the input once before the forward and the output once after it.

- covers: `CaptureOptions.input_transform`, `CaptureOptions.output_transform`
- tasks: capture, transforms
- fixture: toy

```python
opts = tl.options.CaptureOptions(input_transform=lambda t: t * 0, output_transform=lambda out: out * 10)
trace = tl.trace(model, x, capture=opts)
print("input sum", trace.ops["input_1"].out.abs().sum().item())
print("fc1 equals its bias", torch.allclose(trace.ops["linear_1_1"].out[0], model.fc1.bias))
# Output:
# input sum 0.0
# fc1 equals its bias True
```

### options-save-args-rng
Keep each op's non-tensor arguments and random-number state for later replay.

- covers: `CaptureOptions.save_args`, `CaptureOptions.save_rng`
- tasks: capture, replay
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_args=True, save_rng=True))
print("save_args", trace.options.save_args)
print("save_rng", trace.options.save_rng)
# Output:
# save_args True
# save_rng True
```

### options-source-context
Record the source lines around each call; `source_context_lines` sets how many.

- covers: `CaptureOptions.save_source_context`, `CaptureOptions.source_context_lines`
- tasks: capture, source
- fixture: toy

```python
opts = tl.options.CaptureOptions(save_source_context=True, source_context_lines=3)
trace = tl.trace(model, x, capture=opts)
print("save_source_context", trace.options.save_source_context)
print("source_context_lines", trace.options.source_context_lines)
print("forward_signature", trace.modules["fc1"].forward_signature is not None)
# Output:
# save_source_context True
# source_context_lines 3
# forward_signature True
```

### options-output-device-save-mode
Choose where saved tensors live and whether they are copied.

- covers: `CaptureOptions.output_device`, `CaptureOptions.save_mode`
- tasks: activations, memory
- fixture: toy

```python
opts = tl.options.CaptureOptions(output_device="cpu", save_mode="copy")
trace = tl.trace(model, x, capture=opts)
print("output_device", trace.options.output_device, trace.ops["relu_1_2"].out.device)
print("save_mode", trace.options.save_mode)
# Output:
# output_device cpu cpu
# save_mode copy
```

### options-structure-only
Record only the graph, shapes and dtypes; no values are kept.

- covers: `CaptureOptions.structure_only`
- tasks: capture, structure
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(structure_only=True))
print("structure_only", trace.options.structure_only)
print("op_labels", trace.op_labels)
print("saved", trace.ops["relu_1_2"].has_saved_activation)
# Output:
# structure_only True
# op_labels ['input_1:1', 'linear_1_1:1', 'relu_1_2:1', 'linear_2_3:1', 'output_1:1']
# saved False
```

### options-keep-orphans
Keep computations that feed no output, reachable through `trace.graph.orphans`.

- covers: `CaptureOptions.keep_orphans`
- tasks: capture, graph
- fixture: none

```python
class Side(torch.nn.Module):
    def forward(self, x):
        unused = torch.ones(2) * 3
        return x + 1


plain = tl.trace(Side(), torch.zeros(2))
kept = tl.trace(Side(), torch.zeros(2), capture=tl.options.CaptureOptions(keep_orphans=True))
print("default", len(plain.graph.orphans))
print("keep_orphans", len(kept.graph.orphans) > 0)
# Output:
# default 0
# keep_orphans True
```

## Naming, seeding and loops

### options-name-seed
Name a capture and fix its seed so random ops repeat.

- covers: `CaptureOptions.name`, `CaptureOptions.seed`
- tasks: capture, reproducibility
- fixture: toy

```python
opts = tl.options.CaptureOptions(name="baseline", seed=0)
trace = tl.trace(model, x, capture=opts)
print("name", trace.name)
print("seed", trace.options.seed)
# Output:
# name baseline
# seed 0
```

### options-recurrence-detection
With recurrence detection off, a repeated module's passes are not grouped into one layer.

- covers: `CaptureOptions.recurrence_detection`
- tasks: loops, multi-pass
- fixture: loop

```python
on = tl.trace(model, x)
off = tl.trace(model, x, capture=tl.options.CaptureOptions(recurrence_detection=False))
print("on", [layer.num_passes for layer in on.layers])
print("off", [layer.num_passes for layer in off.layers])
# Output:
# on [1, 3, 3, 1]
# off [1, 3, 1, 1, 1, 1]
```

### options-chunk-size
Run the forward in batch chunks of a given size; the trace looks like one forward.

- covers: `CaptureOptions.chunk_size`
- tasks: capture, memory
- fixture: toy

```python
opts = tl.options.CaptureOptions(chunk_size=1)
trace = tl.trace(model, x, capture=opts)
print("chunk_size", opts.chunk_size)
print("output", tuple(trace[trace.output_layers[0]].out.shape))
# Output:
# chunk_size 1
# output (3, 2)
```

### options-carry-state
By default a capture restores buffers; `carry_state=True` keeps what the forward changed.

- covers: `CaptureOptions.carry_state`
- tasks: capture, state, buffers
- fixture: none

```python
torch.manual_seed(0)
net = torch.nn.BatchNorm1d(2).train()
x = torch.randn(4, 2)
tl.trace(net, x)
print("default", net.running_mean.abs().sum().item())
tl.trace(net, x, capture=tl.options.CaptureOptions(carry_state=True))
print("carry_state", net.running_mean.abs().sum().item() > 0)
# Output:
# default 0.0
# carry_state True
```

### options-lookback
Keep a window of recent ops, with their payloads, so `tl.followed_by` can select an op by what comes after it.

- covers: `CaptureOptions.lookback`, `CaptureOptions.lookback_payload_policy`
- tasks: select-save
- fixture: toy

```python
opts = tl.options.CaptureOptions(lookback=2, lookback_payload_policy="detached_raw")
trace = tl.trace(model, x, capture=opts, save=tl.func("linear") & tl.followed_by(tl.func("relu")))
print([label for label in trace.op_labels if trace.ops[label].has_saved_activation])
print("lookback_payload_policy", opts.lookback_payload_policy)
# Output:
# ['linear_1_1:1']
# lookback_payload_policy detached_raw
```

## Checks and extra records

### options-on-nonfinite
Track which ops produced NaN or Inf.

- covers: `CaptureOptions.on_nonfinite`
- tasks: debugging, nonfinite
- fixture: none

```python
class Blow(torch.nn.Module):
    def forward(self, x):
        return torch.log(x) * 2


trace = tl.trace(Blow(), torch.tensor([1.0, 0.0]), capture=tl.options.CaptureOptions(on_nonfinite="track"))
print(trace.nonfinite_ops)
# Output:
# ('log_1_1:1', 'mul_1_2:1')
```

### options-aten-ops
Record the ATen operations each torch call dispatched to.

- covers: `CaptureOptions.aten_ops`
- tasks: capture, aten
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(aten_ops=True))
print("has_aten_ops", trace.has_aten_ops)
print([a.operator for a in trace.ops["relu_1_2"].aten_ops])
# Output:
# has_aten_ops True
# ['relu', 'detach']
```

### options-cache
Cache a capture on disk keyed by model, inputs and options; the second call reuses it.

- covers: `CaptureOptions.cache`, `CaptureOptions.cache_dir`
- tasks: capture, cache
- fixture: toy

```python
import os

opts = tl.options.CaptureOptions(cache=True, cache_dir="tl-cache")
first = tl.trace(model, x, capture=opts)
second = tl.trace(model, x, capture=opts)
print("cache dir made", os.path.isdir("tl-cache"))
print("same ops", first.op_labels == second.op_labels)
# Output:
# cache dir made True
# same ops True
```

### options-memory-probes
Opt in to host peak-memory measurement; the peak varies, so check it exists.

- covers: `CaptureOptions.measure_host_peak_memory`, `CaptureOptions.measure_python_peak_memory`, `CaptureOptions.track_device_memory`
- tasks: memory, profiling
- fixture: toy

```python
opts = tl.options.CaptureOptions(measure_host_peak_memory=True, measure_python_peak_memory=True)
trace = tl.trace(model, x, capture=opts)
print("peak measured", trace.cost.forward_peak_memory is not None)
print("track_device_memory", opts.track_device_memory)
# Output:
# peak measured True
# track_device_memory False
```

### options-session-knobs
Read back profiling and distributed knobs that change what capture pays for, not what a trace means.

- covers: `CaptureOptions.emit_nvtx`, `CaptureOptions.distributed_witness`, `CaptureOptions.save_budget`, `CaptureOptions.verbose`, `CaptureOptions.unwrap_when_done`
- tasks: profiling, options
- fixture: toy

```python
opts = tl.options.CaptureOptions(save_budget=None, unwrap_when_done=True)
trace = tl.trace(model, x, capture=opts)
print("emit_nvtx", opts.emit_nvtx)
print("distributed_witness", opts.distributed_witness)
print("save_budget", opts.save_budget)
print("verbose", opts.verbose)
print("unwrap_when_done", opts.unwrap_when_done)
# Output:
# emit_nvtx False
# distributed_witness none
# save_budget None
# verbose False
# unwrap_when_done True
```

### options-replay-knobs
Read back the options that make a capture replayable or reconstructable.

- covers: `CaptureOptions.replayable`, `CaptureOptions.reconstruction_ready`, `CaptureOptions.capture_container_structure`, `CaptureOptions.compute_input_output_distances`, `CaptureOptions.log_injections`, `CaptureOptions.hooks`, `CaptureOptions.grouping`
- tasks: replay, options
- fixture: toy

```python
opts = tl.options.CaptureOptions(replayable=True, capture_container_structure=True)
trace = tl.trace(model, x, capture=opts)
print("replayable", opts.replayable)
print("reconstruction_ready", opts.reconstruction_ready)
print("capture_container_structure", opts.capture_container_structure)
print("compute_input_output_distances", opts.compute_input_output_distances)
print("log_injections", opts.log_injections)
print("hooks", opts.hooks)
print("grouping", opts.grouping)
# Output:
# replayable True
# reconstruction_ready False
# capture_container_structure True
# compute_input_output_distances True
# log_injections False
# hooks None
# grouping structural
```

### options-bundle-policies
Read back what a saved bundle keeps of the raw input, output and per-layer pictures.

- covers: `CaptureOptions.save_raw_input`, `CaptureOptions.save_transformed_output`, `CaptureOptions.batch_render`, `CaptureOptions.layer_visualizers`, `CaptureOptions.save_visualizations`, `CaptureOptions.payload_policy`, `CaptureOptions.module_identity_mode`
- tasks: bundles, options
- fixture: toy

```python
opts = tl.options.CaptureOptions(save_raw_input=True, batch_render="first")
trace = tl.trace(model, x, capture=opts)
print("save_raw_input", opts.save_raw_input)
print("save_transformed_output", opts.save_transformed_output)
print("batch_render", opts.batch_render)
print("layer_visualizers", opts.layer_visualizers)
print("save_visualizations", opts.save_visualizations)
print("payload_policy", opts.payload_policy)
print("module_identity_mode", opts.module_identity_mode)
# Output:
# save_raw_input True
# save_transformed_output small
# batch_render first
# layer_visualizers None
# save_visualizations False
# payload_policy None
# module_identity_mode None
```

### options-backend-only
Some fields belong to other backends; on torch they keep their defaults.

- covers: `CaptureOptions.jax_control_flow`, `CaptureOptions.jax_max_control_flow_unroll`, `CaptureOptions.jax_static_argnums`, `CaptureOptions.grad_options`
- tasks: backends, options
- fixture: none

```python
opts = tl.options.CaptureOptions()
print("jax_control_flow", opts.jax_control_flow)
print("jax_max_control_flow_unroll", opts.jax_max_control_flow_unroll)
print("jax_static_argnums", opts.jax_static_argnums)
print("grad_options", opts.grad_options)
# Output:
# jax_control_flow unroll
# jax_max_control_flow_unroll 64
# jax_static_argnums None
# grad_options None
```

### options-backend-only-refused
Passing a backend-only field to a torch capture is refused.

- covers: `CaptureOptions.jax_static_argnums`
- tasks: backends, options
- fixture: toy
- raises: TorchLensError

```python
tl.trace(model, x, capture=tl.options.CaptureOptions(jax_static_argnums=0))
```

### options-as-dict
List every capture option and its value as a dict.

- covers: `CaptureOptions.as_dict`
- tasks: options
- fixture: none

```python
values = tl.options.CaptureOptions(seed=1).as_dict()
print(len(values), values["seed"], values["save_grads"])
# Output:
# 57 1 None
```

## Drawing: VisualizationOptions fields are Trace.draw keywords

### options-vis-rolled
Draw a loop rolled (one node per repeated layer) instead of every pass.

- covers: `VisualizationOptions.rolled`
- tasks: draw, loops
- fixture: loop

```python
trace = tl.trace(model, x)
unrolled = trace.draw(path="unrolled", file_format="gv")
rolled = trace.draw(path="rolled", file_format="gv", rolled=True)
print("unrolled nodes", unrolled.dot.count("tanh_1_2"))
print("rolled nodes", rolled.dot.count("tanh_1_2"))
# Output:
# unrolled nodes 12
# rolled nodes 5
```

### options-vis-file-output
Choose the output file name, format and whether a viewer opens.

- covers: `VisualizationOptions.container_path`, `VisualizationOptions.file_format`, `VisualizationOptions.open_viewer`
- tasks: draw
- fixture: toy

```python
import os

opts = tl.options.VisualizationOptions(container_path="graph", file_format="gv", open_viewer=False)
trace = tl.trace(model, x)
drawing = trace.draw(path=opts.container_path, file_format=opts.file_format, open_viewer=opts.open_viewer)
print(drawing.file_format, sorted(os.listdir(".")))
# Output:
# gv ['bank_options_vis_file_output.py', 'graph.gv']
```

### options-vis-direction-layout
Set the layout direction and engine.

- covers: `VisualizationOptions.direction`, `VisualizationOptions.layout`, `VisualizationOptions.renderer`, `VisualizationOptions.order_siblings`
- tasks: draw
- fixture: toy

```python
import re

trace = tl.trace(model, x)
d = trace.draw(path="g", file_format="gv", direction="leftright", layout="dot", renderer="graphviz", order_siblings=False)
print(re.findall(r"rankdir=\"?(\w+)", d.dot))
# Output:
# ['LR']
```

### options-vis-depth-collapse
Hide detail: limit module depth, or collapse modules into single boxes.

- covers: `VisualizationOptions.depth`, `VisualizationOptions.collapse`, `VisualizationOptions.collapse_fn`, `VisualizationOptions.fold_repeats`
- tasks: draw, modules
- fixture: none

```python
block = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
net = torch.nn.Sequential(block, block, torch.nn.Linear(4, 2))
trace = tl.trace(net, torch.randn(2, 4))
full = trace.draw(path="full", file_format="gv")
shallow = trace.draw(path="shallow", file_format="gv", depth=1)
boxed = trace.draw(path="boxed", file_format="gv", collapse_fn=lambda m: m.address == "0")
auto = trace.draw(path="auto", file_format="gv", collapse="auto", fold_repeats=False)
print("depth=1 changes it", shallow.dot != full.dot, len(shallow.dot) < len(full.dot))
print("collapse_fn changes it", boxed.dot != full.dot, len(boxed.dot) < len(full.dot))
print("collapse auto changes it", auto.dot != full.dot)
# Output:
# depth=1 changes it True True
# collapse_fn changes it True True
# collapse auto changes it True
```

### options-vis-style
Restyle the picture: theme, node style, font size and Graphviz attributes for the graph, edges and module boxes.

- covers: `VisualizationOptions.theme`, `VisualizationOptions.node_style`, `VisualizationOptions.font_size`, `VisualizationOptions.graph_overrides`, `VisualizationOptions.edge_overrides`, `VisualizationOptions.module_overrides`
- tasks: draw, style
- fixture: none

```python
import re

block = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
net = torch.nn.Sequential(block, torch.nn.Linear(4, 2))
trace = tl.trace(net, torch.randn(2, 4))
d = trace.draw(path="g", file_format="gv", theme="torchlens", node_style="profiling", font_size=20,
               graph_overrides={"bgcolor": "ivory"}, edge_overrides={"color": "red"}, module_overrides={"pencolor": "blue"})
print("bgcolor", "ivory" in d.dot, "edge color", "red" in d.dot, "module pen", "blue" in d.dot)
print("fontsize 20", bool(re.search(r"fontsize=\"?20", d.dot)))
# Output:
# bgcolor True edge color True module pen True
# fontsize 20 True
```

### options-vis-dpi
`dpi` sets the resolution of raster output such as png; vector and source formats ignore it.

- covers: `VisualizationOptions.dpi`
- tasks: draw, style
- fixture: toy

```python
opts = tl.options.VisualizationOptions(dpi=72, file_format="gv")
trace = tl.trace(model, x)
d = trace.draw(path="g", file_format=opts.file_format, dpi=opts.dpi)
print("dpi", opts.dpi, "in gv source", "dpi=" in d.dot)
# Output:
# dpi 72 in gv source False
```

### options-vis-node-callbacks
Customize or hide nodes with callbacks, and choose the label rows.

- covers: `VisualizationOptions.node_spec_fn`, `VisualizationOptions.skip_fn`, `VisualizationOptions.collapsed_node_spec_fn`, `VisualizationOptions.node_label_fields`
- tasks: draw, style
- fixture: toy

```python
def paint(layer, spec):
    spec.fillcolor = "gold" if "relu" in layer.layer_label else spec.fillcolor
    return spec


trace = tl.trace(model, x)
d = trace.draw(path="g", file_format="gv", node_spec_fn=paint, skip_fn=lambda layer: False, collapsed_node_spec_fn=None, node_label_fields=["shape"])
print("gold relu", "gold" in d.dot)
# Output:
# gold relu True
```

### options-vis-encodings
Color and size nodes by a value, with a legend.

- covers: `VisualizationOptions.color_by`, `VisualizationOptions.size_by`, `VisualizationOptions.size_scale`, `VisualizationOptions.show_legend`, `VisualizationOptions.stack_by`, `VisualizationOptions.node_overlay`
- tasks: draw, style
- fixture: toy

```python
trace = tl.trace(model, x)
base = trace.draw(path="base", file_format="gv")
d = trace.draw(path="g", file_format="gv", color_by=lambda node: float(len(node.layer_label)), size_by="dims", size_scale="sqrt", show_legend=True)
o = trace.draw(path="o", file_format="gv", node_overlay={"relu_1_2": 1.0}, stack_by=False)
print("encoded differs", d.dot != base.dot, "overlay differs", o.dot != base.dot)
# Output:
# encoded differs True overlay differs True
```

### options-vis-buffers
Show buffer nodes always, only when meaningful, or never.

- covers: `VisualizationOptions.show_buffers`
- tasks: draw, buffers
- fixture: none

```python
net = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4)).eval()
trace = tl.trace(net, torch.randn(2, 4))
for mode in ["never", "meaningful", "always"]:
    d = trace.draw(path=mode, file_format="gv", show_buffers=mode)
    print(mode, d.dot.count("running_mean"))
# Output:
# never 1
# meaningful 1
# always 2
```

### options-vis-backward-graph
After a backward, the forward drawing gains gradient edges you can restyle, and `graph="backward"` draws the autograd graph itself.

- covers: `VisualizationOptions.graph`, `VisualizationOptions.grad_edge_overrides`
- tasks: draw, backward
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads=True))
trace.backward(trace[trace.output_layers[0]].out.sum())
fwd = trace.draw(path="fwd", file_format="gv", grad_edge_overrides={"color": "orange"})
bwd = trace.draw(path="bwd", file_format="gv", graph="backward")
print("orange grad edges", "orange" in fwd.dot)
print("backward nodes", "relu_back" in bwd.dot)
# Output:
# orange grad edges True
# backward nodes True
```

### options-vis-intervention-args
Read back how interventions and redundant constructor arguments are drawn.

- covers: `VisualizationOptions.intervention_mode`, `VisualizationOptions.show_cone`, `VisualizationOptions.show_redundant_args`
- tasks: draw, interventions
- fixture: toy

```python
opts = tl.options.VisualizationOptions(intervention_mode="as_node", show_cone=True, show_redundant_args=True)
trace = tl.trace(model, x)
d = trace.draw(path="g", file_format="gv", intervention_mode=opts.intervention_mode, show_cone=opts.show_cone, show_redundant_args=opts.show_redundant_args)
print(opts.intervention_mode, opts.show_cone, opts.show_redundant_args, d.file_format)
# Output:
# as_node True True gv
```

### options-vis-object
Build visualization options once, see which fields you set, and rebuild from a dict.

- covers: `VisualizationOptions.to_dict`, `VisualizationOptions.is_field_explicit`, `VisualizationOptions.from_values`
- tasks: draw, options
- fixture: none

```python
opts = tl.options.VisualizationOptions(rolled=True, depth=2)
values = opts.to_dict()
print("rolled explicit", opts.is_field_explicit("rolled"), "theme explicit", opts.is_field_explicit("theme"))
again = tl.options.VisualizationOptions.from_values(values, frozenset({"rolled", "depth"}))
print(again.rolled, again.depth, again.is_field_explicit("depth"))
# Output:
# rolled explicit True theme explicit False
# True 2 True
```
