# Top level: the root tl.* functions and classes

## The result classes

### toplevel-classes-trace-layer-op
A capture returns a `tl.Trace`; its layers are `tl.Layer` objects and each pass is a `tl.Op`.

- covers: `tl.Trace`, `tl.Layer`, `tl.Op`
- tasks: capture
- fixture: cnn

```python
trace = tl.trace(model, x)
print("Trace", isinstance(trace, tl.Trace))
print("Layer", isinstance(trace.layers["relu_1_2"], tl.Layer))
print("Op", isinstance(trace.ops["relu_1_2"], tl.Op))
# Output:
# Trace True
# Layer True
# Op True
```

### toplevel-classes-selector-selection
A selector is a query (`tl.Selector`); indexing a trace with one gives a `tl.Selection`.

- covers: `tl.Selector`, `tl.Selection`, `tl.units`
- tasks: select
- fixture: cnn

```python
query = tl.units("relu_1_2", [(0, 0, 1, 1)]) | tl.units("relu_1_2", [(0, 1, 2, 2)])
trace = tl.trace(model, x)
selection = trace[query]
print("Selector", isinstance(query, tl.Selector), query.kind)
print("Selection", isinstance(selection, tl.Selection), selection[0].num_selected)
# Output:
# Selector True ACT
# Selection True 2
```

### toplevel-classes-edit-recipe
An edit helper returns a `tl.Edit`; `tl.when` binds it to a site as a `tl.Recipe`, and `&` joins recipes.

- covers: `tl.Edit`, `tl.Recipe`, `tl.when`, `tl.scale`
- tasks: intervene
- fixture: cnn

```python
edit = tl.scale(2.0)
first = tl.when(tl.func("relu"), edit)
both = first & tl.when(tl.module("2"), tl.scale(0.5))
print("Edit", isinstance(edit, tl.Edit), edit.name)
print("Recipe", isinstance(both, tl.Recipe), len(both.rules))
# Output:
# Edit True scale
# Recipe True 2
```

### toplevel-classes-recording
`tl.record` returns a `tl.Recording` holding one record per saved site.

- covers: `tl.Recording`, `tl.record`
- tasks: record-many
- fixture: cnn

```python
recording = tl.record(model, x, save=tl.func("relu"))
print(isinstance(recording, tl.Recording), recording.status, len(recording.records))
print(tuple(recording.records[0].ram_payload.shape))
# Output:
# True complete 1
# (1, 4, 8, 8)
```

## Capturing and one-call helpers

### toplevel-trace-save-all
Capture with every activation saved and list the layers.

- covers: `tl.trace`
- tasks: capture
- fixture: cnn

```python
trace = tl.trace(model, x, save="all")
print(trace.layer_labels)
print(tuple(trace["relu_1_2"].out.shape))
# Output:
# ['input_1', 'conv2d_1_1', 'relu_1_2', 'conv2d_2_3', 'output_1']
# (1, 4, 8, 8)
```

### toplevel-pluck-one-layer
Get one layer's output tensor in one call, without keeping a trace.

- covers: `tl.pluck`
- tasks: extract-layers
- fixture: cnn

```python
relu_out = tl.pluck(model, x, "relu")
print(type(relu_out).__name__, tuple(relu_out.shape))
# Output:
# Tensor (1, 4, 8, 8)
```

### toplevel-summary-text
Print a one-call text summary of the model's forward pass.

- covers: `tl.summary`
- tasks: summary
- fixture: cnn

```python
text = tl.summary(model, x)
print(isinstance(text, str), "conv2d" in text.lower())
# Output:
# True True
```

### toplevel-draw-file
Draw the model graph to an SVG file in the working directory.

- covers: `tl.draw`
- tasks: draw
- fixture: cnn

```python
import os

drawing = tl.draw(model, x, path="net.svg")
print(type(drawing).__name__, drawing.file_format, os.path.exists("net.svg"))
# Output:
# TorchLens graph written to net.svg
# Drawing svg True
```

### toplevel-validate
Check that a capture replays and its metadata holds for this model and input.

- covers: `tl.validate`
- tasks: validate
- fixture: cnn

```python
print(tl.validate(model, x, scope="forward"))
print(tl.validate(model, x, scope="saved"))
# Output:
# True
# True
```

### toplevel-extract-dataset-memory
Extract a module's outputs over a dataset in batches, kept in memory.

- covers: `tl.extract_dataset`
- tasks: extract-dataset
- fixture: cnn

```python
stimuli = torch.randn(4, 3, 8, 8)
out = tl.extract_dataset(model, stimuli, ["1"], batch_size=2, progress=False)
print(tuple(out["1"].shape))
# Output:
# (4, 4, 8, 8)
```

### toplevel-to-disk-stream
Stream the saved activations of a capture to a `.tlspec` file while it runs.

- covers: `tl.to_disk`
- tasks: to-disk
- fixture: cnn

```python
import os

storage = tl.to_disk("run.tlspec")
trace = tl.trace(model, x, storage=storage)
print("retain_in_memory", storage.retain_in_memory)
print("written", os.path.exists("run.tlspec"))
# Output:
# retain_in_memory False
# written True
```

### toplevel-load-roundtrip
Save a trace and load it back.

- covers: `tl.load`
- tasks: save-load
- fixture: cnn

```python
trace = tl.trace(model, x)
trace.save("run.tlspec")
loaded = tl.load("run.tlspec")
print(loaded.layer_labels)
print(torch.equal(loaded["relu_1_2"].out, trace["relu_1_2"].out))
# Output:
# ['input_1', 'conv2d_1_1', 'relu_1_2', 'conv2d_2_3', 'output_1']
# True
```

### toplevel-sweep-values
Capture one edited trace per value at a site, plus an unedited baseline, as a bundle.

- covers: `tl.sweep`
- tasks: sweep
- fixture: cnn

```python
bundle = tl.sweep(model, x, site="relu", values=[0.0, 1.0], include_baseline=True)
print(bundle.names)
# Output:
# ['baseline', 'sweep_0', 'sweep_1']
```

### toplevel-clear-cache
Delete the capture-cache entries in a cache folder; returns how many it removed.

- covers: `tl.clear_cache`
- tasks: cache
- fixture: cnn

```python
opts = tl.options.CaptureOptions(cache=True, cache_dir="cache")
trace = tl.trace(model, x, capture=opts)
print(tl.clear_cache("cache"))
# Output:
# 1
```

### toplevel-release-model
Undo TorchLens's preparation of a model after capturing, so it pickles cleanly.

- covers: `tl.release_model`
- tasks: capture
- fixture: none

```python
import pickle

net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
trace = tl.trace(net, torch.randn(1, 2))
tl.release_model(net)
print(len(pickle.dumps(net)) > 0)
# Output:
# True
```

### toplevel-merge-ranks-refuses-local
`tl.merge_ranks` joins captures from distributed ranks; a plain single-process trace is refused with a typed reason.

- covers: `tl.merge_ranks`
- tasks: distributed
- fixture: cnn

```python
trace = tl.trace(model, x)
try:
    tl.merge_ranks([trace])
except tl.errors.MergeInputError as error:
    print(error.fields["reason"])
# Output:
# not_a_rank_capture
```

### toplevel-merge-report-refuses-local
`tl.merge_report` diagnoses a rank merge without building it; input checks still refuse a non-rank trace.

- covers: `tl.merge_report`
- tasks: distributed
- fixture: cnn

```python
trace = tl.trace(model, x)
try:
    tl.merge_report([trace])
except tl.errors.MergeInputError as error:
    print(error.fields["code"], error.fields["reason"])
# Output:
# merge_input_invalid not_a_rank_capture
```

## Selecting ops by name, module and position

### toplevel-select-label
Select one op by its exact label.

- covers: `tl.label`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.label("relu_1_2")].label)
# Output:
# relu_1_2:1
```

### toplevel-select-label-contains
Select every op whose label contains a substring.

- covers: `tl.label_contains`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.label_contains("conv")].labels)
# Output:
# ('conv2d_1_1:1', 'conv2d_2_3:1')
```

### toplevel-select-label-matches
Select ops whose label matches a regular expression.

- covers: `tl.label_matches`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.label_matches("^relu_[0-9]+_2$")].labels)
# Output:
# ('relu_1_2:1',)
```

### toplevel-select-module
`tl.module` addresses a module's output op.

- covers: `tl.module`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.module("1")].labels)
# Output:
# ('relu_1_2:1',)
```

### toplevel-select-in-module
`tl.in_module` addresses every op inside a module.

- covers: `tl.in_module`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.in_module("0")].labels)
# Output:
# ('conv2d_1_1:1',)
```

### toplevel-select-site
Select ops by structural position, here by function name.

- covers: `tl.site`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.site(func="relu")].labels)
# Output:
# ('relu_1_2:1',)
```

### toplevel-select-where
Select ops with your own predicate on the op.

- covers: `tl.where`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.where(lambda op: op.func_name == "relu", name_hint="relus")].labels)
# Output:
# ('relu_1_2:1',)
```

### toplevel-select-all-sites
`tl.all_sites` matches every op; subtract a selector to drop some.

- covers: `tl.all_sites`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.all_sites() - tl.func("relu")].labels)
# Output:
# ('input_1:1', 'conv2d_1_1:1', 'conv2d_2_3:1', 'output_1:1')
```

### toplevel-select-between
Select the ops that carry influence from one op to another.

- covers: `tl.between`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.between("conv2d_1_1", "conv2d_2_3")].labels)
# Output:
# ('conv2d_1_1:1', 'relu_1_2:1', 'conv2d_2_3:1')
```

### toplevel-select-neighborhood
Select every op within a number of dataflow hops of a seed op.

- covers: `tl.neighborhood`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.neighborhood("relu_1_2", hops=1)].labels)
# Output:
# ('conv2d_1_1:1', 'relu_1_2:1', 'conv2d_2_3:1')
```

### toplevel-select-followed-by
Save only the convs that a relu follows, decided after the relu runs (needs lookback).

- covers: `tl.followed_by`
- tasks: select-save
- fixture: cnn

```python
keep = tl.func("conv2d") & tl.followed_by(tl.func("relu"))
opts = tl.options.CaptureOptions(lookback=4, lookback_payload_policy="detached_raw")
trace = tl.trace(model, x, save=keep, capture=opts)
print([op.label for op in trace.saved_ops])
# Output:
# ['conv2d_1_1:1']
```

### toplevel-select-preceded-by
Save only the convs that come after a relu; the model's output node is saved along with the last conv.

- covers: `tl.preceded_by`
- tasks: select-save
- fixture: cnn

```python
keep = tl.func("conv2d") & tl.preceded_by(tl.func("relu"))
trace = tl.trace(model, x, save=keep, capture=tl.options.CaptureOptions(lookback=4))
print([op.label for op in trace.saved_ops])
# Output:
# ['conv2d_2_3:1', 'output_1:1']
```

### toplevel-select-model-input
Address one entry of a nested (dict) model input.

- covers: `tl.model_input`
- tasks: select
- fixture: none

```python
class DictIn(torch.nn.Module):
    def forward(self, d):
        return d["a"] + d["b"]


inputs = [{"a": torch.zeros(1, 2), "b": torch.ones(1, 2)}]
capture = tl.options.CaptureOptions(capture_container_structure=True)
trace = tl.trace(DictIn(), inputs, capture=capture)
print(trace[tl.model_input("b")].labels)
# Output:
# ('input_2:1',)
```

### toplevel-select-model-output
Address the op that produced one element of the model's output.

- covers: `tl.model_output`
- tasks: select
- fixture: none

```python
class Split(torch.nn.Module):
    def forward(self, v):
        first, second = torch.chunk(v, 2, dim=1)
        return first * 2 + second


trace = tl.trace(Split(), torch.randn(1, 4))
print(trace[tl.model_output(1)].labels)
# Output:
# ('chunk_2_2:1',)
```

### toplevel-select-func-transform
A `torch.func` transform is captured as one boundary op; select it by transform kind.

- covers: `tl.func_transform`
- tasks: select
- fixture: none

```python
class Mapped(torch.nn.Module):
    def forward(self, v):
        return torch.func.vmap(torch.sin)(v)


trace = tl.trace(Mapped(), torch.randn(3, 4))
print(trace[tl.func_transform("vmap")].labels)
# Output:
# ('vmap_1_1:1',)
```

### toplevel-select-facet-head
Facet and head selectors address an attention projection and one head of it.

- covers: `tl.facet`, `tl.head`
- tasks: attention
- fixture: none

```python
query = tl.facet("q").head(3)
head = tl.head(2, "q")
print(query.selector_kind, head.head_index, head.name)
# Output:
# facet 2 q
```

### toplevel-select-at-step
`tl.at_step` addresses one step of a stepped episode; a plain trace has none and refuses with a code.

- covers: `tl.at_step`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
try:
    trace[tl.at_step(0)]
except tl.errors.SiteResolutionError as error:
    print(error.fields["code"])
# Output:
# episode_step_selector_without_episode
```

## Selecting backward-pass sites

### toplevel-select-grad-fn
After a backward, select the autograd nodes by class name.

- covers: `tl.grad_fn`
- tasks: gradients
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"))
trace.backward(trace[trace.output_layers[0]].out.sum())
sites = trace.find_sites(tl.grad_fn("ReluBackward0"), max_fanout=100)
print(len(sites), type(sites[0]).__name__)
# Output:
# 1 GradFn
```

### toplevel-select-grad-input-output
Select backward events by kind: gradients flowing into, or out of, autograd nodes.

- covers: `tl.grad_input`, `tl.grad_output`
- tasks: gradients
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"))
trace.backward(trace[trace.output_layers[0]].out.sum())
print("grad_input", len(trace.find_sites(tl.grad_input(), max_fanout=100)) > 0)
print("grad_output", len(trace.find_sites(tl.grad_output(), max_fanout=100)) > 0)
# Output:
# grad_input True
# grad_output True
```

### toplevel-select-in-backward-pass
Select the backward sites of one backward pass by its number.

- covers: `tl.in_backward_pass`
- tasks: gradients
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"))
loss = trace[trace.output_layers[0]].out.sum()
trace.backward(loss)
print(len(trace.find_sites(tl.in_backward_pass(1), max_fanout=100)) > 0)
# Output:
# True
```

### toplevel-select-without-op
`tl.without_op` matches autograd nodes that have no forward op of their own.

- covers: `tl.without_op`
- tasks: gradients
- fixture: toy

```python
trace = tl.trace(model, x, capture=tl.options.CaptureOptions(save_grads="all"))
trace.backward(trace[trace.output_layers[0]].out.sum())
selector = tl.without_op()
print(selector.selector_kind)
print(len(trace.find_sites(selector, max_fanout=100)))
# Output:
# intervening
# 8
```

### toplevel-grad-site-edit
`tl.grad(site)` addresses a site's gradient: here an edit zeroes the relu's gradient in the backward pass.

- covers: `tl.grad`
- tasks: gradients
- fixture: toy

```python
source = tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True, save_grads="all"))
def loss(result):
    return result[result.output_layers[0]].out.sum()
result = source.do(tl.grad(tl.func("relu")), tl.scale(0), model=model, inputs=x, loss=loss)
print(repr(tl.grad("relu_1_2")))
print(int(torch.count_nonzero(result["relu_1_2"].grad)))
# Output:
# tl.grad('relu_1_2')
# 0
```

## Selecting elements by value

### toplevel-units-explicit
Select explicit elements (units) of one op's output.

- covers: `tl.units`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
selection = trace[tl.units("relu_1_2", [(0, 0, 1, 1), (0, 1, 2, 2)])]
print(selection.labels, selection[0].num_selected)
# Output:
# ('relu_1_2:1',) 2
```

### toplevel-param-region
Select the elements of a named parameter.

- covers: `tl.param`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
selection = trace[tl.param("0.weight")]
print(selection.kind, selection[0].num_selected)
# Output:
# PARAM 108
```

### toplevel-random-like-control
A seeded random selection the same size as a probe, as a control.

- covers: `tl.random_like`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
probe = tl.units("relu_1_2", [(0, 0, 0, 0), (0, 1, 1, 1)])
control = tl.random_like(like=probe, within=tl.func("conv2d"), seed=0)
print(sum(site.num_selected for site in trace[control]))
# Output:
# 2
```

### toplevel-top-k
Select the k highest-valued elements of an op's output.

- covers: `tl.top_k`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.top_k("conv2d_1_1", 3)][0].num_selected)
# Output:
# 3
```

### toplevel-top-fraction
Select the top fraction of an op's elements by value.

- covers: `tl.top_fraction`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
print(trace[tl.top_fraction("conv2d_1_1", 0.5)][0].num_selected)
# Output:
# 128
```

### toplevel-threshold
Select elements strictly above a value.

- covers: `tl.threshold`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
active = trace[tl.threshold("relu_1_2", above=0.0)]
print(active[0].num_selected == int((trace["relu_1_2"].out > 0).sum()))
# Output:
# True
```

### toplevel-sign
Select elements by sign class, here the zeros a relu produced.

- covers: `tl.sign`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
zeros = trace[tl.sign("relu_1_2", "zero")]
print(zeros[0].num_selected == int((trace["relu_1_2"].out == 0).sum()))
# Output:
# True
```

### toplevel-subspace
Select the elements a direction in channel space lives on.

- covers: `tl.subspace`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
direction = torch.tensor([1.0, 0.0, 0.0, 0.0])
region = tl.subspace("relu_1_2", direction, origin="hand-picked", dim=1)
print(trace[region][0].num_selected)
# Output:
# 64
```

### toplevel-changed
Select elements whose value differs from a reference run.

- covers: `tl.changed`
- tasks: compare-runs
- fixture: cnn

```python
trace = tl.trace(model, x)
other = tl.trace(model, x + 1.0)
moved = trace[tl.changed(other, "conv2d_1_1")]
print(moved[0].num_selected > 0)
# Output:
# True
```

### toplevel-top-changed
Select the k elements that changed most against a reference run.

- covers: `tl.top_changed`
- tasks: compare-runs
- fixture: cnn

```python
trace = tl.trace(model, x)
other = tl.trace(model, x + 1.0)
biggest = trace[tl.top_changed(other, "conv2d_2_3", k=5)]
print(sum(site.num_selected for site in biggest))
# Output:
# 5
```

### toplevel-dead
Select relu units that never fired across several runs; the relation says how sure the answer is.

- covers: `tl.dead`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
other = tl.trace(model, -x)
silent = trace[tl.dead([trace, other], within="relu_1_2")]
print(silent[0].provenance.relation)
# Output:
# upper_bound
```

### toplevel-saturated
Select units pinned at a bound (here zero) across several runs.

- covers: `tl.saturated`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
other = tl.trace(model, -x)
pinned = trace[tl.saturated([trace, other], low=0.0, within="relu_1_2")]
print(pinned[0].provenance.relation)
# Output:
# upper_bound
```

### toplevel-low-variance
Select units whose variance across runs is below a threshold.

- covers: `tl.low_variance`
- tasks: select
- fixture: cnn

```python
trace = tl.trace(model, x)
other = tl.trace(model, x + 0.01)
steady = trace[tl.low_variance([trace, other], threshold=1.0, within="relu_1_2")]
print(steady[0].provenance.relation, steady[0].num_selected)
# Output:
# exact 256
```

### toplevel-pass-variance
Select units by their variance across a recurrent layer's passes.

- covers: `tl.pass_variance`
- tasks: loops
- fixture: loop

```python
trace = tl.trace(model, x)
selection = trace[tl.pass_variance(tl.func("linear"), below=1e9)]
print(len(selection), sum(site.num_selected for site in selection))
# Output:
# 3 24
```

### toplevel-stable-across-passes
Select units that stay the same across a recurrent layer's passes, within a tolerance.

- covers: `tl.stable_across_passes`
- tasks: loops
- fixture: loop

```python
trace = tl.trace(model, x)
selection = trace[tl.stable_across_passes(tl.func("linear"), atol=1e9)]
print(len(selection), sum(site.num_selected for site in selection))
# Output:
# 3 24
```

## Edits

### toplevel-edit-zero-ablate
Zero a site's output during capture.

- covers: `tl.zero_ablate`
- tasks: ablate
- fixture: cnn

```python
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.zero_ablate()))
print(bool((edited["relu_1_2"].out == 0).all()), edited["relu_1_2"].is_edited)
# Output:
# True True
```

### toplevel-edit-mean-ablate
Fill a site with the mean of its own values (`dim=None`: one mean over all elements).

- covers: `tl.mean_ablate`
- tasks: ablate
- fixture: cnn

```python
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.mean_ablate(dim=None)))
print(edited["relu_1_2"].out.unique().numel())
# Output:
# 1
```

### toplevel-edit-add
Add a constant to a site's output.

- covers: `tl.add`
- tasks: intervene
- fixture: cnn

```python
clean = tl.trace(model, x)
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.add(1.0)))
print(torch.allclose(edited["relu_1_2"].out, clean["relu_1_2"].out + 1.0))
# Output:
# True
```

### toplevel-edit-add-noise
Add seeded Gaussian noise to a site's output.

- covers: `tl.add_noise`
- tasks: intervene
- fixture: cnn

```python
clean = tl.trace(model, x)
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.add_noise(0.1, seed=0)))
print(bool((edited["relu_1_2"].out != clean["relu_1_2"].out).any()))
# Output:
# True
```

### toplevel-edit-clamp
Clamp a site's values into a range.

- covers: `tl.clamp`
- tasks: intervene
- fixture: cnn

```python
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.clamp(low=0.0, high=0.1)))
print(bool(edited["relu_1_2"].out.max() <= 0.1))
# Output:
# True
```

### toplevel-edit-clip-norm
Rescale a site's output so its norm is at most a bound.

- covers: `tl.clip_norm`
- tasks: intervene
- fixture: cnn

```python
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.clip_norm(1.0)))
print(bool(torch.linalg.vector_norm(edited["relu_1_2"].out) <= 1.0 + 1e-5))
# Output:
# True
```

### toplevel-edit-scale
Multiply a site's output by a factor.

- covers: `tl.scale`
- tasks: intervene
- fixture: cnn

```python
clean = tl.trace(model, x)
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.scale(0.5)))
print(torch.allclose(edited["relu_1_2"].out, clean["relu_1_2"].out * 0.5))
# Output:
# True
```

### toplevel-edit-replace-with
Replace a site's output with a fixed tensor of the same shape.

- covers: `tl.replace_with`
- tasks: intervene
- fixture: cnn

```python
value = torch.ones(1, 4, 8, 8)
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), tl.replace_with(value)))
print(bool((edited["relu_1_2"].out == 1).all()))
# Output:
# True
```

### toplevel-edit-steer
Add a scaled direction along the channel dimension.

- covers: `tl.steer`
- tasks: steer-once
- fixture: cnn

```python
clean = tl.trace(model, x)
steer = tl.steer(torch.ones(4), strength=2.0, dim=1)
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), steer))
print(torch.allclose(edited["relu_1_2"].out, clean["relu_1_2"].out + 2.0))
# Output:
# True
```

### toplevel-edit-project-onto
Keep only the component of each channel vector along a direction.

- covers: `tl.project_onto`
- tasks: intervene
- fixture: cnn

```python
edit = tl.project_onto(torch.ones(4), dim=1)
out = tl.trace(model, x, intervene=tl.when(tl.func("relu"), edit))["relu_1_2"].out
print(torch.allclose(out[:, 0], out[:, 1]))
# Output:
# True
```

### toplevel-edit-project-out
Remove the component of each channel vector along a direction.

- covers: `tl.project_out`
- tasks: intervene
- fixture: cnn

```python
edit = tl.project_out(torch.ones(4), dim=1)
out = tl.trace(model, x, intervene=tl.when(tl.func("relu"), edit))["relu_1_2"].out
print(torch.allclose(out.sum(dim=1), torch.zeros(1, 8, 8), atol=1e-5))
# Output:
# True
```

### toplevel-edit-splice-module
Run a module of your own on a site's output and use its result instead.

- covers: `tl.splice_module`
- tasks: intervene
- fixture: cnn

```python
splice = torch.nn.Conv2d(4, 4, 1)
edit = tl.splice_module(splice, input_route="out")
edited = tl.trace(model, x, intervene=tl.when(tl.func("relu"), edit))
print(tuple(edited["relu_1_2"].out.shape), edited["relu_1_2"].is_edited)
# Output:
# (1, 4, 8, 8) True
```

### toplevel-edit-positions
Restrict an edit to chosen positions along one axis (token positions in a language model).

- covers: `tl.positions`
- tasks: intervene
- fixture: cnn

```python
edit = tl.positions(tl.zero_ablate(), [0, 1], axis=1)
out = tl.trace(model, x, intervene=tl.when(tl.func("relu"), edit))["relu_1_2"].out
print(bool((out[:, :2] == 0).all()), bool((out[:, 2:] != 0).any()))
# Output:
# True True
```

### toplevel-edit-patch-from
Patch a site's values from another trace of the same model.

- covers: `tl.patch_from`
- tasks: patch
- fixture: cnn

```python
donor = tl.trace(model, torch.randn(1, 3, 8, 8))
fork = tl.trace(model, x).fork()
fork.do("relu_1_2", tl.patch_from(donor))
print(torch.equal(fork["relu_1_2"].out, donor["relu_1_2"].out))
# Output:
# True
```
