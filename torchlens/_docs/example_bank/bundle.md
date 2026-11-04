# Bundle: several traces side by side

## Building and reading a bundle

### bundle-build-read
Build a bundle of named traces with a baseline and read it like a mapping.

- covers: `tl.Bundle`, `Bundle.__len__`, `Bundle.__contains__`, `Bundle.__getitem__`, `Bundle.__iter__`, `Bundle.names`, `Bundle.baseline_name`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
print("len", len(bundle))
print("names", bundle.names)
print("baseline_name", bundle.baseline_name)
print("contains", "shifted" in bundle)
print("getitem", type(bundle["clean"]).__name__)
print("iter", [type(member).__name__ for member in bundle])
# Output:
# len 2
# names ['clean', 'shifted']
# baseline_name clean
# contains True
# getitem Trace
# iter ['Trace', 'Trace']
```

### bundle-container-facts
Container-level facts: the member mapping, identity, history, capacity and outcome.

- covers: `Bundle.members`, `Bundle.id`, `Bundle.history`, `Bundle.capacity`, `Bundle.outcome`, `Bundle.is_structurally_consistent`, `Bundle.preserved_sections`, `Bundle.relations`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
print("members", sorted(bundle.members))
print("id", isinstance(bundle.id, str))
print("history", [row.kind for row in bundle.history])
print("capacity", bundle.capacity)
print("outcome", bundle.outcome.status.value)
print("is_structurally_consistent", bundle.is_structurally_consistent)
print("preserved_sections", dict(bundle.preserved_sections))
print("relations", len(bundle.relations))
# Output:
# members ['clean', 'shifted']
# id True
# history []
# capacity None
# outcome complete
# is_structurally_consistent True
# preserved_sections {}
# relations 0
```

### bundle-ops-layers
Read one op or layer across every member at once.

- covers: `Bundle.ops`, `Bundle.layers`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
op = bundle.ops["relu_1_2:1"]
layer = bundle.layers["relu_1_2"]
print(type(op).__name__, type(layer).__name__)
print(type(op.out).__name__, tuple(op.out.shape))
# Output:
# SuperOp SuperLayer
# Tensor (6, 8)
```

### bundle-at-site
`at` returns the cross-member view of one site, from a label or a selector.

- covers: `Bundle.at`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
view = bundle.at("relu_1_2")
print(type(view).__name__, tuple(view.out.shape))
# Output:
# SuperOp (6, 8)
```

### bundle-modules-params-buffers
Cross-member views of modules, module calls, parameters and buffers (the toy model has no buffers).

- covers: `Bundle.modules`, `Bundle.module_calls`, `Bundle.params`, `Bundle.buffers`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
print("modules", bundle.modules.keys())
print("module_calls", bundle.module_calls.keys())
print("params", bundle.params.keys())
print("buffers", bundle.buffers.keys())
# Output:
# modules ['fc1', 'fc2', 'self']
# module_calls ['fc1:1', 'fc2:1', 'self:1']
# params ['fc1.bias', 'fc1.weight', 'fc2.bias', 'fc2.weight']
# buffers []
```

### bundle-grad-fns
After a backward on each member, the bundle shows their autograd nodes and node calls.

- covers: `Bundle.grad_fns`, `Bundle.grad_fn_calls`
- tasks: gradients
- fixture: toy

```python
members = {}
for name, inputs in (("clean", x), ("shifted", x + 1)):
    trace = tl.trace(model, inputs, capture=tl.options.CaptureOptions(save_grads="all"))
    trace.backward(trace[trace.output_layers[0]].out.sum())
    members[name] = trace
bundle = tl.Bundle(members, baseline="clean")
print("grad_fns", bundle.grad_fns.keys())
print("grad_fn_calls", len(bundle.grad_fn_calls.keys()))
# Output:
# grad_fns ['accumulategrad_back_1_4', 'accumulategrad_back_2_8', 'accumulategrad_back_3_9', 'accumulategrad_back_4_11', 'addmm_back_1_3', 'addmm_back_2_7', 'clone_back_1_2', 'relu_back_1_5', 'sum_back_1_1', 't_back_1_6', 't_back_2_10']
# grad_fn_calls 11
```

### bundle-supergraph
The supergraph merges the members' graphs; read its nodes in order.

- covers: `Bundle.supergraph`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
graph = bundle.supergraph
print(type(graph).__name__)
print(list(graph.topological_order))
# Output:
# Supergraph
# ['input_1', 'linear_1_1', 'relu_1_2', 'linear_2_3', 'output_1']
```

## Adding and removing members

### bundle-add
Start empty and add traces one at a time or several at once.

- covers: `Bundle.add`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle()
bundle.add(tl.trace(model, x), names="a")
bundle.add([tl.trace(model, x + 1), tl.trace(model, x + 2)], names=["b", "c"])
print(bundle.names)
# Output:
# ['a', 'b', 'c']
```

### bundle-remove
Remove a member by name; `remove` returns what it took out.

- covers: `Bundle.remove`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"a": tl.trace(model, x), "b": tl.trace(model, x + 1), "c": tl.trace(model, x + 2)})
removed = bundle.remove("b")
print(bundle.names, type(removed).__name__)
# Output:
# ['a', 'c'] Trace
```

### bundle-remove-except
Keep only the listed members.

- covers: `Bundle.remove_except`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"a": tl.trace(model, x), "b": tl.trace(model, x + 1), "c": tl.trace(model, x + 2)})
bundle.remove_except(["a", "c"])
print(bundle.names)
# Output:
# ['a', 'c']
```

### bundle-clear
`clear` removes every member except the baseline.

- covers: `Bundle.clear`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"a": tl.trace(model, x), "b": tl.trace(model, x + 1)}, baseline="a")
bundle.clear()
print(bundle.names)
# Output:
# ['a']
```

### bundle-capacity-evicts
Setting a capacity evicts the oldest members but keeps the baseline.

- covers: `Bundle.capacity`
- tasks: compare-runs
- fixture: toy

```python
traces = {f"m{i}": tl.trace(model, x + i) for i in range(4)}
bundle = tl.Bundle(traces, baseline="m0")
bundle.capacity = 2
print(bundle.capacity, bundle.names)
# Output:
# 2 ['m0', 'm3']
```

## Comparing members

### bundle-compare
Per-site differences between two members (cosine by default).

- covers: `Bundle.compare`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
diffs = bundle.compare("clean", "shifted")
print([(label, round(value, 3)) for label, value in diffs])
# Output:
# [('input_1', 0.259), ('linear_1_1', 0.115), ('relu_1_2', 0.133), ('linear_2_3', 0.15), ('output_1', 0.15)]
```

### bundle-rank-changes
Rank sites by how far each member moved from the baseline.

- covers: `Bundle.rank_changes`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
ranked = bundle.rank_changes(k=3, metric="relative_l2")
print([label for label, _ in ranked])
# Output:
# ['input_1', 'linear_1_1', 'relu_1_2']
```

### bundle-comparability
Ask how two members relate before trusting a comparison; the table holds every pair.

- covers: `Bundle.comparability`, `Bundle.comparability_table`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
print(bundle.comparability("clean", "shifted"))
print(type(bundle.comparability_table).__name__, len(bundle.comparability_table))
# Output:
# Relationship.SAME_OBJECT
# dict 4
```

### bundle-stored-comparison
Store a per-node comparison on the supergraph once, then read it back by name.

- covers: `Bundle.store_comparison`, `Bundle.stored_comparison`, `Bundle.stored_comparison_names`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
name = bundle.store_comparison("relative_l2")
print(name, bundle.stored_comparison_names())
stored = bundle.stored_comparison(name)
print(type(stored).__name__, len(stored))
# Output:
# relative_l2:out@clean ('relative_l2:out@clean',)
# dict 5
```

## Running, editing and mapping over members

### bundle-fork-do
Fork the bundle, then apply one edit to every member of the fork.

- covers: `Bundle.fork`, `Bundle.do`
- tasks: ablate
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
edited = bundle.fork()
edited.do("relu_1_2", tl.zero_ablate())
print(edited.names)
print([bool((member["relu_1_2"].out == 0).all()) for member in edited])
print([bool((member["relu_1_2"].out == 0).all()) for member in bundle])
# Output:
# ['clean', 'shifted']
# [True, True]
# [False, False]
```

### bundle-map
Apply a function to each member and get the results by member name.

- covers: `Bundle.map`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
result = bundle.map(lambda trace: round(float(trace["relu_1_2"].out.sum()), 3))
print(result)
# Output:
# {'clean': 6.511, 'shifted': 8.263}
```

### bundle-run
Run every member's recorded recipe again on a model and input.

- covers: `Bundle.run`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
result = bundle.run(model, x)
print(type(result).__name__, result.names)
# Output:
# Bundle ['clean', 'shifted']
```

### bundle-vary
Give each member its own edit (or `None` for no edit) in one call.

- covers: `Bundle.vary`
- tasks: compare-runs
- fixture: toy

```python
opts = tl.options.CaptureOptions(replayable=True)
bundle = tl.Bundle({"a": tl.trace(model, x, capture=opts).fork(), "b": tl.trace(model, x, capture=opts).fork()}, baseline="a")
bundle.vary({"a": None, "b": tl.when(tl.func("relu"), tl.scale(0.0))})
print(bundle.history[-1].kind, bundle.history[-1].params["outcomes"])
print(bool((bundle["b"]["relu_1_2"].out == 0).all()))
# Output:
# vary {'a': 'identity', 'b': 'completed'}
# True
```

## Saving and drawing

### bundle-save-load
Save a bundle as one `.tlspec` folder and load it back.

- covers: `Bundle.save`
- tasks: save-load
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
bundle.save("pair.tlspec")
loaded = tl.load("pair.tlspec")
print(type(loaded).__name__, loaded.names, loaded.baseline_name)
# Output:
# Bundle ['clean', 'shifted'] clean
```

### bundle-draw
Draw the merged graph of the members to an SVG file.

- covers: `Bundle.draw`
- tasks: draw
- fixture: toy

```python
import os

bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
drawing = bundle.draw("bundle.svg")
print(type(drawing).__name__, os.path.exists("bundle.svg"))
# Output:
# Drawing True
```

### bundle-show-repr
`show` picks a view; `kind="repr"` gives each member's text form without drawing.

- covers: `Bundle.show`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "shifted": tl.trace(model, x + 1)}, baseline="clean")
shown = bundle.show(kind="repr")
print(sorted(shown), type(shown["clean"]).__name__)
# Output:
# ['clean', 'shifted'] str
```

## Relations between members

### bundle-relate-upstream-downstream
Declare that one member was forked from another, then walk the relation both ways.

- covers: `Bundle.relate`, `Bundle.upstream`, `Bundle.downstream`, `Bundle.relations`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"clean": tl.trace(model, x), "changed": tl.trace(model, x + 1)}, baseline="clean")
bundle.relate("forked_from", "changed", "clean", at_step=0)
print(bundle.upstream("changed"), bundle.downstream("clean"))
print(len(bundle.relations))
# Output:
# ('clean',) ('changed',)
# 1
```

### bundle-relation-grades
Grade each declared relation by the evidence behind it.

- covers: `Bundle.relation_grades`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"turn_1": tl.trace(model, x), "turn_2": tl.trace(model, x + 1)})
bundle.relate("successor_of", "turn_2", "turn_1", carry_mode="none")
(grade,) = bundle.relation_grades()
print(grade.grade, grade.members)
# Output:
# disclosed ('turn_2', 'turn_1')
```

### bundle-episode-status
Fold the members of a stepped episode into one status; with no episode evidence it says unknown.

- covers: `Bundle.episode_status`
- tasks: compare-runs
- fixture: toy

```python
bundle = tl.Bundle({"m0": tl.trace(model, x), "m1": tl.trace(model, x + 1)})
result = bundle.episode_status("ep")
print(type(result).__name__, result.status)
# Output:
# EpisodeFoldResult episode_unknown
```

### bundle-why-provenance
Explain how a swept member derives from the baseline, and list each member's lineage.

- covers: `Bundle.why`, `Bundle.provenance`
- tasks: sweep
- fixture: toy

```python
bundle = tl.sweep(model, x, site="relu", values=[0.0], include_baseline=True)
report = bundle.why("sweep_0")
print(report.lineage_status, report.lineage_basis)
rows = bundle.provenance()
print({row["member"]: row["lineage_status"] for row in rows})
# Output:
# exact container_operation
# {'baseline': 'baseline', 'sweep_0': 'exact'}
```

### bundle-effects
A site-sweep experiment keeps one effect row per candidate; `effects` serves that table.

- covers: `Bundle.effects`
- tasks: sweep
- fixture: toy

```python
import torchlens.experiment

base = tl.trace(model, x)
bundle = tl.experiment.sweep_sites(
    base, candidates={"relu": "relu_1_2", "fc2": "linear_2_3"}, edit=tl.zero_ablate(),
    metric=lambda t: float(t[t.output_layers[0]].out.sum()), retain="all", model=model, inputs=x)
view = bundle.effects()
print(bundle.names)
print([(row.candidate_id, row.status) for row in view.rows])
# Output:
# ['baseline', 'relu', 'fc2']
# [('__baseline__', 'completed'), ('relu', 'completed'), ('fc2', 'completed')]
```

## Token-level reads across generation calls

### bundle-token-pairs
Pair the token positions a reader call took from a source call's completion.

- covers: `Bundle.token_pairs`
- tasks: lm
- fixture: none

```python
lm = torch.nn.Sequential(torch.nn.Embedding(32, 8), torch.nn.Linear(8, 32)).eval()
bundle = tl.Bundle()
calls = (("draft", [[1, 2, 20, 21, 7, 8, 9]], [4, 7]), ("critique", [[1, 4, 7, 8, 9, 3, 11, 12]], [6, 8]))
for name, ids, completion in calls:
    trace = tl.trace(lm, torch.tensor(ids))
    facts = {"stream": "input", "batch_row": 0, "completion": completion}
    tl.io.attach_sidecar(trace, "torchlens.boundary_facts", facts)
    bundle.add(trace, names=name)
bundle.relate("fed_by", "critique", "draft", token_spans="auto")
pairs = bundle.token_pairs("critique", "draft")
print(pairs.pairs)
# Output:
# ((4, 2), (5, 3), (6, 4))
```

### bundle-token-divergence
Find where a re-run's prompt and completion part from the original's, by token id.

- covers: `Bundle.token_divergence`
- tasks: lm
- fixture: none

```python
lm = torch.nn.Sequential(torch.nn.Embedding(32, 8), torch.nn.Linear(8, 32)).eval()
bundle = tl.Bundle()
for name, ids in (("draft", [[1, 2, 3, 7, 8, 9]]), ("rerun", [[1, 2, 3, 7, 5, 6]])):
    trace = tl.trace(lm, torch.tensor(ids))
    facts = {"stream": "input", "batch_row": 0, "completion": [3, 6]}
    tl.io.attach_sidecar(trace, "torchlens.boundary_facts", facts)
    bundle.add(trace, names=name)
report = bundle.token_divergence("draft", "rerun")
print(report.prompt_same, report.completion_same, report.completion_divergence)
# Output:
# True False 1
```

### bundle-add-alternative
Merge a second run of a harness, with one call steered, and see which calls changed.

- covers: `Bundle.add_alternative`
- tasks: steer-once
- fixture: none

```python
lm = torch.nn.Sequential(torch.nn.Embedding(32, 8), torch.nn.Linear(8, 32)).eval()
def run(ids, spec=None):
    trace = tl.trace(lm, torch.tensor(ids), intervene=spec) if spec else tl.trace(lm, torch.tensor(ids))
    facts = {"stream": "input", "batch_row": 0, "completion": [3, 6]}
    tl.io.attach_sidecar(trace, "torchlens.boundary_facts", facts)
    return tl.Bundle({"draft": trace})
original = run([[1, 2, 3, 7, 8, 9]])
second = run([[1, 2, 3, 7, 5, 6]], tl.when(tl.module("1"), tl.scale(2.0)))
report = original.add_alternative(second, steered="draft")
print(original.names, [call.status for call in report.calls])
# Output:
# ['draft', 'cf_draft'] ['steered']
```
