# Selection: the resolved rows a selector names

## Reading a selection

### selection-facts
Index a trace with a selector to get a Selection, then read what it holds.

- covers: `Selection.kind`, `Selection.labels`, `Selection.__len__`, `Selection.__bool__`, `Selection.is_empty`, `Selection.selector`, `Selection.digest`
- tasks: select
- fixture: toy

```python
trace = tl.trace(model, x)
selection = trace[tl.func("linear")]
print("kind", selection.kind)
print("labels", selection.labels)
print("len", len(selection))
print("bool", bool(selection))
print("is_empty", selection.is_empty)
print("selector", selection.selector)
print("digest", type(selection.digest).__name__)
# Output:
# kind ACT
# labels ('linear_1_1:1', 'linear_2_3:1')
# len 2
# bool True
# is_empty False
# selector tl.func('linear')
# digest str
```

### selection-iterate-index
Iterate a selection's site entries, or take one by position.

- covers: `Selection.__iter__`, `Selection.__getitem__`
- tasks: select
- fixture: loop

```python
trace = tl.trace(model, x)
selection = trace[tl.func("tanh")]
for entry in selection:
    print(type(entry).__name__)
print(selection[0])
# Output:
# Site
# Site
# Site
# Site ACT tanh_1:1: 8 values, relation exact
#   shape     [2, 4]
#   values    8 selected
#   relation  exact
#   source    selector tl.func('tanh')
#   more      .key  .mask  .provenance  .record
```

### selection-to-pandas
One row per selected site as a pandas table.

- covers: `Selection.to_pandas`
- tasks: select, table
- fixture: toy

```python
trace = tl.trace(model, x)
frame = trace[tl.func("linear")].to_pandas()
print(len(frame))
print(list(frame.columns))
# Output:
# 2
# ['label', 'kind', 'key', 'num_selected', 'numel', 'shape', 'relation', 'source']
```

### selection-align-to
Re-bind a selection made on one trace onto a second run of the same model.

- covers: `Selection.align_to`
- tasks: select, compare
- fixture: toy

```python
first = tl.trace(model, x)
second = tl.trace(model, torch.randn(3, 4))
selection = first[tl.func("relu")]
moved = selection.align_to(second)
print(moved.labels)
# Output:
# ('relu_1_2:1',)
```
