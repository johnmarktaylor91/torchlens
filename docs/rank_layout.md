# Rank Layout

TorchLens uses Graphviz `dot` for local-topology graphs and can switch to an internal
pure-Python rank layout for graphs whose long-range edges make `dot` expensive. No Node.js
layout dependency is required.

## Engine Selection

`layout="auto"` is the default. It estimates layout cost from the render graph
before invoking Graphviz:

```text
num_nodes + sum(rank_span for edges with rank_span > 12)
```

The rank layout is selected above 20,000 cost units. Local 5k-node chains stay on `dot`;
hub-like graphs with a small number of long edges switch to rank.

The render graph is occurrence-aware for repeated argument uses. If one parent tensor feeds
multiple argument slots of the same child, such as `x + x` or `torch.cat([x, x])`, TorchLens
draws one arrow per slot. Commutative ops keep those parallel arrows unlabeled; non-commutative
ops label each arrow with its argument slot.

## Requirements

- Graphviz installed for final DOT/SVG/PDF rendering.

## Usage

```python
log = tl.trace(model, x)
log.draw(layout="auto", path="graph", file_format="svg")
```

Use `layout="dot"` to force Graphviz or `layout="rank"` to force
the rank layout.

Use SVG for very large graphs. PDF renderers can produce empty or impractically large output at
high node counts.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Empty large PDF | Render `file_format="svg"` instead. |
| Slow layout | Reduce nesting/detail with `depth`, rolled mode, or `module=` focus. |
