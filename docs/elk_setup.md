# ELK Setup

TorchLens can use an internal ELK layout backend for large graph layout while it remains an
installable internal backend. Graphviz remains the stable default renderer.

## Requirements

- Node.js available on `PATH`.
- The ELK JavaScript package available to the internal layout bridge.
- Graphviz installed for final DOT/SVG/PDF rendering.

## Local Setup

```bash
npm install elkjs
sudo apt install graphviz
```

If your project uses a local `node_modules` directory, run commands from the repository or set
`NODE_PATH` so Node can resolve `elkjs`.

## Usage

```python
log = tl.log_forward_pass(model, x, vis_opt="none")
log.render_graph(vis_renderer="elk", vis_outpath="graph", vis_fileformat="svg")
```

Use SVG for very large graphs. PDF renderers can produce empty or impractically large output at
high node counts.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `node` not found | Install Node.js and confirm `node --version` works. |
| `elkjs` cannot be resolved | Install `elkjs` in the working directory or configure `NODE_PATH`. |
| Empty large PDF | Render `vis_fileformat="svg"` instead. |
| Slow layout | Reduce nesting/detail or use Graphviz for quick iteration. |
