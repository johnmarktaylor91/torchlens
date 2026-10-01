# Google Model Explorer export

TorchLens exports captured graphs into [Google Model Explorer](https://github.com/google-ai-edge/model-explorer)'s
file-ingest contract: their hierarchy-first viewer becomes a TorchLens
frontend, showing the graph your code ACTUALLY RAN with TorchLens' measured
facts on it. Every spelling on this page is DOCUMENTED-UNSTABLE pending the
naming session. The vendor schema of record is
`model_explorer.graph_builder` / `node_data_builder` from
`ai-edge-model-explorer` 0.1.32; the executed contract oracle is the pinned
`dist/worker.js` from npm `ai-edge-model-explorer-visualizer` 0.1.2
(`tests/test_modelexplorer_assets/`).

```python
import torchlens as tl

log = tl.trace(model, x)
tl.export.model_explorer(log, "model.json", overlays=("standard",))
tl.export.model_explorer_serve(log)          # local app, one line
tl.export.model_explorer_diff((log_a, log_b), "diff/")  # split-pane value diff
payload = tl.export.to_model_explorer_dict(log)          # the pure core
report = tl.export.validate_model_explorer_payload(payload)
```

## What the export guarantees (schema v3)

- **Namespaces are the module hierarchy.** Derived from each op's
  `module_call_stack`, dot-split so unexecuted containers (a `ModuleList`)
  reappear as collapsible levels (`transformer/h/0/attn`). A module address
  that executes more than once in a graph is pass-qualified on EVERY call
  (`blk:1`, `blk:2` -- never a bare call next to a qualified one). `%`, `/`,
  `|`, and control bytes percent-encode reversibly per component. Malformed
  stacks keep the longest verified prefix, disclosed as
  `namespace_fidelity=partial` (`strict_namespace=True` refuses typed).
- **One node-id rule for every graph product:** `site_key` + `|` + the
  1-based graph-local occurrence ordinal, appended ALWAYS. Model Explorer
  silently drops duplicate-id nodes; every alternative spelling measurably
  loses nodes on real ResNets or destroys cross-step comparability. Ids are
  identical across captures of the same architecture (the diff product's
  foundation) and byte-identical after `.tlspec` round trips. Captures
  without site keys mint visibly prefixed `legacy|` ids, stamped
  `id_fidelity=legacy` in the graph's `""` row.
- **Labels are short op types** (`conv2d`, `linear`) so the viewer's
  identical-group ("twin") detection works; the globally unique TorchLens
  spelling stays searchable as the `torchlens_label` attr, the site key as
  `site_key`.
- **Edges are occurrence-preserving and ported.** `x + x` emits two edges;
  positional args keep their recorded position, keyword args their recorded
  kwarg name/path (`inputsMetadata` carries `__tensor_tag`). Every parent
  reference resolves through an explicit two-spelling map; an unresolvable
  parent is the typed failure `model_explorer_parent_unresolvable`, never a
  silent skip. Shapes/dtypes ride `outputsMetadata` slot `"0"` under the
  vendor-resolved key `shape`.
- **Curated attrs, never the firehose:** kind, op, shape, dtype, device,
  module, pass n/N, act_bytes, params, flops, time, saved, nonfinite,
  torchlens_label, site_key -- short strings, omit-when-absent
  (`include_source=True` adds `source` basename:line). The nonfinite row
  appears only where the record actually checked.
- **Group rows for every namespace** plus the `""` root row: the visible
  TorchLens provenance-and-disclosure block (version, backend, capture
  honesty facts, privacy/id/namespace fidelity, every omission count).
- **Multi-graph collections:** unrolled exact execution is `graphs[0]` and
  the default; a rolled projection (`01-rolled`) is appended only when the
  capture has multi-pass ops, with recurrent back-edges removed from
  `incomingEdges` (dagre silently reverses cycles) and re-emitted as the
  named "recurrent feedback" edge overlay (`tasksData`).
- **Episode captures emit ONE collection:** the full exact episode graph
  first (`00-episode`), then zero-padded per-step graphs. Step membership
  joins the persisted module-call ops list with the module-stack cross-check
  asserted equal; stackless driver ops (sampling, cache glue) window-assign
  exactly once under a `driver` namespace; cross-step edges get
  deterministic `proxy|`-prefixed boundary nodes (`boundary_proxies=False`
  drops-plus-discloses); a 12 MB / 128-graph budget binds the step graphs
  with omissions listed in the full graph's `""` row. Emitted tokens ride
  step rows on local exports only.
- **Privacy is ONE mechanism:** `privacy_profile="public"` drops tokens,
  source paths, code lines, and value-derived attrs/providers BY
  CONSTRUCTION, regardless of what the capture retained.

## Node-data overlays (`overlays=`)

A thin serializer over the existing draw() overlay vocabulary -- `time`,
`flops`, `bytes`, `nan`, `magnitude`, `grad_norm`, `intervention`,
`bundle_delta` -- plus `"field:<record field>"` and callable escape hatches.
`overlays=("standard",)` writes time/flops/bytes/nonfinite sidecars
(`<stem>.nodedata.<provider>.json`, vendor `GraphNodeData` format) and a
manifest with per-reason coverage counts. Missing, nonfinite, and
rolled-varying values are omitted and counted, never a zero heatmap;
value-derived providers refuse typed on value-free captures
(`model_explorer_value_not_retained`) -- export never widens the save policy
or runs backward.

## Value diff (`model_explorer_diff`)

Writes `left.json`/`right.json`, per-pane `bundle_delta` node data (L2 norm
of saved-out differences over the id-aligned intersection), an embed config
whose `syncNavigationData` self-activates pane sync, and a README with the
app's split-pane + "Match node id" steps. Aligned ids need zero mapping
artifacts; `mappingEntries` (with `disableMappingFallback=true`) exist ONLY
for legacy-id nodes. One-sided nodes stay native presence differences;
missing values are coverage, never zero.

## Serving (`model_explorer_serve`)

Exports (when given a `Trace`) and calls the public pinned
`model_explorer.visualize(...)` -- `reuse_server=`, `host=`, `port=` forward
verbatim. Needs `pip install ai-edge-model-explorer==0.1.32`
(`model_explorer_serve_unavailable` teaches this at the point of failure).

## Validation report vocabulary

`validate_model_explorer_payload` returns a structured report; its failure
rows carry check codes (`model_explorer_duplicate_node_id`,
`model_explorer_duplicate_graph_id`, `model_explorer_edge_unresolved`,
`model_explorer_zero_edges`, `model_explorer_slot_unreferenced`,
`model_explorer_group_rows_missing`,
`model_explorer_group_row_missing_namespace`) plus disclosure counters
(`stackless_root_ops`, per-graph node/edge counts). `strict=True` raises
`model_explorer_payload_invalid` with the rows on `fields["failures"]`.
