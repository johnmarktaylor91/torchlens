# The Netron export

`tl.export.netron` turns a finished `Trace` into a single JSON file the
[Netron](https://github.com/lutzroeder/netron) viewer opens fully lit:
typed shapes on every edge (Netron's oldest open ask, #71), the module
hierarchy as one-click drill-down boxes, a curated properties panel, and an
optional metrics companion rendered in Netron's own sidebar sections. Every
keyword spelling below is DOCUMENTED-UNSTABLE pending the naming sprint.

```python
import torchlens as tl

log = tl.trace(model, x)
tl.export.netron(log, "model.json")                    # module view (default)
tl.export.netron(log, "model.json", granularity="op")  # every leaf op
tl.export.netron(log, "model.json", attachment=True)   # + metrics companion
tl.export.netron(log, open=True)                       # serve + browser tab
```

The artifact is valid ONNX protobuf JSON (`irVersion` 10): it strict-parses
into `onnx.ModelProto` and passes `onnx.checker.check_model(full_check=True)`.
It is deliberately **not a runnable model** — ops keep their captured
TorchLens names under the custom `ai.torchlens.lossy` domain, module calls
become `ai.torchlens.module` FunctionProtos, and the disclaimer plus a full
honesty block (`torchlens.capture_outcome`, counts, buffer policy,
intervention marks) ride `metadataProps`. Node names stay verbatim trace
labels: `log["conv2d_1_1"]` is the way back from any box to the live API.

## Granularities

- `"module"` (default): per-call FunctionProtos grouped at module depth 1 —
  the architecture-diagram view; deeper structure nests recursively behind
  the drill-down icon; single-op modules are inlined rather than drawn as
  one-box rooms. `depth=` overrides the grouping level and is recorded in
  the metadata.
- `"op"`: every leaf op flat. Real transformers render as a rope at this
  granularity (an enriched flat GPT-2 measures 26 screens tall); the export
  warns past the ~27-rank legibility budget and names the remedy.
- `"rolled"`: repeated passes merged by rolled identity. An edge is typed
  only when every pass agrees on shape and dtype; otherwise
  `shape_variants` / `dtype_variants` disclose the drift and the edge stays
  untyped. Recurrent feedback is disclosed via the `recurrence` attribute —
  a drawn feedback edge would be a dataflow cycle the ONNX checker rejects.

## Buffer density

`show_buffers="never"|"meaningful"|"always"` is TorchLens's existing
tri-state policy. The default `meaningful` also drops counter-update chains
(the zero-dim `add` ops feeding `num_batches_tracked` in train mode) and
disclosures ride the owning module's docString plus the model metadata
(`torchlens.hidden_buffer_count` / `torchlens.hidden_op_count`); `always`
is the forensic escape hatch. Trace under `model.eval()` for inspection
artifacts — it drops the twenty train-mode counter ops natively.

## The metrics companion

`attachment=True` writes `<stem>.attachment.json`, Netron's own
`netron:attachment` sidecar: per-node observed durations, estimated FLOPs,
and captured-tensor bytes render in dedicated "Metadata" and "Metrics"
sidebar sections after you drop the companion onto the open model. The
main artifact stays self-contained — the companion is additive depth. The
writer refuses (code `netron_attachment_invalid`) any row Netron would
silently drop.

## Opening it

- File: open in the Netron desktop app or at `netron.app` — no extra needed.
- `open=True`: serves through the netron package (loopback, ephemeral port)
  and opens a browser tab or notebook widget; needs
  `pip install 'torchlens[netron]'` (refusal code `netron_serve_unavailable`
  teaches this). With `path=None` the artifact serves from memory.

## Reading the render

Nodes draw in Netron's default dark node style (charcoal with white text) —
the same deliberate style Netron uses for its own `torch.export` graphs;
TorchLens ops are custom-domain by design (honesty beats fake category
colours), so structure, typed edges, and density carry the visual load.
Shape labels cost about half again as much height as an unlabeled graph and
are worth it; labels longer than 16 characters (5-D shapes and beyond)
truthfully fall back to the sidebar. The module drill-down affordance is
the small "ƒ" glyph at the right edge of a module box's header — it is only
6x12 px, so know to look for it; drill-down is push navigation with a
breadcrumb, not inline expansion.

## Wording contract

The properties panel is part of the accuracy contract: durations are
*observed*, FLOPs are *estimates*, activation bytes are *captured-tensor*
bytes, module totals say *inclusive*, and a fact that does not apply is
**absent, never zero**. The module-path attribute is deliberately not named
`module` — Netron already labels the ONNX domain "module" in the sidebar.

## Credit

The rendering, the ONNX JSON readers, function-definition navigation, the
`input_names` port hook, the serve/widget grammar, and the attachment
Metadata/Metrics design are Lutz Roeder's Netron. TorchLens contributes
observed execution facts Netron cannot infer; it does not present Netron's
visual grammar as its own. The demand record this design answers lives in
Netron issues #65, #71, #275, #1234, #1240, #1241, #1346, #1358, #1369,
#1370, #1480, #1544, and #1596.

## Upstream asks (filed, expected to land never)

1. A sidecar/`?attachment=` convention in `netron.serve()` so the metrics
   channel needs no manual drop.
2. A larger or labelled function-drill-down affordance.
3. `ai.torchlens.*` category rows in `onnx-metadata.json` (lowest priority;
   colour re-domaining is rejected on measured evidence — it buys a
   dark-on-dark tint shift and forfeits the checker gate).
