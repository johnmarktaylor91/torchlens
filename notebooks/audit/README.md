# TorchLens Human-Facing Audit Notebooks

Coverage-optimized notebooks that exercise EVERY human-facing surface of TorchLens.
These are NOT beginner tutorials (see `notebooks/*.ipynb` for that) — they are a
developer ergonomics review: does every repr/summary/accessor look clean?

## How to review

The committed `.ipynb` are **output-stripped** by the repo's `nbstripout` git filter, so to
see executed output you review the generated artifacts (not the notebooks on GitHub):

1. Regenerate + open the executed HTML: run the export step in `CLAUDE.md` (writes
   `_exports/<notebook>.html`), then scroll each in a browser. _(`_exports/` is gitignored.)_
2. Flip through `visual/visual_audit.pdf` to catch pixel-level rendering nits
   (regenerate with `python visual/generate_visual_pack.py`).
3. Any cell marked **⚠️ GAP** is a surface that errored or felt awkward — those are the
   action items.

> If you'd rather review the executed notebooks directly on GitHub (outputs inline), add
> `notebooks/audit/*.ipynb -filter` to `.gitattributes` to exempt this folder from
> `nbstripout` and re-commit. Trade-off: text outputs land in git (small, all notebooks are
> 22-67 KB) but re-execution produces noisier diffs.

## Coverage matrix

All 13 core notebooks + the visual pack run green. `tl.__all__` coverage is 67/67 names
(selector/action/submodule holes closed). Full triaged findings live in the maintainer's
review report; the per-notebook rough edges are summarized below.

| Notebook | Surfaces covered | Green? | Notable rough edges flagged |
|---|---|---|---|
| `00_setup_and_first_capture` | `tl.trace`, `Trace.__repr__`/`__str__`, `summary()` levels, `layer_labels`/`op_labels`, `trace[label]`, `Op.out`, `tl.peek` | yes | `summary()` levels look identical on tiny models; `trace[int]`/`trace[label]` return an **`Op`** (use `trace.layers[...]` for a `Layer`) though the Op repr header says "Layer N" |
| `01_indexing_and_lookup` | `trace[int/label/substr/module_path]`, `.layers/.ops/.modules/.params/.buffers`, `AmbiguousOpLookupError`, raw labels | yes | `AmbiguousOpLookupError` not importable from `torchlens` (can't `except` it); empty accessors print bare `{}` |
| `02_activations_and_metadata` | `op.out`, `.shape`, `.dtype`, `.device_ref`, `.activation_memory`, `Bytes`/`Duration`/`Flops`/`Macs`, `saved_args`, `arg_expressions`, RNG fields, `code_context` | yes | `Bytes`/`Duration`/`Flops`/`Macs` `__repr__` show a bare number (no unit) |
| `03_the_data_model` | `Op`, `Layer`, `Module`, `ModuleCall`, `Param`, `Buffer`, `GradFn`, `GradFnCall` — `__repr__` + `to_pandas()` + key fields | yes | `Op.__repr__` typo `dype=`; property names vs `to_pandas` column names differ; `Module.to_pandas` is per-layer |
| `04_extraction_surfaces` | `tl.extract`, `tl.batched_extract`, `Container`, `Container.summary()`, `to_pandas`, `output_table` | yes | `Container` has no public constructor path; `Container.summary()==repr()`; `output_table()` needs retained logits |
| `05_save_and_load` | `tl.save`/`tl.load`, `.tlspec` bundle, save levels, round-trip, `PayloadLoadHints` | yes | `.tlspec` is a directory not a file; `lazy=True` needs `materialize_out()` |
| `06_backward_and_gradients` | backward trace, grad fields on `Op`, `GradFn`/`GradFnCall`, `tl.validate` backward parity, `draw_backward` smoke | yes | input `.grad` is None (use `Op.grad`); `GradFn` records empty pre-backward with no hint |
| `07_intervention` | `tl.sites`, selectors (`func`/`in_module`/`where`/`followed_by`/`output`/`label`/`module`/`grad_fn`/`&`/`\|`/`~`), actions (`zero_ablate`/`add`/`replace_with`/`scale`/`steer`/`mean_ablate`/`project_onto`/`swap_with`/...), `when`/`do`, `replay`/`replay_from`/`rerun`, `splice_module`, `bwd_hook`, `grad_*` | yes | **`tl.project_onto` collapses the batch dim** (crashes batch>1, projects globally at batch=1); `bwd_hook` is live-rerun-only |
| `08_bundles_and_cross_trace` | `tl.bundle`/`Bundle`, members, Super* accessors, `bundle.at`, `sweep`, pairwise diff, bundle save/load | yes | `Bundle.__repr__` is a bare object address; `sweep` rejects action helpers as values; `diff_pair` differs by level |
| `09_fastlog_record` | `tl.record`/`fastlog`, `Recording` repr, `to_trace`, `save=` predicate, `dry_run`, `halt`, module events | yes | **`Recording.n_records` returns stale 0** vs `summary()`; `dry_run` uses deprecated `keep_op=`; verbose repr |
| `10_facets` | `tl.facets` namespace, `facet`/`head` selectors, registry discovery, attention facets (guarded), `facet.grad` / reconstruction | yes | `Facet`/`MissingGradient` reprs are bare addresses; `tl.head()` silent on non-attention models |
| `11_visualization_in_workflow` | `draw` (`vis_mode` unrolled/rolled), `node_mode` presets, `module` focus, `show_containers`, `code_panel`, `node_overlay`, `vis_theme`, `show_legend`, `order_siblings`, `tl.viz.*`, `draw_backward`/`draw_combined` | yes | `vis_outpath` double-extension; `node_overlay` rejects callables; profiling node_mode shows doubled time unit (`msms`) |
| `12_validation_stats_reporting` | `tl.validate` (fwd/bwd/saved-out), `tl.report.explain`, `tl.tap`/`tl.record_span`, `tl.debug.*`, `summary` levels, `output_table` | yes | `tl.validate` returns bare bool (no per-check breakdown); `verbose=True` doesn't print; `record_span` yields plain dict |
| `13_tabular_export` | `trace.to_pandas`, per-record `to_pandas`, `torchlens.export` csv/json/parquet, schema/round-trip | yes | `torchlens.export` not in `tl.__all__`; three non-overlapping `to_pandas` schemas |
| `14_huggingface_guarded` _(optional)_ | Guarded HF text/vision trace, bridge helpers, `tl.autoroute` | not built | deferred — optional, needs `transformers` |
| `visual/generate_visual_pack.py` | All `draw()` options x model zoo; each render-path scenario; stapled `visual_audit.pdf` (24 pages) | yes | page 18 profiling doubled-unit nit; otherwise no broken pages |
