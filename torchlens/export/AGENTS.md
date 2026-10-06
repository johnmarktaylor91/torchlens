# export/ - Implementation Guide

Export bridge (`tl.export`, lazy): renders/serializes a FINISHED trace (or
bundle) into third-party formats. The facade (`__init__.py`) re-exports the
per-family modules (`_graphs` viewers, `_trackers`, `_common` shared
helpers, `_netron*` the netron subsystem) and registers every builtin
through the C01 export-target registry door (`_registry.py`, per-member
tier rows: `present` native emitters, `bridge` foreign-peer writers).

## Surface (grouped)

- Visual: `svg` (editable Graphviz SVG), `html` (self-contained page).
- Profiling: `chrome_trace`, `chrome_trace_diff` (bundle), `speedscope`,
  `flamegraph`, `memory_timeline`.
- Tabular/array: `csv`, `parquet`, `json`, `xarray`.
- Experiment trackers: `tensorboard(log, writer)`, `wandb(log, run)`,
  `mlflow(log, client, *, run_id=None)`, `aim(log, run)` — the tracker
  OBJECT is passed in (guarded by `_require_tracker_object`, duck-typed on
  the required method); this package never imports tracker SDKs itself.
  `mlflow` tells the fluent module from an `MlflowClient` by
  `log_metric`'s first parameter (`run_id`): a client needs `run_id=`
  (typed `TypeError` otherwise); the fluent module gets it as a keyword.
- Model viewers: `model_explorer`, `netron`.

## The netron subsystem (F14; kwarg spellings DOCUMENTED-UNSTABLE)

`netron(log, path=None, *, granularity="module", depth=1,
show_buffers="meaningful", attachment=False, open=False, baseline=None)`
emits schema-v2 ONNX protobuf-JSON (irVersion 10, custom domains
`ai.torchlens.lossy` / `ai.torchlens.module`, NEVER re-domained to real
ONNX names — ruling N1). Module split: `_netron.py` entry + validation +
extent warning + serve; `_netron_records.py` op/rolled projections;
`_netron_module.py` per-call FunctionProto projection (depth-1 interim
default per DISSENT D1; ruling N2 makes module the FILE default);
`_netron_fields.py` curated attrs/wording contract (module-path attr is
never named `module`; missing-is-absent-never-zero);
`_netron_emit.py` deterministic serialization; `_netron_attachment.py` the
`netron:attachment` companion with four fail-closed vendor guards. Every
artifact must stay green under strict `onnx.ModelProto` parse +
`check_model(full_check=True)` — the checker catches what netron's
permissive reader swallows, including the function-cycle class that KILLS
netron outright.

## Gotchas

- File-writing exporters take an explicit `path` and return the written
  `Path`; nothing writes to implicit locations.
- Optional heavy dependencies (pandas/pyarrow/xarray/graphviz consumers)
  resolve lazily at call time; add new exporters with the same deferred
  pattern, never module-top imports.
- These renderers are NOT behind the capture-outcome N-gates (those cover
  `tl.save`/replay/validation entries); they read whatever the log exposes.
  Report-side honesty (unverified/halted disclosure) lives in `report/`.

## Tests

`tests/test_exports.py`, `tests/test_export_behaviors.py`,
`tests/test_export_html_minimal.py`, `tests/test_io_export.py`; the netron
acceptance tiers live in `tests/test_netron_export_*.py` (T1 strict parse +
checker contract, T3 executed-vendor node harness
`test_netron_export_harness.mjs`, T4 headless-browser smoke, density/
module/rolled/attachment/composition suites, the packet generator).
