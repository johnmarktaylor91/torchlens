# bridge/ - Implementation Guide

Optional adapters between TorchLens traces and external tools. The package
`__init__.py` lists every adapter in `_BRIDGE_MODULES` and imports them lazily
via module-level `__getattr__`; `import torchlens.bridge` itself pulls in no
optional dependency.

## _utils.py (shared helpers)
- `source_model()` returns the live model from `trace._source_model_ref` and
  raises `ValueError` when the model was garbage-collected.
- `resolve_one_site()`, `out_at()`, `first_input_tensor()`, `tensor_layers()`
  map site-like values (labels/selectors/layer objects) to saved tensor outs.
- `module_for_site(log, site, *, bridge=)` is the ONE module resolver the
  attribution bridges share (Grad-CAM `layer()`, Captum `layer()`): a module
  address returns that module; an op label returns the OUTERMOST module in
  the op's `output_of_module_calls` (`relu_17_66` in resnet18 is `layer4`,
  never the inner `layer4.1.relu`); an op label or pass-qualified address
  whose module runs more than once refuses typed
  (`bridge_module_site_multi_call`), because a module hook sees every call.

## Adapter files and main entry points
- `captum.py`: `attribute()`, `layer()` (extra: `torchlens[captum]`;
  `layer()` is `_utils.module_for_site`).
- `shap.py`: `explain(log, *, background, inputs=None, ...)` (default
  `shap.DeepExplainer`; extra `torchlens[shap]`, `shap>=0.45.1,<1`).
  `background` is REQUIRED keyword-only: the old default made the explained
  inputs their own background, which gives all-zero values for one input.
  `inputs` defaults to the first saved input tensor.
- `sae_lens.py`: `encode()`, `decode()` (extra: `torchlens[sae]`).
- `lit/`: `model(net, tokenizer, *, task=, sites=, ...)` wraps a LIVE model as
  a real `lit_nlp.api.model.Model` (classification + causal LM); `dataset()`,
  `layout()` (extra: `torchlens[lit]`; import-inert, call-time gate; site
  identity = structural `site_key`, never labels; doc:
  `docs/reference/lit_bridge.md`). The old Trace-wrapping stub is deleted --
  real LIT rejected it at construction.
- `hf.py`: `trace_text()`, `trace_image()`, `trace_multimodal()` plus input
  detection helpers (`_is_hf_text_input`, `_is_hf_image_input`,
  `_is_hf_multimodal_input`, ...). This is the autoroute bridge: consumed by
  `autoroute/_builtin_input.py` and imported eagerly by `user_funcs.py`, so it
  must import cleanly without `transformers` installed (gating stays inside
  the functions).
- `huggingface.py`: `push_to_hub()` for artifacts (extra: `torchlens[hf]`).
- `profiler.py`: `execution_trace()`, `join()` correlate a Kineto/Chrome trace
  with captured layers; stdlib-only, no import gate. `join()` (schema
  `torchlens.profiler_join.v2`) assigns each complete event to at most one
  layer: a `record_function` range whose name EQUALS a layer label, else the
  k-th `aten::<func_name>` event (outermost of same-name nesting, time order)
  to the k-th layer of that type, `k mod L` for n repeated forwards. Types
  whose event count is not a multiple of their layer count stay unmatched in
  `mismatched_op_types` (with a `UserWarning`); every unmatched event is
  counted by name in `unmatched_event_counts`. Host-side times: not a rate
  denominator.
- `gradcam.py`: `cam()`, `layer()` (extra: `torchlens[gradcam]`). `cam()`
  splits `**kwargs` by name: keywords the CAM constructor declares
  (`reshape_transform`, ...) go to the constructor, the rest (`aug_smooth`,
  `eigen_smooth`) to the CAM call. `layer()` is `_utils.module_for_site`.
- `brain_score.py`: `per_layer()` takes a CALLABLE offline benchmark (no
  import gate; raises `TypeError` on non-callables). `get_activations_fn()`
  builds the `get_activations(images, layer_names) -> OrderedDict[str,
  np.ndarray]` callable Brain-Score's `ActivationsExtractorHelper` expects
  (offline, no import gate; layer names are module dotted paths or any
  TorchLens lookup, `"logits"` maps to the model output);
  `activations_extractor()` wires it into a real
  `ActivationsExtractorHelper` (import gate: `brainscore_vision`, which
  needs Python >= 3.11 — LIVE-GATE VERIFIED 2026-08-29 against a running
  brainscore-vision 2.3.22 install on py3.12/CPU: resnet18 through a real
  helper + StimulusSet with value/order/coordinate/logits parity).
- `rsatoolbox.py`: `dataset()` (extra: `torchlens[neuro]`).
- `mcp.py`: Model Context Protocol stdio server (`python -m
  torchlens.bridge.mcp`; extra: `torchlens[mcp]`, mcp>=2.0). Read-only tools
  over SAVED `.tlspec` artifacts + environment: `torchlens_doctor`,
  `torchlens_api_map`, `torchlens_overview`, `torchlens_dump`,
  `torchlens_explain`, `torchlens_query_sites`, `torchlens_payload_stats`,
  `torchlens_compare`, `torchlens_schema`, and the F03 ledger family
  (`torchlens_ledger_overview`/`_entry`/`_evidence`). The removed
  `torchlens_load_overview`/`torchlens_agent_dump` spellings were renamed to
  `torchlens_overview`/`torchlens_dump` (F29 remove-and-rename, no aliases).
  The pure layer (`TOOL_SPECS`/`call_tool`) has NO mcp
  dependency and is what tests drive; `_build_server()` wires the mcp>=2.0
  `MCPServer` high-level API (schemas derived from handler signatures — keep
  handler params in sync with `TOOL_SPECS`). No tool executes user code or
  mutates state; live capture stays a Python-process concern.
- `nnsight.py`: `from_trace()` normalizes a cached nnsight-style trace into a
  stable payload schema; offline, no import gate.
- `inseq.py`: `attribute()` (extra: `torchlens[inseq]`).
- `depyf.py`: `dump()` (extra: `torchlens[depyf]`).
- Contrastive steering family (`steering_vectors.py`, `repeng.py`, `dialz.py`):
  each trains the package's own vector from saved activations, never by
  re-running the model. Shared signature: `(log, positive_site,
  negative_site=None, *, negative_log=None, read_token_index=-1, ...)`; the
  negative prompts usually live in a second trace (`negative_log=`, the site
  then defaults to `positive_site`), and one token per prompt is read before
  training (`-1` = last; a sequence = one index per prompt; `None` = unsliced
  `[n, hidden]` outs). Private shared helpers: `steering_vectors._contrastive_rows`,
  `repeng._read_directions` / `_interleave` / `_model_type`. On HF decoders
  trace with `config.use_cache = False`, or `"model.layers.<i>"` is ambiguous
  (hidden state plus KV-cache outputs).
  - `steering_vectors.py`: `vector(..., trainer=None, layer=None,
    layer_type="decoder_block")` (extra `torchlens[steering]`); the default
    trainer is `steering_vectors.mean_aggregator()`; `layer=` adds a real
    `SteeringVector` under `steering_vector`. Bit-identical to
    `train_steering_vector(..., read_token_index=-1, batch_size=1)`.
  - `repeng.py`: `control_vector(..., layer, method="pca_diff",
    model_type=None)` (extra `torchlens[repeng]`) replicates
    `repeng.extract.read_representations` (PCA, sign rule, in-place centring
    for `pca_center`) and returns a real `repeng.ControlVector`.
  - `dialz.py`: `vector(..., layer, method=None, model_type=None)` (extra
    `torchlens[dialz]`, dialz 0.2 through 1.x) does the same against
    `dialz.vector.read_representations` and returns a real
    `dialz.SteeringVector`; `method=None` follows the installed release's
    default (`pca` in 1.x, `pca_diff` in 0.2). The old `analyze()` is removed.
  - repeng/dialz layer `i` matches `hidden_states[i + 1]`, which is the
    output of `model.layers.<i>` except at the last layer, where Hugging Face
    returns the final-norm output (site `"model.norm"`).

## Optional-dependency gating pattern
- Never import an optional dependency at module top level. The pattern is a
  function-local `try: import x / except ImportError: raise ImportError(...)`
  naming the exact extra, e.g.
  "Captum bridge requires the `captum` extra: install torchlens[captum].".
- Tests gate on the dependency with `pytest.importorskip()`; offline adapters
  (`brain_score`, `nnsight`, `profiler`) run without extras.

## Local Invariants / Gotchas
- Adding an adapter requires updating BOTH `_BRIDGE_MODULES` and `__all__` in
  `__init__.py`; a name missing from `_BRIDGE_MODULES` raises
  `AttributeError` on access.
- Bridges that execute the model (`captum`, `shap`, `gradcam`, ...) need the
  source model alive (repeng/dialz read only its `config.model_type`, and
  only when `model_type=` is not given); `tl.release_model()`
  or a dropped reference makes `source_model()` raise.
- Site arguments must carry saved tensor outs; `out_at()` raises `ValueError`
  otherwise. Keep error messages actionable (name the site and the extra).
