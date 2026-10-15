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
  never the inner `layer4.1.relu`); a pass-qualified address resolves through
  the op that call returns the same way (`layer4.1.relu:2` is `layer4`); an
  op label or pass-qualified address whose resolved module runs more than
  once refuses typed (`bridge_module_site_multi_call`, remedy naming the
  nearest enclosing single-call module), because a module hook sees every
  call.

## Adapter files and main entry points
- `captum.py`: `attribute(log, method, target=None, *, inputs=None, **kwargs)`,
  `layer()` (extra: `torchlens[captum]`; `layer()` is `_utils.module_for_site`).
  `target=` is forwarded only when given (LayerActivation and most Neuron*
  classes reject it); without `inputs=` every logged input is forwarded (the
  tensor, or the positional tuple in `forward` order); a keyword-only, nested
  or unsaved input refuses `bridge_captum_inputs_unresolved`.
- `shap.py`: `explain(log, *, background, inputs=None, ...)` (default
  `shap.DeepExplainer`; extra `torchlens[shap]`, `shap>=0.45.1,<1`).
  `background` is REQUIRED keyword-only: the old default made the explained
  inputs their own background, which gives all-zero values for one input.
  `background=None` refuses typed (`bridge_shap_background_missing`) before
  shap is called.
  `inputs` defaults to the first saved input tensor.
- `sae_lens.py`: `encode(log, site, sae)` (the site's latents) and
  `reconstruct(log, site, sae)` (`sae.decode(sae.encode(out))`; a callable
  SAE is called on the out) (extra: `torchlens[sae]`). The old `decode(log,
  site, sae)` passed the site activation to `sae.decode` as if it were
  latents and is removed.
- `pyreft.py`: `read(directory, *, repo=None, revision=None)` returns `LoreftArtifact`
  (`rotation` R `[d, k]`, `weight` W `[k, d]`, `bias` b `[k]` as saved; `layer`,
  `component`, `unit`, `units_per_example`; `provenance` with file sha256s, source
  dtypes, float64 param sha256s). No extra and no peer import: files are read with
  `torch.load(weights_only=True)`; the weight file is `intkey_<key>.bin` under the dotted
  or underscored key. Apply with `fork.do(artifact.subspace(site, positions=mask),
  artifact.edit)` (`tl.subspace` over `R.T` plus `tl.intervention.affine_source`); the
  float32 R is orthonormalised by the subspace step, not here. Refusals
  `pyreft_artifact_invalid`, `pyreft_artifact_unsupported`, `pyreft_positions_invalid`.
  Tests: `tests/test_bridge_pyreft.py` (stand-in artifact); the pinned-pyreft oracle leg
  runs outside the test suite.
- `sae.py`: `splice(model_or_log, inputs, *, site, sae, latents_edit=None,
  tokens=None, exclude_bos=False, keep_latents=True)` returns `SpliceResult`
  (duck-typed SAE, no extra). Metrics are float64 and follow SAELens evals and
  Gemma Scope: `fraction_variance_explained_pooled` (per-dimension centring)
  and `_by_dim` beside the global-mean `fraction_variance_explained`; zero
  denominators give `nan`, `zero_variance=True` and a reason in
  `unavailable_metrics`, never 0.0. `latents`/`edited_latents` are kept by
  default (once-per-session `sae_splice_latents_large` past 256 MiB). With
  `tokens=` the first output shaped `tokens.shape + (vocab,)` is the logits:
  `ce_loss_clean`/`_spliced`/`_zero_ablated`, `ce_loss_recovered`,
  `kl_divergence`; without it they are `None` with a reason. Metric math lives
  in `_sae_metrics.py`. Recipes: `docs/recipes/pretrained_saes.md`; an SAE latent
  as a feature-visualisation objective, `docs/recipes/feature_visualisation.md`.
- `_sae_export.py` (exported by `sae.py`): `write_sae_dashboard(latents, tokens, *,
  latent_ids=None, k=20, path)` and `write_delphi(latents, tokens, *, site, model_name,
  n_splits=5, directory)` write the files SAEDashboard 0.8.0 (`SaeVisData.load_json`)
  and eai-delphi 0.1.3 (`LatentDataset`) read, from dense latents or a dataset
  `UnitTopK`; neither package is imported (json and safetensors only). The layouts copy
  the consumers' own writers (cited in the docstrings); ranking is `tl.stats.TopK`'s.
  The real loaders run in `tests/test_bridge_real_sae_export.py` (peer-leg pins).
  Recipe: `docs/recipes/sae_latent_export.md`.
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
- `hf_attention.py` (re-exported from `hf.py`): `register_attention(model=None)`
  registers the `"torchlens"` attention function AND its matching mask function
  (`torchlens_mask`, transformers' boolean `sdpa_mask` with the causal skip
  off) under one key; `record_attention(model, check_sdpa=False)` records
  per-head probabilities per attention module in an `AttentionRecording`. The
  active recording is a never-rebound `ContextVar`. transformers imports stay
  inside functions (hf.py is imported eagerly). Tests:
  `tests/test_hf_attention_backend.py` (independent float64 reference, SDPA
  parity, mask-trap tripwire) and `tests/test_hf_attention_backend_deploy.py`
  (device_map, disk offload, bitsandbytes on CPU).
- `huggingface.py`: `push_to_hub()` for artifacts (extra: `torchlens[hf]`).
- `huggingface.py`: `push_to_hub()` uploads a `Trace`/`Bundle` as a gzipped
  tar of its portable `.tlspec` bundle (never a pickle; anything else refuses
  `hub_push_artifact_unsupported`); `from_hub()` downloads it, checks every tar
  member (`hub_artifact_unsafe`) and loads eagerly through `tl.load`'s
  default-deny loader. Tests use an `api=` fake exposing `upload_file` /
  `hf_hub_download` (extra: `torchlens[hf]`).
- `profiler.py`: `join()` correlates a Kineto/Chrome trace
  with captured layers; stdlib-only, no import gate. `join()` (schema
  `torchlens.profiler_join.v2`) assigns each complete event to at most one
  layer: a `record_function` range whose name EQUALS a layer label, else the
  k-th outermost `aten::<op>` event (an aten event inside ANY other aten
  event on its thread is internal and never counts; `add` and `__add__` pool
  under `aten::add` in execution order) to the k-th layer of that op, `k mod
  L` for n repeated forwards. Every assigned type must imply the same n
  (payload `forwards`). Types whose event count is not a multiple of their
  layer count, or whose n differs from the n most types share, stay unmatched
  in `mismatched_op_types` (with a `UserWarning`); every unmatched event is
  counted by name in `unmatched_event_counts`. Host-side times: not a rate
  denominator.
- `gradcam.py`: `cam()`, `layer()` (extra: `torchlens[gradcam]`). `cam()`
  splits `**kwargs` by name: keywords the CAM constructor declares
  (`reshape_transform`, ...) go to the constructor, the rest (`aug_smooth`,
  `eigen_smooth`) to the CAM call; `call_kwargs=` is the explicit spelling
  for call options (a key given both ways refuses
  `bridge_gradcam_call_option_conflict`). `layer()` is `_utils.module_for_site`.
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
- `inseq.py`: `attribute()` (extra: `torchlens[inseq]`).
- `depyf.py`: `dump()` (extra: `torchlens[depyf]`).
- Contrastive steering family (`steering_vectors.py`, `repeng.py`, `dialz.py`):
  the direction math lives in `torchlens/intervention/_directions.py`
  (`torchlens.intervention.direction_from_pairs` returning a `Direction`).
  - `steering_vectors.py`: `from_direction(direction, *, layer=None,
    layer_type="decoder_block")` (extra `torchlens[steering]`) returns a real
    `SteeringVector` holding `direction.vector`; `Direction.to_steering_vectors()`
    calls it. Pinned to the tested line, `steering-vectors~=0.12`.
  - `repeng.py`: `from_direction(direction, *, layer=None, model_type=None)`
    (extra `torchlens[repeng]`) returns a real `repeng.ControlVector` (float32
    values; `model_type` defaults to the one the direction recorded, else
    refuses `direction_model_type_unknown`); `Direction.to_repeng()` calls it.
  - `dialz.py`: `from_direction(direction, *, layer=None, model_type=None)`
    (extra `torchlens[dialz]`, pinned to the tested line `dialz>=1.1,<2`)
    returns a real `dialz.SteeringVector`, with the same float32 values and
    `model_type` rule as repeng; `Direction.to_dialz()` calls it.
  - The converters recompute nothing and key the vector by
    `Direction.library_layer(layer)` (the site label's trailing index). The
    real-package tests prove that applying the converted object with the same
    multiplier reproduces the native steer's logits. dialz's and repeng's
    layer modules build a padding mask from the layer's `position_ids`, which
    current transformers passes as `[1, seq]`, so those tests run one prompt
    per forward. dialz's `SteeringModel` loads its checkpoint in float16; the
    real test casts the wrapped model to float32 before tracing.
  - On HF decoders trace with `config.use_cache = False`, or
    `"model.layers.<i>"` is ambiguous (hidden state plus KV-cache outputs).
- `inseq.py`: `attribute(model_or_id, inputs, *, method="integrated_gradients",
  generated_texts=None, attribution_model=None, **kwargs)`; names follow
  inseq 0.7 (`load_model(model, attribution_method)`,
  `AttributionModel.attribute(input_texts, generated_texts=...)`); inseq
  swallows unknown keywords with only a warning, so a wrong spelling is
  silently ignored -- verify keywords against inseq's signatures. `method`
  may be the TorchLens spelling (`input_x_grad` maps to `input_x_gradient`);
  an unknown name refuses `bridge_inseq_method_unknown`. With
  `generated_texts=` the bridge compares `result.info["generated_texts"]`
  with the request and refuses `bridge_inseq_forced_decoding_ignored` on a
  difference (extra: `torchlens[inseq]`).
- `depyf.py`: `dump(model, x, path, **prepare_debug_kwargs) -> list[Path]`
  runs `with depyf.prepare_debug(path): torch.compile(model)(*x)` and
  returns the files written; `path` is required; a cached compile that
  dumps nothing raises `RuntimeError` naming `torch.compiler.reset()`
  (extra: `torchlens[depyf]`).
- `dialz.py`: `from_direction()` (extra: `torchlens[dialz]`).
- `repeng.py`: `from_direction()` (extra: `torchlens[repeng]`).
- `steering_vectors.py`: `from_direction()` (extra: `torchlens[steering]`).

## Optional-dependency gating pattern
- Never import an optional dependency at module top level. The pattern is a
  function-local `try: import x / except ImportError: raise ImportError(...)`
  naming the exact extra, e.g.
  "Captum bridge requires the `captum` extra: install torchlens[captum].".
- Tests gate on the dependency with `pytest.importorskip()`; offline adapters
  (`brain_score`, `profiler`) run without extras. Fake-module
  tests pass while the real package fails (depyf and inseq
  both did), so each bridge also keeps a real-package file
  `tests/test_bridge_real_<bridge>.py` that compares against the package
  used directly.

## Local Invariants / Gotchas
- Adding an adapter requires updating BOTH `_BRIDGE_MODULES` and `__all__` in
  `__init__.py`; a name missing from `_BRIDGE_MODULES` raises
  `AttributeError` on access.
- Bridges that execute the model (`captum`, `shap`, `gradcam`, ...) need the
  source model alive (the steering converters read no model:
  `direction_from_pairs` records its `config.model_type`); `tl.release_model()`
  or a dropped reference makes `source_model()` raise.
- Site arguments must carry saved tensor outs; `out_at()` raises `ValueError`
  otherwise. Keep error messages actionable (name the site and the extra).
