# TorchLens name index (maintainers)

This page is for maintainers and tools, not readers. It lists every public name on one line each
(namespaces, members, record fields and keywords), the full name lists of the experimental
surfaces, and their stability notes; records and keywords continue in
[name_index_keywords.md](name_index_keywords.md). The name checks in `tools/name_lockstep/` and the glossary
tests read it together with the public glossary, so a public name may be listed here, there, or
both. Readers' definitions live in the [glossary](../../reference/glossary.md).

## Experimental name lists

### ATen profile INDEX

The ATen execution profile is recorded by the public switch `CaptureOptions(aten_ops=True)`
(value-free rows) or `CaptureOptions(aten_ops="values")` (rows plus each call's output tensors in
`AtenOp.out`). It is served by the stable accessors `Trace.aten_ops`, `Trace.has_aten_ops`,
`Trace.aten_gaps` and `Op.aten_ops` (entries below this index); an absent profile refuses typed
with `AtenOpsUnavailableError` (`aten_ops_unavailable`). The profile is saved with the trace
(`FieldPolicy.KEEP`), and a loaded profile validates its foreign keys whenever present; older
artifacts contain no primitive profile. The names below are the row-level surface the profile
exposes. Each is documented unstable and may be renamed or removed without a compatibility
alias. No validation or honesty tripwire may be weakened.

<!-- ATEN-UNSTABLE-INDEX:START -->

| Surface | Exact spelling or token | Stability |
| --- | --- | --- |
| Facades, record kind, disclosure, and provenance | `AtenOp`, `OpRef`, `SuperAtenOp`, `primitive_op`, `mode_paused_interior`, `exact_via_aten`, `heuristic` | unstable -- no deprecation shim owed |
| Field-order contract | `PRIMITIVE_OP_FIELD_ORDER` | unstable -- no deprecation shim owed |
| Primitive fields | `label`, `sequence`, `capture_phase`, `forward_pass_index`, `backward_epoch_index`, `owner_func_call_id`, `parent_op_refs`, `parent_grad_fn_call_ref`, `owner_status`, `decomposition_slot`, `namespace`, `operator`, `overload`, `schema`, `schema_fingerprint`, `module_call_stack`, `input_tensor_facts`, `output_tensor_facts`, `mutation_kind`, `view_copy_kind`, `autocast_context`, `dispatch_key_context`, `grad_fn_ref`, `grad_fn_link_status`, `grad_fn_link_provenance`, `algorithmic_flops`, `flop_status`, `flop_formula_source`, `flop_formula_version`, `outcome`, `exception_type`, `execution_context`, `out` | unstable -- no deprecation shim owed |
| Observer-gap fields | `kind`, `capture_phase`, `sequence_before`, `sequence_after`, `owner_func_call_id`, `parent_op_refs`, `reason` | unstable -- no deprecation shim owed |
| Redundant Op-reference fields | `op_row_index`, `op_label`, `func_call_id` | unstable -- no deprecation shim owed |
| Tensor-fact fields | `container_path`, `tensor_impl_capability`, `logical_version`, `storage_alias_group`, `shape`, `stride`, `dtype`, `device`, `layout`, `requires_grad` | unstable -- no deprecation shim owed |
| Execution-context fields | `pytorch_version`, `backend`, `device_model`, `device_capability`, `grad_mode`, `inference_mode`, `module_training_summary`, `autocast`, `deterministic_algorithms`, `tf32_matmul_policy`, `sdpa_policy`, `compile_stance`, `owner_thread_coverage`, `completeness_witness_mode` | unstable -- no deprecation shim owed |
| Super comparison fields | `comparison_status`, `has_observation_gap` | unstable -- no deprecation shim owed |
| Invariant contracts | `primitive_op_invariants`, `non_torch_primitive_op_inert`, `aten_values_consistent` | unstable -- no deprecation shim owed |
| Switch-active load failures | `primitive_op_schema_invalid`, `primitive_op_fk_invalid` | unstable -- no deprecation shim owed |
| Capture phases | `forward`, `backward`, `setup` | unstable -- no deprecation shim owed |
| Mutation classes | `none`, `in_place`, `out_variant`, `metadata_only`, `unknown` | unstable -- no deprecation shim owed |
| View/copy classes | `view`, `copy`, `alias`, `unknown` | unstable -- no deprecation shim owed |
| Owner classes | `forward_op`, `backward_grad_fn_call`, `orphan`, `unresolved` | unstable -- no deprecation shim owed |
| Grad-link classes | `linked`, `unlinked`, `conflict`, `not_applicable` | unstable -- no deprecation shim owed |
| Dispatcher outcomes | `returned`, `raised` | unstable -- no deprecation shim owed |
| FLOP evidence classes | `formula_exact`, `estimated`, `unsupported` | unstable -- no deprecation shim owed |
| Super alignment classes | `all_present_same_schema`, `all_present_different_schema`, `sparse`, `coverage_indeterminate` | unstable -- no deprecation shim owed |
| Execution and disclosure tokens | `forced_eager`, `strict_subclass_constructor` | unstable -- no deprecation shim owed |
| Temporary label grammar | `aten_<sequence>` | unstable -- no deprecation shim owed |

<!-- ATEN-UNSTABLE-INDEX:END -->

### kernel telemetry INDEX

The optional CUDA/CUPTI adapter is the detachable trailing lane of the ATen execution profile. Its
rows are saved with the trace (`FieldPolicy.KEEP`), and the capture core does not import it.
Every spelling below is documented unstable and may be renamed or removed without a
compatibility alias.

<!-- KERNEL-TELEMETRY-UNSTABLE-INDEX:START -->

| Surface | Exact spelling or token | Stability |
| --- | --- | --- |
| Facade | `KernelLaunch` | unstable -- no deprecation shim owed |
| Launch fields | `launch_name`, `device`, `stream`, `duration`, `runtime_correlation`, `attribution_status` | unstable -- no deprecation shim owed |
| Attribution classes | `attributed`, `unavailable`, `ambiguous`, `unattributed` | unstable -- no deprecation shim owed |

<!-- KERNEL-TELEMETRY-UNSTABLE-INDEX:END -->

### attribution kit INDEX

The attribution kit is a detachable orchestration layer over shipped gradient, Selection,
intervention, and receptive-field visualization primitives. Every spelling below is documented
unstable and may be renamed or removed without a compatibility alias.

<!-- ATTRIBUTION-KIT-UNSTABLE-INDEX:START -->

| Surface | Exact spelling or token | Stability |
| --- | --- | --- |
| Integrated Gradients method and controls | `integrated_gradients`, `target`, `num_steps`, `baseline` | unstable -- no deprecation shim owed |
| Completeness evidence | `attribution_sum`, `target_delta`, `completeness_residual` | unstable -- no deprecation shim owed |
| Occlusion method and controls | `occlusion`, `selection`, `score`, `blur_kernel_size` | unstable -- no deprecation shim owed |
| Occlusion baseline policies | `zeros`, `mean`, `blur` | unstable -- no deprecation shim owed |
| Occlusion evidence | `original_score`, `occluded_score`, `selection_digest` | unstable -- no deprecation shim owed |
| CAM method and controls | `grad_cam`, `layer`, `reshape`, `relu`, `overlay`, `image`, `alpha`, `cmap` | unstable -- no deprecation shim owed |
| CAM token grid and layer-choice evidence | `grid_route`, `feature_map`, `derived`, `prefix_tokens`, `grid_evidence`, `layer_source`, `caller`, `trace`, `layer_reason` | unstable -- no deprecation shim owed |
| CAM resolution evidence | `native_map_resolution`, `rendered_map_resolution`, `upsampling`, `bilinear_display_only` | unstable -- no deprecation shim owed |
| Graph overlay bridge and controls | `overlay`, `source`, `reduce`, `abs_sum`, `abs_mean`, `sum`, `max` | unstable -- no deprecation shim owed |
| F06 wrapping contract | `noise_tunnel`, `attribute`, `method`, `method_kwargs` | unstable -- no deprecation shim owed |
| F06 noise tunnel controls and evidence | `num_samples`, `stdevs`, `reduce`, `noise_bank`, `mean_square`, `variance`, `stdevs_resolved`, `residual_of_means` | unstable -- no deprecation shim owed |
| F06 GradientShap controls | `gradient_shap`, `baselines`, `draw_bank`, `store_draws` | unstable -- no deprecation shim owed |
| F06 guided methods and evidence | `guided_backprop`, `deconvolution`, `sites`, `absolute`, `site_census` | unstable -- no deprecation shim owed |
| F06 occlusion map controls | `occlusion_map`, `window`, `strides`, `baseline_value`, `occlude_leaf`, `max_passes`, `average` | unstable -- no deprecation shim owed |
| F06 IG step runner controls and evidence | `step_batch_size`, `step_audit`, `step_audit_seed`, `per_call`, `per_chunk`, `path_evaluations_logical`, `physical_forward_calls` | unstable -- no deprecation shim owed |
| F06 completeness disclosure | `residual_rel`, `target_delta_abs`, `completeness_caveat` | unstable -- no deprecation shim owed |
| F06 metrics | `infidelity`, `sensitivity`, `AttributionMetric`, `perturb`, `noise_std`, `radius`, `gaussian`, `square_removal` | unstable -- no deprecation shim owed |
| F06 token attribution | `text`, `TokenAttribution`, `TokenAttributionPayload`, `keep_special_tokens`, `steps_per_batch`, `max_length`, `pad_token`, `auto`, `converged`, `show` | unstable -- no deprecation shim owed |
| F06 LRP mechanism | `SiteStash`, `stash`, `fetch`, `mark_firing`, `leftovers`, `register_site`, `label_of` | unstable -- no deprecation shim owed |
| Site targets | `SiteTarget`, `unit`, `call` | unstable -- no deprecation shim owed |
| Integer-input evidence | `integer_input_layers` | unstable -- no deprecation shim owed |
| Evaluation curves | `deletion_curve`, `insertion_curve`, `PerturbationCurve`, `CurveStep`, `fractions`, `scores`, `area`, `aopc`, `steps`, `features`, `patch_size`, `auto`, `elements`, `pixels`, `patches`, `tokens`, `num_features`, `index`, `fraction`, `inputs`, `input_kwargs`, `build_baseline_mask`, `intervention` | unstable -- no deprecation shim owed |
| Randomization test | `randomization_test`, `RandomizationTest`, `RandomizationStep`, `randomization`, `cascading`, `independent`, `layers`, `seed`, `keep_values`, `original_attribution`, `randomized_module_addresses`, `similarity`, `similarities`, `spearman`, `spearman_abs`, `ssim`, `values` | unstable -- no deprecation shim owed |
| ROAD curve | `road_curve`, `ranking`, `most_relevant_first`, `least_relevant_first`, `noise_std`, `seed`, `road` | unstable -- no deprecation shim owed |
| Causal audit | `causal_audit`, `CausalAudit`, `CausalAuditUnit`, `CausalAuditSet`, `claim`, `k`, `edit`, `zero`, `mean`, `controls`, `top`, `bottom`, `random`, `num_random_draws`, `per_unit`, `layer`, `site`, `original_score`, `units`, `sets`, `rank_agreement`, `set_of`, `unit`, `claimed_effect`, `measured_effect`, `kind`, `draw`, `intervention`, `component`, `claim_kind`, `constant_row`, `rows_not_audited`, `corrupted_input`, `metric`, `edges`, `edge_table`, `edge_patch`, `example_reduction`, `logits_site`, `sets_not_measured`, `reason`, `trace` | unstable -- no deprecation shim owed |
| Attribution graphs | `build_graph`, `AttributionGraph`, `GraphNode`, `transcoders`, `embedding_module`, `target`, `max_num_logits`, `logit_probability_coverage`, `demeaned_logits`, `logits_at`, `frozen`, `nodes`, `adjacency_matrix`, `node_values`, `constants`, `encoder_bias`, `decoder_bias`, `other`, `active_feature_indices`, `feature_activations`, `transcoder_sites`, `error_sites`, `error_vectors`, `input_token_ids`, `logit_token_ids`, `logit_probabilities`, `capture_status`, `disclosures`, `node_counts`, `node_indices`, `kind`, `site`, `position`, `feature_index`, `token`, `logit`, `error` | unstable -- no deprecation shim owed |
| Attribution graph export | `to_neuronpedia`, `slug`, `model_id`, `tokenizer`, `prompt`, `node_influence_coverage`, `edge_influence_coverage`, `path` | unstable -- no deprecation shim owed |
| Attribution graph overlay and upload | `influence`, `activation`, `upload_to_neuronpedia`, `api_key` | unstable -- no deprecation shim owed |

<!-- ATTRIBUTION-KIT-UNSTABLE-INDEX:END -->

### mechinterp kit additions INDEX

Additions to `torchlens.mechinterp` (the transformer analysis kit). Every spelling below is
documented unstable and may be renamed or removed without a compatibility alias.

<!-- MECHINTERP-KIT-UNSTABLE-INDEX:START -->

| Surface | Exact spelling or token | Stability |
| --- | --- | --- |
| Factored low-rank algebra (vendored from TransformerLens, MIT) | `FactoredMatrix`, `A`, `B`, `AB`, `BA`, `T`, `svd`, `U`, `S`, `V`, `eigenvalues`, `norm`, `make_even`, `corner`, `collapse_l`, `collapse_r`, `unsqueeze`, `pair`, `ldim`, `mdim`, `rdim`, `ndim` | unstable -- no deprecation shim owed |
| QK and OV circuits on verified head weight views | `qk_circuit`, `ov_circuit` | unstable -- no deprecation shim owed |
| Per-neuron MLP rows under the identity gate | `mlp_neuron_contributions`, `layers`, `neurons`, `onto`, `positions` | unstable -- no deprecation shim owed |
| Alias table generation value | `generation="4.x"` (TransformerBridge canonical hook names) | unstable -- no deprecation shim owed |
| Alias table generation value | `generation="nnterp"` (nnterp `StandardizedTransformer` accessor names: `token_embeddings`, `embed_tokens.output`, `layers_input[L]`, `layers_output[L]`, `attentions_input[L]`, `attentions_output[L]`, `attention_probabilities[L]`, `mlps_input[L]`, `mlps_output[L]`, `ln_final.input`, `lm_head.output`, `logits`); `alias_report(trace, generation="nnterp")` lists the trace's sites under these names | unstable -- no deprecation shim owed |
| Alias table generation value | `generation="interp-engine"` (interp-engine point addresses, pinned to interp-engine 1.14.3: `embeddings`, `resid_pre.L`, `attn_in.L`, `value.L`, `attn_scores.L`, `attn_probs.L`, `z.L`, `attn_out.L`, `resid_mid.L`, `mlp_in.L`, `mlp_pre.L`, `mlp_pre_linear.L`, `mlp_act.L`, `mlp_out.L`, `resid_post.L`, `lm_head`); `alias_report(trace, generation="interp-engine")` lists the trace's sites under these names, then one `structurally_absent` entry per interp-engine point no torchlens facet carries, with the reason | unstable -- no deprecation shim owed |
| Appliance on the `trace.using` entry point | `trace.using("mechinterp")` (ruled 3 Nov 2026) with `logit_lens` (ruled); `head_scores`, `qk_circuit`, `ov_circuit` | the method spellings other than `logit_lens` are unstable -- no deprecation shim owed |
| Per-query-head patching under grouped-query attention | `patch_query_heads_grid`, `site`, `heads`, `effect_site` | unstable -- no deprecation shim owed |
| Component edge graph (heads and MLPs as nodes, q/k/v split receivers) | `component_graph`, `ComponentGraph`, `ComponentSender`, `ComponentReceiver`, `ComponentEdge`, `nodes`, `receivers`, `edges`, `layer_index`, `head`, `senders`, `contribution`, `identity_receipt`, `input_label`, `norm_address`, `module_address`, `patch_edges`, `corrupted_graph` | unstable -- no deprecation shim owed |
| Edge attribution patching over the component graph | `edge_scores`, `corrupted_input`, `metric`, `method`, `eap`, `eap_ig`, `steps`, `example_reduction`, `mean`, `sum` | unstable -- no deprecation shim owed |
| Circuit faithfulness areas | `circuit_performance_ratio`, `circuit_model_difference`, `faithfulness`, `fractions`, `log_scale` | unstable -- no deprecation shim owed |
| Fixed-period induction score | `FixedPeriodInduction`, `head_scores(..., kind=FixedPeriodInduction(period=n))`, `period` | unstable -- no deprecation shim owed |
| Ablation controls (baseline, direction and site reruns of one claim) | `ablation_controls`, `AblationControlsReport`, `axis`, `baseline`, `donors`, `direction`, `feature_axis`, `n_directions`, `sites`, `seed`, `min_ratio`, `alpha`, `clean`, `effect`, `rows`, `families`, `direction_p`, `survives` | unstable -- no deprecation shim owed |
| Input feature optimization | `visualize_feature`, `inputs`, `input_shape`, `unit`, `steps`, `lr`, `seed`, `sae` | unstable -- no deprecation shim owed |
<!-- MECHINTERP-KIT-UNSTABLE-INDEX:END -->

## Unstable surfaces (documented-unstable; no deprecation shim owed)

These spellings are documented-unstable: they may rename without a deprecation shim. Each
carries the same tag at its definition.

**splice / SpliceResult (`tl.bridge.sae.splice(model_or_log, inputs, *, site, sae, latents_edit=None, tokens=None, exclude_bos=False, keep_latents=True, log_latents=False)` returns `tl.bridge.sae.SpliceResult`)** — *unstable — no deprecation shim owed*
: The SAE splice experiment: a duck-typed SAE's reconstruction replaces one site and the fork
  replays downstream. Every metric is float64 and follows the SAELens evals and the Gemma Scope
  paper.
  - Variance explained: `fraction_variance_explained_pooled` is the standard reading (each
    feature dimension, the last axis, centred on its own mean over tokens);
    `fraction_variance_explained_by_dim` gives it per dimension; `fraction_variance_explained`
    centres on one global mean. A zero denominator gives `nan`, `zero_variance=True` and a
    reason in `unavailable_metrics`, never 0.0.
  - Latents: `latents` is the encoder output exactly as `sae.encode` returned it, and
    `edited_latents` is the latents after `latents_edit`. Both are kept by default
    (`keep_latents=False` drops them; `sae_splice_latents_large` warns once past 256 MiB).
  - Loss metrics: with `tokens=` (the integer ids the model ran on), the first output shaped
    `tokens.shape + (vocab,)` is read as logits and gives `ce_loss_clean`, `ce_loss_spliced`,
    `ce_loss_zero_ablated`, `ce_loss_recovered` =
    `(CE_zero - CE_spliced) / (CE_zero - CE_clean)` and `kl_divergence` = mean
    `KL(P_clean || P_spliced)`. Without it they are `None` with a reason.
  - `exclude_bos=True` keeps sequence position 0 clean and out of every metric.
  - `log_latents=True` (off by default; live-model door only) makes `spliced` a fresh capture
    with the SAE run inside the model as ordinary ops, so the encoder ops, the latents node and
    the decoder ops are trace nodes between the site and its consumers. `latents_site` names
    the latents node (`None` with logging off). The spliced outputs are the same with logging
    on and off.
  - Recipe: `docs/field-guide/pretrained_saes.md`.

**register / unregister (`tl.bridge.sae.register(name, sae, *, site)`, `tl.bridge.sae.unregister(name)`)** — *unstable — no deprecation shim owed*
: Make (or stop making) an SAE's latents the facet family `name` at `site`. Registration is
  process-wide.
  - Addressing: `tl.facet(name).latent(k)` then addresses latent `k` at the doors a `tl.head`
    selector serves: `Trace.do`, `fork().do`, and, as a bare selector, `trace[selector]` (which
    returns the `Facet` whose `.value` is the read) and `tl.trace(save=selector)` (which keeps
    the ops that read needs).
  - Reading: `trace[site].facets[name]` (every latent), `facets[f"{name}[{k}]"]` or
    `facets[f"{name}[{j},{k}]"]` read the encoder activation.
  - Writing: a write edits the latents and adds the decoder-space change
    `decode(z') - decode(z)` to the site (for an affine decoder, `delta * W_dec[k]`), so the
    reconstruction error is untouched.
  - Sites: a string site must name one op of a layer that runs once
    (`sae_latent_site_ambiguous` otherwise); a selector site is every match, and an edit there
    fans out and reports its count (`facet_selector_fanout`).

**FacetSelector.latent (`tl.facet(name).latent(index)`)** — *unstable — no deprecation shim owed*
: Latent `index` (an int, or a list of ints edited together in one write) of
  the SAE registered as `name`; refuses `sae_latent_family_unknown`,
  `sae_latent_index_invalid` or `sae_latent_index_unknown`.

**reconstruct (`tl.bridge.sae_lens.reconstruct(trace, site, sae)`)** — *unstable — no deprecation shim owed*
: The SAE's reconstruction of a saved site activation, `sae.decode(sae.encode(out))`. A
  callable SAE without that pair is called on the activation (an SAE Lens `SAE`'s forward is
  the same encode then decode).
  - Replaces the removed `decode(log, site, sae)`, which passed the activation to `sae.decode`
    as if it were latents.
  - Related: `tl.bridge.sae_lens.encode(trace, site, sae)` gives the latents.

**write_delphi (`tl.bridge.sae.write_delphi(latents, tokens, *, site, model_name, n_splits=5, directory)`)** — *unstable — no deprecation shim owed*
: Writes SAE latents as the latent cache that delphi's `LatentDataset` loads (eai-delphi
  0.1.3).
  - Output: `directory/site/{start}_{end}.safetensors` splits with `locations` (example,
    position, latent), `float16` `activations` and the `tokens`, plus a `config.json` naming
    `model_name`.
  - Input: `latents` is dense (`SpliceResult.latents`, every activation above delphi's 1e-5)
    or the `UnitTopK` of `tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)` over a dataset
    (only the retained top entries).
  - Recipe: `docs/field-guide/sae_latent_export.md`.

**write_sae_dashboard (`tl.bridge.sae.write_sae_dashboard(latents, tokens, *, latent_ids=None, k=20, path)`)** — *unstable — no deprecation shim owed*
: Writes SAE latents and each latent's top examples as the JSON that SAEDashboard's
  `SaeVisData.load_json` loads (sae-dashboard 0.8.0).
  - Layout: one "TOP ACTIVATIONS" group of whole sequences per latent, ranked by activation,
    then lowest example, then lowest position (the `tl.stats.TopK` order).
  - Dense latents carry every token's activation and give `feature_stats` `max` and
    `frac_nonzero`; a dataset `UnitTopK` carries the ranked token's value only.
  - `latent_ids` defaults to every latent that fired; `sae_export_dashboard_large` warns once
    past 256 MiB.
  - Recipe: `docs/field-guide/sae_latent_export.md`.

**similarity_repository_id (`rdm_evolution(...).similarity_repository_id`, `tl.stats.CKA.similarity_repository_id`, the `SimilarityScore` that `tl.stats.cka` and `tl.stats.CKA.compute()` return, `ScoreInterval.similarity_repository_id`, the `neuro.rdms()` source-mode descriptor)** — *unstable — no deprecation shim owed*
: The similarity-repository catalogue identifier of the exact measure variant a
  TorchLens result computed, with `*` for the implementing repository (for
  example `measure/*/cka-kernel=linear-hsic=gretton-score` for linear CKA,
  `rdm/*/correlation` for a correlation RDM). On `rdm_evolution` results it is
  `None` when the catalogue has no exact match (the `"manhattan"` metric); the
  `neuro.rdms()` descriptor is then absent. The full table is in
  [the brain-benchmarking seam](../../reference/brain_benchmarking_seam.md).

**SimilarityScore (`tl.stats.SimilarityScore`; returned by `tl.stats.cka` and `tl.stats.CKA.compute()`)** — *unstable — no deprecation shim owed*
: A `float` subclass: the score itself (arithmetic, comparison, `repr` and JSON
  encoding are a float's; arithmetic returns a plain `float`) plus
  `similarity_repository_id`, the exact measure variant it computes
  (`measure/*/cka-kernel=linear-hsic=gretton-score` for linear CKA). `to_dict()`
  is the serialized form that keeps the identifier
  (`{"score": ..., "similarity_repository_id": ...}`); pickling keeps it too.

**ScoreInterval (`tl.repgeom.ScoreInterval`; returned by `tl.repgeom.compare_rdms_interval` and `tl.repgeom.cka_interval`)** — *unstable — no deprecation shim owed*
: A TorchLens score with a descriptive uncertainty interval beside it. Descriptive, not a
  test; inference stays with rsatoolbox.
  - Fields: `score` (the unchanged estimator on all stimuli), `low` and `high` (a percentile
    bootstrap over stimuli), `confidence`, `num_resamples`, `num_valid_resamples` (resamples
    with a finite score), `seed`, `basis` (how the interval was made, in words) and
    `similarity_repository_id` (the measure variant `score` computes: the linear CKA
    identifier from `cka_interval`, `None` from `compare_rdms_interval`, whose precomputed
    RDMs do not name their metric).
  - Methods: `width` and `to_dict()` (every field).
  - `rdm_compare_interval(rdm_a, rdm_b, method, num_resamples=1000, seed=0, confidence=0.95)`
    resamples stimuli and recomputes `compare_rdms`, dropping self-pairs;
    `cka_interval(a, b, ...)` does the same for linear CKA.

**ScoreParts (`tl.bridge.brain_score.ScoreParts`; returned by `tl.bridge.brain_score.score_parts` and, per layer, `tl.bridge.brain_score.per_layer`)** — *unstable — no deprecation shim owed*
: A benchmark score split into named parts so a ceiling-normalised value
  never stands in for the raw one: `raw`, `ceiling_normalised`, `ceiling`,
  `error`, `note`. A Brain-Score result that carries a ceiling gives its
  value as `ceiling_normalised` and its raw attribute as `raw`; a result
  without one is raw and `ceiling_normalised` is `None`. TorchLens reads
  these numbers; it computes no ceiling.

**metric_disagreement (`tl.repgeom.metric_disagreement`; returns `tl.repgeom.MetricDisagreement` with `tl.repgeom.DisagreeingPair` rows)** — *unstable — no deprecation shim owed*
: The rank order of sites (layers or models) under each comparison metric on one extraction,
  with every pair of sites that two metrics order oppositely flagged.
  - Input: `scores[metric][site]` (a float or a `ScoreInterval`). Higher scores rank first,
    ties share their average rank, missing scores take no side.
  - Flagged pairs name the site each metric ranks higher and, when the scores carry
    intervals, whether the opposite orders are separated by them (`firm`).
  - `MetricDisagreement.metrics` lists the metric names in input order;
    `MetricDisagreement.ranks` holds `ranks[metric][site]` (1 is the highest score; ties share
    their average rank; a NaN score gets a NaN rank).
  - Each `DisagreeingPair` names its two sites as `site_a` and `site_b` (in the report's site
    order) and, in `higher`, the site each metric scores higher (`None` for a tie or a missing
    score).
  - Methods: `table()` gives the plain-text report and `to_dict()` the data form.

**neuro dependency conflict (refusal code `neuro_dependency_conflict`)** — *unstable — no deprecation shim owed*
: When rsatoolbox is installed but cannot import because another installed package pins
  scikit-learn below what it needs (the `torchlens[neuro,brainscore]` install), `tl.neuro`
  leaves `rdms` and `datasets` out of `__all__` and `dir()`, and touching them raises
  `MissingDependencyError` with this code. The diagnosis reads package metadata only.
  - `exc.fields` carries `dependency_version`, `conflicting`, `conflicting_version` and
    `pinned_by` (`None` when no pinning package is found).

**neuro.compare (`tl.neuro.compare`; needs no extra)** — *unstable — no deprecation shim owed*
: One call that scores every site of one extraction against one target with several
  descriptive analyses: `tl.neuro.compare(source, target, analyses=("rsa", "ridge", "cka"))`.
  Each site's activations are read once from what `source` (a Trace with saved activations,
  an extraction, or a `{site: array}` mapping) already holds; the model is never re-run.
  - `"rsa"` is `compare_rdms` over `rdm(site)` and `rdm(target)` (`rdm_metric`,
    `rsa_method`).
  - `"ridge"` is a cross-validated ridge score: a seeded `num_folds` split, the penalty per
    fold chosen from `ridge_alphas` by exact leave-one-out on the training fold, then the
    mean over target columns of the Pearson r between out-of-fold predictions and the target.
  - `"cka"` is linear CKA. `num_resamples=` adds a `ScoreInterval` per score.
  - Report: `scores`, `intervals`, `settings`, `ridge_alphas_chosen`,
    `similarity_repository_ids`, `table()`, `disagreement()` (a `MetricDisagreement`) and
    `to_dict()`.
  - Descriptive only: inference, variance partitioning, voxel maps and stacked fits stay with
    rsatoolbox, himalaya, scikit-learn and Net2Brain.

**site_key (`Op.graph.site_key`)** — *unstable — no deprecation shim owed*
: The portable identity of one retained op's structural position (`site_key_v1`). Two
  captures of the same program agree on site keys even when their layer labels disagree.
  - Form: a `"s1|"`-prefixed string of the pass-free module address stack, the normalized op
    type, the output slot, and a 1-based occurrence ordinal within one pass-qualified
    innermost module call instance. It is minted from raw records at grouping time on every
    backend. The `"s1"` prefix makes any future re-keying a visible schema event.
  - Properties: policy-independent (identical whether grouping ran, degraded, or was off) and
    process-portable (no barcodes, ids, or arg hashes).
  - Limits: it proves position, never source identity. Cross-capture joins carry a
    per-call-instance cardinality guard, a source-location witness, and a three-tier verdict
    (corroborated / positional / refused). Editing the model's `forward()` changes downstream
    sites: keys bridge captures of the same program, never a program diff.
  - Saving: saved with the trace and recomputed byte-exactly at load; artifacts saved before
    site keys existed are wholly keyless.

**Layer.graph.site_peers** — *unstable — no deprecation shim owed*
: Layers sharing any of this layer's site keys within the trace (the reused-relu cohort
  surface), computed live per call and never persisted. Refuses typed `site_key_unavailable`
  on keyless layers, so a legacy artifact never collapses into a `None`-key
  peer-of-everything.

**Layer.shape_summary** — *unstable — no deprecation shim owed*
: A derived (never persisted) string summarizing one layer's output shapes across its passes.
  - Values: `None` for single-pass and shape-uniform layers; one varying axis renders
    `"A->B"` (monotone) or `"A-B"` (min-max); multi-axis or rank-varying layers render the
    first and last full shapes (`"2x64x8x8->2x512x4x4"`).
  - Distinct from the internal module-run fold summary (`ModuleRepeatFold.shape_summary`),
    which summarizes across a repeated module run.
  - The string legitimately contains `->`; renderers must HTML-escape it at render time
    rather than assume it is absent.

**Op.has_params** / **Layer.has_params** — *unstable — no deprecation shim owed*
: Whether the operation (or any pass of the layer) consumed a parameter tensor.
  Derived from the captured parameter barcodes; read-only.

**Op.has_saved_grad** — *unstable — no deprecation shim owed*
: Whether a gradient was captured and saved for this operation's output.

**Buffer.overwrite_index** — *unstable — no deprecation shim owed*
: The buffer pass of the buffer's most recent graph version (how many times the
  buffer was overwritten when that version was read); `None` when the buffer has
  no graph version.

**Buffer.writer_label** — *unstable — no deprecation shim owed*
: The final label of the op that produced the buffer's current value (the most
  recent write), or `None` when nothing in the forward pass wrote it. Always a
  live lookup key into the trace.

**Buffer.write_ops** — *unstable — no deprecation shim owed*
: The buffer's graph versions produced by writes, excluding the static initial
  read.

**Buffer.grad_memory** / **Param.grad_memory** — *unstable — no deprecation shim owed*
: Size in bytes of the captured gradient: the final observed gradient of the
  buffer, or the parameter's `.grad` tensor.

**Trace.nonfinite_ops** — *unstable — no deprecation shim owed*
: The queryable per-op NaN/Inf record: a tuple of pass-qualified op labels (`Op.label`, each
  a valid `trace[label]` key) whose output held at least one NaN or Inf.
  - On default captures it derives from the memoized saved-payload scan already backing
    `print(trace)`: zero capture-time cost, one scan paid at first query.
  - When the capture ran with `CaptureOptions(on_nonfinite='track')` it serves the
    capture-time verdicts instead, which also cover ops that retained no payload.
  - An empty tuple is only as strong as its coverage; read `Trace.nonfinite_coverage` first.

**Trace.nonfinite_coverage** — *unstable — no deprecation shim owed*
: The frozen evidence disclosure behind `nonfinite_ops`; the programmatic twin of the prose
  coverage-gap note in `first_nonfinite()`.
  - Fields: `basis` (`"capture"` vs `"saved_payloads"`), `checked`, `nonfinite`, `unchecked`
    (dtypes with no runnable finiteness kernel), `unexamined` (ops the scan could not look
    at), and `unmapped` (capture-basis events whose op did not survive postprocessing).

**on_nonfinite='track' (CaptureOptions)** — *unstable — no deprecation shim owed*
: Session-time opt-in: torch capture records a per-op finiteness verdict for every committed
  op output. The default stays `False` (no overhead); the `print(trace)` NOT-CHECKED health
  line names this option.
  - It never changes control flow (`on_nonfinite='raise'` stays the stop-and-throw and is
    independent). Device flags are drained in one batch after the forward, never a per-op
    CUDA synchronization.
  - `structure_only=True` refuses the combination typed.
  - Load restores the default `False`; loaded traces serve the saved-payload basis.

**grouping= (trace kwarg) / trace.options.grouping** — *unstable — no deprecation shim owed; S2-gated vocabulary*
: The grouping-policy knob, with a closed vocabulary. The mirror field
  `trace.options.grouping` records the requested value.
  - Values: `"structural"` (default; call-site grouping, `docs/guides/grouping.md`),
    `"strict_shapes"` (reserved; refuses typed until its own reviewed design lands),
    `"fold_sites"` (the within-capture folding axis; refuses typed on plain captures until
    its design is decided).
  - Refusals: unknown values refuse `grouping_invalid`; legal values that this entry point
    does not accept refuse `grouping_policy_unavailable`.
  - Distinct from the display-only `fold_repeats` viz knob, which folds repeated module runs
    at render time and never changes grouping.

**trace.options.grouping_policy** — *unstable — no deprecation shim owed; S2-gated vocabulary*
: The `grouping_policy_v1` stamp recording how the trace was grouped. It is saved with the
  trace and validated on load.
  - Fields: `policy` (the grouping that actually ran: `structural` / `params_only` / later
    `fold_sites`), `requested` (the knob mirror), `folded_sites` and `site_join` (two
    distinct site-granular axes: grouping folds vs product-layer joins), `detector`,
    `effective`, and `note`.
  - `detector`: `callsite_v1` for call-site grouping, `grouper_v1` for the structural grouper
    that serves captures without call-site evidence, `unknown` when grouping did not run.
  - Loading: loads validate against the exact writer key set, closed vocabularies, and
    coherence rules. A parse failure or incoherence warns once and settles to the one
    canonical degraded form (`policy="unknown"`, `note="grouping_stamp_<reason>"`), which
    round-trips byte-stable and stays degraded: verdicts only worsen across saves. Legacy
    artifacts from before the stamp settle silently to `grouping_stamp_legacy`; degraded
    stamps refuse stamp-consuming operations typed.

**L1 grouping refusal codes** — *unstable — no deprecation shim owed; S2-gated*
: `layer_site_ambiguous`, `site_key_unavailable`, `grouping_invalid`,
  `grouping_policy_unavailable`.

**structure_only (capture kwarg) / trace.options.structure_only** — *unstable — no deprecation shim owed*
: `tl.trace(model, x, capture=CaptureOptions(structure_only=True))` runs the capture under
  the structure-only contract: the op graph, module hierarchy, parameter geometry, and per-op
  shape/dtype are recorded, and every value-bearing claim is a hypothesis. The mirror field
  `trace.options.structure_only` declares the mode.
  - Value payloads are never retained. Value-requiring consumers refuse typed through
    `torchlens._capture.structure_only.require_structure_only_capability`, and
    value-dependent branches refuse with the user's source line (device-neutral).
  - Capability contract: [structure_only_capabilities.md](../../reference/structure_only_capabilities.md).

**Trace.discharge_against(real_trace)** — *unstable — no deprecation shim owed*
: Discharges a structure-only trace's hypotheses against an ordinary settled complete capture
  of the same graph. Returns a frozen `StructureDischarge`: a per-claim table plus an overall
  corroborated/refuted verdict, with the positional join licensed by equal graph-shape
  digests. A refuted discharge flips hypothesis consumers to typed refusals. Neither trace is
  mutated.

**StructureClaimStatus (hypothesis / corroborated / refuted)** — *unstable — no deprecation shim owed*
: The tri-state evidence class of a structure-only trace's value-bearing
  claims; never silently promoted.

**Structure-only refusal codes** — *unstable — no deprecation shim owed; S2-gated*
: `structure_only_option_conflict`, `structure_only_values_unsupported`,
  `value_dependent_branch_unsupported`, `meta_kernel_unavailable`,
  `structure_only_{save,runnable,replay,validation,backward,episode}_unsupported`,
  `structure_only_refuted_hypothesis`, `structure_only_discharge_precondition`,
  `structure_only_type_invalid`.

**Selector / Selection / `__selection__`** — *unstable — no deprecation shim owed (do/Selection spellings slate-ratified)*
: `tl.Selector` is the composable selection query: a frozen, trace-independent syntax tree
  over leaf terms (selector / receptive-field box / gradient-RF / facet / param / unit) and
  boolean combinators. `trace[selector]` (the one resolution door) returns a `tl.Selection`:
  frozen, trace-bound, an ordered tuple of `tl.types.Site(key, mask, provenance)` rows, never
  capped. `tl.Selection` is session-time only, never persisted.
  - Rows: a row's `.record` is the `Op`, `Param` or `EdgeUseRecord` it names, read through a
    weak link to the trace (`selection_trace_released` once the trace is gone). `Site.mask`
    returns a fresh materialization, so mutating it cannot alter the selection.
  - Single-thing spellings: a spelling that names one thing (an exact label, a target spec, a
    facet selector) must match exactly one site and returns that record
    (`site_spelling_not_unique` otherwise).
  - Container: `len`, iteration, `[i]` (a `Site`) and slices (a `Selection`), `.labels`
    (property, pass-qualified op labels), `.selector` (the selector that produced it, or
    `None`), `.kind`, `.is_empty`, `.digest`, `.align_to(other)` and `.to_pandas()`.
  - Display: the repr gives the kind, compact counts and producing selector, for example
    `Selection ACT: 12 sites, 184.3k values, from tl.func('tanh')`. Its card gives exact
    counts and the first five sites. Dense-mask counts that require reading tensor values say
    `not computed`.
  - The former `SiteTable` (`resolve_sites`, `.first`, `.where`) folded into it:
    `selection[0]` and `trace[selector & other]` spell those.
  - Two-level meaning: a selection is a touched-site family plus a selected-element set.
    Zero-mask entries are kept first-class (`len()` and iteration count them; the graph
    producers and `trace.subgraph` do not seed from them). `.is_empty`/`__bool__` are
    element-level; `bool()` on the query type refuses typed (`selection_bool_ambiguous`).
  - Anything region-shaped implements `__selection__`.
  - Exactness: masks are exact as sets; producer inexactness rides the closed
    `provenance.relation` lattice (`exact | upper_bound | lower_bound | unknown`).
  - Kinds: `ACT | PARAM | EDGE` are a closed vocabulary; mixed kinds refuse
    `selection_kind_incompatible`.
  - See also: [Completeness of the selection algebra](../../reference/selection_completeness.md).

**Selection operators `| & - ~` (+ reflected)** — *ratified set (slate 5.4); semantics unstable-documented*
: Set operators on selections. Same-site operands compose as masks; different-site operands
  yield a multi-site selection.
  - `-` never un-touches sites (`fam(A-B) = fam(A)`).
  - `~` is the touched-site mask complement (never predicate negation and never
    model-universe).
  - There is no `__xor__`: `(a - b) | (b - a)` spells it.
  - `BaseSelector` keeps its shipped composite semantics; `selector - selector` desugars to
    `and(a, not(b))`; a selector composed with a region producer defers to the Selection
    algebra.

**tl.units / tl.param / tl.random_like** — *unstable — no deprecation shim owed*
: Stage-1 producer constructors.
  - `units(site, indices)`: an explicit site plus index set.
  - `params(name, mask=None)`: a named-parameter element region.
  - `random_selection(like=, within=, seed=)`: a seeded size-matched control sampled without
    replacement inside `within`; too-small populations refuse `selection_unresolvable` /
    `population_too_small`.

**Value producers: tl.top_k / tl.top_fraction / tl.threshold / tl.sign** — *unstable — no deprecation shim owed*
: Select by what the values are on the resolution trace's retained activations.
  - `top_k(within, k, by='value'|'abs', largest=)`: global ranking across the population with
    a deterministic stable tie-break (canonical site order, then flat index); too few
    rankable elements refuse `population_too_small`.
  - `top_fraction(within, fraction)`: `k = ceil(fraction * population)`.
  - `threshold(within, above=, below=, by=)`: strict open bounds; both together give a band.
  - `sign(within, 'positive'|'negative'|'zero'|'nonzero', atol=)`: `'zero'` is the sparsity
    mask and the single-capture "didn't fire on this input" spelling.
  - `within`: `None` (default population: every retained tensor site, since value claims
    exist only about retained values), a site label string, or any ACT selection.
    PARAM/EDGE populations refuse `selection_kind_incompatible`.
  - Masks are computed at resolve time and are exact as sets about this capture
    (`provenance.relation="exact"`).
  - Refusals: unsaved payloads refuse `value_not_saved`; complex payloads refuse ordered
    comparisons (`value_criterion_invalid`; use `by='abs'`). NaN elements never satisfy a
    criterion.

**Statistical producers: tl.dead / tl.saturated / tl.low_variance** — *unstable — no deprecation shim owed*
: Selections that are inherently multi-sample, over an explicit evidence set: `samples=`
  takes an iterable of Traces (a `Bundle` iterates its members), at least two. The
  single-capture form is deliberately a different spelling (`sign(site, 'zero')`), so one
  name never means two things.
  - The resolution trace supplies geometry and population only; include it in `samples` to
    count it as evidence.
  - `dead(samples, atol=)` selects elements with `|v| <= atol` in every sample.
  - `saturated(samples, low=, high=, atol=)` selects elements pinned within `atol` of the
    same declared bound in every sample (a bound is required, because saturation is relative
    to the nonlinearity's range).
  - The dead and saturated selections make dispositional claims from finite evidence and declare
    `provenance.relation="upper_bound"` (observed-silent is a superset of truly-dead), with
    the sample count disclosed in `provenance.source`.
  - `low_variance(samples, threshold=)` names the sample statistic itself (elementwise
    unbiased variance in float64) and declares `exact`; its touched-family complement
    `~low_variance(...)` is the high-variance selection.
  - Refusals: missing sample sites, unsaved sample payloads, and cross-sample shape drift
    refuse typed (`site_not_in_trace` / `value_not_saved` / `mask_shape_mismatch` with the
    offending sample named); the evidence set never silently shrinks.

**Graph-structural producers: tl.neighborhood / tl.between** — *unstable — no deprecation shim owed*
: Select by where ops sit in the executed graph (DAG).
  - `neighborhood(selector, hops=, direction=)` selects every op within N recorded dataflow
    hops of a seed region (`hops=0` is the seed family itself; `direction` is
    `'both' | 'upstream' | 'downstream'`).
  - `between(sources, sinks)` selects the executed sub-DAG carrying influence from the source
    region to the sink region: exactly the ops on at least one directed source-to-sink path,
    endpoints included. No directed path resolves to the empty selection (emptiness is a
    disclosure, never an error).
  - Operands: site label strings (a bare layer label on a multi-pass layer is the all-passes
    Layer spelling), `Op`/`Layer` handles, or any ACT selection; `between` endpoints also
    accept lists of regions.
  - A seed or endpoint site that carries at least one selected element counts as its whole
    node (a partial mask never shrinks a graph region). A site whose element mask is empty
    seeds nothing: `x - y` keeps every site `x` touched, with an empty mask where `y` removed
    all of it, and those sites are not part of the region the closure grows from, so
    `tl.neighborhood(tl.all_sites() - region, ...)` grows from the sites outside `region`.
  - `trace.subgraph(selection)` takes the same member set. Membership is a structural fact
    about this capture, so entries are whole-site masks with `provenance.relation="exact"`.
  - Refusals: PARAM/EDGE operands refuse `selection_kind_incompatible`; unknown sites refuse
    `site_not_in_trace`; member sites with no output index space refuse `non_tensor_site` /
    `no_index_space`.
  - Both producers are pure functions over one executed-graph substrate
    (`torchlens._selection.graph._ExecutedDag`, shared with the influence-geometry path
    machinery), where a future graph-motif producer would plug in.
  - See also: [Completeness of the selection algebra](../../reference/selection_completeness.md) (why
    `tl.neighborhood` and `tl.between` are necessary).

**TraceSlice / trace.between / trace.subgraph** — *unstable — no deprecation shim owed*
: `trace.between(sources, sinks)` returns the influence region as a sub-graph view: the same
  member set the `tl.between` producer selects (one idea, two binding modes), presented as a
  `TraceSlice`. Session-time only; never persisted.
  - `TraceSlice` is a frozen presenter (composition, never a `Trace`/`Bundle` subclass,
    exactly like `MergedTrace`). It exposes the member `Op` records in execution order, the
    region's internal dataflow edges, and an explicit boundary: every edge crossing into the
    region (`boundary_in_edges`, the external dependencies) or out of it
    (`boundary_out_edges`) is declared, never silently dropped, plus the region's entry and
    exit ops (`source_ops`/`sink_ops`).
  - A slice deliberately offers no save, replay or validate (a region with dangling external
    inputs cannot honour them); `Trace.save` refuses it typed (`slice_save_unsupported`).
  - `__selection__` lifts the member family back into the algebra, so a slice composes under
    `| & - ~` and feeds `do()`.
  - `trace.subgraph(selection)` is the general door: any ACT region (an n-hop neighborhood,
    explicit units, a future motif matcher's hits) presents as the same view via its
    touched-site family.

**Differential producers: tl.changed / tl.top_changed** — *unstable — no deprecation shim owed*
: Select by how values differ between two runs. The subject is the resolution trace (masks
  land on its sites); the reference is one explicit Trace argument (a fork after `do()`, or a
  capture on another input). This is deliberately pairwise: multi-sample dispersion is
  `low_variance(samples=...)`.
  - Delta: directional, `subject - reference`, elementwise in float64. `by='abs'` (default)
    compares `|delta|`; `by='signed'` compares the signed delta (increased/decreased).
  - `changed(reference, within=None, above=, below=)` applies strict bounds. With neither
    given it defaults to `above=0.0`, so bare `changed(reference)` is the "every element that
    moved" intervention-effect mask.
  - `top_changed(reference, within=None, k= | fraction=, by=, largest=)` globally ranks
    deltas with the deterministic stable tie-break (exactly one of `k`/`fraction`;
    `largest=False` selects the least-moved control; too few rankable refuse
    `population_too_small`).
  - Structure checks: the reference must retain a shape-identical payload at every
    population site's pass-qualified address. Missing sites (`site_not_in_trace`), unsaved
    payloads (`value_not_saved`), shape drift (`mask_shape_mismatch`), all naming the
    reference, and structural site-key disagreement (labels that coincide across different
    architectures) refuse typed. Structures are never silently intersected, and resolving
    against the reference itself refuses (vacuous by construction).
  - NaN deltas never satisfy or rank; complex deltas refuse `by='signed'`. Relation is
    `exact` (the pairwise delta between these two captures is complete evidence).
  - PARAM populations refuse `selection_kind_incompatible`: Param records hold a live
    parameter reference, never capture-time payloads, so "weights that shifted between
    checkpoints" cannot be claimed honestly from Trace records (compare runnable-save
    `state_dict_v1` blobs; a capture-time weight differential is a named possibility, not a
    promise).

**Cross-pass producers: tl.stable_across_passes / tl.pass_variance** — *unstable — no deprecation shim owed*
: Select by behaviour across a recurrent layer's passes, pass-qualified throughout. One
  capture is the evidence; population entries group per layer.
  - `stable_across_passes(within=None, atol=, passes=)` selects elements whose range
    (max - min, float64) across the pass window is `<= atol`.
  - `pass_variance(within=None, above=, below=, passes=)` applies strict bounds on the
    unbiased (n-1) cross-pass variance (at least one bound; "variance explodes late" is
    `above=` on a late window).
  - Window: `passes=None` uses every population pass; an explicit window is an iterable of
    >= 2 distinct 1-based pass indices (each must be in the population, else
    `site_not_in_trace`, pass-qualified).
  - Minimum evidence: a layer contributing fewer than two window passes refuses
    `population_too_small` with a teaching message. A single-pass cross-pass claim is
    vacuous, so the default population on a feedforward model refuses loudly rather than
    returning an everything-mask.
  - Refusals: cross-pass shape drift refuses `mask_shape_mismatch`; complex payloads refuse
    `value_criterion_invalid`; NaN at any window pass excludes the element.
  - Masks: the element population is the intersection of the window entries' masks
    (complete evidence per element), and the mask lands on every window pass-site. The claim
    is about the unit across the window, so `do()` edits every window pass. Relation is
    `exact` (the statistic of this capture's passes, as for `low_variance`).

**Subspace producer: tl.subspace** — *unstable — no deprecation shim owed*
: Select the elements a direction (or small subspace) in activation space lives on: probe
  directions, steering vectors, PCA components, SAE decoder rows.
  `subspace(within, basis, *, origin=, method=None, dim=-1, atol=0.0)` takes one direction
  `[d]` or a stack `[k, d]` (canonicalized to float64).
  - Edits: under `do()` the edit acts on the component along the span of the basis rows.
    With `Q` the rows, it acts on the coordinates `c = Q x`, written back as
    `x + Q^T (edit(c) - c)`; the orthogonal complement is unchanged.
  - Verbs: coordinate verbs (`zero_ablate`, `scale`, `clamp`, `add_noise`,
    `mean_ablate(dim=None)`, hook callables, ...) act on `c`; value verbs (`patch_from`, the
    interchange; `replace_with`, `steer`, ...) write a value's component; other verbs refuse
    `subspace_edit_unsupported`.
  - Basis: an orthonormal basis is used as given; any other full-rank basis is
    orthonormalised (Gram-Schmidt in row order, signs kept). Non-finite bases and rows
    entirely at or below `atol` refuse at construction (vacuous direction).
  - `within=` is required (a direction is minted for one representation space) and
    restricts positions; a partial-axis mask refuses `subspace_projection_partial_axis`. A
    subspace combined with set operators, or resolved, refuses in `do()`
    (`subspace_projection_composed`).
  - Resolution: `resolve()` reports the basis's support set: every element whose coordinate
    on the bound axis carries weight `|w| > atol` in at least one basis vector, expanded
    across the site's other axes (a dense direction supports the whole axis); see
    docs/intervention_explainers.md, "Subspace edits: the component along a basis".
    Resolution is geometry-only (no payload reads, so unsaved sites resolve); relation is
    `exact`. PARAM/EDGE populations refuse `selection_kind_incompatible` (a parameter-space
    direction is a named possibility, not a promise).
  - Provenance (mandatory): `origin=` is a required non-empty account of where the basis
    came from. The origin, optional `method=`, geometry, and the basis's sha256 content
    digest are stamped into every resolved entry's `provenance.source` (and so into `do()`
    audit records); the frozen `tl.types.BasisProvenance` record is the programmatic face.
    `BasisProvenance.source_dtype` is the dtype of the caller's basis before the float64
    canonicalisation.
  - Dimension: `dim=` names the bound site axis (default `-1`, the trailing feature axis;
    conv channel directions are `dim=1`). A site whose extent there differs from `d` refuses
    `selection_unresolvable` / `basis_dim_mismatch`, never a silent broadcast, truncation, or
    padding.
  - Portable rotations: with a builtin verb the staged per-site edit is the builtin
    `subspace_edit`, whose stored arguments are the verb, the float64 orthonormal rows it
    applies (a tensor sidecar), the bound `axis`, the rows' `basis_sha256` and the basis
    provenance (`origin`, `method`, `origin_sha256`, `source_dtype`). So
    `tl.io.save_intervention(trace, path, level="portable")` reloads it as the same
    projection, never as the plain verb over the whole site. The target path follows the
    save/load symlink rule: its final component may not be a symlink; a symlinked parent
    passes.
  - A rotation learned elsewhere (DAS, ReFT) is applied as
    `tl.subspace(site, R[:k], origin=...)`; TorchLens applies it and does not fit it.
  - Masked subspaces: a subspace restricted by a `within=` element mask (one position, as
    LoReFT edits) also stores the dense `mask` and its `mask_sha256`, so it still saves
    portably; any other masked edit keeps its portability. A verb that is not builtin
    (`patch_from`, population verbs, hook callables) keeps the edit `opaque_audit`.
  - Load refusals: a malformed field, a malformed stored mask, or a stored keyword the loader
    does not know refuses `subspace_spec_invalid`; rows changed after saving refuse
    `subspace_spec_basis_tampered`; a mask changed after saving refuses
    `subspace_spec_mask_tampered`; non-orthonormal rows refuse
    `subspace_spec_basis_not_orthonormal`.

**SelectionError / selection refusal codes** — *unstable — no deprecation shim owed; S2-gated*
: One carrier class (`torchlens.errors.SelectionError`, catalogued in the intervention error
  catalog) for a closed set of codes.
  - `selection_trace_mismatch`, `selection_bool_ambiguous`, `selection_kind_incompatible`.
  - `selection_unresolvable`, with a closed reason set:
    `site_not_in_trace | value_not_saved | non_tensor_site | no_index_space | mask_shape_mismatch | facet_write_mask_unavailable | population_too_small | multipass_bare_label | value_criterion_invalid | basis_dim_mismatch`.
  - `selection_apply_invalid` (stage 2), `selection_trace_released` (`Site.record` after the
    trace was released), and `slice_save_unsupported` (`Trace.save` on a `TraceSlice`
    presenter).
  - The `multipass_bare_label` reason is the `tl.units` face of the multi-pass bare-label
    ambiguity refusal: a bare layer label addresses only single-pass layers, and each pass
    of a recurrence-grouped layer must be named pass-qualified (`label:pass`, `label:last`,
    or `label:all` for every pass). The string/`tl.label` addressing face is
    `multipass_bare_label_ambiguous` on `AmbiguousSiteError`. Both teaching refusals name
    every pass-qualified spelling, `:last`, `:all` and `trace[label]` (which gives the
    aggregate Layer); these spellings are documented-unstable.

**tl.Edit / do(selection, edit)** — *Edit ratified (slate 5.5, default-keep); mask-application semantics documented-unstable*
: `tl.Edit` is the public edit-object type (the old class spelling is removed, with no
  alias). `trace.do(selection, edit)` applies an edit to a resolved selection under the
  mask-application contract below.
  - Repr: `repr(edit)` prints the public spelling. A builtin edit prints as the public call
    that builds it (`tl.scale(2.0)`, default-valued keywords omitted; evaluating it rebuilds
    an equal edit); any other edit prints as
    `tl.Edit('name', signal=..., portability=...)`.
  - Fields: an `Edit` is a frozen record of its identity: `name`, positional `args`, keyword
    `kwargs` (a tuple of name and value pairs), `signal`, `portability`, `metadata` (name and
    value pairs, such as `("keep_grad", True)` or the selection disclosure a masked `do()`
    adds), `is_batch_independent`, `is_compatible_with_append` (an edit that may change the
    output shape is refused on an append rerun unless this is true) and `selection_recipe`
    (the selection recipe a selection-targeted `do()` attaches; `None` otherwise).
  - The hook `factory` is session-side. `Edit._PORTABLE_STATE_SPEC` is the class-level table
    of how a saved spec keeps each field: `factory` is dropped, `selection_recipe` is saved
    as a recursive blob, every other field is kept.
  - Mask application: the edit hook computes its full replacement exactly as without a mask
    (helpers stay mask-oblivious), then the engine applies
    `torch.where(mask, edited, original)` on a fresh tensor, never in place on, and never a
    view aliasing, the stored capture value. Whole-site masks skip the scatter (the unmasked
    behavior).
  - No broadcasting in v1: shape/dtype/device/broadcast mismatches and ineligible sites
    refuse `selection_apply_invalid` (closed reason set
    `shape | dtype | device | broadcast | not_maskable`).
  - Learned-parameter edits route through parameter substitution on the replay engine (see
    the entry below); rerun/set_only keep refusing typed.
  - Audit: each Selection-targeted do() appends an audit record (query repr, resolve digest
    and per-site relations) to `trace.intervention_audit`. The audit is saved with the trace,
    with its digest relation validated on load, and includes the per-site `source`
    disclosure, the PARAM row kind and PARAM-shaped recipes, and the EVENT
    transaction-envelope row kind with its optional hash-chain extension. Older readers
    refuse to load an artifact saved with selection or param interventions.

**tl.patch_from(source_trace, *, source_site=None, positions=None, position_axis=1)** — *unstable — no deprecation shim owed*
: Edit factory that patches targeted sites with another trace's recorded post-capture values
  (activation patching). With a Selection target only the selected elements are patched.
  - Facet and head targets: with a facet or head target (`tl.head(h, "v")`, `tl.facet(name)`)
    the donor is the same facet view of the source's home-op value, so only that slice is
    copied (a single head of GPT-2's fused `c_attn`, not the whole output); `positions=` then
    indexes the facet slice's axes.
  - Source-to-target mapping (Patchscopes, SelfIE): `source_site=` reads the donor at another
    site of the source trace (a label, pass-qualified on multi-pass layers, or a selector
    naming exactly one op; else `patch_source_site_unresolved`).
  - `positions=` (an int, a `{source_position: target_position}` dict, or a list of pairs;
    negatives count from the end) writes those source positions into those target positions
    along `position_axis`, leaving the other target positions unchanged. The prompts may
    differ in length and a donor batch of one broadcasts.
  - Refusals: `patch_positions_invalid`, `patch_positions_out_of_range`,
    `patch_positions_shape_mismatch`.
  - Portability `opaque_audit`: saved args carry the source trace's identity plus the mapping
    (never the Trace, never tensors); values bind at do() time session-side; no
    executable-save path in v1.

**trace.graph.edges / edge substitution** — *unstable — no deprecation shim owed; S2/S3-gated*
: `trace.graph.edges` returns the dataflow edge family: one `EdgeUseRecord` per
  parent-to-child occurrence (requires an `replayable` capture, else
  `edge_provenance_unavailable`). `do(edge_selection, edit)` replaces the value consumed on
  the edge: only the child's consumption changes, and `parent.out` stays producer truth.
  - Address: the canonical occurrence address is `(child_func_call_id, arg_kind, arg_path)`.
    Edge records lift as EDGE-kind selections (whole-edge granularity; `~` complements within
    the trace's edge family).
  - Engine: ships on the replay engine only (`edge_intervention_engine_unsupported`
    otherwise; a rerun-engine design is a named future possibility).
  - Storage: the substituted value lives in the `Op.graph.edge_substitutions` store, saved with
    the trace together with `Op.edge_replacement_stamps` and `FireRecord.edge_address`. Capture
    truth (`saved_args`, `out_versions_by_child`, `parent.out`) is kept unmodified: the
    pre-edit snapshot that makes divergence decidable.
  - Validation: every substitution entry must be corroborated (FireRecord plus stamp, and
    the stored value must digest to the stamp's `value_digest`;
    `edge_substitution_stamp_mismatch` otherwise), else FAIL. Corroborated children are
    re-executed with the substituted value spliced at the address and must match (verdict
    `edge_intervention_boundary`: a different check, never no check).
  - Passes: commits are pass-qualified. An edge into pass k of a multi-pass child edits
    exactly pass k (the FireRecord, store, and pushed cone all land on `child.label`, never
    the bare `layer_label` that names the last pass).
  - Composition: edits at one occurrence compose like replay hooks (a second edit derives
    from the substituted value; `scale(0.5)` twice is a quarter, every fire is kept). Every
    substitution entry (edge, param, region) is re-spliced whenever the cone recomputes the
    child, so a second edge edit on another arg, an upstream act edit, or a param edit
    reaching the child never silently reverts it.
  - Full validation of edited forks: the replay engine records the capture-time content
    digest of every site's out the first time it overwrites it (session-time, in the replay
    run context). The argument-logging check compares each child's retained `saved_args`
    snapshot against that digest for replay-recomputed parents, the same exact bidirectional
    check, never skipped. So non-identity act, edge, and param forks validate against their
    hand-computed ground truth and the boundary check is reached in a passing validation. A
    recomputed parent with no recordable digest reads `unverified`
    (`replay_recomputed_parent_unattested`), never validated.
  - Saving: the edge carriers are saved with the trace, so ordinary saves proceed. The
    `edge_intervention_save_unsupported` refusal stays as a schema-regression tripwire: it
    fires again at all four save levels, before `artifact_save_level_unsupported`, if the
    carriers ever stop being saved.

**parameter substitution (do over PARAM selections)** — *unstable — no deprecation shim owed*
: `fork.do(tl.param(name, mask=None), edit)` applies the edit "as if" the parameter were
  changed, for replay only: the value each consuming op sees is substituted at its derived
  occurrence address, and the live `nn.Parameter` object is never written (bit-identical
  before and after).
  - Derived addresses: parameters are not in the dataflow edge family (they classify as
    `LiteralTensor` template components), so occurrence addresses are derived from
    `Param.used_by_ops` plus template-component identity/barcode matching. This fails
    closed: `param_substitution_occurrence_underivable` when any consumption cannot be
    addressed (nested container positions, released legacy captures, a bare pass-ambiguous
    multi-pass consumer spelling, or a consumer inventory omitting a pass).
  - Reused parameters (tied weights, one module called at N passes of a recurrence-grouped
    layer) can be substituted: consumers are recorded and staged pass-qualified
    (`label:pass`), the edit lands at every consumption (a parameter has one identity across
    passes), and the pass-qualified replay engine recomputes each pass faithfully.
  - Mechanism: the substitution drives the shipped edge-substitution engine, with
    `Op.graph.edge_substitutions` entries marked `substitution_kind="param"`, edit-then-scatter
    masking over the param index space, and one replay pass over all consumer origins. Cone
    recomputation re-splices every substitution entry (param, region, and edge kinds), so
    later replays never silently revert any edit; a chained `do(tl.param(p), scale(0.5))`
    composes on the staged "as if" value, 0.25, keeping both FireRecords.
  - Validation: the same boundary as edge substitution (`edge_intervention_boundary`: a
    different check, never no check; uncorroborated entries FAIL).
  - Engine: replay engine only; rerun/set_only refuse `param_substitution_engine_unsupported`.
  - Audit: the record (kind `PARAM`) discloses "substituted at consumption … live parameters
    unchanged". The product is derived-class, never a verified reproduction of the original
    weights.
  - Buffer operands (documented-unstable): `do(tl.param("<buffer address>", mask), edit)` on
    a registered buffer (BatchNorm running stats, a rotary cache, recurrent state)
    substitutes the buffer's initial version node, the value its consumers read, and replays
    the cone through the leaf-site path (needs an `replayable=True` capture). In-forward
    writes thread the edited value; the live buffer is never written.

**TapObserver.values(masked=True)** — *unstable — no deprecation shim owed*
: `tap(resolved_selection)` stores each firing site's mask on the
  `TapRecord`; `values(masked=True)` returns fresh masked copies (selected
  elements). `values()` stays exactly the shipped full-snapshot behavior.

**color_by (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: The value source of the colour encoding channel on `Trace.draw`. See the "color_by" entry
  above for semantics.

**size_by / size_scale (draw kwargs)** — *unstable — no deprecation shim owed; keyword-only*
: The size encoding channel on `Trace.draw`. Applied by default: sqrt scaling, a conservative
  area-only mapping, and a typed refusal on rolled varying sources. See the "size_by" and
  "size_scale (size channel)" entries above for semantics.

**view (draw kwarg): the premade views** — *keyword-only*
: `Trace.draw(view=...)` draws one premade view: a bundle of draw settings plus evidence checks
  and a caption, each answering one question. `None` (the default) draws the plain graph, which
  already shows every op.
  - Values: `"overview"` (what is this model?), `"debug"` (what is broken?), `"speed"` (where
    does time go?), `"dims"` (what shape is the data?), `"transformer"` (how is attention
    wired?), `"memory"` (what is big in bytes?), `"sequence"` (how do the passes unfold?),
    `"compute"` (where are the FLOPs?) and `"interventions"` (what did the edits change?).
  - Caption: each view prints any caption lines it needs under the header block. The header
    names the model, the traced network's op count and size (a `module=` draw: that module's;
    never the drawn boxes), the parameters (compact from 1,000, `11.7M params`) and, on an
    edited trace, one edit line.
  - Combining: keywords you pass win over the view's own settings, so combinations are spelled
    with keywords, for example `trace.draw(view="debug", show_edge_shapes=True)`. `theme=`
    stays the colour skin and composes with any view.
  - `"speed"`, `"memory"` and `"compute"` turn on the `node_style="profiling"` rows themselves.
  - `"transformer"` focuses the drawing on the first repeated block that holds attention (for
    GPT-2, `transformer.h.0`) and names it in the caption; an explicit `module=` wins.
  - `"interventions"` outlines every edited op in purple (dashed for an inferred site) and
    every other op whose saved value differs from the clean run (the unedited parent trace) in
    light purple with its largest absolute change; a collapsed module box takes the marks of
    the ops inside it. It writes the full record beside a written figure as
    `<path>.census.txt`.
  - Refusals: a view whose evidence or subject is missing refuses typed with a remedy
    (`view_headline_evidence_missing`, `view_attention_subject_absent`,
    `view_single_pass_no_sequence`, or `surgery_evidence_missing` for `"interventions"` on a
    trace with no recorded edit); any other value refuses `draw_view_invalid`, naming the
    allowed values.
  - See docs/reference/lenses.md.

**ContactSheet (contact-sheet page)** — *documented-unstable*
: What `tl.viz.contact_sheet(view, page=...)` returns: the page's SVG text (a `str`, so
  every string use keeps working) carrying `page` (zero-based index) and `page_count`
  (pages in the full sheet at this `tiles_per_page`, at least 1; the "N of M" footer
  prints the same number).

**stack_by (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: The rank (stacking) encoding channel on `Trace.draw`: explicit, plus a licensed automatic
  choice on plain traces. The `True`/`"auto"` request form is itself an unstable spelling. See
  the "stack_by" entry above.

**show_redundant_args (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: Opt-out for the default-on, checked suppression of redundant constructor-argument label rows.
  See the "show_redundant_args" entry above.

**show_arg_labels (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: `Trace.draw(show_arg_labels=False)` draws no argument-position labels (`arg 0`,
  `arg (0, 1)`) on the edges into multi-input operations such as `cat`; the default `True`
  keeps them. Also `VisualizationOptions(show_arg_labels=False)`. A draw-only flag: capture
  never reads it.

**direction="rightleft" (draw value)** — *unstable — no deprecation shim owed*
: The fourth layout direction on `Trace.draw(direction=)` for the forward, backward and
  combined graphs: edges flow right to left (Graphviz `rankdir=RL`), beside `"bottomup"`,
  `"topdown"` and `"leftright"`. An unknown value refuses `visualization_direction_invalid`
  naming all four.

**path=None (draw default)** — *unstable — no deprecation shim owed*
: Where a bare draw writes. Outside a notebook a bare `Trace.draw()` writes a collision-safe
  file in the current directory and always prints `TorchLens graph written to <path>`, so a
  second bare draw never overwrites the first picture.
  - File names: `modelgraph.<format>`, then `modelgraph-2.<format>`, `modelgraph-3.<format>`.
  - In a Jupyter kernel (the same detection as `tl.draw`) a bare `trace.draw()` writes no file
    and returns the `Drawing` with its rendered bytes (SVG unless `file_format=` says
    otherwise), which the notebook displays inline.
  - An explicit `path=` is always written exactly (overwriting), in a notebook too, and printed
    only on a verbose trace.
  - `Bundle.draw` follows the same rules.

**file_format (draw kwarg)** — *unstable — no deprecation shim owed*
: The path's extension picks the format: `trace.draw("graph.png")` writes PNG, and
  `b.draw("graph.svg")` writes SVG.
  - `file_format=` applies only to a path with no extension
    (`trace.draw("graph", file_format="svg")` writes `graph.svg`) and to a bare call. Its
    default is `"pdf"` (`"svg"` for a notebook bare call and for `Bundle.draw(graph="diff")`).
  - A `file_format` that disagrees with the path's extension refuses with
    `ArgumentConflictError` (`draw_format_mismatch`).
  - Both draw doors share this rule; `VisualizationOptions.file_format` reaches `Trace.draw`
    only when set explicitly.

**show_module_paths / show_grad_values (draw kwargs, `graph="backward"`)** — *unstable — no deprecation shim owed; keyword-only*
: Opt-in label rows on `Trace.draw(graph="backward")`. Both default to `False`, leaving the
  default render unchanged.
  - `show_module_paths=True` adds `module <address>` to each backward node a module owns
    (`param <address>` on AccumulateGrad nodes).
  - `show_grad_values=True` adds `grad norm <x>, absmax <y>` from the node's captured output
    gradient (`grad values not captured` when the capture kept none).

**theme (draw kwarg, `graph="backward"`)** — *unstable — no deprecation shim owed; keyword-only*
: The skin preset on `Trace.draw(graph="backward")`, with the same names the forward picture
  takes: `"torchlens"` (default), `"paper"`, `"dark"`, `"colorblind"`, `"high_contrast"`.
  - Backward graph: the skin sets the page colour, text ink, typography and the backward key's
    ink; backward node fills and gradient arrows keep their colours.
  - Role colours on `Trace.draw`: `torchlens` keeps the released set (light green input,
    salmon output, light and darker grey weights, yellow branch tests, every other node
    white); `colorblind` renders the Okabe-Ito set with grey weights (green input, vermillion
    output); `paper` and `high_contrast` use the legacy set (light green input, light red
    output, grey weights). No skin draws blue.
  - The functional form with grouped options reads it from `VisualizationOptions.theme`.

**theme / show_legend (Bundle.draw kwargs)** — *unstable — no deprecation shim owed*
: `bundle.draw(path, theme="dark", show_legend=True)` draws the bundle picture like
  `Trace.draw`.
  - Style: the skin's page, ink and Helvetica face, and `draw`'s node shapes and fills (oval
    for one function call, box for a leaf module, cylinder for a buffer; the weight fills for
    weights).
  - `show_legend=True` adds the key under the header.
  - Member lists appear only on nodes and arrows that not every member runs.

**Bundled render font** — *unstable — no deprecation shim owed*
: Every Graphviz render resolves the pinned `Helvetica` family to the bundled Liberation Sans
  2.1.5 (SIL Open Font License 1.1, shipped unmodified under `torchlens/_visualization/fonts/`
  with its license), so text metrics no longer depend on which face a machine has installed.
  - It is wired in through a fontconfig file layered over the caller's own, which maps
    `Helvetica` and `Arial` (the family Graphviz's PostScript alias table asks for).
  - The emitted DOT keeps `fontname="Helvetica"`.

**Expansion manifest** — *unstable — no deprecation shim owed*
: The JSON file `<outpath>.expansion.json` (schema `torchlens_expansion_manifest_v1`) that a
  `draw` render writes beside an over-ceiling overview. The `collapse_expansion_manifest`
  warning discloses it.
  - Written for: a floor-fallback plan still above the readable band, or the bounded top-level
    module overview served past the pathological pre-gate.
  - Entries: one per collapsed unit, with a stable `expansion_id`, `kind`, `module`,
    `members`, and the `draw()` keywords (`expand`) that open it.

**render_vector_oversize (warning)** — *unstable — no deprecation shim owed*
: A forward SVG or PDF page past the 14,400 pt per-side vector page ceiling is written exactly as
  requested and warned with its size and the remedies (`collapse='auto'`, `module=`,
  `depth=`): warn and obey, never a refusal.

**"shape_summary" node_label_fields token** — *unstable — no deprecation shim owed*
: Selector token rendering `Layer.shape_summary` (the across-pass shape summary) as a label
  row; skipped when the field is unset. The default label renders the summary automatically on
  rolled varying multi-pass nodes, directly after the title row.

**"shape" node_label_fields token** — *unstable — no deprecation shim owed*
: Selector token rendering the node's output shape row, the same row the default label draws
  (there with the activation memory appended: `(1, 4), 16 B`).
  - Shapes print as Python tuples, except a 0-d output, which reads `scalar` (`scalar, 4 B`).
    When its activation was saved, a 0-d output gains a value row: `= 2.31`, or one value per
    pass on a rolled node (`= 0.87, 1.02, 0.95`; past four passes the first, `...`, the last).
  - A terminal boolean (an if-test) prints its outcome on its first row instead.
  - Kwarg values, parameter brackets and collapse summaries keep `()`.

**"dtype" node_label_fields token** — *unstable — no deprecation shim owed*
: Selector token rendering the node's output dtype as a label row with the
  framework namespace dropped (`"float16"`, `"bfloat16"`; Keras `show_dtype`
  style), for spotting where a mixed-precision graph changes dtype. Skipped
  when the node has no dtype.

**torchlens.preprocessing.imagenet_default()** — *unstable — no deprecation shim owed*
: The explicit opt-in to the TorchLens-authored ImageNet recipe (resize 256, center-crop 224,
  ImageNet mean/std), passed as
  `CaptureOptions(input_transform=torchlens.preprocessing.imagenet_default())`.
  - A PIL image whose model resolves no preprocessing authority (torchvision weights handle,
    timm data config, HF image processor) refuses typed `preprocessing_unresolved` instead of
    receiving this recipe silently.
  - The opted-in capture's `input_preprocessor` keeps `source="imagenet_default"`,
    `verified=False`, status `unverified_fallback`: never an authority.

**torchlens.preprocessing.attach_authority(model, authority)** — *unstable — no deprecation shim owed*
: Records a model's preprocessing authority on the model so discovery finds it. Returns the
  model.
  - Why: a torchvision model built with `weights=` keeps no handle to its weights, and the
    preset cannot be inferred from the class.
  - Without a handle, discovery matches the model's loaded state against the presets whose
    checkpoints are already in the torch hub cache (exact tensor equality, one unique match,
    never a download) and records the match as model metadata with
    `resolution_method="state_match"`. `attach_authority` remains the explicit route when no
    cached checkpoint matches.
  - Afterwards `resolve(model=model)`, PIL-input autorouting and `check_card` resolve the
    attached authority as model metadata.
  - Accepts any recognized authority; an opaque callable refuses `preprocessing_attach_opaque`.

**torchlens.preprocessing.check_card(trace, *, model=None, authority=None, offline=None)** — *unstable — no deprecation shim owed*
: The on-request preprocessing verifier. Nothing runs it automatically.
  - `trace` is a `Trace` (its `input_preprocessor` is read) or a `ResolvedPreprocessing`
    record.
  - It compares the preprocessing a capture recorded (`Trace.provenance.input_preprocessor`)
    with the model's known card (HF image processor, torchvision weights, timm or open_clip
    config, or a named `authority=`) and returns the field-level `PreprocessingComparison`.
  - It warns `preprocessing_card_mismatch` when a field disagrees. An opaque record compares as
    unknown, never as a match.

**torchlens.postprocessing** — *unstable — no deprecation shim owed*
: The public home of custom output detectors: the registry that `Trace.decode_output` and the
  end-of-capture decode consult.
  - Functions: `register(name=, priority=)` (a decorator; lower priority runs first),
    `unregister(name)`, `list(pattern=None)`, `info(name)`, and the `snapshot()`
    save-and-restore context manager.
  - Records: the frozen `Detector` metadata record and the `ResolvedPostprocessing` record a
    detector returns.
  - A detector receives `(outputs, meta)` and returns `ResolvedPostprocessing` or `None`.
  - Registering a name twice refuses `ValueError` naming `unregister`.

**torchlens.preprocessing.register / torchlens.preprocessing.unregister** — *unstable — no deprecation shim owed*
: The public door of the input detector registry, the one the built-in Hugging Face text,
  image and multimodal routes use.
  - Functions: `register(name=, priority=)` (a decorator; lower priority runs first, the
    built-ins sit at 10 to 30) and `unregister(name)`.
  - A detector receives `(model, payload, **kwargs)` before backend routing and returns a
    `Trace` or `None`. It claims the call by re-entering `tl.trace` with what `forward` takes,
    forwarding `kwargs` so the caller's `capture=`, `save=` and the other options survive, and
    declines its own converted input so the re-entry does not loop.
  - An explicit `CaptureOptions(input_transform=...)`, `backend=` or `chunk_size=` skips every
    detector for that call.
  - Registration affects later traces only; registering a name twice refuses `ValueError`
    naming `unregister`.

**image file path as `tl.trace` input** — *unstable — no deprecation shim owed*
: A `str` or `os.PathLike` naming an existing file with an image suffix, or a list of them, is
  opened as an RGB PIL image before routing. So `tl.trace(vision_model, "dog.jpg")` takes the
  image path with resolved preprocessing instead of the text path.
  - Suffixes: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`, `.tif`, `.tiff`.
  - A path string with whitespace is text.
  - Refusals: a missing image path refuses `image_path_missing` unless the model has a text
    tokenizer (then the string is text); an undecodable file refuses `image_path_unreadable`.

**torchlens.bridge.pyreft.read(directory, *, repo=None, revision=None)** — *documented-unstable*
: Read a pyreft LoReFT artifact into a `torchlens.bridge.pyreft.LoreftArtifact`. It never
  imports pyreft or pyvene and never fits anything.
  - Input: the directory `ReftModel.save` writes, holding `config.json` plus
    `intkey_<key>.bin` under the key's dotted or underscored name.
  - Apply it with `fork.do(artifact.subspace(site, positions=mask), artifact.edit)`, which is
    `tl.subspace(tl.units(site, ...), R.T, origin=..., method="loreft")` with
    `tl.intervention.set_affine_coordinates(W, bias=b)`.
  - Precision: pyreft trains `R` in float32 (orthonormal to about `1e-7`), so the subspace
    orthonormalises it once in float64. The saved edit records `origin_sha256` (R as read),
    `basis_sha256` (rows applied), `orthonormalised=True` and `source_dtype="float32"`.
  - Refusals: `pyreft_artifact_invalid` (missing or malformed file) and
    `pyreft_artifact_unsupported` (not exactly one `LoreftIntervention`).

**torchlens.bridge.pyreft.LoreftArtifact** — *documented-unstable*
: Frozen record of one LoReFT intervention as read.
  - Tensors, in their saved dtypes: `rotation` (`R`, `[d, k]`), `weight` (`W`, `[k, d]`),
    `bias` (`b`, `[k]`).
  - Site: `layer`, `component`, `unit` and `units_per_example` as pyreft names the site (for
    example layer 8, `"block_output"`, `"pos"`, 1).
  - Other fields: `rank`; `origin` (the subspace `origin=` text); `provenance` (repo,
    revision, each file's sha256, the weight file name, each tensor's source dtype, the float64
    sha256 of `R`, `W`, `b`, and `orthonormality_max_abs`).
  - `subspace(site, *, positions=None)` returns the selection. `site` is a site label or the
    site from a trace, `trace[label]`; `positions` is a bool mask over the site's axes except
    the trailing feature axis, and `None` is every position. A non-bool mask refuses
    `pyreft_positions_invalid`.
  - Given the site from a trace, its shape is checked at once: a trailing axis that is not the
    artifact's width (or a feature axis that is not last) refuses `pyreft_site_invalid`, and a
    mask that does not cover the other axes refuses `pyreft_positions_invalid`. A label is
    checked at `do()`.
  - `edit` is the float64 `set_affine_coordinates`.

**torchlens.bridge.huggingface_hub.from_hub(repo_id, *, path_in_repo, revision, token, api)** — *unstable — no deprecation shim owed*
: Downloads the gzipped `.tlspec` tar that `push_to_hub` uploads, checks every tar member (no
  links, absolute or parent-escaping names, one top-level bundle), and loads it eagerly onto
  the CPU through `tl.load`'s default-deny loader.
  - `push_to_hub` accepts only a `Trace` or `Bundle` and never uploads a pickle
    (`hub_push_artifact_unsupported`); its default `path_in_repo` is
    `torchlens_artifact.tar.gz`.

**call_kwargs (torchlens.bridge.pytorch_grad_cam.cam)** — *unstable — no deprecation shim owed*
: Mapping forwarded unchanged to the pytorch-grad-cam call (`aug_smooth`,
  `eigen_smooth`); an option given both here and as a keyword refuses
  `bridge_gradcam_call_option_conflict`.

**SummaryCallback (torchlens.callbacks.lightning)** — *unstable — no deprecation shim owed*
: Lightning callback printing `tl.summary` once per fit. Rank zero only; the report stays on
  `.report`.
  - Sample: the module's `example_input_array` (at fit start), else the first training batch
    (before its training step), whose forward input is `input_getter(batch)`, else
    `batch["x"]`, else `batch[0]`, else the batch (a mapping as keyword arguments).
  - Cost: one extra forward in eval mode under no_grad; every module's training flag and the
    RNG state are restored.
  - A rejected sample warns `lightning_summary_skipped`.

**LayerProfilerCallback run_id / disclosure (torchlens.callbacks.lightning)** — *unstable — no deprecation shim owed*
: Every JSONL record carries `run_id` and `disclosure`.
  - `run_id`: fresh at each Lightning `setup`, or fixed by
    `LayerProfilerCallback(..., run_id=...)`.
  - `disclosure`: `PROFILE_DISCLOSURE` (one extra forward per profiled batch, eval mode,
    no_grad, training flags restored).
  - A batch the forward cannot take is skipped with a `lightning_profile_skipped` warning;
    training continues.

**MLflowSink, CometSink (torchlens.trackers)** — *unstable — no deprecation shim owed*
: Forward-watch sinks over a run or experiment the user created; the sink never starts or ends
  it. Both implement the `torchlens.trackers` sink protocol.
  - `MLflowSink(client=None, *, run_id=None)` takes the fluent module, an `MlflowClient` +
    `run_id`, or nothing. It keeps the caller's step; raw histograms are exact JSON artifacts.
  - `CometSink(experiment)` takes an `Experiment` or `OfflineExperiment`; scalars and text
    only (Comet re-bins histograms, so `raw_histogram` is not advertised).
  - `capabilities()`: what the sink accepts (`MLflowSink`: `scalar`, `raw_histogram`,
    `text_manifest`, `run_metadata`; `CometSink` the same without `raw_histogram`).
  - `emit_scalar(point)`: logs one metric at the caller's step.
  - `emit_text(point)`: stores one text payload (an MLflow text artifact, a Comet text log
    tagged in its metadata).
  - `emit_histogram(point)`: stores exact counts and edges as one MLflow JSON artifact;
    `CometSink` refuses it with `tracker_sink_capability_missing`.
  - `flush()`: nothing to force (MLflow writes synchronously, Comet writes at
    `experiment.end()`).
  - `close()`: idempotent; never ends the user's run or experiment.

**tl.export.mlflow / tl.export.aim parity keys** — *unstable — no deprecation shim owed*
: Both exporters log the per-layer table, the report and every honesty fact as
  a string tag (wandb-export parity) and return `table`, `report`, `tags`
  (and `artifacts` for MLflow) beside the summary metrics and `capture_honesty`.

**tl.export.xarray(trace)** — *unstable — no deprecation shim owed*
: Returns an `xarray.Dataset`. It used to return a bare `DataArray`; read today's array as
  `tl.export.xarray(trace)["out"]`.
  - `out`: the NeuroidAssembly `DataArray` of saved non-buffer outs, `presentation` by
    `neuroid` with `layer` / `layer_label` / `neuroid_index` coords.
  - Buffers: each saved buffer layer is its own variable named by its layer label, with own
    dims `<label>_dim_<i>`, no presentation axis, values unchanged, and attrs
    `torchlens_kind="module_state"`, `layer_label`, `buffer_address`. The Dataset's
    `attrs["module_state_variables"]` lists them.
  - Save-all and partial traces both export buffers (the former buffer-layer refusal is
    removed); unsaved layers are listed in `attrs["skipped_unsaved_layers"]`.
  - Leading-size disagreement refuses `xarray_presentation_mismatch`.

**watch(grain="op", allow_forward_wrap=True) (torchlens.trackers)** — *unstable — no deprecation shim owed*
: Op-grain (tier O) watching: per-op activations from the op-tier collector during ordinary
  training.
  - Needs `signals=("activations",)`, `select=` (module path prefixes or a selector-algebra
    object) and `inputs=`.
  - Consent: refuses `watch_forward_wrap_consent_required` without `allow_forward_wrap=True`,
    the consent to wrap the root module's forward.
  - Calibration: attach calibrates the wrap (plain vs wrapped forward, RNG and buffers
    restored) into `session.collector.calibration` and discloses it in `plan_report()`, the
    `torchlens/meta/op_grain_calibration` and `torchlens/run/op_grain_overhead_median_x` rows
    and one `watch_op_grain_cost_disclosed` warning.
  - `facets={op_label: splitter}` adds per-facet series under `<op label>/<facet>`.
  - Unknown `grain=` refuses `watch_grain_invalid`.

**Watch callback environment configuration (TORCHLENS_WATCH_SIGNALS, _SELECT, _EVERY, _HIST_EVERY)** — *unstable — no deprecation shim owed*
: An explicitly constructed `HFTrainerWatchCallback` / `LightningWatchCallback` takes settings
  it was not given from these variables. They are configuration only: the environment never
  activates instrumentation.
  - At attach it prints each setting's source through `print_fn=`: `[argument]`,
    `[env NAME]`, `[derived: logging_steps=N]` or `[default]`.
  - Resolved values and sources stay on `callback.settings`; `hist_every=` is a constructor
    argument too.
  - A malformed variable refuses `watch_env_config_invalid`.

**show_legend=None AUTO value** — *unstable — no deprecation shim owed*
: The tri-state AUTO value on the stable `show_legend` kwarg: no legend unless an encoding
  channel is active, then the channel key alone. A `module=` focused draw also shows the key.
  - The reading key (what `@name`, an op name's two counters, the shape and size line and call
    arguments mean) is opt-in: `show_legend=True` adds it to a focused draw.
  - `True`/`False` keep their stable historical meanings.

**Encoding refusal codes** — *unstable — no deprecation shim owed*
: `encoding_source_invalid`, `encoding_value_invalid`,
  `encoding_callable_error`, `encoding_requires_dot_layout`,
  `size_by_rolled_varying`, `scale_requires_size_by`, `encoding_scale_invalid`,
  `stack_by_auto_underivable`, `stack_by_requires_unrolled`.

**trace.autograd.grad_fn_fire_timings** — *unstable — no deprecation shim owed*
: Live-trace-only backward timing spans, one per fire, keyed like
  `trace.autograd.grad_fn_calls` (`"<grad_fn_label>:<call_index>"`).
  - Both stamps of every span come from `time.perf_counter()`, paired at capture time by a
    per-node keyed LIFO (stale entries are discarded, never paired). Untimed fires read
    `None`, never a false zero.
  - Served from the runtime `GradFnFired` event stream, which never persists: loaded or
    cleaned traces refuse typed `grad_fn_fire_timing_unavailable`.
  - A saved trace keeps the per-fire `perf_counter` pair in the `GradFnCall` timing fields,
    discriminated by `Trace.grad_fn_timing_provenance`.

**Trace.grad_fn_timing_provenance** — *unstable — no deprecation shim owed*
: The marker that says which clock the per-fire timings came from: `"unmeasured"` until a
  timing prehook arms, `"perf_counter"` on the universal path (`"perf_counter_grad_saved_only"`
  is reserved for a fallback mode).
  - Saved with the trace, where the saved `GradFnCall` stamps carry the per-fire
    `perf_counter` pair this marker discriminates. An untimed fire saves `(None, None)` and
    reads a `None` duration.

**Trace.checkpoint_invocation_witness** — *unstable — no deprecation shim owed*
: The projected checkpoint-invocation summary: what the capture saw of activation
  checkpointing, with a completeness verdict scoped to that evidence.
  - Contents: token count, per-token pack counts, unpack window evidence and BACKWARD-DERIVED
    site-key candidates, degrade flags, and the evidence-scoped completeness verdict.
  - Degrade flags: `classifier_unavailable`, `patch_unavailable`, `exotic_subclass`,
    `unmatched_backward_warn`, `reentrant_node_discovered`, `unwitnessed_checkpoint_enter`.
    Any flag withdraws the affirmative no-checkpoint claim.
  - Tokens are minted only for classified non-reentrant `_checkpoint_hook` enters on the armed
    owner thread outside any engine invocation; reentrant checkpointing is token-free and
    affirmatively sentinel-flagged.
  - Saved with the trace and validated against a closed vocabulary on load.
  - A typed checkpoint-ambiguity refusal is not yet shipped.

**trace.grad_fn_site_summary** — *unstable — no deprecation shim owed*
: The grouped-backward floor: a read-only per-`site_key` rollup of GradFn/GradFnCall facts
  (labels, fire counts, pass coverage, and live per-fire timing totals when evidence exists).
  - Grad-fns without a forward op aggregate under the `None` key; op-backed grad-fns on a
    keyless legacy artifact refuse typed `site_key_unavailable`.
  - Accessor-level only: no persisted fields.

**BackwardPass.aten_ops / DispatchedOp** — *unstable — no deprecation shim owed*
: The ATen calls one backward pass dispatched, recorded while the pass ran through a capture
  that requested a backward (`backward_ready=True`, `save_grads=` or `default_grad=`). Each
  `DispatchedOp` is value-free.
  - `DispatchedOp` fields: `operator` (`"aten.mm.default"`), `input_shapes`/`input_dtypes`
    (one entry per positional argument, `None` for non-tensors), `output_shapes`/
    `output_dtypes`, `grad_fn_call_ref` (`(grad_fn_object_id, call_index, pass_index)` of the
    GradFn call it ran under, or `None`), `family` (`"matmul"`, `"convolution"`,
    `"zero_arithmetic"`, `"uncounted"`) and `flops` (fma2 for counted calls, `0` for layout
    calls, `None` outside the counted families).
  - Session-time: `None` on a loaded trace or an implicit pass.
  - Calls made inside TorchLens's own GradFn posthooks are not recorded; gradient-saving
    copies from its tensor hooks appear as zero-arithmetic calls.

**BackwardStatus.executed_flops / executed_flops_per_pass / uncounted_op_counts / convention** — *unstable — no deprecation shim owed*
: In the `backward_observed` state, `tl.report.backward_status(trace)` counts the backward that
  ran over each pass's `aten_ops`.
  - `executed_flops`: exact for the matmul and convolution families at fma2 (`convention`,
    FlopCounterMode's convention), or `None` when a pass has no record on this object.
  - `executed_flops_per_pass`: pairs each pass index with its count; a `create_graph=True`
    gradient and the backward through it are separate passes.
  - `uncounted_op_counts`: every recorded op kind outside the counted families with its call
    count, never zeroed or estimated.
  - Other states: `grad_disabled` still reads `0` and `grad_enabled_unobserved` still reads
    unknown; the hypothetical `backward_estimate` is unchanged.

**BackwardPass.flops_attribution / BackwardFlopsAttribution / BackwardFlopsRow** — *unstable — no deprecation shim owed*
: One backward pass's executed FLOPs, charged to the GradFn call each counted call ran under
  and to the forward op that node differentiates. `None` when the pass has no census on this
  object; a pass that raised has `executed_flops` `None`.
  - `BackwardFlopsAttribution` fields: `pass_number`, `rows`, the `unattributed_flops` bucket
    with every unattributed call named in `unattributed_ops` (the seed gradient, nodes no
    capture walked), `uncounted_ops`, `recompute` (`"labeled"`, `"none_observed"`,
    `"unknown"`), `other_trace_ops` (calls on another capture's nodes in a shared backward,
    left to it) and `source`. Rows plus bucket equal `executed_flops` exactly.
  - `BackwardFlopsRow` (one per GradFn call): `pass_number` (formerly `pass_index`),
    `grad_fn_label`, `call_index`, `op_label` (or `None` for a node with no forward op),
    `flops`, `recompute_flops`.
  - `flops_by_op()` sums rows per forward op.

**Persisted backward FLOPs record (`BackwardPass`)** — *unstable — no deprecation shim owed*
: The pass's counted executed backward and its attribution, saved with the trace as a closed
  record (`convention`, `executed_flops`, `uncounted_ops`, `rows`, `unattributed`; ints,
  `None` and short strings only), so a loaded trace reads the counted answer through
  `flops_attribution` and `tl.report.backward_status`.
  - The record is internal, not a public read; it is `None` on a live pass.
  - A malformed record refuses at load with `artifact_executed_flops_record_invalid`.

**BackwardStatus.flops_attributions / unattributed_flops / recompute_flops** — *unstable — no deprecation shim owed*
: Three fields of `tl.report.backward_status(trace)` in the `backward_observed` state; empty or
  `None` in the other two states.
  - Per counted pass: one `BackwardFlopsAttribution` (the record's constructor keyword is
    `flops_attributions=`).
  - Unattributed: the counted work that ran under no recorded node.
  - Recompute: the part that replayed forward kernels under activation checkpointing (`None`
    when the evidence cannot separate replay from backward, as in a `create_graph` pass that
    checkpoints).

**DispatchedOp.grad_enabled** — *unstable — no deprecation shim owed*
: Whether autograd grad mode was on when the call dispatched during a
  recorded backward. Backward kernels run with grad mode off unless the pass
  builds a graph; checkpoint replays run with it on, which is how a replayed
  forward kernel is labeled recompute.

**CostTreeRow forward/backward/total columns; CostTree.backward_state / total_backward_flops / total_forward_mac_flops / column_conventions** — *unstable — no deprecation shim owed*
: The forward and backward FLOPs columns of `tl.report.cost_tree(trace)`.
  - Self columns: `self_forward_mac_flops` (2 x true MACs), `self_backward_flops` (executed
    backward, matmul and convolution families, fma2) and `self_forward_backward_flops` (their
    sum, one convention).
  - Subtree twins (not additive): `CostTreeRow.subtree_forward_mac_flops`,
    `CostTreeRow.subtree_backward_flops` and `CostTreeRow.subtree_forward_backward_flops`
    (constructor keywords `subtree_forward_mac_flops=`, `subtree_backward_flops=`,
    `subtree_forward_backward_flops=`).
  - Remainder rows: backward work on nodes with no forward op and unattributed calls get their
    own rows, so the self backward column sums to `total_backward_flops`.
  - Without a counted backward the cells are `None` and `backward_state` names the state;
    never 0.
  - `column_conventions` names each FLOPs column's convention.
  - Siblings fold only when their MAC and backward cells agree too.

**until= (run kwarg) / report.truncated / report.stopped_at** — *unstable — no deprecation shim owed*
: `trace.run_transaction(inputs=..., until=...)` executes only the dependency closure (or
  disclosed sequential prefix) needed to reach the named sites.
  - Truncation is a RESULT/REPORT term, never a capture outcome. The report's flat surface is
    `report.truncated` (bool) and `report.stopped_at` (the stop-frontier site label), with the
    full `RunTruncation` disclosure on `report.truncation` (regime, requested sites,
    executed/skipped counts, skipped-site digest).
  - Skipped sites are "not-run", never "passed".
  - With `fast=True` on a live trace the native forward is cut right after the `until=` site,
    which must be a site the capture saved (a module address, `"address:call"`, or a saved op
    label; otherwise `run_capability_unavailable` at stage `fast_until_site_unsaved`).
  - In that fast run the prefix keeps the fast path's per-call guards and comes back
    graph-connected under the default save; the regime is `live_stop_after`; the output and
    the later saved sites read `None` for that run (the next full run refreshes them); and the
    verdict is `unverifiable` because the whole-forward call fingerprint cannot be compared.
  - The fast run refreshes its result trace: `trace.run_transaction(inputs=..., fast=True)` on
    a live trace returns a new Trace and leaves the source untouched. So that verdict marks the
    result trace as every truncated run product is marked (monotonic: save, export or validate
    it before training through `until=`).
  - A loaded artifact's fast tier still refuses `run_fast_until_unsupported`.
  - Full contract: [the runnable tlspec contract](../../reference/runnable_tlspec_contract.md).

**Trace.sites()** — *unstable — no deprecation shim owed*
: The tabular view of a trace's structural sites: one row per distinct `site_key` in
  first-occurrence execution order, aggregating the ops that share the site (`module_site`,
  `layer_type`, `output_slot`, `call_ordinal`, `n_ops`, labels, passes, shapes).
  - Requires the `tabular` extra (pandas).
  - Legacy artifacts without site keys refuse `site_key_unavailable`, never a silently empty
    table.

**Trace.bill_of_materials()** — *unstable — no deprecation shim owed*
: The inventory of what a trace actually contains: a nested, JSON-friendly
  dict of sections (`capture`, `graph`, `parameters`, `buffers`,
  `activations`, `backward`, `annotations`), every figure read from fields
  the trace already carries. The rollup mints no new claims.

**logit_lens (torchlens._semantic)** — *unstable — no deprecation shim owed*
: `torchlens._semantic.logit_lens(trace)` projects each block's residual-stream facet through
  the model's own final norm + unembedding, returning a `LogitLens` (per-layer logits,
  `summary()`, `top_tokens()`).
  - Validation: the reconstructed lens is validated against the captured logits before use and
    refuses `LogitLensError` when the head cannot be represented. Pass `lens=` for a tuned lens
    or `verify=False` to trust the reconstruction explicitly.
  - Vision fallback: with no language-model head, the default lens falls back to a vision head
    recipe, validated the same way: `classification_head` (final norm plus `classifier` over
    the `[CLS]` token; `lens_source="classifier_head"`) or CLIP's `contrastive_image_head`
    (per-layer scores against the captured text features of the label prompts;
    `lens_source="contrastive_head"`).
  - User lens identity: a user lens may carry `model_identity` (an extraction manifest's
    identity record). A differing measured identity refuses `logit_lens_identity_mismatch`
    unless `verify=False`; a lens with no comparable identity warns
    `logit_lens_identity_unrecorded` when its last-layer decode misses the captured logits.
  - Streaming sibling: `logit_lens_predictions(trace, ...)` keeps only per-layer top-k,
    `logsumexp` and requested-token values. Its keyword-only `mask_input=True` bans the model
    input's own token ids before those reductions (the trace's first input must be saved
    integer token ids, else `logit_lens_mask_input_unavailable`).
  - Related: trajectory statistics over many batches are `tl.mechinterp.measure_trajectory`.
    See [facets](../../facets.md).

**render_logit_lens_heatmap (torchlens.transformer)** — *unstable — no deprecation shim owed*
: `torchlens.viz.logit_lens(result, path)` draws a `LogitLens` as a layers-by-positions heatmap
  and returns a `SavedFigure`.
  - Cells: one row per projected block, one column per input position; each cell prints that
    layer's top-1 next-token prediction and is colored on a fixed 0-to-1 probability scale (the
    top-1 probability, or with `value="target_prob", target_token_id=...` a tracked token's).
  - Footer: states the lens source, whether it was validated, and that every row is projected
    through the final norm and head. Only a validated lens lets it say that the last row
    reproduces the model's own output scores.
  - `title=` prefixes the figure title with a subject such as the model's name.
  - Lenses wider than 48 positions refuse `tv_record_invalid` until `positions=` names the
    columns; nothing is cropped silently.

**kv-group receipts (torchlens.transformer)** — *unstable — no deprecation shim owed*
: `tviz.receipt_from_patch_grid(grid, ..., kv_groups=tviz.GqaInfo(...))` builds a
  `CausalReceipt` whose cells are grouped-query KV groups, for grids over a key or value site.
  - The receipt's `kv_groups` field holds the grouping, `unit` reads `"kv group"`, and
    `sharing_lines()` names each group's query heads.
  - `render_receipt_grid` labels the columns by group and prints that no cell is a
    per-query-head effect.
  - `GqaInfo` gains `query_heads(group)` and `group_header(group)`.
  - Refusals: a grid whose head axis exceeds the group count refuses `tv_record_invalid`;
    mixing kv-group and query-head rows refuses `tv_receipt_invalid`; `Annotation.from_receipt`
    refuses a kv-group receipt `tv_annotation_invalid`.
  - A grid from `mechinterp.patch_query_heads_grid` is an ordinary query-head receipt: its
    `effect_site` receipt becomes the receipt's `effect_source`, printed as "effect source:
    ..." on the grid figure, and passing `kv_groups=` for it refuses `tv_receipt_invalid`.

**Receipt cells and patch-grid baselines (torchlens.transformer, torchlens.mechinterp)** — *unstable — no deprecation shim owed*
: How patch-grid results become per-cell effects measured from one baseline run.
  - A `mechinterp.PatchGrid` from `mechinterp.patch_grid`, its presets or
    `mechinterp.patch_query_heads_grid` keeps `values` as the metric of each patched run
    (`to_pandas` names the column `metric`) and records the unpatched baselines as
    `receipts["clean_metric"]` and `receipts["corrupted_metric"]`.
  - `tviz.receipt_from_patch_grid` turns a row into effects: patched metric minus
    `corrupted_metric`, the run every cell patches into. A cell that changes nothing is exactly
    0, the same quantity as the self-patch negative control. A grid without `corrupted_metric`
    refuses `tv_receipt_invalid`.
  - The `CausalReceipt` carries the baseline as `baseline_metric` (`None` when a producer did
    not record one) and names its run in `baseline_run` ("unpatched corrupted run" for a
    patch-grid receipt); its `effects`, `joint_effect` and controls are all deltas from that
    run.
  - The default `intervention` wording is "clean activation of one head patched into the
    corrupted run" ("one kv group" with `kv_groups=`); a row's `fires` is its own cell count
    when the grid's ledger shows one fire per cell.
  - `tl.viz.ablation_grid` titles the figure "measured head effects on METRIC", labels the
    colorbar "change in METRIC", labels the x axis "query head" (or "kv group"), prints "each
    cell: change in METRIC from the unpatched corrupted run (B)" and says in the footer that
    single-head effects need not be additive. It refuses rows measured from different
    baselines (`tv_receipt_invalid`) and reserves the footer's height below the matrix and its
    tick labels.

**Summary columns (`summary(columns=...)`)** — *unstable — no deprecation shim owed*
: The column registry of `tl.summary` / `trace.summary`. See [summary](../../reference/summary.md).
  - Columns: `name`, `input`, `output`, `kernel_size`, `groups`, `params`, `params_pct`,
    `params_trainable`, `flops`, `flops_pct`, `macs`, `train`, `mode`, `passes`, `evidence`.
  - `input` lists the shapes of the distinct tensors entering a row.
  - `output` lists every member shape of the row's output in return order. Past 60 characters
    a run of identical shapes prints once as `shape ×N`, and a list still too long ends in
    `... +K more`; `input` cells compact the same way.
  - `kernel_size` / `groups` read the row's own conv or pooling op.
  - `params_trainable` is the per-path trainable count (`train` stays the yes/no/partial
    `requires_grad` tri-state).
  - `mode` is `module.training` as captured (`train` / `eval` / `mixed`).
  - `flops_pct` is the row's share of the known forward FLOPs.
  - A `+name` delta inserts at the column's registry position.
  - A FLOP cell whose count includes a full-square attention op ends in `^` with a footer note.

**SummaryReport.table (`trace.summary(view="stats").table`)** — *unstable — no deprecation shim owed*
: The typed observation table (`StatsTable`) behind a `view="stats"` report. The report's text
  is one rendering of it.
  - `rows`: `StatsTableRow` records with `state`, `value_stats`, `reason`, `parent_mark`.
  - Also `ok_rows`, `sort_rows(by=..., descending=...)` (sorts on a sampled column are
    disclosed) and `fold_same_as_parent()` (opt-in fold of `= parent` rows with an exact
    count).
  - A report built with any other view carries no table and refuses
    `summary_table_unavailable`.

**Summary Params footer extras** — *unstable — no deprecation shim owed*
: Extras on the summary's Params footer line.
  - `size`: unique parameter storage, dtype true.
  - A `pruned:` note with raw and effective nonzero counts on models reparametrized by
    `torch.nn.utils.prune` (`<name>_orig` + `<name>_mask`).
  - `optimizer state N params, X MB` when the capture was given
    `CaptureOptions(optimizer=...)` (read from the optimizer's live state; omitted on a loaded
    trace).
  - Raw values: `SummaryTotals.params_bytes`, `.params_effective`, `.optimizer_state_params`
    and `.optimizer_state_bytes`. `params_bytes` is the unique parameters' storage;
    `params_effective` is set only on a pruned model (raw total minus mask zeros);
    `optimizer_state_params` and `optimizer_state_bytes` are set only when the capture was
    given an optimizer. Each is `None` otherwise.

**ParamFacts.nbytes / ParamFacts.pruned (`tl.report.PrunedParamFacts`)** — *unstable — no deprecation shim owed*
: FactCore parameter facts (schema version 2): `bytes` is the unique
  parameters' storage; `pruned` is `None` unless prune reparametrizations
  exist, else a `PrunedParamFacts(paths, raw, masked)` naming the pruned
  tensors, their raw element count, and the mask zeros (`masked=None` when
  the capture retained no mask values).

**FlopsReport.attention_full_square_ops / ProfileReport.attention_full_square_ops** — *unstable — no deprecation shim owed*
: The count (the `attention_full_square_ops` field) of `scaled_dot_product_attention` ops in the
  forward FLOP total of a `flops` or a `trace.profile()`.
  - When non-zero, the report or the profile footer prints that attention FLOPs count the full
    query x key square and masks / `is_causal` are not discounted.
  - Every attention FLOP line carries the same note: the `tl.summary` FLOP cells (marked, with
    the note as a legend), the `explain` FLOPs line, and graph labels, where an attention
    node's `flops` label field or `flops` overlay row is followed by the note.

**`decode_style_unknown`** — *unstable — no deprecation shim owed*
: The typed refusal `tl.errors.DecodeStyleUnknownError` (an `InvalidArgumentError` subclass)
  for a `Trace.decode_output(style=...)` that no built-in or registered custom output detector
  resolves, with a did-you-mean over the built-in styles (`classification`, `imagenet`,
  `imagenet1k`, `hf_text`).
  - The same code refuses `style="hf_text"` when no rank-2 or rank-3 logits are retained.
  - `style="hf_text"` without a `tokenizer=` is not a refusal: the rows carry the argmax
    `token_ids` and `text` is `None` (the trace was captured without a tokenizer).
  - Output decoding is read-time only: the capture runs the automatic detector pass with no
    option, and `CaptureOptions` has no decode field.
  - `CaptureOptions(output_style=...)`, `CaptureOptions(output_head=...)` and the same
    keywords on `tl.trace` refuse naming `trace.decode_output(...)`.

**`decode_head_unknown`** — *unstable — no deprecation shim owed*
: `Trace.decode_output(head=...)` named a head that is not one of the trace's output heads
  (the output container paths such as `"logits"`, `"0"`, `"0.logits"`); the refusal lists the
  available heads in `fields["available_heads"]`.

**`decode_head_ambiguous`** — *unstable — no deprecation shim owed*
: `Trace.decode_output(style=...)` found several qualifying output heads and none was named;
  pass `head=` one of the heads the refusal lists.

**tl.bridge.transformers.register_attention(model=None)** — *unstable — no deprecation shim owed*
: Registers TorchLens as a Hugging Face attention backend and returns the key, for
  `from_pretrained(..., attn_implementation="torchlens")`.
  - One call puts a `"torchlens"` attention function in transformers' `AttentionInterface`
    AND its matching mask function in `AttentionMaskInterface` (the boolean `sdpa_mask` with
    the causal skip off), so the model always receives an explicit causal and padding mask.
  - Registering the attention function alone is the mask trap: transformers then builds no
    mask and padding leaks into attention.
  - Never registers over a built-in key; a foreign `"torchlens"` registration refuses
    `hf_attention_key_taken`.
  - With `model=`, also calls `model.set_attn_implementation("torchlens")`.
  - Memory cost: the backend materializes the full `(batch, heads, q_len, kv_len)` probability
    matrix in every attention call, unlike fused SDPA or flash kernels.

**tl.bridge.transformers.record_attention(model, *, check_sdpa=False)** — *unstable — no deprecation shim owed*
: Context manager yielding an `AttentionRecording`: every attention call of a model on the
  `"torchlens"` backend records its per-head probabilities, keyed by the attention module's
  qualified name, without a TorchLens trace.
  - The recorded tensor is exactly the one the backend multiplied by the values (and returns
    as `output_attentions`).
  - Refuses `hf_attention_backend_not_selected` for a model on another backend.
  - `check_sdpa=True` also runs `scaled_dot_product_attention` on each call and refuses
    `hf_attention_sdpa_mismatch` beyond the semantic SDPA reconstruction tolerance, doubling
    the attention cost. Calls with dropout, softcapping or learned sink logits (`s_aux`) are
    left unchecked and say so; emergent attention sinks in trained heads are measured by the
    checks kit's sink probe, see docs/reference/checks_kit.md.
  - Retained memory is `batch * heads * q_len * kv_len * itemsize` bytes per call: for a
    Llama-2-7B shape (32 layers, 32 heads) at 4096 tokens in bfloat16, 1 GiB per layer and
    32 GiB per sequence.

**AttentionRecording / AttentionCall (`tl.bridge.transformers`)** — *unstable — no deprecation shim owed*
: `AttentionRecording` is a read-only mapping from attention-module name to probabilities
  `(batch, heads, q_len, kv_len)`.
  - `recording[name]` serves a module called once and refuses `hf_attention_multiple_calls`
    otherwise; `calls_for(name)` returns every call in order.
  - `.calls` lists all `AttentionCall` records (`module_name`, `probs`, `num_key_value_heads`,
    `mask_kind`, `sdpa_checked`).
  - `.nbytes` is the retained total.

**Report (`tl.report.Report`)** — *unstable — no deprecation shim owed*
: The one typed result of every report-shaped `tl.debug` helper.
  - Lead: `report_type`, `subject` and `status`.
  - Outputs: `to_dict()` (a plain, JSON-friendly payload; tensors and graph records
    summarized, never inlined), `to_pandas()` (the table), and `str(report)` (a bounded card:
    lead line, at most five findings, the exact remainder, the exits).
  - Subclasses: the typed results (`HealthReport`, `DeterminismReport`, `NonfiniteReport`,
    `FindNanReport`, `ParamHealthReport`, ...). Those with a house-voice `__repr__`
    (`HealthReport`, `ParamHealthReport`) keep printing it and serve the card as
    `report.card()`.
  - Table helpers: `compare`, `compare_params`, `dead_neurons`, `grad_health`, `hot_path` and
    `recompute_candidates` return a tabular `Report` whose `.to_pandas()` is the DataFrame
    they used to return (`attrs` included); the report has a table and never is one.
  - Source lines: every finding that names an op names its user source line too. The
    `source_line` field or column holds `"<file>:<line>"` with the full path, and the card
    prints ` at <file>:<line>` relative to the working directory when the file is under it.

**Trace.profile(group_by="source_line") / ProfileReport.group_by** — *unstable — no deprecation shim owed*
: `group_by="source_line"` folds the op rows of `trace.profile()` into one row per user source
  line (`"<file>:<line>"`, the op's recorded call site). Only with `level="op"`.
  - It sums activation bytes, FLOPs and instrumented time; boundary rows fold into one
    `<inputs and outputs>` row, so every additive column sums to the ungrouped op view.
  - `ProfileReport.group_by` records the key (`None` for one row per op).

**tl.summary("<hub model id>", input_shape=...)** — *unstable — no deprecation shim owed*
: A string model is a Hugging Face Hub id (or local config directory):
  `tl.summary` fetches the config only, builds the declared architecture on
  the meta device (no weights downloaded or allocated, `trust_remote_code`
  never set) and takes the weights-free summary path, so it needs
  `input_shape=` or meta inputs. Requires `transformers`.

**Trace.cost.forward_peak_memory / Trace.cost.backward_peak_memory**
: The largest memory growth above a bracket's starting baseline, including memory freed before
  the bracket ends.
  - CPU measures only under `CaptureOptions(measure_host_peak_memory=True)`: it reads the
    process resident high-water mark, reset at each bracket on Linux (the reset also resets a
    caller's own `VmHWM` and `ru_maxrss` readings). Without the opt-in (the default) a CPU
    peak is `None` and its `*_peak_memory_reason` names the opt-in.
  - CUDA reads the device allocator peak; MPS the allocator delta.
  - `backward_peak_memory` is the largest peak above a pass's starting baseline over all
    backward passes (CUDA: the growth of the device high-water mark).

**measure_host_peak_memory (CaptureOptions)** — *unstable — no deprecation shim owed*
: Opt-in (default `False`) for the CPU `Trace.cost.forward_peak_memory` and
  `Trace.cost.backward_peak_memory`.
  - Measuring resets the process-wide resident high-water mark at the forward bracket and at
    each CPU backward pass, which also resets the caller's own `VmHWM` and `ru_maxrss` readings
    and cannot be undone. The default writes nothing and reports CPU peaks as `None` with a
    reason.
  - Mirrored on the trace as `Trace.measure_host_peak_memory`, a session-time knob (not
    persisted; load restores `False`); reruns and refreshes of a trace inherit it.
  - CUDA peaks and the MPS allocator delta ignore it.

**Trace.cost.forward_peak_memory_reason** — *unstable — no deprecation shim owed*
: Why `Trace.cost.forward_peak_memory` is `None`. It is itself `None` when a value was
  measured, and on artifacts that predate the field.
  - A CPU capture without `CaptureOptions(measure_host_peak_memory=True)` records `None` with
    a reason naming that opt-in.
  - A CPU forward that measured zero (resident high-water growth, end-of-forward RSS delta
    and, when opted in, the Python-allocation peak all 0) is recorded as `None` with a reason
    naming `CaptureOptions(measure_python_peak_memory=True)`.
  - Saved with the trace; a loaded reason beside a number refuses
    `artifact_forward_peak_memory_invalid`.
  - CUDA and MPS keep their numeric semantics.

**Trace.backward_peak_memory_reason** — *unstable — no deprecation shim owed*
: Why `Trace.cost.backward_peak_memory` is `None`. It is itself `None` when a value was
  measured, before any backward, and on artifacts that predate the field.
  - A CPU backward on a capture without `CaptureOptions(measure_host_peak_memory=True)`
    records `None` with a reason naming that opt-in.
  - Saved with the trace; a loaded reason that is not a non-empty string, or sits beside a
    number, refuses `artifact_backward_peak_memory_invalid`.
  - CUDA and MPS keep their numeric semantics.

**Trace.cost.total_backward_memory**
: Bytes of the gradient payloads the trace retains, counted once per storage: a payload that is a
  view charges its whole base storage, an expanded gradient its smaller storage.

**trace.ops counting (`len(trace.ops)`)** — *unstable — no deprecation shim owed*
: `len`, iteration, integer indexing, `keys()`/`values()`/`items()` and `to_pandas()` of
  `trace.ops` cover the counted ops (no input, output or buffer rows).
  - Label lookup still resolves every row: `trace.ops["input_1"]` returns the input row and
    `"input_1" in trace.ops` is true although the label is not in `trace.ops.keys()`.
  - The every-row sequence (inputs, outputs and buffers included) is `trace.ops`;
    `len(trace)`, `trace[int]` and `trace[plane]` index it.
  - Only `trace.ops` is the counted view: `input_ops`, `output_ops`, `saved_ops`,
    `saved_grad_ops`, `compute_ops` and `ops_with_params` list exactly their rows, boundary
    rows included.

**`inv_rms` facet (Hugging Face `Gemma2RMSNorm`)** — *unstable — no deprecation shim owed*
: The norm's single `rsqrt` op output, `1 / sqrt(mean(x**2) + eps)`, the per-position factor
  that multiplies the input before the `(1 + weight)` gain.
  - `tl.facet("inv_rms")` names it in every Gemma-2 norm.
  - An attribution graph holds it with
    `tl.attribution.build_graph(..., frozen=[tl.facet("pattern"), tl.facet("inv_rms")])`,
    which also holds every Gemma-2 attention pattern (`Gemma2Attention` serves the attention
    facets; `pattern` is the eager softmax after the score softcap).

**`AttributionGraph.to_neuronpedia` (export in Neuronpedia's attribution-graph schema)** — *unstable — no deprecation shim owed*
: `graph.to_neuronpedia(*, slug, model_id, tokenizer, prompt=None, node_influence_coverage=0.8, edge_influence_coverage=0.98, path=None)`
  returns the graph, pruned by circuit-tracer's influence rule, as a dict in Neuronpedia's
  published attribution-graph schema (and writes it to `path` when given).
  - Arguments: `slug` is the graph's name on Neuronpedia, `model_id` Neuronpedia's model id,
    `tokenizer` anything with `decode(ids) -> str`, `prompt` the prompt string (`None` decodes
    the input tokens), and `node_influence_coverage` and `edge_influence_coverage` the shares
    of total influence kept.
  - Refusal codes: `attribution_graph_export_argument_invalid`,
    `attribution_graph_export_unsupported`.

**`trace.draw(color_by=graph)` with an `AttributionGraph`** — *unstable — no deprecation shim owed*
: `trace.draw(color_by=graph)` paints an attribution graph onto the trace it was built from:
  each site holding feature or error nodes gets the `abs_sum` total of its nodes' influence on
  the logits over every position (circuit-tracer's rule). A site-to-value mapping paints any
  other quantity, feature activations included. Refusal code
  `attribution_overlay_graph_mismatch`.

**`tl.attribution.upload_to_neuronpedia` (publish an exported graph)** — *unstable — no deprecation shim owed*
: `upload_to_neuronpedia(exported, *, api_key)` publishes the dict `to_neuronpedia` returned
  (or the file it wrote) on neuronpedia.org under the account that owns `api_key`, through
  Neuronpedia's documented two-step upload, and returns the graph page's URL.
  - Standard library only; the key goes in the `x-api-key` header and never into a message.
  - Checked before any request: the export's structure (`attribution_graph_contract_invalid`),
    the `slug` and `model_id` characters and the file size
    (`neuronpedia_upload_argument_invalid`).
  - HTTP failures raise `neuronpedia_upload_http_error`; answers without their fields raise
    `neuronpedia_upload_response_invalid`.

## Capture door and options: current spellings

**Capture keywords**
: The keyword spellings that `CaptureOptions` and the capture doors accept for the capture
  knobs. The saved `Trace` keeps its stored field names in this release.
  - `seed=`: sets `CaptureOptions.seed`.
  - `save_args=`: sets `CaptureOptions.save_args`.
  - `save_source_context=`: sets `CaptureOptions.save_source_context`.
  - `save_rng=`: sets `CaptureOptions.save_rng`.
  - `save_transformed_output=`: sets `CaptureOptions.save_transformed_output`.
  - `input_transform=`: sets `CaptureOptions.input_transform`.

**Recording keywords**
: The keywords that `tl.record(...)` and `Recorder(...)` take.
  - `episode=`: accepted only to refuse it with a typed error (`record_episode_unsupported`),
    which points you to `tl.trace(model, x, episode=...)`.
  - `narrate=`: a `tl.options.NarrationOptions`. Methods: `NarrationOptions.as_dict()`,
    `NarrationOptions.from_values()`, `NarrationOptions.is_field_explicit()`.
  - `NarrationOptions.sites` (`sites=`): scopes the narration: `True` for every event,
    `"modules"` for structure lines only, or a live selector. `trace.narrate(select=...)`
    filters after the run.
  - `NarrationOptions.sink` receives the narrated lines and `NarrationOptions.max_lines` caps
    them.
  - `NarrationOptions.on_error_only`: narrate only when the forward raises.
  - `NarrationOptions.tail_on_error`: how many of the last selected events the sink receives
    when the forward raises.
  - `save_raw_grads` is a `CaptureOptions` field; on `tl.record` pass it as
    `capture=CaptureOptions(save_raw_grads=...)`.

**StorageOptions**
: `tl.options.StorageOptions` groups the streaming-save knobs. It is also reachable as
  `torchlens._user_funcs.StorageOptions`.
  - Fields: `StorageOptions.path` (`path=`), `StorageOptions.retain_in_memory`,
    `StorageOptions.on_save` (keyword `on_save=`), `StorageOptions.include_custom_attributes`,
    `StorageOptions.include_buffer_values`.
  - Methods: `StorageOptions.as_dict()`, `StorageOptions.from_values()`,
    `StorageOptions.is_field_explicit()`.

**CaptureOutcome flags**
: The flags and counts that say how a capture ended.
  - Fields: `CaptureOutcome.is_partial`, `CaptureOutcome.is_derived` (`is_derived=`),
    `CaptureOutcome.is_recovered` (`is_recovered=`), `CaptureOutcome.num_committed_ops`,
    `CaptureOutcome.note`.
  - `CaptureStatus.STOPPED` (value `"stopped"`): the status of a capture stopped by
    `stop_after=` or `tl.recording.stop()`.

**CaptureOutcome failure and boundary fields**
: What a capture outcome records about a failure or a stop boundary.
  - `CaptureOutcome.error_type`: the qualified exception type name of a failed capture.
  - `CaptureOutcome.boundary_kind` and `CaptureOutcome.boundary_label`: the event kind and op
    label of the stop boundary.
  - `CaptureOutcome.frontier_labels`: the labels on the halt frontier.
  - Phases: a failed capture names the phase it failed in, one of `CapturePhase.FORWARD`,
    `CapturePhase.FINALIZE`, `CapturePhase.POSTPROCESS` or `CapturePhase.TEARDOWN`.
  - Origins: it also names its diagnostic origin, one of `FailureOrigin.USER_OP`,
    `FailureOrigin.TORCHLENS` or `FailureOrigin.INTERRUPT`.

**Recording state**
: The status, counts and controls of a live `Recording`, plus how a recorder runs a pass.
  - Status: `Recording.is_failed`, `Recording.is_stopped`, `Recording.stop_reason`,
    `Recording.stops_by_pass`, `Recording.is_recovered`.
  - Counts: `Recording.num_ops`, `Recording.num_committed_ops`, `Recording.num_passes`.
  - Access: `Recording.records_in_pass(pass_index)`, `Recording.lazy_trace`,
    `Recording.lean_omissions` (provisional spelling), `Recording.backward(loss)`.
  - `Recorder.__call__` runs one pass (`recorder(x)`) and `Recorder.backward(loss)` records its
    backward.
  - `ActivationRecord.recorded_time` and `GradRecord.recorded_time` (`recorded_time=`) stamp
    when a record was taken.

**Recording types**
: The record, dry-run and stop types of the recording namespace, the per-event save switches on a
  capture spec, and the read protocol that traces and recordings share.
  - `tl.recording.GradRecord`. Fields: `GradRecord.ctx`, `GradRecord.ram_payload`,
    `GradRecord.transformed_ram_payload`, `GradRecord.disk_payload`,
    `GradRecord.transformed_disk_payload`.
  - `tl.recording.DryRun`. Fields: `DryRun.events`, `DryRun.contexts`, `DryRun.decisions`,
    `DryRun.predicate_failures`. Methods: `DryRun.filter(...)`, `DryRun.print_tree()`,
    `DryRun.timeline_html()`, `DryRun.to_pandas()`.
  - Stopping: `tl.recording.stop()` and `tl.recording.StopSignal`.
  - Save switches: `CaptureSpec.save_out` keeps the tensor payload of an event,
    `CaptureSpec.save_metadata` keeps its non-payload metadata, `CaptureSpec.save_grad` keeps a
    payload attached to autograd, and `CaptureSpec.save_mode` picks the retention mode (`copy`,
    `reference`, `view` or `cpu_async`).
  - `GradRecordContext`: the predicate input for a backward gradient event.
    `GradRecordContext.kind` names the record kind. Fields:
    `GradRecordContext.grad_fn_class_name`, `GradRecordContext.backward_call_index`,
    `GradRecordContext.grad_kind` (`grad_input` or `grad_output`),
    `GradRecordContext.grad_input_index`, `GradRecordContext.grad_output_index`,
    `GradRecordContext.module_stack`, `GradRecordContext.has_forward_op`,
    `GradRecordContext.has_op`, `GradRecordContext.order`, `GradRecordContext.event_index`,
    `GradRecordContext.tensor_device`.
  - `GradRecordContext.effective_label` (read-only): the joined forward label when there is
    one, else the autograd-node label.
  - `tl.types.TraceLike`: the read protocol shared by `Trace` and `Recording`. Members:
    `TraceLike.op_events`, `TraceLike.backward_events`, `TraceLike.event_stream`,
    `TraceLike.sibling_events`, `TraceLike.activation_payloads_by_raw_label`.
  - Lookups: `TraceLike.activation_by_raw_label(raw_label)`,
    `TraceLike.activations_by_pass(pass_number)`, `TraceLike.activations_by_address(address)`.
    The pass keyword, spelled as above and in `activation_by_raw_label(pass_number=)`, replaces
    the former `pass_index`.

**Backend registry**
: The functions that look up, add, remove and resolve capture backends.
  - Functions: `tl.backends.spec(name)`, `tl.backends.specs()`, `tl.backends.register(spec)`,
    `tl.backends.unregister(name)`, `tl.backends.resolve(...)`.
  - Each `BackendSpec` carries `maturity` and the `protocol` factory (see "Backends: the
    backend protocol, maturity and conformance").
  - `BackendCapabilities.has_streaming`: the streaming-save capability flag (flag name
    `has_streaming`).
  - `CaptureBackend.cleanup_stopped_forward_session` tears down a forward stopped early.

**Extraction and preprocessing names**
: Current spellings of the row counts, preprocessing and transform names used in extraction.
  - Counts: `BatchEnvelope.num_rows` and `RaggedBatch.num_rows` (`num_rows=`),
    `TokenRows.num_tokens`.
  - Preprocessing: `Resolution.preprocessing` and `SiteMatrix.shaping` (`shaping=`).
  - Transforms: `tl.extract_dataset(..., output_transform=...)` and
    `tl.neuro.plan_extraction(..., output_transform=...)` take the output transform;
    `input_transform=` preprocesses inputs.

**Admin and debugging names**
: Helpers for clearing the cache, registering observers and debugging a capture.
  - `tl.clear_cache()` (also `torchlens._user_funcs.clear_cache`) empties the capture cache.
  - `tl.observers.register_tensor_connection` registers a tensor connection.
  - `tl.debug.isolated_capture` runs a capture in isolation; its parts are
    `tl.debug.preserved_rng_state()` and `tl.debug.clone_input_tree(...)`.
  - `Trace.capture_backward` captures a backward pass for validation. It is a thin `Trace`
    method over `backends/torch/_trace_backward.py`.

### Debugging tools and report results

**DTypeRangeReport (`tl.debug.DTypeRangeReport`)** — *unstable — no deprecation shim owed*
: Coverage-bounded dtype range audit result.

**clone_input_tree (`tl.debug.clone_input_tree`)** — *unstable — no deprecation shim owed*
: Return a detached clone of an input tree for one isolated run: tensors are detached and
  cloned (keeping `requires_grad`), dicts, lists and tuples are rebuilt as new containers, and
  other leaves pass through as the same object.

**dtype_range (`tl.debug.dtype_range`)** — *unstable — no deprecation shim owed*
: Audit saved activations for numeric-range and precision hazards.

**nonfinite_origin (`tl.debug.nonfinite_origin`)** — *unstable — no deprecation shim owed*
: One-call NaN forensics.

**param_health (`tl.debug.param_health`)** — *unstable — no deprecation shim owed*
: Audit parameter/buffer space for numeric-health findings.

**preserved_rng_state (`tl.debug.preserved_rng_state`)** — *unstable — no deprecation shim owed*
: Context manager that saves the Python, NumPy, torch CPU and initialized CUDA RNG states and
  restores them when the block exits, normally or by an exception.

**ExplainReport (`tl.report.ExplainReport`)** — *unstable — no deprecation shim owed*
: The plain-language report trace.explain() returns: a str of multi-section prose.

**FlopsCrossCheck (`tl.report.FlopsCrossCheck`)** — *unstable — no deprecation shim owed*
: Side-by-side dispatch-formula evidence (never an oracle).

**MfuReport (`tl.report.MfuReport`)** — *unstable — no deprecation shim owed*
: The mode-split time-normalized MFU facts.

**OverheadReport (`tl.report.OverheadReport`)** — *unstable — no deprecation shim owed*
: Measured capture costs on the user's model and host.

**RooflineReport (`tl.report.RooflineReport`)** — *unstable — no deprecation shim owed*
: The roofline map: rows, sum/sum aggregate, coverage by reason.

**measure_clean_step (`tl.report.measure_clean_step`)** — *unstable — no deprecation shim owed*
: Measure an uninstrumented forward wall time (the MFU denominator).

**overhead (`tl.report.overhead`)** — *unstable — no deprecation shim owed*
: Measure requested capture tiers on the user's own model/host.

**profile (`tl.report.profile`)** — *unstable — no deprecation shim owed*
: Build a unified op, module, or invocation resource profile.

**DTypeRangeReport fields** — *unstable — no deprecation shim owed*
: `DTypeRangeReport.findings`, `DTypeRangeReport.num_audited_ops`,
  `DTypeRangeReport.num_ops`.

**HealthReport fields** — *unstable — no deprecation shim owed*
: `HealthReport.checks_run`, `HealthReport.findings`, `HealthReport.skipped`.

**MfuReport fields** — *unstable — no deprecation shim owed*
: The fields of the MFU report.
  - Fields: `MfuReport.denominator`, `MfuReport.denominator_scope`, `MfuReport.ideal_seconds`,
    `MfuReport.mfu`, `MfuReport.mode_terms`, `MfuReport.peaks_source`, `MfuReport.step_seconds`.

**NonfiniteReport fields** — *unstable — no deprecation shim owed*
: The fields of the non-finite value report.
  - Fields: `NonfiniteReport.amp_disclosure`, `NonfiniteReport.capture_status`,
    `NonfiniteReport.cost_tier`, `NonfiniteReport.coverage_basis`, `NonfiniteReport.defer_recipe`,
    `NonfiniteReport.has_nonfinite`, `NonfiniteReport.nonfinite_labels`,
    `NonfiniteReport.num_checked_ops`, `NonfiniteReport.num_unchecked_ops`,
    `NonfiniteReport.num_unexamined_ops`, `NonfiniteReport.origin_label`, `NonfiniteReport.phase`,
    `NonfiniteReport.scope_note`, `NonfiniteReport.tensor_role`, `NonfiniteReport.uncertainty_zone`.

**OverheadReport fields** — *unstable — no deprecation shim owed*
: `OverheadReport.num_forward_runs`, `OverheadReport.tiers`.

**ParamHealthReport fields** — *unstable — no deprecation shim owed*
: The fields of the parameter health report.
  - Fields: `ParamHealthReport.checks_run`, `ParamHealthReport.findings`,
    `ParamHealthReport.schema_version`, `ParamHealthReport.skipped`, `ParamHealthReport.to_json`.

**ProfileReport fields** — *unstable — no deprecation shim owed*
: The fields of the resource profile.
  - Fields: `ProfileReport.capture_status`, `ProfileReport.capture_verification_reason`,
    `ProfileReport.column_additivity`, `ProfileReport.evidence_labels`, `ProfileReport.frame`,
    `ProfileReport.is_capture_verified`, `ProfileReport.partition_total`, `ProfileReport.tree`,
    `ProfileReport.unknown_events`.

**RooflineReport fields** — *unstable — no deprecation shim owed*
: The fields of the roofline map.
  - Fields: `RooflineReport.aggregate_intensity`, `RooflineReport.coverage_by_reason`,
    `RooflineReport.excluded_cache_resident`, `RooflineReport.headline_compute_bound`,
    `RooflineReport.headline_memory_bound`, `RooflineReport.ridge_intensity`.

## Visuals and kits spellings
Spellings introduced by the visuals-and-kits rename (attribution, facets, the `tl.viz` and
`tl.neuro` homes, transformer views, statistics and bridges), one line each. The sections above
describe each surface in depth.

### Namespace names
| Spelling | Meaning |
| --- | --- |
| `tl.attribution.Attribution` | Container for provisional input-attribution results |
| `tl.bridge.transformers.trace_image` | Trace an image model with PIL input and resolved preprocessing |
| `tl.bridge.transformers.trace_multimodal` | Trace a Hugging Face multimodal model with dict-keyed input |
| `tl.bridge.transformers.trace_text` | Trace a Hugging Face language model with raw text input |
| `tl.errors.AttributionError` | Error raised for unsupported or invalid attribution requests |
| `tl.errors.MissingFacetError` | Raised when a declared facet needs a different capture to be available |
| `tl.experiment.GridCell` | One cell of a `SweepGrid`: coordinates, status, value, error and member name |
| `tl.experiment.SweepGrid` | The labeled result of `sweep_grid`: float64 `values` shaped by the axes (NaN for refused or failed cells) beside the Bundle |
| `tl.experiment.sweep_grid` | Sweep the product of named axes (sites, at most one axis of edits); one member per cell, baseline first |
| `tl.facets.FacetFunction` | Registered facet recipe metadata |
| `tl.facets.MissingGrad` | Sentinel for a known facet gradient that is not currently saved |
| `tl.facets.child_output_spec` | Recipe helper: a child module's output as a `FacetSpec` (`module, child_name, recipe_id`), or an absence reason |
| `tl.facets.config_value` | Recipe helper: the first available attribute of a `Module` record, checking record attributes and the `custom_attributes` snapshot, or an absence reason |
| `tl.facets.describe` | List registered facet function names and declared facets for a class name |
| `tl.facets.enable_transformer_lens_aliases` | Enable or disable opt-in TransformerLens-style facet aliases |
| `tl.facets.functions` | List registered facet functions, optionally filtered by target scope and class-name glob |
| `tl.facets.parameter_spec` | Recipe helper: a module parameter as a `FacetSpec` (`module, name, recipe_id`), or an absence reason |
| `tl.facets.reset` | Reset the process facet registry to the built-in recipe set, dropping every recipe registered since (bumps the registry version) |
| `tl.mechinterp.AblationControlsReport` | Every control rerun of one ablation claim and the verdict per family (read-only record) |
| `tl.mechinterp.HeadScores` | Per-head taxonomy scores |
| `tl.mechinterp.LogitLensPrediction` | Streaming per-layer prediction summary (one row per projected layer) |
| `tl.mechinterp.LogitLensPredictions` | Result of a streaming :func:`logit_lens_predictions` sweep |
| `tl.mechinterp.PatchedRun` | The receipted product of one lowered counterfactual |
| `tl.mechinterp.SavePlan` | A priced, composable retention plan (mikit) |
| `tl.mechinterp.ablation_controls` | Rerun an ablation claim under the baseline, direction and site controls; returns an `AblationControlsReport` |
| `tl.mechinterp.patch_head` | Patch one head (or its pattern) on a fused model via exact lowering |
| `tl.mechinterp.patching` | Prebuilt activation and attribution patching helpers for facets |
| `tl.neuro.ExtractionPlan` | Immutable measured plan for one whole-model extraction sweep |
| `tl.neuro.PlanSite` | One requested site's measured plan row |
| `tl.neuro.export_npz` | Export extracted features as npz interchange |
| `tl.neuro.plan_extraction` | Build a measured extraction plan from two real probe captures |
| `tl.postprocessing.Detector` | Frozen metadata of one registered output detector (`name`, `priority`, `registration_order`, `func`) |
| `tl.postprocessing.ResolvedPostprocessing` | The output-decoding provenance record a detector returns (same class as `Trace.provenance.output_postprocessor`); fields `source`, `identifier`, `verified`, `config`, `description`, `style`, `selected_output_head`, `label_source`, `label_source_version`, `confidence`, `top_n_captured`, `ambiguous`, filled by keyword (`identifier=`, `description=`, `selected_output_head=`, `label_source=`, `label_source_version=`, `top_n_captured=`) |
| `tl.postprocessing.info` | Describe one registered output detector by name |
| `tl.postprocessing.list` | List registered output detectors, optionally filtered by name glob (`name_glob=`) |
| `tl.postprocessing.register` | Decorator registering a custom output detector (`name=`, `priority=`; lower runs first) |
| `tl.postprocessing.snapshot` | Context manager saving and restoring the output detector registry |
| `tl.postprocessing.unregister` | Remove a registered output detector by name |
| `tl.recording.ModuleStackFrame` | One module frame in a predicate record context stack |
| `tl.recording.cleanup_partial` | Remove partial recording bundles and sibling temp directories |
| `tl.recording.dry_run` | Run predicates over one forward pass without retaining tensor payloads |
| `tl.recording.recover` | Recover a finalized or partial recording bundle |
| `tl.stats.RunningStats` | Always-on scalar spine accumulator (StreamingStat) |
| `tl.stats.TensorStats` | Frozen tensor statistics shared by tensor displays and narration; per-family evidence owns every claim |
| `tl.stats.RunningStatsResult` | Finalized spine statistics for one population |
| `tl.viz.Drawing` | The picture every draw returns: DOT source (or animation HTML), format, written path; `save(path, editable=)`, `show()`, `close()` (releases a retained trace; idempotent; the `with` block exit calls it); a forward drawing also carries `collapse_plan` and `collapse_schedule` |
| `Trace.animate` | `trace.animate(site, path=None, *, open_viewer=False)`: the HTML animation of the repeated ops at `site` (a site label, pass-qualified label or record) as a `Drawing`; the one animation door (the former `animate_ops`), not a `draw` keyword |
| `tl.stats.NarrationObserver` | Live stats-line echo mounted on the shipped observer surface |
| `tl.stats.RunningStatsResult` | Finalized spine statistics for one population |
| `tl.stats.narrate` | Create an echo observer (the live line density) for one site |
| `tl.trackers.HistoryView` | Session-side read view over one run's committed history |
| `tl.trackers.SeriesPoint` | One (step-span, block-truth, observation) sample of a series |
| `tl.viz.NodeSpec` | Graphviz node attributes produced by TorchLens before user customization |
| `tl.viz.attention_atlas` | Render the whole-model attention atlas (BertViz model view, matched) |
| `tl.viz.chans` | Draw a tensor's leading channels as a grid of grayscale tiles |
| `tl.viz.contact_sheet` | Render one page of the ranked overview contact sheet |
| `tl.viz.feature_maps` | Compute and store per-site spatial activation maps for graph nodes |
| `tl.viz.histogram_waterfall` | Render the histogram waterfall (step x signed-log2-bin density) |
| `tl.viz.image_scatter` | Render a bounded 2D scatter with optional PIL thumbnails |
| `tl.viz.intervention_diff` | Diff ``subject`` against ``reference``, site-key-first |
| `tl.viz.lineplot` | Render one or more series as a compact PIL line plot |
| `tl.viz.metric_strip` | Render a loss/entropy strip to paper output |
| `tl.viz.nodes.channel_grid` | Create a grid visualizer for the first ``n`` activation channels |
| `tl.viz.nodes.heatmap` | Create a channel-averaged 2D activation heatmap visualizer |
| `tl.viz.nodes.histogram` | Create a compact value-distribution histogram visualizer |
| `tl.viz.nodes.mds_scatter` | Return a draw-time node callback for stored MDS scatter annotations |
| `tl.viz.nodes.scree` | Return a draw-time node callback for stored scree annotations |
| `tl.viz.nodes.text_table` | Render a text batch as a Graphviz HTML-like table label |
| `tl.viz.prediction_table` | Render the top-k table to paginated paper pages (whole positions) |
| `tl.viz.quantile_fan` | Render the quantile-fan strip for one series (the default view) |
| `tl.viz.rgb` | Draw a three-channel tensor as one RGB picture |
| `tl.viz.site_detail` | Render the per-site detail sheet: every stream on one step axis |
| `tl.viz.timeline` | Render one timeline v2 artifact as deterministic standalone SVG text |
| `tl.viz.token_strip` | Render the paper-format token strip via matplotlib |
### Fields and methods
| Spelling | Meaning |
| --- | --- |
| `AblationControlsReport.baseline` | field of `AblationControlsReport`: the claimed ablation's baseline (`"zero"`, `"mean"` or `"resample"`) |
| `AblationControlsReport.clean` | field of `AblationControlsReport`: the metric on the unedited run |
| `AblationControlsReport.direction_p` | field of `AblationControlsReport`: the direction family's rank p-value, or `None` when no direction was given |
| `AblationControlsReport.effect` | field of `AblationControlsReport`: the claimed ablation's effect, edited metric minus clean metric |
| `AblationControlsReport.families` | field of `AblationControlsReport`: one verdict per family that ran (`family`, `survives`, `reason`) |
| `AblationControlsReport.rows` | field of `AblationControlsReport`: one row per rerun, in run order (`family`, `name`, `effect`, `survives`) |
| `AblationControlsReport.survives` | `True` when every family that ran survived |
| `AbsenceReason.detail` | Human-readable explanation of the absence |
| `AbsenceReason.remedy` | Optional capture instruction for actionable missing-payload cases |
| `AttentionRecording.__iter__` | Iterate the recorded attention-module names in first-call order |
| `AttentionRecording.__len__` | The number of distinct attention modules recorded |
| `Attribution.target_repr` | Compact representation of the scalarization target |
| `AttributionGraph.capture_status` | The capture's completeness verdict, as the trace reports it |
| `AttributionGraph.frozen_sites` | Labels of the sites held at their captured values |
| `AttributionGraph.has_demeaned_logits` | Whether a logit node's value is its logit minus the mean logit |
| `BindReport.door` | The door the call was bound through (`"call"` or `"generate"`) |
| `BindReport.duration_s` | Wall time of the bound call in seconds, or `None` when it was not measured |
| `BindReport.execution_effect` | How a bound call's helpers act: always `"values_replaced_after_execution"` (outputs are replaced after each op runs) |
| `BindReport.fire_records` | The `FireRecord` objects of every firing during the bound call |
| `BindReport.fires` | One plain-dict row per firing during the bound call (`fire_records` carries the typed objects) |
| `BindReport.module_aliases` | Canonical module address to the other addresses the same module object is registered under |
| `BindReport.resolved_static_targets` | Rule id to the module addresses its static anchor resolved to at bind time |
| `BindReport.rule_fire_counts` | Rule id to the number of times it fired during the bound call |
| `BindReport.spec_digest` | Digest of the intervention spec the call was bound with |
| `CausalAuditSet.build_interventions` | The ``(tl.units(label, mask), edit)`` pair that produced ``measured_effect`` |
| `CausalAuditSet.draw_index` | Index of the random draw (``0`` for ``"top"`` and ``"bottom"``) |
| `CausalAuditUnit.build_interventions` | The ``(tl.units(label, mask), edit)`` pair that produced ``measured_effect`` |
| `ComponentStack.topk` | Return the ``k`` largest rows as ``(label, score)`` pairs |
| `ContributionScores.topk` | Return the ``k`` largest contributors at one (position, direction) |
| `CurveStep.build_inputs` | The perturbed inputs of this step, shaped like the caller's ``inputs`` |
| `CurveStep.build_interventions` | The recorded edits of this step, as ``(selection, edit)`` pairs |
| `Drawing.collapse_plan` | The collapse plan a forward draw with `collapse="auto"`, `"max"` or a float level rendered from, computed against the drawing's resolved render context (property); refuses typed (`collapse_plan_unavailable`) for `collapse="none"`, a backward or combined graph, an animation, or a released trace |
| `Drawing.collapse_schedule` | The monotone float collapse schedule of the drawn forward graph (level 0.0 is the full graph, level 1.0 the maximum collapse) under the drawing's resolved render context (property); refuses typed for non-forward drawings and a released trace |
| `Drawing.file_format` | The rendered file format (``"pdf"``, ``"svg"``, ...) |
| `Drawing.receipt` | Execution receipt from the pinned capture policy |
| `ExtractionPlan.batch_size` | field of `ExtractionPlan` |
| `ExtractionPlan.included_sites` | Return the sites in the run's save set |
| `ExtractionPlan.input_signature` | field of `ExtractionPlan` |
| `ExtractionPlan.memory_budget_bytes` | field of `ExtractionPlan` |
| `ExtractionPlan.model_class` | field of `ExtractionPlan` |
| `ExtractionPlan.num_stimuli` | field of `ExtractionPlan` |
| `ExtractionPlan.peak_basis` | field of `ExtractionPlan` |
| `ExtractionPlan.planned_passes` | Return exact planned forwards: warm-up + two probes (already run) + batches |
| `ExtractionPlan.predicted_peak_bytes` | field of `ExtractionPlan` |
| `ExtractionPlan.probe_batch_sizes` | field of `ExtractionPlan` |
| `ExtractionPlan.probe_peak_pairs` | field of `ExtractionPlan` |
| `ExtractionPlan.retained_bytes_per_batch` | Return fitted retained bytes for one batch across included sites |
| `ExtractionPlan.safety_margin` | field of `ExtractionPlan` |
| `ExtractionPlan.total_output_bytes` | Return fitted final artifact bytes across all stimuli |
| `ExtractionPlan.transform_repr` | field of `ExtractionPlan` |
| `FacetFunction.class_names` | Short class names matched by this recipe |
| `FacetFunction.has_predicate` | Whether this recipe has an arbitrary predicate matcher |
| `FacetFunction.qualnames` | Fully-qualified class names matched by this recipe |
| `FacetFunction.target_scope` | Record scope the recipe targets |
| `FacetRegistrySnapshot.functions` | Immutable active recipe entries |
| `FacetRegistrySnapshot.provenance_id` | Stable digest of the recipe identities and their source tier |
| `FacetRegistrySnapshot.version` | Monotonic global registry version at snapshot construction time |
| `FacetSpec.apply_transforms` | Apply the transform chain to a value |
| `FacetSpec.capability_class` | Weakest capability class of the transform chain |
| `FacetSpec.capability_flags` | Weakest-link capability flags of the transform chain |
| `FacetSpec.conflict_alias_group` | Shared-home conflict and alias group for intervention |
| `FacetSpec.function_id` | Recipe identifier that produced this facet |
| `FacetSpec.function_version` | Recipe version that produced this facet |
| `FacetSpec.home` | Runtime home object; intentionally not portable |
| `FacetSpec.home_address` | Stable home address when available |
| `FacetSpec.home_kind` | Kind of home value |
| `FacetSpec.home_label` | Stable home label when available |
| `FacetSpec.output_path` | Structural output path inside a container output |
| `FacetSpec.scatter_update` | Scatter an edited facet slice back into the full home-op output |
| `FacetSpec.split` | Return specs for equal sections along a dimension |
| `FacetSpec.transpose` | Append a transpose primitive |
| `FacetSpec.value_version` | Which home value version the spec reads |
| `FacetSpec.write_mask` | Return a boolean mask of home positions written by this facet |
| `FacetView.function_source` | Return recipe name provenance for this view |
| `FacetView.get` | Return an available facet value, or a default when it is unavailable now |
| `FacetView.has` | Whether a facet is available in the current trace |
| `FacetView.invalidate` | Clear cached computed facet values |
| `FamilyEvidence.policy` | `exact`, `sampled`, `unavailable`, or nonfinite presence-only `verdict`; unavailable evidence carries a reason |
| `FamilyEvidence.is_sampled` | Whether this family rode the seeded gathered sample |
| `FireRecord.call_label` | Label of the call the firing belongs to, or `None` when it is not a call-level site |
| `FireRecord.replaced` | Whether the firing replaced the tensor at its site, or `None` when not recorded |
| `FireRecord.target_label` | Label of the op the firing edited |
| `FireRecord.timing` | Whether the firing ran before (`"pre"`) or after (`"post"`) its op, or `None` when not recorded |
| `FireRecord.tuple_index` | Position of the edited output in the op's gradient tuple for backward firings, or `None` |
| `GraphNode.token_id` | Token id: the input token at ``position`` (token nodes) or the target vocabulary id (logit nodes) |
| `HeadScores.eligible_rows` | field of `HeadScores` |
| `HeadScores.module_addresses` | field of `HeadScores` |
| `HeadScores.topk` | Return the ``k`` highest-scoring heads as ``(address.headN, score)`` |
| `HeadWeightViews.num_kv_heads` | field of `HeadWeightViews` |
| `HistoryView.run_record` | field of `HistoryView` |
| `InputProvenance.has_semantic_values` | True only when the values are caller-supplied real data. Gates derived-semantics claims |
| `InterventionValidationReport.__bool__` | Truth value of the report: `True` when every axis passes |
| `InterventionValidationReport.passed` | Return the aggregate pass/fail result |
| `LogitLens.entries` | Per-layer projections in block execution order |
| `LogitLens.final_logits` | The model's captured real output logits when readable, else ``None`` |
| `LogitLens.is_validated` | Whether the reconstructed lens was numerically validated against the captured last-block ``resid_post`` -> logits pair |
| `LogitLens.stacked` | Return all per-layer logits stacked on a new leading layer axis |
| `LogitLensPrediction.token_ids` | Requested-token ids as given; empty when no tokens were requested |
| `LogitLensPrediction.token_logits` | Requested-token logits, shaped `[batch, n_positions, n_tokens]` |
| `LogitLensPrediction.token_probs` | Requested-token probabilities, shaped `[batch, n_positions, n_tokens]` |
| `LogitLensPrediction.token_ranks` | One-based ranks of the requested tokens as int64; tie convention recorded on the parent result |
| `LogitLensPrediction.top_ids` | Top-k token ids, shaped `[batch, n_positions, k]` |
| `LogitLensPrediction.top_logits` | Top-k logit values, shaped `[batch, n_positions, k]` |
| `LogitLensPredictions.is_validated` | Whether the reconstructed lens passed numeric validation |
| `MissingGrad.raise_error` | Raise the missing-gradient tensor-use error |
| `NarrationObserver.lines` | field of `NarrationObserver` |
| `NarrationObserver.signal` | property of `NarrationObserver`, derived from its `site` (`"backward"` for a `tl.grad(...)` site); not a constructor keyword |
| `NarrationObserver.sink` | field of `NarrationObserver` |
| `NarrationObserver.style` | field of `NarrationObserver` |
| `PatchedRun.metric_value` | field of `PatchedRun` |
| `PatchedRun.read_facet` | Read a facet from the patched rerun, refusing stale virtual reads |
| `PatchedRun.receipt` | field of `PatchedRun` |
| `PlanSite.bytes_at_batch` | Return fitted retained bytes for one batch of ``batch_size`` |
| `PlanSite.bytes_fixed` | Batch-independent byte intercept from the two-point linear fit |
| `PlanSite.bytes_per_stimulus` | Byte slope per stimulus from the two-point fit |
| `PlanSite.exclusion_reason` | Actionable reason when excluded; never silently dropped |
| `PlanSite.is_batch_invariant` | Whether the site's bytes did NOT scale with batch |
| `PlanSite.is_included` | Whether the site is in the run's save set |
| `PlanSite.transform_seconds` | Measured live transform time at the probe batch (the probe executes the transform chain; post-transform bytes and time are measured, never derived) |
| `PlanSite.transformed_bytes` | Post-transform bytes at the probe batch, when a transform ran |
| `PlanSite.transformed_shape` | Post-transform shape at the probe batch, when a transform ran |
| `Population.__len__` | The number of members in the population |
| `ProjectionSource.has_fused_qkv` | Whether the role is a column slice of a fused QKV weight |
| `RandomizationTest.module_addresses` | Randomized modules, top (output side) first |
| `ReadRow.alias_group` | Ordinal of the `(node, slot)` alias group whose sites share one gradient, or `None` when unshared |
| `ReadRow.frozen_reached` | Whether a requested freeze actually fired in the target's backward cone; `None` when no backward touched the site |
| `ReadRow.is_detached` | ``True`` when ``value`` is a detached copy (always true in v1 for carried tensors) |
| `ReadRow.is_frozen_requested` | field of `ReadRow` |
| `ReadRow.is_frozen_resolved` | field of `ReadRow` |
| `ReadRow.is_rescorable` | ``False`` on loaded tables (the runtime registry is not serialized) |
| `ReadRow.policy_digest` | Digest of the resolved frozen policy this row was measured under |
| `ReadRow.reduction` | Named scalar reduction that produced `score`, or `None` |
| `ReadRow.retention` | `'saved'` or `'unsaved'` payload retention at the site |
| `ReadRow.sample_id` | Nullable sample identity for dataset-mean rows (always `None` in v1) |
| `ReadRow.status_reason` | Closed reason detail for the row's status (for example `not_upstream_of_target`), or `None` |
| `ReadRow.target_id` | Stable id of the target this row answers, or `None` for target-free rows |
| `ReadTable.__iter__` | Iterate the row keys `(target_id, kind, address)` in table order |
| `ReadTable.__len__` | The number of rows in the table |
| `ReadTable.aggregate_targets` | Fold multi-target rows into one row per address, explicitly |
| `ReadTable.collapse_alias_groups` | Explicitly collapse alias groups to one row per `(node, slot)` |
| `ReadTable.column` | Return one row field as `key -> value` |
| `ReadTable.is_rescorable` | Whether the table can be re-scored against its live trace |
| `ReadTable.status_counts` | Return the status histogram over all rows |
| `ReadTable.trace_token` | Return the trace validity token captured at read time |
| `Recording.__iter__` | Iterate the retained activation records |
| `Recording.__len__` | The number of retained activation records |
| `Recording.by_pass` | Map from pass number to indexes of its retained records |
| `Recording.orphan_records` | Records that arrived without a matching pass, kept for diagnosis |
| `RegionBoundary.exits` | The region's exit edges, ordered by consuming-op execution order then occurrence order |
| `RegionEdge.addresses` | The canonical edge occurrence addresses |
| `RegionEdge.nested` | Whether any occurrence address has a nested (container) argument |
| `RegionInstance.exits` | The exit edges of this pass instance; they commit atomically |
| `RegionTarget.boundary` | The typed, ordered entry/exit `RegionBoundary` of the region |
| `RegionTarget.region_digest` | Stable digest over members and exit addresses (audit identity) |
| `RegionTarget.source_slice` | The `TraceSlice` this region was derived from |
| `Relationship.DIFF_MODEL` | Evidence level: the bundle members come from different models |
| `Relationship.SAME_MODEL_OBJECT_AT_CAPTURE` | Evidence level: the members were captured from the same model object |
| `Relationship.SAME_OBJECT` | Evidence level: the members are the same object |
| `Relationship.SAME_PARAM_SHAPES` | Evidence level: the members' models share parameter shapes only |
| `Relationship.SHARED_ARCHITECTURE` | Evidence level: the members' models share an architecture |
| `Relationship.SHARED_GRAPH_DIFFERENT_INPUT` | Evidence level: the members share a graph but ran on different inputs |
| `Relationship.SHARED_GRAPH_SAME_INPUT` | Evidence level: the members share a graph and ran on the same input |
| `Relationship.UNKNOWN` | Evidence level: the relationship between the members could not be established |
| `RunningStats.compute` | Return the finalized spine statistics (`RunningStatsResult`) |
| `RunningStats.merge` | Fold another spine into this one |
| `RunningStatsResult.num_finite` | field of `RunningStatsResult` |
| `RunningStatsResult.num_nan` | field of `RunningStatsResult` |
| `RunningStatsResult.num_negative` | field of `RunningStatsResult` |
| `RunningStatsResult.num_neginf` | field of `RunningStatsResult` |
| `RunningStatsResult.num_posinf` | field of `RunningStatsResult` |
| `RunningStatsResult.numel` | field of `RunningStatsResult` |
| `RunningStatsResult.num_zero` | field of `RunningStatsResult` |
| `RunningStatsResult.finite_absmax` | field of `RunningStatsResult` |
| `RunningStatsResult.finite_max` | field of `RunningStatsResult` |
| `RunningStatsResult.finite_min` | field of `RunningStatsResult` |
| `RunningStatsResult.m2` | field of `RunningStatsResult` |
| `RunningStatsResult.reduction_dtype` | field of `RunningStatsResult` |
| `RunningStatsResult.sum_abs` | field of `RunningStatsResult` |
| `RunningStatsResult.sum_squares` | field of `RunningStatsResult` |
| `SavePlan.analyses` | field of `SavePlan` |
| `SavePlan.describe` | Return the full per-site table as text |
| `SavePlan.estimated_bytes` | Return the summed payload estimate for the named sites |
| `SavePlan.module_subtrees` | field of `SavePlan` |
| `SeriesPoint.is_coarsened` | field of `SeriesPoint` |
| `SpecCompat.site_diff` | Human-readable site-level diff; empty when neither side has site keys |
| `TargetManifestDiff.matched_site_keys` | Site keys present in both the saved manifest and the new trace |
| `TargetManifestDiff.missing_labels` | Labels in the saved manifest that the new trace lacks |
| `TargetManifestDiff.missing_site_keys` | Site keys in the saved manifest that the new trace lacks |
| `TargetManifestDiff.new_labels` | Labels in the new trace that the saved manifest lacks |
| `TargetManifestDiff.new_site_keys` | Site keys in the new trace that the saved manifest lacks |
| `TargetManifestDiff.selector_resolution_diffs` | Selector to how its resolution differs between the saved manifest and the new trace |
| `TargetManifestDiff.site_diff_lines` | Render the printed site-level diff (one line per changed site) |
| `TensorStats.mean_evidence` | `FamilyEvidence` with policy, population, sample size, and reason for the mean |
| `TensorStats.nonfinite_evidence` | Exact census evidence, unavailable with a reason, or presence-only `verdict` with counts unknown |
| `TensorStats.finite_min` | Minimum finite value, or `None`; read `extrema_evidence` for its policy |
| `TensorStats.finite_max` | Maximum finite value, or `None`; read `extrema_evidence` for its policy |
| `TensorStats.mean` | Mean, or `None`; read `mean_evidence` for its policy |
| `TensorStats.numel` | Number of elements in the tensor |
| `TensorStats.nbytes` | Tensor payload bytes, or `None` when unknown |
| `TensorStats.has_no_finite_values` | Whether no element is finite (the fully poisoned case) |
| `TensorStats.is_all_false` | bool family: every element false (exact) |
| `TensorStats.is_all_true` | bool family: every element true (exact) |
| `TensorStats.is_all_zero` | Whether every element is exactly zero (exact claim) |
| `TensorStats.is_magnitude_based` | field of `TensorStats` |
| `TensorStats.num_finite` | Elements with finite values, or `None` when any nonfinite count is unknown |
| `TensorStats.num_nan` | Exact count, or `None` when unknown |
| `TensorStats.num_neginf` | Exact count, or `None` when unknown |
| `TensorStats.num_nonfinite` | Total nonfinite elements, or `None` when any count is unknown |
| `TensorStats.num_posinf` | Exact count, or `None` when unknown |
| `TensorStats.num_true` | field of `TensorStats` |
| `TensorStats.num_zero` | field of `TensorStats` |
| `TensorStats.policy` | Read-only family summary: `sampled` if any family is sampled, `reuse` if only reused facts are present, otherwise `exact` |
| `TensorStats.std` | field of `TensorStats` |
| `TensorStats.std_evidence` | field of `TensorStats` |
| `TensorStats.std_se` | field of `TensorStats` |
| `TokenAttribution.completeness` | ``attribution_sum`` / ``target_delta`` / ``completeness_residual`` / ``residual_rel`` / ``target_delta_abs`` |
| `TokenAttribution.convergence` | Grid-pair evidence: requested spelling, evaluated grids, residual and both stability norms per grid, and the stopping statement |
| `TokenAttribution.cost` | Logical/physical call accounting incl. the batching audit evidence |
| `TokenAttribution.display_tokens` | Display strings per raw token (marker cleanup only) |
| `TokenAttribution.footer_lines` | Build the mandatory disclosure footer |
| `TokenAttribution.offsets` | Character offsets per token, or ``None`` when the tokenizer cannot provide them (disclosed) |
| `TokenAttribution.payload` | Build the renderer-neutral payload (the tviz contract) |
| `TokenAttribution.raw_tokens` | Tokenizer tokens, never silently merged |
| `TokenAttribution.special_tokens_mask` | 1 where the token is a tokenizer special, or ``None`` |
| `TokenAttribution.target_repr` | Full target resolution disclosure incl. the decoded token |
| `TokenAttributionPayload.to_markdown` | Render a plain-text fallback table with the footer lines |
| `TrajectoryStats.compute_stats` | Return ``{layer: RunningStatsResult}`` for one statistic (count, min, max, mean, ...) |
| `TrajectoryStats.is_input_masked` | Whether each row's own input tokens were banned before reduction |
| `TrajectoryStats.module_addresses` | Block addresses in depth order (one row per layer) |
| `TrajectoryStats.num_examples` | Number of examples (batch rows) folded |
| `TrajectoryStats.num_positions` | Sequence length every batch shared |
| `ValidationFailure.inf_mismatch` | Whether the saved and recomputed tensors disagree on infinities, or `None` |
| `ValidationFailure.max_abs_diff` | Largest absolute difference between saved and recomputed values, or `None` |
| `ValidationFailure.max_rel_diff` | Largest relative difference between saved and recomputed values, or `None` |
| `ValidationFailure.nan_mismatch` | Whether the saved and recomputed tensors disagree on NaNs, or `None` |
| `ValidationFailure.recomputed_dtype` | Dtype of the recomputed tensor, or `None` |
| `ValidationFailure.recomputed_shape` | Shape of the recomputed tensor, or `None` |
| `ValidationFailure.reduction_depth` | Reduction depth of the failing op used to scale the tolerance, or `None` |
| `ValidationFailure.saved_dtype` | Dtype of the saved tensor, or `None` |
| `ValidationFailure.saved_shape` | Shape of the saved tensor, or `None` |
| `ValidationStatus.__bool__` | Truth value only for completed results: `True` for `passed`, `False` for `failed`; `TypeError` otherwise |
| `ValidationStatus.passed` | Return whether replay validation ran and passed |
| `VisualizationOptions.size_scale` | UNSTABLE size-channel scale transform (``"sqrt"``/``"linear"``); see ``Trace.draw`` |
| `VisualizationOptions.graph` | Which graph the picture draws: ``"forward"`` (default), ``"backward"`` or ``"combined"`` |
| `VisualizationOptions.rolled` | ``True`` draws each repeated module once (the rolled graph); ``False`` (default) draws every pass |
| `VisualizationOptions.open_viewer` | ``True`` opens the written picture in the system viewer; the default never does |
| `VisualizationOptions.file_format` | Format for a ``container_path`` with no extension (default ``"pdf"``); an extension on the path picks the format, and a disagreeing explicit value refuses ``draw_format_mismatch`` |
| `VisualizationOptions.animate` | A site label, pass-qualified label or record: the picture is the HTML animation of the repeated ops at that site (default ``None``) |

#### `tl.Recording`

| Spelling | Meaning |
| --- | --- |
| `Recording.add_grad_record` | Append one retained gradient record and update the gradient indexes |
| `Recording.by_address` | Map from module address to indexes of its retained records |
| `Recording.by_label` | Map from layer label to `(pass, record)` index pairs of its retained records |
| `Recording.by_pass` | Map from pass index to indexes of the retained records captured in that pass |
| `Recording.do` | Cook through `to_trace()` and run `Trace.do` on a fork of the cooked trace: the edited, replayed Trace, with no in-place mutation warning; `recording.do(site, edit, /, *, model=None, inputs=None, intervention=None, loss=None)`, with the site and edit positional-only as on `Trace.do` |
| `Recording.end_times` | Per-pass end timestamps of the recording |
| `Recording.enrich` | Return a new recording with the requested incremental enrichments, or all feasible ones |
| `Recording.error_repr` | Repr of the error that ended a failed recording, or `None` |
| `Recording.error_traceback` | Traceback text of the error that ended a failed recording, or `None` |
| `Recording.fork` | Cook through `to_trace()` and return `Trace.fork(name)` of the cooked trace |
| `Recording.from_capture_events` | Build a lazy recording projection from a predicate capture session |
| `Recording.grad_by_grad_fn_label` | Map from gradient function label to indexes into `grad_records` |
| `Recording.grad_by_label` | Map from layer label to indexes into `grad_records` |
| `Recording.grad_by_pass` | Map from pass index to indexes into `grad_records` |
| `Recording.grad_records` | Retained gradient records, in capture order |
| `Recording.history_size` | Number of earlier events kept for predicate history at recording time |
| `Recording.last_event_func` | Function name of the last captured event on a failed recording, best effort |
| `Recording.last_event_input_meta` | Input metadata of the last captured event on a failed recording, best effort |
| `Recording.last_event_label` | Label of the last captured event on a failed recording, best effort |
| `Recording.last_event_source_line` | Source line of the last captured event on a failed recording, best effort |
| `Recording.last_successful_op_label` | Label of the last op committed before a failure, or `None` |
| `Recording.lean_omissions` | PROVISIONAL SPELLING. What a lean recording left out; `()` for a default recording |
| `Recording.orphan_records` | Payload-free rows (raw label, label, payload reference) of the orphan island ops kept by `keep_orphans=True` |
| `Recording.raw_metadata` | Return payload-free raw metadata rows for every captured event, retained or not; refused with `recording_lean_omitted` on a lean recording |
| `Recording.records` | Retained activation records, in capture order |
| `Recording.recovery_warnings` | Warnings collected while recovering this recording from disk |
| `Recording.save` | Cook through `to_trace()` and write the cooked trace with `Trace.save` (same keywords, same artifact) |
| `Recording.start_times` | Per-pass start timestamps of the recording |
| `Recording.to_agent_json` | Cook through `to_trace()` and return the cooked trace's `Trace.to_agent_json(max_ops=)` dump |

#### `tl.options.RecordingOptions`

| Spelling | Meaning |
| --- | --- |
| `RecordingOptions.default_grad` | Gradient capture decision for events the predicate does not decide; giving it installs gradient hooks for a later `Recording.backward()` |
| `RecordingOptions.default_module` | Capture decision for module events the predicate does not decide |
| `RecordingOptions.default_op` | Capture decision for operation events the predicate does not decide |
| `RecordingOptions.history_size` | Number of earlier events kept for predicate history; 0 to 1024 |
| `RecordingOptions.include_source_events` | Whether source events appear in the predicate lookback of recent ops |
| `RecordingOptions.lean` | PROVISIONAL SPELLING. Journal only the events the selection keeps; the recording names its omissions and refuses the reads that need the full journal |
| `RecordingOptions.max_predicate_failures` | Maximum number of predicate failures stored on the recording; further ones are counted as overflow |
| `RecordingOptions.on_forward_error` | Failed-forward handling: `"raise"` (default), `"attach_partial"` or `"return_partial"` |
| `RecordingOptions.on_predicate_error` | Predicate error handling: `"auto"`, `"accumulate"` or `"fail-fast"` |
| `RecordingOptions.postprocess` | Post-capture enrichment preset applied to the recording; `"none"` by default |

### Keywords
| Spelling | Meaning |
| --- | --- |
| `ce_loss_clean=` | field of `SpliceResult` |
| `ce_loss_spliced=` | field of `SpliceResult` |
| `extra_positional=` | Var-positional catch on `tl.trace` and `tl.record`: only the model and the inputs are positional; any further positional argument refuses typed `extra_positional_input_invalid` (`input_kwargs=` is keyword-only) |
| `axis_roles=` | field of `ReceptiveField` |
| `backward_pass_number=` | field of `FireRecord` |
| `bin_edges=` | field of `HistogramPoint` |
| `can_intervene=` | keyword of `BackendCapabilities` |
| `candidate_num_components=` | keyword of `tl.transforms.srp_fidelity_probe`: the candidate numbers of output components (projection widths k) to probe, one report row each |
| `comet_experiment=` | keyword of `tl.trackers.CometSink`: the Comet experiment to log to |
| `d_head=` | keyword of `FacetSpec.heads`: head dimension; the default `None` infers it from the last axis when the facet is read |
| `edit=` | field of `FireRecord`; keyword of `tl.attribution.causal_audit` (keyword-only; default `'zero'`): `"zero"` (default), `"mean"` (each unit's mean over the batch), or a number or tensor (the baseline activation, broadcast to the site); an EDGE table is patched, not ablated, and takes only the default; keyword of `tl.experiment.sweep_grid` (keyword-only; default `None`); keyword of `tl.experiment.sweep_sites` (keyword-only; required): REQUIRED. The edit applied at each candidate site (which ablation is sound is a live scientific argument — no default); keyword of `tl.intervention.keep_grad` (required): A forward `Edit` (`tl.scale(2.0)`, `tl.patch_from(source)`, ...) or a hook callable `fn(out, *, ctx)`; keyword of `tl.positions` (required): Any same-shape forward edit (`tl.zero_ablate()`, `tl.steer(...)`, `tl.patch_from(...)`, or a hook callable taking `(out, *, ctx)`) |
| `edit_name=` | field of `FireRecord` |
| `extra_facets=` | keyword of `tl.facets.snapshot` |
| `function_id=` | keyword of `FacetSpec.computed`, `FacetSpec.from_home`, `tl.facets.FacetSpec` |
| `function_version=` | keyword of `FacetSpec.computed`, `FacetSpec.from_home`, `tl.facets.FacetSpec` |
| `functions=` | Immutable active recipe entries |
| `grad_fn_label=` | field of `GradFnCall` |
| `has_fused_qkv=` | keyword of `tl.mechinterp.ProjectionSource` |
| `has_semantic_values=` | True only when the values are caller-supplied real data. Gates derived-semantics claims; constructor keyword of `tl.types.InputProvenance` (required): True only when the values are caller-supplied real data. Gates derived-semantics claims |
| `is_capture_complete=` | keyword of `ModuleInputSnapshot` |
| `is_compatible_with_append=` | field of `Edit` |
| `is_detected=` | field of `DoctorCheck` |
| `kl_divergence=` | field of `SpliceResult` |
| `module_pass_number=` | field of `RecordContext` |
| `num_components=` | keyword of `tl.repgeom.classical_mds`, `tl.stats.PCA` and `tl.transforms.srp`: the number of components (output dimensions) to return; constructor keyword of `tl.transforms.SrpSizing` (required): Number of output components: the sized projection width |
| `num_heads=` | keyword of `FacetSpec.heads`: number of attention heads to split into |
| `rolled=` | keyword of `Trace.draw` and `VisualizationOptions`: ``True`` for the rolled graph, ``False`` (default) for every pass; keyword of `Bundle.draw` (keyword-only; default `False`) |
| `open_viewer=` | keyword of `Trace.draw`, `tl.draw` and `VisualizationOptions`: ``True`` opens the written picture; the default never does; keyword of `Bundle.draw` (keyword-only; default `False`); keyword of `Trace.animate` (keyword-only; default `False`): `True` opens the animation in the system viewer; keyword of `tl.viz.bundle_diff` (keyword-only; default `False`): `True` opens the written picture in a viewer; the default never does |
| `size_scale=` | keyword of `Trace.draw` and `VisualizationOptions`: size-channel transform ``"sqrt"`` (default) or ``"linear"``; refuses without `size_by=` |
| `where=` | keyword of `Trace.summary`: regex string (searched in the row name, its module address and, for an op row, the module that ran it), TorchLens selector (``tl.in_module(...)``, ``tl.func(...)``; keeps rows owning a matched op) or ``row -> bool`` callable; presentation-only (whole-model totals preserved and coverage disclosed) |
| `graph=` | keyword of `Trace.draw` and `VisualizationOptions`: ``"forward"`` (default), ``"backward"`` or ``"combined"``; keyword of `Bundle.draw` (keyword-only; default `'forward'`); keyword of `tl.export.html` (keyword-only; default `'collapsed'`): `"collapsed"` (default) / `"flat"` / `"none"` |
| `editable=` | keyword of `Drawing.save`: ``True`` writes the Graphviz source instead of a rendered picture |
| `num_stimuli=` | keyword of `tl.neuro.plan_extraction` |
| `pass_number=` | field of `BackwardPass`, `DiffRow`, `GradRecordContext`, `ModuleStackFrame`, `RecordContext`, `SiteRow`; the one-based backward pass number taken by `for_pass` on `trace.autograd.backward_passes`, `grad_fn.calls` and `op.grads` (was `pass_index=`; JMT 1 Nov 2026 3:00 PM PT) |
| `query_slice=` | keyword of `AttentionView.crop_to`: which query positions to keep, as a slice |
| `sampling=` | field of `FireRecord` |
| `value_fn=` | keyword of `FacetSpec.computed` |

## Current spellings: namespaces, members and keywords
Public names whose spelling or home changed, and names new to the public surface, one line each.
The sections above describe each surface in depth.

### Namespace names
| Spelling | Meaning |
| --- | --- |
| `tl.backends.BackendProtocol` | The one contract every backend fills: capability table, capture, validate, reference output, host copy, realize-once |
| `tl.backends.FrameworkRequirement` | A backend's framework distributions, supported range, tested-through version, CI pin and install extra |
| `tl.backends.REFUSAL_CODES` | Closed registry of backend refusal codes (`exc.code` of `BackendUnsupportedError`) |
| `tl.backends.VALIDATION_REASONS` | Closed list of per-op validation decision reasons every backend's `validate` uses |
| `tl.clip_norm` | Neutral edit that rescales a tensor (on `tl.grad(site)`, the gradient) to at most `max_norm` in `norm_type` norm |
| `tl.debug.BackwardNanReport` | Result of `tl.debug.bisect_nan_backward`: the backward pass walked, the earliest provable non-finite birth and the full transition ledger |
| `tl.debug.BisectNanReport` | Result of `tl.debug.bisect_nan` |
| `tl.debug.BisectPrecisionReport` | Result of `tl.debug.bisect_precision` |
| `tl.debug.CheckFinding` | One finding recorded by a training-loop check |
| `tl.debug.CheckReport` | The findings a `tl.debug.ChecksSession` collected, as a report |
| `tl.debug.ChecksSession` | Capture-free training-loop checks bound to a model and optimizer; see `docs/reference/checks_kit.md` |
| `tl.debug.ClipLedger` | Ledger of gradient-clipping events across steps |
| `tl.debug.ClipLedgerSnapshot` | Point-in-time read of a `tl.debug.ClipLedger` |
| `tl.debug.GuessInputShapeReport` | Result of `tl.debug.guess_input_shape` |
| `tl.debug.LineageReport` | Result of `tl.debug.lineage` |
| `tl.debug.LivenessProbe` | Per-site recorder of zero-fraction and running maximum across fires; a statistic, never a verdict |
| `tl.debug.OptimizerFacts` | Adapter-read facts about one optimizer; `is_known` gates every derived band |
| `tl.debug.RangeProbe` | Opt-in live semantic range check per site |
| `tl.debug.ScaleLedger` | Ledger of grad-scaler scale and skipped-step history across steps |
| `tl.debug.ScaleLedgerSnapshot` | Point-in-time read of a `tl.debug.ScaleLedger` |
| `tl.debug.ScanRow` | Health facts for one unique tensor in a parameter scan; `aliases` keeps every other name bound to the same tensor |
| `tl.debug.Watchdog` | Death-spiral watchdog: warns when no optimizer step is ever accepted |
| `tl.debug.WatchdogSnapshot` | Point-in-time read of a `tl.debug.Watchdog` |
| `tl.debug.guess_input_shape` | Infer a synthetic input shape while leaving the caller's global RNG untouched |
| `tl.differential.DiffReport` | The settled two-capture differential: rows plus honest totals |
| `tl.differential.diff` | Settle two captures into the guarded differential projection |
| `tl.distributed.is_vetted_build` | Capability probe: does this torch build match a censused snapshot? |
| `tl.distributed.verify_membership_lineages` | Audit per-membership lineage evidence across rank cores |
| `tl.errors.AmbiguousAxisError` | Raised when a helper cannot infer a feature axis safely |
| `tl.errors.AmbiguousBackendError` | Raised when backend auto-resolution has multiple equal-priority matches |
| `tl.errors.AmbiguousBatchInputError` | Raised when `chunk_size` cannot infer which input leaf is batched |
| `tl.errors.AmbiguousCallError` | Raised when a module-level query has multiple executed calls |
| `tl.errors.AmbiguousLabelError` | Raised when `Bundle.at` finds a label in multiple accessors |
| `tl.errors.AmbiguousPassError` | Raised when a layer-level query has multiple executed passes |
| `tl.errors.AmbiguousTargetError` | Raised when a projective query has more than one reachable target and none is the primary output. When the other targets are caches (an output role with a path segment `past_key_values`, `presents` or `mems`, or one containing `cache`), a query with no `target=` answers the primary output instead: the role whose last segment is `logits`, else the first non-cache output in model output order. Two real outputs and no cache still raise; `target=` selects any role, caches included |
| `tl.errors.AppendStreamingUnsupportedError` | Raised when append rerun would need to mutate active streamed activation blobs |
| `tl.errors.ArtifactTooNewError` | Raised when an artifact declares a schema newer than this runtime's ceiling |
| `tl.errors.ArtifactTooOldError` | Raised when an artifact predates the supported rehydration floor |
| `tl.errors.BundleComparisonError` | Raised when bundle members fail a gated operation's comparison gate |
| `tl.errors.CaptureGapError` | Reserved enforcement error for an unrepresented torch invocation |
| `tl.errors.CaptureGapWarning` | Shadow-mode report for a possible unrepresented torch invocation |
| `tl.errors.CaptureNotActiveError` | Raised when a capture-only operation is called outside an active capture |
| `tl.errors.DatasetExtractionResumeError` | Raised when a resumable extraction artifact cannot be safely continued |
| `tl.errors.DirectWriteSaveError` | Raised when executable spec save sees direct out writes |
| `tl.errors.InterventionWarning` | Base class for TorchLens intervention warnings |
| `tl.errors.LiveIndexWindowError` | Raised when a live-index query targets an unavailable raw label |
| `tl.errors.MLXValueUnavailableError` | Raised when MLX lazy evaluation makes a predicate value unavailable |
| `tl.errors.MergeConflictError` | The presented rank cores structurally contradict each other |
| `tl.errors.MergeInputError` | A merge input is unusable: not a rank capture, duplicate rank, missing distributed evidence, or an out-of-scope boundary |
| `tl.errors.MergedArtifactError` | Merged artifact save/load failure: schema, integrity, or tamper |
| `tl.errors.MergedSurfaceUnsupportedError` | A refused surface on a merged trace (run/export/validate/selectors) |
| `tl.errors.MergedTraceError` | Base class for merged-trace errors; `fields["code"]` is frozen |
| `tl.errors.NarrationConfigError` | Raised for `narrate=` configuration refusals at capture entry |
| `tl.errors.NarrationStatsError` | Raised when an armed echo stats rung refuses a value computation |
| `tl.errors.NoInfluencePathError` | Raised when two graph points have no directed influence path |
| `tl.errors.OpaqueCallableError` | Raised when an executable intervention save would require opaque code |
| `tl.errors.ParamEvidenceMissingError` | Cross-member parameter read refused: no immutable capture-time evidence |
| `tl.errors.ParamReleasedError` | Raised when a released Param cannot re-fetch its live model parameter |
| `tl.errors.ReceptiveFieldConfigurationError` | Raised when a receptive-field call is misconfigured by its caller |
| `tl.errors.ReceptiveFieldError` | Base class for receptive-field failures |
| `tl.errors.ReceptiveFieldUnavailableError` | Raised when a receptive-field result requires unavailable capture state |
| `tl.errors.ReceptiveFieldValidationError` | Raised when receptive-field cross-validation detects a violation |
| `tl.errors.RenderError` | Raised when a transformer picture cannot be honestly produced |
| `tl.errors.ReplayValueMismatchError` | A recomputed selected activation differed from its archived bytes |
| `tl.errors.SidecarError` | Raised for sidecar-family registration and payload refusals |
| `tl.errors.TensorSubclassUnsupportedError` | A strict Tensor subclass could not be constructed while a mode was active here |
| `tl.errors.TraceCaptureError` | Raised by every capture door when the forward raises a non-TorchLens exception; a `CaptureError` chained `from` the original (`exc.__cause__`, traceback intact, never mutated). `TraceCaptureError.partial_trace`: the `PartialTrace` captured before the failure (or `None`); `TraceCaptureError.origin`: the outcome's `FailureOrigin`. `fields['code']` is `forward_raised` (origin `user_op` or `unknown`) or `capture_internal_error` (origin `torchlens`); `fields` also carries `cause_type`, `last_captured_op` and `num_captured_ops`. TorchLens's own errors propagate unwrapped and carry `partial_trace` too |
| `tl.errors.TraceMutationWarning` | First root-log mutation; Trace mutators operate in place |
| `tl.errors.TransformContractError` | Raised when a transform spec, chain, or plan violates its contract |
| `tl.errors.UserTransformError` | Raised when activation_transform or grad_transform raises |
| `tl.errors.quiet` | `tl.errors.quiet(*codes)`: the one switch that silences TorchLens warnings. `quiet()` silences every TorchLens warning; `quiet("code", ...)` silences only the coded warnings whose `fields["code"]` is listed. A bare call lasts for the session; `with tl.errors.quiet(...):` restores the prior state and warning filters at the end of the block. A later explicit `warnings.simplefilter` takes precedence |
| `tl.experiment.sweep_sites` | Ablate every candidate site, one member per candidate, baseline first |
| `tl.export.describe_target` | Return one export target's registration metadata (tier row included) |
| `tl.export.list_targets` | Return the registered export-target names |
| `tl.export.register_target` | Register one export target through the public door |
| `tl.export.resolve_target` | Return one registered exporter callable; unknown names refuse teaching |
| `tl.export.unregister_target` | Remove one export-target registration |
| `tl.fingerprint.StructuralHashMismatchError` | Raised when a model's provisional structural hash differs from its pin |
| `tl.fingerprint.content` | Return a deterministic SHA-256 digest of nested tensor content |
| `CheckReport.counters` | Event counters the checks session accumulated |
| `CheckReport.finalized_reason` | Why the report was finalized, such as scope exit |
| `CheckReport.ledgers` | The clip, scale and watchdog snapshots collected into the report |
| `ClipLedger.observe_step` | Record one accepted step's post-clip total norm, and the pre-clip norm when the magnitude tier is armed |
| `LivenessProbe.facts` | Per-site liveness statistics recorded so far |
| `LivenessProbe.hook_plan` | The `hooks=` mapping to pass to `CaptureOptions` |
| `RangeProbe.hook_plan` | The `hooks=` mapping to pass to `CaptureOptions` |
| `RangeProbe.observed_fires` | Per-label fire counts, so a site that never fired is visible |
| `Watchdog.on_accepted` | Reset on an accepted step and re-arm the single warning |
| `Watchdog.on_backward` | Record one backward pass; returns `True` exactly once, when the watchdog trips |
| `tl.grad` | Address the gradient of a site's output: `tl.grad("relu_3")`; builds a `GradSelector` |
| `tl.intervention.BundleComparisonError` | The same class as `tl.errors.BundleComparisonError` |
| `tl.intervention.Population` | A prepared, provenance-stamped donor population: `members`, `datums`, `origin`, `kind`, `content_digest`, `digest_kind`, `identity`, `eligible_indices(...)`, `mean(dim=None, ...)`, `std(dim=None, ...)`; built by `tl.intervention.population` |
| `tl.intervention.SamplingRecord` | One realized draw of a stochastic edit, exported beside `FireRecord` (a `FireRecord.sampling` entry): a frozen record with `verb`, `base_seed`, `derived_seed`, `share_draw`, `logical_firing_coordinate`, `eligible_count`, `population_count`, `population_identity`, `donor_group_id`, `draw_indices` and `realized_draw_digest`; the field rows below define each |
| `tl.intervention.label_contains` | Create a label-substring selector |
| `tl.intervention.label_matches` | Create a label regex-pattern selector |
| `tl.intervention.model_input` | Create a nested model-input path selector |
| `tl.intervention.model_output` | Create an output selector |
| `tl.intervention.population` | Build a `Population` from a tensor, a sequence of tensors or a sequence of Traces, with a required `origin=` provenance string |
| `tl.options.HtmlReportOptions` | Options for emitting the offline HTML report (thumbnail budgets, determinism, share-safe stripping) |
| `tl.receptive_field.GradReceptiveField` | Empirical receptive-field influence set measured by autograd |
| `tl.report.DiffReport` | The same class as `tl.differential.DiffReport` |
| `tl.report.InterventionValidationReport` | Five-axis intervention validation result |
| `tl.report.KernelProfileReport` | One door call: one execution, one trace, one joined result |
| `tl.report.MergeReport` | Graph-free diagnostic summary of how rank captures were merged |
| `tl.report.ValidationDiagnostic` | Structured non-failing diagnostic emitted by validation |
| `tl.report.ValidationFailure` | A structured description of the first surfaced validation mismatch |
| `tl.report.ValidationStatus` | Machine-readable replay-validation availability or result |
| `tl.report.compare_topology` | Compare two Traces structurally |
| `tl.report.diff` | The same function as `tl.differential.diff` |
| `tl.report.DoctorReport` | Environment and optional model compatibility report; records are constructed by TorchLens |
| `tl.report.DoctorCheck` | One diagnostic row; constructor keywords are outside the public contract |
| `tl.report.doctor` | Inspect the environment always, and a model/input tree when supplied; never executes the supplied model |
| `tl.report.kernel_profile` | Run one save-nothing capture under the owned session and join it |
| `tl.trackers.HistoryReader` | Read one artifact without loading the run or importing a model |
| `tl.trackers.correct_running_stats` | Apply the closed-form unscale correction to one spine record |
| `tl.trackers.corrected_grad_block` | Return `block` with every scaled gradient observation corrected |
| `tl.trackers.running_stats_scalars` | Derive the portable scalar statistics from one spine record |
| `tl.transformer.ANNOTATION_KINDS` | The closed set of transformer annotation kinds: `descriptive_head_score`, `additive_logit_contribution`, `screening_estimate`, `intervention_effect` |
| `tl.transformer.AttentionView` | One attention pattern in canonical coordinates |
| `tl.transformer.CropInfo` | Crop disclosure: cropping never renormalizes |
| `tl.transformer.EpisodeCoordinates` | Episode coordinates (step and position) carried by transformer view records |
| `tl.transformer.MaskInfo` | Mask provenance for an attention view |
| `tl.transformer.PredictionTable` | Top-k tokens per position |
| `tl.transformer.PredictionTrajectory` | Per-layer prediction trajectory record |
| `tl.transformer.ScoreDecomposition` | Term-complete score decomposition for one (head, dst, src) pair |
| `tl.transformer.TokenAxis` | One labeled token axis |
| `tl.transformer.TokenMetrics` | Per-token metric strip record |
| `tl.transformer.TokenScoreRow` | One labeled row of per-token scores (strip row) |
| `tl.transformer.TokenScores` | Colored-token strip record, single- or multi-row |
| `tl.transformer.attention_view` | Extract one layer's attention pattern as a typed view; `attention_view(generation, layer, step=k)` reads step `k` of a recorded generation |
| `tl.transformer.attention_views` | Extract every (or the named) attention layer's view, in trace order |
| `tl.transformer.bertviz_tuple` | Return attention in the BertViz and Hugging Face `output_attentions` tuple format |
| `tl.transformer.circuitsvis_attention` | Hand the payload to circuitsvis's attention component (notebook only) |
| `tl.transformer.circuitsvis_payload` | Build the CircuitsVis attention payload from typed views |
| `tl.transformer.metric_strip` | Convert a metric record to a strip record for rendering |
| `tl.transformer.prediction_table` | Build the per-position top-k table from the native lens row |
| `tl.transformer.prediction_table_html` | Emit the zero-dependency HTML top-k table |
| `tl.transformer.prediction_trajectory` | Build the trajectory record from a streaming lens result |
| `tl.transformer.receipt_from_patch_grid` | Build a validated receipt from a mech-interp patch grid row |
| `tl.transformer.render_answer_trajectory` | Render the tracked answer token's probability + rank panels |
| `tl.transformer.render_neuron_card` | Render the selected-pair neuron card |
| `tl.transformer.render_prediction_ribbon` | Render the layer-prediction ribbon |
| `tl.transformer.score_decomposition` | Decompose one attention score into its per-dimension terms |
| `tl.transformer.token_metrics` | Compute the loss and entropy strips from ONE logits source |
| `tl.transformer.token_strip_html` | Emit the zero-dependency, self-contained HTML token strip |
| `tl.types.Accessor` | Generic dict-like accessor for ordered TorchLens log objects |
| `tl.types.ContainerLeafOccurrence` | One tensor occurrence at one container path |
| `tl.types.ContainerSpec` | Portable description of an output container seen during capture |
| `tl.types.ContractCheck` | One contract check recorded in a run report |
| `tl.types.InputSpec` | Explicit per-slot input override for the quickstart input grammar |
| `tl.types.MergedFinding` | One structured merge finding |
| `tl.types.NarrationEvent` | One immutable narration record |
| `tl.types.NumericAttestationStatus` | Result states for the optional saved-activation oracle |
| `tl.types.PathFaithfulness` | Honesty states reported by a completed sparse run |
| `tl.types.ReadinessReport` | Complete non-executing readiness report for a Trace provider |
| `tl.types.ReadinessStatus` | Overall non-executing runnable readiness states |
| `tl.types.Relationship` | The derived relationship between two Bundle members, returned by `Bundle.comparability` and `comparability_table` (persisted bundle data; values unchanged) |
| `tl.types.RunProvider` | Providers selected by the `Trace.run` contract |
| `tl.types.RunReport` | Honesty, state, resolution, and contract metadata for one run |
| `tl.types.RunnableCompatibility` | Producer/runtime versions recorded for readiness diagnostics |
| `tl.types.RunnableDiagnostic` | Structured diagnostic shared by preflight, readiness, and run reports |
| `torchlens._user_funcs.CaptureNotActiveError` | The same class as `tl.errors.CaptureNotActiveError` |
| `torchlens._user_funcs.UserTransformError` | The same class as `tl.errors.UserTransformError` |
### Members
| Spelling | Meaning |
| --- | --- |
| `TraceSource.annotated` | Return captured root forward source as numbered text with captured op labels beside each producing line; nested ops use their root forward caller line, synthetic rows are omitted; refuse with `source_unavailable` when no source snapshot exists |
| `tl.mechinterp.visualize_feature` | DOCUMENTED-UNSTABLE: maximize a channel, indexed unit or SAE latent at one module/op site through the shared binding; returns a detached optimized input and `steps + 1` objective floats as a plain tuple; one floating input, unconstrained Adam ascent |
| `source_unavailable` | Typed `InvalidArgumentError` code: no captured root forward source; re-capture a model defined in a source file and save with `include_source=True` |
| `feature_visualization_invalid` | Typed `InvalidArgumentError` code: invalid optimization settings, ambiguous/missing tensor site, or disconnected/non-finite objective; use one floating input and tensor site, valid unit, positive steps and finite lr |
| `Accessor.labels` | The lookup label of every record the collection lists, in order |
| `Accessor.count_by` | Count records by one or more record fields |
| `ModuleAccessor.uncalled` | Addresses of registered modules the captured pass never called |
| `Accessor.search` | Return records whose lookup key contains `needle` |
| `TraceOpAccessor.compute` | The members that are not input, output or buffer rows |
| `TraceOpAccessor.nonfinite` | The members whose output held NaN or Inf |
| `AmbiguousBackendError.default_remedy` | The remedy the error names when none is passed |
| `BackendCapabilities.can_intervene` | Whether live intervention capture is supported (was `interventions`) |
| `BackendCapabilities.has_validation_replay` | Whether the backend can perform replay validation |
| `BackendSpec.maturity` | The backend's launch label, `"preview"` or `"stable"` |
| `BackendSpec.protocol` | Lazy factory returning the backend's `BackendProtocol` object |
| `BackwardEstimate.is_freeze_pattern_captured` | Whether the parameter freeze pattern was captured |
| `BackwardEstimate.is_grad_disabled_at_capture` | Whether grad was disabled when the forward was captured |
| `BackwardEstimate.num_frozen_weight_ops` | Ops whose weight gradient is skipped because the weight is frozen |
| `BackwardEstimate.num_multiplied_ops` | Ops whose backward cost the estimate multiplies from the forward cost |
| `BackwardNanReport.amp_hint` | The GradScaler hint when every finding is an `inf` and no `grad_scale` was given; else `None` |
| `BackwardNanReport.backward_pass` | The one-based backward pass walked |
| `BackwardNanReport.birth` | The earliest provable birth transition (finite in, non-finite out), or `None` |
| `BackwardNanReport.checked_calls` | Calls with saved, checkable gradient payloads |
| `BackwardNanReport.first_bad_thing` | Project this result into the shared first-bad-thing vocabulary |
| `BackwardNanReport.found` | Whether any non-finite gradient was observed in the walked pass |
| `BackwardNanReport.healed` | Labels of transitions where a non-finite arrival was erased (non-finite in, finite out) |
| `BackwardNanReport.message` | Human-readable summary headlining the earliest provable birth |
| `BackwardNanReport.transitions` | The full transition ledger in measured fire order |
| `BackwardNanReport.unchecked_calls` | Calls without saved or checkable gradient payloads |
| `BackwardNanReport.verdict_scope` | `"complete"` when every earlier call was checkable, else `"found_first_among_checked"` |
| `BackwardPass.pass_number` | 1-based number of this backward pass |
| `BackwardStatus.num_backward_passes` | Number of backward passes observed |
| `BackwardStatus.num_grad_fn_records` | Number of GradFn records the status read |
| `BackwardStatus.uncounted_op_counts` | Every recorded op kind outside the counted families with its call count, never zeroed or estimated |
| `BasisProvenance.num_vectors` | Number of basis vectors |
| `BindReport.is_grad_enabled` | Whether grad mode was on when the bound run executed |
| `BindReport.is_model_training` | Whether the model was in training mode when the bound run executed |
| `BindReport.num_fires` | How many times the bound intervention fired |
| `BisectNanReport.first_bad_thing` | Project this result into the shared first-bad-thing vocabulary |
| `BisectNanReport.found` | Whether a saved activation containing NaN or Inf was found |
| `BisectNanReport.message` | Actionable human-readable result summary |
| `BisectPrecisionReport.found` | Whether any compared op diverged beyond tolerance |
| `BisectPrecisionReport.max_abs_err` | First divergent op's maximum absolute error, or `None` |
| `BisectPrecisionReport.max_rel_err` | First divergent op's maximum relative error, or `None` |
| `BisectPrecisionReport.message` | Human-readable summary |
| `Bundle.comparability` | Return the derived relationship for a pair of members (`tl.types.Relationship`) |
| `Bundle.comparability_table` | Return pairwise relationships for all member pairs |
| `Bundle.map` | Apply a function independently to each member; released effect-table candidates read `"unmeasured"`. `fn(member)`; the member name is passed only where `fn` takes it: positionally to a required second parameter or `*args` (`lambda log, name: ...`), by keyword to a parameter called `name`; a defaulted second parameter (`lambda log, k=3: ...`) keeps its default |
| `Bundle.compare` | Per-site differences between two members as `(site_label, distance)` rows; `metric=` and `on=`. Row keys: a multi-pass op gives one row per pass, keyed by its pass-qualified label (`tanh_1_1:2`); a single-pass op keeps its bare layer label (JMT, Sun 1 Nov 2026) |
| `Bundle.rank_changes` | Rank sites by their distance from a baseline member; rows keyed like `Bundle.compare` (one row per pass of a multi-pass op); `by="module"` folds them into one row per module address, scored by its most changed site (every module a site ran inside gets a row, so `transformer.h.9` and `transformer.h.9.mlp` both appear) |
| `Bundle.summary` | The bundle card as text: member count and names, outcome distribution, baseline, relation count, structural consistency |
| `Bundle.to_pandas` | One stacked table: a leading `member` column, then each member's `Trace.to_pandas()` rows in `names` order with a fresh index; an empty bundle gives an empty frame with the `member` column |
| `Bundle.to_agent_json` | The agent dump (schema `"torchlens.agent_bundle.v1"`): `member_count`, `names`, `baseline_name`, `outcome`, `structurally_consistent` and `members` (each member's `Trace.to_agent_json(max_ops=)`) |
| `BundleComparisonError.severity` | The error's severity class |
| `CapabilityRow.is_available` | Whether the capability is available in this environment |
| `CaptureFacts.is_structure_only` | Whether the capture kept structure only, without payloads |
| `CaptureFacts.is_verified` | Whether the capture was verified (`None` when not checked) |
| `CausalReceipt.num_measured_heads` | Return how many displayed heads carry a measured effect |
| `CheckFinding.check_name` | Name of the check that produced the finding |
| `CheckFinding.param_names` | The offending parameter or site names, with every alias of a tied tensor kept |
| `CheckFinding.accepted_step_id` | Identifier of the accepted optimizer step the finding belongs to, or `None` |
| `CheckFinding.attempted_step_id` | Identifier of the attempted step the finding belongs to, or `None` |
| `CheckFinding.backward_id` | Identifier of the backward pass the finding belongs to, or `None` |
| `CheckFinding.collateral_zeroed` | Count of values zeroed as collateral of the problem, or `None` |
| `CheckFinding.follow_up` | The call that localizes the culprit the finding only narrows to a cone |
| `CheckFinding.global_step` | The training step the finding was recorded at, or `None` |
| `CheckFinding.scale_provenance` | Where the grad-scaler scale in the finding came from, or `None` |
| `CheckFinding.step_provenance` | How the step number was known: observed or inferred |
| `CheckFinding.zero_baseline` | Whether the value was compared against a zero baseline, or `None` |
| `ChecksSession.register_nonfinite_grad_check` | Register the live parameter-gradient nonfinite check |
| `ChecksSession.attach` | Install the scheduled gradient, optimizer and step hooks |
| `ChecksSession.detach` | Remove every hook handle; safe to call more than once |
| `ChecksSession.disable` | Suspend all checks until `enable` is called |
| `ChecksSession.enable` | Re-enable checks after `disable` |
| `ChecksSession.estimated_snapshot_bytes` | Clone-tier snapshot budget estimated from element count and dtype size alone |
| `ChecksSession.final_report` | The report finalized at scope exit, when the session was entered as a context manager |
| `ChecksSession.paused` | Context manager that suspends all checks inside its block |
| `ChecksSession.register_change_check` | Register the parameter-change fact check |
| `ChecksSession.register_frozen_check` | Register the declared-frozen-parameter invariant check |
| `ChecksSession.register_magnitude_check` | Arm the pre-clip gradient magnitude check |
| `ChecksSession.register_nonfinite_param_check` | Register the scheduled parameter and buffer nonfinite scan |
| `ChecksSession.register_update_ratio_check` | Register the update-to-weight ratio check |
| `ClipLedgerSnapshot.has_constancy_hint` | Whether the clip ledger saw a constant clipping pattern worth hinting |
| `ClipLedgerSnapshot.accepted_steps_observed` | Number of accepted steps the clip ledger observed |
| `ClipLedgerSnapshot.applied_clip_factors` | The exact applied clip factor per step, when the magnitude tier was armed |
| `ClipLedgerSnapshot.constant_total` | The post-clip total norm when it stayed constant, else `None` |
| `ClipLedgerSnapshot.declared_clip_norm` | The clip norm the caller declared, or `None` |
| `ClipLedgerSnapshot.nonfinite_totals` | Count of steps whose total norm was nonfinite |
| `ClipLedgerSnapshot.saturated_steps` | Count of steps clipped to the declared norm |
| `ClipLedgerSnapshot.saturation_rate` | Fraction of observed steps that saturated the declared clip norm, or `None` |
| `ClipLedgerSnapshot.tier` | Evidence tier of the clip facts: exact factors, declared norm, or constancy hint |
| `CloseReport.is_disabled` | Whether the watcher was disabled |
| `CloseReport.is_final_sample_forced` | Whether close forced a final sample |
| `CostTreeRow.num_unknown_ops` | Directly-owned ops with no cost rule |
| `DeterminismReport.first_problem` | Project the first divergence into the shared vocabulary |
| `DiagnosticFinding.check_name` | Check name (`"nonfinite"`, `"dtype_regime"`, `"layout"` or `"confinement"`) |
| `DiffReport.rows_with_status` | Return the rows carrying one closed-vocabulary status |
| `DiffRow.pass_number` | Pass number of the compared op |
| `DoctorCheck.details` | Diagnostic explanation |
| `DoctorCheck.is_detected` | Detected condition; environment rows use status other than pass |
| `DoctorCheck.key` | Stable row key; environment keys retain probe names except torch_capabilities |
| `DoctorCheck.label` | Readable diagnostic title |
| `DoctorCheck.severity` | ok, info, warning, or error |
| `DoctorCheck.status` | pass, known_broken, scope, or not_tested |
| `DoctorCheck.suggestion` | Remedy text or an empty string |
| `DoctorReport.capability_snapshot` | Full named runtime capability mapping |
| `DoctorReport.check` | Return one diagnostic row by stable key; raises KeyError for an unknown key |
| `DoctorReport.checks` | Ordered tuple of environment and optional model checks |
| `DoctorReport.model_type` | Qualified model type, or None for an environment-only report |
| `DoctorReport.show` | Print and return the fixed-width table; str(report) returns the same text |
| `DoctorReport.to_markdown` | Return the diagnostic table as Markdown |
| `DoctorReport.torch_version` | Installed PyTorch version, always set |
| `DryRun.filter` | Return a new trace with decisions from a new op predicate |
| `Edit.is_compatible_with_append` | Whether the edit keeps the output shape, so an append rerun may stack it with other edits |
| `Edit.name` | The edit's public name (the builder that made it, such as `zero_ablate`) |
| `Edit.signal` | Pass an edit is bound to: `"forward"` for edits that read recorded activations, `None` for neutral edits |
| `EffectsView.rank_changes` | Rank candidates by the size of their change from baseline, descending |
| `EffectsView.to_grid` | Rebuild the `SweepGrid` of a `sweep_grid` operation from its effect table and recorded axes |
| `EffectsView.to_selection` | Convert the view to a selection: the union of its WHERE regions |
| `EpisodeOptions.acknowledge_step_cost` | Accept a step declaration beyond the cost ceiling |
| `EpisodeOptions.crossings` | Declared exogenous crossings: step indices whose inputs come from outside the capture |
| `EpisodeOptions.escalated_from` | Digest of the cheaper capture this capture escalates, when escalating |
| `EpisodeOptions.expected_tokens` | The cheaper capture's per-step tokens, carried so the escalated capture can check fidelity |
| `EpisodeOptions.feed` | Join strictness: `"open"` (default) captures across feed breaks and grades every join; `"closed"` holds joins strict |
| `EpisodeOptions.forced_tokens` | Teacher-forced token sequence fed instead of model emissions |
| `EpisodeOptions.on_feed_break` | What a feed break does: `"disclose"` (default) returns the trace with the break disclosed; `"refuse"` raises |
| `EpisodeOptions.rng` | Seeding discipline; only `"managed"` ships |
| `EpisodeOptions.step_input_from` | Which stepped-call argument carries the step input: a position, a keyword name, or `None` |
| `FindNanReport.bad_tensors` | Tensor positions found non-finite (`("output",)` for an eager finding) |
| `FindNanReport.first_bad_thing` | Project this result into the shared first-bad-thing vocabulary |
| `FindNanReport.found` | Whether a NaN or Inf was found |
| `FindNanReport.label_status` | Whether the reported label is final (`"final"`) or absent (`"none"`) |
| `FindNanReport.message` | Human-readable diagnostic summary |
| `FireRecord.backward_pass_number` | Backward pass the edit fired in, or `None` for a forward fire |
| `FireRecord.edit` | The `Edit` that fired, or `None` for a raw hook |
| `FireRecord.edit_name` | Name of the edit or hook that fired |
| `FireRecord.sampling` | Realized draws of the stochastic verbs during this fire: a tuple of `SamplingRecord`, saved with the trace (`determinism_note` keeps the readable line) |
| `FireRecord.signal` | Pass the fire happened in: `"forward"` or `"backward"` |
| `GqaInfo.num_query_heads` | Number of query heads |
| `GradFnCall.fire_record` | The fire record of the edit that produced this grad-fn call, or `None` |
| `GradFnCall.grad_fn_label` | Label of the grad_fn this call belongs to |
| `GradFnCall.label` | Pass-qualified call label (`<grad_fn_label>:<call_index>`) |
| `GradFnSketch.num_nodes` | Number of walked nodes |
| `GradReceptiveField.batch_support` | Batch indices the influence set reaches, or `None` |
| `GradReceptiveField.cross_batch_influence` | Whether the influence crosses batch elements |
| `GradReceptiveField.io_role` | The far end of the probe, the space `support_mask` covers: the source (model input name or op label) for a receptive probe, the target op label for a projective probe; `source_key` / `target_key` read it direction-stably |
| `GradReceptiveField.num_nonfinite` | Number of NaN or Inf gradient entries; nonzero makes validation indeterminate |
| `GradReceptiveField.rtol` | Relative tolerance for the support threshold |
| `GradReceptiveField.signed_grad` | The signed gradient, kept so a sign-disagreeing adjoint can be checked |
| `GradReceptiveField.source_key` | Return the direction-stable key of the source graph point |
| `GradReceptiveField.spatial_support_mask` | The support mask reduced to the spatial axes, when the layout has them |
| `GradReceptiveField.support_mask` | Boolean mask of input elements above the tolerance threshold |
| `GradReceptiveField.support_ranges` | Per-axis index ranges of the support, or `None` |
| `GradReceptiveField.target_key` | Return the direction-stable key of the target graph point |
| `GradReceptiveField.unit_shape` | Shape of the target unit |
| `GradReceptiveField.warnings` | Warnings raised while measuring |
| `GradRecordContext.pass_number` | Pass number of the op whose gradient is recorded |
| `GridCell.candidate_id` | The cell id: the comma-joined value names in axis order, also the retained member's name |
| `GridCell.coords` | Axis name to value name for this cell |
| `GridCell.error` | The refusal (`grid_cell_unresolved`, `grid_cell_multi_site`) or failure text; None otherwise |
| `GridCell.member_name` | The retained member's name, None when the cell holds no member |
| `GridCell.status` | `completed`, `released`, `refused` or `failed` |
| `GridCell.value` | The metric value, None for a refused or failed cell |
| `GuessInputShapeReport.attempts` | Probe diary as `(shape, outcome)` tuples |
| `GuessInputShapeReport.constraining_module` | Module qualname that constrained the inferred shape, when known |
| `GuessInputShapeReport.constraining_op` | Operation name that constrained the inferred shape, when known |
| `GuessInputShapeReport.dtype` | Input dtype for the primary tensor |
| `GuessInputShapeReport.example_input` | Ready-to-use input object |
| `GuessInputShapeReport.flexible_dims` | Dimensions that are expected to tolerate other sizes |
| `GuessInputShapeReport.found` | Whether a verified input was found |
| `GuessInputShapeReport.message` | Actionable human-readable summary |
| `GuessInputShapeReport.reason` | Failure reason when `found` is false |
| `GuessInputShapeReport.shapes` | Multi-input shapes, when applicable |
| `GuessInputShapeReport.source_line` | Source line for the constraining op, when known |
| `GuessInputShapeReport.strategy` | Strategy that produced the winning input |
| `GuessInputShapeReport.value_range` | Synthetic value recipe: `("uniform", 0, 1)` or `("randint", 0, vocab)` |
| `HealthFacts.num_nonfinite_ops` | Nonfinite ops in the identity partition (alias rows excluded) |
| `HealthFacts.num_unmapped_ops` | Ops the health partition could not map |
| `Histogram.compute` | Return the finalized histogram counts (`HistogramResult`) |
| `HistogramPoint.bin_edges` | Histogram bin edges for the point's counts |
| `HistoryReader.num_chunks` | Number of committed chunks |
| `HookContext.run_state` | Run-scoped scratch dict shared by every hook of one run |
| `HookContext.signal` | Pass the hook is firing in: `"forward"` or `"backward"` |
| `HookContext.state` | Per-(site, hook) scratch dict |
| `HtmlReportOptions.deterministic` | Suppress the manifest timestamp for byte-exact regeneration |
| `HtmlReportOptions.max_array_bytes` | Thumbnail byte budget, enforced before payloads are written |
| `HtmlReportOptions.max_array_sites` | Thumbnail site budget, enforced before payloads are written |
| `HtmlReportOptions.share_safe` | Strip every embedded value |
| `InterventionValidationReport.is_complete` | Whether the validation exercised all requested axes |
| `InterventionValidationReport.is_consistent` | Whether repeated validation decisions agree |
| `InterventionValidationReport.is_invariant` | Whether a baseline forward validation succeeds |
| `InterventionValidationReport.is_local` | Whether intervention checks stay local to the requested model/input |
| `InterventionValidationReport.is_specific` | Whether hook targets can be represented without ambiguity |
| `InterventionWarning.severity` | The warning's severity class (`"recoverable"` by default) |
| `KernelProfileReport.availability` | Profiler session availability in five states (missing is `None`, never zero) |
| `KernelProfileReport.device_time_ns` | Exact-attribution device nanoseconds for one op label |
| `KernelProfileReport.device_time_table` | Per-op exact device time, descending; refuses when unjoined |
| `KernelProfileReport.join` | The profiler join result that maps kernels to ops |
| `KernelProfileReport.session_facts` | Facts about the profiler session |
| `Layer.is_edited` | Whether a user edit replaced the output of any pass of this layer |
| `Layer.short_label` | Short pass-qualified label of a single-pass layer |
| `Layer.short_layer_label` | Short layer label (no pass suffix) |
| `Layer.source_trace` | Back-reference to the owning Trace; a detached Layer refuses instead of returning `None` |
| `LineageReport.message` | Human-readable status |
| `LineageReport.start_label` | Resolved starting op label, or the requested label when lookup failed |
| `MachineBalance.peak_bytes_per_s` | Peak memory bandwidth used for the roofline, in bytes per second |
| `MachineBalance.peak_flops_per_s` | Peak compute rate used for the roofline, in FLOPs per second |
| `MemoryFacts.captured_bytes` | Bytes the capture retained when it ran (an immutable capture fact) |
| `MemoryFacts.captured_saved_ops` | Ops whose payload the capture saved |
| `MemoryFacts.retained_bytes` | Bytes this object holds now (present or lazily loadable payloads); zero on a payload-stripped artifact |
| `MemoryFacts.retained_lazy_ops` | Ops whose payload this object can load lazily |
| `MemoryFacts.retained_present_ops` | Ops whose payload is present in memory |
| `MergeReport.degradations` | Environment degradations recorded for the merge |
| `MergeReport.num_joins` | Number of cross-rank joins made |
| `MergedFinding.from_dict` | Rebuild a finding from `to_payload` output |
| `MergedTrace.at` | Return the SPMD fan: the ops matching `label` on every rank |
| `MergedTrace.degradations` | Environment degradations recorded at load (empty on a live merge) |
| `ModuleCall.has_pre_hook_input_change` | Whether a user forward pre-hook changed the call's effective inputs (`None` when evidence is incomplete) |
| `ModuleCall.label` | Pass-qualified module call label (`address:call_index`) |
| `ModuleCall.parent_label` | Label of the parent ModuleCall in the dynamic call tree |
| `ModuleInputSnapshot.is_capture_complete` | Whether all supported mutation evidence was available for the snapshot |
| `ModuleStackFrame.pass_number` | Call number of this module in the stack |
| `NarrationEvent.has_intervention` | Whether an intervention acted at this event |
| `NarrationEvent.value_stats` | `tl.stats.TensorStats` for this tensor, or `None` when no statistics were requested or available |
| `NarrationEvent.tier` | Capture tier the event came from, such as `post_hoc`, or `None` |
| `Op.edits` | The FireRecords of every edit that fired at this op, in fire order (was `interventions`) |
| `Op.is_edited` | Whether a user edit replaced this op's output |
| `Op.pass_number` | 1-based pass of this op within its layer |
| `Op.short_label` | Short pass-qualified op label |
| `Op.short_layer_label` | Short layer label (no pass suffix) |
| `Op.source_trace` | Back-reference to the owning Trace; a detached Op refuses instead of returning `None` |
| `Op.type` | The op's normalized operation type token (for example `"linear"`); `Layer.layer_type` is the Layer's spelling |
| `OptimizerFacts.is_known` | Whether the optimizer is recognised; an unknown optimizer gives `band_reason` instead of a guessed band |
| `OptimizerFacts.decoupled_decay` | Whether the optimizer applies weight decay decoupled from the gradient step |
| `OptimizerFacts.group_fact` | One per-parameter-group hyperparameter fact, or `None` when unknown |
| `OptimizerFacts.param_group_facts` | The per-parameter-group facts read from the optimizer |
| `PreHookEffect.has_changes` | `True`/`False` when the pre-hook's mutation evidence is complete, otherwise `None` |
| `PreprocessingComparison.applied_source` | How the applied side was obtained: `"resolution"` / `"declaration"` / `"parsed_transform"` / `"undeclared"` / `"opaque_transform"` |
| `PreprocessingComparison.authority_status` | The authority's standing (`authoritative` / `unverified_fallback` / `unknown`); a non-authoritative reference caps the verdict below `verified` |
| `PreprocessingComparison.mismatched_fields` | Names of every mismatched field, in field order |
| `PreprocessingComparison.unknown_fields` | Names of every uncomparable field, in field order |
| `PreprocessingComparison.unknown_reasons` | Deduplicated reasons behind every unknown field, in field order |
| `PreprocessingComparison.verdict` | `"verified"` / `"mismatch"` / `"unknown"` |
| `ProfileReport.capture_verification_reason` | Why the capture is unverified when `is_capture_verified` is `False`, else `None` |
| `ProfileReport.has_rescue_rerun` | Whether the profile needed a rescue rerun |
| `ProfileReport.is_structure_only` | Whether the profiled capture kept structure only |
| `ReceptiveField.axis_roles` | Axis layout of the input grid: per-axis kinds, windowed axes and their provenance |
| `ReceptiveFieldValidation.num_violations` | Number of cross-validation violations found |
| `ReceptiveFieldView.axis_roles` | Axis layout of the viewed op's input grid |
| `ReceptiveFieldView.of` | Many-unit influence region as a trace-bound `tl.Selection`, one region per `within=` site |
| `ReceptiveFieldView.probe_grad` | Empirical gradient probe of one unit's receptive field |
| `ReceptiveFieldView.status` | Field status of the only reachable endpoint, or a read-only mapping from each endpoint's IO role to its status when several are reachable (a projective view of a GPT-2 activation reaches the logits and every later block's cached keys and values) |
| `Recipe.address_class` | Most restrictive address class over all rules (worst wins) |
| `Recipe.decision` | The sole rule's normalized decision (`None` on multi-clause specs) |
| `Recipe.door_pairs` | Lower the spec to `(site, action)` pairs for the hook doors |
| `Recipe.match` | Return the single rule matching one op context, or `None` |
| `Recipe.save` | Write the rules alone as a `kind: "recipe"` `.tlspec` that `tl.load` returns as a Recipe |
| `RecordContext.module_pass_number` | Call number of the innermost module, or `None` |
| `RecordContext.pass_number` | Pass number of the recorded op |
| `Recording.num_predicate_failure_overflows` | Predicate failures beyond the kept list (counted, not stored) |
| `RooflineRow.is_cache_resident_hint` | Whether the row's working set likely fits in cache (a hint, not a measurement) |
| `Rule.action_repr` | Canonical rendering of the action over its current tensor content |
| `Rule.address_class` | How the rule addresses its sites (value-dependent by default) |
| `Rule.decision` | The rule's normalized decision, or `None` |
| `Rule.payload_fidelity` | Audit fidelity of this rule's action payload |
| `Rule.rule_id` | Content-derived rule identity over WHERE and action |
| `Rule.signal` | Pass the rule edits (`"forward"`, `"backward"` or `"both"`), derived from its target |
| `Rule.where_repr` | Canonical rendering of the rule's WHERE clause |
| `RunReport.has_carried_state` | Whether mutations were deliberately left on the live model; the default run restores declared state and reports `False` |
| `RunReport.is_poisoned` | Whether the run left the provider unusable for further runs |
| `RunnableDiagnostic.is_analysis_load_available` | Whether the artifact can still load for analysis only |
| `SamplingRecord.base_seed` | The plan's declared or capture-derived base seed; `None` for an averaging verb |
| `SamplingRecord.derived_seed` | The stateless per-firing seed derived from the base seed, population identity, donor group and coordinate |
| `SamplingRecord.donor_group_id` | Donor-sharing group of the firing (`per_group` plans share one draw across sites) |
| `SamplingRecord.draw_indices` | The full realized index vector: donor member ids, or the row permutation of `permute_batch`; empty when nothing was drawn |
| `SamplingRecord.eligible_count` | Members eligible for the draw after agreement-class filtering |
| `SamplingRecord.logical_firing_coordinate` | Where the draw fired: the logical site, pass and row coordinate the seed derivation used |
| `SamplingRecord.population_count` | Size of the population drawn from |
| `SamplingRecord.population_identity` | Digest identity of the population |
| `SamplingRecord.realized_draw_digest` | Short SHA-256 of the realized index vector |
| `SamplingRecord.share_draw` | Sharing rule of the plan: `per_firing`, `per_rule` or `per_group` |
| `SamplingRecord.verb` | The stochastic verb that drew: `patch_from`, `resample_rows_from`, `permute_batch`, `set_direction_mean` or `mean_ablate` |
| `SavedFigure.disclosure_lines` | Every honesty line the artifact rendered, in order |
| `SavedFigure.num_pages` | Total page count (equals `len(paths)` for file output) |
| `SavedFigure.svg_fonttype` | The matplotlib `svg.fonttype` used (`"path"` by default) |
| `ScaleLedger.is_enabled` | Whether a live, enabled GradScaler is attached |
| `ScaleLedger.observe_accepted_step` | Record one accepted optimizer step |
| `ScaleLedger.observe_backward` | Record one backward's scale reading; count decrements exactly |
| `ScaleLedger.observe_scale` | Fold any unobserved grad-scaler movement into the ledger without counting a backward |
| `ScaleLedgerSnapshot.has_scaler` | Whether a GradScaler was attached |
| `ScaleLedgerSnapshot.is_enabled` | Whether the attached GradScaler was enabled |
| `ScaleLedgerSnapshot.accepted_steps` | Number of optimizer steps the scaler accepted |
| `ScaleLedgerSnapshot.backward_reads` | Number of backward passes observed |
| `ScaleLedgerSnapshot.current_scale` | The grad-scaler scale at snapshot time |
| `ScaleLedgerSnapshot.growth_events` | Number of scale growth events observed |
| `ScaleLedgerSnapshot.inferred_k` | The inferred growth interval, or `None` when it could not be inferred |
| `ScaleLedgerSnapshot.initial_scale` | The grad-scaler scale when observation began |
| `ScaleLedgerSnapshot.preconditions` | The preconditions the ledger checked before inferring scaler behaviour |
| `ScaleLedgerSnapshot.skipped_attempts` | Number of step attempts the scaler skipped |
| `StatsTableRow.value_stats` | `tl.stats.TensorStats` for this op on `state == "ok"` rows, or `None` otherwise (`state` names why the numbers are absent) |
| `ScanRow.is_audited` | False exactly when `skip_reason` says why the row was not audited |
| `ScanRow.is_constant` | Whether the scanned value stayed constant (`None` when unknown) |
| `ScanRow.abs_max` | Largest absolute value of the scanned tensor |
| `ScanRow.bounds_above` | Count of values above the declared upper bound |
| `ScanRow.bounds_below` | Count of values below the declared lower bound |
| `ScanRow.dtype_max` | Largest finite value the tensor dtype can hold |
| `ScanRow.subnormal_fraction` | Fraction of values in the subnormal range |
| `ScanRow.zero_fraction` | Fraction of values equal to zero |
| `Selection.align_to` | Re-bind a resolved selection onto another trace |
| `Selection.labels` | Pass-qualified op label of every row, in row order |
| `Site.record` | The `Op`, `Param` or `EdgeUseRecord` the row names, through a weak trace link |
| `Selection.selector` | The selector that produced the rows, or `None` |
| `Selection.to_pandas` | One row per site: label, kind, key, num_selected, numel, shape, relation, source |
| `Session.to_bundle` | Return a bundle containing all captured invocation logs |
| `Site.num_selected` | Return the number of selected elements at this site |
| `Site.structural_site_key` | The structural site key when present, else `None` |
| `SiteRow.pass_number` | Pass number of the site |
| `SixNDFacts.num_params` | Parameter count used by the 6ND estimate |
| `SpecCompat.has_identical_target_resolution` | Whether the spec resolves to the same targets on both sides |
| `SpliceResult.ce_loss_clean` | Mean next-token cross-entropy of the clean run when `tokens=` is given, else `None` |
| `SpliceResult.ce_loss_spliced` | Mean next-token cross-entropy with the SAE reconstruction spliced in, else `None` |
| `SpliceResult.kl_divergence` | Mean KL divergence from the clean to the spliced next-token distribution, else `None` |
| `SrpSizing.num_components` | Recommended number of projection components |
| `SrpSizing.num_samples` | Number of samples (rows) the projection width was sized for |
| `SummaryTotals.captured_bytes` | Bytes the capture retained when it ran |
| `SummaryTotals.retained_bytes` | Bytes this object holds now |
| `Super.absent_names` | Return bundle member names not represented by this view |
| `Super.is_complete` | Return whether every bundle member is represented |
| `SweepGrid.axes` | Axis name to value names, in declaration order |
| `SweepGrid.baseline_value` | The metric on the un-edited baseline |
| `SweepGrid.bundle` | The Bundle (baseline plus retained members) the grid's effect table lives on |
| `SweepGrid.candidate_id` | The cell id at the given coordinates (every axis by keyword) |
| `SweepGrid.cell` | The `GridCell` at the given coordinates |
| `SweepGrid.cells` | Every cell in product order |
| `SweepGrid.coords` | The coordinates of a cell id (the inverse of `candidate_id`) |
| `SweepGrid.deltas` | `values - baseline_value`; NaN stays NaN |
| `SweepGrid.edit_axis` | The axis holding edits, None when edit= served every cell |
| `SweepGrid.effects` | The `EffectsView` over the grid's effect table |
| `SweepGrid.failed` | Cell ids whose forward or metric raised |
| `SweepGrid.history_id` | The recorded `site_sweep` operation's id |
| `SweepGrid.index` | The integer index tuple into `values` at the given coordinates |
| `SweepGrid.refused` | Cell ids refused at preflight (never run) |
| `SweepGrid.to_pandas` | Long format, one row per cell (needs the `tabular` extra) |
| `SweepGrid.values` | float64 tensor shaped by the axes; NaN for refused or failed cells |
| `TapRecord.signal` | Pass the tapped value belongs to: `"forward"` or `"backward"` |
| `TensorInputObservation.mutation_version` | Captured PyTorch version counter of the input tensor (diagnostic, not portable) |
| `TensorInputObservation.python_object_id` | In-process Python object identity of the input tensor (diagnostic, not portable) |
| `TensorSpec.from_tensor` | Build a spec from a tensor (alternate constructor) |
| `Trace.armed_edits` | Read-only tuple of the gradient rules in the trace's recipe |
| `Trace.input` | The original user input before `input_transform` was applied (was `raw_input`) |
| `Trace.is_replayable` | Whether the capture ran with `CaptureOptions(replayable=True)` (the runnable apparatus: runnable save, edge provenance). Saved-graph replay (`do()` without `model=`) needs only the per-op argument templates every `tl.trace` capture records (`trace.options.save_arg_templates`), so it does not need this flag |
| `Trace.is_stopped` | Whether the capture stopped early (`until=`/`stop_after=`); a stopped trace is partial (was `halted`) |
| `Trace.options.seed` | The torch seed the capture ran under (`seed=` on the capture door), or `None`; saved with the trace |
| `Trace.transformed_output` | Human-readable model output after `output_transform` (was `raw_output`); after a replay rewrites an output op it refuses `output_not_stored` (the rerun engine recomputes it), and a saved such trace stores no transformed output (`None` on load) |
| `TraceSlice.empty` | Return whether the slice has no members (emptiness is disclosure) |
| `ValidationDiagnostic.check_name` | Stable name for the validation diagnostic source |
| `ValidationFailure.check_name` | Which validation phase fired (always set) |
| `ValidationStatus.__bool__` | The pass/fail result as a bool; raises `TypeError` unless replay validation ran (`state` is `passed` or `failed`) |
| `ValidationStatus.available_live` | Build the default status for a live trace with validation support |
| `ValidationStatus.exempted_reason_counts` | Stable reason-code counts for nodes whose replay/perturbation check had a provable exemption |
| `ValidationStatus.from_replay_counts` | Fold replay outcome counts into a trace-level status |
| `ValidationStatus.is_available` | Whether replay validation can run for this trace |
| `ValidationStatus.message` | Human-readable status explanation |
| `ValidationStatus.num_effect_region_ops` | Number of effectful or stateful nodes excluded from pure per-op replay |
| `ValidationStatus.num_failed_ops` | Number of replay-validation failures contributing to a failed result |
| `ValidationStatus.num_pure_unverified_ops` | Number of pure value nodes that classified cleanly but were not replayed, usually because no faithful backend raw-op replay is available |
| `ValidationStatus.num_replayed_ops` | Number of ops or regions replay-validated for a completed result |
| `ValidationStatus.num_unverified_ops` | Aggregate number of nodes or regions excluded from per-op replay |
| `ValidationStatus.passed` | Whether replay validation ran and passed |
| `ValidationStatus.payload_load_status` | Optional payload materialization status attached by `tl.load` |
| `ValidationStatus.reason` | Stable reason code for an unavailable or unverified status (for example `trace_retrace_structure_mismatch`), else `None` |
| `ValidationStatus.state` | Replay-validation state: `"available"`, `"passed"`, `"failed"`, `"unavailable"` or `"unverified"` |
| `ValidationStatus.unavailable_loaded_runtime_stripped` | Build the loaded non-torch unavailable status |
| `ValidationStatus.unavailable_unsupported` | Build an unavailable status for a backend without replay validation |
| `ValidationStatus.unverified_reason_counts` | Stable reason-code counts for nodes that replay could not verify |
| `WatchdogSnapshot.is_tripped` | Whether the watchdog fired |
| `WatchdogSnapshot.backwards_since_accepted` | Backward passes seen since the last accepted step |
### Members by class
Members of classes whose public home changed, one entry per class.

**`tl.types.Container`**
: Computed view over a captured Python output container.
  - `Container.leaf_occurrences`: every tensor occurrence and its container path.
  - `Container.leaves`: the ops whose outputs are the container's tensor leaves.
  - `Container.reconstruct_at`: rebuild the container from a snapshot selector (`site=`, `role=`,
    `values=`).
  - `Container.reconstructable`: whether the view can rebuild the original Python object.
  - `Container.root_id`: identity of the container's root.
  - `Container.root_kind`: what kind of root the container hangs from.
  - `Container.supports_reconstruct`: whether reconstruction is supported for this container.

**`tl.types.ContainerSpec`**
: Portable description of an output container seen during capture.
  - `ContainerSpec.aux_data`: auxiliary data needed to rebuild the container type.
  - `ContainerSpec.child_specs`: the child path components and their nested specs.
  - `ContainerSpec.kind`: the container kind (`tuple`, `list`, `dict`, `namedtuple`, `dataclass`,
    `hf_model_output`, `registered`, `distribution` for a `torch.distributions` object recorded one
    output per parameter at attribute paths such as `.loc`, `literal`, or `opaque`).
  - `ContainerSpec.literal_value`: the stored value for a non-tensor literal leaf.
  - `ContainerSpec.lossy_reconstruction`: whether rebuilding the container loses information.
  - `ContainerSpec.type_module`: module of the container's type.
  - `ContainerSpec.type_qualname`: qualified name of the container's type.

**`tl.types.ContainerLeafOccurrence`**
: One tensor occurrence at one container path.
  - `ContainerLeafOccurrence.occ_index`: index of this occurrence.
  - `ContainerLeafOccurrence.producer_op_label`: label of the op that produced the tensor, or
    `None`.
  - `ContainerLeafOccurrence.tensor_identity`: identity of the tensor, or `None`.

**`tl.types.MergeAlignment`**
: Structural merge verdict; value divergence never changes it.
  - `MergeAlignment.ALIGNED`: the rank captures line up structurally.
  - `MergeAlignment.CONFLICTED`: at least one structural conflict was found.

**`tl.types.MergeValueStatus`**
: Merge-level value verdict derived from the join ledger.
  - `MergeValueStatus.ATTESTED_COMPLETE`: every applicable join boundary is attested.
  - `MergeValueStatus.ATTESTED_PARTIAL`: some applicable join boundaries are attested, not all.
  - `MergeValueStatus.DIVERGENT`: at least one join boundary mismatched across ranks.
  - `MergeValueStatus.UNWITNESSED`: no join boundary is attested.

**`tl.types.MergedFinding`**
: One structured merge finding.
  - `MergedFinding.detail`: human-readable explanation (diagnostics may sharpen it but are never
    authority).
  - `MergedFinding.membership_digest`: the membership the finding concerns, when it is
    membership-scoped.

**`tl.report.MergeReport`**
: Graph-free diagnostic summary of a merge.
  - `MergeReport.alignment`: the effective structural verdict.
  - `MergeReport.conflicts`: the structural conflict findings, which never join and never become
    gaps.
  - `MergeReport.divergences`: the value-divergence findings: ranks disagreeing on a witnessed
    value.
  - `MergeReport.expected_ranks`: the declared world, when one was given.
  - `MergeReport.gaps`: the presence-gap findings: boundaries a rank never recorded.
  - `MergeReport.stored_alignment`: the structural verdict derived at merge time.
  - `MergeReport.value_status`: the merge-level value verdict.

**`tl.types.MergedTrace`**
: N rank-local captures stitched at their collective boundaries.
  - `MergedTrace.alignment`: the effective verdict: the stored one, lowered by load degradations.
  - `MergedTrace.expected_ranks`: the declared world, when one was given (it can only widen
    expectations).
  - `MergedTrace.gaps`: the typed presence-gap ledger.
  - `MergedTrace.happens_before`: partial-order query over two join keys (`left`, `right`).
  - `MergedTrace.join_ops`: resolve a join's boundary ops on every presenting rank.
  - `MergedTrace.joins`: the cross-rank collective joins, in deterministic key order.
  - `MergedTrace.member_outcomes`: read-only map of each rank to its settled capture-outcome status.
  - `MergedTrace.rank_ids`: the global ranks present in the merge.
  - `MergedTrace.cleanup`: drop every rank-trace handle so the member traces can be reclaimed.
  - `MergedTrace.stored_alignment`: the structural verdict derived at merge time.
  - `MergedTrace.stored_value_status`: the value verdict derived at merge time.
  - `MergedTrace.value_status`: the merge-level value verdict from the join ledger.

**`tl.trackers.HistoryReader`**
: Reads one history artifact without loading the run or importing a model.
  - `HistoryReader.observations`: materialize observations, optionally filtered by `step`, `site_id`
    and `stream`; verifies checksums.
  - `HistoryReader.step_blocks`: the committed step blocks, oldest first.

**`tl.trackers.HistoryView`**
: Session-side read view over one run's committed history.
  - `HistoryView.from_blocks`: wrap in-memory committed blocks.
  - `HistoryView.from_collector`: read the live collector's in-memory ring.
  - `HistoryView.from_reader`: build a view from a disk artifact (checksums verified).
  - `HistoryView.series`: one site and stream series in step order, disambiguated by phase.
  - `HistoryView.site_streams`: the (stream, phase) pairs observed for one site.

**`tl.trackers.SeriesPoint`**
: One sample of a series.
  - `SeriesPoint.observation`: the observation record.
  - `SeriesPoint.step_hi`: last step of the span the sample covers.
  - `SeriesPoint.step_lo`: first step of the span the sample covers.

**`tl.types.PartialTrace`**
: Thin wrapper around raw capture state from a failed forward pass.
  - `PartialTrace._FIELD_POLICY`: the declared field policy, so field-declaration checks also cover
    failed partial captures.
  - `PartialTrace.audit`: an evidence-backed health report for the partial capture.
  - `PartialTrace.capability_card`: which family members work on this object.
  - `PartialTrace.from_trace`: build the wrapper from a failed capture's internal state and its
    exception.
  - `PartialTrace.original_exception`: the exception the failed capture raised.
  - `PartialTrace.output_table`: typed refusal: a failed forward has no outputs.
  - `PartialTrace.raw_layers`: the layer entries captured before the failure.
  - `PartialTrace.to_agent_json`: typed refusal: the machine map needs finalized identities.

**`tl.types.InputProvenance`**
: Persistent provenance for how a capture's input came to exist.
  - `InputProvenance.cache_key_version`: version of the input cache key, or `None`.
  - `InputProvenance.cache_status`: whether the input came from the cache, or `None`.
  - `InputProvenance.call_structure`: normalized call description: positional count, keyword names,
    and whether non-tensor arguments are present.
  - `InputProvenance.caveats`: the rendered disclosure sentences.
  - `InputProvenance.d8_meta`: metadata for the reserved meta-device rung, or `None`.
  - `InputProvenance.evidence_grade`: `verified_exact`, `empirically_flexible`, `under_determined`
    or `unsupported`.
  - `InputProvenance.execution_policy`: `"caller_mode"` (the caller's flags are kept) or
    `"eval_no_grad_restored"` (eval and no-grad are pinned, then restored).
  - `InputProvenance.lazy_transitions`: module addresses whose lazy parameters materialized during
    the captured forward.
  - `InputProvenance.rung`: the input ladder rung, 1 to 3.

**`tl.types.ContractCheck`**
: One contract check recorded in a run report.
  - `ContractCheck.diagnostic`: the diagnostic for a failed check, or `None`.

**`tl.types.NumericAttestationStatus`**
: Result states for the optional saved-activation check.
  - `NumericAttestationStatus.ATTESTED`: the run's values match the saved activations.
  - `NumericAttestationStatus.NOT_APPLICABLE`: the check does not apply to this run.
  - `NumericAttestationStatus.NOT_PRESENT`: no saved activations were available to check against.
  - `NumericAttestationStatus.NUMERIC_ATTESTATION_FAILED`: the run's values did not match the saved
    activations.

**`tl.types.PathFaithfulness`**
: Honesty states reported by a completed sparse run.
  - `PathFaithfulness.DIVERGED`: the run left the recorded path.
  - `PathFaithfulness.UNVERIFIABLE`: the path could not be checked.
  - `PathFaithfulness.VERIFIED`: the run followed the recorded path.

**`tl.types.ReadinessReport`**
: Complete non-executing readiness report for a trace provider.
  - `ReadinessReport.capability`: the capability the report is about, or `None`.
  - `ReadinessReport.diagnostics`: the readiness diagnostics.
  - `ReadinessReport.provider`: which provider would serve the run.
  - `ReadinessReport.resolver_records`: how each callable was resolved.
  - `ReadinessReport.state_sources_available`: the state sources the run could use.
  - `ReadinessReport.witness_completeness`: how complete the recorded witnesses are, or `None`.

**`tl.types.ReadinessStatus`**
: Overall non-executing readiness states.
  - `ReadinessStatus.READY`: the trace can run.
  - `ReadinessStatus.UNAVAILABLE`: the trace cannot run.

**`tl.types.RunProvider`**
: Providers selected by `Trace.run`.
  - `RunProvider.LIVE`: run on the live model.
  - `RunProvider.LOADED_ANALYSIS`: run from a loaded analysis trace.
  - `RunProvider.LOADED_SPARSE`: run from a loaded sparse trace.

**`tl.types.RunReport`**
: Honesty, state, resolution and contract metadata for one run.
  - `RunReport.contract_checks`: the contract checks the run recorded.
  - `RunReport.first_mismatch`: the first mismatch diagnostic, or `None`.
  - `RunReport.initializer_policy_version`: version of the initializer policy used to fill slots, or
    `None`.
  - `RunReport.nondeterministic_sources`: sources of nondeterminism the run met.
  - `RunReport.numeric_attestation`: the saved-activation check result.
  - `RunReport.path_faithfulness`: whether the run followed the recorded path.
  - `RunReport.random_filled_slot_ids`: slots filled with random values.
  - `RunReport.readiness`: the readiness report the run started from.
  - `RunReport.state_source`: where the run's state came from.

**`tl.types.RunnableCompatibility`**
: Producer and runtime versions recorded for readiness diagnostics.
  - `RunnableCompatibility.torchlens_version`: the torchlens version that produced the artifact.
  - `RunnableCompatibility.backend_version`: backend version.
  - `RunnableCompatibility.call_recipe_version`: call recipe version.
  - `RunnableCompatibility.callable_ref_schema_version`: callable reference schema version.
  - `RunnableCompatibility.descriptor_version`: descriptor version.
  - `RunnableCompatibility.initializer_policy_version`: initializer policy version.
  - `RunnableCompatibility.python_version`: Python version.

**`tl.types.RunnableDiagnostic`**
: Structured diagnostic shared by preflight, readiness and run reports.
  - `RunnableDiagnostic.affected_op_labels`: labels of the ops the diagnostic concerns.
  - `RunnableDiagnostic.current_runtime`: the runtime now in use, or `None`.
  - `RunnableDiagnostic.details`: extra (key, value) details.
  - `RunnableDiagnostic.detection_stage`: the stage that found the problem.
  - `RunnableDiagnostic.recorded_runtime`: the runtime recorded at capture, or `None`.
  - `RunnableDiagnostic.registry_id`: the registry entry involved, or `None`.
  - `RunnableDiagnostic.resolver_provenance`: how the callable was resolved, or `None`.

**`tl.types.BasisProvenance`**
: Frozen reproducibility record for one subspace basis.
  - `BasisProvenance.space_dim`: dimension of the space the basis lives in.

**`tl.types.NarrationEvent`**
: One immutable narration record. `NarrationEvent.value_stats` holds `tl.stats.TensorStats` or
  `None`; sampled narration omits the finiteness family, while a reused positive verdict renders
  "nonfinite present" without a count.
  - `NarrationEvent.extra_labels`: extra (key, value) labels.
  - `NarrationEvent.late`: ordinal distance disclosed on a buffered release.
  - `NarrationEvent.module_depth`: nesting depth of the module.
  - `NarrationEvent.module_type`: class name of the module, or `None`.
  - `NarrationEvent.ordinal`: position of the event in the narration.
  - `NarrationEvent.output_index`: index of the output, or `None`.
  - `NarrationEvent.source_location`: the source location, or `None`.
  - `NarrationEvent.step_index`: the step index, or `None`.

**`tl.types.TraceSlice`**
: Frozen presenter over one trace's executed sub-DAG.
  - `TraceSlice.as_region`: derive an admissible region that is operated on as one unit; the slice
    views its trace without copying it.

**`tl.transformer.Annotation`**
: One typed annotation attached to an attention picture. `Annotation.legend`: the kind legend line.

**`tl.transformer.AttentionView`**
: One attention pattern in canonical coordinates.
  - `AttentionView.coordinate`: optional shared site coordinate.
  - `AttentionView.crop`: optional crop disclosure (cropping never renormalizes).
  - `AttentionView.crop_to`: return a cropped view with an exact omitted-mass disclosure.
  - `AttentionView.domain`: `"probability"` (fixed 0 to 1) or `"per_panel"`.
  - `AttentionView.gqa`: optional grouped-query-attention disclosure.
  - `AttentionView.head_view`: a single-head view selected by original head index.
  - `AttentionView.key_tokens`: source-axis tokens, a separate record even for self-attention.
  - `AttentionView.provenance_wording`: the rendered provenance wording for the view.
  - `AttentionView.query_tokens`: destination-axis tokens.

**`tl.transformer.CausalReceipt`**
: Validated measured-intervention evidence for an annotation.
  - `CausalReceipt.cell_sum`: the sum of the measured single-head effects.
  - `CausalReceipt.fingerprints`: content fingerprints for the joined artifacts.
  - `CausalReceipt.negative_control`: measured self-patch metric delta (must be about 0).
  - `CausalReceipt.negative_control_tol`: tolerance for the negative control.
  - `CausalReceipt.pattern_source`: where the attention pattern came from (default `"captured"`).
  - `CausalReceipt.positive_control`: measured large-perturbation metric delta (must move).
  - `CausalReceipt.positive_control_min`: smallest movement the positive control must show.

**`tl.transformer.CropInfo`**
: Crop disclosure; cropping never renormalizes.
  - `CropInfo.key_range`: kept source (column) index range.
  - `CropInfo.omitted_keys`: number of source positions cropped away.
  - `CropInfo.omitted_mass_max`: maximum probability mass omitted from any kept destination row,
    measured on the uncropped pattern.
  - `CropInfo.query_range`: kept destination (row) index range.

**`tl.transformer.EpisodeCoordinates`**
: Episode coordinates carried by transformer records.
  - `EpisodeCoordinates.role`: `"prefill"` or `"decode"`.

**`tl.transformer.PredictionTable`**
: Top-k tokens per position.
  - `PredictionTable.context_tokens`: the input token at each position.
  - `PredictionTable.top_probs`: full-vocabulary probabilities, one row of k per position.

**`tl.transformer.PredictionTrajectory`**
: Per-layer prediction trajectory record.
  - `PredictionTrajectory.target_probs`: the tracked token's probability at each layer, or `None`.
  - `PredictionTrajectory.target_ranks`: the tracked token's rank at each layer, or `None`.
  - `PredictionTrajectory.target_token`: optional tracked answer token.
  - `PredictionTrajectory.top_probs`: full-vocabulary probabilities, one row of k per layer.

**`tl.transformer.ScoreDecomposition`**
: Term-complete score decomposition for one (head, destination, source) triple.
  - `ScoreDecomposition.additive_terms`: named additive contributions applied after the scaled dot
    product.
  - `ScoreDecomposition.destination`: destination position.
  - `ScoreDecomposition.key_vector`: the key vector.
  - `ScoreDecomposition.products`: elementwise query-times-key products.
  - `ScoreDecomposition.query_vector`: the query vector.
  - `ScoreDecomposition.reference_score`: the captured score this decomposition claims to equal.
  - `ScoreDecomposition.tolerance`: closure tolerance.

**`tl.transformer.TokenAxis`**
: One labeled token axis. `TokenAxis.role`: `"query"`, `"key"` or `"position"`.

**`tl.transformer.TokenMetrics`**
: Per-token metric strip record (loss, entropy).
  - `TokenMetrics.source_fingerprint`: fingerprint of the one logits tensor both metrics were
    computed from.

**`tl.transformer.TokenScores`**
: Colored-token strip record.
  - `TokenScores.domain`: `"zero_centered_diverging"`, `"unit_interval"` or
    `"magnitude_sequential"`.
  - `TokenScores.from_attribution`: build a strip record from an attribution token payload.

### Record class members

Members of the capture record classes that had no entry of their own. Each row names the member on its
public class; every class below is reached through the home shown in its heading. Rows are in
alphabetical order.

#### `tl.Trace`

| Spelling | Meaning |
| --- | --- |
| `Trace.routing_table` | Lazy list of routing rows across structurally recognized MoE blocks; each row carries token expert indices, gate weights, counts and assignment load fractions; empty for non-MoE traces |
| `Trace.__enter__` | Context-manager entry; returns the Trace itself so `with trace:` binds the same object |
| `Trace.__exit__` | Context-manager exit; detaches the sticky hooks the last `do()` inside the scope staged |
| `Trace.annotate` | Attach user-owned annotation data or an image to the graph nodes picked by a selector |
| `Trace.append` | `trace.append(inputs, *, model=None)`: run the model on one more compatible batch and add its rows to this Trace in place, with the same checks as a rerun; returns `None`. `model` omitted reuses the live model this Trace holds weakly. A batch that changes shapes beyond the batch axis or an edit not marked `is_compatible_with_append` raises `AppendMismatchError`; a batch that would rewrite streamed activation blobs raises `AppendStreamingUnsupportedError` |
| `Trace.archived_activations` | Mapping of label to inspection-only activation payloads from a runnable archive |
| `Trace.attention_blocks` | Yield the Modules that have an available query-projection facet, by the membership rule of `Trace.modules_with_facet("q")`; a reused Module is included, and its `.facets` refuses (`module_call_ambiguous`), so read `module.calls[N].facets` |
| `Trace.backend_runtime_config` | Backend runtime configuration recorded at capture, or `None` |
| `Trace.backend_runtime_device_summary` | Summary of the backend devices in use at trace time, or `None` |
| `Trace.backend_runtime_version` | Version string of the backend runtime that ran the capture, or `None` |
| `Trace.backward` | Run backward from `loss` (a scalar, or a callable mapping `Trace.output` to one, as `loss_fn` does, on every backend that supports backward) and populate this Trace with backward metadata (accepts `save_grads`; other keywords pass to the framework's backward call); returns the Trace |
| `Trace.batch_render` | Policy for rendering a batch of raw input in visualizations; defaults to `"auto"`. A trace captured from one photo (a file path or PIL image) draws a thumbnail of it on the input node; `"shape_only"` turns that off |
| `Trace.buffer_layers` | Labels of the Layers that represent buffers as graph boundary values |
| `Trace.buffer_read_ops` | Labels of buffer Ops that read registered-buffer values into the graph |
| `Trace.buffer_write_ops` | Labels of buffer Ops that record writes to registered buffers |
| `Trace.call_tree` | The root `ModuleCall` of the nested module call tree, as structured data |
| `Trace.capture_end_time` | Wall-clock time (seconds since the epoch) when the capture ended |
| `Trace.capture_guard_passes` | List with one dict per capture guard pass, giving its index, capture mode and owner thread id |
| `Trace.capture_start_time` | Wall-clock time (seconds since the epoch) when the capture started |
| `Trace.capture_tensor_grad_hooks` | Whether forward tensors receive tensor-level backward hooks for implicit backward events and per-op gradients |
| `Trace.check_metadata_invariants` | Run the metadata invariant checks on this completed trace and return whether they pass |
| `Trace.check_plan` | Audit a plan against this trace, resolving its selectors and checking declared shapes, without arming or running anything |
| `Trace.chunked_forward` | Whether the trace was assembled from forward chunks |
| `Trace.cleanup` | Delete log data, break reference cycles and free cached GPU memory |
| `Trace.completeness_diagnostics` | List of reports of ATen dispatch events the completeness witness could not account for; empty when capture was complete |
| `Trace.compute_coverage` | Counts of layer-list rows in each of four compute-coverage categories, as a dict |
| `Trace.decode_output` | Return the decoded output rows captured during tracing, up to `k` ranks per batch item; with `style=`, `head=` or `tokenizer=` decode again on demand from the retained output tensors without touching `decoded_output` (`style="hf_text"` without a tokenizer returns the argmax `token_ids` with `text` `None`); with nothing captured, run the output detectors over the retained output tensors and raise if none is retained or no detector resolves a style |
| `Trace.do` | `trace.do(site, edit, /, *, model=None, inputs=None, intervention=None, loss=None)`: apply an intervention and replay, rerun or set it; the site (a selector-like site, or a mapping or list batch of `(site, edit)` pairs) and the edit (the value, hook or helper applied there, omitted for a batch) are positional-only |
| `Trace.escape_diagnostics` | List of reports of raw torch callables that ran outside their registered wrapper during capture, each with file, line and function |
| `Trace.factcore` | The single numbers substrate for this trace: counts, identity spine, parameter totals, compute aggregation and memory figures |
| `Trace.find_nan` | Return the first NaN or Inf found among saved outputs in execution order |
| `Trace.find_sites` | Return a table of the intervention sites that match a selector |
| `Trace.has_frozen_params` | Whether this Trace contains at least one frozen parameter |
| `Trace.has_trainable_params` | Whether this Trace contains at least one trainable parameter |
| `Trace.health_facts` | Normalized, serializable health observations for this trace, including non-finite value findings and coverage counts |
| `Trace.injected_ops` | Tuple of records for computation run inside intervention hooks when `log_injections=True`; empty if none ran |
| `Trace.input_layers` | Labels of the Layers that are model inputs |
| `Trace.internal_sink_ops` | Labels of Ops that end inside the model without reaching an output |
| `Trace.internal_source_ops` | Labels of Ops whose values originate inside the model, such as buffers and tensors created during forward |
| `Trace.internally_terminated_bool_ops` | Labels of boolean tensor ops that end inside the model without reaching an output |
| `Trace.jax_capture_index_to_final_op_label` | Dict mapping each JAX capture index to the final label of its op in the trace |
| `Trace.jax_equation_captures` | Tuple of the per-equation capture records from tracing the JAX function |
| `Trace.jax_outvar_key_to_capture_index` | Dict mapping a JAX output-variable key to the capture index of the equation that produced it |
| `Trace.last_run_records` | Return the fire records from the most recent replay, rerun or live capture |
| `Trace.layer_labels` | Layer labels in execution order, without pass suffixes such as `:2` |
| `Trace.layer_visualizers` | Optional mapping from site selectors to per-layer thumbnail callables used when rendering; not saved with the trace |
| `Trace.layers_with_params` | Dict mapping each parameter-consuming layer label to its parameter records |
| `Trace.logged_values` | Dict of the values recorded through `log_value` during this capture |
| `Trace.macs_unknown_split_ops` | Labels of compute ops whose exact multiply-accumulate (MAC) split cannot be derived |
| `Trace.model_ops` | Tuple of the trace's ordinary executed op records, excluding any ops injected by intervention hooks |
| `Trace.model_profile` | The computed `ModelProfile` semantic input/output descriptor for this trace, not persisted |
| `Trace.module_calls` | Accessor for per-invocation `ModuleCall` records, looked up by call label or index |
| `Trace.modules_with_facet` | Yield the Modules that have the facet `name` available now: a module called once when its facet view has `name`; a module called more than once when its facet recipe declares `name` and at least one of its calls has it available now. A reused Module's own `.facets` refuses (`module_call_ambiguous`); read each call with `module.calls[N].facets` |
| `Trace.nonfinite_verdict` | Three-state health verdict: `found`, `checked_and_clean` or `not_checked` |
| `Trace.num_context_lines` | Number of source lines captured around each call site for `FuncCallLocation` |
| `Trace.num_edges` | Number of distinct edges in the per-pass Op graph, including boundary edges |
| `Trace.num_input_layers` | Number of distinct input-boundary Layers (a multi-pass Layer counts once) |
| `Trace.num_output_layers` | Number of distinct output-boundary Layers (a multi-pass Layer counts once) |
| `Trace.num_params_frozen` | Number of scalar parameters that do not require gradients |
| `Trace.num_params_trainable` | Number of scalar parameters that require gradients |
| `Trace.op_labels` | Op labels in execution order, always pass-qualified (ending in a pass suffix such as `:1`) |
| `Trace.output_id2label` | Dict mapping output class ids to label strings, used to decode model output, or `None` |
| `Trace.output_layers` | Labels of the Layers that are model outputs |
| `Trace.output_num_classes` | Number of output classes used to decode model output, or `None` |
| `Trace.parent_trace` | The immediate parent Trace in a fork or rerun lineage, or `None` |
| `Trace.preview_fastlog` | Render a preview of what a fastlog predicate would keep on this model graph |
| `Trace.profile_enabled` | Whether this capture was run with profiling requested |
| `Trace.reconstruct_container` | Rebuild a captured container selected by site and role from saved Op values |
| `Trace.reconstruct_output` | Rebuild the traced model's final Python output object from saved Op values |
| `Trace.replace_state_from` | Atomically replace this trace's run state with that of another `Trace` |
| `Trace.root_module` | The `Module` record for the top-level model, `modules["self"]` |
| `Trace.root_trace` | The ultimate root Trace of this fork or rerun lineage, or `None` |
| `Trace.run` | `trace.run(model, x, /, *, chunk_paths=None, replay=None, input_transform=<stored>, output_transform=<stored>, on_zero_fire=<unset>)`: rerun the live model on new input into a new Trace and return it; this Trace is unchanged. The result is a fork named `<name>_fork_<n>`, as the `fast=True` door's result is, carrying this Trace's weak live-model reference and staged edits; keep it with `trace = trace.run(x)`. `Trace.append` is the in-place form. `model` and `x` are positional-only: `model` is the module to run (omitted, the live model this Trace holds weakly is reused, so the caller must keep a strong reference), and `x` is the forward input in every form `tl.trace` takes (a tensor, a positional list or tuple, and for a Hugging Face model a string, PIL image or modality dict); with one argument it is the input. A loaded artifact has no live model to rerun and refuses `run_capability_unavailable`, naming `run_transaction` |
| `Trace.run_transaction` | `trace.run_transaction(inputs=..., *, seed=None, fast=False, carry_state=False, until=None, save=None, on_divergence="raise")`: a verified fresh execution on the live model or a loaded artifact; returns a `RunResult` with its report and leaves this Trace unchanged |
| `Trace.save_visualizations` | Whether rendered visualizations are persisted in bundles |
| `Trace.saved_grad_module_calls` | Accessor for the ModuleCalls whose outputs include saved gradients |
| `Trace.saved_grad_modules` | Accessor for the Modules with at least one saved-gradient ModuleCall |
| `Trace.saved_layers` | Accessor for the Layers containing at least one Op with a saved activation |
| `Trace.saved_module_calls` | Accessor for the ModuleCalls whose outputs include saved activations |
| `Trace.saved_modules` | Accessor for the Modules with at least one saved-activation ModuleCall |
| `Trace.show_call_tree` | Print this Trace's `ModuleCall` tree as an ASCII tree, with options for depth, atomic calls and call indices |
| `Trace.state_history` | Append-only list of lifecycle events for this trace, such as fork, do, run and direct write |
| `Trace.tlspec_version` | Integer version of the `.tlspec` format, used for save and load compatibility |
| `Trace.transforms` | Tuple of the Op records that mark transform boundaries |
| `Trace.uncalled_modules` | Registered module addresses that were not called during the captured pass |
| `Trace.unknown_flop_ops` | Ledger of ops with unknown FLOPs, grouped by op name, with counts and example labels |
| `Trace.validation_replay_status` | Machine-readable status of replay validation: unavailable, unverified, or the last completed result |
| `Trace.walk_calls` | Yield every `ModuleCall` in the call tree in depth-first order |

#### `tl.Op`

| Spelling | Meaning |
| --- | --- |
| `Op.is_router` | Lazy bool identifying gate, normalization and selection ops on a structurally proven expert-routing path; no stored per-op field |
| `Op.expert` | Lazy zero-based expert number in homogeneous container registration order, or `None` outside an individually executed routed expert, including fused expert ops |
| `Op.source.arg_expressions` | Source expressions of this Op's call arguments, parsed lazily from `code_context`; an empty list when unavailable |
| `Op.call.arg_names` | Names of the arguments in this Op's function call |
| `Op.args_summary` | Human-readable summary of this Op's positional arguments |
| `Op.call.args_template` | Structural template of this Op's positional arguments, used to rebuild the call |
| `Op.module_context.atomic_module` | The `Module` record for the atomic module this Op is the output of, or `None` |
| `Op.module_context.atomic_module_address` | Module address, without the pass suffix, of the atomic module this Op outputs from, or `None` |
| `Op.module_context.atomic_module_call` | Stored `ModuleCall` label for an atomic-module output Op, or `None` |
| `Op.atomic_module_call_label` | Stable `ModuleCall` label for an atomic-module output Op, or `None` |
| `Op.backend_address` | The Op's locator in the backend's own graph, such as `jaxpr:<label>` |
| `Op.control_flow.bool_value` | The Python value of this Op's output when it is a scalar bool |
| `Op.buffer_address` | Module address of the buffer this Op reads or writes, such as `bn.running_mean`, or `None` for a non-buffer Op |
| `Op.buffer_pass` | Pass number of this Op's buffer, as used in the `address:pass` lookup key |
| `Op.buffer_replay_validated` | Whether replaying this buffer write as an identity from its source matched the recorded value |
| `Op.graph.buffer_sink_ops` | Accessor over the buffer-write Ops that directly consume this Op |
| `Op.graph.buffer_source` | Label of the Op that wrote this buffer value, or `None` for a static buffer |
| `Op.graph.buffer_source_func_name` | Name of the function that produced this buffer value |
| `Op.graph.buffer_source_ops` | Accessor over the buffer-source Ops that feed this Op |
| `Op.buffer_value_changed` | Whether this buffer write changed the buffer's value |
| `Op.buffer_write_kind` | Kind of write that produced this buffer version, or `None` for a static buffer |
| `Op.children` | Labels of the Ops that directly consume this Op's output |
| `Op.graph.co_parents` | Labels of other Ops that share at least one child with this Op |
| `Op.source.frames` | Call-stack frames showing where this Op ran, outermost first: `[0]` is the call site that started the capture and `[-1]` is the frame that ran the Op; `print(op)` shows the innermost frame outside torch itself |
| `Op.control_flow.arm_children` | Dict mapping each conditional's id to its arm kinds (then, elif_N, else, fallthrough) and the labels of this Op's children inside each arm |
| `Op.control_flow.branch_depth` | Number of conditional arms this Op is nested inside or guarded by (a `fallthrough` arm guards the code after an early-return chain); 0 when it is in no arm |
| `Op.control_flow.branch_stack` | Stack of `(conditional_id, arm_kind)` pairs for the conditional branches this Op is nested inside, outermost first |
| `Op.control_flow.depth` | Number of distinct conditionals this Op takes part in, derived on read from `in_conditionals` |
| `Op.control_flow.in_conditionals` | Ids of the conditionals this Op takes part in, as a test or inside an arm |
| `Op.control_flow.context_kind` | Kind of context (`"if"` or `"while"`) of the conditional that owns this terminal bool, or `None` |
| `Op.control_flow.elif_children` | Dict mapping each elif arm index to the labels of this Op's children in that arm |
| `Op.control_flow.else_children` | Labels of this Op's children that sit in an else arm of a conditional |
| `Op.control_flow.entry_children` | Labels of this Op's children whose edge from it enters a conditional arm |
| `Op.control_flow.then_children` | Labels of this Op's children that sit in a then arm of a conditional |
| `Op.control_flow.wrapper_kind` | Wrapper kind of the conditional that owns this terminal bool, or `None` |
| `Op.container` | Runtime container view for this Op, when available |
| `Op.container_spec` | Rebuildable description of the container that holds this Op's output |
| `Op.containers` | All input and output container views associated with this Op |
| `Op.call` | How this Op's function was called: arguments, saved values, RNG and autocast state, and transform boundaries (`tl.types.OpCall`) |
| `Op.source` | Where this Op's function was called in user code: call-stack frames, argument expressions, assigned variable names and FX identity (`tl.types.OpSource`) |
| `Op.autograd` | Autograd linkage and saved gradients of this Op: the `grad_fn` node, its class names, saved-gradient records and derived gradients (`tl.types.OpAutograd`) |
| `Op.call.call_id` | Identifier of the function call that produced this Op; Ops from one call share it |
| `Op.call.func_transform_kind` | Kind of function transform at this boundary Op, such as `vmap` or `grad`, or `None` |
| `Op.call.saved_args` | Positional argument values saved for this Op's call, or `None` when not saved |
| `Op.call.tl_authored_root` | Whether this Op is the root of a TorchLens-authored call rather than a model call |
| `Op.control_flow` | Conditional-branch facts of this Op: branch membership, terminal bools and arm children (`tl.types.OpControlFlow`) |
| `Op.cost` | Compute, memory and timing facts of this Op (`tl.types.OpCost`) |
| `Op.module_context` | Module-nesting facts of this Op: the module calls active around it, the module calls its output feeds and the module calls it is the output of (`tl.types.OpModuleContext`) |
| `Op.payload` | Saved transformed payloads of this Op and lazy materialization of bundle-backed payloads (`tl.types.OpPayload`) |
| `Op.graph` | Graph position of this Op: parent, child, sibling and co-parent relations, input and output ancestry, distances, buffer sources and sinks, and equivalence (`tl.types.OpGraph`) |
| `Op.cost.activation_memory` | Bytes of this Op's output activation |
| `Op.cost.arithmetic_intensity` | Theoretical forward FLOPs per byte of ideal memory traffic, or `None` |
| `Op.cost.autograd_memory` | Bytes of the tensors autograd saved for backward at this Op |
| `Op.cost.bytes_delta_at_call` | Change in allocated bytes at the time of this Op's function call |
| `Op.cost.bytes_peak_at_call` | Peak allocated bytes at the time of this Op's function call |
| `Op.cost.bytes_read` | Theoretical ideal read-once memory traffic for this operation, in bytes |
| `Op.cost.bytes_written` | Theoretical ideal write-once memory traffic for this operation, in bytes |
| `Op.cost.compute_record` | Two-term compute record (`fma_macs` and `other_flops`), derived on read |
| `Op.cost.duration` | Time spent in this Op's function call, as a `Duration` |
| `Op.cost.flops_backward` | Approximate backward FLOPs for this Op |
| `Op.cost.flops_forward` | Approximate forward FLOPs for this Op |
| `Op.cost.flops_total` | Approximate total FLOPs for this Op |
| `Op.cost.input_memory` | Sum of saved activation bytes across this Op's parent Ops |
| `Op.cost.macs_forward` | True forward multiply-accumulate count for this Op, or `None` |
| `Op.cost.param_memory` | Bytes used by the parameters consumed by this Op |
| `Op.cost.transformed_activation_memory` | Bytes of this Op's transformed output activation, if one was kept |
| `Op.autograd.derived_grad` | Backend-owned intermediate gradient payload for this Op, or `None` |
| `Op.do` | `op.do(edit, /, *, model=None, inputs=None, ...)`: `Trace.do` at this op, forwarding its pass-qualified label; the edit is positional-only |
| `Op.call.dropped_edge_tensor_args` | Argument positions, such as `"arg1"`, whose tensor inputs did not become graph edges at capture |
| `Op.edge_uses` | Tuple of `EdgeUseRecord` entries, one per parent edge use captured for this Op |
| `Op.graph.edge_substitutions` | Recorded parent-edge substitutions for this Op, keyed by argument slot |
| `Op.graph.equivalence_class` | Key of the equivalence class that loop detection assigned to this Op |
| `Op.graph.equivalent_ops` | Labels of other Ops in the same equivalence class (same function and arguments) |
| `Op.call.autocast_state` | Autocast state in effect when this Op's function was called |
| `Op.call.config` | Dict of lightweight hyperparameters captured from the call, such as `kernel_size` and `stride` |
| `Op.call.func_id` | Portable key identifying the captured function this Op ran: its namespace, qualified name and dispatch kind |
| `Op.call.non_tensor_args` | Flat list of all non-tensor arguments of this Op's function call |
| `Op.source.func_qualname` | Qualified name of the function this Op called |
| `Op.call.rng_states` | RNG state snapshots recorded for this Op's function call |
| `Op.source.fx_call_index` | Call index in the `torch.fx`-style identity of this Op, when applicable |
| `Op.fx_label` | A `torch.fx`-style label for this Op, or `None` when module context is unavailable |
| `Op.source.fx_path` | Module path in the `torch.fx`-style identity of this Op, when applicable |
| `Op.get_children` | Return the child `Op` objects of this Op |
| `Op.get_parents` | Return the parent `Op` objects of this Op |
| `Op.autograd.grad_dtype` | Dtype of this Op's saved output gradient |
| `Op.autograd.grad_fn` | The autograd `grad_fn` node of this Op's output tensor, or `None`; for an in-place write through a view, the base's `CopySlices` node (it runs the write's backward), not the view's regenerated node |
| `Op.autograd.grad_fn_class_name` | Class name of this Op's autograd node |
| `Op.autograd.grad_fn_class_qualname` | Qualified class name of this Op's autograd node |
| `Op.autograd.grad_fn_handle` | The live PyTorch autograd node object for this Op; not portable |
| `Op.autograd.grad_for` | Return the saved gradient tensor for one backward pass |
| `Op.autograd.grad_shape` | Shape of this Op's saved output gradient |
| `Op.gradient_memory` | Bytes used by this Op's saved output gradient |
| `Op.gradient_transform` | Transform used for this Op's saved gradient, or `None` when unset |
| `Op.autograd.grads` | Accessor over the per-backward-pass gradient records saved for this Op |
| `Op.graph.has_children` | Whether this Op has any child Ops |
| `Op.graph.has_co_parents` | Whether this Op shares children with other Ops |
| `Op.has_frozen_params` | Whether this Op consumes at least one frozen parameter |
| `Op.has_grad` | Whether a gradient was saved for this Op's output |
| `Op.graph.has_input_ancestor` | Whether any model input is an ancestor of this Op |
| `Op.graph.has_internal_source_ancestor` | Whether any internally initialized source Op is an ancestor of this Op |
| `Op.has_out_variations` | Whether different children observed different versions of this Op's output |
| `Op.graph.has_output_descendant` | Whether any model output is a descendant of this Op |
| `Op.graph.has_parents` | Whether this Op has any parent Ops |
| `Op.has_saved_activation` | Whether this Op's output activation was saved |
| `Op.call.has_saved_args` | Whether this Op's argument values were saved |
| `Op.graph.has_siblings` | Whether this Op shares parents with other Ops |
| `Op.has_trainable_params` | Whether this Op consumes at least one trainable parameter |
| `Op.in_multi_output` | Whether this Op's output came from a list, tuple, dict or set returned by a function |
| `Op.in_submodule` | Whether this Op was computed inside a non-root module's forward |
| `Op.call.injection_provenance` | Record of which intervention injected this Op and where, or `None` for ordinary model Ops |
| `Op.input_activations` | Saved parent activations consumed by this Op, in `parents` order |
| `Op.input_ancestors` | Labels of the model inputs that are ancestors of this Op |
| `Op.input_containers` | Input container views consumed by this Op |
| `Op.module_context.input_to_module_calls` | Labels of the `ModuleCall` records whose input this Op's output fed |
| `Op.module_context.input_to_modules` | Module addresses, without the pass suffix, whose input this Op's output fed |
| `Op.input_was_parameter` | Whether this Op is an input boundary whose value was a model parameter |
| `Op.graph.internal_source_ancestors` | Labels of the internally initialized source Ops that are ancestors of this Op |
| `Op.graph.internal_source_parents` | Labels of the internally initialized source Ops that are direct parents of this Op |
| `Op.module_context.is_atomic_module` | Whether this Op is the sole operation output of an atomic module call |
| `Op.graph.io_role` | Input or output address this boundary Op stands for, or `None` for an interior Op |
| `Op.graph.is_buffer` | Stored flag marking this Op as a buffer boundary; `is_buffer_source` exposes it |
| `Op.is_buffer_source` | Whether this Op represents a buffer boundary (overwrites a buffer) |
| `Op.graph.is_compute` | Whether this Op executed a torch function rather than being a boundary sentinel |
| `Op.is_constant` | Whether this Op is a constant, a closure constant or a literal producer, stamped by the shared constants rule |
| `Op.graph.is_final_output` | Whether this Op is a final model output |
| `Op.control_flow.is_in_conditional` | Whether this Op participates in any conditional role |
| `Op.control_flow.is_in_conditional_body` | Whether this Op sits inside the body of a conditional arm |
| `Op.control_flow.is_in_conditional_evaluation` | Whether this Op computes a conditional arm's condition |
| `Op.is_inplace` | Whether the underlying operation was in-place |
| `Op.is_input` | Whether this Op is a model input |
| `Op.is_internal_sink` | Whether this Op's result ends inside the model without reaching an output |
| `Op.graph.is_internal_source` | Whether this Op was created inside the model, such as by a factory call, rather than fed from an input |
| `Op.call.is_internally_initialized` | Alias of `is_internal_source`: whether this Op was created inside the model rather than fed from an input |
| `Op.module_context.is_module_input` | Whether this Op's output feeds at least one `ModuleCall` as an input |
| `Op.module_context.is_module_output` | Whether this Op is the output of a `ModuleCall` |
| `Op.graph.is_orphan` | Whether this Op is disconnected from both the input and output flows |
| `Op.is_output` | Whether this Op is a model output |
| `Op.graph.is_output_parent` | Whether this Op is directly returned from `forward` |
| `Op.control_flow.is_scalar_bool` | Whether this Op's output is a scalar bool |
| `Op.control_flow.is_terminal_bool` | Whether this Op is a scalar bool that ends a chain of operations and decides control flow |
| `Op.control_flow.is_terminal_conditional_bool` | Whether this terminal bool is the test of a recorded conditional |
| `Op.call.is_func_transform` | Whether this Op is a function-transform boundary node |
| `Op.kwargs_summary` | Human-readable summary of this Op's keyword arguments |
| `Op.call.kwargs_template` | Structural template of this Op's keyword arguments, used to rebuild the call |
| `Op.short_label` | Short Op label without the step index, such as `conv2d_1:2` |
| `Op.short_layer_label` | Short layer label without the pass suffix or step index, such as `conv2d_1` |
| `Op.lookup_keys` | Alternate keys that resolve to this Op in lookups |
| `Op.macs_backward` | Backward MACs for this Op; not derivable, so always `None` |
| `Op.macs_total` | Total MACs for this Op; not derivable, so always `None` |
| `Op.payload.materialize_grad` | Materialize this Op's saved gradient from a lazy bundle reference |
| `Op.payload.materialize_out` | Materialize this Op's saved output from a lazy bundle reference |
| `Op.graph.max_distance_from_input` | Longest graph distance from any model input to this Op |
| `Op.graph.max_distance_to_output` | Longest graph distance from this Op to any model output |
| `Op.graph.min_distance_from_input` | Shortest graph distance from any model input to this Op |
| `Op.graph.min_distance_to_output` | Shortest graph distance from this Op to any model output |
| `Op.module_context.call_depth` | Depth of this Op's active `ModuleCall` nesting (the length of `module_call_stack`) |
| `Op.module_context.call_stack` | Root-first labels of the `ModuleCall`s active when this Op ran |
| `Op.module_context.entry_arg_keys` | Mapping from `ModuleCall` label to the argument keys this Op filled at module entry |
| `Op.multi_output_index` | 0-based position of this Op's output within its multi-output container |
| `Op.multi_output_name` | Semantic name of this Op's output within its multi-output container, such as a dict key or namedtuple field |
| `Op.multi_output_type` | Python class of the multi-output container this Op came from, or `None` |
| `Op.call.non_tensor_kwargs` | Non-tensor keyword arguments of this Op's function call |
| `Op.call.non_tensor_pos_args` | Non-tensor positional arguments of this Op's function call |
| `Op.call.num_args_total` | Total number of arguments in this Op's function call |
| `Op.autograd.num_autograd_tensors` | Number of tensors the captured forward's autograd saved for this Op's backward |
| `Op.graph.num_inputs` | Number of parent Ops of this Op |
| `Op.call.num_kwargs` | Number of keyword arguments in this Op's function call |
| `Op.num_param_tensors_frozen` | Number of frozen parameter tensors consumed by this Op |
| `Op.num_param_tensors_trainable` | Number of trainable parameter tensors consumed by this Op |
| `Op.num_params_frozen` | Total number of frozen parameter elements consumed by this Op |
| `Op.num_params_trainable` | Total number of trainable parameter elements consumed by this Op |
| `Op.call.num_pos_args` | Number of positional arguments in this Op's function call |
| `Op.ordinal_index` | 0-based position of this Op in `trace.ops`; `trace[op.ordinal_index]` returns it |
| `Op.output_containers` | Output container views associated with this Op |
| `Op.output_descendants` | Labels of the model outputs that are descendants of this Op |
| `Op.module_context.output_of_module_calls` | Labels of the `ModuleCall` records whose output this Op is |
| `Op.module_context.output_of_modules` | Module addresses, without the pass suffix, whose output this Op is |
| `Op.param_dtypes` | Dtypes of the parameters consumed by this Op |
| `Op.param_shapes` | Shapes of the parameters consumed by this Op |
| `Op.call.parent_arg_positions` | Mapping from call argument positions, positional and keyword, to the parent Op label used there |
| `Op.graph.parent_param_ops` | Mapping from each consumed parameter's identifying barcode to its pass number |
| `Op.graph.parent_params` | The `nn.Parameter` objects consumed by this Op |
| `Op.parents` | Labels of the Ops that directly feed this Op |
| `Op.raw_index` | 1-based raw capture-time counter; may have gaps after orphan removal |
| `Op.graph.recurrent_ops` | Labels of the Ops in the same recurrent group: equivalent and at a structurally corresponding position |
| `Op.resolver_status` | Status of the backend's address resolution for this Op, such as `resolved` |
| `Op.graph.root_ancestors` | Labels of the root Ops at the start of the graph that this Op descends from |
| `Op.call.saved_kwargs` | Keyword argument values saved for this Op's call, or `None` when not saved |
| `Op.graph.siblings` | Labels of other Ops that share at least one parent with this Op |
| `Op.graph.site_key` | Portable structural-position identity of this Op (`site_key_v1`, a `"s1|"`-prefixed string), or `None` on a keyless artifact |
| `Op.control_flow.terminal_bool_for` | `(conditional_id, arm_index)` when this Op is the terminal bool of a conditional arm |
| `Op.terminal_conditional_id` | Id of the conditional whose test this terminal bool decides, or `None` |
| `Op.call.func_transform_chain` | Tuple of transform tags applied to the function this Op ran |
| `Op.call.func_transform_config` | Configuration of the function transform for this transform-boundary Op |
| `Op.call.func_transform_fn_name` | Name of the function wrapped by the transform at this boundary Op |
| `Op.call.func_transform_fn_qualname` | Qualified name of the function wrapped by the transform at this boundary Op |
| `Op.call.func_transform_fn_source` | Source location of the function wrapped by the transform at this boundary Op |
| `Op.payload.transformed_grad` | This Op's saved gradient after the trace's gradient transform, or `None` |
| `Op.payload.transformed_grad_dtype` | Dtype of this Op's saved transformed gradient |
| `Op.payload.transformed_grad_shape` | Shape of this Op's saved transformed gradient |
| `Op.transformed_gradient_memory` | Bytes used by this Op's saved transformed gradient |
| `Op.payload.transformed_out` | This Op's saved output after the trace's activation transform, or `None` |
| `Op.payload.transformed_out_dtype` | Dtype of this Op's saved transformed output |
| `Op.payload.transformed_out_shape` | Shape of this Op's saved transformed output |
| `Op.type_index` | 1-based position of this Op among Ops of the same type |
| `Op.call.unattributed_tensor_args` | Tensor arguments that could not be attributed to a parent Op |
| `Op.source.var_names` | Variable names the Op's result is assigned to at its call site; empty when unresolved |
| `Op.visualizer_path` | PNG path used for the node thumbnail in graph renders, or `None` |

#### `tl.Layer`

| Spelling | Meaning |
| --- | --- |
| `Layer.call.arg_names` | Parameter names from the signature of the function this layer calls |
| `Layer.backend_address` | Backend-native address of the op or state this layer stands for, as the capturing backend names it, or `None` |
| `Layer.control_flow.bool_value` | The Python `bool` value of a scalar boolean layer, or `None` otherwise |
| `Layer.buffer_address` | Module address of the buffer this layer stands for, such as `bn.running_mean`, or `None` for a non-buffer layer |
| `Layer.buffer_overwrite_index` | Which overwrite of its buffer this layer represents |
| `Layer.graph.buffer_pass` | Sequential number of this buffer write among the writes to the same buffer address, starting at 1, or `None` for non-writes |
| `Layer.buffer_replay_validated` | Whether replaying this buffer write from its source layer reproduced the saved value, or `None` when not a buffer write |
| `Layer.graph.buffer_source` | The label of the op that wrote this buffer value, or `None` for static buffers |
| `Layer.graph.buffer_source_func_name` | Name of the function that produced this buffer write, or `None` when this layer is not a buffer write |
| `Layer.buffer_value_changed` | Whether this buffer write changed the buffer's value, or `None` when this layer is not a buffer write |
| `Layer.buffer_write_kind` | How the buffer was written, one of `reassign`, `inplace`, `fused` or `data_reassign`, or `None` for a non-write |
| `Layer.call_labels` | Labels of the ops (one per pass) that make up this layer |
| `Layer.graph.child_ops_per_layer` | Mapping from each child layer label to the pass numbers of this layer's ops that connect to it |
| `Layer.children` | Union of child layer labels across all ops of this layer, without pass suffixes |
| `Layer.graph.children_per_pass` | Mapping from pass number to the child layer labels (without pass suffixes) of that pass |
| `Layer.co_parents` | Union across ops of the co-parent layer labels, without pass suffixes |
| `Layer.source.frames` | The captured call stack of the user code that ran this layer's function, for single-pass layers, outermost first with the frame that ran it last |
| `Layer.control_flow.arm_children` | Dict mapping each conditional's id to its arm kinds (`then`, `elif_N`, `else`, `fallthrough`) and the labels of this layer's child ops inside each arm |
| `Layer.control_flow.branch_stack_ops` | Dict mapping each distinct conditional branch stack to the sorted pass numbers of this layer that ran under it |
| `Layer.control_flow.depth` | Number of distinct conditionals this layer takes part in, derived on read from `in_conditionals` |
| `Layer.control_flow.in_conditionals` | Ids of the conditionals this layer takes part in, as a test or inside an arm |
| `Layer.control_flow.elif_children` | Dict mapping each `elif` arm index to the labels of this layer's children that sit in that arm |
| `Layer.control_flow.else_children` | List of labels of this layer's child ops that sit in an `else` arm of a conditional |
| `Layer.control_flow.entry_children` | List of labels of this layer's child ops whose edge from it enters a conditional arm |
| `Layer.control_flow.role_stacks` | List of the distinct conditional branch stacks, one per different nesting, that this layer's passes ran under |
| `Layer.control_flow.then_children` | List of labels of this layer's child ops that sit in a `then` arm of a conditional |
| `Layer.call` | How this layer's function was called: argument counts and names, saved arguments, RNG state and parent argument positions (`tl.types.LayerCall`) |
| `Layer.source` | Where this layer's function was called in user code: call-stack frames and FX identity, for single-pass layers (`tl.types.LayerSource`) |
| `Layer.autograd` | Autograd linkage of this layer's output: the `grad_fn` node, its class names and the count of tensors autograd saved (`tl.types.LayerAutograd`) |
| `Layer.call.saved_args` | Positional arguments captured for this layer's function call, for single-pass layers |
| `Layer.control_flow` | Conditional-branch facts of this layer: branch membership, terminal bools and arm children (`tl.types.LayerControlFlow`) |
| `Layer.cost` | Compute, memory and timing facts of this Layer (`tl.types.LayerCost`) |
| `Layer.module_context` | Module-nesting facts of this layer: its nesting depth, whether it is a single-op module and the modules it is the output of (`tl.types.LayerModuleContext`) |
| `Layer.payload` | Saved transformed payloads of this layer, read from its single pass (`tl.types.LayerPayload`) |
| `Layer.graph` | Graph position of this Layer: relations, ancestry, per-pass edge maps and equivalence; per-pass members read the single pass and refuse a multi-pass layer (`tl.types.LayerGraph`) |
| `Layer.cost.activation_memory` | Bytes of the representative pass's output activation |
| `Layer.cost.autograd_memory` | Bytes autograd saved for backward at the representative pass |
| `Layer.cost.duration` | Execution time of this layer's function call, for single-pass layers |
| `Layer.cost.flops_backward` | Backward FLOPs of this layer's representative (first) pass |
| `Layer.cost.flops_forward` | Forward FLOPs of this layer's representative (first) pass |
| `Layer.cost.flops_total` | Representative total FLOPs for this layer |
| `Layer.cost.macs_forward` | True forward multiply-accumulate count of the representative pass, or `None` when unknown |
| `Layer.cost.total_activation_memory` | Sum of activation memory in bytes across all ops of this layer |
| `Layer.cost.total_autograd_memory` | Bytes autograd saved for backward, as carried by this layer |
| `Layer.cost.total_flops` | Sum of forward and backward FLOPs across all ops of this layer |
| `Layer.cost.total_flops_backward` | Sum of backward FLOPs across all ops of this layer |
| `Layer.cost.total_flops_forward` | Sum of forward FLOPs across all ops of this layer |
| `Layer.cost.total_func_duration` | Sum of function-call durations across all ops of this layer |
| `Layer.cost.total_macs` | Always `None`, because total MACs are not derivable |
| `Layer.cost.total_macs_forward` | Sum of true forward MACs across all ops of this layer, or `None` when any pass's split is unknown |
| `Layer.cost.total_param_memory` | Bytes used by this layer's parameters |
| `Layer.cost.transformed_activation_memory` | Bytes of the representative pass's transformed output activation |
| `Layer.do` | `layer.do(edit, /, *, model=None, inputs=None, ...)`: `Trace.do` at this layer; the edit (the value, hook or helper applied there) is positional-only |
| `Layer.edges_vary_across_ops` | Whether the graph edges of this layer differ between its passes |
| `Layer.graph.equivalence_class` | Fingerprint string that groups ops repeating the same operation, used for loop detection |
| `Layer.graph.equivalent_ops` | Labels of the ops that share this layer's equivalence class |
| `Layer.call.config` | Non-tensor configuration of the function call, stored as a dict |
| `Layer.func_qualname` | Qualified name of the function this layer calls |
| `Layer.call.rng_states` | RNG states captured for this layer's function call, for single-pass layers |
| `Layer.source.fx_call_index` | The FX-style call index of a single-op layer |
| `Layer.fx_label` | A `torch.fx`-style label for a single-op layer, or `None` |
| `Layer.source.fx_path` | The FX-style qualified path of a single-op layer, or `None` |
| `Layer.get_children` | Return the child `Layer` objects of this layer |
| `Layer.get_parents` | Return the parent `Layer` objects of this layer |
| `Layer.autograd.grad_fn` | The autograd `grad_fn` node of this layer's output tensor, or `None`; for an in-place write through a view, the base's `CopySlices` node (it runs the write's backward), not the view's regenerated node |
| `Layer.autograd.grad_fn_class_name` | Class name of the autograd node that produced this layer's output, or `None` |
| `Layer.autograd.grad_fn_class_qualname` | Qualified class name of the autograd node that produced this layer's output, or `None` |
| `Layer.autograd.grad_fn_handle` | The live autograd node object for this layer's output, not kept in saved files |
| `Layer.graph.has_children` | Whether any pass of this layer has child layers |
| `Layer.graph.has_co_parents` | Whether this layer has any co-parent layers besides itself |
| `Layer.has_frozen_params` | Whether this layer uses at least one frozen parameter |
| `Layer.has_grad` | Whether the single pass has a saved gradient |
| `Layer.graph.has_input_ancestor` | Whether any model input is an ancestor of this layer |
| `Layer.graph.has_parents` | Whether any pass of this layer has parent layers |
| `Layer.has_saved_activation` | Whether the single pass has a saved output activation |
| `Layer.graph.has_siblings` | Whether this layer has any sibling layers besides itself |
| `Layer.has_trainable_params` | Whether this layer uses at least one trainable parameter |
| `Layer.in_multi_output` | Whether this layer is one output of a function that returns several outputs |
| `Layer.in_submodule` | Whether this layer was computed inside a submodule |
| `Layer.input_ancestors` | Labels of the model-input layers that are ancestors of this layer |
| `Layer.input_to_module_calls` | Labels of the module calls that this layer's output is fed into |
| `Layer.input_was_parameter` | Whether this model input was an `nn.Parameter` |
| `Layer.internal_source_parents` | Labels of the parent layers whose ancestry includes an internal source, a tensor made inside the model rather than passed in |
| `Layer.module_context.is_atomic_module` | Whether this layer's op is the only op of a module call, a single-op leaf module |
| `Layer.graph.io_role` | Input or output address this boundary layer stands for, or `None` for an interior layer |
| `Layer.graph.is_buffer` | Whether this layer is a registered-buffer version node, the flag `is_buffer_source` reports |
| `Layer.is_buffer_source` | Whether this layer represents a buffer overwrite boundary |
| `Layer.graph.is_compute` | Whether this layer's representative op is a compute op |
| `Layer.graph.is_final_output` | Whether this layer is one of the model's final outputs |
| `Layer.control_flow.is_in_conditional` | Whether this layer takes part in any conditional role |
| `Layer.control_flow.is_in_conditional_body` | Whether this layer sits in the body of a conditional arm |
| `Layer.control_flow.is_in_conditional_evaluation` | Whether this layer computes the condition of a conditional arm |
| `Layer.is_inplace` | Whether this layer's function operates in place |
| `Layer.is_input` | Whether this layer is a model input |
| `Layer.graph.is_internal_sink` | Whether this layer's value ends inside the model without reaching any output |
| `Layer.graph.is_internal_source` | Whether this layer creates a tensor inside the model instead of receiving it from a model input |
| `Layer.module_context.is_module_input` | Whether this layer's representative op feeds into at least one `ModuleCall`, for single-op layers |
| `Layer.graph.is_orphan` | Whether any op in this layer is disconnected from the main graph |
| `Layer.is_output` | Whether this layer is a model output |
| `Layer.control_flow.is_scalar_bool` | Whether this layer's value is a scalar boolean |
| `Layer.control_flow.is_terminal_bool` | Whether this layer is a scalar boolean that ends inside the model, such as a branch condition |
| `Layer.short_label` | The short label of a single-pass layer with its pass suffix, such as `conv2d_1:1` |
| `Layer.short_layer_label` | The short layer label without a pass suffix, built from the layer type and its type index, such as `conv2d_1` |
| `Layer.leaf_module_ops` | Set of atomic module calls whose single op belongs to this layer |
| `Layer.lookup_keys` | Keys that look up this layer in the trace, for single-pass layers |
| `Layer.macs_backward` | Always `None`, because backward MACs are not derivable |
| `Layer.macs_total` | Always `None`, because total MACs are not derivable |
| `Layer.module_context.call_depth` | Depth of module nesting around this layer |
| `Layer.multi_output_index` | Position of this layer among the outputs of a multi-output function, or `None` |
| `Layer.multi_output_name` | Name of this output within a multi-output result, derived from its container path, or `None` |
| `Layer.call.num_args_total` | Total number of positional and keyword arguments passed to this layer's function |
| `Layer.call.num_kwargs` | Number of keyword arguments passed to this layer's function |
| `Layer.autograd.num_autograd_tensors` | Number of tensors autograd saved for this layer's backward, as carried by the layer |
| `Layer.num_param_tensors_frozen` | Number of frozen parameter tensors used by this layer |
| `Layer.num_param_tensors_trainable` | Number of trainable parameter tensors used by this layer |
| `Layer.num_params_frozen` | Number of frozen parameter elements used by this layer |
| `Layer.num_params_trainable` | Number of trainable parameter elements used by this layer |
| `Layer.call.num_pos_args` | Number of positional arguments passed to this layer's function |
| `Layer.op_labels` | Labels of the ops that belong to this layer |
| `Layer.ordinal_index` | Ordinal index of this layer, read from its first op |
| `Layer.output_descendants` | Labels of the model-output layers reachable downstream of this layer |
| `Layer.module_context.output_of_module_calls` | Labels of the module calls whose output this layer is |
| `Layer.module_context.output_of_modules` | Addresses of the modules whose output this layer is |
| `Layer.param_dtypes` | Dtypes of the parameters used by this layer |
| `Layer.param_shapes` | Shapes of the parameters used by this layer |
| `Layer.call.parent_arg_positions` | Map from argument kind and position to the parent layer label, merged across ops |
| `Layer.graph.parent_ops_per_layer` | Mapping from each parent layer label to the pass numbers of this layer's ops that connect from it |
| `Layer.parent_params` | The parameter objects that feed this layer's operation |
| `Layer.parents` | Union of parent layer labels across all ops of this layer, without pass suffixes |
| `Layer.graph.parents_per_pass` | Mapping from pass number to the parent layer labels (without pass suffixes) of that pass |
| `Layer.graph.site_peers` | Layers sharing any of this layer's site keys (live index) |
| `Layer.raw_index` | Position of this layer's first op in raw capture order |
| `Layer.resolver_status` | Whether the backend address resolves to a live object, as a string such as `resolved` or `metadata_only` |
| `Layer.call.saved_kwargs` | Keyword arguments captured for this layer's function call, for single-pass layers |
| `Layer.siblings` | Union across ops of the sibling layer labels, without pass suffixes |
| `Layer.control_flow.terminal_bool_for` | The conditional that this terminal boolean decides, or `None` |
| `Layer.total_gradient_memory` | Sum of gradient memory in bytes across all ops of this layer |
| `Layer.total_macs_backward` | Always `None`, because backward MACs are not derivable |
| `Layer.payload.transformed_grad` | Transformed gradient of the single pass, for single-pass layers |
| `Layer.payload.transformed_grad_dtype` | Dtype of the saved transformed gradient, or `None` |
| `Layer.payload.transformed_grad_shape` | Shape of the saved transformed gradient, or `None` |
| `Layer.transformed_gradient_memory` | Bytes used by the saved transformed gradient, or `None` |
| `Layer.payload.transformed_out` | Transformed output of the single pass, for single-pass layers |
| `Layer.payload.transformed_out_dtype` | Dtype of the saved transformed output, or `None` |
| `Layer.payload.transformed_out_shape` | Shape of the saved transformed output, or `None` |
| `Layer.type_index` | Index of this layer among layers of the same type, counted from 1 in execution order |
| `Layer.visualizer_path` | Path of the PNG used as this node's thumbnail in graph renders, or `None` |

#### Conditional records (`trace.graph.conditionals`)

`trace.graph.conditionals` holds one `Conditional` per `if` chain or ternary, each with its
`ConditionalArm`s; an op's `in_conditionals` lists its `ConditionalRoleRef`s. Arms come from the
source and are never invented: `then`, one `elif` per elif, `else` only when the source has one,
and `fallthrough` for the code after a chain with no `else` whose every arm provably ends in
`return`, `raise`, `break` or `continue` (that code runs only when every test was false). The
fall-through arm, its token and drawing label `FALLTHROUGH`, and `Conditional.outcome` are part of
this record; `docs/conditional_attribution.md` has the worked example.

| Spelling | Meaning |
| --- | --- |
| `Conditional.fired_arm` | The arm at `fired_arm_index`, or `None` when no single arm fired |
| `Conditional.fired_arm_kind` | Kind of the one arm whose body was witnessed firing (`then`, `elif`, `else` or `fallthrough`); `None` when no single source arm's body was witnessed firing, and `outcome` says why |
| `Conditional.has_else` | Whether the source has an `else` block; a `fallthrough` arm never counts as one |
| `Conditional.num_arms` | Number of arms, the `fallthrough` arm included |
| `Conditional.outcome` | Which way the conditional went, derived from its arms and never persisted: `then`, `elif`, `else` or `fallthrough` when that arm fired or its witnessed test selected it with no tensor op in it, `skipped` when every test was witnessed false and there is no `else` or `fallthrough` arm, `multiple` when more than one arm fired across passes, `unknown` when no evidence decides it |
| `ConditionalArm.bool_value_at_run` | Value the arm's test took when the capture ran, with a `not` in the test applied, so it never contradicts `fired`; `None` for an `else` or `fallthrough` arm (no test of its own) and for a compound `and`/`or` test, where no single bool decides the arm |
| `ConditionalArm.evaluation_ops` | Labels of the ops that hold the `evaluation` role in this arm: the ops computing its test; always empty for an `else` or `fallthrough` arm, which has no test |
| `ConditionalArm.execution_ops` | Labels of the ops that hold the `body` role in this arm: the ops its entry edges feed |
| `ConditionalArm.kind` | Arm kind: `then`, `elif`, `else` or `fallthrough` |
| `ConditionalRoleRef.arm_kind` | Kind of the arm the role is in: `then`, `elif`, `else` or `fallthrough` |

#### `tl.types.Module`

| Spelling | Meaning |
| --- | --- |
| `Module.routing_table` | Lazy list of routing rows for this MoE block alone, or `None` when routing does not apply; saved raw routing outputs are required |
| `Module.address_children` | Addresses of the modules directly nested under this module in the static module tree |
| `Module.address_depth` | Depth of this module's address in the static module tree, with the root at 0 |
| `Module.address_parent` | Address of this module's parent in the static module tree, or `None` for the root |
| `Module.backward_duration` | Backward duration for this module, or `None` when backward timing is not available |
| `Module.backward_hooks` | The backward hooks registered on this module, as a list of `HookInfo` |
| `Module.backward_pre_hooks` | The backward pre-hooks registered on this module, as a list of `HookInfo` |
| `Module.buffer_layers` | Labels of the buffer layers that belong to this module |
| `Module.call_children` | Addresses of the modules this module calls directly in the dynamic call tree |
| `Module.call_depth` | Depth of this module in the dynamic call stack |
| `Module.call_labels` | Labels of this module's calls |
| `Module.call_parent` | Address of the module that calls this one in the dynamic call tree, or `None` at the root |
| `Module.call_tree` | Return this module's single `ModuleCall` as the root of a nested call tree |
| `Module.class_docstring` | The docstring of this module's class, or `None` |
| `Module.class_name` | The class name of this module |
| `Module.class_qualname` | The qualified class name of this module |
| `Module.class_source_file` | Path of the source file that defines this module's class, or `None` |
| `Module.class_source_line` | Line number where this module's class is defined, or `None` |
| `Module.class_source_location` | The class source location as `file:line`, or `None` |
| `Module.custom_attributes` | User-defined attributes of this module beyond those of a stock `nn.Module` |
| `Module.custom_methods` | Names of user-defined methods of this module beyond those of a stock `nn.Module` |
| `Module.forward_args` | Positional arguments captured for the forward call of a single-call module, or `None` |
| `Module.forward_args_summary` | Human-readable summary of the forward positional arguments of a single-call module |
| `Module.forward_args_template` | Representative structural template of the forward positional arguments |
| `Module.forward_docstring` | The docstring of this module's `forward` method, or `None` |
| `Module.forward_duration` | Wall-clock duration of this module's single forward call |
| `Module.forward_hooks` | The forward hooks registered on this module, as a list of `HookInfo` |
| `Module.forward_kwargs` | Keyword arguments captured for the forward call of a single-call module, or `None` |
| `Module.forward_kwargs_summary` | Human-readable summary of the forward keyword arguments of a single-call module |
| `Module.forward_kwargs_template` | Representative structural template of the forward keyword arguments |
| `Module.forward_pre_hooks` | The forward pre-hooks registered on this module, as a list of `HookInfo` |
| `Module.forward_signature` | The signature of this module's `forward` method, or `None` |
| `Module.forward_source_file` | Path of the source file that defines this module's `forward` method, or `None` |
| `Module.forward_source_line` | Line number where this module's `forward` method is defined, or `None` |
| `Module.forward_source_location` | The `forward` source location as `file:line`, or `None` |
| `Module.full_backward_hooks` | The full backward hooks registered on this module, as a list of `HookInfo` |
| `Module.full_backward_pre_hooks` | The full backward pre-hooks registered on this module, as a list of `HookInfo` |
| `Module.func_calls_duration` | Inclusive torch function duration of this module's single call |
| `Module.grads` | Saved output gradients of a single-call module, as a list |
| `Module.has_backward_hooks` | Whether any backward hook list on this module is nonempty |
| `Module.has_forward_hooks` | Whether any forward hook list on this module is nonempty |
| `Module.has_frozen_params` | Whether this module owns at least one frozen parameter |
| `Module.has_multiple_addresses` | Whether this module appears at more than one address |
| `Module.has_trainable_params` | Whether this module owns at least one trainable parameter |
| `Module.init_docstring` | The docstring of this module's `__init__` method, or `None` |
| `Module.init_signature` | The signature of this module's `__init__` method, or `None` |
| `Module.init_source_file` | Path of the source file that defines this module's `__init__` method, or `None` |
| `Module.init_source_line` | Line number where this module's `__init__` method is defined, or `None` |
| `Module.init_source_location` | The `__init__` source location as `file:line`, or `None` |
| `Module.input_layers` | Labels of the layers that carry inputs into this module |
| `Module.internal_param_memory` | Bytes of parameters across this module's address subtree, including descendants |
| `Module.layer_labels` | Labels of the layers computed inside this module, without pass suffixes |
| `Module.max_descendant_depth` | Deepest nesting of module calls beneath this module |
| `Module.num_descendant_calls` | Total number of module calls nested beneath all calls of this module |
| `Module.num_edges` | Total number of internal, input and output edges of this module |
| `Module.num_input_edges` | Number of edges whose child op is inside this module and whose parent op is outside it |
| `Module.num_internal_edges` | Number of edges whose parent and child ops are both inside this module |
| `Module.num_output_edges` | Number of edges whose parent op is inside this module and whose child op is outside it |
| `Module.num_param_tensors_frozen` | Number of frozen parameter tensors owned by this module |
| `Module.num_param_tensors_trainable` | Number of trainable parameter tensors owned by this module |
| `Module.num_params_frozen` | Number of frozen parameter elements owned by this module |
| `Module.num_params_trainable` | Number of trainable parameter elements owned by this module |
| `Module.num_recursive_param_tensors` | Number of parameter tensors across this module and its address descendants |
| `Module.num_recursive_param_tensors_frozen` | Number of frozen parameter tensors across this module and its address descendants |
| `Module.num_recursive_param_tensors_trainable` | Number of trainable parameter tensors across this module and its address descendants |
| `Module.num_recursive_params` | Number of parameter records across this module and its address descendants |
| `Module.num_recursive_params_frozen` | Number of frozen parameter records across this module and its address descendants |
| `Module.num_recursive_params_trainable` | Number of trainable parameter records across this module and its address descendants |
| `Module.out_shape` | Output shape of a single-call, single-output module |
| `Module.out_shapes` | Output shapes of a single-call module, as a list |
| `Module.output_layers` | Labels of the layers that produce this module's outputs |
| `Module.output_structure` | The structure of the single call's return container, or `None` when ambiguous |
| `Module.outs` | Saved output tensors of a single-call module, as a list |
| `Module.param_memory` | Bytes of parameters at this module's own address, the same for every call |
| `Module.recursive_params` | The parameters owned by this module's address and by all of its address descendants |
| `Module.show_call_tree` | Print the call subtrees of this module as ASCII trees |
| `Module.total_backward_duration` | Total backward duration across calls, or `None` when backward timing is not available |
| `Module.total_flops_backward` | Sum of backward FLOPs across every op this module's calls ran |
| `Module.total_flops_forward` | Sum of forward FLOPs across every op this module's calls ran |
| `Module.total_forward_duration` | Sum of wall-clock forward durations across all calls of this module |
| `Module.total_func_calls_duration` | Sum of inclusive torch function durations across all calls of this module |
| `Module.total_internal_activation_memory` | Sum of internal activation bytes across this module's calls |
| `Module.total_internal_gradient_memory` | Sum of internal gradient bytes across this module's calls |
| `Module.total_macs_backward` | Always `None`, because backward MACs are not derivable |
| `Module.total_macs_forward` | True forward MACs across every op this module's calls ran, counting only ops with a known split |
| `Module.total_output_activation_memory` | Sum of output activation bytes across this module's calls |
| `Module.total_output_gradient_memory` | Sum of output gradient bytes across this module's calls |
| `Module.walk_descendants` | Yield every `ModuleCall` nested beneath any call of this module |

#### `tl.types.ModuleCall`

| Spelling | Meaning |
| --- | --- |
| `ModuleCall.backward_duration` | Backward time for this module call; currently always `None` |
| `ModuleCall.call_children` | Labels of the module calls nested directly inside this call |
| `ModuleCall.call_children_labels` | Labels of the child module calls in the dynamic call tree |
| `ModuleCall.label` | Pass-qualified label of this call, usually `address:N` |
| `ModuleCall.call_parent` | Label of the parent module call in the call tree, or `None` at the root |
| `ModuleCall.call_tree` | This call as the root of its nested `ModuleCall` subtree |
| `ModuleCall.class_name` | Class name of the module invoked by this call |
| `ModuleCall.class_qualname` | Qualified class name of the module invoked by this call |
| `ModuleCall.code_context` | List of `FuncCallLocation` frames for the call stack at the entry to `forward()`, outermost first |
| `ModuleCall.forward_arg_names` | Names of the `forward()` parameters, as resolved from its signature |
| `ModuleCall.forward_args` | Positional arguments passed to `forward()`, or `None` when not captured |
| `ModuleCall.forward_args_summary` | Short text summary of the positional `forward()` arguments |
| `ModuleCall.forward_args_template` | Structural template of the positional `forward()` arguments |
| `ModuleCall.forward_duration` | Wall-clock time in seconds spent in this call's `forward()` |
| `ModuleCall.forward_kwargs` | Keyword arguments passed to `forward()`, or `None` when not captured |
| `ModuleCall.forward_kwargs_summary` | Short text summary of the keyword `forward()` arguments |
| `ModuleCall.forward_kwargs_template` | Structural template of the keyword `forward()` arguments |
| `ModuleCall.forward_pre_hook_effects` | Tuple of `PreHookEffect` records, one per user forward pre-hook that ran on this call |
| `ModuleCall.func_calls_duration` | Total time spent in torch function calls inside this call, summed over their `Op` durations |
| `ModuleCall.grads` | Saved output gradients of this call, one per tensor output in container-path order |
| `ModuleCall.has_frozen_params` | Whether the called module owns any frozen parameters |
| `ModuleCall.has_multiple_addresses` | Whether the called module appears at more than one address |
| `ModuleCall.has_saved_forward_args` | Whether the `forward()` arguments were captured for this call |
| `ModuleCall.has_trainable_params` | Whether the called module owns any trainable parameters |
| `ModuleCall.input_layers` | Labels of the layers at the input boundary of this call |
| `ModuleCall.inputs_after_pre_hooks` | Snapshot of the module inputs after user forward pre-hooks ran, or `None` |
| `ModuleCall.inputs_before_pre_hooks` | Snapshot of the module inputs as passed to `Module.__call__`, before any pre-hook ran, or `None` |
| `ModuleCall.internal_activation_memory` | Activation bytes of the non-output `Op`s executed inside this call |
| `ModuleCall.internal_gradient_memory` | Gradient bytes of the non-output `Op`s executed inside this call |
| `ModuleCall.internal_param_memory` | Parameter bytes summed over the called module and all modules beneath it |
| `ModuleCall.max_descendant_depth` | Depth of the deepest nested `ModuleCall` beneath this call |
| `ModuleCall.num_descendant_calls` | Number of `ModuleCall`s nested under this call, counted recursively |
| `ModuleCall.num_edges` | Total number of internal, input and output edges of this call |
| `ModuleCall.num_forward_args_total` | Total number of positional and keyword arguments passed to `forward()` |
| `ModuleCall.num_forward_kwargs` | Number of keyword arguments passed to `forward()` |
| `ModuleCall.num_forward_pos_args` | Number of positional arguments passed to `forward()` |
| `ModuleCall.num_input_edges` | Number of edges whose child `Op` is inside this call and whose parent `Op` is outside |
| `ModuleCall.num_internal_edges` | Number of edges whose parent and child `Op`s are both inside this call |
| `ModuleCall.num_output_edges` | Number of edges whose parent `Op` is inside this call and whose child `Op` is outside |
| `ModuleCall.num_param_tensors` | Number of parameter tensors owned by the called module |
| `ModuleCall.num_param_tensors_frozen` | Number of frozen parameter tensors owned by the called module |
| `ModuleCall.num_param_tensors_trainable` | Number of trainable parameter tensors owned by the called module |
| `ModuleCall.ordinal_index` | 0-based position of this call among all module calls in the trace |
| `ModuleCall.out_dtype` | Output dtype of a single-output call; raises when the call has several outputs |
| `ModuleCall.out_dtypes` | Output dtypes of this call, one per tensor output in container-path order |
| `ModuleCall.out_shape` | Output shape of a single-output call; raises when the call has several outputs |
| `ModuleCall.out_shapes` | Output shapes of this call, one per tensor output in container-path order |
| `ModuleCall.output_activation_memory` | Activation bytes of the `Op`s whose outputs form this call's return value |
| `ModuleCall.output_gradient_memory` | Gradient bytes of the `Op`s whose outputs form this call's return value |
| `ModuleCall.output_layers` | Labels of the layers at the output boundary of this call |
| `ModuleCall.output_paths` | Tuple of container paths locating each tensor output inside this call's return structure |
| `ModuleCall.output_structure` | `ContainerSpec` describing the shape of this call's return container, or `None` |
| `ModuleCall.outs` | Saved output tensors of this call, one per tensor output in container-path order |
| `ModuleCall.param_memory` | Parameter bytes owned directly by the module invoked by this call |
| `ModuleCall.show_call_tree` | Print this call's subtree as an ASCII tree |
| `ModuleCall.walk_descendants` | Yield this call, if `include_self` is true, and all descendant calls in depth-first order |

#### `tl.types.Param`

| Spelling | Meaning |
| --- | --- |
| `Param.all_module_addresses` | Addresses of every module that owns this parameter, more than one when it is shared |
| `Param.backend_address` | The parameter's address as seen by the backend, normally equal to `address` |
| `Param.barcode` | Identity token for the underlying tensor, shared across aliases of the same parameter |
| `Param.co_parent_params` | Addresses of other parameters used by the same operations as this one |
| `Param.grad_dtype` | Dtype of this parameter's cached gradient, or `None` |
| `Param.grad_shape` | Shape of this parameter's cached gradient, or `None` |
| `Param.grad_memory` | Size in bytes of this parameter's cached gradient |
| `Param.grads` | Accessor over the per-backward-pass gradient increments observed for this parameter |
| `Param.has_grad` | Whether this parameter has a cached gradient |
| `Param.has_multiple_addresses` | Whether this parameter is registered at more than one address |
| `Param.has_optimizer` | Whether optimizer state was detected for this parameter, or `None` when unknown |
| `Param.is_quantized` | Whether this parameter uses a quantized dtype such as `qint8` or `quint8` |
| `Param.is_trainable` | Whether this parameter requires gradients |
| `Param.module_cls` | The live class of the owning module, or `None` when unavailable |
| `Param.param_memory` | Memory used by the parameter tensor, in bytes |
| `Param.resolver_status` | How the backend resolved this parameter, such as `"resolved"` or `"metadata_only"` |
| `Param.used_by_layers` | Labels of the `Layer`s that used this parameter |

#### `tl.types.Buffer`

| Spelling | Meaning |
| --- | --- |
| `Buffer.writer_label` | Label of the producer of the most recent write to this buffer, or `None` |
| `Buffer.final_value` | The buffer value at its last observed version, or `None` when there are no versions |
| `Buffer.grad_dtype` | Dtype of the saved gradient at the final observed version |
| `Buffer.grad_shape` | Shape of the saved gradient at the final observed version |
| `Buffer.grad_memory` | Memory used by the saved gradient at the final observed version |
| `Buffer.has_grad` | Whether the final observed version has a saved gradient |
| `Buffer.has_multiple_addresses` | Whether this buffer is registered at more than one module address |
| `Buffer.has_saved_activation` | Whether the final observed version has a saved activation |
| `Buffer.initial_value` | The buffer value at its first observed version, or `None` when there are no versions |
| `Buffer.is_overwritten` | Whether the buffer was written at least once during the forward pass |
| `Buffer.num_overwrites` | Number of write events recorded for this buffer address |
| `Buffer.value_after` | Return the buffer value after the n-th write, counting writes from 1 |
| `Buffer.value_at` | Return the buffer value at version n, counting versions from 1 |
| `Buffer.versions` | List of `Op` nodes, in order, that record each observed value of this buffer |

#### `tl.types.GradFn`

| Spelling | Meaning |
| --- | --- |
| `GradFn.backward_docstring` | Docstring of the custom autograd `backward` method, or `None` |
| `GradFn.backward_duration` | Elapsed backward time for this GradFn when it has a single measured call, or `None` when ambiguous |
| `GradFn.backward_signature` | Signature of the custom autograd `backward` method as a string, or `None` |
| `GradFn.backward_source_file` | Source file defining the custom autograd `backward` method, or `None` |
| `GradFn.backward_source_line` | Source line of the custom autograd `backward` method, or `None` |
| `GradFn.backward_source_location` | Source location of the custom autograd `backward` method as a string, or `None` |
| `GradFn.call_labels` | Pass-qualified labels of this GradFn's calls |
| `GradFn.children` | Labels of the GradFns that fire after this one in the backward pass |
| `GradFn.class_docstring` | Docstring of the grad-fn class, or `None` |
| `GradFn.class_name` | Name of the PyTorch grad-fn class |
| `GradFn.class_qualname` | Qualified name of the PyTorch grad-fn class, including its module |
| `GradFn.class_source_file` | Source file defining the grad-fn class, or `None` |
| `GradFn.class_source_line` | Source line where the grad-fn class is defined, or `None` |
| `GradFn.class_source_location` | Source location of the grad-fn class as a string, or `None` |
| `GradFn.co_parents` | Labels of the GradFns that feed the same child GradFns as this one |
| `GradFn.creator_object_id` | The `id()` of the object that created this grad-fn, or `None` |
| `GradFn.differentiates` | Label of the forward `Op` this GradFn differentiates, or `None` when unknown |
| `GradFn.forward_docstring` | Docstring of the custom autograd `forward` method, or `None` |
| `GradFn.forward_signature` | Signature of the custom autograd `forward` method as a string, or `None` |
| `GradFn.forward_source_file` | Source file defining the custom autograd `forward` method, or `None` |
| `GradFn.forward_source_line` | Source line of the custom autograd `forward` method, or `None` |
| `GradFn.forward_source_location` | Source location of the custom autograd `forward` method as a string, or `None` |
| `GradFn.grad_fn_object_id` | The `id()` of the grad-fn object at capture time, used as its key for the trace lifetime |
| `GradFn.has_children` | Whether this GradFn has children in the backward graph |
| `GradFn.has_co_parents` | Whether this GradFn has co-parents in the backward graph |
| `GradFn.has_op` | Whether a forward `Op` was captured for this GradFn |
| `GradFn.has_parents` | Whether this GradFn has parents in the backward graph |
| `GradFn.has_saved_call` | Whether any call of this GradFn has recorded call data |
| `GradFn.has_siblings` | Whether this GradFn has siblings in the backward graph |
| `GradFn.init_docstring` | Docstring of the grad-fn class `__init__`, or `None` |
| `GradFn.init_signature` | Signature of the grad-fn class `__init__` as a string, or `None` |
| `GradFn.init_source_file` | Source file defining the grad-fn class `__init__`, or `None` |
| `GradFn.init_source_line` | Source line of the grad-fn class `__init__`, or `None` |
| `GradFn.init_source_location` | Source location of the grad-fn class `__init__` as a string, or `None` |
| `GradFn.is_custom` | Whether this GradFn comes from a user-defined autograd `Function` |
| `GradFn.module_membership_source` | How this GradFn was assigned to a module, such as `"inferred"` or `"paired"`, or `None` |
| `GradFn.next_grad_fn_ids` | Object ids of the grad-fn nodes that follow this one, taken from torch's `next_functions` |
| `GradFn.order` | Integer position of this GradFn in the discovered backward traversal, or `None` when unresolved |
| `GradFn.ordinal_index` | 0-based position of this GradFn in the order grad-fns were discovered in the trace |
| `GradFn.origin_backward_pass` | Index of the backward pass in which this GradFn was first encountered, or `None` |
| `GradFn.parents` | Labels of the GradFns that fire before this one in the backward pass |
| `GradFn.siblings` | Labels of the other GradFns that share a parent with this one |
| `GradFn.total_backward_duration` | Total backward time summed over this GradFn's measured calls |
| `GradFn.type_index` | 1-based index of this GradFn among those of the same normalized type |

#### `tl.types.GradFnCall`

| Spelling | Meaning |
| --- | --- |
| `GradFnCall.backward_duration` | Measured backward time for this call, or `None` when not measured |
| `GradFnCall.backward_pass_number` | 1-based number of the backward pass this call belongs to, or `None` |
| `GradFnCall.label` | Pass-qualified label of this call, in the form `label:call_index` |
| `GradFnCall.grad_inputs` | Gradients flowing out of this grad-fn toward earlier ops (the gradients with respect to its forward inputs), as observed by the backward hook |
| `GradFnCall.grad_outputs` | Gradients flowing into this grad-fn from later ops (the gradients with respect to its forward outputs), as observed by the backward hook |
| `GradFnCall.is_saved` | Whether this call has saved gradient payloads |
| `GradFnCall.ordinal_index` | 0-based position of this call in `trace.autograd.grad_fn_calls` |
| `GradFnCall.timestamp` | Wall-clock time in seconds when the hook fired, or `None` |

#### `tl.types.BackwardPass`

| Spelling | Meaning |
| --- | --- |
| `BackwardPass.backward_call_context` | Source location of the user's `backward` call that opened this pass, or `None` when it was not recorded |
| `BackwardPass.engine_flags` | Keyword flags passed to the autograd backward call, such as `create_graph` and `retain_graph`, or `None` |
| `BackwardPass.implicit` | Whether this pass was inferred from orphan tensor-hook emissions rather than an observed backward call |
| `BackwardPass.inputs_subset` | The `inputs` subset given to the backward call, as a tuple, empty when none was given |
| `BackwardPass.order` | Position of this pass in the discovered backward traversal order, or `None` when it could not be resolved |
| `BackwardPass.order_attribution_coverage` | Fraction of this pass's grad-fn calls whose traversal order was resolved, or `None` when the pass has no calls |
| `BackwardPass.ordinal_index` | 0-based position of this pass among the trace's backward passes |
| `BackwardPass.origin_backward_pass` | Index of the earlier backward pass that created the graph this pass walks, or `None` |
| `BackwardPass.outer_context` | Outer TorchLens trigger context when this pass was nested inside another, or `None` |
| `BackwardPass.peak_memory` | Peak memory observed during this pass in bytes, or `None` when not measured |
| `BackwardPass.root_grad_arguments` | The gradient arguments given to the backward call for its roots, or `None` |
| `BackwardPass.root_grad_fn_ids` | Object ids of the root grad-fn nodes this pass started from |
| `BackwardPass.root_meta` | Tuple of shape, dtype and device descriptions, one per root tensor of the backward call |
| `BackwardPass.save_grads_policy` | Text form of the gradient-saving policy active during this pass, or `None` |
| `BackwardPass.trigger` | The entry point that opened this backward pass, as a string |

#### `tl.types.FuncCallLocation`

| Spelling | Meaning |
| --- | --- |
| `FuncCallLocation.call_line` | The exact source line that ran this frame, or `''` when the text is unavailable. Without `save_source_context=True` the text is read from the file when accessed, so it reflects the file on disk at read time, and only while the file keeps the size and modification time it had at capture |
| `FuncCallLocation.code_context` | The source lines around the call site as a list, or `None` when source is unavailable (read lazily like `call_line`) |
| `FuncCallLocation.code_context_labeled` | The surrounding source context with the call line marked |
| `FuncCallLocation.code_firstlineno` | First source line of the function that contains the call |
| `FuncCallLocation.col_offset` | Column offset of the call within its source line, or `None` when unknown |
| `FuncCallLocation.func_docstring` | Docstring of the called function, or `None` when unavailable |
| `FuncCallLocation.func_qualname` | Qualified name of the called function, or `None` when unknown |
| `FuncCallLocation.func_signature` | Signature of the called function as a string, or `None` when unavailable |
| `FuncCallLocation.line_number` | One-indexed source line of the call |
| `FuncCallLocation.num_context_lines` | Number of source context lines actually loaded around the call |
| `FuncCallLocation.source_context` | The surrounding source context as a single string |
| `FuncCallLocation.source_loading_enabled` | Whether this location holds a source-text snapshot (`True` by default; captures set it from `save_source_context`). When `False` the text is read lazily from the file, and `repr()` names the reason when it cannot be shown (`source changed on disk since capture`, `source file not found`) |
| `FuncCallLocation.to_html_link` | Return a VS Code source-location link for this call, for HTML renderers |

#### `tl.types.PreHookEffect`

| Spelling | Meaning |
| --- | --- |
| `PreHookEffect.after_fingerprint` | Metadata-only fingerprint of the inputs after the hook ran |
| `PreHookEffect.before_fingerprint` | Metadata-only fingerprint of the inputs before the hook ran |
| `PreHookEffect.change_kinds` | Tuple of short strings naming the kinds of change the hook made to the inputs |
| `PreHookEffect.changed_paths` | Tuple of typed input paths that the hook changed |
| `PreHookEffect.exception_message` | Message of the exception that interrupted the hook, or `None` |
| `PreHookEffect.hook` | Descriptive metadata about the hook, as a `HookInfo` |
| `PreHookEffect.incomplete_reasons` | Tuple of stable reason strings for why change evidence or attribution was downgraded |
| `PreHookEffect.registration_id` | PyTorch's registry identifier for the hook, kept for diagnostics |
| `PreHookEffect.with_kwargs` | Whether PyTorch invoked the hook with keyword arguments |

#### `tl.types.ModuleInputSnapshot`

| Spelling | Meaning |
| --- | --- |
| `ModuleInputSnapshot.incomplete_reasons` | Tuple of stable reason strings explaining why the evidence is incomplete |
| `ModuleInputSnapshot.input_structure` | Compact immutable description of the input container topology |
| `ModuleInputSnapshot.tensor_observations` | Ordered `TensorInputObservation`s of the input tensors, keyed by typed input path |

#### `tl.types.TensorInputObservation`

| Spelling | Meaning |
| --- | --- |
| `TensorInputObservation.alias_group` | Snapshot-local identity shared by input paths that alias the same tensor |
| `TensorInputObservation.payload_origin` | Description of how the retained payload was obtained |
| `TensorInputObservation.payload_status` | `"saved"`, or the reason no payload is available |
| `TensorInputObservation.source_op` | Label of the final producer `Op` of this tensor, or `None` when unavailable |

#### `tl.recording.RecordContext`

| Spelling | Meaning |
| --- | --- |
| `RecordContext.backend_requires_isolation` | Whether predicate evaluation must be isolated for backend safety |
| `RecordContext.bool_value` | Scalar boolean value when it can be observed safely, otherwise `None` or a deferred value |
| `RecordContext.event_index` | Chronological index of this event within the active capture |
| `RecordContext.input_output_address` | Structural address of an input or output boundary event, or `None` |
| `RecordContext.is_bottom_level_func` | Whether the event came from a leaf decorated function call, or `None` when not applicable |
| `RecordContext.is_output_parent` | Whether this event is a direct parent of a model output |
| `RecordContext.is_scalar_bool` | Whether the output is a scalar boolean, possibly a deferred value |
| `RecordContext.is_transform` | Whether this event is a captured transform boundary |
| `RecordContext.label_prefix` | Short raw label alias for the event, for example `"relu_1"` |
| `RecordContext.label_raw` | The full raw label of the event, or an empty string when unavailable; never the final label |
| `RecordContext.module_pass_number` | Call index of the nearest active module, or `None` |
| `RecordContext.module_stack` | Active module frames from outermost to innermost |
| `RecordContext.output_of_module_calls` | Labels of the module calls whose output this event is, empty when it is not a module-call output |
| `RecordContext.parent_labels` | Parent labels exposed to the predicate |
| `RecordContext.parent_labels_raw` | Full raw parent labels, when the backend records them separately |
| `RecordContext.raw_index` | One-based operation index used in raw graph labels, or `None` |
| `RecordContext.recent_events` | Bounded chronological lookback of earlier events, as `RecordContext`s |
| `RecordContext.recent_ops` | Operation-only view of the bounded lookback, as `RecordContext`s |
| `RecordContext.sample_id` | Identifier supplied by a batched predicate caller, or `None` |
| `RecordContext.tensor_device` | Backend-neutral reference to the output tensor's device, or `None` |
| `RecordContext.tensor_requires_grad` | Whether the output tensor requires gradients, `None` when unknown, or a deferred value |
| `RecordContext.time_since_pass_start` | Wall-clock seconds elapsed since capture began |
| `RecordContext.type_index` | One-based occurrence index of this event within its `layer_type`, or `None` |
| `RecordContext.window_miss` | Whether a history-dependent selector looked past the retained lookback window |

#### `tl.recording.ModuleStackFrame`

| Spelling | Meaning |
| --- | --- |
| `ModuleStackFrame.module_id` | The `id()` of the module object at this stack frame |

#### `tl.stats.HistogramDescriptor`

| Spelling | Meaning |
| --- | --- |
| `HistogramDescriptor.base` | Logarithm base of the bin grid; only `2` is supported |
| `HistogramDescriptor.bins_per_octave` | Number of bins per factor of two in magnitude |
| `HistogramDescriptor.bins_per_side` | Number of magnitude bins on each sign side |
| `HistogramDescriptor.bucket_edges` | Return the ascending magnitude bin edges for one sign side |
| `HistogramDescriptor.hi_exp` | Upper edge of the grid as a power of the base |
| `HistogramDescriptor.lo_exp` | Lower edge of the grid as a power of the base |

#### `tl.stats.HistogramResult`

| Spelling | Meaning |
| --- | --- |
| `HistogramResult.counts_fit_int32` | Return whether every stored count fits the signed thirty-two-bit integer used by the observation encoding |
| `HistogramResult.neg_counts` | Per-bin counts for the negative side, in ascending magnitude |
| `HistogramResult.pos_counts` | Per-bin counts for the positive side, in ascending magnitude |
| `HistogramResult.specials` | Exact counts of zeros, per-side underflow and overflow, NaN and infinities, keyed by name |

<!-- trace group views: generated by export_groups.py -->
**`tl.types.TraceCost`**
: Compute, memory and timing totals of a Trace; `trace.cost` is the instance a Trace serves.
  - `TraceCost.backward_durations`: seconds per backward pass; `last_backward_duration` /
    `total_backward_duration` (@properties).
  - `TraceCost.backward_peak_memory`: peak memory during forward / backward. On CPU/MPS this is only
    the cheap host RSS (or MPS allocator) delta and legitimately reads `0` for small models;
    `CaptureOptions(measure_python_peak_memory=True)` opts into a `tracemalloc` Python-allocation
    peak at 1.7x-2.5x capture-time cost (session-time only, never persisted).
  - `TraceCost.flops_by_op_type`: group FLOPs by layer type.
  - `TraceCost.forward_duration`: phase timings (`func_calls_duration` = pure model compute proxy).
  - `TraceCost.forward_peak_memory`: peak memory during forward / backward. On CPU/MPS this is only
    the cheap host RSS (or MPS allocator) delta and legitimately reads `0` for small models;
    `CaptureOptions(measure_python_peak_memory=True)` opts into a `tracemalloc` Python-allocation
    peak at 1.7x-2.5x capture-time cost (session-time only, never persisted).
  - `TraceCost.forward_peak_memory_pair`: session-time (live, resident) forward peak pair with
    backend named.
  - `TraceCost.forward_peak_memory_reason`: why the forward peak memory is absent or estimated, if
    it is.
  - `TraceCost.func_calls_duration`: phase timings (`func_calls_duration` = pure model compute
    proxy).
  - `TraceCost.last_backward_duration`: most recent backward-pass duration, if any.
  - `TraceCost.macs_by_kind`: group TRUE forward MACs by layer type.
  - `TraceCost.overhead_duration`: time spent on TorchLens overhead (total minus function calls).
  - `TraceCost.pinned_autograd_memory`: bytes of autograd memory pinned by retained saved tensors.
  - `TraceCost.saved_activation_memory`: bytes for all / actually-saved Op activations.
  - `TraceCost.saved_grad_memory`: bytes for all / actually-saved output gradients.
  - `TraceCost.total_activation_memory`: bytes for all / actually-saved Op activations.
  - `TraceCost.total_autograd_memory`: bytes autograd retained for backward.
  - `TraceCost.total_backward_duration`: sum of all captured backward-pass durations.
  - `TraceCost.total_backward_memory`: total bytes of backward activations retained.
  - `TraceCost.total_flops`: total FLOPs (forward + backward).
  - `TraceCost.total_flops_backward`: total backward FLOPs across all layers (skipping
    None/unknown).
  - `TraceCost.total_flops_forward`: total forward FLOPs (the canonical identity-partition
    aggregation).
  - `TraceCost.total_grad_memory`: bytes for all / actually-saved output gradients.
  - `TraceCost.total_macs`: total MACs are not derivable and always ``None`` (see
    total_macs_backward).
  - `TraceCost.total_macs_backward`: backward MACs are not derivable and always ``None`` (estimates
    only).
  - `TraceCost.total_macs_forward`: total TRUE forward MACs over ops with a known MAC split.
  - `TraceCost.total_param_grad_memory`: bytes for parameter tensors / populated parameter
    gradients.
  - `TraceCost.total_param_memory`: bytes for parameter tensors / populated parameter gradients.

**`tl.types.TraceGraph`**
: Graph-shape facts of a Trace: edges, degrees, branching and conditionals; `trace.graph` is the
  instance a Trace serves.
  - `TraceGraph.branching_factor`: mean fan-out: children (consumers) per compute Op.
  - `TraceGraph.conditional_arm_entry_edges`: entry edges of each conditional arm, keyed by
    conditional.
  - `TraceGraph.conditional_branch_edges`: edges that branch on a conditional.
  - `TraceGraph.conditional_edge_call_indices`: call indices of the ops on each conditional edge.
  - `TraceGraph.conditional_records`: raw conditional-flow records in trace order.
  - `TraceGraph.conditionals`: conditionalAccessor (by int ordinal or stable label).
  - `TraceGraph.edges`: return this trace's dataflow edge family (finalized edge views).
  - `TraceGraph.has_conditional_branching`: whether any layer is in a conditional branch.
  - `TraceGraph.has_conditionals`: whether this Trace contains conditional-flow records.
  - `TraceGraph.is_branching`: whether any layer has more than one child.
  - `TraceGraph.is_dynamic`: whether execution depends on runtime tensor values.
  - `TraceGraph.is_recurrent`: whether any layer has more than one pass.
  - `TraceGraph.manual_tensor_connections`: user-declared tensor connections added to the graph.
  - `TraceGraph.max_in_degree`: maximum number of parents over compute Ops.
  - `TraceGraph.max_ops_per_layer`: maximum number of ops for any layer.
  - `TraceGraph.max_out_degree`: maximum number of children over compute Ops.
  - `TraceGraph.num_buffer_edges`: distinct Op graph edges with at least one buffer endpoint.
  - `TraceGraph.num_compute_edges`: distinct Op graph edges whose endpoints are both compute Ops.
  - `TraceGraph.num_layer_edges`: distinct edges in the aggregate Layer graph.
  - `TraceGraph.op_equivalence_classes`: dict of equivalence-class id → set of Op labels.
  - `TraceGraph.orphans`: access retained orphan island operations by raw or final label.
  - `TraceGraph.relationship_evidence`: evidence behind each inferred op relationship.
  - `TraceGraph.stop_frontier`: early-halt state (partial trace).
  - `TraceGraph.structure_evidence`: evidence behind the structure-only reconstruction.
  - `TraceGraph.tied_param_groups`: alias-address groups of parameters registered at more than one
    path.

**`tl.types.TraceAutograd`**
: Backward-pass records and grad_fn facts of a Trace; `trace.autograd` is the instance a Trace
  serves.
  - `TraceAutograd.backward_passes`: access backward pass records by 0-based position or named pass
    number.
  - `TraceAutograd.backward_root_grad_fn_object_ids`: root grad-fn `object_id` per backward pass;
    `last_backward_root_grad_fn_object_id`.
  - `TraceAutograd.derived_grads`: derivedGradAccessor for leaf-level derived gradients (non-torch
    backends / explicit derivation).
  - `TraceAutograd.disarm`: detach this Trace from global autograd backward interception so plain
    backward calls no longer record into it.
  - `TraceAutograd.escape_detector_backward_coverage`: backward coverage reported by the escape
    detector.
  - `TraceAutograd.grad_fn_calls`: access per-invocation GradFnCall records by qualified label or
    index.
  - `TraceAutograd.grad_fn_fire_timings`: return live per-fire backward timing spans keyed like
    ``grad_fn_calls``.
  - `TraceAutograd.grad_fn_order`: grad_fn labels in backward execution order.
  - `TraceAutograd.grad_fns`: access backward grad_fn_handle metadata by label, index, pass label,
    or substring.
  - `TraceAutograd.has_backward_pass`: backward data / gradient values present.
  - `TraceAutograd.has_grads`: backward data / gradient values present.
  - `TraceAutograd.intermediate_derived_grads`: access exact op-level derived gradient records, when
    a backend provides them.
  - `TraceAutograd.last_backward_pass`: return the most recent backward pass record, if any.
  - `TraceAutograd.last_backward_root_grad_fn_object_id`: most recent backward root grad_fn_handle
    object id, if any.
  - `TraceAutograd.num_backward_edges`: distinct edges in the backward GradFn graph, or ``None`` if
    ungated.
  - `TraceAutograd.saved_grad_fn_calls`: access GradFnCall records with saved gradient inputs or
    outputs.
  - `TraceAutograd.saved_grad_fns`: access GradFns containing at least one saved GradFnCall.

**`tl.types.TraceSource`**
: Where the traced model's source lives: the class, `__init__` and `forward` locations (file, line,
  combined location), the forward signature, the three docstrings, the captured frames, and the live
  class while the model is still alive; `trace.source` is the instance a Trace serves.
  - `TraceSource.class_docstring`: model class location and docs.
  - `TraceSource.class_file`: model class location and docs.
  - `TraceSource.class_line`: model class location and docs.
  - `TraceSource.class_location`: combined model class source location.
  - `TraceSource.class_name`: short Python class name of the source model.
  - `TraceSource.class_qualname`: fully qualified class name; useful for run verification.
  - `TraceSource.cls`: return the live source model class when the model is still alive.
  - `TraceSource.forward_docstring`: `forward()` info.
  - `TraceSource.forward_file`: `forward()` info.
  - `TraceSource.forward_line`: `forward()` info.
  - `TraceSource.forward_location`: combined forward source location.
  - `TraceSource.forward_signature`: `forward()` info.
  - `TraceSource.frames`: python call-stack frames at capture time (`list[FuncCallLocation]`).
  - `TraceSource.init_docstring`: `__init__` info.
  - `TraceSource.init_file`: `__init__` info.
  - `TraceSource.init_line`: `__init__` info.
  - `TraceSource.init_location`: combined model ``__init__`` source location.
  - `TraceSource.init_signature`: `__init__` info.
  - `TraceSource.snapshots`: source snapshots taken for the traced model.

**`tl.types.TraceProvenance`**
: How, when and from what this Trace was captured; `trace.provenance` is the instance a Trace
  serves.
  - `TraceProvenance.backend`: `Literal["torch","tf","jax","mlx","paddle","tinygrad"]` capture
    backend tag. Portable.
  - `TraceProvenance.backend_config`: backend runtime snapshot captured at trace time.
  - `TraceProvenance.backend_version`: backend runtime snapshot captured at trace time.
  - `TraceProvenance.backward_memory_backend`: backend used to measure backward memory.
  - `TraceProvenance.cache_hit`: whether this capture was served from the capture cache.
  - `TraceProvenance.cache_key`: key of the capture-cache entry this capture used or wrote.
  - `TraceProvenance.cache_path`: path of the capture-cache entry, when one was used.
  - `TraceProvenance.cleanup_duration`: phase timings (`func_calls_duration` = pure model compute
    proxy).
  - `TraceProvenance.distributed_scope`: distributed scope of the capture (rank and world facts), if
    any.
  - `TraceProvenance.duration`: total elapsed. `overhead_duration`: TorchLens overhead (capture
    minus func calls).
  - `TraceProvenance.end_time`: epoch timestamps.
  - `TraceProvenance.fingerprint`: hash of captured topology + shape-relevant structure.
  - `TraceProvenance.forward_memory_backend`: memory backend that measured the forward peak.
  - `TraceProvenance.input_annotations`: user-attached metadata (Trace-wide / inputs).
  - `TraceProvenance.input_object_id`: python `id()` of the input at capture. Within-session only.
  - `TraceProvenance.input_preprocessor`: descriptor of the input preprocessor applied before
    capture, if any.
  - `TraceProvenance.input_signature_hash`: hash of input shapes/dtypes/devices.
  - `TraceProvenance.input_structure`: structural descriptor of the model input (shape/dtype/pytree
    spec).
  - `TraceProvenance.is_appended`: chunked-append provenance.
  - `TraceProvenance.jax_closed_jaxpr`: closed jaxpr of the traced JAX function.
  - `TraceProvenance.jax_inlined_call_primitives`: JAX call primitives inlined during capture.
  - `TraceProvenance.kind`: kind of capture (live, cooked from a Recording, loaded).
  - `TraceProvenance.model_object_id`: python `id()` of the model at capture time. Within-session
    only.
  - `TraceProvenance.observer_spans`: observer spans recorded during the capture.
  - `TraceProvenance.output_postprocessor`: output-decoding descriptors.
  - `TraceProvenance.owner_thread_id`: thread id of the thread that owned the capture.
  - `TraceProvenance.owner_thread_qualified`: whether the owner thread was fully qualified for the
    capture.
  - `TraceProvenance.param_hash_full`: parameter-value hashes.
  - `TraceProvenance.param_hash_quick`: parameter-value hashes.
  - `TraceProvenance.param_source`: how params were derived — `"native-module"`, `"pytree-derived"`,
    `"none"`.
  - `TraceProvenance.recording_kept`: whether the capture's Recording was kept.
  - `TraceProvenance.root_entry_point`: persisted identity fact disclosing a bound-method capture
    root (`"bound_method:<owner>.<method>"`), absent on ordinary module-root captures.
  - `TraceProvenance.setup_duration`: phase timings (`func_calls_duration` = pure model compute
    proxy).
  - `TraceProvenance.start_time`: epoch timestamps.
  - `TraceProvenance.summary()`: the capture-provenance block (the relocated summary preamble).
  - `TraceProvenance.thread_activity_detected`: whether other threads were active during the
    capture.
  - `TraceProvenance.thread_count_end`: thread count when the capture ended.
  - `TraceProvenance.thread_count_start`: thread count when the capture started.
  - `TraceProvenance.verification_reason`: machine-readable reason a live capture is not positively
    verified.
  - `TraceProvenance.verified`: whether the live capture is positively verified (``None`` when not
    recorded).

**`TraceProvenance.summary()`**
: `trace.provenance.summary()` returns the capture-provenance text block as a string: the backend, timing, cache, thread, input and parameter facts of the capture, the relocated summary preamble byte-for-byte. The Trace twin of `Bundle.provenance.summary()`. The view type names `TraceProvenance` and `TraceHealth` are confirmed (JMT 1 Nov 2026 3:00 PM PT).

**`tl.types.TraceSettings`**
: The capture options this Trace ran with (the stored snapshot); `trace.options` is the instance a
  Trace serves.
  - `TraceSettings.activation_transform`: callable applied to saved forward outputs before storage.
  - `TraceSettings.backward_ready`: backward machinery prepared at capture. `inference_only`:
    captured under `torch.no_grad()`.
  - `TraceSettings.chunk_size`: chunked-append provenance.
  - `TraceSettings.detach_saved_activations`: saved tensors detached from autograd.
  - `TraceSettings.emit_nvtx`: capture toggles.
  - `TraceSettings.grad_transform`: callable applied to saved gradients.
  - `TraceSettings.grouping`: `"structural"` (default), `"strict_shapes"`/`"fold_sites"` (reserved;
    they refuse typed).
  - `TraceSettings.grouping_policy`: stamp of the layer-grouping policy that produced the layers.
  - `TraceSettings.inference_only`: whether the capture ran under inference mode.
  - `TraceSettings.input_transform`: portable `repr()` of the input / gradient transform callables.
    (`gradient_transform` ruled v12; not yet shipped -- the field itself is still
    `grad_transform`.).
  - `TraceSettings.jax_static_argnums`: static argument positions passed to the JAX capture.
  - `TraceSettings.keep_orphans`: whether orphan island Ops are retained.
  - `TraceSettings.mark_layer_depths`: capture toggles.
  - `TraceSettings.mode`: `"exhaustive"`, `"fast"`, or `"predicate"` (the capture-mechanism word).
  - `TraceSettings.module_identity_mode`: how modules are identified — `"torch_module"`,
    `"pytree_module"`, `"function_root"`, `"object_module"`.
  - `TraceSettings.on_nonfinite`: capture toggles.
  - `TraceSettings.output_device`: device where saved outputs are placed. `save_mode`: tensor
    retention mode (`"copy"`, `"reference"`, `"view"`, `"cpu_async"`).
  - `TraceSettings.recurrence_detection`: loop/recurrence detection on/off.
  - `TraceSettings.save`: the save selector the capture ran with.
  - `TraceSettings.save_arg_templates`: deep-copied Op arguments / structural arg templates.
  - `TraceSettings.save_args`: deep-copied Op arguments / structural arg templates.
  - `TraceSettings.save_grads`: whether operation gradients are retained after backward.
  - `TraceSettings.save_mode`: how saved activations were retained (``copy`` or ``reference``).
  - `TraceSettings.save_raw_activations`: whether raw untransformed tensors are saved.
  - `TraceSettings.save_raw_grads`: whether raw untransformed tensors are saved.
  - `TraceSettings.save_raw_input`: raw-I/O save policy + batch rendering policy for visualization.
  - `TraceSettings.save_rng`: RNG snapshots. `save_code_context`: source-code context.
  - `TraceSettings.save_source_context`: whether source text was snapshotted at capture for each
    frame (otherwise it is read lazily from disk while the file is unchanged).
  - `TraceSettings.save_transformed_output`: raw-I/O save policy + batch rendering policy for
    visualization.
  - `TraceSettings.seed`: RNG seed applied for capture, if any.
  - `TraceSettings.source_context_lines`: number of source-context lines saved per op.
  - `TraceSettings.structure_only`: see Capture products above.
  - `TraceSettings.verbose`: capture toggles.

**`tl.types.TraceHealth`**
: Honesty diagnostics of the capture: witness and escape-detector facts; `trace.health` is the
  instance a Trace serves.
  - `TraceHealth.completeness_callback_ns`: time spent in the completeness-witness callback, in
    nanoseconds.
  - `TraceHealth.completeness_decompositions`: completeness-witness decompositions of the captured
    op stream.
  - `TraceHealth.completeness_mode`: mode the completeness witness ran in.
  - `TraceHealth.completeness_verified`: whether the completeness witness verified the capture.
  - `TraceHealth.escape_detector_callback_ns`: time spent in the escape-detector callback, in
    nanoseconds.
  - `TraceHealth.escape_detector_mode`: mode the escape detector ran in.
  - `TraceHealth.escape_detector_verified`: whether the escape detector verified the capture.
  - `TraceHealth.num_accounted_witness_events`: number of witness events accounted for by captured
    ops.
  - `TraceHealth.num_escape_detector_events`: number of escape-detector events observed.
  - `TraceHealth.num_expected_opaque_witness_events`: number of witness events expected to be
    opaque.
  - `TraceHealth.num_unaccounted_witness_events`: number of witness events no captured op accounts
    for.
  - `TraceHealth.num_witness_events`: number of completeness-witness events observed.

<!-- end trace group views -->
