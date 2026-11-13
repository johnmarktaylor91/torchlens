# Where to start: mechanistic interpretability

## Start here

- [The decoder running example](../field-guide/running_examples.md#the-decoder-on-the-ioi-prompt)
  on the IOI prompt.
- [Intervene on a site and measure the effect](../field-guide/intervene.md): ablation, the
  hook check, and the same edit bound live for sweeps and `generate()`.
- [See the graph and inspect any operation](../field-guide/see_and_inspect.md).

## Going further

- [The mech-interp kit](../reference/mechinterp.md): residual decomposition, direct logit
  contributions, per-head results, the component edge graph, patching grids.
- [Intervention reference](../intervention_api.md) and [the intervention explainers](../intervention_explainers.md).
- Recipes: [ACDC on IOI](../field-guide/acdc_ioi.md), [pretrained SAEs](../field-guide/pretrained_saes.md),
  [tuned lens](../field-guide/tuned_lens.md), [conditional steering](../field-guide/conditional_steering.md).
- [Performance: choosing a fast path](../guides/fast_paths.md).

## Coming from another tool

[TransformerLens](../migration/from_transformerlens.md), [nnsight](../migration/from_nnsight.md),
[pyvene](../migration/from_pyvene.md), [SAELens](../migration/saelens.md), plain
[forward hooks](../migration/from_hooks.md). Where another tool is the better choice:
[where to go instead](../migration/where_to_go_instead.md).
