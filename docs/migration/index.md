# Relation to other packages

How TorchLens relates to the tools people already use. Checked 2026-11-03.

<!-- JMT review: plain factual prose, written for review. -->

- [Which tool for this job?](comparison.md): the dated table, and when five lines of hooks are
  enough.
- [Comparable tools, checked November 2026](landscape.md): each tool's status and source.
- [Using TorchLens with other tools](interop.md): the bridges, by direction.
- [Where to go instead](where_to_go_instead.md): jobs another tool does and TorchLens does not.

## Coming from another tool

Every page has the same sections: the same task in both tools, a translation table, what
TorchLens adds, what the other tool does that TorchLens does not, using both, and semantic
differences.

<!-- coming-from-index: start -->
- PyTorch building blocks: [forward hooks](from_hooks.md), [torch.fx](from_fx.md), [Hugging Face `output_hidden_states` and `output_attentions`](from_hf_kwargs.md).
- Interpretability and intervention: [TransformerLens](from_transformerlens.md), [nnsight](from_nnsight.md), [pyvene](from_pyvene.md), [baukit](baukit.md), [AutoCircuit](autocircuit.md), [SAELens](saelens.md), [RepE](repe.md), [penzai](penzai.md).
- Attribution: [Captum](from_captum.md), [Inseq](inseq.md), [transformers-interpret](from_transformers_interpret.md).
- Feature extraction: [torchvision's feature extractor](from_torchvision_feature_extraction.md), [torchvision's `IntermediateLayerGetter`](from_intermediate_layer_getter.md), [timm `features_only`](from_timm_features_only.md), [thingsvision](from_thingsvision.md), [torchextractor](from_torchextractor.md), [surgeon-pytorch](from_surgeon_pytorch.md), [model zoos](from_model_zoos.md).
- Inspection and drawing: [torchinfo](from_torchinfo.md), [flax tabulate](from_flax_tabulate.md), [torchview](from_torchview.md), [torchviz or hiddenlayer](from_torchviz_hiddenlayer.md), [BertViz or Ecco](from_bertviz_ecco.md), [torchsnooper](coming_from_torchsnooper.md), [lovely-tensors](lovely_stats_line.md).
<!-- coming-from-index: end -->
