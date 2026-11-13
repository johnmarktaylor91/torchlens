# Where to start: NeuroAI

## Start here

- [The ResNet-18 running example](../field-guide/running_examples.md#resnet-18-on-one-image)
  on one image.
- [Extract features for a stimulus set](../field-guide/extract_features.md): per-layer
  activations in stimulus order, written with the model's identity and the preprocessing.
- [See the graph and inspect any operation](../field-guide/see_and_inspect.md).

## Going further

- [The NeuroAI feature-extraction journey](../neuroai/journey.md): load, verify the input
  side, address the layer, extract at scale, hand the features on, prove it.
- [Task RNN circuits](../neuroai/task_rnn_circuits.md): train a recurrent network on a
  memory task, edit windows of timesteps, find its fixed points with `torch.autograd`.
- [A protein language model, traced](../neuroai/protein_language_model.md): ESM-2 attention
  maps, a per-residue readout and a contact map.
- [Loading models for feature extraction](../neuroai/loaders.md) and
  [bring your own alignment](../neuroai/byo_alignment.md).
- [torchlens.neuro](../reference/neuro.md) and [the brain-benchmarking seam](../reference/brain_benchmarking_seam.md):
  rsatoolbox and Brain-Score are where the analysis runs.
- [Receptive and projective fields](../receptive_projective_fields.md).

## Coming from another tool

[thingsvision](../migration/from_thingsvision.md),
[torchvision feature extraction](../migration/from_torchvision_feature_extraction.md),
plain [forward hooks](../migration/from_hooks.md). Where another tool is the better choice:
[where to go instead](../migration/where_to_go_instead.md).
