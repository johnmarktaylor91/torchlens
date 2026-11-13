# TorchLens documentation

## Start

- [The two running examples](field-guide/running_examples.md): a small decoder on the IOI
  prompt and ResNet-18 on one image, each a few lines ending in `trace = tl.trace(model, x)`.
- Core tasks: [see and inspect](field-guide/see_and_inspect.md),
  [extract features](field-guide/extract_features.md),
  [intervene](field-guide/intervene.md).
- [Quickstart: the input ladder](quickstart.md).
- [Also in TorchLens](also_in_torchlens.md): every feature the README tour does not show, grouped
  by task, one line each.

## What it shows

- [The Reproduction Gallery](gallery/index.md): published claims reproduced with TorchLens, each
  checked against an independent oracle with controls, a saved artifact and a stamp.
- [The basis grid](gallery/basis_grid.md): the six Reproduction Gallery entries against the six
  verbs they use.
- [The any-architecture strip](gallery/any_architecture.md): the same three lines on six
  architecture families and one non-PyTorch backend.
- [Capture speed, by kind of use](benchmarks/capture.md): what each way of using TorchLens costs
  against a plain forward, measured, including where it is slow.

## Doors

Four doors, one per community, in no order of rank:

| Door | Start with |
| --- | --- |
| [Mechanistic interpretability](doors/mechinterp.md) | [Intervene](field-guide/intervene.md) on the decoder |
| [NeuroAI](doors/neuroai.md) | [Extract features](field-guide/extract_features.md) on ResNet-18 |
| [Vision and dynamic models](doors/vision.md) | [See and inspect](field-guide/see_and_inspect.md) on ResNet-18 |
| [Agent builders](doors/agents.md) | [TorchLens for AI coding agents](for-ai-agents.md) |

## Relation to other packages

- [Which tool for this job?](migration/comparison.md): a dated table of what TorchLens and seven
  other tools do, including the jobs another tool does and TorchLens does not.
- [Coming from another tool](migration/index.md): one page per tool, in one shared shape.
- [Using TorchLens with other tools](migration/interop.md), by direction.
- [Where to go instead](migration/where_to_go_instead.md).
- [Comparable tools, checked November 2026](migration/landscape.md).
