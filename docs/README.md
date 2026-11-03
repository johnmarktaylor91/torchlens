# TorchLens documentation

<!-- JMT writes this: the pitch and story for the docs landing. -->

## Start

- [The two running examples](field-guide/running_examples.md): a small decoder on the IOI
  prompt and ResNet-18 on one image, each a few lines ending in `trace = tl.trace(model, x)`.
- Core tasks: [see and inspect](field-guide/see_and_inspect.md),
  [extract features](field-guide/extract_features.md),
  [intervene](field-guide/intervene.md).
- [Quickstart: the input ladder](quickstart.md).

## What it shows

- [The basis grid](gallery/basis_grid.md): the six Reproduction Gallery entries against the six
  verbs they use.
- [The any-architecture strip](gallery/any_architecture.md): the same three lines on six
  architecture families and one preview backend.
- [Capture benchmark](benchmarks/capture.md): what capture costs against a plain forward hook,
  measured.

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
