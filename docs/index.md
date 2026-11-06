---
hide:
  - navigation
---

# TorchLens

## The story

## The core calls

| Task | Call |
|---|---|
| See and inspect a model | `trace = tl.trace(model, x)`, then index the trace and call `trace.draw()` ([guide](field-guide/see_and_inspect.md)) |
| Extract features | `tl.extract(model, x, ["layer4"])` ([guide](field-guide/extract_features.md)) |
| Intervene | `trace.do(site, edit)` ([guide](field-guide/intervene.md)) |

## What it shows

## What do you want to do?

## Doors

- [Mechanistic interpretability](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/doors/mechinterp.md)
- [NeuroAI](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/doors/neuroai.md)
- [Vision and dynamic models](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/doors/vision.md)
- [Agent builders](https://github.com/johnmarktaylor91/torchlens/blob/main/docs/doors/agents.md)
