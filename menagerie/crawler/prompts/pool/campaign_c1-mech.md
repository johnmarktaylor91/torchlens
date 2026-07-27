## CAMPAIGN c1-mech -- library-zoo mechanical (author tier: sonnet)

~7,150 models that live in maintained public zoos: `timm`, `segmentation_models_pytorch`,
`transformers`, `ultralytics`, `diffusers`, `mmdetection`, `torchvision`, and a long tail
of smaller registries. These are the crawler's cheapest class **because the real code
already exists and is installed**, not because the standards are lower.

### What "mechanical" actually means here

- **R1 is the expected rung.** The architecture ships materially unmodified in a
  maintained library, so the proposal is the library's own declarative recipe: the exact
  constructor, the exact entrypoint name, and the exact keyword arguments -- with every
  pretrained/weights/checkpoint flag explicitly disabled.
- Your job is to establish, with evidence, that the installed class **is** the published
  architecture: same paper, same variant, same configuration. A same-named entrypoint that
  is actually a different variant is the characteristic failure of this class.
- Pin the library version and the exact symbol path. "timm has it" is not a source;
  `timm==<version>` plus `timm/models/<file>.py::<symbol>` is.

### Do not

- Do not write a from-scratch reimplementation of something a library already ships. If
  the real source exists, use the real source -- an "approximation" is slop and is
  forbidden outright.
- Do not silently escalate a hard model in place. This campaign's author identity is
  frozen to sonnet for its entire run; a model that genuinely needs deeper reasoning gets
  a typed `BLOCKED` result with a clear reason so it can be requeued into `c3-classics`.
  Quietly trying harder here produces a worse proposal *and* a wrong provenance record.

### Budget shape

Target roughly 5 minutes and well under the tool-call grant. If a model in this campaign
is eating the whole grant, that is itself the signal: it is not mechanical. Emit `BLOCKED`
with the reason and move on.
