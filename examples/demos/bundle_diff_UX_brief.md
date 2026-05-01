# Bundle Diff Hero Demo UX Brief

This brief is the acceptance contract for Phase 9 of the TorchLens 2.0 Feature
Overhaul. It must remain stable unless a later plan explicitly revises the hero
demo contract.

## Golden Artifact

- Demo model: `torchvision.models.resnet18(weights=None)`, using random weights only.
- Input: deterministic, via `torch.manual_seed(0); x = torch.randn(1, 3, 224, 224)`.
- Intervention: `model_log.do(tl.module("layer1.0.relu"), tl.zero_ablate())`.
- Output: SVG, approximately 1200x800 px, with an approximately 50KB target size.

## Layout

- Render as two columns: clean on the left, intervention on the right.
- Align rows vertically by node pairs from `bundle.aligned_pairs`.
- Use `bundle.delta_map` for per-node color values.
- Unmatched nodes receive a gray border on the side where their pair is missing.

## Color Scale

- Use a diverging `delta_map` palette from blue→white→red.
- The caption legend defines the scale as per-node L2 norm delta.

## Caption

Clean vs zero_ablate(layer1.0.relu) — top: clean, bottom: ablated. Color: per-node
L2 norm delta.

## Accessibility

- Include a `<title>` element for each rendered node.
- Include an `aria-label` on the figure-level SVG.

## Snapshot Test Strategy

Snapshot tests use two independent acceptance paths and pass if either path succeeds:

1. Semantic SVG normalization followed by byte comparison.
   - Strip Graphviz-generated random IDs, including `id="node\d+"`.
   - Strip timestamps and Graphviz comment IDs.
   - Normalize whitespace.
2. Image threshold comparison.
   - Rasterize the candidate SVG and reference SVG.
   - Compare pixels with similarity threshold `>= 0.95`.
