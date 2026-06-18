# Menagerie Methodology

This catalog was built as a broad cross-lab survey rather than a single-framework scrape.
The goal was coverage of model families across domains while preserving enough constructor
metadata to render random-init TorchLens graphs when public PyTorch code supports it.

## Survey Rounds

The working survey used repeated rounds with a scoreboard:

| Round | Focus | Outcome |
| --- | --- | --- |
| R1 | Seed public PyTorch families from TorchVision, timm, Transformers, Diffusers | Established the core vision, NLP, multimodal, and generative backbone. |
| R2 | OpenMMLab and detection/segmentation/pose/video configs | Added large CV long-tail coverage and exposed config-sketch limitations. |
| R3 | Audio, speech, music, and vocoder ecosystems | Added ASR/TTS/music families with mixed random-init support. |
| R4 | Graph, geometric, and point-cloud libraries | Added GNN and 3D geometry families. |
| R5 | RL, robotics, control, and embodied models | Added policy/model-based families, many as public-code sketches. |
| R6 | Scientific ML: molecular, protein, genomics, weather, PDEs | Added domain-specific architectures and noted many repo-specific constructors. |
| R7 | Recommender, tabular, time-series, and anomaly models | Filled non-vision production modeling families. |
| R8 | Generative long tail: GANs, flows, restoration, compression, NeRF | Expanded families with varied input conventions. |
| R9 | Integrity sweep: dedup, family normalization, grep verification | Corrected taxonomy drift and removed variant inflation. |

## Family Normalization

Rows start from source-provided names, zoos, constructors, domains, eras, and notes. The
canonical builder then:

1. Collapses micro-domains into stable macro-domains.
2. Normalizes family aliases such as ResNet/ResNeXt, ViT/DeiT, U-Net variants, DETR
   variants, and common LLM/audio/generative families.
3. Removes scale/config suffixes where they are ordinary variants.
4. Keeps distinct rows when the same model name has different zoos, constructors, input
   recipes, or dtype requirements.
5. Marks rows as verified when notes or known zoos indicate a random-init construction
   path, while preserving caveats in `notes`.

The normalization code is intentionally conservative. It does not erase differences in
source zoo or constructor recipe just to force a lower row count.

## Integrity Bug Lesson

The most important lesson from the survey was that a large taxonomy can look complete
while still containing integrity bugs:

- Family counts can be inflated by backbone names, resolution suffixes, and scale names.
- A grep hit can be a false positive when the same acronym means different things.
- A verified source roster is not the same as a renderable random-init constructor.
- Config-file recipes often prove that a family exists but do not make it runnable in a
  clean environment.

The corrected workflow therefore separates four questions:

1. Is the family genuinely absent?
2. Is the catalog row distinct from an existing variant?
3. Is public PyTorch code available?
4. Is the row renderable from random init without gated weights or private assets?

## Current Numbers

The corrected public source TSV has 10,216 source rows. In this working tree, the
canonical SQLite database currently preserves 10,216 rows because the deduplication key
keeps distinct zoo/constructor/input recipes. The catalog has roughly 3,200 normalized
families depending on the normalization pass.

The verified flag is not a guarantee that a row will render on a given machine. It means
the source metadata suggests an instantiable recipe. The renderer records actual outcomes
in `manifest.tsv` as `rendered`, `skipped:<reason>`, or `failed:<reason>`.

## Caveats

- Some rows are web-only or config sketches. They document family coverage but should be
  skipped honestly by the renderer until a clean constructor is added.
- JAX-native, TensorFlow-native, or private/gated-weight entries are catalogable but not
  renderable by the current PyTorch gallery pipeline.
- Some public libraries require incompatible dependency stacks. The renderer groups
  dependency installs to amortize work, but separate conda environments may still be the
  right operational choice for full sweeps.
- Input recipes are intentionally small random-init probes, not benchmark-accurate
  training or evaluation setups.

## Search Log (date frontier — read before any future sweep)
- 2026-06-18: EXHAUSTIVE multi-round cross-lab discovery completed. Reached a CLEAN literal-zero at R12
  (both Opus + Codex returned 0 genuinely-new families). Catalog is complete w.r.t. all established AND
  bleeding-edge published architecture FAMILIES **through 2026-06-18** (highest arXiv id folded: 2606.18923
  GrapNet; 2512.24695 HOPE/Nested-Learning). Round scoreboard: R7 +25 / R8 +18 / R9 +1 / R10 +1 / R11
  contested / R12 +0 clean.
- FUTURE delta-sweeps: focus ONLY on arXiv submissions dated 2026-06-18 onward. The back-catalog is
  exhausted; new families now arrive at field-publication rate. Use menagerie/DISCOVER_MODELS.md.
