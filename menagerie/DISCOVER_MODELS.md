# Public Prompt: Discover New Model Families

Use this prompt to periodically expand the TorchLens menagerie without flooding the
catalog with backbone, scale, or configuration variants.

```text
You are extending the public TorchLens model-menagerie catalog.

Goal
Find genuinely new model families that are absent from the current menagerie catalog.
Do not add ordinary variants of families already present.

Inputs
- Current catalog TSV or SQLite database.
- Current canonical family list.
- Public package registries, model zoo docs, papers, and repository model lists.

Method
1. Build the current family baseline.
   - Extract all canonical family names from the catalog.
   - Also extract raw model names and source family strings.
   - Normalize case, punctuation, hyphens, underscores, and common aliases before comparing.

2. Walk the full taxonomy, not just popular computer vision.
   Sweep these slices at minimum:
   - vision classification/backbones
   - detection/tracking
   - segmentation/matting
   - pose/keypoints
   - video/action
   - OCR/document/layout
   - multimodal/vision-language
   - NLP/LLM/text
   - audio/speech/music
   - diffusion/GAN/flow/generative
   - 3D/geometry/NeRF/point clouds
   - graph/geometric learning
   - RL/robotics/control
   - time-series/dynamics
   - recsys/tabular
   - scientific molecular/protein/genomics
   - scientific physics/weather/PDE
   - medical imaging
   - spiking/neuromorphic
   - exotic/algorithmic/other

3. Use grep-verified taxonomy walking.
   For every candidate family:
   - Search the current catalog by family alias, paper acronym, repo name, and obvious spelling variants.
   - Treat a grep hit as present unless inspection shows it is a different family.
   - Record the exact grep/search evidence for both absences and false-positive hits.

4. Fan out by lab, framework, and domain slice.
   For each domain, inspect public rosters from major model zoos and labs relevant to that slice:
   - PyTorch/TorchVision, timm, Hugging Face Transformers/Diffusers
   - OpenMMLab projects
   - Ultralytics and detection/segmentation ecosystems
   - PyG/DGL and graph libraries
   - robotics/RL benchmark repositories
   - audio/speech libraries
   - scientific ML repositories
   - current conference benchmark lists
   Expand from each confirmed absence to nearby sibling projects, then return to grep
   verification before accepting anything.

5. Enforce family-vs-variant discipline.
   Accept:
   - A new architecture family.
   - A new modeling paradigm with a distinct graph structure.
   - A widely recognized family name that is not already represented.
   Reject:
   - Backbone swaps, dataset configs, checkpoint names, input resolutions, patch sizes,
     quantized variants, distilled variants, small/base/large scale variants, and training
     recipes when the family is already represented.
   - Paper names that are only applications of an existing backbone.
   - Repo names that package existing families without a new architecture.

6. For each genuine absence, draft a row candidate.
   Include:
   - `name`
   - `zoo`
   - `constructor_call` for random-init PyTorch construction when public code supports it
   - `input_shape`
   - `input_dtype`
   - `family`
   - `domain`
   - `era`
   - `notes` with evidence, installability, and any caveat
   If only a web/config sketch exists, say that honestly in `notes`.

7. Deduplicate before proposing changes.
   - Compare against canonical family names and raw source strings again.
   - Merge aliases into one family candidate.
   - Keep variants out unless they require materially different construction and represent
     a distinct family.

Stop rule
Continue domain-by-domain sweeps until a complete sweep returns zero genuinely new
families after grep verification. At that point, stop and report:
- domains swept
- sources inspected
- candidates rejected as already present
- candidates rejected as variants
- genuinely new families proposed
- unresolved ambiguous cases

Output
Return a table of only genuine new family candidates plus the grep evidence proving
absence from the current catalog.
```

The important habit is to prove absence against the catalog before adding anything. A
large menagerie becomes useful only if family names stay normalized and variants do not
overwhelm the taxonomy.
