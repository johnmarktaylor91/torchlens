# Menagerie Harvest Sources — the hubs/zoos/datasets we draw on

**Mandate: the TorchLens menagerie should be a SUPERSET of the union of every open model hub, library zoo,
and prior-art architecture dataset below.** Each update sweep should enumerate/diff against these and pull in
any genuinely-new architecture (see `DISCOVER_MODELS.md` for the method; dedup with the architectural hash,
see "Dedup" below). **This list is living — append new hubs/zoos as you find them.**

Discipline: harvest NAMES + recipes (facts) and re-render our OWN eager-faithful graphs; don't redistribute
others' data; cite the prior-art datasets as related work.

## A. General model hubs (highest-yield; enumerate model lists, diff, pull new architectures)
- **HuggingFace Hub** — huggingface.co/models (filter by task/library; `huggingface_hub.list_models`). The single
  largest source; the `transformers`/`diffusers`/`timm`/`sentence-transformers` model-type registries cover most.
- **PyTorch Hub** — pytorch.org/hub (`torch.hub.list(repo)` per published repo).
- **Kaggle Models** — kaggle.com/models (was a near-total re-host of HF in the 2026-06 sweep; check for exclusives).
- **ONNX Model Zoo** — github.com/onnx/models (migrated to HF in 2025-07). Classic long-tail (ESPCN, original
  YOLOv2/Darknet-19, FER+, age-GoogLeNet) we missed via library-only sourcing.
- **TensorFlow Hub / Kaggle-TF** , **OpenVINO Model Zoo** , **ModelScope** (Alibaba; strong for Chinese-lab models),
  **modelzoo.co** aggregator.

## B. Library zoos (we already harvest these via live pip constructors — keep current)
timm · torchvision (+detection/segmentation/video) · transformers (AutoModel by model_type) · diffusers ·
segmentation_models_pytorch · torch_geometric / DGL (graph) · OpenMMLab (mmdet/mmseg/mmpretrain/mmpose/mmagic/
mmaction/mmocr) · detectron2 · ultralytics · monai (medical) · open_clip · fairseq · ESPnet/SpeechBrain/NeMo (speech) ·
RecBole / DeepCTR-Torch / pytorch-tabular / pytorch-widedeep (recsys/tabular) · neuraloperator / DeepXDE / PhysicsNeMo
(operators/sci-ML) · scvi-tools / OpenFold / ESM (bio) · flash-linear-attention (modern SSM/linear-attn).
NOTE: library-only sourcing is our #1 historical blind spot — it MISSES ONNX-only / bespoke-repo / pre-2014 models.
Always pair library enumeration with hub (A) + prior-art (C) harvesting.

## C. Prior-art architecture datasets (use as pre-deduped coverage CHECKLISTS; cite as related work)
- **Younger** (arXiv 2406.15132) — ~7.6k unique from HF+ONNX+PyTorch-Hub+Kaggle. (2026-06 diff: 0 new vs ours.)
- **Neural Architecture Retrieval / NAR** (arXiv 2307.07919) — ~12.5k from HF+PyTorch-Hub. (0 new vs ours.)
- **GitGraph** (arXiv 1801.05159) — TF1-era 2014-17 GitHub roster (face nets, BN-Inception, FractalNet, GNMT...).
  Best older-era source; 24 new in 2026-06.
- **DeepNets-1M** (2110.13100) named eval split · **NAS-Bench-101/201/301**, **NATS-Bench** (synthetic cells —
  methods/diversity-sampling, not name sources).
- **ONNX-Net/ONNX-Bench**, **TpuGraphs**, **ArchBERT**, **NN-Meter** — minor/spot-check.

## D. Leaderboards / aggregators / frontier (recency)
- **Papers With Code** SOTA tables + "newly added" · **HF trending / leaderboards** · **GitHub trending** ·
  **OpenReview** (NeurIPS/ICLR/ICML/CVPR/ACL/...) · major-lab release blogs · arXiv recent (cs.LG/CV/CL/NE/AI,
  eess.*, q-bio, physics.*, ...). New architectures appear constantly — recency is the most reliable new-family source.

## Dedup (how we guarantee "no architecture repeated")
TorchLens gives every trace a **`graph_shape_hash`** (`torchlens.utils.hashing.compute_graph_shape_hash`,
exposed on `Trace.graph_shape_hash`). Verified 2026-06-18: it is **param-invariant, input-resolution-invariant,
batch-invariant, and architecture-discriminative** — i.e. two models share a hash IFF they are the same
architecture (ignoring weights/resolution/batch). It is the canonical menagerie dedup key: render a candidate,
compute its `graph_shape_hash`, and if it collides with an existing entry's hash it is the SAME architecture.
(`compute_raw_event_shape_hash` is a stricter variant if finer dedup is ever needed.) See `dedup_report.py`.

## Update-pipeline checklist (each sweep)
1. Read `data/crawl_history.json` (last-crawl date) and this file.
2. Enumerate/diff A (hubs) + B (libraries) + C (prior-art lists), prioritizing post-last-crawl additions and D (frontier).
3. Run the adversarial sweep in `DISCOVER_MODELS.md`; add genuinely-new architectures (family-not-variant) per its
   "Adding to the database" steps; render them.
4. Dedup the whole corpus by `graph_shape_hash` (`dedup_report.py`) to prove no repeats.
5. Append a `crawls[]` entry + bump `last_exhaustive_crawl`; append any newly-found hubs to this file.


## Early / non-PyTorch framework model zoos (swept 2026-06-21; diff against these in future rounds)
The menagerie should subsume models born in the pre-convergence frameworks, even those never cleanly
ported to PyTorch. Enumerate + diff each; reimplement notable-missing as `classics/`.
- **Caffe Model Zoo** — github.com/BVLC/caffe/wiki/Model-Zoo (vision: detection/segmentation/pose/face).
- **Torch7 / Lua** — the direct PyTorch ancestor; FB/DeepMind/NYU/Karpathy repos (fb.resnet.torch, dpnn,
  element-research/rnn, neural-style, fast-neural-style, char-rnn, NeuralTalk2, Stacked-Hourglass).
- **Theano ecosystem** — Lasagne / Lasagne-Recipes, Blocks+Fuel, Pylearn2, Keras-on-Theano, DyNet (NLP).
- **MatConvNet** (Oxford VGG, MATLAB); **Chainer / ChainerCV** (Preferred Networks; waifu2x, style2paints).
- **CNTK** (Microsoft Cognitive Toolkit); **MXNet / GluonCV** (Amazon/DMLC); **Neon** (Nervana/Intel).
- **darknet** (Redmon, C — YOLO family + classifiers); **Deeplearning4j** (Java); **PaddlePaddle v1** (Baidu).
- **TensorFlow** — TF-Slim model zoo, TF-Hub, **DeepMind Sonnet** (DNC, GQN, Relation/Graph Nets, MERLIN),
  **Magenta** (NSynth, SketchRNN, MusicVAE, Coconet), tensor2tensor, Google-Research models.
