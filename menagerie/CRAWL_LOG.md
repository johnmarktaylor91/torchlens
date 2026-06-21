# Menagerie Crawl Log

This is the durable record of **when the menagerie roster was last exhaustively crawled** for new
architecture families. Discovery agents should read it (and its machine-readable companion
`data/crawl_history.json`) at the start of a sweep so they can focus on what is genuinely new.

> **Last exhaustive crawl: 2026-06-18.**
> A new sweep should PRIORITIZE architecture families first published **after 2026-06-18** (the recent
> big-conference / journal / arXiv frontier) — while remaining free to surface anything missed in ANY
> earlier era, language, or field. The machine-readable source of truth is
> `menagerie/data/crawl_history.json` -> `last_exhaustive_crawl`.

## How to use this when running a sweep
1. Read `last_exhaustive_crawl` from `data/crawl_history.json`.
2. Seed candidates from everything published since then: `python -m menagerie.discover_crawler`
   (it defaults its date window to the last-crawl date).
3. Run the adversarial sweep prompt in `DISCOVER_MODELS.md` (recency-first, but every-axis and
   non-English).
4. Fold finds into the catalog / `classics/` per `DISCOVER_MODELS.md`.
5. **Append a new entry to `data/crawl_history.json` and bump `last_exhaustive_crawl`** to the sweep date,
   and add a one-line row below.

## History
| Date | Type | Result (models / families / classics / verified) |
| --- | --- | --- |
| 2026-06-18 | Exhaustive multi-axis sweep + cross-lab adversarial red-team (non-English included) | 10,742 / 3,746 / 312 / 6,127 |
| 2026-06-21 | **Framework-provenance + broad all-domain conference-proceedings completeness sweep** (part of the 100%-render sprint; IN PROGRESS) | framework-era: +28 new classics, ~72 already-covered; 6 broad domain proceedings sweeps running |


## 2026-06-21 — Framework-provenance + proceedings completeness sweep (part of the 100%-render sprint)
**Why:** to defensibly claim "every notable neural network ever," we explicitly swept two axes the
earlier era/domain crawls did not single out: (a) **architectures born in pre/non-PyTorch frameworks**
that may never have gotten a clean PyTorch port, and (b) a **flavorless, maximally-broad conference-
proceedings sweep** by domain. All sweeps were web-enabled (arxiv/github/paperswithcode/proceedings,
verified live), grep-checked each candidate against the existing catalog first, and reimplemented only
the *notable + genuinely-missing* ones as `classics/` (faithful random-init torch).

**12 parallel codex sweeps run (so future work need not re-tread these exact dimensions):**
- **Framework provenance (6):** Caffe (BVLC Model Zoo) · Torch7/Lua + cuda-convnet · Theano/Lasagne/Blocks/
  Pylearn2/early-Keras/DyNet · MatConvNet/Chainer/CNTK/MXNet-Gluon/Neon/darknet/DL4J/PaddlePaddle-v1 ·
  TensorFlow (TF-Slim/TF-Hub/DeepMind-Sonnet/Magenta) · general-papers (seed: PSGNet).
- **Broad all-domain proceedings (6, no topical bias):** vision · NLP/language · generative · speech/audio ·
  RL+control+world-models+graph/geometric · multimodal+science-ML+tabular/time-series+efficiency.
  Venues: NeurIPS/NIPS, ICML, ICLR, CVPR, ICCV, ECCV, BMVC, ACL, EMNLP, NAACL, AAAI, IJCAI, SIGGRAPH,
  Interspeech, ICASSP, KDD; years ~2012–2024; any original framework.

**Result (framework-provenance sweep, complete):** **28 new** notable architectures found (the bulk —
~72 candidates — were ALREADY in the catalog, confirming good prior coverage). New finds include:
Neural Turing Machine, Generative Query Network (GQN), DeepMind Graph Nets, Stacked Hourglass, Slot
Attention, IODINE, PSGNet, MTCNN, R-FCN, SqueezeDet, DenseBox, HED, ENet, DeViSE, Hypercolumns,
Spatial Transformer (Lasagne), Show-Attend-Tell, RNNG, BiLSTM-CNN-CRF, SoundNet, Grid-LSTM, char-rnn,
waifu2x, and the Magenta suite (SketchRNN, MusicVAE, NSynth, Coconet).
**Broad domain proceedings sweeps: IN PROGRESS** — totals + new finds to be appended at sprint end.

**For future sweeps:** these framework zoos + venue/domain axes are now logged in
`HARVEST_SOURCES.md`. Don't blindly re-sweep them expecting big gains (prior coverage is strong); DO
re-run for anything published AFTER 2026-06-21, and with a more capable model (a smarter auditor finds
more — esp. on the broad-proceedings long tail).
