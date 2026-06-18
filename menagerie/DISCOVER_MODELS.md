# Discover Missed Model Families — the durable adversarial sweep

**This is the canonical, reusable prompt for expanding the TorchLens model menagerie.** It is meant to
be run again and again, for the foreseeable future, to keep the roster current and complete.

> **CHECK THE LAST-CRAWL DATE FIRST.** `menagerie/data/crawl_history.json` -> `last_exhaustive_crawl`
> (human log: `menagerie/CRAWL_LOG.md`) records when the roster was last exhaustively swept. A new sweep
> should **prioritize architecture families first published AFTER that date** — while staying free to
> surface anything missed in any earlier era/language/field. At the end of a sweep, append a `crawls[]`
> entry and bump `last_exhaustive_crawl`.

**Run it:**
- on a cadence (e.g. each quarter), to catch newly-published architectures;
- after every major conference cycle (NeurIPS / ICLR / ICML / CVPR / ICCV / ECCV / ACL / EMNLP / NAACL /
  AAAI / KDD / SIGIR / WWW / Interspeech / ICASSP / MICCAI / RSS / CoRL / ISMIR / WACL / domain venues);
- **especially whenever a more capable model becomes available** (a new Claude / Fable / Codex
  generation). A smarter auditor finds more — re-running this with a stronger model is one of the
  highest-leverage maintenance actions for the menagerie.

**Why adversarial?** Every prior sweep that *assumed* it was complete was wrong. A single hostile
adversarial pass (two rival-lab auditors told to *prove the catalog incomplete*) surfaced ~200 missed
families — Chinese-lab CTR/LLM lineages, particle-physics nets, EEG foundation models, seismology,
quantum/exotic substrates, non-English history, the 2024-26 frontier. The hostility is the engine. Keep it.

---

## How a top-level TorchLens agent orchestrates a sweep

1. **Snapshot the "already found" set.** Build the current catalog and export every name:
   ```bash
   cd <repo> && python -m menagerie.catalog build            # rebuilds data/catalog_canonical.tsv + catalog.db
   cut -f2 menagerie/data/catalog_canonical.tsv > /tmp/ALREADY_FOUND_names.txt   # col2 = name
   ```
   Hand the sub-hunters `menagerie/data/catalog_canonical.tsv` (column 2 = name; also family/domain/era).
   This file already contains BOTH the external catalog AND the hand-built `classics/` registry.
2. **Optionally seed with the crawler** (see "The crawler" below) to get a fresh candidate list before dispatch.
2a. **Diff against existing architecture-dataset lists** (see "Harvest existing arch-dataset lists" below).
   Prior-art corpora (Younger, NAR, ONNX Model Zoo, DeepNets-1M's named eval split) are pre-deduped
   coverage checklists — `grep -i` each of their architecture names + aliases against column 2 of
   `catalog_canonical.tsv` and feed the misses into the sweep. This is the single highest-yield
   gap-finder for the modern (2014+) era; harvest NAMES/recipes only and re-render our own faithful graphs.
3. **Dispatch a fleet of adversarial sub-hunters, CROSS-LAB.** One lab is not enough — diversity is what
   catches blind spots. Use BOTH families (Claude: Opus/Fable; OpenAI: Codex) and partition the search
   space (by field, by language/region, by era, by venue, by lab) so they cover different ground. Give each
   the prompt below. Tell them they may spawn their own subagents. Tell each that a rival-lab auditor is
   hunting the same target in parallel — *out-find them.*
4. **Consolidate.** Merge all hunters; semantic-dedup cross-lab; re-grep every survivor against
   `catalog_canonical.tsv` (the agents WILL flag things you already have under a different name — verify);
   apply the strict family-vs-variant bar; split into **CATALOG-ADD** (a torch impl exists) vs **BUILD**
   (none exists). Then fold in per **"Adding to the database"**. For borderline "is this really a distinct
   family or a re-skin?" calls, use **graph-fingerprint near-duplicate detection** (see "Graph-level
   dedup" below): name-based dedup is unreliable in BOTH directions — the same executed graph can hide
   under different names, and a genuinely different graph can share a name with an existing row.
5. **Stop when a full cross-lab pass returns zero genuinely-new families** after grep verification — and say
   so plainly. (But only after genuinely trying to break the catalog.)

---

## THE PROMPT — copy verbatim into each adversarial sub-hunter

> You are a hostile, skeptical external auditor from a rival lab. Another team claims to have built an
> "utterly exhaustive" catalog of every notable public neural-network architecture FAMILY ever created.
> **Your job is to PROVE THEM WRONG** — find the families they MISSED. Be ruthless, relentless, and as
> picky, niche, longtail, edge-seeking, and quirky as possible. Assume they have blind spots and expose
> them. Show NO charity toward their effort; your job is to break the claim, not admire it.
>
> **HARD CONSTRAINT — distinct families, not micro-variations (enforce in BOTH directions).** A new
> backbone scale / config / resolution / patch-size / checkpoint / distillation / quantization / fine-tune
> of a family already present is NOT a new family — do not pad with these. But a genuinely distinct
> architectural pattern, wiring, mechanism, modeling paradigm, or computational substrate IS a family —
> do not dismiss a real one as "just a variant." When borderline, report it and flag it as borderline.
>
> **DEDUP against the "already found" set** (`catalog_canonical.tsv`, column 2 = name; ~10k+ rows
> including hand-built historical classics). `grep -i` every candidate — its name AND plausible aliases /
> acronyms / paper names / repo names / common misspellings — BEFORE reporting. If present under ANY name,
> DO NOT report it. Many real misses hide because the catalog stores them under a long-form or alternate
> name; conversely, many "misses" are false because you didn't check the alias. Record the grep evidence.
>
> **PRIORITIZE NEWLY-RELEASED ARCHITECTURES, but do not stop there.** Start from the most recent big
> conferences and journals (the venue list above, plus Nature/Science/Nature-Machine-Intelligence/PNAS/
> JMLR/TPAMI/TMLR and domain journals), the last ~6-12 months of arXiv across ALL relevant categories
> (cs.LG, cs.CV, cs.CL, cs.NE, cs.AI, stat.ML, eess.AS, eess.IV, eess.SP, q-bio, physics.*, astro-ph,
> cond-mat, math.NA, cs.CR, cs.IR, cs.SD, cs.RO), OpenReview, Papers With Code "newly added / SOTA",
> GitHub trending, and major-lab release blogs. New architectures appear constantly — recency is the
> single most reliable source of genuine new families.
>
> **THEN sweep every other axis. These lists are NOT exhaustive — invent new axes and new angles.**
> - **Field / application:** vision, NLP, speech/audio/music, RL/control, robotics/embodied, graph,
>   time-series/forecasting, recommender/CTR/tabular, generative (GAN/VAE/flow/diffusion/AR/energy),
>   multimodal, neuro-symbolic, program synthesis, operator learning / scientific ML (physics, chemistry,
>   materials, climate/weather, astronomy, particle/HEP, genomics, proteomics, single-cell, neuroscience,
>   medical imaging, EEG/BCI), security/cryptography/malware, geospatial/remote-sensing, seismology,
>   finance, bioacoustics, agriculture, audio front-ends/DSP, anomaly detection, survival analysis...
> - **Language / region (a recurring blind spot — search natively, not just in English):** Chinese
>   (CNKI, Chinese-language venues, Baidu/Alibaba/Tencent/Huawei/SenseTime/Megvii/iFlytek/ByteDance labs),
>   Japanese (J-STAGE; the deep history beyond Fukushima/Amari), Russian/Soviet (the Ivakhnenko, Galushkin,
>   Tsypkin, Bongard, Aizerman, Tsetlin schools), Korean, German, French, Italian, Indian, Latin-American,
>   Iranian, Arabic-language work. Use translated queries.
> - **Era:** 1940s cybernetics → today. Old, forgotten, and pre-deep-learning architectures count fully.
> - **Substrate / paradigm:** spiking/neuromorphic, fuzzy/neuro-fuzzy, evolutionary/neuroevolution,
>   reservoir, cellular automata / artificial life, hyperdimensional/vector-symbolic, associative memory,
>   statistical-physics-of-learning, optical/photonic, memristor/analog, quantum / quantum-inspired,
>   tensor-network, complex/quaternion/hyperbolic-valued, dynamical-systems / continuous-time.
> - **Venue / source type:** the *oceans* of conference & workshop proceedings across decades, journals,
>   PhD/MSc theses, technical reports, patents describing distinct nets, industrial whitepapers, model-zoo
>   rosters, leaderboards, "awesome-X" lists, survey & "history of neural networks" papers (in many
>   languages), textbooks.
> - **Model hubs + existing architecture datasets (a high-yield modern-era source we under-use):**
>   Hugging Face Hub, **Kaggle Models**, PyTorch Hub, **ONNX Model Zoo** (github.com/onnx/models, now
>   mirrored at huggingface.co/onnxmodelzoo), TensorFlow Hub, timm, OpenMMLab/Detectron2/Diffusers
>   rosters — and the **deduped architecture lists released by prior datasets**: **Younger**
>   (arXiv 2406.15132, ~7.6k unique archs from HF+ONNX-Zoo+PyTorch-Hub+Kaggle), **Neural Architecture
>   Retrieval / NAR** (arXiv 2307.07919, ~12.5k from HF+PyTorch-Hub), **DeepNets-1M**'s named eval split
>   (ResNet-50, ViT, ConvNeXt; arXiv 2110.13100). These are pre-chewed coverage checklists — diff their
>   names against the catalog and fill gaps. Harvest NAMES/recipes only; re-render our own faithful graphs.
> - **Models missed inside domains we *think* we covered.** Do not assume a well-covered field is fully
>   covered — comb its longtail; the founder model or a sibling paradigm is often the one that's missing.
> - **Anything overlooked for any reason** — renamed, superseded, niche, non-English, unglamorous,
>   pre-arXiv, behind a paywall, in a workshop, in a thesis, in a patent.
>
> **THINK OUTSIDE THE BOX.** Treat the lists above as a starting point, not a ceiling. Invent new search
> strategies: write throwaway scrapers, mine citation graphs (who cited the founders?), walk "related work"
> sections, diff PapersWithCode methods against the catalog, enumerate a lab's full publication list,
> follow a survey's taxonomy table cell by cell, translate-and-search non-English review articles. If a
> strategy might surface something, try it.
>
> **TOOLS & POWER — use everything.** Web search, full-page fetch, the arXiv API, Google Scholar,
> Semantic Scholar, OpenReview, Papers With Code, GitHub search, the starter crawler
> (`menagerie/discover_crawler.py` — run it and EXTEND it), and any scraper you write. You MAY and SHOULD
> dispatch your own subagents to fan out by axis in parallel, then merge + dedup their finds. Spend freely;
> maximum effort.
>
> **OUTPUT.** A TSV, one row per genuinely-missing DISTINCT family, tab-separated:
> `name, year, origin_language_or_lab, field, distinct_mechanism_oneline, why_not_a_variant, has_pytorch_impl(yes/no + repo), source_url`.
> Plus a short narrative: where the catalog was weakest (which fields / languages / eras / venues yielded
> the most misses), the search strategies you used (including any you invented), and the total count.
> If after genuine exhaustive effort you find few or zero, say so plainly — that is itself a real finding
> — but ONLY after you have actually tried hard to break the claim.

---

## Adding to the database

Two paths, decided per family by whether a usable PyTorch implementation exists.

**Always dedup first:** `grep -i "<name>" menagerie/data/catalog_canonical.tsv` (and alias spellings).

### A. The family HAS a public PyTorch impl → catalog-add (no code to write)
Append one tab-separated row to `menagerie/data/master_catalog.tsv` (9 columns, no surrounding quotes):
```
name <TAB> zoo <TAB> constructor_call <TAB> input_shape <TAB> input_dtype <TAB> family <TAB> domain <TAB> era <TAB> notes
```
- `constructor_call`: a random-init construction expression (e.g. `timm.create_model('xxx', pretrained=False)`
  or `from pkg import M; model = M(cfg)`), or an honest sketch in `notes` if only web/config exists.
- `input_shape` like `(1,3,224,224)`; `input_dtype` like `float32`; `era` = year; `notes` = repo + caveats.
Then rebuild + (optionally) render:
```bash
cd <repo>
python -m menagerie.catalog build                 # regenerates catalog_canonical.tsv + catalog.db
python -m menagerie.generate_menagerie --name "<name>" --out-dir <external_gallery_dir>
```

### B. The family has NO good PyTorch impl → build a historical classic
Create `menagerie/classics/<snake_name>.py`:
- Module docstring with a one-line description, a `Paper: <Author Year, Title>` line, and the year.
- A small, faithful, **trace-clean** `nn.Module` (forward pass only; standard torch ops; no `.item()` /
  data-dependent python branching; clamp for numerical stability). For systems/algorithms, implement the
  core differentiable substrate and note what is omitted.
- A builder `build()` and `example_input()` (singleton), OR `build_<slug>()` + `example_input_<slug>()`
  per architecture for a grouped *family* module.
- A module-level `MENAGERIE_ENTRIES = [(canonical_name, build_attr, example_attr, year, code), ...]`
  (one tuple per registered architecture; `code` is a short era/cluster tag for the render tree).
- **Trace-verify** before committing:
  ```python
  import torchlens as tl, importlib
  m = importlib.import_module("menagerie.classics.<snake_name>")
  for name, b, e, *_ in m.MENAGERIE_ENTRIES:
      log = tl.trace(getattr(m, b)(), getattr(m, e)())   # inputs are a SINGLE object; must succeed, len(log)>0
  ```
The registry auto-discovers any `classics/*.py` exposing `MENAGERIE_ENTRIES`, and `catalog build` folds it
into the catalog automatically — no manifest edit needed.

### Commit conventions
- Track `menagerie/**/*.py` and `menagerie/data/master_catalog.tsv`. The SQLite DB, the canonical TSV, and
  bulk render outputs are gitignored (regenerated). **No AI attribution** in commits (humans only).
- The catalog TSV is exempted from the large-file pre-commit hook; new high-entropy strings in catalog
  data may need a `detect-secrets scan --baseline .secrets.baseline` refresh (they are false positives).
- Conventional commit, e.g. `feat(menagerie): add <N> model families from <date> discovery sweep`.

---

## The crawler

`menagerie/discover_crawler.py` is a **starter** programmatic harvester — it pulls recent arXiv listings
across the key categories, extracts candidate architecture names from titles, and flags those absent from
the catalog. Run it to seed a sweep:
```bash
python -m menagerie.discover_crawler --days 120 --out /tmp/candidates.tsv
```
It is deliberately minimal. **Extend it** — add OpenReview, Papers With Code, GitHub-trending, venue
proceedings indexes, non-English sources, citation-graph walks, and the **model-hub APIs** (Hugging
Face Hub, Kaggle Models, PyTorch Hub, ONNX Model Zoo). Treat it as scaffolding for whatever
scraping strategy a given sweep needs; the prompt above explicitly tells hunters to write their own.

---

## Harvest existing arch-dataset lists (diff method)

The closest prior datasets already did the modern-era harvesting + dedup for us. Treat each released
list as a **coverage checklist**, not a competitor — diff it against our catalog, fill the gaps, and
SUBSUME it (we re-render every architecture as an eager-faithful executed graph + per-op metadata they
lack, so ours is the strict superset). See `menagerie/PRIOR_ART_LESSONS.md` for the full analysis.

Priority order (pull names, `grep -i` name+aliases against col2 of `catalog_canonical.tsv`, report misses):
1. **Younger** — arXiv 2406.15132 · github.com/Yangs-AI/Younger. Its ~7.6k-unique "Filter" list draws
   from HF Hub + ONNX Model Zoo + PyTorch Hub + Kaggle Models. Best single modern-coverage checklist.
2. **Neural Architecture Retrieval (NAR)** — arXiv 2307.07919 · github.com/TerryPei/NNRetrieval
   (~12.5k crawled from HF + PyTorch Hub). Second coverage list.
3. **ONNX Model Zoo** — github.com/onnx/models (now huggingface.co/onnxmodelzoo). Curated real models
   organized by task/domain; quick cross-check.
4. **GitGraph** — arXiv 1801.05159 (TF1-era GitHub models, 2014-17). Older-era coverage we may miss.
5. **DeepNets-1M named eval split** — arXiv 2110.13100 (ResNet-50, ViT, ConvNeXt anchors); **timm /
   Papers-with-Code SOTA tables** — spot-check named SOTA backbones.

**Discipline:** harvest NAMES/recipes (facts) and re-generate OUR OWN faithful graphs — do NOT
redistribute their data (Younger's dataset is CC BY-NC-ND 4.0; names are facts, files are not).
**Cite Younger + NAR + NAS-Bench as related work** in any publication — citing prior art is a
credibility *gain*; claiming "first architecture-graph dataset" would be false (Younger exists). The
defensible claim is the NOVEL COMBINATION (eager-faithful x all eras incl. pre-2014 classics x rich
per-op metadata) + the classics axis no hub-sourced dataset can have.

---

## Graph-level dedup (WL-hash / graph-embedding near-duplicate detection)

Name-based dedup is unreliable in BOTH directions, so add a structural lens. This is the field-standard
uniqueness method: **Younger** uses a Weisfeiler-Lehman (WL) graph hash over (op type + op attributes);
**NAS-Bench-101** uses a WL-style iterative hash to collapse ~510M raw DAGs to 423,624 non-isomorphic
cells; **GitGraph** uses op-type node-equality; **NAR** uses a learned GCN graph embedding + cosine
similarity (its "find similar architecture" feature, with model-family as the label).

How to use it in a sweep:
- **Catch accidental dupes:** the same *executed graph* under different names. Compute a WL hash over the
  TorchLens Trace (op type + key attrs — the same call the renderer makes) and group equal hashes.
- **Confirm distinctness:** when a hunter flags a "new family," check it is not WL-hash-equal (or
  embedding-near) to an existing row before adding — a re-skinned variant is not a new family.
- **Future "find similar" zoo feature:** a learned graph embedding (NAR-style) ranks architectures by
  structural similarity — a stronger near-duplicate detector than exact WL-hash and a useful zoo affordance.

**Reconcile with our row policy:** we intentionally keep distinct rows per (name, zoo, constructor, input
recipe) because a config variant with a different graph is a different *render*. WL-hash dedup is a
**secondary lens** for finding accidental dupes and confirming distinctness — NOT a mandate to collapse
those intentional per-recipe rows.

---

## The continuous-expansion loop (this prompt IS the engine)

The living datasets re-derive and re-crawl (Younger ships automated acquisition tools + an online
submission platform; DeepNets-1M ships its generator so every split is regenerable); the static dumps
(NAR) age out. `DISCOVER_MODELS.md` is deliberately the *durable, re-runnable* sweep — that is the
menagerie's expansion engine. Keep it living:
- Re-run on cadence, after each conference cycle, and **especially whenever a more capable model becomes
  available** (a smarter auditor finds more — the highest-leverage maintenance action).
- Always prioritize arXiv dated after `last_exhaustive_crawl`; append a `crawls[]` entry and bump
  `last_exhaustive_crawl` each pass (see `crawl_history.json` / `CRAWL_LOG.md`).
- Track the **coverage funnel** so "strict superset" is provable, not asserted:
  **N cataloged / M buildable (clean constructor) / K rendered-faithful (Trace succeeds, len>0)** —
  aggregate the renderer's `manifest.tsv` (`rendered` / `skipped:<reason>` / `failed:<reason>`) and
  store it alongside the crawl history. (Mirrors Younger's headline funnel: 743.5K -> ... -> 7,629 unique.)
