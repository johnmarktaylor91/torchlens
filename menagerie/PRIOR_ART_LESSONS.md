# Prior-Art Lessons for the Menagerie

What the existing neural-architecture datasets teach us about (1) finding models, (2) designing
the dataset, and (3) why we miss things. Distilled from reading the methods and design of the
closest prior work: **Younger** (arXiv 2406.15132, NeurIPS 2024 D&B), **Neural Architecture
Retrieval / NAR** (arXiv 2307.07919, ICLR 2024), **ONNX Model Zoo** (github.com/onnx/models),
**GitGraph** (arXiv 1801.05159), **DeepNets-1M** (arXiv 2110.13100), and **NAS-Bench-101/201/301**
(arXiv 1902.09635 / 2001.00326 / 2008.09777).

> **Framing (read first).** None of these match our full combination — real named archs x all
> eras x EAGER-FAITHFUL executed graph x rich per-op metadata x one schema. They are SOURCES to
> harvest and SUBSUME, not competitors. The menagerie's wedge is: we record what *actually ran*
> op-by-op (eager-faithful), we own the pre-2014 classics axis no hub-sourced dataset can have,
> and we keep rich per-op metadata. "Subsume" means **strict superset**: for every architecture
> these datasets contain we can render the faithful executed graph + per-op metadata they lack.

---

## 1. Curation methods to adopt

Each lesson is mapped to a concrete change in our pipeline (`DISCOVER_MODELS.md`,
`catalog.py`, `discover_crawler.py`, `master_catalog.tsv`).

### 1.1 Add the prior-art datasets' own model lists as explicit harvest sources
Younger draws from **Hugging Face Hub + ONNX Model Zoo + PyTorch Hub + Kaggle Models**; NAR from
**HF + PyTorch Hub**. We already sweep arXiv/venues but under-use the model HUBS as roster sources.
- **Change:** `DISCOVER_MODELS.md` now lists Kaggle Models, ONNX Model Zoo, and the Younger/NAR
  unique-architecture lists themselves as first-class harvest sources to diff against our catalog.
- **Why it matters:** Younger collapsed 743.5K hub models to **7,629 unique architectures** ("<1%
  are unique"). Their *deduped unique list* is a pre-chewed coverage checklist — diffing it against
  `catalog_canonical.tsv` is the single highest-yield gap-finder for the modern era.

### 1.2 "Diff against an existing arch-dataset list" as a standing method
Treat each released dataset as a checklist, not just a citation.
- **Change:** new method in `DISCOVER_MODELS.md` — pull Younger's filtered list, NAR's 12,517-arch
  crawl, the ONNX Zoo taxonomy, DeepNets-1M's named eval split (ResNet-50, ViT, ConvNeXt), and the
  NAS-Bench op vocabularies; `grep -i` each name+aliases against col2 of our catalog; report misses.
- **Discipline:** harvest **names/recipes (facts)** and re-render OUR OWN faithful graphs. Do NOT
  redistribute their data. Younger's dataset is CC BY-NC-ND 4.0 — names are facts, their files aren't.

### 1.3 Canonicalize to ONE IR before dedup, and report the funnel
Younger's leverage came from forcing every framework -> **ONNX -> NetworkX DAG** (collapsing ~2000
framework ops to ~200 ONNX ops) *before* asking "is this unique?". A single normalized graph form is
the precondition for any meaningful uniqueness test.
- **Our analogue:** the **TorchLens Trace** is already our canonical IR (better than ONNX: it's the
  eager-faithful executed graph). Make the catalog dedup operate over the Trace, not over names.
- **Change:** add a **coverage funnel** metric we report and track in `crawl_history.json`:
  `N cataloged / M buildable (clean constructor) / K rendered-faithful (Trace succeeds, len>0)`.
  This makes "superset" *provable* and mirrors Younger's headline Table 3.

### 1.4 Graph-isomorphism / WL-hash dedup as the uniqueness criterion
This is the most-repeated method across the field and we should adopt it as a tool:
- **Younger:** Weisfeiler-Lehman (WL) graph hash over (operator type + operator attributes). Equal
  hash => same architecture. Hyperparameter-sensitive.
- **NAS-Bench-101:** WL-style iterative hash (each node hash folds in its label + sorted in/out
  neighbor hashes, iterated to graph diameter, then MD5 the multiset) collapsed ~510M raw labeled
  DAGs to **423,624** non-isomorphic cells. Cheap O(V*E); confirm rare collisions with an exact check.
- **GitGraph:** node-equality = op-type only (ignore hyperparams) -> 6,863 graphs -> 2,033 unique.
- **Change:** add **"graph-embedding / WL-hash near-duplicate detection over the Trace"** as a
  discovery+dedup method in `DISCOVER_MODELS.md`. Two uses: (a) catch catalog rows that are the same
  *executed graph* under different names; (b) confirm a "new" family is structurally distinct, not a
  re-skinned variant. Decide the hash's iteration objects explicitly: for us, op type + key attrs
  (the same call the renderer makes), so dedup is reproducible.
- **Reconcile with our policy:** we keep distinct rows per (name, zoo, constructor, input recipe) ON
  PURPOSE (a config variant with a different graph is a different *render*). WL-hash dedup is a
  *secondary lens* for finding accidental dupes and confirming distinctness, NOT a mandate to collapse
  our intentional per-recipe rows. State this so the two don't fight (see METHODOLOGY family-norm step).

### 1.5 Learned graph-embedding retrieval -> a "find similar" feature + near-dup detection
NAR's core method: encode each computational graph (motif sampling -> macro-graph -> **GCN** encoder,
512-dim, multi-level contrastive pre-training with **model family** as the similarity label), then rank
by **cosine similarity** for top-K retrieval. Solves "ResNet-50 vs ResNet-101 differ globally but share
identical blocks."
- **Change:** noted as a **future zoo feature** ("find architectures similar to X") and a stronger
  near-duplicate detector than exact WL-hash (catches *near*-isomorphic families). Listed as a method
  in `DISCOVER_MODELS.md`; not a blocking dependency.

### 1.6 Continuous-expansion loop, not a one-shot dump
Younger ships **automated model-acquisition tools** tuned to HF's growth rate + an **online submission
platform** (ONNX -> DAG converter) for collaborative growth; keeps old versions, appends new. DeepNets-1M
ships its **generator** so any split is re-derivable. NAR is a static research dump — and that is its
named weakness.
- **Our analogue is already half-built:** `DISCOVER_MODELS.md` is explicitly the *durable, re-runnable*
  sweep; `crawl_history.json` + `CRAWL_LOG.md` track the date frontier; the crawler seeds candidates.
- **Change:** make the continuous-expansion loop an explicit first-class idea in the prompt: re-run on
  cadence + after each conference cycle + **whenever a more capable model becomes available**, always
  prioritizing arXiv dated after `last_exhaustive_crawl`, and append a `crawls[]` entry each pass.
  (We already do this — the lesson is to KEEP doing it and to frame it as the dataset's living engine.)

### 1.7 Diversity-by-construction for any SYNTHETIC / illustrative coverage
DeepNets-1M generated **1M unique graphs with no dedup pass** by sampling each axis independently and
widely over an extended 15-op vocabulary: depth (4-18 cells), width (16-128 ch), connectivity, stem
type (CIFAR vs ImageNet), normalization (BN on/off). A Hungarian-matching check just *confirmed*
uniqueness. NAS-Bench-301 diversity-samples a 10^18 space by **mixing distributions** (uniform random
for coverage PLUS trajectories from many real optimizers to densify the high-performance region).
- **Change:** if/when we add synthetic or illustrative architectures (a complement to real harvesting),
  make diversity a **design-space property sampled per axis**, not a post-hoc filter. Carve **named
  OOD stress splits** (Wide/Deep/Dense/BN-Free + real anchors) the way DeepNets-1M does — directly
  useful for testing TorchLens capture on extreme topologies. (Noted as a method; real-model harvest
  remains primary.)

### 1.8 Mine GitHub's long tail with task-bucketing + frequency thresholds
GitGraph extracted graphs from committed `.ckpt` files (not by compiling source), bucketed by task,
and used **CloseGraph frequent-subgraph mining at 30% support** to collapse the GitHub long tail to a
few dozen distinct task-relevant graphs.
- **Change:** when a sweep harvests GitHub, bucket candidates by field/task before dedup, and threshold
  by frequency — a motif appearing across many repos is a real family; a one-off fork is noise. Noted
  as a long-tail filtering method.

---

## 2. Metadata / storage / organization / release design notes

What to capture and how to ship the menagerie, informed by what the prior datasets store and release.

### 2.1 What to capture per architecture
Union of the best-of-breed schemas, mapped to what TorchLens already produces per op:

| Field | Who captures it | Our status |
|---|---|---|
| op type per node | all | Trace has it (eager-faithful, finer than ONNX) |
| op attributes / hyperparameters | Younger (WL objects) | Trace has it |
| data-flow edges | all | Trace has it |
| **per-op FLOPs / params** | NAR (per-model totals) | Trace has it **per op** -> our differentiator |
| per-op shapes / dtype | DeepNets-1M (param shapes) | Trace has it |
| depth / recurrence / provenance | none per-op | Trace has it -> our differentiator |
| **provenance: source hub + model ID** | Younger ("complete" tier) | **catalog `zoo` + `notes`; formalize** |
| task / domain label | Younger, NAR, ONNX, GitGraph | catalog `domain`/`family` |
| era / year | none (all 2014+) | catalog `era` -> our differentiator (classics axis) |
| license of source | ONNX (allowlist) | **add to `notes` where known** |
| cost/scale totals | NAR, NAS-Bench | derivable from per-op Trace metadata |

- **Lesson:** Younger deliberately **strips weight values** (privacy/security) and keeps only the
  graph + provenance. We render from **random init**, so we already avoid redistributing weights —
  same privacy posture, by construction. Keep it that way.
- **Action:** formalize **provenance** (source hub + model/repo ID) and **source license** as catalog
  fields/notes — cheap to capture at curation time, expensive to backfill (NAR/Younger lesson).

### 2.2 File format & graph representation
- Prior art: Younger = NetworkX DAGs (from ONNX); NAR = edge-index + one-hot node features
  (PyTorch-Geometric style); DeepNets-1M = HDF5 graphs + JSON meta + a shipped generator; NAS-Bench =
  TFRecord / `.pth` dict behind a query API; GitGraph = MongoDB + graph-tool; ONNX Zoo = protobuf.
- **Our canonical artifact is the TorchLens Trace** (richer than any of the above). For *release*,
  offer downstream-friendly EXPORTS: per-architecture JSON/NetworkX/PyG graph + a per-op metadata
  table, generated FROM the Trace. Keep the catalog TSV (9 cols: name, zoo, constructor_call,
  input_shape, input_dtype, family, domain, era, notes) as the human-curatable index; ship the
  rendered graphs as the dataset.

### 2.3 How to release it
- **Release vehicle: Hugging Face dataset.** ONNX Zoo *migrated off GitHub LFS to HF*
  (`huggingface.co/onnxmodelzoo`, July 2025); GitGraph's 2018 zip links are dead; NAR's Google-Drive
  release has no versioning. **Lesson: host on a durable, versioned vehicle (HF) from day one.** A flat
  GitHub-LFS or Drive dump is a maintenance trap.
- **Version tiers (Younger's three-tier "Dataset Series"):** ship (a) a **Complete** tier with full
  provenance + caveats, (b) a **Filtered/unique** tier (WL-deduped working set), (c) **task/domain
  Split** tiers for downstream consumers. Keep old versions; append new; re-cut only on significant change.
- **Re-derivable, not a frozen blob (DeepNets-1M):** ship the **catalog + the render pipeline** so the
  graphs are regenerable from constructors. The big rendered bundle is a *convenience artifact*, the
  catalog + code is the source of truth. (We already gitignore renders + the SQLite DB and regenerate —
  this is exactly the right posture; just say so in the release.)
- **Queryable-API release pattern (NAS-Bench):** the actual *product* is a stable encoding + a one-call
  lookup. Separate the **identity scheme** (name / WL-hash / Trace key) from the **metadata store** and
  expose one query method. For us: `tl`/catalog already lets you look up by name; a future HF dataset
  should expose query-by-name and query-by-graph-fingerprint.
- **Licensing & citation:** permissive code license (Apache-2.0 / MIT, like NAS-Bench/DeepNets-1M);
  for the *data*, decide deliberately (Younger used CC BY-NC-ND for the dataset, Apache for code). Mint
  a **DOI** (Zenodo or HF) so it's citable. **Cite Younger + NAR + NAS-Bench as related work** (see s4).

### 2.4 Coverage funnel as a first-class metric
Younger's N-collected -> M-convertable -> K-unique table (743.5K -> 341K -> 174K -> 7,629; "<1%
unique") is its most reusable contribution — it documents curation honestly AND surfaces a headline.
- **Action:** report and version our funnel: **N cataloged / M buildable (clean constructor) /
  K rendered-faithful (Trace succeeds)**. The renderer already records `rendered` / `skipped:<reason>`
  / `failed:<reason>` in `manifest.tsv` — aggregate those into the funnel and store it in
  `crawl_history.json`. This is how "strict superset of Younger/NAR" becomes provable rather than asserted.

---

## 3. Why we miss models (likely gap-axes) + lessons for the finding pipeline

The prior datasets reveal where a sweep that *feels* exhaustive still leaks:

1. **Under-using the model HUBS as roster sources.** We sweep arXiv/venues hard but Younger/NAR show
   the hubs (HF, Kaggle Models, PyTorch Hub, ONNX Zoo) hold thousands of *configured, runnable*
   architectures. A paper-centric sweep misses the family that only ever shipped as a hub checkpoint or
   a Kaggle model. **Fix:** harvest hubs explicitly (1.1) and diff the deduped hub lists (1.2).
2. **Alias / canonical-name blindness.** Younger stores by model ID, NAR derives family by **regex over
   operator/repo names**. Many of our "misses" are false (already present under a long-form/alternate
   name) and many real misses hide under an alias we didn't grep. **Fix:** the prompt already mandates
   grepping name+aliases+acronyms+repo-names; reinforce with the regex-to-coarse-label idea for grouping.
3. **Same executed graph under different names = silent dupes; different graph under same name = silent
   misses.** Name-based dedup is unreliable (GitGraph/Younger both moved to graph-level identity).
   **Fix:** WL-hash over the Trace (1.4) as a secondary lens catches both.
4. **The pre-2014 / non-hub long tail.** Every hub-sourced dataset is 2014+ by construction — this is
   exactly the axis we OWN (classics/), and exactly the axis a hub-diffing sweep will under-feed if we
   let hub lists drive the agenda. **Fix:** keep the adversarial era/language sweep primary; hubs are an
   *additive* source, not a replacement.
5. **Frequency/long-tail noise vs real families.** GitHub harvesting drowns in forks. **Fix:** task-
   bucket + frequency-threshold (1.8) so a one-off fork isn't logged as a family.
6. **Treating a release as one-shot.** NAR's static dump is its weakness; the field's living datasets
   (Younger, DeepNets-1M) re-derive and re-crawl. **Fix:** keep `DISCOVER_MODELS.md` a standing,
   re-runnable loop tied to the date frontier (1.6) — re-running with a stronger model is the single
   highest-leverage maintenance action.

---

## 4. General lessons from comparing to prior art

- **Cite related work; do NOT claim false novelty.** Younger (~7.6k unique, 2024) and NAR (~12.5k,
  ICLR'24) exist. Claiming "first architecture-graph dataset" would be a credibility own-goal (cf. the
  perception audit — our bottleneck is the storefront, not the engine). The **defensible** claim is a
  **NOVEL COMBINATION** (eager-faithful executed graph x all eras incl. pre-2014 classics x rich per-op
  metadata x one schema), plus the classics axis that is ours alone. Citing Younger/NAR/NAS-Bench makes
  us MORE credible, not less.
- **"Subsume" means strict superset, provably.** For every architecture in Younger/NAR/ONNX-Zoo we can
  render the eager-faithful executed graph + per-op metadata they lack -> ours is the canonical upgrade.
  Make this measurable via the coverage funnel (2.4): "we cover everything Younger/NAR/ONNX-Zoo do,
  faithfully, plus the entire pre-2014 history they can't."
- **Define "unique" precisely and make it a content property, not a vibe.** Younger = WL hash; NB-101 =
  isomorphism hash; GitGraph = op-type equality; NAR = learned-embedding similarity. Pick the criterion
  on purpose and state its objects. Our policy (distinct rows per recipe + WL-hash as a dedup/distinct
  ness lens) must be written down so it's reproducible (1.4, METHODOLOGY).
- **Capture cheap metadata at curation time.** Provenance (hub + model ID), source license, task label,
  FLOPs/params — all cheap to record when you add the row, expensive to backfill (NAR/Younger). Our
  per-op Trace already exceeds everyone on the metadata axis; the gap is *formalizing provenance +
  license* in the catalog.
- **Ship the generator + a queryable API, host durably, version from day one.** Re-derivable beats
  frozen (DeepNets-1M); queryable-by-key is the product (NAS-Bench); host on HF not GitHub-LFS/Drive
  (ONNX Zoo migration, GitGraph dead links); three version tiers + append-don't-replace (Younger).
- **Honest funnels and OOD splits earn trust.** Younger's "<1% unique" and DeepNets-1M's named stress
  splits are credibility-builders precisely because they're self-critical. Report our funnel and any
  extreme-topology coverage the same way.

---

*Sources: Younger arXiv 2406.15132 + github.com/Yangs-AI/Younger; NAR arXiv 2307.07919 +
github.com/TerryPei/NNRetrieval; ONNX Model Zoo github.com/onnx/models (+ huggingface.co/onnxmodelzoo);
GitGraph arXiv 1801.05159; DeepNets-1M arXiv 2110.13100 + facebookresearch/ppuda; NAS-Bench-101/201/301
arXiv 1902.09635 / 2001.00326 / 2008.09777. Compiled 2026-06-18 from primary-source web research.*
