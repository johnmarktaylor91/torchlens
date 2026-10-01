# Acknowledgments

TorchLens borrows generously from the tools its users already know: familiar
spellings, familiar conventions, familiar workflows -- with the dumb limits
removed and the honesty machinery added. This page credits every project and
paper whose ideas, conventions, or (in the named cases) code shaped a shipped
TorchLens feature, ordered roughly by how much we took. Point-of-use
"inspired by" notes live in the borrowed features' own docs; this page
collects them. "Inspired by" appears only where provenance supports it;
everything else is related prior art, and nothing here implies endorsement.

## License hygiene

Two upstream license situations deserve a plain statement. **baukit** is
unlicensed and **zennit** is LGPL: from both, TorchLens takes ideas and
vocabulary only -- no source copied, ported, generated-from, or vendored.
Permissively licensed work is different: the one named CODE borrow (treescope,
Apache-2.0) is ported with a license notice in the source file
(`torchlens/notebook/_truncation.py`), and Apache-2.0 reuse carries NOTICE
attribution. Do not over-apply the cleanroom rule to permissively licensed
work; do not under-apply it to unlicensed or copyleft work.

## Mechanistic interpretability tooling

- **TransformerLens** (Neel Nanda, Joseph Bloom, and maintainers) -- the
  single largest influence on `torchlens.mechinterp`: the ActivationCache
  analysis-kit design (accumulated residual / decomposition / stacked heads /
  logit attributions), the cached-scale LayerNorm linearization our norm facet
  is validated against bit-exactly, `get_act_name` ergonomics and the `L5H7`
  spelling, `test_prompt`, `head_detector`'s conventions, `FactoredMatrix`
  (to be ported with credit, not reinvented), position and head addressing
  ergonomics (`'last'` as a first-class spelling), the ablation-sweep idiom,
  the "% of the clean/corrupt gap recovered" idiom, its demo notebooks as the
  de-facto acceptance gallery of mechinterp, and its published
  model-compatibility table as prior art for claimed-vs-tested lists.
- **nnsight / NDIF** -- source-local tracing feel, the field's best
  generation-time intervention ergonomics, per-invocation views, the
  scan/validate posture (our plan check persists what their scan evaporates),
  and remote large-model execution -- a real product niche TorchLens
  deliberately cedes.
- **pyvene / pyreft** -- interventions as durable, serializable, shareable
  artifacts (they got there first and their serialization story is real),
  dict-config ergonomics, ragged unit locations, and the honesty bar for
  disclosing intervention coverage per attention implementation.
- **baukit** (David Bau's lab) -- the `Trace`/`TraceDict`/`edit_output` idiom
  that made module-level editing feel easy, `runningstats`/`tally` streaming
  statistics, and the ROME/MEMIT causal-tracing lineage. Ideas only (see
  license hygiene); exercised in CI from a pinned SHA, never vendored.
- **Redwood Research** -- causal scrubbing: the agreement-conditioned
  resample, the resample-over-zero/mean doctrine, run trees, donor-group
  semantics, and the treeification framing the stochastic edit family serves.
- **ACDC** (Conmy et al.) -- the iterated ablate-and-measure loop TorchLens
  deliberately does NOT ship: we enumerate, execute, and record; the search
  policy stays with the user or agent. **EAP / EAP-IG** -- the edge-grain
  attribution requirement the edge family plumbs for.
- **circuit-tracer** (Anthropic) -- the MLP stop-gradient linearization
  discipline the reads design defaults to, the batching-is-mandatory lesson,
  and the nightly parity reference.
- **SAELens** -- SAE pedagogy and the shard-shuffle access pattern the
  extraction views layer preserves.
- **nostalgebraist** -- the logit lens. **Belrose et al.** -- the tuned lens
  (the `lens=` socket). **Elhage et al.** (transformer-circuits) -- QK/OV and
  composition framing. **Wang et al.** (IOI) -- the logit-difference
  convention and the canonical experiment the ledger gallery imitates.
  **ARENA / Callum McDougall** -- the curriculum that made this the standard
  workflow. **VISCNN** -- dataset/channel Taylor importance and the
  keep-conv-nets-first-class guard.

## Attribution

- **Captum** -- the attribution catalog and familiar APIs, matched openly with
  credit: the GradientShap estimator convention, the metric conventions, the
  occlusion averaging, the completeness discipline, the
  `LayerGradientXActivation` vocabulary, and the oracle rows our parity tests
  target.
- **zennit** -- the maintained LRP ecosystem; the TorchLens LRP recipe is
  cleanroom: ideas credited, no code read (see license hygiene).
- **transformers-interpret** -- the two-line token-attribution demand the
  token helper answers. **inseq** -- contrastive-target and generation-step
  vocabulary (reserved in the schema). **pytorch-grad-cam** -- the CAM-family
  expectations.
- Papers: SmoothGrad; VarGrad (Adebayo et al.); Integrated Gradients; SHAP /
  GradientShap; occlusion; guided backpropagation; deconvolution; Grad-CAM;
  LRP; infidelity and sensitivity (Yeh et al.); and Ancona et al. for the
  LRP-0 == input-x-gradient equivalence powering the self-oracle.

## Summaries, reprs, and narration

- **torchinfo** -- the familiar model-summary entry point whose defaults set
  expectations: `input_size=`, `depth=`, the returned statistics object,
  K/M/B/T units, the table-first first screen -- and an honest note that it
  double-counts GPT-2's weight tie, which is why familiarity is borrowed and
  the dumb limits are not.
- **lovely-tensors** (xl0) -- the whole philosophy that a tensor's default
  view should be a human summary: the inline bracketed sparkline, extremes
  bracketing the distribution, element count and memory in the line,
  `all_zeros`/`empty`, omitting the ordinary, the plain-view escape hatch,
  separated +Inf/-Inf, dot-verb ergonomics, and repr-is-a-hot-path cost
  discipline. Our stats line is their line with a model behind it.
- **treescope / penzai** (Daniel D. Johnson, Google DeepMind, Apache-2.0) --
  repr-is-the-product as a stance; external type registries,
  collapsed/expanded balanced layouts, copy-buttons that emit code, faceted
  array visualization, truncation budgets with edge items,
  diverging-around-zero defaults, NaN-as-pattern. Named CODE borrow:
  `infer_balanced_truncation` (including `doubling_bonus`) and the two-stage
  slice-then-convert truncation protocol, ported with license notice.
- **torchsnooper** (zasdfgbnm) -- the compact fixed-order per-line tensor
  summary and the crash-diagnosis workflow; its open issue asking for
  per-line descriptive statistics is the line TorchLens ships. **pysnooper**
  (cool-RR) -- the narration idiom, `normalize=` for diffable transcripts,
  and the depth/prefix/output-control vocabulary.
- **Keras** -- `model.summary()` as the first command anyone types,
  table-first restraint, the authoritative labelled footer, and
  `histogram_freq`. **torchsummary** -- the same expectation, earlier.
  **flax `tabulate`** -- totals-conserving hierarchy, promoted here from
  convention to CI property test. **CLU `parameter_overview`** --
  per-parameter row precedent. **rich** -- the sparse-table vocabulary.

## Visualization and export

- **Graphviz** -- the substrate: `dot -Tjson` (without which the label
  measurement work does not exist), midpoint dummy-node space reservation,
  named HTML-table ports. Network visualizations have been generated with
  Graphviz since TorchLens 1.0.
- **Netron** (Lutz Roeder) -- the renderer, the ONNX JSON readers,
  function-definition navigation, the `input_names` port hook, the
  serve/widget grammar, the attachment Metadata/Metrics design, the
  "one glance tells you the block structure" bar, and the interaction model
  the netron export deliberately borrows -- plus the reporters and commenters
  of netron issues #65, #71, #275, #1234, #1240, #1241, #1346, #1358, #1369,
  #1370, #1480, #1544, and #1596 as the demand record that design answers.
  TorchLens contributes observed execution facts netron cannot infer; it does
  not present netron's visual grammar as its own.
- **Google AI Edge Model Explorer** -- the hierarchy-first viewer and its
  file contracts: the npm visualizer component, `dist/worker.js` (our CI
  oracle), the node-data and edge-overlay formats, split panes, the
  sync-navigation design, the disclosed-truncation settings philosophy, and
  Apache-2.0 licensing that makes the embed story possible. Product copy says
  "Open in Google Model Explorer"; nothing implies TorchLens authored the
  viewer. Upstream issue and PR contributors whose threads shaped the plan:
  issues #82, #669, #365, #50/#561, #494; PRs #503, #537, #553/#393/#377/#684.
- **hiddenlayer** (Waleed Abdulla) -- the declarative graph-transform idea,
  the `>` pattern syntax, and the ConvBnRelu idiom list. We credit the idea,
  write our own code, add `{k}` repetition and composability-with-honesty.
- **torchview** (mert-kurttutan) -- the compact-overview-by-default
  expectation, one-call model-in/graph-out, input+output-shapes-on-edges,
  and zero-memory meta-device drawing as a familiar expectation
  (`device='meta'` is their mechanism and their trap -- TorchLens never moves
  your model). **torchviz** -- notebook-first real-forward drawing lineage.
- **TensorBoard** -- the event format, the `add_histogram_raw`
  precomputed-bucket contract (the receiving API that makes an honest
  low-cost watcher possible), the namespace-collapsing Graphs dashboard, and
  hierarchical expand/collapse as the large-graph interaction model.
- **BertViz** (Jesse Vig) -- the head-view, model-view, and query-key
  teaching tasks. **CircuitsVis** (and its TransformerLens/ARENA lineage) --
  raw-tensor notebook components, and the one supported tviz bridge.
  **Ecco** (Jay Alammar) -- colored tokens, layer predictions, rankings, and
  factor views. **inspectus / labml** -- token tables and loss/entropy
  strips.
- **Google PAIR LIT** -- the typed-Spec architecture (a model declares typed
  output fields and the UI derives which analyses are possible), the
  model-provided-salience pathway, the callable contract validator, the
  prediction cache, the counterfactual editor, and side-by-side comparison.
- **Okabe-Ito** and **ColorBrewer** -- the colorblind-safe palettes.
  **matplotlib** -- the rendering substrate and style-sheets-as-defaults
  precedence. **cmocean** (Kristen Thyng) -- the Balance diverging map
  convention lineage.
- The tree-cut / pruned-search lineage (Li & Abe 1998; Veras & Collins 2017;
  CART pruning) -- the score -> tree-cut -> budget pattern behind smart
  collapse; the novelty is the application, not the pattern.

## Training-time diagnostics and tracking

- **torcheck** (pengyan510) -- the check vocabulary and the register-once /
  named-module / enable-disable ergonomics `torchlens.checks` systematizes.
  **torchtest** -- same niche, honored the same way.
- **Andrej Karpathy**, "A Recipe for Training Neural Networks" -- the raw
  update-to-weight ratio and the sanity-check culture. **Lightning** -- the
  sanity-check-before-fit framing and ThroughputMonitor's clean-step MFU
  recipe.
- **torchexplorer** (Samuel Pfrommer, Apache-2.0) -- the vertical-slice
  distribution-over-time encoding, the instrument-once ergonomic, `sample_n`
  element subsampling, the task-first recipe docs shape, and making per-site
  distribution health a training-time question people expect answered.
- **Weights & Biases** -- attach-once `watch`,
  `wandb.Histogram(np_histogram=)`, `sync_tensorboard`, and (with **MLflow**)
  run-as-first-class-object and the run/experiment distinction behind the
  experiment ledger; ours is local-first, file-based, and intervention-native.
- **Neptune TorchWatcher** -- the activation-watch precedent and per-layer
  stat vocabulary. **ClearML** -- automatic TensorBoard capture and
  `Logger.report_histogram`'s precomputed-values contract. **Hugging Face
  Trainer** -- derived logging cadence and `WANDB_WATCH` as the
  config-not-activation pattern.
- **DDSketch** (Datadog) -- the relative-error logarithmic bin design and its
  accuracy parameterization, adopted as `bins_per_octave` on a universal grid
  while declining the collapsing store.
- **Sacred / Hydra** -- configuration is part of the record, not a comment.
  **DVC** -- experiment records as small reference-bearing artifacts beside
  the data. The folklore `plot_grad_flow` recipe -- acknowledged honestly as
  a widely-copied forum snippet rather than a paper.

## Profiling, cost, and memory

- **PyTorch profiler / Kineto / CUPTI** -- every device number TorchLens will
  ever print comes from their instrumentation; we add addressing, not
  measurement. Also: the profiler's typed record-function SCOPE field,
  forward/backward flow events (an independent native witness used as an
  oracle), `record_function` (the ergonomic `tl.region` is modeled on),
  `torch.profiler._memory_profiler`'s category taxonomy (preserved past its
  deprecation), the CUDA memory snapshot and memory_viz, and
  `torch.utils.benchmark` as the clean-timing authority we hand off to.
- **torch.utils.flop_counter.FlopCounterMode** -- the primary external FLOPs
  oracle and the formula-registry design `register_op_rule` echoes (we found
  and reported a registry gap rather than exploiting it silently). **fvcore**
  (the `unsupported_ops` disclosure shape), **ptflops**, **calflops**,
  **thop** -- conventions, the one-call ergonomic, and the shared
  unit-labeling failures that motivate our convention line.
- **PaLM** (Chowdhery et al., 2022) -- MFU and the MFU/HFU distinction: their
  name, their convention. **Williams, Waterman, Patterson** -- the roofline
  model. **Hugging Face Transformers** -- the 6ND rule. **DeepSpeed** --
  module training-cost expectations. **Horace He** -- the clean-step MFU
  recipe.
- **cProfile / perf / py-spy** -- the exclusive-vs-inclusive
  (tottime/cumtime) distinction that makes percent columns safe on a nested
  table. **NVIDIA Nsight**, **Perfetto / Chrome tracing / speedscope** -- the
  viewers we emit into. **MLCommons Chakra** -- a schema we may later emit,
  disclosed as not-yet-emitted.

## NeuroAI and representational analysis

- **thingsvision** (ViCCo-Group; Muttenthaler & Hebart) -- for making
  preprocessing correctness a first-class concern in NeuroAI feature
  extraction at all; the journey-shaped docs organization; matched
  preprocessing as a method on the thing that knows the model (the instinct
  behind the preprocessing verifier); `show_model()` as the site-discovery
  pattern; one-arg export ergonomics; the Gaussian RDM convention and
  rank-scaled RDM display; gLocal; and the pooling/flatten vocabulary.
- **rsatoolbox** -- the Dataset/RDMs descriptor model, the condensed-RDM
  convention, the comparison vocabulary, and the inference boundary TorchLens
  points at rather than reimplements.
- **Brain-Score** -- the ActivationsExtractorHelper / NeuroidAssembly
  contract the bridge serves. **Net2Brain** (Roig and Cichy labs) --
  npz-as-interchange, the RDMCreator metric set (including manhattan),
  pooling and layer presets, the tidy results contract, and fold-local SRP
  practice as prior art. **DeepJuice** (Conwell et al.) -- the memory-limited
  layer-batched extraction idea, the degrees-of-freedom-equating
  up-projection, human-unit memory budgets, and the industrial-scale framing
  behind the streaming seam. Their public tutorial lists torchlens in its
  requirements: a gift, not a rivalry.
- **CORnet** -- the recurrent real-model case. **netrep** -- the shape-metric
  home we point at. **himalaya** -- the encoding-model home we point at
  (with **sklearn, scipy, POT, nilearn, brainML Stacking** -- the tools our
  typed refusals name). **The THINGS initiative** -- the stimulus-management
  ecosystem framing.
- Papers: **Kriegeskorte, Mur & Bandettini 2008**, **Nili et al. 2014**,
  **Khaligh-Razavi & Kriegeskorte 2014** -- the RSA protocol, the
  noise-ceiling procedure we refuse and point at, the tau-a discipline, and
  the layer sweep the gallery reproduces. **Kornblith et al. 2019** -- CKA.
- **scikit-learn** -- `SparseRandomProjection`, the
  `johnson_lindenstrauss_min_dim` sizing-helper practice, `check_estimator`
  as the conformance-suite idea, and the conventions that make our numbers
  familiar (same-seed non-equivalence documented). **Achlioptas 2003** and
  **Li, Hastie & Church 2006** -- the sparse projection constructions we
  implement; **Johnson-Lindenstrauss** (with Dasgupta/Kumar/Sarlos 2010 and
  Kane/Nelson 2014 named as literature, not warrant) -- with the honest note
  that the guarantee covers Euclidean distances, and the measured residual on
  correlation-distance RDMs is the statement of what it does not cover.
- **sentence-transformers** and **Hugging Face** -- masked-mean reference
  semantics. **timm / torchvision** -- pooling precedent and named
  weight/transform presets. **COCO** -- the diverse-stimulus corpus (with the
  licensing caveat that keeps it un-vendorable).

## Persistence, formats, and ecosystem policy

- **safetensors / Hugging Face** -- keyed row slicing, tokenizer conventions,
  the precedent that a load path can be provably rather than documentedly
  safe, the `hf-internal-testing/tiny-random-*` real-class/tiny-config corpus
  pattern, the immutable-revision offline-cache workflow, explicit
  remote-code trust (tightened here to distribution-scoped activation), the
  chat-template and KV-cache conventions that make token-prefix boundary
  evidence checkable, and Accelerate's `init_empty_weights()` conventions.
- **Apache Arrow** -- the packed values/offsets/row-shapes ragged carrier and
  the format-stability norm. **HDF5 / NumPy / SciPy** -- physical format
  conventions (with zarr's consolidated-metadata idea). **zlib / CRC-32** --
  the dependency-free corruption checksum. **Merkle / BLAKE3 / Zstd** -- the
  reason cryptographic identity is affordable as a default.
- **StableHLO / OpenXLA** -- dated serialization windows, governed
  compatibility corpora, and the per-PR serialized-corpus test. **NumPy
  NEP 29 / SPEC 0** -- date-derived support policy. **ONNX** -- version
  adapters and the familiar backend-runner shape. **protobuf** -- the
  never-reuse-a-field rule. **PyPA / importlib.metadata** -- entry-point
  metadata. **pytest** -- plugin mechanics AND the autoload cautionary tale.
  **OpenTelemetry** -- per-kind groups and capability matrices. **direnv**,
  git `safe.directory`, **VS Code Workspace Trust** -- the trust-ledger
  pattern. **mypy** -- cited honestly as the counter-precedent: a project
  config file that DOES execute code, the design declined. **Khronos** --
  conformance program shape.
- **The Model Context Protocol** (Anthropic) -- tool annotations
  (`readOnlyHint`/`idempotentHint` as the machine-readable form of a safety
  claim, adopted exactly) and the resources model. **JSON Schema Draft
  2020-12** -- the manifests' practice. **ruff `--output-format=json`** and
  **pytest `--json-report`** -- the CLI shape worth copying: a stable machine
  format beside a human default, exit codes that mean something to CI.

## Testing practice

- **Hypothesis / QuickCheck** -- property-based testing culture. **mutmut /
  cosmic-ray** -- mutation testing. The **metamorphic-testing literature**.
  **NIST ACTS / IPOG** covering-array practice (Cohen et al. lineage) -- the
  pair accounting behind composition coverage. **nbclient** -- the notebook
  execution harness. **pytest-randomly** -- already in the repo's DNA.

## PyTorch itself

Beyond everything named above: `__torch_function__` (the capture substrate),
the optimizer hook API (`register_post_accumulate_grad_hook`,
`register_multi_grad_hook`), gradient clipping's
`clip_grad_norm_(error_if_nonfinite=)` (the upstream tripwire our remedy
strings point at), `GradScaler`'s public backoff semantics,
`torch.autograd.graph.saved_tensors_hooks` (the independent storage-identity
oracle: our accounting is checked against torch's, not against itself),
`torch.autograd.graph.Node.register_prehook`/`register_hook` (the public seam
that makes backward device attribution possible), `detect_anomaly` (the
authority for backward traceback diagnostics, named inside our own report),
`fork_rng`, `torch.nn.attention.sdpa_kernel`, `ModuleTracker`,
`torch.autograd._make_grads` (whose seed recipe is specified precisely enough
to reproduce bit-for-bit), `torch.load(mmap=True)`, the DataLoader/collate
seam, the lazy-module `materialize()` mechanism and machine-readable
pending-tensor allowlist, and `torch.compiler.set_stance`. Related prior art
in the extraction niche: **torch.fx**, **torchvision feature_extraction**,
**torchextractor**, **surgeon-pytorch**.

## People

The development of TorchLens benefitted greatly from discussions with
Nikolaus Kriegeskorte, George Alvarez, Alfredo Canziani, Tal Golan, and the
Visual Inference Lab at Columbia University. Thank you to Kale Kundert for
helpful discussion and code contributions enabling PyTorch Lightning
compatibility. Logo created by Nikolaus Kriegeskorte.

## Maintenance rule

Any change landing a borrowed idea adds or extends a row on this page in the
same change, plus a point-of-use "inspired by" note in the feature's own doc.
The lint in `tests/test_docs_regen_acknowledgments.py` enforces the page's
existence, the README link, and the presence of the load-bearing entries; it
is a floor, not the rule -- the rule is sporting, specific credit at the
moment of borrowing.
