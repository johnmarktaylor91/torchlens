# Paper replications

Each script here states one published claim, measures it on GPT-2 small with existing
TorchLens calls, and prints the claim, the measured numbers and a verdict. The verdict
rule was fixed before the first run and is printed as `RULE:`; a replication that misses
its rule prints `VERDICT: did not replicate`, and the rule is never moved to pass.

| Script | Paper | Claim tested | Verdict rule |
|---|---|---|---|
| `induction_heads.py` | Olsson et al. 2022, "In-context Learning and Induction Heads" | the heads with the top prefix-matching scores on repeated random tokens | TorchLens's top 3 heads equal an independent Hugging Face oracle's top 3 (5.5, 6.9, 7.10), and all 144 scores are within 1e-4 of it |
| `ioi_name_movers.py` | Wang et al. 2022, "Interpretability in the Wild" (arXiv 2211.00593) | the Name Mover Heads 9.9, 9.6 and 10.0 write the IO-minus-S logit difference directly | all three are in the top 5 heads by mean direct logit contribution over 50 IOI prompts |
| `causal_tracing.py` | Meng et al. 2022, "Locating and Editing Factual Associations in GPT" (arXiv 2202.05262) | restoring MLP outputs has its largest effect at the last subject token in early-to-middle layers | the largest average indirect effect of restoring one MLP output is at the last subject token, in layers 0-7 |

Run one with `python examples/replications/<script>.py --out DIR`. GPT-2 small
(`openai-community/gpt2`, about 550 MB, public) is downloaded at a pinned revision on first
use. `--out` receives a JSON summary and a portable `.tlspec` of one representative run,
loadable with `tl.load`. The bundles keep every activation: about 370 MB for the induction
run (one 101-token sequence) and about 50 MB for one IOI prompt.

The TorchLens calls used: `tl.trace` with `tl.options.CaptureOptions`,
`torchlens.mechinterp.head_scores` with `torchlens.mechinterp.FixedPeriodInduction`,
`torchlens.mechinterp.direct_logit_contributions` over `torchlens.mechinterp.full_decomposition`,
`trace.fork`, `attach_hooks` with a `tl.facet(...).in_module(...)` selector, `run`, and
`Trace.save`.

## Data

`data/` holds the vendored inputs; `make_data.py` rebuilds the two derived files from
their pinned public sources and checks each source's sha256:

- `ioi_rows.json`: the first 50 rows of MIB's IOI `train.json` (Hugging Face dataset
  `mib-bench/ioi`, MIT) whose indirect object and subject are single GPT-2 tokens.
- `known_facts.json`: the first 30 records of ROME's `known_1000.json` (MIT) that GPT-2
  small completes correctly with its greedy next token. The causal-tracing verdict uses the
  first 10; the other 20 only widen its reported grids.
- `induction_oracle.json`: per-head induction scores of GPT-2 small computed with plain
  Hugging Face transformers (eager attention, float64), and the input protocol.

## Differences from the papers

- Causal tracing: the paper measured GPT-2 XL (48 layers) over 1,000 facts and restored a
  run of 10 MLP layers; this script's verdict uses GPT-2 small (12 layers), 10 facts and one
  MLP layer per restoration. `--window 3 --facts 30` adds a reported, non-gating grid that
  restores 3 consecutive MLP layers (25% of depth, against the paper's 10 of 48, about 21%)
  over 30 facts; it never changes the verdict. The GPT-2 small picture differs from the
  paper's: the single-layer effect at the last subject token sits at layer 0 (0.0063 over
  10 facts, 0.0111 over 30) and is near zero elsewhere, and with a 3-layer window the
  largest cell is the last token at layer 9 (0.0112), not a mid-layer band at the last
  subject token. The script's docstring lists both grids' rows. The noise follows the paper: Gaussian, 3 times the standard deviation
  of the subject tokens' embeddings, 10 samples per fact.
- IOI: the paper's logit difference (3.56) is over its own 100,000-prompt sample; MIB's
  prompts use different templates, so the script prints its own mean beside the paper's
  and does not gate on it.
