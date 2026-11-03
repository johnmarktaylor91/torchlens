# IOI circuit recovery with EAP-IG

Wang et al.'s 26-head indirect-object-identification circuit is the membership oracle for
an EAP-IG ranking on GPT-2 small; this entry's measurement is pending.

| Field | Value |
| --- | --- |
| Claim | Wang et al., *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*, ICLR 2023, arXiv 2211.00593 |
| Script and run line | `python examples/gallery/ioi_circuit.py --out outputs/ioi_circuit` |
| Pinned inputs | GPT-2 small revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; MIB IOI revision `e5f3468f3af4c0883be35cd3bced8c711c95d286`; 1,000 training rows; five steps; seed 0 |
| Verdict | Not run yet; no recorded machine, date, version or commit |
| Cost | Not measured |

## Oracles

Rank each head by the sum of the absolute scores of its outgoing edges, including every
head in the comparison. The head ROC AUC against Wang membership must be at least 0.80.
The script verifies the pinned training file's SHA-256 before selecting the first 1,000
rows with equal token lengths for the indirect-object, subject and random names.
Tokenization follows the MIB convention: BOS, right padding and the last real position.
The metric is minus the IO-minus-subject logit difference.

Head recall at 26 and head-only edge recall at 182 are reported without gating. The
182-edge oracle expands the head-class connections in the published ACDC IOI oracle at
commit `bc99ace817974b5584b7ee203d596a8e2bbcd399`. Exact edge-patching rank agreement is
not measured within this entry's CPU budget. No measured result is available yet.

## Controls

Rules are fixed before running. On the first 32 selected rows, independently replacing
every name occurrence with a seeded ordinary random token removes the repeated-name IOI
structure. Its head ROC AUC must lie in [0.40, 0.60]. Reversing clean and corrupt inputs,
while retaining the metric, must give a Pearson correlation of edge scores at most -0.90
with the normal run on the same rows. A failed control fails the entry. Neither control
has been measured yet.

## Artifact and replay

The script saves `ioi_circuit.tlspec`, a representative clean trace containing the complete
aggregate edge-score table in its annotations, and `edge_scores.json`. Reloading the trace
recomputes the top-k ranking from the saved scores and checks exact head-name agreement.
This is ranking replay, using a representative capture; the intermediate integration
captures are not archived. Artifact bytes and hashes are recorded by the run.

## Basis-grid row

Core verbs: capture, read, save. The script counts its marked TorchLens region, containing
the batch scorer and artifact round trip, and its plain region, containing the membership
ROC and oracle-edge expansion. These regions perform different jobs; their lengths are
not an implementation-efficiency comparison. Reference implementation length is not
measured. Architecture: GPT-2 small.

## The stamp

```text
No completed measurement. The script prints STAMP and writes ioi_circuit.json on completion.
```

## Vary it

Use `--k` to change the reported head count, `--steps` to change integration resolution and
`--rows` to change the data prefix. The full oracle requires 1,000 rows and five steps;
`--quick` exercises two real-model rows at one step and cannot pass that full oracle.

```bash
python examples/gallery/ioi_circuit.py --out outputs/ioi_circuit --rows 1000 --steps 5 --k 26
```

The executed stand-in uses a seeded two-layer GPT-2 with no downloaded weights. It scores
every component edge and checks that each score is available.

```python
# recipe-test: tool=transformers expected=True
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens.mechinterp as mi

torch.manual_seed(0)
config = GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=32,
                   n_positions=16, use_cache=False)
config._attn_implementation = "eager"
model = GPT2LMHeadModel(config).eval()
clean = torch.tensor([[3, 7, 1, 9]])
corrupt = torch.tensor([[5, 7, 2, 9]])
table = mi.edge_scores(model, clean, corrupted_input=corrupt,
                       metric=lambda logits: logits[:, -1, 1] - logits[:, -1, 2],
                       steps=1)
scores = {row.address[0]: row.score for row in table.rows()}
RESULT = len(scores) == 57 and all(value is not None for value in scores.values())
```
