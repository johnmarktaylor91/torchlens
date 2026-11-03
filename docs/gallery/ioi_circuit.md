# IOI circuit recovery with EAP-IG

Wang et al.'s 26-head indirect-object-identification circuit is the membership oracle for
an EAP-IG ranking on GPT-2 small; the fixed 1,000-row run passed with head ROC AUC 0.9263.

| Field | Value |
| --- | --- |
| Claim | Wang et al., *Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small*, ICLR 2023, arXiv 2211.00593 |
| Script and run line | `python examples/gallery/ioi_circuit.py --out outputs/ioi_circuit` |
| Pinned inputs | GPT-2 small revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`; MIB IOI revision `e5f3468f3af4c0883be35cd3bced8c711c95d286`; 1,000 training rows; five steps; seed 0 |
| Verdict | Passed; x86_64 CPU; 3 November 2026, 8:25 AM PT; TorchLens 2.36.3; commit `fd37ce87dcae9391c5dbfabafcc6dfd5ae5ab9f5` |
| Cost | 6,085.8 s script wall time; two CPU cores; 6,800,248 KiB process peak RSS |

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
not measured within this entry's CPU budget. Head ROC AUC was **0.926336**, passing the
0.80 bar. Head recall at 26 was 0.769231; edge recall at 182 was 0.395604. The pinned
training file SHA-256 was `2ee41861d725eee2b71da3846872d5e5ea29d9d8c9aa3a8954f8430584b2f243`.

## Controls

Rules are fixed before running. On the first 32 selected rows, independently replacing
every name occurrence with a seeded ordinary random token removes the repeated-name IOI
structure. Its head ROC AUC must lie in [0.40, 0.60]. Reversing clean and corrupt inputs,
while retaining the metric, must give a Pearson correlation of edge scores at most -0.90
with the normal run on the same rows. A failed control fails the entry. Both controls
passed: scrambled-name AUC was **0.403846**, passing [0.40, 0.60];
reversed-role Pearson was **-0.996254**, passing the -0.90 bar. Both used 32 rows.

## Artifact and replay

The script saves `ioi_circuit.tlspec`, a representative clean trace containing the complete
aggregate edge-score table in its annotations, and `edge_scores.json`. Reloading the trace
recomputes the top-k ranking from the saved scores and checks exact head-name agreement.
This is ranking replay, using a representative capture; the intermediate integration
captures are not archived. The portable directory totals **52,632,361 bytes**, with the
gallery helper's sorted-path and content SHA-256
`e7b5583962688481f0a99f39b3c14a9c98e6964607ba1a3c6465aa6eec7c2a2d`.
Reload reproduced all 26 ranked head names exactly. This verifies saved-score replay,
not an independent rerun of the model or all interpolation captures.

The plotting helpers write `ioi_roc.png` / `.svg` and `ioi_circuit.png` / `.svg` from
the saved scores. The graph shows the 20 strongest absolute head-to-head q/k/v edges,
with Wang's head classes colored and all other heads grey. Arrow width encodes absolute
attribution, without asserting that an edge is necessary. The ROC uses all 144 heads
against the 26-head membership set. These are single-run diagnostics, with no uncertainty
intervals; they do not replace the oracle or controls.

## Basis-grid row

Core verbs: capture, read, save. The script counts its marked TorchLens region, containing
the batch scorer and artifact round trip, and its plain region, containing the membership
ROC and oracle-edge expansion. These regions perform different jobs; their lengths are
not an implementation-efficiency comparison. The measured counts were 48 TorchLens
lines and 19 plain lines at the stamped commit; the later closure-binding lint fix adds
seven TorchLens signature lines without changing the computation. Reference implementation length is not
measured. Architecture: GPT-2 small.

## The stamp

```text
{"commit": "fd37ce87dcae9391c5dbfabafcc6dfd5ae5ab9f5", "date_utc": "2026-11-03T16:25:03Z", "device": "cpu", "libraries": {"numpy": "2.2.6", "torchvision": "0.28.0+cpu", "transformers": "5.18.0"}, "machine": "x86_64", "platform": "Linux-5.15.0-139-generic-x86_64-with-glibc2.31", "python": "3.10.19", "torch": "2.13.0+cpu", "torch_threads": 2, "torchlens": "2.36.3", "wall_seconds": 6085.8}
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
