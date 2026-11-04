# A protein language model: ESM-2 traced

TorchLens on a biology model: ESM-2, a protein language model trained to fill in masked amino
acids, traced on one protein. The smallest ESM-2 (6 layers, 8 million parameters) runs on a CPU
in a few seconds. The page shows what one trace captures, reads a per-residue score straight
from a captured site, and turns the captured attention maps into a predicted residue contact
map.

**Requirements:** `pip install torchlens transformers matplotlib`. The first run downloads the
31 MB checkpoint `facebook/esm2_t6_8M_UR50D` (MIT license) from the Hugging Face Hub, pinned to
one revision; no GPU.

## 1. Load and trace

The protein is human ubiquitin, 76 residues. The tokenizer adds a start and an end token, so
the model sees 78 positions. Eager attention makes each layer's attention probabilities an
ordinary `softmax` call TorchLens can capture.

```python
# recipe-test: tool=esm expected=('COMPLETE', (1, 78, 33))
import torch
from transformers import AutoTokenizer, EsmForMaskedLM

import torchlens as tl

MODEL_ID = "facebook/esm2_t6_8M_UR50D"
REVISION = "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
model = EsmForMaskedLM.from_pretrained(
    MODEL_ID, revision=REVISION, attn_implementation="eager", dtype=torch.float32
).eval()
tokens = tokenizer(UBIQUITIN, return_tensors="pt")

trace = tl.trace(model, (), input_kwargs=dict(tokens))
logits = trace["output_1"].out
print(f"{len(trace.layers)} layers, {sum(1 for m in trace.modules if m.num_calls)} modules called")
RESULT = (trace.outcome.status.name, tuple(logits.shape))
```

The trace holds every operation of the forward pass (391 of them) with its output, inside the
module that ran it. The output site holds the logits: one score for each of the 33 tokens in
the vocabulary at each of the 78 positions.

## 2. The attention maps are captured sites

Each of the six layers computes attention probabilities for 20 heads with one `softmax`.
Selecting by function finds all six; selecting by function inside a module finds one layer's.
The captured maps are the same tensors the model returns when asked for its attentions.

```python
# recipe-test: tool=esm expected=(6, (1, 20, 78, 78), True)
attention = trace.find_sites(tl.func("softmax"))
last = trace.find_sites(tl.in_module("esm.encoder.layer.5.attention") & tl.func("softmax"))

with torch.no_grad():
    returned = model(**tokens, output_attentions=True).attentions
maps = [site.out for site in attention]
same = all(torch.allclose(m, r) for m, r in zip(maps, returned))
RESULT = (len(maps), tuple(last.first().out.shape), same)
```

## 3. A per-residue readout

The model's log-probability of the native amino acid at each position is a per-residue score
of how expected that residue is in its context. It comes from the captured output site with no
second forward pass. Without masking, the most likely amino acid is the native one at 72 of
the 76 positions.

```python
# recipe-test: tool=esm expected=(76, 72)
residue_logits = logits[0, 1:-1]  # drop the start and end tokens
native = tokens["input_ids"][0, 1:-1]
native_logp = residue_logits.log_softmax(dim=-1).gather(1, native[:, None])[:, 0]
recovered = residue_logits.argmax(dim=-1) == native
print(f"mean native log-probability {native_logp.mean():.3f}")
RESULT = (len(native_logp), int(recovered.sum()))
```

## 4. Contacts from the captured attention

ESM-2 carries a small logistic-regression head that predicts which residue pairs touch in the
folded protein from the attention maps of every layer and head (Rao et al., 2021). Feeding it
the captured maps gives the same contact map as the model's own `predict_contacts`.

```python
# recipe-test: tool=esm expected=((76, 76), True)
with torch.no_grad():
    contacts = model.esm.contact_head(tokens["input_ids"], torch.stack(maps, dim=1))[0]
    reference = model.esm.predict_contacts(tokens["input_ids"], tokens["attention_mask"])[0]
RESULT = (tuple(contacts.shape), torch.allclose(contacts, reference, atol=1e-6))
```

## 5. Picture

The top panel is the per-residue score, with the four positions where the model prefers a
different amino acid marked in red. The bottom panel is the contact map from the captured
attention.

```python
# recipe-test: tool=esm expected=True
from pathlib import Path

import matplotlib.pyplot as plt

fig, (top, bottom) = plt.subplots(
    2, 1, figsize=(7, 8.6), gridspec_kw={"height_ratios": [1, 3.4]}, layout="constrained"
)
positions = torch.arange(1, len(UBIQUITIN) + 1)
colors = ["#B0B0B0" if hit else "#c0392b" for hit in recovered.tolist()]
top.bar(positions, native_logp.detach(), color=colors, width=0.85)
top.set_xlim(0.5, len(UBIQUITIN) + 0.5)
top.set_xlabel("residue")
top.set_ylabel("log p(native)")
top.text(0.01, 0.06, "red: another amino acid ranked first", transform=top.transAxes,
         ha="left", color="#c0392b", fontsize=9)
top.set_title("Per-residue score from the captured output site", loc="left", fontsize=11)
for side in ("top", "right"):
    top.spines[side].set_visible(False)

image = bottom.imshow(
    contacts, cmap="Greys", vmin=0, vmax=1,
    extent=(0.5, len(UBIQUITIN) + 0.5, len(UBIQUITIN) + 0.5, 0.5),
)
bottom.set_xlabel("residue")
bottom.set_ylabel("residue")
bottom.set_title("Contact probability from the captured attention maps", loc="left", fontsize=11)
fig.colorbar(image, ax=bottom, shrink=0.8, label="contact probability")
fig.savefig("esm2_ubiquitin.png", dpi=150, facecolor="white")
plt.close(fig)
RESULT = Path("esm2_ubiquitin.png").stat().st_size > 0
```

![Per-residue native log-probability for ubiquitin (four mispredicted residues in red) above
the predicted contact map](../images/esm2/esm2_ubiquitin.png)

The strongest contacts away from the diagonal pair residues 2 to 6 with 12 to 16, the hairpin
formed by ubiquitin's first two beta strands, and residue 44 with residue 70, its third and
fifth strands. The block around residues 21 to 33 is short-range contacts inside its alpha
helix. The model learned these from sequences alone; it never saw a structure.

## 6. Edit and read again

Interventions work on this trace as on any other model. Zeroing the last layer's attention
probabilities and replaying the forward pass from there lowers the number of positions where
the native amino acid ranks first from 72 to 67.

```python
# recipe-test: tool=esm expected=67
edited = trace.fork().do(last, tl.zero_ablate())
edited_logits = edited["output_1"].out[0, 1:-1]
RESULT = int((edited_logits.argmax(dim=-1) == native).sum())
```

## Where to go next

- [The NeuroAI journey](../neuroai/journey.md) and [task RNN circuits](../neuroai/task_rnn_circuits.md)
  use the same capture and edit calls on vision models and trained recurrent networks.
- The larger ESM-2 checkpoints (35M to 15B parameters) have the same module names, so the
  selectors on this page carry over unchanged.
