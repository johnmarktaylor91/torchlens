"""Find the GPT-2 small attention heads that carry "France" into the answer " Paris".

Patching each of the 144 heads' values from "The capital of France is" into
"The capital of Italy is" ranks the heads by how far they move the prediction from
" Rome" toward " Paris", and patching just the top two flips the model to " Paris".

Run (downloads GPT-2 small, about 550 MB, on first use)::

    python examples/quick/paris_vs_rome.py
"""

from transformers import AutoTokenizer, GPT2LMHeadModel

import torchlens as tl

model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
tok = AutoTokenizer.from_pretrained("gpt2")


def paris_vs_rome(trace):
    """How much more the model predicts " Paris" than " Rome" at the last position."""
    logits = trace["lm_head"].out[0, -1]
    return (logits[tok.encode(" Paris")] - logits[tok.encode(" Rome")]).item()


# TorchLens tokenizes the text automatically for Hugging Face models
france = tl.trace(model, "The capital of France is")
italy = tl.trace(model, "The capital of Italy is")

# the values of all 144 attention heads
heads = {
    (b, h): tl.module(f"transformer.h.{b}.attn") & tl.head(h, "v")
    for b in range(12)
    for h in range(12)
}
# patch each head from France into Italy
effects = {
    bh: paris_vs_rome(italy.fork().do(head, tl.patch_from(france))) for bh, head in heads.items()
}
print(max(effects, key=effects.get))  # (9, 8): the head that matters most

# the payoff: copy the Italy run and patch in just those two heads from France, in one line
both = italy.fork().do(heads[9, 8] | heads[10, 0], tl.patch_from(france))
# (-5.25, 2.99): the model now prefers Paris over Rome
print((paris_vs_rome(italy), paris_vs_rome(both)))
