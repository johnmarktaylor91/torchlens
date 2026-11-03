# Bundles plus a two-model harness

A comparison of a clean run with an edited run is only worth reading when the two runs line up
site by site and differ by the stated edit alone; and in a two-model pipeline, where one model
drafts text that a second model reads, an edit to the drafter can reach the critic only through
the tokens the critic is given. This entry is a method demonstration rather than the
reproduction of one paper; the closest published setup is the clean-versus-ablated comparison
common to activation-patching work. It bundles a clean and a zero-ablated run of GPT-2 small
and of ResNet-18 and checks every module's difference against two plain hook runs, then records
GPT-2 small drafting a sentence and DistilGPT2 judging it as one bundle, steers the drafter, and
checks that the recorded tokens, the `fed_by` token spans and the reported reach of the steer
all match a hand-run pipeline. The entry passes: every bundle difference matches the plain hooks
to exactly 0, the hand-run tokens agree, the spans point at the right critic tokens, and the
controls refuse as declared.

| Item | Recorded setup |
| --- | --- |
| Claim | A bundle compares two runs only when they line up and differ by the stated edit; a steer reaches a second model only through the tokens it reads. |
| Script and run line | `python examples/gallery/bundles_harness.py --out outputs/bundles_harness` |
| Pinned inputs | Drafter `openai-community/gpt2` at revision `607a30d7`, critic `distilbert/distilgpt2` at revision `2290a626`, float32, eager attention; torchvision ResNet-18 `IMAGENET1K_V1` on a seed-0 random input; prompt "The Eiffel Tower is in the city of"; greedy decoding, 12 draft tokens, 8 critique tokens; ablations at `transformer.h.6.mlp` and `layer1.0.relu`; steer: a seed-0 unit direction added at `transformer.h.6.mlp`, strength 120 (weak control 0.001). |
| Verdict | Passed; Linux x86_64 CPU, 3 November 2026, TorchLens 2.36.3, commit `df03308e`. |
| Cost | 141 s measured analysis; 2 min 29 s process wall time on one CPU thread, 2.7 GiB peak memory. |

## Oracles

The rules are the ones registered for this entry before any run.

| Oracle | Rule | Measured | Outcome |
| --- | --- | --- | --- |
| Bundle difference, GPT-2 | At every module call recorded by both the bundle and two plain hook runs with one tensor output, (ablated - clean) from the bundle equals (ablated - clean) from the hooks: max abs difference 0 in float32 | 137 module calls compared, 59 changed by the ablation, max abs difference 0.0 | Pass |
| Bundle difference, ResNet-18 | Same rule | 75 module calls compared, 69 changed, max abs difference 0.0 | Pass |
| Bundle compare | `Bundle.compare('clean', 'ablated', metric='relative_l2')` at the `lm_head` output equals the relative L2 distance of the plain hook runs' logits within 1e-6 (relative) | 0.04855974763631821 from both | Pass |
| Composition | Running the drafter, decoding, building the critic prompt and running the critic by hand (plain greedy loops, no TorchLens) gives exactly the tokens the bundle recorded: draft, critic prompt and critique, unsteered and steered | all four token sequences agree in both conditions | Pass |
| `fed_by` spans | The critique `fed_by` draft row found with `token_spans='auto'` has basis `matched`, and every span's draft token ids equal the critic-prompt token ids it points at in the hand-run pipeline | basis `matched`, coverage 1.0, one span `[10, 22, 2, 14]`, ids agree | Pass |
| Steer reaches the critic | A steer that changes the draft is reported as `steered` for the draft and `absorbed` or `changed` for the critique, and the critique's prompt divergence is the first position where the two hand-run critic prompts differ | draft `steered`, critique `changed`, prompt divergence 2 from both | Pass |

The first run of this script failed the GPT-2 bundle difference with a largest difference of
57.9: on a fork, zeroing the layer-6 MLP's output also rewrote the value recorded for the MLP's
inner dropout module, whose output is the same operation. A post-capture module edit is now
recorded as its own replacement operation after the module's last operation, the shape a
capture-time edit always had, so the inner module keeps the value a plain hook leaves it. No
oracle or control moved.

## Controls

| Control | Rule | Measured | Outcome |
| --- | --- | --- | --- |
| Weak steer | A steer too small to change the drafted tokens (strength 1e-3) leaves the draft tokens equal and reports the critique `unchanged` | draft unchanged; critique `unchanged` | Pass |
| Different inputs refuse | Comparing two GPT-2 runs on prompts of different token counts refuses with `bundle_gate_input_values_differ` or `metric_shape_mismatch` | refused: `InvalidArgumentError`, code `metric_shape_mismatch` (10 against 7 tokens) | Pass |
| Ordering refuses | Comparing the critique with the draft it read refuses with `bundle_gate_ordering_topology` (members joined by `fed_by` are not comparison operands) | refused: `BundleComparisonError`, code `bundle_gate_ordering_topology` | Pass |

Two prompts of the same token count but different words are compared by design (the gate
checks shapes, not values): that comparison ran and returned 585 rows, and is reported here, not
graded.

## Artifact and replay

The whole bundle (clean, ablated, draft, critique and the steered alternatives) is saved as a
portable `bundles_harness.tlspec` (350,959,201 bytes, sha256 `5f49d4b6e407530c`...). The
script reloads it and replays the `fed_by` spans, the prompt divergence and the `lm_head`
comparison from the reloaded copy: all three match the in-memory values.

## Basis-grid row

| Verbs | TorchLens lines | Plain-PyTorch lines | Shared | Architectures |
| --- | --- | --- | --- | --- |
| capture, select, read, do, compare, save | 44 | 41 | 19 | GPT-2 small, ResNet-18, DistilGPT2 (critic) |

Counts are nonblank, noncomment, non-docstring lines of the script's marked regions. The
TorchLens region builds the bundles, the ablation, the harness and the steered alternative;
the plain region is the hook-based differences and the hand-run two-model pipeline. No count
against an original reference implementation was measured.

## The stamp

```text
{
  "commit": "df03308eeb64948b6820a8ad92a675b29eebe7a2",
  "date_utc": "2026-11-03T19:21:38Z",
  "device": "cpu",
  "libraries": {
    "numpy": "2.2.6",
    "torchvision": "0.28.0+cpu",
    "transformers": "5.18.0"
  },
  "machine": "x86_64",
  "platform": "Linux-5.15.0-139-generic-x86_64-with-glibc2.31",
  "python": "3.10.19",
  "torch": "2.13.0+cpu",
  "torch_threads": 1,
  "torchlens": "2.36.3",
  "wall_seconds": 141.0
}
```

## Vary it

Use `--steer-layer 4` to steer a different GPT-2 block, `--critic gpt2` to make a second GPT-2
the critic, or `--draft-tokens 20` for a longer draft. These small examples use a randomly
initialized module and need no downloads. The first builds a clean and an ablated run of a
nested module, bundles them, and checks the difference at every module against plain hooks,
including the inner module that shares the ablated module's last operation.

```python
# recipe-test: tool=torchlens expected=True
import torch
from torch import nn
import torchlens as tl


class Nested(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Linear(8, 16)
        self.mlp = nn.Sequential(nn.Linear(16, 16), nn.ReLU())
        self.head = nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(torch.relu(self.inp(x))))


torch.manual_seed(0)
model = Nested().eval()
x = torch.randn(2, 8)
clean = tl.trace(model, x, save="all", capture=tl.options.CaptureOptions(replayable=True))
ablated = clean.fork("ablated")
ablated.do(tl.module("mlp"), tl.zero_ablate())
bundle = tl.Bundle({"clean": clean, "ablated": ablated}, baseline="clean")

plain: dict[str, list[torch.Tensor]] = {}
handles = [
    m.register_forward_hook(lambda mod, a, out, n=n: plain.setdefault(n, []).append(out))
    for n, m in model.named_modules()
    if n
]
with torch.no_grad():
    model(x)
    handles.append(model.mlp.register_forward_hook(lambda mod, a, out: torch.zeros_like(out)))
    model(x)
for h in handles:
    h.remove()
RESULT = all(
    torch.equal(
        bundle["ablated"].modules[n].calls[0].outs[0] - bundle["clean"].modules[n].calls[0].outs[0],
        plain[n][1] - plain[n][0],
    )
    for n in plain
)
```

```python
# recipe-test: tool=torchlens expected=True
rows = bundle.compare("clean", "ablated", metric="relative_l2")
RESULT = any(float(distance) > 0 for site, distance in rows)
```

The second example records a two-step pipeline as one bundle with a `fed_by` row whose spans are
given by hand, as the gallery script gets them from `token_spans='auto'`.

```python
# recipe-test: tool=torchlens expected=True
lm = nn.Sequential(nn.Embedding(9, 4), nn.ReLU(), nn.Linear(4, 9))
draft_ids = torch.tensor([[1, 2, 3, 5, 6, 7]])  # prompt [1, 2, 3], then the draft [5, 6, 7]
critique_ids = torch.tensor([[4, 5, 6, 7, 8]])  # the critique prompt quotes the draft
harness = tl.Bundle()
harness.add(tl.trace(lm, draft_ids), names="draft")
harness.add(tl.trace(lm, critique_ids), names="critique")
draft_output = {"stream": "input", "batch_row": 0, "completion": [3, 6]}
tl.io.attach_sidecar(harness["draft"], "torchlens.boundary_facts", draft_output)
harness.relate("fed_by", "critique", "draft", token_spans=[[3, 6, 1, 4]], label="critique")
row = harness.relations[0]
RESULT = row.params["basis"] == "verified" and harness.upstream("critique") == ("draft",)
```
