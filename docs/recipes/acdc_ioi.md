# ACDC circuit discovery as a driver loop

ACDC (Conmy et al., 2023) finds a circuit by pruning edges: walk the
receivers from the output back to the input, and for each receiver try
removing each incoming edge (patching it to its value on a corrupted
input); keep the removal when the metric moves by less than a threshold
`tau`. TorchLens does not ship a circuit-search loop. It enumerates the
edges (`torchlens.mechinterp.component_graph`), executes each patch
(`torchlens.mechinterp.patch_edges`, edge substitution on the replay engine) and
records each decision (`torchlens.experiment.ledger`); the loop and every prune
decision are yours, and live on this page.

Why each edge test goes through edge substitution and not `site_sweep`:
`site_sweep` applies one edit at every candidate site, but the value an
edge patch writes depends on the edge's sender (receiver `t` reads
`input - s + s_corrupt`), and every edge into `t` writes at the same site,
`t`'s input. One edit per site cannot express that, so the loop calls
`patch_edges` for every test. `site_sweep` runs once, beside the loop, as a
head-knockout pass whose table is reported next to the circuit; it never
changes which edges are pruned.

The page runs on a two-layer GPT-2 stand-in with random weights (two heads
per layer, eight dimensions, 57 graph edges of which 46 are not bias edges)
and walks only its last two receivers, `logits` and `m1` (13 edge tests),
so it finishes in seconds on CPU. The full walk is the same loop over
`reversed(list(graph.receivers.values()))`; on GPT-2 small that is 32,491
edge tests, each a multi-receiver patch: run it as a background job.

## Setup: the stand-in, the two inputs and the metric

The metric is ACDC's default: the KL divergence from the full model's
next-token distribution to the circuit's, at the last position, averaged
over the batch. The full model has KL 0; pruning raises it.

```python
# recipe-test: tool=transformers expected=True
import torch
from transformers import GPT2Config, GPT2LMHeadModel

import torchlens as tl
import torchlens.mechinterp

config = GPT2Config(n_layer=2, n_head=2, n_embd=8, vocab_size=32, n_positions=16, use_cache=False)
config._attn_implementation = "eager"
torch.manual_seed(0)
model = GPT2LMHeadModel(config).eval().double()
with torch.no_grad():
    for param in model.parameters():
        param.mul_(3.0)  # sharper attention, so the edges differ in importance
clean_ids = torch.tensor([[3, 7, 1, 9, 4, 3], [2, 6, 1, 8, 5, 2]])
corrupt_ids = torch.tensor([[5, 7, 2, 9, 8, 5], [4, 6, 3, 8, 1, 4]])

ready = tl.options.CaptureOptions(layers_to_save="all", intervention_ready=True)
clean = tl.trace(model, clean_ids, capture=ready)
corrupt_trace = tl.trace(model, corrupt_ids, capture=ready)
graph = torchlens.mechinterp.component_graph(clean)
corrupt = torchlens.mechinterp.component_graph(corrupt_trace)
full_logp = torch.log_softmax(clean.modules["lm_head"].out[:, -1], dim=-1)


def kl_to_full(logits):
    """KL(full || circuit) at the last position, averaged over the batch."""
    logp = torch.log_softmax(logits[:, -1], dim=-1)
    return float((full_logp.exp() * (full_logp - logp)).sum(dim=-1).mean())


def circuit_kl(pruned):
    """Patch every pruned edge to its corrupted sender in one call; return the KL."""
    patched = torchlens.mechinterp.patch_edges(clean, pruned, corrupted_graph=corrupt)
    return kl_to_full(patched.modules["lm_head"].out)


edges = [e for e in graph.edges.values() if e.sender.kind != "bias"]
print(len(graph.edges), len(edges), circuit_kl([]))
RESULT = (len(graph.edges), len(edges), circuit_kl([])) == (57, 46, 0.0)
```

## The head-knockout pass (`site_sweep`, reported beside the circuit)

Arm the ledger first, so the sweep lands in its first entry as one material
step. Each candidate zero-ablates one head's value vectors on a real
forward.

```python
# recipe-test: tool=transformers expected=True
from torchlens.experiment import head_ablation_candidates, ledger, site_sweep

led = ledger(
    "acdc_ledger.jsonl",
    question="Which edges of the stand-in carry its next-token prediction?",
    hypothesis="A sparse subgraph keeps the KL to the full model below tau",
    metric="KL(full || circuit) at the last position, batch mean",
)
candidates = {}
for layer in range(2):
    plan = head_ablation_candidates(clean, f"transformer.h.{layer}.attn", heads=2)
    candidates.update({f"a{layer}.{key}": site for key, site in plan.items()})
knockout = site_sweep(
    clean,
    candidates=candidates,
    edit=tl.zero_ablate(),
    metric=lambda member: kl_to_full(member[member.output_layers[0]].out),
    retain="none",
    model=model,
    x=clean_ids,
)
knockout_kl = {row.candidate_id: round(row.value, 4) for row in knockout.effects().rows
               if row.candidate_id != "__baseline__"}
print(knockout_kl)
led.observe(f"head knockout KL: {knockout_kl}")
led.verdict("not_assessed", text="reported beside the ACDC circuit; not a pruning input")
led.close_entry()
RESULT = sorted(knockout_kl) == ["a0.h0", "a0.h1", "a1.h0", "a1.h1"]
```

## The ACDC loop

Receivers in reverse topological order (`logits` first; here only the last
two), each receiver's senders latest first; bias edges are skipped, since a bias is the same on
both inputs and patching it changes nothing. Every test opens one ledger
entry, records the observed KL change and sets the verdict: `refutes` (the
edge is not needed: pruned) or `supports` (kept).

```python
# recipe-test: tool=transformers expected=True
tau = 0.002
pruned = []
current = circuit_kl(pruned)
walk = [graph.receivers["logits"], graph.receivers["m1"]]  # full walk: reversed(graph.receivers)
tested = []
for receiver in walk:
    for sender in reversed(receiver.senders):
        if sender.kind == "bias":
            continue
        edge = graph.edges[f"{sender.name}->{receiver.name}"]
        tested.append(edge.name)
        trial = circuit_kl(pruned + [edge])
        led.entry(
            question=f"Is {edge.name} needed?",
            hypothesis=f"{edge.name} carries part of the prediction",
            metric="KL(full || circuit) change when the edge is patched",
        )
        change = trial - current
        led.observe(f"KL {current:.6f} -> {trial:.6f}", value=change)
        if change < tau:
            pruned.append(edge)
            current = trial
            led.verdict("refutes", text=f"pruned: the KL change is below tau={tau}",
                        basis=[f"KL change {change:.6f}"])
        else:
            led.verdict("supports", text=f"kept: the KL change reaches tau={tau}",
                        basis=[f"KL change {change:.6f}"])
        led.close_entry()

pruned_names = {e.name for e in pruned}
kept = [name for name in tested if name not in pruned_names]
print(f"kept {len(kept)} of {len(tested)} tested edges, final KL {current:.6f}")
print(kept)
RESULT = (len(tested), 0 < len(kept) < len(tested)) == (13, True)
```

## On IOI with GPT-2 small

The same loop runs unchanged on GPT-2 small: load `openai-community/gpt2`
with `attn_implementation="eager"`, take the clean prompts and their
corrupted counterparts from the MIB IOI task (`mib-bench/ioi`, the
`s2_io_flip_counterfactual` column, padded to one width), and keep the
metric (or use minus the logit difference of the indirect object over the
subject). Each test patches every pruned edge in one multi-receiver call,
so the cost of a test grows with the number of receivers already touched;
budget a background job. The measured cost on 20 rows and 4 CPU cores (0.9 s
for a test into `logits`, 138 s for one that touches every receiver, about
450 hours for the full walk) is in the
[mechinterp reference](../reference/mechinterp.md), under the circuit test
cases on GPT-2 small.

## The record

Every decision is a closed ledger entry with a verdict, and the kept set is
exactly the edges whose verdict is `supports`. Re-running the circuit the
ledger describes (its `refutes` edges patched, in ledger order) in one call
reproduces the loop's last KL exactly.

```python
# recipe-test: tool=transformers expected=True
decisions = [entry for entry in led.entries().values() if (entry.question or "").startswith("Is ")]
verdicts = [(entry.question[3:-8], entry.verdict["token"]) for entry in decisions]
from_ledger_kept = [name for name, token in verdicts if token == "supports"]
from_ledger_pruned = [graph.edges[name] for name, token in verdicts if token == "refutes"]
led.close()
again = circuit_kl(from_ledger_pruned)
print(len(decisions), sorted(from_ledger_kept) == sorted(kept), again == current)
RESULT = (len(decisions), sorted(from_ledger_kept) == sorted(kept), again == current) == (13, True, True)
```

Printed output on CPU (float64): filled in from the page's test run.

The page's test (`tests/test_circ3_acdc_recipe.py`) runs these fences in
order, re-runs the pruning loop independently of the ledger and checks that
it prunes the same edges, and checks that every decision entry carries a
verdict.
