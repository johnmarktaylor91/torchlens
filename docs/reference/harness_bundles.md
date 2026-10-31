# Harness bundles

*Spellings documented-unstable, like the episode surface.*

A harness runs several model calls where a later prompt contains an earlier output: a draft
and its critique, a judge reading candidates, a router picking a tool, two agents debating.
TorchLens records each call you choose to trace as a member of one `Bundle`, and each flow of
text from one call into another as a `fed_by` relation row between the two members. There is no
separate harness type: the bundle and its relation table are the harness graph.

Scope: local PyTorch models only. A member is a captured eager forward of an `nn.Module` (or a
bound method of one) in this process. Hosted APIs, serving engines, other processes and calls you
did not trace are not members; text they produce can still appear in a local prompt, where no
span covers it.

## Recording a flow

```python
import torch

import torchlens as tl


class TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(9, 4)
        self.head = torch.nn.Linear(4, 9)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.embed(ids)))


torch.manual_seed(0)
lm = TinyLM()
draft_ids = torch.tensor([[1, 2, 3, 5, 6, 7]])  # prompt [1, 2, 3], then the draft [5, 6, 7]
critique_ids = torch.tensor([[4, 5, 6, 7, 8]])  # the critique prompt quotes the draft

b = tl.Bundle()
b.add(tl.trace(lm, draft_ids), names="draft")
b.add(tl.trace(lm, critique_ids), names="critique")
# Bind which tokens of the draft's stream are its output (positions 3 to 6).
draft_output = {"stream": "input", "batch_row": 0, "completion": [3, 6]}
tl.io.attach_sidecar(b["draft"], "torchlens.boundary_facts", draft_output)
b.relate("fed_by", "critique", "draft", token_spans=[[3, 6, 1, 4]], label="critique")

row = b.member_relations[0]
assert (row.from_member, row.to_member) == ("critique", "draft")
assert row.params["basis"] == "verified" and row.params["coverage"] == 1.0
assert b.upstream("critique") == ("draft",)
assert b.downstream("draft") == ("critique",)
```

`b.relate(kind, later, earlier, **params)` reads as the sentence it records: "critique fed_by
draft". The later member comes first, as on every ordering relation (`successor_of` too), so
`upstream` walks from a reader to what fed it and `downstream` the other way.

## What `fed_by` means

`fed_by` is content provenance only: the later call's prompt contains the earlier call's output.
It is not "the earlier call caused the later one".

A router is the case where the two differ. A small router model returns `1`; your code then picks
the lookup branch and calls the writer with `"Answer using this context: Paris"`. The router's
output never enters the writer's prompt, so `writer fed_by router` would be a false statement,
and an empty span list would not repair it (an empty list means a real feed whose alignment is
unknown). Record that harness as:

- `b.relate("selected_by", "writer", "router", chosen=1, candidates=[0, 1])`: the router's decision
  selected the writer call;
- `fed_by` only for text that really flows, such as `writer fed_by lookup` when the looked-up
  text came from a traced call.

## What `selected_by` means

"writer selected_by router" records a control decision: the earlier call's output decided whether
the later call ran, or which input it got. The decider's text need not appear in the later prompt,
and the row never says the decider caused the later call's content. The direction is the one every
ordering kind uses: the selected call comes first, the decider second, so `upstream("writer")`
reaches the router and `downstream("router")` lists every call its decision selected (and what
those calls fed).

Its params are a closed set, all optional:

- `chosen`: the decision as one JSON scalar (route `1`, candidate `"c2"`);
- `candidates`: the options the decider chose among, distinct JSON scalars; `chosen` must be one of
  them when both are given. They may be member names, routes or tool names; TorchLens never
  resolves them as members;
- `evidence`: why it chose, as the graded evidence envelope `successor_of` takes, for example the
  router's scores. Measurement grades you write are stored as `disclosed`.

A judge that picks a winner among candidate calls selects the call that uses the winner:
`b.relate("selected_by", "final", "judge", chosen="c2", candidates=["c1", "c2", "c3"])`, plus
`final fed_by c2` for the winning text the final prompt quotes. A malformed param (a list as
`chosen`, a repeated candidate, `chosen` outside `candidates`) refuses
`bundle_relation_schema_invalid` and leaves the table unchanged. `label` is free human text on
`fed_by` rows (`"critique"`, `"judge"`); nothing in TorchLens reads it.

## Token spans, and when a row is verified

`token_spans` rows are four integers, half-open: source start, source end, destination start,
destination end. Four integers alone are not a checkable claim, so each member can carry a
token-stream binding: a `token_boundary_v1` descriptor in the reserved `torchlens.boundary_facts`
sidecar family naming

- `stream`: which captured tensor the coordinates index (`"input"`, the first integer-dtype input
  of the capture; or `"output"`);
- `batch_row`, and the measured `length` and `ids_digest` (the digest of the row's token ids),
  which you may leave out: `relate` fills them from the retained ids;
- `completion`, the member's own output range inside that stream, which a source needs before
  any of its rows can verify; `excluded_ids`, token ids outside the eligible set (padding,
  special tokens); and `tokenizer` / `template` identity strings.

Attach a descriptor before relating with
`torchlens.io.attach_sidecar(trace, "torchlens.boundary_facts", {...})`, as above. When you relate
with `basis` omitted and a member has none, TorchLens derives a default input-stream binding for a
batch-of-one capture whose ids are retained; it has no completion range, so a source bound only
that way leaves its rows declared.

`basis` says what the row stands on:

- `"verified"`: TorchLens checked it in this process. Both members are bound, each binding's
  digest equals the member's retained ids, the source binding names its completion range, every
  source span lies inside it, and the token ids agree span by span. Shared prompt tokens alone
  never verify a feed.
- `"declared"`: anything less. Retokenized or paraphrased text, a batch row TorchLens cannot pick,
  or ids that were not retained all leave a row declared. A row without a binding can never read
  as verified.
- `"matched"`: TorchLens found the spans itself (`token_spans="auto"`, below) and they verify.

A row claiming `"verified"` or `"matched"` is checked again wherever it enters a bundle (relate,
the constructor, loading an artifact). Missing evidence turns it into `"declared"`; evidence that
contradicts it refuses with `bundle_relation_span_unverified`. Spans past a member's token stream
refuse with `bundle_relation_span_out_of_bounds`.

`coverage` on a verified row is computed: the covered share of the source's eligible tokens, where
the eligible set is the bound completion range minus `excluded_ids`. On a declared row, `coverage`
is whatever you declared. Span rows need start < end on both sides; spans are measured against a
member's stream only when its binding names the length.

## Finding the spans automatically

`b.relate("fed_by", "critique", "draft", token_spans="auto")` asks TorchLens to find the draft's
output inside the critique's prompt, by integer token ids only (no tokenizer is involved):

- The source's eligible output is its bound `completion` range minus `excluded_ids`, split into
  runs wherever an excluded id sits. Each run is searched as an exact run of ids in the reader's
  prompt: the reader's bound stream before its own `completion`, when its binding names one.
- A run that is not found whole is retried with up to two boundary tokens trimmed per side, fewest
  trimmed tokens first (a quoted output often retokenizes at its edges). A trimmed run keeps at
  least two tokens and at least half its length; trimmed tokens are simply not covered, and the
  row's `coverage` says so.
- At the first trim level with any hit there must be exactly one placement. A repeated quote, or a
  tie between two trims, refuses `bundle_relation_auto_spans_ambiguous`: declare the spans
  yourself. No placement at all refuses `bundle_relation_auto_spans_unmatched`; missing evidence
  (no source completion range, token ids not retained) refuses
  `bundle_relation_auto_spans_unavailable`.
- The row is written with `basis="matched"` and a computed `coverage`. `basis` and `coverage`
  cannot be passed alongside `"auto"`.

A `matched` row is attested by re-derivation wherever it enters a bundle (relate, the constructor,
fork, loading an artifact): its spans must verify exactly like `verified` spans, and the matcher,
re-run over the members' retained token ids, must return the same spans. Spans that verify but are
not what the matcher finds are stored as `verified` (checked, but not found by TorchLens); spans
that contradict the ids refuse `bundle_relation_span_unverified`; missing evidence leaves the row
`declared`. Saved bundles keep each member's token-id input and its binding, so a matched row
re-attests after `tl.save` and `tl.load`, lazy or not.

## Reading aligned activations

`b.token_pairs(reader, source)` is the one verified alignment read. Its arguments follow `relate`:
the reader (the later call) first, then the source. It returns index pairs
`(source_pos, reader_pos)` for token positions TorchLens established one to one, and reports
everything else:

- Only spans of `verified` or `matched` rows that re-check now are paired.
- Declared spans stay span correspondences and are never converted to pairs. Two source tokens
  that became one reader token, or a reader token that fuses a template delimiter with copied
  text, are both declared spans and show up in `excluded` with a reason.
- Excluded ids (padding, special tokens), positions claimed by two spans, and streams whose
  bindings name different tokenizers are reported, not paired.
- Equal lengths are never evidence of matching meaning; only equal ids at checked positions are.
- `coverage` is the paired share of the source's eligible tokens; every eligible source token left
  unpaired appears in `excluded`.

Slicing is yours: index each member's activations with the pair positions.

```python
import torch

import torchlens as tl


class TinyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(32, 8)
        self.block = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, 32)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.block(self.embed(ids))))


torch.manual_seed(0)
lm = TinyLM()
draft = tl.trace(lm, torch.tensor([[1, 2, 3, 7, 8, 9, 10]]))  # prompt, then the output 7..10
binding = {"stream": "input", "batch_row": 0, "completion": [3, 7]}
tl.io.attach_sidecar(draft, "torchlens.boundary_facts", binding)
critique = tl.trace(lm, torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11]]))
b = tl.Bundle({"draft": draft, "critique": critique})
b.relate("fed_by", "critique", "draft", token_spans="auto", label="critique")
assert b.member_relations[0].params["basis"] == "matched"

pairs = b.token_pairs("critique", "draft")
assert pairs.pairs == ((3, 3), (4, 4), (5, 5), (6, 6)) and pairs.coverage == 1.0
in_draft = b["draft"]["block"].out[0, pairs.source_positions]
in_critique = b["critique"]["block"].out[0, pairs.reader_positions]
per_token = torch.nn.functional.cosine_similarity(in_draft, in_critique, dim=-1)
assert per_token.shape == (4,)

# A retokenized quote: source tokens 7, 8 became one reader token (29). Declared, never paired.
merged = tl.trace(lm, torch.tensor([[4, 5, 29, 9, 10, 11]]))
b.add(merged, names="merged")
b.relate("fed_by", "merged", "draft", token_spans=[[3, 5, 2, 3]], basis="declared")
assert b.token_pairs("merged", "draft").pairs == ()
# Averaging the two source tokens is YOUR reduction: a descriptive comparison, not an alignment.
group_mean = b["draft"]["block"].out[0, 3:5].mean(dim=0)
descriptive = torch.nn.functional.cosine_similarity(group_mean, b["merged"]["block"].out[0, 2], dim=0)
```

Any reduction over a many-to-one group (a mean, a sum, the last token) is a choice you make; name
it where you report the number and call the result a descriptive comparison.

## Reading the graph

- `b.upstream(name)` / `b.downstream(name)`: transitive walks over the ordering kinds (`fed_by`,
  `selected_by`, `successor_of`, `forked_from`, `escalates`), nearest member first.
- `b.node(site, members=["a", "b"])`: one site across a subset of members, so a foreign member such
  as a small router model does not disable the view for the rest.
- `tl.show_bundle_graph(b, nodes="members")`: one node per member and one edge per relation row,
  drawn from the earlier member to the later one and labelled with the kind, the row's `label`, on
  `fed_by` its basis and coverage, and on `selected_by` the `chosen` decision. Solid edges carry token spans TorchLens checked
  (`verified` or `matched`); every other row is dashed. `nodes="ops"` (the default) keeps the
  bundle's union op graph.
- Comparisons: members joined by an ordering path (including `fed_by` and `selected_by`) are not
  comparison operands, and `delta_map`, `norm_delta`, `output_delta` and `compare` run the same
  comparison gate as `compare_at`. Two calls on different prompts refuse `bundle_gate_input_values_differ`
  instead of reporting distances between unrelated inputs.
- `tl.load(path, lazy=True)` on a saved bundle loads every member lazily (tensors stay on disk
  until read); `map_location` and `materialize_nested` reach the members too.

Artifacts carrying `fed_by` or `selected_by` rows need a reader of tlspec 10 or later; earlier
readers refuse the kinds, and an artifact stamped with an earlier version that carries either kind
refuses `bundle_relation_schema_invalid` (no writer of that version could have produced it).

## Four harnesses

Each recipe generates natively, then captures one teacher-forced forward over prompt plus output:
generation runs at native speed (TorchLens's wrappers pass straight through when nothing is being
logged), and the capture costs one forward's overhead. The captured activations are those of a
cache-free forward over the same tokens: for a causal decoder that is the same computation as the
generation loop, not the same bytes. When per-step decode internals matter, capture the episode
instead (`docs/reference/episode_capture.md`).

The recipes use a tokenizer-free stand-in model so they run anywhere; with a real model and
tokenizer, `generate` is `model.generate` and the template text goes through the tokenizer. The
helper `run_and_trace` binds each member's completion range, which is what makes its output
findable and verifiable.

### Self-consistency voting

```python
from collections import Counter

import torch

import torchlens as tl


class TinyLM(torch.nn.Module):
    def __init__(self, vocab: int = 256) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 16)
        self.block = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.block(self.embed(ids))))


def generate(lm, ids, *, seed, n=6):  # native sampling; TorchLens is not logging here
    rng = torch.Generator().manual_seed(seed)
    for _ in range(n):
        probs = torch.softmax(lm(ids)[:, -1], dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1, generator=rng)], dim=1)
    return ids


def run_and_trace(lm, prompt, *, seed):
    out = generate(lm, prompt, seed=seed)
    trace = tl.trace(lm, out)  # one teacher-forced forward over prompt + output
    binding = {"stream": "input", "batch_row": 0, "completion": [prompt.shape[1], out.shape[1]]}
    tl.io.attach_sidecar(trace, "torchlens.boundary_facts", binding)
    return out, trace


torch.manual_seed(0)
lm, question = TinyLM(), torch.tensor([[11, 12, 13, 14]])
outs, members = {}, {}
for k in range(4):
    outs[f"s{k}"], members[f"s{k}"] = run_and_trace(lm, question, seed=k)
b = tl.Bundle(members)
for k in range(1, 4):
    b.relate("alternative_of", f"s{k}", "s0")  # peers: not an ordering kind, so still comparable
answers = {name: int(out[0, -1]) % 3 for name, out in outs.items()}  # your parser
winner = Counter(answers.values()).most_common(1)[0][0]
for name in b.names:
    b[name].annotations["harness"] = {"answer": answers[name], "with_majority": answers[name] == winner}

# The shared prompt is a free determinism check: every sample saw the same four tokens.
prompt_acts = [b[name]["block"].out[0, :4] for name in b.names]
assert all(torch.allclose(prompt_acts[0], acts) for acts in prompt_acts[1:])
b.save(f"{DOCS_TMPDIR}/vote.tlspec")
loaded = tl.load(f"{DOCS_TMPDIR}/vote.tlspec", lazy=True)
assert loaded["s2"].annotations["harness"] == b["s2"].annotations["harness"]
```

### Critique and revise

```python
import torch

import torchlens as tl


class TinyLM(torch.nn.Module):
    def __init__(self, vocab: int = 256) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 16)
        self.block = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.block(self.embed(ids))))


def generate(lm, ids, *, seed, n=6):
    rng = torch.Generator().manual_seed(seed)
    for _ in range(n):
        probs = torch.softmax(lm(ids)[:, -1], dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1, generator=rng)], dim=1)
    return ids


def run_and_trace(lm, prompt, *, seed):
    out = generate(lm, prompt, seed=seed)
    trace = tl.trace(lm, out)
    binding = {"stream": "input", "batch_row": 0, "completion": [prompt.shape[1], out.shape[1]]}
    tl.io.attach_sidecar(trace, "torchlens.boundary_facts", binding)
    return out, trace


def chat(*parts):  # stand-in for a chat template: role tokens around quoted text
    return torch.cat([torch.tensor([[1, 2]]), *parts, torch.tensor([[3]])], dim=1)


torch.manual_seed(0)
lm, q = TinyLM(), torch.tensor([[20, 21, 22]])
d_out, draft = run_and_trace(lm, chat(q), seed=1)
draft_text = d_out[:, chat(q).shape[1] :]
c_out, critique = run_and_trace(lm, chat(q, torch.tensor([[4]]), draft_text), seed=2)
critique_text = c_out[:, c_out.shape[1] - 6 :]
r_prompt = chat(q, torch.tensor([[4]]), draft_text, torch.tensor([[5]]), critique_text)
_, revise = run_and_trace(lm, r_prompt, seed=3)

b = tl.Bundle({"draft": draft, "critique": critique, "revise": revise})
b.relate("fed_by", "critique", "draft", token_spans="auto", label="critique")
b.relate("fed_by", "revise", "draft", token_spans="auto")
b.relate("fed_by", "revise", "critique", token_spans="auto", label="revision")
assert b.downstream("draft") == ("critique", "revise")
assert all(row.params["basis"] == "matched" for row in b.member_relations)

# Does the critic encode the draft the way the drafter did? Same model, aligned tokens only.
pairs = b.token_pairs("critique", "draft")
assert pairs.coverage == 1.0
drafted = b["draft"]["block"].out[0, pairs.source_positions]
read = b["critique"]["block"].out[0, pairs.reader_positions]
similarity = torch.nn.functional.cosine_similarity(drafted, read, dim=-1)
```

### Router with tool use

```python
import torch

import torchlens as tl


class TinyLM(torch.nn.Module):
    def __init__(self, vocab: int = 256) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 16)
        self.block = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.block(self.embed(ids))))


class Router(torch.nn.Module):  # a small, different model: ids in, two route scores out
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(256, 4)
        self.score = torch.nn.Linear(4, 2)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.score(self.embed(ids).mean(dim=1))


torch.manual_seed(0)
big, router, q = TinyLM(), Router(), torch.tensor([[30, 31, 32]])
route = tl.trace(router, q)
choice = int(route.output_ops[0].out.argmax())  # the decision; no router TEXT reaches the answer
tools = {0: torch.tensor([[200, 201]]), 1: torch.tensor([[210]])}  # Python tools, not members
answer = tl.trace(big, torch.cat([torch.tensor([[1]]), q, tools[choice]], dim=1))

b = tl.Bundle({"route": route, "answer": answer})
b.relate("selected_by", "answer", "route", chosen=choice, candidates=[0, 1])  # control, not content
assert b.upstream("answer") == ("route",)
assert b.member_relations[0].params["chosen"] == choice
# The tool's tokens in the answer prompt are covered by no span: text from outside any traced
# call. Reads scoped with b.node(site, members=[...]) skip the foreign router model.
```

### Two-agent debate, then a judge

```python
import torch

import torchlens as tl


class TinyLM(torch.nn.Module):
    def __init__(self, vocab: int = 256) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, 16)
        self.block = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.block(self.embed(ids))))


def generate(lm, ids, *, seed, n=6):
    rng = torch.Generator().manual_seed(seed)
    for _ in range(n):
        probs = torch.softmax(lm(ids)[:, -1], dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1, generator=rng)], dim=1)
    return ids


def run_and_trace(lm, prompt, *, seed):
    out = generate(lm, prompt, seed=seed)
    trace = tl.trace(lm, out)
    binding = {"stream": "input", "batch_row": 0, "completion": [prompt.shape[1], out.shape[1]]}
    tl.io.attach_sidecar(trace, "torchlens.boundary_facts", binding)
    return out, trace


def debate_prompt(role_token, history):  # every earlier turn, quoted between separators
    parts = [torch.tensor([[role_token]])]
    for _, text in history:
        parts += [torch.tensor([[9]]), text]
    return torch.cat(parts, dim=1)


torch.manual_seed(0)
lm, turns, history = TinyLM(), {}, [("question", torch.tensor([[40, 41, 42]]))]
for r in range(2):
    for side, role in (("A", 5), ("B", 6)):
        prompt = debate_prompt(role, history)
        out, turns[f"{side}{r}"] = run_and_trace(lm, prompt, seed=10 * r + role)
        history.append((f"{side}{r}", out[:, prompt.shape[1] :]))
_, turns["judge"] = run_and_trace(lm, debate_prompt(7, history), seed=99)

b = tl.Bundle(turns)
names = list(turns)
for i, name in enumerate(names[1:], start=1):
    for earlier in names[:i]:
        label = "judge" if name == "judge" else None
        params = {"label": label} if label else {}
        b.relate("fed_by", name, earlier, token_spans="auto", **params)
assert set(b.upstream("judge")) == {"A0", "B0", "A1", "B1"}
assert all(row.params["basis"] == "matched" for row in b.member_relations)
# How B encodes A's argument, against A's own encoding of it (same model, aligned tokens only).
pairs = b.token_pairs("B1", "A1")
a_side = b["A1"]["block"].out[0, pairs.source_positions]
b_side = b["B1"]["block"].out[0, pairs.reader_positions]
```

## Steer one call and re-run your harness

TorchLens does not push a steer through later calls: it never sees your harness's prompts, parsing
or branching, so it cannot rebuild them. Instead you run your own harness twice, once as it is and
once with one call steered, and hand the second run to the first bundle. TorchLens pairs the calls
by the member names your harness gives them (never by order), checks the pairing, merges the
second run's members as `cf_<name>` and reports, for every call, whether it was steered and
whether its prompt and its output changed, each from which token.

```python
import torch

import torchlens as tl


class TinyLM(torch.nn.Module):  # causal: position t mixes in a running mean of positions <= t
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(64, 16)
        self.block = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, 64)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        steps = torch.arange(1, ids.shape[1] + 1, dtype=x.dtype).view(1, -1, 1)
        x = x + x.cumsum(dim=1) / steps
        return self.head(torch.relu(self.block(x)))


def generate(model, ids, seed):  # native sampling; `model` may be a steered executor
    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(6):
            probs = torch.softmax(model(ids)[:, -1], dim=-1)
            ids = torch.cat([ids, torch.multinomial(probs, 1, generator=rng)], dim=1)
    return ids


def chat(*parts):  # stand-in chat template: role tokens around the text
    return torch.cat([torch.tensor([[1, 2]]), *parts, torch.tensor([[3]])], dim=1)


def critique_loop(lm, q, *, intervene):  # your harness; intervene maps a call name to a spec
    b = tl.Bundle()

    def call(name, prompt, seed):
        spec = intervene.get(name)
        out = generate(spec.bind(lm) if spec else lm, prompt, seed)  # steered generation
        trace = tl.trace(lm, out, intervene=spec) if spec else tl.trace(lm, out)
        binding = {"stream": "input", "batch_row": 0, "completion": [prompt.shape[1], out.shape[1]]}
        tl.io.attach_sidecar(trace, "torchlens.boundary_facts", binding)
        b.add(trace, names=name)
        return out[:, prompt.shape[1] :]

    draft = call("draft", chat(q), seed=1)  # seed each call, not once per harness
    critique = call("critique", chat(q, torch.tensor([[4]]), draft), seed=2)
    call("revise", chat(q, torch.tensor([[4]]), draft, torch.tensor([[5]]), critique), seed=3)
    b.relate("fed_by", "critique", "draft", token_spans="auto")
    b.relate("fed_by", "revise", "draft", token_spans="auto")
    b.relate("fed_by", "revise", "critique", token_spans="auto")
    return b


torch.manual_seed(0)
lm, q = TinyLM().eval(), torch.tensor([[20, 21, 22]])
steer = tl.when(tl.module("block"), tl.steer(torch.randn(16), strength=6.0, dim=-1))

b = critique_loop(lm, q, intervene={})
report = b.add_alternative(critique_loop(lm, q, intervene={"draft": steer}), steered=("draft",))
print(report.table())
calls = {row.call: row for row in report.calls}
assert calls["draft"].status == "steered" and calls["draft"].prompt_divergence is None
assert {calls["critique"].status, calls["revise"].status} <= {"absorbed", "changed"}
assert b.downstream("draft") == ("critique", "revise")  # what the steer makes stale

d = b.token_divergence("critique", "cf_critique")
assert d.prompt_divergence == 6 + calls["draft"].completion_divergence  # where the quote parts
start, end, cf_start, cf_end = d.shared[0]  # the stretch both prompts share, as a span row
same = b["critique"]["block"].out[0, start:end] - b["cf_critique"]["block"].out[0, cf_start:cf_end]
assert same.abs().max() == 0  # causal model: identical tokens before the divergence
```

The table has one line per call:

- `steered`: a call you declared in `steered=`; its prompt is the same, its output moved.
- `absorbed`: the prompt changed (it quotes the steered text) but the output did not. This is the
  common case: the critique read a different draft from that token on, its activations moved, and
  its sampled text happened not to change.
- `changed`: both the prompt and the output changed.
- `unchanged`: neither changed.
- `added` / `dropped`: the call ran in only one run, because the harness took another branch (a
  router chose differently). Its copy is added unpaired, or the original stays unpaired.

The merge writes one `alternative_of` row from each copy to its original (peers, so the copies
stay out of `downstream("draft")`), copies the second run's own relation rows onto the copies (so
`downstream("cf_draft")` is `("cf_critique", "cf_revise")`), and records one `add` operation with
the statuses. `b.token_divergence(original, rerun)` reads any two members bound with a completion
range: `prompt_divergence` and `completion_divergence` are the first positions whose token ids
differ (the completion's counted from its first generated token), `lengths` gives prompt and
completion sizes, and `shared` lists the stretches both runs share (the common prompt prefix, the
common prompt tail after the quote, the common output prefix) as four-int rows in the
`token_spans` grammar. Comparisons after the divergence are descriptive: one paired run is one
sample, so run several seeds before reading a steer's effect into a number.

`add_alternative` checks everything before it changes the bundle, and a refusal leaves it as it
was:

| Case | Code |
|---|---|
| A steered call is missing from either run | `alternative_target_missing` |
| A declared call was captured without a spec that fired | `alternative_target_unsteered` |
| Another call of the second run was captured under a spec | `alternative_steer_undeclared` |
| The spec edits chosen positions or decode steps (`at_positions`, `at_step`) | `alternative_capture_position_dependent` |
| An unsteered call saw the same prompt and wrote different text | `alternative_unchanged_call_diverged` |
| A `cf_<name>` copy would reuse a member name | `alternative_member_name_taken` (pass `prefix=`) |
| A paired call has no binding with a completion range, or its ids are not retained | `bundle_token_divergence_unavailable` |

The fifth row is the noise tripwire: a harness reseeded between runs, or nondeterministic kernels,
change text the steer never reached. Seed each call, keep the batch the same and use deterministic
algorithms. A position-dependent steer refuses because one teacher-forced forward does not
reproduce generation under it (capture the episode instead, `docs/reference/episode_capture.md`);
a position-uniform steer captured teacher-forced under the same spec matches the steered
generation. Effects through hosted or untraced calls show only as uncovered tokens, and that the
second run used the same code, templates and weights stays yours to keep true.

## Capture overhead

Measured once, on Wed 14 Oct 2026: gpt2 (the 124M-parameter Hugging Face checkpoint, eval mode) on
one batch of 64 token ids, CPU only, on a shared Linux worker (x86_64, two torch threads; Python
3.10, torch 2.13.0, transformers 5.18.0, TorchLens 2.36.1). Each setting ran twice to warm up, then
seven times, interleaved with the others so drift on the host hits every setting alike. Native is
one `model(ids)` forward under `torch.no_grad()`; each capture is one `tl.trace(model, ids,
capture=...)`.

| Setting | Median | Range over 7 runs | Median / native |
|---|---|---|---|
| Native forward | 0.108 s | 0.106 to 0.110 s | 1.0 |
| `layers_to_save="all"` (the default) | 1.620 s | 1.577 to 1.792 s | 15.0 |
| `layers_to_save=["transformer.h.8"]` (one block, 92 of 567 ops saved) | 1.583 s | 1.549 to 1.818 s | 14.7 |
| `layers_to_save=[]` (metadata only) | 1.470 s | 1.452 to 1.654 s | 13.6 |

Wall-clock time on a shared host is noisy: other jobs ran beside this one, and single runs moved by
up to 15 percent. Read the ratios as the size of the cost, not as a benchmark. Most of the cost is
recording every operation's metadata, so retaining fewer activations saves little time; it saves
memory. A harness capture adds this cost once per traced call, on top of native generation.

A `layers_to_save` selection also decides whether the member keeps its token-id input: only the
default `"all"` retained it in this measurement's checks, and without retained ids
`token_spans="auto"` refuses `bundle_relation_auto_spans_unavailable` and spans stay declared.
Capture harness members with the default retention, or with a selection that names the input
layer as well.
