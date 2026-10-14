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

- `b.relate("successor_of", "writer", "router")` for the order of the calls;
- the route decision in your own metadata, for example
  `b["writer"].annotations["harness"] = {"route_from": "router", "route": 1}`;
- `fed_by` only for text that really flows, such as `writer fed_by lookup` when the looked-up
  text came from a traced call.

TorchLens has no control-dependency relation kind, so it cannot answer "which calls did this
decision select". `label` is free human text (`"critique"`, `"judge"`, `"routed:calc"`); nothing
in TorchLens reads it.

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
- `"matched"`: reserved for automatic span finding, which is not available yet; a `"matched"`
  claim that re-checks is stored as `"verified"`.

A row claiming `"verified"` or `"matched"` is checked again wherever it enters a bundle (relate,
the constructor, loading an artifact). Missing evidence turns it into `"declared"`; evidence that
contradicts it refuses with `bundle_relation_span_unverified`. Spans past a member's token stream
refuse with `bundle_relation_span_out_of_bounds`.

`coverage` on a verified row is computed: the covered share of the source's eligible tokens, where
the eligible set is the bound completion range minus `excluded_ids`. On a declared row, `coverage`
is whatever you declared. Span rows need start < end on both sides; spans are measured against a
member's stream only when its binding names the length.

## Reading the graph

- `b.upstream(name)` / `b.downstream(name)`: transitive walks over the ordering kinds (`fed_by`,
  `successor_of`, `forked_from`, `escalates`), nearest member first.
- `b.node(site, members=["a", "b"])`: one site across a subset of members, so a foreign member such
  as a small router model does not disable the view for the rest.
- Comparisons: members joined by an ordering path (including `fed_by`) are not comparison
  operands, and `delta_map`, `norm_delta`, `output_delta` and `compare` run the same comparison
  gate as `compare_at`. Two calls on different prompts refuse `bundle_gate_input_values_differ`
  instead of reporting distances between unrelated inputs.

## Not available yet

- `token_spans="auto"` (finding the earlier output inside the later prompt) refuses with
  `bundle_relation_auto_spans_unavailable`.
- A members-level picture of the harness graph.
- Lazy loading that reaches bundle members.

Artifacts carrying `fed_by` rows need a reader of the current tlspec version; earlier readers
refuse the kind.
