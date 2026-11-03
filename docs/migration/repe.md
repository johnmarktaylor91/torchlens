# Coming from RepE

This page is for people who read and steer a model with representation-engineering (RepE)
control vectors, in the repeng or the original RepE style, and want the same workflow in
TorchLens. TorchLens fits the direction itself and can hand it to repeng
(`direction.to_repeng()`, `torchlens[repeng]` extra, `repeng~=0.4`). repeng is not executed
here; the TorchLens side runs on a tiny model built in the fence. Checked 2026-11-03.

## The same task in both

Fit a direction from positive and negative prompts at one layer, then steer with it.

### In repeng

repeng trains a control vector from a dataset of contrastive pairs and applies it through a
`ControlModel`. This block is not executed here (it needs a tokenizer and a real checkpoint);
check the call against your repeng version:

```text
from repeng import ControlModel, ControlVector, DatasetEntry

model = ControlModel(base_model, layer_ids=[0])
vector = ControlVector.train(model, tokenizer, dataset)   # dataset: list of DatasetEntry
model.set_control(vector, coeff=4.0)
```

### In TorchLens

`fit_direction` reads the saved activations of the positive and negative traces at a site and
fits the direction (`"mean_diff"`, `"pca"` or `"logistic"`); `direction.steer(strength)` is the
edit, placed with `tl.when(tl.module(site), ...)`.

```python
# migration-test: tool=torchlens expected=((32,), 'mean_diff', True)
import torch
import torchlens as tl
from torchlens.intervention import fit_direction
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
lm = LlamaForCausalLM(
    LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
).eval()
lm.config.use_cache = False  # traces read plain forwards
site = "model.layers.0.mlp"
pos_ids = torch.randint(0, 128, (4, 6))
neg_ids = torch.randint(0, 128, (4, 6))
pos = tl.trace(lm, pos_ids, save="all")
neg = tl.trace(lm, neg_ids, save="all")
direction = fit_direction(pos, neg, site)

steer = tl.when(tl.module(site), direction.steer(4.0))
with torch.no_grad():
    plain = lm(pos_ids).logits
    steered = steer.bind(lm)(pos_ids).logits
RESULT = (tuple(direction.vector.shape), direction.method, not torch.equal(plain, steered))
```

## Translation table

| RepE operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Apply a representation direction | `tl.steer(direction, strength=...)` | Equivalent for visible activation sites. |
| Remove a direction | `tl.project_out(direction)` | Equivalent helper. |
| Project onto a direction | `tl.project_onto(direction)` | Equivalent helper. |
| Choose layer/token positions | Discover the activation site, then use tensor-shaped directions or custom hooks | Equivalent when shape is visible; no named token abstraction. |
| Run controlled generations | `spec.bind(model).generate(ids, ...)` steers every decoding step, KV cache included | Equivalent for local generation; see `docs/guides/fast_paths.md`. |
| Compare control strengths | Fork logs and call a metric on a `tl.Bundle` of the steered variants | Equivalent analysis pattern. |
| Save steering recipe | `.tlspec/` portable save when direction tensors and built-ins are used | Equivalent publication path. |
| Train/read representation directions | `torchlens.intervention.fit_direction(log_pos, log_neg, "model.layers.<i>", method="pca")` fits the direction from saved last-token activations (`"mean_diff"`, `"pca"` or `"logistic"`), and `direction.to_repeng()` returns a real `repeng.ControlVector` (`direction.to_steering_vectors()` does the same for steering-vectors, and `direction.to_dialz()` for dialz) | Changed: TorchLens fits the direction itself instead of replicating `ControlVector.train`, so the vector differs from repeng's `pca_diff`; `ControlModel.set_control(vector, coeff=m)` on the converted vector reproduces the logits of `direction.steer(m)` at that layer. Padded batches read the last real token when the trace saved its `attention_mask` input; without a mask, `read_token_index=-1` reads the last position, a pad token on right-padded rows. |
| Batch evaluation | `tl.when(site, action).bind(model)` to steer many forwards, or `trace.run(inputs=batch)` per batch | Equivalent; see `docs/guides/fast_paths.md`. |
| Intervene inside fused attention | Default fused SDPA load: by-head and by-pattern patching through `torchlens.mechinterp.patch_head` (no eager reload); eager-attention load: the softmax and score ops are ordinary intervention sites | Head-level and pattern-level edits work on fused models; other edits inside the fused kernel need an eager recapture. |

## What TorchLens adds

- Any executed op is a site, so the same direction can be read or applied at a module output,
  a head, or a position (`tl.module`, `tl.in_module`, `tl.units`).
- Steering through the model's own `generate()` keeps the edit on every decoding step, KV cache
  included: `spec.bind(model).generate(ids, ...)`.
- Provenance with the vector: `fit_direction` records the method, the number of pairs and
  the site with the direction (`direction.provenance`), and a `do()` edit is stamped into the
  trace's audit record like any other.
- Removing or isolating a direction with `tl.project_out(direction)` and
  `tl.project_onto(direction)`.
- A tested route to repeng, steering-vectors and dialz objects from one fitted direction
  (`direction.to_repeng()`, `direction.to_steering_vectors()`, `direction.to_dialz()`).

## What RepE does that TorchLens does not

- repeng's `ControlVector.train` and `ControlModel` are a packaged training and application
  loop; TorchLens fits the direction itself with a different procedure, so the vectors differ.
- The original RepE reading and control toolkits (pipelines for honesty and similar concepts)
  are not reproduced; TorchLens gives you the sites and edits to build them.
- **Ember** (Goodfire) is a hosted alternative: a hosted API that steers a model through its
  learned features, with the model running on Goodfire's servers. Its Python SDK repository
  (`goodfire-ai/goodfire-sdk`) is archived.
- **EasySteer** is the serving-speed path: steering inside vLLM. Build and check a steering
  vector in TorchLens (where every site is visible and the effect can be measured), then serve
  it in EasySteer when generation speed matters. See also
  [where to go instead](where_to_go_instead.md).

## Using both

### repeng feeds TorchLens

There is no bridge from a repeng `ControlVector` into a TorchLens edit; convert the
layer's direction to a tensor yourself and pass it to `tl.steer(direction, strength=...)`,
which accepts a plain tensor direction.

### TorchLens feeds repeng

`direction.to_repeng()` returns a real `repeng.ControlVector`;
`direction.to_steering_vectors()` does the same for steering-vectors and `direction.to_dialz()`
for dialz. `ControlModel.set_control(vector, coeff=m)` on the converted vector reproduces the
logits of `direction.steer(m)` at that layer (see the "Train/read representation directions"
row). For serving, build and check the vector in TorchLens, then serve it in EasySteer.

## Semantic differences

- **Direction differs.** TorchLens fits the direction itself instead of replicating
  `ControlVector.train`, so the vector differs from repeng's `pca_diff`.
- **Padded batches.** The last real token is read when the trace saved its `attention_mask`
  input; without a mask, `read_token_index=-1` reads the last position, a pad token on
  right-padded rows.
- **Strength.** `direction.steer(m)` adds `m` times the vector on the trailing axis;
  repeng's `coeff` is the same multiplier on its own vector.
- **Fused attention.** Head-level and pattern-level edits work on fused models through
  `torchlens.mechinterp.patch_head`; other edits inside the fused kernel need an
  eager recapture.
