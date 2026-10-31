# RepE to TorchLens v2 Migration

| RepE operation | TorchLens v2 idiom | Parity |
| --- | --- | --- |
| Apply a representation direction | `tl.steer(direction, magnitude=...)` | Equivalent for visible activation sites. |
| Remove a direction | `tl.project_off(direction)` | Equivalent helper. |
| Project onto a direction | `tl.project_onto(direction)` | Equivalent helper. |
| Choose layer/token positions | Discover the activation site, then use tensor-shaped directions or custom hooks | Equivalent when shape is visible; no named token abstraction. |
| Run controlled generations | `torchlens.intervention.steer_generate(model, ids, spec)` steers every decoding step, KV cache included | Equivalent for local generation; see `docs/guides/fast_paths.md`. |
| Compare control strengths | Fork logs or use `Bundle.joint_metric` across steered variants | Equivalent analysis pattern. |
| Save steering recipe | `.tlspec/` portable save when direction tensors and built-ins are used | Equivalent publication path. |
| Train/read representation directions | `torchlens.intervention.fit_direction(log_pos, log_neg, "model.layers.<i>", method="pca")` fits the direction from saved last-token activations (`"mean_diff"`, `"pca"` or `"logistic"`), and `direction.to_repeng()` returns a real `repeng.ControlVector` (`direction.to_steering_vectors()` does the same for steering-vectors, and `direction.to_dialz()` for dialz) | Changed: TorchLens fits the direction itself instead of replicating `ControlVector.train`, so the vector differs from repeng's `pca_diff`; `ControlModel.set_control(vector, coeff=m)` on the converted vector reproduces the logits of `direction.steer(m)` at that layer. Padded batches read the last real token when the trace saved its `attention_mask` input; without a mask, `read_token_index=-1` reads the last position, a pad token on right-padded rows. |
| Batch evaluation | `tl.when(site, action).bind(model)` to steer many forwards, or `trace.run(inputs=batch)` per batch | Equivalent; see `docs/guides/fast_paths.md`. |
| Intervene inside fused attention | Default fused SDPA load: by-head and by-pattern patching through `torchlens.mechinterp.lowered_counterfactual` (no eager reload); eager-attention load: the softmax and score ops are ordinary intervention sites | Head-level and pattern-level edits work on fused models; other edits inside the fused kernel need an eager recapture. |

## Other steering tools

- **Ember** (Goodfire) is a hosted alternative: a hosted API that steers a model through
  its learned features, with the model running on Goodfire's servers. Its Python SDK repository
  (`goodfire-ai/goodfire-sdk`) is archived.
- **EasySteer** is the serving-speed path: steering inside vLLM. Build and check a
  steering vector in TorchLens (where every site is visible and the effect can be
  measured), then serve it in EasySteer when generation speed matters.
