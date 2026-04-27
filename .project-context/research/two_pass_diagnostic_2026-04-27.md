## TorchLens Two-Pass ResNet50 Failure Diagnostic (2026-04-27)

### Summary
Root cause: TorchLens fast Pass 2 incorrectly inserts an extra synthetic `identity` op after the first in-place `ReLU` in ResNet50. Pass 1 records operation 7 as `maxpool2d_1_7_raw`, but Pass 2 records operation 7 as `identity_1_7_raw`, so the raw-label structural map lookup fails before Pass 2 completes.

This is a TorchLens bookkeeping bug in fast-mode module pass-through detection, not a PyTorch nondeterminism issue and not a bad selector.

### Reproduction
Minimal failing case, run against a clean `HEAD` export at `/tmp/two_pass_diag/head_src` because the live workspace has pre-existing dirty edits in `constants.py` / `layer_pass_log.py` that currently break Pass 1 construction.

```python
import torchvision.models as M
import torch, torchlens as tl

model = M.resnet50(weights=None).eval()
x = torch.randn(2, 3, 224, 224)

tl.log_forward_pass(model, x, layers_to_save=["conv2d_1_1"])
```

Failure:

```text
ValueError: The computational graph changed for this forward pass compared to the original call to log_forward_pass (either due to different inputs or a different random seed), so save_new_activations failed. Please re-run log_forward_pass with the desired inputs.
```

Raise site: `torchlens/capture/output_tensors.py:692`, inside `log_function_output_tensors_fast`, at the check:

```python
if tensor_label_raw not in self._raw_to_final_layer_labels:
    raise ValueError(...)
```

### Root Cause
The first failing Pass 2 op is:

```text
func_name: identity
layer_type: identity
creation_order: 7
layer_type_num: 1
tensor_label_raw: identity_1_7_raw
parent_raw_labels: ["relu_1_6_raw"]
parent_final_labels: ["relu_1_3"]
```

Pass 1 recorded the same structural position as:

```text
creation_order: 7
layer_type: maxpool2d
tensor_label_raw: maxpool2d_1_7_raw
final label: maxpool2d_1_4
parents: ["relu_1_3"]
```

So the concrete mismatch is:

```text
Pass 1: tensor_label_raw=maxpool2d_1_7_raw, layer_type=maxpool2d, func=maxpool2d
Pass 2: tensor_label_raw=identity_1_7_raw, layer_type=identity, func=identity
```

The parent is not the real discrepancy at this point: both point back to `relu_1_3`.

Why it happens:

`torchlens/decoration/model_prep.py:746-760` fast mode computes `input_tensor_labels` after `orig_forward(*args, **kwargs)` returns. For an in-place module like `nn.ReLU(inplace=True)`, the input tensor object is also the output tensor object, and the decorated in-place ReLU has already updated that object’s `tl_tensor_label_raw` to `relu_1_6_raw`.

Fast mode then thinks the output tensor label is one of the input labels and treats the module as a pass-through, forcing `_decorated_identity(t)`. Exhaustive mode does not make this mistake because it captures input labels before running the module.

This is TorchLens-internal state/ordering bookkeeping around live tensor attributes, not a real model graph change.

### Reproducibility Scope
- Smaller models:
  - `Conv2d -> ReLU(inplace=False) -> Conv2d -> Flatten -> Linear`: two-pass succeeds.
  - Same model with `ReLU(inplace=True)`: two-pass fails with the same graph-changed error.
- ResNet50 with all `nn.ReLU` modules changed to `inplace=False`: two-pass succeeds.
  - `relu_modules_changed 17`
  - `two_pass_ok 283 ['conv2d_1_1', 'linear_1_175', 'output_1']`
- Known-good selector from Pass 1:
  - Pass 1 verified `conv2d_1_1`, `relu_1_3`, `maxpool2d_1_4`, and `linear_1_175`.
  - All tested selectors still failed: `['conv2d_1_1']`, `['relu_1_3']`, `['maxpool2d_1_4']`, `[1]`, `[-1]`, `['linear_1_175']`.
- Determinism:
  - Raw PyTorch ResNet50 forward is deterministic in eval mode.
  - Two raw forwards produced `allclose=True`, `max_abs_diff=0.0`.
  - The TorchLens failure reproduced deterministically across repeated attempts.

### Structural Check Correctness
The check at `output_tensors.py:692` is not overly strict here. It is correctly detecting that Pass 2 has an extra operation that Pass 1 did not record.

The bug is earlier: fast-mode module pass-through detection creates a spurious `identity` node. Ignoring `identity` in structural comparison would hide real label drift and would leave subsequent counter alignment wrong.

### History / Regression Notes
Requested history checks:

```text
git log --follow -- torchlens/user_funcs.py | grep -i "two[- ]pass"
42c047f fix(io): IO-S9 path-traversal hardening + two-pass streaming rejection + docs security warning + preflight/reader cleanup
```

`output_tensors.py` recent history starts with:

```text
0d9a41d feat(capture): add autograd_saved_bytes field tracking per-op autograd memory
f589700 feat(data-classes): add extra_data dict to layer logs and input_metadata to ModelLog
5713a1e feat(backward): first-class backward-pass capture ...
1b19663 feat(fastlog): high-throughput activation recording for dynamic models (#155)
...
```

More directly relevant blame:

- `output_tensors.py:695-701` missing raw-map check comes from `2cfa90a`.
- `model_prep.py:747-760` fast-mode input-label/pass-through logic is blamed mostly to `326b8a90` with comment movement at `b5da8b8`.
- This makes `model_prep.py` fast-mode pass-through detection the likely regression site, not the selector resolution code in `user_funcs.py`.

### Suggested Fix Direction
Fix Pass 2 setup / fast module wrapper behavior, not the structural comparison.

High-level options:

1. Capture fast-mode module input tensor labels before `orig_forward(*args, **kwargs)`, matching exhaustive mode’s ordering.
2. For in-place modules, compare against pre-forward labels/object state rather than post-forward mutated `tl_tensor_label_raw`.
3. Keep `_decorated_identity()` for true `nn.Identity` and genuine pass-through modules, but avoid treating in-place mutation outputs as pass-throughs.

### Effort Estimate
Medium.

The likely fix is localized to `torchlens/decoration/model_prep.py:746-760`, but it needs careful coverage because that logic exists to preserve fast/exhaustive alignment for true pass-through modules. The comparison logic in `torchlens/capture/output_tensors.py:692` should probably remain strict.

### Followup Test Recommendation
Add a regression test using a small model with `nn.ReLU(inplace=True)`:

```text
Conv2d -> ReLU(inplace=True) -> Conv2d -> Flatten -> Linear
```

Assert that:

- `tl.log_forward_pass(model, x, layers_to_save=['conv2d_1_1'])` succeeds.
- No extra `identity_*` raw label appears after the in-place ReLU unless exhaustive mode also records one.
- A ResNet-style smoke test with `torchvision.models.resnet50(weights=None).eval()` and `layers_to_save=['conv2d_1_1']` succeeds when torchvision is available.

### Artifact Paths
- `/tmp/two_pass_diag/minimal_repro_head.log`: clean-HEAD minimal ResNet50 failure traceback.
- `/tmp/two_pass_diag/diagnose_resnet50.py`: temporary monkey-patch diagnostic script.
- `/tmp/two_pass_diag/resnet50_diagnostic.json`: first failing Pass 2 op and Pass 1 label-map snapshot.
- `/tmp/two_pass_diag/pass1_labels.log`: Pass 1 first raw labels; confirms no `identity_*` labels.
- `/tmp/two_pass_diag/pass1_key_entries.log`: Pass 1 details for `relu_1_3`, `maxpool2d_1_4`, `conv2d_2_5`.
- `/tmp/two_pass_diag/small_models.log`: small CNN non-inplace passes, inplace fails.
- `/tmp/two_pass_diag/selector_matrix.log`: selector irrelevance matrix.
- `/tmp/two_pass_diag/raw_forward_resnet.log`: raw PyTorch determinism check.
- `/tmp/two_pass_diag/resnet50_noninplace.log`: ResNet50 succeeds after setting ReLUs to `inplace=False`.
- `/tmp/two_pass_diag/head_src/`: clean `git archive HEAD` export used for diagnostics.
