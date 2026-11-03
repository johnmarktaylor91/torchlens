# Compat matrix: current-release rows

This page records rows of the TorchLens compat matrix that were measured against
releases newer than the pinned CI environment (python 3.10, torch 2.13.0,
transformers 5.18.0). Each row is a capture plus validation run, executed by a
test in `tests/real_model/test_compat_matrix_rows.py`, or a named export check.
A row says exactly what ran, on which versions and which kind of host; a row that
did not run says so and never shows as passing.

The broader support statement (torch 2.1 to 2.12+ for eager capture, the torch
version table, the limitations catalog) lives in
[limitations.md](limitations.md) and the torch capability rows of `tl.report.doctor()`.

## Rows measured 2026-11-03

| Row | Test | torch | transformers | python | Host | Result |
|---|---|---|---|---|---|---|
| Router-logit outputs (`output_router_logits=True`) | `test_router_logit_outputs_capture_exactly_and_validate[mixtral, qwen2_moe, olmoe]` | 2.14.1+cpu | 5.19.0 | 3.11.15 | Linux x86_64 CPU worker | pass (3 of 3) |
| DTensor tensor parallelism, two-process CPU `gloo` mesh | `test_dtensor_tensor_parallel_on_a_two_process_cpu_mesh_refuses_typed` | 2.14.1+cpu | 5.19.0 | 3.11.15 | Linux x86_64 CPU worker | pass: typed refusal on both ranks (torch-native sharding) |
| `flash_attention_4` | `test_flash_attention_4_capture_and_validate` | 2.14.1+cu126 | 5.19.0 | 3.11.2 | Linux x86_64, one NVIDIA L40S (compute capability 8.9), flash-attn-4 4.0.0b37 | pass under `torch.no_grad()`: exact output, each attention call one `autogradfunction` boundary op (exit `flash_attn.cute`), forward validation passes, witnessed trace verified; with autograd enabled the call's second output (log-sum-exp) fails the perturbation check on its v edge (strict expected failure) |
| Router-logit outputs | as above | 2.14.1+cu126 | 5.19.0 | 3.11.2 | Linux x86_64 GPU host (CPU tensors) | pass (3 of 3) |
| DTensor tensor parallelism, two-process CPU `gloo` mesh | as above | 2.14.1+cu126 | 5.19.0 | 3.11.2 | Linux x86_64 GPU host, CUDA hidden from the ranks | pass |
| Router-logit outputs | as above | 2.14.1 | 5.19.0 | 3.11.14 | macOS arm64 (Apple silicon) | pass (3 of 3) |
| DTensor tensor parallelism, two-process CPU `gloo` mesh | as above | 2.14.1 | 5.19.0 | 3.11.14 | macOS arm64 (Apple silicon) | pass: typed refusal on both ranks (torch-native sharding) |
| DTensor tensor parallelism loaded by transformers (`tp_plan="auto"`) | `test_transformers_tp_plan_load_on_a_two_process_cpu_mesh_refuses_typed` | 2.14.1+cpu | 5.19.0 | 3.11.15 | Linux x86_64 CPU worker | pass: typed refusal on both ranks; the sharded logits match the unsharded load |

### Router-logit outputs

Three small mixture-of-experts causal LMs (Mixtral, Qwen2-MoE and OLMoE
architectures; two layers, four experts, top-2 routing, constructed weights, no
download) run with `output_router_logits=True` under both the `eager` and the
`sdpa` attention implementations. Each trace returns the exact model output:
the logits and every per-layer router-logit tensor are bit-equal to the direct
call, each router-logit tensor is an output op of the trace, and forward replay
validation (`tl.validate(..., scope="forward")`) passes.

### DTensor tensor parallelism

Two processes form a CPU `gloo` device mesh. A two-linear MLP is sharded over it
through `torch.distributed.tensor.parallel.parallelize_module` (column-wise then
row-wise), and the sharded forward runs (the parameters are `DTensor`s). On every rank, capture
refuses at entry with the typed `tl.errors.DistributedCaptureUnsupportedError`,
whose findings name the `dtensor` and `tensor_parallel` kinds. This refusal is
the supported behaviour today: capturing below the `DTensor` dispatch layer would
record rank-local shims instead of the real ops. A capture design for sharded
modules is planned but not built; until it is, capture a rank-local dense copy of
the module.

A model loaded by transformers with
`distributed_config=transformers.DistributedConfig(tp_plan="auto")` is the
row's second half. A tiny random-weight Llama checkpoint, pinned by revision and
digest in the test artifact registry, loads sharded on each rank (the attention
projections are `DTensor`s), its logits match the same checkpoint loaded
unsharded, and capture refuses with the same typed error and kinds. Transformers
applies a `tp_plan` only when the rank layout is in the process environment
(`RANK`, `WORLD_SIZE`, `LOCAL_RANK`); without it the checkpoint loads unsharded
and there is no tensor parallelism to test. Transformers refuses a `tp_plan` on
any machine with an MPS device, so this half does not run on macOS.
On macOS, `gloo` needs the loopback interface for a local mesh
(`GLOO_SOCKET_IFNAME=lo0`; the test sets it when unset), else the shard scatter
waits out its timeout.

### `flash_attention_4`

`attn_implementation="flash_attention_4"` needs a CUDA device that the
FlashAttention 4 package supports (compute capability 8.0 and newer; transformers
refuses it on CPU). FlashAttention 4 (`flash_attn.cute`) allocates its output with
`torch.empty` and fills it from a CuTe kernel reached through TVM FFI, below the torch
function layer TorchLens wraps and without the numpy or dlpack export the dynamic
foreign-kernel detector sees, so a plain capture would record the allocation in place
of the attention. TorchLens therefore records every `autograd.Function` of
`flash_attn.cute.interface` (`FlashAttnFunc`, `FlashAttnVarlenFunc`) as one
`autogradfunction` boundary op with `transform_kind="autograd.Function"` and
`transform_config["exits"] == ["flash_attn.cute"]`: its parents are the projected q, k
and v, its output the attention output, and it replays by calling the Function's
`apply` again (the treatment numpy and dlpack round trips already get). The op is
disclosed once per Function class per capture by the `foreign_kernel_boundary`
warning. On the measured GPU the trace returns the exact logits, and with the
completeness witness armed for every capture
(`torchlens.backends.torch.wrappers.wrap_torch(completeness_witness=True)`) the trace
is verified (`trace.provenance.verified is True`, no `CaptureGapWarning`). Under
`torch.no_grad()` the kernel returns the attention output alone and `tl.validate(...,
scope="forward")` passes with the dispatch census balanced (nothing exempted). With
autograd enabled the kernel also returns its log-sum-exp, so each call is two sibling
ops sharing one `func_call_id` (index 0 the bfloat16 attention output, index 1 the
float32 log-sum-exp that nothing consumes); the log-sum-exp depends on q and k only,
and the perturbation check reports its v edge as insensitive, so forward validation
fails there. That failure is correct about the edge and is pinned as a strict expected
failure (`test_flash_attention_4_grad_enabled_validates`) until a rule for multi-output
boundary calls is decided; nothing is exempted to hide it. The same
model on `sdpa` or `eager` attention, same GPU and dtype (bfloat16), captures and
validates op by op. `tl.report.doctor()` reports whether the package is
installed as `HAS_FLASH_ATTENTION_4_INTERFACE` (read from the package's search path;
nothing is imported). The row's test asserts the boundary ops, the exact output and
the passing validation under `torch.no_grad()`; a companion test pins the verified
marking under the armed witness.

## Model Explorer with adapter 0.2.x

`ai-edge-model-explorer-adapter` 0.2.0 (August 2026) replaced the adapter's
Python extension with a C ABI shared library. Every `ai-edge-model-explorer`
release the `model-explorer` extra admits (`>=0.1.33,<0.2`) depends on the 0.2
adapter line: 0.1.33 needs `~=0.2.0` and 0.1.34 needs `~=0.2.1`. TorchLens
writes Model Explorer JSON graph collections, which the app loads through its
built-in JSON path; the adapter's converters handle other formats and are not on
TorchLens' export path.

Measured 2026-11-03 on a Linux x86_64 CPU worker in the pinned CI environment
(python 3.10, torch 2.13.0, transformers 5.18.0):

| Model Explorer | Adapter | Model Explorer test files | ResNet-18 export | GPT-2 small export | Served read-back |
|---|---|---|---|---|---|
| 0.1.33 (installed without dependencies, as the nightly contract job does) | not installed | 61 passed | pass | pass | not run |
| 0.1.33 | 0.2.0 | 61 passed | pass | pass | pass |
| 0.1.33 | 0.2.1 | 61 passed | pass | pass | not run |
| 0.1.34 | 0.2.1 | 61 passed | pass | pass | pass |

Each export check traces the model (ResNet-18 with random weights on a
1x3x224x224 input; GPT-2 small from its default config with random weights on 8
token ids), writes the file with `tl.export.model_explorer`, passes
`tl.export.validate_model_explorer_payload`, rebuilds every graph through the
vendor's `graph_builder` dataclasses (no duplicate node ids). The served
read-back starts `tl.export.model_explorer_serve` on the file, checks the health
and index routes, and reads the graph back over the server's file route equal to
the file. ResNet-18 exports 151 nodes and 158 edges in the
execution graph (about 564 KB); GPT-2 small exports 567 nodes and 638 edges
(about 1.39 MB).
