# S0.J-slim Findings

## Round 1 - jaxpr-first feasibility probe

Date: 2026-06-12
Branch: `capture-unification`
Probe: `.research/spikes/jax/probe_jaxpr_s0j_round1.py`
Run command:

```bash
nice -n 19 ionice -c3 /tmp/jaxtg/jax-venv/bin/python .research/spikes/jax/probe_jaxpr_s0j_round1.py --json
```

### Exact pins

The isolated venv is `/tmp/jaxtg/jax-venv`.

| Package | Version |
|---|---:|
| Python | 3.11.6 |
| jax | 0.10.1 |
| jaxlib | 0.10.1 |
| numpy | 2.4.6 |

Install command used:

```bash
nice -n 19 ionice -c3 /tmp/jaxtg/jax-venv/bin/python -m pip install -U "jax[cpu]"
```

Host note: CPU JAX was used. JAX printed that a GPU may be present but CUDA-enabled
`jaxlib` is not installed, so it fell back to CPU.

### Deliverables advanced

| Deliverable | Round 1 status | Evidence |
|---|---|---|
| (a) Closed-jaxpr derivation | PASS | `jax.make_jaxpr(fn, static_argnums=...)` over `fn(params, *inputs)` produced a `ClosedJaxpr`; declared static arg removed from dynamic invars; hidden constant surfaced through `ClosedJaxpr.consts`. |
| (b) Interpreter capture | PASS | Prototype binds each equation primitive on concrete inputs, stores per-equation inputs/outputs, and matched direct output on accepted cases. |
| (c) Replay + perturbation | PASS | Single `add` equation replayed from saved inputs; parent perturbation changed output; wrong parent wiring changed output and assertion proved validation failure. |
| (d) Nested-jaxpr rejection | PASS | `scan`, `cond`, `while`, `remat`, nested `jit`/pjit-like call, `custom_jvp`, and `custom_vjp` were detected and rejected. |
| (e) Effects scanning | PASS | `pure_callback`, `io_callback`, and `debug.callback` were detected and rejected. `pure_callback` had no closed-jaxpr effect marker on this pin, so primitive-name scanning is required. |
| (f) Real corpus acceptance | PARTIAL PASS | 13/16 useful accepted captures = 81.25%. Rejections count against the rate. |
| (g) Interpretation overhead | PASS | Toy median overhead 2.40x; MLP median overhead 3.76x. |

### Declared-call inventory

Fixture signature: `fn(params, x, scale)` with `static_argnums=(2,)`.

| Field | Value |
|---|---:|
| `ClosedJaxpr.consts` count | 1 |
| const shapes | `(2, 4)` |
| dynamic invars | 3 |
| outvars | 1 |
| equations | 5 |
| primitives | `dot_general`, `add`, `add`, `tanh`, `mul` |
| captured equations | 5 |
| replay candidate | equation 1, primitive `add`, parent shapes `(2, 4)`, `(2, 4)` |
| wrong parent wiring failed | true |
| parent perturbation detected | true |

### Nested/effect probe table

| Probe | Detected primitive(s) | Interpreter rejected |
|---|---|---:|
| `scan` | `scan` | true |
| `cond` | `cond` | true |
| `while` | `while` | true |
| `remat` | `remat2` | true |
| pjit-like nested call via `jax.jit` | `jit` | true |
| `custom_jvp` | `custom_jvp_call` | true |
| `custom_vjp` | `custom_vjp_call` | true |
| `pure_callback` | `pure_callback` | true |
| `io_callback` | `io_callback` plus `IO` effect | true |
| `debug.callback` | `debug_callback` plus `Debug` effect | true |

Scanner notes:

- Nested jaxprs appear as both `core.ClosedJaxpr` and raw `core.Jaxpr` values in
  primitive params on `jax==0.10.1`; `remat2` used the raw form.
- `lax.cond` stores branches in tuple params, so recursive param traversal is
  necessary.
- `pure_callback` did not populate `ClosedJaxpr.effects`; primitive-name scanning
  is necessary for M1 rejection.

### Corpus accepted-capture rate

Useful accepted capture rate: 13 / 16 = 81.25%.

| Case | Accepted | Equations | Consts | Notes |
|---|---:|---:|---:|---|
| `mlp` | true | 7 | 0 | Primitive `maximum` activation; accepted. |
| `cnn` | false | n/a | n/a | Rejected: `conv_general_dilated` lowered through nested `custom_jvp_call` on this pin. |
| `attention` | true | 17 | 0 | Accepted; softmax expanded into primitive equations. |
| `operator_heavy` | true | 5 | 0 | Accepted. |
| `method_spellings` | true | 3 | 0 | Accepted; method spelling lowered normally. |
| `stochastic_typed_key` | false | n/a | n/a | Rejected: random normal lowered through nested `jit`. |
| `stochastic_legacy_key` | false | n/a | n/a | Rejected: random uniform lowered through nested `jit`. |
| `pytree_multi_output` | true | 3 | 0 | Accepted; output treedef reconstructed from direct call. |
| `reductions` | true | 5 | 0 | Accepted. |
| `broadcasting` | true | 3 | 0 | Accepted. |
| `slicing` | true | 3 | 0 | Accepted. |
| `einsum` | true | 1 | 0 | Accepted as `dot_general`. |
| `nn_funcs` | true | 10 | 0 | Accepted; `gelu`/`sigmoid` expanded without nested calls. |
| `dtype_cast` | true | 2 | 0 | Accepted. |
| `static_scale` | true | 2 | 0 | Accepted with declared static arg. |
| `const_capture` | true | 1 | 1 | Accepted; const surfaced through `ClosedJaxpr.consts`. |

### Overhead

| Scale | Equations | Repeats | Direct median (s) | Interpreted median (s) | Overhead |
|---|---:|---:|---:|---:|---:|
| toy | 4 | 200 | 0.0000570104 | 0.000136792 | 2.40x |
| MLP | 7 | 100 | 0.0000695150 | 0.000261049 | 3.76x |

### Feasibility lean

Jaxpr-first looks feasible for a raw-function M1 core if M1 keeps the plan's
strict nested-jaxpr rejection rule. The interpreter path works on the pinned
version, captures payloads, and supports a real replay/perturbation validation
primitive. The main risk exposed by this round is not interpretation itself; it is
that useful raw JAX spellings can lower through nested call primitives on this pin.
Convolution and random primitives are the immediate coverage gaps to investigate
in rounds 2-3.

### Remaining work for S0.J-slim

- Decide whether random primitives nested under `jit` can be accepted as sanctioned
  boundary expansions or must remain M1 rejects despite explicit keys.
- Investigate convolution's `custom_jvp_call` lowering and whether a primitive-safe
  path exists without violating the nested rejection rule.
- Expand corpus to 20-25 cases after the stochastic/CNN decision, including more
  parameter pytrees and shape-polymorphic-looking spellings if available.
- Add an import-time compatibility probe sketch for `core.ClosedJaxpr`,
  `core.Jaxpr`, primitive binding, effects, and callback primitive names.

## Round 2 - safe pure-call inlining probe

Date: 2026-06-12
Branch: `capture-unification`
Probe: `.research/spikes/jax/probe_jaxpr_s0j_round2.py`
Run command:

```bash
nice -n 19 ionice -c3 /tmp/jaxtg/jax-venv/bin/python .research/spikes/jax/probe_jaxpr_s0j_round2.py --json
```

### Exact pins

The isolated venv remains `/tmp/jaxtg/jax-venv`.

| Package | Version |
|---|---:|
| Python | 3.11.6 |
| jax | 0.10.1 |
| jaxlib | 0.10.1 |
| numpy | 2.4.6 |

Host note: CPU JAX was used. JAX again printed that a GPU may be present but
CUDA-enabled `jaxlib` is not installed, so it fell back to CPU.

### Safe inlining policy tested

Round 2 keeps the strict round-1 interpreter semantics as the baseline and adds a
separate graph-construction interpreter that recursively expands only allowlisted
pure library-internal call primitives:

- `jit` names: `_bernoulli`, `_normal`, `_normal_real`, `_one_hot`, `_randint`,
  `_uniform`, `_where`, `clip`, `relu`.
- `custom_jvp_call`: accepted only when its `call_jaxpr` consists of allowlisted
  library `jit` calls. This accepted JAX's `jax.nn.relu` wrapper but rejected the
  user-authored `custom_jvp` fixture.
- Still rejected: `scan`, `cond`, `while`, `remat2`, user `custom_jvp`, and
  `custom_vjp`.

This is an S0.J spike allowlist, not a final M1 compatibility table.

### Deliverables advanced

| Deliverable | Round 2 status | Evidence |
|---|---|---|
| (a) Closed-jaxpr derivation | CARRIED | Reused round-1 declared-call contract and `ClosedJaxpr.consts`; expanded corpus includes const capture and one-hot const. |
| (b) Interpreter capture | ADVANCED | Added inlining interpreter that flattens accepted nested call jaxprs into captured primitive equations. |
| (c) Replay + perturbation | ADVANCED | Replayed an inlined `max` from CNN/ReLU, perturbed its parent, and wrong parent wiring changed the replay result. |
| (d) Nested-jaxpr rejection | ADVANCED | `scan`, `cond`, `while`, user `custom_jvp`, and `custom_vjp` still reject with asserted errors. |
| (e) Effects scanning | CARRIED | Round-2 interpreter still calls the round-1 effect scanner before execution. |
| (f) Real corpus acceptance | PASS | Expanded corpus accepted 23/23 = 100% under the declared-call contract with safe inlining. |
| (g) Interpretation overhead | PASS | Toy median overhead 2.86x; MLP median overhead 4.02x; CNN timing measured at 0.92x due to tiny CPU/JAX dispatch noise. |

### Replay and perturbation proof on inlined equation

| Field | Value |
|---|---|
| Fixture | CNN block with `lax.conv_general_dilated` + `jax.nn.relu` |
| Outer primitives | `conv_general_dilated`, `custom_jvp_call`, `reduce_sum`, `div` |
| Inlined calls | `custom_jvp_call`, `jit` |
| Replay candidate | inlined equation index 1, primitive `max` |
| Source path | `root` -> `1:custom_jvp_call` -> `0:jit` -> `0:max` |
| Candidate input shapes | `(1, 5, 5, 2)`, `()` |
| Parent perturbation detected | true |
| Wrong parent wiring failed | true |

### Rejection boundary table

| Probe | Outer primitive(s) | Rejection |
|---|---|---|
| `scan` | `slice`, `squeeze`, `scan` | `unsupported nested primitive: scan` |
| `cond` | `slice`, `squeeze`, `gt`, `convert_element_type`, `cond` | `unsupported nested primitive: cond` |
| `while` | `while` | `unsupported nested primitive: while` |
| user `custom_jvp` | `custom_jvp_call` | `unsupported nested call primitive: custom_jvp_call name=None` |
| `custom_vjp` | `custom_vjp_call` | `unsupported nested primitive: custom_vjp_call` |

### Corpus accepted-capture rate

Useful accepted capture rate with safe inlining: 23 / 23 = 100%.

| Case | Accepted | Captured equations | Inlined equations | Inlined calls | Notes |
|---|---:|---:|---:|---|---|
| `mlp` | true | 7 | 0 | - | Accepted baseline MLP. |
| `cnn` | true | 4 | 1 | `custom_jvp_call`, `jit` | Round-1 reject now accepted; ReLU expanded to `max`. |
| `attention` | true | 17 | 0 | - | Accepted. |
| `operator_heavy` | true | 5 | 0 | - | Accepted. |
| `method_spellings` | true | 3 | 0 | - | Accepted. |
| `stochastic_typed_key` | true | 14 | 13 | `jit` | Round-1 reject now accepted. |
| `stochastic_legacy_key` | true | 15 | 13 | `jit` | Round-1 reject now accepted. |
| `pytree_multi_output` | true | 3 | 0 | - | Accepted. |
| `reductions` | true | 5 | 0 | - | Accepted. |
| `broadcasting` | true | 3 | 0 | - | Accepted. |
| `slicing` | true | 3 | 0 | - | Accepted. |
| `einsum` | true | 1 | 0 | - | Accepted as `dot_general`. |
| `nn_funcs` | true | 10 | 0 | - | Accepted. |
| `dtype_cast` | true | 2 | 0 | - | Accepted. |
| `static_scale` | true | 2 | 0 | - | Accepted with declared static arg. |
| `const_capture` | true | 1 | 0 | - | Accepted; one `ClosedJaxpr.consts` entry. |
| `depthwise_conv` | true | 1 | 0 | - | Accepted as `conv_general_dilated`. |
| `pointwise_conv_relu` | true | 2 | 1 | `custom_jvp_call`, `jit` | ReLU expanded to `max`. |
| `dropout_like_explicit_key` | true | 18 | 17 | `jit` | Explicit-key random + `where` accepted. |
| `randint_index_explicit_key` | true | 43 | 34 | `jit` | Explicit-key integer random + helper `clip` accepted. |
| `layer_norm` | true | 15 | 0 | - | Accepted. |
| `one_hot_take` | true | 6 | 4 | `jit` | One-hot helper expanded. |
| `nested_param_mlp` | true | 4 | 0 | - | Accepted with nested pytree params. |

### Overhead

| Scale | Outer equations | Captured equations | Repeats | Direct median (s) | Interpreted median (s) | Overhead |
|---|---:|---:|---:|---:|---:|---:|
| toy | 4 | 4 | 200 | 0.0000530580 | 0.000151599 | 2.86x |
| MLP | 7 | 7 | 100 | 0.0000767636 | 0.000308278 | 4.02x |
| CNN | 4 | 4 | 50 | 0.000435303 | 0.000399090 | 0.92x |

The CNN ratio is below 1.0 on this toy CPU run and should be treated as timing
noise from tiny workloads and JAX dispatch, not as an optimization claim.

### Feasibility lean

Jaxpr-first remains feasible, and round 2 materially improves the M1 story:
conv/ReLU and explicit-key random do not require reopening the wrapper approach on
this pin. A versioned allowlist for pure library call expansion is enough for the
observed gaps, provided M1 keeps the rejection boundary explicit for control flow
and user custom derivative semantics.

### Remaining work for S0.J-slim

- Add an import-time compatibility probe sketch for the versioned allowlist:
  `ClosedJaxpr`, primitive binding, callback/effect names, and required helper
  `jit` names.
- Decide at impact gate whether S0.J closes after this round or spends the third
  budgeted round turning the spike allowlist into a draft M1 compatibility table.
