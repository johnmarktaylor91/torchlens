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
