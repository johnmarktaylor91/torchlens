# JAX backend

The JAX backend captures raw functions through their jaxpr, with parameter records
derived from explicit pytree leaves. Equinox and Flax NNX module roots also carry
module attribution. Static-label payload selection, portable array payloads, and
derived gradients are supported. Transformed roots and unsupported effects refuse;
supported control-flow regions disclose partial replay verification.

## Tutorial

Install the runtime first:

```bash
pip install "torchlens[jax]"
```

```python
# tutorial: executed by tests/test_jax_docs_page.py under the backend_jax marker
import jax.numpy as jnp
import torchlens as tl

def fn(params, x):
    return jnp.tanh(x @ params["w"] + params["b"])

params = {"w": jnp.ones((3, 2)), "b": jnp.zeros((2,))}
x = (params, jnp.ones((4, 3)))
model = fn
trace = tl.trace(model, x, backend="jax")
print(trace.summary())
status = trace.validation_replay_status
print(status.state)
```

## Can and cannot

| Capability | Status | Note |
|---|---|---|
| backward_capture | unsupported | True backward graphs are deferred. |
| has_validation_replay | supported | Replay distinguishes verified operations and partial regions. |
| fastlog | unsupported | Sparse recording is not declared. |
| can_intervene | unsupported | Public static-label live interventions refuse. |
| rng_replay | unsupported | RNG replay is not declared. |
| payload_materialization | supported | Portable forward and derived array payloads are materialized. |
| has_streaming | unsupported | Streaming capture refuses. |
| structure_only_capture | unsupported | Structure-only capture is not declared. |
| intermediate_derived_grads | supported | Exact intermediate gradients use a separate AD replay. |

### Known gaps

- Closed-over array constants in raw function roots are recorded as `constant` source
  ops with a saved payload (the shared constants rule); the derived-gradient path
  (`grad_options=`) still refuses them, so pass them as explicit pytree leaves there.
- Loaded traces report replay validation `unavailable`.
- The ecosystem roster is parked until Flax checkpoints have a dependency path:
  transformers 5 removed its Flax models.

## Framework requirement

- Supported range: `jax>=0.4.30`.
- Tested through: JAX `0.10.2`.
- Pinned CI version: JAX `0.10.2`, equinox `0.13.8`, flax `0.12.8`.
