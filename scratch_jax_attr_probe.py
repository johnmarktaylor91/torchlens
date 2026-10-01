import equinox as eqx
import jax
import jax.numpy as jnp

import torchlens as tl


class SimpleEquinoxMlp(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self):
        key1, key2 = jax.random.split(jax.random.PRNGKey(10))
        self.fc1 = eqx.nn.Linear(3, 4, key=key1)
        self.fc2 = eqx.nn.Linear(4, 2, key=key2)

    def __call__(self, x):
        return self.fc2(jax.nn.relu(self.fc1(x)))


model = SimpleEquinoxMlp()
trace = tl.trace(model, jnp.ones(3, dtype=jnp.float32), backend="jax")

print("modules:", [m.address for m in trace.modules])
print("fc1.layer_labels:", trace.modules["fc1"].layer_labels)
print("fc1.num_layers:", trace.modules["fc1"].num_layers)
for op in trace.layer_list:
    print(
        op.label,
        "module=",
        op.module,
        "modules=",
        op.modules,
        "module_call_stack=",
        op.module_call_stack,
    )
