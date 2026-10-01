import warnings

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

print("module addresses:", [m.address for m in trace.modules])
print("fc1 address_children:", trace.modules["self"].address_children)
print("fc1.num_calls:", trace.modules["fc1"].num_calls)
print(
    "fc1 call keys:",
    list(trace.modules._pass_dict.keys()) if hasattr(trace.modules, "_pass_dict") else "n/a",
)
for op in trace.layer_list:
    print(op.label, "modules=", op.modules, "module=", op.module)

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    fc1_labels = trace.resolve_sites(tl.in_module("fc1"), max_fanout=8).labels()
    print("fc1_labels:", fc1_labels)
    for warning in w:
        print("WARNING:", warning.category, warning.message)
