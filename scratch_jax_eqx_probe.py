import equinox as eqx
import jax
import jax.numpy as jnp

import torchlens as tl
from torchlens.backends.jax.backend import JAXBackend


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

backend = JAXBackend()
print("equations ok:", backend._validate_jax_equations(trace))
print("regions ok:", backend._validate_jax_regions(trace))
print("validate_forward_pass([]):", trace.validate_forward_pass([]))
print("validation_replay_status:", trace.validation_replay_status)
