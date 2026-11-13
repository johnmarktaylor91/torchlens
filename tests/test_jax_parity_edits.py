"""JAX parity: capture-time ``trace(intervene=...)`` edits equation outputs."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, cast

import numpy as np
import pytest

import torchlens as tl
from torchlens.backends.jax import GradOptions
from torchlens.errors import BackendUnsupportedError

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.backend_jax, pytest.mark.rare]

_W = np.asarray([[0.2, -0.4], [0.7, 0.3], [-0.5, 0.1]], dtype=np.float32)
_B = np.asarray([0.05, -0.1], dtype=np.float32)
_X = np.asarray([[1.0, -2.0, 0.5], [0.3, 0.1, -0.8]], dtype=np.float32)


@pytest.fixture(autouse=True)
def _jax_intervene_declared() -> Iterator[None]:
    """Declare ``can_intervene`` on the JAX spec for the test (the spec flip is a registry edit)."""

    spec = tl.backends.spec("jax")
    previous = spec.capabilities.can_intervene
    object.__setattr__(spec.capabilities, "can_intervene", True)
    try:
        yield
    finally:
        object.__setattr__(spec.capabilities, "can_intervene", previous)


def _model(x: Any) -> Any:
    """Return ``tanh(x @ W + b) * 2``."""

    return jnp.tanh(x @ jnp.asarray(_W) + jnp.asarray(_B)) * 2.0


def _two_tanh(x: Any) -> Any:
    """Return a model with two tanh calls."""

    return jnp.tanh(jnp.tanh(x) + 1.0)


def _out(trace: Any) -> np.ndarray:
    """Return the trace's (single) output as NumPy."""

    return np.asarray(trace[trace.output_layers[0]].out)


def _trace(model: Any, x: Any, intervene: Any, **kwargs: Any) -> Any:
    """Trace ``model`` on the JAX backend with ``intervene``."""

    return tl.trace(cast(Any, model), x, backend="jax", intervene=intervene, **kwargs)


def _edited(trace: Any) -> list[Any]:
    """Return the edited ops of a trace."""

    return [op for op in trace.ops if op.is_edited]


def _pre(x: Any) -> Any:
    """Return the tanh input of :func:`_model`."""

    return x @ jnp.asarray(_W) + jnp.asarray(_B)


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (tl.zero_ablate(), lambda x: jnp.zeros((2, 2))),
        (tl.scale(0.5), lambda x: jnp.tanh(_pre(x))),
        (tl.add(1.0), lambda x: (jnp.tanh(_pre(x)) + 1.0) * 2.0),
        (tl.clamp(low=-0.1, high=0.1), lambda x: jnp.clip(jnp.tanh(_pre(x)), -0.1, 0.1) * 2.0),
        (
            tl.mean_ablate(dim=None),
            lambda x: jnp.full((2, 2), jnp.mean(jnp.tanh(_pre(x)))) * 2.0,
        ),
        (
            tl.mean_ablate(dim=1),
            lambda x: (
                jnp.broadcast_to(jnp.mean(jnp.tanh(_pre(x)), axis=1, keepdims=True), (2, 2)) * 2.0
            ),
        ),
        (tl.replace_with(jnp.ones((2, 2))), lambda x: jnp.full((2, 2), 2.0)),
        (lambda out, *, ctx: out * 3.0, lambda x: jnp.tanh(_pre(x)) * 6.0),
    ],
)
def test_jax_intervene_helpers_edit_and_validate(action: Any, expected: Any) -> None:
    """Each supported helper (and a callable hook) edits the tanh output; replay validates."""

    x = jnp.asarray(_X)
    trace = _trace(_model, x, tl.when(tl.func("tanh"), action))
    np.testing.assert_allclose(_out(trace), np.asarray(expected(x)), rtol=1e-6, atol=1e-6)
    edited = _edited(trace)
    assert [op.func_name for op in edited] == ["tanh"]
    assert edited[0].fire_results
    assert trace.validate_forward_pass([])


def test_jax_intervene_label_selector_targets_one_site() -> None:
    """``tl.label("tanh_2")`` edits only the second tanh: simulated labels match emission."""

    x = jnp.asarray(_X)
    trace = _trace(_two_tanh, x, tl.when(tl.label("tanh_2"), tl.zero_ablate()))
    np.testing.assert_allclose(_out(trace), 0.0)
    first = _trace(_two_tanh, x, tl.when(tl.label("tanh_1"), tl.zero_ablate()))
    np.testing.assert_allclose(_out(first), np.tanh(1.0), rtol=1e-6)
    assert trace.validate_forward_pass([]) and first.validate_forward_pass([])


def test_jax_intervene_value_dependent_predicate() -> None:
    """A predicate reading concrete values (a scalar boolean) fires and edits the branch."""

    def gate(x: Any) -> Any:
        """Return ``h`` when ``sum(tanh(x)) > 0`` else ``-h``."""

        h = jnp.tanh(x)
        return jnp.where(jnp.sum(h) > 0, h, -h)

    x = jnp.abs(jnp.asarray(_X))
    seen: list[Any] = []

    def predicate(ctx: Any) -> Any:
        """Zero the scalar boolean gate when it is concretely True."""

        seen.append((ctx.is_scalar_bool, ctx.bool_value))
        return tl.zero_ablate() if ctx.is_scalar_bool and ctx.bool_value else None

    trace = _trace(gate, x, predicate)
    np.testing.assert_allclose(_out(trace), -np.tanh(np.asarray(x)), rtol=1e-6)
    assert (True, True) in seen
    assert trace.validate_forward_pass([])


def test_jax_intervene_matches_torch_twin() -> None:
    """The edited JAX output equals the torch backend's edited output on the same model."""

    torch = pytest.importorskip("torch")

    class TorchModel(torch.nn.Module):
        """Torch twin of :func:`_model`."""

        def forward(self, x: Any) -> Any:
            """Return ``tanh(x @ W + b) * 2``."""

            return torch.tanh(x @ torch.from_numpy(_W) + torch.from_numpy(_B)) * 2.0

    action = tl.when(tl.func("tanh"), tl.scale(0.25))
    jax_trace = _trace(_model, jnp.asarray(_X), action)
    torch_trace = tl.trace(TorchModel(), torch.from_numpy(_X), intervene=action)
    torch_out = torch_trace[torch_trace.output_layers[0]].out.detach().numpy()
    np.testing.assert_allclose(_out(jax_trace), torch_out, rtol=1e-5, atol=1e-6)
    assert len(_edited(jax_trace)) == len(_edited(torch_trace)) == 1


def _equinox_mlp() -> Any:
    """Return a tiny deterministic Equinox MLP."""

    eqx = pytest.importorskip("equinox")

    class Mlp(eqx.Module):
        """Two-layer Equinox MLP."""

        fc1: Any
        fc2: Any

        def __init__(self) -> None:
            """Initialize deterministic layers."""

            key1, key2 = jax.random.split(jax.random.PRNGKey(3))
            self.fc1 = eqx.nn.Linear(3, 4, key=key1)
            self.fc2 = eqx.nn.Linear(4, 2, key=key2)

        def __call__(self, x: Any) -> Any:
            """Return the MLP output."""

            return self.fc2(jax.nn.relu(self.fc1(x)))

    return Mlp()


def test_jax_intervene_in_module_on_equinox() -> None:
    """``tl.in_module("fc1")`` reaches equations inside the named_scope of an Equinox layer."""

    model = _equinox_mlp()
    x = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32)
    trace = _trace(model, x, tl.when(tl.in_module("fc1") & tl.func("add"), tl.zero_ablate()))
    np.testing.assert_allclose(_out(trace), np.asarray(model.fc2.bias), rtol=1e-6)
    assert trace.validate_forward_pass([])


def test_jax_intervene_in_module_on_nnx() -> None:
    """``tl.in_module("fc1")`` reaches equations inside a Flax NNX layer."""

    nnx = pytest.importorskip("flax.nnx")

    class Mlp(nnx.Module):
        """Two-layer NNX MLP."""

        def __init__(self) -> None:
            """Initialize deterministic layers."""

            rngs = nnx.Rngs(4)
            self.fc1 = nnx.Linear(3, 4, rngs=rngs)
            self.fc2 = nnx.Linear(4, 2, rngs=rngs)

        def __call__(self, x: Any) -> Any:
            """Return the MLP output."""

            return self.fc2(jax.nn.relu(self.fc1(x)))

    model = Mlp()
    x = jnp.asarray([0.5, -1.0, 2.0], dtype=jnp.float32)
    trace = _trace(model, x, tl.when(tl.in_module("fc1") & tl.func("add"), tl.zero_ablate()))
    bias = model.fc2.bias
    np.testing.assert_allclose(_out(trace), np.asarray(getattr(bias, "value", bias)), rtol=1e-6)
    assert trace.validate_forward_pass([])


def test_jax_intervene_zero_matches_warns() -> None:
    """A selector that matches nothing warns like torch and leaves the output unedited."""

    x = jnp.asarray(_X)
    with pytest.warns(UserWarning, match="matched zero sites"):
        trace = _trace(_model, x, tl.when(tl.func("cos"), tl.zero_ablate()))
    np.testing.assert_allclose(_out(trace), np.asarray(_model(x)), rtol=1e-6)
    assert not _edited(trace)


def _refusal_code(model: Any, intervene: Any, **kwargs: Any) -> str | None:
    """Return the typed refusal code of one intervene trace."""

    with pytest.raises((BackendUnsupportedError, tl.errors.PredicateError)) as excinfo:
        _trace(model, jnp.asarray(_X), intervene, **kwargs)
    error = cast(Any, excinfo.value)
    return getattr(error, "code", None) or error.fields.get("code")


def test_jax_intervene_typed_refusals() -> None:
    """Unsupported shapes refuse typed rather than editing silently or half-working."""

    edit = tl.zero_ablate()
    assert _refusal_code(_model, tl.when(tl.grad(tl.func("tanh")), edit)) == (
        "backward_capture_unsupported"
    )
    assert _refusal_code(_model, tl.when(tl.module("fc1"), edit)) == "trace_option_unsupported"
    assert _refusal_code(_model, lambda ctx: 42) == "predicate_return_invalid"
    assert _refusal_code(_model, tl.when(tl.func("tanh"), lambda out, *, ctx: out[:1])) == (
        "intervention_result_invalid"
    )
    assert _refusal_code(_model, tl.when(tl.func("tanh"), tl.steer(np.ones(2, np.float32)))) == (
        "trace_option_unsupported"
    )
    loss = GradOptions(params=None, loss_fn=lambda out: jnp.sum(out))
    assert _refusal_code(
        _model,
        tl.when(tl.func("tanh"), edit),
        capture=tl.options.CaptureOptions(grad_options=loss),
    ) == ("intervention_with_derived_grads_unsupported")


def test_jax_intervene_multi_output_equation_refuses() -> None:
    """Editing a multi-output equation refuses typed."""

    def sort_model(x: Any) -> Any:
        """Return both outputs of a two-operand sort."""

        keys, values = jax.lax.sort((x[0], x[1]), num_keys=1)
        return keys + values

    assert _refusal_code(sort_model, tl.when(tl.func("sort"), tl.zero_ablate())) == (
        "trace_option_unsupported"
    )


def _tanh_op(trace: Any) -> Any:
    """Return the edited tanh op."""

    return next(op for op in _edited(trace) if op.func_name == "tanh")


def test_jax_intervene_tamper_stripped_appliers_fails_validation() -> None:
    """An edited op whose capture lost its appliers fails replay validation."""

    import dataclasses

    from torchlens.backends.jax.jaxpr import JaxEquationCapture

    trace = _trace(_model, jnp.asarray(_X), tl.when(tl.func("tanh"), tl.zero_ablate()))
    captures = list(trace.jax_equation_captures)
    index = next(i for i, c in enumerate(captures) if getattr(c, "edit_appliers", ()))
    plain = {
        f.name: getattr(captures[index], f.name) for f in dataclasses.fields(JaxEquationCapture)
    }
    captures[index] = JaxEquationCapture(**plain)
    trace.jax_equation_captures = tuple(captures)
    assert not trace.validate_forward_pass([])


def test_jax_intervene_tamper_stripped_op_edit_flag_fails_validation() -> None:
    """Op-level edit stamps are load-bearing corroboration of the capture's appliers."""

    trace = _trace(_model, jnp.asarray(_X), tl.when(tl.func("tanh"), tl.zero_ablate()))
    op = _tanh_op(trace)
    op.is_edited = False
    assert not trace.validate_forward_pass([])
