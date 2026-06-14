"""JAX backend hardening and public-surface matrix tests."""

from __future__ import annotations

import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import torchlens as tl
from torchlens._io.bundle import _build_manifest, _write_payload_blob
from torchlens._io.payload_codec import get_payload_codec
from torchlens._io.scrub import scrub_for_save
from torchlens._io.tensor_policy import FailReason, Ok
from torchlens._io.tlspec import _TlSpecWriter
from torchlens.backends import (
    BackendRuntimeCompatibilityError,
    BackendUnsupportedError,
    get_backend_spec,
)
from torchlens.backends.jax import capabilities

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
lax = pytest.importorskip("jax.lax")
random = pytest.importorskip("jax.random")


pytestmark = pytest.mark.backend_jax


def _params() -> dict[str, Any]:
    """Return deterministic JAX parameter leaves.

    Returns
    -------
    dict[str, Any]
        Parameter pytree.
    """

    return {"w": jnp.ones((3, 2), dtype=jnp.float32), "b": jnp.zeros((2,), dtype=jnp.float32)}


def _model(params: dict[str, Any], x: Any) -> Any:
    """Return a tiny JAX model output.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input array.

    Returns
    -------
    Any
        Model output.
    """

    return jnp.tanh(x @ params["w"] + params["b"])


def _identity_key(_: None, key: Any) -> Any:
    """Return a JAX PRNG key payload unchanged.

    Parameters
    ----------
    _
        Unused parameter placeholder.
    key
        JAX PRNG key payload.

    Returns
    -------
    Any
        Input key.
    """

    return key


def _identity_array(_: None, value: Any) -> Any:
    """Return a JAX array payload unchanged.

    Parameters
    ----------
    _
        Unused parameter placeholder.
    value
        JAX array payload.

    Returns
    -------
    Any
        Input array.
    """

    return value


def _split_key(_: None, key: Any) -> Any:
    """Return a batched typed-key array from a scalar JAX PRNG key.

    Parameters
    ----------
    _
        Unused parameter placeholder.
    key
        Scalar typed JAX PRNG key.

    Returns
    -------
    Any
        Split typed-key array.
    """

    return random.split(key, 4)


def _sample_key_pair(key: Any) -> Any:
    """Draw a deterministic two-value sample from a JAX PRNG key.

    Parameters
    ----------
    key
        Scalar typed JAX PRNG key.

    Returns
    -------
    Any
        Sample values drawn from ``key``.
    """

    return random.normal(key, (2,), dtype=jnp.float32)


def _named_sharded_array() -> tuple[Any, np.ndarray, int]:
    """Return a fully addressable NamedSharding array over local JAX devices.

    Returns
    -------
    tuple[Any, np.ndarray, int]
        Sharded JAX array, expected host values, and local device count.
    """

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    local_devices = list(jax.local_devices())
    device_count = len(local_devices)
    host_value = np.arange(device_count * 4, dtype=np.float32).reshape(device_count, 4)
    mesh = Mesh(np.asarray(local_devices), ("x",))
    sharding = NamedSharding(mesh, PartitionSpec("x"))
    sharded = jax.device_put(jnp.asarray(host_value), sharding)
    return sharded, host_value, device_count


def _trace(**kwargs: Any) -> Any:
    """Trace the shared tiny JAX model.

    Parameters
    ----------
    **kwargs
        Public trace keyword overrides.

    Returns
    -------
    Any
        Captured JAX trace.
    """

    return tl.trace(
        cast(Any, _model),
        (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
        backend="jax",
        **kwargs,
    )


def _write_jax_codec_fixture(path: Path) -> tuple[Any, np.ndarray]:
    """Write a materialized JAX bundle through the private codec writer path.

    Parameters
    ----------
    path
        Destination tlspec directory.

    Returns
    -------
    tuple[Any, np.ndarray]
        Source trace and expected first saved payload values.
    """

    trace = _trace()
    scrubbed_state, blob_specs, unsupported_tensors = scrub_for_save(
        trace,
        include_outs=True,
        include_grads=False,
        include_saved_args=False,
        include_rng_states=False,
        backend_name="jax",
        payload_materialization=True,
    )
    if not blob_specs:
        raise AssertionError("JAX codec fixture expected at least one saved payload.")

    path.mkdir()
    (path / "blobs").mkdir()
    codec = get_payload_codec("jax")
    tensor_entries = [
        _write_payload_blob(tmp_path=path, blob_spec=blob_spec, codec=codec)
        for blob_spec in blob_specs
    ]
    manifest = _build_manifest(
        trace=trace,
        tensor_entries=tensor_entries,
        unsupported_tensors=unsupported_tensors,
    )
    _TlSpecWriter.write_trace_manifest(
        path=path / "manifest.json",
        trace=trace,
        legacy_manifest=manifest,
        save_level="portable",
    )
    with (path / "metadata.pkl").open("wb") as handle:
        pickle.dump(scrubbed_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _mark_fixture_manifest_materialized(path)
    return trace, np.asarray(blob_specs[0].value)


def _mark_fixture_manifest_materialized(path: Path) -> None:
    """Patch a direct-writer fixture manifest to declare materialized payloads."""

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["body_format"] = "safetensors"
    manifest["payload_policy"] = {
        "policy": "array_payloads",
        "materialization_supported": True,
        "payload_kinds": sorted({entry["kind"] for entry in manifest["tensors"]}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _first_saved_op(trace: Any) -> Any:
    """Return the first op with a saved activation."""

    for op in trace.layer_list:
        if op.has_saved_activation:
            return op
    raise AssertionError("trace has no saved activation op")


def _saved_op_by_func_name(trace: Any, func_name: str) -> Any:
    """Return the first saved op with the requested function name.

    Parameters
    ----------
    trace
        Loaded or live trace to inspect.
    func_name
        Function name to match.

    Returns
    -------
    Any
        First matching saved op.
    """

    for op in trace.layer_list:
        if op.func_name == func_name and op.has_saved_activation:
            return op
    raise AssertionError(f"trace has no saved {func_name!r} activation op")


def test_jax_payload_codec_encodes_numpy_manifest_fields() -> None:
    """JAX payload codec should expose logical and transport metadata."""

    codec = get_payload_codec("jax")
    value = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

    assert codec.can_encode(value)
    assert isinstance(codec.validate_for_save(value, strict=True), Ok)
    encoded = codec.to_numpy(value)
    np.testing.assert_array_equal(encoded.array, np.asarray(value))
    assert encoded.logical_dtype == "float32"
    assert encoded.logical_device

    fields = codec.manifest_fields(value, encoded)
    assert fields["logical_backend"] == "jax"
    assert fields["codec"] == "numpy_safetensors_v1"
    assert fields["logical_dtype"] == "float32"
    assert fields["logical_device"] == encoded.logical_device
    assert fields["transport_backend"] == "safetensors.torch"
    assert fields["transport_dtype"] == "float32"
    assert isinstance(fields["codec_metadata"], dict)

    restored = codec.from_numpy(encoded.array, fields, map_location=None, strict_runtime=True)
    np.testing.assert_array_equal(np.asarray(restored), encoded.array)


def test_jax_codec_fixture_loads_eager_payloads_as_jax_arrays(tmp_path: Path) -> None:
    """Direct-writer JAX codec fixtures should load eager outs as JAX arrays."""

    path = tmp_path / "jax_codec_roundtrip.tlspec"
    _, expected = _write_jax_codec_fixture(path)

    loaded = tl.load(path)
    loaded_op = _first_saved_op(loaded)

    assert loaded.backend == "jax"
    assert isinstance(loaded_op.out, jax.Array)
    assert getattr(loaded, "payload_load_status") == "loaded_device_best_effort"
    np.testing.assert_allclose(np.asarray(loaded_op.out), expected)


def test_jax_codec_fixture_loads_lazy_refs_as_jax_arrays(tmp_path: Path) -> None:
    """Lazy JAX codec fixtures should defer payload decode until materialization."""

    path = tmp_path / "jax_codec_lazy.tlspec"
    _, expected = _write_jax_codec_fixture(path)

    loaded = tl.load(path, lazy=True)
    loaded_op = _first_saved_op(loaded)

    assert loaded_op.out is None
    assert loaded_op.out_ref is not None
    assert loaded_op.out_ref.logical_backend == "jax"
    materialized = loaded_op.out_ref.materialize()
    assert isinstance(materialized, jax.Array)
    np.testing.assert_allclose(np.asarray(materialized), expected)


def test_jax_codec_runtime_missing_loads_audit_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing JAX runtime during decode should preserve metadata and lazy refs."""

    path = tmp_path / "jax_codec_missing_runtime.tlspec"
    _write_jax_codec_fixture(path)
    codec = get_payload_codec("jax")

    def _missing_runtime(
        array: np.ndarray,
        entry: Any,
        *,
        map_location: Any,
        strict_runtime: bool,
    ) -> Any:
        """Pretend the JAX runtime is unavailable during payload decode."""

        del array, entry, map_location, strict_runtime
        raise BackendRuntimeCompatibilityError("jax runtime unavailable")

    monkeypatch.setattr(codec, "from_numpy", _missing_runtime)

    loaded = tl.load(path)
    loaded_op = _first_saved_op(loaded)

    assert getattr(loaded, "payload_load_status") == "audit_only_missing_runtime"
    assert loaded_op.has_saved_activation is True
    assert loaded_op.out is None
    assert loaded_op.out_ref is not None
    with pytest.raises(BackendRuntimeCompatibilityError, match="jax runtime unavailable"):
        loaded_op.out_ref.materialize()


def test_jax_payload_codec_rejects_unknown_prng_key_dtype_tag() -> None:
    """JAX typed PRNG key reconstruction should fail closed for unknown tags."""

    codec = get_payload_codec("jax")
    with pytest.raises(BackendRuntimeCompatibilityError, match="unknown"):
        codec.from_numpy(
            np.asarray([0, 0], dtype=np.uint32),
            {
                "logical_dtype": "key<unknown>",
                "logical_device": "TFRT_CPU_0",
                "codec_metadata": {"jax_prng_key_typed": True},
            },
            map_location=None,
            strict_runtime=True,
        )


def test_jax_typed_prng_scalar_key_tlspec_round_trips(tmp_path: Path) -> None:
    """Scalar typed JAX PRNG keys should round-trip through public tlspec I/O."""

    key = random.key(123)
    trace = tl.trace(cast(Any, _identity_key), (None, key), backend="jax")
    path = tmp_path / "jax_typed_prng_scalar.tlspec"

    trace.save(path, level="portable")
    loaded = tl.load(path)
    loaded_key = _first_saved_op(loaded).out

    assert str(loaded_key.dtype) == "key<fry>"
    assert loaded_key.shape == ()
    np.testing.assert_array_equal(
        np.asarray(random.key_data(loaded_key)),
        np.asarray(random.key_data(key)),
    )
    expected_sample = random.normal(key, (3,), dtype=jnp.float32)
    loaded_sample = random.normal(loaded_key, (3,), dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(loaded_sample), np.asarray(expected_sample))


def test_jax_typed_prng_split_key_array_tlspec_round_trips(tmp_path: Path) -> None:
    """Batched typed JAX PRNG keys should round-trip through public tlspec I/O."""

    key = random.key(123)
    trace = tl.trace(cast(Any, _split_key), (None, key), backend="jax")
    expected_keys = random.split(key, 4)
    path = tmp_path / "jax_typed_prng_split.tlspec"

    trace.save(path, level="portable")
    loaded = tl.load(path)
    loaded_keys = _saved_op_by_func_name(loaded, "random_split").out

    assert str(loaded_keys.dtype) == "key<fry>"
    assert loaded_keys.shape == (4,)
    np.testing.assert_array_equal(
        np.asarray(random.key_data(loaded_keys)),
        np.asarray(random.key_data(expected_keys)),
    )
    sample_fn = jax.vmap(_sample_key_pair)
    np.testing.assert_allclose(
        np.asarray(sample_fn(loaded_keys)),
        np.asarray(sample_fn(expected_keys)),
    )


def test_jax_old_style_prng_key_tlspec_round_trips_as_uint32_array(tmp_path: Path) -> None:
    """Old-style JAX PRNGKey arrays should remain ordinary uint32 array payloads."""

    key = random.PRNGKey(123)
    trace = tl.trace(cast(Any, _identity_key), (None, key), backend="jax")
    path = tmp_path / "jax_old_style_prng_key.tlspec"

    trace.save(path, level="portable")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    loaded = tl.load(path)
    loaded_key = _first_saved_op(loaded).out
    input_entries = [
        entry for entry in manifest["tensors"] if entry.get("label") == "input_1_1_raw:1"
    ]

    assert str(loaded_key.dtype) == "uint32"
    assert loaded_key.shape == (2,)
    np.testing.assert_array_equal(np.asarray(loaded_key), np.asarray(key))
    assert input_entries
    assert input_entries[0]["logical_dtype"] == "uint32"
    assert "jax_prng_key_typed" not in input_entries[0]["codec_metadata"]


def test_jax_sharded_array_tlspec_round_trips_value_with_audit_metadata(
    tmp_path: Path,
) -> None:
    """Fully addressable sharded JAX arrays should save/load by value."""

    sharded, expected, device_count = _named_sharded_array()
    trace = tl.trace(cast(Any, _identity_array), (None, sharded), backend="jax")
    path = tmp_path / "jax_sharded_value.tlspec"
    codec = get_payload_codec("jax")

    assert isinstance(codec.validate_for_save(sharded, strict=True), Ok)

    trace.save(path, level="portable")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    loaded = tl.load(path)
    loaded_array = _first_saved_op(loaded).out
    sharded_entries = [
        entry
        for entry in manifest["body_index"]
        if entry.get("codec_metadata", {}).get("jax_sharding_kind") == "NamedSharding"
    ]

    assert sharded_entries
    metadata = sharded_entries[0]["codec_metadata"]
    assert metadata["jax_sharding_mesh_axis_names"] == ["x"]
    assert metadata["jax_sharding_partition_spec"] == ["x"]
    assert metadata["jax_sharding_device_count"] == device_count
    assert metadata["jax_sharding_addressable_device_count"] == device_count
    assert metadata["jax_sharding_is_fully_addressable"] is True
    assert loaded.backend == "jax"
    assert isinstance(loaded_array, jax.Array)
    assert loaded_array.shape == sharded.shape
    assert loaded_array.dtype == sharded.dtype
    np.testing.assert_array_equal(np.asarray(loaded_array), expected)


def test_jax_sharded_array_default_load_does_not_recreate_sharding(
    tmp_path: Path,
) -> None:
    """Sharding metadata should be audit-only and inert during default load."""

    sharded, expected, _device_count = _named_sharded_array()
    trace = tl.trace(cast(Any, _identity_array), (None, sharded), backend="jax")
    path = tmp_path / "jax_sharded_default_load.tlspec"

    trace.save(path, level="portable")
    loaded = tl.load(path)
    loaded_array = _first_saved_op(loaded).out

    assert type(sharded.sharding).__name__ == "NamedSharding"
    assert type(loaded_array.sharding).__name__ != "NamedSharding"
    np.testing.assert_array_equal(np.asarray(loaded_array), expected)


def test_jax_payload_codec_strict_rejects_unaddressable_and_object_like_arrays() -> None:
    """JAX codec strict validation should reject unsafe array payload forms."""

    class _FakeUnaddressableJaxArray:
        """Minimal JAX-shaped value for unaddressable sharding rejection."""

        __module__ = "jax._src.array"
        dtype = "float32"
        is_fully_addressable = False
        sharding = None

    class _FakeObjectJaxArray:
        """Minimal JAX-shaped value for object dtype rejection."""

        __module__ = "jax._src.array"
        dtype = "object"
        is_fully_addressable = True
        sharding = None

    codec = get_payload_codec("jax")

    unaddressable_value = _FakeUnaddressableJaxArray()
    unaddressable = codec.validate_for_save(unaddressable_value, strict=True)
    assert isinstance(unaddressable, FailReason)
    assert "fully addressable" in unaddressable.text
    assert "multi-host or unaddressable" in unaddressable.text
    with pytest.raises(TypeError, match="fully addressable"):
        codec.to_numpy(unaddressable_value)

    object_like = codec.validate_for_save(_FakeObjectJaxArray(), strict=True)
    assert isinstance(object_like, FailReason)
    assert "object" in object_like.text


@pytest.mark.parametrize(
    ("builder", "pattern"),
    (
        (
            lambda: jax.jit(_model),
            "transformed callable.*root model",
        ),
        (
            lambda: jax.vmap(_model, in_axes=({"w": None, "b": None}, 0)),
            "transformed callable.*jax\\.vmap",
        ),
        (
            lambda: jax.grad(lambda params, x: jnp.sum(_model(params, x))),
            "transformed callable.*jax\\.grad",
        ),
    ),
)
def test_jax_rejects_transformed_callable_as_model(
    builder: Callable[[], Callable[..., Any]],
    pattern: str,
) -> None:
    """Transformed root callables should fail with raw-function guidance."""

    with pytest.raises(BackendUnsupportedError, match=pattern):
        tl.trace(
            cast(Any, builder()),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
        )


def test_jax_rejects_root_capture_inside_jit() -> None:
    """Calling TorchLens capture under a root JAX transform should fail loudly."""

    def traced_under_jit(x: Any) -> Any:
        """Attempt a nested TorchLens capture under ``jax.jit``."""

        tl.trace(cast(Any, _model), (_params(), x), backend="jax")
        return x

    with pytest.raises(BackendUnsupportedError, match="inside jax\\.jit.*concrete"):
        jax.jit(traced_under_jit)(jnp.ones((2, 3), dtype=jnp.float32))


@pytest.mark.parametrize(
    ("fn", "kwargs", "pattern"),
    (
        (
            lambda params, x: lax.scan(lambda carry, item: (carry + item, carry), x[0], x)[1],
            {"jax_control_flow": "reject"},
            "unsupported nested primitive: scan",
        ),
        (
            lambda params, x: lax.cond(
                x.sum() > 0,
                lambda y: y @ params["w"],
                lambda y: (y @ params["w"]) + params["b"],
                x,
            ),
            {"jax_control_flow": "reject"},
            "unsupported nested primitive: cond",
        ),
        (
            lambda params, x: lax.while_loop(
                lambda state: state[0] < 2,
                lambda state: (state[0] + 1, state[1] + 1),
                (0, x),
            )[1],
            {"jax_control_flow": "reject"},
            "unsupported nested primitive: while",
        ),
    ),
)
def test_jax_rejects_nested_jaxpr_primitives(
    fn: Callable[[dict[str, Any], Any], Any],
    kwargs: dict[str, Any],
    pattern: str,
) -> None:
    """Nested jaxpr control-flow primitives should name the unsupported primitive."""

    with pytest.raises(ValueError, match=pattern):
        tl.trace(
            cast(Any, fn),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
            **kwargs,
        )


def test_jax_accepts_default_unrolled_cond_and_while() -> None:
    """Default JAX control-flow policy should unroll supported cond and while primitives."""

    def uses_cond(params: dict[str, Any], x: Any) -> Any:
        """Return a conditional model output."""

        return lax.cond(
            x.sum() > 0,
            lambda y: y @ params["w"],
            lambda y: (y @ params["w"]) + params["b"],
            x,
        )

    def uses_while(params: dict[str, Any], x: Any) -> Any:
        """Return a while-loop model output."""

        return (
            lax.while_loop(
                lambda state: state[0] < 2,
                lambda state: (state[0] + 1, state[1] + 1),
                (0, x),
            )[1]
            @ params["w"]
        )

    args = (_params(), jnp.ones((2, 3), dtype=jnp.float32))
    cond_trace = tl.trace(cast(Any, uses_cond), args, backend="jax")
    while_trace = tl.trace(cast(Any, uses_while), args, backend="jax")

    assert any(
        op.annotations.get("jax_capture_kind") == "cond_decision" for op in cond_trace.layer_list
    )
    assert any(
        op.annotations.get("jax_capture_kind") == "while_decision" for op in while_trace.layer_list
    )
    assert cond_trace.validate_forward_pass([]) is True
    assert while_trace.validate_forward_pass([]) is True


def test_jax_rejects_custom_vjp_nested_primitive() -> None:
    """User custom VJP call primitives should be rejected for M1."""

    @jax.custom_vjp
    def custom_square(x: Any) -> Any:
        """Return a custom-VJP square."""

        return x * x

    def fwd(x: Any) -> tuple[Any, Any]:
        """Return custom VJP forward output and residual."""

        return custom_square(x), x

    def bwd(residual: Any, grad: Any) -> tuple[Any]:
        """Return custom VJP backward output."""

        return (2 * residual * grad,)

    custom_square.defvjp(fwd, bwd)

    def uses_custom_vjp(params: dict[str, Any], x: Any) -> Any:
        """Return output through a custom-VJP function."""

        del params
        return custom_square(x)

    with pytest.raises(ValueError, match="unsupported nested primitive: custom_vjp_call"):
        tl.trace(
            cast(Any, uses_custom_vjp),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
        )


def test_jax_rejects_nested_jit_closure_constants() -> None:
    """Nested JIT calls with closed constants should fail the recursive purity gate."""

    hidden = jnp.ones((2,), dtype=jnp.float32)

    @jax.jit
    def uses_hidden(x: Any) -> Any:
        """Return output using a hidden closed constant."""

        return x + hidden

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return output through a JIT closure."""

        del params
        return uses_hidden(x)

    with pytest.raises(ValueError, match="unsupported nested call primitive: jit"):
        tl.trace(cast(Any, model), ({}, jnp.ones((2,), dtype=jnp.float32)), backend="jax")


def test_jax_rejects_debug_print_inside_nested_jit() -> None:
    """Nested JIT calls with debug effects should remain rejected."""

    @jax.jit
    def prints_value(x: Any) -> Any:
        """Return a value after emitting a JAX debug print effect."""

        jax.debug.print("value {}", x[0])
        return x + 1.0

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return output through an effectful JIT."""

        del params
        return prints_value(x)

    with pytest.raises(ValueError, match="unsupported.*effect"):
        tl.trace(cast(Any, model), ({}, jnp.ones((2,), dtype=jnp.float32)), backend="jax")


def test_jax_rejects_donated_nested_jit_args() -> None:
    """Nested JIT calls with donated inputs should not be inlined."""

    donated_add = jax.jit(lambda value: value + 1.0, donate_argnums=(0,))

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return output through a donated-input JIT."""

        del params
        return donated_add(x)

    with pytest.raises(ValueError, match="unsupported nested call primitive: jit"):
        tl.trace(cast(Any, model), ({}, jnp.ones((2,), dtype=jnp.float32)), backend="jax")


def test_jax_rejects_explicit_sharded_nested_jit() -> None:
    """Nested JIT calls with explicit sharding topology should not be inlined."""

    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    mesh = Mesh(np.asarray(jax.devices()), ("x",))
    sharding = NamedSharding(mesh, PartitionSpec("x"))
    sharded_add = jax.jit(lambda value: value + 1.0, in_shardings=sharding, out_shardings=sharding)

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return output through an explicitly sharded JIT."""

        del params
        return sharded_add(x)

    with pytest.raises(ValueError, match="unsupported nested call primitive: jit"):
        tl.trace(cast(Any, model), ({}, jnp.ones((2,), dtype=jnp.float32)), backend="jax")


def test_jax_rejects_remat2_nested_primitive() -> None:
    """Rematerialization uses a bare Jaxpr frame and remains out of scope."""

    @jax.checkpoint
    def checkpointed_add(x: Any) -> Any:
        """Return a checkpointed value."""

        return x + 1.0

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return output through a checkpointed function."""

        del params
        return checkpointed_add(x)

    with pytest.raises(ValueError, match="unsupported nested primitive: remat2"):
        tl.trace(cast(Any, model), ({}, jnp.ones((2,), dtype=jnp.float32)), backend="jax")


def test_jax_rejects_shard_map_nested_primitive() -> None:
    """Shard-map nested jaxprs should remain rejected as distributed regions."""

    shard_map_module = pytest.importorskip("jax.experimental.shard_map")
    from jax.sharding import Mesh, PartitionSpec

    mesh = Mesh(np.asarray(jax.devices()), ("x",))
    sharded_map = shard_map_module.shard_map(
        lambda value: value + 1.0,
        mesh=mesh,
        in_specs=PartitionSpec("x"),
        out_specs=PartitionSpec("x"),
    )

    def model(params: dict[str, Any], x: Any) -> Any:
        """Return output through a shard-map boundary."""

        del params
        return sharded_map(x)

    with pytest.raises(ValueError, match="unsupported nested jaxpr in primitive: shard_map"):
        tl.trace(cast(Any, model), ({}, jnp.ones((1,), dtype=jnp.float32)), backend="jax")


def test_jax_rejects_callback_effects() -> None:
    """JAX callback effects should be rejected with effect wording."""

    def callback(value: Any) -> Any:
        """Return callback value unchanged."""

        return value

    def uses_callback(params: dict[str, Any], x: Any) -> Any:
        """Return output through ``jax.pure_callback``."""

        del params
        return jax.pure_callback(callback, jax.ShapeDtypeStruct(x.shape, x.dtype), x)

    with pytest.raises(ValueError, match="unsupported.*effect"):
        tl.trace(cast(Any, uses_callback), (_params(), jnp.ones((2, 3))), backend="jax")


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    (
        ({"layers_to_save": ["tanh"]}, "full-save only"),
        ({"lookback": 1}, "full-save only"),
        (
            {"intervene": tl.when(tl.func("tanh"), tl.zero_ablate())},
            "intervene.*predicate-time concrete values",
        ),
        ({"halt": tl.func("tanh")}, "halt.*predicate-time concrete values"),
        ({"save_grads": True}, "GradOptions"),
    ),
)
def test_jax_rejects_save_shaping_kwargs(kwargs: dict[str, Any], pattern: str) -> None:
    """Unsupported save-shaping and runtime mutation kwargs should fail before capture."""

    with pytest.raises(BackendUnsupportedError, match=pattern):
        _trace(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "pattern"),
    (
        ({"payload_policy": "full"}, "payload_policy.*not implemented"),
        ({"save_preview": True}, "save_preview.*not implemented"),
    ),
)
def test_jax_rejects_declared_future_public_options(
    kwargs: dict[str, Any],
    pattern: str,
) -> None:
    """Declared unimplemented public-option spine knobs should reject."""

    with pytest.raises(BackendUnsupportedError, match=pattern):
        _trace(**kwargs)


def test_jax_record_backend_jax_rejected() -> None:
    """Sparse ``tl.record`` should stay torch-only for backend v1."""

    with pytest.raises(BackendUnsupportedError, match="torch-only.*tl\\.trace"):
        tl.record(
            cast(Any, _model),
            (_params(), jnp.ones((2, 3), dtype=jnp.float32)),
            backend="jax",
        )


def test_jax_module_predicate_selects_function_root() -> None:
    """Module predicate capture should select ops inside the function root."""

    trace = _trace(save=tl.in_module("self"))

    assert trace.num_saved_ops == len(trace.layer_list)
    assert trace.validate_forward_pass([]) is True


def test_jax_rejects_closed_over_array_params() -> None:
    """Closed-over array parameters should be passed as explicit pytree leaves."""

    hidden = jnp.ones((3, 2), dtype=jnp.float32)

    def uses_hidden(params: dict[str, Any], x: Any) -> Any:
        """Return output from a hidden array."""

        del params
        return x @ hidden

    with pytest.raises(ValueError, match="closed-jaxpr constants.*explicit"):
        tl.trace(cast(Any, uses_hidden), ({}, jnp.ones((2, 3))), backend="jax")


def test_jax_public_surface_matrix(tmp_path: Path) -> None:
    """Assert supported and unsupported public surfaces on a real JAX trace."""

    trace = _trace()

    assert trace.backend == "jax"
    assert trace.validate_forward_pass([]) is True
    assert trace.validation_replay_status.state == "passed"
    assert trace.validation_replay_status.source == "live"
    assert isinstance(trace.summary(), str)
    assert len(trace.to_pandas()) == len(trace.layer_list)
    assert trace.modules["self"].address == "self"
    assert trace.module_identity_mode == "function_root"
    assert trace.param_source == "pytree-derived"
    assert set(trace.params.keys()) == {"w", "b"}
    assert trace.has_backward_pass is False
    assert trace.num_backward_passes == 0
    assert trace.num_backward_edges is None

    outpath = tmp_path / "jax_graph"
    dot = trace.draw(vis_outpath=str(outpath), vis_save_only=True, vis_fileformat="dot")
    assert isinstance(dot, str)

    audit_path = tmp_path / "jax_audit.tlspec"
    trace.save(audit_path, level="audit")
    loaded = tl.load(audit_path)
    assert loaded.backend == "jax"
    assert loaded.param_source == "pytree-derived"
    assert all(op.out is None for op in loaded.layer_list)

    portable_path = tmp_path / "jax_portable.tlspec"
    expected = np.asarray(trace[trace.output_layers[0]].out)
    trace.save(portable_path)
    loaded_portable = tl.load(portable_path)
    loaded_out = loaded_portable[loaded_portable.output_layers[0]].out
    assert loaded_portable.backend == "jax"
    assert getattr(loaded_portable, "payload_load_status") == "loaded_device_best_effort"
    assert isinstance(loaded_out, jax.Array)
    assert loaded_out.shape == expected.shape
    assert str(loaded_out.dtype) == str(expected.dtype)
    np.testing.assert_allclose(np.asarray(loaded_out), expected)

    loaded_status = loaded_portable.validate_forward_pass([])
    assert loaded_status is loaded_portable.validation_replay_status
    assert loaded_status.state == "unavailable"
    assert loaded_status.available is False
    assert loaded_status.reason == "loaded_trace_runtime_capture_stripped"
    assert loaded_status.payload_load_status == "loaded_device_best_effort"
    with pytest.raises(TypeError, match="not a boolean"):
        bool(loaded_status)
    with pytest.raises(BackendUnsupportedError, match="backward capture.*derived_grads"):
        trace.log_backward(cast(Any, trace[trace.output_layers[0]].out))
    with pytest.raises(ValueError, match="derived_grads"):
        _ = trace.backward_passes
    with pytest.raises(ValueError, match="derived_grads"):
        _ = trace[0].grads


def test_jax_capability_flags_match_preview_contract() -> None:
    """JAX capability exports should match the M1 preview surface."""

    spec = get_backend_spec("jax")

    assert spec.capabilities.backward_capture is False
    assert spec.capabilities.validation_replay is True
    assert spec.capabilities.fastlog is False
    assert spec.capabilities.interventions is False
    assert spec.capabilities.payload_materialization is True
    assert spec.capabilities.module_identity_modes == ("function_root", "pytree_module")
    assert spec.serialization_policy.payload_policy == "array_payloads"
    assert spec.serialization_policy.body_format == "safetensors"
    assert capabilities.supports_backward_capture is False
    assert capabilities.supports_validation_replay is True
    assert capabilities.supports_fastlog is False
    assert capabilities.supports_intervention is False
    assert capabilities.supports_payload_materialization is True
    assert capabilities.module_identity_modes == ("function_root", "pytree_module")
    assert capabilities.payload_policy == "array_payloads"
    assert capabilities.trace_options == (
        "jax_static_argnums",
        "grad_options",
        "jax_control_flow",
        "jax_max_control_flow_unroll",
        "module_identity_mode",
    )
