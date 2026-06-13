"""JAX backend capture tests."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest

import torchlens as tl
from torchlens.backends import BackendUnsupportedError
from torchlens.backends.jax.backend import (
    _control_parent_labels,
    _data_parent_arg_positions,
    _data_parent_labels,
)
from torchlens.intervention.types import EdgeUseRecord
from torchlens.postprocess.graph_traversal import _remove_orphan_nodes

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
lax = pytest.importorskip("jax.lax")
random = pytest.importorskip("jax.random")


pytestmark = pytest.mark.backend_jax


def _mlp(params: dict[str, Any], x: Any) -> Any:
    """Return a tiny JAX MLP output.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input array.

    Returns
    -------
    Any
        Output array.
    """

    hidden = jnp.tanh(x @ params["w1"] + params["b1"])
    return hidden @ params["w2"]


def _relu_block(params: dict[str, Any], x: Any) -> Any:
    """Return an output that lowers through JAX's library ReLU wrapper.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input array.

    Returns
    -------
    Any
        Output array.
    """

    return jax.nn.relu(x @ params["w"])


def _attention_block(params: dict[str, Any], x: Any) -> Any:
    """Return a small single-head attention output.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input sequence.

    Returns
    -------
    Any
        Attention block output.
    """

    q = x @ params["wq"]
    k = x @ params["wk"]
    v = x @ params["wv"]
    scores = (q @ jnp.swapaxes(k, -1, -2)) / jnp.sqrt(jnp.asarray(q.shape[-1], dtype=x.dtype))
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ v


def _operator_heavy(_: None, x: Any) -> Any:
    """Return an operator-heavy expression.

    Parameters
    ----------
    _
        Unused parameter tree placeholder.
    x
        Input array.

    Returns
    -------
    Any
        Combined expression result.
    """

    return ((x + 2.0) * (x - 1.0)) / jnp.maximum(x, 1.0)


def _method_spellings(_: None, x: Any) -> Any:
    """Return a result using array method spellings.

    Parameters
    ----------
    _
        Unused parameter tree placeholder.
    x
        Input array.

    Returns
    -------
    Any
        Method-spelling result.
    """

    return x.reshape((2, 3)).transpose().sum(axis=0)


def _reductions(_: None, x: Any) -> Any:
    """Return reductions over an input array.

    Parameters
    ----------
    _
        Unused parameter tree placeholder.
    x
        Input array.

    Returns
    -------
    Any
        Reduction result.
    """

    return x.sum(axis=0) + x.mean(axis=1, keepdims=True)


def _broadcasting(_: None, x: Any) -> Any:
    """Return a broadcasting-heavy expression.

    Parameters
    ----------
    _
        Unused parameter tree placeholder.
    x
        Input array.

    Returns
    -------
    Any
        Broadcast result.
    """

    return x + jnp.arange(x.shape[-1], dtype=x.dtype)


def _slicing(_: None, x: Any) -> Any:
    """Return a dynamic-slice-style expression.

    Parameters
    ----------
    _
        Unused parameter tree placeholder.
    x
        Input array.

    Returns
    -------
    Any
        Sliced result.
    """

    return x[:, 1:] * 2.0


def _einsum(_: None, x: Any) -> Any:
    """Return an einsum-lowered matrix product.

    Parameters
    ----------
    _
        Unused parameter tree placeholder.
    x
        Input array.

    Returns
    -------
    Any
        Einsum result.
    """

    return jnp.einsum("ij,jk->ik", x, x.T)


def _dtype_cast(_: None, x: Any) -> Any:
    """Return a dtype-cast expression.

    Parameters
    ----------
    _
        Unused parameter tree placeholder.
    x
        Input array.

    Returns
    -------
    Any
        Cast result.
    """

    return x.astype(jnp.float32) + 1.0


def _depthwise_conv(params: dict[str, Any], x: Any) -> Any:
    """Return a grouped convolution output.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input image batch.

    Returns
    -------
    Any
        Convolution result.
    """

    return lax.conv_general_dilated(
        x,
        params["kernel"],
        (1, 1),
        "SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=2,
    )


def _pointwise_relu(params: dict[str, Any], x: Any) -> Any:
    """Return a pointwise convolution followed by library ReLU.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input image batch.

    Returns
    -------
    Any
        Activated convolution result.
    """

    y = lax.conv_general_dilated(
        x, params["kernel"], (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC")
    )
    return jax.nn.relu(y)


def _dropout_like(_: None, key: Any, x: Any) -> Any:
    """Return explicit-key dropout-like output.

    Parameters
    ----------
    _
        Unused parameter tree placeholder.
    key
        Explicit JAX random key.
    x
        Input array.

    Returns
    -------
    Any
        Dropout-like result.
    """

    keep = random.bernoulli(key, 0.75, x.shape)
    return jnp.where(keep, x / 0.75, 0.0)


def _randint_index(_: None, key: Any, x: Any) -> Any:
    """Return an explicit-key random index selection.

    Parameters
    ----------
    _
        Unused parameter tree placeholder.
    key
        Explicit JAX random key.
    x
        Input array.

    Returns
    -------
    Any
        Selected row.
    """

    index = random.randint(key, (), 0, x.shape[0])
    return x[index]


def _layer_norm(params: dict[str, Any], x: Any) -> Any:
    """Return a layer-normalization expression.

    Parameters
    ----------
    params
        Parameter pytree.
    x
        Input array.

    Returns
    -------
    Any
        Normalized result.
    """

    centered = x - x.mean(axis=-1, keepdims=True)
    variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
    return centered * lax.rsqrt(variance + 1e-5) * params["scale"] + params["bias"]


def _one_hot_take(params: dict[str, Any], x: Any) -> Any:
    """Return one-hot matrix selection.

    Parameters
    ----------
    params
        Explicit pytree containing row indices.
    x
        Input matrix.

    Returns
    -------
    Any
        One-hot selected rows.
    """

    return jax.nn.one_hot(params["indices"], x.shape[-1]) @ x.T


def _params() -> dict[str, Any]:
    """Build a nested parameter pytree.

    Returns
    -------
    dict[str, Any]
        Parameter pytree.
    """

    return {
        "w1": jnp.ones((3, 4)),
        "b1": jnp.zeros((4,)),
        "w2": jnp.ones((4, 2)),
    }


def _nested_params() -> dict[str, Any]:
    """Build a nested parameter pytree.

    Returns
    -------
    dict[str, Any]
        Nested parameter pytree.
    """

    return {
        "encoder": {"w": jnp.ones((3, 4)), "b": jnp.zeros((4,))},
        "head": {"w": jnp.ones((4, 2))},
    }


def _attention_params() -> dict[str, Any]:
    """Build parameter leaves for the attention corpus case.

    Returns
    -------
    dict[str, Any]
        Attention parameter pytree.
    """

    return {
        "wq": jnp.eye(4, dtype=jnp.float32),
        "wk": jnp.eye(4, dtype=jnp.float32),
        "wv": jnp.eye(4, dtype=jnp.float32),
    }


def _path_token(component: object) -> tuple[str, object]:
    """Return a comparable token for a TorchLens output-path component.

    Parameters
    ----------
    component
        Output-path component.

    Returns
    -------
    tuple[str, object]
        Component type name and stored key/index value.
    """

    if hasattr(component, "key"):
        return (type(component).__name__, getattr(component, "key"))
    if hasattr(component, "index"):
        return (type(component).__name__, getattr(component, "index"))
    return (type(component).__name__, component)


def _path_tokens(path: tuple[object, ...]) -> tuple[tuple[str, object], ...]:
    """Return comparable tokens for a TorchLens output path.

    Parameters
    ----------
    path
        Output container path.

    Returns
    -------
    tuple[tuple[str, object], ...]
        Comparable path tokens.
    """

    return tuple(_path_token(component) for component in path)


def _trace_jax(model: Callable[..., Any], args: tuple[Any, ...], **kwargs: Any) -> Any:
    """Trace a JAX callable through the public API.

    Parameters
    ----------
    model
        JAX callable under test.
    args
        Public positional input tuple.
    **kwargs
        Additional public trace keyword arguments.

    Returns
    -------
    Any
        Captured JAX trace.
    """

    return tl.trace(cast(Any, model), args, backend="jax", **kwargs)


def _jax_equation_op(trace: Any, primitive: str) -> Any:
    """Return the first JAX equation op with a matching primitive name.

    Parameters
    ----------
    trace
        Captured JAX trace.
    primitive
        JAX primitive name to find.

    Returns
    -------
    Any
        Matching TorchLens op.
    """

    for op in trace.layer_list:
        if op.func_name == primitive and "jax_source_path" in op.annotations:
            return op
    raise AssertionError(f"missing JAX primitive op: {primitive}")


def _fake_raw_node(
    label: str,
    *,
    parents: list[str] | None = None,
    children: list[str] | None = None,
    is_output: bool = False,
) -> SimpleNamespace:
    """Build a minimal raw graph node for postprocess pruning tests.

    Parameters
    ----------
    label
        Raw node label.
    parents
        Parent labels.
    children
        Child labels.
    is_output
        Whether the node is an output node.

    Returns
    -------
    SimpleNamespace
        Node carrying the fields used by ``_remove_orphan_nodes``.
    """

    return SimpleNamespace(
        _label_raw=label,
        label=label,
        parents=[] if parents is None else parents,
        children=[] if children is None else children,
        is_output=is_output,
        is_orphan=False,
        has_saved_activation=False,
        out_ref=None,
        out=None,
        func_call_id=None,
    )


def test_jax_trace_captures_equation_ops_and_params() -> None:
    """JAX raw functions should produce equation-backed TorchLens traces."""

    params = _params()
    x = jnp.ones((2, 3))

    trace = _trace_jax(_mlp, (params, x))

    assert trace.backend == "jax"
    assert trace.module_identity_mode == "function_root"
    assert trace.param_source == "pytree-derived"
    assert trace.modules["self"].address == "self"
    assert set(trace.params.keys()) == {"w1", "b1", "w2"}
    assert trace.params["w1"].backend_address == "pytree:w1"
    assert trace.params["w1"].resolver_status == "metadata_only"
    primitive_names = [op.func_name for op in trace.layer_list]
    assert "dot_general" in primitive_names
    assert "tanh" in primitive_names
    assert all(op.has_saved_activation for op in trace.layer_list)
    assert trace.validate_forward_pass([])


def test_jax_trace_preserves_nested_param_paths() -> None:
    """Nested pytree parameter leaves should be addressable without raising."""

    def nested_mlp(params: dict[str, Any], x: Any) -> Any:
        """Return an MLP result from nested parameters."""

        hidden = jnp.tanh(x @ params["encoder"]["w"] + params["encoder"]["b"])
        return hidden @ params["head"]["w"]

    trace = _trace_jax(nested_mlp, (_nested_params(), jnp.ones((2, 3))))

    assert set(trace.params.keys()) == {"encoder.w", "encoder.b", "head.w"}
    assert trace.params["encoder.w"].backend_address == "pytree:encoder.w"
    assert trace.param_source == "pytree-derived"


def test_jax_trace_preserves_output_pytree_paths() -> None:
    """Structured JAX outputs should preserve output container paths."""

    def multi_output(params: dict[str, Any], x: Any) -> dict[str, Any]:
        """Return a nested output pytree."""

        hidden = jnp.tanh(x @ params["w1"] + params["b1"])
        logits = hidden @ params["w2"]
        return {"logits": logits, "hidden": (hidden, hidden + 1)}

    trace = _trace_jax(multi_output, (_params(), jnp.ones((2, 3))))
    outputs = [trace[label] for label in trace.output_layers]
    output_paths = {_path_tokens(tuple(output.container_path)) for output in outputs}

    assert len(outputs) == 3
    assert {output.multi_output_index for output in outputs} == {0, 1, 2}
    assert all(output.in_multi_output for output in outputs)
    assert output_paths == {
        (("DictKey", "hidden"), ("TupleIndex", 0)),
        (("DictKey", "hidden"), ("TupleIndex", 1)),
        (("DictKey", "logits"),),
    }
    assert trace.validate_forward_pass([])


def test_jax_trace_accepts_declared_static_argnums() -> None:
    """Declared static positional args should be excluded from dynamic jaxpr inputs."""

    def scaled(params: dict[str, Any], x: Any, scale: int) -> Any:
        """Return a statically scaled JAX result."""

        return (x @ params["w1"]) * scale

    trace = _trace_jax(
        scaled,
        (_params(), jnp.ones((2, 3)), 3),
        jax_static_argnums=2,
    )

    assert trace.jax_static_argnums == (2,)
    assert "mul" in {op.func_name for op in trace.layer_list}
    assert trace.validate_forward_pass([])


def test_jax_trace_inlines_allowlisted_pure_library_calls() -> None:
    """JAX library pure-call wrappers should inline into primitive equation ops."""

    params = {"w": jnp.ones((3, 3))}
    x = jnp.ones((2, 3))

    trace = _trace_jax(_relu_block, (params, x))

    assert "custom_jvp_call" in trace.jax_inlined_call_primitives
    assert any(capture.inlined for capture in trace.jax_equation_captures)
    assert "max" in {op.func_name for op in trace.layer_list}
    assert trace.validate_forward_pass([])


def test_jax_repeated_op_block_groups_into_passes() -> None:
    """Repeated equivalent JAX op blocks should share multi-pass layer groups."""

    def repeated_block(_: None, x: Any) -> Any:
        """Return two structurally identical add/mul blocks."""

        y = x + 1.0
        y = y * 2.0
        y = y + 1.0
        return y * 2.0

    trace = _trace_jax(
        repeated_block,
        (None, jnp.ones((2, 3), dtype=jnp.float32)),
    )
    add_ops = [op for op in trace.layer_list if op.func_name == "add"]
    mul_ops = [op for op in trace.layer_list if op.func_name == "mul"]

    assert [op.pass_index for op in add_ops] == [1, 2]
    assert [op.pass_index for op in mul_ops] == [1, 2]
    assert {op.layer_label for op in add_ops} == {add_ops[0].layer_label}
    assert {op.layer_label for op in mul_ops} == {mul_ops[0].layer_label}
    assert all(op.num_passes == 2 for op in (*add_ops, *mul_ops))
    assert add_ops[0].recurrent_ops == [op.label for op in add_ops]
    assert mul_ops[0].recurrent_ops == [op.label for op in mul_ops]
    assert trace.layer_num_calls[add_ops[0].layer_label] == 2
    assert trace.layer_num_calls[mul_ops[0].layer_label] == 2
    assert trace.validate_forward_pass([])


def test_jax_validate_public_entry_returns_real_bool() -> None:
    """Public validation should capture and validate JAX traces."""

    assert tl.validate(
        cast(Any, _mlp),
        (_params(), jnp.ones((2, 3))),
        scope="forward",
        backend="jax",
    )


def test_jax_validation_fails_when_equation_output_is_corrupted() -> None:
    """Replay validation should catch a corrupted saved equation output."""

    trace = _trace_jax(_mlp, (_params(), jnp.ones((2, 3))))
    tanh_op = _jax_equation_op(trace, "tanh")

    tanh_op._internal_set("out", tanh_op.out + jnp.asarray(0.5, dtype=tanh_op.out.dtype))

    assert trace.validate_forward_pass([], validate_metadata=False) is False


def test_jax_validation_fails_when_parent_edge_is_rewired_wrong() -> None:
    """Replay validation should catch graph parent wiring that points to the wrong payload."""

    def add_then_square(_: None, x: Any, y: Any) -> Any:
        """Return a result with same-shaped parent inputs."""

        summed = x + y
        return summed * summed

    trace = _trace_jax(
        add_then_square,
        (
            None,
            jnp.ones((2, 3), dtype=jnp.float32),
            jnp.full((2, 3), 3.0, dtype=jnp.float32),
        ),
    )
    add_op = _jax_equation_op(trace, "add")
    wrong_parent = add_op.parent_arg_positions["args"][0]

    add_op.parent_arg_positions["args"][1] = wrong_parent
    add_op.parents = [wrong_parent]

    assert trace.validate_forward_pass([], validate_metadata=False) is False


def test_jax_synthetic_control_parent_is_not_a_value_replay_parent() -> None:
    """Synthetic control edges should be topology-only for JAX validation."""

    def add_then_square(_: None, x: Any, y: Any) -> Any:
        """Return a square with an injectable control-parent fixture.

        Parameters
        ----------
        _
            Unused params placeholder.
        x
            First input array.
        y
            Second input array.

        Returns
        -------
        Any
            Squared sum.
        """

        summed = x + y
        return summed * summed

    trace = _trace_jax(
        add_then_square,
        (
            None,
            jnp.ones((2, 3), dtype=jnp.float32),
            jnp.full((2, 3), 3.0, dtype=jnp.float32),
        ),
    )
    add_op = _jax_equation_op(trace, "add")
    mul_op = _jax_equation_op(trace, "mul")
    control_parent = next(op for op in trace.layer_list if op.is_input)

    mul_op.parents.append(control_parent._label_raw)
    control_parent.children.append(mul_op._label_raw)
    mul_op._internal_set(
        "_edge_uses",
        [
            *mul_op._edge_uses,
            EdgeUseRecord(
                parent_label=control_parent._label_raw,
                child_label=mul_op._label_raw,
                arg_kind="positional",
                arg_path=(),
                view_or_copy="unknown",
                parent_func_call_id=control_parent.func_call_id,
                child_func_call_id=cast(int, mul_op.func_call_id),
                edge_use="control",
            ),
        ],
    )

    assert control_parent._label_raw in _control_parent_labels(mul_op)
    assert control_parent._label_raw not in _data_parent_labels(mul_op)
    assert _data_parent_arg_positions(mul_op) == {0: add_op.label, 1: add_op.label}
    assert trace.validate_forward_pass([], validate_metadata=False)


def test_synthetic_control_parent_is_retained_by_orphan_pruning() -> None:
    """Orphan pruning should traverse control parents as topology edges."""

    class FakeTrace:
        """Minimal raw graph holder for the pruning contract."""

        def __init__(self) -> None:
            """Initialize a tiny graph with one control-only parent.

            Returns
            -------
            None
                The raw graph fields are populated in place.
            """

            decision = _fake_raw_node("decision")
            child = _fake_raw_node("child", parents=["decision"], children=["output"])
            output = _fake_raw_node("output", parents=["child"], is_output=True)
            orphan = _fake_raw_node("orphan")
            decision.children.append("child")
            self._raw_layer_labels_list = ["decision", "child", "output", "orphan"]
            self._raw_layer_dict = OrderedDict(
                (node._label_raw, node) for node in (decision, child, output, orphan)
            )
            self.input_layers: list[str] = []
            self.output_layers = ["output"]
            self.buffer_layers: list[str] = []
            self.keep_orphans = False
            self._orphan_labels: list[str] = []

        def _batch_remove_log_entries(
            self, _orphan_entries: list[SimpleNamespace], *, remove_references: bool
        ) -> None:
            """Accept the postprocess pruning callback.

            Parameters
            ----------
            _orphan_entries
                Orphan entries selected for removal.
            remove_references
                Whether references should be scrubbed.

            Returns
            -------
            None
                This fixture does not need reference scrubbing.
            """

            return None

    fake_trace = FakeTrace()

    _remove_orphan_nodes(fake_trace)  # type: ignore[arg-type]

    assert fake_trace._raw_layer_labels_list == ["decision", "child", "output"]
    assert fake_trace._orphan_labels == ["orphan"]


def test_jax_validation_fails_when_saved_payload_is_dropped() -> None:
    """Replay validation should catch a dropped JAX equation payload."""

    trace = _trace_jax(_mlp, (_params(), jnp.ones((2, 3))))
    tanh_op = _jax_equation_op(trace, "tanh")

    tanh_op._internal_set("has_saved_activation", False)

    assert trace.validate_forward_pass([], validate_metadata=False) is False


def test_jax_trace_accepts_s0j_extended_corpus_subset() -> None:
    """Representative S0.J corpus cases should capture through public JAX tracing."""

    cases: tuple[tuple[str, Callable[..., Any], tuple[Any, ...], set[str]], ...] = (
        (
            "attention",
            _attention_block,
            (_attention_params(), jnp.ones((2, 3, 4), dtype=jnp.float32)),
            {"dot_general", "div"},
        ),
        (
            "operator_heavy",
            _operator_heavy,
            (None, jnp.linspace(1.0, 3.0, 6, dtype=jnp.float32).reshape(2, 3)),
            {"add", "mul", "div", "max"},
        ),
        (
            "method_spellings",
            _method_spellings,
            (None, jnp.arange(6, dtype=jnp.float32)),
            {"reshape", "transpose", "reduce_sum"},
        ),
        (
            "reductions",
            _reductions,
            (None, jnp.arange(6, dtype=jnp.float32).reshape(2, 3)),
            {"reduce_sum", "div"},
        ),
        (
            "broadcasting",
            _broadcasting,
            (None, jnp.ones((2, 3), dtype=jnp.float32)),
            {"add"},
        ),
        (
            "slicing",
            _slicing,
            (None, jnp.arange(8, dtype=jnp.float32).reshape(2, 4)),
            {"slice", "mul"},
        ),
        (
            "einsum",
            _einsum,
            (None, jnp.arange(6, dtype=jnp.float32).reshape(2, 3)),
            {"dot_general"},
        ),
        (
            "dtype_cast",
            _dtype_cast,
            (None, jnp.arange(4, dtype=jnp.int32)),
            {"convert_element_type", "add"},
        ),
        (
            "depthwise_conv",
            _depthwise_conv,
            (
                {"kernel": jnp.ones((3, 3, 1, 2), dtype=jnp.float32) / 9.0},
                jnp.ones((1, 4, 4, 2), dtype=jnp.float32),
            ),
            {"conv_general_dilated"},
        ),
        (
            "pointwise_conv_relu",
            _pointwise_relu,
            (
                {"kernel": jnp.ones((1, 1, 2, 3), dtype=jnp.float32) / 2.0},
                jnp.ones((1, 4, 4, 2), dtype=jnp.float32),
            ),
            {"conv_general_dilated", "max"},
        ),
        (
            "dropout_like_explicit_key",
            _dropout_like,
            (None, random.key(42), jnp.ones((2, 3), dtype=jnp.float32)),
            {"random_bits", "lt", "select_n"},
        ),
        (
            "randint_index_explicit_key",
            _randint_index,
            (
                None,
                random.PRNGKey(7),
                jnp.arange(12, dtype=jnp.float32).reshape(4, 3),
            ),
            {"random_bits", "dynamic_slice"},
        ),
        (
            "layer_norm",
            _layer_norm,
            (
                {"scale": jnp.ones((4,), dtype=jnp.float32), "bias": jnp.zeros((4,))},
                jnp.arange(8, dtype=jnp.float32).reshape(2, 4),
            ),
            {"reduce_sum", "rsqrt"},
        ),
        (
            "one_hot_take",
            _one_hot_take,
            ({"indices": jnp.asarray([0, 2, 1], dtype=jnp.int32)}, jnp.eye(4, dtype=jnp.float32)),
            {"broadcast_in_dim", "eq", "dot_general"},
        ),
    )

    for name, fn, args, expected_primitives in cases:
        trace = _trace_jax(fn, args)
        primitive_names = {op.func_name for op in trace.layer_list}

        assert expected_primitives <= primitive_names, name
        assert trace.validate_forward_pass([]), name


def test_jax_trace_rejects_save_shaping_kwargs() -> None:
    """JAX preview should reject selective-save shaping."""

    with pytest.raises(BackendUnsupportedError, match="full-save only"):
        _trace_jax(_mlp, (_params(), jnp.ones((2, 3))), layers_to_save=["tanh"])


def test_jax_trace_unrolls_scan_and_groups_body_iterations() -> None:
    """``lax.scan`` should unroll with helper nodes and recurrent body grouping."""

    def uses_scan(params: dict[str, Any], xs: Any) -> Any:
        """Return carry and stacked ys from a simple scan."""

        def body(carry: Any, x_value: Any) -> tuple[Any, Any]:
            """Return one scan body step."""

            scaled = x_value * params["scale"]
            carry_next = carry + scaled
            y_value = carry_next - params["scale"]
            return carry_next, y_value

        return lax.scan(body, params["carry0"], xs)

    trace = _trace_jax(
        uses_scan,
        (
            {"scale": jnp.asarray(2.0, dtype=jnp.float32), "carry0": jnp.asarray(0.0)},
            jnp.arange(4, dtype=jnp.float32),
        ),
    )
    scan_reads = [
        op for op in trace.layer_list if op.annotations.get("jax_capture_kind") == "scan_read"
    ]
    scan_stacks = [
        op for op in trace.layer_list if op.annotations.get("jax_capture_kind") == "scan_stack"
    ]
    body_muls = [
        op
        for op in trace.layer_list
        if op.func_name == "mul" and "/body/" in op.annotations.get("jax_source_path", "")
    ]

    assert len(scan_reads) == 4
    assert len(scan_stacks) == 1
    assert len(body_muls) == 4
    assert {op.layer_label for op in body_muls} == {body_muls[0].layer_label}
    assert [op.pass_index for op in body_muls] == [1, 2, 3, 4]
    assert {op.num_passes for op in body_muls} == {4}
    assert trace.validate_forward_pass([]) is True


def test_jax_trace_keeps_two_scan_groups_separate() -> None:
    """Two scan sites with the same body primitive should not overgroup."""

    def two_scans(params: dict[str, Any], xs: Any) -> Any:
        """Return outputs from two separate scans."""

        def body(carry: Any, x_value: Any) -> tuple[Any, Any]:
            """Return one additive body step."""

            carry_next = carry + x_value
            return carry_next, carry_next

        _carry_a, ys_a = lax.scan(body, params["carry0"], xs)
        _carry_b, ys_b = lax.scan(body, params["carry1"], xs)
        return ys_a + ys_b

    trace = _trace_jax(
        two_scans,
        (
            {"carry0": jnp.asarray(0.0), "carry1": jnp.asarray(10.0)},
            jnp.arange(3, dtype=jnp.float32),
        ),
    )
    first_scan_adds = [
        op
        for op in trace.layer_list
        if op.func_name == "add" and "root/0:scan/" in op.annotations.get("jax_source_path", "")
    ]
    second_scan_adds = [
        op
        for op in trace.layer_list
        if op.func_name == "add" and "root/1:scan/" in op.annotations.get("jax_source_path", "")
    ]

    assert len(first_scan_adds) == 3
    assert len(second_scan_adds) == 3
    assert {op.layer_label for op in first_scan_adds}.isdisjoint(
        {op.layer_label for op in second_scan_adds}
    )
    assert {op.num_passes for op in first_scan_adds} == {3}
    assert {op.num_passes for op in second_scan_adds} == {3}
    assert trace.validate_forward_pass([]) is True


def test_jax_trace_keeps_scan_body_primitive_separate_from_outside() -> None:
    """A primitive outside scan should not group with the scan body primitive."""

    def scan_then_add(params: dict[str, Any], xs: Any) -> Any:
        """Return scan output followed by an outside add."""

        def body(carry: Any, x_value: Any) -> tuple[Any, Any]:
            """Return one additive body step."""

            carry_next = carry + x_value
            return carry_next, carry_next

        _carry, ys = lax.scan(body, params["carry0"], xs)
        return ys + params["bias"]

    trace = _trace_jax(
        scan_then_add,
        (
            {"carry0": jnp.asarray(0.0), "bias": jnp.asarray(1.0)},
            jnp.arange(3, dtype=jnp.float32),
        ),
    )
    body_adds = [
        op
        for op in trace.layer_list
        if op.func_name == "add" and "/body/" in op.annotations.get("jax_source_path", "")
    ]
    outside_adds = [
        op
        for op in trace.layer_list
        if op.func_name == "add" and "/body/" not in op.annotations.get("jax_source_path", "")
    ]

    assert len(body_adds) == 3
    assert len(outside_adds) == 1
    assert {op.layer_label for op in body_adds}.isdisjoint({op.layer_label for op in outside_adds})
    assert {op.num_passes for op in body_adds} == {3}
    assert outside_adds[0].num_passes == 1
    assert trace.validate_forward_pass([]) is True


def test_jax_trace_scan_reject_policy_and_max_unroll_guard() -> None:
    """JAX scan policy controls unroll vs historical rejection and length guard."""

    def uses_scan(params: dict[str, Any], xs: Any) -> Any:
        """Return a scan output."""

        def body(carry: Any, x_value: Any) -> tuple[Any, Any]:
            """Return one additive body step."""

            carry_next = carry + x_value
            return carry_next, carry_next

        return lax.scan(body, params["carry0"], xs)

    args = ({"carry0": jnp.asarray(0.0)}, jnp.arange(4, dtype=jnp.float32))

    with pytest.raises(ValueError, match="unsupported nested primitive: scan"):
        _trace_jax(uses_scan, args, jax_control_flow="reject")
    with pytest.raises(ValueError, match="length exceeds jax_max_control_flow_unroll"):
        _trace_jax(uses_scan, args, jax_max_control_flow_unroll=3)


def test_jax_trace_rejects_unimplemented_nested_control_flow() -> None:
    """Unsupported nested jaxprs should raise actionable errors."""

    def uses_cond(params: dict[str, Any], x: Any) -> Any:
        """Return a conditional JAX result."""

        return jax.lax.cond(
            x.sum() > 0,
            lambda y: y @ params["w1"],
            lambda y: (y @ params["w1"]) - params["b1"],
            x,
        )

    with pytest.raises(ValueError, match="unsupported nested primitive: cond"):
        _trace_jax(uses_cond, (_params(), jnp.ones((2, 3))))


def test_jax_trace_rejects_hidden_consts() -> None:
    """Hidden closure constants should be rejected as undeclared leaves."""

    hidden = jnp.ones((3, 3))

    def uses_hidden(params: dict[str, Any], x: Any) -> Any:
        """Return a result using a closure constant."""

        del params
        return x @ hidden

    with pytest.raises(ValueError, match="closed-jaxpr constants"):
        _trace_jax(uses_hidden, ({}, jnp.ones((2, 3))))
