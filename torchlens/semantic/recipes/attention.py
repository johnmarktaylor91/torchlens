"""Built-in semantic recipes for attention modules.

Detection doctrine (walkthrough A-I item 3 / tviz FIX-P): whether an attention
module ran FUSED or EAGER is decided from the CAPTURED GRAPH -- a real
``scaled_dot_product_attention`` op means fused, a real
``matmul -> softmax -> matmul`` score path means eager -- corroborated by the
model's own config (``_attn_implementation``) for the no-graph-evidence case.
Class names route records to extraction bodies (which children hold q/k/v),
but NEVER decide fusedness: transformers 5.x unified the per-backend
subclasses into one class per family, so any name-keyed fused gate reports
wrong menus on current models.

Per-head facet layouts (served; ``AttentionHeadView`` slices accordingly):
``q``/``k``/``v`` are ``[batch, pos, head, d_head]``; ``scores``/``pattern``
are ``[batch, head, dst, src]``; ``z`` is ``[batch, head, pos, d_head]``;
``result`` is ``[batch, pos, head, d_model]`` (per-QUERY-head contributions to
the output projection, pre-bias, validated against the captured projection
output before anything is served).
"""

from __future__ import annotations

from typing import Any, Literal

import torch

from ..facets import AbsenceReason, Facet, FacetSpec, MissingFacet, register
from ..reconstruction import find_sdpa_op
from ._helpers import (
    add_if_present,
    attention_implementation,
    child_module,
    child_output_spec,
    config_object_value,
    config_value,
    first_input_spec,
    fused_sdpa_facet,
    fused_sdpa_pattern,
    module_op_records,
    module_output_spec,
    needs_capture,
    op_output_readable,
    parameter_spec,
    reshape_heads,
    structural,
)

# Candidate attribute/config names for head counts + head dim. These are looked up
# via config_value (which checks record attrs AND the captured custom_attributes
# snapshot) and then via the captured HF config object itself -- modern
# transformers attention modules keep head geometry ONLY on ``self.config``.
# (num_heads cannot be recovered from tensor shapes alone: a q-projection's output
# dim is num_heads * d_head, an unfactorable product, so it MUST come from a field.)
_HEAD_COUNT_NAMES = (
    "n_heads",
    "num_heads",
    "num_attention_heads",
    "n_head",
    "nhead",
    "nheads",
)
_KV_HEAD_COUNT_NAMES = ("num_key_value_heads", "n_kv_heads", "num_kv_heads", "n_kv_head")
_HEAD_DIM_NAMES = (
    "head_dim",
    "attention_head_size",
    "d_kv",
    "d_head",
    "head_size",
    "key_value_proj_dim",  # T5's attr; T5 decouples inner_dim from d_model
)
# hidden // n_heads is only a valid d_head derivation for families whose
# attention inner dim IS the hidden size; d_model-style names (T5, Whisper)
# stay OUT of this ladder because their inner_dim is decoupled and the guess
# would be silently wrong (measured: T5 d_model=64 / d_kv=16).
_HIDDEN_SIZE_NAMES = ("dim", "embed_dim", "hidden_size", "all_head_size")

#: One facet vocabulary for every attention recipe. Which entries resolve to
#: values (vs typed absences) is decided per trace from the captured graph --
#: the menu itself never depends on a class name (the dead fused-class gate).
_ATTENTION_FACETS = (
    "q",
    "k",
    "v",
    "attn_out",
    "input",
    "n_heads",
    "n_q_heads",
    "n_kv_heads",
    "d_head",
    "head",
    "scores",
    "pattern",
    "z",
    "result",
)

#: Ops that pass a value through unchanged for anchoring purposes (dtype casts,
#: layout no-ops, dropout -- whose output IS the operand the next matmul
#: consumes, in either train or eval mode).
_VALUE_PRESERVING_FUNCS = frozenset(
    {"to", "type", "type_as", "float", "half", "contiguous", "clone", "dropout"}
)

#: Batched-matrix-product ops that can realize ``pattern @ V``.
_MATMUL_FUNCS = frozenset({"matmul", "bmm", "baddbmm", "einsum", "__matmul__"})


def _with_attention_common(
    result: dict[str, Any],
    module: Any,
    n_q_heads: int | None,
    n_kv_heads: int | None,
    d_head: int | None,
) -> dict[str, Any]:
    """Attach common attention facets to a recipe result."""

    add_if_present(result, "attn_out", module_output_spec(module, "attention"))
    add_if_present(result, "input", first_input_spec(module, "attention"))
    if n_q_heads is not None:
        result["n_q_heads"] = n_q_heads
        result["n_heads"] = n_q_heads
    if n_kv_heads is not None:
        result["n_kv_heads"] = n_kv_heads
    if d_head is not None:
        result["d_head"] = d_head
    result["head"] = module.facets.head
    sdpa_op = find_sdpa_op(module)
    if sdpa_op is not None:
        # Fused, proven by the captured graph: scores/pattern/z/result are
        # checked reconstructions, read-only by design.
        add_if_present(
            result, "scores", fused_sdpa_facet(module, "scores", "attention_reconstruction")
        )
        add_if_present(result, "pattern", fused_sdpa_pattern(module))
        add_if_present(result, "z", fused_sdpa_facet(module, "z", "attention_reconstruction"))
        add_if_present(
            result, "result", fused_sdpa_facet(module, "result", "attention_reconstruction")
        )
        return result
    anchors = _eager_attention_anchors(module, n_q_heads)
    if anchors is not None:
        scores_op, pattern_op, z_op = anchors
        add_if_present(result, "scores", _op_facet_spec(scores_op, "attention_eager"))
        add_if_present(result, "pattern", _op_facet_spec(pattern_op, "attention_eager"))
        add_if_present(result, "z", _op_facet_spec(z_op, "attention_eager"))
        add_if_present(result, "result", _eager_result_spec(module, z_op))
        return result
    reason = _no_attention_evidence(module)
    for name in ("scores", "pattern", "z", "result"):
        add_if_present(result, name, reason)
    return result


def _no_attention_evidence(module: Any) -> AbsenceReason:
    """Return the teaching absence for a module with no captured score path."""

    label = getattr(module, "address", "<unknown>")
    implementation = attention_implementation(module)
    if implementation is not None and implementation not in ("eager", "sdpa"):
        return needs_capture(
            f"attention scores/pattern/z were not captured: the model declares "
            f"attn_implementation={implementation!r} at {label}, an external fused kernel "
            "whose internals TorchLens cannot observe or reconstruct. To READ "
            "pattern/scores/z, reload the model with attn_implementation='sdpa' and "
            "capture with reconstruction_ready=True; to EDIT them, reload with "
            "attn_implementation='eager' so they are real, editable ops.",
            "attn_implementation='sdpa' + reconstruction_ready=True (read) or 'eager' (write)",
        )
    declared = (
        f" (the config declares attn_implementation={implementation!r})"
        if implementation is not None
        else ""
    )
    return structural(
        f"no attention score computation was captured inside {label}{declared}: neither "
        "a fused scaled_dot_product_attention op nor an eager matmul->softmax->matmul "
        "score path appears in this module's captured graph"
    )


def _op_shape(op: Any) -> tuple[int, ...] | None:
    """Return an op's recorded output shape metadata when available."""

    shape = getattr(op, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None


def _single_softmax_pattern(module: Any, n_q_heads: int | None) -> Any | None:
    """Return the module's single shape-admissible softmax op, or ``None``."""

    ops = module_op_records(module)
    softmaxes = [op for op in ops if getattr(op, "func_name", None) == "softmax"]
    if len(softmaxes) != 1:
        return None
    pattern_op = softmaxes[0]
    pattern_shape = _op_shape(pattern_op)
    if pattern_shape is not None:
        if len(pattern_shape) < 3:
            return None
        if n_q_heads is not None and pattern_shape[-3] != n_q_heads:
            return None
    return pattern_op


def _softmax_scores_parent(
    trace: Any, pattern_op: Any, pattern_shape: tuple[int, ...] | None
) -> Any | None:
    """Return the shape-agreeing single dataflow parent of the softmax, or ``None``."""

    parents = tuple(getattr(pattern_op, "parents", ()) or ())
    if len(parents) != 1:
        return None
    try:
        scores_op = trace.ops[parents[0]]
    except (KeyError, TypeError):
        return None
    scores_shape = _op_shape(scores_op)
    if pattern_shape is not None and scores_shape is not None and scores_shape != pattern_shape:
        return None
    return scores_op


def _eager_attention_anchors(module: Any, n_q_heads: int | None) -> tuple[Any, Any, Any] | None:
    """Return ``(scores_op, pattern_op, z_op)`` for an eager attention module.

    Anchors are graph-derived, never index- or name-hardcoded: the pattern is
    the module's single softmax op, the scores are its direct dataflow parent
    (post-scale, post-mask -- whatever fed the softmax), and z is the
    matmul-family op that consumes the pattern (through value-preserving
    casts/dropout), verified by shape against the pattern. Any structural
    mismatch returns ``None`` and the caller serves a typed absence -- the
    recipe refuses rather than guessing.
    """

    trace = getattr(module, "trace", None)
    if trace is None:
        return None
    pattern_op = _single_softmax_pattern(module, n_q_heads)
    if pattern_op is None:
        return None
    pattern_shape = _op_shape(pattern_op)
    scores_op = _softmax_scores_parent(trace, pattern_op, pattern_shape)
    if scores_op is None:
        return None
    z_op = _pattern_consuming_matmul(trace, module, pattern_op, pattern_shape)
    if z_op is None:
        return None
    return (scores_op, pattern_op, z_op)


def _pattern_consuming_matmul(
    trace: Any, module: Any, pattern_op: Any, pattern_shape: tuple[int, ...] | None
) -> Any | None:
    """Return the matmul op consuming the attention pattern, or ``None``.

    Walks the pattern's dataflow children through value-preserving ops
    (casts, dropout) staying inside the module, and accepts the first
    matmul-family op whose output shape agrees with the pattern's head and
    destination dimensions (``[..., H, dst, d_head]``).
    """

    # Relation edges carry LAYER labels (no pass qualifier) while op labels are
    # pass-qualified; membership admits both spellings.
    module_labels: set[str] = set()
    for member in module_op_records(module):
        member_label = getattr(member, "label", None)
        if isinstance(member_label, str):
            module_labels.add(member_label)
            module_labels.add(member_label.split(":")[0])
    frontier = [pattern_op]
    seen: set[str] = set()
    while frontier:
        current = frontier.pop(0)
        for child_label in tuple(getattr(current, "children", ()) or ()):
            if child_label in seen or child_label not in module_labels:
                continue
            seen.add(child_label)
            try:
                child = trace.ops[child_label]
            except (KeyError, TypeError):
                continue
            func_name = getattr(child, "func_name", None)
            if func_name in _MATMUL_FUNCS:
                child_shape = _op_shape(child)
                if (
                    pattern_shape is not None
                    and child_shape is not None
                    and (
                        len(child_shape) < 3
                        or child_shape[-3] != pattern_shape[-3]
                        or child_shape[-2] != pattern_shape[-2]
                    )
                ):
                    continue
                return child
            if func_name in _VALUE_PRESERVING_FUNCS:
                frontier.append(child)
    return None


def _op_facet_spec(op: Any, recipe_id: str) -> FacetSpec | AbsenceReason:
    """Return an op-anchored facet spec, or a needs-capture absence."""

    label = getattr(op, "label", "<unknown>")
    if not op_output_readable(op):
        return needs_capture(
            f"attention op {label!r} was not saved",
            f"save=... including {label!r}",
        )
    return FacetSpec.from_home(op, home_kind="op", recipe_id=recipe_id)


_OUTPUT_PROJECTION_CHILD_NAMES = ("o_proj", "c_proj", "out_proj", "dense", "out_lin", "o")

#: Weight-storage orientation by projection class: transformers' Conv1D stores
#: ``[in, out]``; ``nn.Linear`` stores ``[out, in]``. Never guessed for other
#: classes -- a square weight cannot disambiguate, and a wrong orientation
#: produces plausibly-ranked WRONG per-head rows (mikit F5, measured 18.80
#: sum-check error on GPT-2).
_PROJECTION_ORIENTATION_BY_CLASS: dict[str, Literal["in_out", "out_in"]] = {
    "Conv1D": "in_out",
    "Linear": "out_in",
}


def _eager_result_spec(module: Any, z_op: Any) -> FacetSpec | AbsenceReason:
    """Return the computed per-head ``result`` spec for an eager module.

    ``result[b, s, h, :] = z[b, h, s, :] @ W_O[h]`` per QUERY head, pre-bias.
    No model materializes per-head outputs, so this is a computed read-only
    view on every implementation (mikit D10); the read validates the summed
    rows (+ bias) against the CAPTURED output-projection value and refuses on
    mismatch rather than serving a wrong decomposition.
    """

    projection = _output_projection_child(module)
    if projection is None:
        return structural(
            "per-head result needs an output projection child "
            f"({'/'.join(_OUTPUT_PROJECTION_CHILD_NAMES)}) inside the attention module"
        )
    weight_spec = parameter_spec(projection, "weight", "attention_eager")
    if isinstance(weight_spec, AbsenceReason):
        return weight_spec
    if not op_output_readable(z_op):
        z_label = getattr(z_op, "label", "<unknown>")
        return needs_capture(
            f"per-head result needs the attention z op {z_label!r}, which was not saved",
            f"save=... including {z_label!r}",
        )
    projection_out = module_output_spec(projection, "attention_eager")
    if isinstance(projection_out, AbsenceReason):
        return needs_capture(
            "per-head result validates against the output projection's captured "
            f"output, which is unavailable: {projection_out.detail}",
            projection_out.save_hint or "save=... including the output projection output",
        )
    bias_spec = parameter_spec(projection, "bias", "attention_eager")
    bias = bias_spec if isinstance(bias_spec, FacetSpec) else None
    z_spec = FacetSpec.from_home(z_op, home_kind="op", recipe_id="attention_eager")
    projection_class = str(getattr(projection, "class_name", "") or "")

    def _compute() -> torch.Tensor | MissingFacet:
        """Compute the checked per-head result at read time."""

        return _eager_result_checked(z_spec, weight_spec, bias, projection_out, projection_class)

    return FacetSpec.computed(_compute, recipe_id="attention_eager", recipe_version="a01")


def _output_projection_child(module: Any) -> Any | None:
    """Return the attention module's output-projection child record."""

    for name in _OUTPUT_PROJECTION_CHILD_NAMES:
        child = child_module(module, name)
        if child is not None:
            return child
    return None


def _projection_orientation(
    projection_class: str, weight: torch.Tensor, n_heads: int, d_head: int
) -> Literal["in_out", "out_in"] | None:
    """Return the weight orientation, from the class table or shape proof."""

    orientation = _PROJECTION_ORIENTATION_BY_CLASS.get(projection_class)
    if orientation is not None:
        return orientation
    if weight.ndim != 2:
        return None
    in_dim = n_heads * d_head
    first_matches = weight.shape[0] == in_dim
    second_matches = weight.shape[1] == in_dim
    if first_matches and not second_matches:
        return "in_out"
    if second_matches and not first_matches:
        return "out_in"
    return None


def _eager_result_checked(
    z_spec: FacetSpec,
    weight_spec: FacetSpec,
    bias_spec: FacetSpec | None,
    projection_out_spec: FacetSpec,
    projection_class: str,
) -> torch.Tensor | MissingFacet:
    """Compute per-head projection contributions and validate before serving."""

    z = Facet(z_spec).value
    weight = Facet(weight_spec).value
    if not isinstance(z, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return MissingFacet("result computation missing prerequisite: tensor z and weight.")
    if z.ndim < 3 or weight.ndim != 2:
        return MissingFacet(
            "result computation missing prerequisite: z rank >= 3 and a 2-D projection weight."
        )
    n_heads = z.shape[-3]
    d_head = z.shape[-1]
    orientation = _projection_orientation(projection_class, weight, n_heads, d_head)
    if orientation is None:
        return MissingFacet(
            f"result computation refused: output projection class {projection_class!r} has an "
            "unknown weight orientation and the weight shape cannot prove it (a square weight "
            "cannot disambiguate [in, out] from [out, in]; a wrong guess yields plausibly-"
            "ranked WRONG per-head rows)."
        )
    in_dim = weight.shape[0] if orientation == "in_out" else weight.shape[1]
    if in_dim != n_heads * d_head:
        return MissingFacet(
            f"result computation missing prerequisite: projection input dimension {in_dim} "
            f"does not equal heads*d_head {n_heads * d_head}."
        )
    if orientation == "in_out":
        blocks = weight.reshape(n_heads, d_head, weight.shape[1])
        einsum_expr = "...hsd,hde->...she"
    else:
        blocks = weight.reshape(weight.shape[0], n_heads, d_head)
        einsum_expr = "...hsd,ehd->...she"
    result = torch.einsum(einsum_expr, z, blocks)
    target = Facet(projection_out_spec).value
    if isinstance(target, torch.Tensor):
        summed = result.sum(dim=-2)
        if bias_spec is not None:
            bias = Facet(bias_spec).value
            if isinstance(bias, torch.Tensor):
                summed = summed + bias
        # Cancellation-aware summation-error bound, LOCAL to this identity
        # (never the shared replay tolerance table, whose denormal atol is
        # load-bearing THERE and input-dependently refuses correct sums here,
        # the mikit F4 class): both sides fold the same |z||W| products in
        # different reduction orders, so the honest per-element bound scales
        # with eps times the SUM OF PRODUCT MAGNITUDES folded -- the per-head
        # partial sums (and the target itself) can cancel internally to near
        # zero while the folded magnitudes stay large. Orientation/omission/
        # scale defects sit orders of magnitude above this bound; head
        # PERMUTATION is sum-invariant by construction and is caught by the
        # test-side head-identity oracle, never by this check.
        product_magnitude = torch.einsum(einsum_expr, z.abs(), blocks.abs()).sum(dim=-2)
        eps = torch.finfo(target.dtype).eps
        factor = 4.0 * (n_heads * d_head + 2)
        bound = factor * eps * (product_magnitude + target.abs())
        if not bool(((summed.to(target.dtype) - target).abs() <= bound).all()):
            return MissingFacet(
                "result computation validation failed: summed per-head contributions (+ bias) "
                "did not match the captured output projection value."
            )
    return result


def _attention_config(module: Any) -> tuple[int | None, int | None, int | None]:
    """Return ``(n_q_heads, n_kv_heads, d_head)`` from common HF attention configs.

    Three sources, in evidence order: the module record's own attributes, the
    captured ``custom_attributes`` snapshot (both via ``config_value``), and
    the captured HF config object -- where modern transformers keep head
    geometry exclusively (config-based detection; a missing third tier is why
    Llama-class q/k/v were served as false structural absences).
    """

    cls = getattr(module, "cls", None)

    def _lookup(*names: str) -> int | None:
        """Resolve the first int through record attrs, snapshot, then config."""

        value = config_value(module, *names)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if cls is not None:
            value = config_value(cls, *names)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        value = config_object_value(module, *names)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    n_q_heads = _lookup(*_HEAD_COUNT_NAMES)
    n_kv_heads = _lookup(*_KV_HEAD_COUNT_NAMES)
    if n_kv_heads is None:
        n_kv_heads = n_q_heads
    d_head = _lookup(*_HEAD_DIM_NAMES)
    if d_head is None:
        hidden_size = _lookup(*_HIDDEN_SIZE_NAMES)
        if hidden_size is not None and n_q_heads:
            d_head = hidden_size // n_q_heads
    return (n_q_heads, n_kv_heads, d_head)


@register(
    class_name=(
        "MultiHeadSelfAttention",
        "DistilBertSdpaAttention",
        "DistilBertFlashAttention2",
        "DistilBertSelfAttention",
    ),
    target_scope="module",
    facets=_ATTENTION_FACETS,
)
def distilbert_attention(module: Any) -> dict[str, Any]:
    """Return facets for DistilBERT attention modules.

    The class name has changed across transformers releases, so match all
    forms: ``MultiHeadSelfAttention`` (eager base, transformers 4.x), the
    per-backend ``DistilBertSdpaAttention`` / ``DistilBertFlashAttention2``
    subclasses (4.x), and ``DistilBertSelfAttention`` (transformers 5.x, one
    unified class selecting the backend at runtime). All forms project q/k/v
    through the same ``q_lin``/``k_lin``/``v_lin`` children, so one extraction
    body serves every implementation; fusedness is decided from the captured
    graph, never from these names.
    """

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(
        result,
        "q",
        reshape_heads(
            child_output_spec(module, "q_lin", "distilbert_attention"), n_q_heads, d_head
        ),
    )
    add_if_present(
        result,
        "k",
        reshape_heads(
            child_output_spec(module, "k_lin", "distilbert_attention"), n_kv_heads, d_head
        ),
    )
    add_if_present(
        result,
        "v",
        reshape_heads(
            child_output_spec(module, "v_lin", "distilbert_attention"), n_kv_heads, d_head
        ),
    )
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(
    class_name=("GPT2Attention", "GPT2SdpaAttention", "GPT2FlashAttention2"),
    target_scope="module",
    facets=_ATTENTION_FACETS,
)
def gpt2_attention(module: Any) -> dict[str, Any]:
    """Return facets for GPT-2 fused-QKV attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    c_attn_out = child_output_spec(module, "c_attn", "gpt2_attention")
    result: dict[str, Any] = {}
    if isinstance(c_attn_out, FacetSpec) and n_q_heads is not None:
        q_raw, k_raw, v_raw = c_attn_out.split(3, dim=-1)
        add_if_present(result, "q", reshape_heads(q_raw, n_q_heads, d_head))
        add_if_present(result, "k", reshape_heads(k_raw, n_kv_heads, d_head))
        add_if_present(result, "v", reshape_heads(v_raw, n_kv_heads, d_head))
    elif isinstance(c_attn_out, AbsenceReason):
        add_if_present(result, "q", c_attn_out)
        add_if_present(result, "k", c_attn_out)
        add_if_present(result, "v", c_attn_out)
    add_if_present(result, "attn_out", child_output_spec(module, "c_proj", "gpt2_attention"))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(
    class_name=("BertSelfAttention", "BertSdpaSelfAttention"),
    target_scope="module",
    facets=_ATTENTION_FACETS,
)
def bert_self_attention(module: Any) -> dict[str, Any]:
    """Return facets for BERT self-attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(
        result,
        "q",
        reshape_heads(child_output_spec(module, "query", "bert_self_attention"), n_q_heads, d_head),
    )
    add_if_present(
        result,
        "k",
        reshape_heads(child_output_spec(module, "key", "bert_self_attention"), n_kv_heads, d_head),
    )
    add_if_present(
        result,
        "v",
        reshape_heads(
            child_output_spec(module, "value", "bert_self_attention"), n_kv_heads, d_head
        ),
    )
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(
    class_name=(
        "LlamaAttention",
        "MistralAttention",
        "LlamaSdpaAttention",
        "MistralSdpaAttention",
        "LlamaFlashAttention2",
        "MistralFlashAttention2",
        "ViTAttention",
    ),
    target_scope="module",
    facets=_ATTENTION_FACETS,
)
def gqa_attention(module: Any) -> dict[str, Any]:
    """Return facets for ``q_proj``/``k_proj``/``v_proj``/``o_proj`` attention.

    Covers the transformers 5.x unified classes (``LlamaAttention`` /
    ``MistralAttention``, backend chosen at runtime), the 4.x per-backend
    subclasses, and the 5.x ``ViTAttention`` (same child layout, MHA with
    ``n_kv_heads == n_q_heads``); fusedness comes from the captured graph,
    never these names.
    """

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(
        result,
        "q",
        reshape_heads(child_output_spec(module, "q_proj", "gqa_attention"), n_q_heads, d_head),
    )
    add_if_present(
        result,
        "k",
        reshape_heads(child_output_spec(module, "k_proj", "gqa_attention"), n_kv_heads, d_head),
    )
    add_if_present(
        result,
        "v",
        reshape_heads(child_output_spec(module, "v_proj", "gqa_attention"), n_kv_heads, d_head),
    )
    add_if_present(result, "attn_out", child_output_spec(module, "o_proj", "gqa_attention"))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(class_name="T5Attention", target_scope="module", facets=_ATTENTION_FACETS)
def t5_attention(module: Any) -> dict[str, Any]:
    """Return facets for T5 attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(
        result,
        "q",
        reshape_heads(child_output_spec(module, "q", "t5_attention"), n_q_heads, d_head),
    )
    add_if_present(
        result,
        "k",
        reshape_heads(child_output_spec(module, "k", "t5_attention"), n_kv_heads, d_head),
    )
    add_if_present(
        result,
        "v",
        reshape_heads(child_output_spec(module, "v", "t5_attention"), n_kv_heads, d_head),
    )
    add_if_present(result, "attn_out", child_output_spec(module, "o", "t5_attention"))
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)


@register(
    class_name=("ViTSelfAttention", "ViTSdpaSelfAttention"),
    target_scope="module",
    facets=_ATTENTION_FACETS,
)
def vit_self_attention(module: Any) -> dict[str, Any]:
    """Return facets for ViT self-attention modules."""

    n_q_heads, n_kv_heads, d_head = _attention_config(module)
    result: dict[str, Any] = {}
    add_if_present(
        result,
        "q",
        reshape_heads(child_output_spec(module, "query", "vit_self_attention"), n_q_heads, d_head),
    )
    add_if_present(
        result,
        "k",
        reshape_heads(child_output_spec(module, "key", "vit_self_attention"), n_kv_heads, d_head),
    )
    add_if_present(
        result,
        "v",
        reshape_heads(child_output_spec(module, "value", "vit_self_attention"), n_kv_heads, d_head),
    )
    return _with_attention_common(result, module, n_q_heads, n_kv_heads, d_head)
