"""Parameter facets + head-indexed weight views (mikit item 5, D17 plumbing).

Head-indexed views over the model's projection weights, derived from the
captured graph and VERIFIED against captured payloads before anything is
handed out:

- ``w_q`` / ``w_k`` / ``w_v`` as ``[heads, d_model, d_head]`` (k/v carry the
  KV-group count under GQA -- a naive repeat would answer a different
  question), ``w_o`` as ``[q_heads, d_head, d_model]``.
- Orientation metadata: HF ``Conv1D`` stores ``[in, out]``, ``nn.Linear``
  ``[out, in]`` -- the exact confusion behind the measured 18.80 per-head
  sum-check error (fix F5); orientation is decided by MODULE CLASS, and an
  unknown class with a square weight refuses rather than guesses.
- Fused-QKV slicing evidence: the roles' facet ops are traced back to their
  projection weight; when all three share one weight, the slice boundaries
  are derived from the captured split siblings' ``multi_output_index`` order
  and recorded shapes -- never from a hardcoded q|k|v convention.
- Shared-storage identity: roles whose views alias one underlying storage
  are disclosed (tied/fused weights).

The views are lens-side COPIES of live parameter reads (``fold_norm=``-style
folding happens on copies downstream); the user's model is never rewritten.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import torch

from ..semantic.tolerances import within_reconstruction_tolerance
from ._errors import refuse

__all__ = ["HeadWeightViews", "ProjectionSource", "head_weight_views"]

#: Module classes whose 2D weight is stored [in_features, out_features].
_IN_OUT_CLASSES = frozenset({"Conv1D"})

#: Module classes whose 2D weight is stored [out_features, in_features].
_OUT_IN_CLASSES = frozenset({"Linear", "NonDynamicallyQuantizableLinear", "Linear8bitLt"})

_ANCESTRY_LIMIT = 8192


@dataclass(frozen=True)
class ProjectionSource:
    """Provenance for one projection role's weight view.

    Parameters
    ----------
    role:
        ``"q"`` / ``"k"`` / ``"v"`` / ``"o"``.
    module_address:
        Owning module of the underlying parameter.
    class_name:
        Owning module class (the orientation authority).
    orientation:
        ``"in_out"`` (Conv1D) or ``"out_in"`` (Linear).
    fused_qkv:
        Whether the role is a column slice of a fused QKV weight.
    slice_cols:
        ``(start, stop)`` column slice into the math-oriented fused weight,
        ``None`` for unfused roles.
    verification:
        ``"payload_verified"`` or ``"unavailable"`` (payloads not saved).
    """

    role: str
    module_address: str
    class_name: str
    orientation: str
    fused_qkv: bool
    slice_cols: tuple[int, int] | None
    verification: str


@dataclass(frozen=True)
class HeadWeightViews:
    """Head-indexed projection weight views for one attention module.

    ``w_q``: ``[n_q_heads, d_model, d_head]``; ``w_k`` / ``w_v``:
    ``[n_kv_heads, d_model, d_head]``; ``w_o``: ``[n_q_heads, d_head,
    d_model]``. Biases are head-indexed where the parameter exists
    (``b_o`` stays ``[d_model]`` -- it is not attributable per head; mikit
    D10 keeps it a verified remainder, never spread).
    """

    w_q: torch.Tensor
    w_k: torch.Tensor
    w_v: torch.Tensor
    w_o: torch.Tensor
    b_q: torch.Tensor | None
    b_k: torch.Tensor | None
    b_v: torch.Tensor | None
    b_o: torch.Tensor | None
    n_q_heads: int
    n_kv_heads: int
    d_head: int
    module_address: str
    sources: dict[str, ProjectionSource]
    shared_storage: tuple[tuple[str, ...], ...]


def _facet(view: Any, name: str) -> Any:
    """Read one facet from a view, refusing typed on absence."""

    try:
        return view[name]
    except KeyError:
        refuse(
            code="mi_head_geometry_unavailable",
            message=f"The attention module exposes no {name!r} facet.",
            remedy="capture with the builtin attention recipes active (import torchlens."
            "semantic) or register a recipe producing the attention facet names",
            facet=name,
        )


def _weight_params(trace: Any, attn_address: str) -> list[Any]:
    """Return 2D-weight Param records in the attention module subtree."""

    params = []
    for module in trace.modules:
        address = str(getattr(module, "address", ""))
        if address != attn_address and not address.startswith(attn_address + "."):
            continue
        for param in getattr(module, "params", ()) or ():
            if getattr(param, "name", None) == "weight" and len(param.shape) == 2:
                params.append((module, param))
    return params


def _ancestor_labels(trace: Any, start_label: str, within_address: str) -> list[str]:
    """Return ancestor op labels of ``start_label`` inside one module subtree.

    BFS over parent edges, expanding only ops whose module stack includes the
    attention module (the projection sits inside it; the walk never escapes
    into the residual stream).
    """

    seen: set[str] = set()
    order: list[str] = []
    frontier: deque[str] = deque([start_label])
    while frontier:
        if len(seen) > _ANCESTRY_LIMIT:
            break
        label = frontier.popleft()
        if label in seen:
            continue
        seen.add(label)
        try:
            op = trace.ops[label]
        except (KeyError, ValueError, RuntimeError):
            order.append(label)
            continue
        order.append(str(getattr(op, "label", label)))
        stack = [str(entry).rpartition(":")[0] or str(entry) for entry in (op.modules or ())]
        if label != start_label and not any(
            entry == within_address or entry.startswith(within_address + ".") for entry in stack
        ):
            continue
        frontier.extend(str(parent) for parent in (getattr(op, "parents", ()) or ()))
    return order


def _projection_for(
    trace: Any,
    facet_value: Any,
    attn_address: str,
    weight_params: list[Any],
) -> tuple[Any, Any, list[str]]:
    """Return (module, param, ancestry) of the projection feeding one facet."""

    spec = getattr(facet_value, "spec", None)
    home_label = getattr(spec, "home_label", None) if spec is not None else None
    if home_label is None:
        home = getattr(spec, "home", None)
        home_label = str(getattr(home, "label", ""))
    ancestry = _ancestor_labels(trace, str(home_label), attn_address)
    ancestry_set = set(ancestry)
    hits = []
    for module, param in weight_params:
        use_ops = set()
        for label in getattr(param, "used_by_ops", ()) or ():
            try:
                use_ops.add(str(trace.ops[str(label)].label))
            except (KeyError, ValueError, RuntimeError):
                use_ops.add(str(label))
        used_here = use_ops & ancestry_set
        if used_here:
            hits.append((min(ancestry.index(label) for label in used_here), module, param))
    if not hits:
        refuse(
            code="mi_projection_unresolvable",
            message=f"No 2D-weight parameter inside {attn_address!r} is an ancestor of the "
            f"facet op {home_label!r}.",
            remedy="capture with default options (the projection op must be recorded); "
            "external-kernel projections are a named frontier",
            attn_address=attn_address,
            facet_op=str(home_label),
        )
    hits.sort(key=lambda item: item[0])
    _, module, param = hits[0]
    return module, param, ancestry


def _orientation(class_name: str, weight_shape: tuple[int, int], d_model: int) -> str:
    """Return the weight orientation for a module class, refusing on doubt."""

    if class_name in _IN_OUT_CLASSES:
        return "in_out"
    if class_name in _OUT_IN_CLASSES:
        return "out_in"
    rows, cols = weight_shape
    if rows == d_model and cols != d_model:
        return "in_out"
    if cols == d_model and rows != d_model:
        return "out_in"
    refuse(
        code="mi_orientation_unknown",
        message=f"Cannot decide the weight orientation of {class_name!r} with square shape "
        f"{weight_shape} (the exact confusion behind the measured 18.80 per-head error).",
        remedy="only Conv1D ([in, out]) and Linear-family ([out, in]) classes are known; "
        "report the module class so its orientation can be pinned",
        class_name=class_name,
        weight_shape=list(weight_shape),
    )
    raise AssertionError("unreachable")


def _math_weight(param: Any, orientation: str) -> torch.Tensor:
    """Return the live weight read in math orientation ``[in, out]``."""

    value = param.value
    return value if orientation == "in_out" else value.transpose(0, 1)


def _spec_split_slice(facet_value: Any, fused_out: int) -> tuple[int, int] | None:
    """Derive the fused-QKV column slice from the facet's own transform chain.

    The attention recipe discloses its selection as a ``split`` transform
    primitive ``(n_chunks, dim, index)`` on the facet spec -- recipe-derived
    evidence, never a hardcoded q|k|v order. Equal-chunk splits only; other
    shapes fall through to the captured-split-sibling scan.
    """

    spec = getattr(facet_value, "spec", None)
    for transform in getattr(spec, "transforms", ()) or ():
        if getattr(transform, "kind", None) != "split":
            continue
        args = tuple(getattr(transform, "args", ()) or ())
        if len(args) != 3:
            return None
        n_chunks, _dim, index = (int(entry) for entry in args)
        if n_chunks <= 0 or fused_out % n_chunks != 0 or not 0 <= index < n_chunks:
            return None
        width = fused_out // n_chunks
        return (index * width, (index + 1) * width)
    return None


def _fused_slice(
    trace: Any, ancestry: list[str], weight_use_label: str, out_width: int
) -> tuple[int, int] | None:
    """Derive the column slice for one role of a fused QKV projection.

    Walks the role's ancestry from the facet op toward the weight-use op and
    reads the FIRST captured split/select on the way: split siblings order by
    ``multi_output_index`` with recorded shapes giving exact chunk widths
    (GQA's unequal chunks included). Returns ``None`` when no slicing
    evidence exists (the caller refuses).
    """

    for label in ancestry:
        if label == weight_use_label:
            break
        try:
            op = trace.ops[label]
        except (KeyError, ValueError):
            continue
        if str(getattr(op, "func_name", "")) != "split":
            continue
        siblings = [op]
        parent_labels = tuple(getattr(op, "parents", ()) or ())
        if parent_labels:
            parent = trace.ops[str(parent_labels[0])]
            for child_label in getattr(parent, "children", ()) or ():
                child = trace.ops[str(child_label)]
                if str(getattr(child, "func_name", "")) == "split" and child.label != op.label:
                    siblings.append(child)
        siblings.sort(key=lambda item: int(getattr(item, "multi_output_index", 0) or 0))
        start = 0
        for sibling in siblings:
            width = int(tuple(sibling.shape)[-1])
            if str(sibling.label) == str(op.label):
                return (start, start + width)
            start += width
    return None


def _head_view(weight: torch.Tensor, n_heads: int, d_head: int, *, out_proj: bool) -> torch.Tensor:
    """Reshape a math-oriented weight into its head-indexed view (a copy)."""

    if out_proj:
        return weight.reshape(n_heads, d_head, -1).clone()
    return weight.reshape(-1, n_heads, d_head).permute(1, 0, 2).clone()


def _saved_out(op: Any) -> torch.Tensor | None:
    """Return an op's saved output payload, or ``None``."""

    out = getattr(op, "out", None)
    return out if isinstance(out, torch.Tensor) else None


def _verify_qkv(
    trace: Any,
    facet_value: Any,
    weight_view: torch.Tensor,
    bias_view: torch.Tensor | None,
    weight_use_label: str,
) -> str:
    """Verify one role's head view: input @ W (+ b) reproduces the facet.

    Runs only when the projection input payload and the facet value are both
    readable; a failed check REFUSES (a wrong slice or orientation must never
    ride into DLA or a lowered counterfactual).
    """

    try:
        use_op = trace.ops[weight_use_label]
    except (KeyError, ValueError):
        return "unavailable"
    parent_payloads = [
        _saved_out(trace.ops[str(parent)]) for parent in (getattr(use_op, "parents", ()) or ())
    ]
    inputs = [value for value in parent_payloads if value is not None and value.dim() >= 2]
    facet_tensor = facet_value.value if hasattr(facet_value, "value") else facet_value
    if not inputs or not isinstance(facet_tensor, torch.Tensor):
        return "unavailable"
    hidden = next((value for value in inputs if value.shape[-1] == weight_view.shape[1]), None)
    if hidden is None:
        return "unavailable"
    n_heads, _, d_head = weight_view.shape
    flat = weight_view.permute(1, 0, 2).reshape(hidden.shape[-1], n_heads * d_head)
    recon = hidden @ flat
    if bias_view is not None:
        recon = recon + bias_view.reshape(-1)
    recon = recon.reshape(-1, n_heads, d_head)
    target = facet_tensor
    if target.dim() == 4 and target.shape[1] == n_heads and target.shape[3] == d_head:
        # [b, head, pos, d_head] layout: bring heads next to d_head first.
        target = target.permute(0, 2, 1, 3)
    if target.numel() != recon.numel():
        return "unavailable"
    target = target.reshape(-1, n_heads, d_head)
    if not within_reconstruction_tolerance(recon, target, reduction_length=int(hidden.shape[-1])):
        refuse(
            code="mi_weight_view_unverified",
            message="The derived head-indexed weight view does not reproduce the captured "
            "projection output (wrong orientation, slice order, or convention).",
            remedy="report the module class/architecture; the view is refused rather than "
            "served wrong",
            weight_use_op=weight_use_label,
        )
    return "payload_verified"


def _storage_groups(views: dict[str, torch.Tensor | None]) -> tuple[tuple[str, ...], ...]:
    """Group roles whose SOURCE parameters share one storage (disclosure)."""

    by_ptr: dict[int, list[str]] = {}
    for role, tensor in views.items():
        if tensor is None:
            continue
        by_ptr.setdefault(tensor.untyped_storage().data_ptr(), []).append(role)
    return tuple(tuple(sorted(group)) for group in by_ptr.values() if len(group) > 1)


def head_weight_views(trace: Any, attention: Any) -> HeadWeightViews:  # noqa: PLR0915 -- one verified derivation per role, deliberately linear and auditable
    """Derive verified head-indexed weight views for one attention module.

    Parameters
    ----------
    trace:
        A finished torchlens trace.
    attention:
        Attention module record or address (a module exposing the attention
        facet vocabulary: ``q``/``k``/``v``, head geometry).

    Returns
    -------
    HeadWeightViews
        Verified views + orientation/GQA/fusion/shared-storage metadata.
    """

    module = trace.modules[attention] if isinstance(attention, str) else attention
    address = str(getattr(module, "address", ""))
    view = module.facets
    n_q_heads = int(_facet(view, "n_q_heads"))
    n_kv_heads = int(_facet(view, "n_kv_heads"))
    d_head = int(_facet(view, "d_head"))
    weight_params = _weight_params(trace, address)
    if not weight_params:
        refuse(
            code="mi_projection_unresolvable",
            message=f"No 2D weight parameters found inside {address!r}.",
            remedy="pass the attention module record (not a parent container)",
            attn_address=address,
        )

    roles: dict[str, dict[str, Any]] = {}
    for role in ("q", "k", "v"):
        facet_value = _facet(view, role)
        proj_module, param, ancestry = _projection_for(trace, facet_value, address, weight_params)
        roles[role] = {
            "module": proj_module,
            "param": param,
            "ancestry": ancestry,
            "facet": facet_value,
        }

    weight_ids = {role: id(info["param"].value) for role, info in roles.items()}
    fused = len(set(weight_ids.values())) == 1
    d_model_hint = int(roles["q"]["param"].shape[0])

    tensors: dict[str, torch.Tensor] = {}
    biases: dict[str, torch.Tensor | None] = {}
    sources: dict[str, ProjectionSource] = {}
    source_weights: dict[str, torch.Tensor | None] = {}
    for role in ("q", "k", "v"):
        info = roles[role]
        param = info["param"]
        proj_module = info["module"]
        class_name = str(getattr(proj_module, "class_name", ""))
        n_heads_role = n_q_heads if role == "q" else n_kv_heads
        orientation = _orientation(class_name, tuple(param.shape), d_model_hint)
        math_weight = _math_weight(param, orientation)
        bias_param = next(
            (p for p in (getattr(proj_module, "params", ()) or ()) if p.name == "bias"), None
        )
        bias_full = bias_param.value if bias_param is not None else None
        weight_use = str(param.used_by_ops[0]) if getattr(param, "used_by_ops", None) else ""
        slice_cols: tuple[int, int] | None = None
        if fused:
            slice_cols = _spec_split_slice(info["facet"], int(math_weight.shape[1]))
            if slice_cols is None:
                slice_cols = _fused_slice(
                    trace, info["ancestry"], weight_use, n_heads_role * d_head
                )
            if slice_cols is None:
                refuse(
                    code="mi_qkv_slicing_unprovable",
                    message=f"The fused QKV weight's {role!r} slice has no captured "
                    "split/select evidence.",
                    remedy="capture with default options so the QKV split ops are recorded; "
                    "hardcoding a q|k|v order is exactly what this refusal prevents",
                    role=role,
                    attn_address=address,
                )
            math_role = math_weight[:, slice_cols[0] : slice_cols[1]]
            bias_role = bias_full[slice_cols[0] : slice_cols[1]] if bias_full is not None else None
        else:
            math_role = math_weight
            bias_role = bias_full
        weight_view = _head_view(math_role, n_heads_role, d_head, out_proj=False)
        bias_view = (
            bias_role.reshape(n_heads_role, d_head).clone() if bias_role is not None else None
        )
        verification = _verify_qkv(trace, info["facet"], weight_view, bias_view, weight_use)
        tensors[f"w_{role}"] = weight_view
        biases[f"b_{role}"] = bias_view
        source_weights[role] = param.value
        sources[role] = ProjectionSource(
            role=role,
            module_address=str(getattr(proj_module, "address", "")),
            class_name=class_name,
            orientation=orientation,
            fused_qkv=fused,
            slice_cols=slice_cols,
            verification=verification,
        )

    o_module, o_param, o_source = _output_projection(trace, module, address, weight_params)
    o_class = str(getattr(o_module, "class_name", ""))
    o_orientation = _orientation(o_class, tuple(o_param.shape), d_model_hint)
    o_math = _math_weight(o_param, o_orientation)  # [H*D, E]
    w_o = _head_view(o_math, n_q_heads, d_head, out_proj=True)
    o_bias_param = next(
        (p for p in (getattr(o_module, "params", ()) or ()) if p.name == "bias"), None
    )
    b_o = o_bias_param.value.clone() if o_bias_param is not None else None
    source_weights["o"] = o_param.value
    sources["o"] = ProjectionSource(
        role="o",
        module_address=str(getattr(o_module, "address", "")),
        class_name=o_class,
        orientation=o_orientation,
        fused_qkv=False,
        slice_cols=None,
        verification=o_source,
    )

    return HeadWeightViews(
        w_q=tensors["w_q"],
        w_k=tensors["w_k"],
        w_v=tensors["w_v"],
        w_o=w_o,
        b_q=biases["b_q"],
        b_k=biases["b_k"],
        b_v=biases["b_v"],
        b_o=b_o,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        module_address=address,
        sources=sources,
        shared_storage=_storage_groups(source_weights),
    )


def _output_projection(
    trace: Any, module: Any, address: str, weight_params: list[Any]
) -> tuple[Any, Any, str]:
    """Locate the output projection: the weight op on the attn_out path."""

    view = module.facets
    attn_out = _facet(view, "attn_out")
    proj_module, param, _ancestry = _projection_for(trace, attn_out, address, weight_params)
    return proj_module, param, "dataflow_anchored"
