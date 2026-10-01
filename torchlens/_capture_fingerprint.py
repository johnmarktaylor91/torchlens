"""Capture-cache content fingerprints and instance-attribute key fragments.

Split out of ``_capture_state_helpers.py`` under the R43 file-size ratchet:
the content-hash / code-digest / attribute-fragment family that feeds the
capture-cache key. The historical import surface
(``torchlens._capture_state_helpers``) re-exports every name here.
"""

import hashlib
import itertools
import os
import types
from typing import Any

import torch
from torch import nn

from . import _state
from ._transport import digest_byte_view


def _iter_tensor_inputs(obj: Any) -> list[torch.Tensor]:
    """Collect tensor leaves from a nested input object.

    Parameters
    ----------
    obj:
        Arbitrary nested input object.

    Returns
    -------
    list[torch.Tensor]
        Tensor leaves in traversal order.
    """

    tensors: list[torch.Tensor] = []
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, dict):
        for key in sorted(obj.keys(), key=repr):
            tensors.extend(_iter_tensor_inputs(obj[key]))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors.extend(_iter_tensor_inputs(item))
    return tensors


def _hash_tensor_content(tensor: torch.Tensor) -> str:
    """Return a content hash for a tensor.

    Parameters
    ----------
    tensor:
        Tensor to hash.

    Returns
    -------
    str
        SHA-256 digest over tensor metadata and CPU bytes. Metadata includes
        the ORIGINAL tensor's device and ``requires_grad`` flag: a CPU->CUDA
        move or a freeze between ``cache=True`` runs changes ``device_ref``,
        timing/memory, and grad_fn metadata on the capture, so it must be a
        cache miss even though the bytes match.

        The digest frames the LOGICAL dtype, captured before the
        bf16 -> float32 transport upcast numpy requires: framing the
        post-upcast dtype made a bfloat16 tensor collide with the float32
        tensor of the same values, so ``cache=True`` could serve the WRONG
        trace across dtypes (same fix as ``op.py::_tensor_content_hash``).
    """

    with _state.pause_logging():
        # Frame the LOGICAL dtype (b5-opus-R35-1 twin; same rule as the
        # op.py dedup digest) so a bfloat16 input can never hash identically
        # to the float32 tensor of the same values -- a dtype change must be
        # a capture-cache MISS. r7 R35 (fable): the old bf16->f32 transport
        # upcast copy is GONE -- the shared byte view transports bf16 (and
        # float8 friends) natively, and this digest is process-local cache
        # keying, so the byte change is invisible. r8 R35: the transport +
        # uint8 reinterpret live in ONE authority (``_transport``), which
        # also resolves lazy conj/neg bits -- the hand-rolled view here
        # crashed on ``x.conj()`` inputs.
        shape = tuple(tensor.shape)
        logical_dtype = str(tensor.dtype)
        payload = digest_byte_view(tensor)
    hasher = hashlib.sha256()
    hasher.update(
        repr(
            (
                shape,
                logical_dtype,
                str(tensor.device),
                bool(tensor.requires_grad),
            )
        ).encode("utf-8")
    )
    hasher.update(payload)
    return hasher.hexdigest()


def _hash_input_tensor_value(tensor: torch.Tensor) -> str:
    """Return a VALUE-LEVEL digest of one input tensor.

    Frames the shape, the LOGICAL dtype (pre-transport, so a bfloat16 input
    can never collide with the float32 tensor of the same values), and the
    raw payload bytes. Unlike :func:`_hash_tensor_content` (a capture-cache
    key), this digest deliberately EXCLUDES device placement and
    ``requires_grad``: it backs the Bundle comparison gate's value-level
    input-identity predicate (A-GATE), where "the same input values" must
    compare equal across a save/load round trip or a device move. The
    persisted ``input_digest`` carrier (A-GATE/digest, after C07-X) reuses
    this exact derivation, so live-derived and persisted digests stay
    comparable.

    Parameters
    ----------
    tensor:
        Input tensor to digest.

    Returns
    -------
    str
        SHA-256 hex digest over ``(shape, logical_dtype)`` plus payload bytes.
    """

    with _state.pause_logging():
        shape = tuple(tensor.shape)
        logical_dtype = str(tensor.dtype)
        payload = digest_byte_view(tensor)
    hasher = hashlib.sha256()
    hasher.update(repr((shape, logical_dtype)).encode("utf-8"))
    hasher.update(payload)
    return hasher.hexdigest()


_INPUT_FRAGMENT_DEPTH_CEILING = 64


def _forward_input_fragment(value: Any, depth: int = 0) -> object:
    """Return a full-structure key fragment for the forward inputs.

    The capture-cache key used to reduce inputs to their TENSOR leaves only
    (the deleted ``_hash_nested_tensor_content``), so a changed non-tensor
    forward input (``use_relu=False`` after a ``use_relu=True`` capture), a
    changed kwarg NAME over the same tensor value, or a restructured input
    container keyed identically to the original capture and ``cache=True``
    served the WRONG trace (r8 b4 R39, opus end-to-end repro). This walker
    frames the FULL nested structure: containers recursively (dict keys AND
    values, sequences in order, sets order-insensitively), tensors by
    content hash, and every non-container non-tensor leaf through
    :func:`_attribute_state_fragment` (scalars by value, callables by code
    digest, opaque objects by type identity -- the documented boundary).

    Unlike attribute fragments there is NO item ceiling: nothing is
    truncated -- every element is hashed, so a tail difference cannot
    false-hit -- and the depth ceiling is generous (forward inputs
    legitimately nest deeper than instance attributes). Past the ceiling
    the fragment never matches (always-miss, never a truncated false hit).
    """

    if depth > _INPUT_FRAGMENT_DEPTH_CEILING:
        return _never_matching_fragment("input-depth-ceiling")
    if isinstance(value, torch.Tensor):
        # The attribute reducer's tensor branch is exactly the contract
        # needed here (content hash; meta by metadata; unhashable content
        # mints never-matching); it does not recurse, so the fresh depth
        # budget is irrelevant.
        return _attribute_state_fragment(value)
    # The concrete container type participates (unlike attribute fragments):
    # ``forward(x, [1, 2])`` and ``forward(x, (1, 2))`` can branch on
    # isinstance and trace different programs, so they must key apart.
    if isinstance(value, dict):
        return (
            "dict",
            type(value).__name__,
            len(value),
            tuple(
                sorted(
                    (
                        repr(_forward_input_fragment(key, depth + 1)),
                        repr(_forward_input_fragment(item, depth + 1)),
                    )
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        return (
            "sequence",
            type(value).__name__,
            len(value),
            tuple(_forward_input_fragment(item, depth + 1) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        return (
            "set",
            type(value).__name__,
            len(value),
            tuple(sorted(repr(_forward_input_fragment(item, depth + 1)) for item in value)),
        )
    return _attribute_state_fragment(value)


def _fingerprint_model_content(model: nn.Module) -> str:
    """Fingerprint model tensor contents for the capture cache.

    Parameters
    ----------
    model:
        Model to fingerprint.

    Returns
    -------
    str
        SHA-256 digest.
    """

    hasher = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        hasher.update(name.encode("utf-8"))
        hasher.update(_hash_tensor_content(tensor).encode("utf-8"))
    # ``state_dict()`` detaches, so a live parameter's requires_grad flag never
    # reaches the tensor hash: fold the flags explicitly (freezing params
    # changes captured grad metadata and must be a cache miss).
    for name, parameter in model.named_parameters():
        hasher.update(repr((name, bool(parameter.requires_grad))).encode("utf-8"))
    for module_name, module in model.named_modules():
        hasher.update(repr((module_name, bool(module.training))).encode("utf-8"))
        for buffer_name in sorted(module._non_persistent_buffers_set):
            buffer = module._buffers.get(buffer_name)
            hasher.update(repr((module_name, buffer_name)).encode("utf-8"))
            if isinstance(buffer, torch.Tensor):
                hasher.update(_hash_tensor_content(buffer).encode("utf-8"))
            else:
                hasher.update(repr(buffer).encode("utf-8"))
    return hasher.hexdigest()


def _hash_code_object_into(hasher: Any, code: types.CodeType, depth: int = 0) -> None:
    """Fold one code object's behavioral surface into ``hasher``.

    Covers the compiled bytecode, referenced names, and constants (recursing
    into nested code objects such as comprehensions and local closures), which
    is what changes when a ``forward`` implementation is edited. Line-number
    tables and file paths are deliberately excluded so moving a file or adding
    a comment does not invalidate the cache.
    """

    if depth > 8:
        hasher.update(b"<code-depth-ceiling>")
        return
    hasher.update(code.co_code)
    hasher.update(repr(code.co_names).encode("utf-8"))
    hasher.update(repr(code.co_varnames).encode("utf-8"))
    hasher.update(repr(code.co_freevars).encode("utf-8"))
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            _hash_code_object_into(hasher, const, depth + 1)
        else:
            hasher.update(repr(const).encode("utf-8"))


def _callable_code_digest(func: Any, depth: int = 0) -> str:
    """Digest one callable's implementation for the capture-cache key.

    Plain functions (and bound/unbound methods) hash their code object,
    default-argument reprs, and CLOSURE cell contents. Two factory-built
    forwards share one code object while their closure cells configure
    DIFFERENT traced programs (``def make(k): def f(x): return x * k``), so
    omitting the cells collided them on one key and ``cache=True`` served
    the WRONG activations (r8 b4 R39, fable probe). Cell values reduce
    through :func:`_attribute_state_fragment` under the shared depth
    ceiling; an unhashable cell mints a never-matching fragment there
    (always-miss, never a false hit), and a not-yet-filled cell hashes a
    stable empty token (two digests taken while the cell is unfilled are
    indistinguishable by construction). Callables without a Python code
    object (C builtins, scripted callables) fall back to a stable
    module/qualname token -- NEVER ``repr(func)``, whose memory address
    would break cross-process key stability.
    """

    hasher = hashlib.sha256()
    target = getattr(func, "__func__", func)
    code = getattr(target, "__code__", None)
    if code is not None and isinstance(code, types.CodeType):
        _hash_code_object_into(hasher, code)
        for attribute in ("__defaults__", "__kwdefaults__"):
            try:
                hasher.update(repr(getattr(target, attribute, None)).encode("utf-8"))
            except Exception:
                hasher.update(f"<unreprable-{attribute}>".encode())
        closure = getattr(target, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    contents = cell.cell_contents
                except ValueError:
                    hasher.update(b"<empty-cell>")
                    continue
                hasher.update(repr(_attribute_state_fragment(contents, depth + 1)).encode("utf-8"))
    else:
        module = getattr(target, "__module__", None) or type(target).__module__
        qualname = getattr(target, "__qualname__", None) or type(target).__qualname__
        hasher.update(f"<no-code:{module}.{qualname}>".encode())
    return hasher.hexdigest()


_ATTRIBUTE_FRAGMENT_DEPTH_CEILING = 4
_ATTRIBUTE_FRAGMENT_ITEM_CEILING = 256

# Per-process salt + counter minting NEVER-MATCHING fragments for values the
# key cannot soundly cover: tensor attributes whose content cannot be read,
# and containers whose size exceeds the item ceiling (truncating them would
# make two containers differing only past the cut key identically). A stable
# content-blind fragment in either case would false-HIT on changed values,
# which the key contract forbids; the salt keeps the token unique across
# processes (and across pid reuse), the counter within this process.
_ATTRIBUTE_MISS_SALT = os.urandom(8).hex()
_ATTRIBUTE_MISS_COUNTER = itertools.count()


def _never_matching_fragment(kind: str) -> object:
    """Mint a fragment that can never equal any other fragment (always-miss)."""

    return (kind, _ATTRIBUTE_MISS_SALT, next(_ATTRIBUTE_MISS_COUNTER))


def _attribute_state_fragment(value: Any, depth: int = 0) -> object:
    """Return a bounded, address-free key fragment for one instance attribute.

    Plain instance attributes routinely determine the traced program
    (``self.num_layers``, ``self.scale``, ``self.use_checkpoint``), so they
    must participate in the capture-cache key. Values are reduced to stable
    primitives: scalars by value, tensors/arrays by content hash, callables by
    code digest, containers element-wise under the depth ceiling (containers
    larger than the item ceiling mint a never-matching token -- always-miss,
    never a truncated fragment that could false-HIT on a tail difference),
    and any other object by TYPE identity only -- an opaque object's internal
    state is a documented boundary of the signature (changing it without
    changing type keeps the key; conservative for false hits on the covered
    kinds, never address-churning).
    """

    if depth > _ATTRIBUTE_FRAGMENT_DEPTH_CEILING:
        # Same rule as the item ceiling: a STABLE ceiling token would make two
        # attributes identical down to the ceiling but differing BELOW it key
        # identically (false HIT). Never-match instead.
        return _never_matching_fragment("<attr-depth-ceiling>")
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return value
    if isinstance(value, torch.Tensor):
        if value.is_meta:
            # Meta tensors carry NO bytes: shape/dtype/device metadata IS
            # their entire observable content, so a stable metadata fragment
            # cannot be content-blind for them.
            return (
                "tensor-meta",
                tuple(value.shape),
                str(value.dtype),
                str(value.device),
            )
        try:
            return ("tensor", _hash_tensor_content(value))
        except Exception:
            # Content-unreadable tensor (sparse/exotic layout or backend):
            # degrading to a stable shape/dtype fragment made two
            # DIFFERENT-content tensors key identically -- a false cache HIT,
            # inverting this signature's "false hits never" contract. Mint a
            # never-matching token instead: this capture can never hit any
            # other entry (conservative always-miss, cache utility traded for
            # correctness).
            return _never_matching_fragment("tensor-unhashable")
    if isinstance(value, (torch.dtype, torch.device, torch.Size)):
        return ("torch-value", str(value))
    if isinstance(value, nn.Module):
        cls = type(value)
        return ("module", f"{cls.__module__}.{cls.__qualname__}")
    if isinstance(value, dict):
        # Truncating an over-ceiling container would leave items past the cut
        # OUT of the key, so two dicts differing only there would key
        # identically and serve each other's cached trace (false HIT, proven:
        # a 300-key config dict differing at insertion position 258 hit the
        # stale entry). Over-ceiling containers therefore always miss.
        if len(value) > _ATTRIBUTE_FRAGMENT_ITEM_CEILING:
            return _never_matching_fragment("dict-over-item-ceiling")
        return (
            "dict",
            len(value),
            tuple(
                sorted(
                    (
                        repr(_attribute_state_fragment(key, depth + 1)),
                        repr(_attribute_state_fragment(item, depth + 1)),
                    )
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, (list, tuple)):
        if len(value) > _ATTRIBUTE_FRAGMENT_ITEM_CEILING:
            return _never_matching_fragment("sequence-over-item-ceiling")
        return (
            "sequence",
            len(value),
            tuple(_attribute_state_fragment(item, depth + 1) for item in value),
        )
    if isinstance(value, (set, frozenset)):
        if len(value) > _ATTRIBUTE_FRAGMENT_ITEM_CEILING:
            return _never_matching_fragment("set-over-item-ceiling")
        member_reprs = sorted(repr(_attribute_state_fragment(item, depth + 1)) for item in value)
        return ("set", len(value), tuple(member_reprs))
    if type(value).__module__ == "numpy" and hasattr(value, "tobytes"):
        # grind-r5 b7 R22 (fable+opus): object-dtype buffers serialize raw
        # POINTERS -- content-blind (in-place mutation keeps the pointers) and
        # address-churning cross-process -- so they are unhashable here, and a
        # failed read must mint the never-matching token like every ceiling
        # branch in this file. Falling through to the stable ("object", cls)
        # tail made two DIFFERENT-content arrays key identically: a false
        # cache HIT serving the WRONG trace under cache=True.
        if getattr(getattr(value, "dtype", None), "hasobject", False):
            return _never_matching_fragment("ndarray-object-dtype")
        try:
            # Buffer-protocol digest (r7 R35-3): ``tobytes`` materialized a
            # whole-payload copy; a non-contiguous array contiguizes first
            # (same C-order bytes, so the digest is unchanged) and a
            # contiguous one hashes zero-copy.
            import numpy as _np

            contiguous = _np.ascontiguousarray(value)
            digest = hashlib.sha256(contiguous.data).hexdigest()
            return ("ndarray", tuple(getattr(value, "shape", ())), str(value.dtype), digest)
        except Exception:
            return _never_matching_fragment("ndarray-unreadable")
    if callable(value):
        # Thread the CURRENT depth through: the digest folds closure cells
        # back through this fragment reducer, so a self-referential closure
        # (an inner function holding itself) must consume the shared depth
        # budget and terminate at the ceiling instead of recursing forever.
        return ("callable", _callable_code_digest(value, depth))
    cls = type(value)
    return ("object", f"{cls.__module__}.{cls.__qualname__}")


# Hook dicts that fire during (or around) the captured forward/backward and
# therefore change what a capture observes. State-dict/load hooks are excluded:
# they cannot affect the traced program. The ``*_with_kwargs`` /
# ``*_always_called`` companions are flag dicts keyed by handle id; ids come
# from a process-global counter and are NOT stable across processes, so only
# their VALUES are folded, aligned by registration order.
