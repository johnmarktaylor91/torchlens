"""Device/layout-aware host transport for tensor digest and codec paths.

The historical spelling ``tensor.detach().cpu().contiguous()`` is a no-op on the
common already-contiguous CPU path, but for a CROSS-DEVICE source with a
dense-permuted layout (channels_last / transposed-dense) ``.cpu()`` preserves the
permuted strides (``Tensor.cpu`` defaults to ``memory_format=preserve_format``),
so the trailing ``.contiguous()`` forces a SECOND full materialization — a 2x
transient host peak on exactly the payloads GPU CNN workflows produce
(disputed-r2 b5/R35).

The naive one-line replacement ``.to("cpu", memory_format=contiguous_format)`` is
NOT behavior-identical either: on a same-device CPU noncontiguous strided view it
returns the ALIASED noncontiguous tensor (0 bytes allocated) where the old idiom
returned a contiguous copy. Hence this device/layout-aware branch instead of any
blind spelling. Real-CUDA D2H attestation for the fused single-copy claim rides
the standing GPU probe list.
"""

from __future__ import annotations

import torch

from ._utils.tensor_leaves import flat_byte_view


def to_cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    """Return a detached, CPU-resident, contiguous view or copy of ``tensor``.

    Semantics match ``tensor.detach().cpu().contiguous()`` exactly — same
    LOGICAL values, standard-contiguous result, no autograd tape — while a
    cross-device dense-permuted source pays ONE host materialization instead
    of two.

    Lazy conjugate/negative dispatch bits are RESOLVED here (r8 R35, the
    one digest authority): every digest/codec consumer downstream
    reinterprets the result as a ``torch.uint8`` view, which torch refuses
    on a tensor with the conj or neg bit set, so an unresolved
    ``x.conj()`` input crashed three fingerprint sites with a bare
    RuntimeError. Resolving materializes the logical values (what every
    consumer means by "the tensor's content") and is a no-op for the
    common bit-free path.

    Parameters
    ----------
    tensor:
        Source tensor on any device, any strided layout.

    Returns
    -------
    torch.Tensor
        Detached contiguous CPU tensor holding the LOGICAL values (conj/neg
        bits resolved). Already-contiguous CPU inputs without lazy bits come
        back as a zero-copy detached view of the same storage, like the
        historical idiom.
    """

    detached = tensor.detach()
    if detached.is_conj():
        detached = detached.resolve_conj()
    if detached.is_neg():
        detached = detached.resolve_neg()
    if detached.device.type == "cpu":
        # Same-device: ``.contiguous()`` is the single-copy (or no-op) path.
        # ``.to()`` must NOT be used here — on a noncontiguous CPU strided
        # view it aliases the source without contiguizing.
        return detached.contiguous()
    moved = detached.to("cpu", memory_format=torch.contiguous_format)
    if not moved.is_contiguous():
        # Fail-safe for any copy-machinery edge that ignores the requested
        # memory format: never hand back a noncontiguous transport tensor.
        moved = moved.contiguous()
    return moved


def digest_byte_view(tensor: torch.Tensor) -> memoryview:
    """Return the buffer-protocol byte view of one tensor's logical content.

    The shared uint8-reinterpret idiom behind every content digest
    (``to_cpu_contiguous(t).reshape(-1).view(torch.uint8).numpy().data``),
    folded into ONE authority next to the transport so a digest site can
    never reintroduce the unresolved-conj/neg crash by hand-rolling the
    view. The flat view comes from ``_utils.tensor_leaves.flat_byte_view``
    (stride-safe for one-element tensors). The returned memoryview keeps the
    backing ndarray (and thus the transport tensor's storage) alive; no
    whole-payload byte copy is made.

    Parameters
    ----------
    tensor:
        Dense strided tensor on any device (lazy conj/neg bits allowed).

    Returns
    -------
    memoryview
        Read-through byte view of the CPU-contiguous logical payload.
    """

    return flat_byte_view(to_cpu_contiguous(tensor)).numpy().data
