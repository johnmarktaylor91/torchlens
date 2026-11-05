"""Transcoder specs for the attribution graph: what each feature reads and writes.

A transcoder here is any object with ``encode(x)``, ``W_enc``, ``b_enc``,
``W_dec`` and ``b_dec`` (the SAE Lens orientation: ``W_enc`` is
``(d_in, n_features)`` and ``W_dec`` is ``(n_features, d_out)``), the same
object ``tl.splice_module(transcoder, input_route="in")`` splices in for an MLP.
A per-layer transcoder writes the output of the MLP whose input it reads; a
cross-layer transcoder gives ``W_dec`` and ``b_dec`` as mappings from MLP
module names to tensors, one entry per MLP output its features write.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .._result import AttributionError

__all__ = ["Reader", "normalize_transcoders"]

_DOCS_POINTER = "See docs/reference/attribution.md, section Attribution graphs."


@dataclass(frozen=True)
class Reader:
    """One transcoder: the MLP input it reads and the MLP outputs it writes.

    Attributes
    ----------
    site
        MLP module name whose call input the features read.
    transcoder
        The user's transcoder object (its ``encode`` gives activations).
    encoder
        ``(d_in, n_features)`` encoder matrix.
    encoder_bias
        ``(n_features,)`` encoder bias.
    decoders
        ``(write site, (n_features, d_out) decoder)`` pairs in the order given.
    decoder_biases
        ``(write site, (d_out,) bias)`` pairs in the order given.
    """

    site: str
    transcoder: Any
    encoder: torch.Tensor
    encoder_bias: torch.Tensor
    decoders: tuple[tuple[str, torch.Tensor], ...]
    decoder_biases: tuple[tuple[str, torch.Tensor], ...]


def _transcoder_refusal(site: str, why: str, *, code: str) -> AttributionError:
    """Build the coded refusal for a transcoder the graph cannot read."""

    return AttributionError(
        f"The transcoder for {site!r} cannot be read as a transcoder: {why}. "
        "Remedy: pass an object with encode(x), W_enc (d_in, n_features), b_enc "
        "(n_features,), W_dec (n_features, d_out) and b_dec (d_out,), or W_dec and "
        "b_dec as mappings from MLP module names to tensors for a cross-layer "
        f"transcoder. {_DOCS_POINTER}",
        code=code,
    )


def _tensor(transcoder: Any, site: str, name: str) -> torch.Tensor:
    """Return one weight of ``transcoder`` as a detached tensor."""

    value = getattr(transcoder, name, None)
    if not isinstance(value, torch.Tensor):
        raise _transcoder_refusal(
            site,
            f"{name} is {type(value).__name__}, not a tensor",
            code="attribution_graph_transcoder_invalid",
        )
    return value.detach()


def _per_write(transcoder: Any, site: str, name: str) -> tuple[tuple[str, torch.Tensor], ...]:
    """Return ``name`` as ``(write site, tensor)`` pairs (a tensor writes ``site``)."""

    value = getattr(transcoder, name, None)
    if isinstance(value, torch.Tensor):
        return ((site, value.detach()),)
    if isinstance(value, Mapping) and value:
        pairs = []
        for write, tensor in value.items():
            if not isinstance(write, str) or not isinstance(tensor, torch.Tensor):
                raise _transcoder_refusal(
                    site,
                    f"{name} maps {write!r} to {type(tensor).__name__}",
                    code="attribution_graph_transcoder_invalid",
                )
            pairs.append((write, tensor.detach()))
        return tuple(pairs)
    raise _transcoder_refusal(
        site,
        f"{name} is {type(value).__name__}, not a tensor or a mapping",
        code="attribution_graph_transcoder_invalid",
    )


def _reader(site: str, transcoder: Any) -> Reader:
    """Validate one transcoder and return its reader spec."""

    if not callable(getattr(transcoder, "encode", None)):
        raise _transcoder_refusal(
            site, "it has no callable encode", code="attribution_graph_transcoder_invalid"
        )
    encoder = _tensor(transcoder, site, "W_enc")
    encoder_bias = _tensor(transcoder, site, "b_enc")
    if encoder.dim() != 2 or encoder_bias.shape != (encoder.shape[1],):
        raise _transcoder_refusal(
            site,
            f"W_enc {tuple(encoder.shape)} and b_enc {tuple(encoder_bias.shape)} do not "
            "agree on (d_in, n_features) and (n_features,)",
            code="attribution_graph_transcoder_invalid",
        )
    decoders = _per_write(transcoder, site, "W_dec")
    decoder_biases = _per_write(transcoder, site, "b_dec")
    for write, decoder in decoders:
        if decoder.dim() != 2 or decoder.shape[0] != encoder.shape[1]:
            raise _transcoder_refusal(
                site,
                f"W_dec for {write!r} has shape {tuple(decoder.shape)}; expected "
                f"({encoder.shape[1]}, d_out)",
                code="attribution_graph_transcoder_invalid",
            )
    if {write for write, _ in decoder_biases} != {write for write, _ in decoders}:
        raise _transcoder_refusal(
            site,
            "W_dec and b_dec name different MLP outputs",
            code="attribution_graph_transcoder_invalid",
        )
    return Reader(site, transcoder, encoder, encoder_bias, decoders, decoder_biases)


def normalize_transcoders(transcoders: Any) -> tuple[Reader, ...]:
    """Validate the ``transcoders`` mapping and return one reader per entry.

    Parameters
    ----------
    transcoders
        Mapping from MLP module name to transcoder object.

    Returns
    -------
    tuple[Reader, ...]
        Readers in the mapping's order.

    Raises
    ------
    AttributionError
        ``attribution_graph_transcoder_invalid`` when the mapping is empty or
        a transcoder lacks a weight, has inconsistent shapes or names no
        MLP output to write.
    """

    if not isinstance(transcoders, Mapping) or not transcoders:
        raise _transcoder_refusal(
            "<transcoders>",
            f"transcoders must be a non-empty mapping of MLP module names; got "
            f"{type(transcoders).__name__}",
            code="attribution_graph_transcoder_invalid",
        )
    return tuple(_reader(str(site), transcoder) for site, transcoder in transcoders.items())
