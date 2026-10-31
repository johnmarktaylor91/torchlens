"""SAE Lens bridge helpers."""

from __future__ import annotations

from typing import Any

from ._utils import out_at


def encode(trace: Any, site: Any, sae: Any) -> Any:
    """Encode a TorchLens out with an SAE Lens-compatible SAE.

    Parameters
    ----------
    trace:
        TorchLens ``Trace`` containing saved outs.
    site:
        Layer label, selector, or layer object to encode.
    sae:
        SAE object exposing ``encode`` or a callable SAE.

    Returns
    -------
    Any
        Encoded out returned by the SAE.

    Raises
    ------
    ImportError
        If SAE Lens is unavailable.
    TypeError
        If ``sae`` cannot encode an out.
    """

    try:
        import sae_lens  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SAE Lens bridge requires the `sae` extra: install torchlens[sae]."
        ) from exc

    out = out_at(trace, site)
    if hasattr(sae, "encode"):
        return sae.encode(out)
    if callable(sae):
        return sae(out)
    raise TypeError("SAE Lens bridge expected an SAE with encode(...) or a callable SAE.")


def reconstruct(trace: Any, site: Any, sae: Any) -> Any:
    """Return an SAE Lens-compatible SAE's reconstruction of a TorchLens out.

    The site's activation is encoded, then the latents are decoded:
    ``sae.decode(sae.encode(out))``. A callable without ``encode`` and
    ``decode`` is called on the activation, since an SAE Lens ``SAE``'s
    forward pass is that same encode-then-decode. To decode latents you
    already hold, call ``sae.decode(latents)`` directly; to measure the
    reconstruction's causal effect, use ``torchlens.bridge.sae.splice``.

    Parameters
    ----------
    trace:
        TorchLens ``Trace`` containing saved outs.
    site:
        Layer label, selector, or layer object to reconstruct.
    sae:
        SAE object exposing ``encode`` and ``decode``, or a callable SAE.

    Returns
    -------
    Any
        Reconstruction returned by the SAE.

    Raises
    ------
    ImportError
        If SAE Lens is unavailable.
    TypeError
        If ``sae`` cannot reconstruct an out.
    """

    try:
        import sae_lens  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SAE Lens bridge requires the `sae` extra: install torchlens[sae]."
        ) from exc

    out = out_at(trace, site)
    if callable(getattr(sae, "encode", None)) and callable(getattr(sae, "decode", None)):
        return sae.decode(sae.encode(out))
    if callable(sae):
        return sae(out)
    raise TypeError(
        "SAE Lens bridge expected an SAE with encode(...) and decode(...) or a callable SAE."
    )


__all__ = ["encode", "reconstruct"]
