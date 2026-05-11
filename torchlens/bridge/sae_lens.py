"""SAE Lens bridge helpers."""

from __future__ import annotations

from typing import Any

from ._utils import out_at


def encode(log: Any, site: Any, sae: Any) -> Any:
    """Encode a TorchLens out with an SAE Lens-compatible SAE.

    Parameters
    ----------
    log:
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

    out = out_at(log, site)
    if hasattr(sae, "encode"):
        return sae.encode(out)
    if callable(sae):
        return sae(out)
    raise TypeError("SAE Lens bridge expected an SAE with encode(...) or a callable SAE.")


__all__ = ["encode"]
