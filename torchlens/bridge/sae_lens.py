"""SAE Lens bridge helpers."""

from __future__ import annotations

from typing import Any

from ._utils import activation_at


def encode(log: Any, site: Any, sae: Any) -> Any:
    """Encode a TorchLens activation with an SAE Lens-compatible SAE.

    Parameters
    ----------
    log:
        TorchLens ``ModelLog`` containing saved activations.
    site:
        Layer label, selector, or layer object to encode.
    sae:
        SAE object exposing ``encode`` or a callable SAE.

    Returns
    -------
    Any
        Encoded activation returned by the SAE.

    Raises
    ------
    ImportError
        If SAE Lens is unavailable.
    TypeError
        If ``sae`` cannot encode an activation.
    """

    try:
        import sae_lens  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SAE Lens bridge requires the `sae` extra: install torchlens[sae]."
        ) from exc

    activation = activation_at(log, site)
    if hasattr(sae, "encode"):
        return sae.encode(activation)
    if callable(sae):
        return sae(activation)
    raise TypeError("SAE Lens bridge expected an SAE with encode(...) or a callable SAE.")


__all__ = ["encode"]
