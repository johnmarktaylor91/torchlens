"""Fitted-PCA payload + digest-checked persisted sidecar (transforms memo P6).

The additive ``tl.stats.PCA`` upgrade three panels depend on: the fitted
result carries ``mean``, ``n_samples``, ``n_features``, and a content
``digest`` alongside the (deterministically sign-canonicalized) components,
and the fitted arrays persist to a standalone sidecar file under that
digest — because a solver seed does not reconstruct a degenerate eigenspace,
the ARRAYS are the identity. Loading recomputes the digest and REFUSES a
mismatch; nothing here executes or imports code from the artifact.

Spellings DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from .._errors import _actionable_message, _ActionableErrorMixin
from ..errors._base import ConfigurationError

__tl_layer__ = "L5"

__all__ = ["FittedArtifactError", "FittedPCA", "load_fitted", "save_fitted"]

#: Sidecar schema id (self-owned artifact; NOT a tlspec field family).
FITTED_SIDECAR_SCHEMA = "tl_pca_fitted_v1"


class FittedArtifactError(_ActionableErrorMixin, ConfigurationError, RuntimeError):
    """Raised when a fitted-projection payload or sidecar violates its contract."""

    def __init__(self, problem: str, *, code: str, remedy: str, **context: object) -> None:
        """Initialize an actionable fitted-artifact refusal.

        Parameters
        ----------
        problem:
            Description of the rejected object or operation and its cause.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured, non-authoritative diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


def fitted_digest(
    components: torch.Tensor, mean: torch.Tensor, explained_variance: torch.Tensor
) -> str:
    """Digest the fitted arrays (float64 canonical bytes + geometry header).

    Parameters
    ----------
    components:
        ``(k, d)`` component rows.
    mean:
        ``(d,)`` feature mean.
    explained_variance:
        ``(k,)`` eigenvalues.

    Returns
    -------
    str
        ``sha256:<hex>`` content digest — the array identity the sidecar and
        every consumer key on.
    """

    header = json.dumps(
        {
            "schema": FITTED_SIDECAR_SCHEMA,
            "k": int(components.shape[0]),
            "d": int(components.shape[1]) if components.dim() == 2 else 0,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    hasher = hashlib.sha256()
    hasher.update(header)
    for tensor in (components, mean, explained_variance):
        hasher.update(tensor.detach().to("cpu", torch.float64).contiguous().numpy().tobytes())
    return "sha256:" + hasher.hexdigest()


@dataclass(frozen=True, eq=False)
class FittedPCA:
    """The frozen fitted-PCA payload (memo P6 result upgrade).

    Attributes
    ----------
    components:
        ``(k, d)`` float64 component rows, deterministically
        sign-canonicalized (the largest-magnitude entry of each row is
        positive).
    mean:
        ``(d,)`` float64 feature mean the fit centered on.
    explained_variance:
        ``(k,)`` float64 eigenvalues, descending.
    n_samples:
        Number of rows the fit consumed.
    n_features:
        Feature width ``d``.
    fit_scope:
        Recorded disclosure of WHAT was fit on (fitting across held-out
        stimuli leaks analysis information; the scope travels with the
        arrays).
    digest:
        ``sha256:`` content digest of the arrays (:func:`fitted_digest`).
    """

    components: torch.Tensor
    mean: torch.Tensor
    explained_variance: torch.Tensor
    n_samples: int
    n_features: int
    fit_scope: str
    digest: str

    def __repr__(self) -> str:
        """Bounded identity line: geometry, scope, digest -- never the arrays (D31)."""

        k = int(self.components.shape[0]) if self.components.dim() >= 1 else 0
        return (
            f"FittedPCA(k={k}, n_features={self.n_features}, n_samples={self.n_samples}, "
            f"fit_scope={self.fit_scope!r}, digest={self.digest!r})"
        )


def save_fitted(fitted: FittedPCA, path: str | Path) -> Path:
    """Persist a fitted payload as the digest-checked ``tl_pca_fitted_v1`` sidecar.

    Parameters
    ----------
    fitted:
        The fitted payload.
    path:
        Destination file path (conventionally ``*.safetensors``).

    Returns
    -------
    Path
        The written path.
    """

    from safetensors.torch import save_file

    if not isinstance(fitted, FittedPCA):
        raise FittedArtifactError(
            f"save_fitted() needs a FittedPCA payload; got {type(fitted).__name__}.",
            code="pca_fitted_record_invalid",
            remedy="pass tl.stats.PCA(...).fitted(...)",
            value_type=type(fitted).__name__,
        )
    destination = Path(path)
    metadata = {
        "schema": FITTED_SIDECAR_SCHEMA,
        "digest": fitted.digest,
        "n_samples": str(fitted.n_samples),
        "n_features": str(fitted.n_features),
        "fit_scope": fitted.fit_scope,
    }
    save_file(
        {
            "components": fitted.components.detach().to("cpu", torch.float64).contiguous(),
            "mean": fitted.mean.detach().to("cpu", torch.float64).contiguous(),
            "explained_variance": fitted.explained_variance.detach()
            .to("cpu", torch.float64)
            .contiguous(),
        },
        str(destination),
        metadata=metadata,
    )
    return destination


def load_fitted(path: str | Path) -> FittedPCA:
    """Load a ``tl_pca_fitted_v1`` sidecar, recomputing and CHECKING the digest.

    No code is imported or executed; the arrays are the identity, and a
    digest mismatch refuses instead of serving silently different numbers.

    Parameters
    ----------
    path:
        Sidecar file path.

    Returns
    -------
    FittedPCA
        The rehydrated fitted payload.

    Raises
    ------
    FittedArtifactError
        ``pca_fitted_record_invalid`` on a malformed sidecar;
        ``pca_fitted_digest_mismatch`` when the recomputed array digest does
        not match the recorded one.
    """

    from safetensors import safe_open

    source = Path(path)
    try:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            tensor_names = handle.keys()
            tensors = {key: handle.get_tensor(key) for key in tensor_names}
    except Exception as exc:
        raise FittedArtifactError(
            f"Fitted sidecar at {source} is not a readable safetensors file: {exc}.",
            code="pca_fitted_record_invalid",
            remedy="pass a sidecar written by tl.stats.save_fitted()",
            path=str(source),
        ) from exc
    if metadata.get("schema") != FITTED_SIDECAR_SCHEMA or not {
        "components",
        "mean",
        "explained_variance",
    } <= set(tensors):
        raise FittedArtifactError(
            f"Fitted sidecar at {source} does not carry the "
            f"{FITTED_SIDECAR_SCHEMA!r} schema and its three arrays.",
            code="pca_fitted_record_invalid",
            remedy="pass a sidecar written by tl.stats.save_fitted()",
            path=str(source),
            schema=metadata.get("schema"),
        )
    components = tensors["components"].to(torch.float64)
    mean = tensors["mean"].to(torch.float64)
    explained = tensors["explained_variance"].to(torch.float64)
    recorded = metadata.get("digest", "")
    recomputed = fitted_digest(components, mean, explained)
    if recomputed != recorded:
        raise FittedArtifactError(
            f"Fitted sidecar at {source} records digest {recorded!r} but its "
            f"arrays hash to {recomputed!r}; serving silently different "
            "numbers is refused (the arrays ARE the identity — a solver seed "
            "does not reconstruct a degenerate eigenspace).",
            code="pca_fitted_digest_mismatch",
            remedy="re-save the sidecar from the original fit, or re-fit",
            path=str(source),
            recorded=recorded,
            recomputed=recomputed,
        )
    try:
        n_samples = int(metadata.get("n_samples", "0"))
        n_features = int(metadata.get("n_features", str(int(mean.numel()))))
    except ValueError as exc:
        raise FittedArtifactError(
            f"Fitted sidecar at {source} carries non-integer sample/feature counts.",
            code="pca_fitted_record_invalid",
            remedy="pass a sidecar written by tl.stats.save_fitted()",
            path=str(source),
        ) from exc
    return FittedPCA(
        components=components,
        mean=mean,
        explained_variance=explained,
        n_samples=n_samples,
        n_features=n_features,
        fit_scope=metadata.get("fit_scope", "unspecified"),
        digest=recorded,
    )
