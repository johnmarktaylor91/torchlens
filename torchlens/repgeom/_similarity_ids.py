"""Similarity-repository identifiers for TorchLens's own geometry measures.

The similarity-repository (``nacloos/similarity-repository``) is the
catalogue of representational-similarity measures: each implementation is
registered under ``<type>/<repository>/<measure name>``, and the measure name
spells out the exact variant (for CKA, the kernel and the HSIC estimator).
TorchLens records the name of the variant it computes, with ``*`` for the
repository part, so ``similarity.make(identifier)`` selects every catalogued
implementation of the same variant.

Identifiers were taken from the repository at
:data:`SIMILARITY_REPOSITORY_COMMIT` (``similarity/types/__init__.py``) and
checked numerically against its implementations. ``None`` means the
catalogue has no exact match for that TorchLens variant; no near match is
substituted.
"""

from __future__ import annotations

__tl_layer__ = "L5"

SIMILARITY_REPOSITORY = "nacloos/similarity-repository"
SIMILARITY_REPOSITORY_COMMIT = (
    "035b3bee2b5815481a2433641e2813d4a2314507"  # pragma: allowlist secret
)

#: ``tl.repgeom.rdm(metric=...)`` -> RDM identifier.
RDM_IDS: dict[str, str | None] = {
    "euclidean": "rdm/*/euclidean",
    "cosine": "rdm/*/cosine",
    "correlation": "rdm/*/correlation",
    "gaussian": "rdm/*/gaussian",
    "manhattan": None,
}

#: ``rdm_compare(rdm(x, metric), rdm(y, metric), method)`` -> RSA identifier.
RSA_IDS: dict[tuple[str, str], str | None] = {
    ("euclidean", "pearson"): "measure/*/rsa-rdm=euclidean-compare=pearson",
    ("euclidean", "spearman"): "measure/*/rsa-rdm=euclidean-compare=spearman",
    ("euclidean", "kendall"): None,
    ("cosine", "pearson"): "measure/*/rsa-rdm=cosine-compare=pearson",
    ("cosine", "spearman"): "measure/*/rsa-rdm=cosine-compare=spearman",
    ("cosine", "kendall"): None,
    ("correlation", "pearson"): "measure/*/rsa-rdm=correlation-compare=pearson",
    ("correlation", "spearman"): "measure/*/rsa-rdm=correlation-compare=spearman",
    ("correlation", "kendall"): "measure/*/rsa-rdm=correlation-compare=tau_a",
    ("gaussian", "pearson"): "measure/*/rsa-rdm=gaussian-compare=pearson",
    ("gaussian", "spearman"): "measure/*/rsa-rdm=gaussian-compare=spearman",
    ("gaussian", "kendall"): None,
    ("manhattan", "pearson"): None,
    ("manhattan", "spearman"): None,
    ("manhattan", "kendall"): None,
}

#: ``tl.stats.cka`` / ``tl.stats.CKA``: linear kernel, biased (Gretton) HSIC.
LINEAR_CKA_ID = "measure/*/cka-kernel=linear-hsic=gretton-score"


def rdm_id(metric: str) -> str | None:
    """Return the similarity-repository identifier of one RDM metric.

    Parameters
    ----------
    metric:
        ``tl.repgeom`` distance metric name.

    Returns
    -------
    str | None
        The catalogue identifier, or ``None`` when it has no exact match.
    """

    return RDM_IDS.get(metric)
