"""Refuse pickled module paths that only torchlens 2.x wrote.

The restricted unpickler calls :func:`refuse_pre_3_module_path` for every global
it resolves, so a 2.x file fails with the one load-boundary error instead of an
unknown-global refusal.
"""

from __future__ import annotations

from .format_errors import ArtifactTooOldError

#: Top-level torchlens packages the 3.0 rename made private, renamed or folded away. A
#: pickled module path under one of them was written by torchlens 2.x, and this release
#: carries no reader for 2.x files: such a path refuses with the one load-boundary error.
_PRE_3_PACKAGES = frozenset(
    {
        "accessors",
        "autoroute",
        "brainpipe",
        "constants",
        "data_classes",
        "ecosystem",
        "hash",
        "ir",
        "kernel_telemetry",
        "merged",
        "notebook",
        "observability",
        "observe",
        "partial",
        "postprocess",
        "quantities",
        "quickstart",
        "runnable",
        "selection",
        "selection_compare",
        "selection_graph",
        "selection_subspace",
        "selection_values",
        "snoop",
        "trace_slice",
        "tviz",
        "utils",
    }
)


def refuse_pre_3_module_path(module: str) -> None:
    """Refuse a pickled module path that only torchlens 2.x wrote.

    Parameters
    ----------
    module:
        Pickled module path as written in the artifact.

    Raises
    ------
    ArtifactTooOldError
        When ``module`` lives under a package that left the public surface in 3.0.
    """

    root, _, rest = module.partition(".")
    if root != "torchlens" or rest.partition(".")[0] not in _PRE_3_PACKAGES:
        return
    from .format_contract import MIN_TLSPEC_VERSION

    remedy = (
        "save it again with torchlens 2.36.x, which can read it (pip install "
        "'torchlens>=2.36,<3'), or capture the model again with this release and save that."
    )
    raise ArtifactTooOldError(
        f"This artifact was saved by torchlens 2.x (it names the module {module}); "
        f"torchlens 3 does not read files saved by torchlens 2.x. Remedy: {remedy}",
        code="artifact_version_below_floor",
        observed=f"module_path={module}",
        floor_tlspec_version=MIN_TLSPEC_VERSION,
        floor_torchlens_version="3.0.0",
        path=None,
        remedy=remedy,
    )
