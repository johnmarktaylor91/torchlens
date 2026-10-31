"""The one refusal for a file saved by torchlens 2.x.

This release carries no reader for 2.x files. Every door that can recognise one raises
:func:`pre_3_artifact_error`: the restricted unpickler through
:func:`refuse_pre_3_module_path` for each global it resolves, and the manifest checks for
a format stamp older than the current one. A 2.x file fails with this one error instead
of an unknown-global or missing-field refusal.
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
        "multi_trace",
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

#: Private modules that held the 2.x record classes before they moved: a pickle written by
#: an earlier 2.x release names them, and nothing in 3 resolves them.
_PRE_3_PRIVATE_MODULES = frozenset(
    {
        "torchlens._data_classes.model_log",
        "torchlens._data_classes.layer_pass_log",
        "torchlens._data_classes.op_log",
        "torchlens._data_classes.module_log",
        "torchlens._data_classes.grad_fn_pass_log",
        "torchlens._data_classes.grad_fn_call_log",
    }
)

#: Individual globals whose module still exists but whose class name only 2.x wrote.
_PRE_3_GLOBALS = frozenset({("torchlens.intervention.types", "HelperSpec")})

_REMEDY = (
    "save it again with torchlens 2.36.x, which can read it (pip install "
    "'torchlens>=2.36,<3'), or capture the model again with this release and save that."
)


_MODELLOG_216_NOTE = (
    "A portable ModelLog bundle of torchlens 2.16 (io_format_version 2) is the exception: "
    "torchlens 2.36.x cannot read it either, and only torchlens 2.17.0 does (pip install "
    "torchlens==2.17.0 in a separate environment)."
)


def pre_3_artifact_error(
    observed: str,
    *,
    path: str | None = None,
    remedy_note: str = "",
    code: str = "artifact_version_below_floor",
) -> ArtifactTooOldError:
    """Build the typed refusal for an artifact saved by torchlens 2.x.

    Parameters
    ----------
    observed:
        What marks the artifact as 2.x (a pickled module path, a format stamp, a manifest
        spelling), rendered as ``name=value`` text.
    path:
        Artifact path, when the caller has it in scope.
    remedy_note:
        One extra sentence appended to the remedy when the standard one is not enough for
        this origin (the 2.16 ModelLog format).
    code:
        Always ``"artifact_version_below_floor"``; raise sites pass it explicitly so the
        S-17 census sees the code where the raise happens.

    Returns
    -------
    ArtifactTooOldError
        The refusal, ready to raise; ``fields["code"]`` is ``artifact_version_below_floor``.
    """

    from .format_contract import MIN_TLSPEC_VERSION

    remedy = f"{_REMEDY} {remedy_note}" if remedy_note else _REMEDY
    return ArtifactTooOldError(
        f"This artifact was saved by torchlens 2.x ({observed}); torchlens 3 does not read "
        f"files saved by torchlens 2.x. Remedy: {remedy}",
        code=code,
        observed=observed,
        floor_tlspec_version=MIN_TLSPEC_VERSION,
        floor_torchlens_version="3.0.0",
        path=path,
        remedy=remedy,
    )


def modellog_216_error(
    path: str, *, code: str = "artifact_version_below_floor"
) -> ArtifactTooOldError:
    """Build the 2.x refusal for a portable ModelLog bundle of torchlens 2.16.

    Parameters
    ----------
    path:
        Artifact path.
    code:
        Always ``"artifact_version_below_floor"``, passed explicitly by each raise site.

    Returns
    -------
    ArtifactTooOldError
        The refusal, ready to raise; its remedy also names the one release that reads it.
    """

    return pre_3_artifact_error(
        "the portable ModelLog format of torchlens 2.16",
        path=path,
        remedy_note=_MODELLOG_216_NOTE,
        code=code,
    )


def refuse_pre_3_module_path(module: str, name: str = "") -> None:
    """Refuse a pickled module path that only torchlens 2.x wrote.

    Parameters
    ----------
    module:
        Pickled module path as written in the artifact.
    name:
        Pickled class or function name, when the caller has it.

    Raises
    ------
    ArtifactTooOldError
        When ``module`` lives under a package that left the public surface in 3.0, or
        names a private module or class that only 2.x wrote.
    """

    root, _, rest = module.partition(".")
    if root == "torchlens" and (
        rest.partition(".")[0] in _PRE_3_PACKAGES
        or module in _PRE_3_PRIVATE_MODULES
        or (module, name) in _PRE_3_GLOBALS
    ):
        raise pre_3_artifact_error(f"module_path={module}", code="artifact_version_below_floor")
