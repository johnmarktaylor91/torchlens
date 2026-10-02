"""Stimulus-indexed annotation gating shared by the *_evolution verbs.

Split out of ``torchlens.repgeom`` (lane A11): the eligibility gate that
stopped the fabricated buffer pseudo-RDMs (neuro MEMO D4/D5), the gated
ATOMIC annotation commit (D4), and the verb-aware shared diagnostics. The
public verbs re-export these privately through ``torchlens.repgeom``.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any

import torch

from ..backends.registry import TORCH_BACKEND_NAME
from ..errors._base import TorchLensWarning
from ..utils._multipass_access import get_multipass_attr

#: Verb-specific vocabulary for the shared site-selection diagnostics, so an
#: error raised for ``rdm_evolution`` never names ``mds_evolution`` (the
#: historical shared-helper defect). ``feature_map_evolution`` deliberately
#: keeps the ``mds_evolution`` default and recasts the text itself.
_VERB_VOCABULARY: dict[str, dict[str, str]] = {
    "mds_evolution": {"aggregate": "aggregate MDS", "layers": "MDS"},
    "rdm_evolution": {"aggregate": "an aggregate RDM", "layers": "RDM"},
    "scree_evolution": {"aggregate": "an aggregate scree spectrum", "layers": "scree"},
}


def _expected_stimulus_counts(trace: Any) -> frozenset[int]:
    """Return the leading-axis sizes of the capture's input sites.

    Shape metadata survives even when payloads are unsaved, so this evidence
    is available on every trace. An empty set means the stimulus count is
    underivable and shape equality cannot be defended.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.

    Returns
    -------
    frozenset[int]
        Leading dimensions observed across input-boundary layers.
    """

    from ..errors._base import TorchLensError

    # Absent/odd input evidence must read as "no evidence", never kill the
    # sweep; lookup surfaces raise TorchLensError subclasses plus the builtin
    # container/lookup families.
    evidence_errors = (TorchLensError, KeyError, ValueError, TypeError, AttributeError)
    counts: set[int] = set()
    try:
        input_labels = list(getattr(trace, "input_layers", []) or [])
    except evidence_errors:
        return frozenset()
    for label in input_labels:
        try:
            shape = getattr(trace[label], "shape", None)
        except evidence_errors:
            shape = None
        if shape and len(shape) >= 1 and int(shape[0]) > 0:
            counts.add(int(shape[0]))
    return frozenset(counts)


def _site_ineligibility(site: Any, expected_counts: frozenset[int]) -> str | None:
    """Return why a site is not stimulus-indexed, or ``None`` when eligible.

    Evidence order (neuro MEMO D4): record kind first (a buffer overwrite is
    never stimulus-indexed), then defended shape equality against the
    capture's input-site leading axis. When the stimulus count is underivable
    (no input evidence) shape equality cannot be defended and only the record
    kind gates. The one narrow spurious-pass class is a NON-batched tensor
    whose leading axis coincidentally equals the stimulus count and is not
    flagged as a buffer source -- named here rather than solved.

    Parameters
    ----------
    site:
        Layer or op record under consideration.
    expected_counts:
        Leading-axis sizes of the capture's input sites (may be empty).

    Returns
    -------
    str | None
        Human-readable ineligibility reason, or ``None`` when no evidence
        disqualifies the site.
    """

    # get_multipass_attr: a rolled multi-pass Layer raises the varying-field
    # ValueError tripwire on plain attribute reads; absent/varying evidence
    # must read as "no evidence", never kill the sweep.
    if bool(get_multipass_attr(site, "is_buffer_source", False, multipass=None)):
        return "the site is a buffer overwrite (is_buffer_source=True), not stimulus-indexed"
    shape = get_multipass_attr(site, "shape", None, multipass=None)
    if shape is None:
        return None
    if len(shape) < 1:
        return "the site's output is 0-dimensional, so it has no leading stimulus axis"
    if expected_counts and int(shape[0]) not in expected_counts:
        expected = sorted(expected_counts)
        return (
            f"the site's leading axis is {int(shape[0])} but this capture ran "
            f"{expected[0] if len(expected) == 1 else expected} stimuli"
        )
    return None


def _raise_ineligible_site(verb: str, label: str, reason: str) -> None:
    """Refuse an explicitly requested non-stimulus-indexed site, teaching.

    Parameters
    ----------
    verb:
        Public verb name the caller invoked.
    label:
        Label of the refused site.
    reason:
        Evidence-naming ineligibility reason from :func:`_site_ineligibility`.

    Raises
    ------
    ValueError
        Always; an explicit request for an ineligible site must never
        fabricate a stimulus-space result.
    """

    raise ValueError(
        f"{verb} cannot process explicitly selected site {label!r}: {reason}. "
        f"Axis 0 of each processed activation must index the capture's "
        f"stimuli; computing over anything else fabricates a stimulus-space "
        f"result. Select stimulus-indexed activation sites, or omit save= to "
        f"sweep every eligible saved site."
    )


def _warn_skipped_sites(verb: str, skipped: OrderedDict[str, str]) -> None:
    """Emit the one summarized skip disclosure for a default sweep.

    Parameters
    ----------
    verb:
        Public verb name the caller invoked.
    skipped:
        Mapping of skipped site label to its ineligibility reason.
    """

    if not skipped:
        return
    preview = [f"{label} ({reason})" for label, reason in list(skipped.items())[:3]]
    more = len(skipped) - len(preview)
    suffix = f" and {more} more" if more > 0 else ""
    warnings.warn(
        TorchLensWarning(
            f"{verb} skipped {len(skipped)} saved site(s) that are not "
            f"stimulus-indexed: {'; '.join(preview)}{suffix}. Remedy: select "
            f"stimulus-indexed sites explicitly; an explicit selection of a "
            f"skipped site raises the full explanation",
            code="annotation_sweep_sites_skipped",
        ),
        stacklevel=4,
    )


def _commit_annotation_tensors(trace: Any, staged: OrderedDict[str, torch.Tensor]) -> None:
    """Atomically commit staged annotation tensors to a torch trace.

    The gated-write contract (neuro MEMO D4): annotation tensors are staged
    while a sweep runs and committed only after the WHOLE sweep succeeds, so
    a failure at site k never leaves sites 0..k-1 annotated (a forced late
    failure used to leave exactly the fabricated blobs behind). Every payload
    is validated BEFORE the first write, so the commit loop itself cannot
    fail partway.

    Parameters
    ----------
    trace:
        Trace to annotate.
    staged:
        Ordered mapping of annotation blob key to tensor payload.

    Returns
    -------
    None
        The trace is mutated in place through ``_annotation_blobs``.

    Raises
    ------
    ValueError
        If the trace is not a torch trace or any payload is not portable
        (raised before anything is written).
    TypeError
        If any staged payload is not a torch tensor.
    """

    if not staged:
        return
    backend_name = str(getattr(trace, "backend", TORCH_BACKEND_NAME))
    if backend_name != TORCH_BACKEND_NAME:
        raise ValueError(
            "Tensor annotation blobs are supported only for torch traces in this "
            f"release; this trace uses backend={backend_name!r}."
        )
    try:
        validate_tensor = trace._validate_annotation_tensor
    except AttributeError:
        validate_tensor = None
    if not callable(validate_tensor):
        raise ValueError("trace does not support validated tensor annotation blobs.")
    for key, tensor in staged.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"_commit_annotation_tensors requires torch.Tensor payloads; "
                f"key {key!r} staged {type(tensor).__name__}."
            )
        validate_tensor(tensor)
    try:
        blobs_absent = trace._annotation_blobs is None
    except AttributeError:
        blobs_absent = True
    if blobs_absent:
        trace._annotation_blobs = {}
    for key, tensor in staged.items():
        trace._annotation_blobs[key] = tensor
    try:
        mark_mutated = trace._mark_annotations_mutated
    except AttributeError:
        mark_mutated = None
    if callable(mark_mutated):
        mark_mutated()


def _raise_recurrent_layer_requires_pass(layer: Any, *, verb: str = "mds_evolution") -> None:
    """Raise the public recurrent-layer selector error, naming the real verb.

    Parameters
    ----------
    layer:
        Aggregate recurrent layer.
    verb:
        Public verb name the caller invoked.

    Raises
    ------
    ValueError
        Always raised with a pass-selection diagnostic.
    """

    vocabulary = _VERB_VOCABULARY.get(verb, _VERB_VOCABULARY["mds_evolution"])
    layer_label = str(getattr(layer, "layer_label"))
    num_passes = int(getattr(layer, "num_passes", 0))
    raise ValueError(
        f"{verb} cannot compute {vocabulary['aggregate']} for recurrent layer "
        f"{layer_label!r} with {num_passes} passes; select a pass "
        f"(layer is recurrent), for example tl.label('{layer_label}:1')."
    )


def _raise_unsaved_activation(label: str, *, verb: str = "mds_evolution") -> None:
    """Raise the public unsaved-activation error, naming the real verb.

    Parameters
    ----------
    label:
        Layer or op label selected by the caller.
    verb:
        Public verb name the caller invoked.

    Raises
    ------
    ValueError
        Always raised with capture guidance.
    """

    vocabulary = _VERB_VOCABULARY.get(verb, _VERB_VOCABULARY["mds_evolution"])
    raise ValueError(
        f"{verb} requires saved activations for {label!r}; capture with "
        f"save= covering the {vocabulary['layers']} layers before calling {verb}."
    )
