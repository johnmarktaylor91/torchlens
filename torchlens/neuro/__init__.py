"""The seam between a captured network and neuroscience tooling: hands per-site representations to rsatoolbox and Brain-Score with provenance, and raises teaching refusals for work those tools own.

Importing this package must not import ``rsatoolbox``; each name checks its optional dependency when it is first used."""

from __future__ import annotations

from importlib.util import find_spec as _find_spec

from .._utils.facade import DependencyGate as _DependencyGate
from ._conflicts import (
    check_rsatoolbox_conflict as _check_rsatoolbox_conflict,
    rsatoolbox_conflict as _rsatoolbox_conflict,
)

_INSTALL_NEURO = 'pip install "torchlens[neuro]"'
_INSTALL_BRAINSCORE = 'pip install "torchlens[brainscore]" (Python >= 3.11)'

#: ``__all__``/``dir()`` advertise only the names whose foreign dependency
#: is PRESENT (probed with ``find_spec`` at import time -- a metadata search
#: that executes nothing): a name that cannot resolve in this environment is
#: not a real public name of this environment, and ``hasattr`` already
#: answers ``False`` for it. Access to an unadvertised gated name still
#: teaches the exact install command (facade step 4 re-probes per access).
#: An installed rsatoolbox that the metadata-only conflict check diagnoses
#: as unimportable (the [neuro,brainscore] scikit-learn pin) is not
#: advertised either; access raises ``neuro_dependency_conflict``.
__all__ = (
    ["ExtractionPlan", "PlanSite", "compare", "export_npz", "plan_extraction"]
    + (
        ["datasets", "rdms"]
        if _find_spec("rsatoolbox") is not None and _rsatoolbox_conflict() is None
        else []
    )
    + (
        ["activations_extractor", "get_activations_fn"]
        if _find_spec("brainscore_vision") is not None
        else []
    )
)
__all__.sort()

#: Real active names (facade step 4), each behind its own dependency gate.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    # Ungated: numpy-only, so it needs no extra.
    "compare": ("torchlens.neuro._compare", "compare"),
    # The extraction planner and the npz handoff (needs no extra).
    "ExtractionPlan": ("torchlens.neuro._brainpipe", "ExtractionPlan"),
    "PlanSite": ("torchlens.neuro._brainpipe", "PlanSite"),
    "export_npz": ("torchlens.neuro._brainpipe", "export_npz"),
    "plan_extraction": ("torchlens.neuro._brainpipe", "plan_extraction"),
    "datasets": ("torchlens.neuro._datasets", "datasets"),
    "rdms": ("torchlens.neuro._rdms", "rdms"),
    "activations_extractor": ("torchlens.bridge.brain_score", "activations_extractor"),
    "get_activations_fn": ("torchlens.bridge.brain_score", "get_activations_fn"),
}

#: Redirect table: names that live elsewhere
#: in torchlens and need NO extra. A pinned test walks this table and
#: asserts every named spelling resolves, so redirects cannot rot.
_REDIRECTS: dict[str, str] = {
    "rdm": (
        "use tl.repgeom.rdm (needs no extra); neuro.rdms() computes through "
        "it and hands the result to rsatoolbox with provenance descriptors"
    ),
    "cka": "use tl.stats.cka (needs no extra; linear, biased, CPU float64)",
    "mds": (
        "use tl.repgeom.classical_mds (needs no extra); non-metric MDS is "
        "deliberately not implemented -- see scikit-learn's MDS"
    ),
    "classical_mds": "use tl.repgeom.classical_mds (needs no extra)",
    "scree": "use tl.repgeom.scree (needs no extra)",
    "pca": (
        "use tl.repgeom.scree for the eigenspectrum and "
        "tl.repgeom.effective_dimensionality for participation-ratio "
        "summaries (needs no extra)"
    ),
    "effective_dimensionality": ("use tl.repgeom.effective_dimensionality (needs no extra)"),
    "procrustes_align": "use tl.repgeom.procrustes_align (needs no extra)",
    "extract_dataset": "use tl.extract_dataset (needs no extra)",
    "load_extraction": "use tl.load (needs no extra)",
    "brain_score": (
        "the verified adapters are neuro.activations_extractor and "
        "neuro.get_activations_fn (live-gate verified against "
        "brainscore-vision 2.3.22, Python 3.12, CPU, and scored end to end "
        "through a ModelCommitment plugin on brainscore-vision 2.4.1, "
        "Python 3.11, CPU); the underlying module "
        "is tl.bridge.brain_score, and the offline per_layer recipe "
        "deliberately keeps its bridge-only spelling"
    ),
}

#: Refuse-and-point table: each entry names
#: the ask, the owning neighbour, the missing ingredient, and the torchlens
#: spelling that feeds it.
_REFUSALS: dict[str, str] = {
    "noise_ceiling": (
        "a noise ceiling is conceptually undefined for a deterministic "
        "forward pass -- one stimulus yields ONE measurement, so there is no "
        "repeat variance to ceiling against (undefined, not unimplemented). "
        "For brain data, rsatoolbox.inference owns ceilings; feed it "
        "Datasets from neuro.datasets()"
    ),
    "bootstrap": (
        "bootstrap inference over stimuli/subjects belongs to "
        "rsatoolbox.inference; torchlens hands it descriptor-complete RDMs "
        "via neuro.rdms(). For a DESCRIPTIVE stimulus-bootstrap interval on "
        "torchlens's own scores, use tl.repgeom.compare_rdms_interval or "
        "tl.repgeom.cka_interval (needs no extra)"
    ),
    "permutation_test": (
        "permutation-based hypothesis inference belongs to "
        "rsatoolbox.inference; torchlens hands it descriptor-complete RDMs "
        "via neuro.rdms()"
    ),
    "crossnobis": (
        "crossnobis needs repeats plus a noise covariance, which a "
        "deterministic forward does not have; rsatoolbox's distance/noise "
        "tools own it. CAUTION: rsatoolbox's mahalanobis with no noise "
        "argument SILENTLY returns scaled euclidean (measured, 0.1.5 and "
        "0.3.2). The legitimate model-side recipe: a genuinely stochastic "
        "model under tl.add_noise / trace.do has real repeats "
        "(Khaligh-Razavi 2014's noise-matching step)"
    ),
    "mahalanobis": (
        "mahalanobis needs a noise covariance estimate; rsatoolbox's "
        "distance/noise tools own it. CAUTION: rsatoolbox's mahalanobis "
        "with no noise argument SILENTLY returns scaled euclidean "
        "(measured, 0.1.5 and 0.3.2). For a stochastic model under "
        "tl.add_noise / trace.do, real repeats exist and the recipe applies"
    ),
    "searchlight": (
        "searchlight needs brain-volume neighborhoods (rsatoolbox plus "
        "nilearn/nibabel own that); tl.repgeom.rdm_evolution is the "
        "network-side analogue, and neuro.datasets() preserves factual "
        "unit coordinates in channel descriptors for a possible model-side "
        "searchlight later"
    ),
    "kendall": (
        "kendall's tau carries the tau-a vs tau-b tie trap (tau-b "
        "understates agreement when model RDMs contain ties); "
        "rsatoolbox.rdm.compare ships tau-a. Feed it RDMs from neuro.rdms()"
    ),
    "fit_model": (
        "model fitting and weighted-model RSA belong to "
        "rsatoolbox.inference and rsatoolbox.model; torchlens hands them "
        "descriptor-complete RDMs via neuro.rdms()"
    ),
    "encoding_model": (
        "encoding-model fitting, voxel maps, variance partitioning and "
        "stacked fits belong to Brain-Score, himalaya and Net2Brain; "
        "tl.extract_dataset writes the stimuli-x-features matrices they "
        "consume. For a DESCRIPTIVE cross-validated ridge score per site, "
        "use neuro.compare(source, target, analyses=('ridge',))"
    ),
    "ridge": (
        "ridge fitting with alpha inference, voxel maps and variance "
        "partitioning belongs to himalaya (and Brain-Score's pipelines); "
        "tl.extract_dataset writes the stimuli-x-features matrices it "
        "consumes. For a DESCRIPTIVE cross-validated ridge score per site, "
        "use neuro.compare(source, target, analyses=('ridge',))"
    ),
    "cca": (
        "CCA/SVCCA/PWCCA and shape metrics live in netrep and documented "
        "recipes; tl.stats metric functions stay pure over matched "
        "[n_obs x n_features] pairs, which is exactly what those tools "
        "consume"
    ),
    "svcca": (
        "SVCCA is a different statistic, not a CKA kernel; netrep and "
        "documented recipes own it. tl.extract_dataset / torchlens.features "
        "produce the matched [n_obs x n_features] pairs it consumes"
    ),
    "pwcca": (
        "PWCCA lives in netrep and documented recipes; tl.extract_dataset "
        "/ torchlens.features produce the matched [n_obs x n_features] pairs it "
        "consumes"
    ),
    "nonmetric_mds": (
        "non-metric MDS belongs to scikit-learn (sklearn.manifold.MDS with "
        "metric=False); torchlens keeps classical MDS at "
        "tl.repgeom.classical_mds"
    ),
    "model_zoo": (
        "matched preprocessing, model zoos, and gLocal weights belong to "
        "thingsvision; torchlens captures whatever model YOU construct, "
        "with torchlens.preprocessing.resolve() verifying the preprocessing "
        "authority your loader ships"
    ),
    "brain_data": (
        "brain data and stimulus sets belong to their owning ecosystems "
        "(rsatoolbox datasets, Brain-Score benchmarks, NSD/THINGS "
        "releases); torchlens owns the network side of the seam only"
    ),
    "stimulus_set": (
        "stimulus sets belong to Brain-Score's ecosystem; torchlens "
        "records YOUR stimuli's ids and order in the extraction manifest "
        "(tl.extract_dataset(..., stimulus_ids=...))"
    ),
}

#: Per-name dependency gates (facade step 4). Redirect/refusal rows need no
#: gate -- teaching an absent name must never demand a foreign package.
_DEPENDENCIES: dict[str, _DependencyGate] = {
    "datasets": _DependencyGate("rsatoolbox", _INSTALL_NEURO, _check_rsatoolbox_conflict),
    "rdms": _DependencyGate("rsatoolbox", _INSTALL_NEURO, _check_rsatoolbox_conflict),
    # Gated on brainscore_vision ALONE (memo D17/4.3): brainscore_core is
    # named nowhere -- pinning brainscore-vision gets core for free.
    "activations_extractor": _DependencyGate("brainscore_vision", _INSTALL_BRAINSCORE),
    "get_activations_fn": _DependencyGate("brainscore_vision", _INSTALL_BRAINSCORE),
}


def __getattr__(name: str) -> object:
    """Resolve attributes through the shared five-step facade order.

    Parameters
    ----------
    name:
        Requested module attribute name.

    Returns
    -------
    object
        The resolved attribute for real active names.

    Raises
    ------
    AttributeError
        Per the five-step contract: plain for underscore and unknown names,
        typed teaching subclasses for redirect/refusal rows, and a typed
        dependency error (AttributeError lineage, ImportError semantics)
        for gated names whose dependency is absent.
    """

    from .._utils.facade import resolve_facade_attr

    return resolve_facade_attr(
        owner=__name__,
        name=name,
        module_globals=globals(),
        lazy_attrs=_LAZY_ATTRS,
        redirects=_REDIRECTS,
        refusals=_REFUSALS,
        dependencies=_DEPENDENCIES,
    )


def __dir__() -> list[str]:
    """Return the real public names of the namespace and nothing else.

    Returns
    -------
    list[str]
        Sorted public names whose dependency is present; redirect and
        refusal rows are teaching surfaces, not attributes, and
        implementation imports are not advertised.
    """

    from .._utils.facade import facade_dir

    return facade_dir(globals(), __all__)


# ``from __future__ import annotations`` binds ``annotations`` as a reachable
# module attribute; nothing reads the binding (the future feature is a
# compile-time flag), so unbind it -- the root facade's own idiom.
del annotations
