"""torchlens.neuro: the treaty desk between network and brain (neuro memo).

torchlens owns everything that touches the network; rsatoolbox and
Brain-Score own everything that touches the brain or the hypothesis. This
namespace hands representations across that seam with provenance that
cannot lie, and teaches where everything else lives. The map, in five
sections:

1. EXTRACT (upstream, needs no extra): capture with ``tl.trace(model, x,
   save=...)`` or write a disk artifact with ``tl.extract_dataset(...,
   output_dir=...)`` and read it back with ``tl.load_extraction``. The
   manifest records stimulus ids, ordering, dtypes, and preprocessing
   provenance -- everything downstream descriptors are built from.
2. GEOMETRY (needs no extra): RDMs and friends live in their canonical
   homes -- ``tl.repgeom.rdm`` / ``classical_mds`` / ``scree`` /
   ``effective_dimensionality`` / ``procrustes_align``, and ``tl.stats.cka``.
   Nothing is re-exported here: an extras-gated alias for a numpy-only
   function would poison the dependency gate and split the docs home.
3. HANDOFF (this namespace; requires the ``neuro`` extra = rsatoolbox):
   ``neuro.datasets(source, ...)`` turns every stimulus-indexed site of a
   Trace, LoadedExtraction, or extraction directory into per-site
   rsatoolbox Datasets whose descriptors carry the full identity story;
   ``neuro.rdms(source, ...)`` computes RDMs through the canonical
   tl.repgeom arithmetic and records the metric it ACTUALLY used -- plus a
   matrix mode that converts precomputed matrices and REQUIRES the declared
   measure.
4. SCORE (requires the ``brainscore`` extra = brainscore-vision, Python
   >= 3.11): ``neuro.activations_extractor(model, preprocessing, ...)``
   builds a real Brain-Score ``ActivationsExtractorHelper`` over TorchLens
   capture, and ``neuro.get_activations_fn(model, ...)`` is the underlying
   per-batch callable. Both are exposed here because the live end-to-end
   gate PASSED (2026-08-29): a real resnet18 checkpoint through a real
   ``ActivationsExtractorHelper`` and ``StimulusSet`` against a live
   brainscore-vision 2.3.22 install on Python 3.12, with values,
   presentation order, layer coordinates, logits, dotted module paths, a
   functional-op mapping, a short final batch, and CPU behavior matching
   direct torchlens extraction, plus default sites on a PARTIALLY saved
   trace. CUDA is not claimed (no covered runner). The offline
   ``per_layer`` recipe deliberately gets no neuro name (a generic
   score-a-callable loop must not borrow a benchmark's authority).
5. REFUSALS: noise ceilings, crossnobis/mahalanobis, searchlight,
   bootstrap/permutation/model fitting, encoding models, CCA/SVCCA/PWCCA,
   non-metric MDS, model zoos, and brain data itself are deliberately NOT
   here. Each attempted name raises a teaching refusal naming the owning
   neighbour, the missing ingredient, and the torchlens spelling that
   feeds it. (One reverse fact: rsatoolbox has no cosine RDM; ours is real
   and tested.)

Import time stays inert: this module must NOT import ``rsatoolbox`` (the
``neuro`` extra's foreign third-party dependency) as a side effect of
``import torchlens.neuro``. A bare package import can happen incidentally --
e.g. a portable ``.tlspec`` bundle's metadata unpickler resolving a pickled
global whose module path names ``torchlens.neuro`` -- and an eager
import-time dependency check would run that foreign code with no trust
opt-in.

Attribute access resolves through the shared five-step facade order (neuro
memo D15; ``torchlens.utils.facade``). Dependency checks are PER-NAME,
probed with ``importlib.util.find_spec`` so answering never executes
foreign code, and every typed refusal subclasses ``AttributeError`` so
``hasattr``/IPython canary probes degrade instead of erroring. The facade
is pickle-safe: resolving this module imports nothing foreign.

The neuro extra installs ``rsatoolbox`` (``pip install "torchlens[neuro]"``).
Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from importlib.util import find_spec as _find_spec

from ..utils.facade import DependencyGate as _DependencyGate

_INSTALL_NEURO = 'pip install "torchlens[neuro]"'
_INSTALL_BRAINSCORE = 'pip install "torchlens[brainscore]" (Python >= 3.11)'

#: ``__all__``/``dir()`` advertise only the names whose foreign dependency
#: is PRESENT (probed with ``find_spec`` at import time -- a metadata search
#: that executes nothing): a name that cannot resolve in this environment is
#: not a real public name of this environment, and ``hasattr`` already
#: answers ``False`` for it. Access to an unadvertised gated name still
#: teaches the exact install command (facade step 4 re-probes per access).
__all__ = (["datasets", "rdms"] if _find_spec("rsatoolbox") is not None else []) + (
    ["activations_extractor", "get_activations_fn"]
    if _find_spec("brainscore_vision") is not None
    else []
)
__all__.sort()

#: Real active names (facade step 4), each behind its own dependency gate.
_LAZY_ATTRS: dict[str, tuple[str, str | None]] = {
    "datasets": ("torchlens.neuro._datasets", "datasets"),
    "rdms": ("torchlens.neuro._rdms", "rdms"),
    "activations_extractor": ("torchlens.bridge.brain_score", "activations_extractor"),
    "get_activations_fn": ("torchlens.bridge.brain_score", "get_activations_fn"),
}

#: Redirect table (facade step 2; neuro memo 4.5): names that live elsewhere
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
    "load_extraction": "use tl.load_extraction (needs no extra)",
    "brain_score": (
        "the verified adapters are neuro.activations_extractor and "
        "neuro.get_activations_fn (live-gate verified against "
        "brainscore-vision 2.3.22, Python 3.12, CPU); the underlying module "
        "is tl.bridge.brain_score, and the offline per_layer recipe "
        "deliberately keeps its bridge-only spelling"
    ),
}

#: Refuse-and-point table (facade step 3; neuro memo 4.5): each entry names
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
        "via neuro.rdms()"
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
        "model under tl.noise / trace.do has real repeats "
        "(Khaligh-Razavi 2014's noise-matching step)"
    ),
    "mahalanobis": (
        "mahalanobis needs a noise covariance estimate; rsatoolbox's "
        "distance/noise tools own it. CAUTION: rsatoolbox's mahalanobis "
        "with no noise argument SILENTLY returns scaled euclidean "
        "(measured, 0.1.5 and 0.3.2). For a stochastic model under "
        "tl.noise / trace.do, real repeats exist and the recipe applies"
    ),
    "searchlight": (
        "searchlight needs brain-volume neighborhoods (rsatoolbox plus "
        "nilearn/nibabel own that); tl.repgeom.rdm_evolution is the "
        "network-side analogue, and neuro.datasets() preserves factual "
        "unit coordinates in channel descriptors for a possible model-side "
        "searchlight later"
    ),
    "compare": (
        "RDM comparison beyond pearson/spearman (whitened cosine, tau, "
        "rho-a, ...) is rsatoolbox.rdm.compare's mature vocabulary; feed it "
        "RDMs from neuro.rdms()"
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
        "encoding models / ridge regression to brain responses belong to "
        "Brain-Score and himalaya; tl.extract_dataset writes the "
        "stimuli-x-features matrices they consume"
    ),
    "ridge": (
        "ridge regression to brain responses belongs to himalaya (and "
        "Brain-Score's pipelines); tl.extract_dataset writes the "
        "stimuli-x-features matrices it consumes"
    ),
    "cca": (
        "CCA/SVCCA/PWCCA and shape metrics live in netrep and documented "
        "recipes; tl.stats metric functions stay pure over matched "
        "[n_obs x n_features] pairs, which is exactly what those tools "
        "consume"
    ),
    "svcca": (
        "SVCCA is a different statistic, not a CKA kernel; netrep and "
        "documented recipes own it. tl.extract_dataset / tl.features "
        "produce the matched [n_obs x n_features] pairs it consumes"
    ),
    "pwcca": (
        "PWCCA lives in netrep and documented recipes; tl.extract_dataset "
        "/ tl.features produce the matched [n_obs x n_features] pairs it "
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
        "with tl.preprocessing.resolve() verifying the preprocessing "
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
    "datasets": _DependencyGate("rsatoolbox", _INSTALL_NEURO),
    "rdms": _DependencyGate("rsatoolbox", _INSTALL_NEURO),
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

    from ..utils.facade import resolve_facade_attr

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

    from ..utils.facade import facade_dir

    return facade_dir(globals(), __all__)


# ``from __future__ import annotations`` binds ``annotations`` as a reachable
# module attribute; nothing reads the binding (the future feature is a
# compile-time flag), so unbind it -- the root facade's own idiom.
del annotations
