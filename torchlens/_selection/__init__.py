"""Selection: naming sites and resolving them on a trace.

Layout (import order top to bottom; nothing here imports ``torchlens.intervention``
at module level, so every module imports standalone):

* ``errors``: the ``SelectionError`` and ``SiteResolutionError`` families.
* ``target``: ``TargetSpec`` / ``FrozenTargetSpec``, the persisted form of a selector.
* ``mask``: the dense/slice mask algebra behind element selections.
* ``selector_base`` / ``selectors``: the selector grammar (``tl.label`` and friends).
* ``module_alias_guard`` / ``resolver``: resolving selectors to sites on a trace.
* ``selection``: ``Selection``, its set algebra and provenance; ``align``, ``values``,
  ``compare``, ``graph``, ``subspace``, ``do_plan``: operations on a resolved selection.

Public doors keep their spellings: ``torchlens.intervention`` re-exports the selector
constructors and ``torchlens.intervention.types`` / ``.errors`` re-export the target
specs and refusal classes defined here.
"""
