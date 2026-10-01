"""The layer map: one declared stratum per package/module (C01 item 1).

The ten-layer model (architecture memo 3.1). A module's layer resolves
module attribute first (``__tl_layer__`` in the module), then the most
specific row here. ``FACADE`` is a ROLE, not a layer: thin, lazy,
logic-free modules whose deferred upward imports are the one legal upward
door (Rule F). The assurance plane (tests/) never appears here.

This table is CONTRACT DATA (org rule 11): the layer lint
(tests/test_arch_spine_layer_lint.py) derives the eager-upward inversion
baseline from it, and lane F35's placement-table coverage sweep audits it
against every memo build list. Spellings are DOCUMENTED-UNSTABLE pending
the naming sprint; the SEPARATIONS are the decision.
"""

from __future__ import annotations

__tl_layer__ = "L0"

#: Layer rank used by the lint: lower may never eagerly import higher.
LAYER_ORDER: dict[str, int] = {
    "L0": 0,
    "L1": 1,
    "L2": 2,
    "L3": 3,
    "L4": 4,
    "L5": 5,
    "L6": 6,
    "L7": 7,
    "L8": 8,
    "L9": 9,
}

#: Roles outside the numeric stack: FACADE modules take Rule F (deferred
#: upward imports legal); their EAGER imports are judged at the layer of
#: their most specific numbered ancestor row, or exempt when none applies.
LAYER_ROLES = frozenset({"FACADE"})

#: Most-specific-prefix layer map, keyed by dotted module path relative to
#: ``torchlens`` ('' = the package root). Module-granular rows encode the
#: per-module censuses the memo ruled (the _io four-way split, the fastlog
#: dispositions); package rows carry the rest.
PACKAGE_LAYERS: dict[str, str] = {
    # -- FACADE-role modules (Rule F) ------------------------------------
    "": "FACADE",
    "types": "FACADE",
    "io": "FACADE",
    "user_funcs": "FACADE",
    "experimental": "FACADE",
    # quickstart is a FACADE-role package (lane F17): its verbs sit over the
    # capture engine, the inference search, and the renderer, with upward
    # imports deferred per Rule F; its interior modules declare their own
    # __tl_layer__ (plan/grammar L2, resolver/gate L3).
    "quickstart": "FACADE",
    # -- L0 BASIS ---------------------------------------------------------
    "_errors": "L0",
    "errors": "L0",
    "_vocab": "L0",
    "_literals": "L0",
    "constants": "L0",
    "quantities": "L0",
    "_architecture": "L0",
    "_deprecations": "L0",
    # -- L1 PRODUCT -------------------------------------------------------
    "data_classes": "L1",
    "_trace_core": "L1",
    "schemas": "L1",
    "accessors": "L1",
    "_registry": "L1",
    "hash": "L1",
    "_io": "L1",
    "_io.format_errors": "L0",
    "_io.bundle": "L3",
    "_io.scrub": "L3",
    "_io.tlspec": "L3",
    "_io.rehydrate": "L3",
    "_io.lazy": "L3",
    "_io.runnable": "L3",
    "_io.runnable_load": "L3",
    "_io.accessor_rebuild": "L3",
    "_io.forgery_validation": "L3",
    # W051-IO: the AUD-CODE 3.0 load validators split out of forgery_validation
    # (R43 size cap) sit at ITS tier; the module imports its shared refusal/
    # census helpers back from it eagerly.
    "_io._load_validators": "L3",
    # -- L2 ENGINE ----------------------------------------------------------
    "capture": "L2",
    "backends": "L2",
    "ir": "L2",
    "fastlog": "L2",
    "fastlog.types": "L1",
    "fastlog.recover": "L3",
    "fastlog.cleanup": "L3",
    "options": "L2",
    "observers": "L2",
    "_state": "L2",
    "kernel_telemetry": "L2",
    "_options_validation": "L2",
    "_option_receipt": "L2",
    "_save_budget": "L2",
    "_distributed": "L2",
    "_model_wrappers": "L2",
    "_robustness": "L2",
    "_input_coerce": "L2",
    "_input_walk": "L2",
    "_chunking": "L2",
    "_chunked_capture_helpers": "L2",
    "_capture_state_helpers": "L2",
    "_capture_fingerprint": "L2",
    "utils": "L1",
    # -- L3 SETTLE ----------------------------------------------------------
    "postprocess": "L3",
    "_data_substrate": "L3",
    "validation": "L3",
    "merged": "L3",
    "partial": "L3",
    "distributed": "L3",
    "compat": "L3",
    "captured_run": "L3",
    "_fast_run": "L3",
    "runnable": "L3",
    "_transport": "L3",
    "_split_rebind": "L3",
    "_selection_align": "L4",
    "_source_links": "L3",
    "_training_validation": "L3",
    "_user_public_impls": "L3",
    # -- L4 ALGEBRA -----------------------------------------------------------
    "intervention": "L4",
    "bundle": "L4",
    "selection": "L4",
    "selection_values": "L4",
    "selection_compare": "L4",
    "selection_graph": "L4",
    "selection_subspace": "L4",
    "trace_slice": "L4",
    "facets": "L4",
    "autoroute": "L4",
    "transforms": "L4",
    # -- L5 READS ------------------------------------------------------------
    "stats": "L5",
    "observability": "L5",
    "receptive_field": "L5",
    "attribution": "L5",
    "repgeom": "L5",
    "repgeom._trace_views": "L6",
    "repgeom._node_visuals": "L6",
    "dataset_extraction": "L5",
    "_extraction": "L5",
    "debug": "L5",
    "checks": "L5",
    # -- L6 APPLIANCES ---------------------------------------------------------
    "semantic": "L6",
    "notebook": "L6",
    "neuro": "L6",
    # -- L7 PRESENT --------------------------------------------------------------
    "visualization": "L7",
    "viz": "L7",
    "report": "L7",
    "export": "L7",
    "export._trackers": "L8",
    "export._graphs": "L8",
    # -- L8 BRIDGES ----------------------------------------------------------------
    "bridge": "L8",
    "callbacks": "L8",
    "trackers": "L8",
    # -- L9 RECIPES -------------------------------------------------------------------
    "examples": "L9",
}

#: Packages licensed to touch torch privates: direct ``torch._*`` use or
#: consumption of the sanctioned torch-compat chokepoint
#: (``torchlens/utils/_torch_compat.py``). The license is a PER-MODULE
#: contract orthogonal to layer (memo 3.2); this count is printed by the
#: lint and may only SHRINK -- a new package touching torch privates
#: without a row here is RED. Target: burn down toward engine- and
#: persistence-adjacent packages only (measured census 2026-08-27, C01
#: item 12).
TORCH_PRIVATE_LICENSED_PACKAGES: frozenset[str] = frozenset(
    {
        # F04 one-backward reads: attribution/onebackward gates on the
        # HAS_GRADIENT_EDGE / HAS_NODE_PREHOOK capability flags, which live
        # at the ONE sanctioned probe chokepoint (utils/_torch_compat); the
        # package holds no direct torch._ touches of its own.
        "attribution",
        # L8 floor fix: receptive_field's multi-axis any() reduction routes
        # through tensor_any_over_dims() at the ONE sanctioned probe
        # chokepoint (utils/_torch_compat); the package holds no direct
        # torch._ touches of its own.
        "receptive_field",
        # F27 Kineto join + memory-parity oracle: observability consumes the
        # HAS_KINETO_INMEMORY_EVENTS / HAS_KINETO_EVENT_SCOPE /
        # HAS_MEMORY_PROFILE capability flags and their accessors, all living
        # at the ONE sanctioned probe chokepoint (utils/_torch_compat); the
        # package holds no direct torch._ touches of its own.
        "observability",
        "backends",
        "capture",
        "compat",
        "constants",
        "data_classes",
        "debug",
        "distributed",
        "_distributed",
        "errors",
        "fastlog",
        "intervention",
        "ir",
        "_io",
        "kernel_telemetry",
        "merged",
        "postprocess",
        "runnable",
        "utils",
        "validation",
        "_capture_state_helpers",
        "_robustness",
        "_runnable_call_outputs",
        "_runnable_execution",
        "_runnable_input_metadata",
        "_runnable_state",
        "_runnable_state_context",
        "_runnable_verification",
        "_runnable_witness_contracts",
        "_training_validation",
        "_fast_run",
        "user_funcs",
        # floor2 fix (2026-10-01): the runnable-save CPU float8 path
        # consumes get_cpu_float8_deterministic_fill_support() at the ONE
        # sanctioned probe chokepoint (utils/_torch_compat); the module
        # holds no direct torch._ touches of its own.
        "_user_public_impls",
    }
)


def layer_for_module(dotted: str) -> str | None:
    """Resolve one torchlens-relative module path to its declared layer.

    Module ``__tl_layer__`` attributes take precedence at runtime; this
    function is the static-table half used by AST lints (which also read
    the attribute from source).

    Parameters
    ----------
    dotted:
        Module path relative to ``torchlens`` (e.g. ``"fastlog.recover"``).

    Returns
    -------
    str | None
        The most specific declared layer or role, or ``None`` when no row
        covers the module (a lint finding, not a default).
    """

    probe = dotted
    while True:
        if probe in PACKAGE_LAYERS:
            return PACKAGE_LAYERS[probe]
        if "." not in probe:
            return PACKAGE_LAYERS.get("")
        probe = probe.rsplit(".", 1)[0]
