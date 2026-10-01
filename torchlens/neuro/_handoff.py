"""Shared handoff machinery for the ``torchlens.neuro`` treaty desk (F22).

One enumeration + shaping + provenance substrate consumed by both public
verbs (:func:`torchlens.neuro.datasets`, :func:`torchlens.neuro.rdms`) and
by the legacy ``bridge.rsatoolbox.dataset`` delegation. Site eligibility is
the CORE stimulus-indexed gate (``torchlens.repgeom._annotation_gate``,
lane A11) -- the neuro package never re-derives "which sites index
stimuli"; it consumes the one evidence order (record kind first, defended
shape equality second) that stopped the fabricated buffer pseudo-RDMs
(neuro memo D4/D5). Shaping is the ONE recorded feature-matrix operation
(:mod:`torchlens.features`, tvscope B7) -- no hand-rolled flatten enters an
rsatoolbox handoff.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from .._errors import _actionable_message, _ActionableErrorMixin
from ..errors._base import ConfigurationError

#: Observation-descriptor key of the ALWAYS-written original-row-index
#: descriptor (neuro memo D10): rsatoolbox's ``calc_rdm(descriptor=...)``
#: returns rows in SORTED descriptor order, so presentation order must ride
#: a descriptor of our own that is permuted along with the data (measured
#: recoverable at rsatoolbox 0.1.5 and 0.3.2). rsatoolbox's own ``index``
#: descriptor is NOT trusted: at 0.3.2 it is a post-sort arange, at 0.1.5
#: it does not exist.
PRESENTATION_INDEX_KEY = "tl_presentation_index"

#: Descriptor value recorded when no pooling/readout was applied (memo D3:
#: the readout is scientific provenance and is ALWAYS recorded; disclosed
#: flattening is the honest launch default).
FLATTEN_POOL = "flatten"


class NeuroHandoffError(_ActionableErrorMixin, ConfigurationError, ValueError):
    """Typed refusal from the neuro handoff surface (memo 4.1/4.2).

    Codes: ``neuro_pool_vocabulary_unavailable``,
    ``neuro_obs_descriptor_invalid``, ``neuro_stimulus_ids_mismatch``,
    ``neuro_site_ambiguous``, ``neuro_source_invalid``,
    ``neuro_sites_inconsistent_rows``, ``neuro_rdms_mode_conflict``,
    ``neuro_matrix_invalid``.
    """

    def __init__(self, problem: str, *, code: str, remedy: str, **context: object) -> None:
        """Initialize a typed handoff refusal.

        Parameters
        ----------
        problem:
            What made the handoff unsafe.
        code:
            Stable machine-readable refusal code.
        remedy:
            Concrete caller action that resolves the refusal.
        **context:
            Structured diagnostic context.
        """

        super().__init__(
            _actionable_message(problem, remedy),
            code=code,
            remedy=remedy,
            **cast(dict[str, Any], context),
        )


@dataclass(frozen=True)
class SiteOutcome:
    """One per-site ledger row of a neuro sweep (memo D5/D6).

    Attributes
    ----------
    site:
        Canonical site key of the row.
    outcome:
        ``"computed"`` or ``"skipped"`` (explicit requests never skip --
        an ineligible explicit request raises instead).
    reason:
        The evidence-naming skip reason, or ``None`` when computed.
    """

    site: str
    outcome: str
    reason: str | None = None


class SiteDatasets(OrderedDict):
    """Insertion-ordered site -> Dataset mapping carrying its sweep ledger.

    The per-site report is exposed explicitly from day one on new neuro
    functions (memo D6): ``.ledger`` is a tuple of :class:`SiteOutcome`
    rows covering every considered site (computed AND skipped). The class
    is a plain ``OrderedDict`` subclass measured pickle/copy/deepcopy-safe
    with the attribute intact and ``==``-equal to the plain mapping.
    """

    ledger: tuple[SiteOutcome, ...] = ()


def require_rsatoolbox() -> Any:
    """Import and return rsatoolbox, teaching the extra when absent.

    Returns
    -------
    Any
        The imported ``rsatoolbox`` package.

    Raises
    ------
    ImportError
        When rsatoolbox is not installed; the message names the exact
        extra spelling.
    """

    try:
        import rsatoolbox
    except ImportError as exc:
        raise ImportError(
            'torchlens.neuro requires rsatoolbox: install it with `pip install "torchlens[neuro]"`.'
        ) from exc
    return rsatoolbox


def rsatoolbox_version() -> str:
    """Return the installed rsatoolbox version via package metadata.

    Neither validated rsatoolbox version (0.1.5 floor, 0.3.x current)
    exposes ``__version__`` (measured), so the recorded version comes from
    ``importlib.metadata``.

    Returns
    -------
    str
        The installed distribution version, or ``"unknown"`` when metadata
        is unavailable (e.g. a path-injected source tree).
    """

    import importlib.metadata

    try:
        return importlib.metadata.version("rsatoolbox")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def check_pool(pool: Any) -> str:
    """Validate the ``pool=`` argument against the launch vocabulary.

    neuro mints no pooling vocabulary of its own (memo D3): readout presets
    belong to the shared extraction/readout layer, which has not shipped a
    vocabulary yet, so the only accepted value is ``None`` -> disclosed
    flattening. The choice is ALWAYS recorded in descriptors.

    Parameters
    ----------
    pool:
        User-supplied pooling/readout request.

    Returns
    -------
    str
        The recorded pool descriptor value (``"flatten"``).

    Raises
    ------
    NeuroHandoffError
        ``neuro_pool_vocabulary_unavailable`` for any non-``None`` value.
    """

    if pool is None:
        return FLATTEN_POOL
    raise NeuroHandoffError(
        f"pool={pool!r} is not available: the shared extraction/readout "
        "preset vocabulary has not shipped, and torchlens.neuro deliberately "
        "mints no pooling spellings of its own (the readout choice is "
        "scientific provenance -- on a real ViT, flatten vs CLS vs mean-patch "
        "readouts of the SAME site agree at only ~0.33-0.38 second-order).",
        code="neuro_pool_vocabulary_unavailable",
        remedy=(
            "pass pool=None (disclosed flattening, recorded as pool='flatten') "
            "and apply readouts upstream at extraction time"
        ),
        pool=repr(pool),
    )


def widen_for_handoff(matrix: torch.Tensor) -> tuple[torch.Tensor, str | None]:
    """Apply the handoff dtype rule and report the cast (memo D12).

    Captured dtypes pass through; ``float16``/``bfloat16`` widen to
    ``float32`` (never ``float64`` -- float64 doubles peak handoff memory
    while changing real RDMs by exactly zero, measured). Every cast is
    recorded in descriptors.

    Parameters
    ----------
    matrix:
        The shaped 2-D feature matrix.

    Returns
    -------
    tuple[torch.Tensor, str | None]
        The (possibly widened) matrix and a ``"<from>-><to>"`` cast record,
        or ``None`` when no cast was applied.
    """

    if matrix.dtype in (torch.float16, torch.bfloat16):
        source = str(matrix.dtype).removeprefix("torch.")
        return matrix.to(torch.float32), f"{source}->float32"
    return matrix, None


def validate_obs(
    obs: Any,
    n_rows: int,
    *,
    site: str,
) -> dict[str, Any]:
    """Validate user condition/run/repeat observation descriptors (memo 4.1).

    A misaligned condition vector is the panel's highest-severity failure
    class: plausible, scientifically wrong, invisible -- so every column is
    length-checked BEFORE rsatoolbox receives it.

    Parameters
    ----------
    obs:
        Mapping of observation-descriptor name to per-stimulus sequence, or
        ``None``.
    n_rows:
        Stimulus-row count of the site being handed off.
    site:
        Site key, named in refusals.

    Returns
    -------
    dict[str, Any]
        Validated name -> list columns (empty when ``obs`` is ``None``).

    Raises
    ------
    NeuroHandoffError
        ``neuro_obs_descriptor_invalid`` on a non-mapping, a reserved-name
        collision, or a mis-lengthed column.
    """

    if obs is None:
        return {}
    if not hasattr(obs, "items"):
        raise NeuroHandoffError(
            f"obs= must be a mapping of descriptor name to per-stimulus "
            f"sequence; got {type(obs).__name__}.",
            code="neuro_obs_descriptor_invalid",
            remedy="pass e.g. obs={'condition': [...one entry per stimulus...]}",
            site=site,
        )
    validated: dict[str, Any] = {}
    for key, column in obs.items():
        name = str(key)
        if name == PRESENTATION_INDEX_KEY:
            raise NeuroHandoffError(
                f"obs descriptor name {name!r} is reserved: it is the "
                "always-written presentation-order index that makes sorted "
                "rsatoolbox outputs recoverable.",
                code="neuro_obs_descriptor_invalid",
                remedy="rename the descriptor column",
                site=site,
            )
        values = list(column)
        if len(values) != n_rows:
            raise NeuroHandoffError(
                f"obs descriptor {name!r} has {len(values)} entries for "
                f"{n_rows} stimulus rows at site {site!r}; a misaligned "
                "condition vector is plausible, scientifically wrong, and "
                "invisible downstream.",
                code="neuro_obs_descriptor_invalid",
                remedy="pass exactly one entry per stimulus row, in row order",
                site=site,
                descriptor=name,
                n_entries=len(values),
                n_rows=n_rows,
            )
        validated[name] = values
    return validated


def resolve_stimulus_ids(
    stimulus_ids: Any,
    recorded_ids: list[str] | None,
    n_rows: int,
    *,
    site: str,
) -> tuple[list[str], str]:
    """Resolve row identity per the authority rule (memo 4.1).

    Manifest-recorded ids are authoritative on an artifact: an explicitly
    supplied list must MATCH them and acts as validation, never silent
    replacement. Explicit ids are accepted on a bare Trace after exact
    length validation. With neither, a positional arange is allowed only
    under a descriptor disclosing that identity is synthetic. Repeated ids
    are valid (condition averaging is the downstream meaning).

    Parameters
    ----------
    stimulus_ids:
        Explicit per-row identifiers, or ``None``.
    recorded_ids:
        Ids recorded by the source artifact, or ``None`` (bare Trace).
    n_rows:
        Stimulus-row count of the site.
    site:
        Site key, named in refusals.

    Returns
    -------
    tuple[list[str], str]
        The resolved ids and the identity provenance token:
        ``"recorded"``, ``"user_validated"``, ``"user_supplied"``, or
        ``"synthetic_positional"``.

    Raises
    ------
    NeuroHandoffError
        ``neuro_stimulus_ids_mismatch`` on a length mismatch or a
        disagreement with recorded ids.
    """

    explicit = None if stimulus_ids is None else [str(item) for item in stimulus_ids]
    if explicit is not None and len(explicit) != n_rows:
        raise NeuroHandoffError(
            f"{len(explicit)} stimulus ids were supplied for {n_rows} "
            f"stimulus rows at site {site!r}.",
            code="neuro_stimulus_ids_mismatch",
            remedy="pass exactly one identifier per stimulus row, in row order",
            site=site,
            n_ids=len(explicit),
            n_rows=n_rows,
        )
    if recorded_ids is not None:
        if explicit is not None and explicit != list(recorded_ids):
            mismatches = [
                index
                for index, (given, recorded) in enumerate(zip(explicit, recorded_ids, strict=False))
                if given != recorded
            ][:3]
            raise NeuroHandoffError(
                f"explicit stimulus_ids disagree with the ids recorded by the "
                f"extraction artifact (first differing rows: {mismatches}); "
                "recorded ids are authoritative and an explicit list acts as "
                "validation, never silent replacement.",
                code="neuro_stimulus_ids_mismatch",
                remedy=(
                    "drop stimulus_ids= to use the recorded ids, or re-extract "
                    "with the intended ids"
                ),
                site=site,
                first_mismatch_rows=mismatches,
            )
        return list(recorded_ids), "recorded" if explicit is None else "user_validated"
    if explicit is not None:
        return explicit, "user_supplied"
    return [str(index) for index in range(n_rows)], "synthetic_positional"


def _source_kind(source: Any) -> str:
    """Classify a handoff source as ``"trace"`` or ``"extraction"``.

    Parameters
    ----------
    source:
        Trace, ``LoadedExtraction``, or extraction-directory path.

    Returns
    -------
    str
        The source kind token.

    Raises
    ------
    NeuroHandoffError
        ``neuro_source_invalid`` for unsupported source types.
    """

    from ..dataset_extraction import LoadedExtraction

    if isinstance(source, (LoadedExtraction, str, Path)):
        return "extraction"
    if hasattr(source, "layers") and hasattr(source, "layer_logs"):
        return "trace"
    raise NeuroHandoffError(
        f"unsupported source type {type(source).__name__}: expected a "
        "TorchLens Trace, a LoadedExtraction, or an extraction-directory "
        "path.",
        code="neuro_source_invalid",
        remedy=(
            "capture with tl.trace(...) or extract to disk with "
            "tl.extract_dataset(..., output_dir=...) and pass the directory"
        ),
        source_type=type(source).__name__,
    )


@dataclass(frozen=True)
class SitePayload:
    """One eligible site's shaped payload plus identity facts.

    Attributes
    ----------
    key:
        Canonical, collision-free site key (pass-qualified for multi-pass
        selections, memo D13).
    requested:
        The lookup the caller typed (equals ``key`` on default sweeps).
    matrix:
        The shaped 2-D "stimuli x features" tensor (pre-widening).
    per_stimulus_shape:
        Original per-stimulus shape (input shape minus the leading axis).
    source_dtype:
        Captured payload dtype string.
    layer_label:
        Aggregate layer label of the site.
    pass_index:
        1-based pass index, or ``None`` for single-pass layers.
    site_key:
        The portable structural site key when derivable, else a
        ``"unavailable(<reason>)"`` disclosure.
    row_ids:
        Recorded per-row ids when the source carries them, else ``None``.
    input_preprocessing_verdict:
        The artifact's input-preprocessing verdict when carried, else
        ``None``.
    """

    key: str
    requested: str
    matrix: torch.Tensor
    per_stimulus_shape: tuple[int, ...]
    source_dtype: str
    layer_label: str
    pass_index: int | None
    site_key: str
    row_ids: list[str] | None
    input_preprocessing_verdict: str | None


def _record_site_key(record: Any) -> str:
    """Read a record's portable structural site key, disclosing failures.

    Parameters
    ----------
    record:
        Layer or op record.

    Returns
    -------
    str
        The ``site_key`` value, or ``"unavailable(<reason>)"`` -- keyless
        legacy artifacts and within-call-recurrence layer ambiguity refuse
        typed upstream, and the handoff records the fact instead of
        fabricating an identity.
    """

    from ..errors._base import TorchLensError

    try:
        value = getattr(record, "site_key", None)
    except (TorchLensError, AttributeError, ValueError) as exc:
        return f"unavailable({getattr(exc, 'fields', {}).get('code', type(exc).__name__)})"
    return str(value) if value else "unavailable(site_key_missing)"


def _shape_payload(payload: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Run the ONE recorded shaping op over a site payload.

    Parameters
    ----------
    payload:
        Saved activation tensor (leading axis = stimuli).

    Returns
    -------
    tuple[torch.Tensor, tuple[int, ...]]
        The 2-D matrix and the original per-stimulus shape.
    """

    from ..features import as_matrix

    matrix, record = as_matrix(payload.detach().cpu())
    return matrix, tuple(record.input_shape[1:])


def _trace_site_payload(record: Any, key: str, requested: str, verb: str) -> SitePayload:
    """Build a :class:`SitePayload` from one eligible trace record.

    Parameters
    ----------
    record:
        Single-pass layer or pass-qualified op with a saved payload.
    key:
        Canonical site key for the handoff mapping.
    requested:
        The caller's typed lookup.
    verb:
        Public verb name used in diagnostics.

    Returns
    -------
    SitePayload
        The shaped payload with identity facts.
    """

    from ..repgeom._annotation_gate import _raise_unsaved_activation

    out = getattr(record, "out", None) if getattr(record, "has_saved_activation", False) else None
    if not isinstance(out, torch.Tensor):
        _raise_unsaved_activation(key, verb=verb)
    matrix, per_stimulus_shape = _shape_payload(cast(torch.Tensor, out))
    num_passes = int(getattr(record, "num_passes", 1))
    pass_index = getattr(record, "pass_index", None)
    return SitePayload(
        key=key,
        requested=requested,
        matrix=matrix,
        per_stimulus_shape=per_stimulus_shape,
        source_dtype=str(cast(torch.Tensor, out).dtype).removeprefix("torch."),
        layer_label=str(getattr(record, "layer_label", key)),
        pass_index=int(pass_index) if num_passes > 1 and pass_index is not None else None,
        site_key=_record_site_key(record),
        row_ids=None,
        input_preprocessing_verdict=None,
    )


def enumerate_trace_sites(
    trace: Any,
    sites: Any,
    *,
    verb: str,
) -> tuple[list[SitePayload], list[SiteOutcome]]:
    """Enumerate eligible trace sites through the CORE gate (memo 4.4).

    Default sweeps (``sites=None``) cover every stimulus-indexed saved
    site -- single-pass layers under their layer label and multi-pass
    layers as pass-qualified canonical keys automatically (memo D13: a
    CORnet-S user wants ``V1:1..V1:T``, not a refusal) -- skipping
    ineligible sites into the ledger with ONE summarized disclosure.
    Explicit requests refuse actionably: an ineligible site raises the
    core evidence-naming error, and a bare multi-pass layer lookup raises
    naming the pass-qualified alternatives.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    sites:
        ``None`` for the default sweep, or an iterable of lookups (a
        single string is accepted).
    verb:
        Public verb name used in diagnostics.

    Returns
    -------
    tuple[list[SitePayload], list[SiteOutcome]]
        Eligible shaped payloads and the full per-site ledger.

    Raises
    ------
    ValueError
        Core refusals for explicitly requested ineligible or unsaved
        sites.
    NeuroHandoffError
        ``neuro_site_ambiguous`` for a bare multi-pass layer lookup.
    """

    from ..repgeom._annotation_gate import (
        _expected_stimulus_counts,
        _raise_ineligible_site,
        _site_ineligibility,
    )

    expected_counts = _expected_stimulus_counts(trace)
    if sites is None:
        return _default_trace_sweep(trace, expected_counts, verb=verb)

    payloads: list[SitePayload] = []
    ledger: list[SiteOutcome] = []
    requested_list = [sites] if isinstance(sites, str) else list(sites)
    seen: set[str] = set()
    for requested in requested_list:
        lookup = str(requested)
        record = trace[lookup]
        layer_label = str(getattr(record, "layer_label", lookup))
        if int(getattr(record, "num_passes", 1)) > 1 and not _is_pass_qualified(record):
            alternatives = [str(op.label) for op in getattr(record, "ops", {}).values()]
            raise NeuroHandoffError(
                f"site {lookup!r} names a recurrent layer with "
                f"{int(getattr(record, 'num_passes', 0))} passes; a bare layer "
                "lookup is ambiguous across passes.",
                code="neuro_site_ambiguous",
                remedy=f"request a pass-qualified site: {alternatives}",
                site=lookup,
                alternatives=alternatives,
            )
        reason = _site_ineligibility(record, expected_counts)
        if reason is not None:
            _raise_ineligible_site(verb, _safe_label(record) or lookup, reason)
        # Canonical key rule (memo D13): pass-qualified keys ONLY for
        # multi-pass layers; single-pass sites use the bare layer label so
        # explicit requests, default sweeps, and the repgeom verbs agree.
        if int(getattr(record, "num_passes", 1)) > 1:
            key = _safe_label(record) or layer_label
        else:
            key = layer_label
        if key in seen:
            continue
        seen.add(key)
        payloads.append(_trace_site_payload(record, key, lookup, verb))
        ledger.append(SiteOutcome(key, "computed"))
    return payloads, ledger


def _default_trace_sweep(
    trace: Any,
    expected_counts: frozenset[int],
    *,
    verb: str,
) -> tuple[list[SitePayload], list[SiteOutcome]]:
    """Run the ``sites=None`` default sweep over every saved trace site.

    Single-pass layers enter under their layer label; multi-pass layers
    emit pass-qualified canonical keys automatically, each pass gated
    independently (memo D13). Ineligible sites skip into the ledger with
    ONE summarized disclosure.

    Parameters
    ----------
    trace:
        Captured TorchLens trace.
    expected_counts:
        Leading-axis sizes of the capture's input sites.
    verb:
        Public verb name used in diagnostics.

    Returns
    -------
    tuple[list[SitePayload], list[SiteOutcome]]
        Eligible shaped payloads and the full per-site ledger.
    """

    from ..repgeom._annotation_gate import _site_ineligibility, _warn_skipped_sites

    payloads: list[SitePayload] = []
    ledger: list[SiteOutcome] = []
    skipped: OrderedDict[str, str] = OrderedDict()
    for layer in trace.layers:
        layer_label = str(layer.layer_label)
        saved_ops = [
            op for op in layer.ops.values() if bool(getattr(op, "has_saved_activation", False))
        ]
        if not saved_ops:
            continue
        multi_pass = int(getattr(layer, "num_passes", 1)) > 1
        # Multi-pass layers contribute each saved pass as its own site;
        # single-pass layers contribute the aggregate layer once.
        candidates = (
            [(op, str(op.label)) for op in saved_ops] if multi_pass else [(layer, layer_label)]
        )
        for record, key in candidates:
            reason = _site_ineligibility(record, expected_counts)
            if reason is not None:
                skipped[key] = reason
                ledger.append(SiteOutcome(key, "skipped", reason))
                continue
            payloads.append(_trace_site_payload(record, key, key, verb))
            ledger.append(SiteOutcome(key, "computed"))
    _warn_skipped_sites(verb, skipped)
    return payloads, ledger


def _safe_label(record: Any) -> str | None:
    """Read a record's op label without tripping the multi-pass guard.

    Parameters
    ----------
    record:
        Resolved trace record (op, single-pass layer, or aggregate
        multi-pass layer -- whose varying-field tripwire raises on plain
        ``label`` reads).

    Returns
    -------
    str | None
        The single op label, or ``None`` when the record aggregates
        several passes.
    """

    from ..utils._multipass_access import get_multipass_attr

    label = get_multipass_attr(record, "label", None, multipass=None)
    return str(label) if label is not None else None


def _is_pass_qualified(record: Any) -> bool:
    """Return whether a resolved record addresses one pass, not a group.

    Parameters
    ----------
    record:
        Resolved trace record.

    Returns
    -------
    bool
        ``True`` for pass-qualified op records; ``False`` for aggregate
        multi-pass layers.
    """

    label = _safe_label(record)
    return label is not None and label != str(getattr(record, "layer_label", ""))


def enumerate_extraction_sites(
    source: Any,
    sites: Any,
    *,
    verb: str,
) -> tuple[list[SitePayload], list[SiteOutcome], Any]:
    """Enumerate eligible extraction-artifact sites (memo 4.4 file route).

    Evidence order for the file route: the manifest's declared stimulus
    count and axis semantics are FIRST-CLASS evidence (batch axis 0 by the
    artifact contract), so a stored output whose leading axis disagrees
    with the manifest's ``n_stimuli`` is skipped (default sweep) or refused
    (explicit request). A detached artifact resolves stored output keys
    only, and refusals say so.

    Parameters
    ----------
    source:
        ``LoadedExtraction`` or extraction-directory path.
    sites:
        ``None`` for every stored output key, or an iterable of keys.
    verb:
        Public verb name used in diagnostics.

    Returns
    -------
    tuple[list[SitePayload], list[SiteOutcome], Any]
        Eligible payloads, the per-site ledger, and the loaded artifact.

    Raises
    ------
    torchlens.features.FeatureShapingError
        For an explicitly requested key absent from the artifact.
    ValueError
        For an explicitly requested stored output that is not
        stimulus-indexed.
    """

    from ..dataset_extraction import LoadedExtraction, load_extraction
    from ..features import site_matrix
    from ..repgeom._annotation_gate import _raise_ineligible_site, _warn_skipped_sites

    loaded = source if isinstance(source, LoadedExtraction) else load_extraction(source)
    manifest_n = (loaded.manifest.get("stimulus_provenance") or {}).get("n_stimuli")
    expected_rows = int(manifest_n) if manifest_n else None

    if sites is None:
        requested_keys = list(loaded.activations)
        explicit = False
    else:
        requested_keys = [sites] if isinstance(sites, str) else [str(key) for key in sites]
        explicit = True

    payloads: list[SitePayload] = []
    ledger: list[SiteOutcome] = []
    skipped: OrderedDict[str, str] = OrderedDict()
    for key in requested_keys:
        shaped = site_matrix(loaded, key)
        n_rows = int(shaped.matrix.shape[0])
        if expected_rows is not None and n_rows != expected_rows:
            reason = (
                f"the stored output's leading axis is {n_rows} but the "
                f"artifact manifest records {expected_rows} stimuli"
            )
            if explicit:
                _raise_ineligible_site(verb, key, reason)
            skipped[key] = reason
            ledger.append(SiteOutcome(key, "skipped", reason))
            continue
        # A carried preprocessing block with no verdict field reads as
        # "unknown" (the legacy bridge's spelling), never a silent omission.
        verdict = (
            (shaped.input_preprocessing or {}).get("verdict", "unknown")
            if shaped.input_preprocessing is not None
            else None
        )
        payloads.append(
            SitePayload(
                key=key,
                requested=key,
                matrix=shaped.matrix,
                per_stimulus_shape=tuple(shaped.record.input_shape[1:]),
                source_dtype=str(shaped.matrix.dtype).removeprefix("torch."),
                layer_label=key,
                pass_index=None,
                site_key=str(
                    ((loaded.manifest.get("sites") or {}).get(key) or {}).get("site_key")
                    or "unavailable(site_key_missing)"
                ),
                row_ids=shaped.row_ids,
                input_preprocessing_verdict=str(verdict) if verdict is not None else None,
            )
        )
        ledger.append(SiteOutcome(key, "computed"))
    if not explicit:
        _warn_skipped_sites(verb, skipped)
    return payloads, ledger, loaded


def base_descriptors(
    payload: SitePayload, *, source_kind: str, trace: Any | None
) -> dict[str, Any]:
    """Build the shared per-site Dataset/RDM descriptor block (memo 4.1).

    Parameters
    ----------
    payload:
        The shaped site payload.
    source_kind:
        ``"trace"`` or ``"extraction"``.
    trace:
        The source trace for trace-route facts (model identity,
        intervention marker), or ``None`` on the file route.

    Returns
    -------
    dict[str, Any]
        Descriptor mapping: identity story first, provenance facts after.
    """

    import torchlens

    from ..features import SHAPING_OP

    descriptors: dict[str, Any] = {
        "source": "torchlens",
        "site": payload.key,
        "requested": payload.requested,
        "site_key": payload.site_key,
        "layer_label": payload.layer_label,
        "pool": FLATTEN_POOL,
        # The ONE recorded shaping operation (torchlens.features): the
        # legacy bridge published these two keys and readers consume them.
        "shaping": SHAPING_OP,
        "source_kind": source_kind,
        # A STRING spelling: rsatoolbox's descriptor promotion treats any
        # non-string iterable as a per-element column and refuses a length
        # mismatch, so a tuple/list here would break calc_rdm on our own
        # Datasets (measured at 0.3.2).
        "per_stimulus_shape": str(tuple(payload.per_stimulus_shape)),
        "source_dtype": payload.source_dtype,
        "tl_version": str(getattr(torchlens, "__version__", "unknown")),
        "rsatoolbox_version": rsatoolbox_version(),
    }
    if payload.pass_index is not None:
        descriptors["pass_index"] = payload.pass_index
    if trace is not None:
        model_name = getattr(trace, "model_class_name", None)
        if model_name:
            # The reserved multi-model slot (memo section 8): one factual
            # model identity per Dataset today, a stack key tomorrow.
            descriptors["model"] = str(model_name)
        if getattr(trace, "intervention_audit", None):
            descriptors["tl_intervened"] = True
    if payload.input_preprocessing_verdict is not None:
        descriptors["input_preprocessing_verdict"] = payload.input_preprocessing_verdict
    return descriptors
