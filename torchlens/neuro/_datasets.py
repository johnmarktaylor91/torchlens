"""``torchlens.neuro.datasets``: the per-site rsatoolbox Dataset handoff (F22).

Memo 4.1 (item 5): descriptors are the product. rsatoolbox promotes Dataset
descriptors into RDM descriptors automatically (measured at 0.1.5 and
0.3.2), so everything written here survives into the user's figures --
preventing unauditable RDM stacks is the point, not the lines saved.

Every spelling is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._handoff import (
    PRESENTATION_INDEX_KEY,
    SiteDatasets,
    SitePayload,
    _source_kind,
    base_descriptors,
    check_pool,
    enumerate_extraction_sites,
    enumerate_trace_sites,
    require_rsatoolbox,
    resolve_stimulus_ids,
    validate_obs,
    widen_for_handoff,
)


def datasets(
    source: Any,
    sites: Any = None,
    *,
    pool: Any = None,
    obs: Any = None,
    stimulus_ids: Any = None,
) -> SiteDatasets:
    """Convert stimulus-indexed sites into per-site rsatoolbox Datasets.

    Parameters
    ----------
    source:
        A TorchLens ``Trace`` with saved activations, a
        ``LoadedExtraction``, or an extraction-artifact directory path
        (delegates to ``load_extraction``; a detached artifact resolves
        stored output keys only, and refusals say so).
    sites:
        ``None`` sweeps every stimulus-indexed saved site through the core
        enumeration -- never every saved tensor; ineligible sites are
        skipped into the ledger with one summarized disclosure. An explicit
        lookup (string or iterable) REFUSES ineligible sites actionably,
        and a bare multi-pass layer lookup refuses naming the
        pass-qualified alternatives.
    pool:
        Readout/pooling request. Launch vocabulary: ``None`` only --
        disclosed flattening, always recorded as ``pool="flatten"``.
    obs:
        Optional mapping of observation-descriptor name to per-stimulus
        sequence (conditions, runs, repeats); every column is
        length-checked before rsatoolbox receives it.
    stimulus_ids:
        Explicit per-row identifiers. On an extraction artifact the
        recorded ids are authoritative and this list acts as validation;
        on a bare Trace it is accepted after exact length validation.
        Absent both, a positional index is used and DISCLOSED as
        synthetic. Repeated ids are valid (they mean condition averaging
        downstream).

    Returns
    -------
    SiteDatasets
        Insertion-ordered mapping of canonical site key to
        ``rsatoolbox.data.Dataset``, carrying the per-site sweep ledger on
        ``.ledger``. Every Dataset always contains the presentation-order
        observation descriptor ``tl_presentation_index`` (rsatoolbox's
        ``calc_rdm(descriptor=...)`` sorts rows; this index is measured to
        be permuted along with the data and therefore fully recoverable).

    Raises
    ------
    ImportError
        When rsatoolbox is not installed.
    NeuroHandoffError
        Typed refusals: unsupported source, unavailable pool vocabulary,
        mis-lengthed ``obs=`` columns or ``stimulus_ids=``, id
        disagreement with a recorded artifact, ambiguous multi-pass
        lookup.
    ValueError
        Core refusals for explicitly requested ineligible or unsaved
        sites.
    """

    rsatoolbox = require_rsatoolbox()
    pool_token = check_pool(pool)
    del pool_token  # recorded per-site by base_descriptors; validated here.
    kind = _source_kind(source)

    trace = source if kind == "trace" else None
    if kind == "trace":
        payloads, ledger = enumerate_trace_sites(source, sites, verb="neuro.datasets")
    else:
        payloads, ledger, _loaded = enumerate_extraction_sites(source, sites, verb="neuro.datasets")

    result = SiteDatasets()
    for payload in payloads:
        result[payload.key] = _build_dataset(
            rsatoolbox,
            payload,
            source_kind=kind,
            trace=trace,
            obs=obs,
            stimulus_ids=stimulus_ids,
        )
    result.ledger = tuple(ledger)
    return result


def _build_dataset(  # noqa: PLR0913 -- mirrors the datasets() door's knob set verbatim (one pass-through worker)
    rsatoolbox: Any,
    payload: SitePayload,
    *,
    source_kind: str,
    trace: Any | None,
    obs: Any,
    stimulus_ids: Any,
) -> Any:
    """Build one descriptor-complete rsatoolbox Dataset from a payload.

    Parameters
    ----------
    rsatoolbox:
        The imported rsatoolbox package.
    payload:
        Shaped site payload with identity facts.
    source_kind:
        ``"trace"`` or ``"extraction"``.
    trace:
        Source trace on the in-memory route, else ``None``.
    obs:
        User observation-descriptor mapping (validated per site).
    stimulus_ids:
        Explicit per-row identifiers (resolved per the authority rule).

    Returns
    -------
    Any
        ``rsatoolbox.data.Dataset`` with the full identity story.
    """

    matrix, cast_record = widen_for_handoff(payload.matrix)
    n_rows, n_features = int(matrix.shape[0]), int(matrix.shape[1])

    ids, identity = resolve_stimulus_ids(stimulus_ids, payload.row_ids, n_rows, site=payload.key)
    obs_descriptors: dict[str, Any] = {
        "stimulus_id": np.asarray(ids),
        "tl_stimulus_identity": np.asarray([identity] * n_rows),
        # ALWAYS written (memo D10): the original-row-index descriptor that
        # survives rsatoolbox's condition sorting.
        PRESENTATION_INDEX_KEY: np.arange(n_rows),
    }
    for name, column in validate_obs(obs, n_rows, site=payload.key).items():
        obs_descriptors[name] = np.asarray(column)

    # A neutral flat feature index -- never "neuroid": an artificial unit is
    # not a biological one (memo 4.1). Unravelled coordinates ride along
    # where the recorded per-stimulus shape makes them factual (the
    # searchlight plumbing).
    channel_descriptors: dict[str, Any] = {"feature_index": np.arange(n_features)}
    shape = payload.per_stimulus_shape
    if shape and int(np.prod(shape)) == n_features:
        coordinates = np.unravel_index(np.arange(n_features), shape)
        for axis, coordinate in enumerate(coordinates):
            channel_descriptors[f"unit_coord_{axis}"] = coordinate

    descriptors = base_descriptors(payload, source_kind=source_kind, trace=trace)
    if cast_record is not None:
        descriptors["handoff_cast"] = cast_record

    return rsatoolbox.data.Dataset(
        measurements=matrix.numpy(),
        descriptors=descriptors,
        obs_descriptors=obs_descriptors,
        channel_descriptors=channel_descriptors,
    )


def single_dataset(source: Any, site: str | None = None) -> Any:
    """Build ONE rsatoolbox Dataset (the legacy bridge's delegation target).

    The single-Dataset special case of :func:`datasets`, kept so
    ``torchlens.bridge.rsatoolbox.dataset`` shares one flattening and
    descriptor code path with the plural constructor (memo D2). The
    historical silent flatten is now the disclosed ``pool="flatten"``
    descriptor, the ``"neuroid"`` channel label is retired (an artificial
    unit is not a biological one), and presentation-order recoverability is
    preserved: the always-written ``tl_presentation_index`` observation
    descriptor supersedes the accidental integer ``presentation`` column,
    which is still written for existing readers.

    Parameters
    ----------
    source:
        Trace (in-memory route), ``LoadedExtraction``, or extraction
        directory path (file route).
    site:
        One explicit site lookup, or ``None`` for the historical
        final-output behavior (Trace route only).

    Returns
    -------
    Any
        One ``rsatoolbox.data.Dataset``.

    Raises
    ------
    ImportError
        When rsatoolbox is not installed.
    NeuroHandoffError
        ``neuro_source_invalid`` when ``site=None`` is combined with a
        file-route source (an artifact stores no final-output marker).
    ValueError
        Core refusals for an explicitly requested ineligible or unsaved
        site, and the historical no-tensor-output error.
    """

    from ._handoff import NeuroHandoffError, _shape_payload, _source_kind

    rsatoolbox = require_rsatoolbox()
    kind = _source_kind(source)

    if site is None:
        if kind != "trace":
            raise NeuroHandoffError(
                "site=None selects the final model output, which only a "
                "live Trace carries; extraction artifacts store explicit "
                "output keys.",
                code="neuro_source_invalid",
                remedy="pass site=<stored output key> for extraction artifacts",
            )
        out = _final_trace_output(source)
        matrix, per_stimulus_shape = _shape_payload(out)
        payload = SitePayload(
            key="<final_output>",
            requested="<final_output>",
            matrix=matrix,
            per_stimulus_shape=per_stimulus_shape,
            source_dtype=str(out.dtype).removeprefix("torch."),
            layer_label="<final_output>",
            pass_index=None,
            site_key="unavailable(final_output_alias)",
            row_ids=None,
            input_preprocessing_verdict=None,
        )
        trace = source
    else:
        if kind == "trace":
            payloads, _ledger = enumerate_trace_sites(
                source, [str(site)], verb="bridge.rsatoolbox.dataset"
            )
            trace = source
        else:
            payloads, _ledger, _loaded = enumerate_extraction_sites(
                source, [str(site)], verb="bridge.rsatoolbox.dataset"
            )
            trace = None
        payload = payloads[0]

    built = _build_dataset(
        rsatoolbox,
        payload,
        source_kind=kind,
        trace=trace,
        obs=None,
        stimulus_ids=None,
    )
    # Legacy compat column: the accidental integer "presentation" descriptor
    # is what has been hiding the sorting trap in shipped code; keep it for
    # existing readers while tl_presentation_index is the taught spelling.
    built.obs_descriptors["presentation"] = np.arange(int(built.measurements.shape[0]))
    return built


def _final_trace_output(log: Any) -> Any:
    """Return the first final output tensor of a trace (legacy behavior).

    Parameters
    ----------
    log:
        TorchLens ``Trace``.

    Returns
    -------
    Any
        Final output tensor.

    Raises
    ------
    ValueError
        If no tensor output activation is available (the historical
        message, preserved).
    """

    import torch

    for label in getattr(log, "output_layers", []) or []:
        out = getattr(log[label], "out", None)
        if isinstance(out, torch.Tensor):
            return out
    for layer in reversed(getattr(log, "layer_list", [])):
        out = getattr(layer, "out", None)
        if getattr(layer, "is_output", False) and isinstance(out, torch.Tensor):
            return out
    raise ValueError("Could not find a tensor output out for rsatoolbox export.")


__all__ = ["datasets", "single_dataset"]
