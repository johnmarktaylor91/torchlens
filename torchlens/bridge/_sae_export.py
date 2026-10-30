"""Write SAE latents and top examples in the formats SAEDashboard and delphi read.

Two writers, one per consumer, each taking the latents of one batch (a dense
``[examples, sequence, latents]`` tensor such as ``SpliceResult.latents``) or a
per-latent top-k over a whole dataset (the ``UnitTopK`` that
``tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)`` returns from
``tl.aggregate``), together with the token ids of the examples:

- ``write_delphi`` writes the latent cache delphi's ``LatentDataset`` loads
  (eai-delphi 0.1.3): one folder per site holding ``{start}_{end}.safetensors``
  splits with ``locations``, ``activations`` and ``tokens``, and a
  ``config.json``. The layout copies ``delphi/latents/cache.py``
  (``LatentCache.save_splits`` and ``save_config``) and is read by
  ``delphi/latents/loader.py`` (``TensorBuffer.load``, ``LatentDataset``).
- ``write_sae_dashboard`` writes the JSON that SAEDashboard's
  ``SaeVisData.load_json`` reads (sae-dashboard 0.8.0,
  ``sae_dashboard/sae_vis_data.py``): ``feature_data_dict`` with each
  latent's top-activation sequences and ``feature_stats``, renderable with
  ``sae_dashboard.data_writing_fns.save_feature_centric_vis``.

Neither consumer package is imported here: the files are written with the
standard library and ``safetensors``. Every spelling is DOCUMENTED-UNSTABLE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .._errors import InvalidArgumentError

#: delphi's own nonzero test when it caches latents
#: (``delphi/latents/cache.py``, ``get_nonzeros_batch``: ``batch.abs() > 1e-5``).
_DELPHI_NONZERO = 1e-5

#: SAEDashboard's nonzero test for ``frac_nonzero``
#: (``sae_dashboard/utils_fns.py``, ``FeatureStatistics.create``).
_DASHBOARD_NONZERO = 1e-6

#: SAEDashboard rounds sequence activations to four decimals
#: (``sae_dashboard/sequence_data_generator.py``).
_DASHBOARD_DECIMALS = 4

#: Estimated JSON size above which ``write_sae_dashboard`` warns once per session.
_DASHBOARD_WARN_BYTES = 256 * 1024**2

_DOCS = "docs/field-guide/sae_latent_export.md"


def _is_unit_topk(latents: Any) -> bool:
    """Return whether ``latents`` is a ``tl.stats`` per-unit top-k result."""

    from ..stats._per_unit import UnitTopK

    return isinstance(latents, UnitTopK)


def _check_tokens(writer: str, tokens: Any) -> torch.Tensor:
    """Return ``tokens`` as an ``int64 [examples, sequence]`` CPU tensor or refuse."""

    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 2
        or tokens.dtype.is_floating_point
        or tokens.dtype == torch.bool
    ):
        shape = tuple(tokens.shape) if isinstance(tokens, torch.Tensor) else type(tokens).__name__
        raise InvalidArgumentError(
            f"{writer} needs integer token ids shaped [examples, sequence]; got {shape}",
            code="sae_export_tokens_invalid",
            remedy=f"pass the input_ids the model ran on, one row per example; see {_DOCS}",
        )
    return tokens.detach().to(device="cpu", dtype=torch.long)


def _dense_latents(writer: str, latents: Any, tokens: torch.Tensor) -> torch.Tensor:
    """Return dense latents as ``float64 [examples, sequence, latents]`` or refuse."""

    if not isinstance(latents, torch.Tensor) or latents.ndim not in (2, 3):
        shape = tuple(latents.shape) if isinstance(latents, torch.Tensor) else type(latents)
        raise InvalidArgumentError(
            f"{writer} takes latents shaped [examples, sequence, latents] (or [examples, "
            f"latents]) or a tl.stats.TopK(..., unit_axis=-1) result; got {shape}",
            code="sae_export_latents_invalid",
            remedy="pass SpliceResult.latents, sae.encode(activations), or the UnitTopK that "
            f"tl.aggregate returns for tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode); see {_DOCS}",
        )
    dense = latents.detach().to(device="cpu", dtype=torch.float64)
    if dense.ndim == 2:
        dense = dense.unsqueeze(1)
    if tuple(dense.shape[:2]) != tuple(tokens.shape):
        raise InvalidArgumentError(
            f"{writer}: latents cover {tuple(dense.shape[:2])} [examples, sequence] but tokens "
            f"are {tuple(tokens.shape)}",
            code="sae_export_tokens_invalid",
            remedy="pass the token ids of exactly the examples the latents were read from",
        )
    return dense


def _topk_entries(
    writer: str, top: Any, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(example, position, latent, value)`` of every retained top-k entry."""

    positions = top.positions
    if positions.ndim != 3 or positions.shape[-1] > 1:
        raise InvalidArgumentError(
            f"{writer} reads one sequence axis; this top-k has {int(positions.shape[-1])} "
            "position axes",
            code="sae_export_latents_invalid",
            remedy="aggregate a token-level site shaped [examples, sequence, features] with "
            "tl.stats.TopK(k, unit_axis=-1)",
        )
    units, k = (int(size) for size in top.values.shape)
    latent = torch.arange(units).repeat_interleave(k)
    example = top.example_ids.reshape(-1)
    position = positions[..., 0].reshape(-1) if positions.shape[-1] else torch.zeros_like(example)
    value = top.values.reshape(-1).to(torch.float64)
    kept = (example >= 0) & ~value.isnan()
    example, position, latent, value = example[kept], position[kept], latent[kept], value[kept]
    if example.numel() and (
        int(example.max()) >= tokens.shape[0] or int(position.max()) >= tokens.shape[1]
    ):
        raise InvalidArgumentError(
            f"{writer}: the top-k names example {int(example.max())} and position "
            f"{int(position.max())}, outside tokens shaped {tuple(tokens.shape)}",
            code="sae_export_tokens_invalid",
            remedy="pass the token ids of the whole dataset in the order tl.aggregate saw it "
            "(row i is example id i)",
        )
    return example, position, latent, value


def _delphi_entries(
    writer: str, latents: Any, tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return delphi ``locations [n, 3]``, ``activations [n]`` and the latent width."""

    if _is_unit_topk(latents):
        example, position, latent, value = _topk_entries(writer, latents, tokens)
        width = int(latents.values.shape[0])
        kept = value.abs() > _DELPHI_NONZERO
        locations = torch.stack([example[kept], position[kept], latent[kept]], dim=1)
        return locations, value[kept], width
    dense = _dense_latents(writer, latents, tokens)
    locations = torch.nonzero(dense.abs() > _DELPHI_NONZERO)
    activations = dense[locations[:, 0], locations[:, 1], locations[:, 2]]
    return locations, activations, int(dense.shape[-1])


def _location_array(locations: torch.Tensor) -> Any:
    """Return delphi's location array: ``uint16`` when every column fits, else ``uint32``."""

    import numpy as np

    array = locations.numpy()
    if array.size == 0 or int(array.max()) < 2**16:
        return array.astype(np.uint16)
    return array.astype(np.uint32)


def write_delphi(  # noqa: PLR0913 -- the two subjects plus the cache's where (site), how (model name, splits) and output (directory)
    latents: Any,
    tokens: torch.Tensor,
    *,
    site: str,
    model_name: str,
    n_splits: int = 5,
    directory: str | Path,
) -> Path:
    """Write SAE latents as the latent cache delphi's ``LatentDataset`` loads.

    The layout is eai-delphi 0.1.3's own (``delphi/latents/cache.py``,
    ``LatentCache.save_splits`` and ``save_config``): ``directory/site/``
    holds one ``{start}_{end}.safetensors`` file per latent split (``end``
    inclusive, the splits ``torch.linspace(0, width, n_splits + 1)``) with
    ``locations`` (``[n, 3]`` rows of example, position, latent minus
    ``start``; ``uint16`` when every value fits, else ``uint32``),
    ``activations`` (``float16``, as delphi stores them) and ``tokens``
    (``[examples, sequence]``), plus ``config.json`` with delphi's cache
    fields and ``model_name``. delphi reads it with
    ``LatentDataset(directory, sampler_cfg, constructor_cfg, tokenizer=...,
    modules=[site])``.

    Parameters
    ----------
    latents:
        Dense latents ``[examples, sequence, latents]`` (or ``[examples,
        latents]``, one position), for example ``SpliceResult.latents``:
        every activation with absolute value above ``1e-5`` is written,
        delphi's own threshold. Or the ``UnitTopK`` of
        ``tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)`` over a
        dataset: only the retained top-``k`` activations per latent are
        written, so delphi's per-latent frequencies count those.
    tokens:
        Integer token ids ``[examples, sequence]``; row ``i`` is example
        ``i`` (the example ids a ``UnitTopK`` names).
    site:
        The site the latents were read at; it names the cache folder and is
        the module name delphi lists (``modules=[site]``).
    model_name:
        Hugging Face id of the model, which delphi uses to load the
        tokenizer when none is passed.
    n_splits:
        Number of latent splits, delphi's ``CacheConfig.n_splits``.
    directory:
        Cache root (delphi's ``raw_dir``); created if missing.

    Returns
    -------
    Path
        The site folder written.

    Raises
    ------
    InvalidArgumentError
        If ``latents`` is neither shape (``sae_export_latents_invalid``) or
        ``tokens`` is not an integer ``[examples, sequence]`` tensor that
        covers them (``sae_export_tokens_invalid``).
    """

    from safetensors.numpy import save_file

    writer = "tl.bridge.sae.write_delphi"
    token_ids = _check_tokens(writer, tokens)
    locations, activations, width = _delphi_entries(writer, latents, token_ids)
    folder = Path(directory) / str(site)
    folder.mkdir(parents=True, exist_ok=True)
    boundaries = torch.linspace(0, width, steps=int(n_splits) + 1).long()
    for start, stop in zip(boundaries[:-1].tolist(), (boundaries[1:] - 1).tolist(), strict=True):
        mask = (locations[:, 2] >= start) & (locations[:, 2] <= stop)
        split = locations[mask].clone()
        split[:, 2] -= start
        save_file(
            {
                "locations": _location_array(split),
                "activations": activations[mask].to(torch.float16).numpy(),
                "tokens": token_ids.numpy(),
            },
            str(folder / f"{start}_{stop}.safetensors"),
        )
    config = {
        "dataset_repo": "",
        "dataset_split": "",
        "dataset_name": "",
        "dataset_column": "",
        "batch_size": int(token_ids.shape[0]),
        "cache_ctx_len": int(token_ids.shape[1]),
        "n_tokens": int(token_ids.numel()),
        "n_splits": int(n_splits),
        "model_name": str(model_name),
    }
    (folder / "config.json").write_text(json.dumps(config, indent=4), encoding="utf-8")
    return folder


def _dashboard_topk(dense: torch.Tensor, k: int) -> Any:
    """Return the per-latent top-``k`` of dense latents, ranked as ``tl.stats.TopK`` ranks."""

    from ..stats._per_unit import TopK

    tracker = TopK(k, unit_axis=-1)
    tracker.update(dense)
    return tracker.result()


def _warn_large_dashboard(latents: int, k: int, sequence: int) -> None:
    """Warn once per session when the dashboard JSON would pass ``_DASHBOARD_WARN_BYTES``."""

    from .sae import _warn_once

    code = "sae_export_dashboard_large"
    estimate = latents * k * sequence * 24
    if estimate <= _DASHBOARD_WARN_BYTES:
        return
    _warn_once(
        code,
        f"write_sae_dashboard writes {latents} latents x {k} sequences x {sequence} tokens, "
        f"about {estimate / 1024**2:.0f} MiB of JSON, above the "
        f"{_DASHBOARD_WARN_BYTES // 1024**2} MiB guard; every latent is written. This warning "
        f"fires once per session; see {_DOCS}. Remedy: write_sae_dashboard(..., "
        "latent_ids=[0, 1, 2]) for the latents you will open, or a smaller k=",
        "write_sae_dashboard",
        stacklevel=4,
    )


def _sequence_record(
    example: int, position: int, token_row: list[int], acts: list[float]
) -> dict[str, Any]:
    """Return one SAEDashboard ``SequenceData`` dict (its field names, its defaults)."""

    return {
        "original_index": example,
        "qualifying_token_index": position,
        "token_ids": token_row,
        "feat_acts": acts,
        "loss_contribution": [],
        "token_logits": [0.0] * len(token_row),
        "top_token_ids": [],
        "top_logits": [],
        "bottom_token_ids": [],
        "bottom_logits": [],
    }


def _feature_record(sequences: list[dict[str, Any]], top_value: float) -> dict[str, Any]:
    """Return one SAEDashboard ``FeatureData`` dict holding the top-activation group."""

    title = f"TOP ACTIVATIONS<br>MAX = {top_value:.3f}"
    return {
        "sequence_data": {"seq_group_data": [{"title": title, "seq_data": sequences}]},
    }


def _resolve_latent_ids(latent_ids: Any, top: Any) -> list[int]:
    """Return the latents to write: the given ids, else every latent that fired."""

    units = int(top.values.shape[0])
    if latent_ids is None:
        fired = (top.values.nan_to_num(0.0).abs() > _DASHBOARD_NONZERO).any(dim=1)
        return [int(index) for index in torch.nonzero(fired).flatten().tolist()]
    ids = [int(index) for index in latent_ids]
    bad = [index for index in ids if not 0 <= index < units]
    if bad:
        raise InvalidArgumentError(
            f"write_sae_dashboard(latent_ids=...) names {bad[:5]}, outside the {units} latents",
            code="sae_export_latents_invalid",
            remedy=f"pass latent indices in range({units})",
        )
    return ids


def write_sae_dashboard(  # noqa: PLR0913 -- the two subjects plus what (latent ids, k) and output (path)
    latents: Any,
    tokens: torch.Tensor,
    *,
    latent_ids: Any = None,
    k: int = 20,
    path: str | Path,
) -> Path:
    """Write SAE latents and top examples as the JSON SAEDashboard loads.

    The file is what ``SaeVisData.save_json`` writes in sae-dashboard 0.8.0
    (``sae_dashboard/sae_vis_data.py``): ``feature_data_dict`` maps each
    latent to its ``FeatureData``, whose ``sequence_data`` holds one
    "TOP ACTIVATIONS" group of up to ``k`` sequences ranked by activation
    (ties toward the lowest example, then position, as ``tl.stats.TopK``
    ranks), and ``feature_stats`` holds each latent's ``max`` and
    ``frac_nonzero`` when the latents are dense. Load it with
    ``SaeVisData.load_json(path, cfg=SaeVisConfig(hook_point=..., features=...),
    model=..., encoder=...)`` and render with ``save_feature_centric_vis``;
    whole sequences are written, so render with
    ``SequencesConfig(buffer=None)``, which bolds the maximum token.
    Components TorchLens does not compute (logit tables and histograms,
    activation histograms, loss effects) keep SAEDashboard's empty
    defaults, and each token's ``token_logits`` entry is 0.0.

    Parameters
    ----------
    latents:
        Dense latents ``[examples, sequence, latents]`` (or ``[examples,
        latents]``), for example ``SpliceResult.latents``: each sequence
        carries the latent's activation at every token. Or the ``UnitTopK``
        of ``tl.stats.TopK(k, unit_axis=-1, encoder=sae.encode)`` over a
        dataset: each sequence carries the ranked token's activation and
        0.0 elsewhere, as SAEDashboard's own ``compute_buffer=False``
        display does.
    tokens:
        Integer token ids ``[examples, sequence]``; row ``i`` is example
        ``i``.
    latent_ids:
        Latents to write; default every latent that fired (an activation
        with absolute value above ``1e-6``).
    k:
        Sequences per latent for dense latents (SAEDashboard's
        ``top_acts_group_size`` default, 20); a ``UnitTopK`` keeps its own
        ``k``.
    path:
        Output ``.json`` file.

    Returns
    -------
    Path
        The file written.

    Raises
    ------
    InvalidArgumentError
        If ``latents`` is neither shape or ``latent_ids`` names a latent out
        of range (``sae_export_latents_invalid``), or ``tokens`` does not
        cover the latents (``sae_export_tokens_invalid``).
    """

    writer = "tl.bridge.sae.write_sae_dashboard"
    token_ids = _check_tokens(writer, tokens)
    dense: torch.Tensor | None = None
    if _is_unit_topk(latents):
        top = latents
        _topk_entries(writer, top, token_ids)
    else:
        dense = _dense_latents(writer, latents, token_ids)
        top = _dashboard_topk(dense, int(k))
    chosen = _resolve_latent_ids(latent_ids, top)
    _warn_large_dashboard(len(chosen), int(top.values.shape[1]), int(token_ids.shape[1]))
    rows = token_ids.tolist()
    features: dict[str, Any] = {}
    stats: dict[str, list[Any]] = {"max": [], "frac_nonzero": []}
    for unit in chosen:
        sequences = []
        for value, example, position in zip(
            top.values[unit].tolist(),
            top.example_ids[unit].tolist(),
            _positions(top, unit),
            strict=True,
        ):
            if example < 0 or math.isnan(value):
                continue
            if dense is not None:
                acts = [round(v, _DASHBOARD_DECIMALS) for v in dense[example, :, unit].tolist()]
            else:
                acts = [0.0] * len(rows[example])
                acts[position] = round(value, _DASHBOARD_DECIMALS)
            sequences.append(_sequence_record(example, position, rows[example], acts))
        features[str(unit)] = _feature_record(sequences, float(top.values[unit, 0].nan_to_num(0.0)))
        if dense is not None:
            column = dense[..., unit].reshape(-1)
            stats["max"].append(float(column.max()))
            stats["frac_nonzero"].append(float((column.abs() > _DASHBOARD_NONZERO).double().mean()))
    if dense is None:
        stats = {"max": [], "frac_nonzero": []}
    payload = {
        "feature_data_dict": features,
        "feature_stats": {
            **stats,
            "skew": [],
            "kurtosis": [],
            "quantile_data": [],
            "quantiles": [],
        },
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _positions(top: Any, unit: int) -> list[int]:
    """Return the sequence position of each of ``unit``'s top-k entries (0 without one)."""

    positions = top.positions[unit]
    if positions.shape[-1] == 0:
        return [0] * int(positions.shape[0])
    return [int(item) for item in positions[:, 0].tolist()]
