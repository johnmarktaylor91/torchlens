"""Sparse random projection: two constructions, one honest contract (memo B6).

``srp(n_components=REQUIRED, seed=0, density="auto",
construction="very_sparse_fixed", mode="flatten", share="by_extent")``.

- The default ``very_sparse_fixed`` construction holds EXACTLY
  ``m = round(density * D)`` distinct nonzeros per output column via
  deterministic rejection (realized nnz EQUALS ``k * m``), with hash-derived
  signs keyed to positions and values ``+/- 1/sqrt(density * k)``. It is NOT
  the i.i.d. Bernoulli matrix of the JL literature (measured per-column
  nonzero variance 0.41-0.53 against the hypothesis' 44-316), so it records
  ``distance_claim="empirical_only"`` at every density — the empirical
  oracle suite is its entire warrant (T-C13: the citation is selected WITH
  the construction).
- ``iid_bernoulli`` IS the i.i.d. matrix (O1f proves the distribution
  identity) but ALSO records ``empirical_only`` until the section 11
  literature task verifies Achlioptas 2003 against the paper itself; its
  O(D*k) generation cost rides the plan's disclosures.
- ``share="by_extent"`` (default): equal-width sites share one projection —
  determinism over statistics (a matrix keyed on a user's dict label means
  renaming a key changes your numbers). ``by_site`` adds the stable
  structural site key; when only the output-key label is available the
  weaker source is DISCLOSED in the realized facts.
- Flatten-mode SRP binds ONE input extent at first use per (chain, site)
  and REFUSES drift: projecting different rows through DIFFERENT matrices
  as a function of ``batch_size`` was the measured D-5 laundering (values
  moved 132-153% with two of six rows bit-identical).

Every spelling here is DOCUMENTED-UNSTABLE pending the naming sprint.
"""

from __future__ import annotations

import hashlib
import math
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch

from ._context import TransformContext
from ._errors import TransformContractError
from ._registry import _register_builtin
from ._spec import (
    PlannedStep,
    TensorSpec,
    TransformDefinition,
    TransformSpec,
    canonical_json,
    freeze_params,
)
from ._srp_hash import (
    ALGORITHM_VERSION,
    MatrixHeader,
    digest_fixed_matrix,
    generate_bernoulli_columns,
    generate_fixed_columns,
)

__tl_layer__ = "L4"

__all__ = ["srp", "srp_realized_facts", "srp_verify_matrix"]

#: Closed construction vocabulary (memo decision 7; T-C13).
_CONSTRUCTIONS: tuple[str, ...] = ("very_sparse_fixed", "iid_bernoulli")

#: Closed projection-mode vocabulary.
_MODES: tuple[str, ...] = ("flatten", "features")

#: Closed matrix-sharing vocabulary (memo decision 5).
_SHARES: tuple[str, ...] = ("by_extent", "by_site")

#: Dense materialization ceiling (bytes) below which the planner picks the
#: plain dense path; above it, sparse_csr (CPU/CUDA) or dense_chunked.
_DENSE_PATH_BYTES = 64 * 1024 * 1024

#: Column-chunk byte target for the dense_chunked path.
_CHUNK_BYTES = 32 * 1024 * 1024

#: Matrix cache bounds (entries and total bytes) — bounded, LRU-evicted.
_CACHE_MAX_ENTRIES = 8
_CACHE_MAX_BYTES = 128 * 1024 * 1024

#: Extent-binding registry bound (drift protection for the D-5 laundering).
_BINDINGS_MAX_ENTRIES = 4096

_HALF_DTYPES = (torch.float16, torch.bfloat16)


class _MatrixCache:
    """Bounded LRU cache of generated matrices (positions + signs on CPU)."""

    def __init__(self, max_entries: int, max_bytes: int) -> None:
        """Initialize the cache.

        Parameters
        ----------
        max_entries:
            Entry-count bound.
        max_bytes:
            Total-byte bound over cached index/sign tensors.
        """

        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._entries: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
        self._bytes = 0

    @staticmethod
    def _entry_bytes(entry: dict[str, Any]) -> int:
        """Byte size of one cached entry's tensors."""

        total = 0
        for value in entry.values():
            if isinstance(value, torch.Tensor):
                total += value.numel() * value.element_size()
        return total

    def get(self, key: tuple[Any, ...]) -> dict[str, Any] | None:
        """Return a cached entry (refreshing recency) or ``None``."""

        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    def put(self, key: tuple[Any, ...], entry: dict[str, Any]) -> None:
        """Insert an entry, evicting LRU entries past either bound."""

        if key in self._entries:
            self._bytes -= self._entry_bytes(self._entries.pop(key))
        self._entries[key] = entry
        self._bytes += self._entry_bytes(entry)
        while self._entries and (
            len(self._entries) > self._max_entries or self._bytes > self._max_bytes
        ):
            _, evicted = self._entries.popitem(last=False)
            self._bytes -= self._entry_bytes(evicted)

    def clear(self) -> None:
        """Drop every cached entry."""

        self._entries.clear()
        self._bytes = 0


_MATRIX_CACHE = _MatrixCache(_CACHE_MAX_ENTRIES, _CACHE_MAX_BYTES)

#: First-use extent bindings: (spec INSTANCE id, site identity) -> extent.
#: Instance-keyed on purpose: a fresh spec (a fresh run) legitimately binds a
#: new extent, while ONE chain applied to heterogeneous SITES binds per site
#: (the D-3 mixed-rank sweep); only a ragged re-resolution of the SAME
#: (instance, site) is the D-5 laundering and refuses.
_EXTENT_BINDINGS: OrderedDict[tuple[int, str | None], int] = OrderedDict()


def _reset_srp_state() -> None:
    """Test door: drop the matrix cache and every extent binding."""

    _MATRIX_CACHE.clear()
    _EXTENT_BINDINGS.clear()


def _purge_bindings(spec_id: int) -> None:
    """GC hook: drop every binding row for a collected spec instance."""

    for key in [key for key in _EXTENT_BINDINGS if key[0] == spec_id]:
        del _EXTENT_BINDINGS[key]


def _srp_normalize(params: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize srp params.

    Parameters
    ----------
    params:
        Raw params (``n_components``, ``density``, ``construction``,
        ``mode``, ``share``).

    Returns
    -------
    dict[str, Any]
        Normalized canonical-JSON-portable params.
    """

    n_components = params.get("n_components")
    if isinstance(n_components, bool) or not isinstance(n_components, int) or n_components < 1:
        raise TransformContractError(
            f"srp(n_components={n_components!r}) needs a positive integer output "
            "width; there is no default — srp_dims_for() is the labeled sizing "
            "heuristic that feeds it.",
            code="transform_params_invalid",
            remedy="pass n_components as a positive int (srp_dims_for can size it)",
            n_components=n_components,
        )
    density = params.get("density", "auto")
    if density != "auto":
        if isinstance(density, bool) or not isinstance(density, (int, float)):
            raise TransformContractError(
                f"srp(density={density!r}) must be 'auto' or a float in (0, 1].",
                code="transform_params_invalid",
                remedy="pass density='auto' (-> 1/sqrt(D)) or an explicit float in (0, 1]",
                density=density,
            )
        density = float(density)
        if not (0.0 < density <= 1.0) or not math.isfinite(density):
            raise TransformContractError(
                f"srp(density={density!r}) is outside (0, 1].",
                code="transform_params_invalid",
                remedy="pass a density in (0, 1]",
                density=density,
            )
    construction = params.get("construction", "very_sparse_fixed")
    if construction not in _CONSTRUCTIONS:
        raise TransformContractError(
            f"srp(construction={construction!r}) is not in the closed vocabulary "
            f"{_CONSTRUCTIONS}; a construction change is never a silent "
            "substitution (T-C13).",
            code="transform_params_invalid",
            remedy="pass construction='very_sparse_fixed' or 'iid_bernoulli'",
            construction=construction,
        )
    mode = params.get("mode", "flatten")
    if mode not in _MODES:
        raise TransformContractError(
            f"srp(mode={mode!r}) is not in the closed vocabulary {_MODES}.",
            code="transform_params_invalid",
            remedy="pass mode='flatten' (project everything per row) or 'features' (last axis)",
            mode=mode,
        )
    share = params.get("share", "by_extent")
    if share not in _SHARES:
        raise TransformContractError(
            f"srp(share={share!r}) is not in the closed vocabulary {_SHARES}.",
            code="transform_params_invalid",
            remedy="pass share='by_extent' (default) or 'by_site'",
            share=share,
        )
    return {
        "n_components": n_components,
        "density": density,
        "construction": str(construction),
        "mode": str(mode),
        "share": str(share),
    }


def _resolve_extent(spec: TransformSpec, input_spec: TensorSpec) -> int:
    """Resolve the projected input extent ``D`` from a concrete input spec.

    Parameters
    ----------
    spec:
        The srp spec.
    input_spec:
        Incoming tensor description.

    Returns
    -------
    int
        The projected extent (flatten: product of per-stimulus extents;
        features: the last-axis extent).
    """

    params = spec.params_dict()
    shape = input_spec.shape
    if len(shape) < 2:
        raise TransformContractError(
            f"srp needs at least one per-stimulus axis; got rank {len(shape)}.",
            code="transform_plan_invalid",
            remedy="project tensors of rank >= 2 (the stimulus axis plus features)",
            rank=len(shape),
        )
    if params["mode"] == "features":
        last = shape[-1]
        if last is None:
            raise TransformContractError(
                "srp(mode='features') cannot bind an UNKNOWN last-axis extent; "
                "the matrix is a function of the extent it projects.",
                code="transform_plan_invalid",
                remedy="plan with a concrete last-axis extent",
            )
        return int(last)
    tail = shape[1:]
    if any(extent is None for extent in tail):
        raise TransformContractError(
            "srp(mode='flatten') cannot bind UNKNOWN per-stimulus extents; the "
            "matrix is a function of the flattened extent it projects.",
            code="transform_plan_invalid",
            remedy="plan with concrete per-stimulus extents",
        )
    product = 1
    for tail_extent in tail:
        product *= int(tail_extent)  # type: ignore[arg-type]  # None refused above
    return product


def _resolve_density(spec: TransformSpec, extent: int) -> tuple[float, int]:
    """Resolve the requested density against a concrete extent.

    Parameters
    ----------
    spec:
        The srp spec.
    extent:
        Projected input extent ``D``.

    Returns
    -------
    tuple[float, int]
        ``(realized_density, nonzeros_per_column)`` with
        ``realized_density == m / D`` exactly.
    """

    params = spec.params_dict()
    requested = params["density"]
    density = 1.0 / math.sqrt(extent) if requested == "auto" else float(requested)
    m = max(1, round(density * extent))
    if m > extent:
        raise TransformContractError(
            f"srp density {density} at extent {extent} asks for {m} distinct "
            f"nonzeros per column, which exceeds the extent.",
            code="transform_plan_invalid",
            remedy="lower the density",
            extent=extent,
            nonzeros_per_column=m,
        )
    return m / extent, m


def _site_identity(spec: TransformSpec, ctx: TransformContext | None) -> tuple[str | None, str]:
    """Resolve the site identity used for by_site keying and drift binding.

    Parameters
    ----------
    spec:
        The srp spec (for refusal text).
    ctx:
        Optional context carrying site identity.

    Returns
    -------
    tuple[str | None, str]
        ``(site_value, source)`` where source is ``"site_key"``,
        ``"site_label"`` (the DISCLOSED weaker fallback), or ``"none"``.
    """

    if ctx is not None and ctx.site_key:
        return ctx.site_key, "site_key"
    if ctx is not None and ctx.site_label:
        return ctx.site_label, "site_label"
    return None, "none"


def _effective_seed(
    spec: TransformSpec,
    extent: int,
    m: int,
    ctx: TransformContext | None,
) -> tuple[int, str, str]:
    """Derive the effective seed key (memo section 6, sharing and seeding).

    The by_extent key is a canonical function of (base seed,
    algorithm_version, input extent, ordered projected roles where declared,
    output extent, realized nonzeros, share mode) — never a user-visible
    label, never the chain-step index. by_site ADDS the stable site
    identity, with the output-key fallback disclosed as the weaker source.

    Parameters
    ----------
    spec:
        The srp spec.
    extent:
        Projected input extent ``D``.
    m:
        Realized nonzeros per column.
    ctx:
        Optional context (roles + site identity).

    Returns
    -------
    tuple[int, str, str]
        ``(effective_seed, share_key_json, share_key_source)``.
    """

    params = spec.params_dict()
    roles: list[str] | None = None
    if ctx is not None and ctx.roles is not None:
        roles = list(ctx.roles.axes[1:]) if params["mode"] == "flatten" else [ctx.roles.axes[-1]]
    key: list[Any] = [
        "srp_effective_seed_v1",
        ALGORITHM_VERSION,
        params["construction"],
        spec.seed,
        extent,
        params["n_components"],
        m,
        params["share"],
        roles,
    ]
    site_value, site_source = _site_identity(spec, ctx)
    if params["share"] == "by_site":
        if site_value is None:
            raise TransformContractError(
                "srp(share='by_site') has no site identity to key the matrix "
                "on: the context carries neither a structural site key nor an "
                "output-key label; silently falling back to a shared matrix "
                "would contradict the declared sharing policy.",
                code="transform_site_identity_unavailable",
                remedy=(
                    "run through the extraction engine (which supplies the "
                    "site label), pass TransformContext(site_label=...), or "
                    "declare share='by_extent'"
                ),
            )
        key.append([site_source, site_value])
    else:
        site_source = "shared"
    key_json = canonical_json(key)
    digest = hashlib.sha256(key_json.encode("ascii")).digest()
    seed = int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF
    return seed, key_json, site_source


def _binding_key(spec: TransformSpec, ctx: TransformContext | None) -> tuple[int, str | None]:
    """Extent-binding registry key: (spec instance id, site identity)."""

    site_value, _ = _site_identity(spec, ctx)
    return id(spec), site_value


def _bind_extent(spec: TransformSpec, ctx: TransformContext | None, extent: int) -> None:
    """Bind the first-use extent and REFUSE drift (the D-5 guard).

    Parameters
    ----------
    spec:
        The srp spec.
    ctx:
        Optional context carrying site identity.
    extent:
        The concrete projected extent of this batch.

    Raises
    ------
    TransformContractError
        ``transform_extent_drift`` when a later batch resolves a different
        extent for the same (spec, site) — projecting different rows
        through different matrices as a function of batch composition is
        the measured D-5 laundering.
    """

    key = _binding_key(spec, ctx)
    bound = _EXTENT_BINDINGS.get(key)
    if bound is None:
        if not any(existing[0] == key[0] for existing in _EXTENT_BINDINGS):
            weakref.finalize(spec, _purge_bindings, key[0])
        _EXTENT_BINDINGS[key] = extent
        while len(_EXTENT_BINDINGS) > _BINDINGS_MAX_ENTRIES:
            _EXTENT_BINDINGS.popitem(last=False)
        return
    if bound != extent:
        raise TransformContractError(
            f"srp bound input extent {bound} at first use but this batch "
            f"resolves extent {extent}; projecting different rows through "
            "DIFFERENT matrices as a function of batch composition is the "
            "measured D-5 laundering (values moved 132-153% with a "
            "batch-size change), so extent drift refuses instead.",
            code="transform_extent_drift",
            remedy=(
                "normalize the shape first (a pooling step, or padding to a "
                "fixed width), use mode='features' when only trailing "
                "extents vary, or re-extract into a fresh run"
            ),
            bound_extent=bound,
            observed_extent=extent,
        )


def _generation_disclosures(spec: TransformSpec, extent: int, m: int) -> tuple[str, ...]:
    """Plan-time cost disclosures (the iid_bernoulli O(D*k) surprise killer)."""

    params = spec.params_dict()
    if params["construction"] == "iid_bernoulli":
        cells = extent * int(params["n_components"])
        return (
            f"iid_bernoulli generation is O(D*k) = {cells:,} hash draws "
            "(measured 12.8x-170x the fixed-construction cost, growing with "
            "D); it runs once and is cached",
        )
    return ()


def _srp_plan(
    spec: TransformSpec, input_spec: TensorSpec, ctx: TransformContext | None
) -> PlannedStep:
    """Plan srp: bind the extent, predict (batch, k) or (..., k) output.

    Parameters
    ----------
    spec:
        The srp spec.
    input_spec:
        Incoming tensor description.
    ctx:
        Optional context (roles, site identity).

    Returns
    -------
    PlannedStep
        Predicted output with generation-cost disclosures.
    """

    params = spec.params_dict()
    if input_spec.dtype not in (
        "torch.float16",
        "torch.bfloat16",
        "torch.float32",
        "torch.float64",
    ):
        raise TransformContractError(
            f"srp projects float tensors; got dtype {input_spec.dtype}.",
            code="transform_plan_invalid",
            remedy="cast to a float dtype first (chain a cast step)",
            dtype=input_spec.dtype,
        )
    extent = _resolve_extent(spec, input_spec)
    _density, m = _resolve_density(spec, extent)
    _bind_extent(spec, ctx, extent)
    k = int(params["n_components"])
    if params["mode"] == "features":
        out_shape = (*input_spec.shape[:-1], k)
    else:
        out_shape = (input_spec.shape[0], k)
    out_dtype = input_spec.dtype
    if input_spec.dtype in ("torch.float16", "torch.bfloat16"):
        out_dtype = "torch.float32"
    return PlannedStep(
        name=spec.name,
        version=spec.version,
        output=TensorSpec(shape=out_shape, dtype=out_dtype),
        stream_safe=True,
        may_alias=False,
        context_capable=True,
        disclosures=_generation_disclosures(spec, extent, m),
    )


def _multiply_path(extent: int, k: int, device: torch.device, workspace: int | None) -> str:
    """Choose the multiply path deterministically from recorded inputs.

    Parameters
    ----------
    extent:
        Input extent ``D``.
    k:
        Output extent.
    device:
        Device the batch lives on.
    workspace:
        Optional declared byte budget (``ctx.workspace``).

    Returns
    -------
    str
        ``"dense"`` | ``"sparse_csr"`` | ``"dense_chunked"``.
    """

    budget = _DENSE_PATH_BYTES if workspace is None else min(workspace, _DENSE_PATH_BYTES)
    if extent * k * 4 <= budget:
        return "dense"
    if device.type in ("cpu", "cuda"):
        return "sparse_csr"
    return "dense_chunked"


def _matrix_entry(spec: TransformSpec, extent: int, ctx: TransformContext | None) -> dict[str, Any]:
    """Generate (or fetch) the canonical matrix arrays for this spec+extent.

    Parameters
    ----------
    spec:
        The srp spec.
    extent:
        Projected input extent ``D``.
    ctx:
        Optional context (share keying).

    Returns
    -------
    dict[str, Any]
        Cached entry: CPU positions/signs (+ COO for bernoulli), scale,
        digest, effective seed, share key facts, realized density/nnz.
    """

    params = spec.params_dict()
    density, m = _resolve_density(spec, extent)
    seed, share_key, share_source = _effective_seed(spec, extent, m, ctx)
    k = int(params["n_components"])
    construction = str(params["construction"])
    cache_key = (construction, ALGORITHM_VERSION, seed, extent, k, m)
    cached = _MATRIX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if construction == "very_sparse_fixed":
        scale = 1.0 / math.sqrt(density * k)
        positions, signs = generate_fixed_columns(seed, extent, range(k), m)
        digest = digest_fixed_matrix(
            positions,
            signs,
            MatrixHeader(
                construction=construction,
                extent=extent,
                n_components=k,
                nonzeros_per_column=m,
                scale=scale,
            ),
        )
        entry: dict[str, Any] = {
            "construction": construction,
            "positions": positions,
            "signs": signs,
            "scale": scale,
            "digest": digest,
            "effective_seed": seed,
            "share_key": share_key,
            "share_key_source": share_source,
            "realized_density": density,
            "realized_nnz": k * m,
            "nonzeros_per_column": m,
            "extent": extent,
            "n_components": k,
        }
    else:
        density_requested = params["density"]
        bern_density = (
            1.0 / math.sqrt(extent) if density_requested == "auto" else float(density_requested)
        )
        col_ids, positions, signs = generate_bernoulli_columns(seed, extent, range(k), bern_density)
        scale = 1.0 / math.sqrt(bern_density * k)
        nnz = int(positions.numel())
        digest = digest_fixed_matrix(
            positions.reshape(1, -1),
            signs.reshape(1, -1),
            MatrixHeader(
                construction=construction,
                extent=extent,
                n_components=k,
                nonzeros_per_column=-1,
                scale=scale,
            ),
        )
        entry = {
            "construction": construction,
            "col_ids": col_ids,
            "positions": positions,
            "signs": signs,
            "scale": scale,
            "digest": digest,
            "effective_seed": seed,
            "share_key": share_key,
            "share_key_source": share_source,
            "realized_density": nnz / (extent * k) if extent * k else 0.0,
            "realized_nnz": nnz,
            "nonzeros_per_column": None,
            "extent": extent,
            "n_components": k,
        }
    _MATRIX_CACHE.put(cache_key, entry)
    return entry


def _coo_triplets(entry: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return flat (column_ids, positions, signs) triplets for an entry."""

    if entry["construction"] == "very_sparse_fixed":
        k, m = entry["positions"].shape
        col_ids = torch.arange(k, dtype=torch.int64).repeat_interleave(m)
        return col_ids, entry["positions"].reshape(-1), entry["signs"].reshape(-1)
    return entry["col_ids"], entry["positions"], entry["signs"]


def _project_rows(rows: torch.Tensor, entry: dict[str, Any], path: str) -> torch.Tensor:
    """Project a (n, D) row matrix through the cached matrix along ``path``.

    Parameters
    ----------
    rows:
        Float (n, D) matrix (fp32-accumulated for half inputs upstream).
    entry:
        Cached matrix entry.
    path:
        Chosen multiply path.

    Returns
    -------
    torch.Tensor
        (n, k) projected rows.
    """

    device = rows.device
    dtype = rows.dtype
    extent = int(entry["extent"])
    k = int(entry["n_components"])
    scale = float(entry["scale"])
    col_ids, positions, signs = _coo_triplets(entry)
    if path == "dense":
        weights = torch.zeros((extent, k), dtype=dtype, device=device)
        weights[positions.to(device), col_ids.to(device)] = (
            signs.to(device=device, dtype=dtype) * scale
        )
        return rows @ weights
    if path == "sparse_csr":
        with warnings.catch_warnings():
            # torch warns once that CSR support is beta; the PLANNER chose
            # this path, so the ambient torch notice is not the user's to
            # field — parity with dense is gate-tested instead.
            warnings.simplefilter("ignore", UserWarning)
            matrix = torch.sparse_coo_tensor(
                torch.stack([col_ids, positions]).to(device),
                (signs.to(dtype) * scale).to(device),
                size=(k, extent),
                device=device,
                # In-range by construction (positions come from mod-D draws
                # and column ids from arange); the explicit opt-out silences
                # torch's implicit-disable warning without paying the check.
                check_invariants=False,
            ).to_sparse_csr()
            return torch.sparse.mm(matrix, rows.T).T
    # dense_chunked: O(chunk * k) matrix memory, accumulation over D-chunks.
    chunk = max(1, _CHUNK_BYTES // max(1, k * 4))
    out = torch.zeros((rows.shape[0], k), dtype=dtype, device=device)
    device_cols = col_ids.to(device)
    device_pos = positions.to(device)
    device_vals = signs.to(device=device, dtype=dtype) * scale
    for lo in range(0, extent, chunk):
        hi = min(lo + chunk, extent)
        in_chunk = (device_pos >= lo) & (device_pos < hi)
        if not bool(in_chunk.any().item()):
            continue
        block = torch.zeros((hi - lo, k), dtype=dtype, device=device)
        block[device_pos[in_chunk] - lo, device_cols[in_chunk]] = device_vals[in_chunk]
        out += rows[:, lo:hi] @ block
    return out


def _srp_apply(
    spec: TransformSpec, tensor: torch.Tensor, ctx: TransformContext | None
) -> torch.Tensor:
    """Apply srp: bind extent, fetch the matrix, project (T-C7 accumulation).

    Parameters
    ----------
    spec:
        The srp spec.
    tensor:
        Batch tensor with the stimulus axis leading.
    ctx:
        Optional context (share keying, workspace).

    Returns
    -------
    torch.Tensor
        Projected tensor: flatten mode ``(batch, k)``; features mode keeps
        the leading axes and replaces the last extent with ``k``. Half
        inputs accumulate and RETURN fp32 (disclosed in the plan).
    """

    params = spec.params_dict()
    if not tensor.is_floating_point():
        raise TransformContractError(
            f"srp projects float tensors; got dtype {tensor.dtype}.",
            code="transform_plan_invalid",
            remedy="cast to a float dtype first (chain a cast step)",
            dtype=str(tensor.dtype),
        )
    input_spec = TensorSpec.of(tensor)
    extent = _resolve_extent(spec, input_spec)
    _bind_extent(spec, ctx, extent)
    entry = _matrix_entry(spec, extent, ctx)
    workspace = ctx.workspace if ctx is not None else None
    path = _multiply_path(extent, int(params["n_components"]), tensor.device, workspace)
    acc = tensor.float() if tensor.dtype in _HALF_DTYPES else tensor
    if params["mode"] == "features":
        lead = acc.shape[:-1]
        rows = acc.reshape(-1, extent)
        projected = _project_rows(rows, entry, path)
        return projected.reshape(*lead, int(params["n_components"]))
    rows = acc.reshape(acc.shape[0], -1)
    return _project_rows(rows, entry, path)


def srp(
    n_components: int,
    seed: int | None = None,
    density: str | float = "auto",
    construction: str = "very_sparse_fixed",
    mode: str = "flatten",
    share: str = "by_extent",
) -> TransformSpec:
    """Build a seeded sparse random projection step (memo section 6).

    Parameters
    ----------
    n_components:
        REQUIRED output width; :func:`~torchlens.transforms.srp_dims_for`
        is the labeled sizing heuristic that feeds it.
    seed:
        Base seed. Omitting it records ``seed=0`` with
        ``seed_source="library_default"``; passing it (even ``0``) records
        ``"explicit"``. NOTE: under ``share="by_extent"`` five runs with the
        SAME seed share ONE projection — independence needs DISTINCT seeds
        under either policy.
    density:
        ``"auto"`` (-> ``1/sqrt(D)``) or an explicit float in ``(0, 1]``;
        the realized integer density is recorded per site.
    construction:
        ``"very_sparse_fixed"`` (default; exact per-column counts, cheap,
        ``distance_claim="empirical_only"``) or ``"iid_bernoulli"`` (the
        i.i.d. matrix, O(D*k) generation, also ``empirical_only`` until the
        literature task resolves).
    mode:
        ``"flatten"`` (bind ONE flattened extent, refuse drift) or
        ``"features"`` (project the last axis).
    share:
        ``"by_extent"`` (default; equal-width sites share one matrix) or
        ``"by_site"`` (adds the stable site identity to the key).

    Returns
    -------
    TransformSpec
        The frozen srp step with seed + seed_source recorded (T-C4).
    """

    params = _srp_normalize(
        {
            "n_components": n_components,
            "density": density,
            "construction": construction,
            "mode": mode,
            "share": share,
        }
    )
    if seed is None:
        realized_seed, source = 0, "library_default"
    else:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TransformContractError(
                f"srp(seed={seed!r}) must be an int.",
                code="transform_params_invalid",
                remedy="pass an integer seed",
                seed=seed,
            )
        realized_seed, source = seed, "explicit"
    return TransformSpec(
        name="srp",
        version=ALGORITHM_VERSION,
        params=freeze_params(params),
        seed=realized_seed,
        seed_source=source,
    )


def srp_realized_facts(
    spec: TransformSpec,
    input_spec: TensorSpec | torch.Tensor,
    ctx: TransformContext | None = None,
) -> dict[str, Any]:
    """Return the per-site realized facts the manifest embeds (memo 3.17).

    Parameters
    ----------
    spec:
        The srp spec.
    input_spec:
        Concrete input description (or a tensor to describe).
    ctx:
        Optional context (share keying, workspace, device).

    Returns
    -------
    dict[str, Any]
        Realized facts: input extent, effective seed, construction,
        algorithm_version, realized density, realized nnz, matrix digest,
        multiply path, share key/policy/source, distance_claim.
    """

    if isinstance(input_spec, torch.Tensor):
        device = input_spec.device
        input_spec = TensorSpec.of(input_spec)
    else:
        device = torch.device("cpu")
    params = spec.params_dict()
    extent = _resolve_extent(spec, input_spec)
    entry = _matrix_entry(spec, extent, ctx)
    workspace = ctx.workspace if ctx is not None else None
    return {
        "input_extent": extent,
        "effective_seed": entry["effective_seed"],
        "construction": entry["construction"],
        "algorithm_version": ALGORITHM_VERSION,
        "realized_density": entry["realized_density"],
        "realized_nnz": entry["realized_nnz"],
        "nonzeros_per_column": entry["nonzeros_per_column"],
        "matrix_digest": entry["digest"],
        "multiply_path": _multiply_path(extent, int(params["n_components"]), device, workspace),
        "share": params["share"],
        "share_key": entry["share_key"],
        "share_key_source": entry["share_key_source"],
        "seed": spec.seed,
        "seed_source": spec.seed_source,
        "distance_claim": "empirical_only",
    }


def srp_verify_matrix(
    spec: TransformSpec,
    input_spec: TensorSpec | torch.Tensor,
    ctx: TransformContext | None = None,
) -> str:
    """Re-derive the canonical matrix from scratch and check its digest.

    Parameters
    ----------
    spec:
        The srp spec.
    input_spec:
        Concrete input description (or a tensor to describe).
    ctx:
        Optional context (share keying).

    Returns
    -------
    str
        The verified ``sha256:`` digest.

    Raises
    ------
    TransformContractError
        ``transform_matrix_verification_failed`` when the regenerated
        canonical matrix does not reproduce the cached digest — a
        construction drift, which is never silent (T-C13).
    """

    if isinstance(input_spec, torch.Tensor):
        input_spec = TensorSpec.of(input_spec)
    extent = _resolve_extent(spec, input_spec)
    entry = _matrix_entry(spec, extent, ctx)
    density, m = _resolve_density(spec, extent)
    params = spec.params_dict()
    k = int(params["n_components"])
    if entry["construction"] == "very_sparse_fixed":
        positions, signs = generate_fixed_columns(int(entry["effective_seed"]), extent, range(k), m)
        fresh = digest_fixed_matrix(
            positions,
            signs,
            MatrixHeader(
                construction="very_sparse_fixed",
                extent=extent,
                n_components=k,
                nonzeros_per_column=m,
                scale=float(entry["scale"]),
            ),
        )
    else:
        requested = params["density"]
        bern_density = 1.0 / math.sqrt(extent) if requested == "auto" else float(requested)
        _, positions, signs = generate_bernoulli_columns(
            int(entry["effective_seed"]), extent, range(k), bern_density
        )
        fresh = digest_fixed_matrix(
            positions.reshape(1, -1),
            signs.reshape(1, -1),
            MatrixHeader(
                construction="iid_bernoulli",
                extent=extent,
                n_components=k,
                nonzeros_per_column=-1,
                scale=float(entry["scale"]),
            ),
        )
    if fresh != entry["digest"]:
        raise TransformContractError(
            "SRP matrix re-derivation did not reproduce the recorded digest; "
            "the construction drifted, and a cross-version construction "
            "change is an algorithm_version value, never silent (T-C13).",
            code="transform_matrix_verification_failed",
            remedy="report this with the recorded realized facts",
            recorded=entry["digest"],
            rederived=fresh,
        )
    return fresh


_register_builtin(
    TransformDefinition(
        name="srp",
        version=ALGORITHM_VERSION,
        normalize_params=_srp_normalize,
        plan_fn=_srp_plan,
        apply_fn=_srp_apply,
        context_capable=True,
        stream_safe=True,
        zero_param_preset=False,
        stochastic=True,
    )
)
