"""Direct logit contributions (mikit D9; TLens ``logit_attrs``).

CONVENTION, in one sentence: DLA is a linear decomposition of the ACTUAL
logits at the actual operating point (frozen norm scale computed from the
captured norm input), not a set of counterfactuals.

The D9 rules, all load-bearing and all measured:

- DIRECTIONS-FIRST: build ``gamma * (W_U[:, answer] - W_U[:, vs])`` before
  any vocabulary-sized tensor can exist; peak memory scales with requested
  tokens, never vocabulary.
- MINIMAL CONSTANT ROW: only quantities entering AFTER the pre-norm residual
  (final-norm beta + unembedding bias, projected). Writer-side biases are
  already inside the literal captured writers -- double-counting them costs
  a measured 6.35 logits while still ranking plausibly.
- IDENTITY ALWAYS ON: ``sum(rows) + constant == the captured native logit /
  logit-diff`` runs on every call at the selected positions (a wrong fold
  costs 8.7-120 logits and stays silent otherwise). No ``validate=False``.
- POSITION HONESTY: positions are validated against the LOGITS tensor's own
  position count through the recipe's ``logits_position_map`` (the
  ``logits_to_keep`` slice is latent on plain forwards and ACTIVE under
  generation); unmappable mismatches refuse.

Spellings DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

from typing import Any

import torch

from ._anchors import LMHeadAnchor, resolve_lm_head
from ._errors import refuse
from ._records import ComponentStack, ContributionScores
from ._residual import residual_decomposition

__all__ = ["direct_logit_contributions", "token_directions"]


def _math_unembed(anchor: LMHeadAnchor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return (W_U ``[d_model, vocab]``, unembed bias ``[vocab]`` or None).

    Orientation is decided against the captured logits' vocab axis; a square
    unembedding refuses rather than guesses.
    """

    view = anchor.facets()
    logits = view["logits"].value
    try:
        weight = view["unembed_weight"].value
    except KeyError:
        # Fallback: read the head module's own 2D parameter (TLens's Unembed
        # spells it W_U; the orientation check below stays the authority).
        head_params = list(getattr(anchor.head, "params", ()) or ())
        weight_param = next((p for p in head_params if len(p.shape) == 2), None)
        if weight_param is None:
            refuse(
                code="mi_target_unresolvable",
                message="The unembedding head exposes neither an unembed_weight facet "
                "nor a 2D parameter.",
                remedy="register a facet recipe producing the language_model_head names",
            )
        weight = weight_param.value
    vocab = int(logits.shape[-1])
    bias = None
    try:
        bias_facet = view["unembed_bias"]
        bias = bias_facet.value if hasattr(bias_facet, "value") else None
    except KeyError:
        head_params = list(getattr(anchor.head, "params", ()) or ())
        vocab_hint = int(logits.shape[-1])
        bias_param = next(
            (p for p in head_params if len(p.shape) == 1 and int(p.shape[0]) == vocab_hint),
            None,
        )
        bias = bias_param.value if bias_param is not None else None
    rows, cols = int(weight.shape[0]), int(weight.shape[1])
    if cols == vocab and rows != vocab:
        w_u = weight
    elif rows == vocab and cols != vocab:
        w_u = weight.transpose(0, 1)
    else:
        refuse(
            code="mi_orientation_unknown",
            message=f"The unembedding weight shape {tuple(weight.shape)} cannot be oriented "
            f"against the vocab axis ({vocab}).",
            remedy="report the head module class so its orientation can be pinned",
            weight_shape=[rows, cols],
            vocab=vocab,
        )
        raise AssertionError("unreachable")
    return w_u, bias


def token_directions(trace: Any, tokens: Any) -> torch.Tensor:
    """Return unembedding directions ``W_U[:, tokens]`` as ``[d_model, n]``.

    The escape-hatch helper (mikit roster): raw, un-folded directions; DLA
    folds gamma itself via the validated norm reconstruction.

    Parameters
    ----------
    trace:
        A finished torchlens trace with an unembedding head.
    tokens:
        Token id or sequence of token ids.
    """

    anchor = resolve_lm_head(trace)
    w_u, _bias = _math_unembed(anchor)
    index = torch.as_tensor([tokens] if isinstance(tokens, int) else list(tokens), dtype=torch.long)
    return w_u.index_select(1, index)


def _map_positions(
    positions: Any, residual_len: int, logits_len: int, position_map: Any
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate positions against BOTH axes; return (residual, logits) indices.

    ``position_map`` is the recipe's ``logits_position_map`` IndexMapRecord;
    identity maps pass through; a slice-derived map translates residual
    positions to logits rows; a residual position outside the kept set
    refuses (mikit D9 position honesty).
    """

    raw = [positions] if isinstance(positions, int) else list(positions)
    residual_positions = [p % residual_len if p < 0 else p for p in raw]
    for p in residual_positions:
        if not 0 <= p < residual_len:
            refuse(
                code="mi_position_unmappable",
                message=f"Position {p} lies outside the residual sequence (length {residual_len}).",
                remedy=f"pass positions inside [-{residual_len}, {residual_len})",
                position=p,
            )
    if residual_len == logits_len:
        return tuple(residual_positions), tuple(residual_positions)
    kept: list[int] | None = None
    derivation = getattr(position_map, "derivation", None)
    kept_by_dim = getattr(position_map, "kept_positions_by_dim", None)
    if derivation == "slice" and kept_by_dim:
        for dim_positions in kept_by_dim:
            if dim_positions is not None and len(dim_positions) == logits_len:
                kept = [int(entry) for entry in dim_positions]
                break
    if kept is None:
        refuse(
            code="mi_position_unmappable",
            message=f"The logits tensor has {logits_len} positions but the residual stream "
            f"has {residual_len}, and no recorded index map explains the difference "
            "(logits_to_keep is ACTIVE under generation).",
            remedy="trace a plain forward, or select positions the recorded "
            "logits_position_map can translate",
            residual_len=residual_len,
            logits_len=logits_len,
        )
    logits_rows = []
    for p in residual_positions:
        if p not in kept:
            refuse(
                code="mi_position_unmappable",
                message=f"Residual position {p} was dropped by the recorded logits_to_keep "
                f"slice (kept: {kept}).",
                remedy="select positions among the kept set, or trace a plain forward",
                position=p,
                kept=kept,
            )
        logits_rows.append(kept.index(p))
    return tuple(residual_positions), tuple(logits_rows)


def _as_token_tuple(tokens: Any, vocab: int, argname: str) -> tuple[int, ...]:
    """Normalize a token spec to a tuple of in-vocab ids."""

    ids = [tokens] if isinstance(tokens, int) else [int(entry) for entry in tokens]
    for token_id in ids:
        if not 0 <= token_id < vocab:
            refuse(
                code="mi_token_id_invalid",
                message=f"{argname}= token id {token_id} lies outside the vocabulary ({vocab}).",
                remedy=f"pass ids inside [0, {vocab})",
                token_id=token_id,
                vocab=vocab,
            )
    return tuple(ids)


def direct_logit_contributions(
    trace: Any,
    components: ComponentStack | None = None,
    *,
    answer: Any,
    vs: Any = None,
    positions: Any = -1,
) -> ContributionScores:
    """Attribute the model's own logits to internal writers (mikit D9).

    Parameters
    ----------
    trace:
        A finished torchlens trace with an unembedding head.
    components:
        Any COMPLETE or CLOSED-CONSTANT :class:`ComponentStack` (default:
        ``residual_decomposition(trace)``). Admissibility is verified by the
        identity, not trusted; diagnostic stacks refuse.
    answer:
        Answer token id(s) -- the logit(s) to decompose.
    vs:
        Optional comparison token id(s); when given, rows decompose the
        LOGIT DIFF ``answer - vs`` (the recommended, norm-robust form).
    positions:
        Position index or indices (default: the last position).

    Returns
    -------
    ContributionScores
        Per-component rows + the ONE verified constant row + the captured
        native logits, with the in-API identity receipt.
    """

    anchor = resolve_lm_head(trace)
    if components is None:
        components = residual_decomposition(trace)
    _require_admissible(components)

    norm = anchor.norm_reconstruction()
    w_u, unembed_bias = _math_unembed(anchor)
    answer_ids, vs_ids, directions = _token_directions_for(w_u, answer, vs)
    folded = norm.folded_directions(directions)  # [d_model, n_dirs]

    logits = anchor.facets()["logits"].value
    view = anchor.facets()
    try:
        position_map = view["logits_position_map"]
    except KeyError:
        position_map = None
    residual_len = int(norm.input.shape[1])
    logits_len = int(logits.shape[-2]) if logits.dim() >= 2 else 1
    residual_positions, logits_rows = _map_positions(
        positions, residual_len, logits_len, position_map
    )
    pos_index = torch.as_tensor(residual_positions, dtype=torch.long)
    scale = norm.scale.index_select(1, pos_index)  # [b, n_pos, 1]

    target_value = components.target_value
    full_stack = (
        target_value is not None
        and tuple(target_value.shape) == tuple(norm.input.shape)
        and torch.equal(target_value.detach(), norm.input.detach())
    )

    def _project(value: torch.Tensor) -> torch.Tensor:
        """Push one component through the frozen linearization + directions."""

        if tuple(value.shape) != tuple(norm.input.shape):
            value = value.expand(norm.input.shape)
        selected = value.index_select(1, pos_index).to(torch.float32)
        if norm.centered:
            selected = selected - selected.mean(dim=-1, keepdim=True)
        return (selected / scale) @ folded  # [b, n_pos, n_dirs]

    rows = []
    coordinates = []
    for row in components.rows:
        rows.append(_project(row.value))
        coordinates.append(row.coordinate)

    shape = (norm.input.shape[0], len(residual_positions), folded.shape[1])
    if full_stack:
        constant = _constant_row(shape, norm, directions, unembed_bias, answer_ids, vs_ids)
        native = _native_logits(logits, logits_rows, answer_ids, vs_ids)
    else:
        # Partial stack (e.g. head contributions): the exactness claim is
        # LINEARITY -- rows must sum to the projection of the stack's own
        # target region; the full-logit constant row does not belong here.
        constant = torch.zeros(shape, dtype=torch.float32)
        assert target_value is not None  # noqa: S101 -- proven by _require_admissible
        native = _project(target_value)

    values = torch.stack(rows)  # [n_rows, b, n_pos, n_dirs]
    reconstructed = values.sum(dim=0) + constant
    residual = float((reconstructed - native).detach().abs().max())
    tolerance = _identity_tolerance(values)
    if residual > tolerance:
        refuse(
            code="mi_dla_identity_failed",
            message=f"sum(rows) + constant misses the captured native logits by {residual:.3e} "
            f"(tolerance {tolerance:.3e}). A wrong norm fold costs 8.7-120 logits while "
            "still ranking components plausibly; this table is refused, not served.",
            remedy="this is a kit or capture defect, not a user error: report it with "
            "tl.compat.report(model, x)",
            max_abs_residual=residual,
            tolerance=tolerance,
        )

    return ContributionScores(
        tuple(coordinates),
        values,
        constant=constant,
        native=native,
        answer_tokens=answer_ids,
        vs_tokens=vs_ids,
        positions=tuple(residual_positions),
        identity_receipt={
            "check": "sum_rows_plus_constant_vs_native"
            if full_stack
            else "sum_rows_vs_projected_target (partial stack)",
            "result": "verified",
            "max_abs_residual": residual,
            "tolerance": tolerance,
        },
    )


def _require_admissible(components: ComponentStack) -> None:
    """Refuse diagnostic / CLOSED-UNRESOLVED / target-less stacks (D7)."""

    if components.diagnostic_only or components.grading == "closed_unresolved":
        refuse(
            code="mi_stack_not_admissible",
            message="DLA accepts only COMPLETE or CLOSED-CONSTANT stacks; this stack is "
            f"{components.grading!r} (diagnostic_only={components.diagnostic_only}).",
            remedy="fix the capture so strict decomposition closes, or decompose a "
            "different target; strict=False stacks are diagnosis, not attribution",
            grading=components.grading,
        )
    if components.target_value is None:
        refuse(
            code="mi_stack_not_admissible",
            message="DLA needs a stack with a target value to verify against.",
            remedy="produce the stack from a payload-bearing capture",
        )


def _token_directions_for(
    w_u: torch.Tensor, answer: Any, vs: Any
) -> tuple[tuple[int, ...], tuple[int, ...] | None, torch.Tensor]:
    """Validate token specs and build the raw directions FIRST (D9).

    No vocabulary-sized intermediate ever exists: the directions matrix is
    ``[d_model, n_requested]``.
    """

    vocab = int(w_u.shape[1])
    answer_ids = _as_token_tuple(answer, vocab, "answer")
    vs_ids = _as_token_tuple(vs, vocab, "vs") if vs is not None else None
    if vs_ids is not None and len(vs_ids) != len(answer_ids):
        refuse(
            code="mi_token_id_invalid",
            message=f"answer has {len(answer_ids)} token(s) but vs has {len(vs_ids)}; "
            "logit diffs are pairwise.",
            remedy="pass equal-length answer/vs token sequences",
        )
    directions = w_u.index_select(1, torch.as_tensor(answer_ids, dtype=torch.long))
    if vs_ids is not None:
        directions = directions - w_u.index_select(1, torch.as_tensor(vs_ids, dtype=torch.long))
    return answer_ids, vs_ids, directions


def _constant_row(  # noqa: PLR0913 -- the D9 constant row's evidence set, spelled out
    shape: tuple[int, int, int],
    norm: Any,
    directions: torch.Tensor,
    unembed_bias: torch.Tensor | None,
    answer_ids: tuple[int, ...],
    vs_ids: tuple[int, ...] | None,
) -> torch.Tensor:
    """Build the ONE minimal constant row (final-norm beta + unembed bias).

    Writer-side biases are already inside the literal captured writers;
    adding them here is the measured 6.35-logit double count (D9).
    """

    constant = torch.zeros(shape, dtype=torch.float32)
    if norm.beta is not None:
        constant = constant + (norm.beta.to(torch.float32) @ directions.to(torch.float32))
    if unembed_bias is not None:
        bias_term = unembed_bias.index_select(0, torch.as_tensor(answer_ids, dtype=torch.long))
        if vs_ids is not None:
            bias_term = bias_term - unembed_bias.index_select(
                0, torch.as_tensor(vs_ids, dtype=torch.long)
            )
        constant = constant + bias_term.to(torch.float32)
    return constant


def _native_logits(
    logits: torch.Tensor,
    logits_rows: tuple[int, ...],
    answer_ids: tuple[int, ...],
    vs_ids: tuple[int, ...] | None,
) -> torch.Tensor:
    """Read the model's own captured logits at the selected coordinates."""

    logits_index = torch.as_tensor(logits_rows, dtype=torch.long)
    answer_index = torch.as_tensor(answer_ids, dtype=torch.long)
    native = logits.index_select(-2, logits_index).index_select(-1, answer_index)
    if vs_ids is not None:
        native = native - logits.index_select(-2, logits_index).index_select(
            -1, torch.as_tensor(vs_ids, dtype=torch.long)
        )
    native = native.to(torch.float32)
    if native.dim() == 2:  # [n_pos, n_dirs] logits without a batch axis
        native = native.unsqueeze(0)
    return native


def _identity_tolerance(values: torch.Tensor) -> float:
    """Return the DLA identity tolerance (the measured ~1.5e-05 class).

    The DLA path genuinely re-associates (per-component contractions summed
    in a different order than the model's matmul), so the gate cannot be
    bitwise. The budget is the cancellation-aware model: accumulated |addend|
    magnitude times reduction-depth ULP headroom.
    """

    magnitude = float(values.detach().abs().sum(dim=0).max()) + 1.0
    n_addends = max(2, values.shape[0] * 8)
    eps = float(torch.finfo(torch.float32).eps)
    import math

    return 16.0 * (1.0 + math.log2(n_addends)) * eps * magnitude
