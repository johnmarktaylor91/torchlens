"""The token-attribution two-liner (attrib memo D22-D29).

.. code-block:: python

    result = tl.attribution.text(model, tokenizer, "The Eiffel Tower is in", target=" Paris")
    result.show()

Mechanism (D22): input Integrated Gradients through HF ``inputs_embeds`` --
measured bit-identical to embedding-output replacement -- with positions,
attention masks, and ALL integer inputs held fixed; the signed attribution is
sum-reduced over the embedding width so per-token scores retain completeness
accounting.

``baseline="auto"`` is TASK-AWARE (D23) because the measurement forced it:
uniform zeros converges to 0.001% on gpt2 and NEVER converges on bert out to
n=1024, while pad + special-token scaffolding converges monotonically to
0.097%. Decoder-only causal LMs zero all prompt-token embeddings; encoder
classifiers / masked LMs with a reliable pad token and special-token mask pad
content tokens while KEEPING special tokens at their true embeddings
(``keep_special_tokens=True``); missing or unreliable family metadata, pad
token, or special mask falls back to zeros-all with a NAMED disclosure --
a missing pad token is never guessed.

``n_steps=128`` is the fixed default (D24, the recorded 2-1);
``n_steps="auto"`` opts into the dual-criterion ladder 64 -> 128 -> 256 ->
512, stopping ONLY when residual_rel <= 1% AND max(L1, L2) successive-grid
change <= 1% (residual-only stopping is measured-unsound and may never
return; 5%/5% falsely certifies -- never ship 5%). Rank stability is BANNED
as a convergence signal (D25): top-k agreement goes stable exactly where the
certificate is garbage. ``converged=False`` is a primary path (D26) with
first-class rendered treatment.

Rendering: tviz owns the one colored-token renderer; this module ships the
renderer-neutral typed payload and an ESCAPED-TABLE fallback only (D28).
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._result import AttributionError, AttributionWarning

_AUTO_LADDER = (64, 128, 256, 512)
_RESIDUAL_TOLERANCE = 0.01
_STABILITY_TOLERANCE = 0.01


class _LogitsOnly(Module):
    """Adapter returning ONLY the logits tensor from an HF-style model.

    Keeps int targets, output splitting under step batching, and the IG
    machinery on plain tensors regardless of the host model's output class.
    """

    def __init__(self, model: Module) -> None:
        """Wrap the host model."""

        super().__init__()
        self.model = model

    def forward(self, **kwargs: Any) -> Tensor:
        """Run the host forward and extract its logits tensor."""

        output = self.model(**kwargs)
        logits = getattr(output, "logits", None)
        if logits is None and isinstance(output, (tuple, list)) and output:
            logits = output[0]
        if not isinstance(logits, Tensor):
            raise AttributionError(
                "text() could not locate a logits tensor on the model "
                "output. Remedy: use a transformers-style model whose "
                "forward returns .logits.",
                code="text_model_unsupported",
            )
        return logits


def _detect_family(model: Module) -> tuple[str, str]:
    """Classify the model's task family for the auto baseline (D23).

    Parameters
    ----------
    model
        HF-style model.

    Returns
    -------
    tuple[str, str]
        ``("decoder" | "encoder" | "unknown", reason)``.
    """

    name = type(model).__name__
    config = getattr(model, "config", None)
    if "ForCausalLM" in name or "LMHeadModel" in name:
        return "decoder", f"model class {name} is a causal LM head"
    if config is not None and getattr(config, "is_decoder", False):
        return "decoder", "config.is_decoder is True"
    for tag in (
        "ForMaskedLM",
        "ForSequenceClassification",
        "ForTokenClassification",
        "ForQuestionAnswering",
        "ForMultipleChoice",
    ):
        if tag in name:
            return "encoder", f"model class {name} carries the {tag} head"
    return "unknown", f"model class {name} matches no known task-family pattern"


def _resolve_baseline(
    baseline: Any,
    family: str,
    family_reason: str,
    embeddings: Tensor,
    input_ids: Tensor,
    special_mask: Tensor | None,
    embedding_layer: Module,
    pad_token_id: int | None,
    keep_special_tokens: bool,
) -> tuple[Tensor, dict[str, Any]]:
    """Resolve the baseline spelling into concrete embeddings + disclosure.

    Parameters
    ----------
    baseline
        ``"auto"`` (task-aware), ``"zeros"``, ``"pad_token"``, a ``[1, L, D]``
        tensor, or a list of aligned baseline token ids.
    family
        Detected task family.
    family_reason
        Why the family was detected (rides the disclosure).
    embeddings
        True input embeddings ``[1, L, D]``.
    input_ids
        Token ids ``[1, L]``.
    special_mask
        Special-token mask ``[1, L]`` (1 = special), or ``None`` when the
        tokenizer could not provide one.
    embedding_layer
        The model's input embedding module.
    pad_token_id
        The tokenizer's pad id, or ``None``.
    keep_special_tokens
        Whether pad-style baselines scaffold special tokens at their true
        embeddings (the only converging encoder baseline; D23).

    Returns
    -------
    tuple[Tensor, dict[str, Any]]
        Baseline embeddings and the full resolution disclosure (policy,
        reason, baseline ids where applicable, the scaffold mask).

    Raises
    ------
    AttributionError
        On unknown spellings, a requested pad baseline without a pad token,
        or misaligned tensor/id baselines.
    """

    length = int(input_ids.shape[1])

    def _pad_scaffold(policy: str, reason: str) -> tuple[Tensor, dict[str, Any]]:
        """Pad content tokens; keep special tokens at true embeddings."""

        if pad_token_id is None:
            raise AttributionError(
                "baseline='pad_token' requires a tokenizer pad token, and "
                "this tokenizer has none; a missing pad token is never "
                "guessed. Remedy: set tokenizer.pad_token (e.g. to the eos "
                "token) explicitly, or use baseline='zeros'.",
                code="text_baseline_unresolvable",
            )
        pad_embedding = embedding_layer(torch.tensor([pad_token_id], device=input_ids.device)).to(
            embeddings.dtype
        )
        base = pad_embedding.expand(length, -1).unsqueeze(0).clone()
        scaffold = torch.zeros(length, dtype=torch.bool)
        if keep_special_tokens and special_mask is not None:
            keep = special_mask[0].to(torch.bool)
            base[0, keep] = embeddings[0, keep].detach()
            scaffold = keep.cpu()
        baseline_ids = [
            int(input_ids[0, position]) if scaffold[position] else pad_token_id
            for position in range(length)
        ]
        return base.detach(), {
            "policy": policy,
            "reason": reason,
            "baseline_ids": baseline_ids,
            "scaffolded_positions": scaffold.nonzero().flatten().tolist(),
        }

    if baseline == "auto":
        if family == "decoder":
            return torch.zeros_like(embeddings), {
                "policy": "zeros_all",
                "reason": f"auto: decoder family ({family_reason}); zeros "
                "converges on measured decoders (0.001% on gpt2) while pad "
                "does not (20.9%)",
                "baseline_ids": None,
                "scaffolded_positions": [],
            }
        if family == "encoder" and pad_token_id is not None and special_mask is not None:
            return _pad_scaffold(
                "pad_content_plus_scaffold",
                f"auto: encoder family ({family_reason}); pad + scaffolding "
                "is the only measured converging encoder baseline (monotone "
                "to 0.097% at n=1024 on bert) while zeros never converges",
            )
        missing = []
        if family == "unknown":
            missing.append("task-family metadata")
        if pad_token_id is None:
            missing.append("a pad token")
        if special_mask is None:
            missing.append("a special-token mask")
        return torch.zeros_like(embeddings), {
            "policy": "zeros_all",
            "reason": "auto FALLBACK: missing "
            + " and ".join(missing or ["reliable family metadata"])
            + "; zeros-all is the named fallback (a missing pad token is "
            "never guessed)",
            "baseline_ids": None,
            "scaffolded_positions": [],
        }
    if baseline == "zeros":
        return torch.zeros_like(embeddings), {
            "policy": "zeros_all",
            "reason": "explicit baseline='zeros'",
            "baseline_ids": None,
            "scaffolded_positions": [],
        }
    if baseline == "pad_token":
        return _pad_scaffold("pad_content_plus_scaffold", "explicit baseline='pad_token'")
    if isinstance(baseline, Tensor):
        if tuple(baseline.shape) != tuple(embeddings.shape):
            raise AttributionError(
                f"baseline tensor of shape {tuple(baseline.shape)} must match "
                f"the input embeddings {tuple(embeddings.shape)}. Remedy: "
                "align the baseline with the tokenized sequence.",
                code="text_baseline_unresolvable",
            )
        return baseline.detach().to(embeddings.dtype), {
            "policy": "explicit_tensor",
            "reason": "user-provided baseline embeddings",
            "baseline_ids": None,
            "scaffolded_positions": [],
        }
    if isinstance(baseline, (list, tuple)) and all(isinstance(item, int) for item in baseline):
        if len(baseline) != length:
            raise AttributionError(
                f"baseline id list has {len(baseline)} entries but the "
                f"tokenized input has {length}. Remedy: align the baseline "
                "ids with the tokenized sequence.",
                code="text_baseline_unresolvable",
            )
        ids = torch.tensor([list(baseline)], device=input_ids.device)
        return embedding_layer(ids).detach().to(embeddings.dtype), {
            "policy": "explicit_ids",
            "reason": "user-provided aligned baseline token ids",
            "baseline_ids": list(baseline),
            "scaffolded_positions": [],
        }
    raise AttributionError(
        f"baseline must be 'auto', 'zeros', 'pad_token', an embeddings "
        f"tensor, or aligned token ids; got {type(baseline).__name__}. "
        "Remedy: choose a documented spelling.",
        code="text_baseline_unresolvable",
    )


def _resolve_target(
    target: Any,
    logits: Tensor,
    tokenizer: Any,
    attention_mask: Tensor,
    family: str,
) -> tuple[Any, str, int | None]:
    """Resolve the target spelling into a scalarizer + disclosure (D29).

    Parameters
    ----------
    target
        ``None`` (the model's own argmax, disclosed), a bare int (vocab id at
        the final non-padding position on rank-3 causal logits, with the
        ambiguity WARN band), a ``(position, vocab)`` tuple, a token string,
        a ``{"target": ..., "foil": ...}`` contrastive pair, or a callable
        ``logits -> scalar``.
    logits
        The unperturbed logits (rank 2 or 3).
    tokenizer
        Tokenizer for string targets and decoded disclosure.
    attention_mask
        ``[1, L]`` attention mask locating the final non-padding position.
    family
        Detected task family.

    Returns
    -------
    tuple[Any, str, int | None]
        The scalarizer callable, the full ``target_repr`` disclosure, and the
        resolved position (rank-3 targets).

    Raises
    ------
    AttributionError
        On unresolvable spellings.
    """

    def _decode(token_id: int) -> str:
        """Decode one id for disclosure text (never load-bearing)."""

        try:
            return tokenizer.decode([token_id])
        except Exception:  # noqa: BLE001 - decode is disclosure-only
            return "<undecodable>"

    def _token_to_id(token: str) -> int:
        """Resolve a single-token string to its one vocab id or refuse."""

        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            raise AttributionError(
                f"target token string {token!r} tokenizes to {len(ids)} ids; "
                "a target must be ONE vocabulary item. Remedy: pass a single "
                "token (mind leading spaces on BPE vocabularies), a "
                "(position, vocab) tuple, or a callable.",
                code="text_target_unresolvable",
            )
        return int(ids[0])

    if callable(target) and not isinstance(target, (str, tuple)):
        return target, getattr(target, "__name__", repr(target)), None

    if logits.ndim == 3:
        final_position = int(attention_mask[0].nonzero().max())
        sequence_length = int(logits.shape[1])

        def _positional(position: int, vocab: int) -> tuple[Any, str, int]:
            """Build a (position, vocab) scalarizer with full disclosure."""

            def scorer(out: Tensor) -> Tensor:
                """Select one (position, vocab) logit."""

                return out[0, position, vocab]

            representation = f"logits[0, {position}, {vocab}] ({_decode(vocab)!r})"
            return scorer, representation, position

        if target is None:
            vocab = int(logits[0, final_position].argmax())
            scorer, representation, position = _positional(final_position, vocab)
            return (
                scorer,
                f"default: model argmax next token -- {representation}",
                position,
            )
        if isinstance(target, bool):
            raise AttributionError(
                "target must not be a bool. Remedy: pass an int vocab id, a "
                "(position, vocab) tuple, a token string, or a callable.",
                code="text_target_unresolvable",
            )
        if isinstance(target, int):
            # Bare int = vocab id at the final non-padding position; the ONE
            # ambiguous band is an int that is also a valid position index.
            if 0 <= target < sequence_length:
                import warnings

                warnings.warn(
                    AttributionWarning(
                        f"bare int target {target} is read as a VOCAB id at "
                        f"the final non-padding position ({final_position}), "
                        f"but {target} is also a valid position index for "
                        f"this {sequence_length}-token input. Remedy: pass "
                        "the explicit (position, vocab) tuple if a position "
                        "was meant.",
                        code="text_bare_int_target_ambiguous",
                    ),
                    stacklevel=3,
                )
            return _positional(final_position, target)
        if isinstance(target, str):
            return _positional(final_position, _token_to_id(target))
        if (
            isinstance(target, tuple)
            and len(target) == 2
            and all(isinstance(item, int) for item in target)
        ):
            return _positional(int(target[0]), int(target[1]))
        if isinstance(target, dict) and {"target", "foil"} <= set(target):
            target_id = (
                _token_to_id(target["target"])
                if isinstance(target["target"], str)
                else int(target["target"])
            )
            foil_id = (
                _token_to_id(target["foil"])
                if isinstance(target["foil"], str)
                else int(target["foil"])
            )

            def contrastive(out: Tensor) -> Tensor:
                """Target logit minus foil logit at the final position."""

                return out[0, final_position, target_id] - out[0, final_position, foil_id]

            return (
                contrastive,
                f"contrastive logits[0, {final_position}, {target_id} "
                f"({_decode(target_id)!r})] - logits[0, {final_position}, "
                f"{foil_id} ({_decode(foil_id)!r})]",
                final_position,
            )
        raise AttributionError(
            f"could not resolve target {target!r} for rank-3 logits. "
            "Remedy: pass an int vocab id, a (position, vocab) tuple, a "
            "single-token string, a contrastive {'target':..., 'foil':...} "
            "pair, or a callable.",
            code="text_target_unresolvable",
        )

    # Rank-2 logits: classification head.
    if target is None:
        class_index = int(logits[0].argmax())

        def class_scorer(out: Tensor) -> Tensor:
            """Select the argmax class logit."""

            return out[0, class_index]

        return (
            class_scorer,
            f"default: model argmax class -- logits[0, {class_index}]",
            None,
        )
    if isinstance(target, int) and not isinstance(target, bool):

        def index_scorer(out: Tensor) -> Tensor:
            """Select the requested class logit."""

            return out[0, target]

        return index_scorer, f"logits[0, {target}]", None
    raise AttributionError(
        f"could not resolve target {target!r} for rank-2 logits ({family} "
        "family). Remedy: pass an int class index or a callable.",
        code="text_target_unresolvable",
    )


@dataclass(frozen=True)
class TokenAttributionPayload:
    """Renderer-neutral token-score payload (the tviz contract, D28).

    Attributes
    ----------
    raw_tokens
        Tokenizer tokens, never merged.
    display_tokens
        Per-token display strings (marker cleanup only, never merging).
    scores
        Signed per-token scores (zero-centered diverging domain requested).
    tooltips
        Per-token tooltip strings.
    footer_lines
        Disclosure lines the renderer must show (target, method, exact
        baseline, steps, |target_delta|, residual, stability, convergence
        state, truncation).
    score_domain
        ``"zero_centered_diverging"`` -- the renderer's color-domain request.
    """

    raw_tokens: list[str]
    display_tokens: list[str]
    scores: list[float]
    tooltips: list[str]
    footer_lines: list[str]
    score_domain: str = "zero_centered_diverging"

    def to_html(self) -> str:
        """Render the ESCAPED-table fallback (until the tviz renderer lands).

        Returns
        -------
        str
            Self-contained escaped HTML table with the footer lines; no
            network, no JS, no second renderer ambition.
        """

        limit = max((abs(score) for score in self.scores), default=0.0) or 1.0
        cells = []
        for token, score, tooltip in zip(
            self.display_tokens, self.scores, self.tooltips, strict=True
        ):
            intensity = min(1.0, abs(score) / limit)
            channel = int(255 - 128 * intensity)
            color = (
                f"rgb({channel},255,{channel})" if score >= 0 else f"rgb(255,{channel},{channel})"
            )
            cells.append(
                f'<td style="background:{color}" title="{_html.escape(tooltip)}">'
                f"{_html.escape(token)}</td>"
            )
        footer = "".join(f"<div>{_html.escape(line)}</div>" for line in self.footer_lines)
        return '<table role="presentation"><tr>' + "".join(cells) + "</tr></table>" + footer

    def _repr_html_(self) -> str:
        """Notebook display hook for the escaped-table fallback."""

        return self.to_html()

    def to_text(self) -> str:
        """Render a plain-text fallback table with the footer lines."""

        rows = [
            f"{token!r}: {score:+.6f}"
            for token, score in zip(self.display_tokens, self.scores, strict=True)
        ]
        return "\n".join(rows + [""] + self.footer_lines)


@dataclass(frozen=True)
class TokenAttributionResult:
    """Frozen result of the token-attribution two-liner (attrib memo D28).

    Attributes
    ----------
    text
        The original input text.
    input_ids
        Tokenized ids (list of ints).
    raw_tokens
        Tokenizer tokens, never silently merged.
    display_tokens
        Display strings per raw token (marker cleanup only).
    offsets
        Character offsets per token, or ``None`` when the tokenizer cannot
        provide them (disclosed).
    special_tokens_mask
        1 where the token is a tokenizer special, or ``None``.
    scores
        Signed per-token attribution scores (sum over embedding width).
    values
        Full ``(L, D)`` signed attribution tensor.
    target_repr
        Full target resolution disclosure incl. the decoded token.
    baseline
        The exact resolved baseline disclosure (policy, reason, ids, mask).
    n_steps
        The FINAL grid the scores were computed at.
    converged
        ``True``/``False`` under the dual criterion, or ``None`` when a fixed
        integer grid was requested (no stability information exists -- the
        knob is named in the footer).
    convergence
        Grid-pair evidence: requested spelling, evaluated grids, residual and
        both stability norms per grid, and the stopping statement.
    completeness
        ``attribution_sum`` / ``target_delta`` / ``completeness_residual`` /
        ``residual_rel`` / ``target_delta_abs``.
    cost
        Logical/physical call accounting incl. the batching audit evidence.
    provenance
        Model/tokenizer identities and the transformers version.
    truncation
        ``None`` or the explicit-truncation record (original length,
        retained span).
    """

    text: str
    input_ids: list[int]
    raw_tokens: list[str]
    display_tokens: list[str]
    offsets: list[tuple[int, int]] | None
    special_tokens_mask: list[int] | None
    scores: Tensor
    values: Tensor
    target_repr: str
    baseline: dict[str, Any]
    n_steps: int
    converged: bool | None
    convergence: dict[str, Any]
    completeness: dict[str, Any]
    cost: dict[str, Any]
    provenance: dict[str, Any]
    truncation: dict[str, Any] | None = None
    _payload_cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def footer_lines(self) -> list[str]:
        """Build the mandatory disclosure footer (D26-D28)."""

        lines = [
            f"target: {self.target_repr}",
            "method: integrated_gradients (midpoint Riemann) over inputs_embeds",
            f"baseline: {self.baseline['policy']} -- {self.baseline['reason']}",
            f"n_steps: {self.n_steps}",
            f"|target_delta|: {self.completeness['target_delta_abs']:.6g}",
            f"residual_rel: {self.completeness['residual_rel']:.4%}",
        ]
        stability = self.convergence.get("stability")
        if stability is not None:
            lines.append(
                f"stability max(L1, L2): {stability:.4%} at grid pair "
                f"{self.convergence.get('grid_pair')}"
            )
        if self.converged is None:
            lines.append(
                "convergence: not evaluated (fixed n_steps; pass "
                "n_steps='auto' for the dual-criterion certificate)"
            )
        elif self.converged:
            lines.append(
                "convergence: met criteria at grid pair "
                f"{self.convergence.get('grid_pair')} (residual AND "
                "successive-grid stability <= 1%) -- a detector, not a proof"
            )
        else:
            lines.append(
                "convergence: NOT CERTIFIED -- the auto ladder hit its cap "
                f"({self.convergence.get('grids_evaluated')}); this is what "
                "an under-resolved integral honestly looks like"
            )
        if self.truncation is not None:
            lines.append(
                f"truncation: kept {self.truncation['retained_length']} of "
                f"{self.truncation['original_length']} tokens"
            )
        return lines

    def payload(self) -> TokenAttributionPayload:
        """Build the renderer-neutral payload (the tviz contract)."""

        scores = [float(score) for score in self.scores]
        tooltips = [
            f"{raw!r} id={token_id} score={score:+.6g}"
            for raw, token_id, score in zip(self.raw_tokens, self.input_ids, scores, strict=True)
        ]
        return TokenAttributionPayload(
            raw_tokens=list(self.raw_tokens),
            display_tokens=list(self.display_tokens),
            scores=scores,
            tooltips=tooltips,
            footer_lines=self.footer_lines(),
        )

    def show(self) -> TokenAttributionPayload:
        """Return the displayable payload (escaped-table fallback).

        In a notebook the returned payload renders as the escaped HTML
        table; elsewhere print ``.to_text()``.
        """

        return self.payload()

    def __repr__(self) -> str:
        """Compact repr with the headline numbers, never the tensors."""

        return (
            f"TokenAttributionResult(tokens={len(self.raw_tokens)}, "
            f"n_steps={self.n_steps}, converged={self.converged}, "
            f"residual_rel={self.completeness['residual_rel']:.4%}, "
            f"target={self.target_repr!r})"
        )


def _display_tokens(raw_tokens: list[str]) -> list[str]:
    """Clean tokenizer markers per token WITHOUT merging wordpieces (D28).

    Parameters
    ----------
    raw_tokens
        Tokenizer tokens.

    Returns
    -------
    list[str]
        One display string per raw token.
    """

    display = []
    for token in raw_tokens:
        cleaned = token.replace("Ġ", " ").replace("Ċ", "\\n")
        if cleaned.startswith("##"):
            cleaned = "·" + cleaned[2:]
        display.append(cleaned)
    return display


def _dual_criterion_met(residual_rel: float, stability: float | None) -> bool:
    """The ONLY shipped stopping rule: residual AND stability, both <= 1%.

    Residual-only stopping is measured-unsound and may never return (the
    V17/T1 case: bert/zeros reads 1.80% at n=16 by sign-cancellation luck and
    26.7% one grid later; bert/pad-plain passes a 5% criterion at n=64 and
    reads 9.8% at n=256). Thresholds may TIGHTEN, never loosen (release gate,
    memo D24). ``stability`` is ``None`` on the first ladder grid -- one grid
    can never certify, because stability needs two grids by construction.

    Parameters
    ----------
    residual_rel
        The completeness residual ratio at the current grid.
    stability
        ``max(L1, L2)`` successive-grid change, or ``None`` on the first grid.

    Returns
    -------
    bool
        Whether the dual criterion certifies this grid pair.
    """

    if stability is None:
        return False
    return residual_rel <= _RESIDUAL_TOLERANCE and stability <= _STABILITY_TOLERANCE


def _stability_change(previous: Tensor, current: Tensor) -> float:
    """Successive-grid change: max of relative L1 and L2 norms (D24).

    The gate is ``max(L1, L2)`` because the two norms give OPPOSITE answers
    at a threshold on real cases (L2 sees one token moving, L1 sees diffuse
    drift).

    Parameters
    ----------
    previous
        Per-token scores at grid m.
    current
        Per-token scores at grid n.

    Returns
    -------
    float
        ``max(||s_n - s_m||_1 / ||s_n||_1, ||s_n - s_m||_2 / ||s_n||_2)``.
    """

    diff = (current - previous).to(torch.float64)
    current64 = current.to(torch.float64)
    l1 = float(diff.abs().sum() / current64.abs().sum().clamp(min=1e-30))
    l2 = float(diff.norm() / current64.norm().clamp(min=1e-30))
    return max(l1, l2)


def text(
    model: Module,
    tokenizer: Any,
    text: str,
    *,
    target: Any = None,
    baseline: Any = "auto",
    n_steps: int | str = 128,
    steps_per_batch: int | None = 8,
    keep_special_tokens: bool = True,
    max_length: int | None = None,
) -> TokenAttributionResult:
    """Per-token attribution for a text input: the two-liner (memo D22-D28).

    Parameters
    ----------
    model
        HF-style model whose forward accepts ``inputs_embeds`` and returns
        logits.
    tokenizer
        Matching tokenizer.
    text
        Input text.
    target
        ``None`` (the model's own argmax, disclosed), an int (vocab id at the
        final non-padding position for causal LMs -- the position-ambiguity
        band WARNS), a ``(position, vocab)`` tuple, a single-token string, a
        contrastive ``{"target": ..., "foil": ...}`` pair, or a callable
        ``logits -> scalar``.
    baseline
        ``"auto"`` (task-aware, always resolved to a printed concrete
        baseline), ``"zeros"``, ``"pad_token"``, an embeddings tensor, or
        aligned baseline token ids.
    n_steps
        Fixed integer grid (default 128) or ``"auto"`` for the dual-criterion
        ladder 64 -> 128 -> 256 -> 512.
    steps_per_batch
        Path points stacked per forward (default 8, gated by the randomized
        batching audit); ``None``/``1`` for sequential.
    keep_special_tokens
        Whether pad-style baselines keep special tokens at their true
        embeddings (scaffolding; the only measured converging encoder
        baseline).
    max_length
        Optional EXPLICIT truncation; records the original length and the
        retained span, and warns visibly. Never applied silently.

    Returns
    -------
    TokenAttributionResult
        Frozen result; ``result.show()`` renders the escaped-table fallback.

    Raises
    ------
    AttributionError
        On unresolvable baselines/targets, unsupported models, or batching
        audit failures.
    """

    import warnings

    from torchlens.attribution._core import (
        _completeness_extra,
        integrated_gradients,
    )

    if not isinstance(text, str) or not text:
        raise AttributionError(
            "text() requires a non-empty string. Remedy: pass the input text.",
            code="text_input_invalid",
        )
    if n_steps != "auto" and (
        isinstance(n_steps, bool) or not isinstance(n_steps, int) or n_steps <= 0
    ):
        raise AttributionError(
            "n_steps must be a positive int or 'auto'. Remedy: keep the "
            "default 128, or opt into the dual-criterion ladder with "
            "n_steps='auto'.",
            code="text_input_invalid",
        )

    encode_kwargs: dict[str, Any] = {"return_tensors": "pt"}
    if max_length is not None:
        encode_kwargs.update({"truncation": True, "max_length": max_length})
    try:
        encoding = tokenizer(
            text, return_special_tokens_mask=True, return_offsets_mapping=True, **encode_kwargs
        )
        offsets_available = True
    except Exception:  # noqa: BLE001 - slow tokenizers lack offsets
        encoding = tokenizer(text, return_special_tokens_mask=True, **encode_kwargs)
        offsets_available = False
    input_ids = encoding["input_ids"]
    attention_mask = encoding.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    special_mask = encoding.get("special_tokens_mask")

    truncation_record: dict[str, Any] | None = None
    if max_length is not None:
        full_ids = tokenizer(text)["input_ids"]
        if len(full_ids) > int(input_ids.shape[1]):
            truncation_record = {
                "original_length": len(full_ids),
                "retained_length": int(input_ids.shape[1]),
                "retained_span": [0, int(input_ids.shape[1])],
            }
            warnings.warn(
                AttributionWarning(
                    f"text() truncated the input from {len(full_ids)} to "
                    f"{int(input_ids.shape[1])} tokens under the explicit "
                    "max_length; scores cover the retained span only. "
                    "Remedy: raise max_length to cover the full input.",
                    code="text_input_truncated",
                ),
                stacklevel=2,
            )

    embedding_layer = cast(Module, cast(Any, model).get_input_embeddings())
    with torch.no_grad():
        embeddings = embedding_layer(input_ids).detach()
    family, family_reason = _detect_family(model)
    baseline_embeddings, baseline_disclosure = _resolve_baseline(
        baseline,
        family,
        family_reason,
        embeddings,
        input_ids,
        special_mask,
        embedding_layer,
        getattr(tokenizer, "pad_token_id", None),
        keep_special_tokens,
    )

    adapter = _LogitsOnly(model)
    fixed_kwargs: dict[str, Any] = {"attention_mask": attention_mask}
    for extra_key in ("token_type_ids", "position_ids"):
        if extra_key in encoding:
            fixed_kwargs[extra_key] = encoding[extra_key]
    with torch.no_grad():
        reference_logits = adapter(inputs_embeds=embeddings, **fixed_kwargs)
    scorer, target_representation, _position = _resolve_target(
        target, reference_logits, tokenizer, attention_mask, family
    )

    def _run(grid: int) -> tuple[Tensor, dict[str, Any]]:
        """Run inputs_embeds IG at one grid; return (L, D) values + extra."""

        result = integrated_gradients(
            adapter,
            (),
            {"inputs_embeds": embeddings, **fixed_kwargs},
            target=scorer,
            n_steps=grid,
            baseline={
                "inputs": (),
                "input_kwargs": {
                    "inputs_embeds": baseline_embeddings,
                    **dict(fixed_kwargs.items()),
                },
            },
            step_batch_size=steps_per_batch,
        )
        values = result.values
        if isinstance(values, Tensor):
            grid_values = values[0]
        else:
            raise AttributionError(
                "text() expected one attributed embeddings leaf",
                code="text_input_invalid",
            )
        return grid_values.detach(), result.extra

    grids_evaluated: list[int] = []
    residuals: list[float] = []
    stability: float | None = None
    grid_pair: tuple[int, int] | None = None
    converged: bool | None
    physical_calls = 0

    if n_steps == "auto":
        previous_scores: Tensor | None = None
        values_by_grid: Tensor | None = None
        extra_by_grid: dict[str, Any] = {}
        converged = False
        for grid in _AUTO_LADDER:
            values_by_grid, extra_by_grid = _run(grid)
            physical_calls += int(extra_by_grid.get("physical_forward_calls", 0))
            grids_evaluated.append(grid)
            residuals.append(float(extra_by_grid["residual_rel"]))
            scores_now = values_by_grid.sum(dim=-1)
            if previous_scores is not None:
                stability = _stability_change(previous_scores, scores_now)
                grid_pair = (grids_evaluated[-2], grid)
                # Stop ONLY on the DUAL criterion: residual AND stability
                # (residual-only stopping is measured-unsound, D24; rank
                # stability is banned outright, D25).
                if _dual_criterion_met(residuals[-1], stability):
                    converged = True
                    break
            previous_scores = scores_now
        if values_by_grid is None:
            raise AttributionError(
                "internal error: the auto ladder evaluated no grid. This is a "
                "TorchLens contract breach, not a user error. Remedy: report "
                "this as a bug.",
                code="text_auto_ladder_empty",
            )
        final_values, final_extra = values_by_grid, extra_by_grid
        final_grid = grids_evaluated[-1]
    else:
        final_values, final_extra = _run(int(n_steps))
        physical_calls += int(final_extra.get("physical_forward_calls", 0))
        grids_evaluated.append(int(n_steps))
        residuals.append(float(final_extra["residual_rel"]))
        final_grid = int(n_steps)
        converged = None

    scores = final_values.sum(dim=-1)
    completeness = _completeness_extra(final_extra["attribution_sum"], final_extra["target_delta"])
    # Explicit baseline spellings remain available and ALL warn when the
    # achieved residual exceeds the tolerance (D23) -- the user chose the
    # baseline, so the certificate shortfall is disclosed loudly.
    if baseline != "auto" and completeness["residual_rel"] > _RESIDUAL_TOLERANCE:
        warnings.warn(
            AttributionWarning(
                f"text() achieved residual_rel "
                f"{completeness['residual_rel']:.2%} under the explicit "
                f"baseline {baseline_disclosure['policy']!r}, above the 1% "
                "tolerance. Remedy: raise n_steps (or pass n_steps='auto'), "
                "or choose a task-appropriate baseline (baseline='auto').",
                code="text_residual_above_tolerance",
            ),
            stacklevel=2,
        )

    raw_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    offsets = None
    if offsets_available and "offset_mapping" in encoding:
        offsets = [tuple(pair) for pair in encoding["offset_mapping"][0].tolist()]

    result = TokenAttributionResult(
        text=text,
        input_ids=input_ids[0].tolist(),
        raw_tokens=list(raw_tokens),
        display_tokens=_display_tokens(list(raw_tokens)),
        offsets=offsets,
        special_tokens_mask=(special_mask[0].tolist() if special_mask is not None else None),
        scores=scores,
        values=final_values,
        target_repr=target_representation,
        baseline=baseline_disclosure,
        n_steps=final_grid,
        converged=converged,
        convergence={
            "requested": n_steps,
            "grids_evaluated": grids_evaluated,
            "residual_rel_by_grid": residuals,
            "stability": stability,
            "grid_pair": grid_pair,
            "criteria": "residual_rel <= 1% AND max(L1, L2) successive "
            "change <= 1%; rank stability is banned as a signal",
        },
        completeness=completeness,
        cost={
            "physical_forward_calls": physical_calls,
            "steps_per_batch": steps_per_batch,
            "step_audit": final_extra.get("step_audit"),
            "path_evaluations_logical": sum(grids_evaluated),
        },
        provenance={
            "model_class": type(model).__name__,
            "model_name_or_path": getattr(getattr(model, "config", None), "_name_or_path", None),
            "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
            "transformers_version": _transformers_version(),
            "task_family": family,
            "task_family_reason": family_reason,
        },
        truncation=truncation_record,
    )
    return result


def _transformers_version() -> str | None:
    """Return the installed transformers version, or ``None``."""

    try:
        import transformers

        return str(transformers.__version__)
    except Exception:  # noqa: BLE001 - transformers optional
        return None


__all__ = ["TokenAttributionPayload", "TokenAttributionResult", "text"]
