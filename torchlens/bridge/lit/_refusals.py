"""Typed teaching refusals for the LIT bridge.

Every refusal the bridge can issue lives here as one function with one stable
machine-readable code, so the S-17 census, the error-refusal contract doc, and
the raise sites stay in lockstep. Codes are DOCUMENTED-UNSTABLE spellings
pending naming-session ratification (LIT-panel memo, sections 3 and 11).
"""

from __future__ import annotations

from typing import Any, NoReturn

from torchlens._errors import (
    ArgumentTypeError,
    InvalidArgumentError,
    MissingDependencyError,
    RecordBindingError,
)


def refuse_lit_missing(cause: BaseException) -> NoReturn:
    """Refuse a bridge call because ``lit_nlp`` is not installed.

    Parameters
    ----------
    cause:
        The underlying import failure.

    Raises
    ------
    MissingDependencyError
        Always; code ``lit_dependency_missing``.
    """

    raise MissingDependencyError(
        "LIT bridge requires the `lit` extra: the `lit_nlp` package is not importable.",
        code="lit_dependency_missing",
        remedy="pip install torchlens[lit] (isolated environment recommended; installing "
        "lit-nlp moves numpy to the 1.x line and adds ~83 packages)",
        dependency="lit_nlp",
        install="pip install torchlens[lit]",
    ) from cause


def refuse_trace_passed(value: Any) -> NoReturn:
    """Refuse a finished Trace where a live model is required.

    LIT calls ``predict()`` on NEW inputs -- edited text, counterfactuals -- so
    a historical trace is insufficient by construction (LIT-panel memo D2):
    nothing can execute edited text without the live model.

    Parameters
    ----------
    value:
        The Trace-like object the caller passed.

    Raises
    ------
    ArgumentTypeError
        Always; code ``lit_trace_not_executable``.
    """

    raise ArgumentTypeError(
        "tl.bridge.lit.model() needs the LIVE model and tokenizer, but received a "
        f"finished {type(value).__name__}. A historical trace cannot execute the "
        "edited text LIT sends to predict().",
        code="lit_trace_not_executable",
        remedy="Call tl.bridge.lit.model(net, tokenizer, task=..., sites=...) with the "
        "live nn.Module; the bridge traces it per request.",
    )


def refuse_task_invalid(task: Any) -> NoReturn:
    """Refuse an unknown ``task=`` value.

    Parameters
    ----------
    task:
        The rejected task spelling.

    Raises
    ------
    InvalidArgumentError
        Always; code ``lit_task_invalid``.
    """

    raise InvalidArgumentError(
        f"task={task!r} is not a supported LIT bridge task.",
        code="lit_task_invalid",
        remedy='Pass task="classification" (sequence classification) or '
        'task="causal_lm" (deterministic generation).',
        supported=("classification", "causal_lm"),
    )


def refuse_tokenizer_invalid(problem: str, remedy: str) -> NoReturn:
    """Refuse an unusable tokenizer.

    Parameters
    ----------
    problem:
        What is wrong with the tokenizer (missing, uncallable, or unpaddable).
    remedy:
        The exact fix.

    Raises
    ------
    InvalidArgumentError
        Always; code ``lit_tokenizer_invalid``.
    """

    raise InvalidArgumentError(problem, code="lit_tokenizer_invalid", remedy=remedy)


def refuse_sites_unspecified(candidates: tuple[str, ...]) -> NoReturn:
    """Refuse a factory call with no ``sites=`` (DEFAULT-SITES refusal branch).

    Explicit ``sites=`` is the contract (LIT-panel memo D7); the memo's
    [UI-SPRINT] item 4 records both one-line branches and this build ships the
    teaching-refusal branch: name the preset, list the candidates, never guess.

    Parameters
    ----------
    candidates:
        Candidate block-stack module addresses discovered on the probe trace.

    Raises
    ------
    InvalidArgumentError
        Always; code ``lit_sites_unspecified``.
    """

    listing = ", ".join(candidates) if candidates else "(no repeated-module stack found)"
    raise InvalidArgumentError(
        "tl.bridge.lit.model() requires an explicit sites= selection; the bridge never "
        f"guesses which activations to expose. Candidate block stacks: {listing}.",
        code="lit_sites_unspecified",
        remedy='Pass sites="blocks" for the gate-passed repeated-block preset, an '
        "explicit tuple of module addresses, or sites=() for no embedding fields.",
        candidates=candidates,
    )


def refuse_blocks_preset_unavailable(detail: str, candidates: tuple[str, ...]) -> NoReturn:
    """Refuse the ``sites="blocks"`` preset when no homogeneous stack exists.

    Parameters
    ----------
    detail:
        Why discovery failed (none found, or an unresolvable tie).
    candidates:
        Candidate module addresses to name in the teaching message.

    Raises
    ------
    InvalidArgumentError
        Always; code ``lit_blocks_preset_unavailable``.
    """

    listing = ", ".join(candidates) if candidates else "(none)"
    raise InvalidArgumentError(
        f'sites="blocks" could not discover one homogeneous repeated-module stack: '
        f"{detail}. Candidate addresses: {listing}.",
        code="lit_blocks_preset_unavailable",
        remedy="Pass explicit module addresses via sites=(...); candidates above are "
        "the module-call metadata this probe saw.",
        candidates=candidates,
    )


def refuse_site_unresolvable(site: str, available: tuple[str, ...]) -> NoReturn:
    """Refuse an explicit site that matches nothing on the probe trace.

    Parameters
    ----------
    site:
        The requested site spelling.
    available:
        A bounded sample of module addresses that do exist.

    Raises
    ------
    InvalidArgumentError
        Always; code ``lit_site_unresolvable``.
    """

    raise InvalidArgumentError(
        f"sites entry {site!r} matched no module-output op on the probe trace.",
        code="lit_site_unresolvable",
        remedy="Pass a module address from the traced module tree (for example "
        f"one of: {', '.join(available)}), optionally pass-qualified as 'address:N'.",
        site=site,
        available=available,
    )


def refuse_site_ambiguous(site: str, passes: tuple[int, ...]) -> NoReturn:
    """Refuse an explicit site whose module-output op has multiple passes.

    Explicit sites carry their pass in the key or refuse (LIT-panel memo D8);
    only the ``blocks`` preset applies the tested last-pass rule.

    Parameters
    ----------
    site:
        The requested site spelling.
    passes:
        The 1-based pass indexes available at this site.

    Raises
    ------
    InvalidArgumentError
        Always; code ``lit_site_ambiguous``.
    """

    spellings = ", ".join(f"'{site}:{p}'" for p in passes)
    raise InvalidArgumentError(
        f"sites entry {site!r} is pass-ambiguous: the module fires {len(passes)} times "
        "per forward and an unqualified spelling would silently pick one.",
        code="lit_site_ambiguous",
        remedy=f"Pass one pass-qualified spelling: {spellings}.",
        site=site,
        passes=passes,
    )


def refuse_site_key_drift(field_name: str, address: str, site_key: str) -> NoReturn:
    """Refuse a predict-time trace that does not reproduce a pinned site.

    The refusal names both the module address and the structural key (LIT-panel
    memo D4): ``site_key`` is root-relative, so a different wrapper class or a
    module-tree rename correctly lands here rather than silently rebinding.

    Parameters
    ----------
    field_name:
        The LIT output field pinned to this site.
    address:
        The pinned module address.
    site_key:
        The pinned structural site key.

    Raises
    ------
    RecordBindingError
        Always; code ``lit_site_key_drift``.
    """

    raise RecordBindingError(
        f"LIT field {field_name!r} is pinned to module {address!r} with structural "
        f"site key {site_key!r}, but this forward pass did not reproduce that site.",
        code="lit_site_key_drift",
        remedy="Rebuild the LIT model wrapper for this model object (the module tree "
        "or wrapper class changed since construction); do not rebind by label.",
        field_name=field_name,
        address=address,
        site_key=site_key,
    )


def refuse_pooling_invalid(pooling: Any, supported: tuple[str, ...]) -> NoReturn:
    """Refuse an unknown pooling strategy spelling.

    Parameters
    ----------
    pooling:
        The rejected pooling value.
    supported:
        The closed set of named strategies.

    Raises
    ------
    InvalidArgumentError
        Always; code ``lit_pooling_invalid``.
    """

    raise InvalidArgumentError(
        f"pooling={pooling!r} is not a supported pooling strategy.",
        code="lit_pooling_invalid",
        remedy=f"Pass one of {supported} or a callable "
        "(tensor, attention_mask) -> [batch, features].",
        supported=supported,
    )


def refuse_pooling_unsupported(field_name: str, detail: str) -> NoReturn:
    """Refuse a site payload the pooling contract cannot honestly reduce.

    Never batch-zero, never PAD positions averaged in, never silent flattening
    (LIT-panel memo D10): a payload outside the supported shape contract is a
    refusal, not a guess.

    Parameters
    ----------
    field_name:
        The LIT output field whose payload is unpoolable.
    detail:
        The observed payload problem (non-tensor, unsupported rank, ...).

    Raises
    ------
    RecordBindingError
        Always; code ``lit_pooling_unsupported``.
    """

    raise RecordBindingError(
        f"LIT field {field_name!r} cannot be pooled: {detail}.",
        code="lit_pooling_unsupported",
        remedy="Select a site whose output is a [batch, ...] tensor with batch on "
        "axis 0 (rank 2-4), or pass a custom pooling callable for this geometry.",
        field_name=field_name,
    )


def refuse_attention_requested() -> NoReturn:
    """Refuse the ``attention=`` argument (LIT-panel memo D12).

    LIT removed its attention-visualization module upstream on 2024-06-20; the
    ``AttentionHeads`` type still serializes -- into nothing. Shipping a field no
    panel renders is the deleted stub's sin at smaller scale, so the argument
    refuses with the removal cited.

    Raises
    ------
    InvalidArgumentError
        Always; code ``lit_attention_unsupported``.
    """

    raise InvalidArgumentError(
        "attention= is not supported: LIT deleted its attention-visualization panel "
        "upstream (PAIR-code/lit, 2024-06-20), so an AttentionHeads field would "
        "serialize into a void no panel renders.",
        code="lit_attention_unsupported",
        remedy="Use torchlens facets/visualization for attention analysis; expose "
        "activations to LIT via sites= Embeddings fields instead.",
    )


def refuse_labels_invalid(problem: str, remedy: str) -> NoReturn:
    """Refuse a ``labels=`` override that contradicts the model config.

    Parameters
    ----------
    problem:
        The observed mismatch.
    remedy:
        The exact fix.

    Raises
    ------
    InvalidArgumentError
        Always; code ``lit_labels_invalid``.
    """

    raise InvalidArgumentError(problem, code="lit_labels_invalid", remedy=remedy)


def refuse_salience_unavailable(detail: str) -> NoReturn:
    """Refuse ``salience=True`` when the token-embedding site cannot be found.

    Parameters
    ----------
    detail:
        Why the input-embedding module could not be resolved.

    Raises
    ------
    InvalidArgumentError
        Always; code ``lit_salience_unavailable``.
    """

    raise InvalidArgumentError(
        f"salience=True needs the model's input-embedding module and it could not be "
        f"resolved: {detail}.",
        code="lit_salience_unavailable",
        remedy="Expose get_input_embeddings() on the model (the HF convention) or "
        "construct with salience=False.",
    )


def refuse_model_output_unsupported(task: str, detail: str) -> NoReturn:
    """Refuse a model output the task adapter cannot honestly serve.

    Parameters
    ----------
    task:
        The adapter task name.
    detail:
        The observed output problem.

    Raises
    ------
    RecordBindingError
        Always; code ``lit_model_output_unsupported``.
    """

    raise RecordBindingError(
        f"The {task} adapter cannot read this model's output: {detail}.",
        code="lit_model_output_unsupported",
        remedy="The adapter needs logits of shape [batch, num_labels] "
        "(classification) or [batch, seq, vocab] (causal_lm), as a tensor, "
        "a .logits attribute, or the first tuple element.",
        task=task,
    )


def refuse_capture_incomplete(status: Any) -> NoReturn:
    """Refuse a predict-time capture that did not settle COMPLETE.

    Parameters
    ----------
    status:
        The settled capture status.

    Raises
    ------
    RecordBindingError
        Always; code ``lit_capture_incomplete``.
    """

    raise RecordBindingError(
        f"The traced forward settled {status!r}, not COMPLETE; LIT rows will not be "
        "fabricated from a partial capture.",
        code="lit_capture_incomplete",
        remedy="Inspect the capture failure (trace.outcome) and re-run; the bridge "
        "only serves fields from COMPLETE captures.",
        status=str(status),
    )
