"""LIT (Google PAIR Language Interpretability Tool) bridge.

The real two-task bridge (LIT-panel memo, 2026-08-26): ``model()`` wraps a
LIVE torch model + tokenizer as a genuine ``lit_nlp.api.model.Model`` so LIT's
editor, projector, metrics, and side-by-side panels drive it over real HTTP.
TorchLens differentiation: LIT users hand-write per-model wrappers to expose
one or two fields; this factory generates the wrapper for any torch model
exposing any traced site, with no ``output_hidden_states`` plumbing and no
model-code changes.

Import-inert without the peer (the L8 rule): ``import torchlens.bridge.lit``
never imports ``lit_nlp``; the factory gates at call time. Every public
spelling here is DOCUMENTED-UNSTABLE pending naming-session ratification
(memo section 11). The bridge is EXPERIMENTAL: upstream lit-nlp has been
dormant externally since 2024-12 (its typed Model API is exactly why the
bridge is safe to code against).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn

from . import _pooling, _refusals, _runtime, _sites

__all__ = ["dataset", "layout", "model"]

_PROBE_TEXT = "torchlens LIT bridge probe."
_TASKS = ("classification", "causal_lm")


def _require_lit_nlp() -> None:
    """Import-gate the foreign peer at call time.

    Raises
    ------
    torchlens._errors.MissingDependencyError
        Code ``lit_dependency_missing`` when ``lit_nlp`` is absent.
    """

    try:
        import lit_nlp.api.model  # noqa: F401
    except ImportError as exc:
        _refusals.refuse_lit_missing(exc)


def _validate_net(net: Any) -> nn.Module:
    """Demand a live ``nn.Module``, teaching the Trace case explicitly.

    Parameters
    ----------
    net:
        The first factory argument.

    Returns
    -------
    nn.Module
        The validated live model.

    Raises
    ------
    torchlens._errors.ArgumentTypeError
        Code ``lit_trace_not_executable`` (memo D2).
    """

    if not isinstance(net, nn.Module):
        _refusals.refuse_trace_passed(net)
    return net


def _resolve_labels(net: nn.Module, labels: Iterable[str] | None) -> tuple[str, ...]:
    """Resolve and validate the classifier label vocabulary (memo D19).

    Parameters
    ----------
    net:
        The live model (its ``config.id2label`` is the default vocabulary).
    labels:
        Optional user override, validated against the config.

    Returns
    -------
    tuple[str, ...]
        The label vocabulary in class-index order.

    Raises
    ------
    torchlens._errors.InvalidArgumentError
        Code ``lit_labels_invalid`` on a count mismatch or missing vocabulary.
    """

    config = getattr(net, "config", None)
    id2label = getattr(config, "id2label", None) or {}
    defaults = tuple(str(id2label[key]) for key in sorted(id2label, key=int))
    if labels is None:
        if not defaults:
            _refusals.refuse_labels_invalid(
                "the model config exposes no id2label vocabulary, so labels= is "
                "required for the classification adapter",
                remedy='Pass labels=("negative", "positive", ...) in class-index order.',
            )
        return defaults
    override = tuple(str(label) for label in labels)
    if defaults and len(override) != len(defaults):
        _refusals.refuse_labels_invalid(
            f"labels= carries {len(override)} names but the model config declares "
            f"{len(defaults)} classes ({', '.join(defaults)})",
            remedy="Pass exactly one label per class, in class-index order.",
        )
    return override


def _resolve_salience_layer(net: nn.Module) -> str:
    """Resolve the input-embedding module name for the salience field.

    Parameters
    ----------
    net:
        The live model.

    Returns
    -------
    str
        The ``named_modules`` name of ``net.get_input_embeddings()``.

    Raises
    ------
    torchlens._errors.InvalidArgumentError
        Code ``lit_salience_unavailable``.
    """

    getter = getattr(net, "get_input_embeddings", None)
    embedding = getter() if callable(getter) else None
    if isinstance(embedding, nn.Module):
        for module_name, module in net.named_modules():
            if module is embedding:
                return module_name
    _refusals.refuse_salience_unavailable(
        "get_input_embeddings() is missing or does not return a module owned by the model"
    )


def _pin_sites(probe: Any, sites: Any) -> list[_sites.SiteSpec]:
    """Pin the requested sites on the probe trace (memo D7).

    Parameters
    ----------
    probe:
        The unpadded probe trace.
    sites:
        ``"blocks"``, an iterable of explicit spellings, ``()`` for no
        embedding fields, or None (refused: the bridge never guesses).

    Returns
    -------
    list[_sites.SiteSpec]
        The pinned site quads.
    """

    if sites is None:
        _refusals.refuse_sites_unspecified(_candidate_stack(probe))
    if isinstance(sites, str):
        if sites != "blocks":
            _refusals.refuse_site_unresolvable(sites, ("blocks",))
        return _sites.pin_blocks(probe)
    return _sites.pin_explicit(probe, tuple(str(site) for site in sites))


def _candidate_stack(probe: Any) -> tuple[str, ...]:
    """Best-effort block-stack candidates for the DEFAULT-SITES refusal.

    Parameters
    ----------
    probe:
        The unpadded probe trace.

    Returns
    -------
    tuple[str, ...]
        Discovered stack addresses, or empty when discovery itself refuses.
    """

    try:
        return _sites.discover_block_stack(probe)
    except Exception:  # noqa: BLE001 -- refusal path only feeds the message
        return ()


def _save_selection(probe: Any, specs: list[_sites.SiteSpec]) -> Any:
    """Build the predict-time selective ``save=`` selection (memo D18).

    Saves the pinned sites' modules plus the modules feeding the model output
    (so ``reconstruct_output()`` stays serviceable). Falls back to the
    save-everything default when any output feed sits outside every module --
    more memory, never wrong.

    Parameters
    ----------
    probe:
        The unpadded probe trace.
    specs:
        The pinned site quads.

    Returns
    -------
    Any
        A composed selector, or None for the save-everything default.
    """

    from torchlens import in_module

    label_map: dict[str, Any] = {}
    for op in probe.layer_list:
        label_map[str(op.label)] = op
        label_map[str(op.label).rsplit(":", 1)[0]] = op

    addresses = {spec.module_address for spec in specs}
    for out_op in probe.output_ops:
        for parent_label in out_op.parents:
            parent = label_map.get(str(parent_label))
            stack = tuple(getattr(parent, "module_call_stack", ())) if parent else ()
            if not stack:
                return None
            addresses.add(stack[-1].rsplit(":", 1)[0])

    selection = None
    for address in sorted(addresses):
        selector = in_module(address)
        selection = selector if selection is None else (selection | selector)
    return selection


def model(
    net: Any,
    tokenizer: Any = None,
    *,
    task: str = "classification",
    sites: Any = None,
    pooling: Any = "mean_masked",
    labels: Iterable[str] | None = None,
    salience: bool = False,
    name: str | None = None,
    max_minibatch_size: int = 8,
    max_new_tokens: int = 16,
    top_k: int = 10,
    attention: Any = None,
) -> Any:
    """Wrap a live torch model as a real LIT ``Model`` (memo D1-D11, D18-D19).

    Parameters
    ----------
    net:
        The LIVE torch model. A finished Trace refuses: LIT calls ``predict()``
        on NEW inputs (edited text, counterfactuals), which a historical trace
        cannot execute (memo D2).
    tokenizer:
        The checkpoint's tokenizer. Never mutated; batches are padded manually
        with ``pad_token_id`` (falling back to ``eos_token_id``).
    task:
        ``"classification"`` or ``"causal_lm"`` (memo D3).
    sites:
        Which activations to expose as LIT ``Embeddings`` fields: the
        gate-passed ``"blocks"`` preset, explicit module addresses (optionally
        pass-qualified ``"address:N"``), or ``()`` for none. Required -- the
        bridge refuses rather than guesses (memo D7; [UI-SPRINT] item 4).
    pooling:
        ``"mean_masked"`` / ``"first_token"`` / ``"last_unmasked"`` or a
        callable ``(tensor, attention_mask) -> [batch, features]`` (memo D10).
    labels:
        Classifier vocabulary override, validated against ``config.id2label``
        (memo D19; the textattack SST-2 checkpoint ships useless LABEL_0/1).
    salience:
        Opt-in model-provided ``TokenSalience`` from the torchlens attribution
        kit (memo D11; ``autorun=False`` is fixed cost semantics).
    name:
        Display name; defaults to the model class name.
    max_minibatch_size:
        LIT minibatch bound (memo D9: batching is what makes warm_start fast).
    max_new_tokens:
        Fixed deterministic-generation length (causal LM only).
    top_k:
        Next-token candidates per row (causal LM only).
    attention:
        REFUSED (memo D12): LIT deleted its attention panel upstream; the type
        serializes into a void.

    Returns
    -------
    Any
        A ``lit_nlp.api.model.Model`` instance for the requested task.

    Raises
    ------
    torchlens._errors.MissingDependencyError
        Code ``lit_dependency_missing`` when ``lit_nlp`` is absent.
    torchlens._errors.ArgumentTypeError
        Code ``lit_trace_not_executable`` for a Trace or non-module ``net``.
    torchlens._errors.InvalidArgumentError
        The construction-refusal family (task, tokenizer, sites, pooling,
        labels, salience, attention codes; see the error-refusal contract).
    """

    if attention is not None:
        _refusals.refuse_attention_requested()
    live = _validate_net(net)
    _require_lit_nlp()
    if task not in _TASKS:
        _refusals.refuse_task_invalid(task)
    _runtime.validate_tokenizer(tokenizer)
    pooling_spec = _pooling.validate_pooling(pooling)
    display_name = name if name is not None else type(live).__name__

    resolved_labels = _resolve_labels(live, labels) if task == "classification" else ()
    salience_layer = _resolve_salience_layer(live) if salience else None

    probe_ids = _runtime.encode_texts(tokenizer, [_PROBE_TEXT])
    device = _runtime.model_device(live)
    input_ids, mask = _runtime.pad_batch(
        probe_ids, _runtime.pad_token_id(tokenizer) or 0, "right", device
    )
    probe_kwargs: dict[str, Any] = {"input_ids": input_ids, "attention_mask": mask}
    if task == "causal_lm":
        probe_kwargs["use_cache"] = False
    with _runtime.operation_state(live), torch.no_grad():
        probe = _runtime.traced_forward(live, probe_kwargs, save=None)

    specs = _pin_sites(probe, sites)
    from ._adapters import AdapterConfig, TorchLensLitCausalLM, TorchLensLitClassifier

    config = AdapterConfig(
        name=display_name,
        specs=tuple(specs),
        pooling=pooling_spec,
        salience_layer=salience_layer,
        save_selection=_save_selection(probe, specs),
        max_minibatch_size=max_minibatch_size,
    )
    if task == "classification":
        return TorchLensLitClassifier(live, tokenizer, config, resolved_labels)
    return TorchLensLitCausalLM(live, tokenizer, config, max_new_tokens, top_k)


def dataset(
    examples: Iterable[Any],
    *,
    labels: Iterable[str] | None = None,
    description: str = "TorchLens LIT bridge dataset.",
) -> Any:
    """Build a tiny native LIT dataset (memo D19).

    Parameters
    ----------
    examples:
        Strings, or dicts carrying ``text`` (and optionally ``label``).
    labels:
        Classifier vocabulary; adds a ``label`` CategoryLabel field whose
        missing values default to the first entry.
    description:
        Human-readable dataset description.

    Returns
    -------
    Any
        A ``lit_nlp.api.dataset.Dataset``.

    Raises
    ------
    torchlens._errors.MissingDependencyError
        Code ``lit_dependency_missing`` when ``lit_nlp`` is absent.
    """

    _require_lit_nlp()
    from ._adapters import build_dataset

    vocab = tuple(str(label) for label in labels) if labels is not None else None
    return build_dataset(examples, vocab, description)


def layout() -> Any:
    """Build the pure-Python LIT layout foregrounding TorchLens fields.

    Returns
    -------
    Any
        A ``lit_nlp.api.layout.LitCanonicalLayout`` (memo D16).

    Raises
    ------
    torchlens._errors.MissingDependencyError
        Code ``lit_dependency_missing`` when ``lit_nlp`` is absent.
    """

    _require_lit_nlp()
    from ._adapters import build_layout

    return build_layout()
