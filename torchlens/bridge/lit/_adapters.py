"""Real ``lit_nlp.api.model.Model`` adapters over a live torch model.

This module imports ``lit_nlp`` at module level and is therefore only imported
AFTER the factory's dependency gate; ``import torchlens.bridge.lit`` stays
import-inert without the peer (the L8 rule).

Two task adapters ship in v1 (LIT-panel memo D3): sequence classification and
basic causal LM with deterministic generation. Every output field routes
through the field-converter registry (memo section 8 plumbing) -- which ships
with ZERO attention entries: LIT deleted its attention panel upstream and a
field no panel renders must not exist (memo D12).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import torch
from lit_nlp.api import dtypes as lit_dtypes, model as lit_model, types as lit_types
from torch import nn

from . import _pooling, _refusals, _runtime, _sites


@dataclass(frozen=True)
class FieldConverter:
    """One LIT output-field family the bridge can emit.

    Parameters
    ----------
    kind:
        Registry key (``"multiclass_preds"``, ``"embeddings"``, ...).
    tasks:
        Adapter tasks the family applies to.
    build_spec:
        ``config -> LitType`` factory for the output-spec entry.
    """

    kind: str
    tasks: tuple[str, ...]
    build_spec: Callable[[dict[str, Any]], Any]


def _multiclass_spec(config: dict[str, Any]) -> Any:
    """Build the classifier probability field spec.

    Parameters
    ----------
    config:
        Requires ``labels`` (vocab) and ``label_field`` (dataset parent).

    Returns
    -------
    Any
        ``lit_types.MulticlassPreds``.
    """

    return lit_types.MulticlassPreds(
        vocab=list(config["labels"]), parent=str(config["label_field"])
    )


def _tokens_spec(config: dict[str, Any]) -> Any:
    """Build the wordpiece tokens field spec.

    Parameters
    ----------
    config:
        Requires ``text_field`` (input parent).

    Returns
    -------
    Any
        ``lit_types.Tokens``.
    """

    return lit_types.Tokens(parent=str(config["text_field"]))


def _generated_text_spec(config: dict[str, Any]) -> Any:
    """Build the deterministic-generation field spec.

    Parameters
    ----------
    config:
        Requires ``text_field`` (input parent).

    Returns
    -------
    Any
        ``lit_types.GeneratedText``.
    """

    return lit_types.GeneratedText(parent=str(config["text_field"]))


def _token_top_k_spec(config: dict[str, Any]) -> Any:
    """Build the next-token top-k field spec.

    Parameters
    ----------
    config:
        Requires ``tokens_field`` (the aligned Tokens output field).

    Returns
    -------
    Any
        ``lit_types.TokenTopKPreds``.
    """

    return lit_types.TokenTopKPreds(align=str(config["tokens_field"]))


def _embeddings_spec(config: dict[str, Any]) -> Any:
    """Build one pooled-activation field spec.

    Parameters
    ----------
    config:
        Unused; present for the uniform registry signature.

    Returns
    -------
    Any
        ``lit_types.Embeddings``.
    """

    del config
    return lit_types.Embeddings()


def _token_salience_spec(config: dict[str, Any]) -> Any:
    """Build the model-provided token-salience field spec.

    ``autorun=False`` is cost semantics, not naming (memo D11): salience costs
    a backward pass per minibatch, and ``autorun=True`` would fire it for the
    whole dataset at load.

    Parameters
    ----------
    config:
        Unused; present for the uniform registry signature.

    Returns
    -------
    Any
        ``lit_types.TokenSalience`` (signed).
    """

    del config
    return lit_types.TokenSalience(autorun=False, signed=True)


# The v1 field-converter registry (memo section 8): predictions, embeddings,
# and salience route through it. Deliberately ZERO attention entries -- landing
# attention requires a maintained visible renderer upstream plus real
# facets-pattern, alignment, shape, and browser-render tests (memo D12);
# successful serialization is explicitly insufficient.
FIELD_CONVERTERS: dict[str, FieldConverter] = {
    "multiclass_preds": FieldConverter("multiclass_preds", ("classification",), _multiclass_spec),
    "tokens": FieldConverter("tokens", ("classification", "causal_lm"), _tokens_spec),
    "generated_text": FieldConverter("generated_text", ("causal_lm",), _generated_text_spec),
    "token_top_k": FieldConverter("token_top_k", ("causal_lm",), _token_top_k_spec),
    "embeddings": FieldConverter("embeddings", ("classification", "causal_lm"), _embeddings_spec),
    "token_salience": FieldConverter(
        "token_salience", ("classification", "causal_lm"), _token_salience_spec
    ),
}

_TEXT_FIELD = "text"
_LABEL_FIELD = "label"
_SALIENCE_FIELD = "tl_salience"


@dataclass(frozen=True)
class AdapterConfig:
    """Construction-time state shared by both task adapters.

    Parameters
    ----------
    name:
        Display name LIT shows for the model.
    specs:
        The pinned site quads (memo D4).
    pooling:
        Validated pooling strategy.
    salience_layer:
        ``named_modules`` name of the input-embedding module, or None when
        salience is off.
    save_selection:
        TorchLens ``save=`` value for predict-time traces (None = save all).
    max_minibatch_size:
        LIT minibatch bound.
    """

    name: str
    specs: tuple[_sites.SiteSpec, ...]
    pooling: _pooling.PoolingSpec
    salience_layer: str | None
    save_selection: Any
    max_minibatch_size: int


class _TorchLensLitModel(lit_model.BatchedModel):
    """Shared base for the TorchLens LIT adapters.

    TorchLens wraps a LIVE model: every ``predict_minibatch`` tokenizes the
    request, runs ONE padded traced forward (memo D9), re-resolves the pinned
    sites structurally (memo D4/D5), and reads activations per-op (memo D6).
    """

    def __init__(self, net: nn.Module, tokenizer: Any, config: AdapterConfig) -> None:
        """Initialize the adapter around the live model.

        Parameters
        ----------
        net:
            The live torch model.
        tokenizer:
            The checkpoint's tokenizer (never mutated by the bridge).
        config:
            Frozen construction-time adapter state.
        """

        self._net = net
        self._tokenizer = tokenizer
        self._config = config
        pad_id = _runtime.pad_token_id(tokenizer)
        if pad_id is None:
            _refusals.refuse_tokenizer_invalid(
                "tokenizer exposes neither pad_token_id nor eos_token_id, so ragged LIT "
                "requests cannot be padded",
                remedy="Use a tokenizer with a pad or eos token (the factory validates "
                "this before construction).",
            )
        self._pad_id: int = pad_id

    @classmethod
    def init_spec(cls) -> dict[str, Any] | None:
        """Return None: this wrapper is not constructable from a JSON spec.

        The factory takes live Python objects (model, tokenizer), so LIT's UI
        cannot instantiate new copies; returning the explicit literal keeps
        LIT's constructor-inference warning out of the logs (memo D19).

        Returns
        -------
        dict[str, Any] | None
            Always None.
        """

        return None

    def description(self) -> str:
        """Return the human-readable model description LIT displays.

        Returns
        -------
        str
            Display name plus the honest wrapper description.
        """

        return (
            f"{self._config.name}: live torch model served by TorchLens "
            "(experimental bridge; activations exposed per traced site)."
        )

    @property
    def supports_concurrent_predictions(self) -> bool:
        """Return False: TorchLens tracing is not re-entrant (memo D18).

        Returns
        -------
        bool
            Always False; two browser tabs must serialize.
        """

        return False

    def max_minibatch_size(self) -> int:
        """Return the LIT minibatch bound.

        Returns
        -------
        int
            The configured minibatch size.
        """

        return self._config.max_minibatch_size

    def input_spec(self) -> dict[str, Any]:
        """Return the LIT input spec.

        Returns
        -------
        dict[str, Any]
            One required ``text`` segment.
        """

        return {_TEXT_FIELD: lit_types.TextSegment()}

    def _tokens_rows(self, id_lists: list[list[int]]) -> list[list[str]]:
        """Convert per-row token ids to wordpiece strings.

        Parameters
        ----------
        id_lists:
            Unpadded per-row token ids.

        Returns
        -------
        list[list[str]]
            Per-row token strings.
        """

        convert = getattr(self._tokenizer, "convert_ids_to_tokens", None)
        if convert is not None:
            return [list(convert(ids)) for ids in id_lists]
        return [[self._tokenizer.decode([i]) for i in ids] for ids in id_lists]

    def _embedding_columns(self, log: Any, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """Resolve pinned sites on a predict-time trace and pool them.

        Parameters
        ----------
        log:
            The finished predict-time trace.
        attention_mask:
            ``[batch, tokens]`` mask of the traced request.

        Returns
        -------
        dict[str, torch.Tensor]
            ``field_name -> [batch, features]`` pooled activations.
        """

        resolved = _sites.resolve_pinned(log, list(self._config.specs))
        return {
            field: _pooling.pool(field, op.out, attention_mask, self._config.pooling)
            for field, op in resolved.items()
        }

    def _salience_values(
        self,
        input_kwargs: dict[str, Any],
        target: Callable[[Any], torch.Tensor],
    ) -> torch.Tensor:
        """Compute signed per-token salience at the input-embedding site.

        Semantics per memo D11: signed activation-times-gradient of the target
        logits with respect to the input-embedding output, summed over the
        hidden axis -- never called "integrated gradients". Computed with one
        extra forward + backward per minibatch. (The attribution kit's
        ``layer_attribution`` refuses integer-only inputs today -- it demands a
        floating-point INPUT leaf -- so the bridge computes the identical
        activation_x_grad semantic at the embedding site directly; routing
        through the kit is a named seam for the attribution lane.)

        Parameters
        ----------
        input_kwargs:
            Forward kwargs for the salience run (same padded batch).
        target:
            ``model_output -> scalar`` selecting the per-row target logits.

        Returns
        -------
        torch.Tensor
            Signed ``[batch, tokens]`` salience.
        """

        layer = self._config.salience_layer
        if layer is None:  # construction arms salience or leaves this None
            _refusals.refuse_salience_unavailable("salience was not armed at construction")
        embedding = self._net.get_submodule(layer)
        captured: list[torch.Tensor] = []

        def hook(module: nn.Module, args: Any, output: Any) -> torch.Tensor:
            """Re-root the embedding output as the differentiation leaf.

            Parameters
            ----------
            module:
                The embedding module (unused).
            args:
                The module call args (unused).
            output:
                The embedding output tensor.

            Returns
            -------
            torch.Tensor
                The detached, grad-requiring replacement tensor.
            """

            del module, args
            leaf = output.detach().requires_grad_(True)
            captured.append(leaf)
            return leaf

        handle = embedding.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                output = self._net(**input_kwargs)
                scalar = target(output)
                if not captured:
                    _refusals.refuse_pooling_unsupported(
                        _SALIENCE_FIELD,
                        f"the input-embedding module {layer!r} did not fire during "
                        "the salience forward",
                    )
                gradients = torch.autograd.grad(scalar, captured)
        finally:
            handle.remove()
        contributions = [
            (leaf * gradient).sum(dim=-1)
            for leaf, gradient in zip(captured, gradients, strict=True)
        ]
        return torch.stack(contributions, dim=0).sum(dim=0).detach()

    def _salience_rows(
        self,
        tokens_rows: list[list[str]],
        salience: torch.Tensor,
        lengths: list[int],
    ) -> list[Any]:
        """Package per-row salience as LIT ``TokenSalience`` values.

        Parameters
        ----------
        tokens_rows:
            Per-row wordpiece strings (unpadded).
        salience:
            Signed ``[batch, padded_tokens]`` salience (right-padded geometry).
        lengths:
            Per-row real-token counts.

        Returns
        -------
        list[Any]
            One ``lit_dtypes.TokenSalience`` per row.
        """

        rows: list[Any] = []
        for i, tokens in enumerate(tokens_rows):
            values = salience[i, : lengths[i]].detach().cpu().to(torch.float64).numpy()
            rows.append(lit_dtypes.TokenSalience(tokens=tokens, salience=values))
        return rows


class TorchLensLitClassifier(_TorchLensLitModel):
    """Sequence-classification adapter (memo build item 5).

    Right-padded batched predict throughout; probabilities from the SAME
    traced forward that serves the activation fields; mask-aware pooling.
    """

    def __init__(
        self,
        net: nn.Module,
        tokenizer: Any,
        config: AdapterConfig,
        labels: tuple[str, ...],
    ) -> None:
        """Initialize the classification adapter.

        Parameters
        ----------
        net:
            The live torch model.
        tokenizer:
            The checkpoint's tokenizer.
        config:
            Shared adapter configuration.
        labels:
            Validated class-label vocabulary (memo D19 ``labels=`` override).
        """

        super().__init__(net, tokenizer, config)
        self._labels = labels

    def output_spec(self) -> dict[str, Any]:
        """Return the LIT output spec, routed through the converter registry.

        Returns
        -------
        dict[str, Any]
            Probabilities, tokens, pooled activations, and opt-in salience.
        """

        registry = FIELD_CONVERTERS
        spec: dict[str, Any] = {
            "probas": registry["multiclass_preds"].build_spec(
                {"labels": self._labels, "label_field": _LABEL_FIELD}
            ),
            "tokens": registry["tokens"].build_spec({"text_field": _TEXT_FIELD}),
        }
        for site in self._config.specs:
            spec[site.field_name] = registry["embeddings"].build_spec({})
        if self._config.salience_layer is not None:
            spec[_SALIENCE_FIELD] = registry["token_salience"].build_spec({})
        return spec

    def predict_minibatch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Serve one LIT minibatch from one traced forward (memo D9).

        Parameters
        ----------
        inputs:
            LIT example rows carrying ``text``.

        Returns
        -------
        list[dict[str, Any]]
            One output row per example, following ``output_spec()``.
        """

        texts = [str(row.get(_TEXT_FIELD, "")) for row in inputs]
        id_lists = _runtime.encode_texts(self._tokenizer, texts)
        device = _runtime.model_device(self._net)
        input_ids, mask = _runtime.pad_batch(id_lists, self._pad_id, "right", device)
        forward_kwargs = {"input_ids": input_ids, "attention_mask": mask}

        with _runtime.operation_state(self._net):
            with torch.no_grad():
                log = _runtime.traced_forward(
                    self._net, forward_kwargs, self._config.save_selection
                )
            logits = _runtime.extract_logits("classification", log.reconstruct_output())
            probas = torch.softmax(logits.detach().to(torch.float64), dim=-1)
            pooled = self._embedding_columns(log, mask)
            salience = self._classifier_salience(forward_kwargs, logits)

        return self._package_rows(id_lists, mask, probas, pooled, salience)

    def _classifier_salience(
        self, forward_kwargs: dict[str, Any], logits: torch.Tensor
    ) -> torch.Tensor | None:
        """Compute salience toward each row's predicted class, if enabled.

        Parameters
        ----------
        forward_kwargs:
            The padded request batch.
        logits:
            ``[batch, num_labels]`` traced logits.

        Returns
        -------
        torch.Tensor | None
            Signed ``[batch, tokens]`` salience, or None when off.
        """

        if self._config.salience_layer is None:
            return None
        predicted = logits.argmax(dim=-1).detach()

        def target(output: Any) -> torch.Tensor:
            """Sum each row's predicted-class logit.

            Parameters
            ----------
            output:
                The attribution forward's model output.

            Returns
            -------
            torch.Tensor
                Scalar target.
            """

            out_logits = _runtime.extract_logits("classification", output)
            return out_logits.gather(1, predicted.unsqueeze(1)).sum()

        return self._salience_values(forward_kwargs, target)

    def _package_rows(
        self,
        id_lists: list[list[int]],
        mask: torch.Tensor,
        probas: torch.Tensor,
        pooled: dict[str, torch.Tensor],
        salience: torch.Tensor | None,
    ) -> list[dict[str, Any]]:
        """Assemble per-example LIT output rows.

        Parameters
        ----------
        id_lists:
            Unpadded per-row token ids.
        mask:
            The right-padded attention mask.
        probas:
            ``[batch, num_labels]`` probabilities.
        pooled:
            Pooled activation columns.
        salience:
            Optional signed salience.

        Returns
        -------
        list[dict[str, Any]]
            LIT output rows.
        """

        tokens_rows = self._tokens_rows(id_lists)
        lengths = _runtime.unpadded_lengths(mask)
        salience_rows = (
            self._salience_rows(tokens_rows, salience, lengths) if salience is not None else None
        )
        rows: list[dict[str, Any]] = []
        for i in range(len(id_lists)):
            row: dict[str, Any] = {
                "probas": probas[i].cpu().numpy(),
                "tokens": tokens_rows[i],
            }
            for field, column in pooled.items():
                row[field] = column[i].detach().cpu().to(torch.float64).numpy()
            if salience_rows is not None:
                row[_SALIENCE_FIELD] = salience_rows[i]
            rows.append(row)
        return rows


@dataclass(frozen=True)
class _GenerationProducts:
    """Correlated per-request causal-LM products the row assembler consumes.

    Parameters
    ----------
    generated:
        Per-row generated continuations.
    top_k_rows:
        Per-row, per-token descending next-token candidates.
    pooled:
        Pooled activation columns.
    salience:
        Optional signed next-token salience.
    """

    generated: list[str]
    top_k_rows: list[list[list[tuple[str, float]]]]
    pooled: dict[str, torch.Tensor]
    salience: torch.Tensor | None


class TorchLensLitCausalLM(_TorchLensLitModel):
    """Basic causal-LM adapter with deterministic generation (memo item 6).

    The two-tokenization coherent set (memo D9): LEFT-padded batch for ONE
    ``generate()`` call (byte-identical to unbatched HF generation), and
    RIGHT-padded batch for ONE traced forward with ``use_cache=False`` serving
    next-token top-k, activations, and salience. The halves never mix.
    """

    def __init__(
        self,
        net: nn.Module,
        tokenizer: Any,
        config: AdapterConfig,
        max_new_tokens: int,
        top_k: int,
    ) -> None:
        """Initialize the causal-LM adapter.

        Parameters
        ----------
        net:
            The live torch model (must expose HF-style ``generate``).
        tokenizer:
            The checkpoint's tokenizer.
        config:
            Shared adapter configuration.
        max_new_tokens:
            Fixed deterministic-generation length.
        top_k:
            Next-token candidates served per row.
        """

        super().__init__(net, tokenizer, config)
        self._max_new_tokens = max_new_tokens
        self._top_k = top_k

    def output_spec(self) -> dict[str, Any]:
        """Return the LIT output spec, routed through the converter registry.

        Returns
        -------
        dict[str, Any]
            Generation, tokens, top-k, pooled activations, opt-in salience.
        """

        registry = FIELD_CONVERTERS
        spec: dict[str, Any] = {
            "generated_text": registry["generated_text"].build_spec({"text_field": _TEXT_FIELD}),
            "tokens": registry["tokens"].build_spec({"text_field": _TEXT_FIELD}),
            "top_tokens": registry["token_top_k"].build_spec({"tokens_field": "tokens"}),
        }
        for site in self._config.specs:
            spec[site.field_name] = registry["embeddings"].build_spec({})
        if self._config.salience_layer is not None:
            spec[_SALIENCE_FIELD] = registry["token_salience"].build_spec({})
        return spec

    def predict_minibatch(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Serve one LIT minibatch: one generation, one traced forward.

        Parameters
        ----------
        inputs:
            LIT example rows carrying ``text``.

        Returns
        -------
        list[dict[str, Any]]
            One output row per example, following ``output_spec()``.
        """

        texts = [str(row.get(_TEXT_FIELD, "")) for row in inputs]
        id_lists = _runtime.encode_texts(self._tokenizer, texts)
        device = _runtime.model_device(self._net)

        with _runtime.operation_state(self._net):
            generated = self._generate(id_lists, device)
            trace_ids, trace_mask = _runtime.pad_batch(id_lists, self._pad_id, "right", device)
            forward_kwargs = {
                "input_ids": trace_ids,
                "attention_mask": trace_mask,
                "use_cache": False,
            }
            with torch.no_grad():
                log = _runtime.traced_forward(
                    self._net, forward_kwargs, self._config.save_selection
                )
            logits = _runtime.extract_logits("causal_lm", log.reconstruct_output())
            lengths = _runtime.unpadded_lengths(trace_mask)
            next_logits = self._next_token_logits(logits, lengths)
            top_k_rows = self._top_k_rows(logits, lengths)
            pooled = self._embedding_columns(log, trace_mask)
            salience = self._lm_salience(forward_kwargs, next_logits, lengths)

        return self._package_rows(
            id_lists,
            trace_mask,
            _GenerationProducts(
                generated=generated, top_k_rows=top_k_rows, pooled=pooled, salience=salience
            ),
        )

    def _generate(self, id_lists: list[list[int]], device: torch.device) -> list[str]:
        """Run ONE batched deterministic generation, left-padded (memo D9).

        Parameters
        ----------
        id_lists:
            Unpadded per-row token ids.
        device:
            Model device.

        Returns
        -------
        list[str]
            Per-row generated continuations (specials skipped).
        """

        gen_ids, gen_mask = _runtime.pad_batch(id_lists, self._pad_id, "left", device)
        generate = cast(Any, self._net).generate
        with torch.no_grad():
            output = generate(
                input_ids=gen_ids,
                attention_mask=gen_mask,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self._max_new_tokens,
                pad_token_id=self._pad_id,
            )
        continuations = output[:, gen_ids.shape[1] :]
        return [
            self._tokenizer.decode(row.tolist(), skip_special_tokens=True) for row in continuations
        ]

    def _next_token_logits(self, logits: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        """Read each row's next-token logits at ``[i, L_i - 1]``.

        Under right padding without ``position_ids`` the last REAL token's
        position carries the next-token distribution (memo D9 coherent set).

        Parameters
        ----------
        logits:
            ``[batch, seq, vocab]`` traced logits.
        lengths:
            Per-row real-token counts.

        Returns
        -------
        torch.Tensor
            ``[batch, vocab]`` next-token logits.
        """

        index = torch.tensor([n - 1 for n in lengths], dtype=torch.int64, device=logits.device)
        rows = torch.arange(logits.shape[0], device=logits.device)
        return logits[rows, index]

    def _lm_salience(
        self,
        forward_kwargs: dict[str, Any],
        next_logits: torch.Tensor,
        lengths: list[int],
    ) -> torch.Tensor | None:
        """Compute salience toward each row's argmax next token, if enabled.

        Parameters
        ----------
        forward_kwargs:
            The right-padded traced-request kwargs.
        next_logits:
            ``[batch, vocab]`` next-token logits from the traced forward.
        lengths:
            Per-row real-token counts.

        Returns
        -------
        torch.Tensor | None
            Signed ``[batch, tokens]`` salience, or None when off.
        """

        if self._config.salience_layer is None:
            return None
        targets = next_logits.argmax(dim=-1).detach()
        positions = torch.tensor([n - 1 for n in lengths], dtype=torch.int64, device=targets.device)

        def target(output: Any) -> torch.Tensor:
            """Sum each row's argmax next-token logit.

            Parameters
            ----------
            output:
                The attribution forward's model output.

            Returns
            -------
            torch.Tensor
                Scalar target.
            """

            out_logits = _runtime.extract_logits("causal_lm", output)
            rows = torch.arange(out_logits.shape[0], device=out_logits.device)
            return out_logits[rows, positions, targets].sum()

        return self._salience_values(forward_kwargs, target)

    def _top_k_rows(
        self, logits: torch.Tensor, lengths: list[int]
    ) -> list[list[list[tuple[str, float]]]]:
        """Build per-token descending ``(token, probability)`` TUPLES.

        LIT's ``TokenTopKPreds`` is token-aligned: one descending candidate
        list per token in the aligned ``tokens`` field (position ``t`` carries
        the model's distribution for the token AFTER ``t``; the last real
        position is the next-token prediction). The validator rejects list
        candidates (memo D19): each candidate must be a tuple with a string
        first element, in descending order.

        Parameters
        ----------
        logits:
            ``[batch, seq, vocab]`` traced logits (right-padded geometry).
        lengths:
            Per-row real-token counts.

        Returns
        -------
        list[list[list[tuple[str, float]]]]
            Per-row, per-token descending candidates.
        """

        probs = torch.softmax(logits.detach().to(torch.float64), dim=-1)
        top = probs.topk(self._top_k, dim=-1)
        convert = getattr(self._tokenizer, "convert_ids_to_tokens", None)
        rows: list[list[list[tuple[str, float]]]] = []
        for i, length in enumerate(lengths):
            positions: list[list[tuple[str, float]]] = []
            for t in range(length):
                id_list = [int(x) for x in top.indices[i, t].tolist()]
                if convert is not None:
                    names = [str(token) for token in convert(id_list)]
                else:
                    names = [self._tokenizer.decode([x]) for x in id_list]
                values = [float(v) for v in top.values[i, t].tolist()]
                positions.append(list(zip(names, values, strict=True)))
            rows.append(positions)
        return rows

    def _package_rows(
        self,
        id_lists: list[list[int]],
        trace_mask: torch.Tensor,
        products: _GenerationProducts,
    ) -> list[dict[str, Any]]:
        """Assemble per-example LIT output rows.

        Parameters
        ----------
        id_lists:
            Unpadded per-row token ids.
        trace_mask:
            The right-padded attention mask.
        products:
            The bundled per-request generation products.

        Returns
        -------
        list[dict[str, Any]]
            LIT output rows.
        """

        tokens_rows = self._tokens_rows(id_lists)
        lengths = _runtime.unpadded_lengths(trace_mask)
        salience_rows = (
            self._salience_rows(tokens_rows, products.salience, lengths)
            if products.salience is not None
            else None
        )
        rows: list[dict[str, Any]] = []
        for i in range(len(id_lists)):
            row: dict[str, Any] = {
                "generated_text": products.generated[i],
                "tokens": tokens_rows[i],
                "top_tokens": products.top_k_rows[i],
            }
            for field, column in products.pooled.items():
                row[field] = column[i].detach().cpu().to(torch.float64).numpy()
            if salience_rows is not None:
                row[_SALIENCE_FIELD] = salience_rows[i]
            rows.append(row)
        return rows


def _spec_dict(examples: Iterable[Any], labels: tuple[str, ...] | None) -> dict[str, Any]:
    """Build the dataset spec for the tiny native helper.

    Parameters
    ----------
    examples:
        Normalized example rows (unused; present for future field inference).
    labels:
        Classifier vocabulary, or None for LM datasets.

    Returns
    -------
    dict[str, Any]
        The LIT dataset spec.
    """

    del examples
    spec: dict[str, Any] = {_TEXT_FIELD: lit_types.TextSegment()}
    if labels is not None:
        spec[_LABEL_FIELD] = lit_types.CategoryLabel(vocab=list(labels))
    return spec


def build_dataset(examples: Iterable[Any], labels: tuple[str, ...] | None, description: str) -> Any:
    """Build the tiny native LIT dataset (memo D19).

    Parameters
    ----------
    examples:
        Strings, or dicts carrying ``text`` (and optionally ``label``).
    labels:
        Classifier vocabulary (rows default to its first entry), or None.
    description:
        Human-readable dataset description.

    Returns
    -------
    Any
        A ``lit_nlp.api.dataset.Dataset``.
    """

    from lit_nlp.api import dataset as lit_dataset

    class _TorchLensLitDataset(lit_dataset.Dataset):
        """In-memory dataset built by ``tl.bridge.lit.dataset()``."""

        @classmethod
        def init_spec(cls) -> dict[str, Any] | None:
            """Return None: built from live Python rows, not a JSON spec.

            Returns
            -------
            dict[str, Any] | None
                Always None (silences LIT's constructor-inference warning).
            """

            return None

    rows: list[dict[str, Any]] = []
    for example in examples:
        row = dict(example) if isinstance(example, dict) else {_TEXT_FIELD: str(example)}
        if labels is not None:
            row.setdefault(_LABEL_FIELD, labels[0])
        rows.append(row)
    return _TorchLensLitDataset(
        spec=_spec_dict(rows, labels), examples=rows, description=description
    )


def build_layout() -> Any:
    """Build the pure-Python LIT layout foregrounding TorchLens fields.

    Custom ``LitCanonicalLayout``s are the whole frontend leverage (memo D16):
    LIT panels are compiled TypeScript, but layouts are plain Python and need
    no client rebuild.

    Returns
    -------
    Any
        A ``lit_nlp.api.layout.LitCanonicalLayout``.
    """

    from lit_nlp.api import layout as lit_layout

    modules = lit_layout.modules
    return lit_layout.LitCanonicalLayout(
        upper={
            "Main": [
                modules.DocumentationModule,
                modules.EmbeddingsModule,
                *lit_layout.DEFAULT_MAIN_GROUP,
            ]
        },
        lower={
            "Predictions": [*lit_layout.MODEL_PREDS_MODULES, modules.ScalarModule],
            "Salience": [modules.SalienceMapModule],
            "Metrics": [modules.MetricsModule, modules.ConfusionMatrixModule],
            "Counterfactuals": [modules.GeneratorModule],
        },
        description=(
            "TorchLens layout: embedding projector and model-provided salience "
            "foregrounded next to the standard editor and metrics."
        ),
    )
