"""LIT bridge contract tests against REAL lit-nlp (lane F31; PR-leg models).

These are the tests the deleted stub never had: every assertion runs against
the real ``lit_nlp`` 1.3.1 API -- the validator, LitApp construction, and the
served field contracts -- on the PR-leg tiny checkpoints
(``hf-internal-testing/tiny-random-distilbert``, ``sshleifer/tiny-gpt2``).
Skips honestly when lit-nlp, transformers, or the cached checkpoints are
absent; the nightly CI leg installs the real stack (M(lit) build item 8).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens._errors import InvalidArgumentError

lit_model_api = pytest.importorskip("lit_nlp.api.model")
transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.heavy

_TINY_CLS = "hf-internal-testing/tiny-random-distilbert"
_TINY_LM = "sshleifer/tiny-gpt2"


def _load(name: str, cls: Any) -> tuple[Any, Any]:
    """Load a cached checkpoint pair, skipping when unavailable offline.

    Parameters
    ----------
    name:
        Checkpoint name.
    cls:
        Transformers auto-model class.

    Returns
    -------
    tuple[Any, Any]
        ``(model.eval(), tokenizer)``.
    """

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        net = cls.from_pretrained(name).eval()
    except OSError as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"checkpoint {name} not cached: {exc}")
    return net, tokenizer


@pytest.fixture(scope="module")
def stack() -> dict[str, Any]:
    """Load BOTH model variants before the first trace in this process.

    Returns
    -------
    dict[str, Any]
        Both nets and tokenizers plus the constructed wrappers.
    """

    cls_net, cls_tok = _load(_TINY_CLS, transformers.AutoModelForSequenceClassification)
    lm_net, lm_tok = _load(_TINY_LM, transformers.AutoModelForCausalLM)
    classifier = tl.bridge.lit.model(
        cls_net,
        cls_tok,
        task="classification",
        sites="blocks",
        labels=("neg", "pos"),
        salience=True,
    )
    lm = tl.bridge.lit.model(
        lm_net,
        lm_tok,
        task="causal_lm",
        sites="blocks",
        salience=True,
        max_new_tokens=5,
        top_k=5,
    )
    return {
        "cls_net": cls_net,
        "cls_tok": cls_tok,
        "lm_net": lm_net,
        "lm_tok": lm_tok,
        "classifier": classifier,
        "lm": lm,
    }


def test_wrapper_is_real_lit_model(stack: dict[str, Any]) -> None:
    """Both adapters subclass the REAL ``lit_nlp.api.model.Model`` (memo D1)."""

    assert isinstance(stack["classifier"], lit_model_api.Model)
    assert isinstance(stack["lm"], lit_model_api.Model)
    assert stack["classifier"].supports_concurrent_predictions is False
    assert stack["classifier"].init_spec() is None


def test_validator_passes_on_non_empty_dataset(stack: dict[str, Any]) -> None:
    """``validate_model(report_all=True)`` passes with REAL examples (VQ8)."""

    from lit_nlp.lib import validation

    ds = tl.bridge.lit.dataset(["good movie", "terrible movie"], labels=("neg", "pos"))
    validation.validate_model(stack["classifier"], ds, report_all=True)
    lm_ds = tl.bridge.lit.dataset(["The quick brown fox", "Hello there"])
    validation.validate_model(stack["lm"], lm_ds, report_all=True)


def test_validator_gate_is_not_vacuous(stack: dict[str, Any]) -> None:
    """The validator REJECTS a stub-shaped model on the same dataset (VQ11).

    An empty example list vacuously accepts even the deleted stub, so this
    gate proves its own teeth: the known-bad shape (plain-dict specs) must
    fail against real examples for the green above to mean anything.
    """

    from lit_nlp.lib import validation

    class _StubShaped(lit_model_api.Model):
        """The deleted stub's shape: dict specs instead of LitTypes."""

        def input_spec(self) -> dict[str, Any]:
            """Return a plain-dict (invalid) input spec.

            Returns
            -------
            dict[str, Any]
                Not LitTypes.
            """

            return {"text": {"required": False}}

        def output_spec(self) -> dict[str, Any]:
            """Return a plain-dict (invalid) output spec.

            Returns
            -------
            dict[str, Any]
                Not LitTypes.
            """

            return {"layer_labels": {"required": False}}

        def predict(self, inputs: Any, **kw: Any) -> Any:
            """Echo metadata rows like the stub did.

            Parameters
            ----------
            inputs:
                LIT rows.
            **kw:
                Unused.

            Returns
            -------
            Any
                Constant rows.
            """

            del kw
            return [{"layer_labels": ["x"]} for _ in inputs]

    ds = tl.bridge.lit.dataset(["good movie"], labels=("neg", "pos"))
    # lit-nlp 1.3.x rejects dict specs with AttributeError (no LitType methods);
    # ValueError is the validator's own refusal channel for spec mismatches.
    with pytest.raises((AttributeError, ValueError)):
        validation.validate_model(_StubShaped(), ds, report_all=True)


def test_litapp_accepts_wrapper(stack: dict[str, Any]) -> None:
    """LitApp constructs with the wrapper (the VQ1 inversion)."""

    import lit_nlp
    from lit_nlp import app as lit_app

    client_root = os.path.join(os.path.dirname(lit_nlp.__file__), "client", "build", "default")
    ds = tl.bridge.lit.dataset(["good", "bad"], labels=("neg", "pos"))
    app = lit_app.LitApp(
        models={"tl": stack["classifier"]},
        datasets={"d": ds},
        client_root=client_root,
        layouts={"torchlens": tl.bridge.lit.layout()},
    )
    assert app is not None


def test_classifier_probabilities_match_direct(stack: dict[str, Any]) -> None:
    """Wrapper probabilities match a direct HF forward (thread-tolerant).

    Tolerance per memo D20: traced-vs-direct numeric gates use ~1e-3 headroom
    because thread-count-dependent GEMM reduction order alone produces
    ~5e-4 gaps; this is NOT a pooling gate (those assert exactly).
    """

    texts = ["a fine day", "a much longer and stranger example sentence"]
    rows = list(stack["classifier"].predict([{"text": t} for t in texts]))
    enc = stack["cls_tok"](texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        direct = torch.softmax(stack["cls_net"](**enc).logits, dim=-1)
    for i, row in enumerate(rows):
        got = torch.tensor(row["probas"], dtype=torch.float64)
        assert torch.allclose(got, direct[i].to(torch.float64), atol=1e-3)
        assert row["tokens"] == stack["cls_tok"].convert_ids_to_tokens(
            stack["cls_tok"](texts[i])["input_ids"]
        )


def test_ragged_batch_serves_all_fields(stack: dict[str, Any]) -> None:
    """A ragged 3-row request re-resolves every pinned site (memo D4/D9)."""

    texts = ["one", "a somewhat longer middle example", "two words"]
    rows = list(stack["classifier"].predict([{"text": t} for t in texts]))
    spec = stack["classifier"].output_spec()
    emb_fields = [k for k in spec if k.startswith("tl_block_")]
    assert emb_fields
    for row, text in zip(rows, texts, strict=True):
        n_tokens = len(stack["cls_tok"](text)["input_ids"])
        for field in emb_fields:
            assert row[field].ndim == 1
        assert row["tl_salience"].salience.shape == (n_tokens,)
        assert len(row["tl_salience"].tokens) == n_tokens


def test_predict_is_deterministic(stack: dict[str, Any]) -> None:
    """Two identical requests return byte-identical rows (cache-safe)."""

    request = [{"text": "determinism check"}]
    first = list(stack["classifier"].predict(request))[0]
    second = list(stack["classifier"].predict(request))[0]
    assert (first["probas"] == second["probas"]).all()
    for key in first:
        if key.startswith("tl_block_"):
            assert (first[key] == second[key]).all()


def test_lm_generation_parity_ragged(stack: dict[str, Any]) -> None:
    """Batched left-padded generation is byte-identical to unbatched (D9)."""

    prompts = ["The quick brown fox", "Hi", "A longer prompt that keeps going on"]
    rows = list(stack["lm"].predict([{"text": p} for p in prompts]))
    for prompt, row in zip(prompts, rows, strict=True):
        ids = stack["lm_tok"](prompt, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            out = stack["lm_net"].generate(
                input_ids=ids,
                do_sample=False,
                num_beams=1,
                max_new_tokens=5,
                pad_token_id=stack["lm_tok"].eos_token_id,
            )
        reference = stack["lm_tok"].decode(
            out[0, ids.shape[1] :].tolist(), skip_special_tokens=True
        )
        assert row["generated_text"] == reference


def test_lm_top_k_is_descending_tuples(stack: dict[str, Any]) -> None:
    """Per-token top-k candidates are descending (token, prob) TUPLES (D19)."""

    rows = list(stack["lm"].predict([{"text": "The quick brown fox"}]))
    top = rows[0]["top_tokens"]
    assert len(top) == len(rows[0]["tokens"])
    for candidates in top:
        assert all(isinstance(c, tuple) and isinstance(c[0], str) for c in candidates)
        probs = [c[1] for c in candidates]
        assert probs == sorted(probs, reverse=True)


def test_state_restoration_after_predict(stack: dict[str, Any]) -> None:
    """Training flags and parameters are bytewise unchanged (memo D18)."""

    net = stack["cls_net"]
    net.train()
    try:
        before_modes = [m.training for m in net.modules()]
        before = [p.detach().clone() for p in net.parameters()]
        list(stack["classifier"].predict([{"text": "state guard check"}]))
        assert [m.training for m in net.modules()] == before_modes
        for prior, param in zip(before, net.parameters(), strict=True):
            assert torch.equal(prior, param)
        assert all(p.grad is None for p in net.parameters())
    finally:
        net.eval()


def test_sites_unspecified_refusal_names_candidates(stack: dict[str, Any]) -> None:
    """No ``sites=`` refuses and names the discovered block stack (D7)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.bridge.lit.model(stack["cls_net"], stack["cls_tok"], labels=("a", "b"))
    assert excinfo.value.fields["code"] == "lit_sites_unspecified"
    assert any("transformer.layer" in c for c in excinfo.value.fields["candidates"])


def test_explicit_empty_sites_means_no_embedding_fields(stack: dict[str, Any]) -> None:
    """``sites=()`` is the explicit no-embeddings spelling."""

    wrapper = tl.bridge.lit.model(stack["cls_net"], stack["cls_tok"], sites=(), labels=("a", "b"))
    assert not [k for k in wrapper.output_spec() if k.startswith("tl_block_")]


def test_labels_override_count_mismatch_refuses(stack: dict[str, Any]) -> None:
    """A wrong-arity ``labels=`` refuses against config.id2label (D19)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.bridge.lit.model(stack["cls_net"], stack["cls_tok"], sites=(), labels=("only-one",))
    assert excinfo.value.fields["code"] == "lit_labels_invalid"


def test_task_vocabulary_closed(stack: dict[str, Any]) -> None:
    """Unknown tasks refuse with the closed vocabulary."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        tl.bridge.lit.model(stack["cls_net"], stack["cls_tok"], task="regression")
    assert excinfo.value.fields["code"] == "lit_task_invalid"


def test_no_attention_field_anywhere(stack: dict[str, Any]) -> None:
    """No AttentionHeads in any served spec, and no registry entry (D12).

    Evidence for the exclusion (round-2 card / memo D12): LIT deleted its
    attention module upstream on 2024-06-20 -- the type still validates and
    serializes, INTO NOTHING (no panel renders it). Absence here is the
    feature; re-adding an attention entry requires a maintained visible
    renderer upstream plus real alignment/browser-render tests, never just
    successful serialization.
    """

    from lit_nlp.api import types as lit_types

    from torchlens.bridge.lit._adapters import FIELD_CONVERTERS

    assert "attention" not in FIELD_CONVERTERS
    for wrapper_key in ("classifier", "lm"):
        for value in stack[wrapper_key].output_spec().values():
            assert not isinstance(value, lit_types.AttentionHeads)


def test_interpreter_compatibility_required_present_broken_absent(
    stack: dict[str, Any],
) -> None:
    """Required interpreters apply; broken triggers stay hidden (memo D20).

    Required-present / broken-absent, never the exact observed set -- freezing
    LIT's internal interpreter list into our contract would pin internals we
    do not own.
    """

    import lit_nlp
    from lit_nlp import app as lit_app

    client_root = os.path.join(os.path.dirname(lit_nlp.__file__), "client", "build", "default")
    ds = tl.bridge.lit.dataset(["good", "bad"], labels=("neg", "pos"))
    app = lit_app.LitApp(
        models={"tl": stack["classifier"]}, datasets={"d": ds}, client_root=client_root
    )
    info = app._get_info(None)  # noqa: SLF001 -- the served contract under test
    compat = info["models"]["tl"]["interpreters"]
    for required in ("Model-provided salience", "pca", "nearest neighbors", "LIME"):
        assert required in compat, (required, compat)
    for broken in ("Integrated Gradients", "TCAV", "attention"):
        assert broken not in compat, (broken, compat)
