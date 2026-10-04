"""Real-model transform rows (memo 10.2; lane F19). Offline, cache-gated.

The realism rule discharged on the transforms library itself: real
checkpoints, real tokenizers, real ragged shapes — every row skips typed
where its cached artifact is absent and NEVER downloads. Rows owned
elsewhere are not faked here: the Qwen kill/resume harvest and the
cross-process opaque-resume refusal are extraction-engine territory (F18)
— this suite pins the library-side identity and disclosure contracts the
engine keys on; the transformers-4.x leg of every LM row runs in the F36
version-leg venue (one environment cannot host two transformers).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

import torchlens as tl
from tests.transforms_corpus.loader import (
    build_fixture,
    load_vendored,
    resnet50_checkpoint_cached,
)
from torchlens.stats import PCA, load_fitted, save_fitted
from torchlens.transforms import (
    BTD,
    SpecialTokenFacts,
    TransformContext,
    TransformContractError,
    cast,
    chain,
    cls_token,
    flatten,
    pca_apply,
    pipeline_from_record,
    pool_tokens,
    srp,
    unit_norm,
)

pytestmark = [pytest.mark.slow, pytest.mark.real_model]


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real-checkpoint rows run offline: the cache serves the weights or the test
    skips; only missing tokenizer files are fetched (see ``_hf_model``)."""

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


#: Tokenizer files another test may have left out when it cached only weights.
_TOKENIZER_FILE_PATTERNS = [
    "tokenizer*",
    "vocab*",
    "merges*",
    "special_tokens_map.json",
    "added_tokens.json",
    "*.model",
]


@contextlib.contextmanager
def _hub_online() -> Iterator[None]:
    """Lift the module's offline mode for one deliberate tokenizer fetch."""

    import huggingface_hub.constants as hub_constants

    saved_env = {
        key: os.environ.pop(key, None) for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    saved_flag = hub_constants.HF_HUB_OFFLINE
    hub_constants.HF_HUB_OFFLINE = False
    try:
        yield
    finally:
        hub_constants.HF_HUB_OFFLINE = saved_flag
        for key, value in saved_env.items():
            if value is not None:
                os.environ[key] = value


def _encodes(tokenizer) -> bool:
    """Return whether ``tokenizer`` turns text into a non-empty id sequence."""

    return bool(tokenizer("hi")["input_ids"])


def _cached_or_fetched_tokenizer(transformers, name: str):
    """Load ``name``'s tokenizer, fetching only its tokenizer files when missing.

    Another test can cache a checkpoint's weights without its tokenizer
    (``from_pretrained`` of the model alone). Offline, transformers then builds
    a tokenizer that encodes every string to zero ids instead of raising, so
    the missing files are fetched once and the tokenizer reloaded. Only a
    machine that is offline AND lacks the tokenizer files skips.
    """

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(name)
        if _encodes(tokenizer):
            return tokenizer
    except (OSError, ValueError):
        pass
    import huggingface_hub

    offline_reason = (
        f"{name} tokenizer files are not in the HF cache and this machine is offline "
        "(the hub is unreachable); the cached weights alone cannot tokenize"
    )
    try:
        with _hub_online():
            snapshot = huggingface_hub.snapshot_download(
                name, allow_patterns=_TOKENIZER_FILE_PATTERNS
            )
    except Exception as exc:  # noqa: BLE001 - any fetch failure means no network here
        pytest.skip(f"{offline_reason}: {type(exc).__name__}")
    # An unreachable hub does not always raise: snapshot_download can fall back to
    # the partial local snapshot, so check that tokenizer files actually arrived.
    snapshot_dir = Path(snapshot)
    if not any(any(snapshot_dir.glob(pattern)) for pattern in _TOKENIZER_FILE_PATTERNS):
        pytest.skip(offline_reason)
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    assert _encodes(tokenizer), f"{name} tokenizer still encodes to zero ids after the fetch"
    return tokenizer


def _hf_model(name: str):
    """Load a cached HF model + tokenizer or skip typed (weights never download)."""

    transformers = pytest.importorskip("transformers")
    try:
        model = transformers.AutoModel.from_pretrained(name)
    except (OSError, ValueError) as exc:  # cache miss, offline -> honest skip
        pytest.skip(f"{name} not in the offline HF cache: {type(exc).__name__}")
    tokenizer = _cached_or_fetched_tokenizer(transformers, name)
    model.eval()
    return model, tokenizer


SENTENCES = [
    "hi",
    "the cat sat on the mat",
    "a very long sentence about representational geometry and pooling",
    "torchlens traces forward passes",
    "short",
    "mask aware token statistics are the honest default on padded batches",
    "seven",
    "pooling before projecting preserves more correlation structure",
]


# --- 10.2 row 1: resnet50 heterogeneous per-site chains through the engine ----


def test_resnet50_heterogeneous_extraction_end_to_end(tmp_path) -> None:
    """D-3 un-broken: one run, mixed-rank sites, per-site chains, one record.

    Real torchvision resnet50 (IMAGENET1K_V2), disk-mode extraction with a
    per-site Mapping; the manifest carries one tl_transform_pipeline_v1
    record per site; rehydration reproduces the canonical chains; and the
    engine-produced values equal the library chains applied to the raw
    activations (T-C10 makes batching irrelevant).
    """

    pytest.importorskip("torchvision")
    if not resnet50_checkpoint_cached():
        pytest.skip("resnet50 IMAGENET1K_V2 checkpoint not cached")
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    stimuli = torch.nn.functional.interpolate(
        build_fixture(load_vendored(), n_rows=8), size=(224, 224), mode="bilinear"
    )
    # NOTE: the classic fourth site "fc" hits the documented
    # trace(resnet50)["fc"] AmbiguousOpLookupError (extract memo's census
    # note, extract-owned seam); layer2 keeps the sweep heterogeneous.
    chains = {
        "layer2": chain(unit_norm(axis=1), cast(torch.bfloat16)),
        "layer3": chain(flatten(), srp(256, seed=1)),
        "layer4": chain(srp(512, seed=2)),
        "avgpool": chain(flatten(), unit_norm()),
    }
    out_dir = tmp_path / "artifact"
    tl.extract_dataset(
        model,
        stimuli,
        layers=["layer2", "layer3", "layer4", "avgpool"],
        batch_size=4,
        output_dir=out_dir,
        transform=dict(chains),
        progress=False,
    )
    from torchlens.dataset_extraction import load_extraction

    loaded = load_extraction(out_dir)
    tensors = loaded.activations if hasattr(loaded, "activations") else loaded
    assert set(chains) <= set(tensors)
    assert tensors["layer3"].shape == (8, 256)
    assert tensors["layer4"].shape == (8, 512)
    assert tensors["layer2"].dtype == torch.bfloat16
    manifest = loaded.manifest if hasattr(loaded, "manifest") else None
    assert manifest is not None
    per_site = manifest["signature"]["transform_pipeline"]["per_site"]
    for site, pipeline in chains.items():
        record = per_site[site]
        assert record["schema"] == "tl_transform_pipeline_v1"
        assert record["resume_verifiable"] is True
        assert pipeline_from_record(record).canonical_chain() == pipeline.canonical_chain()
    # Numeric identity: engine-applied chains == library chains on raw sites.
    raw = tl.extract_dataset(model, stimuli, layers=["layer4"], batch_size=8, progress=False)
    manual = pipeline_from_record(per_site["layer4"]).apply(
        raw["layer4"], TransformContext(site_label="layer4")
    )
    assert torch.allclose(manual, tensors["layer4"], atol=1e-5)


# --- 10.2 row 3: ragged real-tokenizer laundering + partition invariance -------


def test_bert_ragged_flatten_srp_refuses_and_pooled_partitions_agree() -> None:
    """The D-5 launch gate on real ragged shapes: drift REFUSES before any
    shard; pooled (mask-aware) rows are partition-invariant (T-C10)."""

    model, tokenizer = _hf_model("distilgpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Per-batch tokenization (the ragged reality): widths differ by batch.
    first = tokenizer(SENTENCES[:2], return_tensors="pt", padding=True)
    second = tokenizer(SENTENCES[2:5], return_tensors="pt", padding=True)
    with torch.no_grad():
        hidden_a = model(**first).last_hidden_state
        hidden_b = model(**second).last_hidden_state
    assert hidden_a.shape[1] != hidden_b.shape[1]  # genuinely ragged
    launderer = srp(64, seed=0)  # flatten mode binds the first extent
    launderer.apply(hidden_a, TransformContext(site_label="h"))
    with pytest.raises(TransformContractError) as excinfo:
        launderer.apply(hidden_b, TransformContext(site_label="h"))
    assert excinfo.value.fields["code"] == "transform_extent_drift"
    # Fixed global padding + mask-aware pooling: batch splits are irrelevant.
    batch = tokenizer(SENTENCES, return_tensors="pt", padding="max_length", max_length=24)
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"]
    pipeline = chain(pool_tokens("mean"), srp(64, seed=3, mode="features"))
    whole = pipeline.apply(hidden, TransformContext(roles=BTD, mask=mask))
    parts = torch.cat(
        [
            pipeline.apply(hidden[:3], TransformContext(roles=BTD, mask=mask[:3])),
            pipeline.apply(hidden[3:], TransformContext(roles=BTD, mask=mask[3:])),
        ]
    )
    assert torch.allclose(whole, parts, atol=1e-6)


# --- 10.2 row 4: LM pooling references, left AND right padding ------------------


def test_gpt2_family_pooling_matches_unpadded_references() -> None:
    """Masked mean/first/last equal per-sentence unpadded references under
    BOTH padding sides (left padding passes explicit position_ids so valid
    hidden states are comparable — the A11 engine fix's contract)."""

    model, tokenizer = _hf_model("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    references = {}
    with torch.no_grad():
        for index, sentence in enumerate(SENTENCES[:6]):
            single = tokenizer(sentence, return_tensors="pt")
            states = model(**single).last_hidden_state[0]
            references[index] = {
                "mean": states.mean(dim=0),
                "first": states[0],
                "last": states[-1],
            }
    for side in ("right", "left"):
        tokenizer.padding_side = side
        batch = tokenizer(SENTENCES[:6], return_tensors="pt", padding=True)
        mask = batch["attention_mask"]
        position_ids = (mask.cumsum(dim=1) - 1).clamp_min(0)
        with torch.no_grad():
            hidden = model(**batch, position_ids=position_ids).last_hidden_state
        ctx = TransformContext(roles=BTD, mask=mask)
        pooled_mean = pool_tokens("mean").apply(hidden, ctx)
        pooled_first = pool_tokens("first").apply(hidden, ctx)
        pooled_last = pool_tokens("last").apply(hidden, ctx)
        for index in references:
            assert torch.allclose(pooled_mean[index], references[index]["mean"], atol=2e-4), (
                side,
                index,
            )
            assert torch.allclose(pooled_first[index], references[index]["first"], atol=2e-4), (
                side,
                index,
            )
            assert torch.allclose(pooled_last[index], references[index]["last"], atol=2e-4), (
                side,
                index,
            )
        # The naive mask-blind mean is a DIFFERENT number on padded batches.
        naive = hidden.mean(dim=1)
        assert not torch.allclose(naive[0], references[0]["mean"], atol=1e-3)


# --- 10.2 rows 2 + 4: ViT axis contract; distilbert CLS vs gpt2 CLS -------------


def test_vit_axis_contract_resolves_or_refuses() -> None:
    """HF ViT: no roles -> refusal; declared roles -> pooling + exact CLS."""

    transformers = pytest.importorskip("transformers")
    try:
        vit = transformers.AutoModel.from_pretrained("google/vit-base-patch16-224")
    except (OSError, ValueError) as exc:
        pytest.skip(f"vit-base not in the offline HF cache: {type(exc).__name__}")
    vit.eval()
    pixels = torch.nn.functional.interpolate(load_vendored()[:4], size=(224, 224), mode="bilinear")
    pixels = (pixels - 0.5) / 0.5  # the ViT processor's normalization
    with torch.no_grad():
        hidden = vit(pixel_values=pixels).last_hidden_state  # (4, 197, 768)
    with pytest.raises(TransformContractError) as excinfo:
        pool_tokens("mean", assume_no_padding=True).apply(hidden, None)
    assert excinfo.value.fields["code"] == "transform_axis_roles_unavailable"
    ctx = TransformContext(
        roles=BTD,
        special_tokens=SpecialTokenFacts(has_cls=True, source="vit: prepended class token"),
    )
    pooled = pool_tokens("mean", assume_no_padding=True).apply(hidden, ctx)
    assert pooled.shape == (4, 768)
    cls_out = cls_token(assume_no_padding=True).apply(hidden, ctx)
    assert torch.equal(cls_out, hidden[:, 0])


def test_distilbert_cls_succeeds_and_gpt2_cls_refuses() -> None:
    """The memo's discriminative CLS pair, from REAL tokenizer facts."""

    model, tokenizer = _hf_model("distilbert-base-uncased")
    batch = tokenizer(SENTENCES[:4], return_tensors="pt", padding=True)
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    facts = SpecialTokenFacts(
        has_cls=tokenizer.cls_token_id is not None, source="tokenizer:distilbert-base-uncased"
    )
    assert facts.has_cls is True
    ctx = TransformContext(roles=BTD, mask=batch["attention_mask"], special_tokens=facts)
    cls_out = cls_token().apply(hidden, ctx)
    assert torch.equal(cls_out, hidden[:, 0])  # right padding: first valid == 0
    gpt2_tokenizer = pytest.importorskip("transformers").AutoTokenizer.from_pretrained("distilgpt2")
    gpt2_facts = SpecialTokenFacts(
        has_cls=gpt2_tokenizer.cls_token_id is not None, source="tokenizer:distilgpt2"
    )
    assert gpt2_facts.has_cls is False
    with pytest.raises(TransformContractError) as excinfo:
        cls_token().apply(hidden, TransformContext(roles=BTD, special_tokens=gpt2_facts))
    assert excinfo.value.fields["code"] == "transform_special_tokens_unavailable"


# --- 10.2 row 5: PCA composition on real activations ------------------------------


def test_pca_composition_fit_persist_reload_apply(tmp_path) -> None:
    """Fit on real pooled activations, persist, reload, apply == fp64 ref;
    the same protocol composes after SRP reduction."""

    pytest.importorskip("torchvision")
    if not resnet50_checkpoint_cached():
        pytest.skip("resnet50 IMAGENET1K_V2 checkpoint not cached")
    from tests.transforms_corpus.loader import resnet50_features

    features = resnet50_features(build_fixture(load_vendored(), n_rows=24))["avgpool"]
    estimator = PCA(8)
    estimator.update(features)
    fitted = estimator.fitted(fit_scope="24 augmented fixture rows, avgpool")
    path = save_fitted(fitted, tmp_path / "resnet_pca.safetensors")
    reloaded = load_fitted(path)
    out = pca_apply(reloaded).apply(features[:6], None)
    reference = (features[:6].double() - fitted.mean) @ fitted.components.T
    assert torch.allclose(out.double(), reference, atol=1e-5)
    # After SRP: the fitted protocol composes with reduced-width features.
    reducer = srp(128, seed=5)
    reduced = reducer.apply(features, TransformContext(site_label="avgpool"))
    estimator2 = PCA(4)
    estimator2.update(reduced)
    fitted2 = estimator2.fitted(fit_scope="post-srp reduced features")
    pipeline = chain(reducer, pca_apply(fitted2))
    composed = pipeline.apply(features, TransformContext(site_label="avgpool"))
    assert composed.shape == (24, 4)


# --- opaque steps: the disclosure the engine's strict resume rule keys on ---------


def test_opaque_chain_discloses_identification_only(tmp_path) -> None:
    """An opaque step flips resume_verifiable to False in the REAL manifest."""

    pytest.importorskip("torchvision")
    if not resnet50_checkpoint_cached():
        pytest.skip("resnet50 IMAGENET1K_V2 checkpoint not cached")
    from torchvision.models import ResNet50_Weights, resnet50

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    stimuli = torch.nn.functional.interpolate(
        build_fixture(load_vendored(), n_rows=4), size=(224, 224), mode="bilinear"
    )
    out_dir = tmp_path / "opaque"
    tl.extract_dataset(
        model,
        stimuli,
        layers=["avgpool"],
        batch_size=4,
        output_dir=out_dir,
        transform=lambda tensor: tensor.flatten(1),
        progress=False,
    )
    from torchlens.dataset_extraction import load_extraction

    loaded = load_extraction(out_dir)
    manifest = loaded.manifest if hasattr(loaded, "manifest") else None
    assert manifest is not None
    record = manifest["signature"]["transform_pipeline"]
    if "per_site" in record:
        record = record["per_site"]["avgpool"]
    assert record["resume_verifiable"] is False
    assert record["steps"][0]["kind"] == "opaque"
    assert record["steps"][0]["identity"] == "identification_only"
