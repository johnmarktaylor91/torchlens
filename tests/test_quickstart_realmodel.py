"""Quickstart real-model rows (F17 B17; memo section 6).

Real checkpoints, never toy stand-ins: resnet18 (IMAGENET1K_V1), gpt2 (real
tokenizer), CLIP (the multi-input boundary rider). Missing libraries skip;
missing checkpoints on a provisioned box are setup failures, never skips.
All cost assertions are counters, never wall-clock (memo D11).
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.quickstart import (
    SynthesizedValueReadWarning,
    state_dict_hash,
    trace_input_provenance,
)
from torchlens.user_funcs import render

pytestmark = pytest.mark.heavy


@pytest.fixture(scope="module")
def resnet18_eval() -> nn.Module:
    """Real pretrained resnet18, eval mode (the README's own model)."""

    torchvision = pytest.importorskip("torchvision")
    return torchvision.models.resnet18(weights="IMAGENET1K_V1").eval()


class TestResnet18Rungs:
    """All three rungs on the README model; matched policy; restoration."""

    def test_declared_and_inferred_agree_with_gold_op_count(self, resnet18_eval: nn.Module) -> None:
        """Matched-policy rungs capture the same graph (op-count witness)."""

        torch.manual_seed(0)
        gold = tl.trace(resnet18_eval, torch.rand(1, 3, 224, 224))
        declared = tl.trace(resnet18_eval, input_size=(1, 3, 224, 224))
        inferred = tl.trace(resnet18_eval)
        assert gold.num_tensors == declared.num_tensors == inferred.num_tensors
        assert trace_input_provenance(gold) is None
        assert trace_input_provenance(declared).origin == "declared"
        assert trace_input_provenance(inferred).origin == "inferred"

    def test_render_and_summary_restore_state_hash(
        self, resnet18_eval: nn.Module, tmp_path
    ) -> None:
        """Memo D8: the pinned surfaces leave the model bit-identical."""

        hash_before = state_dict_hash(resnet18_eval)
        result = render(resnet18_eval, input_size=(1, 3, 224, 224), file=str(tmp_path / "r18.svg"))
        tl.summary(resnet18_eval, input_size=(1, 3, 224, 224))
        assert state_dict_hash(resnet18_eval) == hash_before
        assert result.receipt.state_verified is True
        assert result.total_ops == 151  # the eval-mode regression fixture count

    def test_train_mode_gold_capture_warns_and_differs(self) -> None:
        """The 211-vs-151 measurement is a regression fixture, not a bug.

        torchvision delivers models in train mode; a train-mode gold capture
        keeps the caller's mode (tl.trace is the power surface), moves
        BatchNorm running stats, and warns once per process
        (batchnorm_train_stats_mutated, landed by the summary lane).
        """

        torchvision = pytest.importorskip("torchvision")
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")  # train mode
        assert model.training
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            log = tl.trace(model, torch.rand(1, 3, 224, 224))
        assert log.num_tensors == 211
        eval_log = tl.trace(model.eval(), torch.rand(1, 3, 224, 224))
        assert eval_log.num_tensors == 151


class TestGpt2Rungs:
    """String gold rung, declared ids rung, provenance persistence."""

    @pytest.fixture(scope="class")
    def gpt2(self) -> nn.Module:
        """Real gpt2 checkpoint (cached)."""

        transformers = pytest.importorskip("transformers")
        return transformers.AutoModelForCausalLM.from_pretrained("gpt2").eval()

    def test_string_is_a_gold_input_with_tokenizer_provenance(self, gpt2: nn.Module) -> None:
        """A prompt string is rung 1; the tokenizer record stays authoritative."""

        pytest.importorskip("transformers")
        log = tl.trace(gpt2, "The quick brown fox")
        record = log.input_preprocessor
        assert record is not None
        assert record.source != "torchlens.quickstart.input_resolver"
        assert trace_input_provenance(log) is None  # reads as gold
        from torchlens.errors import TorchLensError

        with pytest.raises(TorchLensError) as excinfo:
            tl.trace(gpt2, "The quick brown fox", input_size=(1, 16))
        assert excinfo.value.fields["code"] == "input_rung_conflict"

    @pytest.mark.slow
    def test_declared_ids_are_in_executed_vocab_and_survive_save_load(
        self, gpt2: nn.Module, tmp_path
    ) -> None:
        """Memo section 6 gpt2 rows: int64 in-vocab ids; the record survives
        the bundle round trip field-for-field; the first raw read AFTER the
        round trip still warns once (the handoff case)."""

        log = tl.trace(gpt2, input_size=(1, 16), save=tl.func("linear"))
        provenance = trace_input_provenance(log)
        assert provenance is not None and provenance.origin == "declared"
        recipe = provenance.recipes[0]
        assert recipe["dtype"] == "torch.int64"
        assert recipe["recipe"] == "randint"
        assert 0 < recipe["high"] <= 50257
        with pytest.raises(Exception) as excinfo:
            log.decode_output()
        assert excinfo.value.fields["code"] == "nongold_semantics_unavailable"

        path = tmp_path / "gpt2_declared.tlspec"
        tl.save(log, str(path))
        loaded = tl.load(str(path))
        loaded_provenance = trace_input_provenance(loaded)
        assert loaded_provenance is not None
        assert dataclasses.asdict(loaded_provenance) == dataclasses.asdict(provenance)
        saved_layers = [
            layer_label
            for layer_label in loaded.layer_labels
            if "linear" in layer_label and ":" not in layer_label
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _ = loaded[saved_layers[0]].out
            _ = loaded[saved_layers[0]].out
        codes = [
            w.message.fields.get("code")
            for w in caught
            if isinstance(w.message, SynthesizedValueReadWarning)
        ]
        assert codes == ["nongold_raw_value_read"]


class TestClipBoundaryRider:
    """Memo section 6: the multi-input model is reachable TODAY via rung 2."""

    @pytest.fixture(scope="class")
    def clip(self) -> nn.Module:
        """Real CLIP checkpoint (cached)."""

        transformers = pytest.importorskip("transformers")
        return transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()

    @pytest.mark.slow
    def test_zero_arg_refusal_names_multi_input_and_quotes_the_probe(self, clip: nn.Module) -> None:
        """The CLIP fix: multi_input_required, never a geometric misdiagnosis."""

        from torchlens.debug._infer_input_shape import infer_input_shape

        result = infer_input_shape(clip)
        assert result.found is False
        assert result.reason == "multi_input_required"
        assert "NoneType" in result.message
        assert "input_size" in result.message

    @pytest.mark.slow
    def test_mapping_form_reaches_clip_with_correct_recipes(self, clip: nn.Module) -> None:
        """Keyword-mapping input_size= succeeds with per-input dtype recipes."""

        log = tl.trace(
            clip,
            input_size={
                "input_ids": (1, 16),
                "pixel_values": (1, 3, 224, 224),
                "attention_mask": (1, 16),
            },
        )
        provenance = trace_input_provenance(log)
        assert provenance is not None
        by_keyword = {r["keyword"]: r for r in provenance.recipes}
        assert by_keyword["input_ids"]["dtype"] == "torch.int64"
        assert by_keyword["pixel_values"]["dtype"] in ("torch.float32", "torch.float64")
        assert by_keyword["attention_mask"]["recipe"] == "ones"
