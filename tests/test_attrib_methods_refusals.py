"""F06: provocation rows for every attribution refusal code (S-17 coverage).

Every stable code the lane introduced ships PROVOKED (error-code coverage
gate): each test genuinely reaches its raise site and branches on
``exc.fields["code"]``, never message text. Codes provoked in the feature
suites are not duplicated here; this file holds the residue.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import torchlens.attribution as attribution
from torchlens.attribution import AttributionError

pytestmark = pytest.mark.smoke


class _DictOut(nn.Module):
    """Model returning a mapping (no bare logits tensor)."""

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Wrap the value in a dict."""

        return {"value": x.sum(dim=(1, 2, 3))}


class _DetachedOut(nn.Module):
    """Model whose output is detached from the input graph."""

    def forward(self, x: Tensor) -> Tensor:
        """Break the graph deliberately."""

        return (x * 2.0).detach()


class _AxisEater(nn.Module):
    """Model whose inner layer reshapes the batch axis away."""

    def __init__(self) -> None:
        """Build the eater and head."""

        super().__init__()
        self.eater = _Reshaper()
        self.head = nn.Linear(6, 2, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        """Collapse the batch axis inside ``eater`` then project."""

        return self.head(self.eater(x))


class _Reshaper(nn.Module):
    """Reshapes any input to one fixed row (drops the stacked axis)."""

    def forward(self, x: Tensor) -> Tensor:
        """Reshape to (1, 6) regardless of the incoming batch."""

        return x.reshape(1, -1)[:, :6]


def test_attribution_target_invalid_provoked() -> None:
    """Int target on a mapping output refuses with the target-invalid code."""

    with pytest.raises(AttributionError) as excinfo:
        attribution.occlusion_map(
            _DictOut(), torch.ones(1, 1, 2, 2, dtype=torch.float64), target=0, window=(2, 2)
        )
    assert excinfo.value.fields["code"] == "attribution_target_invalid"


def test_attribution_target_not_differentiable_provoked() -> None:
    """A detached output under stacked IG refuses with the differentiability code."""

    with pytest.raises(AttributionError) as excinfo:
        attribution.integrated_gradients(
            _DetachedOut(),
            torch.ones(1, 3, dtype=torch.float64),
            target=0,
            n_steps=4,
            step_batch_size=2,
            step_audit="off",
        )
    assert excinfo.value.fields["code"] == "attribution_target_not_differentiable"


def test_guided_sites_invalid_provoked() -> None:
    """sites= outside the closed vocabulary refuses typed."""

    with pytest.raises(AttributionError) as excinfo:
        attribution.guided_backprop(
            nn.Sequential(nn.ReLU()),
            torch.ones(1, 3, dtype=torch.float64),
            target=0,
            sites="everything",
        )
    assert excinfo.value.fields["code"] == "guided_sites_invalid"


def test_noise_aggregation_invalid_provoked() -> None:
    """An unknown aggregation token refuses typed."""

    with pytest.raises(AttributionError) as excinfo:
        attribution.noise_tunnel(
            torch.ones(1, 3, dtype=torch.float64),
            method=attribution.saliency,
            model=nn.Linear(3, 2, dtype=torch.float64),
            target=0,
            stdevs=0.1,
            aggregation="median",
        )
    assert excinfo.value.fields["code"] == "noise_aggregation_invalid"


def test_noise_bank_invalid_provoked() -> None:
    """A stored noise bank that does not cover every draw refuses typed."""

    with pytest.raises(AttributionError) as excinfo:
        attribution.noise_tunnel(
            torch.ones(1, 3, dtype=torch.float64),
            method=attribution.saliency,
            model=nn.Linear(3, 2, dtype=torch.float64),
            target=0,
            stdevs=0.1,
            n_samples=3,
            noise_bank=[[torch.zeros(1, 3, dtype=torch.float64)]],  # covers 1 of 3
        )
    assert excinfo.value.fields["code"] == "noise_bank_invalid"


def test_noise_stdevs_invalid_provoked() -> None:
    """A negative noise scale refuses typed."""

    with pytest.raises(AttributionError) as excinfo:
        attribution.noise_tunnel(
            torch.ones(1, 3, dtype=torch.float64),
            method=attribution.saliency,
            model=nn.Linear(3, 2, dtype=torch.float64),
            target=0,
            stdevs=-0.1,
        )
    assert excinfo.value.fields["code"] == "noise_stdevs_invalid"


def test_step_batch_layer_firings_inconsistent_provoked() -> None:
    """A layer that eats the stacked batch axis refuses under step batching."""

    model = _AxisEater()
    with torch.no_grad():
        model.head.weight.copy_(torch.arange(12, dtype=torch.float64).reshape(2, 6) / 5.0)
        model.head.bias.zero_()
    with pytest.raises(AttributionError) as excinfo:
        attribution.layer_integrated_gradients(
            model,
            torch.ones(1, 6, dtype=torch.float64),
            target=0,
            layer="eater",
            n_steps=4,
            step_batch_size=2,
            step_audit="off",
        )
    assert excinfo.value.fields["code"] == "step_batch_layer_firings_inconsistent"


def test_text_input_invalid_provoked() -> None:
    """Empty text refuses typed before any model work."""

    with pytest.raises(AttributionError) as excinfo:
        attribution.text(nn.Linear(2, 2), object(), "")
    assert excinfo.value.fields["code"] == "text_input_invalid"


def test_text_model_unsupported_provoked() -> None:
    """A model whose output carries no logits tensor refuses typed."""

    class _NoLogits(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wte = nn.Embedding(8, 4, dtype=torch.float64)

        def get_input_embeddings(self) -> nn.Module:
            return self.wte

        def forward(self, **_kwargs: object) -> dict[str, str]:
            return {"nothing": "here"}

    class _Tok:
        pad_token_id = 0
        name_or_path = "tok"

        def __call__(self, text: str, **kwargs: object) -> dict[str, Tensor]:
            ids = [[1, 2, 3]]
            out: dict[str, Tensor] = {"input_ids": torch.tensor(ids)}
            if kwargs.get("return_tensors") == "pt":
                out["attention_mask"] = torch.ones(1, 3, dtype=torch.long)
                if kwargs.get("return_special_tokens_mask"):
                    out["special_tokens_mask"] = torch.zeros(1, 3, dtype=torch.long)
            return out

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return [str(i) for i in ids]

    with pytest.raises(AttributionError) as excinfo:
        attribution.text(_NoLogits(), _Tok(), "a b c", n_steps=2)
    assert excinfo.value.fields["code"] == "text_model_unsupported"


class _TinyHead(nn.Module):
    """Minimal named-layer model for the layer-path internal tripwires."""

    def __init__(self) -> None:
        """Build the single named layer."""

        super().__init__()
        self.lin = nn.Linear(3, 2, dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        """Project the input."""

        return self.lin(x)


def test_layer_path_gradient_missing_provoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A require-gradient capture returning no gradients trips the tripwire."""

    from torchlens.attribution import _layer as layer_module

    real = layer_module._capture_layer_activation_for_leaves

    def _gradless(*args: object, **kwargs: object) -> tuple[object, object, object]:
        activations, gradients, scalar = real(*args, **kwargs)
        if kwargs.get("require_gradient"):
            gradients = None
        return activations, gradients, scalar

    monkeypatch.setattr(layer_module, "_capture_layer_activation_for_leaves", _gradless)
    with pytest.raises(AttributionError) as excinfo:
        attribution.layer_integrated_gradients(
            _TinyHead(), torch.ones(1, 3, dtype=torch.float64), target=0, layer="lin", n_steps=2
        )
    assert excinfo.value.fields["code"] == "layer_path_gradient_missing"


def test_layer_conductance_edge_missing_provoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A path run losing its requested right-edge activations trips the tripwire."""

    from torchlens.attribution import _layer as layer_module

    def _edgeless(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {
            "baseline_activations": (torch.zeros(1, 2, dtype=torch.float64),),
            "gradients_by_step": [],
            "right_activations_by_step": None,
        }

    monkeypatch.setattr(layer_module, "_layer_path_run", _edgeless)
    with pytest.raises(AttributionError) as excinfo:
        attribution.layer_conductance(
            _TinyHead(), torch.ones(1, 3, dtype=torch.float64), target=0, layer="lin", n_steps=2
        )
    assert excinfo.value.fields["code"] == "layer_conductance_edge_missing"


def test_noise_tunnel_route_unbound_provoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sugar-route binding without a bound method/model trips the tripwire."""

    from torchlens.attribution import _binder, _noise_tunnel

    def _unbound(**_kwargs: object) -> _binder._BoundMethod:
        return _binder._BoundMethod(
            call=lambda _inputs, _input_kwargs: None,  # type: ignore[arg-type,return-value]
            kind="layer",
            method_name="grad_cam",
            settings={},
            known_stochastic=False,
            frozen_randomness=False,
            model_identity=None,
        )

    monkeypatch.setattr(_noise_tunnel, "_bind_method", _unbound)
    with pytest.raises(AttributionError) as excinfo:
        attribution.noise_tunnel(torch.ones(1, 2, dtype=torch.float64), n_samples=1, stdevs=0.1)
    assert excinfo.value.fields["code"] == "noise_tunnel_route_unbound"


def test_text_auto_ladder_empty_provoked(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty auto ladder (no grid ever evaluated) trips the tripwire."""

    from torchlens.attribution import _text as text_module

    vocab = ["<pad>", "the", "cat", "sat"]

    class _MiniForCausalLM(nn.Module):
        class _Config:
            _name_or_path = "mini-causal"
            is_decoder = True

        def __init__(self) -> None:
            super().__init__()
            self.config = self._Config()
            self.wte = nn.Embedding(len(vocab), 4, dtype=torch.float64)
            self.head = nn.Linear(4, len(vocab), dtype=torch.float64)

        def get_input_embeddings(self) -> nn.Module:
            return self.wte

        def forward(
            self,
            inputs_embeds: Tensor | None = None,
            attention_mask: Tensor | None = None,
            **_ignored: object,
        ) -> tuple[Tensor]:
            del attention_mask
            assert inputs_embeds is not None
            return (self.head(torch.sin(inputs_embeds)),)

    class _MiniTokenizer:
        pad_token_id = 0
        name_or_path = "mini-tok"

        def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
            ids = [vocab.index(word) for word in text.lower().split()]
            if kwargs.get("truncation") and kwargs.get("max_length") is not None:
                ids = ids[: int(kwargs["max_length"])]  # type: ignore[call-overload]
            encoding: dict[str, object] = {"input_ids": ids}
            if kwargs.get("return_tensors") == "pt":
                encoding = {
                    "input_ids": torch.tensor([ids]),
                    "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                }
                if kwargs.get("return_special_tokens_mask"):
                    encoding["special_tokens_mask"] = torch.zeros(1, len(ids), dtype=torch.long)
            return encoding

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [vocab.index(word) for word in text.lower().split()]

        def decode(self, ids: list[int]) -> str:
            return " ".join(vocab[index] for index in ids)

        def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
            return [vocab[index] for index in ids]

    monkeypatch.setattr(text_module, "_AUTO_LADDER", ())
    with pytest.raises(AttributionError) as excinfo:
        attribution.text(_MiniForCausalLM(), _MiniTokenizer(), "the cat sat", n_steps="auto")
    assert excinfo.value.fields["code"] == "text_auto_ladder_empty"
