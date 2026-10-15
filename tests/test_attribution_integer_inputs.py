"""Integer inputs for the layer methods, and the input methods' coded refusal.

A layer method given only integer inputs (token ids) attributes from the
first floating-point values computed from them, the embedding output. Each
result must equal the same method run on a model that takes that embedding
output as its floating-point input.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import torch
from torch import Tensor, nn

import torchlens as tl
from torchlens.attribution import AttributionError

_ATOL = 1e-6


class _Body(nn.Module):
    """Transformer encoder layer and a pooled linear head over embedded tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True
        )
        self.head = nn.Linear(8, 3)

    def forward(self, embedded: Tensor) -> Tensor:
        """Encode the tokens and score the mean token."""

        return self.head(self.encoder(embedded).mean(dim=1))


class _TokenModel(nn.Module):
    """Token ids, an embedding, then the shared body."""

    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        self.embed = nn.Embedding(11, 8)
        self.body = body

    def forward(self, ids: Tensor) -> Tensor:
        """Embed the ids and run the body."""

        return self.body(self.embed(ids))


class _FloatModel(nn.Module):
    """The same body, taking the embedding output as its input."""

    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        self.body = body

    def forward(self, embedded: Tensor) -> Tensor:
        """Run the body on floating-point embeddings."""

        return self.body(embedded)


class _GridBody(nn.Module):
    """Convolution over an ``N, C, H, W`` token map and a linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 6, kernel_size=3, padding=1)
        self.head = nn.Linear(6, 3)

    def forward(self, grid: Tensor) -> Tensor:
        """Convolve, pool and score."""

        return self.head(torch.relu(self.conv(grid)).mean(dim=(2, 3)))


class _GridTokenModel(nn.Module):
    """Token ids laid out ``N, H, W`` (a discrete image), embedded to ``N, C, H, W``."""

    def __init__(self, body: nn.Module) -> None:
        super().__init__()
        self.embed = nn.Embedding(7, 4)
        self.body = body

    def forward(self, ids: Tensor) -> Tensor:
        """Embed each grid cell and run the convolutional body."""

        return self.body(self.embed(ids).permute(0, 3, 1, 2))


class _CastModel(nn.Module):
    """Integer features cast to float inside forward: no embedding module."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 3)

    def forward(self, ids: Tensor) -> Tensor:
        """Cast and score."""

        return self.fc(ids.float())


def _token_pair() -> tuple[_TokenModel, _FloatModel, Tensor]:
    """Return the token model, its float twin sharing the body, and token ids."""

    torch.manual_seed(0)
    body = _Body()
    token_model = _TokenModel(body).eval()
    float_model = _FloatModel(body).eval()
    ids = torch.randint(1, 11, (2, 5))
    return token_model, float_model, ids


def _remedy(error: AttributionError, pattern: str) -> str:
    """Return the remedy spelling the refusal message gives."""

    match = re.search(pattern, str(error))
    assert match is not None, str(error)
    return match.group(1)


_LAYER_METHODS = {
    "layer_attribution": {},
    "layer_integrated_gradients": {"n_steps": 8},
    "layer_conductance": {"n_steps": 8},
}


@pytest.mark.parametrize("method_name", sorted(_LAYER_METHODS))
def test_layer_method_on_integer_ids_matches_float_embedding_run(method_name: str) -> None:
    """Each layer method on token ids equals the same method on the embedding output."""

    token_model, float_model, ids = _token_pair()
    method = getattr(tl.attribution, method_name)
    options = _LAYER_METHODS[method_name]
    from_ids = method(token_model, ids, target=1, layer="body.encoder", **options)
    embedded = token_model.embed(ids).detach()
    from_floats = method(float_model, embedded, target=1, layer="body.encoder", **options)
    torch.testing.assert_close(from_ids.values, from_floats.values, atol=_ATOL, rtol=0)
    assert from_ids.extra["integer_input_layers"] == ("embed",)
    assert from_floats.extra["integer_input_layers"] == ()
    assert bool(from_ids.values.abs().sum() > 0)


def test_layer_integrated_gradients_baseline_ids_match_embedded_baseline() -> None:
    """Integer baseline ids start the path at their own embedding."""

    token_model, float_model, ids = _token_pair()
    baseline_ids = torch.zeros_like(ids)
    from_ids = tl.attribution.layer_integrated_gradients(
        token_model, ids, target=2, layer="body.encoder", baseline=baseline_ids, n_steps=8
    )
    from_floats = tl.attribution.layer_integrated_gradients(
        float_model,
        token_model.embed(ids).detach(),
        target=2,
        layer="body.encoder",
        baseline=token_model.embed(baseline_ids).detach(),
        n_steps=8,
    )
    torch.testing.assert_close(from_ids.values, from_floats.values, atol=_ATOL, rtol=0)
    torch.testing.assert_close(
        torch.as_tensor(from_ids.extra["target_delta"]),
        torch.as_tensor(from_floats.extra["target_delta"]),
        atol=_ATOL,
        rtol=0,
    )


def test_grad_cam_on_integer_token_grid_matches_float_embedding_run() -> None:
    """grad_cam on an ``N, H, W`` id grid equals grad_cam on its embedded map."""

    torch.manual_seed(0)
    body = _GridBody()
    token_model = _GridTokenModel(body).eval()
    float_model = _FloatModel(body).eval()
    ids = torch.randint(0, 7, (2, 5, 6))
    from_ids = tl.attribution.grad_cam(token_model, ids, target=0, layer="body.conv", overlay=False)
    embedded = token_model.embed(ids).permute(0, 3, 1, 2).detach()
    from_floats = tl.attribution.grad_cam(
        float_model, embedded, target=0, layer="body.conv", overlay=False
    )
    assert tuple(from_ids.values.shape) == (2, 1, 5, 6)
    torch.testing.assert_close(from_ids.values, from_floats.values, atol=_ATOL, rtol=0)
    assert from_ids.extra["integer_input_layers"] == ("embed",)


_INPUT_METHODS: dict[str, dict[str, Any]] = {
    "saliency": {},
    "input_x_grad": {},
    "integrated_gradients": {"n_steps": 4},
    "smoothgrad": {"n_samples": 2},
    "gradient_shap": {"baselines": torch.zeros(1, 5, dtype=torch.long), "n_samples": 2},
    "guided_backprop": {},
}


@pytest.mark.parametrize("method_name", sorted(_INPUT_METHODS))
def test_input_method_refuses_integer_ids_and_its_remedy_runs(method_name: str) -> None:
    """Input methods refuse token ids with a code, and the remedy they spell runs."""

    token_model, _float_model, ids = _token_pair()
    method = getattr(tl.attribution, method_name)
    with pytest.raises(AttributionError) as caught:
        method(token_model, ids, target=1, **_INPUT_METHODS[method_name])
    assert caught.value.code == "attribution_input_not_float"
    message = str(caught.value)
    assert method_name in message
    assert "torch.int64 tensor of shape (2, 5)" in message
    spelling = _remedy(caught.value, r"e\.g\. (tl\.attribution\.\w+\(.*?\))\. See")
    assert spelling == (
        "tl.attribution.layer_integrated_gradients(model, inputs, target=1, layer='embed')"
    )
    result = eval(spelling, {"tl": tl, "model": token_model, "inputs": ids})  # noqa: S307
    assert tuple(result.values.shape) == (2, 5, 8)


def test_layer_path_refuses_integer_inputs_without_an_embedding_and_its_remedy_runs() -> None:
    """Path methods refuse integers no module embeds; the path-free remedy runs."""

    torch.manual_seed(0)
    model = _CastModel().eval()
    ids = torch.randint(0, 5, (2, 6))
    with pytest.raises(AttributionError) as caught:
        tl.attribution.layer_integrated_gradients(model, ids, target=0, layer="fc", n_steps=4)
    assert caught.value.code == "layer_path_no_embedding"
    spelling = _remedy(caught.value, r"e\.g\. (tl\.attribution\.\w+\(.*?\)), or")
    assert spelling == "tl.attribution.layer_attribution(model, inputs, target=0, layer='fc')"
    result = eval(spelling, {"tl": tl, "model": model, "inputs": ids})  # noqa: S307
    assert tuple(result.values.shape) == (2, 3)
    assert result.extra["integer_input_layers"] == ()


def test_layer_path_refuses_baseline_ids_that_do_not_mirror_inputs() -> None:
    """Baseline ids of another shape refuse; the spelled baseline runs."""

    token_model, _float_model, ids = _token_pair()
    with pytest.raises(AttributionError) as caught:
        tl.attribution.layer_integrated_gradients(
            token_model, ids, target=1, layer="body.encoder", baseline=ids[:, :3], n_steps=4
        )
    assert caught.value.code == "layer_path_baseline_ids_invalid"
    spelling = _remedy(caught.value, r"e\.g\. (baseline=torch\.zeros_like\(inputs\))")
    result = eval(  # noqa: S307
        f"tl.attribution.layer_integrated_gradients(model, inputs, target=1, "
        f"layer='body.encoder', {spelling}, n_steps=4)",
        {"tl": tl, "torch": torch, "model": token_model, "inputs": ids},
    )
    assert tuple(result.values.shape) == (2, 5, 8)


class _SequenceGridModel(nn.Module):
    """Token ids ``N, T`` folded into a map inside forward: no integer grid input."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(7, 4)
        self.body = _GridBody()

    def forward(self, ids: Tensor) -> Tensor:
        """Embed, fold 30 tokens onto a 5x6 map and run the body."""

        grid = self.embed(ids).reshape(ids.shape[0], 5, 6, 4).permute(0, 3, 1, 2)
        return self.body(grid)


def test_grad_cam_refuses_integer_inputs_without_a_grid_and_its_remedy_runs() -> None:
    """grad_cam on a token sequence has no input grid to lay the map on."""

    torch.manual_seed(0)
    model = _SequenceGridModel().eval()
    ids = torch.randint(0, 7, (2, 30))
    with pytest.raises(AttributionError) as caught:
        tl.attribution.grad_cam(model, ids, target=0, layer="body.conv", overlay=False)
    assert caught.value.code == "grad_cam_integer_grid_invalid"
    spelling = _remedy(caught.value, r"e\.g\. (tl\.attribution\.\w+\(.*?\))\. See")
    result = eval(spelling, {"tl": tl, "model": model, "inputs": ids, "target": 0})  # noqa: S307
    assert tuple(result.values.shape) == (2, 6, 5, 6)


class _ShrinkingModel(nn.Module):
    """Embeds a different batch after its first call: control flow changes between passes."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(7, 4)
        self.fc = nn.Linear(4, 2)
        self.calls = 0

    def forward(self, ids: Tensor) -> Tensor:
        """Embed the ids on the first call and a doubled batch afterwards."""

        self.calls += 1
        tokens = ids if self.calls == 1 else torch.cat([ids, ids])
        return self.fc(self.embed(tokens).mean(dim=1))[: ids.shape[0]]


def test_integer_route_refuses_embedding_that_changes_between_passes() -> None:
    """An embedding whose output shape changes between passes refuses, coded."""

    model = _ShrinkingModel().eval()
    ids = torch.randint(0, 7, (2, 6))
    with pytest.raises(AttributionError) as caught:
        tl.attribution.layer_attribution(model, ids, target=0, layer="fc")
    assert caught.value.code == "layer_path_embedding_inconsistent"
