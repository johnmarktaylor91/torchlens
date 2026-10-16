"""SAE latents as a facet family: reads, edits and attribution targets.

``tl.bridge.sae.register("res", sae, site="fc1")`` makes the SAE's latents the
facet family ``"res"`` at ``fc1``; ``tl.facet("res").latent(k)`` addresses
latent ``k`` at the doors a ``tl.head`` selector serves (``Trace.set``,
``attach_hooks``, ``Trace.do``, ``fork().do``, ``tl.do``) and
``trace["fc1"].facets["res[k]"]`` reads it. An edit adds the decoder-space
change of the edited latents to the site and leaves the reconstruction error
alone. ``tl.attribution.SiteTarget("fc1", sae=sae, unit=k)`` attributes latent
``k``.

The fixture is hand-set: ``fc1`` is the identity, so the site is the input,
and every weight, input and latent is a small dyadic rational, so float32
arithmetic is exact. Each expected value is computed here in plain Python from
the weights, never through TorchLens or the SAE's own methods. Tolerances,
fixed before the first run: ``atol=1e-6, rtol=0`` for every comparison.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.attribution import AttributionError

X = [[1.0, 2.0, -1.0, 3.0], [2.0, -1.0, 4.0, 0.0]]
W_ENC = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -0.5, 1.0], [0.0, 0.0, 0.5]]
B_ENC = [0.0, 0.0, 0.25]
W_DEC = [[1.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]]
B_DEC = [0.5, 0.0, 0.0, 0.0]
W_OUT = [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]
TOL = {"atol": 1e-6, "rtol": 0.0}


def _pre(row: list[float]) -> list[float]:
    """Encoder pre-activations of one site row, by hand."""

    return [sum((row[i] - B_DEC[i]) * W_ENC[i][k] for i in range(4)) + B_ENC[k] for k in range(3)]


def _latents(row: list[float]) -> list[float]:
    """Encoder activations of one site row, by hand."""

    return [max(0.0, value) for value in _pre(row)]


def _decode(z: list[float]) -> list[float]:
    """Decoder output of one latent row, by hand."""

    return [sum(z[k] * W_DEC[k][j] for k in range(3)) + B_DEC[j] for j in range(4)]


def _readout(row: list[float]) -> list[float]:
    """Model output of one site row, by hand."""

    return [sum(W_OUT[o][j] * row[j] for j in range(4)) for o in range(2)]


class _HandSAE(nn.Module):
    """A ReLU SAE with the hand-set weights above."""

    def __init__(self) -> None:
        super().__init__()
        self.W_enc = nn.Parameter(torch.tensor(W_ENC))
        self.b_enc = nn.Parameter(torch.tensor(B_ENC))
        self.W_dec = nn.Parameter(torch.tensor(W_DEC))
        self.b_dec = nn.Parameter(torch.tensor(B_DEC))

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.relu((h - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec


class _Model(nn.Module):
    """``fc1`` (identity) then a hand-set readout."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.eye(4))
            self.fc1.bias.zero_()
            self.fc2.weight.copy_(torch.tensor(W_OUT))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


@pytest.fixture()
def setup() -> Iterator[tuple[_Model, torch.Tensor, _HandSAE]]:
    """The model, its input and the SAE registered as ``"res"`` at ``fc1``."""

    model, sae = _Model().eval(), _HandSAE()
    tl.bridge.sae.register("res", sae, site="fc1")
    try:
        yield model, torch.tensor(X), sae
    finally:
        tl.bridge.sae.unregister("res")


def _trace(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    return tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))


def _site_out(trace: tl.Trace) -> torch.Tensor:
    return trace["fc1"].out


def _model_out(trace: tl.Trace) -> torch.Tensor:
    return trace[trace.output_layers[0]].out


def _latent_column(k: int) -> torch.Tensor:
    return torch.tensor([_latents(row)[k] for row in X])


# --- item 3: reads at every door ------------------------------------------------------


def test_facet_read_door_returns_the_encoder_activation(setup) -> None:
    """``trace[site].facets`` serves every latent, one latent and a chosen set."""

    model, x, _sae = setup
    facets = _trace(model, x)["fc1"].facets
    assert "res" in facets.keys()
    torch.testing.assert_close(facets["res"].value, torch.tensor([_latents(r) for r in X]), **TOL)
    torch.testing.assert_close(facets["res[1]"].value, _latent_column(1), **TOL)
    expected_set = torch.tensor([[_latents(r)[0], _latents(r)[2]] for r in X])
    torch.testing.assert_close(facets["res[0,2]"].value, expected_set, **TOL)


def _recording_hook(seen: list[torch.Tensor]):
    def _hook(latents: torch.Tensor, *, hook=None) -> torch.Tensor:
        seen.append(latents.detach().clone())
        return latents

    return _hook


_IN_PLACE = tl.options.InterventionOptions(confirm_mutation=True)

_DOORS = {
    "attach_hooks": lambda trace, site, hook: trace.attach_hooks(site, hook) and trace.push(),
    "trace_do": lambda trace, site, hook: trace.do(site, hook, intervention=_IN_PLACE),
    "fork_do": lambda trace, site, hook: trace.fork("door").do(site, hook),
    "tl_do": lambda trace, site, hook: tl.do(trace, site, hook, intervention=_IN_PLACE),
    "set": lambda trace, site, hook: trace.fork("door").set(site, hook).push(),
}


@pytest.mark.parametrize("door", sorted(_DOORS))
def test_every_head_door_hands_the_edit_the_latent_activation(setup, door) -> None:
    """The edit at each door the head selector serves receives latent 1's activation."""

    model, x, _sae = setup
    seen: list[torch.Tensor] = []
    _DOORS[door](_trace(model, x), tl.facet("res").latent(1), _recording_hook(seen))
    assert seen, f"the {door} door never handed the edit a latent"
    torch.testing.assert_close(seen[0], _latent_column(1), **TOL)


def test_unknown_latent_refuses_and_its_remedy_runs(setup) -> None:
    model, x, _sae = setup
    with pytest.raises(InvalidArgumentError) as caught:
        tl.facet("res").latent(3)
    assert caught.value.fields["code"] == "sae_latent_index_unknown"
    assert "tl.facet('res').latent(2)" in str(caught.value)
    seen: list[torch.Tensor] = []
    _trace(model, x).fork("edit").do(tl.facet("res").latent(2), _recording_hook(seen))
    torch.testing.assert_close(seen[0], _latent_column(2), **TOL)


def test_unknown_sae_refuses_and_its_remedy_runs(setup) -> None:
    model, x, sae = setup
    with pytest.raises(InvalidArgumentError) as caught:
        tl.facet("res9").latent(0)
    assert caught.value.fields["code"] == "sae_latent_family_unknown"
    assert "tl.bridge.sae.register('res9', sae, site=" in str(caught.value)
    tl.bridge.sae.register("res9", sae, site="fc1")
    try:
        seen: list[torch.Tensor] = []
        _trace(model, x).fork("edit").do(tl.facet("res9").latent(0), _recording_hook(seen))
        torch.testing.assert_close(seen[0], _latent_column(0), **TOL)
    finally:
        tl.bridge.sae.unregister("res9")


class _Loop(nn.Module):
    """``fc1`` runs twice: a string site naming it is not one op."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(torch.tanh(self.fc1(x))))


def test_one_latent_spelling_on_a_layer_that_runs_twice_refuses_with_a_set_spelling() -> None:
    """A string site must name one op; the selector spelling its refusal names runs."""

    torch.manual_seed(0)
    model, x, sae = _Loop().eval(), torch.randn(2, 4), _HandSAE()
    tl.bridge.sae.register("loop", sae, site="fc1")
    try:
        with pytest.raises(InvalidArgumentError) as caught:
            _trace(model, x).fork("edit").do(tl.facet("loop").latent(0), tl.zero_ablate())
        assert caught.value.fields["code"] == "sae_latent_site_ambiguous"
        assert "site=tl.module('fc1')" in str(caught.value)
    finally:
        tl.bridge.sae.unregister("loop")
    tl.bridge.sae.register("loop", sae, site=tl.module("fc1"))
    try:
        clean = _trace(model, x)
        edited = clean.fork("all_passes").do(tl.facet("loop").latent(2), tl.zero_ablate())
        assert not torch.equal(_model_out(edited), _model_out(clean))
    finally:
        tl.bridge.sae.unregister("loop")


class _TwoSites(nn.Module):
    """Two 4-wide layers an SAE of width 4 can read."""

    def __init__(self) -> None:
        super().__init__()
        self.fc_a = nn.Linear(4, 4)
        self.fc_b = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc_b(torch.tanh(self.fc_a(x))))


def test_a_set_site_spelling_fans_out_and_reports_its_count() -> None:
    torch.manual_seed(0)
    model, x, sae = _TwoSites().eval(), torch.randn(2, 4), _HandSAE()
    tl.bridge.sae.register("both", sae, site=tl.module("fc_a") | tl.module("fc_b"))
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _trace(model, x).fork("edit").do(tl.facet("both").latent(0), tl.zero_ablate())
        counts = [
            getattr(w.message, "fields", {}).get("site_count")
            for w in caught
            if getattr(w.message, "fields", {}).get("code") == "facet_selector_fanout"
        ]
        assert counts == [2]
        # Scoped to one module, the same latent is one write and no fan-out.
        with warnings.catch_warnings(record=True) as scoped:
            warnings.simplefilter("always")
            _trace(model, x).fork("one").do(
                tl.facet("both").latent(0).in_module("fc_a"), tl.zero_ablate()
            )
        scoped_codes = [getattr(w.message, "fields", {}).get("code") for w in scoped]
        assert "facet_selector_fanout" not in scoped_codes
    finally:
        tl.bridge.sae.unregister("both")


def test_bad_names_and_indices_refuse_with_codes(setup) -> None:
    _model, _x, sae = setup
    with pytest.raises(InvalidArgumentError) as name_taken:
        tl.bridge.sae.register("res", sae, site="fc1")
    assert name_taken.value.fields["code"] == "sae_family_name_taken"
    with pytest.raises(InvalidArgumentError) as bad_name:
        tl.bridge.sae.register("res[1]", sae, site="fc1")
    assert bad_name.value.fields["code"] == "sae_family_name_invalid"
    with pytest.raises(InvalidArgumentError) as bad_index:
        tl.facet("res").latent(-1)
    assert bad_index.value.fields["code"] == "sae_latent_index_invalid"
    with pytest.raises(InvalidArgumentError) as unknown:
        tl.bridge.sae.unregister("nope")
    assert unknown.value.fields["code"] == "sae_family_unknown"
    with pytest.raises(TypeError):
        tl.bridge.sae.register("plain", object(), site="fc1")


def test_latent_selector_repr_is_its_spelling(setup) -> None:
    assert repr(tl.facet("res").latent(2)) == "tl.facet('res').latent(2)"
    assert repr(tl.facet("res").latent([0, 2])) == "tl.facet('res').latent([0,2])"
    assert repr(tl.facet("res").latent([1])) == "tl.facet('res').latent([1])"


# --- item 4: writes add a decoder direction and keep the error term -------------------


@pytest.mark.parametrize("delta", [1.5, -0.25])
def test_latent_edit_adds_delta_times_the_decoder_row(setup, delta) -> None:
    model, x, sae = setup
    clean = _trace(model, x)
    edited = clean.fork("steer").do(tl.facet("res").latent(1), tl.add(delta))
    expected_site = [[h[j] + delta * W_DEC[1][j] for j in range(4)] for h in X]
    torch.testing.assert_close(_site_out(edited), torch.tensor(expected_site), **TOL)
    torch.testing.assert_close(
        _site_out(edited) - _site_out(clean), delta * torch.tensor([W_DEC[1]] * 2), **TOL
    )
    # The reconstruction error is carried unchanged: site' - decode(z') == site - decode(z).
    for row, edited_row in zip(X, expected_site):
        z = _latents(row)
        z_edited = [z[0], z[1] + delta, z[2]]
        error = [row[j] - _decode(z)[j] for j in range(4)]
        error_edited = [edited_row[j] - _decode(z_edited)[j] for j in range(4)]
        assert error_edited == pytest.approx(error, abs=1e-6)
    # Downstream equals a hand-applied edit of the same vector on an unspliced model.
    direction = delta * sae.W_dec[1].detach()
    handle = model.fc1.register_forward_hook(lambda _m, _a, out: out + direction)
    try:
        with torch.no_grad():
            hand = model(x)
    finally:
        handle.remove()
    torch.testing.assert_close(_model_out(edited), hand, **TOL)
    torch.testing.assert_close(
        _model_out(edited), torch.tensor([_readout(r) for r in expected_site]), **TOL
    )


def test_zero_ablating_a_latent_removes_its_decoder_contribution(setup) -> None:
    model, x, _sae = setup
    edited = _trace(model, x).fork("edit").do(tl.facet("res").latent(0), tl.zero_ablate())
    expected = [[h[j] - _latents(h)[0] * W_DEC[0][j] for j in range(4)] for h in X]
    torch.testing.assert_close(_site_out(edited), torch.tensor(expected), **TOL)


def test_a_latent_set_is_one_simultaneous_write(setup) -> None:
    model, x, _sae = setup
    edited = _trace(model, x).fork("edit").do(tl.facet("res").latent([0, 2]), tl.add(1.0))
    expected = [[h[j] + W_DEC[0][j] + W_DEC[2][j] for j in range(4)] for h in X]
    torch.testing.assert_close(_site_out(edited), torch.tensor(expected), **TOL)


def test_set_writes_a_latent_value(setup) -> None:
    model, x, _sae = setup
    edited = _trace(model, x).fork("pin").set(tl.facet("res").latent(1), torch.full((2,), 4.0))
    expected = [[h[j] + (4.0 - _latents(h)[1]) * W_DEC[1][j] for j in range(4)] for h in X]
    edited.push()
    torch.testing.assert_close(_site_out(edited), torch.tensor(expected), **TOL)


# --- item 5: an SAE latent as an attribution target -----------------------------------


def test_sae_latent_is_an_attribution_target_with_its_closed_form(setup) -> None:
    """d z_1 / d x is W_enc[:, 1] where the latent is active, zero where it is not."""

    model, x, sae = setup
    target = tl.attribution.SiteTarget("fc1", sae=sae, unit=1)
    result = tl.attribution.input_x_grad(model, x, target=target)
    expected = [[row[i] * W_ENC[i][1] if _pre(row)[1] > 0 else 0.0 for i in range(4)] for row in X]
    torch.testing.assert_close(result.values, torch.tensor(expected), **TOL)
    saliency = tl.attribution.saliency(model, x, target=target)
    expected_abs = [[abs(W_ENC[i][1]) if _pre(row)[1] > 0 else 0.0 for i in range(4)] for row in X]
    torch.testing.assert_close(saliency.values, torch.tensor(expected_abs), **TOL)


def test_sae_target_refuses_an_object_without_a_codec(setup) -> None:
    with pytest.raises(AttributionError) as caught:
        tl.attribution.SiteTarget("fc1", sae=object(), unit=0)
    assert caught.value.fields["code"] == "attribution_site_sae_invalid"


def test_sae_target_unit_outside_the_latents_refuses_naming_the_sae_spelling(setup) -> None:
    model, x, sae = setup
    with pytest.raises(AttributionError) as caught:
        tl.attribution.saliency(model, x, target=tl.attribution.SiteTarget("fc1", sae=sae, unit=7))
    assert caught.value.fields["code"] == "attribution_site_unit_invalid"
    assert "unit=0, sae=sae" in str(caught.value)
