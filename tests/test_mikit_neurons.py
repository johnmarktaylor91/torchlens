"""Per-neuron MLP contribution rows under the D7 identity gate (R2-26, N-MI-2).

The rows must sum to the captured MLP output they decompose, and each row
must equal an independent hand computation from the model's own modules
(GPT-2's Conv1D ``c_fc``/``c_proj`` with GELU; Llama's gated
``gate_proj``/``up_proj``/``down_proj``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import torchlens as tl
import torchlens.mechinterp as mi
from torchlens.mechinterp._errors import MechInterpError

sys.path.insert(0, str(Path(__file__).resolve().parent / "real_model" / "r0"))

TOL = {"rtol": 1e-5, "atol": 1e-5}


def _capture(builder_name: str):
    import families

    model = getattr(families, builder_name)("eager")
    torch.manual_seed(5)
    x = torch.randint(0, 500, (2, 6))
    trace = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    return model, trace


@pytest.fixture(scope="module")
def gpt2_bundle():
    model, trace = _capture("build_gpt2")
    try:
        yield model, trace
    finally:
        trace.cleanup()


@pytest.fixture(scope="module")
def llama_bundle():
    model, trace = _capture("build_llama")
    try:
        yield model, trace
    finally:
        trace.cleanup()


def _mlp_io(trace, address):
    module = trace.modules[address]
    inp = module.facets["input"]
    out = module.facets["output"]
    return getattr(inp, "value", inp), getattr(out, "value", out)


def test_neuron_rows_sum_to_mlp_output_gpt2(gpt2_bundle):
    model, trace = gpt2_bundle
    d_mlp = model.transformer.h[0].mlp.c_fc.nf
    for layer in range(model.config.n_layer):
        address = f"transformer.h.{layer}.mlp"
        stack = mi.mlp_neuron_contributions(trace, address)
        mlp_in, mlp_out = _mlp_io(trace, address)
        # d_mlp neuron rows + the verified c_proj bias constant.
        assert len(stack) == d_mlp + 1
        assert stack.grading == "closed_constant" and not stack.diagnostic_only
        assert stack.identity_receipt["per_layer"][0]["remainder"] == "verified_bias"
        assert stack.identity_receipt["per_layer"][0]["hidden_source"] == (
            "captured_projection_input"
        )
        torch.testing.assert_close(stack.sum(), mlp_out, **TOL)
        assert torch.equal(stack.target_value, mlp_out)
        # Independent oracle from the model's own modules.
        mlp = model.transformer.h[layer].mlp
        with torch.no_grad():
            hidden = mlp.act(mlp.c_fc(mlp_in))
        w_out = mlp.c_proj.weight.detach()  # Conv1D: [d_mlp, d_model]
        for n in (0, 7, d_mlp - 1):
            row = stack.rows[n]
            assert row.coordinate.label == f"{address}.neuron{n}"
            assert row.coordinate.kind == "mlp"
            torch.testing.assert_close(row.value, hidden[..., n : n + 1] * w_out[n], **TOL)
        assert stack.rows[-1].coordinate.kind == "constant"
        torch.testing.assert_close(
            stack.rows[-1].value, mlp.c_proj.bias.detach().expand_as(mlp_out), **TOL
        )


def test_neuron_rows_sum_to_mlp_output_gated_llama(llama_bundle):
    model, trace = llama_bundle
    stack = mi.mlp_neuron_contributions(trace)
    d_mlp = model.config.intermediate_size
    n_layers = model.config.num_hidden_layers
    # Llama's down_proj has no bias: the remainder is proven zero, grading complete.
    assert len(stack) == n_layers * d_mlp
    assert stack.grading == "complete"
    assert [r["remainder"] for r in stack.identity_receipt["per_layer"]] == [
        "verified_zero"
    ] * n_layers
    outputs = [_mlp_io(trace, f"model.layers.{i}.mlp")[1] for i in range(n_layers)]
    torch.testing.assert_close(stack.sum(), outputs[0] + outputs[1], **TOL)
    mlp = model.model.layers[1].mlp
    mlp_in, mlp_out = _mlp_io(trace, "model.layers.1.mlp")
    with torch.no_grad():
        hidden = mlp.act_fn(mlp.gate_proj(mlp_in)) * mlp.up_proj(mlp_in)
    w_out = mlp.down_proj.weight.detach().T  # Linear: [d_model, d_mlp] stored
    layer1 = stack.rows[d_mlp : 2 * d_mlp]
    torch.testing.assert_close(sum(r.value for r in layer1), mlp_out, **TOL)
    for n in (0, 31, d_mlp - 1):
        torch.testing.assert_close(layer1[n].value, hidden[..., n : n + 1] * w_out[n], **TOL)


def test_neuron_subset_folds_rest_and_onto_contracts(gpt2_bundle):
    model, trace = gpt2_bundle
    address = "transformer.h.1.mlp"
    _, mlp_out = _mlp_io(trace, address)
    subset = mi.mlp_neuron_contributions(trace, 1, neurons=[5, 0, 5])
    assert subset.labels == (
        f"{address}.neuron0",
        f"{address}.neuron5",
        f"{address}.other_neurons",
        f"{address}.c_proj.bias",
    )
    torch.testing.assert_close(subset.sum(), mlp_out, **TOL)
    full = mi.mlp_neuron_contributions(trace, address)
    torch.testing.assert_close(subset.rows[1].value, full.rows[5].value, **TOL)
    direction = torch.randn(model.config.n_embd, 3, generator=torch.Generator().manual_seed(1))
    contracted = mi.mlp_neuron_contributions(
        trace, address, neurons=[0], onto=direction, positions=[0, 2]
    )
    expected = (mlp_out @ direction)[:, [0, 2]]
    assert contracted.identity_receipt["contracted"] is True
    assert tuple(contracted.rows[0].value.shape) == (2, 2, 3)
    torch.testing.assert_close(contracted.target_value, expected, **TOL)
    torch.testing.assert_close(contracted.sum(), expected, **TOL)


def test_neuron_refusals(gpt2_bundle):
    _, trace = gpt2_bundle
    cases = [
        ({"layers": 99}, "mi_neuron_geometry_unavailable"),
        ({"layers": "transformer.h.0.attn"}, "mi_neuron_geometry_unavailable"),
        ({"layers": 0, "neurons": [10**6]}, "mi_neuron_index_invalid"),
        ({"layers": 0, "neurons": [True]}, "mi_neuron_index_invalid"),
    ]
    for kwargs, code in cases:
        with pytest.raises(MechInterpError) as excinfo:
            mi.mlp_neuron_contributions(trace, **kwargs)
        assert excinfo.value.fields["code"] == code
        assert excinfo.value.fields["remedy"]
