"""Lane F22 neuro.datasets handoff contract (neuro MEMO 4.1, rows T7/T10/T14).

Pins:

- Default sweeps cover only STIMULUS-INDEXED saved sites through the core
  gate (buffer overwrites skipped with ONE summarized disclosure); explicit
  ineligible requests refuse actionably.
- Descriptors are the product: site, requested lookup, site_key, layer
  label, ALWAYS-present ``pool="flatten"``, source kind, versions,
  per-stimulus shape, dtype, casts.
- The presentation-order observation descriptor ``tl_presentation_index``
  is ALWAYS written (D10) alongside stimulus identity provenance.
- The stimulus-id authority rule (memo 4.1): recorded artifact ids are
  authoritative and an explicit list acts as VALIDATION; explicit ids on a
  bare Trace need exact length; positional identity is disclosed synthetic.
- obs= columns are length-checked BEFORE rsatoolbox receives them (the
  panel's highest-severity failure class), and the reserved presentation
  key refuses.
- Dtype rule (D12/T10): f16/bf16 widen to f32 (never f64) with the cast
  recorded; the widened RDM path matches a float32-native handoff.
- Channel descriptors use a neutral ``feature_index`` (never "neuroid")
  plus factual unravelled coordinates.
- Ledger mechanics (D6/T14): the returned mapping ==-equals the plain
  dict, survives pickle/copy with its ledger, and one call emits exactly
  ONE skip warning regardless of skip count.
"""

from __future__ import annotations

import copy
import pickle
import warnings
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors._base import TorchLensWarning
from torchlens.neuro._handoff import NeuroHandoffError

pytest.importorskip("rsatoolbox")

from torchlens.neuro._datasets import datasets  # noqa: E402


class _BNModel(nn.Module):
    """Toy model whose BatchNorm mints buffer sites on save-everything."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.bn(self.conv(x)))
        return self.fc(hidden.mean(dim=(2, 3)))


def _full_trace(n_stimuli: int = 6) -> tl.Trace:
    """Save-everything trace with buffer sites present."""

    torch.manual_seed(0)
    model = _BNModel().eval()
    x = torch.randn(n_stimuli, 3, 8, 8)
    return tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))


def _extraction_dir(tmp_path: Path, n_stimuli: int = 6) -> Path:
    """Write a real extraction artifact with recorded stimulus ids."""

    torch.manual_seed(0)
    model = _BNModel().eval()
    stimuli = [torch.randn(3, 8, 8) for _ in range(n_stimuli)]
    out_dir = tmp_path / "artifact"
    # NOTE: the readout head is deliberately absent from the layer list --
    # the 'fc' lookup ambiguity ("Layer not found" masking the ambiguous-op
    # error) is the extract lane's named fix (neuro memo section 9) and has
    # not landed on this base.
    tl.extract_dataset(
        model,
        stimuli,
        ["conv", "bn"],
        batch_size=3,
        output_dir=out_dir,
        stimulus_ids=[f"img_{index:02d}" for index in range(n_stimuli)],
        progress=False,
    )
    return out_dir


@pytest.mark.heavy
def test_default_sweep_skips_buffers_with_one_warning() -> None:
    """T7/T14: buffers never enter the mapping; ONE summarized warning."""

    log = _full_trace()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = datasets(log)
    skip_warnings = [
        w for w in caught if isinstance(w.message, TorchLensWarning) and "skipped" in str(w.message)
    ]
    assert len(skip_warnings) == 1
    assert all("buffer" not in key for key in result)
    skipped = [row for row in result.ledger if row.outcome == "skipped"]
    assert skipped and all(row.reason for row in skipped)
    assert {row.outcome for row in result.ledger} == {"computed", "skipped"}
    assert list(result) == [row.site for row in result.ledger if row.outcome == "computed"]


@pytest.mark.heavy
def test_explicit_ineligible_site_refuses_actionably() -> None:
    """D5: an explicit buffer request raises the evidence-naming error."""

    log = _full_trace()
    with pytest.raises(ValueError, match="buffer overwrite"):
        datasets(log, "buffer_1")


@pytest.mark.heavy
def test_descriptor_contract_and_presentation_index() -> None:
    """4.1: identity story present; tl_presentation_index always written."""

    log = _full_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = datasets(log, ["relu_1_3"], stimulus_ids=[f"s{index}" for index in range(6)])
    dataset = result["relu_1_3"]
    descriptors = dataset.descriptors
    assert descriptors["site"] == "relu_1_3"
    assert descriptors["requested"] == "relu_1_3"
    assert descriptors["pool"] == "flatten"
    assert descriptors["source_kind"] == "trace"
    assert descriptors["source_dtype"] == "float32"
    assert descriptors["model"] == "_BNModel"
    assert "site_key" in descriptors and "tl_version" in descriptors
    assert descriptors["rsatoolbox_version"] not in ("", None)
    obs = dataset.obs_descriptors
    assert list(obs["tl_presentation_index"]) == list(range(6))
    assert list(obs["stimulus_id"]) == [f"s{index}" for index in range(6)]
    assert obs["tl_stimulus_identity"][0] == "user_supplied"
    channels = dataset.channel_descriptors
    assert "feature_index" in channels
    assert "neuroid" not in channels
    assert "unit_coord_0" in channels


@pytest.mark.heavy
def test_obs_validation_refuses_before_rsatoolbox() -> None:
    """Mis-lengthed obs columns and the reserved key refuse typed."""

    log = _full_trace()
    with pytest.raises(NeuroHandoffError) as excinfo:
        datasets(log, ["relu_1_3"], obs={"condition": ["a", "b"]})
    assert excinfo.value.fields["code"] == "neuro_obs_descriptor_invalid"
    with pytest.raises(NeuroHandoffError) as excinfo:
        datasets(log, ["relu_1_3"], obs={"tl_presentation_index": list(range(6))})
    assert excinfo.value.fields["code"] == "neuro_obs_descriptor_invalid"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = datasets(log, ["relu_1_3"], obs={"condition": list("aabbcc")})
    assert list(result["relu_1_3"].obs_descriptors["condition"]) == list("aabbcc")


@pytest.mark.heavy
def test_stimulus_id_length_and_synthetic_disclosure() -> None:
    """Explicit ids validate length; absent ids disclose synthetic identity."""

    log = _full_trace()
    with pytest.raises(NeuroHandoffError) as excinfo:
        datasets(log, ["relu_1_3"], stimulus_ids=["only", "two"])
    assert excinfo.value.fields["code"] == "neuro_stimulus_ids_mismatch"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = datasets(log, ["relu_1_3"])
    obs = result["relu_1_3"].obs_descriptors
    assert obs["tl_stimulus_identity"][0] == "synthetic_positional"


@pytest.mark.heavy
def test_extraction_route_recorded_ids_are_authoritative(tmp_path: Path) -> None:
    """Memo 4.1: recorded ids serve, and explicit ids act as validation."""

    artifact = _extraction_dir(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = datasets(artifact)
    assert set(result) == {"conv", "bn"}
    dataset = result["conv"]
    assert list(dataset.obs_descriptors["stimulus_id"]) == [
        f"img_{index:02d}" for index in range(6)
    ]
    assert dataset.obs_descriptors["tl_stimulus_identity"][0] == "recorded"
    assert dataset.descriptors["source_kind"] == "extraction"
    with pytest.raises(NeuroHandoffError) as excinfo:
        datasets(artifact, ["conv"], stimulus_ids=[f"wrong_{index}" for index in range(6)])
    assert excinfo.value.fields["code"] == "neuro_stimulus_ids_mismatch"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        validated = datasets(
            artifact, ["conv"], stimulus_ids=[f"img_{index:02d}" for index in range(6)]
        )
    assert validated["conv"].obs_descriptors["tl_stimulus_identity"][0] == "user_validated"


@pytest.mark.heavy
def test_pool_vocabulary_refuses_non_none() -> None:
    """D3: neuro mints no pooling vocabulary; non-None refuses typed."""

    log = _full_trace()
    with pytest.raises(NeuroHandoffError) as excinfo:
        datasets(log, ["relu_1_3"], pool="cls")
    assert excinfo.value.fields["code"] == "neuro_pool_vocabulary_unavailable"


@pytest.mark.heavy
def test_bf16_widen_to_f32_with_recorded_cast() -> None:
    """D12/T10: bf16 reaches rsatoolbox as float32 with the cast recorded."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).to(torch.bfloat16).eval()
    x = torch.randn(5, 3, dtype=torch.bfloat16)
    log = tl.trace(model, x, capture=tl.options.CaptureOptions(layers_to_save="all"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = datasets(log, ["relu_1_2"])
    dataset = result["relu_1_2"]
    assert dataset.measurements.dtype.name == "float32"
    assert dataset.descriptors["handoff_cast"] == "bfloat16->float32"
    assert dataset.descriptors["source_dtype"] == "bfloat16"

    float_model = nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()
    float_model.load_state_dict({key: value.float() for key, value in model.state_dict().items()})
    float_log = tl.trace(
        float_model, x.float(), capture=tl.options.CaptureOptions(layers_to_save="all")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        native = datasets(float_log, ["relu_1_2"])
    import numpy as np

    assert np.allclose(dataset.measurements, native["relu_1_2"].measurements, atol=2e-2)


@pytest.mark.heavy
def test_ledger_mechanics_pickle_copy_equality() -> None:
    """T14: ==-equal to the plain dict; ledger survives pickle and copy."""

    log = _full_trace()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = datasets(log)
    plain = dict(result)
    assert result == plain
    restored = pickle.loads(pickle.dumps(result))
    assert restored.ledger == result.ledger
    assert list(restored) == list(result)
    for clone in (copy.copy(result), copy.deepcopy(result)):
        assert clone.ledger == result.ledger


def test_unsupported_source_refuses_typed() -> None:
    """Unsupported source types refuse with neuro_source_invalid."""

    with pytest.raises(NeuroHandoffError) as excinfo:
        datasets(42)
    assert excinfo.value.fields["code"] == "neuro_source_invalid"


@pytest.mark.heavy
def test_multipass_bare_lookup_refuses_with_alternatives() -> None:
    """D13: a bare multi-pass layer lookup names pass-qualified spellings."""

    class Recurrent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for _ in range(3):
                x = torch.tanh(self.fc(x))
            return x

    torch.manual_seed(0)
    log = tl.trace(
        Recurrent().eval(),
        torch.randn(5, 4),
        capture=tl.options.CaptureOptions(layers_to_save="all"),
    )
    with pytest.raises(NeuroHandoffError) as excinfo:
        datasets(log, "linear_1_1")
    assert excinfo.value.fields["code"] == "neuro_site_ambiguous"
    alternatives = excinfo.value.fields["alternatives"]
    assert any(":" in alternative for alternative in alternatives)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        swept = datasets(log)
    assert any(key.endswith(":1") for key in swept)
    assert any(key.endswith(":3") for key in swept)
