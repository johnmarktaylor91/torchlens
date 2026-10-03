"""Extraction v2 typed-refusal provocations (lane F18; extract memo D18).

Every new ``extraction_*`` refusal code lands with a provoking assertion or
the repo's error-contract coverage gate goes red. This file holds the
CHEAP, capture-free provocations: closed-vocabulary knobs, unit-level
geometry refusals, and the in-memory false-affordance gates. Engine-loop
refusals (ragged drift, selector attestation, resume) ride their feature
files.
"""

from __future__ import annotations

import pytest
import torch
from support.fp8_guard import permit_cpu_float8_allocation
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._extraction import (
    apply_pool,
    cast_for_store,
    classify_callable,
    register_pure_module,
    to_padded,
    trim_batch,
)
from torchlens._extraction.ragged import RaggedBatch
from torchlens.dataset_extraction import extract_dataset, relabel_extraction

pytestmark = pytest.mark.smoke


def _model() -> nn.Module:
    """Return a tiny two-layer model for knob-validation calls."""

    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(3, 4), nn.ReLU()).eval()


def _code(excinfo: pytest.ExceptionInfo) -> str:
    """Return the typed refusal code off one raised exception."""

    return excinfo.value.fields["code"]


# --- closed-vocabulary knobs ---------------------------------------------------


def test_ragged_vocabulary_refuses_bools_with_teaching() -> None:
    """``ragged=`` is an enum, not a bool (D4): True refuses typed."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), ["relu"], ragged=True, progress=False)
    assert _code(excinfo) == "extraction_ragged_invalid"
    assert "enum, not a bool" in str(excinfo.value)


def test_as_captured_is_reserved_and_refuses() -> None:
    """``ragged='as_captured'`` is the reserved never-default mode (item 19)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), ["relu"], ragged="as_captured", progress=False)
    assert _code(excinfo) == "extraction_as_captured_unavailable"


def test_checksums_vocabulary_refuses() -> None:
    """``checksums=`` outside fast/crypto/none refuses typed (D7)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _model(),
            torch.randn(4, 3),
            ["relu"],
            output_dir="/tmp/never-created-f18",
            checksums="paranoid",
            progress=False,
        )
    assert _code(excinfo) == "extraction_checksums_invalid"


def test_shard_format_vocabulary_refuses(tmp_path) -> None:
    """``shard_format=`` outside safetensors/pt refuses typed (item 11)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _model(),
            torch.randn(4, 3),
            ["relu"],
            output_dir=tmp_path,
            shard_format="npz",
            progress=False,
        )
    assert _code(excinfo) == "extraction_shard_format_invalid"


def test_disk_only_options_refuse_in_memory_mode() -> None:
    """Disk-only knobs without output_dir are false affordances (item 4)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), ["relu"], pipeline_id="exp-1", progress=False)
    assert _code(excinfo) == "extraction_disk_only_option"
    assert "pipeline_id" in str(excinfo.value)
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), ["relu"], ragged="trim", progress=False)
    assert _code(excinfo) == "extraction_disk_only_option"


# --- user output keys (item 4) --------------------------------------------------


@pytest.mark.parametrize(
    "bad_key",
    ["", "a/b", "a\\b", "..", "x" * 300, "nul\x00byte"],
    ids=["empty", "slash", "backslash", "dotdot", "overlong", "nul"],
)
def test_output_key_validation_refuses_path_capable_keys(bad_key: str) -> None:
    """Path-capable or unbounded mapping keys refuse AT CALL TIME."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), {bad_key: "relu"}, progress=False)
    assert _code(excinfo) == "extraction_output_key_invalid"


def test_output_key_validation_refuses_non_strings() -> None:
    """Non-string mapping keys refuse typed."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), {7: "relu"}, progress=False)
    assert _code(excinfo) == "extraction_output_key_invalid"


def test_sanitize_key_is_injective_across_a_label_corpus() -> None:
    """T-KEYS: sanitization round-trips a hostile label corpus collision-free."""

    from torchlens._extraction import sanitize_key

    corpus = [f"layer_{i}" for i in range(400)] + [
        "relu:1",
        "relu:2",
        "conv%41",
        "conv%3A1",
        "attn.q",
        "attn q",
        "ünïcode",
        "conv[0]",
        "conv(0)",
    ]
    sanitized = [sanitize_key(key) for key in corpus]
    assert len(set(sanitized)) == len(set(corpus)), "sanitization collided"
    for stem in sanitized:
        assert "/" not in stem and "\x00" not in stem and ".." not in stem


# --- collate (item 5) ------------------------------------------------------------


def test_collate_non_callable_refuses() -> None:
    """A non-callable collate= refuses typed."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), [torch.randn(3)] * 4, ["relu"], collate=42, progress=False)
    assert _code(excinfo) == "extraction_collate_invalid"


def test_collate_bare_tuple_refuses_as_ambiguous() -> None:
    """A USER collate returning a bare tuple refuses naming the envelope."""

    def bad_collate(items):
        """Return an ambiguous bare tuple."""

        return (torch.stack(items),)

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _model(), [torch.randn(3)] * 4, ["relu"], collate=bad_collate, progress=False
        )
    assert _code(excinfo) == "extraction_collate_ambiguous"
    assert "BatchEnvelope" in str(excinfo.value)


def test_tokenizer_unresolvable_refuses_with_all_three_doors_named() -> None:
    """Text stimuli with no resolvable tokenizer refuse typed (D9)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), ["a", "b"], ["relu"], progress=False)
    assert _code(excinfo) == "extraction_tokenizer_unresolvable"
    assert "hf_collate" in str(excinfo.value)


class _PadlessTokenizer:
    """A GPT-2-shaped tokenizer stub: no pad token configured."""

    pad_token = None
    eos_token = "<eos>"
    padding_side = "right"
    name_or_path = "stub-tokenizer"
    vocab_size = 10

    def __call__(self, texts, **kwargs):
        """Refuse like HF does when padding without a pad token."""

        raise AssertionError("tokenization must not be reached without a pad token")


def test_missing_pad_token_refuses_with_eos_opt_in_named() -> None:
    """D9: a missing pad token refuses; EOS-as-pad is the explicit opt-in."""

    from torchlens.dataset_extraction import hf_collate

    collate = hf_collate(_PadlessTokenizer())
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), ["a", "b", "c"], ["relu"], collate=collate, progress=False)
    assert _code(excinfo) == "extraction_pad_token_missing"
    assert "pad_token='eos'" in str(excinfo.value)


def test_pad_token_vocabulary_refuses_unknown_opt_in() -> None:
    """The only pad-token opt-in is 'eos'; anything else refuses."""

    from torchlens.dataset_extraction import hf_collate

    collate = hf_collate(_PadlessTokenizer(), pad_token="bos")
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), ["a", "b", "c"], ["relu"], collate=collate, progress=False)
    assert _code(excinfo) == "extraction_pad_token_missing"


# --- pool (item 8) ----------------------------------------------------------------


def test_pool_vocabulary_refusals() -> None:
    """``pool=`` outside the preset vocabulary refuses typed (D10)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), ["relu"], pool="nope", progress=False)
    assert _code(excinfo) == "extraction_pool_invalid"
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _model(),
            torch.randn(4, 3),
            ["relu"],
            pool={"preset": "token_mean", "bogus": 1},
            progress=False,
        )
    assert _code(excinfo) == "extraction_pool_invalid"
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), ["relu"], pool=lambda t: t, progress=False)
    assert _code(excinfo) == "extraction_pool_invalid"


def test_pool_axis_ambiguity_refuses() -> None:
    """spatial_* on a 2D site refuses instead of guessing an axis (D10)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), ["relu"], pool="spatial_mean", progress=False)
    assert _code(excinfo) == "extraction_pool_axes_ambiguous"


def test_pool_mask_required_refuses_without_mask() -> None:
    """Mask-aware token pooling without a mask refuses (measured 20.66%)."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(5, 5), nn.ReLU()).eval()
    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(model, torch.randn(4, 7, 5), ["relu"], pool="token_mean", progress=False)
    assert _code(excinfo) == "extraction_pool_mask_required"
    assert "20.66%" in str(excinfo.value)


def test_pool_fp8_compute_refuses_unit() -> None:
    """fp8 pool compute refuses typed rather than silently upcasting (D11)."""

    with permit_cpu_float8_allocation():
        tensor = torch.randn(2, 3, 4).to(torch.float8_e4m3fn)
    with pytest.raises(InvalidArgumentError) as excinfo:
        apply_pool("k", tensor, {"global": {"preset": "token_mean", "unmasked": True}}, None)
    assert _code(excinfo) == "extraction_pool_dtype_unsupported"


def test_pool_last_token_never_runs_unmasked() -> None:
    """'last' means last VALID token; the unmasked override cannot define it."""

    tensor = torch.randn(2, 3, 4)
    with pytest.raises(InvalidArgumentError) as excinfo:
        apply_pool("k", tensor, {"global": {"preset": "last_token", "unmasked": True}}, None)
    assert _code(excinfo) == "extraction_pool_mask_required"


# --- dtype (item 9) -----------------------------------------------------------------


def test_dtype_vocabulary_refuses() -> None:
    """``dtype=`` outside torch.dtype/name spellings refuses typed (D11)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), ["relu"], dtype="floof", progress=False)
    assert _code(excinfo) == "extraction_dtype_invalid"


def test_dtype_unstorable_refuses_unit() -> None:
    """A cast the safetensors shard codec cannot store refuses typed."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        cast_for_store("k", torch.randn(2, 3), {"global": torch.complex128}, "safetensors")
    assert _code(excinfo) == "extraction_dtype_unsupported"


# --- ragged carriers (item 10) --------------------------------------------------------


def test_noncontiguous_mask_refuses_trimming_unit() -> None:
    """T-TRIM: a hand-built interior-gap mask refuses (start, extent) slicing."""

    from torchlens._extraction import mask_row_geometry

    mask = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 0]])
    geometry = mask_row_geometry(mask)
    assert geometry["contiguous"] is False
    with pytest.raises(InvalidArgumentError) as excinfo:
        trim_batch("k", torch.randn(2, 4, 3), geometry)
    assert _code(excinfo) == "extraction_ragged_mask_noncontiguous"


def test_trim_geometry_mismatch_refuses_unit() -> None:
    """A tensor whose axis 1 cannot carry the mask geometry refuses."""

    from torchlens._extraction import mask_row_geometry

    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    with pytest.raises(InvalidArgumentError) as excinfo:
        trim_batch("k", torch.randn(2, 2, 3), mask_row_geometry(mask))
    assert _code(excinfo) == "extraction_ragged_geometry_mismatch"


def test_to_padded_max_len_shorter_than_a_row_refuses() -> None:
    """to_padded never truncates: a short max_len refuses typed (D4)."""

    carrier = RaggedBatch(
        values=torch.randn(5, 2),
        offsets=torch.tensor([0, 3, 5]),
        row_shapes=((3, 2), (2, 2)),
    )
    with pytest.raises(InvalidArgumentError) as excinfo:
        to_padded([carrier], max_len=2)
    assert _code(excinfo) == "extraction_ragged_geometry_mismatch"


# --- callable identity (item 15) -----------------------------------------------------


def test_register_pure_module_refusals() -> None:
    """register_pure_module refuses dotted names and uninstallable modules."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        register_pure_module("torch.nn")
    assert _code(excinfo) == "extraction_pure_module_invalid"
    with pytest.raises(InvalidArgumentError) as excinfo:
        register_pure_module("definitely_not_installed_f18_module")
    assert _code(excinfo) == "extraction_pure_module_invalid"


def test_register_pure_module_flips_classification() -> None:
    """The registered escape flips a named partial into a measured complete."""

    import json as json_module

    from torchlens._extraction.callable_identity import _REGISTERED_PURE_MODULES

    def uses_json(tensor):
        """Reference a stdlib module OUTSIDE the launch allowlist."""

        return tensor * len(json_module.dumps({}))

    before = classify_callable(uses_json)
    assert before["classification"] == "partial"
    assert any("module(json" in ref for ref in before["opaque_references"])
    register_pure_module("json")
    try:
        after = classify_callable(uses_json)
        assert after["classification"] == "complete"
        assert "json" in after["pure_modules"]
    finally:
        _REGISTERED_PURE_MODULES.pop("json", None)


# --- relabel (item 7) ------------------------------------------------------------------


def test_relabel_refuses_incomplete_and_mislengthed(tmp_path) -> None:
    """relabel_extraction is complete-only and cardinality-checked (D2)."""

    model = _model()
    extract_dataset(
        model,
        torch.randn(4, 3),
        ["relu"],
        batch_size=2,
        output_dir=tmp_path,
        progress=False,
        stimulus_ids=["a", "b", "c", "d"],
    )
    with pytest.raises(InvalidArgumentError) as excinfo:
        relabel_extraction(tmp_path, ["x", "y"])
    assert _code(excinfo) == "extraction_relabel_invalid"
    import json

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["status"] = "in_progress"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(InvalidArgumentError) as excinfo:
        relabel_extraction(tmp_path, ["w", "x", "y", "z"])
    assert _code(excinfo) == "extraction_relabel_invalid"


def test_relabel_rewrites_sidecar_with_audit(tmp_path) -> None:
    """The audited verb relabels, records both digests, and changes identity."""

    import json

    model = _model()
    extract_dataset(
        model,
        torch.randn(4, 3),
        ["relu"],
        batch_size=2,
        output_dir=tmp_path,
        progress=False,
        stimulus_ids=["a", "b", "c", "d"],
    )
    old_digest = json.loads((tmp_path / "manifest.json").read_text())["signature"][
        "stimulus_ids_digest"
    ]
    audit = relabel_extraction(tmp_path, ["w", "x", "y", "z"])
    assert audit["prior_ids_digest"] == old_digest
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["signature"]["stimulus_ids_digest"] == audit["new_ids_digest"]
    assert manifest["relabel_audit"][0]["kind"] == "relabel"
    sidecar = json.loads((tmp_path / "stimulus_ids.json").read_text())
    assert sidecar["ids"] == ["w", "x", "y", "z"]


def test_stimulus_ids_reject_empty_and_non_strings() -> None:
    """Identifiers are non-empty strings; duplicates are legal (D2)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _model(),
            torch.randn(2, 3),
            ["relu"],
            output_dir="/tmp/never-created-f18-ids",
            stimulus_ids=["a", ""],
            progress=False,
        )
    assert _code(excinfo) == "extraction_stimulus_ids_invalid"


# --- selector doors (item 13) ------------------------------------------------------------


def test_selector_mixed_mapping_refuses() -> None:
    """A mapping mixing selectors and strings refuses (one door per request)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _model(),
            torch.randn(4, 3),
            {"a": tl.func("relu"), "b": "relu"},
            progress=False,
        )
    assert _code(excinfo) == "extraction_selector_mixed_unsupported"


def test_selector_partial_units_teaches_postprocess() -> None:
    """An element-level Selection refuses with the postprocess remedy (D12)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(_model(), torch.randn(4, 3), tl.units("relu_1_2", [(0, 0)]), progress=False)
    assert _code(excinfo) == "extraction_selector_partial_units"
    assert "transform" in str(excinfo.value)


def test_selector_transform_mapping_refuses() -> None:
    """Selector layers + per-site transform Mapping refuses (keys unknowable)."""

    with pytest.raises(InvalidArgumentError) as excinfo:
        extract_dataset(
            _model(),
            torch.randn(4, 3),
            tl.func("relu"),
            transform={"a": torch.abs},
            progress=False,
        )
    assert _code(excinfo) == "extraction_selector_transform_mapping_unsupported"
