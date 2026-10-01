"""W051-EXTRACT: resume identity defects (audit 2.10a / 2.10c / 2.10d).

(a) The D6 identity record measures ``state_dict()`` only, so same weights +
    different hyperparameters resumed to a MIXED ``status=complete`` artifact;
    the signature now carries a structural record compared on resume.
(c) The callable-identity fold repr'd ``co_consts``; a nested code object reprs
    as an ADDRESS, so every resume with a genexpr/nested-lambda/comprehension
    transform refused ``extraction_resume_callable_mismatch``.
(d) bf16 stimuli crashed untyped through ``tensor.numpy()``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

from torchlens._extraction.callable_identity import classify_callable
from torchlens._extraction.dtype_policy import tensor_payload_bytes
from torchlens._extraction.reader import open_extraction
from torchlens._extraction.resume import model_structure_record
from torchlens.dataset_extraction import DatasetExtractionResumeError, extract_dataset
from torchlens.utils._torch_compat import (
    get_cpu_float8_deterministic_fill_support,
    get_cpu_half_kernels_support,
)

pytestmark = pytest.mark.smoke


class _Net(nn.Module):
    def __init__(self, padding_mode: str = "zeros", act: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, padding_mode=padding_mode)
        self.act = act()
        self.conv2 = nn.Conv2d(4, 5, 3, padding=1)
        self.fc = nn.Linear(5 * 8 * 8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv2(self.act(self.conv1(x))).flatten(1))


class _Interrupt(Exception):
    pass


def _dying(x: torch.Tensor, stop_at: int):  # type: ignore[no-untyped-def]
    for index in range(x.shape[0]):
        if index == stop_at:
            raise _Interrupt()
        yield x[index]


def _same_weights_variants() -> tuple[nn.Module, nn.Module, nn.Module]:
    torch.manual_seed(0)
    a = _Net().eval()
    b = _Net(padding_mode="reflect").eval()
    b.load_state_dict(a.state_dict())
    c = _Net(act=nn.GELU).eval()
    c.load_state_dict(a.state_dict())
    return a, b, c


def test_structure_record_separates_hyperparameter_twins() -> None:
    a, b, c = _same_weights_variants()
    ra, rb, rc = (model_structure_record(m) for m in (a, b, c))
    assert ra["algorithm_id"] == "tl_model_structure_v1"
    assert ra["n_modules"] == 5
    assert ra["digest"] != rb["digest"], "padding_mode lives in extra_repr"
    assert ra["digest"] != rc["digest"], "activation class lives in the qualname"
    assert model_structure_record(_Net().eval())["digest"] == ra["digest"], "weights never enter"


@pytest.mark.parametrize("variant", ["reflect_padding", "gelu_activation"])
def test_resume_refuses_same_weights_different_architecture(tmp_path: Path, variant: str) -> None:
    """p3_identity: the mixed-artifact hazard refuses typed under the identity code."""

    a, b, c = _same_weights_variants()
    other = b if variant == "reflect_padding" else c
    x = torch.randn(20, 3, 8, 8)
    layers = {"c1": "conv1", "c2": "conv2"}
    with pytest.raises(_Interrupt):
        extract_dataset(a, _dying(x, 10), layers, batch_size=5, output_dir=tmp_path, progress=False)
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        extract_dataset(
            other, list(x), layers, batch_size=5, output_dir=tmp_path, progress=False, resume=True
        )
    fields = excinfo.value.fields
    assert fields["code"] == "extraction_resume_model_identity_mismatch"
    assert fields["mismatched_fields"] == ["model_structure"]
    assert fields["recorded_structure"]["digest"] != fields["requested_structure"]["digest"]
    # The matching construction still resumes, row-exact against the interrupted prefix.
    extract_dataset(
        a, list(x), layers, batch_size=5, output_dir=tmp_path, progress=False, resume=True
    )
    got = open_extraction(tmp_path).materialize()
    with torch.no_grad():
        assert torch.equal(got["c1"], a.conv1(x))


def test_resume_of_a_manifest_without_structure_record_discloses(tmp_path: Path) -> None:
    """Older artifacts (no record) resume with an audit row, never a forged comparison."""

    a, _b, _c = _same_weights_variants()
    x = torch.randn(10, 3, 8, 8)
    with pytest.raises(_Interrupt):
        extract_dataset(
            a, _dying(x, 5), {"c1": "conv1"}, batch_size=5, output_dir=tmp_path, progress=False
        )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["signature"]["model_structure"]
    manifest_path.write_text(json.dumps(manifest))
    extract_dataset(
        a, list(x), {"c1": "conv1"}, batch_size=5, output_dir=tmp_path, progress=False, resume=True
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "complete"
    kinds = [row["kind"] for row in manifest["run"]["resume_audit"]]
    assert "model_structure_unrecorded" in kinds


_NESTED_SOURCES = {
    "genexpr": (
        "def fn(t):\n"
        "    return torch.stack([torch.sum(t[i]) for i in range(t.shape[0])]) + sum(x for x in [1])\n"
    ),
    "nested_lambda": "def fn(t):\n    f = lambda z: z * 2\n    return f(t)\n",
    "listcomp": "def fn(t):\n    return torch.stack([row.mean(dim=-1) for row in t.unbind(0)])\n",
}


def _compile(source: str):  # type: ignore[no-untyped-def]
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(source, "<w051>", "exec"), namespace)  # noqa: S102 - test-local source
    return namespace["fn"]


@pytest.mark.parametrize("name", sorted(_NESTED_SOURCES))
def test_callable_digest_is_stable_across_distinct_code_objects(name: str) -> None:
    """Two compilations = two nested code objects at two addresses = ONE digest."""

    first, second = _compile(_NESTED_SOURCES[name]), _compile(_NESTED_SOURCES[name])
    assert first.__code__ is not second.__code__
    r1, r2 = classify_callable(first), classify_callable(second)
    assert r1["classification"] == "complete"
    assert r1["digest"] == r2["digest"]


def test_callable_digest_folds_nested_code_structurally() -> None:
    """The nested body is folded, not ignored: changing it changes the digest."""

    base = classify_callable(_compile("def fn(t):\n    f = lambda z: z * 2\n    return f(t)\n"))
    edited = classify_callable(_compile("def fn(t):\n    f = lambda z: z * 3\n    return f(t)\n"))
    assert base["digest"] != edited["digest"]


def test_resume_with_a_comprehension_transform_completes(tmp_path: Path) -> None:
    """End to end (p4c): the once-refused ordinary transform now resumes."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    x = torch.arange(60, dtype=torch.float32).reshape(20, 3)
    transform = _compile(_NESTED_SOURCES["listcomp"])
    with pytest.raises(_Interrupt):
        extract_dataset(
            model,
            _dying(x, 10),
            {"h": "relu_1_2"},
            batch_size=5,
            output_dir=tmp_path,
            progress=False,
            transform=transform,
        )
    # A fresh compilation stands in for the second process.
    extract_dataset(
        model,
        list(x),
        {"h": "relu_1_2"},
        batch_size=5,
        output_dir=tmp_path,
        progress=False,
        transform=_compile(_NESTED_SOURCES["listcomp"]),
        resume=True,
    )
    assert json.loads((tmp_path / "manifest.json").read_text())["status"] == "complete"
    assert open_extraction(tmp_path).materialize()["h"].shape == (20,)


def test_v1_callable_record_refuses_as_incomparable_not_as_behavior_change(
    tmp_path: Path,
) -> None:
    """An artifact recorded under encoding v1 discloses the encoding change on resume."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).eval()
    x = torch.arange(60, dtype=torch.float32).reshape(20, 3)
    transform = _compile(_NESTED_SOURCES["listcomp"])
    with pytest.raises(_Interrupt):
        extract_dataset(
            model,
            _dying(x, 10),
            {"h": "relu_1_2"},
            batch_size=5,
            output_dir=tmp_path,
            progress=False,
            transform=transform,
        )
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    slots = manifest["signature"]["callable_identity"]["slots"]
    assert list(slots) == ["transform:h:0"]
    slot = slots["transform:h:0"]
    assert slot["algorithm_version"] == 2
    slot["algorithm_version"] = 1
    slot["digest"] = "blake2b:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(DatasetExtractionResumeError) as excinfo:
        extract_dataset(
            model,
            list(x),
            {"h": "relu_1_2"},
            batch_size=5,
            output_dir=tmp_path,
            progress=False,
            transform=transform,
            resume=True,
        )
    fields = excinfo.value.fields
    assert fields["code"] == "extraction_resume_callable_mismatch"
    assert fields["encoding_changed"] is True
    assert (fields["recorded_algorithm_version"], fields["current_algorithm_version"]) == (1, 2)
    assert "INCOMPARABLE" in str(excinfo.value)
    assert "behavior changed" not in str(excinfo.value)


@pytest.mark.skipif(
    not get_cpu_float8_deterministic_fill_support(),
    reason="CPU Float8 empty-fill under deterministic mode postdates the torch 2.1 floor",
)
def test_tensor_payload_bytes_covers_numpy_less_dtypes() -> None:
    bf16 = torch.tensor([1.0, -2.0, 3.5], dtype=torch.bfloat16)
    assert tensor_payload_bytes(bf16) == bf16.view(torch.int16).numpy().tobytes()
    assert tensor_payload_bytes(torch.empty(0, dtype=torch.bfloat16)) == b""
    assert len(tensor_payload_bytes(torch.tensor(2.0, dtype=torch.bfloat16))) == 2
    fp8 = torch.tensor([1.0, 0.5], dtype=torch.float8_e4m3fn)
    assert len(tensor_payload_bytes(fp8)) == 2
    with pytest.raises(TypeError):
        bf16.numpy()  # the historical crash site


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
        pytest.param(
            torch.float16,
            marks=pytest.mark.skipif(
                not get_cpu_half_kernels_support(),
                reason="CPU addmm for float16 postdates the torch 2.1 floor",
            ),
        ),
    ],
)
def test_low_precision_stimuli_extract_on_both_paths(tmp_path: Path, dtype: torch.dtype) -> None:
    """p9b_bf16: tensor stimuli, item stimuli, disk and in-memory, plus a bf16 closure."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).to(dtype).eval()
    x = torch.randn(7, 3).to(dtype)
    disk = extract_dataset(
        model, x, {"h": "relu_1_2"}, batch_size=3, output_dir=tmp_path / "t", progress=False
    )
    assert len(disk) == 3
    extract_dataset(
        model,
        [x[i] for i in range(7)],
        {"h": "relu_1_2"},
        batch_size=3,
        output_dir=tmp_path / "i",
        progress=False,
    )
    memory = extract_dataset(model, x, {"h": "relu_1_2"}, batch_size=3, progress=False)
    assert memory["h"].shape == (7, 4) and memory["h"].dtype == dtype
    scale = torch.tensor([2.0], dtype=dtype)
    assert classify_callable(lambda t: t * scale)["classification"] == "complete"
