"""Hub push and pull through the safe ``.tlspec`` path (walk U14.2).

``push_to_hub`` uploads a gzipped tar of the portable bundle and never a raw
pickle; ``from_hub`` downloads it, checks every tar member, and loads it
through ``torchlens.load``'s default-deny loader. A local fake Hub stands in
for the network: no request leaves the process and no credential is used.
"""

from __future__ import annotations

import ast
import inspect
import io
import tarfile
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import ArgumentTypeError
from torchlens._io import TorchLensIOError
from torchlens.bridge import huggingface_hub


class _LocalHub:
    """``HfApi``-shaped fake: uploads land in a dict, downloads come from disk."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[tuple[str, str], bytes] = {}
        self.downloads: list[dict[str, Any]] = []

    def create_repo(self, *, repo_id: str, private: bool | None, exist_ok: bool) -> None:
        """Accept any repo creation."""

    def upload_file(
        self, *, path_or_fileobj: str, path_in_repo: str, repo_id: str, commit_message: str
    ) -> str:
        """Store the uploaded bytes under ``(repo_id, path_in_repo)``."""

        self.files[(repo_id, path_in_repo)] = Path(path_or_fileobj).read_bytes()
        return f"local://{repo_id}/{path_in_repo}"

    def hf_hub_download(self, *, repo_id: str, filename: str, revision: str | None) -> str:
        """Write the stored bytes to a local cache file and return its path."""

        self.downloads.append({"repo_id": repo_id, "filename": filename, "revision": revision})
        target = self.root / repo_id.replace("/", "--") / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.files[(repo_id, filename)])
        return str(target)


class _Tiny(nn.Module):
    """Two-layer MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the MLP."""

        return self.fc2(torch.relu(self.fc1(x)))


@pytest.fixture
def trace() -> Any:
    """A fully saved trace of the tiny MLP."""

    torch.manual_seed(0)
    return tl.trace(
        _Tiny(), torch.randn(2, 4), capture=tl.options.CaptureOptions(layers_to_save="all")
    )


def test_from_hub_round_trips_a_trace_through_the_safe_loader(trace: Any, tmp_path: Path) -> None:
    hub = _LocalHub(tmp_path / "cache")
    pushed = huggingface_hub.push_to_hub(trace, "me/demo", api=hub)
    assert pushed["format"] == "tar.gz"
    assert pushed["path_in_repo"] == "torchlens_artifact.tar.gz"

    loaded = huggingface_hub.from_hub("me/demo", api=hub, revision="main")

    assert hub.downloads == [
        {"repo_id": "me/demo", "filename": "torchlens_artifact.tar.gz", "revision": "main"}
    ]
    assert isinstance(loaded, type(trace))
    assert [op.layer_label for op in loaded.layer_list] == [
        op.layer_label for op in trace.layer_list
    ]
    for original, restored in zip(trace.layer_list, loaded.layer_list, strict=True):
        assert torch.equal(restored.out, original.out)


def test_push_writes_no_pickle(trace: Any, tmp_path: Path) -> None:
    hub = _LocalHub(tmp_path / "cache")
    huggingface_hub.push_to_hub(trace, "me/demo", api=hub)
    payload = hub.files[("me/demo", "torchlens_artifact.tar.gz")]

    # The upload is a gzipped tar of the .tlspec bundle, never a pickle stream.
    assert payload[:2] == b"\x1f\x8b"
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        names = tar.getnames()
    assert {name.split("/")[0] for name in names} == {"bundle"}
    assert "bundle/manifest.json" in names
    # The module has no pickle path left at all.
    tree = ast.parse(inspect.getsource(huggingface_hub))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "pickle" not in imported
    # A plain object is refused before anything is uploaded.
    with pytest.raises(ArgumentTypeError) as info:
        huggingface_hub.push_to_hub({"weights": [1, 2]}, "me/other", api=hub)
    assert info.value.fields["code"] == "hub_push_artifact_unsupported"
    assert ("me/other", "torchlens_artifact.tar.gz") not in hub.files


def _tar_bytes(add: Any) -> bytes:
    """Build a gzipped tar by calling ``add(tar)``."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        add(tar)
    return buffer.getvalue()


def _file(tar: tarfile.TarFile, name: str, data: bytes = b"{}") -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _symlink(tar: tarfile.TarFile) -> None:
    info = tarfile.TarInfo("bundle/manifest.json")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"
    tar.addfile(info)


@pytest.mark.parametrize(
    ("payload", "problem"),
    [
        (b"not a tar at all", "not a readable gzipped tar"),
        (_tar_bytes(lambda tar: _file(tar, "bundle/../../escape.json")), "escapes the bundle"),
        (_tar_bytes(lambda tar: _file(tar, "/abs/manifest.json")), "escapes the bundle"),
        (_tar_bytes(_symlink), "link or special file"),
        (
            _tar_bytes(lambda tar: (_file(tar, "a/manifest.json"), _file(tar, "b/x.json"))),
            "expected one top-level bundle",
        ),
    ],
    ids=["not-gzip", "parent-escape", "absolute", "symlink", "two-roots"],
)
def test_from_hub_refuses_an_unsafe_download(payload: bytes, problem: str, tmp_path: Path) -> None:
    hub = _LocalHub(tmp_path / "cache")
    hub.files[("me/bad", "torchlens_artifact.tar.gz")] = payload
    with pytest.raises(TorchLensIOError, match=problem) as info:
        huggingface_hub.from_hub("me/bad", api=hub)
    assert info.value.fields["code"] == "hub_artifact_unsafe"
    assert not (tmp_path / "escape.json").exists()
