"""``tl.bridge.huggingface.load_from_hub``: the load side of ``push_to_hub``.

Every test runs offline against an in-memory Hub double (and, for the default
path, a mock ``huggingface_hub`` module): a pushed bundle round-trips back to
a ``Trace`` through the governed ``tl.load`` door, a pushed raw pickle is
refused without being unpickled, and hostile tar members never reach disk.
"""

from __future__ import annotations

import io
import pickle
import sys
import tarfile
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import TorchLensIOError
from torchlens.bridge import huggingface


class _MemoryHub:
    """``HfApi`` double: uploads land in memory, downloads come back from it."""

    def __init__(self, download_dir: Path) -> None:
        """Remember where downloads are written."""

        self.download_dir = download_dir
        self.files: dict[tuple[str, str], bytes] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create_repo(self, **kwargs: Any) -> None:
        """Record repository creation."""

        self.calls.append(("create_repo", kwargs))

    def upload_file(self, **kwargs: Any) -> str:
        """Keep the uploaded bytes under ``(repo_id, path_in_repo)``."""

        self.files[(kwargs["repo_id"], kwargs["path_in_repo"])] = Path(
            kwargs["path_or_fileobj"]
        ).read_bytes()
        return "https://huggingface.co/" + kwargs["repo_id"]

    def hf_hub_download(self, **kwargs: Any) -> str:
        """Write the stored bytes to the download directory and return the path."""

        self.calls.append(("hf_hub_download", dict(kwargs)))
        payload = self.files[(kwargs["repo_id"], kwargs["filename"])]
        target = self.download_dir / kwargs["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return str(target)


def _trace() -> tl.Trace:
    """A seeded two-layer capture with saved activations."""

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)).eval()
    return tl.trace(model, torch.randn(2, 4))


def _tarball(members: dict[str, bytes], *, extra: tarfile.TarInfo | None = None) -> bytes:
    """Build a gzipped tar from ``name -> bytes`` plus an optional raw member."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra is not None:
            tar.addfile(extra)
    return buffer.getvalue()


def test_push_then_load_round_trips_a_trace(tmp_path: Path) -> None:
    """A pushed Trace comes back as a Trace with the same layers and outputs."""

    hub = _MemoryHub(tmp_path / "downloads")
    log = _trace()
    pushed = huggingface.push_to_hub(log, "someone/trace", api=hub)
    assert pushed["format"] == "tar.gz"

    loaded = huggingface.load_from_hub("someone/trace", api=hub, extract_dir=tmp_path / "bundle")
    assert isinstance(loaded, tl.Trace)
    assert [layer.layer_label for layer in loaded.layer_list] == [
        layer.layer_label for layer in log.layer_list
    ]
    label = loaded.layer_list[-1].layer_label
    assert torch.equal(loaded[label].out, log[label].out)
    # The bundle was extracted where asked, and only there.
    assert (tmp_path / "bundle" / "bundle" / "manifest.json").is_file()
    download_call = next(kwargs for name, kwargs in hub.calls if name == "hf_hub_download")
    assert download_call == {
        "repo_id": "someone/trace",
        "filename": "torchlens_artifact.tar.gz",
        "revision": None,
        "cache_dir": None,
    }


def test_load_kwargs_reach_the_governed_load_door(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lazy=`` and friends are forwarded to ``tl.load`` unchanged."""

    hub = _MemoryHub(tmp_path / "downloads")
    huggingface.push_to_hub(_trace(), "someone/trace", api=hub)
    seen: dict[str, Any] = {}

    def fake_load(path: Path, **kwargs: Any) -> str:
        seen["path"] = path
        seen["kwargs"] = kwargs
        return "loaded"

    import torchlens._io.bundle as bundle_module

    monkeypatch.setattr(bundle_module, "load", fake_load)
    result = huggingface.load_from_hub(
        "someone/trace", api=hub, extract_dir=tmp_path / "x", lazy=True, map_location="cpu"
    )
    assert result == "loaded"
    assert seen["kwargs"] == {"lazy": True, "map_location": "cpu"}
    assert (seen["path"] / "manifest.json").is_file()


def test_default_path_uses_a_mock_huggingface_hub(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without ``api=``, ``HfApi(token=...)`` is built and ``hf_hub_download`` called."""

    hub = _MemoryHub(tmp_path / "downloads")
    huggingface.push_to_hub(_trace(), "someone/trace", api=hub, path_in_repo="runs/t.tar.gz")
    built: list[dict[str, Any]] = []

    class HfApi:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

        def hf_hub_download(self, **kwargs: Any) -> str:
            return hub.hf_hub_download(**kwargs)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=HfApi))
    loaded = huggingface.load_from_hub(
        "someone/trace",
        path_in_repo="runs/t.tar.gz",
        revision="v1",
        token="hf_token",
        cache_dir=tmp_path / "cache",
        extract_dir=tmp_path / "out",
    )
    assert isinstance(loaded, tl.Trace)
    assert built == [{"token": "hf_token"}]
    assert hub.calls[-1] == (
        "hf_hub_download",
        {
            "repo_id": "someone/trace",
            "filename": "runs/t.tar.gz",
            "revision": "v1",
            "cache_dir": str(tmp_path / "cache"),
        },
    )


def test_missing_huggingface_hub_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the package and without ``api=`` the ImportError names the extra."""

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r"torchlens\[hf\]"):
        huggingface.load_from_hub("someone/trace")


class _Tripwire:
    """Pickles fine; unpickling it is the failure the load side must never cause."""

    def __reduce__(self) -> Any:
        return (_explode, ())


def _explode() -> None:
    raise AssertionError("load_from_hub unpickled a raw artifact")


def test_raw_pickle_artifact_is_refused_without_unpickling(tmp_path: Path) -> None:
    """A ``.pkl`` pushed from a plain object refuses typed; nothing is unpickled."""

    hub = _MemoryHub(tmp_path / "downloads")
    hub.files[("someone/plain", "torchlens_artifact.pkl")] = pickle.dumps(_Tripwire())
    with pytest.raises(TorchLensIOError) as info:
        huggingface.load_from_hub(
            "someone/plain", path_in_repo="torchlens_artifact.pkl", api=hub, extract_dir=tmp_path
        )
    assert info.value.fields["code"] == "hub_artifact_not_a_bundle"
    assert info.value.fields["remedy"]


def test_tarball_that_is_not_gzip_tar_is_refused(tmp_path: Path) -> None:
    """A ``.tar.gz`` name over non-tar bytes refuses with the same code."""

    hub = _MemoryHub(tmp_path / "downloads")
    hub.files[("someone/bad", "torchlens_artifact.tar.gz")] = b"not a tarball"
    with pytest.raises(TorchLensIOError) as info:
        huggingface.load_from_hub("someone/bad", api=hub, extract_dir=tmp_path / "x")
    assert info.value.fields["code"] == "hub_artifact_not_a_bundle"


def test_tarball_without_a_bundle_directory_is_refused(tmp_path: Path) -> None:
    """A tarball with no (or two) ``manifest.json`` directories is not a bundle."""

    hub = _MemoryHub(tmp_path / "downloads")
    hub.files[("someone/none", "torchlens_artifact.tar.gz")] = _tarball({"readme.txt": b"hi"})
    with pytest.raises(TorchLensIOError) as info:
        huggingface.load_from_hub("someone/none", api=hub, extract_dir=tmp_path / "none")
    assert info.value.fields["code"] == "hub_artifact_not_a_bundle"

    hub.files[("someone/two", "torchlens_artifact.tar.gz")] = _tarball(
        {"a/manifest.json": b"{}", "b/manifest.json": b"{}"}
    )
    with pytest.raises(TorchLensIOError) as info:
        huggingface.load_from_hub("someone/two", api=hub, extract_dir=tmp_path / "two")
    assert info.value.fields["code"] == "hub_artifact_not_a_bundle"


@pytest.mark.parametrize(
    "member_name",
    ["../escape.txt", "/abs/escape.txt", "bundle/../../escape.txt"],
)
def test_path_escaping_members_are_refused_before_extraction(
    tmp_path: Path, member_name: str
) -> None:
    """A member whose path leaves the extraction directory refuses; nothing is written."""

    hub = _MemoryHub(tmp_path / "downloads")
    hub.files[("someone/evil", "torchlens_artifact.tar.gz")] = _tarball(
        {"bundle/manifest.json": b"{}", member_name: b"owned"}
    )
    extract_dir = tmp_path / "extract"
    with pytest.raises(TorchLensIOError) as info:
        huggingface.load_from_hub("someone/evil", api=hub, extract_dir=extract_dir)
    assert info.value.fields["code"] == "hub_artifact_tar_unsafe"
    assert info.value.fields["remedy"]
    assert not list(extract_dir.iterdir())
    assert not (tmp_path / "escape.txt").exists()


def test_link_members_are_refused(tmp_path: Path) -> None:
    """Symlink and hard-link members are refused even when their names look tame."""

    link = tarfile.TarInfo("bundle/manifest.json")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    hub = _MemoryHub(tmp_path / "downloads")
    hub.files[("someone/link", "torchlens_artifact.tar.gz")] = _tarball(
        {"bundle/blob.bin": b"x"}, extra=link
    )
    extract_dir = tmp_path / "extract"
    with pytest.raises(TorchLensIOError) as info:
        huggingface.load_from_hub("someone/link", api=hub, extract_dir=extract_dir)
    assert info.value.fields["code"] == "hub_artifact_tar_unsafe"
    assert not list(extract_dir.iterdir())
