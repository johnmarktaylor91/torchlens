"""Hugging Face Hub publishing and loading helpers.

Both directions go through the portable ``.tlspec`` bundle: ``push_to_hub``
uploads a gzipped tar of the privacy-scrubbed bundle (never a raw pickle),
and ``from_hub`` downloads that tar, extracts it under strict member checks,
and loads it through the default-deny ``.tlspec`` loader (``torchlens.load``).
"""

from __future__ import annotations

import io
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

_DEFAULT_PATH_IN_REPO = "torchlens_artifact.tar.gz"
_BUNDLE_ARCNAME = "bundle"
#: Upper bounds on a downloaded artifact's tar: a Hub file is untrusted input.
_MAX_TAR_MEMBERS = 100_000
_MAX_TAR_BYTES = 64 * 1024**3


def push_to_hub(
    log_or_bundle_or_spec: Any,
    repo_id: str,
    *,
    token: str | None = None,
    private: bool | None = None,
    path_in_repo: str = _DEFAULT_PATH_IN_REPO,
    commit_message: str = "Add TorchLens artifact",
    create_repo: bool = True,
    dry_run: bool = False,
    api: Any | None = None,
    save_level: str = "portable",
) -> dict[str, Any]:
    """Upload a TorchLens ``Trace`` or ``Bundle`` to the Hugging Face Hub.

    The artifact is saved as a privacy-scrubbed portable ``.tlspec`` bundle
    and uploaded as one gzipped tar; :func:`from_hub` reads it back through
    the safe loader. Nothing is ever uploaded as a raw pickle: an object with
    no ``.tlspec`` save path is refused.

    Parameters
    ----------
    log_or_bundle_or_spec:
        ``Trace`` or ``Bundle`` to publish.
    repo_id:
        Target Hugging Face repository ID.
    token:
        Optional Hub token.
    private:
        Optional repository privacy flag used when creating the repo.
    path_in_repo:
        Destination filename inside the repository.
    commit_message:
        Commit message for the upload.
    create_repo:
        Whether to create the repo before upload.
    dry_run:
        If True, serialize locally and return planned upload metadata without
        contacting the Hub.
    api:
        Optional ``HfApi``-compatible object for tests or advanced callers.
    save_level:
        Public ``.tlspec`` save level of the uploaded bundle (``"audit"``,
        ``"executable_with_callables"``, or ``"portable"``).

    Returns
    -------
    dict[str, Any]
        Upload metadata including ``repo_id``, ``path_in_repo``,
        ``size_bytes`` and ``format`` (always ``"tar.gz"``).

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is unavailable.
    ArgumentTypeError
        ``hub_push_artifact_unsupported`` when the object is neither a
        ``Trace`` nor a ``Bundle`` (raw pickling is never a fallback).
    TorchLensIOError
        If the ``.tlspec`` bundle save fails. ``push_to_hub`` never silently
        substitutes a metadata-only stub for genuine artifact content.
    """

    payload = _artifact_bytes(log_or_bundle_or_spec, save_level=save_level)

    if dry_run:
        return {
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "size_bytes": len(payload),
            "format": "tar.gz",
            "dry_run": True,
        }

    if api is None:
        api = _hf_api(token)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / Path(path_in_repo).name
        artifact_path.write_bytes(payload)
        size_bytes = artifact_path.stat().st_size
        if create_repo:
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
        upload_result = api.upload_file(
            path_or_fileobj=str(artifact_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            commit_message=commit_message,
        )

    return {
        "repo_id": repo_id,
        "path_in_repo": path_in_repo,
        "size_bytes": size_bytes,
        "format": "tar.gz",
        "dry_run": False,
        "upload_result": upload_result,
    }


def from_hub(
    repo_id: str,
    *,
    path_in_repo: str = _DEFAULT_PATH_IN_REPO,
    revision: str | None = None,
    token: str | None = None,
    api: Any | None = None,
) -> Any:
    """Download a TorchLens artifact from the Hub and load it safely.

    The file must be the gzipped ``.tlspec`` tar that :func:`push_to_hub`
    writes. Its members are checked before extraction (regular files and
    directories under one top-level bundle directory only: no links, devices,
    absolute paths or ``..``), and the bundle is loaded eagerly onto the CPU through
    ``torchlens.load``, whose metadata unpickler is default-deny and never
    imports foreign custom callables. The extracted copy is deleted after the
    load.

    Parameters
    ----------
    repo_id:
        Source Hugging Face repository ID.
    path_in_repo:
        Artifact filename inside the repository.
    revision:
        Optional branch, tag or commit to download from.
    token:
        Optional Hub token (used only when ``api`` is not given).
    api:
        Optional ``HfApi``-compatible object exposing ``hf_hub_download``,
        for tests, mirrors or advanced callers.
    Returns
    -------
    Trace | Bundle | InterventionSpec
        Whatever the ``.tlspec`` manifest names, as ``torchlens.load`` returns.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is unavailable and no ``api`` is given.
    TorchLensIOError
        ``hub_artifact_unsafe`` when the download is not a readable gzipped
        tar or a member fails the extraction checks; any ``.tlspec`` load
        refusal propagates unchanged.
    """

    from .._io.bundle import load as _load

    if api is None:
        api = _hf_api(token)
    downloaded = Path(
        api.hf_hub_download(repo_id=repo_id, filename=path_in_repo, revision=revision)
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = _extract_bundle(downloaded, Path(tmpdir))
        return _load(bundle_dir, lazy=False, map_location="cpu")


def _hf_api(token: str | None) -> Any:
    """Return a ``huggingface_hub.HfApi`` or raise naming the ``hf`` extra."""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Hugging Face publishing requires the `hf` extra: install torchlens[hf]."
        ) from exc
    return HfApi(token=token)


def _extract_bundle(archive: Path, destination: Path) -> Path:
    """Extract a pushed ``.tlspec`` tar after checking every member.

    Parameters
    ----------
    archive:
        Downloaded gzipped tar.
    destination:
        Empty directory to extract into.

    Returns
    -------
    Path
        The single top-level bundle directory.

    Raises
    ------
    TorchLensIOError
        ``hub_artifact_unsafe`` on an unreadable archive, a link or special
        member, an absolute or parent-escaping name, more than one top-level
        entry, or a member count or total size over the bounds.
    """

    from .._io import TorchLensIOError

    root, problem = _checked_extract(archive, destination)
    if problem is not None:
        raise TorchLensIOError(
            f"from_hub refused {archive.name}: {problem}. Only the gzipped `.tlspec` "
            "tar written by push_to_hub is loaded. Remedy: point path_in_repo at an "
            "artifact uploaded with torchlens.bridge.huggingface.push_to_hub, or "
            "download the file yourself and inspect it before loading it with "
            "torchlens.load.",
            code="hub_artifact_unsafe",
            remedy=(
                "point path_in_repo at an artifact uploaded with push_to_hub, or "
                "download and inspect the file before loading it with torchlens.load"
            ),
        )
    return destination / root


def _checked_extract(archive: Path, destination: Path) -> tuple[str, str | None]:
    """Check every member, then extract; return ``(root, problem)``."""

    try:
        with tarfile.open(archive, mode="r:gz") as tar:
            members = tar.getmembers()
            problem = _members_problem(members)
            if problem is None:
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(destination, filter="data")
                else:  # pragma: no cover - interpreters without PEP 706
                    tar.extractall(destination)  # noqa: S202 -- every member checked above
    except (tarfile.TarError, OSError, EOFError) as exc:
        return "", f"not a readable gzipped tar ({exc})"
    if problem is not None:
        return "", problem
    root = PurePosixPath(members[0].name).parts[0]
    if not (destination / root).is_dir():
        return "", "the top-level entry is not a bundle directory"
    return root, None


def _members_problem(members: list[tarfile.TarInfo]) -> str | None:
    """Return why a member list is unsafe to extract, or ``None``."""

    problem = next((p for p in map(_member_problem, members) if p is not None), None)
    if problem is not None:
        return problem
    roots = {PurePosixPath(member.name).parts[0] for member in members}
    if len(roots) != 1:
        return f"expected one top-level bundle, found {len(roots)}"
    if len(members) > _MAX_TAR_MEMBERS:
        return f"{len(members)} members exceed the bound"
    if sum(member.size for member in members) > _MAX_TAR_BYTES:
        return "total size exceeds the bound"
    return None


def _member_problem(member: tarfile.TarInfo) -> str | None:
    """Return why one tar member is unsafe to extract, or ``None``."""

    if not (member.isfile() or member.isdir()):
        return f"member {member.name!r} is a link or special file"
    parts = PurePosixPath(member.name).parts
    if not parts or member.name.startswith("/") or ".." in parts or "\\" in member.name:
        return f"member name {member.name!r} escapes the bundle"
    return None


def _artifact_bytes(log_or_bundle_or_spec: Any, *, save_level: str = "portable") -> bytes:
    """Serialize a ``Trace``/``Bundle`` as a gzipped tar of its ``.tlspec`` bundle.

    Parameters
    ----------
    log_or_bundle_or_spec:
        Artifact object.
    save_level:
        Public ``.tlspec`` save level.

    Returns
    -------
    bytes
        The gzipped tar. The bundle is the same privacy-scrubbed portable
        save :func:`torchlens.save` / ``Bundle.save`` write (raw pickling of a
        ``Trace`` would embed ``$HOME``, the username and absolute source
        paths into a public upload).

    Raises
    ------
    ArgumentTypeError
        ``hub_push_artifact_unsupported`` for any other object.
    TorchLensIOError
        If the bundle save fails.
    """

    from .._errors import ArgumentTypeError
    from .._io import TorchLensIOError

    saver = _resolve_bundle_saver(log_or_bundle_or_spec)
    if saver is None:
        raise ArgumentTypeError(
            "push_to_hub uploads only a Trace or a Bundle (as a portable `.tlspec` "
            f"bundle); got {type(log_or_bundle_or_spec).__name__}. Raw pickling is "
            "never used for Hub uploads.",
            code="hub_push_artifact_unsupported",
            remedy=(
                "push a Trace or a Bundle; save an intervention recipe with "
                "torchlens.io.save_intervention(trace, path) and share that "
                "`.tlspec` directory; share plain data in a format of your choice"
            ),
            received_type=type(log_or_bundle_or_spec).__name__,
        )
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / _BUNDLE_ARCNAME
            saver(bundle_dir, level=save_level, overwrite=True)
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                tar.add(bundle_dir, arcname=_BUNDLE_ARCNAME)
            return buffer.getvalue()
    except Exception as bundle_error:
        raise TorchLensIOError(
            "push_to_hub could not serialize this "
            f"{type(log_or_bundle_or_spec).__name__} artifact through the "
            f"privacy-scrubbed portable `.tlspec` bundle path ({bundle_error!r}). "
            "Refusing to upload a metadata-only stub instead of the real artifact."
        ) from bundle_error


def _resolve_bundle_saver(log_or_bundle_or_spec: Any) -> Any | None:
    """Return a ``save(path, *, level, overwrite)`` callable for real objects.

    Parameters
    ----------
    log_or_bundle_or_spec:
        Candidate artifact object.

    Returns
    -------
    Any | None
        A bound ``save`` callable matching the ``.tlspec`` bundle contract
        (``Trace``/``Bundle`` both expose one), or ``None`` if the object has
        no known portable-bundle save path.
    """

    from ..data_classes.trace import Trace

    if isinstance(log_or_bundle_or_spec, Trace):
        from .._io.bundle import save as _save_trace_bundle

        def _save_trace(path: Path, *, level: str, overwrite: bool) -> None:
            """Save the captured ``Trace`` as a portable ``.tlspec`` bundle.

            Parameters
            ----------
            path:
                Destination bundle directory.
            level:
                Public ``.tlspec`` save level.
            overwrite:
                Whether to overwrite an existing bundle at ``path``.
            """

            _save_trace_bundle(log_or_bundle_or_spec, path, level=level, overwrite=overwrite)

        return _save_trace

    from ..bundle import Bundle

    if isinstance(log_or_bundle_or_spec, Bundle):
        return log_or_bundle_or_spec.save

    return None


__all__ = ["from_hub", "push_to_hub"]
