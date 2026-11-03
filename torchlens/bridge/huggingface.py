"""Hugging Face Hub publishing and loading helpers."""

from __future__ import annotations

import io
import pickle
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

_DEFAULT_PATH_IN_REPO = "torchlens_artifact.pkl"
_DEFAULT_TARBALL_PATH_IN_REPO = "torchlens_artifact.tar.gz"
_BUNDLE_TARBALL_SUFFIXES = (".tar.gz", ".tgz")
_HF_EXTRA_HINT = "Hugging Face Hub access requires the `hf` extra: install torchlens[hf]."


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
    """Upload a real TorchLens artifact to the Hugging Face Hub.

    Parameters
    ----------
    log_or_bundle_or_spec:
        ``Trace``, ``Bundle``, or ``InterventionSpec``-like object to publish.
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
        Public ``.tlspec`` save level used when the artifact must be
        serialized through the portable-bundle scrub path (``"audit"``,
        ``"executable_with_callables"``, or ``"portable"``). Only consulted
        when ``log_or_bundle_or_spec`` is a ``Trace``/``Bundle`` that cannot
        be pickled directly (see ``_artifact_bytes``).

    Returns
    -------
    dict[str, Any]
        Upload metadata including ``repo_id``, ``path_in_repo``, and ``format``
        (``"pickle"`` or ``"tar.gz"``). When the artifact must be packed as a
        gzipped tar bundle and the caller left ``path_in_repo`` at its default,
        the destination name is switched to a ``.tar.gz`` extension so a bundle
        is never uploaded under a ``.pkl`` name.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is unavailable.
    TorchLensIOError
        If the artifact cannot be serialized at all. ``push_to_hub`` never
        silently substitutes a metadata-only stub for genuine artifact
        content -- either the real artifact is uploaded, or this raises.
    """

    if api is None and not dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(_HF_EXTRA_HINT) from exc
        api = HfApi(token=token)

    payload, artifact_format = _artifact_bytes(log_or_bundle_or_spec, save_level=save_level)
    if artifact_format == "tar.gz" and path_in_repo == _DEFAULT_PATH_IN_REPO:
        path_in_repo = _DEFAULT_TARBALL_PATH_IN_REPO

    if dry_run:
        return {
            "repo_id": repo_id,
            "path_in_repo": path_in_repo,
            "size_bytes": len(payload),
            "format": artifact_format,
            "dry_run": True,
        }

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / Path(path_in_repo).name
        artifact_path.write_bytes(payload)
        size_bytes = artifact_path.stat().st_size
        if api is None:
            raise RuntimeError("A Hugging Face API object is required when dry_run=False.")
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
        "format": artifact_format,
        "dry_run": False,
        "upload_result": upload_result,
    }


def load_from_hub(
    repo_id: str,
    *,
    path_in_repo: str = _DEFAULT_TARBALL_PATH_IN_REPO,
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    extract_dir: str | Path | None = None,
    api: Any | None = None,
    **load_kwargs: Any,
) -> Any:
    """Download a TorchLens bundle published with :func:`push_to_hub` and load it.

    The reverse of :func:`push_to_hub`: the file is fetched through
    ``huggingface_hub``, the gzipped tar is extracted after a member check,
    and the ``.tlspec`` bundle directory inside it is opened with the same
    governed :func:`torchlens.load` door a local bundle goes through (manifest,
    schema and integrity gates, custom-callable trust flags). Nothing is ever
    unpickled outside that door: a ``.pkl`` artifact, which ``push_to_hub``
    writes only for plain objects with no bundle path, is refused typed.

    Parameters
    ----------
    repo_id:
        Hugging Face repository ID the artifact was pushed to.
    path_in_repo:
        Filename inside the repository; ``push_to_hub`` names a bundle
        ``torchlens_artifact.tar.gz`` by default.
    revision:
        Optional git revision (branch, tag or commit) to download from.
    token:
        Optional Hub token for private repositories.
    cache_dir:
        Optional ``huggingface_hub`` download cache directory.
    extract_dir:
        Directory the tarball is extracted into. Defaults to a fresh
        temporary directory, which is left in place so a ``lazy=True`` load
        keeps a readable bundle behind it; pass a directory to choose where
        the bundle lives.
    api:
        Optional ``HfApi``-compatible object exposing ``hf_hub_download`` for
        tests or advanced callers (the same seam ``push_to_hub`` takes).
    **load_kwargs:
        Forwarded to :func:`torchlens.load` (``lazy``, ``map_location``,
        ``trust_custom_callables``, ``allowed_custom_callable_modules``,
        ``unknown_relations``, ...).

    Returns
    -------
    Any
        The loaded ``Trace``, ``Bundle`` or ``InterventionSpec``, exactly as
        :func:`torchlens.load` returns it for the extracted bundle.

    Raises
    ------
    ImportError
        If ``huggingface_hub`` is unavailable and no ``api`` was given.
    TorchLensIOError
        ``hub_artifact_not_a_bundle`` when the downloaded file is not a
        gzipped tar holding one ``.tlspec`` bundle (for example a ``.pkl``
        pushed from a plain object); ``hub_artifact_tar_unsafe`` when a tar
        member would write outside the extraction directory or is not a plain
        file or directory. Load-time refusals of the bundle itself carry the
        :func:`torchlens.load` codes unchanged.
    """

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ImportError(_HF_EXTRA_HINT) from exc
        api = HfApi(token=token)

    downloaded = Path(
        api.hf_hub_download(
            repo_id=repo_id,
            filename=path_in_repo,
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
        )
    )
    _refuse_non_bundle_download(repo_id, path_in_repo, downloaded)
    destination = (
        Path(tempfile.mkdtemp(prefix="torchlens-hub-"))
        if extract_dir is None
        else Path(extract_dir)
    )
    bundle_root = _extract_bundle_tarball(downloaded, destination, repo_id=repo_id)

    from .._io.bundle import load as _load_bundle

    return _load_bundle(bundle_root, **load_kwargs)


def _refuse_non_bundle_download(repo_id: str, path_in_repo: str, downloaded: Path) -> None:
    """Refuse a downloaded artifact that is not a gzipped-tar bundle.

    Parameters
    ----------
    repo_id:
        Repository the file came from (for the message).
    path_in_repo:
        Requested filename.
    downloaded:
        Local path ``hf_hub_download`` returned.

    Raises
    ------
    TorchLensIOError
        ``hub_artifact_not_a_bundle`` for any other filename or a file that
        is not a readable gzipped tar.
    """

    from .._io import TorchLensIOError

    if not path_in_repo.endswith(_BUNDLE_TARBALL_SUFFIXES) or not tarfile.is_tarfile(downloaded):
        raise TorchLensIOError(
            f"load_from_hub fetched {path_in_repo!r} from {repo_id!r}, which is not a "
            "gzipped tar of a `.tlspec` bundle; push_to_hub writes other formats "
            "(a raw `.pkl`) only for plain objects, and load_from_hub never unpickles",
            code="hub_artifact_not_a_bundle",
            remedy="publish a Trace or Bundle with push_to_hub (it uploads a .tar.gz "
            "bundle), or download the file with huggingface_hub.hf_hub_download and open "
            "it yourself",
        )


def _extract_bundle_tarball(tarball: Path, destination: Path, *, repo_id: str) -> Path:
    """Extract a bundle tarball into ``destination`` and return the bundle root.

    Every member must be a plain file or directory whose path stays inside
    ``destination``; links, devices, absolute names and ``..`` components are
    refused before anything is written. Pythons that ship
    ``tarfile.data_filter`` apply it as a second guard.

    Parameters
    ----------
    tarball:
        Gzipped tar downloaded from the Hub.
    destination:
        Directory to extract into (created if missing).
    repo_id:
        Repository the file came from (for messages).

    Returns
    -------
    Path
        The extracted ``.tlspec`` bundle directory (the one holding
        ``manifest.json``).

    Raises
    ------
    TorchLensIOError
        ``hub_artifact_tar_unsafe`` for a refused member;
        ``hub_artifact_not_a_bundle`` when no single bundle directory results.
    """

    from .._io import TorchLensIOError

    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(tarball, mode="r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            unsafe = _unsafe_member_reason(member, root)
            if unsafe is not None:
                raise TorchLensIOError(
                    f"load_from_hub refused tar member {member.name!r} from {repo_id!r}: {unsafe}",
                    code="hub_artifact_tar_unsafe",
                    remedy="re-publish the bundle with push_to_hub, which writes only plain "
                    "files and directories under one bundle directory",
                )
        if hasattr(tarfile, "data_filter"):
            tar.extractall(root, members=members, filter="data")
        else:  # pragma: no cover - Pythons without the backported filter
            tar.extractall(root, members=members)

    if (root / "manifest.json").is_file():
        return root
    candidates = [entry for entry in root.iterdir() if (entry / "manifest.json").is_file()]
    if len(candidates) != 1:
        raise TorchLensIOError(
            f"load_from_hub extracted {tarball.name!r} from {repo_id!r} but found "
            f"{len(candidates)} `.tlspec` bundle directories (a manifest.json) inside it, "
            "not one",
            code="hub_artifact_not_a_bundle",
            remedy="publish a Trace or Bundle with push_to_hub (it uploads one bundle "
            "directory per tarball), or download the file with "
            "huggingface_hub.hf_hub_download and open it yourself",
        )
    return candidates[0]


def _unsafe_member_reason(member: tarfile.TarInfo, root: Path) -> str | None:
    """Return why a tar member may not be extracted under ``root``, or ``None``.

    Parameters
    ----------
    member:
        Tar member being checked.
    root:
        Resolved extraction root.

    Returns
    -------
    str | None
        Reason text for a refused member, ``None`` for a plain file or
        directory that stays inside ``root``.
    """

    if not (member.isfile() or member.isdir()):
        return "only plain files and directories are extracted"
    name = PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts:
        return "its path would escape the extraction directory"
    target = (root / Path(*name.parts)).resolve() if name.parts else root
    if target != root and root not in target.parents:
        return "its path would escape the extraction directory"
    return None


def _artifact_bytes(
    log_or_bundle_or_spec: Any, *, save_level: str = "portable"
) -> tuple[bytes, str]:
    """Serialize an artifact for Hub upload and report its serialized format.

    Parameters
    ----------
    log_or_bundle_or_spec:
        Artifact object.
    save_level:
        Public ``.tlspec`` save level used for the portable-bundle fallback
        path (see below).

    Returns
    -------
    tuple[bytes, str]
        The serialized bytes and a format tag. A ``Trace``/``Bundle`` -- which
        raw ``pickle.dumps`` would embed ``$HOME``, the username, and absolute
        source/bundle/visualizer paths into (R62-1: a privacy leak to the one
        PUBLIC sharing surface) -- is ALWAYS serialized through the same real,
        privacy-scrubbed ``.tlspec`` portable bundle path used by
        :func:`torchlens.save`/``Bundle.save`` and returned as a gzipped tar
        archive with the tag ``"tar.gz"``. The scrub is real, not a metadata
        stand-in. Only an object with NO portable-bundle save path (e.g. a plain
        dict of user data the caller chose to push) is pickled directly and
        tagged ``"pickle"``.

    Raises
    ------
    TorchLensIOError
        If the artifact cannot be serialized at all (the portable-bundle path
        itself fails, or a no-bundle object fails to pickle).
    """

    from .._io import TorchLensIOError

    # Privacy-first (R62-1): any artifact with a portable-bundle save path
    # (Trace/Bundle) is serialized through the SCRUBBED bundle path, never raw
    # ``pickle.dumps`` -- raw pickle leaks $HOME, the username, and absolute
    # source paths to a public hub. Raw pickle is reserved for objects that have
    # no bundle save path and thus carry no scrubbable TorchLens internals.
    saver = _resolve_bundle_saver(log_or_bundle_or_spec)
    if saver is not None:
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                bundle_dir = Path(tmpdir) / "bundle"
                saver(bundle_dir, level=save_level, overwrite=True)
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
                    tar.add(bundle_dir, arcname=bundle_dir.name)
                return buffer.getvalue(), "tar.gz"
        except Exception as bundle_error:
            raise TorchLensIOError(
                "push_to_hub could not serialize this "
                f"{type(log_or_bundle_or_spec).__name__} artifact for upload through "
                f"the privacy-scrubbed portable `.tlspec` bundle path ({bundle_error!r}). "
                "Refusing to fall back to raw pickle (which would leak local paths) or "
                "to silently upload a metadata-only stub instead of the real artifact."
            ) from bundle_error

    try:
        return pickle.dumps(log_or_bundle_or_spec), "pickle"
    except Exception as direct_pickle_error:
        raise TorchLensIOError(
            "push_to_hub could not serialize this "
            f"{type(log_or_bundle_or_spec).__name__} artifact for upload "
            f"({direct_pickle_error!r}) and it has no portable `.tlspec` bundle "
            "save path to fall back to. Refusing to silently upload a metadata-only "
            "stub instead of the real artifact."
        ) from direct_pickle_error


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


__all__ = ["load_from_hub", "push_to_hub"]
