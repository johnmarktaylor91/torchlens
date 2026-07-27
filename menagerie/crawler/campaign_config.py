"""Private persisted campaign configuration for unattended crawler resumes."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from menagerie.crawler.identity import canonical_json_bytes, stable_hash

CAMPAIGN_CONFIG_FORMAT = "menagerie.crawler.campaign-config.v1"


@dataclass(frozen=True)
class CampaignConfig:
    """Resolved operator inputs that must survive a clean wake environment."""

    repo_root: Path
    intake_root: Path
    target: str
    run_id: str
    author_command: tuple[str, ...]
    checker_command: tuple[str, ...]
    environment_command: tuple[str, ...]
    notify_command: Optional[tuple[str, ...]]
    public_mirror: Path
    private_mirror: Path
    review_checkpoint_at: Optional[int]
    progress_milestones: tuple[int, ...]
    phase: Optional[str]
    only_status: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the closed JSON representation.

        Returns
        -------
        dict[str, Any]
            Canonical persisted campaign configuration.
        """

        return {
            "format": CAMPAIGN_CONFIG_FORMAT,
            "repo_root": str(self.repo_root),
            "intake_root": str(self.intake_root),
            "target": self.target,
            "run_id": self.run_id,
            "author_command": list(self.author_command),
            "checker_command": list(self.checker_command),
            "environment_command": list(self.environment_command),
            "notify_command": (
                list(self.notify_command) if self.notify_command is not None else None
            ),
            "mirrors": {
                "public": str(self.public_mirror),
                "private": str(self.private_mirror),
            },
            "review_checkpoint_at": self.review_checkpoint_at,
            "progress_milestones": list(self.progress_milestones),
            "phase": self.phase,
            "only_status": self.only_status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignConfig":
        """Parse and validate one persisted campaign configuration.

        Parameters
        ----------
        value:
            Decoded JSON object.

        Returns
        -------
        CampaignConfig
            Validated resolved campaign configuration.
        """

        expected = {
            "format",
            "repo_root",
            "intake_root",
            "target",
            "run_id",
            "author_command",
            "checker_command",
            "environment_command",
            "notify_command",
            "mirrors",
            "review_checkpoint_at",
            "progress_milestones",
            "phase",
            "only_status",
        }
        if set(value) != expected or value.get("format") != CAMPAIGN_CONFIG_FORMAT:
            raise ValueError("campaign config has an invalid field contract")
        mirrors = value.get("mirrors")
        if not isinstance(mirrors, Mapping) or set(mirrors) != {"public", "private"}:
            raise ValueError("campaign config mirrors are invalid")
        author = _command(value.get("author_command"), "author")
        checker = _command(value.get("checker_command"), "checker")
        environment = _command(value.get("environment_command"), "environment")
        raw_notify = value.get("notify_command")
        notify = None if raw_notify is None else _command(raw_notify, "notify")
        milestones = value.get("progress_milestones")
        if not isinstance(milestones, list) or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1 for item in milestones
        ):
            raise ValueError("campaign config progress milestones are invalid")
        review = value.get("review_checkpoint_at")
        if review is not None and (
            not isinstance(review, int) or isinstance(review, bool) or review < 0
        ):
            raise ValueError("campaign config review checkpoint is invalid")
        target = value.get("target")
        phase = value.get("phase")
        only_status = value.get("only_status")
        run_id = value.get("run_id")
        if target not in {"osx-arm64", "linux-x86_64-cuda"}:
            raise ValueError("campaign config target is invalid")
        if phase not in {None, "pytorch", "native-tail"}:
            raise ValueError("campaign config phase is invalid")
        if only_status not in {
            None,
            "deferred:*",
            "deferred:needs-cuda",
            "deferred:needs-x86",
        }:
            raise ValueError("campaign config only-status is invalid")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("campaign config run ID is invalid")
        repo_root = _absolute_path(value.get("repo_root"), "repo root")
        intake_root = _absolute_path(value.get("intake_root"), "intake root")
        public_mirror = _absolute_path(mirrors.get("public"), "public mirror")
        private_mirror = _absolute_path(mirrors.get("private"), "private mirror")
        return cls(
            repo_root=repo_root,
            intake_root=intake_root,
            target=target,
            run_id=run_id,
            author_command=author,
            checker_command=checker,
            environment_command=environment,
            notify_command=notify,
            public_mirror=public_mirror,
            private_mirror=private_mirror,
            review_checkpoint_at=review,
            progress_milestones=tuple(milestones),
            phase=phase,
            only_status=only_status,
        )


def default_campaign_config_path(repo_root: Path, intake_root: Path) -> Path:
    """Return the private config path for one exact intake.

    Parameters
    ----------
    repo_root, intake_root:
        Resolved repository and immutable intake roots.

    Returns
    -------
    pathlib.Path
        Campaign-specific gitignored configuration path.
    """

    suffix = stable_hash(str(intake_root.resolve())).removeprefix("sha256:")[:12]
    return (
        repo_root.resolve()
        / ".crawl-local"
        / "campaign-configs"
        / (f"{intake_root.name}-{suffix}.json")
    )


def write_campaign_config(path: Path, config: CampaignConfig) -> None:
    """Atomically persist one mode-0600 campaign configuration.

    Parameters
    ----------
    path, config:
        Destination and fully resolved configuration.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(config.to_dict()) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_campaign_config(path: Path) -> CampaignConfig:
    """Load a private mode-safe campaign configuration.

    Parameters
    ----------
    path:
        Persisted configuration path.

    Returns
    -------
    CampaignConfig
        Validated configuration.
    """

    mode = path.stat().st_mode & 0o777
    if mode & 0o077:
        raise ValueError(f"campaign config must be mode 0600: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"campaign config is unreadable: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("campaign config must contain one JSON object")
    return CampaignConfig.from_dict(value)


def resolve_command(command: Sequence[str], *, cwd: Path) -> tuple[str, ...]:
    """Resolve an argv command to stable absolute executable and path arguments.

    Parameters
    ----------
    command:
        Nonempty parsed command.
    cwd:
        Base for relative existing path arguments.

    Returns
    -------
    tuple[str, ...]
        Clean-environment-safe command.
    """

    if not command:
        raise ValueError("cannot resolve an empty command")
    executable = command[0]
    candidate = Path(executable).expanduser()
    if candidate.is_absolute() or candidate.parent != Path("."):
        resolved_executable = candidate.resolve()
        if not resolved_executable.is_file():
            raise ValueError(f"command executable is not a file: {executable}")
    else:
        found = shutil.which(executable)
        if found is None:
            raise ValueError(f"command executable is not resolvable: {executable}")
        resolved_executable = Path(found).resolve()
    resolved: list[str] = [str(resolved_executable)]
    for argument in command[1:]:
        candidate = Path(argument).expanduser()
        rooted = candidate if candidate.is_absolute() else cwd / candidate
        resolved.append(str(rooted.resolve()) if rooted.exists() else argument)
    return tuple(resolved)


def _command(value: Any, name: str) -> tuple[str, ...]:
    """Validate one persisted argv array.

    Parameters
    ----------
    value, name:
        Candidate value and diagnostic field name.

    Returns
    -------
    tuple[str, ...]
        Validated immutable argv.
    """

    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"campaign config {name} command is invalid")
    command = tuple(value)
    if not Path(command[0]).is_absolute():
        raise ValueError(f"campaign config {name} executable must be absolute")
    return command


def _absolute_path(value: Any, name: str) -> Path:
    """Validate one persisted absolute path.

    Parameters
    ----------
    value, name:
        Candidate path value and diagnostic name.

    Returns
    -------
    pathlib.Path
        Absolute path.
    """

    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError(f"campaign config {name} must be absolute")
    return Path(value)


def _fsync_directory(path: Path) -> None:
    """Fsync a directory after an atomic configuration replacement.

    Parameters
    ----------
    path:
        Directory containing the replaced file.
    """

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
