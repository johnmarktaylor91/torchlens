"""Execution-phase offline environment and policy tripwires."""

from __future__ import annotations

import ast
import builtins
import importlib.abc
import importlib.machinery
import os
import shutil
import socket
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, IO, Mapping, Optional, Sequence, Union


_CREDENTIAL_MARKERS = (
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "CREDENTIAL",
    "COOKIE",
)
_CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".h5", ".weights")
_SAFE_INHERITED_KEYS = (
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "LANG",
    "LC_ALL",
    "TZ",
    "SYSTEMROOT",
    "WINDIR",
    "TMPDIR",
)


class PolicyViolation(RuntimeError):
    """Raised immediately when execution violates a closed worker policy."""

    def __init__(self, reason_code: str, detail: str) -> None:
        """Initialize a structured policy violation.

        Parameters
        ----------
        reason_code:
            Closed policy failure reason.
        detail:
            Non-secret diagnostic.
        """

        super().__init__(detail)
        self.reason_code = reason_code


@dataclass
class PolicyObservation:
    """Worker-side policy tripwire observations.

    Parameters
    ----------
    network_attempted, checkpoint_or_weight_read_attempted,
    write_outside_scratch_attempted, credentials_present,
    torchlens_import_attempted, cache_read_attempted:
        Closed attempt flags used by the driver.
    socket_targets, checkpoint_paths, write_paths:
        Sanitized attempted targets.
    """

    network_attempted: bool = False
    socket_targets: list[str] = field(default_factory=list)
    checkpoint_or_weight_read_attempted: bool = False
    checkpoint_paths: list[str] = field(default_factory=list)
    write_outside_scratch_attempted: bool = False
    write_paths: list[str] = field(default_factory=list)
    credentials_present: bool = False
    torchlens_import_attempted: bool = False
    cache_read_attempted: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible policy observation.

        Returns
        -------
        dict[str, Any]
            Complete closed observation payload.
        """

        return {
            "network_attempted": self.network_attempted,
            "socket_targets": list(self.socket_targets),
            "checkpoint_or_weight_read_attempted": self.checkpoint_or_weight_read_attempted,
            "checkpoint_paths": list(self.checkpoint_paths),
            "write_outside_scratch_attempted": self.write_outside_scratch_attempted,
            "write_paths": list(self.write_paths),
            "credentials_present": self.credentials_present,
            "torchlens_import_attempted": self.torchlens_import_attempted,
            "cache_read_attempted": self.cache_read_attempted,
        }


def _contains_credential_name(name: str) -> bool:
    """Return whether an environment key appears credential-bearing.

    Parameters
    ----------
    name:
        Environment variable name.

    Returns
    -------
    bool
        True for secret-like names.
    """

    upper = name.upper()
    token_or_auth = (
        upper.endswith("TOKEN")
        or "_TOKEN_" in upper
        or upper.endswith("_AUTH")
        or "_AUTH_" in upper
    )
    return (
        token_or_auth
        or any(marker in upper for marker in _CREDENTIAL_MARKERS)
        or upper
        in {
            "SSH_AUTH_SOCK",
            "GIT_ASKPASS",
            "AWS_PROFILE",
        }
    )


def build_safe_environment(
    scratch_root: Path, *, base_environment: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    """Build a credential-free offline child environment with empty caches.

    Parameters
    ----------
    scratch_root:
        Writable worker root that will contain all fresh cache directories.
    base_environment:
        Environment to filter. Defaults to the current process environment.

    Returns
    -------
    dict[str, str]
        Allowlisted child environment with offline flags.
    """

    source = os.environ if base_environment is None else base_environment
    safe = {
        key: value
        for key, value in source.items()
        if key in _SAFE_INHERITED_KEYS and not _contains_credential_name(key)
    }
    cache_root = scratch_root / "caches"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "HOME": scratch_root / "home",
        "TMPDIR": scratch_root / "tmp",
        "TEMP": scratch_root / "tmp",
        "TMP": scratch_root / "tmp",
        "XDG_CACHE_HOME": cache_root / "xdg",
        "TORCH_HOME": cache_root / "torch",
        "HF_HOME": cache_root / "huggingface",
        "HUGGINGFACE_HUB_CACHE": cache_root / "huggingface-hub",
        "TRANSFORMERS_CACHE": cache_root / "transformers",
        "KERAS_HOME": cache_root / "keras",
        "JAX_CACHE_DIR": cache_root / "jax",
        "PADDLE_HOME": cache_root / "paddle",
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    safe.update({name: str(path) for name, path in cache_paths.items()})
    safe.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "WANDB_DISABLED": "true",
            "COMET_DISABLE_AUTO_LOGGING": "1",
            "NO_PROXY": "*",
            "no_proxy": "*",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "MENAGERIE_EXECUTION_OFFLINE": "1",
        }
    )
    return safe


def static_source_check(path: Path) -> None:
    """Reject TorchLens imports and opaque execution calls in Python source.

    Parameters
    ----------
    path:
        Python source file to inspect.

    Raises
    ------
    PolicyViolation
        If the source imports TorchLens or calls eval/exec/compile.
    """

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise PolicyViolation("opaque-code", f"cannot statically inspect {path}: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name == "torchlens" or alias.name.startswith("torchlens.") for alias in node.names
        ):
            raise PolicyViolation("torchlens-import", f"TorchLens import in {path}")
        if isinstance(node, ast.ImportFrom) and (
            node.module == "torchlens" or str(node.module).startswith("torchlens.")
        ):
            raise PolicyViolation("torchlens-import", f"TorchLens import in {path}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile"}:
                raise PolicyViolation("opaque-code", f"{node.func.id}() call in {path}")


class _TorchLensBlocker(importlib.abc.MetaPathFinder):
    """Runtime import finder that fails closed on TorchLens imports."""

    def __init__(self, observation: PolicyObservation) -> None:
        """Initialize the import blocker.

        Parameters
        ----------
        observation:
            Mutable policy observation.
        """

        self.observation = observation

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None,
    ) -> Optional[importlib.machinery.ModuleSpec]:
        """Reject TorchLens and ignore every other import.

        Parameters
        ----------
        fullname:
            Requested module name.
        path:
            Parent import path.
        target:
            Optional reload target.

        Returns
        -------
        importlib.machinery.ModuleSpec | None
            Always ``None`` for allowed imports.
        """

        del path, target
        if fullname == "torchlens" or fullname.startswith("torchlens."):
            self.observation.torchlens_import_attempted = True
            raise PolicyViolation("torchlens-import", f"blocked import of {fullname}")
        return None


class ExecutionPolicy(AbstractContextManager[PolicyObservation]):
    """In-process socket, write, checkpoint, and TorchLens tripwires.

    Parameters
    ----------
    scratch_root:
        Sole writable filesystem root.
    additional_write_roots:
        Other explicit result roots, normally the atomic receipt directory.
    """

    def __init__(self, scratch_root: Path, *additional_write_roots: Path) -> None:
        """Initialize inactive tripwires.

        Parameters
        ----------
        scratch_root:
            Primary writable root.
        *additional_write_roots:
            Explicit additional writable roots.
        """

        self.allowed_roots = tuple(
            path.resolve() for path in (scratch_root, *additional_write_roots)
        )
        self.observation = PolicyObservation(
            credentials_present=any(_contains_credential_name(name) for name in os.environ)
        )
        self._original_open = builtins.open
        self._original_os_open = os.open
        self._original_connect = socket.socket.connect
        self._original_connect_ex = socket.socket.connect_ex
        self._original_create_connection = socket.create_connection
        self._import_blocker = _TorchLensBlocker(self.observation)

        def blocked_connect(socket_instance: socket.socket, address: Any) -> Any:
            """Reject a method-form socket connection.

            Parameters
            ----------
            socket_instance:
                Socket receiving the blocked call.
            address:
                Attempted target.

            Returns
            -------
            Any
                Never returns.
            """

            return self._blocked_connect(socket_instance, address)

        def blocked_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
            """Reject a module-level socket connection.

            Parameters
            ----------
            address:
                Attempted target.
            *args, **kwargs:
                Standard socket connection options.

            Returns
            -------
            Any
                Never returns.
            """

            return self._blocked_create_connection(address, *args, **kwargs)

        self._blocked_connect_function = blocked_connect
        self._blocked_create_connection_function = blocked_create_connection

    def _path_allowed(self, value: Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]) -> bool:
        """Return whether a path is beneath an allowed write root.

        Parameters
        ----------
        value:
            Filesystem path.

        Returns
        -------
        bool
            True for an allowed path.
        """

        candidate = Path(os.fsdecode(value)).resolve()
        if candidate == Path(os.devnull).resolve():
            return True
        return any(candidate == root or root in candidate.parents for root in self.allowed_roots)

    def _audit_path(self, value: Any, *, writing: bool) -> None:
        """Audit one Python-level file access.

        Parameters
        ----------
        value:
            File path or descriptor.
        writing:
            Whether the operation can modify bytes.
        """

        if isinstance(value, int):
            return
        path_text = os.fsdecode(value)
        if not writing and path_text.lower().endswith(_CHECKPOINT_SUFFIXES):
            self.observation.checkpoint_or_weight_read_attempted = True
            self.observation.checkpoint_paths.append(path_text)
            raise PolicyViolation("checkpoint-read", f"blocked checkpoint read: {path_text}")
        if writing and not self._path_allowed(value):
            self.observation.write_outside_scratch_attempted = True
            self.observation.write_paths.append(path_text)
            raise PolicyViolation("write-outside-scratch", f"blocked write: {path_text}")

    def _open(
        self,
        file: Any,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
        closefd: bool = True,
        opener: Any = None,
    ) -> IO[Any]:
        """Audit and delegate the built-in open function.

        Parameters
        ----------
        file, mode, buffering, encoding, errors, newline, closefd, opener:
            Standard ``open`` arguments.

        Returns
        -------
        IO[Any]
            Open file object.
        """

        self._audit_path(file, writing=any(flag in mode for flag in "wax+"))
        return self._original_open(
            file,
            mode,
            buffering,
            encoding,
            errors,
            newline,
            closefd,
            opener,
        )

    def _os_open(self, path: Any, flags: int, mode: int = 0o777, *, dir_fd: Any = None) -> int:
        """Audit and delegate ``os.open``.

        Parameters
        ----------
        path, flags, mode, dir_fd:
            Standard ``os.open`` arguments.

        Returns
        -------
        int
            Open file descriptor.
        """

        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        self._audit_path(path, writing=bool(flags & write_flags))
        return self._original_os_open(path, flags, mode, dir_fd=dir_fd)

    def _blocked_connect(self, _socket: socket.socket, address: Any) -> Any:
        """Record and reject a socket connection attempt.

        Parameters
        ----------
        _socket:
            Socket instance.
        address:
            Attempted target.

        Returns
        -------
        Any
            Never returns.
        """

        self.observation.network_attempted = True
        self.observation.socket_targets.append(repr(address))
        raise PolicyViolation("network-attempt", f"blocked socket target {address!r}")

    def _blocked_create_connection(self, address: Any, *args: Any, **kwargs: Any) -> Any:
        """Record and reject ``socket.create_connection``.

        Parameters
        ----------
        address:
            Attempted target.
        *args, **kwargs:
            Ignored socket arguments.

        Returns
        -------
        Any
            Never returns.
        """

        del args, kwargs
        self.observation.network_attempted = True
        self.observation.socket_targets.append(repr(address))
        raise PolicyViolation("network-attempt", f"blocked socket target {address!r}")

    def __enter__(self) -> PolicyObservation:
        """Activate every tripwire.

        Returns
        -------
        PolicyObservation
            Mutable observation populated by attempts.
        """

        setattr(builtins, "open", self._open)
        setattr(os, "open", self._os_open)
        setattr(socket.socket, "connect", self._blocked_connect_function)
        setattr(socket.socket, "connect_ex", self._blocked_connect_function)
        setattr(socket, "create_connection", self._blocked_create_connection_function)
        sys.meta_path.insert(0, self._import_blocker)
        return self.observation

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Restore patched process functions.

        Parameters
        ----------
        exc_type, exc_value, traceback:
            Context-manager exception state.
        """

        setattr(builtins, "open", self._original_open)
        setattr(os, "open", self._original_os_open)
        setattr(socket.socket, "connect", self._original_connect)
        setattr(socket.socket, "connect_ex", self._original_connect_ex)
        setattr(socket, "create_connection", self._original_create_connection)
        if self._import_blocker in sys.meta_path:
            sys.meta_path.remove(self._import_blocker)
        return None
