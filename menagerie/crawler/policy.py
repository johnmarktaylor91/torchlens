"""Execution-phase offline environment and policy tripwires."""

from __future__ import annotations

import ast
import builtins
import importlib.abc
import importlib.machinery
import io
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any, IO, Literal, Mapping, Optional, Sequence, Union


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
_RUNTIME_SOURCE_SUFFIXES = frozenset(
    {
        ".a",
        ".c",
        ".cc",
        ".cpp",
        ".cu",
        ".cuh",
        ".dylib",
        ".h",
        ".hpp",
        ".py",
        ".pyc",
        ".pyd",
        ".pyi",
        ".pyx",
        ".so",
    }
)
_SPECIAL_READ_ROOTS = (Path("/dev"), Path("/proc"), Path("/sys"))
_SYSTEM_READ_FILES = frozenset(
    {
        Path("/etc/group"),
        Path("/etc/hosts"),
        Path("/etc/ld.so.cache"),
        Path("/etc/locale.alias"),
        Path("/etc/localtime"),
        Path("/etc/nsswitch.conf"),
        Path("/etc/passwd"),
        Path("/etc/resolv.conf"),
    }
)
_SAFE_INHERITED_KEYS = (
    "PATH",
    "PYTHONHOME",
    "LANG",
    "LC_ALL",
    "TZ",
    "SYSTEMROOT",
    "WINDIR",
    "TMPDIR",
)

SandboxKind = Literal["sandbox-exec", "bubblewrap"]


@dataclass(frozen=True)
class OperatingSystemSandbox:
    """A capability-probed OS sandbox executable.

    Parameters
    ----------
    kind:
        Closed sandbox implementation name.
    executable:
        Absolute path to the sandbox launcher.
    """

    kind: SandboxKind
    executable: str


class SandboxUnavailableError(RuntimeError):
    """Raised when worker execution has no usable OS isolation boundary."""


def _normalized_write_roots(write_roots: Sequence[Path]) -> tuple[Path, ...]:
    """Return sorted, resolved writable roots without redundant descendants.

    Parameters
    ----------
    write_roots:
        Candidate sandbox write roots.

    Returns
    -------
    tuple[Path, ...]
        Minimal deterministic set of writable roots.
    """

    resolved = sorted({path.resolve() for path in write_roots}, key=lambda path: str(path))
    roots: list[Path] = []
    for path in resolved:
        if not any(path == root or root in path.parents for root in roots):
            roots.append(path)
    return tuple(roots)


def generate_macos_sandbox_profile(
    write_roots: Sequence[Path],
    *,
    allowed_read_paths: Sequence[Path] = (),
    runtime_read_roots: Sequence[Path] = (),
) -> str:
    """Generate a deterministic sandbox-exec profile for offline execution.

    Parameters
    ----------
    write_roots:
        Sole filesystem roots where writes are permitted.
    allowed_read_paths:
        Exact source/input paths or directories permitted for data reads.
    runtime_read_roots:
        Environment/source roots restricted to executable-code file suffixes.

    Returns
    -------
    str
        Complete Seatbelt profile denying network, undeclared reads, and other writes.
    """

    roots = _normalized_write_roots(write_roots)
    lines = [
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny file-read-data)",
        "(deny file-write*)",
        '(allow file-read* (subpath "/System"))',
        '(allow file-read* (subpath "/usr/lib"))',
        '(allow file-read* (subpath "/Library/Apple"))',
        '(allow file-read* (subpath "/private/etc"))',
        '(allow file-read* (subpath "/dev"))',
        '(allow file-write* (literal "/dev/null"))',
    ]
    read_paths = tuple(dict.fromkeys(path.resolve() for path in (*roots, *allowed_read_paths)))
    for path in read_paths:
        encoded = json.dumps(str(path), ensure_ascii=True)
        lines.append(f"(allow file-read* (literal {encoded}))")
        if path in roots or path.is_dir():
            lines.append(f"(allow file-read* (subpath {encoded}))")
    runtime_suffixes = "a|c|cc|cpp|cu|cuh|dylib|h|hpp|metallib|py|pyc|pyd|pyi|pyx|so"
    for root in tuple(dict.fromkeys(path.resolve() for path in runtime_read_roots)):
        pattern = f"^{re.escape(str(root))}/.*\\.(?:{runtime_suffixes})$"
        lines.append(f"(allow file-read* (regex #{json.dumps(pattern)}))")
    for root in roots:
        encoded = json.dumps(str(root), ensure_ascii=True)
        lines.append(f"(allow file-write* (literal {encoded}))")
        lines.append(f"(allow file-write* (subpath {encoded}))")
    return "\n".join(lines) + "\n"


def _probe_command(argv: Sequence[str]) -> bool:
    """Return whether a sandbox capability probe exits successfully.

    Parameters
    ----------
    argv:
        Exact probe command.

    Returns
    -------
    bool
        True only for a prompt zero exit.
    """

    try:
        completed = subprocess.run(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _python_probe_argv() -> tuple[str, ...]:
    """Return a minimal Python startup command for sandbox probes.

    Returns
    -------
    tuple[str, ...]
        Isolated interpreter command that also verifies OS randomness access.
    """

    return (sys.executable, "-I", "-S", "-c", "import os; os.urandom(1)")


def _linux_compute_devices() -> tuple[Path, ...]:
    """Return explicit accelerator device paths needed by Linux workers.

    Returns
    -------
    tuple[Path, ...]
        Existing CUDA/DRM compute device paths in deterministic order.
    """

    candidates = {
        *Path("/dev").glob("nvidia*"),
        Path("/dev/dri"),
        Path("/dev/kfd"),
    }
    return tuple(sorted((path for path in candidates if path.exists()), key=lambda path: str(path)))


def _linux_dynamic_runtime_files(executable: Path) -> tuple[Path, ...]:
    """Return exact host runtime files required to start one Linux executable.

    Parameters
    ----------
    executable:
        Interpreter executable that will run inside the minimal namespace.

    Returns
    -------
    tuple[Path, ...]
        Existing absolute ELF loader and shared-library paths reported by ``ldd``.

    Raises
    ------
    SandboxUnavailableError
        If the runtime dependency inventory cannot be obtained safely.
    """

    ldd = shutil.which("ldd")
    if ldd is None:
        raise SandboxUnavailableError("Linux runtime dependency inventory is unavailable")
    try:
        completed = subprocess.run(
            (ldd, str(executable)),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=5,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SandboxUnavailableError("Linux runtime dependency inventory failed") from exc
    if completed.returncode != 0:
        raise SandboxUnavailableError("Linux runtime dependency inventory was incomplete")
    paths: list[Path] = []
    for line in completed.stdout.splitlines():
        match = re.search(r"(?:=>\s+)?(/[^\s]+)\s+\(0x[0-9a-fA-F]+\)", line)
        if match is None:
            continue
        path = Path(match.group(1))
        if not path.is_file():
            raise SandboxUnavailableError(f"missing Linux runtime dependency: {path}")
        paths.append(path)
    if not paths:
        raise SandboxUnavailableError("Linux runtime dependency inventory was empty")
    return tuple(dict.fromkeys(paths))


def _linux_environment_prefix(executable: Path) -> Path:
    """Return the closed interpreter/environment prefix for a Linux executable.

    Parameters
    ----------
    executable:
        Worker interpreter path.

    Returns
    -------
    pathlib.Path
        Conventional environment prefix containing the interpreter.

    Raises
    ------
    SandboxUnavailableError
        If the executable is absent or has no bounded environment prefix.
    """

    resolved = executable.resolve()
    if not resolved.is_file():
        raise SandboxUnavailableError("worker interpreter is unavailable")
    prefix = (
        resolved.parent.parent if resolved.parent.name in {"bin", "Scripts"} else resolved.parent
    )
    if prefix == Path("/"):
        raise SandboxUnavailableError("worker interpreter prefix would expose the host root")
    return prefix


def _linux_minimal_read_mounts(
    argv: Sequence[str], cwd: Path, allowed_read_paths: Sequence[Path]
) -> tuple[Path, ...]:
    """Build the minimal read-only Linux namespace mount inventory.

    Parameters
    ----------
    argv:
        Original child command whose first entry is the worker interpreter.
    cwd:
        Parent-verified source root used by the child.
    allowed_read_paths:
        Exact request, adapter, and declared-input paths.

    Returns
    -------
    tuple[Path, ...]
        Minimal environment, verified-source, declared-input, and ELF runtime mounts.

    Raises
    ------
    SandboxUnavailableError
        If any required mount is missing or would expose the host root.
    """

    if not argv:
        raise SandboxUnavailableError("worker command is empty")
    executable = Path(argv[0]).resolve()
    source_root = cwd.resolve()
    candidates = [
        _linux_environment_prefix(executable),
        source_root,
        *allowed_read_paths,
        *_linux_dynamic_runtime_files(executable),
    ]
    normalized: list[Path] = []
    for candidate in candidates:
        path = candidate.absolute()
        if path == Path("/"):
            raise SandboxUnavailableError("minimal namespace refuses a host-root read mount")
        if not path.exists():
            continue
        if any(path == root or root in path.parents for root in normalized):
            continue
        normalized = [root for root in normalized if not (root == path or path in root.parents)]
        normalized.append(path)
    if source_root not in normalized and not any(
        source_root == root or root in source_root.parents for root in normalized
    ):
        raise SandboxUnavailableError("verified source root is unavailable")
    return tuple(normalized)


def _bubblewrap_argv(
    executable: str,
    argv: Sequence[str],
    cwd: Path,
    write_roots: Sequence[Path],
    allowed_read_paths: Sequence[Path],
) -> tuple[str, ...]:
    """Build a bubblewrap command with a default-invisible host filesystem.

    Parameters
    ----------
    executable:
        Bubblewrap executable.
    argv:
        Original worker command.
    cwd:
        Parent-verified source root.
    write_roots:
        Sole writable scratch and result roots.
    allowed_read_paths:
        Exact declared source/input paths outside the source root.

    Returns
    -------
    tuple[str, ...]
        Complete minimal-filesystem bubblewrap command.
    """

    wrapped: list[str] = [
        executable,
        "--unshare-net",
        "--unshare-ipc",
        "--unshare-pid",
        "--die-with-parent",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
    ]
    for path in _linux_minimal_read_mounts(argv, cwd, allowed_read_paths):
        wrapped.extend(("--ro-bind", str(path), str(path)))
    for device in _linux_compute_devices():
        wrapped.extend(("--dev-bind", str(device), str(device)))
    wrapped.extend(("--remount-ro", "/dev"))
    for root in _normalized_write_roots(write_roots):
        wrapped.extend(("--bind", str(root), str(root)))
    wrapped.extend(("--remount-ro", "/"))
    wrapped.extend(("--chdir", str(cwd.resolve()), "--", *argv))
    return tuple(wrapped)


def _probe_bubblewrap(executable: str) -> bool:
    """Return whether bubblewrap can create the required namespaces and mounts.

    Parameters
    ----------
    executable:
        Bubblewrap executable.

    Returns
    -------
    bool
        True when the complete boundary can launch a no-op.
    """

    try:
        with tempfile.TemporaryDirectory(prefix="menagerie-bwrap-probe-") as temporary:
            root = Path(temporary).resolve()
            return _probe_command(
                _bubblewrap_argv(
                    executable,
                    _python_probe_argv(),
                    root,
                    (root,),
                    (),
                )
            )
    except (OSError, SandboxUnavailableError):
        return False


def _probe_sandbox_exec(executable: str) -> bool:
    """Return whether macOS sandbox-exec accepts the generated profile.

    Parameters
    ----------
    executable:
        sandbox-exec executable.

    Returns
    -------
    bool
        True when Seatbelt launches a no-op under the profile.
    """

    profile = generate_macos_sandbox_profile(
        (),
        allowed_read_paths=(Path(sys.executable),),
        runtime_read_roots=(Path(sys.prefix), Path.cwd()),
    )
    return _probe_command((executable, "-p", profile, *_python_probe_argv()))


@lru_cache(maxsize=4)
def detect_os_sandbox(system_name: Optional[str] = None) -> Optional[OperatingSystemSandbox]:
    """Detect a working fail-closed OS sandbox for the current platform.

    Parameters
    ----------
    system_name:
        Optional platform override used by deterministic tests.

    Returns
    -------
    OperatingSystemSandbox | None
        First working preferred sandbox, or ``None`` when execution must fail closed.
    """

    detected_system = platform.system() if system_name is None else system_name
    if detected_system == "Darwin":
        sandbox_exec = shutil.which("sandbox-exec")
        audit_log = shutil.which("log")
        if sandbox_exec is not None and audit_log is not None and _probe_sandbox_exec(sandbox_exec):
            return OperatingSystemSandbox("sandbox-exec", sandbox_exec)
        return None
    if detected_system != "Linux":
        return None
    bubblewrap = shutil.which("bwrap")
    if bubblewrap is not None and _probe_bubblewrap(bubblewrap):
        return OperatingSystemSandbox("bubblewrap", bubblewrap)
    # The legacy unshare fallback cannot construct a default-invisible root without
    # an additional pivot-root helper. Refuse it instead of exposing the host read-only.
    return None


def wrap_with_os_sandbox(
    sandbox: OperatingSystemSandbox,
    argv: Sequence[str],
    cwd: Path,
    write_roots: Sequence[Path],
    *,
    macos_profile_path: Optional[Path] = None,
    allowed_read_paths: Sequence[Path] = (),
) -> tuple[str, ...]:
    """Wrap a child command in a capability-probed OS sandbox.

    Parameters
    ----------
    sandbox:
        Selected sandbox implementation.
    argv:
        Original child argv.
    cwd:
        Read-only child working directory.
    write_roots:
        Sole writable scratch/result roots.
    macos_profile_path:
        Generated profile path required by sandbox-exec.
    allowed_read_paths:
        Exact source/input paths exposed inside a minimal Linux namespace.

    Returns
    -------
    tuple[str, ...]
        Wrapped command suitable for ``subprocess.Popen``.

    Raises
    ------
    SandboxUnavailableError
        If the selected implementation lacks required launch data.
    """

    roots = _normalized_write_roots(write_roots)
    if sandbox.kind == "sandbox-exec":
        if macos_profile_path is None:
            raise SandboxUnavailableError("sandbox-exec profile path is required")
        return (sandbox.executable, "-f", str(macos_profile_path), *argv)
    if sandbox.kind == "bubblewrap":
        return _bubblewrap_argv(
            sandbox.executable,
            argv,
            cwd,
            roots,
            allowed_read_paths,
        )
    raise SandboxUnavailableError("unsupported OS sandbox implementation")


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
            "PYTHONDONTWRITEBYTECODE": "1",
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
    allowed_read_paths:
        Explicit source and input files or directories authorized by the parent request.
    """

    def __init__(
        self,
        scratch_root: Path,
        *additional_write_roots: Path,
        allowed_read_paths: Sequence[Path] = (),
    ) -> None:
        """Initialize inactive tripwires.

        Parameters
        ----------
        scratch_root:
            Primary writable root.
        *additional_write_roots:
            Explicit additional writable roots.
        allowed_read_paths:
            Exact source/input paths that model construction may read.
        """

        self.allowed_roots = tuple(
            path.resolve() for path in (scratch_root, *additional_write_roots)
        )
        self.allowed_read_paths = tuple(path.resolve() for path in allowed_read_paths)
        self.environment_search_paths = frozenset(
            Path(value).resolve() for value in sys.path if isinstance(value, str) and value
        )
        self.environment_roots = tuple(
            dict.fromkeys((Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()))
        )
        self.observation = PolicyObservation(
            credentials_present=any(_contains_credential_name(name) for name in os.environ)
        )
        self._original_open = builtins.open
        self._original_io_open = io.open
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

    def _read_path_allowed(
        self, value: Union[str, bytes, os.PathLike[str], os.PathLike[bytes]]
    ) -> bool:
        """Return whether a read is explicit input/source or runtime support.

        Parameters
        ----------
        value:
            Filesystem path opened for reading.

        Returns
        -------
        bool
            True only for the closed read allowlist.
        """

        candidate = Path(os.fsdecode(value)).resolve()
        if candidate == Path(os.devnull).resolve() or candidate in _SYSTEM_READ_FILES:
            return True
        if any(candidate == root or root in candidate.parents for root in _SPECIAL_READ_ROOTS):
            return True
        if any(
            candidate == allowed or allowed in candidate.parents
            for allowed in self.allowed_read_paths
        ):
            return True
        if candidate in self.environment_search_paths:
            return True
        if any(candidate == root or root in candidate.parents for root in self.environment_roots):
            return True
        if any(candidate == root or root in candidate.parents for root in self.allowed_roots):
            return True
        if any(part.endswith((".dist-info", ".egg-info")) for part in candidate.parts):
            return True
        name = candidate.name.lower()
        if name.startswith("__editable__.") and name.endswith(".__path_hook__"):
            return True
        if name.startswith("python") and name.endswith(".zip") and "lib" in candidate.parts:
            return True
        return candidate.suffix.lower() in _RUNTIME_SOURCE_SUFFIXES or ".so." in name

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
        if not writing and not self._read_path_allowed(value):
            self.observation.checkpoint_or_weight_read_attempted = True
            self.observation.checkpoint_paths.append(path_text)
            raise PolicyViolation(
                "checkpoint-read", f"blocked undeclared model-data read: {path_text}"
            )
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
        setattr(io, "open", self._open)
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
        setattr(io, "open", self._original_io_open)
        setattr(os, "open", self._original_os_open)
        setattr(socket.socket, "connect", self._original_connect)
        setattr(socket.socket, "connect_ex", self._original_connect_ex)
        setattr(socket, "create_connection", self._original_create_connection)
        if self._import_blocker in sys.meta_path:
            sys.meta_path.remove(self._import_blocker)
        return None
