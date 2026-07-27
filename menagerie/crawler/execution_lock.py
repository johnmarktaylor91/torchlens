"""Host-global execution serialization for crawler model workers."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import pwd
import stat
from typing import Optional


def global_execution_flock_path() -> Path:
    """Return the one execution lock path shared by every campaign clone.

    Returns
    -------
    pathlib.Path
        Operator-account-global lock path outside every campaign runtime root.

    The path is derived from the kernel user identity rather than ``HOME`` or a
    campaign environment variable. Separate launchd environments and clones
    therefore cannot accidentally select different locks.
    """

    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    return account_home / ".cache" / "torchlens" / "menagerie-crawler" / "execution.flock"


def acquire_global_execution_flock(path: Optional[Path] = None) -> int:
    """Acquire the blocking host-global execution flock.

    The returned descriptor must remain open for the full worker lifetime. Production
    callers also inherit it into the worker process, so an abrupt driver death cannot
    release the slot while its detached child is still running.

    Parameters
    ----------
    path:
        Explicit test/operator path. The production default is shared across clones.

    Returns
    -------
    int
        Open descriptor owning the exclusive kernel flock.

    Raises
    ------
    OSError
        If the lock path cannot be created safely or the kernel lock cannot be acquired.
    """

    lock_path = (path or global_execution_flock_path()).resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o600)
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise OSError(f"global execution flock is not a regular file: {lock_path}")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def release_global_execution_flock(descriptor: Optional[int]) -> None:
    """Release one host-global execution descriptor.

    Parameters
    ----------
    descriptor:
        Descriptor returned by :func:`acquire_global_execution_flock`, or ``None``.
    """

    if descriptor is None:
        return
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)
