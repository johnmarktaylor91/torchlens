"""Parse the torch version a saved artifact recorded (never the live runtime).

Runnable compatibility metadata carries the producer's torch version string; alias
bounds compare its major/minor prefix. The live runtime is feature-detected in
``_torch_compat``, never version-parsed.
"""

from __future__ import annotations


def _torch_minor_version(version: str | None) -> tuple[int, int] | None:
    """Parse a torch major/minor prefix without inspecting the live runtime.

    Parameters
    ----------
    version:
        Producer torch version from runnable compatibility metadata.

    Returns
    -------
    tuple[int, int] | None
        Major/minor pair, or ``None`` when producer metadata is absent or not
        parseable (for example a legacy placeholder).
    """

    if version is None:
        return None
    components = version.split("+", maxsplit=1)[0].split(".")
    if len(components) < 2 or not components[0].isdigit() or not components[1].isdigit():
        return None
    return int(components[0]), int(components[1])
