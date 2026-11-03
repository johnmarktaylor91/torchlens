"""Structural stdlib/builtin denial for resolved callables.

A resolved callable whose real top-level module is a Python standard-library or
builtin module is denied on the trusted foreign resolution path, except the
pure-forward ``operator`` roots. Split out of ``_callable_safety`` (which applies it);
stdlib-only leaf.
"""

from __future__ import annotations

import sys

# STRUCTURAL close of the denylist-completeness class (r31). The explicit
# ``_DENIED_MODULES`` denylist above is inherently incomplete: successive audits kept
# finding *more* resolvable stdlib/builtin gadgets (r26 exec/spawn; r28
# ``_frozen_importlib``; r30 ``_imp`` / ``zipimport`` / ``io`` / ``linecache`` /
# ``fileinput`` / ``py_compile`` / ``compileall`` / ``gc`` / ``mmap`` / ``fcntl`` ...).
# Chasing modules one at a time never terminates. Instead we close the CLASS with a
# POSITIVE structural rule: on the TRUSTED foreign resolution path, DENY any resolved
# callable whose REAL top-level module is a Python STANDARD-LIBRARY or BUILTIN module.
# The detector is the authoritative ``sys.stdlib_module_names`` (frozenset, py3.10+)
# UNION ``sys.builtin_module_names``. Trust means "run this user recipe", NEVER "import
# a stdlib module": a stdlib/builtin owner is never a legitimate user recipe, so it is
# denied EVEN under ``trust_custom_callables`` or an explicit module allowlist. The
# explicit ``_DENIED_MODULES`` denylist above is KEPT as belt-and-suspenders.
#
# CARVE-OUTS (must STILL resolve): the pure-forward ``operator`` root -- ``operator``
# is stdlib and ``operator.neg`` et al. resolve to the C accelerator ``_operator``
# (builtin) -- is admitted here so the legitimate ``operator:neg`` custom ref survives.
# The torch namespaces (torch / torch.Tensor / torch.nn.functional) are NOT stdlib, so
# they are naturally allowed and need no carve-out. First-party ``torchlens.*`` (incl.
# appliance ``torchlens.neuro`` / ``torchlens._notebook`` under explicit trust) is not
# stdlib either, so it is naturally allowed. A user's ``allowed_custom_callable_modules``
# entry that names a NON-stdlib user package still resolves; but a user CANNOT re-allow
# a stdlib module (an allowlist entry naming e.g. ``io`` stays DENIED -- that is the
# whole point of "denied even under trust").
_ALLOWED_STDLIB_ROOTS: frozenset[str] = frozenset({"operator", "_operator"})


def _compute_stdlib_and_builtin_top_level() -> frozenset[str]:
    """Return the top-level Python stdlib + builtin module-name detector set.

    Uses the authoritative ``sys.stdlib_module_names`` (py3.10+) UNION
    ``sys.builtin_module_names``. On pre-3.10 interpreters (no
    ``stdlib_module_names``) it falls back to ``sys.builtin_module_names`` UNION the
    top-level names enumerated from the standard-library directory; the explicit
    ``_DENIED_MODULES`` denylist remains the belt-and-suspenders floor on that path.
    """

    names: set[str] = set(sys.builtin_module_names)
    stdlib = getattr(sys, "stdlib_module_names", None)
    if stdlib is not None:
        names |= set(stdlib)
        return frozenset(names)
    # pre-3.10 fallback: enumerate top-level module/package names from the stdlib dir
    # (``os.__file__``'s directory is the pure stdlib root, NOT site-packages).
    try:
        import os

        std_dir = os.path.dirname(os.__file__ or "")
        if std_dir:
            for entry in os.listdir(std_dir):
                if entry.endswith(".py"):
                    names.add(entry[:-3])
                elif "." not in entry and not entry.startswith("_"):
                    names.add(entry)
    except OSError:  # pragma: no cover - defensive; stdlib dir is always readable here.
        pass
    return frozenset(names)


_STDLIB_AND_BUILTIN_TOP_LEVEL: frozenset[str] = _compute_stdlib_and_builtin_top_level()


def is_denied_stdlib_or_builtin_module(module: str) -> bool:
    """Return whether a REAL module identity is a denied stdlib / builtin module.

    Keyed on the TOP-LEVEL package of ``module`` (so submodules such as
    ``importlib.util`` or ``os.path`` are covered). The pure-forward ``operator`` /
    ``_operator`` carve-out is admitted (returns ``False``); everything else whose
    top-level name is in the stdlib/builtin detector set is denied. First-party
    (``torchlens.*``) and third-party (``torch`` / ``numpy`` / user packages) modules
    are not in the detector set and are therefore allowed by this gate.
    """

    if not module:
        return False
    top_level = module.split(".", 1)[0]
    if top_level in _ALLOWED_STDLIB_ROOTS:
        return False
    return top_level in _STDLIB_AND_BUILTIN_TOP_LEVEL
