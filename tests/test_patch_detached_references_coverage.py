"""Coverage-preservation tests for detached torch reference patching."""

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import torch

from torchlens import _state
from torchlens.backends.torch._tl import is_decorated_function
from torchlens.backends.torch.wrappers import (
    clear_patch_detached_references_cache,
    patch_detached_references,
    unwrap_torch,
    wrap_torch,
)


def test_patch_detached_references_rewrites_non_torch_detached_refs() -> None:
    """Patch module, class, and function-default refs in a non-torch module."""

    mod_name = "_tl_adversarial_non_torch_detached_refs"
    sys.modules.pop(mod_name, None)
    unwrap_torch()
    clear_patch_detached_references_cache()

    mod = types.ModuleType(mod_name)
    sys.modules[mod_name] = mod
    try:
        exec(
            """
from typing import Any
from torch import relu, sigmoid, tanh

module_level_ref = relu


class Holder:
    class_level_ref = sigmoid


def uses_default(x: Any, op: Any = tanh) -> Any:
    \"\"\"Apply the default torch function reference.\"\"\"

    return op(x)
""",
            mod.__dict__,
        )
        original_module_ref = mod.module_level_ref
        original_class_ref = mod.Holder.class_level_ref
        original_default_ref = mod.uses_default.__defaults__[0]

        wrap_torch()

        expected_module_ref: Any = _state._orig_to_decorated[id(original_module_ref)]
        expected_class_ref: Any = _state._orig_to_decorated[id(original_class_ref)]
        expected_default_ref: Any = _state._orig_to_decorated[id(original_default_ref)]

        assert mod.module_level_ref is expected_module_ref
        assert mod.Holder.class_level_ref is expected_class_ref
        assert mod.uses_default.__defaults__ == (expected_default_ref,)
        assert is_decorated_function(mod.module_level_ref)
        assert is_decorated_function(mod.Holder.class_level_ref)
        assert is_decorated_function(mod.uses_default.__defaults__[0])
    finally:
        sys.modules.pop(mod_name, None)
        clear_patch_detached_references_cache()
        wrap_torch()


def _import_temp_module(tmp_path: Path, mod_name: str, source: str) -> types.ModuleType:
    """Import a temporary module from source text.

    Parameters
    ----------
    tmp_path:
        Directory where the module file should be created.
    mod_name:
        Temporary module name.
    source:
        Python source code to write.

    Returns
    -------
    types.ModuleType
        Imported module object.
    """

    module_path = tmp_path / f"{mod_name}.py"
    module_path.write_text(source, encoding="utf-8")
    sys.modules.pop(mod_name, None)
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        return importlib.import_module(mod_name)
    finally:
        sys.path.remove(str(tmp_path))


def test_default_policy_preserves_source_level_detached_refs(tmp_path: Path) -> None:
    """Default source prefilter should still patch source-level torch refs."""

    mod_name = "_tl_source_level_detached_refs"
    unwrap_torch()
    clear_patch_detached_references_cache()
    mod = _import_temp_module(
        tmp_path,
        mod_name,
        """
from typing import Any
from torch import relu, sigmoid, tanh

module_level_ref = relu


class Holder:
    class_level_ref = sigmoid


def uses_default(x: Any, op: Any = tanh) -> Any:
    \"\"\"Apply the default torch function reference.\"\"\"

    return op(x)
""",
    )
    try:
        original_module_ref = mod.module_level_ref
        original_class_ref = mod.Holder.class_level_ref
        original_default_ref = mod.uses_default.__defaults__[0]

        wrap_torch()

        assert mod.module_level_ref is _state._orig_to_decorated[id(original_module_ref)]
        assert mod.Holder.class_level_ref is _state._orig_to_decorated[id(original_class_ref)]
        assert mod.uses_default.__defaults__ == (
            _state._orig_to_decorated[id(original_default_ref)],
        )
        assert is_decorated_function(mod.module_level_ref)
        assert is_decorated_function(mod.Holder.class_level_ref)
        assert is_decorated_function(mod.uses_default.__defaults__[0])
    finally:
        sys.modules.pop(mod_name, None)
        clear_patch_detached_references_cache()
        wrap_torch()


def test_default_policy_skips_torch_free_deep_scan_and_full_patches(
    tmp_path: Path,
) -> None:
    """Torch-free source should skip Level 2/3 by default and patch with full=True."""

    mod_name = "_tl_torch_free_dynamic_detached_ref"
    unwrap_torch()
    clear_patch_detached_references_cache()
    mod = _import_temp_module(
        tmp_path,
        mod_name,
        """
class Holder:
    pass
""",
    )
    try:
        original_relu = torch.relu
        mod.Holder.class_level_ref = original_relu

        wrap_torch()

        assert vars(mod.Holder)["class_level_ref"] is original_relu
        clear_patch_detached_references_cache()
        patch_detached_references(full=True)

        patched_ref = vars(mod.Holder)["class_level_ref"]
        assert patched_ref is _state._orig_to_decorated[id(original_relu)]
        assert is_decorated_function(patched_ref)
    finally:
        sys.modules.pop(mod_name, None)
        clear_patch_detached_references_cache()
        wrap_torch()
