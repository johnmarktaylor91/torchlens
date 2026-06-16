"""Coverage-preservation tests for detached torch reference patching."""

import sys
import types
from typing import Any

from torchlens import _state
from torchlens.backends.torch._tl import is_decorated_function
from torchlens.backends.torch.wrappers import (
    clear_patch_detached_references_cache,
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
