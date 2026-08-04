"""Closed declarative R1 recipes and audited typed adapter loading."""

from __future__ import annotations

import ast
import importlib
import importlib.machinery
import importlib.metadata
import inspect
import re
import sys
from dataclasses import dataclass, field as dataclass_field
from functools import lru_cache
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, Callable, Mapping, Optional, Sequence, Union

from menagerie.crawler.identity import canonical_json_bytes, compute_recipe_revision, hash_bytes


class RecipeError(ValueError):
    """Raised when a recipe is opaque, executable text, or violates its typed contract."""


BuildModel = Callable[[], object]
MakeDummyCall = Callable[[int, str], tuple[tuple[object, ...], dict[str, object]]]

CONSTRUCT_NODE_KEY = "__construct__"
"""Marker key that turns a declarative kwargs mapping into a construct node."""

MAX_CONSTRUCT_DEPTH = 4
"""Maximum construct-node nesting depth admitted by the declarative grammar."""

MAX_CONSTRUCT_NODES = 16
"""Maximum total construct nodes admitted by one declarative recipe."""

MAX_POST_CONSTRUCT_CALLS = 8
"""Maximum bounded post-construction configuration calls per recipe."""

_GENERIC_CONTAINER_SYMBOLS = frozenset(
    {"Sequential", "ModuleList", "ModuleDict", "ParameterList", "ParameterDict", "Module"}
)

CONSTRUCT_MODULE_ALLOWLIST = frozenset({"torch", "torch.nn", "torch.nn.utils.rnn"})
"""Exact auxiliary modules a construct node may name besides the pinned distribution.

A construct node's ``module`` is imported and its ``symbol`` is called, so the set
of reachable callables *is* the grammar's real bound. Without this the grammar is
a bounded syntax over an unbounded callable namespace: ``subprocess.run`` parses,
imports, and executes before any downstream type error is reached.

The bound is therefore "the artifact the recipe pins, plus a named auxiliary
allowlist". Membership is EXACT module names, not prefixes, and is derived from
the shapes that actually occur: ``torch`` (``torch.zeros``), ``torch.nn``
(``TransformerEncoderLayer``), and ``torch.nn.utils.rnn``
(``pack_padded_sequence``). Exactness keeps the auxiliary surface minimal --
``torch.hub``, ``torch.jit``, and ``torch.utils.collect_env`` all expose
general-purpose code/state loaders and none of them are on this list.

Extend this deliberately, one module at a time, with a test that names the real
recipe shape requiring it. Widening it to a top-level package prefix would hand
every construct node the whole of torch again.
"""


def _is_generic_container_reference(module: str, symbol: str) -> bool:
    """Return whether a declared module/symbol names a generic torch container.

    Parameters
    ----------
    module, symbol:
        Declared import module and constructor attribute.

    Returns
    -------
    bool
        True when the reference names a composition container rather than a
        published architecture entrypoint.
    """

    return symbol in _GENERIC_CONTAINER_SYMBOLS and (
        module == "torch.nn" or module.startswith("torch.nn.")
    )


@lru_cache(maxsize=None)
def _distribution_top_level_packages(distribution: str) -> frozenset[str]:
    """Return the top-level import packages the named distribution installs.

    This inverts the interpreter's package-to-distribution map so a distribution
    whose import package differs from its distribution name (``opencv-python``
    ships ``cv2``) still resolves without importing anything.

    Parameters
    ----------
    distribution:
        Declared pinned distribution name.

    Returns
    -------
    frozenset[str]
        Top-level package names attributed to the distribution, empty when the
        distribution is not installed in this interpreter.
    """

    wanted = canonical_distribution_name(distribution)
    return frozenset(
        package
        for package, names in _packages_distributions_snapshot().items()
        if any(canonical_distribution_name(name) == wanted for name in names)
    )


def assert_construct_namespace(module: str, *, distribution: str, context: str) -> None:
    """Refuse a construct-node module outside the pinned distribution and allowlist.

    This runs BEFORE the module is imported. Import order matters: importing an
    author-named module already executes that module's top-level code, so a
    post-import check would be a check applied after the thing it exists to
    prevent. The declared string is the only artifact available early enough.

    Two namespaces are admitted. The pinned distribution's own packages are
    admitted because that distribution is exactly what the recipe pins, versions,
    and digests -- running its code is the point of the R1 rung. Everything else
    must be an exact member of :data:`CONSTRUCT_MODULE_ALLOWLIST`.

    Parameters
    ----------
    module:
        Declared dotted module name of the construct node.
    distribution:
        Declared pinned distribution of the enclosing recipe.
    context:
        Location used in refusal messages.

    Raises
    ------
    RecipeError
        If the module belongs to neither admitted namespace.
    """

    if module in CONSTRUCT_MODULE_ALLOWLIST:
        return
    top_level = module.split(".")[0]
    if canonical_distribution_name(top_level) == canonical_distribution_name(distribution):
        return
    if top_level in _distribution_top_level_packages(distribution):
        return
    raise RecipeError(
        f"construct node at {context} names module {module!r}, which is outside the "
        f"pinned distribution {distribution!r} and the allowlisted modules "
        f"{sorted(CONSTRUCT_MODULE_ALLOWLIST)!r}"
    )


def _reject_container_instance(value: object, description: str) -> None:
    """Refuse a materialized object whose type is a generic torch container.

    The root model and every construct-node result are materialized the same way,
    so they get the same instance check. Checking only the root leaves the exact
    asymmetry composition laundering needs: a returned-object check on the root
    and a constructor-symbol-only check on everything the root is built from.

    Parameters
    ----------
    value:
        Materialized object.
    description:
        Leading refusal clause naming what constructed the object.

    Raises
    ------
    RecipeError
        If the object's type is one of the generic torch containers.
    """

    value_type = type(value)
    module_name = str(getattr(value_type, "__module__", "") or "")
    if module_name.split(".")[0] != "torch":
        return
    import torch

    containers: tuple[object, ...] = (
        torch.nn.Sequential,
        torch.nn.ModuleList,
        torch.nn.ModuleDict,
        torch.nn.ParameterList,
        torch.nn.ParameterDict,
        torch.nn.Module,
    )
    if any(value_type is container for container in containers):
        raise RecipeError(
            f"{description} {module_name}.{value_type.__qualname__}; "
            "a composed container cannot claim R1"
        )


def _reject_container_constructor(constructor: object, context: str) -> None:
    """Refuse a resolved constructor whose identity is a generic torch container.

    Name-based parse rules are defeated by re-export laundering, so this check
    compares the resolved object identity against the actual torch container
    classes whenever the resolved constructor originates from torch.

    Parameters
    ----------
    constructor:
        Resolved callable constructor.
    context:
        Human-readable reference used in the refusal.

    Raises
    ------
    RecipeError
        If the constructor is one of the generic torch.nn containers.
    """

    module_name = str(getattr(constructor, "__module__", "") or "")
    if module_name.split(".")[0] != "torch":
        return
    import torch

    containers: tuple[object, ...] = (
        torch.nn.Sequential,
        torch.nn.ModuleList,
        torch.nn.ModuleDict,
        torch.nn.ParameterList,
        torch.nn.ParameterDict,
        torch.nn.Module,
    )
    if any(constructor is container for container in containers):
        raise RecipeError(
            f"generic torch.nn containers are refused in declarative R1 recipes: {context}"
        )


def _validate_construct_node_spec(
    spec: Any, context: str, *, distribution: str
) -> Mapping[str, Any]:
    """Validate the exact three-field construct-node payload.

    Parameters
    ----------
    spec:
        Candidate ``{"module", "symbol", "kwargs"}`` payload.
    context:
        Location used in refusal messages.
    distribution:
        Pinned distribution bounding the construct node's callable namespace.
        Keyword-only and required so no call site can resolve a construct node
        against an unbounded namespace by omission.

    Returns
    -------
    Mapping[str, Any]
        The validated construct payload.

    Raises
    ------
    RecipeError
        If the payload deviates from the closed construct-node shape or names a
        module outside the bounded namespace.
    """

    if not isinstance(spec, Mapping) or set(spec) != {"module", "symbol", "kwargs"}:
        raise RecipeError(
            f"construct node at {context} must declare exactly module, symbol, and kwargs"
        )
    module = spec["module"]
    symbol = spec["symbol"]
    if not isinstance(module, str) or not all(
        part.isidentifier() for part in module.split(".")
    ):
        raise RecipeError(f"construct node at {context} module must be a dotted identifier")
    if not isinstance(symbol, str) or not symbol.isidentifier():
        raise RecipeError(f"construct node at {context} symbol must be a direct identifier")
    if _is_generic_container_reference(module, symbol):
        raise RecipeError(
            "generic torch.nn containers are refused in declarative R1 recipes: "
            f"{module}.{symbol}"
        )
    assert_construct_namespace(module, distribution=distribution, context=context)
    kwargs = spec["kwargs"]
    if not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
        raise RecipeError(f"construct node at {context} kwargs must be a string-keyed mapping")
    return spec


def _walk_construct_values(
    value: Any, *, context: str, depth: int, counter: list[int], distribution: str
) -> None:
    """Recursively validate construct nodes inside declarative JSON values.

    Parameters
    ----------
    value:
        JSON-compatible declarative value.
    context:
        Location used in refusal messages.
    depth:
        Number of enclosing construct nodes.
    counter:
        One-slot mutable total construct-node count.
    distribution:
        Pinned distribution bounding every construct node's callable namespace.

    Raises
    ------
    RecipeError
        If a construct node is malformed, too deep, too numerous, or names a
        module outside the bounded namespace.
    """

    if isinstance(value, Mapping):
        if CONSTRUCT_NODE_KEY in value:
            if set(value) != {CONSTRUCT_NODE_KEY}:
                raise RecipeError(
                    f"construct node at {context} must carry only the "
                    f"{CONSTRUCT_NODE_KEY!r} key"
                )
            if depth + 1 > MAX_CONSTRUCT_DEPTH:
                raise RecipeError(
                    f"construct graph exceeds the maximum depth of {MAX_CONSTRUCT_DEPTH}"
                )
            counter[0] += 1
            if counter[0] > MAX_CONSTRUCT_NODES:
                raise RecipeError(
                    f"construct graph exceeds the maximum of {MAX_CONSTRUCT_NODES} nodes"
                )
            spec = _validate_construct_node_spec(
                value[CONSTRUCT_NODE_KEY], context, distribution=distribution
            )
            for name, child in spec["kwargs"].items():
                _walk_construct_values(
                    child,
                    context=f"{context}.{name}",
                    depth=depth + 1,
                    counter=counter,
                    distribution=distribution,
                )
            return
        for name, child in value.items():
            _walk_construct_values(
                child,
                context=f"{context}.{name}",
                depth=depth,
                counter=counter,
                distribution=distribution,
            )
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_construct_values(
                child,
                context=f"{context}[{index}]",
                depth=depth,
                counter=counter,
                distribution=distribution,
            )


def _reject_construct_nodes(value: Any, context: str) -> None:
    """Refuse construct nodes anywhere inside a plain-JSON-only value.

    Parameters
    ----------
    value:
        JSON-compatible value that must stay free of construct nodes.
    context:
        Location used in refusal messages.

    Raises
    ------
    RecipeError
        If any nested mapping carries the construct marker key.
    """

    if isinstance(value, Mapping):
        if CONSTRUCT_NODE_KEY in value:
            raise RecipeError(f"{context} must be plain JSON without construct nodes")
        for name, child in value.items():
            _reject_construct_nodes(child, f"{context}.{name}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_construct_nodes(child, f"{context}[{index}]")


@lru_cache(maxsize=1)
def _packages_distributions_snapshot() -> Mapping[str, tuple[str, ...]]:
    """Return the interpreter's frozen top-level-package to distribution map.

    Returns
    -------
    Mapping[str, tuple[str, ...]]
        Immutable copy of ``importlib.metadata.packages_distributions()``. The
        environment cannot change inside one worker process, so one snapshot is
        both correct and cheap.
    """

    return {
        name: tuple(values)
        for name, values in importlib.metadata.packages_distributions().items()
    }


def assert_model_provenance(model: object, distribution: str) -> None:
    """Fail closed unless the constructed model's class is defined by the pinned distribution.

    Parse-time grammar rules bound what a declarative recipe can *express*; this
    runtime tripwire bounds what is actually *constructed*. It closes composition
    laundering: a model assembled from generic primitives, or resolved through a
    factory into another package's class, cannot carry the R1 label of the
    declared distribution.

    Parameters
    ----------
    model:
        Constructed native model object.
    distribution:
        Declared pinned distribution name.

    Raises
    ------
    RecipeError
        If the model type is a generic container, its defining module cannot be
        attributed to any installed distribution, or the attribution does not
        include the declared distribution.
    """

    model_type = type(model)
    module_name = str(getattr(model_type, "__module__", "") or "")
    qualified = f"{module_name}.{model_type.__qualname__}"
    top_level = module_name.split(".")[0]
    _reject_container_instance(model, "declarative recipe constructed a generic container")
    if not top_level or top_level in {"builtins", "__main__"}:
        raise RecipeError(
            f"constructed model type {qualified} has no importable defining module "
            "and cannot claim R1"
        )
    wanted = canonical_distribution_name(distribution)
    attributed = _packages_distributions_snapshot().get(top_level, ())
    if not any(canonical_distribution_name(name) == wanted for name in attributed):
        described = sorted(attributed) if attributed else "no installed distribution"
        raise RecipeError(
            f"constructed model type {qualified} is not defined by the pinned "
            f"distribution {distribution!r}: attributed to {described}"
        )


def _is_disabling_pretrained_value(value: Any) -> bool:
    """Return whether a declarative value explicitly disables pretrained assets.

    Parameters
    ----------
    value:
        JSON-compatible constructor value.

    Returns
    -------
    bool
        True only for conventional explicit opt-out values.
    """

    return (
        value is None
        or value is False
        or (isinstance(value, str) and value.strip().lower() in {"", "none", "random"})
    )


def validate_pretrained_disable_fields(
    kwargs: Mapping[str, Any], disable_fields: Sequence[str]
) -> None:
    """Validate explicit pretrained opt-outs against constructor keyword values.

    Parameters
    ----------
    kwargs:
        Complete declarative constructor keyword mapping.
    disable_fields:
        Fields claimed to disable pretrained assets.

    Raises
    ------
    RecipeError
        If a field is duplicated, absent, or does not carry a disabling value.
    """

    if len(set(disable_fields)) != len(disable_fields):
        raise RecipeError("pretrained_disable_fields must not contain duplicates")
    for field in disable_fields:
        if field not in kwargs:
            raise RecipeError(
                f"pretrained disable field {field!r} is absent from constructor kwargs"
            )
        if not _is_disabling_pretrained_value(kwargs[field]):
            raise RecipeError(
                f"pretrained disable field {field!r} does not carry a disabling value"
            )


_HASH_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def canonical_distribution_name(value: str) -> str:
    """Return one comparable package name for environment-inventory lookup.

    Parameters
    ----------
    value:
        Declared distribution name.

    Returns
    -------
    str
        Case-folded name with ``_``/``.`` normalized to ``-``.
    """

    return value.strip().casefold().replace("_", "-").replace(".", "-")


def resolve_environment_artifact_digest(
    packages: Sequence[Mapping[str, Any]],
    *,
    distribution: str,
    version: str,
) -> Optional[str]:
    """Derive the installed-distribution artifact digest from an exact inventory.

    The author stage has no package inventory, no environment identity, and no
    interpreter, so it cannot know this digest; the routed environment's exact
    resolved export does. This resolves the one row naming ``distribution`` and
    returns its recorded artifact digest.

    Parameters
    ----------
    packages:
        Exact ``name``/``version``/``sha256`` rows from the routed intent's
        resolved export or the materialized prefix inventory.
    distribution, version:
        Declared recipe distribution and version.

    Returns
    -------
    str | None
        The canonical ``sha256:``-prefixed artifact digest, or ``None`` when the
        routed environment names no matching distribution. ``None`` is an honest
        "not derivable here", never a fabricated digest.

    Raises
    ------
    RecipeError
        If the inventory names the distribution more than once, records a
        noncanonical digest, or records a version that contradicts the recipe.
    """

    wanted = canonical_distribution_name(distribution)
    matches = [
        row
        for row in packages
        if isinstance(row, Mapping)
        and isinstance(row.get("name"), str)
        and canonical_distribution_name(str(row["name"])) == wanted
    ]
    if not matches:
        return None
    digests = {str(row.get("sha256")) for row in matches}
    versions = {str(row.get("version")) for row in matches}
    if len(digests) != 1 or len(versions) != 1:
        raise RecipeError(
            f"environment inventory names distribution {distribution!r} ambiguously"
        )
    digest = digests.pop()
    if _HASH_PATTERN.fullmatch(digest) is None:
        raise RecipeError(
            f"environment inventory digest for {distribution!r} is not a canonical sha256"
        )
    resolved_version = versions.pop()
    if resolved_version != str(version).strip():
        raise RecipeError(
            f"recipe declares {distribution!r} version {version!r} but the routed "
            f"environment installs {resolved_version!r}"
        )
    return digest


def bind_library_artifact_digest(
    implementation: dict[str, Any],
    packages: Sequence[Mapping[str, Any]],
) -> bool:
    """Fill the machine-derived R1 artifact digest, refusing a conflicting claim.

    The digest identifies the installed distribution, which only the machine can
    derive, so this never trusts an author value over its own derivation and
    never silently overwrites one either: a conflicting supplied digest is a
    typed refusal, so the check cannot be left structurally dead.

    Parameters
    ----------
    implementation:
        Mutable proposal implementation block.
    packages:
        Exact package rows for the routed environment.

    Returns
    -------
    bool
        Whether the recipe bytes changed and dependent identities must rebind.

    Raises
    ------
    RecipeError
        If a supplied digest conflicts with the derived one, is malformed, or the
        inventory itself is ambiguous.
    """

    if implementation.get("recipe_type") != "declarative-library":
        return False
    recipe = implementation.get("library_recipe")
    if not isinstance(recipe, dict):
        return False
    distribution = recipe.get("distribution")
    version = recipe.get("version")
    if not isinstance(distribution, str) or not isinstance(version, str):
        raise RecipeError("declarative recipe lacks a distribution and version to resolve")
    supplied = recipe.get("artifact_sha256")
    if supplied is not None and (
        not isinstance(supplied, str) or _HASH_PATTERN.fullmatch(supplied) is None
    ):
        raise RecipeError("supplied artifact_sha256 is not a canonical sha256 digest")
    derived = resolve_environment_artifact_digest(
        packages, distribution=distribution, version=version
    )
    if supplied is not None and derived is not None and supplied != derived:
        raise RecipeError(
            "supplied artifact_sha256 conflicts with the routed environment: "
            f"supplied {supplied}, derived {derived}"
        )
    if supplied is not None:
        return False
    if derived is None:
        if "artifact_sha256" in recipe:
            return False
        recipe["artifact_sha256"] = None
        return True
    recipe["artifact_sha256"] = derived
    return True


@dataclass(frozen=True)
class PostConstructCall:
    """One bounded declarative post-construction configuration call.

    Parameters
    ----------
    method:
        Public method name invoked on the constructed model.
    args, kwargs:
        Plain-JSON call arguments; construct nodes are refused here.
    """

    method: str
    args: tuple[Any, ...] = ()
    kwargs: Mapping[str, Any] = dataclass_field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any, context: str) -> "PostConstructCall":
        """Validate and construct one closed post-construction call.

        Parameters
        ----------
        value:
            Candidate call mapping.
        context:
            Location used in refusal messages.

        Returns
        -------
        PostConstructCall
            Strict configuration call.
        """

        if not isinstance(value, Mapping):
            raise RecipeError(f"{context} must be an object")
        unknown = set(value) - {"method", "args", "kwargs"}
        if unknown:
            raise RecipeError(f"{context} has unknown fields: {sorted(unknown)!r}")
        method = value.get("method")
        if not isinstance(method, str) or not method.isidentifier():
            raise RecipeError(f"{context} method must be a direct identifier")
        if method.startswith("_"):
            raise RecipeError(f"{context} method must be a public name, not {method!r}")
        args = value.get("args", [])
        if not isinstance(args, (list, tuple)):
            raise RecipeError(f"{context} args must be a list")
        kwargs = value.get("kwargs", {})
        if not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
            raise RecipeError(f"{context} kwargs must be a string-keyed mapping")
        try:
            canonical_json_bytes({"args": list(args), "kwargs": dict(kwargs)})
        except (TypeError, ValueError) as exc:
            raise RecipeError(f"{context} arguments must be JSON-compatible") from exc
        _reject_construct_nodes(list(args), f"{context}.args")
        _reject_construct_nodes(dict(kwargs), f"{context}.kwargs")
        return cls(method=method, args=tuple(args), kwargs=dict(kwargs))

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible call mapping.

        Returns
        -------
        dict[str, Any]
            Post-construction call payload.
        """

        return {"method": self.method, "args": list(self.args), "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class DeclarativeRecipe:
    """Closed R1 library constructor description.

    Parameters
    ----------
    distribution, version:
        Exact package distribution and version.
    module, symbol:
        Importable module and direct constructor attribute.
    kwargs:
        JSON-compatible constructor keyword arguments. A mapping value carrying
        the single ``__construct__`` key is a construct node resolved from a
        declared module/symbol at build time; everything else is literal JSON.
    artifact_sha256:
        Optional exact installed-artifact hash.
    pretrained_disable_fields:
        Keyword names explicitly set to disable pretrained assets.
    post_construct:
        Bounded declarative configuration calls applied after construction.
    entrypoint:
        Optional public non-``forward`` call method delegated through the
        crawler-owned transparent adapter. ``None`` means native ``forward``.
    """

    distribution: str
    version: str
    module: str
    symbol: str
    kwargs: Mapping[str, Any]
    artifact_sha256: Optional[str] = None
    pretrained_disable_fields: tuple[str, ...] = ()
    post_construct: tuple[PostConstructCall, ...] = ()
    entrypoint: Optional[str] = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DeclarativeRecipe":
        """Validate and construct a closed declarative recipe.

        Parameters
        ----------
        value:
            Candidate recipe mapping.

        Returns
        -------
        DeclarativeRecipe
            Strict declarative recipe.
        """

        allowed = {
            "distribution",
            "version",
            "artifact_sha256",
            "module",
            "symbol",
            "kwargs",
            "pretrained_disable_fields",
            "post_construct",
            "entrypoint",
        }
        unknown = set(value) - allowed
        if unknown:
            raise RecipeError(f"unknown declarative recipe fields: {sorted(unknown)!r}")
        required = {"distribution", "version", "module", "symbol", "kwargs"}
        missing = required - set(value)
        if missing:
            raise RecipeError(f"missing declarative recipe fields: {sorted(missing)!r}")
        strings = {name: value[name] for name in required - {"kwargs"}}
        if any(not isinstance(item, str) or not item for item in strings.values()):
            raise RecipeError("distribution, version, module, and symbol must be non-empty strings")
        module = str(value["module"])
        symbol = str(value["symbol"])
        if not all(part.isidentifier() for part in module.split(".")) or not symbol.isidentifier():
            raise RecipeError("module and symbol must be direct Python identifiers")
        if _is_generic_container_reference(module, symbol):
            raise RecipeError(
                "generic torch.nn containers are refused in declarative R1 recipes: "
                f"{module}.{symbol}"
            )
        kwargs = value["kwargs"]
        if not isinstance(kwargs, Mapping) or not all(isinstance(key, str) for key in kwargs):
            raise RecipeError("kwargs must be a string-keyed mapping")
        try:
            canonical_json_bytes(kwargs)
        except (TypeError, ValueError) as exc:
            raise RecipeError(
                "kwargs must contain only JSON-compatible declarative values"
            ) from exc
        node_counter = [0]
        for name, child in kwargs.items():
            _walk_construct_values(
                child,
                context=f"kwargs.{name}",
                depth=0,
                counter=node_counter,
                distribution=str(value["distribution"]),
            )
        entrypoint = value.get("entrypoint")
        if entrypoint is not None:
            if not isinstance(entrypoint, str) or not entrypoint.isidentifier():
                raise RecipeError("entrypoint must be a direct method identifier or null")
            if entrypoint.startswith("_"):
                raise RecipeError(f"entrypoint must be a public method, not {entrypoint!r}")
        raw_post_construct = value.get("post_construct", [])
        if not isinstance(raw_post_construct, (list, tuple)):
            raise RecipeError("post_construct must be a list of configuration calls")
        if len(raw_post_construct) > MAX_POST_CONSTRUCT_CALLS:
            raise RecipeError(
                f"post_construct exceeds the maximum of {MAX_POST_CONSTRUCT_CALLS} calls"
            )
        post_construct = tuple(
            PostConstructCall.from_mapping(item, f"post_construct[{index}]")
            for index, item in enumerate(raw_post_construct)
        )
        disable_fields = value.get("pretrained_disable_fields", [])
        if not isinstance(disable_fields, (list, tuple)) or not all(
            isinstance(field, str) and field for field in disable_fields
        ):
            raise RecipeError("pretrained_disable_fields must be a string sequence")
        validate_pretrained_disable_fields(kwargs, disable_fields)
        artifact = value.get("artifact_sha256")
        if artifact is not None and not isinstance(artifact, str):
            raise RecipeError("artifact_sha256 must be a string or null")
        return cls(
            distribution=str(value["distribution"]),
            version=str(value["version"]),
            module=module,
            symbol=symbol,
            kwargs=dict(kwargs),
            artifact_sha256=artifact,
            pretrained_disable_fields=tuple(disable_fields),
            post_construct=post_construct,
            entrypoint=entrypoint,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a canonical JSON-compatible recipe mapping.

        The two grammar-extension fields are emitted only when they deviate from
        their defaults so that every pre-extension recipe keeps its exact
        historical recipe-revision hash.

        Returns
        -------
        dict[str, Any]
            Declarative recipe payload.
        """

        payload: dict[str, Any] = {
            "distribution": self.distribution,
            "version": self.version,
            "artifact_sha256": self.artifact_sha256,
            "module": self.module,
            "symbol": self.symbol,
            "kwargs": dict(self.kwargs),
            "pretrained_disable_fields": list(self.pretrained_disable_fields),
        }
        if self.post_construct:
            payload["post_construct"] = [call.to_dict() for call in self.post_construct]
        if self.entrypoint is not None:
            payload["entrypoint"] = self.entrypoint
        return payload


@dataclass(frozen=True)
class LoadedRecipe:
    """Executable constructor obtained from a safe recipe form.

    Parameters
    ----------
    kind:
        ``declarative-library`` or ``typed-adapter``.
    build_model:
        Zero-argument random-initialized constructor.
    make_dummy_call:
        Typed adapter input builder, absent for declarative R1 recipes.
    recipe_revision:
        Source-bound recipe hash when a source identity was supplied.
    module:
        Loaded typed module, absent for declarative recipes.
    adapter_sha256:
        Digest observed from the one exact adapter byte string executed by the loader.
    entrypoint:
        Declared public non-``forward`` call method for declarative recipes;
        the executor delegates through the crawler-owned transparent adapter.
    """

    kind: str
    build_model: BuildModel
    make_dummy_call: Optional[MakeDummyCall]
    recipe_revision: str
    module: Optional[ModuleType]
    adapter_sha256: Optional[str] = None
    entrypoint: Optional[str] = None


def reject_opaque_recipe(value: Mapping[str, Any]) -> None:
    """Reject every legacy expression, statement, eval, or exec recipe form.

    Parameters
    ----------
    value:
        Candidate recipe mapping.

    Raises
    ------
    RecipeError
        If the mapping contains an opaque executable recipe marker.
    """

    forbidden_keys = {"code", "expression", "statement", "eval", "exec", "callable_string"}
    offending = forbidden_keys.intersection(value)
    recipe_type = str(value.get("type", ""))
    if offending or recipe_type in {"statement", "expression", "exec-string", "eval-string"}:
        details = sorted(offending) or [recipe_type]
        raise RecipeError(f"opaque executable recipes are forbidden: {details!r}")


def _materialize_construct_value(value: Any, context: str, *, distribution: str) -> Any:
    """Resolve construct nodes inside one declarative value at build time.

    Parameters
    ----------
    value:
        Validated JSON-compatible declarative value.
    context:
        Location used in refusal messages.
    distribution:
        Pinned distribution bounding every construct node's callable namespace.

    Returns
    -------
    Any
        The literal value, or the object built by the declared construct graph.

    Raises
    ------
    RecipeError
        If a construct node names a module outside the bounded namespace, cannot
        be resolved to a callable non-container symbol, or builds a generic
        container instance.
    """

    if isinstance(value, Mapping):
        if CONSTRUCT_NODE_KEY in value:
            spec = _validate_construct_node_spec(
                value.get(CONSTRUCT_NODE_KEY), context, distribution=distribution
            )
            constructed_module = importlib.import_module(str(spec["module"]))
            constructed_symbol = getattr(constructed_module, str(spec["symbol"]), None)
            if constructed_symbol is None or not callable(constructed_symbol):
                raise RecipeError(
                    f"construct node at {context}: {spec['module']}.{spec['symbol']} "
                    "is not a callable constructor"
                )
            _reject_container_constructor(
                constructed_symbol, f"{spec['module']}.{spec['symbol']}"
            )
            materialized = {
                name: _materialize_construct_value(
                    child, f"{context}.{name}", distribution=distribution
                )
                for name, child in spec["kwargs"].items()
            }
            constructed = constructed_symbol(**materialized)
            _reject_container_instance(
                constructed, f"construct node at {context} constructed a generic container"
            )
            return constructed
        return {
            name: _materialize_construct_value(
                child, f"{context}.{name}", distribution=distribution
            )
            for name, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _materialize_construct_value(
                child, f"{context}[{index}]", distribution=distribution
            )
            for index, child in enumerate(value)
        ]
    return value


def resolve_input_constructor(spec: Mapping[str, Any], *, distribution: str) -> object:
    """Materialize one declared constructed input leaf without authored code.

    This implements the input-leaf ``distribution: "constructor"`` reservation:
    the value is built by importing a declared module, resolving a declared
    symbol, and calling it with JSON-only (possibly construct-node) kwargs.

    Parameters
    ----------
    spec:
        Exact ``{"module", "symbol", "kwargs"}`` input constructor payload.
    distribution:
        Pinned distribution of the recipe this input is being built for; bounds
        the constructor's callable namespace exactly as it bounds a recipe's
        construct nodes. A constructed input leaf runs the same import-then-call
        machinery, so leaving it unbounded would reopen the hole on the input
        side alone.

    Returns
    -------
    object
        The constructed input value.

    Raises
    ------
    RecipeError
        If the payload deviates from the closed grammar, names a module outside
        the bounded namespace, resolves to or builds a generic container, or
        produces ``None``.
    """

    validated = _validate_construct_node_spec(
        spec, "input_contract constructor", distribution=distribution
    )
    counter = [0]
    for name, child in validated["kwargs"].items():
        _walk_construct_values(
            child,
            context=f"input constructor kwargs.{name}",
            depth=1,
            counter=counter,
            distribution=distribution,
        )
    try:
        canonical_json_bytes(dict(validated["kwargs"]))
    except (TypeError, ValueError) as exc:
        raise RecipeError(
            "input constructor kwargs must contain only JSON-compatible values"
        ) from exc
    constructed_module = importlib.import_module(str(validated["module"]))
    constructed_symbol = getattr(constructed_module, str(validated["symbol"]), None)
    if constructed_symbol is None or not callable(constructed_symbol):
        raise RecipeError(
            f"input constructor {validated['module']}.{validated['symbol']} "
            "is not a callable constructor"
        )
    _reject_container_constructor(
        constructed_symbol, f"{validated['module']}.{validated['symbol']}"
    )
    value = constructed_symbol(
        **{
            name: _materialize_construct_value(
                child, f"input constructor kwargs.{name}", distribution=distribution
            )
            for name, child in validated["kwargs"].items()
        }
    )
    if value is None:
        raise RecipeError(
            f"input constructor {validated['module']}.{validated['symbol']} produced None"
        )
    _reject_container_instance(
        value,
        f"input constructor {validated['module']}.{validated['symbol']} "
        "constructed a generic container",
    )
    return value


def load_declarative_recipe(
    value: Union[DeclarativeRecipe, Mapping[str, Any]], *, source_identity: str = "unbound"
) -> LoadedRecipe:
    """Load a closed R1 recipe without evaluating source text.

    Parameters
    ----------
    value:
        Validated recipe object or mapping.
    source_identity:
        Exact source identity to bind into the recipe revision.

    Returns
    -------
    LoadedRecipe
        Direct import/getattr constructor wrapper.
    """

    if isinstance(value, Mapping):
        reject_opaque_recipe(value)
        recipe = DeclarativeRecipe.from_mapping(value)
    else:
        recipe = value
    module = importlib.import_module(recipe.module)
    constructor = getattr(module, recipe.symbol, None)
    if constructor is None or not callable(constructor):
        raise RecipeError(f"{recipe.module}.{recipe.symbol} is not a callable constructor")
    _reject_container_constructor(constructor, f"{recipe.module}.{recipe.symbol}")
    if recipe.pretrained_disable_fields:
        try:
            parameters = inspect.signature(constructor).parameters
        except (TypeError, ValueError) as exc:
            raise RecipeError(
                "cannot verify pretrained disable fields against constructor signature"
            ) from exc
        unsupported = [
            field for field in recipe.pretrained_disable_fields if field not in parameters
        ]
        if unsupported:
            raise RecipeError(
                "pretrained disable fields are not explicit constructor parameters: "
                f"{unsupported!r}"
            )

    def build_model() -> object:
        """Invoke the direct library constructor with declarative kwargs.

        Construct-node kwargs are resolved from their declared modules -- bounded
        to the pinned distribution and :data:`CONSTRUCT_MODULE_ALLOWLIST`, and
        refused if they build a generic container -- bounded post-construction
        configuration calls are applied, and the runtime provenance tripwire
        refuses any constructed model whose class is not defined by the pinned
        distribution.

        Returns
        -------
        object
            Constructed random-initialized model.
        """

        materialized = {
            name: _materialize_construct_value(
                child, f"kwargs.{name}", distribution=recipe.distribution
            )
            for name, child in recipe.kwargs.items()
        }
        model = constructor(**materialized)
        for index, call in enumerate(recipe.post_construct):
            method = getattr(model, call.method, None)
            if not callable(method):
                raise RecipeError(
                    f"post_construct[{index}] method {call.method!r} is not callable "
                    "on the constructed model"
                )
            method(*call.args, **dict(call.kwargs))
        assert_model_provenance(model, recipe.distribution)
        return model

    revision = compute_recipe_revision(recipe.to_dict(), source_identity)
    return LoadedRecipe(
        "declarative-library", build_model, None, revision, None, entrypoint=recipe.entrypoint
    )


def _check_adapter_source(source: str, path: Path) -> ast.Module:
    """Statically reject opaque execution and TorchLens imports in an adapter.

    Parameters
    ----------
    source:
        Exact Python adapter source.
    path:
        Source path used in diagnostics.

    Returns
    -------
    ast.Module
        Parsed module syntax tree.
    """

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise RecipeError(f"typed adapter has invalid syntax: {exc}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "torchlens" or alias.name.startswith("torchlens.")
                for alias in node.names
            ):
                raise RecipeError("typed adapters must not import TorchLens")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torchlens" or str(node.module).startswith("torchlens."):
                raise RecipeError("typed adapters must not import TorchLens")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "compile"}:
                raise RecipeError(f"typed adapters must not call {node.func.id}()")
    return tree


def _validate_typed_function(
    function: Callable[..., Any], name: str, expected_parameters: tuple[str, ...]
) -> None:
    """Validate one typed adapter entry point.

    Parameters
    ----------
    function:
        Candidate function.
    name:
        Required public symbol name.
    expected_parameters:
        Exact positional parameter names.
    """

    if not inspect.isfunction(function):
        raise RecipeError(f"{name} must be a Python function")
    signature = inspect.signature(function)
    if tuple(signature.parameters) != expected_parameters:
        raise RecipeError(f"{name} must have parameters {expected_parameters!r}")
    annotations = function.__annotations__
    if "return" not in annotations or any(
        parameter not in annotations for parameter in expected_parameters
    ):
        raise RecipeError(f"{name} must carry parameter and return type annotations")


def load_typed_adapter(
    path: Path,
    *,
    source_identity: str = "unbound",
    expected_recipe_revision: Optional[str] = None,
    expected_adapter_sha256: Optional[str] = None,
) -> LoadedRecipe:
    """Statically check and import a typed R2--R4 adapter module.

    Parameters
    ----------
    path:
        Exact adapter Python file.
    source_identity:
        Exact source identity to bind into the recipe revision.
    expected_recipe_revision:
        Parent-authorized revision that the observed bytes must reproduce before import.
    expected_adapter_sha256:
        Optional direct digest of the parent-authorized adapter bytes.

    Returns
    -------
    LoadedRecipe
        Typed constructor and dummy-call functions.
    """

    source_bytes = path.read_bytes()
    observed_adapter_sha256 = hash_bytes(source_bytes)
    revision = compute_recipe_revision(
        {"recipe_type": "typed-adapter", "path": path.name},
        source_identity,
        adapter_bytes=source_bytes,
    )
    if expected_adapter_sha256 is not None and observed_adapter_sha256 != expected_adapter_sha256:
        raise RecipeError(
            "typed adapter digest mismatch: "
            f"expected {expected_adapter_sha256}, observed {observed_adapter_sha256}"
        )
    if expected_recipe_revision is not None and revision != expected_recipe_revision:
        raise RecipeError(
            "typed adapter recipe revision mismatch: "
            f"expected {expected_recipe_revision}, observed {revision}"
        )
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecipeError("typed adapter must be UTF-8 Python source") from exc
    _check_adapter_source(source, path)
    module_name = f"menagerie_crawler_adapter_{observed_adapter_sha256[7:23]}"
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[module_name] = module
    try:
        loader = importlib.machinery.SourceFileLoader(module_name, str(path))
        code = loader.source_to_code(source_bytes, str(path))
        FunctionType(code, module.__dict__)()
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    build_model = getattr(module, "build_model", None)
    make_dummy_call = getattr(module, "make_dummy_call", None)
    if not callable(build_model) or not callable(make_dummy_call):
        raise RecipeError("typed adapter must define build_model and make_dummy_call")
    _validate_typed_function(build_model, "build_model", ())
    _validate_typed_function(make_dummy_call, "make_dummy_call", ("seed", "device"))
    return LoadedRecipe(
        "typed-adapter",
        build_model,
        make_dummy_call,
        revision,
        module,
        observed_adapter_sha256,
    )


def load_recipe(
    value: Mapping[str, Any],
    *,
    source_identity: str = "unbound",
    expected_recipe_revision: Optional[str] = None,
) -> LoadedRecipe:
    """Load one explicitly tagged closed recipe form.

    Parameters
    ----------
    value:
        Mapping with kind ``declarative-library`` or ``typed-adapter``.
    source_identity:
        Source identity bound into the recipe revision.
    expected_recipe_revision:
        Parent-authorized revision required before a typed adapter is imported.

    Returns
    -------
    LoadedRecipe
        Executable typed recipe contract.
    """

    reject_opaque_recipe(value)
    kind = value.get("kind") or value.get("recipe_type")
    if kind == "declarative-library":
        payload = value.get("recipe", value)
        if not isinstance(payload, Mapping):
            raise RecipeError("declarative recipe payload must be an object")
        if payload is value:
            payload = {
                key: item for key, item in value.items() if key not in {"kind", "recipe_type"}
            }
        loaded = load_declarative_recipe(payload, source_identity=source_identity)
        if (
            expected_recipe_revision is not None
            and loaded.recipe_revision != expected_recipe_revision
        ):
            raise RecipeError(
                "declarative recipe revision mismatch: "
                f"expected {expected_recipe_revision}, observed {loaded.recipe_revision}"
            )
        return loaded
    if kind == "typed-adapter":
        path_value = value.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise RecipeError("typed-adapter recipe requires a path")
        digest_value = value.get("adapter_sha256")
        if digest_value is not None and not isinstance(digest_value, str):
            raise RecipeError("typed-adapter adapter_sha256 must be a string when supplied")
        return load_typed_adapter(
            Path(path_value),
            source_identity=source_identity,
            expected_recipe_revision=expected_recipe_revision,
            expected_adapter_sha256=digest_value,
        )
    raise RecipeError(f"unsupported recipe kind: {kind!r}")
