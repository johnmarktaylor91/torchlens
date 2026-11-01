"""Static opcode pre-scan for the restricted bundle unpickler.

The loader runs untrusted ``metadata.pkl`` bytes through the C ``pickle.Unpickler``
for speed. The C machine exposes only ``find_class`` to a subclass, so the opcode-level
checks the pure-Python machine used to enforce on BUILD / REDUCE / NEWOBJ / NEWOBJ_EX /
INST / OBJ are established here first, on a symbolic walk of the opcode stream
(:func:`pickletools.genops` plus a stack model), before any object is built:

* every global reference is resolved through the caller's ``find_class`` policy
  (refusals fire here, before the C machine starts);
* every construction opcode whose class and arguments are statically known runs the
  allocation belt now, with the same refusal text the pure-Python handlers used;
* every construction whose callable arrives through ``find_class`` is marked so the C
  phase hands the callable back wrapped in a guard that runs the belt on the real
  arguments at call time (:class:`torchlens._io._safe_unpickle._GuardedCallable`);
* every BUILD target is classified: a constant or container can never be a tensor, a
  resolved global or a statically typed construction is checked here, and a REDUCE
  result is flagged so its guard checks the real result type before BUILD can reach it;
* stream corruption (stack underflow, missing memo keys, truncation, unknown opcodes)
  is refused as :class:`pickle.UnpicklingError`, the family the pure-Python boundary
  normalised to;
* persistent ids, extension codes and out-of-band buffers are refused, as both machines
  refuse them without the corresponding hooks.

A stream shape the static walk cannot classify (a callable of unknown provenance, a
constructor with non-literal arguments, a global used both as a value and as a
callable) raises :class:`StreamNeedsInterpreter`; the unpickler then runs the stream
through the pure-Python machine, which enforces the same checks per opcode. No
TorchLens writer produces such a stream, so the fast path is the normal path and the
interpreter is the exact fallback, never a weaker one.

Two checks here are deliberately stricter than the pure-Python machine was. A GLOBAL or
INST opcode whose module line contains a space is refused (no Python module is spelled
that way, and the two-line form must parse unambiguously), and an explicit memo PUT
index above the running opcode count is refused (a pickler assigns memo indices
sequentially; the C machine sizes its memo table from the index, so an absurd index is
an allocation amplifier the dict-backed pure-Python memo never had).
"""

from __future__ import annotations

import copyreg
import pickle
import pickletools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "GuardedCallable",
    "OpcodeCeilingExceeded",
    "PrescanPlan",
    "ReduceSite",
    "StreamNeedsInterpreter",
    "count_opcodes",
    "prescan_stream",
]

#: Refusal-text prefix shared with the pure-Python machine's load boundary.
_CORRUPT = "corrupt or malformed pickle stream: "
#: Largest literal argument tree concretised for a static construction check; bigger
#: argument trees are not what a constructor call looks like and take the interpreter.
_MAX_STATIC_NODES = 256

ConstructionRefusal = Callable[[str, Any, Any, Any], "str | None"]
BuildRefusal = Callable[[Any], "str | None"]
BuildTypeRefusal = Callable[[type], "str | None"]


class StreamNeedsInterpreter(Exception):
    """The static walk cannot classify the stream; the pure-Python machine must run it."""


class OpcodeCeilingExceeded(pickle.UnpicklingError):
    """The stream holds more opcodes than the caller's allocation ceiling allows."""


@dataclass(eq=False)
class ReduceSite:
    """What the C phase must enforce at one REDUCE opcode, in stream order."""

    index: int
    func_pair: tuple[str, str] | None
    derived_from: int
    build_target: bool = False
    result_callable: bool = False


@dataclass(eq=False)
class PrescanPlan:
    """Outcome of a successful pre-scan, consumed by the C phase."""

    opcode_count: int
    resolved: dict[tuple[str, str], Any]
    wrap_pairs: frozenset[tuple[str, str]]
    reduce_sites: list[ReduceSite]
    construction_refusal: ConstructionRefusal
    build_refusal: BuildRefusal
    cursor: int = 0
    global_lookups: int = field(default=0)

    def take_reduce_site(self) -> ReduceSite:
        """Return the next REDUCE site; refuse when the stream calls more than planned."""

        if self.cursor >= len(self.reduce_sites):
            raise pickle.UnpicklingError(
                _CORRUPT + "a guarded callable was invoked more often than the pre-scan planned."
            )
        site = self.reduce_sites[self.cursor]
        self.cursor += 1
        return site


class GuardedCallable:
    """Call-time belt around a callable the C machine will invoke for a REDUCE.

    The C ``pickle.Unpickler`` runs REDUCE in C, so the belt the pure-Python
    ``load_reduce`` applied before the call moves into the callable itself: the guard
    takes the next planned REDUCE site, verifies it is the planned callable, runs the
    construction belt on the real argument tuple, calls through, and then applies what
    the pre-scan asked of the result: the BUILD refusal when a later BUILD targets it,
    or a nested guard when a later REDUCE calls it.
    """

    __slots__ = ("_origin", "_pair", "_plan", "_real")

    def __init__(
        self,
        real: Any,
        plan: PrescanPlan,
        *,
        pair: tuple[str, str] | None = None,
        origin: int = -1,
    ) -> None:
        self._real = real
        self._plan = plan
        self._pair = pair
        self._origin = origin

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        plan = self._plan
        site = plan.take_reduce_site()
        if self._pair is not None:
            planned = site.func_pair == self._pair
        else:
            planned = site.func_pair is None and site.derived_from == self._origin
        if not planned:
            raise _corrupt("a REDUCE invoked a callable the pre-scan did not plan for this site")
        reason = plan.construction_refusal("REDUCE", self._real, args, kwargs or None)
        if reason is not None:
            raise pickle.UnpicklingError(reason)
        result = self._real(*args, **kwargs)
        if site.build_target:
            build_reason = plan.build_refusal(result)
            if build_reason is not None:
                raise pickle.UnpicklingError(build_reason)
        if site.result_callable:
            return GuardedCallable(result, plan, origin=site.index)
        return result


class _Sym:
    """One symbolic stack value.

    ``kind`` is ``const`` (``value`` holds the literal), ``global`` (``pair`` names it,
    ``value`` is the resolved object), ``container`` (``value`` is the container kind
    name, ``items`` its member symbols; a dict stores alternating keys and values) or
    ``obj`` (``producer`` is the callable or class symbol, ``via`` the opcode name,
    ``site`` the REDUCE site index). ``roles`` records how the value is consumed.
    """

    __slots__ = ("items", "kind", "pair", "producer", "roles", "site", "value", "via")

    def __init__(
        self,
        kind: str,
        value: Any = None,
        *,
        pair: tuple[str, str] | None = None,
        items: list[_Sym] | None = None,
        producer: _Sym | None = None,
        via: str = "",
        site: int = -1,
    ) -> None:
        self.kind = kind
        self.value = value
        self.pair = pair
        self.items = items
        self.producer = producer
        self.via = via
        self.site = site
        self.roles: set[str] | None = None

    def tag(self, role: str) -> None:
        """Record one consumption role (``value``, ``func`` or ``cls``)."""

        if self.roles is None:
            self.roles = {role}
        else:
            self.roles.add(role)


class _NotStatic(Exception):
    """A symbol tree holds a value that is only known at run time."""


def _corrupt(detail: str) -> pickle.UnpicklingError:
    return pickle.UnpicklingError(_CORRUPT + detail)


def _concrete(sym: _Sym, budget: list[int]) -> Any:
    """Return the literal Python value a symbol tree denotes, or raise ``_NotStatic``."""

    budget[0] -= 1
    if budget[0] < 0:
        raise _NotStatic("argument tree too large for a static check")
    if sym.kind == "const" or sym.kind == "global":
        return sym.value
    if sym.kind != "container":
        raise _NotStatic("argument built at run time")
    items = sym.items or []
    if sym.value == "dict":
        return {
            _concrete(items[index], budget): _concrete(items[index + 1], budget)
            for index in range(0, len(items), 2)
        }
    members = [_concrete(item, budget) for item in items]
    if sym.value == "tuple":
        return tuple(members)
    if sym.value == "list":
        return members
    try:
        return set(members) if sym.value == "set" else frozenset(members)
    except TypeError as exc:
        raise _NotStatic("unhashable literal member") from exc


class _Walker:
    """Symbolic stack machine over one pickle stream."""

    def __init__(
        self,
        data: bytes,
        *,
        resolve: Callable[[str, str], Any],
        construction_refusal: ConstructionRefusal,
        build_refusal: BuildRefusal,
        build_type_refusal: BuildTypeRefusal,
        max_opcodes: int | None,
    ) -> None:
        self.data = data
        self.resolve = resolve
        self.construction_refusal = construction_refusal
        self.build_refusal = build_refusal
        self.build_type_refusal = build_type_refusal
        self.max_opcodes = max_opcodes
        self.stack: list[_Sym] = []
        self.metastack: list[list[_Sym]] = []
        self.memo: dict[int, _Sym] = {}
        self.resolved: dict[tuple[str, str], Any] = {}
        self.global_syms: list[_Sym] = []
        self.reduce_sites: list[ReduceSite] = []
        self.reduce_funcs: list[_Sym] = []
        self.opcode_count = 0

    # -- stack primitives -------------------------------------------------------------

    def pop(self, opcode: str) -> _Sym:
        if not self.stack:
            raise _corrupt(f"stack underflow on {opcode}")
        return self.stack.pop()

    def top(self, opcode: str) -> _Sym:
        if not self.stack:
            raise _corrupt(f"stack underflow on {opcode}")
        return self.stack[-1]

    def pop_mark(self, opcode: str) -> list[_Sym]:
        if not self.metastack:
            raise _corrupt(f"{opcode} without a preceding MARK")
        items = self.stack
        self.stack = self.metastack.pop()
        return items

    def push_const(self, value: Any) -> None:
        self.stack.append(_Sym("const", value))

    def push_container(self, kind: str, items: list[_Sym]) -> None:
        for item in items:
            item.tag("value")
        self.stack.append(_Sym("container", kind, items=items))

    # -- globals ----------------------------------------------------------------------

    def read_global_lines(self, pos: int, opcode: str) -> tuple[str, str]:
        """Parse the two newline-terminated lines after a GLOBAL / INST opcode byte."""

        first_end = self.data.find(b"\n", pos + 1)
        second_end = self.data.find(b"\n", first_end + 1) if first_end >= 0 else -1
        if first_end < 0 or second_end < 0:
            raise _corrupt(f"truncated {opcode} operand")
        try:
            module = self.data[pos + 1 : first_end].decode("utf-8")
            name = self.data[first_end + 1 : second_end].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _corrupt(f"{opcode} operand is not valid UTF-8") from exc
        if " " in module:
            raise _corrupt(f"{opcode} module line contains a space")
        return module, name

    def resolve_global(self, module: str, name: str) -> _Sym:
        pair = (module, name)
        if pair in self.resolved:
            obj = self.resolved[pair]
        else:
            obj = self.resolve(module, name)
            self.resolved[pair] = obj
        sym = _Sym("global", obj, pair=pair)
        self.global_syms.append(sym)
        return sym

    # -- construction opcodes ---------------------------------------------------------

    def static_construction(
        self, opcode: str, cls_sym: _Sym, args_sym: _Sym, kwargs_sym: _Sym | None
    ) -> None:
        """Run the allocation belt now for a construction with literal arguments."""

        if cls_sym.kind != "global":
            raise StreamNeedsInterpreter(f"{opcode} on a class of unknown provenance")
        try:
            budget = [_MAX_STATIC_NODES]
            args = _concrete(args_sym, budget)
            kwargs = None if kwargs_sym is None else _concrete(kwargs_sym, budget)
        except _NotStatic as why:
            raise StreamNeedsInterpreter(f"{opcode} with non-literal arguments: {why}") from None
        reason = self.construction_refusal(opcode, cls_sym.value, args, kwargs)
        if reason is not None:
            raise pickle.UnpicklingError(reason)
        cls_sym.tag("cls")

    def op_reduce(self) -> None:
        args = self.pop("REDUCE")
        func = self.pop("REDUCE")
        if args.kind != "container" or args.value != "tuple":
            raise StreamNeedsInterpreter("REDUCE whose argument is not a literal tuple")
        args.tag("value")
        if func.kind == "global":
            site = ReduceSite(len(self.reduce_sites), func.pair, -1)
        elif func.kind == "obj" and func.via == "REDUCE":
            site = ReduceSite(len(self.reduce_sites), None, func.site)
        else:
            raise StreamNeedsInterpreter("REDUCE on a callable of unknown provenance")
        func.tag("func")
        self.reduce_sites.append(site)
        self.reduce_funcs.append(func)
        self.stack.append(_Sym("obj", producer=func, via="REDUCE", site=site.index))

    def op_newobj(self, opcode: str) -> None:
        kwargs = None
        if opcode == "NEWOBJ_EX":
            kwargs = self.pop(opcode)
            if kwargs.kind != "container" or kwargs.value != "dict":
                raise StreamNeedsInterpreter("NEWOBJ_EX whose keywords are not a literal dict")
            kwargs.tag("value")
        args = self.pop(opcode)
        cls = self.pop(opcode)
        if args.kind != "container" or args.value != "tuple":
            raise StreamNeedsInterpreter(f"{opcode} whose argument is not a literal tuple")
        args.tag("value")
        self.static_construction(opcode, cls, args, kwargs)
        self.stack.append(_Sym("obj", producer=cls, via=opcode))

    def op_inst(self, pos: int) -> None:
        module, name = self.read_global_lines(pos, "INST")
        items = self.pop_mark("INST")
        cls = self.resolve_global(module, name)
        self.static_construction("INST", cls, _Sym("container", "tuple", items=items), None)
        for item in items:
            item.tag("value")
        self.stack.append(_Sym("obj", producer=cls, via="INST"))

    def op_obj(self) -> None:
        items = self.pop_mark("OBJ")
        if not items:
            raise _corrupt("OBJ without a class on the stack")
        cls, args = items[0], items[1:]
        self.static_construction("OBJ", cls, _Sym("container", "tuple", items=args), None)
        for item in args:
            item.tag("value")
        self.stack.append(_Sym("obj", producer=cls, via="OBJ"))

    def op_build(self) -> None:
        state = self.pop("BUILD")
        target = self.top("BUILD")
        state.tag("value")
        if target.kind == "global":
            reason = self.build_refusal(target.value)
            if reason is not None:
                raise pickle.UnpicklingError(reason)
        elif target.kind == "obj":
            if target.via == "REDUCE":
                self.reduce_sites[target.site].build_target = True
            else:
                producer = target.producer
                if producer is None or producer.kind != "global":
                    raise StreamNeedsInterpreter("BUILD on an object of unknown provenance")
                if isinstance(producer.value, type):
                    reason = self.build_type_refusal(producer.value)
                    if reason is not None:
                        raise pickle.UnpicklingError(reason)

    def op_stack_global(self) -> None:
        name = self.pop("STACK_GLOBAL")
        module = self.pop("STACK_GLOBAL")
        if name.kind != "const" or module.kind != "const":
            raise StreamNeedsInterpreter("STACK_GLOBAL with non-literal operands")
        if type(name.value) is not str or type(module.value) is not str:
            raise pickle.UnpicklingError("STACK_GLOBAL requires str")
        self.stack.append(self.resolve_global(module.value, name.value))

    # -- the walk ---------------------------------------------------------------------

    def step(self, name: str, arg: Any, pos: int) -> bool:  # noqa: C901 -- one opcode table
        """Apply one opcode; return ``True`` at STOP."""

        if name in _CONST_OPCODES:
            self.push_const(arg)
        elif name == "MEMOIZE":
            self.memo[len(self.memo)] = self.top(name)
        elif name in _GET_OPCODES:
            sym = self.memo.get(arg)
            if sym is None:
                raise _corrupt(f"missing memo key {arg!r}")
            self.stack.append(sym)
        elif name in _PUT_OPCODES:
            if arg < 0:
                raise _corrupt("negative PUT argument")
            if arg > self.opcode_count:
                raise _corrupt("memo PUT index above the opcode count")
            self.memo[arg] = self.top(name)
        elif name == "MARK":
            self.metastack.append(self.stack)
            self.stack = []
        elif name in _EMPTY_CONTAINER_OPCODES:
            self.push_container(_EMPTY_CONTAINER_OPCODES[name], [])
        elif name in _TUPLE_N_OPCODES:
            count = _TUPLE_N_OPCODES[name]
            if len(self.stack) < count:
                raise _corrupt(f"stack underflow on {name}")
            items = self.stack[-count:]
            del self.stack[-count:]
            self.push_container("tuple", items)
        elif name in _MARK_CONTAINER_OPCODES:
            items = self.pop_mark(name)
            if name == "DICT" and len(items) % 2:
                raise _corrupt("DICT with an odd number of items")
            self.push_container(_MARK_CONTAINER_OPCODES[name], items)
        elif name == "APPEND":
            value = self.pop(name)
            self.extend_container(self.top(name), [value], "list")
        elif name == "APPENDS":
            items = self.pop_mark(name)
            self.extend_container(self.top(name), items, "list")
        elif name == "SETITEM":
            value = self.pop(name)
            key = self.pop(name)
            self.extend_container(self.top(name), [key, value], "dict")
        elif name == "SETITEMS":
            items = self.pop_mark(name)
            if len(items) % 2:
                raise _corrupt("SETITEMS with an odd number of items")
            self.extend_container(self.top(name), items, "dict")
        elif name == "ADDITEMS":
            items = self.pop_mark(name)
            self.extend_container(self.top(name), items, "set")
        elif name == "POP":
            if self.stack:
                self.stack.pop()
            else:
                self.pop_mark(name)
        elif name == "POP_MARK":
            self.pop_mark(name)
        elif name == "DUP":
            self.stack.append(self.top(name))
        elif name == "GLOBAL":
            module, global_name = self.read_global_lines(pos, name)
            self.stack.append(self.resolve_global(module, global_name))
        elif name == "STACK_GLOBAL":
            self.op_stack_global()
        elif name == "REDUCE":
            self.op_reduce()
        elif name in ("NEWOBJ", "NEWOBJ_EX"):
            self.op_newobj(name)
        elif name == "INST":
            self.op_inst(pos)
        elif name == "OBJ":
            self.op_obj()
        elif name == "BUILD":
            self.op_build()
        elif name in ("PERSID", "BINPERSID"):
            raise pickle.UnpicklingError(
                "A load persistent id instruction was encountered, "
                "but no persistent_load function was specified."
            )
        elif name in ("EXT1", "EXT2", "EXT4"):
            if arg in copyreg._inverted_registry:
                raise StreamNeedsInterpreter("registered pickle extension code")
            raise _corrupt(f"unregistered extension code {arg!r}")
        elif name in ("NEXT_BUFFER", "READONLY_BUFFER"):
            raise pickle.UnpicklingError(
                "pickle stream refers to out-of-band data but no *buffers* argument was given"
            )
        elif name == "PROTO":
            if arg > pickle.HIGHEST_PROTOCOL:
                raise _corrupt(f"unsupported pickle protocol: {arg}")
        elif name == "FRAME":
            pass
        elif name == "STOP":
            self.pop(name).tag("value")
            return True
        else:  # pragma: no cover - every pickletools opcode is listed above
            raise _corrupt(f"unsupported opcode {name}")
        return False

    def extend_container(self, target: _Sym, items: list[_Sym], kind: str) -> None:
        """Record items added to a container; other targets fail or act at run time."""

        for item in items:
            item.tag("value")
        if target.kind == "container" and target.value == kind and target.items is not None:
            target.items.extend(items)

    def run(self) -> PrescanPlan:
        ops = pickletools.genops(self.data)
        while True:
            opcode, arg, pos = _next_opcode(ops)
            self.opcode_count += 1
            if self.max_opcodes is not None and self.opcode_count > self.max_opcodes:
                raise OpcodeCeilingExceeded(
                    _CORRUPT + "opcode count exceeds the allocation ceiling"
                )
            if self.step(opcode.name, arg, pos):
                break
        return self.finish()

    def finish(self) -> PrescanPlan:
        """Decide which callables the C phase wraps; refuse mixed roles to the interpreter."""

        wrap_pairs: set[tuple[str, str]] = set()
        for site, func in zip(self.reduce_sites, self.reduce_funcs, strict=True):
            if func.roles != {"func"}:
                raise StreamNeedsInterpreter("a callable is also used as a value or a class")
            if func.kind == "global" and func.pair is not None:
                wrap_pairs.add(func.pair)
            else:
                self.reduce_sites[site.derived_from].result_callable = True
        for sym in self.global_syms:
            if sym.pair in wrap_pairs and sym.roles != {"func"}:
                raise StreamNeedsInterpreter("a global is used both as a callable and otherwise")
        return PrescanPlan(
            opcode_count=self.opcode_count,
            resolved=self.resolved,
            wrap_pairs=frozenset(wrap_pairs),
            reduce_sites=self.reduce_sites,
            construction_refusal=self.construction_refusal,
            build_refusal=self.build_refusal,
        )


_CONST_OPCODES = frozenset(
    {
        "INT",
        "BININT",
        "BININT1",
        "BININT2",
        "LONG",
        "LONG1",
        "LONG4",
        "STRING",
        "BINSTRING",
        "SHORT_BINSTRING",
        "BINBYTES",
        "SHORT_BINBYTES",
        "BINBYTES8",
        "BYTEARRAY8",
        "NONE",
        "NEWTRUE",
        "NEWFALSE",
        "UNICODE",
        "BINUNICODE",
        "SHORT_BINUNICODE",
        "BINUNICODE8",
        "FLOAT",
        "BINFLOAT",
    }
)
_GET_OPCODES = frozenset({"GET", "BINGET", "LONG_BINGET"})
_PUT_OPCODES = frozenset({"PUT", "BINPUT", "LONG_BINPUT"})
_EMPTY_CONTAINER_OPCODES = {
    "EMPTY_LIST": "list",
    "EMPTY_TUPLE": "tuple",
    "EMPTY_DICT": "dict",
    "EMPTY_SET": "set",
}
_TUPLE_N_OPCODES = {"TUPLE1": 1, "TUPLE2": 2, "TUPLE3": 3}
_MARK_CONTAINER_OPCODES = {
    "TUPLE": "tuple",
    "LIST": "list",
    "DICT": "dict",
    "FROZENSET": "frozenset",
}


def _next_opcode(ops: Any) -> tuple[Any, Any, int]:
    """Return the next ``(opcode, arg, pos)``; map reader failures to corruption."""

    try:
        return next(ops)
    except StopIteration:
        raise _corrupt("pickle exhausted before seeing STOP") from None
    except (ValueError, EOFError, OverflowError) as exc:
        # ``pickletools.genops`` reports truncation, unknown opcodes and bad operands
        # with these builtins; the C machine would refuse the same bytes.
        raise _corrupt(f"{type(exc).__name__}: {exc}") from exc


def count_opcodes(data: bytes, max_opcodes: int) -> int:
    """Return the opcode count of ``data``, refusing past ``max_opcodes``."""

    count = 0
    ops = pickletools.genops(data)
    while True:
        opcode, _arg, _pos = _next_opcode(ops)
        count += 1
        if count > max_opcodes:
            raise OpcodeCeilingExceeded(_CORRUPT + "opcode count exceeds the allocation ceiling")
        if opcode.name == "STOP":
            return count


def prescan_stream(
    data: bytes,
    *,
    resolve: Callable[[str, str], Any],
    construction_refusal: ConstructionRefusal,
    build_refusal: BuildRefusal,
    build_type_refusal: BuildTypeRefusal,
    max_opcodes: int | None = None,
) -> PrescanPlan:
    """Walk ``data`` symbolically and return the plan the C phase enforces.

    Parameters
    ----------
    data:
        The complete pickle stream.
    resolve:
        The ``find_class`` policy; called once per distinct ``(module, name)``.
    construction_refusal:
        ``(opcode, callable_or_class, args, kwargs) -> reason | None``: the allocation
        belt for a construction with literal arguments.
    build_refusal:
        ``(target_object) -> reason | None`` for a BUILD applied to a resolved global.
    build_type_refusal:
        ``(target_class) -> reason | None`` for a BUILD applied to an instance a class
        construction produced.
    max_opcodes:
        Opcode-count ceiling; exceeding it raises :class:`OpcodeCeilingExceeded`.

    Raises
    ------
    pickle.UnpicklingError
        A refusal (allowlist, belt, BUILD) or stream corruption.
    StreamNeedsInterpreter
        The stream must run through the pure-Python machine instead.
    """

    walker = _Walker(
        data,
        resolve=resolve,
        construction_refusal=construction_refusal,
        build_refusal=build_refusal,
        build_type_refusal=build_type_refusal,
        max_opcodes=max_opcodes,
    )
    return walker.run()
