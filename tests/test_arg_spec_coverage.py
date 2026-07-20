"""Coverage gate for decorated torch argument extraction specs."""

from __future__ import annotations

import pytest
import torch

from torchlens.capture.arg_positions import (
    FUNC_ARG_SPECS,
    ArgSpec,
    _cache_dynamic_spec,
    _normalize_func_name,
)
from torchlens.constants import get_orig_torch_funcs


_HIGH_CONFIDENCE_STATIC_NAMES = frozenset(
    {
        "adaptivemaxpool1dwithindices",
        "adaptivemaxpool2dwithindices",
        "adaptivemaxpool3dwithindices",
        "addr",
        "airyai",
        "alignas",
        "aminmax",
        "argwhere",
        "besselj0",
        "besselj1",
        "bessely0",
        "bessely1",
        "binarycrossentropy",
        "bucketize",
        "cdist",
        "chainmatmul",
        "chebyshevpolynomialt",
        "chebyshevpolynomialu",
        "chebyshevpolynomialv",
        "chebyshevpolynomialw",
        "choleskyex",
        "choleskyinverse",
        "cond",
        "corrcoef",
        "cov",
        "cumulativetrapezoid",
        "dequantize",
        "diff",
        "dropout1d",
        "entr",
        "erfcx",
        "fakequantizeperchannelaffine",
        "fakequantizepertensoraffine",
        "frexp",
        "gammainc",
        "gammaincc",
        "gammaln",
        "gcd",
        "geqrf",
        "ger",
        "glu",
        "gradient",
        "gumbelsoftmax",
        "hermitepolynomialh",
        "hermitepolynomialhe",
        "hfftn",
        "hspmm",
        "igamma",
        "igammac",
        "ihfftn",
        "inv",
        "invex",
        "isconj",
        "isin",
        "isinference",
        "isnonzero",
        "issetto",
        "laguerrepolynomiall",
        "lcm",
        "ldexp",
        "ldlfactor",
        "ldlfactorex",
        "ldlsolve",
        "legendrepolynomialp",
        "lobpcg",
        "localresponsenorm",
        "logaddexp",
        "logaddexp2",
        "logndtr",
        "lppool3d",
        "lufactor",
        "lufactorex",
        "map",
        "modifiedbesseli0",
        "modifiedbesseli1",
        "modifiedbesselk0",
        "modifiedbesselk1",
        "moduleload",
        "multigammaln",
        "multinomial",
        "nanquantile",
        "ndtr",
        "ndtri",
        "normalize",
        "orgqr",
        "ormqr",
        "pairwisedistance",
        "pcalowrank",
        "pdist",
        "pinv",
        "poisson",
        "psi",
        "quantile",
        "quantizeperchannel",
        "quantizepertensor",
        "quantizepertensordynamic",
        "randintlike",
        "renorm",
        "resize",
        "scaledmodifiedbesselk0",
        "scaledmodifiedbesselk1",
        "shiftedchebyshevpolynomialt",
        "shiftedchebyshevpolynomialu",
        "shiftedchebyshevpolynomialv",
        "shiftedchebyshevpolynomialw",
        "smm",
        "softmin",
        "solveex",
        "solvetriangular",
        "sspaddmm",
        "stdmean",
        "svdlowrank",
        "svdvals",
        "tensorinv",
        "tensorsolve",
        "trapezoid",
        "trapz",
        "tripletmarginwithdistanceloss",
        "truncnormal",
        "unravelindex",
        "unsafechunk",
        "unsafesplit",
        "varmean",
        "vecdot",
        "zeta",
    }
)

_KNOWN_UNSUPPORTED_ARG_SPEC_REASONS = {
    "addbatchdim": "internal/private helper left on dynamic fallback until independently validated",
    "adddocstr": "internal/private helper left on dynamic fallback until independently validated",
    "addmmactivation": "internal/private helper left on dynamic fallback until independently validated",
    "addrelu": "internal/private helper left on dynamic fallback until independently validated",
    "apply": "demoted fragment: no operator schema available",
    "array": "Python/operator protocol helper with nonstandard callable metadata",
    "arraywrap": "Python/operator protocol helper with nonstandard callable metadata",
    "asarray": "C-level tensor factory with non-introspectable signature (wrapped for capture coverage)",
    "emptypermuted": "C-level tensor factory with non-introspectable signature (wrapped for capture coverage)",
    "emptystrided": "C-level tensor factory with non-introspectable signature (wrapped for capture coverage)",
    "frombuffer": "C-level tensor factory with non-introspectable signature (wrapped for capture coverage)",
    "assertasync": "metadata/control helper with no validated tensor-input schema",
    "assertscalar": "metadata/control helper with no validated tensor-input schema",
    "asserttensormetadata": "metadata/control helper with no validated tensor-input schema",
    "backward": "autograd/backward helper left on dynamic fallback pending gradient schema audit",
    "batchnormbackwardelemt": "autograd/backward helper left on dynamic fallback pending gradient schema audit",
    "batchnormbackwardreduce": "autograd/backward helper left on dynamic fallback pending gradient schema audit",
    "clearnonserializablecacheddata": "metadata/control helper with no validated tensor-input schema",
    "cudnnrnn": "demoted fragment: schema mismatch: missing_positions=[5, 15], extra_positions=[14], missing_names=[]",
    "debughasinternaloverlap": "internal/private helper left on dynamic fallback until independently validated",
    "deepcopy": "demoted fragment: no operator schema available",
    "dimarange": "Python/operator protocol helper with nonstandard callable metadata",
    "dimi": "Python/operator protocol helper with nonstandard callable metadata",
    "dimorder": "Python/operator protocol helper with nonstandard callable metadata",
    "dimv": "Python/operator protocol helper with nonstandard callable metadata",
    "dsmm": "demoted fragment: no operator schema available",
    "fanmode": "internal/private helper left on dynamic fallback until independently validated",
    "functionalassertasync": "internal/private helper left on dynamic fallback until independently validated",
    "functionalassertscalar": "internal/private helper left on dynamic fallback until independently validated",
    "functionalsymconstrainrangeforsize": "internal/private helper left on dynamic fallback until independently validated",
    "getdevice": "metadata/control helper with no validated tensor-input schema",
    "getsoftmaxdim": "metadata/control helper with no validated tensor-input schema",
    "h": "demoted fragment: no operator schema available",
    "handletorchfunction": "internal/private helper left on dynamic fallback until independently validated",
    "hsmm": "demoted fragment: no operator schema available",
    "ipu": "demoted fragment: no operator schema available",
    "issamesize": "metadata/control helper with no validated tensor-input schema",
    "isshared": "metadata/control helper with no validated tensor-input schema",
    "isview": "metadata/control helper with no validated tensor-input schema",
    "iszerotensor": "metadata/control helper with no validated tensor-input schema",
    "lazyclone": "internal/private helper left on dynamic fallback until independently validated",
    "linalgcheckerrors": "internal/private helper left on dynamic fallback until independently validated",
    "linalgdet": "internal/private helper left on dynamic fallback until independently validated",
    "linalgeigh": "internal/private helper left on dynamic fallback until independently validated",
    "linalgslogdet": "internal/private helper left on dynamic fallback until independently validated",
    "linalgsolveex": "internal/private helper left on dynamic fallback until independently validated",
    "linalgsvd": "internal/private helper left on dynamic fallback until independently validated",
    "logsoftmaxbackwarddata": "autograd/backward helper left on dynamic fallback pending gradient schema audit",
    "luwithinfo": "internal/private helper left on dynamic fallback until independently validated",
    "makedeprecate": "metadata/control helper with no validated tensor-input schema",
    "makedual": "metadata/control helper with no validated tensor-input schema",
    "map2": "internal/private helper left on dynamic fallback until independently validated",
    "maskedscale": "internal/private helper left on dynamic fallback until independently validated",
    "maskedsoftmax": "internal/private helper left on dynamic fallback until independently validated",
    "mixeddtypeslinear": "internal/private helper left on dynamic fallback until independently validated",
    "mh": "demoted fragment: no operator schema available",
    "mkldnnreshape": "internal/private helper left on dynamic fallback until independently validated",
    "mkldnntranspose": "internal/private helper left on dynamic fallback until independently validated",
    "mod": "internal/private helper left on dynamic fallback until independently validated",
    "mt": "internal/private helper left on dynamic fallback until independently validated",
    "mtia": "demoted fragment: no operator schema available",
    "nonlinearitytype": "internal/private helper left on dynamic fallback until independently validated",
    "op": "Python/operator protocol helper with nonstandard callable metadata",
    "optional": "Python/operator protocol helper with nonstandard callable metadata",
    "overload": "Python/operator protocol helper with nonstandard callable metadata",
    "packpaddedsequence": "internal/private helper left on dynamic fallback until independently validated",
    "padpackedsequence": "internal/private helper left on dynamic fallback until independently validated",
    "print": "internal/private helper left on dynamic fallback until independently validated",
    "propagatexladata": "demoted fragment: schema mismatch: missing_positions=[], extra_positions=[1], missing_names=[]",
    "removebatchdim": "metadata/control helper with no validated tensor-input schema",
    "reversed": "internal/private helper left on dynamic fallback until independently validated",
    "saddmm": "demoted fragment: no operator schema available",
    "setstate": "metadata/control helper with no validated tensor-input schema",
    "softmaxbackwarddata": "autograd/backward helper left on dynamic fallback pending gradient schema audit",
    "sparselogsoftmaxbackwarddata": "demoted fragment: schema mismatch: missing_positions=[], extra_positions=[1], missing_names=[]",
    "sparsesoftmaxbackwarddata": "demoted fragment: schema mismatch: missing_positions=[], extra_positions=[1], missing_names=[]",
    "spmm": "demoted fragment: no operator schema available",
    "standardgammagrad": "demoted fragment: schema mismatch: missing_positions=[], extra_positions=[1], missing_names=[]",
    "symite": "Python/operator protocol helper with nonstandard callable metadata",
    "symsqrt": "Python/operator protocol helper with nonstandard callable metadata",
    "symsum": "Python/operator protocol helper with nonstandard callable metadata",
    "testautogradmultipledispatch": "autograd/backward helper left on dynamic fallback pending gradient schema audit",
    "testautogradmultipledispatchview": "autograd/backward helper left on dynamic fallback pending gradient schema audit",
    "testautogradmultipledispatchviewcopy": "autograd/backward helper left on dynamic fallback pending gradient schema audit",
    "testchecktensor": "internal/private helper left on dynamic fallback until independently validated",
    "testfunctorchfallback": "internal/private helper left on dynamic fallback until independently validated",
    "testparallelmaterialize": "internal/private helper left on dynamic fallback until independently validated",
    "testserializationsubcmul": "internal/private helper left on dynamic fallback until independently validated",
    "unpackdual": "internal/private helper left on dynamic fallback until independently validated",
    "wrappedlinearprepack": "internal/private helper left on dynamic fallback until independently validated",
    "wrappedquantizedlinearprepacked": "demoted fragment: schema mismatch: missing_positions=[], extra_positions=[4, 5], missing_names=[]",
    "xpu": "demoted fragment: no operator schema available",
}

_KNOWN_UNSUPPORTED_ARG_SPECS = frozenset(_KNOWN_UNSUPPORTED_ARG_SPEC_REASONS)

# Torch adds/removes internal/private helpers across releases, so a few known-unsupported
# names are decorated on some torch versions and absent on others (e.g. "op" is absent on
# torch 2.8). These are torch-version differences, not stale entries; keep them out of the
# strict "every known entry is decorated on THIS torch" guard below.
_TORCH_VERSION_VARYING_UNSUPPORTED = frozenset({"op"})

_EXPECTED_KWARGS = {
    "addr": ("input", "vec1", "vec2"),
    "alignas": ("self", "other"),
    "binarycrossentropy": ("input", "target", "weight"),
    "bucketize": ("input", "boundaries"),
    "cdist": ("x1", "x2"),
    "choleskyinverse": ("L",),
    "cov": ("input", "fweights", "aweights"),
    "cumulativetrapezoid": ("y", "x"),
    "diff": ("input", "prepend", "append"),
    "fakequantizeperchannelaffine": ("input", "scale", "zero_point"),
    "fakequantizepertensoraffine": ("input", "scale", "zero_point"),
    "ger": ("input", "vec2"),
    "gumbelsoftmax": ("logits",),
    "hspmm": ("mat1", "mat2"),
    "isin": ("elements", "test_elements"),
    "issetto": ("self", "tensor"),
    "ldlsolve": ("LD", "pivots", "B"),
    "lobpcg": ("A", "B", "X", "iK"),
    "map": ("self", "tensor"),
    "moduleload": ("self", "other"),
    "orgqr": ("input", "tau"),
    "ormqr": ("input", "tau", "other"),
    "pairwisedistance": ("x1", "x2"),
    "quantizeperchannel": ("input", "scales", "zero_points"),
    "quantizepertensor": ("input", "scale", "zero_point"),
    "smm": ("input", "mat"),
    "solveex": ("A", "B"),
    "solvetriangular": ("A", "B"),
    "sspaddmm": ("input", "mat1", "mat2"),
    "svdlowrank": ("A", "M"),
    "tensorsolve": ("A", "B"),
    "trapezoid": ("y", "x"),
    "trapz": ("y", "x"),
    "tripletmarginwithdistanceloss": ("anchor", "positive", "negative"),
    "truncnormal": ("tensor",),
    "unravelindex": ("indices",),
    "vecdot": ("x", "y"),
}


def _decorated_normalized_names() -> set[str]:
    """Return normalized function names from the actual decorated set.

    Returns
    -------
    set[str]
        Normalized names derived from the actual first-wrap function target set.
    """

    return {_normalize_func_name(func_name.strip("_")) for _, func_name in get_orig_torch_funcs()}


def test_every_decorated_arg_spec_is_static_or_explicitly_unsupported() -> None:
    """Every decorated function has a static spec or a known unsupported entry."""

    decorated_names = _decorated_normalized_names()
    static_names = set(FUNC_ARG_SPECS)
    missing = decorated_names - static_names

    assert missing <= _KNOWN_UNSUPPORTED_ARG_SPECS
    # Known-unsupported entries not decorated on this torch version are torch-version
    # differences (see _TORCH_VERSION_VARYING_UNSUPPORTED); anything else undecorated is a
    # stale/typo entry and must be caught.
    undecorated_known = _KNOWN_UNSUPPORTED_ARG_SPECS - decorated_names
    assert undecorated_known <= _TORCH_VERSION_VARYING_UNSUPPORTED, (
        "known-unsupported arg-spec entries are undecorated on this torch and not marked "
        f"version-varying: {sorted(undecorated_known - _TORCH_VERSION_VARYING_UNSUPPORTED)}"
    )
    assert not (_KNOWN_UNSUPPORTED_ARG_SPECS & static_names)


def test_high_confidence_static_fills_remain_covered() -> None:
    """High-confidence schema-derived fills must stay in the static table."""

    decorated_names = _decorated_normalized_names()

    assert _HIGH_CONFIDENCE_STATIC_NAMES <= decorated_names
    assert _HIGH_CONFIDENCE_STATIC_NAMES <= set(FUNC_ARG_SPECS)
    assert not (_HIGH_CONFIDENCE_STATIC_NAMES & _KNOWN_UNSUPPORTED_ARG_SPECS)


@pytest.mark.parametrize("func_name,expected_kwargs", sorted(_EXPECTED_KWARGS.items()))
def test_high_confidence_tensor_kwarg_names_match_schema(
    func_name: str, expected_kwargs: tuple[str, ...]
) -> None:
    """Selected static specs must retain their schema-backed tensor kwarg names."""

    spec = FUNC_ARG_SPECS[func_name]

    assert set(expected_kwargs) <= set(spec.tensor_kwargs)


def test_dynamic_fallback_cache_is_self_consistent_for_unlisted_schema() -> None:
    """Tier-3 BFS fallback records observed tensor positions and kwarg names."""

    normalized_name = "torchlenslocalargspeccoverage"
    tensor_arg = torch.randn(2, 3)
    tensor_kwarg = torch.randn(2, 3)
    found_tensors = [tensor_arg, tensor_kwarg]
    found_params: list[torch.nn.Parameter] = []

    _cache_dynamic_spec(
        normalized_name,
        (tensor_arg,),
        {"right": tensor_kwarg},
        found_tensors,
        found_params,
    )
    spec = FUNC_ARG_SPECS.get(normalized_name)
    dynamic_spec = __import__("torchlens")._state._dynamic_arg_specs.pop(normalized_name)

    assert spec is None
    assert dynamic_spec == ArgSpec(positions=(0,), tensor_kwargs=("right",))


def test_no_binary_factory_argspec_key_collision() -> None:
    """Binary-op and factory-func arg-spec tables must never share a normalized key.

    They assign contradictory specs (tensor parents at positions 0,1 vs. NO tensor parents);
    a shared key + dict last-writer-wins silently drops parents -- the historic
    __rand__/torch.rand "rand" collision. arg_positions enforces this at import; assert it here.
    """
    from torchlens.capture.arg_positions import _BINARY_FUNCS, _FACTORY_FUNCS

    assert set(_BINARY_FUNCS) & set(_FACTORY_FUNCS) == set()


@pytest.mark.parametrize(
    "dunder,forward",
    [
        ("__rand__", "and"),
        ("__ror__", "or"),
        ("__rxor__", "xor"),
        ("__radd__", "add"),
        ("__rmul__", "mul"),
    ],
)
def test_commutative_reflected_dunders_normalize_to_forward_op(dunder: str, forward: str) -> None:
    """COMMUTATIVE reflected dunders normalize to their forward op (swapped order is irrelevant).

    (``__rand__`` -> ``and``, matching ``__and__``; the ``torch.rand`` factory keeps ``rand``.)
    """
    assert _normalize_func_name(dunder) == forward
    assert _normalize_func_name("rand") == "rand"


@pytest.mark.parametrize(
    "dunder,expected",
    [
        ("__rsub__", "rsub"),
        ("__rtruediv__", "rtruediv"),
        ("__rfloordiv__", "rfloordiv"),
        ("__rmod__", "rmod"),
        ("__rpow__", "rpow"),
        ("__rmatmul__", "rmatmul"),
        ("__rlshift__", "rlshift"),
        ("__rrshift__", "rrshift"),
    ],
)
def test_noncommutative_reflected_dunders_keep_reflected_name(dunder: str, expected: str) -> None:
    """NON-commutative reflected dunders keep their r-prefixed name (``other <op> self``: operands
    are swapped, so the ``r`` is meaningful -- e.g. ``rsub`` matches the real ``torch.rsub``)."""
    assert _normalize_func_name(dunder) == expected


def test_reflected_bitwise_and_captures_tensor_parent() -> None:
    """Regression: reflected ``int & tensor`` (Tensor.__rand__) captures its tensor parent and
    is labeled by the forward op (``and``), not the ``rand`` factory that silently dropped it.
    """
    import torchlens as tl

    class _Reflected(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = x.int() + 1  # traced int tensor
            z = 5 & y  # reflected: y.__rand__(5) -- parent MUST be y
            return z.float().sum()

    trace = tl.trace(_Reflected(), torch.randint(0, 7, (4,)))
    and_ops = [label for label in trace.layer_labels if label.startswith("and_")]
    assert and_ops, "reflected & should produce an 'and_' op"
    assert not [label for label in trace.layer_labels if label.startswith("rand_")], (
        "reflected & must not be mislabeled as the 'rand' factory"
    )
    op = trace[and_ops[0]]
    assert op.parents, "reflected & op must capture its tensor parent, not be orphaned"
    assert op.is_internal_source is False
