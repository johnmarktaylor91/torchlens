# FAITHFUL PORT of nikhilxb/ncap-quadruped @ main (original framework: PyTorch, but with a custom
# in-repo `circuits`/`tonic`/`gym.spaces.Dict` harness that is not installable/reasonably vendorable)
# https://github.com/nikhilxb/ncap-quadruped
# Paper: Bhattasali, Zador, Engel. "Neural Circuit Architectural Priors for Quadruped Locomotion."
# NeurIPS 2022. Project page: https://ncap-quadruped.github.io/
#
# This ports the "Circuit" (NCAP) modular quadruped architecture from
# `ncap/quadruped/models/modular.py`: a biologically-inspired spinal-cord neural circuit composed
# of a Rhythm Generation (RG) oscillator circuit (Danner et al. 2019), per-limb Pattern Formation
# (PF) and Afferent Feedback (AF) linear networks, and a Brainstem Command (BC) signal, as used by
# the `spinal_bccmd` fixed-speed config (`configs/quadruped/model/es/unitree_a1/spinal_bccmd.yaml`).
#
# Ported faithfully, unit-for-unit and connection-for-connection, from `libs/circuits/{units,
# connector,signal,simulator}.py` (the `Basic`/`Oscillator` backward-Euler circuit dynamics and
# weighted-delay `Connector`) and `ncap/quadruped/models/modular.py` (`RhythmGenerationCircuit`
# wiring + `init_rg_default` biases, `PatternFormationNetwork`/`AfferentFeedbackNetwork`/
# `LimbSubnetwork` with `init_pf_flxext`/`init_af_flxext` weight init, `BrainstemCommandSignal`).
#
# Dropped as RL-harness plumbing (not architecture): `gym.spaces.Dict` observation/action space
# bookkeeping, the `tonic` RL library's `DeterministicPolicyHead` wrapper class (inlined below as
# the `nn.Linear` + activation it actually constructs), the Hydra config system, the
# `UnflatNormalizer`/`NegPos` observation normalizer, and the `Constrainer` (a training-time-only
# gradient-clamp utility applied between optimizer steps, not part of the forward computation).
# The `forward()` here performs one Euler-discretized circuit step (matching `Simulator.step()`)
# given the current oscillator state and per-limb observations, and returns the next state plus
# the per-limb joint actions -- this is the same operation the original repeatedly calls once per
# environment timestep during a rollout.

import math
import typing as T

import torch
import torch.nn as nn

LimbName = T.Literal["FL", "FR", "HL", "HR"]

# ==================================================================================================
# `circuits` library primitives (ported from libs/circuits/{units,connector,signal}.py)


def unitrelu(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0, max=1)


def halflinear(v: torch.Tensor, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return (v > 0) * (a + 1) / 2


def _linear(start: float, end: float, z: torch.Tensor) -> torch.Tensor:
    return start * (1 - z) + end * z


class BasicUnit:
    """A basic neuron with an internal voltage variable (ported from `circuits.units.Basic`)."""

    def __init__(
        self,
        bias: float = 0.0,
        voltage_time: float = 50.0,
        activation: T.Callable[[torch.Tensor], torch.Tensor] = unitrelu,
    ):
        assert -1 <= bias <= 1
        self.bias = bias
        self.voltage_time = voltage_time
        self.activation = activation

    def step(
        self, v: torch.Tensor, z_in: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.clamp(z_in + self.bias, min=-1, max=1)
        kv = 4 / self.voltage_time * dt
        v_next = (v + z * kv) / (1 + kv)
        value = self.activation(v_next)
        return v_next, value


class OscillatorUnit:
    """A bursting neuron switching active/quiet states (ported from `circuits.units.Oscillator`)."""

    def __init__(
        self,
        bias: float = 0.0,
        adaptation_time: float = 2000.0,
        active_time: float = 500.0,
        quiet_time: float = 1000.0,
        active_scale: float = 0.5,
        quiet_scale: float = 0.1,
        tonic_threshold: float = 1.0,
        activation: T.Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ] = halflinear,
    ):
        assert -1 <= bias <= 1
        self.bias = bias
        self.adaptation_time = adaptation_time
        self.active_time = active_time
        self.quiet_time = quiet_time
        self.active_scale = active_scale
        self.quiet_scale = quiet_scale
        self.tonic_threshold = tonic_threshold
        self.activation = activation

        ka = 4 * active_time / adaptation_time
        kq = 4 * quiet_time / adaptation_time
        self.active_bound0 = (1 - math.exp(kq)) / (1 - math.exp(ka + kq))
        self.quiet_bound0 = self.active_bound0 * math.exp(ka)
        self.active_bound1 = (1 - math.exp(kq * quiet_scale)) / (
            1 - math.exp(ka * active_scale + kq * quiet_scale)
        )
        self.quiet_bound1 = self.active_bound1 * math.exp(ka * active_scale)

    def init_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        v0 = -torch.ones(1)
        a0 = torch.rand(1) * (self.quiet_bound0 - self.active_bound0) * 0.1 + self.active_bound0
        return v0, a0

    def step(
        self, v: torch.Tensor, a: torch.Tensor, z_in: torch.Tensor, dt: float
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = z_in + self.bias
        z = torch.clamp(x, min=0, max=1)

        k = 4 / self.adaptation_time * dt
        a_next = (v < 0) * (a + k) / (1 + k) + (v > 0) * a / (1 + k)
        a_next = torch.clamp(a_next, min=0, max=1)

        active_bound = _linear(self.active_bound0, self.active_bound1, z)
        quiet_bound = _linear(self.quiet_bound0, self.quiet_bound1, z)
        active_to_quiet = (a_next <= active_bound) & (x <= self.tonic_threshold)
        quiet_to_active = (a_next >= quiet_bound) & (x >= 0)
        v_next = v.clone()
        v_next = torch.where(active_to_quiet, -torch.ones_like(v_next), v_next)
        v_next = torch.where(quiet_to_active, torch.ones_like(v_next), v_next)

        value = self.activation(v_next, a_next, z)
        return v_next, a_next, value


class Synapse(T.NamedTuple):
    source: str  # name of source unit/signal in the RG circuit state dict
    weight: float
    delay: int = 1  # in timesteps; every synapse in `init_rg_default` uses the default delay=1


# ==================================================================================================
# Rhythm Generation (RG) circuit (ported from `ncap.quadruped.models.RhythmGenerationCircuit` +
# `init_rg_default`, `Danner et al. 2019`-style spinal oscillator network).

_LIMBS: tuple[LimbName, ...] = ("FL", "FR", "HL", "HR")


def _init_rg_default_params() -> dict[str, float]:
    """Bias/weight values from `init_rg_default` (fixed-speed default RG initialization)."""
    p: dict[str, float] = {}
    for limb in _LIMBS:
        p[f"bias_flx_{limb}"] = 0.1
        p[f"bias_ext_{limb}"] = 0.8
        p[f"bias_v0d_{limb}"] = -0.1
        p[f"bias_v0v_{limb}"] = -0.1
        p[f"bias_v3f_{limb}"] = -0.4
        p[f"bias_v3e_{limb}"] = -0.0
        p[f"bias_v3a_{limb}"] = -0.4
        p[f"bias_in2_{limb}"] = -0.1
        p[f"osc_flxext_i_{limb}"] = -1.0
        p[f"osc_extflx_i_{limb}"] = -0.1
        p[f"cross_flxflx_i_{limb}"] = -1.5
        p[f"cross_flxflx_e_{limb}"] = 0.3
        p[f"cross_extext_e_{limb}"] = 0.05
        p[f"sided_flxflx_i_{limb}"] = -0.1
        p[f"sided_extflx_e_{limb}"] = 0.1
        p[f"sidea_extflx_e_{limb}"] = 0.1
        p[f"diagd_flxflx_i_{limb}"] = -0.8
        p[f"diagd_flxflx_e_{limb}"] = 0.2
        p[f"diagd_in2v0d_i_{limb}"] = -0.2
        p[f"diaga_flxflx_e_{limb}"] = 0.2
        p[f"diaga_v3ain2_e_{limb}"] = 0.8
        p[f"aff_flx_{limb}"] = 1.0
        p[f"aff_ext_{limb}"] = 1.0
    for side in ("L", "R"):
        p[f"freq_osc_e_{side}"] = 1.0
        p[f"freq_cross_e_{side}"] = 1.0
        p[f"freq_diaga_e_{side}"] = 0.8
        p[f"sync_cross_i_{side}"] = -0.8
        p[f"sync_diagdi_i_{side}"] = -0.5
        p[f"sync_diagde_i_{side}"] = -0.5
    return p


_RG_OSC_KWARGS = dict(
    adaptation_time=1400.0,
    active_time=200.0,
    quiet_time=1200.0,
    active_scale=100 / 200,
    quiet_scale=60 / 1200,
    tonic_threshold=3.0,
)
_RG_BASIC_KWARGS = dict(voltage_time=30.0)


class RhythmGenerationCircuit(nn.Module):
    """Spinal-cord Rhythm Generation (RG) circuit coordinating gait between all 4 limbs.

    Ported unit-for-unit and connection-for-connection from
    `ncap.quadruped.models.RhythmGenerationCircuit` (default/"Danner" wiring, `init_rg_default`).
    """

    def __init__(self, timestep: float = 30.0):
        super().__init__()
        self.timestep = timestep
        p = _init_rg_default_params()
        self.p = p

        # Flexor/extensor oscillator+basic units per limb.
        self.flx: dict[LimbName, OscillatorUnit] = {}
        self.ext: dict[LimbName, BasicUnit] = {}
        for limb in _LIMBS:
            osc_kwargs = dict(_RG_OSC_KWARGS)
            if limb in ("FL", "HL"):
                osc_kwargs["adaptation_time"] = osc_kwargs["adaptation_time"] * 1.05
            self.flx[limb] = OscillatorUnit(bias=p[f"bias_flx_{limb}"], **osc_kwargs)
            self.ext[limb] = BasicUnit(bias=p[f"bias_ext_{limb}"], **_RG_BASIC_KWARGS)

        # Interneurons (all `Basic` units, same voltage_time/activation as ext units).
        interneuron_names = []
        for limb in _LIMBS:
            interneuron_names += [f"cross_v0d_{limb}", f"cross_v3f_{limb}", f"cross_v3e_{limb}"]
        for limb in ("FL", "FR"):
            interneuron_names += [f"diagd_v0d_{limb}", f"diagd_v0v_{limb}", f"diagd_in2_{limb}"]
        for limb in ("HL", "HR"):
            interneuron_names += [f"diaga_v3a_{limb}"]
        self.inter: dict[str, BasicUnit] = {
            name: BasicUnit(bias=p[self._bias_key(name)], **_RG_BASIC_KWARGS)
            for name in interneuron_names
        }

        # Synapse wiring: dst_unit_name -> list of Synapse(source_name, weight). `source_name` may
        # reference a flx/ext/inter unit's *output value*, or an external "aff_flx_{limb}"/
        # "aff_ext_{limb}"/"freq_{L,R}"/"sync_{L,R}" input signal. See `_wire` for the exact,
        # faithful connection list (ported from `RhythmGenerationCircuit.__init__`).
        self._syn = self._wire(p)

    @staticmethod
    def _bias_key(unit_name: str) -> str:
        # e.g. "cross_v0d_FL" -> "bias_v0d_FL"; "diagd_in2_FL" -> "bias_in2_FL"; etc.
        parts = unit_name.split("_")
        limb = parts[-1]
        mid = "_".join(parts[1:-1])
        return f"bias_{mid}_{limb}"

    def _all_unit_names(self) -> list[str]:
        names = []
        for limb in _LIMBS:
            names += [f"flx_{limb}", f"ext_{limb}"]
            names += [f"cross_v0d_{limb}", f"cross_v3f_{limb}", f"cross_v3e_{limb}"]
        for limb in ("FL", "FR"):
            names += [f"diagd_v0d_{limb}", f"diagd_v0v_{limb}", f"diagd_in2_{limb}"]
        for limb in ("HL", "HR"):
            names += [f"diaga_v3a_{limb}"]
        return names

    def _wire(self, p: dict[str, float]) -> dict[str, list[Synapse]]:
        """Exact connection list from `RhythmGenerationCircuit.__init__` in `modular.py`."""
        syn: dict[str, list[Synapse]] = {name: [] for name in self._all_unit_names()}

        def add(dst: str, src: str, weight: float):
            syn[dst].append(Synapse(src, weight))

        # Oscillator connections (osc): flx <-> ext mutual inhibition.
        for a in _LIMBS:
            add(f"flx_{a}", f"ext_{a}", p[f"osc_extflx_i_{a}"])
            add(f"ext_{a}", f"flx_{a}", p[f"osc_flxext_i_{a}"])

        # Cross-side connections (cross).
        for a, b in (("FR", "FL"), ("FL", "FR"), ("HR", "HL"), ("HL", "HR")):
            add(f"cross_v3f_{a}", f"flx_{a}", 1.0)
            add(f"cross_v0d_{a}", f"flx_{a}", 1.0)
            add(f"cross_v3e_{a}", f"ext_{a}", 1.0)
            add(f"flx_{b}", f"cross_v0d_{a}", p[f"cross_flxflx_i_{b}"])
            add(f"flx_{b}", f"cross_v3f_{a}", p[f"cross_flxflx_e_{b}"])
            add(f"ext_{b}", f"cross_v3e_{a}", p[f"cross_extext_e_{b}"])

        # Same-side, descending connections (sided).
        for a, b in (("FL", "HL"), ("FR", "HR")):
            add(f"flx_{b}", f"ext_{a}", p[f"sided_extflx_e_{b}"])
            add(f"flx_{b}", f"flx_{a}", p[f"sided_flxflx_i_{b}"])

        # Same-side, ascending connections (sidea).
        for a, b in (("HL", "FL"), ("HR", "FR")):
            add(f"flx_{b}", f"ext_{a}", p[f"sidea_extflx_e_{b}"])

        # Diagonal, descending connections (diagd).
        for a, b in (("FL", "HR"), ("FR", "HL")):
            add(f"diagd_v0d_{a}", f"flx_{a}", 1.0)
            add(f"diagd_v0v_{a}", f"flx_{a}", 1.0)
            add(f"flx_{b}", f"diagd_v0d_{a}", p[f"diagd_flxflx_i_{b}"])
            add(f"flx_{b}", f"diagd_v0v_{a}", p[f"diagd_flxflx_e_{b}"])
            add(f"diagd_in2_{a}", f"cross_v3f_{a}", 1.0)
            add(f"diagd_v0d_{a}", f"diagd_in2_{a}", p[f"diagd_in2v0d_i_{a}"])

        # Diagonal, ascending connections (diaga).
        for a, b in (("HL", "FR"), ("HR", "FL")):
            add(f"diaga_v3a_{a}", f"flx_{a}", 1.0)
            add(f"flx_{b}", f"diaga_v3a_{a}", p[f"diaga_flxflx_e_{b}"])
            add(f"diagd_in2_{b}", f"diaga_v3a_{a}", p[f"diaga_v3ain2_e_{b}"])

        # Afferent connections (aff): external per-limb sensory feedback signals.
        for a in _LIMBS:
            add(f"flx_{a}", f"aff_flx_{a}", p[f"aff_flx_{a}"])
            add(f"ext_{a}", f"aff_ext_{a}", p[f"aff_ext_{a}"])

        # Command connections (freq, sync): external brainstem command signals.
        for a in ("L", "R"):
            add(f"flx_F{a}", f"freq_{a}", p[f"freq_osc_e_{a}"])
            add(f"flx_H{a}", f"freq_{a}", p[f"freq_osc_e_{a}"])
            add(f"cross_v3f_F{a}", f"freq_{a}", p[f"freq_cross_e_{a}"])
            add(f"cross_v3f_H{a}", f"freq_{a}", p[f"freq_cross_e_{a}"])
            add(f"diaga_v3a_H{a}", f"freq_{a}", p[f"freq_diaga_e_{a}"])
            add(f"cross_v0d_F{a}", f"sync_{a}", p[f"sync_cross_i_{a}"])
            add(f"cross_v0d_H{a}", f"sync_{a}", p[f"sync_cross_i_{a}"])
            add(f"diagd_v0d_F{a}", f"sync_{a}", p[f"sync_diagdi_i_{a}"])
            add(f"diagd_v0v_F{a}", f"sync_{a}", p[f"sync_diagde_i_{a}"])
        return syn

    def init_state(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for limb in _LIMBS:
            v0, a0 = self.flx[limb].init_state()
            state[f"flx_{limb}_v"] = v0.expand(batch_size, 1).clone()
            state[f"flx_{limb}_a"] = a0.expand(batch_size, 1).clone()
            state[f"ext_{limb}_v"] = torch.zeros(batch_size, 1)
        for name in self.inter:
            state[f"{name}_v"] = torch.zeros(batch_size, 1)
        return state

    def step(
        self,
        state: dict[str, torch.Tensor],
        aff: dict[str, torch.Tensor],
        cmd: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Advances every unit in the circuit by one Euler timestep (matches `Simulator.step()`:
        every unit reads the *previous* timestep's values, then all values update simultaneously).
        """
        dt = self.timestep
        values: dict[str, torch.Tensor] = {}
        for limb in _LIMBS:
            values[f"flx_{limb}"] = self.flx[limb].activation(
                state[f"flx_{limb}_v"],
                state[f"flx_{limb}_a"],
                torch.zeros_like(state[f"flx_{limb}_v"]),
            )
            values[f"ext_{limb}"] = self.ext[limb].activation(state[f"ext_{limb}_v"])
        for name in self.inter:
            values[name] = self.inter[name].activation(state[f"{name}_v"])
        values.update(aff)
        values.update(cmd)

        def integrate(dst: str) -> torch.Tensor:
            total = torch.zeros_like(state[f"flx_{_LIMBS[0]}_v"])
            for syn in self._syn[dst]:
                total = total + syn.weight * values[syn.source]
            return total

        new_state: dict[str, torch.Tensor] = {}
        for limb in _LIMBS:
            v, a = state[f"flx_{limb}_v"], state[f"flx_{limb}_a"]
            v_next, a_next, _ = self.flx[limb].step(v, a, integrate(f"flx_{limb}"), dt)
            new_state[f"flx_{limb}_v"] = v_next
            new_state[f"flx_{limb}_a"] = a_next

            v_ext = state[f"ext_{limb}_v"]
            v_ext_next, _ = self.ext[limb].step(v_ext, integrate(f"ext_{limb}"), dt)
            new_state[f"ext_{limb}_v"] = v_ext_next
        for name in self.inter:
            v_next, _ = self.inter[name].step(state[f"{name}_v"], integrate(name), dt)
            new_state[f"{name}_v"] = v_next

        outputs = {
            limb: torch.concat(
                (
                    self.flx[limb].activation(
                        new_state[f"flx_{limb}_v"],
                        new_state[f"flx_{limb}_a"],
                        torch.zeros_like(new_state[f"flx_{limb}_v"]),
                    ),
                    self.ext[limb].activation(new_state[f"ext_{limb}_v"]),
                ),
                dim=-1,
            )
            for limb in _LIMBS
        }
        return new_state, outputs


# ==================================================================================================
# Pattern Formation (PF) / Afferent Feedback (AF) linear heads (ported from
# `ncap.quadruped.models.LimbSubnetwork` + `init_pf_flxext`/`init_af_flxext`, `mode="share-cross"`).

_PF_FLX = {"FL": [-0.5, -0.2], "HL": [-0.5, -0.2]}
_PF_EXT = {"FL": [+0.3, +0.8], "HL": [+0.3, +0.8]}


class LimbSubnetworkHead(nn.Module):
    """Linear head that a limb's observations+RG-input map to joint actions. `LimbSubnetwork` in
    the original repo wraps `tonic.torch.models.DeterministicPolicyHead`, which is exactly a
    `nn.Sequential(nn.Linear(in, out, bias), activation())` -- inlined here directly."""

    def __init__(
        self, in_size: int, out_size: int, activation: type[nn.Module] = nn.Tanh, bias: bool = True
    ):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size, bias=bias)
        self.activation = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(x))


def _init_pf_head(head: LimbSubnetworkHead, osize: int, zsize: int, repeat: int, flx, ext):
    """Ported from `init_pf_flxext`: zero-init then write flx/ext gains into the last
    `2*repeat` input columns (the RG rhythm-generation input, split neg/pos rectified)."""
    with torch.no_grad():
        head.linear.bias.zero_()
        head.linear.weight.zero_()
        ncols = head.linear.weight.shape[1]
        block = head.linear.weight[:, :ncols]
        block[:, -2 * repeat : -repeat] = (
            torch.tensor(flx, dtype=torch.float32).reshape(-1, 1) / repeat
        )
        block[:, -repeat:] = torch.tensor(ext, dtype=torch.float32).reshape(-1, 1) / repeat


# ==================================================================================================
# Top-level Quadruped NCAP actor

_ACTION_SIZE_PER_LIMB = 2  # [thigh, calf] joint targets, matching `spinal_bccmd.yaml` action_sizes
_OBS_SIZE_PER_LIMB = 20  # matches the `af` flx/ext observation vector length in the config
_AF_REPEAT = 8
_PF_REPEAT = 8


class QuadrupedNCAP(nn.Module):
    """Neural Circuit Architectural Prior (NCAP) quadruped locomotion controller: RG oscillator
    circuit + per-limb Afferent Feedback + Pattern Formation linear heads + a fixed Brainstem
    Command signal, faithfully ported from `nikhilxb/ncap-quadruped`'s `spinal_bccmd` config."""

    def __init__(self, timestep: float = 30.0, bc_command: float = 0.5):
        super().__init__()
        self.rg = RhythmGenerationCircuit(timestep=timestep)

        # Afferent Feedback: `mode="share-cross"` -> one shared head for {FL,HL}, one for {FR,HR}.
        # Output is 2 limbs x [flx, ext] x repeat wide; chunked in 2 below to recover per-limb blocks.
        af_out = 2 * 2 * _AF_REPEAT
        self.af_cross = LimbSubnetworkHead(
            _OBS_SIZE_PER_LIMB * 2, af_out, activation=nn.Identity, bias=False
        )
        with torch.no_grad():
            self.af_cross.linear.weight.zero_()

        # Pattern Formation: `mode="share-cross"` -> one shared head for {FL,HL}, one for {FR,HR}.
        pf_in = _OBS_SIZE_PER_LIMB * 2 + 2 * _PF_REPEAT
        pf_out = _ACTION_SIZE_PER_LIMB
        self.pf_cross = LimbSubnetworkHead(pf_in, pf_out, activation=nn.Tanh, bias=True)
        _init_pf_head(
            self.pf_cross,
            _OBS_SIZE_PER_LIMB,
            _ACTION_SIZE_PER_LIMB,
            _PF_REPEAT,
            _PF_FLX["FL"],
            _PF_EXT["FL"],
        )

        # Brainstem Command: fixed learnable scalar broadcast to freq/sync of both sides.
        self.bc_command = nn.Parameter(torch.tensor([bc_command], dtype=torch.float32))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observations: (batch, 4, _OBS_SIZE_PER_LIMB) per-limb proprioceptive observations,
                ordered [FL, FR, HL, HR].
        Returns:
            actions: (batch, 4, _ACTION_SIZE_PER_LIMB) per-limb joint targets, same limb order.
        """
        batch = observations.shape[0]
        obs = {limb: observations[:, i, :] for i, limb in enumerate(_LIMBS)}

        # Brainstem command broadcast to freq/sync for both sides (fixed signal, `repeat_agg=sum`).
        cmd_scalar = self.bc_command.sum().expand(batch, 1)
        cmd = {
            "freq_L": cmd_scalar,
            "sync_L": cmd_scalar,
            "freq_R": cmd_scalar,
            "sync_R": cmd_scalar,
        }

        # Afferent feedback per limb (share-cross: FL&HL share a head, FR&HR share a head).
        af_fl_hl_in = torch.concat((obs["FL"], obs["HL"]), dim=-1)
        af_fr_hr_in = torch.concat((obs["FR"], obs["HR"]), dim=-1)
        af_fl_hl = self.af_cross(af_fl_hl_in)
        af_fr_hr = self.af_cross(af_fr_hr_in)
        y_fl, y_hl = torch.chunk(af_fl_hl, 2, dim=-1)
        y_fr, y_hr = torch.chunk(af_fr_hr, 2, dim=-1)
        af_out = {"FL": y_fl, "FR": y_fr, "HL": y_hl, "HR": y_hr}

        def agg(y: torch.Tensor) -> torch.Tensor:
            # repeat_agg="sum": collapse the `_AF_REPEAT`-wide flx/ext blocks back to [flx, ext].
            flx, ext = y.split(_AF_REPEAT, dim=-1)
            return torch.concat(
                (flx.sum(dim=-1, keepdim=True), ext.sum(dim=-1, keepdim=True)), dim=-1
            )

        aff = {}
        for limb in _LIMBS:
            a = agg(af_out[limb])
            aff[f"aff_flx_{limb}"] = a[:, :1]
            aff[f"aff_ext_{limb}"] = a[:, 1:]

        # One Euler step of the RG circuit given current afferent+command inputs.
        state = self.rg.init_state(batch_size=batch)
        _, rg_out = self.rg.step(state, aff, cmd)

        # Pattern formation per limb (share-cross), rectified obs + repeated RG rhythm input.
        def rectify(x: torch.Tensor) -> torch.Tensor:
            return torch.concat((torch.clamp(-x, min=0), torch.clamp(x, min=0)), dim=-1)

        pf_fl_hl_in = torch.concat(
            (rectify(obs["FL"]), rg_out["FL"].repeat_interleave(_PF_REPEAT, dim=-1)), dim=-1
        )
        pf_fr_hr_in = torch.concat(
            (rectify(obs["FR"]), rg_out["FR"].repeat_interleave(_PF_REPEAT, dim=-1)), dim=-1
        )
        pf_hl_in = torch.concat(
            (rectify(obs["HL"]), rg_out["HL"].repeat_interleave(_PF_REPEAT, dim=-1)), dim=-1
        )
        pf_hr_in = torch.concat(
            (rectify(obs["HR"]), rg_out["HR"].repeat_interleave(_PF_REPEAT, dim=-1)), dim=-1
        )

        y_fl_action = self.pf_cross(pf_fl_hl_in)
        y_fr_action = self.pf_cross(pf_fr_hr_in)
        y_hl_action = self.pf_cross(pf_hl_in)
        y_hr_action = self.pf_cross(pf_hr_in)

        return torch.stack([y_fl_action, y_fr_action, y_hl_action, y_hr_action], dim=1)


# ==================================================================================================
# MENAGERIE staging entry points

MENAGERIE_ZOO = "ported-pytorch"


def build_ncap_quadruped():
    return QuadrupedNCAP()


def example_input_ncap_quadruped():
    return torch.randn(2, 4, _OBS_SIZE_PER_LIMB)


MENAGERIE_ENTRIES = [
    (
        "NCAP (Neural Circuit Architectural Prior)",
        build_ncap_quadruped,
        example_input_ncap_quadruped,
        2022,
        "ported-pytorch",
    ),
]
