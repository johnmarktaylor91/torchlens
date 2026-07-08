# SOURCE: vendored from https://github.com/dixantmittal/mctsnet @ master
# MCTSNet: "Learning to Search with MCTSnets" (Guez et al., ICML 2018,
# arXiv:1802.04697). Differentiable Monte-Carlo Tree Search whose Memory,
# Policy, Backup, and Readout sub-networks are learned end-to-end; the tree
# search itself is a real Python control-flow procedure over real learned
# tensors (RockSample POMDP simulator), not a fixed-shape single forward
# pass -- this vendored module drives the real `MCTSnet.search()` loop.
#
# Vendored real repo code, combined verbatim from:
#   modules/commons.py  (ResidualLinear, ResidualConv, hidden_size/d_memory)
#   modules/memory.py   (Memory)
#   modules/policy.py   (Policy)
#   modules/backup.py   (Backup)
#   modules/readout.py  (Readout)
#   solvers/mctsnet.py  (MCTSnet.search, minus save/load)
#   simulators/rocksample.py (RockSample: the real repo's smallest/default
#     simulator -- MCTSnet needs a concrete SIMULATOR to define n_actions and
#     tensor_shape())
#   simulators/base.py  (BaseSimulator, parent of RockSample)
#   utils/basic.py      (manhattan, to_one_hot; tensor_cache pre-fill)
#   utils/prepare_input.py (prepare_input_for_f_backup)
# Only non-architectural portability fixes applied:
#   - the original code imports `recordclass.recordclass` (a pip package not
#     in the TorchLens base env) purely as a mutable-namedtuple container for
#     grid `(x, y)` positions and tree-node bookkeeping (`Variables`,
#     `Tensors`, `Node`); replaced with a tiny local `_MutableRecord` helper
#     (stdlib-only) providing the same attribute-mutation semantics. This
#     touches zero neural-network computation -- it is bookkeeping plumbing
#     identical in effect to the original recordclass instances.
#   - `Device.get_device()` (a process-global CPU/GPU switch) inlined to
#     always target the input tensor's device; no behavior change on CPU.
#   - `environment.SIMULATOR` (import-time global set to `RockSample` in the
#     real repo) kept as a plain module-level binding.
# No layer, head, or search/backup/policy/readout dataflow was changed from
# the real implementation.

from copy import deepcopy

import torch as t
import torch.nn as nn
from torch.distributions import Categorical

MENAGERIE_ZOO = "vendored-pytorch"


# ---- recordclass substitute (bookkeeping only, no architecture) ----
class _MutableRecord:
    """Minimal stand-in for `recordclass.recordclass`: a mutable, attribute-
    addressable record. Supports the same iter/index access the real repo
    code uses on Vector/Variables/Tensors instances."""

    __slots__ = ()

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    def __iter__(self):
        return iter(getattr(self, s) for s in self.__slots__)

    def __getitem__(self, i):
        return getattr(self, self.__slots__[i])


def _mutable_record(name, fields):
    slots = tuple(fields.split())

    def __init__(self, *args, **kwargs):
        for s, v in zip(slots, args):
            object.__setattr__(self, s, v)
        for k, v in kwargs.items():
            object.__setattr__(self, k, v)

    ns = {"__slots__": slots, "__init__": __init__}
    return type(name, (_MutableRecord,), ns)


# ---- simulators/rocksample.py + simulators/base.py (verbatim logic) ----
Vector = _mutable_record("Vector", "x y")


class State:
    __slots__ = ("rover", "rocks", "quality")

    def __init__(self, rover, rocks, quality):
        self.rover = rover
        self.rocks = rocks
        self.quality = quality

    def __iter__(self):
        return iter((self.rover, self.rocks, self.quality))


def manhattan(a, b):
    return abs(a.x - b.x) + abs(a.y - b.y)


class RockSample:
    MAP_SIZE = 7
    ROCKS = [Vector(1, 3), Vector(2, 2), Vector(3, 4), Vector(5, 5)]
    NUM_OF_ROCKS = len(ROCKS)

    n_actions = 5

    ACTIONS_UP = 0
    ACTIONS_DOWN = 1
    ACTIONS_LEFT = 2
    ACTIONS_RIGHT = 3
    ACTIONS_SAMPLE = 4

    ILLEGAL_ACTION_REWARD = -100
    BAD_ROCK_REWARD = -10
    GOOD_ROCK_REWARD = 10
    EXIT_REWARD = 10
    MOVE_REWARD = 0

    @staticmethod
    def reset():
        import numpy as np

        rocks = deepcopy(RockSample.ROCKS)
        rover = Vector(0, RockSample.MAP_SIZE // 2)
        quality = np.random.RandomState(0).binomial(1, 0.5, RockSample.NUM_OF_ROCKS).tolist()
        return State(rover, rocks, quality)

    @staticmethod
    def simulate(state, action):
        rover = Vector(state.rover.x, state.rover.y)
        rocks = deepcopy(state.rocks)
        quality = deepcopy(state.quality)

        reward = RockSample.MOVE_REWARD
        terminal = False

        if action == RockSample.ACTIONS_UP:
            if rover.y == 0:
                reward = RockSample.ILLEGAL_ACTION_REWARD
            rover.y = max(rover.y - 1, 0)

        elif action == RockSample.ACTIONS_DOWN:
            if rover.y == RockSample.MAP_SIZE - 1:
                reward = RockSample.ILLEGAL_ACTION_REWARD
            rover.y = min(rover.y + 1, RockSample.MAP_SIZE - 1)

        elif action == RockSample.ACTIONS_LEFT:
            if rover.x == 0:
                reward = RockSample.ILLEGAL_ACTION_REWARD
            rover.x = max(rover.x - 1, 0)

        elif action == RockSample.ACTIONS_RIGHT:
            if rover.x >= RockSample.MAP_SIZE - 1:
                reward = RockSample.EXIT_REWARD
                terminal = True
            rover.x = min(rover.x + 1, RockSample.MAP_SIZE)

        elif action == RockSample.ACTIONS_SAMPLE:
            if rover in rocks:
                idx = rocks.index(rover)
                observation = quality[idx]
                reward = RockSample.GOOD_ROCK_REWARD * observation + RockSample.BAD_ROCK_REWARD * (
                    1 - observation
                )
                quality[idx] = 0
            else:
                reward = RockSample.ILLEGAL_ACTION_REWARD

        return State(rover, rocks, quality), reward, terminal

    @staticmethod
    def tensor_shape():
        return 3, RockSample.MAP_SIZE, RockSample.MAP_SIZE + 1

    @staticmethod
    def state_to_tensor(state):
        rover, rocks, qualities = state

        tensor = t.zeros((3, RockSample.MAP_SIZE, RockSample.MAP_SIZE + 1))

        tensor[0, rover.y, rover.x] = 1
        for rock, quality in zip(rocks, qualities):
            tensor[1, rock.y, rock.x] = 1
            tensor[2, rock.y, rock.x] = 1 if quality == 1 else -1

        return t.constant_pad_nd(tensor, (1, 1, 1, 1), value=-1)


SIMULATOR = RockSample

# ---- utils/basic.py (verbatim; pre-filled one-hot cache) ----
_tensor_cache = {}
for _n_classes in range(1, 20):
    for _i in range(_n_classes):
        _onehot = t.zeros(_n_classes)
        _onehot.scatter_(0, t.tensor(_i), 1)
        _tensor_cache["{}_{}".format(_i, _n_classes)] = _onehot


def to_one_hot(index, n_classes):
    return _tensor_cache["{}_{}".format(index, n_classes)]


# ---- modules/commons.py (verbatim) ----
hidden_size = 32
d_belief = 32
d_memory = 32


class ResidualLinear(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.linear1 = nn.Linear(size, size)
        self.linear2 = nn.Linear(size, size)

    def forward(self, x):
        y = t.relu(self.linear1(x))
        y = t.relu(self.linear2(y))

        return x + y


class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        y = t.relu(self.conv1(x))
        y = t.relu(self.conv2(y))

        return x + y


# ---- modules/memory.py (verbatim) ----
class Memory(nn.Module):
    def __init__(self):
        super().__init__()
        channels, _, _ = SIMULATOR.tensor_shape()
        self.memory = nn.Sequential(
            nn.Conv2d(channels, 16, 1, 1),
            ResidualConv(16),
            nn.Conv2d(16, 32, 1, 1),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, d_memory),
        )

    def forward(self, x):
        return self.memory(x.unsqueeze(0)).squeeze()


# ---- modules/policy.py (verbatim) ----
class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        self.memory = nn.Linear(d_memory, hidden_size)
        self.children_memory = nn.Linear(d_memory * SIMULATOR.n_actions, hidden_size)

        self.policy = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            ResidualLinear(hidden_size),
            nn.Linear(hidden_size, SIMULATOR.n_actions),
        )

    def forward(self, node):
        memory = t.relu(self.memory(node.tensors.memory))
        children = t.relu(self.children_memory(t.cat(node.tensors.children)))

        x = t.cat([memory, children], dim=0)

        return self.policy(x)


# ---- modules/backup.py (verbatim) ----
class Backup(nn.Module):
    def __init__(self):
        super().__init__()

        self.memory = nn.Linear(d_memory, hidden_size)
        self.child_memory = nn.Linear(d_memory, hidden_size)
        self.action = nn.Linear(SIMULATOR.n_actions, hidden_size)
        self.reward = nn.Linear(1, hidden_size)

        d_input = 4 * hidden_size

        self.forget = nn.Sequential(
            nn.Linear(d_input, hidden_size),
            nn.ReLU(),
            ResidualLinear(hidden_size),
            nn.Linear(hidden_size, d_memory),
        )

        self.update = nn.Sequential(
            nn.Linear(d_input, hidden_size),
            nn.ReLU(),
            ResidualLinear(hidden_size),
            nn.Linear(hidden_size, d_memory),
        )

        self.info = nn.Sequential(
            nn.Linear(d_input, hidden_size),
            nn.ReLU(),
            ResidualLinear(hidden_size),
            nn.Linear(hidden_size, d_memory),
        )

    def forward(self, memory, child_memory, action, reward):
        e_memory = t.relu(self.memory(memory))
        e_child_memory = t.relu(self.child_memory(child_memory))
        e_action = t.relu(self.action(action))
        e_reward = t.relu(self.reward(reward))

        phi = t.cat((e_memory, e_child_memory, e_action, e_reward), dim=0)

        forget = t.sigmoid(self.forget(phi))
        update = t.tanh(self.update(phi))
        info = t.relu(self.info(phi))

        return memory * forget + update * info


# ---- modules/readout.py (verbatim) ----
class Readout(nn.Module):
    def __init__(self):
        super().__init__()
        self.readout = nn.Sequential(
            nn.Linear(d_memory, hidden_size),
            nn.ReLU(),
            ResidualLinear(hidden_size),
            nn.Linear(hidden_size, SIMULATOR.n_actions),
        )

    def forward(self, x):
        return self.readout(x)


# ---- utils/prepare_input.py (verbatim, Device inlined to tensor device) ----
def prepare_input_for_f_backup(node, action, reward):
    memory = node.tensors.memory
    child_memory = node.variables.children[action].tensors.memory

    action = to_one_hot(action, SIMULATOR.n_actions).to(memory.device)
    reward = t.tensor([reward]).float().to(memory.device)

    return memory, child_memory, action, reward


Path = _mutable_record("Path", "node action reward")
Variables = _mutable_record("Variables", "state children")
Tensors = _mutable_record("Tensors", "memory children")
Node = _mutable_record("Node", "variables tensors")


# ---- solvers/mctsnet.py: MCTSnet (verbatim search logic, minus save/load) ----
class MCTSnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.f_memory = Memory()
        self.f_policy = Policy()
        self.f_backup = Backup()
        self.f_readout = Readout()

        self.tensor_cache = {}

    def state_to_tensor(self, state):
        key = str(state)
        tensor = self.tensor_cache.get(key)
        if tensor is None:
            tensor = SIMULATOR.state_to_tensor(state)
            self.tensor_cache[key] = tensor

        return tensor

    def new_node(self, state):
        variables = Variables(state=state, children={})

        tensor_memory = self.f_memory(self.state_to_tensor(state))

        tensors = Tensors(
            memory=tensor_memory,
            children=[t.zeros_like(tensor_memory) for i in range(SIMULATOR.n_actions)],
        )

        return Node(variables=variables, tensors=tensors)

    def search(self, state, n_simulations, training):
        root = self.new_node(state)

        predictions = [self.f_readout(root.tensors.memory)]
        logits = []
        actions = []

        for i in range(n_simulations):
            node = root

            path = []
            logits_m = []
            actions_m = []

            terminal = False
            while not terminal:
                p_actions = self.f_policy(node)
                action = Categorical(logits=p_actions).sample().item()

                if training:
                    logits_m.append(p_actions)
                    actions_m.append(action)

                next_state, reward, terminal = SIMULATOR.simulate(node.variables.state, action)
                path.append(Path(node, action, reward))

                if node.variables.children.get(action) is None:
                    node.variables.children[action] = self.new_node(next_state)
                    break
                else:
                    node = node.variables.children[action]

            for node, action, reward in reversed(path):
                node.tensors.memory = self.f_backup(
                    *prepare_input_for_f_backup(node, action, reward)
                )
                node.tensors.children[action] = node.variables.children[action].tensors.memory

            predictions.append(self.f_readout(root.tensors.memory))

            logits.append(logits_m)
            actions.append(actions_m)

        return Categorical(logits=predictions[-1]).sample().item(), (predictions, logits, actions)

    def forward(self, state, n_simulations=3, training=True):
        action, aux = self.search(state, n_simulations, training)
        predictions, logits, actions = aux
        return t.stack(predictions), action


def build_mctsnet():
    return MCTSnet()


def example_input_mctsnet():
    state = SIMULATOR.reset()
    return (state,)


MENAGERIE_ENTRIES = [
    (
        "MCTSNet",
        build_mctsnet,
        example_input_mctsnet,
        2018,
        MENAGERIE_ZOO,
    ),
]
