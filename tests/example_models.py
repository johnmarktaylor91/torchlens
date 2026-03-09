import numpy as np
import torch
from torch import nn


# TODO:
#  1) add tests for
#       a) different uber models that combines everything (branching, looping, conditional, etc.),
#       b) model architectures like resnets, googlenet, transformers, GNNs, GANs, etc.
#  2) Make things nicely organized, ordered, named.


#  ***********************
#  **** Simple Models ****
#  ***********************
class SimpleFF(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + 1
        x = x * 2
        return x


#  ********************************
#  **** Special Case Functions ****
#  ********************************


class InPlaceFuncs(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_module_inplace = nn.ReLU(inplace=True)
        self.relu_module_newtensor = nn.ReLU(inplace=False)

    def forward(self, x):
        x1 = x + 1
        x2 = self.relu_module_inplace(x1)
        y1 = x1 * 10
        y2 = x2 + 3
        x3 = torch.log(x2)
        x4 = self.relu_module_newtensor(x3)
        _x5 = x4 * 2 + y1 + y2
        return x


class GeluModel(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + 1
        y = x[0]
        y = torch.nn.functional.gelu(y)
        y = y + 2
        return y


class SimpleInternallyGenerated(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + torch.ones(x.shape)
        x = x * 2
        return x


class NewTensorInside(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x * 2
        y = torch.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        y = y * 3
        x = x + torch.mean(y)
        x = torch.log(x)
        return x


class TensorFromNumpy(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x * 2
        y = torch.from_numpy(np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])).type(x.dtype)
        y = y * 3
        x = x + torch.mean(y)
        x = torch.log(x)
        return x


class SimpleRandom(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + 2
        x = torch.log(x) + torch.rand(x.shape)
        x = x + 3
        return x


class DropoutModelReal(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x * 2
        x = self.relu(x)
        x = self.dropout(x)
        x = x * 2
        return x


class DropoutModelDummyZero(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(p=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x * 2
        x = self.relu(x)
        x = self.dropout(x)
        x = x * 2
        return x


class BatchNormModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x * 2
        x = self.relu(x)
        x = self.batchnorm(x)
        x = x * 2
        return x


class ConcatTensors(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = torch.log(x)
        y = torch.sin(x)
        z = torch.tan(x)
        x = torch.cat([x, y, z], dim=0)
        return x


class SplitTensor(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x * 2
        a, b, c, d = torch.split(x, 56, dim=2)
        x = (a + b) * (c + d)
        return x


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x


class IdentityModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = Identity()

    def forward(self, x):
        x = x + 1
        x = x * 2
        x = self.identity(x)
        x = x * 2
        return x


class AssignTensor(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = torch.log(x)
        x[1, 1] = 5
        x[2] = 1
        return x


class GetAndSetItem(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = torch.log(x)
        x[2, 2, 0, 1] = x[3, 2, 1, 0]
        x = x * 2
        return x


class GetItemTracking(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = torch.log(x)
        x = x * 0
        y1 = x[0]
        y1[:] = 1
        y1 = y1.mean()
        y2 = x[1]
        y2[:] = 1
        y2 = y2.mean()
        y3 = x[3]
        y3[:] = 1
        y3 = y3.mean()
        xmean = x.mean()
        z = y1 + y2 + y3 + xmean
        return z


class InPlaceZeroTensor(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = torch.log(x)
        y = x[1:3, 2:4]
        y.zero_()
        return x


class SliceOperations(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = torch.log(x)
        y = x[1:3, 2:4]
        y.zero_()
        y = y + 1
        y = torch.log(y)
        x = x**2
        return x


class DummyOperations(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = torch.log(x)
        y = torch.rand(x.shape)
        y = y + 1
        y = y * 0
        x = x * y
        x = x + 0
        x = torch.sin(x)
        x = x * 1
        return x


class SameTensorArg(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * 2
        x = x + x
        x = torch.sin(x)
        return x


#  ************************************
#  **** Special Case Architectures ****
#  ************************************


class MultiInputs(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x, y, z):
        a = x + y
        b = torch.log(z)
        x = a**b
        return x


class ListInput(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_list):
        x, y, z = input_list
        a = x + y
        b = torch.log(z)
        x = a**b
        return x


class DictInput(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_dict):
        x, y, z = input_dict["x"], input_dict["y"], input_dict["z"]
        a = x + y
        b = torch.log(z)
        x = a**b
        return x


class NestedInput(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_dict):
        list1, list2 = input_dict["list1"], input_dict["list2"]
        a, b = list1
        c, d = list2
        t1 = a * d
        t2 = c * d + b
        t = t1 + t2
        return t


class MultiOutputs(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x * 2
        a = torch.log(x)
        b = torch.sin(x)
        c = torch.tan(x)
        b = b * 3
        return a, b, c


class ListOutput(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x * 2
        a = torch.log(x)
        b = torch.sin(x)
        c = torch.tan(x)
        b = b * 3
        return [a, b, c]


class DictOutput(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x * 2
        a = torch.log(x)
        b = torch.sin(x)
        c = torch.tan(x)
        b = b * 3
        return {"a": a, "b": b, "c": c}


class NestedOutput(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x * 2
        a = torch.log(x)
        b = torch.sin(x)
        c = torch.tan(x)
        b = b * 3
        return {"a": [a, b], "b": c, "c": [a, b, c], "d": [[a, b]]}


class BufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buffer1", torch.ones(12, 12))
        self.register_buffer("buffer2", torch.rand(12, 12))

    def forward(self, x):
        x = x + self.buffer1
        x = x ** (self.buffer2 * (3 + torch.rand(12, 12)))
        x = x * 2
        return x


class BufferRewriteModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("buffer1", torch.rand(12, 12))
        self.register_buffer("buffer2", torch.rand(12, 12))

    def forward(self, x):
        x = torch.sin(x)
        x = x + self.buffer1
        x = x * self.buffer2
        self.buffer1 = torch.rand(12, 12)
        self.buffer2 = x**2
        x = self.buffer1 + self.buffer2
        return x


class BufferRewriteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.buffer_mod = BufferRewriteModule()

    def forward(self, x):
        x = torch.cos(x)
        x = self.buffer_mod(x)
        x = x * 4
        x = self.buffer_mod(x)
        x = self.buffer_mod(x)
        x = x + 1
        x = self.buffer_mod(x)
        x = self.buffer_mod(x)
        x = self.buffer_mod(x)
        x = x * 2
        return x


class SimpleBranching(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + 1
        y = x * 2
        y = y + 3
        y = torch.log(y)
        z = x**2
        z = torch.sin(z)
        x = x + y + z
        return x


class ConditionalBranching(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        if torch.mean(x) > 0:
            x = torch.sin(x)
            x = x + 2
            x = x * 3
        else:
            x = torch.cos(x)
            x = x + 1
            x = x * 4
            x = x**2
        return x


class ConditionalAlwaysTrue(nn.Module):
    """Condition always True — only THEN branch executes."""

    @staticmethod
    def forward(x):
        if torch.mean(torch.abs(x)) >= 0:  # always true (abs >= 0)
            x = torch.sin(x)
            x = x + 1
        else:
            x = torch.cos(x)
        return x


class ConditionalAlwaysFalse(nn.Module):
    """Condition always False — only ELSE branch executes."""

    @staticmethod
    def forward(x):
        if torch.mean(torch.abs(x)) < 0:  # always false (abs >= 0)
            x = torch.sin(x)
        else:
            x = torch.cos(x)
            x = x + 1
        return x


class ConditionalNested(nn.Module):
    """Nested if-then (condition inside condition)."""

    @staticmethod
    def forward(x):
        if torch.mean(x) > -1000:
            x = x + 1
            if torch.sum(x) > -1000:
                x = x * 2
            else:
                x = x * 3
        else:
            x = x - 1
        return x


class ConditionalChainedBools(nn.Module):
    """Two boolean conditions checked before branching."""

    @staticmethod
    def forward(x):
        cond1 = torch.mean(x) > 0
        cond2 = torch.sum(x) > 0
        if cond1 and cond2:
            x = torch.sin(x)
        else:
            x = torch.cos(x)
        return x


class ConditionalNoBranch(nn.Module):
    """Bool computed but never used for branching (no THEN -> should clear IF)."""

    @staticmethod
    def forward(x):
        _ = torch.mean(x) > 0  # computed but not used for control flow
        x = torch.sin(x) + 1
        return x


class ConditionalMultipleBranches(nn.Module):
    """Two separate if-then blocks in sequence."""

    @staticmethod
    def forward(x):
        if torch.mean(x) > 0:
            x = torch.sin(x)
        else:
            x = torch.cos(x)
        if torch.sum(x) > 0:
            x = x + 1
        else:
            x = x - 1
        return x


class ConditionalWithModules(nn.Module):
    """Branches using nn.Linear layers."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 5, bias=False)
        self.linear2 = nn.Linear(5, 5, bias=False)

    def forward(self, x):
        if torch.mean(x) > 0:
            x = self.linear1(x)
        else:
            x = self.linear2(x)
        return x


class ConditionalIdentity(nn.Module):
    """Condition but both branches do same thing (still valid IF/THEN)."""

    @staticmethod
    def forward(x):
        if torch.mean(x) > 0:
            x = x + 1
        else:
            x = x + 1
        return x


class RepeatedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + 1
        x = self.relu(x)
        y = x + 2
        z = x * 3
        y = self.relu(y)
        x = self.relu(y + z)
        return x


class Level01(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = torch.tan(x)
        x = torch.sin(x)
        return x


class Level11(nn.Module):
    def __init__(self):
        super().__init__()
        self.level01 = Level01()

    def forward(self, x):
        x = x + 1
        x = x * 2
        x = self.level01(x)
        return x


class Level12(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x**3 + torch.ones(x.shape)
        x = x / 5
        return x


class Level21(nn.Module):
    def __init__(self):
        super().__init__()
        self.level11 = Level11()
        self.level12 = Level12()

    def forward(self, x):
        x = x + 1
        x = self.level11(x) + torch.rand(x.shape)
        x = x + 9
        x = self.level12(x)
        x = x / 5
        return x


class Level22(nn.Module):
    def __init__(self):
        super().__init__()
        self.level1_1 = Level11()
        self.level1_2 = Level12()

    def forward(self, x):
        x = x + 9
        x = self.level1_2(x)
        x = x - 5
        x = self.level1_2(self.level1_2(x))
        x = x / 5
        return x


class NestedModules(nn.Module):
    def __init__(self):
        super().__init__()
        self.level21 = Level21()
        self.level22 = Level22()

    def forward(self, x):
        x = torch.cos(x)
        x1 = self.level21(x)
        x2 = self.level21(x)
        x3 = self.level21(x) + self.level21.level12(x)
        x = x1 * x2 + x3
        x = self.level22(x)
        return x


class OrphanTensors(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + 1
        x = x * 2
        z = torch.ones(5, 5)
        z = z + 1
        a = z * 2
        b = z**2
        _c = a + b
        x = x**2
        return x


class SimpleLoopNoParam(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + 2
        for _ in range(3):
            x = torch.log(x)
            x = torch.sin(x)
        x = x + 3
        return x


class SameOpRepeat(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        for _ in range(8):
            x += 1
        x = torch.log(x)
        x = torch.flatten(x)
        for _ in range(8):
            x = self.fc(x)
        return x


class RepeatedOpTypeInLoop(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        for _ in range(8):
            x = x + 1
            x = torch.sin(x)
            x = torch.cos(x)
            x = x * 2
            x = torch.sin(x)
            x = torch.exp(x)
        x = torch.flatten(x)
        return x


class VaryingLoopNoParam1(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + 2
        for i in range(8):
            x = torch.log(x)
            x = x + torch.ones(x.shape)
            x = torch.sin(x)
            if i % 2 == 0:
                y = x + 3
                y = torch.sin(y)
                y = y**2
            x = torch.sin(x)
            if i % 2 == 1:
                _z = x + 3
                _z = torch.cos(x)
        x = x + 3
        return x


class VaryingLoopNoParam2(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + 2
        for i in range(8):
            x = torch.log(x)
            x = x + torch.ones(x.shape)
            x = torch.sin(x)
            if i in [0, 3, 4]:
                y = x + 3
                y = torch.sin(y)
                y = y**2
            else:
                y = x * torch.rand(x.shape)
                y = torch.cos(y)
            x = x + y
            x = torch.log(x)
        x = x + 3
        return x


class VaryingLoopWithParam(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        x = self.fc(x)
        x = x + 1
        x = x * 2
        x = self.fc(x)
        x = x + 1
        x = x * 2
        x = self.fc(x)
        x = x + 1
        x = x * 2
        x = x**3
        x = self.fc(x)
        return x


class LoopingInternalFuncs(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + 3
        for _ in range(3):
            x = x * 2
            y = torch.rand(x.shape)
            z = torch.ones(x.shape)
            y = y + 1
            y = y + z
            x = x / y
            x = torch.sin(x)
        return x


class LoopingFromInputs1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 3)

    @staticmethod
    def forward(x, y, z):
        start = torch.ones(x.shape)
        x = torch.sin(start) + x
        x = torch.log(x)
        x = torch.sin(x)
        x = x + y
        x = torch.log(x)
        x = torch.sin(x)
        x = x + z
        x = torch.log(x)
        x = torch.sin(x)
        return x


class LoopingFromInputs2(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        buffer = torch.zeros(x[0].shape)
        for i in range(len(x)):
            buffer = buffer + x[i]
            buffer = buffer + 1
            buffer = buffer * 2
            buffer = torch.sin(buffer)
        return buffer


class LoopingInputsAndOutputs(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        buffer = torch.zeros(x[0].shape)
        outputs = []
        for i in range(len(x)):
            buffer = buffer + x[i]
            buffer = buffer + 1
            buffer = buffer * 2
            buffer = torch.sin(buffer)
            outputs.append(buffer)
        return outputs


class StochasticLoop(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        while True:
            x = x + torch.rand(x.shape).abs()
            if torch.mean(x) > 100:
                break
        x = x * 2
        return x


class RecurrentParamsSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        x = x + 1
        x = self.fc1(x)
        x = x * 2
        x = self.fc1(x)
        x = torch.log(x)
        x = torch.tan(x)
        x = self.fc1(x)
        x = x * 2
        x = self.fc1(x)
        x = torch.log(x)
        return x


class RecurrentParamsComplex(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        x = x + 1
        x = self.fc1(x)
        x = x * 2
        x = self.fc2(x)
        x = torch.log(x)
        x = torch.sin(x)
        x = self.fc2(x)
        x = torch.log(x)
        x = self.fc1(x)
        x = x + 1
        x = x * 2
        x = x + 1
        x = x * 2
        x = self.fc2(x)
        x = torch.log(x)
        x = self.fc1(x)
        x = x + 2
        x = self.fc2(x)
        x = torch.log(x)
        x = self.fc2(x)
        x = torch.log(x)
        x = self.fc1(x)
        x = x + 1
        x = x * 2
        x = self.fc2(x)
        y = torch.log(x)
        y = torch.sin(y)
        y = self.fc1(y)
        y = y + 3
        z = torch.tan(x)
        z = torch.log(z)
        z = self.fc2(x) + self.fc2(z)
        a = y * z
        return a


class LoopingParamsDoubleNested(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        x = self.fc1(x)
        x = x + 1
        x = self.fc2(x)
        x = x * 2
        x = self.fc2(x)
        x = x * 2
        x = self.fc1(x)
        x = x + 1
        x = self.fc2(x)
        x = x * 2
        x = self.fc2(x)
        x = x * 2
        x = self.fc1(x)
        x = x + 1
        x = self.fc2(x)
        x = x * 2
        x = self.fc2(x)
        x = x * 2
        x = self.fc1(x)
        return x


class ModuleLoopingClash1(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + 2
        x = torch.sin(x)
        x = self.relu(x)
        x = torch.log(x)
        for _ in range(
            4
        ):  # this tests clashes between what counts as "same"--module-based or looping-based
            x = self.relu(x)
            x = x + 1
        x = torch.cos(x)
        x = self.relu(x)
        return x


class ModuleLoopingClash2(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + 2
        x = torch.sin(x)
        x = self.relu(x)
        x = torch.log(x)
        for i in range(6):
            if i % 2 == 0:
                x = self.relu(x)  # these should be counted as different
            else:
                x = nn.functional.relu(x)
            x = x + 1
        x = torch.cos(x)
        x = self.relu(x)
        return x


class ModuleLoopingClash3(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + 2
        x = torch.sin(x)
        x = self.relu(x)
        x = torch.log(x)
        for i in range(6):
            if i % 2 == 0:
                x += nn.functional.relu(x) + self.relu(torch.rand(x.shape))
            x = x + 1
        x = torch.cos(x)
        x = self.relu(x)
        return x


#  ****************************
#  **** RNN/LSTM Architectures ****
#  ****************************
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 10)
        self.label = nn.Linear(10, 5)
        self.hidden_size = 10

    def forward(self, x):
        batch_size = x.shape[0]
        h_0 = torch.zeros(1, batch_size, self.hidden_size)
        c_0 = torch.zeros(1, batch_size, self.hidden_size)

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))

        return self.label(final_hidden_state[-1])


class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.RNN(5, 10)
        self.label = nn.Linear(10, 5)
        self.hidden_size = 10

    def forward(self, x):
        batch_size = x.shape[0]
        h_0 = torch.zeros(1, batch_size, self.hidden_size)

        output, final_hidden_state = self.rnn(x, h_0)

        return self.label(final_hidden_state[-1])


class UberModel1(nn.Module):
    def __init__(self):
        """
        Network for testing complex topology (e.g., multiple inputs, outputs, terminal nodes, buffer nodes, etc.)
        """
        super().__init__()
        self.buffer = torch.ones(5, 5)

    @staticmethod
    def forward(x):
        x, y, z = x
        x = x + 1
        y = y * 2
        y = y**3
        w = torch.rand(5, 5)
        w = w * 2
        w = w + 4
        wa = torch.cos(w)
        wa = torch.sin(wa)
        wb = torch.log(w)
        wb = torch.tan(wb)
        wb = wb + 1
        w = wa * wb
        w2 = w - 3
        w2 = w2 + 4
        w2 = w2 * 5
        w3 = torch.zeros(5, 5)
        w4 = torch.ones(5, 5)
        w3 = w3 + 1
        w3 = w3 * w4
        w2 = w2 + torch.zeros(5, 5)
        u = x + y + w + z
        v = u + 4
        v = u * 8
        v2 = v + 2
        v2 = v * 3
        _v3 = v2.sum() > 5
        m = torch.ones(5)
        _m1 = m * 2
        _m2 = m + 3
        v = torch.cos(v)
        return u, v, y


class UberModel2(nn.Module):
    def __init__(self):
        """Conv, relu, pool, fc, output."""
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.fc = nn.Linear(18, 3)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 1)
        x = x + torch.ones(1, 1, 3, 3) + torch.rand(1, 1, 3, 3)
        y = torch.ones(1, 2, 3)
        y = y + 1
        x = self.identity(x)
        x = x.flatten()
        x = self.fc(x)
        x = self.identity(x)
        return x


class UberModel3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        x = self.fc(x)
        x = x + 1
        x = x * 2
        x = self.fc(x)
        x = x + 1
        x = x * 2
        x = self.fc(x)
        x = x + 1
        x = x * 2
        x = x**3
        x = self.fc(x)
        x = x + 2
        x = x * 3
        x = self.fc(x)
        x = x + 1
        x = x * 2
        x = self.fc(x)
        x = x * 2
        x = x + 3
        x = self.fc(x)
        return x


class UberModel4(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)

    def forward(self, x):
        x = x + 1
        x = x * 2
        x = x - 1
        x = x + 1
        x = x * 2
        x = x - 1
        x = x + 1
        x = x * 2
        x = x - 1
        for i in range(3):
            x = torch.cos(x)
            x = self.fc1(x)
            x = torch.sin(x)
            for j in range(2):
                x = x - 4
                x = x * 2
                x = self.fc2(x)
                x = x - 4
            x = torch.log(x)
            x = self.fc1(x)
        x = x * 6
        x = torch.tanh(x)
        return x


class UberModel5(nn.Module):
    def __init__(self):
        """Conv, relu, pool, fc, output."""
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.fc = nn.Linear(9, 3)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 1)
        x = self.identity(x)
        x = x.flatten()
        x = self.fc(x)
        return x


class UberModel6(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x = x + 1
        x = x * 2
        x = x + 1
        x = x * 2
        x = x + 1
        x = x * 2
        x = x + 3
        x = x * 9
        x = x + 1
        x = x * 2
        x = x + 1
        x = x * 2
        x = torch.cos(x)
        x = torch.sin(x)
        x = torch.tan(x)
        x = torch.cos(x)
        x = torch.sin(x)
        x = torch.tan(x)
        x = torch.cos(x)
        x = torch.sin(x)
        x = x + 1
        x = x * 2
        return x


class UberModel7(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        x = x + 1
        x = torch.sin(x)
        x = x * 2
        x = self.fc(x)
        x = torch.cos(x)
        x = x - 5
        x = x * 2
        x = self.fc(x)
        x = torch.cos(x)
        x = x - 3
        x = x * 2
        x = self.fc(x)
        x = torch.cos(x)
        return x


class UberModel8(nn.Module):
    def __init__(self):
        """Conv, relu, pool, fc, output."""
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.fc = nn.Linear(18, 3)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 1)
        y = torch.ones(1, 1, 3, 3) + torch.zeros(1, 1, 3, 3)
        z = torch.rand(1, 1, 3, 3) ** y
        x = x * z
        x = self.identity(x)
        x = x.flatten()
        x = self.fc(x)
        return x


class UberModel9(nn.Module):
    def __init__(self):
        """Conv, relu, pool, fc, output."""
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.fc = nn.Linear(9, 3)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x2 = x * 2
        _x3 = x2 + 4
        y1 = x + 1
        y2 = y1 * 2
        y31 = y2 + 1
        y32 = y2 * 3
        y321 = y32 + 4
        y322 = torch.sin(y321)
        y323 = torch.cos(y322)
        y324 = torch.tan(y323)
        y4 = y324 + y31
        z1 = x * 2
        z2 = z1 * 4
        a = torch.ones(3, 3)
        b = torch.zeros(3, 3)
        z3 = z2 + a + b
        w1 = x**3
        x = torch.sum(torch.stack([y4, z3, w1]))
        return x


class NestedParamFreeLoops(nn.Module):
    """Nested loops where the inner loop has operations that are equivalent across
    all levels AND operations that differ per level due to level-dependent constants.

    Structure:
    - Outer loop: 4 iterations with state feedback (coords updated each iteration)
    - Inner loop: 3 levels, each doing:
        - A normalization step that divides by a LEVEL-DEPENDENT constant
          (this creates different operation_equivalence_types per level)
        - A core operation (sin) that is THE SAME equivalence type across all levels
        - An accumulation step

    The key challenge for the loop finder: the 12 sin operations (4 outer x 3 inner)
    all have the same equivalence type, but they are surrounded by operations that
    have DIFFERENT equivalence types per inner level (because the divisor constant
    changes). This means the BFS subgraph expansion cannot match operations across
    inner levels, fragmenting the sin ops into groups of 4 (one per inner level
    position) instead of one group of 12.

    This replicates the topology of RAFT Large's correlation pyramid, where
    grid_sample normalizes by spatial dimensions that change at each pyramid level.
    """

    @staticmethod
    def forward(x):
        coords = torch.zeros_like(x)
        divisors = [2.0, 4.0, 8.0]  # different constant per inner level

        for _ in range(4):
            # State-dependent value that changes each outer iteration
            offset = coords - x

            # Inner loop: 3 levels with level-dependent normalization
            accum = torch.zeros_like(x)
            scaled = offset
            for divisor in divisors:
                # This division uses a DIFFERENT non-tensor arg per level,
                # giving each level a different operation_equivalence_type
                normalized = scaled / divisor

                # This sin has the SAME equivalence type across all levels
                # (same func, no non-tensor args, same shape/dtype)
                result = torch.sin(normalized)

                accum = accum + result
                scaled = scaled * 0.5

            # Post-inner-loop processing
            coords = coords + torch.tanh(accum)

        return coords


class PropertyModel(nn.Module):
    def __init__(self):
        """Conv, relu, pool, fc, output."""
        super().__init__()

    def forward(self, x):
        r = x.real
        i = x.imag
        t = torch.rand(4, 4)
        t = t * 3
        t = t.data
        t2 = t.T
        m = torch.rand(4, 4, 4)
        m = m**2
        m2 = m.mT.mean()
        out = r * i / m2 + t2.mean()
        return out


#  ****************************************************
#  **** Edge-Case Loop Detection Test Models ****
#  ****************************************************


class ParallelLoops(nn.Module):
    """Two independent loops on separate branches, same op types."""

    @staticmethod
    def forward(x):
        a = x + 1
        b = x + 2
        for _ in range(3):
            a = torch.sin(a)
        for _ in range(3):
            b = torch.sin(b)
        return a + b


class SharedParamLoopExternal(nn.Module):
    """Same Linear used both inside and outside a loop."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        x = self.fc(x)
        for _ in range(3):
            x = self.fc(x)
        x = self.fc(x)
        return x


class InterleavedSharedParamLoops(nn.Module):
    """Same Linear used in two distinct loops with different surrounding ops."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        for _ in range(3):
            x = self.fc(x)
            x = torch.relu(x)
        for _ in range(3):
            x = self.fc(x)
            x = torch.sigmoid(x)
        return x


class NestedLoopsIndependentParams(nn.Module):
    """Outer loop uses fc_outer, inner loop uses fc_inner. Different params per level."""

    def __init__(self):
        super().__init__()
        self.fc_outer = nn.Linear(5, 5)
        self.fc_inner = nn.Linear(5, 5)

    def forward(self, x):
        for _ in range(3):
            x = self.fc_outer(x)
            for _ in range(2):
                x = self.fc_inner(x)
        return x


class SelfFeedingNoParam(nn.Module):
    """Output of each iteration feeds directly as input to next, no params."""

    @staticmethod
    def forward(x):
        for _ in range(4):
            x = torch.sin(x)
            x = torch.cos(x)
        return x


class DiamondLoop(nn.Module):
    """Loop body has a diamond: split into two paths, then merge."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        for _ in range(3):
            x = self.fc(x)
            a = torch.sin(x)
            b = torch.cos(x)
            x = a + b
        return x


class AccumulatorLoop(nn.Module):
    """Loop appends to output list each iteration."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        outputs = []
        for _ in range(3):
            x = self.fc(x)
            x = torch.relu(x)
            outputs.append(x)
        return torch.stack(outputs)


class SingleIterationLoop(nn.Module):
    """Loop that runs exactly once — should NOT be detected as multi-pass."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        for _ in range(1):
            x = self.fc(x)
            x = torch.relu(x)
        return x


class LongLoop(nn.Module):
    """Many iterations to test O(n^2) pairwise merge performance."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        for _ in range(20):
            x = self.fc(x)
            x = torch.relu(x)
        return x


class DataDependentBranchLoop(nn.Module):
    """Branch inside loop — some iterations take different paths."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        for i in range(4):
            x = self.fc(x)
            if i % 2 == 0:
                x = torch.relu(x)
            else:
                x = torch.sigmoid(x)
        return x


class SequentialParamFreeLoops(nn.Module):
    """Two back-to-back param-free loops with identical ops.
    Tests that adjacency correctly separates them."""

    @staticmethod
    def forward(x):
        for _ in range(3):
            x = torch.sin(x)
            x = torch.cos(x)
        x = x + 1
        for _ in range(3):
            x = torch.sin(x)
            x = torch.cos(x)
        return x


# =============================================================================
# Conditional Diffusion UNet
# =============================================================================
# Adapted from TeaPearce/Conditional_Diffusion_MNIST:
# https://github.com/TeaPearce/Conditional_Diffusion_MNIST


class _ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.same_channels = in_channels == out_channels
        self.is_res = is_res
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.is_res:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if self.same_channels:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            return x2


class _UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(_ResidualConvBlock(in_channels, out_channels), nn.MaxPool2d(2))

    def forward(self, x):
        return self.model(x)


class _UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            _ResidualConvBlock(out_channels, out_channels),
            _ResidualConvBlock(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x


class _EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)


class ContextUnet(nn.Module):
    """Conditional UNet for diffusion models."""

    def __init__(self, in_channels, n_feat=256, n_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.n_feat = n_feat
        self.n_classes = n_classes

        self.init_conv = _ResidualConvBlock(in_channels, n_feat, is_res=True)

        self.down1 = _UnetDown(n_feat, n_feat)
        self.down2 = _UnetDown(n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(7), nn.GELU())

        self.timeembed1 = _EmbedFC(1, 2 * n_feat)
        self.timeembed2 = _EmbedFC(1, 1 * n_feat)
        self.contextembed1 = _EmbedFC(n_classes, 2 * n_feat)
        self.contextembed2 = _EmbedFC(n_classes, 1 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 7, 7),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = _UnetUp(4 * n_feat, n_feat)
        self.up2 = _UnetUp(2 * n_feat, n_feat)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.ReLU(),
            nn.Conv2d(n_feat, self.in_channels, 3, 1, 1),
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        down1 = self.down1(x)
        down2 = self.down2(down1)
        hiddenvec = self.to_vec(down2)

        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)

        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = -1 * (1 - context_mask)
        c = c * context_mask

        cemb1 = self.contextembed1(c).view(-1, self.n_feat * 2, 1, 1)
        temb1 = self.timeembed1(t).view(-1, self.n_feat * 2, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.n_feat, 1, 1)
        temb2 = self.timeembed2(t).view(-1, self.n_feat, 1, 1)

        up1 = self.up0(hiddenvec)
        up2 = self.up1(cemb1 * up1 + temb1, down2)
        up3 = self.up2(cemb2 * up2 + temb2, down1)
        out = self.out(torch.cat((up3, x), 1))
        return out


#  ****************************************************
#  **** View Mutation / Child Tensor Variation Models ****
#  ****************************************************


class ViewMutationUnsqueeze(nn.Module):
    """Mutation through unsqueeze view: y = x.unsqueeze(0); y.fill_(0); return x.
    The fill_ mutates x's storage through the view, so x's tensor_contents at
    logging time differs from what children actually receive."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 8)

    def forward(self, x):
        x = self.linear(x)
        y = x.unsqueeze(0)
        y.fill_(0)
        return x


class ViewMutationReshape(nn.Module):
    """Mutation through reshape view: y = x.reshape(1, -1); y.zero_(); return x.
    The zero_ mutates x's storage through the reshaped view."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 8)

    def forward(self, x):
        x = self.linear(x)
        y = x.reshape(1, -1)
        y.zero_()
        return x


class ViewMutationTranspose(nn.Module):
    """Mutation through transpose view: y = x.t(); y.fill_(42); return x.
    The fill_ mutates x's storage through the transposed view."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 8)

    def forward(self, x):
        x = self.linear(x)
        y = x.t()
        y.fill_(42)
        return x


class MultipleViewMutations(nn.Module):
    """Multiple views mutated independently: y = x[0:2]; z = x[2:4];
    y.zero_(); z.fill_(1); return x.  Two different slices mutated."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 8)

    def forward(self, x):
        x = self.linear(x)
        y = x[0:2]
        z = x[2:4]
        y.zero_()
        z.fill_(1)
        return x


class ChainedViewMutation(nn.Module):
    """Mutation through chained views: y = x.unsqueeze(0); z = y.squeeze(0);
    z.zero_(); return x.  The zero_ propagates through two levels of views."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 8)

    def forward(self, x):
        x = self.linear(x)
        y = x.unsqueeze(0)
        z = y.squeeze(0)
        z.zero_()
        return x


class OutputMatchesParent(nn.Module):
    """No mutation: output IS the same tensor as the linear output.
    Verifies no false-positive child tensor variations are detected."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 8)

    def forward(self, x):
        x = self.linear(x)
        y = x + 1
        z = y * 2
        return z


class TensorWrapper:
    """Simulates complex tensor wrappers (like ESCNN's GeometricTensor) that
    cause copy.deepcopy to hang.  The circular reference makes deepcopy
    loop infinitely, while the .tensor attribute lets torchlens extract
    the underlying data."""

    def __init__(self, tensor):
        self.tensor = tensor
        self._self_ref = self  # circular reference → deepcopy hangs


class WrappedInputModel(nn.Module):
    """Model that receives a TensorWrapper and extracts its .tensor.

    Used to test that torchlens can handle non-deepcopy-safe input
    arguments without hanging (issue #18).
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, wrapper):
        return self.linear(wrapper.tensor)


class TupleInputModel(nn.Module):
    """Model that takes a single tuple of tensors as its only argument.

    Used to test that torchlens does not incorrectly unpack the tuple into
    multiple positional args (issue #43).
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        a, b = x
        return self.linear(a) + self.linear(b)


class FunctionalAfterSubmodule(nn.Module):
    """Container module with a functional op (relu) after a leaf submodule (linear).

    The relu should render as an oval in the graph, not a box (issue #48).
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        return torch.relu(self.linear(x))


class StochasticDepthModel(nn.Module):
    """Model with stochastic depth (dropout-like skip) that changes the
    computational graph between forward passes unless RNG state is restored.

    Used to test that the two-pass architecture handles stochastic models
    correctly (issue #58).
    """

    def __init__(self, drop_prob=0.5):
        super().__init__()
        self.linear1 = nn.Linear(5, 5)
        self.linear2 = nn.Linear(5, 5)
        self.drop_prob = drop_prob

    def forward(self, x):
        x = self.linear1(x)
        # Stochastic depth: randomly skip linear2
        if self.training and torch.rand(1).item() < self.drop_prob:
            return x
        return self.linear2(x)


#  *****************************************************
#  **** Aesthetic Testing Models (8-dim for speed) ****
#  *****************************************************


class AestheticDeepNested(nn.Module):
    """Tests nesting depth visualization at levels 0, 1, 2, 3, full.
    Root → outer (Linear+mul) → mid (add) → inner (Linear+relu)
    """

    def __init__(self):
        super().__init__()
        self.outer = _AestheticOuter()

    def forward(self, x):
        return self.outer(x)


class _AestheticOuter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)
        self.mid = _AestheticMid()

    def forward(self, x):
        x = self.linear(x)
        x = x * 2
        return self.mid(x)


class _AestheticMid(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _AestheticInner()

    def forward(self, x):
        x = x + 1
        return self.inner(x)


class _AestheticInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        x = self.linear(x)
        return torch.relu(x)


class AestheticSharedModule(nn.Module):
    """Tests rolled vs unrolled visualization. Single Linear module called 4 times with relu between."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x)
        x = self.fc(x)
        x = torch.relu(x)
        x = self.fc(x)
        x = torch.relu(x)
        x = self.fc(x)
        return x


class AestheticBufferBranch(nn.Module):
    """Tests buffer visibility toggle and branching. Has a registered buffer, a Linear, and a sin branch that merges."""

    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.ones(8) * 0.5)
        self.linear = nn.Linear(8, 8)

    def forward(self, x):
        x = x * self.scale
        a = self.linear(x)
        b = torch.sin(x)
        return a + b


class AestheticKitchenSink(nn.Module):
    """Combines nesting + shared modules + buffers + branching. Exercises everything in one graph."""

    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.randn(8))
        self.shared_fc = nn.Linear(8, 8)
        self.nest = _AestheticKitchenNest()

    def forward(self, x):
        x = x + self.bias
        a = self.shared_fc(x)
        b = self.nest(x)
        x = a + b
        x = self.shared_fc(x)
        return torch.relu(x)


class _AestheticKitchenNest(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))


class AestheticFrozenMix(nn.Module):
    """Has both frozen and trainable params for testing frozen-param aesthetics."""

    def __init__(self):
        super().__init__()
        self.frozen_fc = nn.Linear(8, 8)
        self.trainable_fc = nn.Linear(8, 8)
        # Freeze the first layer
        for param in self.frozen_fc.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.frozen_fc(x)
        x = torch.relu(x)
        x = self.trainable_fc(x)
        return x


# =============================================================================
# Group A: Attention & Transformers
# =============================================================================


class MultiheadAttentionModel(nn.Module):
    """nn.MultiheadAttention with multi-output (attn_output, attn_weights)."""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=False)

    def forward(self, x):
        attn_output, _attn_weights = self.attn(x, x, x)
        return attn_output


class ScaledDotProductAttentionModel(nn.Module):
    """F.scaled_dot_product_attention (fused kernel)."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (batch, heads, seq_len, head_dim) = (2, 4, 10, 8)
        return torch.nn.functional.scaled_dot_product_attention(x, x, x)


class TransformerEncoderModel(nn.Module):
    """nn.TransformerEncoder + nn.TransformerEncoderLayer, 2 layers."""

    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=16, nhead=2, dim_feedforward=32, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, x):
        return self.encoder(x)


class TransformerDecoderModel(nn.Module):
    """nn.TransformerDecoder with cross-attention, multi-input."""

    def __init__(self):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=16, nhead=2, dim_feedforward=32, batch_first=False
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

    def forward(self, tgt, memory):
        return self.decoder(tgt, memory)


class EmbeddingPositionalModel(nn.Module):
    """nn.Embedding with sinusoidal positional encoding."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 16)
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        # x: integer tokens (2, 10)
        emb = self.embedding(x)
        # Simple sin/cos positional encoding
        seq_len = emb.size(1)
        pos = torch.arange(seq_len, dtype=torch.float, device=emb.device).unsqueeze(1)
        pe = torch.sin(pos * 0.01)
        emb = emb + pe
        return self.fc(emb)


class EinsumModel(nn.Module):
    """torch.einsum with multiple contraction patterns."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (2, 3, 4)
        # Batch matrix multiply
        y = torch.einsum("bij,bjk->bik", x, x.transpose(-1, -2))
        # Batch trace
        z = torch.einsum("bii->b", y)
        return z


# =============================================================================
# Group B: Container Modules
# =============================================================================


class ModuleListModel(nn.Module):
    """nn.ModuleList iterated in a loop."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(5, 5) for _ in range(3)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


class ModuleListIndexedModel(nn.Module):
    """nn.ModuleList with sparse index access (skip middle)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(5, 5) for _ in range(4)])

    def forward(self, x):
        x = self.layers[0](x)
        x = self.layers[3](x)
        return x


class ModuleDictModel(nn.Module):
    """nn.ModuleDict with string-key dispatch."""

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleDict(
            {
                "path_a": nn.Linear(5, 5),
                "path_b": nn.Linear(5, 5),
            }
        )

    def forward(self, x):
        a = self.branches["path_a"](x)
        b = self.branches["path_b"](x)
        return a + b


class VarArgsModel(nn.Module):
    """*args forward signature, variable-length inputs."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, *args):
        out = args[0]
        for t in args[1:]:
            out = out + t
        return self.fc(out)


class KwargsModel(nn.Module):
    """**kwargs forward, keyword-only tensors."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, **kwargs):
        out = kwargs["a"] + kwargs["b"]
        return self.fc(out)


# =============================================================================
# Group C: Conditional & Dynamic Ops
# =============================================================================


class TorchWhereModel(nn.Module):
    """torch.where() conditional tensor selection."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        condition = x > 0.5
        return torch.where(condition, x, torch.zeros_like(x))


class ScatterGatherModel(nn.Module):
    """torch.scatter + torch.gather with index tensors."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (3, 5)
        idx = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 2, 2, 2, 2]])
        gathered = torch.gather(x, 1, idx)
        out = torch.zeros_like(x)
        out = out.scatter(1, idx, gathered)
        return out


class NoGradBlockModel(nn.Module):
    """torch.no_grad() context inside forward."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc(x)
        with torch.no_grad():
            scale = (x * 2).mean()
        return x * scale


class WhileLoopModel(nn.Module):
    """Data-dependent while loop (converging x*0.9)."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        count = 0
        while x.mean() > 0.1 and count < 10:
            x = x * 0.9
            count += 1
        return x


class NestedConditionalLoopModel(nn.Module):
    """Conditionals inside loops (different graph per iteration)."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        for i in range(4):
            if i % 2 == 0:
                x = torch.sin(x)
            else:
                x = torch.cos(x)
            if i > 1:
                x = x + 0.1
        return x


# =============================================================================
# Group D: Normalization & Conv Variants
# =============================================================================


class LayerNormModel(nn.Module):
    """nn.LayerNorm."""

    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(8)
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        x = self.ln(x)
        return self.fc(x)


class GroupNormModel(nn.Module):
    """nn.GroupNorm."""

    def __init__(self):
        super().__init__()
        self.gn = nn.GroupNorm(4, 8)
        self.conv = nn.Conv2d(8, 8, 3, padding=1)

    def forward(self, x):
        x = self.gn(x)
        return self.conv(x)


class InstanceNormModel(nn.Module):
    """nn.InstanceNorm2d."""

    def __init__(self):
        super().__init__()
        self.inorm = nn.InstanceNorm2d(3)
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        x = self.inorm(x)
        return self.conv(x)


class Conv1dModel(nn.Module):
    """nn.Conv1d + nn.AdaptiveAvgPool1d."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class Conv3dModel(nn.Module):
    """nn.Conv3d + nn.AdaptiveAvgPool3d."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# =============================================================================
# Group E: Residual & Parameter Sharing
# =============================================================================


class ResidualBlockModel(nn.Module):
    """ResNet-style Conv→BN→ReLU→Conv→BN + skip."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class SharedParamBranchModel(nn.Module):
    """Same nn.Linear called in two independent branches with different inputs."""

    def __init__(self):
        super().__init__()
        self.shared_fc = nn.Linear(5, 5)

    def forward(self, x):
        a = self.shared_fc(x + 1)
        b = self.shared_fc(x * 2)
        return a + b


class ModelCallingModelModel(nn.Module):
    """One nn.Module calling another full nn.Module as submodule."""

    def __init__(self):
        super().__init__()
        self.sub = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8))
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        x = x.flatten(1)
        x = self.sub(x)
        return self.head(x)


class BidirectionalGRUModel(nn.Module):
    """nn.GRU(bidirectional=True), concatenated hidden states."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(8, 4, batch_first=False, bidirectional=True)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        # x: (seq_len=5, batch=2, features=8)
        output, _hidden = self.gru(x)
        # Use last timestep
        return self.fc(output[-1])


# =============================================================================
# Group F: In-Place & Type Operations
# =============================================================================


class InPlaceChainModel(nn.Module):
    """Chained in-place: add_(), mul_(), relu_() in sequence."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        y = x.clone()
        y.add_(1)
        y.mul_(2)
        y.relu_()
        return y


class TypeCastChainModel(nn.Module):
    """.float() → .long() → .float() → linear."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        x = x.float()
        x = (x * 100).long()
        x = x.float() / 100
        return self.fc(x)


class LikeOpsModel(nn.Module):
    """full_like, rand_like, randn_like (untested *_like ops)."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        a = torch.full_like(x, 0.5)
        b = torch.rand_like(x)
        c = torch.randn_like(x)
        return x + a + b + c


class MultiTensorReturnModel(nn.Module):
    """torch.chunk, torch.unbind — ops returning multiple tensors."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (3, 4) → chunk into 3 pieces
        chunks = torch.chunk(x, 3, dim=0)
        return chunks[0] + chunks[1] + chunks[2]


class MixedDtypeModel(nn.Module):
    """Int index tensors mixed with float computation."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (4, 4) float
        idx = torch.arange(4)
        selected = x[idx]
        return selected.sum()


# =============================================================================
# Group G: Scalar & Broadcasting
# =============================================================================


class ScalarTensorModel(nn.Module):
    """0-dim scalar tensors as arguments in binary ops."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        scale = torch.tensor(2.0)
        bias = torch.tensor(0.5)
        return x * scale + bias


class BroadcastingModel(nn.Module):
    """Mismatched-shape broadcasting: (3,4,5)+(1,4,1)*(5,)."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (3, 4, 5)
        b = torch.ones(1, 4, 1)
        c = torch.ones(5)
        return (x + b) * c


class PackedSequenceModel(nn.Module):
    """pack_padded_sequence / pad_packed_sequence + LSTM."""

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(8, 4, batch_first=False)
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        # x: (5, 3, 8) — seq_len=5, batch=3, features=8
        # Simulate lengths: [5, 3, 2]
        lengths = torch.tensor([5, 3, 2])
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=True)
        output, (_h, _c) = self.lstm(packed)
        padded, _lens = nn.utils.rnn.pad_packed_sequence(output)
        return self.fc(padded[-1])


class CustomAutogradModel(nn.Module):
    """torch.autograd.Function subclass in forward."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        x = _DoubleFunction.apply(x)
        return x


class _DoubleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x * 2

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 2


# =============================================================================
# Group H: Exemption Registry Stress Tests
# =============================================================================


class CrossEntropyModel(nn.Module):
    """F.cross_entropy — STRUCTURAL_ARG_POSITIONS[1]."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 10)

    def forward(self, x):
        logits = self.fc(x)
        targets = torch.zeros(logits.size(0), dtype=torch.long)
        return torch.nn.functional.cross_entropy(logits, targets)


class IndexSelectModel(nn.Module):
    """torch.index_select — STRUCTURAL_ARG_POSITIONS[2]."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (5, 3)
        idx = torch.tensor([0, 2, 4])
        return torch.index_select(x, 0, idx)


class InterpolateModel(nn.Module):
    """F.interpolate with scale_factor — CUSTOM_EXEMPTION_CHECKS."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (1, 1, 4, 4)
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")


class ForwardHooksModel(nn.Module):
    """register_forward_hook on submodule — hooks during logging."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)
        self._hook_output = None
        self.fc.register_forward_hook(self._save_hook)

    def _save_hook(self, module, input, output):
        self._hook_output = output

    def forward(self, x):
        return self.fc(x)


# =============================================================================
# Group I: Architecture Patterns (built-in, no external deps)
# =============================================================================


class SimpleVAE(nn.Module):
    """Encoder→reparameterize→decoder, sampling in forward."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32 * 7 * 7, 16)
        self.fc_logvar = nn.Linear(32 * 7 * 7, 16)
        self.decoder_fc = nn.Linear(16, 32 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        h = self.decoder_fc(z).view(-1, 32, 7, 7)
        return self.decoder(h)


class SimpleGenerator(nn.Module):
    """GAN generator: Linear→BN→ReLU→Linear→Tanh."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z).view(-1, 1, 28, 28)


class SimpleDiscriminator(nn.Module):
    """GAN discriminator: Conv→LeakyReLU→Conv→Sigmoid."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    """Encoder-decoder with skip connections."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d1 = self.dec1(torch.cat([self.up(b), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e1], dim=1))
        return self.out(d2)


class TemporalConvNet(nn.Module):
    """TCN with dilated causal convolutions (dilation 1, 2, 4)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 8, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(8, 8, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(8, 8, 3, padding=4, dilation=4)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        return self.fc(x.mean(dim=-1))


class ESPCNSuperRes(nn.Module):
    """Sub-pixel convolution (nn.PixelShuffle)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return self.shuffle(x)


class SimplePointNet(nn.Module):
    """PointNet: Conv1d shared MLPs→global max pool→FC."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # x: (batch, 3, num_points) = (2, 3, 128)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.max(dim=-1)[0]  # global max pool
        return self.fc(x)


class ActorCritic(nn.Module):
    """Shared backbone with two heads (actor+critic), multi-output."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 32), nn.ReLU())
        self.actor = nn.Linear(32, 2)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.actor(h), self.critic(h)


class TwoTowerRecommender(nn.Module):
    """Independent parallel branches with dot-product interaction."""

    def __init__(self):
        super().__init__()
        self.user_tower = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
        self.item_tower = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))

    def forward(self, user, item):
        u = self.user_tower(user)
        i = self.item_tower(item)
        return (u * i).sum(dim=-1)


class SimpleDepthEstimator(nn.Module):
    """Encoder-decoder for single-channel depth prediction."""

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


# =============================================================================
# Group Z: EVIL ADVERSARIAL EDGE CASES
# =============================================================================


class SameTensorAllArgs(nn.Module):
    """Z1: torch.addmm(x, x, x) — same tensor in ALL 3 arg positions.
    Attacks parent_layer_arg_locs dedup."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (4, 4)
        return torch.addmm(x, x, x)


class ViewChainMutateMiddle(nn.Module):
    """Z2: Creates 3 views, mutates MIDDLE view in-place, reads from FIRST view."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (4, 4)
        y = x.clone()
        v1 = y.view(2, 8)
        v2 = y.view(4, 4)
        _v3 = y.view(8, 2)
        v2.fill_(0)
        return v1.sum()


class SelfCachingModel(nn.Module):
    """Z4: Stores intermediate results on self.cache during forward."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)
        self.cache = None

    def forward(self, x):
        self.cache = self.fc(x)
        intermediate = torch.relu(self.cache)
        return intermediate + self.cache


class DeleteTensorMidForward(nn.Module):
    """Z5: Creates tensor, uses it, dels it, creates new tensor."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        temp = self.fc(x)
        result = temp * 2
        del temp
        other = self.fc(x + 1)
        return result + other


class DynamicModuleCreation(nn.Module):
    """Z6: Creates a NEW nn.Linear INSIDE forward and uses it."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        x = self.fc(x)
        dynamic_layer = nn.Linear(5, 5)
        dynamic_layer.weight = self.fc.weight
        dynamic_layer.bias = self.fc.bias
        return dynamic_layer(x)


class NonPersistentBuffer(nn.Module):
    """Z8: register_buffer with persistent=False, then uses buffer in forward."""

    def __init__(self):
        super().__init__()
        self.register_buffer("buf", torch.ones(5) * 0.5, persistent=False)
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        return self.fc(x * self.buf)


class ContiguousNoOp(nn.Module):
    """Z9: Calls x.contiguous() on already-contiguous tensor."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(5, 5)

    def forward(self, x):
        x = self.fc(x)
        x = x.contiguous()
        return x * 2


class StackViewsOfSameTensor(nn.Module):
    """Z10: torch.stack([x[0], x[1], x[2]]) — stacking views of same parent."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # x: (4, 4) — stack first 3 rows
        return torch.stack([x[0], x[1], x[2]])


class SaturatedSoftmax(nn.Module):
    """Z13: F.softmax(x * 1000) — extreme values saturate softmax."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return torch.nn.functional.softmax(x * 1000, dim=-1)


class DuplicateValueParents(nn.Module):
    """Z15: Two DIFFERENT parent tensors with IDENTICAL values."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        a = torch.full_like(x, 0.5)
        b = torch.full_like(x, 0.5)
        return x * a + x * b


class EmptyTensorChain(nn.Module):
    """Z16: Operations on empty tensors (numel==0)."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        empty = x[:0]  # empty tensor
        reshaped = empty.reshape(0, x.size(1))
        unsqueezed = reshaped.unsqueeze(0)
        # Return something non-empty so the model has an output
        return x.sum() + unsqueezed.sum()


class ClampNarrowRange(nn.Module):
    """Z17: torch.clamp(x, 0.499, 0.501) — perturbation lands in same range."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return torch.clamp(x, 0.499, 0.501)


class RoundNearIntegers(nn.Module):
    """Z18: torch.round(x) where input is near-integer values."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        # Scale to near-integers
        near_int = x * 0.0001 + torch.arange(x.numel(), dtype=x.dtype).reshape(x.shape)
        return torch.round(near_int)


class BoolCastExploit(nn.Module):
    """Z19: (x > 0).long() * 10 — bool intermediate cast to long."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        mask = (x > 0).long()
        return mask * 10


class FakeLoopSameOpType(nn.Module):
    """Z20: relu→matmul→relu→matmul alternating ops that look like a loop but aren't."""

    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(4, 4))
        self.w2 = nn.Parameter(torch.randn(4, 4))

    def forward(self, x):
        a = torch.relu(x)
        b = torch.matmul(a, self.w1)
        c = torch.relu(b + 1)  # different args — not a loop
        d = torch.matmul(c, self.w2)  # different weight
        return d


class AutocastMidForward(nn.Module):
    """Z22: torch.amp.autocast('cpu') context wrapping part of forward."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc1(x)
        with torch.amp.autocast("cpu"):
            x = self.fc2(x)
        return x


# =============================================================================
# Group J: Autoencoders
# =============================================================================


class VanillaAutoencoder(nn.Module):
    """Standard deterministic autoencoder: encoder→bottleneck→decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with symmetric encoder-decoder."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class SparseAutoencoder(nn.Module):
    """Autoencoder with L1-penalized bottleneck (sparsity via activation)."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        # L1 sparsity applied externally during training; forward is standard
        return self.decoder(z)


class DenoisingAutoencoder(nn.Module):
    """Autoencoder that adds noise in forward (trains to reconstruct clean input)."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Add noise during forward (denoising objective)
        noisy = x + 0.1 * torch.randn_like(x)
        z = self.encoder(noisy)
        return self.decoder(z)


class VQVAE(nn.Module):
    """Vector-quantized VAE: encoder→nearest codebook→decoder.

    Straight-through estimator for gradients past the argmin.
    """

    def __init__(self, num_embeddings=16, embedding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, embedding_dim, 3, stride=2, padding=1),
        )
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embedding_dim, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        ze = self.encoder(x)  # (B, D, H, W)
        B, D, H, W = ze.shape
        flat = ze.permute(0, 2, 3, 1).reshape(-1, D)  # (B*H*W, D)
        # Nearest codebook lookup
        dists = torch.cdist(flat, self.codebook.weight)
        indices = dists.argmin(dim=-1)
        zq = self.codebook(indices).view(B, H, W, D).permute(0, 3, 1, 2)
        # Straight-through estimator
        zq_st = ze + (zq - ze).detach()
        return self.decoder(zq_st)


class BetaVAE(nn.Module):
    """Beta-VAE with tunable KL weight (same architecture as VAE, different loss weighting)."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, 16)
        self.fc_logvar = nn.Linear(128, 16)
        self.decoder = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z)


class ConditionalVAE(nn.Module):
    """Conditional VAE: label embedding concatenated to both encoder and decoder."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.label_embed = nn.Embedding(num_classes, 16)
        self.encoder = nn.Sequential(
            nn.Linear(784 + 16, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, 16)
        self.fc_logvar = nn.Linear(128, 16)
        self.decoder = nn.Sequential(
            nn.Linear(16 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid(),
        )

    def forward(self, x, label):
        label_emb = self.label_embed(label)
        h = self.encoder(torch.cat([x, label_emb], dim=-1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(torch.cat([z, label_emb], dim=-1))


# =============================================================================
# Group K: State Space Models
# =============================================================================


class SimpleSSM(nn.Module):
    """Minimal state-space model: discretized linear recurrence + output projection.

    Implements x_k = A x_{k-1} + B u_k; y_k = C x_k
    Uses a sequential scan (no parallel associative scan for simplicity).
    """

    def __init__(self, input_dim=8, state_dim=16, output_dim=8):
        super().__init__()
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)
        self.C = nn.Parameter(torch.randn(output_dim, state_dim) * 0.1)
        self.state_dim = state_dim

    def forward(self, u):
        # u: (batch, seq_len, input_dim)
        B, T, _ = u.shape
        x = torch.zeros(B, self.state_dim, device=u.device, dtype=u.dtype)
        outputs = []
        for t in range(T):
            x = torch.tanh(x @ self.A.T + u[:, t] @ self.B.T)
            outputs.append(x @ self.C.T)
        return torch.stack(outputs, dim=1)


class SelectiveSSM(nn.Module):
    """Simplified Mamba-style selective SSM: input-dependent gating.

    Key innovation vs simple SSM: A/B/C parameters are projected from input,
    making the system input-dependent (selective).
    """

    def __init__(self, d_model=16, d_state=8):
        super().__init__()
        self.d_state = d_state
        self.proj_in = nn.Linear(d_model, d_model * 2)
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.proj_dt = nn.Linear(d_model, d_model)
        self.proj_B = nn.Linear(d_model, d_state)
        self.proj_C = nn.Linear(d_model, d_state)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.proj_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, T, D = x.shape
        xz = self.proj_in(x)
        x_branch, z = xz.chunk(2, dim=-1)
        # Conv1d expects (B, D, T)
        x_conv = self.conv1d(x_branch.transpose(1, 2)).transpose(1, 2)
        x_conv = torch.nn.functional.silu(x_conv)
        # Input-dependent parameters
        dt = torch.nn.functional.softplus(self.proj_dt(x_conv))
        B_param = self.proj_B(x_conv)
        C_param = self.proj_C(x_conv)
        A = -torch.exp(self.A_log)
        # Sequential scan
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            dB = dt[:, t].unsqueeze(-1) * B_param[:, t].unsqueeze(1)
            h = h * dA + x_conv[:, t].unsqueeze(-1) * dB
            y = (h * C_param[:, t].unsqueeze(1)).sum(dim=-1)
            outputs.append(y)
        y = torch.stack(outputs, dim=1)
        y = y * torch.nn.functional.silu(z)
        return self.proj_out(y)


class GatedSSMBlock(nn.Module):
    """SSM block with pre-norm and residual connection (Mamba block pattern)."""

    def __init__(self, d_model=16, d_state=8):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state)

    def forward(self, x):
        return x + self.ssm(self.norm(x))


class StackedSSM(nn.Module):
    """Multi-layer SSM with embedding and classification head."""

    def __init__(self, vocab_size=100, d_model=16, d_state=8, n_layers=2, n_classes=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([GatedSSMBlock(d_model, d_state) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, seq_len) integer tokens
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.head(h.mean(dim=1))


# =============================================================================
# Group L: Additional Architecture Patterns
# =============================================================================


class SiameseNetwork(nn.Module):
    """Siamese network: shared encoder, L2 distance output."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        return torch.norm(z1 - z2, dim=-1)


class MLPMixer(nn.Module):
    """MLP-Mixer: token-mixing + channel-mixing MLPs, no attention."""

    def __init__(self, num_patches=16, d_model=32, num_classes=10, depth=2):
        super().__init__()
        self.patch_embed = nn.Linear(12, d_model)  # 2x2 patches from 3-channel
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(d_model),
                        "token_mix": nn.Sequential(
                            nn.Linear(num_patches, num_patches),
                            nn.GELU(),
                            nn.Linear(num_patches, num_patches),
                        ),
                        "norm2": nn.LayerNorm(d_model),
                        "channel_mix": nn.Sequential(
                            nn.Linear(d_model, d_model * 2),
                            nn.GELU(),
                            nn.Linear(d_model * 2, d_model),
                        ),
                    }
                )
            )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 3, 8, 8) -> 16 patches of 2x2x3=12
        B = x.shape[0]
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (B, 3, 4, 4, 2, 2)
        patches = patches.contiguous().view(B, -1, 12)  # (B, 16, 12)
        h = self.patch_embed(patches)  # (B, 16, d_model)
        for block in self.blocks:
            # Token mixing
            residual = h
            h = block["norm1"](h)
            h = h.transpose(1, 2)  # (B, d_model, num_patches)
            h = block["token_mix"](h)
            h = h.transpose(1, 2) + residual
            # Channel mixing
            residual = h
            h = block["norm2"](h)
            h = block["channel_mix"](h) + residual
        h = self.norm(h)
        return self.head(h.mean(dim=1))


class SimpleGCN(nn.Module):
    """Graph Convolutional Network (Kipf & Welling): A_hat @ X @ W per layer.

    Takes adjacency + features as separate inputs.
    """

    def __init__(self, in_features=8, hidden=16, out_features=4):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden, bias=False)
        self.w2 = nn.Linear(hidden, out_features, bias=False)

    def forward(self, x, adj):
        # x: (num_nodes, in_features), adj: (num_nodes, num_nodes)
        h = torch.relu(adj @ self.w1(x))
        return adj @ self.w2(h)


class SimpleGAT(nn.Module):
    """Graph Attention Network: attention-weighted message passing."""

    def __init__(self, in_features=8, hidden=16, out_features=4):
        super().__init__()
        self.W = nn.Linear(in_features, hidden, bias=False)
        self.a = nn.Linear(2 * hidden, 1, bias=False)
        self.out_proj = nn.Linear(hidden, out_features)

    def forward(self, x, adj):
        # x: (N, in_features), adj: (N, N)
        N = x.shape[0]
        h = self.W(x)  # (N, hidden)
        # Compute attention for all pairs
        h_i = h.unsqueeze(1).expand(-1, N, -1)  # (N, N, hidden)
        h_j = h.unsqueeze(0).expand(N, -1, -1)  # (N, N, hidden)
        e = torch.nn.functional.leaky_relu(self.a(torch.cat([h_i, h_j], dim=-1)).squeeze(-1), 0.2)
        # Mask non-edges
        e = e.masked_fill(adj == 0, float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        alpha = alpha.masked_fill(torch.isnan(alpha), 0)
        out = alpha @ h  # (N, hidden)
        return self.out_proj(out)


class SimpleDiffusion(nn.Module):
    """Minimal denoising diffusion: noise predictor network.

    Takes noisy input + timestep embedding, predicts noise.
    """

    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
        )
        self.net = nn.Sequential(
            nn.Linear(784 + 32, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 784),
        )

    def forward(self, x, t):
        # x: (B, 784), t: (B, 1)
        t_emb = self.time_embed(t)
        return self.net(torch.cat([x, t_emb], dim=-1))


class SimpleNormalizingFlow(nn.Module):
    """Normalizing flow with affine coupling layers."""

    def __init__(self, dim=8, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "scale_net": nn.Sequential(
                            nn.Linear(dim // 2, dim),
                            nn.ReLU(),
                            nn.Linear(dim, dim // 2),
                        ),
                        "shift_net": nn.Sequential(
                            nn.Linear(dim // 2, dim),
                            nn.ReLU(),
                            nn.Linear(dim, dim // 2),
                        ),
                    }
                )
            )

    def forward(self, x):
        # x: (B, dim)
        for layer in self.layers:
            x1, x2 = x.chunk(2, dim=-1)
            s = layer["scale_net"](x1)
            t = layer["shift_net"](x1)
            x2 = x2 * torch.exp(s) + t
            x = torch.cat([x2, x1], dim=-1)  # swap halves
        return x


class CapsuleNetwork(nn.Module):
    """Simplified capsule network with dynamic routing."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        # Primary capsules: 8 capsule types, each producing 4-dim vectors
        self.primary_caps = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10 * 8)  # 10 classes, 8-dim capsules
        self.out = nn.Linear(80, 10)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        h = torch.relu(self.conv1(x))
        h = torch.relu(self.primary_caps(h))
        h = h.flatten(1)
        caps = self.fc(h).view(-1, 10, 8)
        # Squash activation (capsule nonlinearity)
        norms = caps.norm(dim=-1, keepdim=True)
        squashed = (norms**2 / (1 + norms**2)) * (caps / (norms + 1e-8))
        return self.out(squashed.flatten(1))


# =============================================================================
# Group M: Attention Variants
# =============================================================================


class MultiQueryAttentionModel(nn.Module):
    """Multi-Query Attention: single K/V head shared across all Q heads."""

    def __init__(self, dim=32, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, self.head_dim)
        self.v_proj = nn.Linear(dim, self.head_dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).unsqueeze(1)
        v = self.v_proj(x).unsqueeze(1)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.head_dim**0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class GroupedQueryAttentionModel(nn.Module):
    """Grouped-Query Attention: K/V shared within groups of Q heads."""

    def __init__(self, dim=32, n_heads=4, n_kv_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        reps = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(reps, dim=1)
        v = v.repeat_interleave(reps, dim=1)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.head_dim**0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class RoPEAttentionModel(nn.Module):
    """Self-attention with Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim=32, n_heads=4, max_seq=64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        freqs = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(max_seq).float()
        angles = torch.outer(t, freqs)
        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def _apply_rope(self, x):
        T = x.shape[2]
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
        out1 = x1 * cos - x2 * sin
        out2 = x1 * sin + x2 * cos
        return torch.stack([out1, out2], dim=-1).flatten(-2)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = self._apply_rope(q), self._apply_rope(k)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / (self.head_dim**0.5), dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out_proj(out)


class ALiBiAttentionModel(nn.Module):
    """ALiBi: linear bias on attention scores based on distance."""

    def __init__(self, dim=32, n_heads=4, max_seq=32):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        slopes = torch.pow(2, -torch.arange(1, n_heads + 1).float() * 8 / n_heads)
        self.register_buffer("slopes", slopes)

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn = q @ k.transpose(-2, -1) / (self.head_dim**0.5)
        positions = torch.arange(T, device=x.device)
        bias = -(positions.unsqueeze(0) - positions.unsqueeze(1)).abs().float()
        bias = bias.unsqueeze(0) * self.slopes.view(1, -1, 1, 1)
        attn = torch.softmax(attn + bias, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.out(out)


class SlotAttentionModel(nn.Module):
    """Slot Attention: iterative competitive binding to discrete slots."""

    def __init__(self, n_slots=4, dim=32, n_iter=3):
        super().__init__()
        self.n_slots = n_slots
        self.n_iter = n_iter
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Linear(dim * 2, dim))
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x):
        B, N, D = x.shape
        slots = self.slots_mu.expand(B, self.n_slots, -1) + 0.1 * torch.randn(
            B, self.n_slots, D, device=x.device
        )
        x = self.norm_input(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        for _ in range(self.n_iter):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.q_proj(slots)
            attn = torch.softmax(q @ k.transpose(-2, -1) / (D**0.5), dim=-1)
            updates = attn @ v
            slots = self.gru(updates.reshape(-1, D), slots_prev.reshape(-1, D)).view(
                B, self.n_slots, D
            )
            slots = slots + self.mlp(slots)
        return slots


class CrossAttentionModel(nn.Module):
    """Cross-attention: Q from learned latents, K/V from input (Perceiver-style)."""

    def __init__(self, dim=16, n_latents=4):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, dim))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        q = self.q_proj(latents)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (x.shape[-1] ** 0.5), dim=-1)
        return self.out_proj(attn @ v)


# =============================================================================
# Group N: Gating & Skip Patterns
# =============================================================================


class HighwayNetwork(nn.Module):
    """Highway Network: T(x)*H(x) + (1-T(x))*x gated skip connections."""

    def __init__(self, dim=32, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "transform": nn.Linear(dim, dim),
                        "gate": nn.Linear(dim, dim),
                    }
                )
            )

    def forward(self, x):
        for layer in self.layers:
            h = torch.relu(layer["transform"](x))
            t = torch.sigmoid(layer["gate"](x))
            x = t * h + (1 - t) * x
        return x


class SqueezeExcitationBlock(nn.Module):
    """SE Block: global pool -> FC -> sigmoid -> channel-wise rescaling."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.se_fc1 = nn.Linear(16, 4)
        self.se_fc2 = nn.Linear(4, 16)
        self.out = nn.Conv2d(16, 8, 1)

    def forward(self, x):
        h = torch.relu(self.conv(x))
        se = h.mean(dim=[2, 3])
        se = torch.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))
        h = h * se.unsqueeze(-1).unsqueeze(-1)
        return self.out(h)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable: groups=in_channels depthwise + 1x1 pointwise."""

    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(3, 3, 3, padding=1, groups=3)
        self.pointwise = nn.Conv2d(3, 16, 1)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = torch.relu(self.depthwise(x))
        x = torch.relu(self.bn(self.pointwise(x)))
        return self.fc(self.pool(x).flatten(1))


class InvertedResidualBlock(nn.Module):
    """MobileNetV2: expand -> depthwise -> project with residual."""

    def __init__(self, dim=16, expand=4):
        super().__init__()
        mid = dim * expand
        self.conv_in = nn.Conv2d(3, dim, 3, padding=1)
        self.expand = nn.Conv2d(dim, mid, 1)
        self.depthwise = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.project = nn.Conv2d(mid, dim, 1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.bn2 = nn.BatchNorm2d(mid)
        self.bn3 = nn.BatchNorm2d(dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, 10)

    def forward(self, x):
        x = torch.relu(self.conv_in(x))
        residual = x
        x = torch.relu(self.bn1(self.expand(x)))
        x = torch.relu(self.bn2(self.depthwise(x)))
        x = self.bn3(self.project(x))
        x = x + residual
        return self.fc(self.pool(x).flatten(1))


class FeaturePyramidNet(nn.Module):
    """FPN: multi-scale backbone with lateral + top-down connections."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())
        self.c3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.lat2 = nn.Conv2d(32, 64, 1)
        self.lat1 = nn.Conv2d(16, 64, 1)
        self.out3 = nn.Conv2d(64, 64, 3, padding=1)
        self.out2 = nn.Conv2d(64, 64, 3, padding=1)
        self.out1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        p3 = self.out3(c3)
        p2 = self.out2(
            self.lat2(c2) + nn.functional.interpolate(p3, size=c2.shape[2:], mode="nearest")
        )
        p1 = self.out1(
            self.lat1(c1) + nn.functional.interpolate(p2, size=c1.shape[2:], mode="nearest")
        )
        return self.fc(self.pool(p1).flatten(1))


# =============================================================================
# Group O: Generative & Self-Supervised
# =============================================================================


class HierarchicalVAE(nn.Module):
    """Two-level hierarchical VAE with top-down conditional prior."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(16, 32)
        self.enc2 = nn.Linear(32, 16)
        self.mu2 = nn.Linear(16, 8)
        self.logvar2 = nn.Linear(16, 8)
        self.prior1 = nn.Linear(8, 32)
        self.mu1 = nn.Linear(32, 8)
        self.logvar1 = nn.Linear(32, 8)
        self.dec = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))

    def _reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h1 = torch.relu(self.enc1(x))
        h2 = torch.relu(self.enc2(h1))
        z2 = self._reparam(self.mu2(h2), self.logvar2(h2))
        h_prior = torch.relu(self.prior1(z2))
        z1 = self._reparam(self.mu1(h_prior), self.logvar1(h_prior))
        return self.dec(torch.cat([z1, z2], dim=-1))


class GatedConvModel(nn.Module):
    """WaveNet-style gated convolutions: tanh(conv_f) * sigmoid(conv_g)."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv1d(3, 16, 1)
        self.conv_f1 = nn.Conv1d(16, 16, 3, padding=1, dilation=1)
        self.conv_g1 = nn.Conv1d(16, 16, 3, padding=1, dilation=1)
        self.conv_f2 = nn.Conv1d(16, 16, 3, padding=2, dilation=2)
        self.conv_g2 = nn.Conv1d(16, 16, 3, padding=2, dilation=2)
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        x = self.conv_in(x)
        x = torch.tanh(self.conv_f1(x)) * torch.sigmoid(self.conv_g1(x))
        x = torch.tanh(self.conv_f2(x)) * torch.sigmoid(self.conv_g2(x))
        return self.out(x.mean(dim=-1))


class MaskedConvModel(nn.Module):
    """PixelCNN-style: mask applied to conv weights for autoregressive ordering."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=False)
        self.out = nn.Conv2d(16, 1, 1)
        mask = torch.ones(3, 3)
        mask[1, 2] = 0
        mask[2, :] = 0
        self.register_buffer("mask", mask)

    def forward(self, x):
        w1 = self.conv1.weight * self.mask
        h = torch.relu(nn.functional.conv2d(x, w1, padding=1))
        w2 = self.conv2.weight * self.mask
        h = torch.relu(nn.functional.conv2d(h, w2, padding=1))
        return torch.sigmoid(self.out(h))


class SimCLRModel(nn.Module):
    """SimCLR: shared encoder + projection head on two views."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
        self.projector = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))

    def forward(self, x1, x2):
        z1 = nn.functional.normalize(self.projector(self.encoder(x1)), dim=-1)
        z2 = nn.functional.normalize(self.projector(self.encoder(x2)), dim=-1)
        return (z1 * z2).sum(dim=-1)


class StopGradientModel(nn.Module):
    """BYOL-style asymmetric: online encoder + predictor vs. detached target."""

    def __init__(self):
        super().__init__()
        self.online = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))
        self.predictor = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 16))
        self.target = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))

    def forward(self, x):
        online_out = self.predictor(self.online(x))
        target_out = self.target(x).detach()
        return nn.functional.mse_loss(online_out, target_out)


class AdaINModel(nn.Module):
    """Adaptive Instance Normalization: style transfer building block."""

    def __init__(self):
        super().__init__()
        self.content_enc = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.style_enc = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.style_scale = nn.Linear(16, 16)
        self.style_shift = nn.Linear(16, 16)
        self.decoder = nn.Sequential(nn.Conv2d(16, 3, 3, padding=1), nn.Sigmoid())

    def forward(self, content, style):
        c = self.content_enc(content)
        s = self.style_enc(style).flatten(1)
        scale = self.style_scale(s).unsqueeze(-1).unsqueeze(-1)
        shift = self.style_shift(s).unsqueeze(-1).unsqueeze(-1)
        mean = c.mean(dim=[2, 3], keepdim=True)
        std = c.std(dim=[2, 3], keepdim=True) + 1e-6
        styled = (c - mean) / std * scale + shift
        return self.decoder(styled)


# =============================================================================
# Group P: Exotic Architectures
# =============================================================================


class HyperNetwork(nn.Module):
    """One network generates weights for another (weight prediction)."""

    def __init__(self):
        super().__init__()
        self.weight_gen = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 8 * 16))
        self.bias_gen = nn.Linear(4, 16)
        self.main_fc = nn.Linear(16, 4)

    def forward(self, x, cond):
        w = self.weight_gen(cond).view(16, 8)
        b = self.bias_gen(cond).squeeze(0)
        h = torch.relu(x @ w.t() + b)
        return self.main_fc(h)


class DEQModel(nn.Module):
    """Deep Equilibrium Model: fixed-point iteration z* = f(z*) in forward."""

    def __init__(self):
        super().__init__()
        self.inject = nn.Linear(8, 16)
        self.f = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.out = nn.Linear(16, 4)

    def forward(self, x):
        z = self.inject(x)
        for _ in range(5):
            z = self.f(z) + self.inject(x)
        return self.out(z)


class SimpleNeuralODE(nn.Module):
    """Neural ODE via Euler integration: z(t+dt) = z(t) + dt*f(z(t))."""

    def __init__(self):
        super().__init__()
        self.dynamics = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 8))
        self.out = nn.Linear(8, 4)

    def forward(self, x):
        z = x
        dt = 0.1
        for _ in range(10):
            z = z + dt * self.dynamics(z)
        return self.out(z)


class MemoryAugmentedNet(nn.Module):
    """NTM-style differentiable read/write on external memory matrix."""

    def __init__(self, mem_size=8, mem_dim=16, input_dim=8):
        super().__init__()
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.controller = nn.Linear(input_dim, 32)
        self.read_key = nn.Linear(32, mem_dim)
        self.write_key = nn.Linear(32, mem_dim)
        self.write_val = nn.Linear(32, mem_dim)
        self.out = nn.Linear(mem_dim, 4)

    def forward(self, x):
        B = x.shape[0]
        memory = torch.zeros(B, self.mem_size, self.mem_dim, device=x.device)
        h = torch.relu(self.controller(x))
        wk = self.write_key(h)
        wv = self.write_val(h)
        w_attn = torch.softmax(memory @ wk.unsqueeze(-1), dim=1).squeeze(-1)
        memory = memory + w_attn.unsqueeze(-1) * wv.unsqueeze(1)
        rk = self.read_key(h)
        r_attn = torch.softmax(memory @ rk.unsqueeze(-1), dim=1).squeeze(-1)
        read = (r_attn.unsqueeze(-1) * memory).sum(dim=1)
        return self.out(read)


class SwiGLUFFN(nn.Module):
    """SwiGLU gated linear unit (LLaMA/Mistral FFN): silu(W1*x) * W3*x."""

    def __init__(self, dim=16, hidden=32):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, dim)
        self.w3 = nn.Linear(dim, hidden)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


# =============================================================================
# Group Q: Graph Neural Networks (expanded, pure-torch)
# =============================================================================


class GraphSAGEModel(nn.Module):
    """GraphSAGE: mean-aggregate neighbors, concat with self, project."""

    def __init__(self, in_dim=8, hidden=16, out_dim=4):
        super().__init__()
        self.W1 = nn.Linear(in_dim * 2, hidden)
        self.W2 = nn.Linear(hidden * 2, out_dim)

    def _aggregate(self, x, adj):
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        return (adj @ x) / deg

    def forward(self, x, adj):
        neigh = self._aggregate(x, adj)
        h = torch.relu(self.W1(torch.cat([x, neigh], dim=-1)))
        neigh2 = self._aggregate(h, adj)
        return self.W2(torch.cat([h, neigh2], dim=-1))


class GINModel(nn.Module):
    """Graph Isomorphism Network: MLP((1+eps)*h + sum(neighbors))."""

    def __init__(self, in_dim=8, hidden=16, out_dim=4):
        super().__init__()
        self.eps1 = nn.Parameter(torch.zeros(1))
        self.mlp1 = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.eps2 = nn.Parameter(torch.zeros(1))
        self.mlp2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x, adj):
        h = self.mlp1((1 + self.eps1) * x + adj @ x)
        return self.mlp2((1 + self.eps2) * h + adj @ h)


class EdgeConvModel(nn.Module):
    """DGCNN EdgeConv: MLP(x_i || x_j - x_i), max over neighbors."""

    def __init__(self, in_dim=3, hidden=16, out_dim=4):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(in_dim * 2, hidden), nn.ReLU())
        self.fc = nn.Linear(hidden, out_dim)

    def forward(self, x, adj):
        N = x.shape[0]
        xi = x.unsqueeze(1).expand(-1, N, -1)
        xj = x.unsqueeze(0).expand(N, -1, -1)
        edge_feat = torch.cat([xi, xj - xi], dim=-1)
        edge_out = self.edge_mlp(edge_feat) * adj.unsqueeze(-1)
        h = edge_out.max(dim=1)[0]
        return self.fc(h)


class GraphTransformerModel(nn.Module):
    """Graph Transformer: multi-head attention masked by adjacency."""

    def __init__(self, in_dim=8, hidden=16, n_heads=2, out_dim=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.norm1 = nn.LayerNorm(in_dim)
        self.q = nn.Linear(in_dim, hidden)
        self.k = nn.Linear(in_dim, hidden)
        self.v = nn.Linear(in_dim, hidden)
        self.out_proj = nn.Linear(hidden, in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        N = x.shape[0]
        h = self.norm1(x)
        q = self.q(h).view(N, self.n_heads, self.head_dim)
        k = self.k(h).view(N, self.n_heads, self.head_dim)
        v = self.v(h).view(N, self.n_heads, self.head_dim)
        attn = torch.einsum("ihd,jhd->ijh", q, k) / (self.head_dim**0.5)
        attn = attn.masked_fill(adj.unsqueeze(-1) == 0, float("-inf"))
        attn = torch.softmax(attn, dim=1)
        attn = attn.masked_fill(torch.isnan(attn), 0)
        out = torch.einsum("ijh,jhd->ihd", attn, v).reshape(N, -1)
        x = x + self.out_proj(out)
        x = x + self.ffn(self.norm2(x))
        return self.fc(x)


# =============================================================================
# Group R: Additional Computational Patterns
# =============================================================================


class MoEModel(nn.Module):
    """Mixture of Experts with top-2 routing."""

    def __init__(self, dim=16, n_experts=4):
        super().__init__()
        self.gate = nn.Linear(dim, n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, dim))
                for _ in range(n_experts)
            ]
        )
        self.out = nn.Linear(dim, 4)

    def forward(self, x):
        scores = torch.softmax(self.gate(x), dim=-1)
        topk_vals, topk_idx = scores.topk(2, dim=-1)
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        result = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i).any(dim=-1)
            if mask.any():
                expert_out = expert(x)
                weight = (topk_vals * (topk_idx == i).float()).sum(dim=-1)
                result = result + weight.unsqueeze(-1) * expert_out
        return self.out(result)


class SpatialTransformerNet(nn.Module):
    """Learned affine transform on feature maps (Jaderberg et al.)."""

    def __init__(self):
        super().__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.fc_loc = nn.Sequential(nn.Linear(16 * 4 * 4, 32), nn.ReLU(), nn.Linear(32, 6))
        self.classifier = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, 10),
        )
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        theta = self.fc_loc(self.localization(x).flatten(1)).view(-1, 2, 3)
        grid = nn.functional.affine_grid(theta, x.size(), align_corners=False)
        x = nn.functional.grid_sample(x, grid, align_corners=False)
        return self.classifier(x)


class DuelingDQN(nn.Module):
    """Dueling DQN: separate value and advantage streams."""

    def __init__(self, obs_dim=8, n_actions=4):
        super().__init__()
        self.features = nn.Sequential(nn.Linear(obs_dim, 32), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        self.advantage = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, n_actions))

    def forward(self, x):
        h = self.features(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=-1, keepdim=True)


class RMSNormModel(nn.Module):
    """RMS Normalization (LLaMA/Mistral): normalize by RMS, no mean centering."""

    def __init__(self, dim=16):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.fc1 = nn.Linear(dim, 32)
        self.fc2 = nn.Linear(32, dim)

    def _rms_norm(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return x / rms * self.weight

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(self._rms_norm(x))))


class SparsePrunedModel(nn.Module):
    """Structured pruning: binary masks on weight matrices."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 8)
        self.register_buffer("mask1", (torch.rand(32, 16) > 0.5).float())
        self.register_buffer("mask2", (torch.rand(8, 32) > 0.5).float())

    def forward(self, x):
        w1 = self.fc1.weight * self.mask1
        h = torch.relu(nn.functional.linear(x, w1, self.fc1.bias))
        w2 = self.fc2.weight * self.mask2
        return nn.functional.linear(h, w2, self.fc2.bias)


class FourierMixingModel(nn.Module):
    """FNet-style: FFT token mixing replaces attention."""

    def __init__(self, dim=16):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim))
        self.out = nn.Linear(dim, 4)

    def forward(self, x):
        x = self.embed(x)
        h = torch.fft.fft(torch.fft.fft(x, dim=-1), dim=-2).real
        x = self.norm1(x + h)
        x = self.norm2(x + self.ffn(x))
        return self.out(x.mean(dim=1))


# ── Group S: Gap-fill models ─────────────────────────────────────────────


class LeNet5(nn.Module):
    """The original CNN (LeCun 1998). Pure sequential conv→pool→conv→pool→FC."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.functional.avg_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = nn.functional.avg_pool2d(x, 2)
        x = x.flatten(1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM — forward + backward, concatenated outputs."""

    def __init__(self, input_dim=8, hidden_dim=16, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class Seq2SeqWithAttention(nn.Module):
    """Encoder-decoder with Bahdanau (additive) attention."""

    def __init__(self, input_dim=8, hidden_dim=16, output_dim=4, seq_len=6):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder_cell = nn.LSTMCell(output_dim + hidden_dim, hidden_dim)
        self.attn_W = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.seq_len = seq_len

    def forward(self, x):
        enc_out, (h, c) = self.encoder(x)
        h, c = h.squeeze(0), c.squeeze(0)
        batch = x.size(0)
        dec_input = torch.zeros(batch, self.out_proj.out_features, device=x.device)
        outputs = []
        for _ in range(self.seq_len):
            h_expand = h.unsqueeze(1).expand_as(enc_out)
            energy = torch.tanh(self.attn_W(torch.cat([enc_out, h_expand], dim=-1)))
            scores = self.attn_v(energy).squeeze(-1)
            weights = torch.softmax(scores, dim=-1)
            context = (weights.unsqueeze(-1) * enc_out).sum(dim=1)
            h, c = self.decoder_cell(torch.cat([dec_input, context], dim=-1), (h, c))
            dec_input = self.out_proj(h)
            outputs.append(dec_input)
        return torch.stack(outputs, dim=1)


class TripletNetwork(nn.Module):
    """Three shared-weight paths: anchor, positive, negative for metric learning."""

    def __init__(self, input_dim=16, embed_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, embed_dim))

    def forward(self, anchor, positive, negative):
        e_a = self.encoder(anchor)
        e_p = self.encoder(positive)
        e_n = self.encoder(negative)
        return e_a, e_p, e_n


class BarlowTwinsModel(nn.Module):
    """Twin encoders with cross-correlation redundancy reduction."""

    def __init__(self, input_dim=16, proj_dim=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, proj_dim))

    def forward(self, x1, x2):
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
        cross_corr = (z1_norm.T @ z2_norm) / z1.size(0)
        return cross_corr


class DeepCrossNetwork(nn.Module):
    """Explicit feature crossing layers + deep MLP (recommender pattern)."""

    def __init__(self, input_dim=16, num_cross_layers=3, deep_dim=32):
        super().__init__()
        self.cross_weights = nn.ParameterList(
            [nn.Parameter(torch.randn(input_dim)) for _ in range(num_cross_layers)]
        )
        self.cross_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(input_dim)) for _ in range(num_cross_layers)]
        )
        self.deep = nn.Sequential(
            nn.Linear(input_dim, deep_dim),
            nn.ReLU(),
            nn.Linear(deep_dim, deep_dim),
            nn.ReLU(),
        )
        self.final = nn.Linear(input_dim + deep_dim, 1)

    def forward(self, x):
        x0 = x
        x_cross = x
        for w, b in zip(self.cross_weights, self.cross_biases):
            x_cross = x0 * (x_cross * w).sum(dim=-1, keepdim=True) + b + x_cross
        x_deep = self.deep(x)
        return self.final(torch.cat([x_cross, x_deep], dim=-1))


class AxialAttentionModel(nn.Module):
    """Decomposed attention: height-axis then width-axis (sequential)."""

    def __init__(self, dim=16, heads=2):
        super().__init__()
        self.embed = nn.Conv2d(3, dim, 1)
        self.height_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.width_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.out = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim, 4)

    def forward(self, x):
        x = self.embed(x)
        B, C, H, W = x.shape
        # Height-axis attention: (B*W, H, C)
        xh = x.permute(0, 3, 2, 1).reshape(B * W, H, C)
        xh = self.norm1(xh + self.height_attn(xh, xh, xh)[0])
        xh = xh.reshape(B, W, H, C).permute(0, 3, 2, 1)
        # Width-axis attention: (B*H, W, C)
        xw = xh.permute(0, 2, 3, 1).reshape(B * H, W, C)
        xw = self.norm2(xw + self.width_attn(xw, xw, xw)[0])
        xw = xw.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.fc(self.out(xw).flatten(1))


class CBAMBlock(nn.Module):
    """CBAM: channel attention (pool→FC→sigmoid) + spatial attention (pool→conv→sigmoid)."""

    def __init__(self, channels=16, reduction=4):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 3, padding=1)
        # Channel attention
        self.ca_fc1 = nn.Linear(channels, channels // reduction)
        self.ca_fc2 = nn.Linear(channels // reduction, channels)
        # Spatial attention
        self.sa_conv = nn.Conv2d(2, 1, 7, padding=3)
        self.out = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 4)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        # Channel attention
        avg_pool = x.mean(dim=[2, 3])
        max_pool = x.amax(dim=[2, 3])
        ca = torch.sigmoid(
            self.ca_fc2(torch.relu(self.ca_fc1(avg_pool)))
            + self.ca_fc2(torch.relu(self.ca_fc1(max_pool)))
        )
        x = x * ca.unsqueeze(-1).unsqueeze(-1)
        # Spatial attention
        sa_avg = x.mean(dim=1, keepdim=True)
        sa_max = x.amax(dim=1, keepdim=True)
        sa = torch.sigmoid(self.sa_conv(torch.cat([sa_avg, sa_max], dim=1)))
        x = x * sa
        return self.fc(self.out(x).flatten(1))


# =============================================================================
# Group T: Additional Pattern Coverage
# =============================================================================


class GRUModel(nn.Module):
    """Simple GRU forward pass."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=8, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 4)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class NiNModel(nn.Module):
    """Network in Network: 1x1 conv (mlpconv) + global average pooling."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.mlpconv1 = nn.Conv2d(16, 16, 1)  # 1x1 conv
        self.mlpconv2 = nn.Conv2d(16, 16, 1)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.mlpconv3 = nn.Conv2d(8, 4, 1)  # final 1x1 → num_classes channels

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.mlpconv1(x))
        x = torch.relu(self.mlpconv2(x))
        x = torch.relu(self.conv2(x))
        x = self.mlpconv3(x)
        x = x.mean(dim=[2, 3])  # global average pooling
        return x


class ChannelShuffleModel(nn.Module):
    """Channel shuffle operation (ShuffleNet-style)."""

    def __init__(self, groups=2):
        super().__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, groups=1)
        self.gconv = nn.Conv2d(16, 16, 1, groups=groups)
        self.conv2 = nn.Conv2d(16, 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.gconv(x)
        # Channel shuffle
        B, C, H, W = x.shape
        x = x.view(B, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, C, H, W)
        x = torch.relu(x)
        return (
            self.conv2(self.pool(x).flatten(1, -1).unsqueeze(-1).unsqueeze(-1))
            .squeeze(-1)
            .squeeze(-1)
        )


class PixelShuffleModel(nn.Module):
    """Sub-pixel convolution upsampling (PixelShuffle)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)  # 16 * 4 for 2x upscale
        self.shuffle = nn.PixelShuffle(2)  # 64 → 16 channels, 2x spatial
        self.conv3 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.shuffle(x)
        x = self.conv3(x)
        return self.pool(x).flatten(1)


class PartialConvModel(nn.Module):
    """Partial convolution: mask-aware conv (inpainting pattern)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.mask_conv_weight = nn.Parameter(torch.ones(16, 3, 3, 3) / 27.0, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(16))
        self.fc = nn.Linear(16, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Simulate partial convolution: apply conv only where mask=1
        mask = (x.sum(dim=1, keepdim=True) > 0).float()
        mask3 = mask.expand_as(x)
        x_masked = x * mask3
        out = self.conv(x_masked)
        # Update mask: count valid pixels in each receptive field
        with torch.no_grad():
            mask_sum = torch.nn.functional.conv2d(
                mask, torch.ones(1, 1, 3, 3, device=x.device), padding=1
            )
            mask_ratio = 9.0 / (mask_sum + 1e-8)
            new_mask = (mask_sum > 0).float()
        out = out * mask_ratio * new_mask + self.bias.view(1, -1, 1, 1)
        return self.fc(self.pool(torch.relu(out)).flatten(1))


class FiLMModel(nn.Module):
    """Feature-wise Linear Modulation: external signal conditions features."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        # Conditioning network: produces scale (gamma) and shift (beta)
        self.cond_fc = nn.Linear(8, 32)  # 8-dim conditioning → 16 gamma + 16 beta
        self.conv2 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, cond):
        feat = torch.relu(self.conv(x))
        # Generate FiLM parameters from conditioning signal
        film_params = self.cond_fc(cond)
        gamma = film_params[:, :16].unsqueeze(-1).unsqueeze(-1)
        beta = film_params[:, 16:].unsqueeze(-1).unsqueeze(-1)
        # Apply FiLM: scale and shift
        feat = gamma * feat + beta
        return self.pool(self.conv2(torch.relu(feat))).flatten(1)


class CoordinateAttentionModel(nn.Module):
    """Coordinate attention: factorized H/W pooling for spatial attention."""

    def __init__(self, channels=16, reduction=4):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 3, padding=1)
        mid = max(channels // reduction, 4)
        self.fc_shared = nn.Conv2d(channels, mid, 1)
        self.fc_h = nn.Conv2d(mid, channels, 1)
        self.fc_w = nn.Conv2d(mid, channels, 1)
        self.out = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 4)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        B, C, H, W = x.shape
        # Pool along W → (B, C, H, 1), pool along H → (B, C, 1, W)
        h_pool = x.mean(dim=3, keepdim=True)  # (B, C, H, 1)
        w_pool = x.mean(dim=2, keepdim=True)  # (B, C, 1, W)
        # Concatenate along spatial dim, share FC
        h_pool_t = h_pool.permute(0, 1, 3, 2)  # (B, C, 1, H)
        cat = torch.cat([w_pool, h_pool_t], dim=3)  # (B, C, 1, W+H)
        cat = torch.relu(self.fc_shared(cat))
        w_attn, h_attn = cat.split([W, H], dim=3)
        w_attn = torch.sigmoid(self.fc_w(w_attn))  # (B, C, 1, W)
        h_attn = torch.sigmoid(self.fc_h(h_attn.permute(0, 1, 3, 2)))  # (B, C, H, 1)
        x = x * w_attn * h_attn
        return self.fc(self.out(x).flatten(1))


class DifferentialAttentionModel(nn.Module):
    """Differential attention: subtract two attention patterns for noise cancellation."""

    def __init__(self, dim=16, num_heads=2):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.q1 = nn.Linear(dim, dim)
        self.k1 = nn.Linear(dim, dim)
        self.q2 = nn.Linear(dim, dim)
        self.k2 = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.scale = dim**-0.5
        self.lambda_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x = self.embed(x)
        q1, k1 = self.q1(x), self.k1(x)
        q2, k2 = self.q2(x), self.k2(x)
        v = self.v(x)
        attn1 = torch.softmax(q1 @ k1.transpose(-2, -1) * self.scale, dim=-1)
        attn2 = torch.softmax(q2 @ k2.transpose(-2, -1) * self.scale, dim=-1)
        # Differential: subtract attention patterns
        diff_attn = attn1 - self.lambda_param * attn2
        out = diff_attn @ v
        return self.out(out)


class RelativePositionAttentionModel(nn.Module):
    """T5-style relative position bias in self-attention."""

    def __init__(self, dim=16, num_heads=2, max_len=32, num_buckets=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.rel_bias = nn.Embedding(num_buckets, num_heads)
        self.num_buckets = num_buckets
        self.scale = self.head_dim**-0.5

    def _relative_position_bucket(self, rel_pos):
        return torch.clamp(rel_pos + self.num_buckets // 2, 0, self.num_buckets - 1)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = q @ k.transpose(-2, -1) * self.scale
        # Relative position bias
        pos = torch.arange(S, device=x.device)
        rel_pos = pos.unsqueeze(0) - pos.unsqueeze(1)
        buckets = self._relative_position_bucket(rel_pos)
        bias = self.rel_bias(buckets).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out(out)


class EarlyExitModel(nn.Module):
    """Early exit: multiple classifier heads at different depths."""

    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.block2 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.block3 = nn.Sequential(nn.Linear(32, 32), nn.ReLU())
        self.exit1 = nn.Linear(32, 4)
        self.exit2 = nn.Linear(32, 4)
        self.exit3 = nn.Linear(32, 4)

    def forward(self, x):
        x1 = self.block1(x)
        out1 = self.exit1(x1)
        x2 = self.block2(x1)
        out2 = self.exit2(x2)
        x3 = self.block3(x2)
        out3 = self.exit3(x3)
        return out1, out2, out3


class MultiScaleParallelModel(nn.Module):
    """HRNet-like parallel multi-resolution streams with fusion."""

    def __init__(self):
        super().__init__()
        # High-res stream
        self.high_conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.high_conv2 = nn.Conv2d(8, 8, 3, padding=1)
        # Low-res stream (downsampled)
        self.low_conv1 = nn.Conv2d(3, 16, 3, stride=2, padding=1)
        self.low_conv2 = nn.Conv2d(16, 16, 3, padding=1)
        # Fusion: low→high (upsample + 1x1), high→low (stride + 1x1)
        self.low_to_high = nn.Conv2d(16, 8, 1)
        self.high_to_low = nn.Conv2d(8, 16, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8 + 16, 4)

    def forward(self, x):
        hi = torch.relu(self.high_conv1(x))
        lo = torch.relu(self.low_conv1(x))
        hi = torch.relu(self.high_conv2(hi))
        lo = torch.relu(self.low_conv2(lo))
        # Cross-resolution fusion
        lo_up = torch.nn.functional.interpolate(
            self.low_to_high(lo), size=hi.shape[2:], mode="nearest"
        )
        h_down = self.high_to_low(torch.nn.functional.avg_pool2d(hi, 2))
        hi = hi + lo_up
        lo = lo + h_down
        h_pool = self.pool(hi).flatten(1)
        l_pool = self.pool(lo).flatten(1)
        return self.fc(torch.cat([h_pool, l_pool], dim=1))


class GumbelVQModel(nn.Module):
    """Gumbel-Softmax vector quantization (differentiable discrete tokens)."""

    def __init__(self, input_dim=16, codebook_size=8, embed_dim=8):
        super().__init__()
        self.encoder = nn.Linear(input_dim, codebook_size)
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        self.decoder = nn.Linear(embed_dim, input_dim)

    def forward(self, x):
        logits = self.encoder(x)
        # Gumbel-Softmax: differentiable discrete sampling
        soft = torch.nn.functional.gumbel_softmax(logits, tau=1.0, hard=True)
        # Lookup: weighted sum of codebook entries
        quantized = soft @ self.codebook.weight
        return self.decoder(quantized)


class EndToEndMemoryNetwork(nn.Module):
    """End-to-end memory network: multi-hop attention over memory slots."""

    def __init__(self, vocab_size=32, embed_dim=16, mem_slots=8, hops=3):
        super().__init__()
        self.hops = hops
        self.embed_a = nn.Embedding(vocab_size, embed_dim)
        self.embed_c = nn.Embedding(vocab_size, embed_dim)
        self.query_embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, 4)

    def forward(self, query, memory):
        # query: (B,) int, memory: (B, M) int
        u = self.query_embed(query)  # (B, D)
        for _ in range(self.hops):
            m = self.embed_a(memory)  # (B, M, D)
            p = torch.softmax((u.unsqueeze(1) * m).sum(-1), dim=-1)  # (B, M)
            c = self.embed_c(memory)  # (B, M, D)
            o = (p.unsqueeze(-1) * c).sum(1)  # (B, D)
            u = u + o
        return self.fc(u)


class RBFNetwork(nn.Module):
    """Radial Basis Function network: Gaussian kernel hidden layer."""

    def __init__(self, input_dim=8, num_centers=16, output_dim=4):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_centers, input_dim))
        self.log_sigmas = nn.Parameter(torch.zeros(num_centers))
        self.fc = nn.Linear(num_centers, output_dim)

    def forward(self, x):
        # Compute RBF activations: exp(-||x - c||^2 / (2*sigma^2))
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)  # (B, K, D)
        dist_sq = (diff**2).sum(-1)  # (B, K)
        sigma_sq = torch.exp(self.log_sigmas) ** 2 * 2
        rbf = torch.exp(-dist_sq / (sigma_sq + 1e-8))
        return self.fc(rbf)


class SIRENModel(nn.Module):
    """SIREN: sinusoidal activations for coordinate-based MLP."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self.omega = 30.0

    def forward(self, coords):
        x = torch.sin(self.omega * self.fc1(coords))
        x = torch.sin(self.omega * self.fc2(x))
        return self.fc3(x)


class MultiTaskModel(nn.Module):
    """Multi-task: shared trunk + task-specific heads."""

    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.head_cls = nn.Linear(32, 4)  # classification head
        self.head_reg = nn.Linear(32, 1)  # regression head
        self.head_embed = nn.Linear(32, 8)  # embedding head

    def forward(self, x):
        feat = self.shared(x)
        return self.head_cls(feat), self.head_reg(feat), self.head_embed(feat)


class WideAndDeepModel(nn.Module):
    """Wide & Deep: wide linear + deep MLP merged."""

    def __init__(self, input_dim=16):
        super().__init__()
        # Wide: linear path
        self.wide = nn.Linear(input_dim, 4)
        # Deep: MLP path
        self.deep = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        return self.wide(x) + self.deep(x)


class ChebGCN(nn.Module):
    """Chebyshev polynomial spectral GCN (manual implementation)."""

    def __init__(self, in_features=8, hidden=16, out_features=4, K=3):
        super().__init__()
        self.K = K
        self.theta = nn.ParameterList(
            [nn.Parameter(torch.randn(in_features, hidden)) for _ in range(K)]
        )
        self.fc = nn.Linear(hidden, out_features)

    def forward(self, x, adj):
        # Chebyshev polynomial: T0=I, T1=adj, Tk=2*adj*T_{k-1} - T_{k-2}
        T = [x]  # T0(adj) @ x = x
        if self.K > 1:
            T.append(adj @ x)  # T1(adj) @ x
        for k in range(2, self.K):
            T.append(2 * adj @ T[-1] - T[-2])
        out = sum(T[k] @ self.theta[k] for k in range(self.K))
        return self.fc(torch.relu(out))


class PrototypicalNetwork(nn.Module):
    """Prototypical network: compute class prototypes, classify by distance."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 16))

    def forward(self, support, support_labels, query):
        # support: (N*K, D), support_labels: (N*K,), query: (Q, D)
        s_embed = self.encoder(support)
        q_embed = self.encoder(query)
        # Compute prototypes per class
        classes = support_labels.unique()
        prototypes = torch.stack([s_embed[support_labels == c].mean(0) for c in classes])
        # Negative squared distance
        dists = -(torch.cdist(q_embed, prototypes) ** 2)
        return dists


class ECAModel(nn.Module):
    """Efficient Channel Attention: 1D conv instead of FC for channel attention."""

    def __init__(self, channels=16, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(3, channels, 3, padding=1)
        self.eca = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 4)

    def forward(self, x):
        x = torch.relu(self.conv(x))
        # ECA: GAP → 1D conv on channel dim → sigmoid → scale
        attn = self.pool(x).squeeze(-1).squeeze(-1)  # (B, C)
        attn = self.eca(attn.unsqueeze(1)).squeeze(1)  # (B, C)
        attn = torch.sigmoid(attn).unsqueeze(-1).unsqueeze(-1)
        x = x * attn
        return self.fc(self.pool(x).flatten(1))


# ============================================================================
# Group U: Final Coverage — Novel Computational Patterns
# ============================================================================


class LinearAttentionModel(nn.Module):
    """Kernel-based linear attention: phi(Q) @ (phi(K)^T @ V).

    Avoids softmax entirely — O(N*d^2) instead of O(N^2*d).
    Tests the associative-reorder pattern that is fundamentally different
    from standard scaled-dot-product attention.
    """

    def __init__(self, dim=16, heads=2, seq_len=8):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, 4)

    def forward(self, x):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, S, heads, head_dim)
        q = q.transpose(1, 2)  # (B, heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Kernel feature map: elu(x) + 1 (positive features)
        q = torch.nn.functional.elu(q) + 1
        k = torch.nn.functional.elu(k) + 1
        # Linear attention: phi(Q) @ (phi(K)^T @ V) — RIGHT associativity
        kv = torch.matmul(k.transpose(-2, -1), v)  # (B, h, d, d)
        out = torch.matmul(q, kv)  # (B, h, S, d)
        # Normalize by sum of keys
        k_sum = k.sum(dim=-2, keepdim=True)  # (B, h, 1, d)
        denom = torch.matmul(q, k_sum.transpose(-2, -1)).clamp(min=1e-6)
        out = out / denom
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out(out.mean(dim=1))


class SimpleFNO(nn.Module):
    """Fourier Neural Operator: FFT -> spectral weights -> iFFT.

    Tests the spectral convolution pattern: transform to frequency domain,
    multiply by learned complex weights, transform back. Fundamentally
    different from spatial convolution.
    """

    def __init__(self, in_channels=3, width=8, modes=4):
        super().__init__()
        self.modes = modes
        self.lift = nn.Linear(in_channels, width)
        # Complex spectral weights (learnable Fourier coefficients)
        self.spectral_weight = nn.Parameter(
            torch.randn(width, width, modes, dtype=torch.cfloat) * 0.02
        )
        self.project = nn.Linear(width, 4)

    def forward(self, x):
        # x: (B, N, in_channels) — 1D function on N grid points
        x = self.lift(x)  # (B, N, width)
        x_ft = torch.fft.rfft(x, dim=1)  # (B, N//2+1, width)
        # Multiply only the first `modes` frequencies by learned weights
        out_ft = torch.zeros_like(x_ft)
        modes = min(self.modes, x_ft.shape[1])
        out_ft[:, :modes, :] = torch.einsum(
            "bmi,iom->bmo", x_ft[:, :modes, :], self.spectral_weight[:, :, :modes]
        )
        # Back to physical space
        x = torch.fft.irfft(out_ft, n=x.shape[1], dim=1)  # (B, N, width)
        x = torch.relu(x)
        return self.project(x.mean(dim=1))


class PerceiverModel(nn.Module):
    """Perceiver: cross-attention from input->latent, then self-attention on latents.

    Tests the asymmetric cross-attention bottleneck pattern: arbitrary-length
    input is compressed into a fixed-size latent array, then processed
    with self-attention. Distinct from encoder-decoder cross-attention
    because latent is LEARNED, not derived from input.
    """

    def __init__(self, input_dim=16, latent_dim=8, num_latents=4, heads=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, latent_dim))
        # Cross-attention: Q from latent, K/V from input
        self.cross_q = nn.Linear(latent_dim, latent_dim)
        self.cross_kv = nn.Linear(input_dim, latent_dim * 2)
        # Self-attention on latents
        self.self_attn = nn.MultiheadAttention(latent_dim, heads, batch_first=True)
        self.ff = nn.Linear(latent_dim, latent_dim)
        self.out = nn.Linear(latent_dim, 4)
        self.scale = latent_dim**-0.5

    def forward(self, x):
        B = x.shape[0]
        latents = self.latents.expand(B, -1, -1)
        # Cross-attention: latents attend to input
        q = self.cross_q(latents)
        kv = self.cross_kv(x)
        k, v = kv.chunk(2, dim=-1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        latents = latents + torch.matmul(attn, v)
        # Self-attention on latents
        sa_out, _ = self.self_attn(latents, latents, latents)
        latents = latents + sa_out
        latents = latents + torch.relu(self.ff(latents))
        return self.out(latents.mean(dim=1))


class ASPPModel(nn.Module):
    """Atrous Spatial Pyramid Pooling: parallel dilated convs at multiple rates.

    Tests multi-rate parallel branches — each branch sees the same input
    through a different dilation rate, then all branches are concatenated.
    Used in DeepLab for multi-scale context aggregation.
    """

    def __init__(self, in_channels=3, mid=8, rates=(1, 6, 12)):
        super().__init__()
        self.branches = nn.ModuleList(
            [nn.Conv2d(in_channels, mid, 3, padding=r, dilation=r) for r in rates]
        )
        # Global average pooling branch
        self.gap_conv = nn.Conv2d(in_channels, mid, 1)
        self.fuse = nn.Conv2d(mid * (len(rates) + 1), 4, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        branch_outs = [torch.relu(b(x)) for b in self.branches]
        # GAP branch: pool -> 1x1 conv -> upsample
        gap = self.pool(x)
        gap = torch.relu(self.gap_conv(gap))
        gap = torch.nn.functional.interpolate(
            gap, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        branch_outs.append(gap)
        fused = torch.cat(branch_outs, dim=1)
        return self.fuse(fused).mean(dim=(2, 3))


class ControlNetModel(nn.Module):
    """ControlNet pattern: frozen encoder + trainable copy + zero-conv injection.

    Tests the parallel-copy-with-injection pattern: a trainable copy of an
    encoder processes conditioning, then its features are injected into the
    frozen encoder via zero-initialized convolutions. The zero init ensures
    training starts from the pretrained model's behavior.
    """

    def __init__(self, channels=8):
        super().__init__()
        # "Frozen" main encoder (we just don't update it, structurally same)
        self.main_conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.main_conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.main_fc = nn.Linear(channels, 4)
        # Trainable copy (ControlNet branch)
        self.ctrl_conv1 = nn.Conv2d(3, channels, 3, padding=1)
        self.ctrl_conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        # Zero convolutions for injection (initialized to zero)
        self.zero_conv1 = nn.Conv2d(channels, channels, 1)
        self.zero_conv2 = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.zero_conv1.weight)
        nn.init.zeros_(self.zero_conv1.bias)
        nn.init.zeros_(self.zero_conv2.weight)
        nn.init.zeros_(self.zero_conv2.bias)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Main path
        h1 = torch.relu(self.main_conv1(x))
        # ControlNet path (parallel copy)
        c1 = torch.relu(self.ctrl_conv1(x))
        # Inject via zero conv
        h1 = h1 + self.zero_conv1(c1)
        h2 = torch.relu(self.main_conv2(h1))
        c2 = torch.relu(self.ctrl_conv2(c1))
        h2 = h2 + self.zero_conv2(c2)
        return self.main_fc(self.pool(h2).flatten(1))


class SimpleEGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network: message passing with coordinate updates.

    Tests the equivariant pattern where both node features AND spatial coordinates
    are updated during message passing. Messages depend on distances (invariant),
    and coordinate updates are weighted by relative positions (equivariant).
    Takes TWO inputs: (features, coordinates).
    """

    def __init__(self, node_dim=8, hidden_dim=16, out_dim=4):
        super().__init__()
        # Edge MLP: maps (hi, hj, ||xi-xj||^2) -> message
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        # Coordinate update weight (scalar per edge)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 1),
        )
        self.out = nn.Linear(node_dim, out_dim)

    def forward(self, h, x):
        # h: (B, N, node_dim) node features
        # x: (B, N, 3) coordinates
        B, N, _ = h.shape
        # All-pairs: compute messages for complete graph
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        xi = x.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, 3)
        xj = x.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, 3)
        # Squared distances (E(n) invariant)
        diff = xi - xj  # (B, N, N, 3)
        dist_sq = (diff**2).sum(dim=-1, keepdim=True)  # (B, N, N, 1)
        # Edge messages
        edge_input = torch.cat([hi, hj, dist_sq], dim=-1)
        msg = self.edge_mlp(edge_input)  # (B, N, N, hidden)
        # Aggregate messages (sum over neighbors)
        agg = msg.sum(dim=2)  # (B, N, hidden)
        # Update node features
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        # Update coordinates (equivariant: weighted sum of relative positions)
        coord_weights = self.coord_mlp(msg).squeeze(-1)  # (B, N, N)
        # x_update = sum_j w_ij * (x_i - x_j)
        x_update = (diff * coord_weights.unsqueeze(-1)).sum(dim=2)  # (B, N, 3)
        x = x + x_update
        # Readout: mean pool features
        return self.out(h.mean(dim=1))


class MAMLInnerLoop(nn.Module):
    """MAML-style higher-order gradients: gradient computation within forward pass.

    Tests the pattern where torch.autograd.grad is called INSIDE the forward
    method to compute an inner-loop gradient update, and the outer loss
    backpropagates THROUGH that inner gradient. This is the "gradient of
    gradient" pattern — the most demanding test for activation tracking.
    """

    def __init__(self, in_dim=8, hidden=16, out_dim=4):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(in_dim, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.randn(hidden, out_dim) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        # "Support set" forward pass (inner loop)
        h = torch.relu(x @ self.w1 + self.b1)
        support_out = h @ self.w2 + self.b2
        # Inner loss (MSE to zero — dummy task)
        inner_loss = (support_out**2).mean()
        # Inner gradient step (differentiable!)
        grads = torch.autograd.grad(
            inner_loss, [self.w1, self.b1, self.w2, self.b2], create_graph=True
        )
        # Fast weights (one gradient step)
        w1_fast = self.w1 - 0.01 * grads[0]
        b1_fast = self.b1 - 0.01 * grads[1]
        w2_fast = self.w2 - 0.01 * grads[2]
        b2_fast = self.b2 - 0.01 * grads[3]
        # "Query set" forward with fast weights (outer loop)
        h2 = torch.relu(x @ w1_fast + b1_fast)
        return h2 @ w2_fast + b2_fast


class TinyNeRF(nn.Module):
    """Minimal differentiable volumetric renderer (NeRF-style).

    Tests the differentiable rendering pattern: sample points along rays,
    query an MLP for (density, color), then composite via volume rendering
    equation (alpha compositing). The ray marching loop with learned
    density/color is unique to neural radiance fields.
    """

    def __init__(self, pos_dim=3, hidden=32, num_samples=8):
        super().__init__()
        self.num_samples = num_samples
        # Simple MLP: position -> (density, RGB)
        self.net = nn.Sequential(
            nn.Linear(pos_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.density_head = nn.Linear(hidden, 1)
        self.color_head = nn.Linear(hidden, 3)

    def forward(self, ray_origins, ray_dirs):
        # ray_origins: (B, 3), ray_dirs: (B, 3)
        B = ray_origins.shape[0]
        # Sample points along each ray
        t_vals = torch.linspace(0.0, 1.0, self.num_samples, device=ray_origins.device)
        t_vals = t_vals.unsqueeze(0).expand(B, -1)  # (B, num_samples)
        # Points: o + t*d for each sample
        pts = ray_origins.unsqueeze(1) + t_vals.unsqueeze(-1) * ray_dirs.unsqueeze(1)  # (B, S, 3)
        # Query MLP for all points
        feats = self.net(pts.reshape(B * self.num_samples, -1))
        raw_density = self.density_head(feats).reshape(B, self.num_samples)
        raw_color = self.color_head(feats).reshape(B, self.num_samples, 3)
        # Volume rendering (alpha compositing)
        density = torch.relu(raw_density)
        color = torch.sigmoid(raw_color)
        # Spacing between samples
        dists = t_vals[:, 1:] - t_vals[:, :-1]
        dists = torch.cat([dists, torch.ones(B, 1, device=dists.device) * 1e-3], dim=-1)
        # Alpha = 1 - exp(-density * dist)
        alpha = 1.0 - torch.exp(-density * dists)
        # Transmittance: cumulative product of (1 - alpha)
        transmittance = torch.cumprod(
            torch.cat([torch.ones(B, 1, device=alpha.device), 1.0 - alpha[:, :-1]], dim=-1),
            dim=-1,
        )
        # Weighted sum of colors
        weights = alpha * transmittance  # (B, S)
        rgb = (weights.unsqueeze(-1) * color).sum(dim=1)  # (B, 3)
        return rgb


# **********************************************
# **** Random Graph Model (for scale tests) ****
# **********************************************


class _SequentialBlock(nn.Module):
    """~3 TorchLens nodes: Linear -> ReLU -> Linear."""

    APPROX_NODES = 3

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _ResidualBlock(nn.Module):
    """~4 TorchLens nodes: Linear -> ReLU -> Linear + skip."""

    APPROX_NODES = 4

    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))) + x


class _BranchMergeBlock(nn.Module):
    """~8 TorchLens nodes: input -> [branch1, branch2, branch3] -> cat -> Linear."""

    APPROX_NODES = 8

    def __init__(self, dim):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.branch3 = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.merge = nn.Linear(dim * 3, dim)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        return self.merge(torch.cat([b1, b2, b3], dim=-1))


class _AttentionBlock(nn.Module):
    """~9 TorchLens nodes: Q/K/V projections -> matmul -> softmax -> matmul -> out."""

    APPROX_NODES = 9

    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.scale = dim**0.5

    def forward(self, x):
        # Treat batch dim as sequence for simplicity.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)


_BLOCK_TYPES = [_SequentialBlock, _ResidualBlock, _BranchMergeBlock, _AttentionBlock]


class RandomGraphModel(nn.Module):
    """Programmatically generated model with controlled node count.

    Args:
        target_nodes: Approximate number of TorchLens nodes to generate.
        nesting_depth: Levels of nn.Module nesting (1-5).
        seed: RNG seed for deterministic structure.
        branch_probability: Fraction of blocks that use branch/attention types.
        hidden_dim: Hidden dimension for all layers.
    """

    def __init__(
        self,
        target_nodes: int = 1000,
        nesting_depth: int = 2,
        seed: int = 42,
        branch_probability: float = 0.3,
        hidden_dim: int = 64,
    ):
        super().__init__()
        import random as _random

        rng = _random.Random(seed)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)

        # Compute how many blocks we need.
        # Weighted average: branch_probability controls mix of complex vs simple blocks.
        simple_avg = (_SequentialBlock.APPROX_NODES + _ResidualBlock.APPROX_NODES) / 2
        complex_avg = (_BranchMergeBlock.APPROX_NODES + _AttentionBlock.APPROX_NODES) / 2
        avg_nodes_per_block = (
            1 - branch_probability
        ) * simple_avg + branch_probability * complex_avg
        # Subtract 2 fixed overhead nodes (model input + input_proj).
        effective_target = max(1, target_nodes - 2)
        num_blocks = max(1, round(effective_target / avg_nodes_per_block))

        # Choose block types.
        blocks = []
        for _ in range(num_blocks):
            if rng.random() < branch_probability:
                block_cls = rng.choice([_BranchMergeBlock, _AttentionBlock])
            else:
                block_cls = rng.choice([_SequentialBlock, _ResidualBlock])
            blocks.append(block_cls(hidden_dim))

        # Distribute across nesting depth using ModuleList wrappers.
        nesting_depth = max(1, min(nesting_depth, 5))
        if nesting_depth == 1:
            self.layers = nn.ModuleList(blocks)
        else:
            # Split blocks into groups and nest them.
            group_size = max(1, len(blocks) // nesting_depth)
            nested = nn.ModuleList()
            for i in range(0, len(blocks), group_size):
                group = nn.ModuleList(blocks[i : i + group_size])
                nested.append(group)
            self.layers = nested

        self.nesting_depth = nesting_depth

    def forward(self, x):
        x = self.input_proj(x)
        if self.nesting_depth == 1:
            for layer in self.layers:
                x = layer(x)
        else:
            for group in self.layers:
                for layer in group:
                    x = layer(x)
        return x
