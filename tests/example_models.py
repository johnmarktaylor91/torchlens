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
