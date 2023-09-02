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
        x5 = x4 * 2 + y1 + y2
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
        self.dropout = nn.Dropout(p=.5)
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
        x = x ** 2
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
        x = a ** b
        return x


class ListInput(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_list):
        x, y, z = input_list
        a = x + y
        b = torch.log(z)
        x = a ** b
        return x


class DictInput(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_dict):
        x, y, z = input_dict['x'], input_dict['y'], input_dict['z']
        a = x + y
        b = torch.log(z)
        x = a ** b
        return x


class NestedInput(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_dict):
        list1, list2 = input_dict['list1'], input_dict['list2']
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
        return {'a': a, 'b': b, 'c': c}


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
        return {'a': [a, b], 'b': c, 'c': [a, b, c], 'd': [[a, b]]}


class BufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('buffer1', torch.ones(12, 12))
        self.register_buffer('buffer2', torch.rand(12, 12))

    def forward(self, x):
        x = x + self.buffer1
        x = x ** (self.buffer2 * (3 + torch.rand(12, 12)))
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
        z = x ** 2
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
            x = x ** 2
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
        x = x ** 3 + torch.ones(x.shape)
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
        b = z ** 2
        c = a + b
        x = x ** 2
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
                y = y ** 2
            x = torch.sin(x)
            if i % 2 == 1:
                z = x + 3
                z = torch.cos(x)
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
                y = y ** 2
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
        x = x ** 3
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
            x = x + torch.rand(x.shape)
            if torch.mean(x) > 0:
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
        for _ in range(4):  # this tests clashes between what counts as "same"--module-based or looping-based
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
#  **** Uber Architectures ****
#  ****************************

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
        y = y ** 3
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
        v3 = v2.sum() > 5
        m = torch.ones(5)
        m1 = m * 2
        m2 = m + 3
        v = torch.cos(v)
        return u, v, y


class UberModel2(nn.Module):
    def __init__(self):
        """Conv, relu, pool, fc, output.

        """
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
        x = x ** 3
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
        """Conv, relu, pool, fc, output.

        """
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
        """Conv, relu, pool, fc, output.

        """
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
        """Conv, relu, pool, fc, output.

        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, padding=1)
        self.fc = nn.Linear(9, 3)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv1(x)
        x2 = x * 2
        x3 = x2 + 4
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
        w1 = x ** 3
        x = torch.sum(torch.stack([y4, z3, w1]))
        return x
