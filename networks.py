from torch import nn
import torch
from typing import Dict

torch.TEST_ATTRIBUTE = 'networks_script'

# TODO: devise a set of test networks to make all the different edgecases bulletproof.
# TODO: and somewhat overlapping: make some good demos

"""
Cases to test:

- Nested looping
- Complex branching 
- Internally generated and internally terminated tensors 
- Multiple inputs and outputs 
- Nested modules with stuff before and after 
"""


# For simplicity, let's make all inputs 3x3.

class SimpleFF(nn.Module):
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


class SimpleFFInternalFuncs(nn.Module):
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
        x = x + torch.ones(1, 1, 3, 3) + torch.rand(1, 1, 3, 3)
        y = torch.ones(1, 2, 3)
        y = y + 1
        x = self.identity(x)
        x = x.flatten()
        x = self.fc(x)
        return x


class SomeComplexInternalFuncs(nn.Module):
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
        y = torch.ones(1, 1, 3, 3) + torch.zeros(1, 1, 3, 3)
        z = torch.rand(1, 1, 3, 3) ** y
        x = x * z
        x = self.identity(x)
        x = x.flatten()
        x = self.fc(x)
        return x


class TwoWayBranching(nn.Module):
    def __init__(self):
        """Conv, relu, pool, fc, output.

        """
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.fc = nn.Linear(9, 3)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        y = x + 1
        y = y + 2
        z = x * 2
        x = nn.functional.relu(y ** z)
        return x


class ThreeWayBranching(nn.Module):
    def __init__(self):
        """Conv, relu, pool, fc, output.

        """
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.fc = nn.Linear(9, 3)
        self.identity = nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        a = x + 1
        b = x * 2
        b = b * 3
        c = x ** 2
        c = c ** 3
        c = c ** 4
        x = a + b + c
        x = nn.functional.relu(x)
        return x


class BranchingWithInternalFuncs(nn.Module):
    def __init__(self):
        """Conv, relu, pool, fc, output.

        """
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.fc = nn.Linear(9, 3)
        self.identity = nn.Identity()

    def forward(self, x):
        if torch.sum(x) <= 0:
            x = x * 2
            return x
        x = self.conv(x)
        y = x + 1
        z = x * 2
        b = torch.ones(1, 1, 3, 3)
        c = torch.zeros(1, 1, 3, 3)
        d = torch.rand(1, 1, 3, 3)
        e = b * c + d
        x = y + z + e
        x = nn.functional.relu(x)
        return x


class NestedComplexBranching(nn.Module):
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


class ConditionalBranching(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        if torch.sum(x) > 0:
            x = x + 1
            x = x + 1
            x = x + 1
        else:
            x = x * 2
            x = x * 2
            x = x * 2
        return x


class SimpleRecurrent(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        x = x + 1
        x = self.fc(x)
        x = x * 2
        x = self.fc(x)
        x = x ** 3
        x = self.fc(x)
        return x


class RecurrentSameFuncs(nn.Module):
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
        x = self.fc(x)
        return x


class RecurrentDiffFuncsSimple(nn.Module):
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


class RecurrentDiffFuncs(nn.Module):
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


class RecurrentSimpleInternal(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        x = self.fc(x)
        x = x + 1
        x = x * 2 + torch.ones(5, 5)
        x = self.fc(x)
        x = x + 1
        x = x * 2
        x = self.fc(x)
        return x


class RecurrentRepeatedInternal(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        x = self.fc(x)
        x = x + 1
        x = x * 2 + torch.ones(5, 5)
        x = self.fc(x)
        x = x + 1
        x = x * 2 + torch.ones(5, 5)
        x = self.fc(x)
        x = x + 1
        x = x * 2
        x = self.fc(x)
        x = x + 1
        x = x * 2 + torch.ones(5, 5)
        x = self.fc(x)
        x = x + 1
        x = x * 2
        x = self.fc(x)
        x = x * 2
        x = x + 3
        x = self.fc(x)
        return x


class RecurrentWithBranching(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        x = self.fc(x)
        x1 = x + 1
        x2 = x * 2
        x = x1 ** x2
        x = self.fc(x)
        x1 = x + 1
        x2 = x * 2
        x = x1 ** x2
        x = self.fc(x)
        x1 = x + 1
        x2 = x * 2
        x = x1 ** x2
        x = self.fc(x)
        return x


class RecurrentDoubleNested(nn.Module):
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


class RecurrentNestedInternal(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        x = self.fc1(x)
        x = x + 1
        x = self.fc2(x)
        x = x * 2 + torch.ones(5, 5)
        x = self.fc2(x)
        x = x * 2 + torch.ones(5, 5)
        x = self.fc1(x)
        x = x + 1
        x = self.fc2(x)
        x = x * 2 + torch.ones(5, 5)
        x = self.fc2(x)
        x = x * 2
        x = self.fc1(x)
        x = x + 1 + torch.ones(5, 5)
        x = self.fc2(x)
        x = x * 2
        x = self.fc2(x)
        x = x * 2
        x = self.fc1(x)
        return x


# Test how the nested module functionality works. TODO: Try with trickier nesting/branching to be sure nothing breaks.

class Level0_1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.tan(x)
        x = torch.sin(x)
        return x


class Level1_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.level0_1 = Level0_1()

    def forward(self, x):
        x = x + 1
        x = x * 2
        x = self.level0_1(x)
        return x


class Level1_2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x ** 3
        x = x / 5
        return x


class Level2_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.level1_1 = Level1_1()
        self.level1_2 = Level1_2()

    def forward(self, x):
        x = x + 1
        x = self.level1_1(x)
        x = x + 9
        x = self.level1_2(x)
        x = x / 5
        return x


class Level2_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.level1_1 = Level1_1()
        self.level1_2 = Level1_2()

    def forward(self, x):
        x = x + 9
        x = self.level1_2(x)
        x = x - 5
        x = self.level1_2(self.level1_2(x))
        x = x / 5
        return x


class NestedModulesSimple(nn.Module):
    def __init__(self):
        super().__init__()
        self.level2_1 = Level2_1()
        self.level2_2 = Level2_2()

    def forward(self, x):
        x = torch.cos(x)
        x1 = self.level2_1(x)
        x2 = self.level2_1(x)
        x3 = self.level2_1(x)
        x = x1 * x2 + x3
        x = self.level2_2(x)
        return x

# Next: recurrent with internal branching.

# And recurrent with internal branching and internally generated.

# And some nested loops.
