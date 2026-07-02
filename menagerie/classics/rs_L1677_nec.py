# SOURCE: vendored from mjacar/pytorch-nec @ main (models.py)
# https://github.com/mjacar/pytorch-nec -- community PyTorch implementation of "Neural
# Episodic Control" (Pritzel et al., ICML 2017). DeepMind's own paper code was never
# open-sourced; this is the canonical, widely-cited community port (see also
# EndingCredits/Neural-Episodic-Control for a second independent port of the same
# architecture). NEC's forward-pass network is the CNN state-embedding encoder `DQN`
# below (Atari 84x84x4 -> conv -> conv -> fc -> embedding head); the rest of the NEC
# agent (`dnd.py`'s Differentiable Neural Dictionary, `nec_agent.py`'s NECAgent) is
# episodic-memory training/control logic built on non-tensor kNN lookups (via pyflann)
# over Python-level key/value stores, not a `nn.Module.forward` graph, so it is not a
# traceable capture unit -- the module below is exactly and only `models.DQN`, the real
# embedding network class, transcribed verbatim (import list trimmed to what's used).
import torch
import torch.nn as nn
import torch.nn.functional as F

MENAGERIE_ZOO = "vendored-pytorch"


class DQN(nn.Module):
    def __init__(self, embedding_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc = nn.Linear(2592, 256)
        self.head = nn.Linear(256, embedding_size)

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = F.relu(self.conv2(out))
        out = F.relu(self.fc(out.view(out.size(0), -1)))
        out = self.head(out)
        return out


# ---- staging build/example helpers ----
def build_nec_dqn():
    torch.manual_seed(0)
    return DQN(embedding_size=32)


def example_input_nec_dqn():
    torch.manual_seed(0)
    # Real repo's Atari preprocessing stacks 4 frames of an 84x84 grayscale
    # observation (see utils/atari_wrapper.py) -> (N, 4, 84, 84).
    return (torch.randn(1, 4, 84, 84),)


MENAGERIE_ENTRIES = [
    (
        "NeuralEpisodicControl-DQNEmbedding",
        build_nec_dqn,
        example_input_nec_dqn,
        2017,
        "vendored-pytorch",
    ),
]
