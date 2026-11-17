"""Network definitions for C11 and the two EXP066 CNNs."""

import torch
from torch import nn
from torch.nn import functional as F

DILATIONS = {"control": (3, 3, 3), "candidate": (1, 3, 5)}


def make_cnn_gru():
    """6,215 parameters. Left padding prevents future context within the window."""
    import torch
    from torch import nn
    from torch.nn import functional as functional

    class CausalConv(nn.Module):
        def __init__(self, cin, cout, kernel, stride=1):
            super().__init__()
            self.left = kernel - 1
            self.conv = nn.Conv1d(cin, cout, kernel, stride=stride)

        def forward(self, value):
            return self.conv(functional.pad(value, (self.left, 0)))

    class CNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                CausalConv(6, 16, 5),
                nn.ReLU(),
                CausalConv(16, 24, 5, stride=2),
                nn.ReLU(),
            )
            self.recurrent = nn.GRU(24, 24, batch_first=True)
            self.head = nn.Linear(24, 7)

        def forward(self, value):
            sequence = self.encoder(value.transpose(1, 2)).transpose(1, 2)
            sequence, _ = self.recurrent(sequence)
            return self.head(sequence[:, -1])

    return CNNGRU()


class CausalConv(nn.Conv1d):
    def forward(self, value):
        left = self.dilation[0] * (self.kernel_size[0] - 1)
        return super().forward(F.pad(value, (left, 0)))


class ScaleCNN(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.stem = CausalConv(6, 16, 5)
        self.branches = nn.ModuleList(
            CausalConv(16, 8, 7, dilation=d) for d in DILATIONS[arm]
        )
        self.mix = nn.Conv1d(24, 24, 1)
        self.head = nn.Linear(24, 7)

    def features(self, value):
        if value.ndim != 3 or value.shape[1:] != (125, 6) or value.dtype != torch.float32:
            raise ValueError("Expected float32 N × 125 × 6")
        stem = self.stem(value.transpose(1, 2)).relu()
        return self.mix(torch.cat([branch(stem) for branch in self.branches], dim=1)).relu()

    def forward(self, value):
        return self.head(self.features(value).mean(dim=2))
