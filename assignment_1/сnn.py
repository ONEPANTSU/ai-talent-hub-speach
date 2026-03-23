from math import gcd
import torch.nn as nn

from melbanks import LogMelFilterBanks


def _safe_groups(channels: int, groups: int) -> int:
    while channels % groups != 0:
        groups -= 1
    return groups


class SimpleCNN(nn.Module):
    CH = 32

    def __init__(self, n_mels: int = 80, groups: int = 1):
        super().__init__()
        self.n_mels = n_mels
        self.groups = groups
        self.mel = LogMelFilterBanks(n_mels=n_mels)

        g = _safe_groups(gcd(n_mels, self.CH), groups)

        self.features = nn.Sequential(
            nn.Conv1d(n_mels, self.CH, kernel_size=3, padding=1, groups=g),
            nn.BatchNorm1d(self.CH),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(self.CH, 1)

    def forward(self, x):
        x = self.mel(x)            # (batch, n_mels, frames)
        x = self.features(x)       # (batch, CH, 1)
        x = x.squeeze(-1)          # (batch, CH)
        return self.classifier(x).squeeze(-1)
