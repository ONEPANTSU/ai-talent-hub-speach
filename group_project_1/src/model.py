from __future__ import annotations

from typing import List

import torch
from torch import nn


def _separable_conv1d(in_ch: int, out_ch: int, kernel: int, padding: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, in_ch, kernel_size=kernel, padding=padding, groups=in_ch, bias=False),
        nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
    )


class SubBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float, last: bool = False):
        super().__init__()
        self.conv = _separable_conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True) if not last else nn.Identity()
        self.drop = nn.Dropout(dropout) if not last else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


class BBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, repeats: int, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(repeats):
            ic = in_ch if i == 0 else out_ch
            is_last = (i == repeats - 1)
            layers.append(SubBlock(ic, out_ch, kernel, dropout, last=is_last))
        self.main = nn.Sequential(*layers)

        self.residual = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.main(x)
        r = self.residual(x)
        return self.drop(self.act(y + r))


class QuartznetASR(nn.Module):
    def __init__(self, n_mels: int = 64, vocab_size: int = 22, dropout: float = 0.15):
        super().__init__()

        # C1 — входной stride=2 по времени
        self.c1 = nn.Sequential(
            nn.Conv1d(n_mels, 128, kernel_size=33, stride=2, padding=16, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        cfg = [
            (256, 33, 5),
            (256, 39, 5),
            (256, 51, 5),
            (384, 63, 5),
            (384, 75, 5),
        ]
        blocks: List[nn.Module] = []
        prev = 128
        for out_ch, k, r in cfg:
            blocks.append(BBlock(prev, out_ch, k, r, dropout))
            prev = out_ch
        self.blocks = nn.Sequential(*blocks)

        self.c2 = nn.Sequential(
            _separable_conv1d(prev, 512, kernel=29, padding=14),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.c3 = nn.Sequential(
            nn.Conv1d(512, 640, kernel_size=1, bias=False),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head = nn.Conv1d(640, vocab_size, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, mels: torch.Tensor) -> torch.Tensor:
        x = self.c1(mels)
        x = self.blocks(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.head(x)           # (B, V, T')
        return x.transpose(1, 2)   # (B, T', V)

    def output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        return (input_lengths + 1) // 2


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = QuartznetASR(n_mels=64, vocab_size=22)
    n = count_params(m)
    print(f"Params: {n:,}  (limit 5,000,000)")
    x = torch.randn(2, 64, 400)
    y = m(x)
    print("Output:", y.shape)  # (2, ~200, 22)
