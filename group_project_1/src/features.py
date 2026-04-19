from __future__ import annotations

import torch
from torch import nn
import torchaudio


class LogMelFilterBanks(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,           # 25 ms @ 16 kHz
        hop_length: int = 160,      # 10 ms @ 16 kHz
        n_mels: int = 64,
        f_min: float = 0.0,
        f_max: float | None = None,
    ) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max if f_max is not None else sample_rate / 2,
            power=2.0,
            center=True,
            pad_mode="reflect",
            norm=None,
            mel_scale="htk",
        )
        self.eps = 1e-6

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        spec = self.mel(wav)          # (B, n_mels, F)
        return torch.log(spec + self.eps)
