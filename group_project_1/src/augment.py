from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import torch
import torchaudio

class SpeedPerturb:
    def __init__(self, sample_rate: int = 16000, speeds=(0.9, 1.0, 1.1), p: float = 0.5):
        self.sr = sample_rate
        self.speeds = speeds
        self.p = p
        self._resamplers = {
            s: torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=int(sample_rate * s),
            )
            for s in speeds if s != 1.0
        }

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return wav
        s = random.choice(self.speeds)
        if s == 1.0:
            return wav
        return self._resamplers[s](wav)


class RandomGain:
    def __init__(self, min_db: float = -6.0, max_db: float = 6.0, p: float = 0.5):
        self.min_db = min_db
        self.max_db = max_db
        self.p = p

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return wav
        db = random.uniform(self.min_db, self.max_db)
        return wav * (10 ** (db / 20))


class GaussianNoise:
    def __init__(self, min_snr_db: float = 10.0, max_snr_db: float = 30.0, p: float = 0.3):
        self.min_snr = min_snr_db
        self.max_snr = max_snr_db
        self.p = p

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return wav
        snr = random.uniform(self.min_snr, self.max_snr)
        sig_power = wav.pow(2).mean().clamp_min(1e-10)
        noise_power = sig_power / (10 ** (snr / 10))
        noise = torch.randn_like(wav) * noise_power.sqrt()
        return wav + noise


class BackgroundNoise:
    def __init__(
        self,
        noise_dir: Optional[str],
        sample_rate: int = 16000,
        min_snr_db: float = 5.0,
        max_snr_db: float = 20.0,
        p: float = 0.3,
    ):
        self.sr = sample_rate
        self.min_snr = min_snr_db
        self.max_snr = max_snr_db
        self.p = p
        self.files: List[Path] = []
        if noise_dir:
            root = Path(noise_dir)
            if root.exists():
                self.files = [
                    p for p in root.rglob("*")
                    if p.suffix.lower() in {".wav", ".flac", ".mp3", ".ogg"}
                ]

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.files or random.random() > self.p:
            return wav
        f = random.choice(self.files)
        try:
            from .dataset import _load_audio_any
            noise, sr = _load_audio_any(f)
            if sr != self.sr:
                noise = torchaudio.functional.resample(noise, sr, self.sr)
            noise = noise.mean(dim=0)  # моно
        except Exception:
            return wav
        T = wav.shape[-1]
        if noise.numel() < T:
            reps = T // noise.numel() + 1
            noise = noise.repeat(reps)
        start = random.randint(0, noise.numel() - T)
        noise = noise[start:start + T]

        sig_power = wav.pow(2).mean().clamp_min(1e-10)
        noise_power = noise.pow(2).mean().clamp_min(1e-10)
        snr = random.uniform(self.min_snr, self.max_snr)
        scale = (sig_power / (noise_power * (10 ** (snr / 10)))).sqrt()
        return wav + noise * scale


class WaveAugPipeline:
    def __init__(
        self,
        sample_rate: int = 16000,
        noise_dir: Optional[str] = None,
        use_speed: bool = True,
        use_gain: bool = True,
        use_gauss: bool = True,
        use_bg_noise: bool = True,
    ):
        self.ops = []
        if use_speed:
            self.ops.append(SpeedPerturb(sample_rate=sample_rate))
        if use_gain:
            self.ops.append(RandomGain())
        if use_bg_noise and noise_dir:
            self.ops.append(BackgroundNoise(noise_dir=noise_dir, sample_rate=sample_rate))
        if use_gauss:
            self.ops.append(GaussianNoise())

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        for op in self.ops:
            wav = op(wav)
        return wav


class SpecAugment(torch.nn.Module):
    def __init__(
        self,
        freq_masks: int = 2,
        freq_mask_param: int = 15,
        time_masks: int = 2,
        time_mask_param: int = 35,
        time_mask_ratio: float = 0.05,
    ):
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_mask_param = freq_mask_param
        self.time_masks = time_masks
        self.time_mask_param = time_mask_param
        self.time_mask_ratio = time_mask_ratio

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
        B, F, T = spec.shape
        spec = spec.clone()
        for b in range(B):
            for _ in range(self.freq_masks):
                f = random.randint(0, self.freq_mask_param)
                if f == 0 or f >= F:
                    continue
                f0 = random.randint(0, F - f)
                spec[b, f0:f0 + f, :] = 0.0
            max_t = min(self.time_mask_param, int(T * self.time_mask_ratio) + 1)
            for _ in range(self.time_masks):
                t = random.randint(0, max_t)
                if t == 0 or t >= T:
                    continue
                t0 = random.randint(0, T - t)
                spec[b, :, t0:t0 + t] = 0.0
        return spec
