from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from .text_norm import number_to_words


def _load_audio_any(path) -> Tuple[torch.Tensor, int]:
    path = str(path)
    try:
        wav, sr = torchaudio.load(path)
        return wav.to(torch.float32), sr
    except Exception:
        pass
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True)
        wav = torch.from_numpy(data.T).to(torch.float32)
        return wav, sr
    except Exception:
        pass
    import librosa
    import numpy as np
    data, sr = librosa.load(path, sr=None, mono=False)
    if data.ndim == 1:
        data = data[None, :]
    wav = torch.from_numpy(np.ascontiguousarray(data)).to(torch.float32)
    return wav, int(sr)


class NumbersASRDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        vocab: List[str],
        sample_rate: int = 16000,
        wave_aug: Optional[Callable] = None,
        max_seconds: float = 8.0,
    ):
        self.data_root = Path(data_root)
        self.df = pd.read_csv(csv_path)
        self.df["filename"] = self.df["filename"].astype(str)
        self.sr = sample_rate
        self.vocab = vocab
        self.char2idx: Dict[str, int] = {c: i for i, c in enumerate(vocab)}
        self.aug = wave_aug
        self.max_samples = int(max_seconds * sample_rate)

        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.df)

    def _resample(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sr:
            return wav
        if sr not in self._resamplers:
            self._resamplers[sr] = torchaudio.transforms.Resample(sr, self.sr)
        return self._resamplers[sr](wav)

    def _encode(self, text: str) -> torch.LongTensor:
        ids = [self.char2idx[c] for c in text if c in self.char2idx]
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.LongTensor, str, str, int]:
        row = self.df.iloc[idx]
        path = self.data_root / row["filename"]
        wav, sr = _load_audio_any(path)
        if wav.dim() == 2:
            wav = wav.mean(dim=0)  # mono
        wav = self._resample(wav, sr)

        if self.aug is not None:
            wav = self.aug(wav)

        if wav.numel() > self.max_samples:
            wav = wav[:self.max_samples]

        peak = wav.abs().max().clamp_min(1e-6)
        wav = wav / peak * 0.9

        num = int(row["transcription"])
        text = number_to_words(num)
        target = self._encode(text)

        spk = str(row.get("spk_id", "unknown"))
        return wav, target, spk, text, num


def collate_fn(batch):
    wavs, targets, spks, texts, nums = zip(*batch)
    wav_lens = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
    max_wl = int(wav_lens.max().item())
    wav_batch = torch.zeros(len(wavs), max_wl)
    for i, w in enumerate(wavs):
        wav_batch[i, :w.numel()] = w

    tgt_lens = torch.tensor([t.numel() for t in targets], dtype=torch.long)
    max_tl = int(tgt_lens.max().item()) if tgt_lens.numel() > 0 else 1
    tgt_batch = torch.zeros(len(targets), max_tl, dtype=torch.long)
    for i, t in enumerate(targets):
        tgt_batch[i, :t.numel()] = t

    return {
        "wavs": wav_batch,          # (B, T_audio)
        "wav_lens": wav_lens,       # (B,)
        "targets": tgt_batch,       # (B, T_tgt)
        "target_lens": tgt_lens,    # (B,)
        "spks": list(spks),
        "texts": list(texts),
        "nums": list(nums),
    }
