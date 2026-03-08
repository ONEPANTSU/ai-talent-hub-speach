from pathlib import Path

import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset


class YesNoDataset(Dataset):

    LABELS = {"yes": 1, "no": 0}
    AUDIO_LEN = 16000  # 1 sec at 16 kHz

    def __init__(self, subset: str, data_root: str = "./assigment_1/data"):
        Path(data_root).mkdir(parents=True, exist_ok=True)
        print(f"  Loading SPEECHCOMMANDS ({subset})...", end=" ", flush=True)
        self._dataset = torchaudio.datasets.SPEECHCOMMANDS(
            root=data_root, download=True, subset=subset
        )
        self._indices = [
            i for i, filepath in enumerate(self._dataset._walker)
            if Path(filepath).parent.name in self.LABELS
        ]
        print(f"{len(self._indices)} samples")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        waveform, sr, label, *_ = self._dataset[self._indices[idx]]
        if waveform.shape[1] < self.AUDIO_LEN:
            waveform = nn.functional.pad(waveform, (0, self.AUDIO_LEN - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.AUDIO_LEN]
        return waveform.squeeze(0), self.LABELS[label]
