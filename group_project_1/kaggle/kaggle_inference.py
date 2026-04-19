import subprocess, sys, os
from pathlib import Path

try:
    import kenlm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "kenlm"])
    import kenlm

try:
    from pyctcdecode import build_ctcdecoder
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           "--no-deps", "pyctcdecode", "pygtrie"])
    from pyctcdecode import build_ctcdecoder

INPUT_ROOT = Path("/kaggle/input")

WEIGHTS_INPUT = None
for p in INPUT_ROOT.rglob("best.ckpt"):
    parent = p.parent
    if (parent / "scripts" / "infer.py").exists() and (parent / "src").exists():
        WEIGHTS_INPUT = str(parent)
        break
assert WEIGHTS_INPUT is not None, "artifacts/ не найдено. Подключи training notebook в Input."
print("WEIGHTS_INPUT:", WEIGHTS_INPUT)

sample_sub = next(INPUT_ROOT.rglob("sample_submission.csv"), None)
assert sample_sub is not None
SAMPLE_SUB = str(sample_sub)
COMP_INPUT = str(sample_sub.parent)
print("COMP_INPUT:", COMP_INPUT)

sys.path.insert(0, WEIGHTS_INPUT)

import pandas as pd
sub_df = pd.read_csv(SAMPLE_SUB)
ID_COL = sub_df.columns[0]
PRED_COL = sub_df.columns[1]
has_prefix = any("test/" in f for f in sub_df[ID_COL].tolist())

CKPT = f"{WEIGHTS_INPUT}/best.ckpt"
LM = f"{WEIGHTS_INPUT}/numbers_3gram.arpa"
has_lm = Path(LM).exists()
print(f"CKPT={CKPT}\nLM={LM if has_lm else '<none>'}")

test_dir = Path(COMP_INPUT) / "test"
test_files = sorted(
    [f for f in test_dir.glob("*") if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
)
print(f"files: {len(test_files)}")

import torch
import torchaudio
from src.model import QuartznetASR
from src.features import LogMelFilterBanks
from src.dataset import _load_audio_any
from src.text_norm import words_to_number

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

ck = torch.load(CKPT, map_location=device, weights_only=False)
vocab = ck["vocab"]
model = QuartznetASR(n_mels=64, vocab_size=len(vocab)).to(device)
model.load_state_dict(ck["model"])
model.eval()
mel = LogMelFilterBanks(n_mels=64).to(device)

labels = [""] + list(vocab[1:])
decoder = build_ctcdecoder(
    labels,
    kenlm_model_path=LM if has_lm else None,
    alpha=0.5,
    beta=1.0,
)

def load_16k(path):
    wav, sr = _load_audio_any(path)
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    peak = wav.abs().max().clamp_min(1e-6)
    return wav / peak * 0.9

@torch.no_grad()
def logprobs_batch(wavs):
    lens = [w.numel() for w in wavs]
    max_l = max(lens)
    batch = torch.zeros(len(wavs), max_l, device=device)
    for i, w in enumerate(wavs):
        batch[i, :w.numel()] = w.to(device)
    mels = mel(batch)
    logits = model(mels)
    lp = torch.log_softmax(logits, dim=-1).cpu()
    out = []
    for i, L in enumerate(lens):
        t_mel = L // 160 + 1
        t_out = (t_mel + 1) // 2
        out.append(lp[i, :min(t_out, lp.size(1)), :].numpy())
    return out

import time
BATCH = 8
preds = []
buf_wavs, buf_names = [], []
t0 = time.time()

def flush():
    if not buf_wavs:
        return
    lps = logprobs_batch(buf_wavs)
    for name, lp in zip(buf_names, lps):
        hyp = decoder.decode(lp, beam_width=32)
        num = words_to_number(hyp)
        if num == 0:
            num = 1000
        preds.append((name, num, hyp))
    buf_wavs.clear(); buf_names.clear()

for i, f in enumerate(test_files):
    try:
        buf_wavs.append(load_16k(f))
        buf_names.append(f.name)
    except Exception as e:
        preds.append((f.name, 1000, ""))
        continue
    if len(buf_wavs) >= BATCH:
        flush()
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(test_files)}  elapsed={time.time()-t0:.0f}s")
flush()

print(f"decoded {len(preds)} in {time.time()-t0:.0f}s")

rows = []
for name, num, _ in preds:
    fname = f"test/{name}" if has_prefix else name
    rows.append({ID_COL: fname, PRED_COL: num})
sub_out = pd.DataFrame(rows)
OUT = "/kaggle/working/submission.csv"
sub_out.to_csv(OUT, index=False)
print(f"\nWrote {OUT}  ({len(sub_out)} rows)")
print(sub_out.head(10))
print(f"unique: {sub_out[PRED_COL].nunique()}  fallback_1000: {(sub_out[PRED_COL]==1000).sum()}")