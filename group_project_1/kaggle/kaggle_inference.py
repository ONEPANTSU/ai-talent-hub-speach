import subprocess, sys, os
from pathlib import Path

try:
    import kenlm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "kenlm"])
    import kenlm

try:
    from pyctcdecode import build_ctcdecoder
    print("pyctcdecode already installed")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pyctcdecode"])
    from pyctcdecode import build_ctcdecoder
    print("pyctcdecode installed")

from pathlib import Path
test_dir = Path(COMP_INPUT) / "test"
test_files = sorted(
    [f for f in test_dir.glob("*") if f.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg"}]
)
print(f"files to decode: {len(test_files)}")

import torch
from src.model import QuartznetASR
from src.features import LogMelFilterBanks
from src.dataset import _load_audio_any
from src.text_norm import words_to_number
import torchaudio

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

ck = torch.load(CKPT, map_location=device, weights_only=False)
vocab = ck["vocab"]
print("vocab size:", len(vocab))

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
print("decoder ready, with_lm =", has_lm)

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
t0 = time.time()
buf_wavs, buf_names = [], []

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
    buf_wavs.clear()
    buf_names.clear()

for i, f in enumerate(test_files):
    try:
        wav = load_16k(f)
    except Exception as e:
        print(f"[!] {f.name}: {e}")
        preds.append((f.name, 1000, ""))
        continue
    buf_wavs.append(wav)
    buf_names.append(f.name)
    if len(buf_wavs) >= BATCH:
        flush()
    if (i + 1) % 100 == 0:
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(test_files) - i - 1)
        print(f"  {i+1}/{len(test_files)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s")
flush()

print(f"\ndecoded {len(preds)} files in {time.time()-t0:.0f}s")
print("first 15 predictions:")
for name, num, hyp in preds[:15]:
    print(f"  {name}: {num}   (hyp='{hyp[:50]}')")

sample_filenames = set(sub_df["filename"].tolist())
has_prefix = any("test/" in f for f in sample_filenames)
print(f"\nsample_submission uses 'test/' prefix: {has_prefix}")

rows = []
for name, num, _ in preds:
    fname = f"test/{name}" if has_prefix else name
    rows.append({ID_COL: fname, PRED_COL: num})

sub_out = pd.DataFrame(rows)
OUT = "/kaggle/working/submission.csv"
sub_out.to_csv(OUT, index=False)
print(f"\nWrote {OUT}  ({len(sub_out)} rows)")
print(sub_out.head(10))
print("\ndistribution check:")
print(f"  unique predictions: {sub_out[PRED_COL].nunique()}")
print(f"  min: {sub_out[PRED_COL].min()}  max: {sub_out[PRED_COL].max()}")
print(f"  fallback 1000 count: {(sub_out[PRED_COL] == 1000).sum()}")