import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.text_norm import number_to_words


def make_wav(path, n_samples, sr):
    t = np.arange(n_samples) / sr
    freqs = np.random.choice([200, 400, 800, 1600], size=3)
    sig = sum(np.sin(2 * np.pi * f * t) for f in freqs) * 0.3
    sig = sig.astype(np.float32)
    sf.write(str(path), sig, sr)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="asr_smoke_"))
    print(f"tmp: {tmp}")
    data = tmp / "data"
    (data / "train").mkdir(parents=True)
    (data / "dev").mkdir(parents=True)

    rng = np.random.default_rng(0)
    train_rows = []
    dev_rows = []
    SPEAKERS_TRAIN = ["spk_A", "spk_B", "spk_C"]
    SPEAKERS_DEV = ["spk_D", "spk_E"]

    for i in range(40):
        n = int(rng.integers(1000, 999_999))
        sr = int(rng.choice([16000, 22050, 24000]))
        dur = rng.uniform(1.0, 3.0)
        fname = f"train/t{i:04d}.wav"
        make_wav(data / fname, int(dur * sr), sr)
        train_rows.append([fname, n, SPEAKERS_TRAIN[i % len(SPEAKERS_TRAIN)], "male", "wav", sr])

    for i in range(10):
        n = int(rng.integers(1000, 999_999))
        sr = int(rng.choice([16000, 22050, 24000]))
        dur = rng.uniform(1.0, 3.0)
        fname = f"dev/d{i:04d}.wav"
        make_wav(data / fname, int(dur * sr), sr)
        dev_rows.append([fname, n, SPEAKERS_DEV[i % len(SPEAKERS_DEV)], "female", "wav", sr])

    def write_csv(path, rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "transcription", "spk_id", "gender", "ext", "samplerate"])
            w.writerows(rows)

    train_csv = tmp / "train.csv"
    dev_csv = tmp / "dev.csv"
    write_csv(train_csv, train_rows)
    write_csv(dev_csv, dev_rows)

    out_dir = tmp / "ckpt"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    print("\n=== smoke: run 2 epochs of training ===")
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train.py"),
            "--train_csv", str(train_csv),
            "--dev_csv", str(dev_csv),
            "--data_root", str(data),
            "--out_dir", str(out_dir),
            "--epochs", "2",
            "--batch_size", "4",
            "--num_workers", "0",
            "--lr", "3e-4",
        ],
        cwd=ROOT,
        env=env,
    )
    if r.returncode != 0:
        print("TRAIN FAILED")
        sys.exit(1)

    print("\n=== smoke: run inference (greedy, no LM) ===")
    submission_csv = tmp / "submission.csv"
    sample_sub = tmp / "sample_submission.csv"
    import pandas as pd
    dev_df = pd.read_csv(dev_csv)
    sub_df = pd.DataFrame({"filename": dev_df["filename"], "transcription": 0})
    sub_df.to_csv(sample_sub, index=False)

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "infer.py"),
            "--ckpt", str(out_dir / "last.ckpt"),
            "--test_csv", str(sample_sub),
            "--data_root", str(data),
            "--out", str(submission_csv),
            "--decode", "greedy",
        ],
        cwd=ROOT,
        env=env,
    )
    if r.returncode != 0:
        print("INFER FAILED")
        sys.exit(1)

    print("\n=== smoke: check submission.csv ===")
    out = pd.read_csv(submission_csv)
    print(out.head(10))
    assert len(out) == len(dev_df)
    assert "filename" in out.columns and "transcription" in out.columns
    assert out["transcription"].dtype.kind in "iuf"
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
