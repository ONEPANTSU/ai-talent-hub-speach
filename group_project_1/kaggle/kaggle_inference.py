import subprocess, sys, os

def pip_install_from_dataset(wheel_glob):
    import glob
    wheels = glob.glob(wheel_glob)
    if wheels:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", wheels[0]])

pip_install_from_dataset("/kaggle/input/*/kenlm*.whl")

COMP_INPUT = "/kaggle/input/asr-2026-spoken-numbers-recognition-challenge"  # соревнование
WEIGHTS_INPUT = "/kaggle/input/asr-numbers-weights"

sys.path.insert(0, WEIGHTS_INPUT)

import pandas as pd
from pathlib import Path

cand = list(Path(COMP_INPUT).rglob("sample_submission.csv"))
assert cand, f"sample_submission.csv не найден в {COMP_INPUT}"
SAMPLE_SUB = str(cand[0])
print("sample_submission:", SAMPLE_SUB)
sub_df = pd.read_csv(SAMPLE_SUB)
print("columns:", list(sub_df.columns))
print("head:\n", sub_df.head())

DATA_ROOT = COMP_INPUT

CKPT = f"{WEIGHTS_INPUT}/best.ckpt"
LM = f"{WEIGHTS_INPUT}/numbers_3gram.arpa"
OUT = "/kaggle/working/submission.csv"

for p in (CKPT,):
    assert Path(p).exists(), f"Отсутствует: {p}"
has_lm = Path(LM).exists()
print("checkpoint:", CKPT)
print("LM:", LM if has_lm else "<не найден — используем greedy или beam без LM>")

ID_COL = sub_df.columns[0]
PRED_COL = sub_df.columns[1] if len(sub_df.columns) > 1 else "transcription"
print(f"id_col={ID_COL}  pred_col={PRED_COL}")

cmd = [
    sys.executable,
    f"{WEIGHTS_INPUT}/scripts/infer.py",
    "--ckpt", CKPT,
    "--test_csv", SAMPLE_SUB,
    "--data_root", DATA_ROOT,
    "--out", OUT,
    "--id_col", ID_COL,
    "--pred_col", PRED_COL,
    "--batch_size", "8",
    "--decode", "beam" if has_lm else "greedy",
    "--beam_size", "32",
    "--alpha", "0.5",
    "--beta", "1.0",
]
if has_lm:
    cmd += ["--lm", LM]

print("running:", " ".join(cmd))
subprocess.check_call(cmd)

final = pd.read_csv(OUT)
print(f"\nsubmission rows: {len(final)}")
print(final.head(10))
print("\ndone:", OUT)
