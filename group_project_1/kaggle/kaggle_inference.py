import subprocess, sys, os
from pathlib import Path

try:
    import kenlm
    print("kenlm already installed")
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "kenlm"])
    import kenlm
    print("kenlm installed")

INPUT_ROOT = Path("/kaggle/input")

best_ckpt = next(INPUT_ROOT.rglob("best.ckpt"), None)
assert best_ckpt is not None, "best.ckpt не найден"
WEIGHTS_INPUT = str(best_ckpt.parent)
print("WEIGHTS_INPUT:", WEIGHTS_INPUT)

sample_sub = next(INPUT_ROOT.rglob("sample_submission.csv"), None)
assert sample_sub is not None, "sample_submission.csv не найден"
SAMPLE_SUB = str(sample_sub)
COMP_INPUT = str(sample_sub.parent)
print("SAMPLE_SUB:", SAMPLE_SUB)
print("COMP_INPUT:", COMP_INPUT)

sys.path.insert(0, WEIGHTS_INPUT)
INFER_SCRIPT = str(Path(WEIGHTS_INPUT) / "scripts" / "infer.py")
assert Path(INFER_SCRIPT).exists(), f"infer.py не найден: {INFER_SCRIPT}"

import pandas as pd
sub_df = pd.read_csv(SAMPLE_SUB)
print("columns:", list(sub_df.columns))
print(sub_df.head())
print(f"rows: {len(sub_df)}")

first_path = Path(COMP_INPUT) / sub_df.iloc[0, 0]
print(f"first audio: {first_path}  exists: {first_path.exists()}")

CKPT = f"{WEIGHTS_INPUT}/best.ckpt"
LM = f"{WEIGHTS_INPUT}/numbers_3gram.arpa"
OUT = "/kaggle/working/submission.csv"

has_lm = Path(LM).exists()
print("LM:", LM if has_lm else "<no>")

ID_COL = sub_df.columns[0]
PRED_COL = sub_df.columns[1]
print(f"id_col={ID_COL}  pred_col={PRED_COL}")

cmd = [
    sys.executable, INFER_SCRIPT,
    "--ckpt", CKPT,
    "--test_csv", SAMPLE_SUB,
    "--data_root", COMP_INPUT,
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

print("\nrunning:", " ".join(cmd))
subprocess.check_call(cmd)

final = pd.read_csv(OUT)
print(f"\nsubmission rows: {len(final)}")
print(final.head(20))