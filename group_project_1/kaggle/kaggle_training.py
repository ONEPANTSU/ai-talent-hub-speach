import os, subprocess, sys
from pathlib import Path

REPO = "https://github.com/ONEPANTSU/ai-talent-hub-speach.git"
REPO_DIR = "/kaggle/working/repo"
PROJECT_PATH = "/kaggle/working/repo/group_project_1"
if not Path(REPO_DIR).exists():
    subprocess.check_call(["git", "clone", REPO, REPO_DIR])

os.chdir(PROJECT_PATH)
sys.path.insert(0, PROJECT_PATH)

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "-r", "requirements.txt"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "https://github.com/kpu/kenlm/archive/master.zip"])

INPUT_ROOT = Path("/kaggle/input")
print("Ищу train.csv и dev.csv в /kaggle/input ...")

TRAIN_CSV = None
DEV_CSV = None
for p in INPUT_ROOT.rglob("train.csv"):
    TRAIN_CSV = p
    break
for p in INPUT_ROOT.rglob("dev.csv"):
    DEV_CSV = p
    break

assert TRAIN_CSV is not None, "train.csv не найден"
assert DEV_CSV is not None, "dev.csv не найден"

COMP_INPUT = str(TRAIN_CSV.parent)
print("TRAIN_CSV: ", TRAIN_CSV)
print("DEV_CSV:   ", DEV_CSV)
print("COMP_INPUT:", COMP_INPUT)

import pandas as pd
df = pd.read_csv(TRAIN_CSV)
print("train rows:", len(df))
print(df.head(2))
first_audio = Path(COMP_INPUT) / df.iloc[0]["filename"]
print("first audio exists:", first_audio.exists(), "->", first_audio)

OUT_DIR = "/kaggle/working/ckpt"
LM_DIR = "/kaggle/working/lm"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LM_DIR, exist_ok=True)

print("\n=== training KenLM ===")
CORPUS = f"{LM_DIR}/corpus.txt"
LM_ARPA = f"{LM_DIR}/numbers_3gram.arpa"
subprocess.check_call([sys.executable, "scripts/make_lm_corpus.py", "--out", CORPUS])
subprocess.check_call([sys.executable, "scripts/make_arpa.py",
                       "--corpus", CORPUS, "--out", LM_ARPA, "--order", "3"])

print("\n=== training acoustic model ===")
cmd = [
    sys.executable, "scripts/train.py",
    "--train_csv", str(TRAIN_CSV),
    "--dev_csv",   str(DEV_CSV),
    "--data_root", COMP_INPUT,
    "--out_dir",   OUT_DIR,
    "--epochs",    "60",
    "--batch_size","32",
    "--num_workers","4",
    "--lr",        "3e-4",
    "--amp",
]
subprocess.check_call(cmd)

import shutil
ART = Path("/kaggle/working/artifacts")
ART.mkdir(exist_ok=True)
shutil.copy(f"{OUT_DIR}/best.ckpt", ART / "best.ckpt")
shutil.copy(f"{OUT_DIR}/last.ckpt", ART / "last.ckpt")
shutil.copy(f"{OUT_DIR}/vocab.json", ART / "vocab.json")
shutil.copy(LM_ARPA, ART / "numbers_3gram.arpa")
shutil.copytree(f"{REPO_DIR}/src", ART / "src", dirs_exist_ok=True)
shutil.copytree(f"{REPO_DIR}/scripts", ART / "scripts", dirs_exist_ok=True)

print("\nDONE. Артефакты:", ART)
print("Содержимое:")
for p in sorted(ART.rglob("*")):
    if p.is_file():
        print(" ", p, f"{p.stat().st_size/1e6:.2f} MB")
