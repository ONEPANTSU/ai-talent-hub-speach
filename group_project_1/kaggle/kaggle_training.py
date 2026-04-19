import os, subprocess, sys
from pathlib import Path

REPO = "https://github.com/<your-username>/<your-repo>.git"
REPO_DIR = "/kaggle/working/repo"
if not Path(REPO_DIR).exists():
    subprocess.check_call(["git", "clone", REPO, REPO_DIR])

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "-r", "requirements.txt"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "https://github.com/kpu/kenlm/archive/master.zip"])

COMP_NAMES = [
    "asr-2026-spoken-numbers-recognition-challenge",
]
COMP_INPUT = None
for name in COMP_NAMES:
    p = Path(f"/kaggle/input/{name}")
    if p.exists():
        COMP_INPUT = str(p)
        break
assert COMP_INPUT, "Данные соревнования не подключены. Add Input > Competition."
print("COMP_INPUT:", COMP_INPUT)

TRAIN_CSV = next(Path(COMP_INPUT).rglob("train.csv"), None)
DEV_CSV = next(Path(COMP_INPUT).rglob("dev.csv"), None)
assert TRAIN_CSV and DEV_CSV, f"train.csv/dev.csv не найдены в {COMP_INPUT}"
print("TRAIN_CSV:", TRAIN_CSV)
print("DEV_CSV:", DEV_CSV)

OUT_DIR = "/kaggle/working/ckpt"
LM_DIR = "/kaggle/working/lm"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LM_DIR, exist_ok=True)

print("\n=== training KenLM ===")
CORPUS = f"{LM_DIR}/corpus.txt"
LM_ARPA = f"{LM_DIR}/numbers_3gram.arpa"
subprocess.check_call([sys.executable, "scripts/make_lm_corpus.py", "--out", CORPUS])
subprocess.check_call([sys.executable, "scripts/train_kenlm.py",
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
