"""
Собираем KenLM-бинарники (если их нет) и тренируем 3-gram на корпусе.

Использование:
  # 1) сгенерировать корпус
  python scripts/make_lm_corpus.py --out lm/corpus.txt
  # 2) обучить LM
  python scripts/train_kenlm.py --corpus lm/corpus.txt --out lm/numbers_3gram.arpa

На Kaggle (интернет включён при обучении, но не при инференсе):
собираем KenLM один раз и сохраняем в kaggle-датасет; в инференс-ноутбуке
просто читаем готовый .arpa.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def build_kenlm(build_root: Path) -> Path:
    """Клонируем KenLM и собираем lmplz. Возвращает путь к lmplz."""
    build_dir = build_root / "build"
    lmplz = build_dir / "bin" / "lmplz"
    if lmplz.exists():
        return lmplz

    if not build_root.exists():
        subprocess.run(
            ["git", "clone", "--depth=1", "https://github.com/kpu/kenlm", str(build_root)],
            check=True,
        )
    # На некоторых системах (Homebrew Boost ≥1.90) KenLM ругается на libboost_system.
    cmakelists = build_root / "CMakeLists.txt"
    txt = cmakelists.read_text(encoding="utf-8")
    if "\n  system\n" in txt:
        cmakelists.write_text(txt.replace("\n  system\n", "\n"), encoding="utf-8")

    build_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cmake", "-DCOMPILE_TESTS=OFF", ".."], cwd=build_dir, check=True)
    subprocess.run(["make", "-j4", "lmplz", "build_binary"], cwd=build_dir, check=True)
    return lmplz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True, help="output .arpa path")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--build_root", default="/tmp/kenlm_build")
    args = ap.parse_args()

    lmplz = build_kenlm(Path(args.build_root))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    with open(args.corpus, "rb") as fin, open(args.out, "wb") as fout:
        subprocess.run(
            [str(lmplz), "-o", str(args.order), "--discount_fallback"],
            stdin=fin,
            stdout=fout,
            check=True,
        )
    print(f"LM written: {args.out}  size={Path(args.out).stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
