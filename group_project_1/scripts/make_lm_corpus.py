from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.text_norm import number_to_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="lm_corpus.txt")
    ap.add_argument("--min_n", type=int, default=1000)
    ap.add_argument("--max_n", type=int, default=999_999)
    args = ap.parse_args()

    out = Path(args.out)
    n_lines = 0
    with out.open("w", encoding="utf-8") as f:
        for n in range(args.min_n, args.max_n + 1):
            line = number_to_words(n)
            f.write(line + "\n")
            n_lines += 1
    print(f"Wrote {n_lines:,} lines to {out.resolve()}")
    print(f"Size: {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
