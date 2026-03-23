import argparse
import gzip
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from run_experiments import (
    decode_logits_dataset,
    load_samples,
    precompute_logits,
    resolve_existing_path,
    save_df,
)
from wav2vec2decoder import Wav2Vec2Decoder


def train_financial_lm(corpus_path: Path, output_gz: Path) -> None:
    build_root = Path("/tmp/kenlm_build")
    build_dir = build_root / "build"
    lmplz_bin = build_dir / "bin" / "lmplz"

    if not lmplz_bin.exists():
        if not build_root.exists():
            subprocess.run(
                ["git", "clone", "--depth=1", "https://github.com/kpu/kenlm", str(build_root)],
                check=True,
            )
        # Homebrew Boost 1.90+: no separate libboost_system; KenLM CMake still lists it.
        cmakelists = build_root / "CMakeLists.txt"
        txt = cmakelists.read_text(encoding="utf-8")
        if "\n  system\n" in txt:
            cmakelists.write_text(txt.replace("\n  system\n", "\n"), encoding="utf-8")
        build_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["cmake", "-DCOMPILE_TESTS=OFF", ".."],
            cwd=build_dir,
            check=True,
        )
        subprocess.run(["make", "-j4", "lmplz", "build_binary"], cwd=build_dir, check=True)

    arpa_path = output_gz.with_suffix("").with_suffix(".arpa")
    with corpus_path.open("rb") as fin, arpa_path.open("wb") as fout:
        subprocess.run(
            [str(lmplz_bin), "-o", "3", "--discount_fallback"],
            stdin=fin,
            stdout=fout,
            check=True,
        )

    with arpa_path.open("rb") as f_in, gzip.open(output_gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_dir", default="data/librispeech_test_other")
    parser.add_argument("--earnings_dir", default="data/earnings22_test")
    parser.add_argument("--earnings_corpus", default="data/earnings22_train/corpus.txt")
    parser.add_argument("--lm_3gram", default="lm/3-gram.pruned.1e-7.arpa.gz")
    parser.add_argument("--lm_4gram", default="lm/4-gram.arpa.gz")
    parser.add_argument("--lm_financial", default="lm/financial-3gram.arpa.gz")
    parser.add_argument("--output_dir", default="results_remaining")
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--skip-task5",
        action="store_true",
        help="Skip 3-gram vs 4-gram comparison (use while 4-gram.arpa.gz is still downloading).",
    )
    parser.add_argument(
        "--skip-task9",
        action="store_true",
        help="Skip full LM comparison (saves time when you only need Task 5).",
    )
    parser.add_argument(
        "--min-4gram-bytes",
        type=int,
        default=800_000_000,
        help="If 4-gram file is smaller than this, skip Task 5 (incomplete download). Set 0 to disable.",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    args.lm_3gram = resolve_existing_path(args.lm_3gram)
    args.lm_4gram = resolve_existing_path(args.lm_4gram)
    args.lm_financial = resolve_existing_path(args.lm_financial)
    corpus = Path(resolve_existing_path(args.earnings_corpus))

    # Task 8: train financial LM if absent
    financial_gz = Path(args.lm_financial)
    if not financial_gz.exists():
        train_financial_lm(corpus, financial_gz)

    libri_samples = load_samples(args.librispeech_dir)
    libri_logits, libri_refs = precompute_logits(libri_samples)

    earn_logits, earn_refs = [], []
    if not args.skip_task9:
        earn_samples = load_samples(args.earnings_dir)
        earn_logits, earn_refs = precompute_logits(earn_samples)

    # Task 5: compare 3-gram vs 4-gram on LibriSpeech with best alpha/beta
    four_g = Path(args.lm_4gram)
    skip5 = args.skip_task5
    if not skip5 and args.min_4gram_bytes > 0 and four_g.exists():
        if four_g.stat().st_size < args.min_4gram_bytes:
            print(
                f"Skipping Task 5: {four_g} is {four_g.stat().st_size} bytes "
                f"(expected >= {args.min_4gram_bytes} when download is complete)."
            )
            skip5 = True
    if not skip5:
        task5_rows = []
        for lm_name, lm_path in [("3gram", args.lm_3gram), ("4gram", args.lm_4gram)]:
            dec = Wav2Vec2Decoder(
                lm_model_path=lm_path,
                beam_width=args.beam_width,
                alpha=args.alpha,
                beta=args.beta,
            )
            sf = decode_logits_dataset(dec, libri_logits, libri_refs, "beam_lm")
            rs = decode_logits_dataset(dec, libri_logits, libri_refs, "beam_lm_rescore")
            task5_rows.append({"lm": lm_name, "method": "beam_lm", **sf})
            task5_rows.append({"lm": lm_name, "method": "beam_lm_rescore", **rs})
        save_df(pd.DataFrame(task5_rows), out / "task5_3gram_vs_4gram.csv")

    # Task 9: both LMs on both domains with best methods
    if not args.skip_task9:
        task9_rows = []
        for lm_name, lm_path in [
            ("librispeech_3gram", args.lm_3gram),
            ("financial_3gram", str(financial_gz)),
        ]:
            dec = Wav2Vec2Decoder(
                lm_model_path=lm_path,
                beam_width=args.beam_width,
                alpha=args.alpha,
                beta=args.beta,
            )
            for dataset_name, logits, refs in [
                ("librispeech_test_other", libri_logits, libri_refs),
                ("earnings22_test", earn_logits, earn_refs),
            ]:
                sf = decode_logits_dataset(dec, logits, refs, "beam_lm")
                rs = decode_logits_dataset(dec, logits, refs, "beam_lm_rescore")
                task9_rows.append({"dataset": dataset_name, "lm": lm_name, "method": "beam_lm", **sf})
                task9_rows.append({"dataset": dataset_name, "lm": lm_name, "method": "beam_lm_rescore", **rs})
        save_df(pd.DataFrame(task9_rows), out / "task9_full_lm_comparison.csv")
    else:
        print("Skipped Task 9 (--skip-task9).")

    if args.skip_task5 and args.skip_task9:
        print("Warning: both Task 5 and Task 9 skipped; no new artifacts written.")

    print(f"Saved remaining task artifacts to: {out.resolve()}")


if __name__ == "__main__":
    main()
