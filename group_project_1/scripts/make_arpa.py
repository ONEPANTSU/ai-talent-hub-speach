from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path


def tokenize(line: str):
    return line.strip().split()


def count_ngrams(corpus_path: Path, order: int):
    counts = [Counter() for _ in range(order)]
    contexts_to_words = [defaultdict(set) for _ in range(order)]

    n_sentences = 0
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            toks = tokenize(line)
            if not toks:
                continue
            n_sentences += 1
            toks = ["<s>"] + toks + ["</s>"]
            for n in range(1, order + 1):
                for i in range(len(toks) - n + 1):
                    gram = tuple(toks[i:i + n])
                    counts[n - 1][gram] += 1
                    if n > 1:
                        ctx = gram[:-1]
                        w = gram[-1]
                        contexts_to_words[n - 1][ctx].add(w)
    return counts, contexts_to_words, n_sentences


def witten_bell_probs(counts, contexts_to_words, order):
    probs = [dict() for _ in range(order)]
    backoffs = [dict() for _ in range(order - 1)]

    unigram_counts = counts[0]
    total = sum(unigram_counts.values())
    V = len(unigram_counts)
    denom = total + V
    for gram, c in unigram_counts.items():
        probs[0][gram] = math.log10(c / denom)

    for n in range(2, order + 1):
        ngram_counts = counts[n - 1]
        ctx_counts = counts[n - 2]
        T = {ctx: len(ws) for ctx, ws in contexts_to_words[n - 1].items()}

        for gram, c in ngram_counts.items():
            ctx = gram[:-1]
            c_ctx = ctx_counts[ctx]
            t_ctx = T[ctx]
            if c_ctx + t_ctx == 0:
                continue
            p = c / (c_ctx + t_ctx)
            probs[n - 1][gram] = math.log10(p)

        for ctx, t_ctx in T.items():
            c_ctx = ctx_counts[ctx]
            if c_ctx + t_ctx == 0:
                continue
            lam = t_ctx / (c_ctx + t_ctx)
            if lam > 0:
                backoffs[n - 2][ctx] = math.log10(lam)

    return probs, backoffs


def write_arpa(out_path: Path, probs, backoffs, order):
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n\\data\\\n")
        for n in range(1, order + 1):
            f.write(f"ngram {n}={len(probs[n-1])}\n")

        for n in range(1, order + 1):
            f.write(f"\n\\{n}-grams:\n")
            # сортируем для детерминизма
            for gram, logp in sorted(probs[n - 1].items()):
                tokens = " ".join(gram)
                if n < order:
                    # backoff вес этой n-gram'ы как контекста для (n+1)-gram
                    bo = backoffs[n - 1].get(gram, 0.0) if n <= order - 1 else 0.0
                    f.write(f"{logp:.6f}\t{tokens}\t{bo:.6f}\n")
                else:
                    f.write(f"{logp:.6f}\t{tokens}\n")
        f.write("\n\\end\\\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", type=int, default=3)
    args = ap.parse_args()

    corpus = Path(args.corpus)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"counting n-grams up to order={args.order} from {corpus} ...", flush=True)
    counts, ctx2w, n_sent = count_ngrams(corpus, args.order)
    print(f"  sentences: {n_sent:,}")
    for i, c in enumerate(counts):
        print(f"  {i+1}-grams: {len(c):,}")

    print("computing Witten-Bell probabilities ...", flush=True)
    probs, backoffs = witten_bell_probs(counts, ctx2w, args.order)

    print(f"writing ARPA to {out} ...", flush=True)
    write_arpa(out, probs, backoffs, args.order)
    print(f"done. size = {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()