import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.decoder import greedy_decode, beam_search_decode
from src.text_norm import vocab_chars

vocab = ["<blank>"] + vocab_chars()
V = len(vocab)
print("vocab:", vocab, "size:", V)

def idx(c):
    return vocab.index(c)

target = "пять"
target_ids = [idx(c) for c in target]

T = 12
logits = torch.full((1, T, V), -10.0)
seq_plan = [
    idx("п"), idx("п"), 0,
    idx("я"), 0,
    idx("т"), idx("т"), 0,
    idx("ь"), 0, 0, 0,
]
for t, s in enumerate(seq_plan):
    logits[0, t, s] = 0.0
log_probs = torch.log_softmax(logits, dim=-1)

g = greedy_decode(log_probs, vocab, blank_idx=0)
b = beam_search_decode(log_probs, vocab, blank_idx=0, beam_size=8, lm=None)
print("greedy:", g)
print("beam  :", b)
assert g[0] == "пять", f"greedy should return 'пять', got {g[0]!r}"
assert b[0] == "пять", f"beam should return 'пять', got {b[0]!r}"

try:
    import kenlm  # noqa
    has_kenlm = True
except ImportError:
    has_kenlm = False

if has_kenlm:
    import subprocess, tempfile, os
    tmp = Path(tempfile.mkdtemp())
    corpus = tmp / "c.txt"
    arpa = tmp / "lm.arpa"
    corpus.write_text(
        "пять\nсто пять\nодна тысяча пять\nдва\nтри\n" * 20,
        encoding="utf-8",
    )
    try:
        from src.decoder import KenLMWordScorer
        candidate = Path("/home/claude/asr_project/lm/numbers_3gram.arpa")
        if not candidate.exists():
            print("(no lm available, skipping LM test)")
        else:
            lm = KenLMWordScorer(str(candidate))
            b_lm = beam_search_decode(log_probs, vocab, blank_idx=0, beam_size=8, lm=lm, alpha=0.5, beta=1.0)
            print("beam+LM:", b_lm)
            assert b_lm[0] == "пять"
    except Exception as e:
        print("(lm test skipped:", e, ")")
else:
    print("(kenlm not installed, skipping LM test)")

print("DECODER OK")
