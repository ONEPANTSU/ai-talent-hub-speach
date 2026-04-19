from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch


def greedy_decode(log_probs: torch.Tensor, vocab: Sequence[str], blank_idx: int = 0) -> List[str]:
    ids = log_probs.argmax(dim=-1).cpu().tolist()
    results: List[str] = []
    for seq in ids:
        out = []
        prev = -1
        for i in seq:
            if i != prev and i != blank_idx:
                out.append(vocab[i])
            prev = i
        results.append("".join(out))
    return results



@dataclass
class Beam:
    prefix: Tuple[int, ...] = ()
    p_b: float = 0.0
    p_nb: float = -float("inf")
    lm_score: float = 0.0
    n_words: int = 0


def _logsumexp2(a: float, b: float) -> float:
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class KenLMWordScorer:
    def __init__(self, arpa_path: str):
        import kenlm  # отложенный импорт
        self.model = kenlm.Model(arpa_path)

    @staticmethod
    def _log10_to_ln(x: float) -> float:
        return x * math.log(10)

    def word_score(self, prev_words: List[str], word: str) -> float:
        state_in = self.model.BeginSentenceState()
        state_out = self._kenlm_state()
        for w in prev_words:
            score = self.model.BaseScore(state_in, w, state_out)
            state_in, state_out = state_out, self._kenlm_state()
        score = self.model.BaseScore(state_in, word, state_out)
        return self._log10_to_ln(score)

    def _kenlm_state(self):
        import kenlm
        return kenlm.State()

    def full_score(self, sentence: str) -> float:
        return self._log10_to_ln(self.model.score(sentence, bos=True, eos=True))


def beam_search_decode(
    log_probs: torch.Tensor,
    vocab: Sequence[str],
    blank_idx: int = 0,
    beam_size: int = 32,
    lm: Optional[KenLMWordScorer] = None,
    alpha: float = 0.5,     # вес LM
    beta: float = 1.0,      # бонус за слово
    space_char: str = " ",
) -> List[str]:
    B, T, V = log_probs.shape
    space_idx = vocab.index(space_char)
    results: List[str] = []
    for b in range(B):
        logp = log_probs[b].cpu().numpy()
        beams: List[Beam] = [Beam(prefix=(), p_b=0.0, p_nb=-float("inf"), lm_score=0.0, n_words=0)]
        for t in range(T):
            # отберём top-k символов для скорости
            topk = logp[t].argsort()[-min(8, V):][::-1]
            new_beams = {}
            for beam in beams:
                for s in topk:
                    p = float(logp[t][s])
                    if s == blank_idx:
                        key = beam.prefix
                        nb = new_beams.get(key)
                        new_p_b = _logsumexp2(beam.p_b, beam.p_nb) + p
                        if nb is None:
                            nb = Beam(prefix=key,
                                      p_b=new_p_b,
                                      p_nb=-float("inf"),
                                      lm_score=beam.lm_score,
                                      n_words=beam.n_words)
                            new_beams[key] = nb
                        else:
                            nb.p_b = _logsumexp2(nb.p_b, new_p_b)
                    else:
                        last = beam.prefix[-1] if beam.prefix else -1
                        if s == last:
                            key_same = beam.prefix
                            nb = new_beams.get(key_same)
                            add_p = beam.p_nb + p
                            if nb is None:
                                nb = Beam(prefix=key_same, p_b=-float("inf"),
                                          p_nb=add_p,
                                          lm_score=beam.lm_score,
                                          n_words=beam.n_words)
                                new_beams[key_same] = nb
                            else:
                                nb.p_nb = _logsumexp2(nb.p_nb, add_p)

                            key_new = beam.prefix + (s,)
                            nb2 = new_beams.get(key_new)
                            add_p2 = beam.p_b + p
                            lm_add = 0.0
                            n_words_new = beam.n_words
                            if nb2 is None:
                                nb2 = Beam(prefix=key_new, p_b=-float("inf"),
                                           p_nb=add_p2,
                                           lm_score=beam.lm_score + lm_add,
                                           n_words=n_words_new)
                                new_beams[key_new] = nb2
                            else:
                                nb2.p_nb = _logsumexp2(nb2.p_nb, add_p2)
                        else:
                            key_new = beam.prefix + (s,)
                            nb = new_beams.get(key_new)
                            add_p = _logsumexp2(beam.p_b, beam.p_nb) + p

                            lm_add = 0.0
                            n_words_new = beam.n_words
                            if lm is not None and s == space_idx:
                                word = _last_word(beam.prefix, vocab, space_idx)
                                if word:
                                    prev_words = _prev_words(beam.prefix, vocab, space_idx)
                                    lm_add = lm.word_score(prev_words, word)
                                    n_words_new = beam.n_words + 1

                            if nb is None:
                                nb = Beam(prefix=key_new,
                                          p_b=-float("inf"),
                                          p_nb=add_p,
                                          lm_score=beam.lm_score + lm_add,
                                          n_words=n_words_new)
                                new_beams[key_new] = nb
                            else:
                                nb.p_nb = _logsumexp2(nb.p_nb, add_p)

            def _total(bm: Beam) -> float:
                acoustic = _logsumexp2(bm.p_b, bm.p_nb)
                return acoustic + alpha * bm.lm_score + beta * bm.n_words

            beams = sorted(new_beams.values(), key=_total, reverse=True)[:beam_size]

        best = None
        best_score = -float("inf")
        for bm in beams:
            acoustic = _logsumexp2(bm.p_b, bm.p_nb)
            lm_bonus = bm.lm_score
            n_words = bm.n_words
            if lm is not None:
                last_word = _last_word(bm.prefix, vocab, space_idx)
                if last_word:
                    prev_words = _prev_words(bm.prefix, vocab, space_idx)
                    lm_bonus = bm.lm_score + lm.word_score(prev_words, last_word)
                    n_words = bm.n_words + 1
            total = acoustic + alpha * lm_bonus + beta * n_words
            if total > best_score:
                best_score = total
                best = bm
        results.append("".join(vocab[i] for i in best.prefix) if best else "")
    return results


def _last_word(prefix: Tuple[int, ...], vocab: Sequence[str], space_idx: int) -> str:
    if not prefix:
        return ""
    end = len(prefix)
    start = end
    for i in range(end - 1, -1, -1):
        if prefix[i] == space_idx:
            break
        start = i
    return "".join(vocab[i] for i in prefix[start:end])


def _prev_words(prefix: Tuple[int, ...], vocab: Sequence[str], space_idx: int) -> List[str]:
    text = "".join(vocab[i] for i in prefix)
    parts = text.split(" ")
    if parts and parts[-1] != "":
        parts = parts[:-1]
    return [p for p in parts if p]
