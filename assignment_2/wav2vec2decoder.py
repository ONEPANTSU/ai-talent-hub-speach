import math
import heapq
from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    _processor_cache = {}
    _model_cache = {}
    _lm_cache = {}

    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
        ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        if model_name not in self._processor_cache:
            self._processor_cache[model_name] = Wav2Vec2Processor.from_pretrained(model_name)
        if model_name not in self._model_cache:
            self._model_cache[model_name] = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.processor = self._processor_cache[model_name]
        self.model = self._model_cache[model_name]

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        if lm_model_path:
            if lm_model_path not in self._lm_cache:
                self._lm_cache[lm_model_path] = kenlm.Model(lm_model_path)
            self.lm_model = self._lm_cache[lm_model_path]
        else:
            self.lm_model = None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        best_path = torch.argmax(log_probs, dim=-1).tolist()

        decoded_ids: List[int] = []
        prev = None
        for token_id in best_path:
            if token_id == self.blank_token_id:
                prev = token_id
                continue
            if token_id != prev:
                decoded_ids.append(token_id)
            prev = token_id

        return self._ids_to_text(decoded_ids)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        time_steps, vocab_size = log_probs.shape

        # Prefix beam search:
        # beams[prefix] = (log p(prefix, ends_with_blank), log p(prefix, ends_nonblank))
        beams = {(): (0.0, float("-inf"))}

        for t in range(time_steps):
            frame = log_probs[t].tolist()
            next_beams = {}

            for prefix, (p_b, p_nb) in beams.items():
                prefix_total = _log_add(p_b, p_nb)

                # Extend with blank (prefix unchanged)
                blank_lp = frame[self.blank_token_id]
                n_b, n_nb = next_beams.get(prefix, (float("-inf"), float("-inf")))
                n_b = _log_add(n_b, prefix_total + blank_lp)
                next_beams[prefix] = (n_b, n_nb)

                last_token = prefix[-1] if prefix else None

                for c in range(vocab_size):
                    if c == self.blank_token_id:
                        continue

                    c_lp = frame[c]
                    if c == last_token:
                        # Repeating char without blank: keep same prefix from non-blank path
                        n_b, n_nb = next_beams.get(prefix, (float("-inf"), float("-inf")))
                        n_nb = _log_add(n_nb, p_nb + c_lp)
                        next_beams[prefix] = (n_b, n_nb)

                        # From blank path, we can append repeated token as a new symbol
                        new_prefix = prefix + (c,)
                        n_b2, n_nb2 = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                        n_nb2 = _log_add(n_nb2, p_b + c_lp)
                        next_beams[new_prefix] = (n_b2, n_nb2)
                    else:
                        new_prefix = prefix + (c,)
                        n_b, n_nb = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                        n_nb = _log_add(n_nb, prefix_total + c_lp)
                        next_beams[new_prefix] = (n_b, n_nb)

            # Keep only top-N beams by total log-prob
            top_items = heapq.nlargest(
                self.beam_width,
                next_beams.items(),
                key=lambda item: _log_add(item[1][0], item[1][1]),
            )
            beams = dict(top_items)

        ranked = sorted(
            (
                (list(prefix), _log_add(scores[0], scores[1]))
                for prefix, scores in beams.items()
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        if return_beams:
            return ranked
        return self._ids_to_text(ranked[0][0] if ranked else [])

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")
        log_probs = torch.log_softmax(logits, dim=-1)
        time_steps, vocab_size = log_probs.shape

        beams = {(): (0.0, float("-inf"))}

        for t in range(time_steps):
            frame = log_probs[t].tolist()
            next_beams = {}

            for prefix, (p_b, p_nb) in beams.items():
                prefix_total = _log_add(p_b, p_nb)

                blank_lp = frame[self.blank_token_id]
                n_b, n_nb = next_beams.get(prefix, (float("-inf"), float("-inf")))
                n_b = _log_add(n_b, prefix_total + blank_lp)
                next_beams[prefix] = (n_b, n_nb)

                last_token = prefix[-1] if prefix else None
                for c in range(vocab_size):
                    if c == self.blank_token_id:
                        continue

                    c_lp = frame[c]
                    if c == last_token:
                        n_b, n_nb = next_beams.get(prefix, (float("-inf"), float("-inf")))
                        n_nb = _log_add(n_nb, p_nb + c_lp)
                        next_beams[prefix] = (n_b, n_nb)

                        new_prefix = prefix + (c,)
                        n_b2, n_nb2 = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                        n_nb2 = _log_add(n_nb2, p_b + c_lp)
                        next_beams[new_prefix] = (n_b2, n_nb2)
                    else:
                        new_prefix = prefix + (c,)
                        n_b, n_nb = next_beams.get(new_prefix, (float("-inf"), float("-inf")))
                        n_nb = _log_add(n_nb, prefix_total + c_lp)
                        next_beams[new_prefix] = (n_b, n_nb)

            # Apply LM shallow fusion only for pruning/ranking
            def fused_score(item):
                prefix, (s_b, s_nb) = item
                acoustic = _log_add(s_b, s_nb)
                text = self._ids_to_text(list(prefix))
                lm = self.lm_model.score(text, bos=False, eos=False) * math.log(10)
                words = len(text.split()) if text else 0
                return acoustic + self.alpha * lm + self.beta * words

            top_items = heapq.nlargest(self.beam_width, next_beams.items(), key=fused_score)
            beams = dict(top_items)

        best_prefix = max(
            beams.items(),
            key=lambda item: (
                _log_add(item[1][0], item[1][1])
                + self.alpha * self.lm_model.score(self._ids_to_text(list(item[0])), bos=False, eos=False) * math.log(10)
                + self.beta * (len(self._ids_to_text(list(item[0])).split()) if item[0] else 0)
            ),
        )[0]
        return self._ids_to_text(list(best_prefix))

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")
        best_text = ""
        best_score = float("-inf")

        for token_ids, acoustic_score in beams:
            text = self._ids_to_text(token_ids)
            lm_score = self.lm_model.score(text, bos=False, eos=False) * math.log(10)
            num_words = len(text.split()) if text else 0
            total = acoustic_score + self.alpha * lm_score + self.beta * num_words
            if total > best_score:
                best_score = total
                best_text = text

        return best_text

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------

def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    # Reference transcripts are lowercase to match the evaluation manifests.
    # examples/ clips are for quick debugging only — use data/librispeech_test_other/
    # and data/earnings22_test/ for all reported metrics.
    test_samples = [
        ("examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
        ("examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
        ("examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
        ("examples/sample6.wav", "at this time all participants are in a listen only mode"),
        ("examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
        ("examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    ]

    decoder = Wav2Vec2Decoder(lm_model_path=None)  # set lm_model_path for Tasks 4+

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)