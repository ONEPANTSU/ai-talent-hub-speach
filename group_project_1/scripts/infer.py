from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torchaudio

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.decoder import KenLMWordScorer, beam_search_decode, greedy_decode
from src.features import LogMelFilterBanks
from src.model import QuartznetASR
from src.text_norm import words_to_number
from src.dataset import _load_audio_any


def _build_pyctc_decoder(vocab, lm_path: Optional[str], alpha: float, beta: float):
    try:
        from pyctcdecode import build_ctcdecoder
    except Exception:
        return None
    labels = [""] + [c for c in vocab[1:]]  # blank заменяем на ""
    try:
        if lm_path and Path(lm_path).exists():
            decoder = build_ctcdecoder(labels, kenlm_model_path=lm_path, alpha=alpha, beta=beta)
        else:
            decoder = build_ctcdecoder(labels)
        return decoder
    except Exception as e:
        print(f"[!] pyctcdecode build failed: {e}. Falling back to custom beam.", flush=True)
        return None


def load_audio_16k(path: Path) -> torch.Tensor:
    wav, sr = _load_audio_any(path)
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    # peak-normalize
    peak = wav.abs().max().clamp_min(1e-6)
    wav = wav / peak * 0.9
    return wav


@torch.no_grad()
def batched_logprobs(
    model: QuartznetASR,
    mel: LogMelFilterBanks,
    wavs: List[torch.Tensor],
    device: str,
) -> List[torch.Tensor]:
    """Паддим батч и считаем log_softmax логиты, возвращаем список (T_i, V) без паддинга."""
    lens = [w.numel() for w in wavs]
    max_l = max(lens)
    B = len(wavs)
    batch = torch.zeros(B, max_l)
    for i, w in enumerate(wavs):
        batch[i, :w.numel()] = w
    batch = batch.to(device)
    mels = mel(batch)
    logits = model(mels)                           # (B, T', V)
    log_probs = torch.log_softmax(logits, dim=-1)  # (B, T', V)

    out = []
    for i in range(B):
        t_mel = lens[i] // 160 + 1
        t_out = (t_mel + 1) // 2
        t_out = min(t_out, log_probs.size(1))
        out.append(log_probs[i, :t_out, :].cpu())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lm", default=None, help="path to .arpa (optional)")
    ap.add_argument("--test_csv", required=True,
                    help="csv со списком файлов (напр. sample_submission.csv)")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--id_col", default="filename",
                    help="колонка с относительным путём к аудио в test_csv")
    ap.add_argument("--pred_col", default="transcription",
                    help="название колонки для предсказания в submission")
    ap.add_argument("--decode", choices=["greedy", "beam"], default="beam")
    ap.add_argument("--beam_size", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_mels", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    ck = torch.load(args.ckpt, map_location=device)
    vocab: List[str] = ck["vocab"]
    print("vocab:", vocab, flush=True)

    model = QuartznetASR(n_mels=args.n_mels, vocab_size=len(vocab)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    mel = LogMelFilterBanks(n_mels=args.n_mels).to(device)

    pyctc = None
    lm: Optional[KenLMWordScorer] = None
    if args.decode == "beam":
        pyctc = _build_pyctc_decoder(vocab, args.lm, args.alpha, args.beta)
        if pyctc is None:
            if args.lm and Path(args.lm).exists():
                try:
                    lm = KenLMWordScorer(args.lm)
                    print(f"Loaded KenLM (custom path): {args.lm}", flush=True)
                except Exception as e:
                    print(f"[!] KenLM load failed: {e}. No-LM beam.", flush=True)
        else:
            print(f"Using pyctcdecode beam (alpha={args.alpha} beta={args.beta} lm={bool(args.lm)})", flush=True)

    df = pd.read_csv(args.test_csv)
    print(f"test rows: {len(df)}", flush=True)
    data_root = Path(args.data_root)

    preds = []
    wavs_buf: List[torch.Tensor] = []
    idx_buf: List[int] = []

    def flush():
        if not wavs_buf:
            return
        lps = batched_logprobs(model, mel, wavs_buf, device)
        for i, lp in zip(idx_buf, lps):
            if args.decode == "greedy":
                hyp = greedy_decode(lp.unsqueeze(0), vocab, blank_idx=0)[0]
            elif pyctc is not None:
                hyp = pyctc.decode(lp.numpy(), beam_width=args.beam_size)
            else:
                hyp = beam_search_decode(
                    lp.unsqueeze(0), vocab, blank_idx=0,
                    beam_size=args.beam_size, lm=lm,
                    alpha=args.alpha, beta=args.beta,
                )[0]
            num = words_to_number(hyp)
            if num == 0:
                num = 1000
            preds.append((i, num, hyp))
        wavs_buf.clear()
        idx_buf.clear()

    for i, row in df.iterrows():
        fname = str(row[args.id_col])
        path = data_root / fname
        if not path.exists():
            alt = list(data_root.rglob(Path(fname).name))
            if alt:
                path = alt[0]
            else:
                print(f"[!] missing audio: {fname}", flush=True)
                preds.append((i, 1000, ""))
                continue
        wavs_buf.append(load_audio_16k(path))
        idx_buf.append(i)
        if len(wavs_buf) >= args.batch_size:
            flush()
            if (i + 1) % 100 == 0:
                print(f"  decoded {i+1}/{len(df)}", flush=True)
    flush()

    preds.sort(key=lambda x: x[0])
    out_df = df.copy()
    out_df[args.pred_col] = [p[1] for p in preds]
    keep = [c for c in out_df.columns if c in (args.id_col, args.pred_col)]
    if args.id_col not in keep:
        keep = [args.id_col] + keep
    out_df = out_df[keep]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  ({len(out_df)} rows)", flush=True)

    print("first 10 predictions:")
    for idx, (_, num, hyp) in enumerate(preds[:10]):
        print(f"  {df.iloc[idx][args.id_col]}: {num}   (hyp='{hyp}')")


if __name__ == "__main__":
    main()
