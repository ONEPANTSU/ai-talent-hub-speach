"""
Тренировка Quartznet-CTC на русских спонсированных числах.

Запуск (Kaggle):
  python scripts/train.py \
      --train_csv /kaggle/input/<data>/train.csv \
      --dev_csv   /kaggle/input/<data>/dev.csv \
      --data_root /kaggle/input/<data> \
      --out_dir   /kaggle/working/ckpt \
      --epochs    40 \
      --batch_size 32 \
      --lr        3e-4

Каждую эпоху:
- считает train loss
- на dev считает greedy CER и WER (на числах-после-денормализации)
- сохраняет best чекпоинт по dev harmonic-mean CER по inD/ooD спикерам
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# локальный импорт пакета src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.augment import SpecAugment, WaveAugPipeline
from src.dataset import NumbersASRDataset, collate_fn
from src.decoder import greedy_decode
from src.features import LogMelFilterBanks
from src.model import QuartznetASR, count_params
from src.text_norm import number_to_words, vocab_chars, words_to_number


def cer(hyp: str, ref: str) -> float:
    """Character error rate (Levenshtein / len(ref))."""
    if not ref:
        return 1.0 if hyp else 0.0
    # DP Левенштейн
    prev = list(range(len(hyp) + 1))
    for j, cr in enumerate(ref, 1):
        curr = [j]
        for i, ch in enumerate(hyp, 1):
            curr.append(min(curr[-1] + 1, prev[i] + 1, prev[i - 1] + (0 if ch == cr else 1)))
        prev = curr
    return prev[-1] / len(ref)


def num_cer(pred_num: int, ref_num: int) -> float:
    """CER посчитанный на ЦИФРОВЫХ записях — это метрика задания."""
    return cer(str(pred_num), str(ref_num))


def build_vocab():
    """[blank] + символы; blank=0."""
    chars = vocab_chars()
    return ["<blank>"] + chars


class WarmupCosine:
    def __init__(self, optimizer, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.opt = optimizer
        self.base_lr = base_lr
        self.warm = max(1, warmup_steps)
        self.total = total_steps
        self.min_lr = min_lr
        self.step_n = 0

    def step(self):
        self.step_n += 1
        if self.step_n <= self.warm:
            lr = self.base_lr * self.step_n / self.warm
        else:
            t = (self.step_n - self.warm) / max(1, self.total - self.warm)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * min(1.0, t)))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


@torch.no_grad()
def evaluate(model, mel, dev_loader, vocab, device) -> dict:
    model.eval()
    per_spk = {}          # spk_id -> {"cer_sum": float, "n": int, "num_err": int, "exact": int}
    total_cer = 0.0
    total_n = 0
    exact = 0
    for batch in dev_loader:
        wavs = batch["wavs"].to(device)
        wav_lens = batch["wav_lens"].to(device)
        mels = mel(wavs)                                  # (B, n_mels, T)
        logits = model(mels)                              # (B, T', V)
        log_probs = torch.log_softmax(logits, dim=-1)
        hyps = greedy_decode(log_probs, vocab, blank_idx=0)

        for hyp, text, num, spk in zip(hyps, batch["texts"], batch["nums"], batch["spks"]):
            pred_num = words_to_number(hyp)
            c = num_cer(pred_num, num)
            total_cer += c
            total_n += 1
            if pred_num == num:
                exact += 1
            if spk not in per_spk:
                per_spk[spk] = {"cer_sum": 0.0, "n": 0, "exact": 0}
            per_spk[spk]["cer_sum"] += c
            per_spk[spk]["n"] += 1
            per_spk[spk]["exact"] += int(pred_num == num)

    mean_cer = total_cer / max(1, total_n)
    per_spk_cer = {s: v["cer_sum"] / v["n"] for s, v in per_spk.items()}
    # гармоническое среднее по спикерам — прокси к kaggle-метрике
    if per_spk_cer:
        vals = [max(v, 1e-6) for v in per_spk_cer.values()]
        hmean = len(vals) / sum(1 / v for v in vals)
    else:
        hmean = float("inf")

    return {
        "mean_cer": mean_cer,
        "exact_acc": exact / max(1, total_n),
        "per_spk_cer": per_spk_cer,
        "hmean_cer": hmean,
        "n": total_n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--dev_csv", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--noise_dir", default=None, help="Папка с шумами для BG-noise aug (опционально)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=15.0)
    ap.add_argument("--amp", action="store_true", help="mixed precision")
    ap.add_argument("--n_mels", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    vocab = build_vocab()
    print("vocab:", vocab, "size:", len(vocab), flush=True)
    # сохраняем vocab — нужен при инференсе
    with open(out_dir / "vocab.json", "w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    wave_aug = WaveAugPipeline(
        sample_rate=16000,
        noise_dir=args.noise_dir,
        use_speed=True, use_gain=True, use_gauss=True, use_bg_noise=args.noise_dir is not None,
    )

    train_ds = NumbersASRDataset(args.train_csv, args.data_root, vocab, wave_aug=wave_aug)
    dev_ds = NumbersASRDataset(args.dev_csv, args.data_root, vocab, wave_aug=None)
    print(f"train: {len(train_ds)}  dev: {len(dev_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
    )

    mel = LogMelFilterBanks(n_mels=args.n_mels).to(device)
    spec_aug = SpecAugment(freq_masks=2, freq_mask_param=15, time_masks=2, time_mask_param=40).to(device)
    model = QuartznetASR(n_mels=args.n_mels, vocab_size=len(vocab), dropout=args.dropout).to(device)
    print(f"model params: {count_params(model):,}  (limit 5,000,000)", flush=True)
    assert count_params(model) <= 5_000_000, "Model exceeds 5M parameter limit!"

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = WarmupCosine(optimizer, args.lr, int(total_steps * args.warmup_ratio), total_steps)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")

    start_epoch = 0
    best_hmean = float("inf")
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optim"])
        scheduler.step_n = ck.get("step_n", 0)
        start_epoch = ck.get("epoch", 0)
        best_hmean = ck.get("best_hmean", float("inf"))
        print(f"resumed from {args.resume}, epoch={start_epoch}, best_hmean={best_hmean:.4f}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0
        n_batches = 0
        for step, batch in enumerate(train_loader):
            wavs = batch["wavs"].to(device, non_blocking=True)
            wav_lens = batch["wav_lens"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            tgt_lens = batch["target_lens"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp and device == "cuda"):
                mels = mel(wavs)
                mels = spec_aug(mels)
                logits = model(mels)                     # (B, T', V)
                log_probs = torch.log_softmax(logits, dim=-1)
                # CTCLoss ждёт (T, B, V)
                log_probs_t = log_probs.transpose(0, 1)
                # длины вывода: T' = T/2 (из-за C1 stride=2)
                feat_lens = (wav_lens // 160 + 1)           # число hop'ов в мел-спеке
                out_lens = model.output_lengths(feat_lens)
                out_lens = out_lens.clamp(max=log_probs.size(1))
                loss = ctc(log_probs_t, targets, out_lens, tgt_lens)

            if torch.isnan(loss) or torch.isinf(loss):
                # пропускаем батч, но не прерываемся
                print(f"  [!] skipping NaN/Inf batch at epoch={epoch} step={step}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += loss.item()
            n_batches += 1

            if step % 50 == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"  ep{epoch} step{step}/{len(train_loader)} loss={loss.item():.3f} lr={lr:.2e}", flush=True)

        train_loss = running / max(1, n_batches)
        metrics = evaluate(model, mel, dev_loader, vocab, device)
        dt = time.time() - t0
        print(
            f"[epoch {epoch}] train_loss={train_loss:.3f} "
            f"dev_cer={metrics['mean_cer']:.4f} dev_exact={metrics['exact_acc']:.4f} "
            f"hmean_cer={metrics['hmean_cer']:.4f} "
            f"per_spk={ {k: round(v,3) for k, v in metrics['per_spk_cer'].items()} } "
            f"({dt:.1f}s)",
            flush=True,
        )

        # сохраняем last + best
        ck = {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "step_n": scheduler.step_n,
            "epoch": epoch + 1,
            "best_hmean": best_hmean,
            "vocab": vocab,
            "args": vars(args),
            "metrics": metrics,
        }
        torch.save(ck, out_dir / "last.ckpt")
        if metrics["hmean_cer"] < best_hmean:
            best_hmean = metrics["hmean_cer"]
            ck["best_hmean"] = best_hmean
            torch.save(ck, out_dir / "best.ckpt")
            print(f"  [*] new best hmean_cer={best_hmean:.4f}  -> saved best.ckpt", flush=True)

    print("Training done. Best hmean_cer:", best_hmean)


if __name__ == "__main__":
    main()
