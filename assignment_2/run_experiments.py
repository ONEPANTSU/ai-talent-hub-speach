import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import jiwer
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchaudio
from tqdm.auto import tqdm

from wav2vec2decoder import Wav2Vec2Decoder


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _resolve_audio_path(manifest_path: Path, audio_field: str) -> Path:
    raw = Path(audio_field)
    if raw.is_absolute() and raw.exists():
        return raw.resolve()

    candidates = [
        manifest_path.parent / raw,
        manifest_path.parent.parent / raw,
        manifest_path.parent.parent.parent / raw,
        Path.cwd() / raw,
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    return (manifest_path.parent / raw).resolve()


def _read_manifest(manifest_path: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    suffix = manifest_path.suffix.lower()
    base_dir = manifest_path.parent

    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        with manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=sep)
            for row in reader:
                text = row.get("text") or row.get("transcript") or row.get("sentence")
                audio = row.get("audio") or row.get("path") or row.get("wav")
                if not text or not audio:
                    continue
                audio_path = _resolve_audio_path(manifest_path, audio)
                samples.append((audio_path, _normalize_text(text)))
    elif suffix == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text") or obj.get("transcript") or obj.get("sentence")
                audio = obj.get("audio") or obj.get("path") or obj.get("wav")
                if not text or not audio:
                    continue
                audio_path = _resolve_audio_path(manifest_path, audio)
                samples.append((audio_path, _normalize_text(text)))

    return samples


def _read_librispeech_trans_txt(data_dir: Path) -> List[Tuple[Path, str]]:
    samples: List[Tuple[Path, str]] = []
    for trans in data_dir.rglob("*.trans.txt"):
        with trans.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                key, text = line.split(" ", 1)
                wav_path = trans.parent / f"{key}.wav"
                if wav_path.exists():
                    samples.append((wav_path.resolve(), _normalize_text(text)))
    return samples


def load_samples(data_dir: str) -> List[Tuple[Path, str]]:
    root = Path(data_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")

    # Strategy 1: explicit manifest files
    manifest_names = [
        "manifest.csv",
        "manifest.tsv",
        "metadata.csv",
        "metadata.tsv",
        "test.csv",
        "test.tsv",
        "test.jsonl",
    ]
    for name in manifest_names:
        manifest = root / name
        if manifest.exists():
            samples = _read_manifest(manifest)
            if samples:
                return samples

    # Strategy 2: LibriSpeech style transcripts
    ls_samples = _read_librispeech_trans_txt(root)
    if ls_samples:
        return ls_samples

    # Strategy 3: wav + same-stem txt
    samples: List[Tuple[Path, str]] = []
    for wav in root.rglob("*.wav"):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            continue
        text = _normalize_text(txt.read_text(encoding="utf-8"))
        samples.append((wav.resolve(), text))

    if not samples:
        raise RuntimeError(
            f"Could not infer transcript format in {root}. "
            "Provide manifest.(csv/tsv/jsonl) or wav+txt pairs."
        )
    return sorted(samples, key=lambda x: str(x[0]))


def resolve_existing_path(path_like: str) -> str:
    p = Path(path_like)
    if p.exists():
        return str(p.resolve())
    alt = Path("assignment_2") / p
    if alt.exists():
        return str(alt.resolve())
    return str(p)


def decode_dataset(
    decoder: Wav2Vec2Decoder,
    samples: Iterable[Tuple[Path, str]],
    method: str,
    max_samples: int | None = None,
) -> Dict[str, float]:
    refs: List[str] = []
    hyps: List[str] = []

    for idx, (audio_path, ref) in enumerate(samples):
        if max_samples is not None and idx >= max_samples:
            break
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        hyp = _normalize_text(decoder.decode(audio, method=method))
        refs.append(ref)
        hyps.append(hyp)

    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "num_samples": len(refs),
    }


def precompute_logits(
    samples: Iterable[Tuple[Path, str]],
    model_name: str = "facebook/wav2vec2-base-100h",
    max_samples: int | None = None,
) -> Tuple[List[torch.Tensor], List[str]]:
    bootstrap = Wav2Vec2Decoder(model_name=model_name, lm_model_path=None, beam_width=1)
    logits_list: List[torch.Tensor] = []
    refs: List[str] = []

    for idx, (audio_path, ref) in enumerate(
        tqdm(samples, desc="Precompute logits", unit="sample")
    ):
        if max_samples is not None and idx >= max_samples:
            break
        audio, sr = torchaudio.load(audio_path)
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        inputs = bootstrap.processor(audio, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = bootstrap.model(inputs.input_values.squeeze(0)).logits[0].cpu()
        logits_list.append(logits)
        refs.append(ref)
    return logits_list, refs


def decode_logits_dataset(
    decoder: Wav2Vec2Decoder,
    logits_list: List[torch.Tensor],
    refs: List[str],
    method: str,
) -> Dict[str, float]:
    hyps: List[str] = []
    for logits in tqdm(logits_list, desc=f"Decode [{method}]", unit="sample", leave=False):
        scaled = logits / decoder.temperature
        if method == "greedy":
            hyp = decoder.greedy_decode(scaled)
        elif method == "beam":
            hyp = decoder.beam_search_decode(scaled)
        elif method == "beam_lm":
            hyp = decoder.beam_search_with_lm(scaled)
        elif method == "beam_lm_rescore":
            beams = decoder.beam_search_decode(scaled, return_beams=True)
            hyp = decoder.lm_rescore(beams)
        else:
            raise ValueError(f"Unknown method: {method}")
        hyps.append(_normalize_text(hyp))

    return {
        "wer": jiwer.wer(refs, hyps),
        "cer": jiwer.cer(refs, hyps),
        "num_samples": len(refs),
    }


def save_df(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def plot_heatmap(df: pd.DataFrame, title: str, output_path: Path) -> None:
    pivot = df.pivot(index="beta", columns="alpha", values="wer")
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])
    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="WER")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_bar(df: pd.DataFrame, x: str, y: str, hue: str, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    pivot = df.pivot_table(index=x, columns=hue, values=y, aggfunc="mean")
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel(y.upper())
    ax.grid(axis="y", alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_task7b_wer_vs_temperature(
    task7b_df: pd.DataFrame,
    libri_greedy_wer_flat: float,
    output_path: Path,
    title: str = "Task 7b: WER vs temperature (Earnings22)",
) -> None:
    """WER vs T for greedy and beam+LM on OOD data; horizontal line = Libri greedy plateau."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    pivot = task7b_df.pivot(index="temperature", columns="method", values="wer").sort_index()
    temps = list(pivot.index)
    ax.plot(temps, pivot["greedy"].values, marker="o", linewidth=2, label="Greedy (Earnings22)")
    ax.plot(temps, pivot["beam_lm"].values, marker="s", linewidth=2, label="Beam + 3-gram shallow fusion (Earnings22)")
    ax.axhline(
        libri_greedy_wer_flat,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"Greedy LibriSpeech (flat vs T, WER≈{libri_greedy_wer_flat:.3f})",
    )
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("WER")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_dir", default="data/librispeech_test_other")
    parser.add_argument("--earnings_dir", default="data/earnings22_test")
    parser.add_argument("--base_lm", default="lm/3-gram.pruned.1e-7.arpa.gz")
    parser.add_argument("--financial_lm", default="lm/financial-3gram.arpa.gz")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--beam_width", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    args.base_lm = resolve_existing_path(args.base_lm)
    args.financial_lm = resolve_existing_path(args.financial_lm)
    libri_samples = load_samples(args.librispeech_dir)
    earnings_samples = load_samples(args.earnings_dir)
    libri_logits, libri_refs = precompute_logits(libri_samples, max_samples=args.max_samples)
    earnings_logits, earnings_refs = precompute_logits(earnings_samples, max_samples=args.max_samples)

    pbar_tasks = tqdm(total=7, desc="Experiment tasks", unit="task")

    # Task 1/2 baseline methods
    baseline_rows = []
    for method in ["greedy", "beam"]:
        decoder = Wav2Vec2Decoder(
            lm_model_path=None,
            beam_width=args.beam_width,
            alpha=0.0,
            beta=0.0,
            temperature=1.0,
        )
        metrics = decode_logits_dataset(decoder, libri_logits, libri_refs, method=method)
        baseline_rows.append({"dataset": "librispeech_test_other", "method": method, **metrics})
    baseline_df = pd.DataFrame(baseline_rows)
    save_df(baseline_df, out / "task1_task2_baseline.csv")
    pbar_tasks.update(1)

    # Task 2: beam width sweep
    beam_rows = []
    for bw in [1, 3, 10, 50]:
        decoder = Wav2Vec2Decoder(lm_model_path=None, beam_width=bw)
        m = decode_logits_dataset(decoder, libri_logits, libri_refs, method="beam")
        beam_rows.append({"beam_width": bw, **m})
    beam_df = pd.DataFrame(beam_rows)
    save_df(beam_df, out / "task2_beam_width_sweep.csv")
    pbar_tasks.update(1)

    # Task 3: temperature sweep with greedy
    temp_rows = []
    for t in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        decoder = Wav2Vec2Decoder(lm_model_path=None, beam_width=1, temperature=t)
        m = decode_logits_dataset(decoder, libri_logits, libri_refs, method="greedy")
        temp_rows.append({"temperature": t, **m})
    temp_df = pd.DataFrame(temp_rows)
    save_df(temp_df, out / "task3_temperature_sweep_librispeech.csv")
    pbar_tasks.update(1)

    # Task 4/6: alpha-beta sweep for shallow fusion + rescoring
    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
    betas = [0.0, 0.5, 1.0, 1.5]

    sf_rows = []
    rs_rows = []
    for a in alphas:
        for b in betas:
            decoder = Wav2Vec2Decoder(
                lm_model_path=args.base_lm,
                beam_width=args.beam_width,
                alpha=a,
                beta=b,
            )
            sf = decode_logits_dataset(decoder, libri_logits, libri_refs, method="beam_lm")
            rs = decode_logits_dataset(decoder, libri_logits, libri_refs, method="beam_lm_rescore")
            sf_rows.append({"alpha": a, "beta": b, **sf})
            rs_rows.append({"alpha": a, "beta": b, **rs})

    sf_df = pd.DataFrame(sf_rows)
    rs_df = pd.DataFrame(rs_rows)
    save_df(sf_df, out / "task4_shallow_fusion_grid.csv")
    save_df(rs_df, out / "task6_rescoring_grid.csv")
    plot_heatmap(sf_df, "Task 4: Shallow Fusion WER", out / "task4_shallow_fusion_heatmap.png")
    plot_heatmap(rs_df, "Task 6: Rescoring WER", out / "task6_rescoring_heatmap.png")
    pbar_tasks.update(1)

    best_sf = sf_df.sort_values("wer", ascending=True).iloc[0]
    best_rs = rs_df.sort_values("wer", ascending=True).iloc[0]

    # Task 7: compare 4 methods on both domains (best SF/RS on libri)
    compare_rows = []
    methods = [
        ("greedy", {"lm_model_path": None, "alpha": 0.0, "beta": 0.0}),
        ("beam", {"lm_model_path": None, "alpha": 0.0, "beta": 0.0}),
        ("beam_lm", {"lm_model_path": args.base_lm, "alpha": float(best_sf["alpha"]), "beta": float(best_sf["beta"])}),
        ("beam_lm_rescore", {"lm_model_path": args.base_lm, "alpha": float(best_rs["alpha"]), "beta": float(best_rs["beta"])}),
    ]
    for dataset_name, logits_set, refs_set in [
        ("librispeech_test_other", libri_logits, libri_refs),
        ("earnings22_test", earnings_logits, earnings_refs),
    ]:
        for method, cfg in methods:
            decoder = Wav2Vec2Decoder(
                lm_model_path=cfg["lm_model_path"],
                beam_width=args.beam_width,
                alpha=cfg["alpha"],
                beta=cfg["beta"],
            )
            m = decode_logits_dataset(decoder, logits_set, refs_set, method=method)
            compare_rows.append({"dataset": dataset_name, "method": method, **m})
    compare_df = pd.DataFrame(compare_rows)
    save_df(compare_df, out / "task7_domain_shift_comparison.csv")
    pbar_tasks.update(1)

    # Task 7b: temperature sweep on earnings for greedy + best shallow-fusion
    temp7b_rows = []
    for t in [0.5, 1.0, 1.5, 2.0]:
        greedy_decoder = Wav2Vec2Decoder(lm_model_path=None, beam_width=1, temperature=t)
        greedy_m = decode_logits_dataset(greedy_decoder, earnings_logits, earnings_refs, "greedy")
        temp7b_rows.append({"method": "greedy", "temperature": t, **greedy_m})

        sf_decoder = Wav2Vec2Decoder(
            lm_model_path=args.base_lm,
            beam_width=args.beam_width,
            alpha=float(best_sf["alpha"]),
            beta=float(best_sf["beta"]),
            temperature=t,
        )
        sf_m = decode_logits_dataset(sf_decoder, earnings_logits, earnings_refs, "beam_lm")
        temp7b_rows.append({"method": "beam_lm", "temperature": t, **sf_m})
    temp7b_df = pd.DataFrame(temp7b_rows)
    save_df(temp7b_df, out / "task7b_temperature_earnings.csv")
    libri_greedy_flat = float(temp_df["wer"].iloc[0])
    plot_task7b_wer_vs_temperature(
        temp7b_df,
        libri_greedy_flat,
        out / "task7b_wer_vs_temperature_earnings.png",
    )
    pbar_tasks.update(1)

    # Task 9: compare both LMs with best methods on both domains
    lm_rows = []
    lm_cfgs = [("librispeech_3gram", args.base_lm)]
    if Path(args.financial_lm).exists():
        lm_cfgs.append(("financial_3gram", args.financial_lm))
    else:
        print(f"Skipping financial LM comparison: file not found at {args.financial_lm}")
    for lm_name, lm_path in lm_cfgs:
        for dataset_name, logits_set, refs_set in [
            ("librispeech_test_other", libri_logits, libri_refs),
            ("earnings22_test", earnings_logits, earnings_refs),
        ]:
            for method, best_cfg in [("beam_lm", best_sf), ("beam_lm_rescore", best_rs)]:
                decoder = Wav2Vec2Decoder(
                    lm_model_path=lm_path,
                    beam_width=args.beam_width,
                    alpha=float(best_cfg["alpha"]),
                    beta=float(best_cfg["beta"]),
                )
                m = decode_logits_dataset(decoder, logits_set, refs_set, method=method)
                lm_rows.append(
                    {
                        "dataset": dataset_name,
                        "lm": lm_name,
                        "method": method,
                        **m,
                    }
                )
    lm_df = pd.DataFrame(lm_rows)
    save_df(lm_df, out / "task9_lm_comparison.csv")
    plot_bar(lm_df, x="dataset", y="wer", hue="lm", title="Task 9 WER by Domain and LM", output_path=out / "task9_wer_bar.png")
    plot_bar(lm_df, x="dataset", y="cer", hue="lm", title="Task 9 CER by Domain and LM", output_path=out / "task9_cer_bar.png")
    pbar_tasks.update(1)
    pbar_tasks.close()

    print(f"Saved all experiment artifacts to: {out.resolve()}")


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    main()
