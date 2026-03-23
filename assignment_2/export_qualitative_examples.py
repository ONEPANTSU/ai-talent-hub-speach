#!/usr/bin/env python3
"""
Task 6: export 5–10 LibriSpeech examples where LM changes the hypothesis vs plain beam.

Reads best (alpha, beta) for shallow fusion and rescoring from results_full grids.
Writes results_full/task6_qualitative_examples.md and .csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from run_experiments import (
    _normalize_text,
    load_samples,
    precompute_logits,
    resolve_existing_path,
)
from wav2vec2decoder import Wav2Vec2Decoder


def _per_sample_hyps(decoder: Wav2Vec2Decoder, logits_list: list, method: str) -> list[str]:
    hyps: list[str] = []
    for logits in logits_list:
        scaled = logits / decoder.temperature
        if method == "beam":
            hyp = decoder.beam_search_decode(scaled)
        elif method == "beam_lm":
            hyp = decoder.beam_search_with_lm(scaled)
        elif method == "beam_lm_rescore":
            beams = decoder.beam_search_decode(scaled, return_beams=True)
            hyp = decoder.lm_rescore(beams)
        else:
            raise ValueError(method)
        hyps.append(_normalize_text(hyp))
    return hyps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--librispeech_dir", default="data/librispeech_test_other")
    p.add_argument("--base_lm", default="lm/3-gram.pruned.1e-7.arpa.gz")
    p.add_argument("--results_full", default="results_full")
    p.add_argument("--beam_width", type=int, default=10)
    p.add_argument("--max_samples", type=int, default=200)
    p.add_argument("--num_examples", type=int, default=8)
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    rf = here / args.results_full
    sf_grid = pd.read_csv(rf / "task4_shallow_fusion_grid.csv")
    rs_grid = pd.read_csv(rf / "task6_rescoring_grid.csv")
    best_sf = sf_grid.sort_values("wer").iloc[0]
    best_rs = rs_grid.sort_values("wer").iloc[0]

    lm_path = resolve_existing_path(args.base_lm)
    libri_dir = str(here / args.librispeech_dir) if not Path(args.librispeech_dir).is_absolute() else args.librispeech_dir
    samples = load_samples(libri_dir)[: args.max_samples]
    logits_list, refs = precompute_logits(samples, max_samples=args.max_samples)

    dec_beam = Wav2Vec2Decoder(
        lm_model_path=None, beam_width=args.beam_width, alpha=0.0, beta=0.0, temperature=1.0
    )
    dec_sf = Wav2Vec2Decoder(
        lm_model_path=lm_path,
        beam_width=args.beam_width,
        alpha=float(best_sf["alpha"]),
        beta=float(best_sf["beta"]),
        temperature=1.0,
    )
    dec_rs = Wav2Vec2Decoder(
        lm_model_path=lm_path,
        beam_width=args.beam_width,
        alpha=float(best_rs["alpha"]),
        beta=float(best_rs["beta"]),
        temperature=1.0,
    )

    h_beam = _per_sample_hyps(dec_beam, logits_list, "beam")
    h_sf = _per_sample_hyps(dec_sf, logits_list, "beam_lm")
    h_rs = _per_sample_hyps(dec_rs, logits_list, "beam_lm_rescore")

    picked: list[int] = []
    for i in range(len(refs)):
        if h_beam[i] != h_sf[i] or h_beam[i] != h_rs[i] or h_sf[i] != h_rs[i]:
            picked.append(i)
        if len(picked) >= args.num_examples:
            break

    rows = []
    lines = [
        "# Task 6 — qualitative examples (LibriSpeech test-other)",
        "",
        f"Best shallow fusion: α={best_sf['alpha']}, β={best_sf['beta']} (WER={best_sf['wer']:.4f}).",
        f"Best rescoring: α={best_rs['alpha']}, β={best_rs['beta']} (WER={best_rs['wer']:.4f}).",
        "",
    ]

    for rank, i in enumerate(picked, start=1):
        wav = samples[i][0]
        rows.append(
            {
                "idx": i,
                "audio": str(wav),
                "ref": refs[i],
                "beam": h_beam[i],
                "shallow_fusion": h_sf[i],
                "rescore": h_rs[i],
            }
        )
        lines.append(f"## Example {rank} (`{wav.name}`)")
        lines.append("")
        lines.append(f"- **REF:**  {refs[i]}")
        lines.append(f"- **BEAM:** {h_beam[i]}")
        lines.append(f"- **SF:**   {h_sf[i]}")
        lines.append(f"- **RS:**   {h_rs[i]}")
        lines.append("")

    if not picked:
        lines.append("_No differing hypotheses found in scanned subset; increase max_samples or check decoders._")
        lines.append("")

    md_path = rf / "task6_qualitative_examples.md"
    csv_path = rf / "task6_qualitative_examples.csv"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Wrote {md_path} and {csv_path} ({len(picked)} examples)")


if __name__ == "__main__":
    torch.set_num_threads(max(1, torch.get_num_threads() // 2))
    main()
