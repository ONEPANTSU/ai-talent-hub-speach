#!/usr/bin/env python3
"""Regenerate Task 7b figure from existing CSVs (no model run)."""
from pathlib import Path

import pandas as pd

from run_experiments import plot_task7b_wer_vs_temperature


def main() -> None:
    here = Path(__file__).resolve().parent
    t7b = pd.read_csv(here / "results_full" / "task7b_temperature_earnings.csv")
    t3 = pd.read_csv(here / "results_full" / "task3_temperature_sweep_librispeech.csv")
    libri_flat = float(t3["wer"].iloc[0])
    out = here / "results_full" / "task7b_wer_vs_temperature_earnings.png"
    plot_task7b_wer_vs_temperature(t7b, libri_flat, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
