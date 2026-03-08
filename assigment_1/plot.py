import csv
from pathlib import Path

import torch
import torchaudio
import matplotlib.pyplot as plt

from melbanks import LogMelFilterBanks


PLOT_DIR = Path("assigment_1/plots")


def _read_metric(log_dir: str, metric: str) -> tuple[list[int], list[float]]:
    metrics_path = Path(log_dir) / "metrics.csv"
    steps, values = [], []
    if metrics_path.exists():
        with open(metrics_path) as f:
            for row in csv.DictReader(f):
                if row.get(metric):
                    steps.append(int(row["step"]))
                    values.append(float(row[metric]))
    return steps, values


def plot_melbanks_vs_torchaudio(signal):
    PLOT_DIR.mkdir(exist_ok=True)

    melspec = torchaudio.transforms.MelSpectrogram(hop_length=160, n_mels=80)(signal)
    ref = torch.log(melspec + 1e-6)

    ours = LogMelFilterBanks()(signal)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].imshow(ref.squeeze().detach().numpy(), aspect="auto", origin="lower")
    axes[0].set_title("torchaudio MelSpectrogram (log)")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Mel bin")

    axes[1].imshow(ours.squeeze().detach().numpy(), aspect="auto", origin="lower")
    axes[1].set_title("LogMelFilterBanks (ours)")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mel bin")

    diff = (ref - ours).abs().squeeze().detach().numpy()
    im = axes[2].imshow(diff, aspect="auto", origin="lower")
    axes[2].set_title(f"Absolute difference (max={diff.max():.2e})")
    axes[2].set_xlabel("Frame")
    axes[2].set_ylabel("Mel bin")
    fig.colorbar(im, ax=axes[2])

    fig.suptitle("LogMelFilterBanks vs torchaudio.transforms.MelSpectrogram", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "melbanks_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] melbanks comparison saved to {PLOT_DIR}/melbanks_comparison.png")


def plot_nmels_comparison(results: list[dict]):
    PLOT_DIR.mkdir(exist_ok=True)

    mels = [r["n_mels"] for r in results]
    accs = [r["test_acc"] for r in results]

    fig, ax = plt.subplots()
    bars = ax.bar([str(m) for m in mels], accs, color="steelblue")
    ax.bar_label(bars, fmt="%.3f")
    ax.set_xlabel("n_mels")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy vs n_mels")
    ax.set_ylim(0, 1.05)
    fig.savefig(PLOT_DIR / "nmels_vs_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    for r in results:
        steps, losses = _read_metric(r["log_dir"], "train_loss")
        if steps:
            ax.plot(steps, losses, label=f'n_mels={r["n_mels"]}')
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss Curves (varying n_mels)")
    ax.legend()
    fig.savefig(PLOT_DIR / "nmels_train_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plots] n_mels plots saved to {PLOT_DIR}/")


def plot_groups_comparison(results: list[dict]):
    PLOT_DIR.mkdir(exist_ok=True)

    labels = [str(r["groups"]) for r in results]

    epoch_times = []
    for r in results:
        _, times = _read_metric(r["log_dir"], "epoch_time")
        epoch_times.append(sum(times) / len(times) if times else 0)

    test_accs = [r["test_acc"] for r in results]
    params = [r["params"] for r in results]
    macs = [r["macs"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    bars = axes[0, 0].bar(labels, epoch_times, color="coral")
    axes[0, 0].bar_label(bars, fmt="%.1f")
    axes[0, 0].set_xlabel("groups")
    axes[0, 0].set_ylabel("Avg Epoch Time (s)")
    axes[0, 0].set_title("Epoch Time vs Groups")

    bars = axes[0, 1].bar(labels, params, color="mediumseagreen")
    axes[0, 1].bar_label(bars, fmt="%d")
    axes[0, 1].set_xlabel("groups")
    axes[0, 1].set_ylabel("Parameters")
    axes[0, 1].set_title("Params vs Groups\n(P = in*out*k / groups)")

    bars = axes[1, 0].bar(labels, macs, color="mediumpurple")
    axes[1, 0].bar_label(bars, fmt="%.0f")
    axes[1, 0].set_xlabel("groups")
    axes[1, 0].set_ylabel("MACs")
    axes[1, 0].set_title("FLOPs (MACs) vs Groups")

    bars = axes[1, 1].bar(labels, test_accs, color="steelblue")
    axes[1, 1].bar_label(bars, fmt="%.3f")
    axes[1, 1].set_xlabel("groups")
    axes[1, 1].set_ylabel("Test Accuracy")
    axes[1, 1].set_title("Test Accuracy vs Groups")
    axes[1, 1].set_ylim(0, 1.05)

    fig.suptitle("Group Convolution: efficiency vs quality trade-off", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "groups_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    for r in results:
        steps, losses = _read_metric(r["log_dir"], "train_loss")
        if steps:
            ax.plot(steps, losses, label=f'groups={r["groups"]}')
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Train Loss Curves (varying groups)")
    ax.legend()
    fig.savefig(PLOT_DIR / "groups_train_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plots] groups plots saved to {PLOT_DIR}/")