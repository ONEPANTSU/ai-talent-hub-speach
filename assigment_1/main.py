import warnings
import pytorch_lightning as pl

from utils import make_dataloaders, count_parameters, count_flops, print_layer_breakdown
from plot import  plot_nmels_comparison, plot_groups_comparison
from classifier import LitClassifier


warnings.filterwarnings("ignore")


def run_experiment(
    n_mels: int = 80,
    groups: int = 1,
    max_epochs: int = 10,
    batch_size: int = 64,
    num_workers: int = 4,
):
    train_dl, val_dl, test_dl = make_dataloaders(batch_size, num_workers)

    lit = LitClassifier(n_mels=n_mels, groups=groups)

    n_params = count_parameters(lit.model)
    macs, _ = count_flops(lit.model)
    print(f"\n--- n_mels={n_mels}, groups={groups} ---")
    print(f"  Total params : {n_params:,}")
    print(f"  Total MACs   : {macs:,.0f}")
    print_layer_breakdown(lit.model)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=pl.loggers.CSVLogger("./assigment_1/logs", name=f"nmels{n_mels}_g{groups}"),
    )
    trainer.fit(lit, train_dl, val_dl)
    test_result = trainer.test(lit, test_dl)

    return {
        "n_mels": n_mels,
        "groups": groups,
        "params": n_params,
        "macs": macs,
        "test_acc": test_result[0]["test_acc"],
        "log_dir": trainer.logger.log_dir,
    }



def main():
    MAX_EPOCHS = 5
    BATCH_SIZE = 64
    NUM_WORKERS = 4

    print("=" * 60)
    print("Experiment 1: varying n_mels (groups=1)")
    print("=" * 60)
    nmels_results = []
    for n_mels in [20, 40, 80]:
        res = run_experiment(
            n_mels=n_mels, groups=1,
            max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        )
        nmels_results.append(res)
    plot_nmels_comparison(nmels_results)


    groups_nmels = 80
    print("\n" + "=" * 60)
    print(f"Experiment 2: group convolution (n_mels={groups_nmels})")
    print("=" * 60)

    groups_results = []
    for g in [1, 2, 4, 8, 16]:
        res = run_experiment(
            n_mels=groups_nmels, groups=g,
            max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        )
        groups_results.append(res)
    plot_groups_comparison(groups_results)


    all_results = nmels_results + groups_results
    print("\n" + "=" * 70)
    print(f"  {'exp':<8} {'n_mels':>6} {'groups':>6} {'params':>8} "
          f"{'MACs':>12} {'test_acc':>9}")
    print("  " + "-" * 66)
    for r in all_results:
        exp = "nmels" if r in nmels_results else "groups"
        print(
            f"  {exp:<8} {r['n_mels']:>6} {r['groups']:>6} {r['params']:>8,} "
            f"{r['macs']:>12,.0f} {r['test_acc']:>9.4f}"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
