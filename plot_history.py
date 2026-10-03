"""Plot train/val loss curves from a run's history.csv.

Usage:
    python plot_history.py saved_models/<run1> [saved_models/<run2> ...]

Reads the per-epoch history.csv written by Trainer.fit() (epoch, train_loss,
val_loss, metrics, lr, elapsed_seconds), draws train/val loss curves, marks the
best-val epoch, and writes history_plot.png next to the csv.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+", help="run dirs containing history.csv")
    args = p.parse_args()

    for d in args.run_dirs:
        path = os.path.join(d, "history.csv")
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        if not rows:
            print(f"SKIP {path}: no rows")
            continue
        epochs = [int(r["epoch"]) for r in rows]
        best = min(rows, key=lambda r: float(r["val_loss"]))

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, [float(r["train_loss"]) for r in rows], label="train_loss")
        plt.plot(epochs, [float(r["val_loss"]) for r in rows], label="val_loss")
        plt.axvline(int(best["epoch"]), color="gray", ls="--", alpha=0.6)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.yscale("log")
        plt.legend()
        plt.title(os.path.basename(d.rstrip("/\\")) +
                  f"  best val_loss={float(best['val_loss']):.4f} @ ep {best['epoch']}")
        plt.tight_layout()
        out = os.path.join(d, "history_plot.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print(f"{out}: best epoch {best['epoch']}, val_loss {float(best['val_loss']):.5f}")


if __name__ == "__main__":
    main()
