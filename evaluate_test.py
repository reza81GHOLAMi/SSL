"""Final test evaluation as a SEPARATE execution from training.

Training (main.py) never touches the test set when a val split exists —
best-epoch selection runs on validation. This script loads a checkpoint
(default: best.pth from the run's checkpoint dir) and evaluates it once on
the test split, printing metrics and appending them to
results/<run_name>_testEval.txt.

Usage:
    python evaluate_test.py saved_models/<run_name>            # dir -> best.pth
    python evaluate_test.py saved_models/<run_name>/best.pth   # explicit file
    python evaluate_test.py --checkpoint ... --no-side
"""
import argparse
import os
import re
import sys
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader

from operation import AudioSpectrogramDataset
from trainer import Trainer


def resolve_checkpoint(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        best = os.path.join(path, "best.pth")
        if os.path.isfile(best):
            return best
        # Fall back to the highest-numbered *_final.pth
        finals = [f for f in os.listdir(path) if re.fullmatch(r"\d+_final\.pth", f)]
        if finals:
            return os.path.join(path, max(finals, key=lambda f: int(f.split("_")[0])))
        sys.exit(f"ERROR: no best.pth or *_final.pth in {path}")
    sys.exit(f"ERROR: checkpoint not found: {path}")


def evaluate(ckpt_path, use_side=None, dataset_path=None, out_file=None):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["config"])
    cfg["pretrained"] = False
    cfg["device"] = "cpu"
    cfg["wandb"] = False
    # Route the appended results to the dedicated testEval file instead of
    # the training results file
    cfg["results_file"] = out_file
    if use_side is not None:
        cfg["use_side"] = use_side
    if dataset_path is not None:
        cfg["dataset_path"] = dataset_path

    # Checkpoints saved before the configurable-target feature always have the
    # legacy 6-output head: distance, height, azimuth, side
    head_size = checkpoint["classifier_state_dict"]["fc2.weight"].shape[0]
    if "use_hypotenuse" not in cfg and head_size == 6:
        cfg["use_distance"] = True
        cfg["use_height"] = True
        cfg["use_hypotenuse"] = False
        cfg["use_side"] = True

    dataset_test = AudioSpectrogramDataset(cfg, mode="test")
    test_loader = DataLoader(dataset_test, cfg["batch_size"], shuffle=False, drop_last=False,
                             num_workers=cfg.get("num_workers", 2))

    trainer = Trainer(cfg, test_loader, test_loader)
    trainer.segmentate.load_state_dict(checkpoint["classifier_state_dict"])
    trainer.segmentate.eval()

    val_loss = checkpoint.get("val_loss")
    print(f"\n=== {ckpt_path} (epoch {checkpoint['epoch']}"
          + (f", val_loss {val_loss:.6f}" if val_loss is not None else ", no val_loss stored") + ") ===")
    print(f"Test files: {len(dataset_test)}")
    trainer.test_loop(checkpoint["epoch"])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", nargs="?", default=None,
                        help="Checkpoint .pth file or run directory (dir -> best.pth)")
    parser.add_argument("--checkpoint", dest="checkpoint_opt", default=None,
                        help="Same as the positional argument")
    parser.add_argument("--no-side", action="store_true",
                        help="Exclude the side (d) terms from loss/metrics")
    parser.add_argument("--dataset-path", default=None,
                        help="Override dataset root (default: stored in checkpoint config)")
    args = parser.parse_args()

    target = args.checkpoint or args.checkpoint_opt
    if not target:
        parser.error("provide a checkpoint file or run directory")
    ckpt_path = resolve_checkpoint(target)

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run_name = checkpoint.get("run_name") or "unknown_run"
    out_file = os.path.join("results", f"{run_name}_testEval.txt")

    evaluate(ckpt_path,
             use_side=False if args.no_side else None,
             dataset_path=args.dataset_path,
             out_file=out_file)
    print(f"Results appended to {out_file}")


if __name__ == "__main__":
    main()
