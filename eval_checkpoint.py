"""Evaluate a saved checkpoint on the full test split (post-training test).

Usage: python eval_checkpoint.py [--no-side] ckpt1.pth [ckpt2.pth ...]
Rebuilds datasets/loaders from the config stored inside the checkpoint.
--no-side excludes the side (d) terms from the loss and metrics.
"""
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch.utils.data import DataLoader

from operation import AudioSpectrogramDataset
from trainer import Trainer


def evaluate(ckpt_path, use_side=None):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(checkpoint["config"])
    cfg["pretrained"] = False
    cfg["device"] = "cpu"
    cfg["wandb"] = False
    # Don't append evaluation output to the run's results file
    cfg["results_file"] = None
    if use_side is not None:
        cfg["use_side"] = use_side

    # Checkpoints saved before the configurable-target feature (no use_* keys in
    # their stored config) always have the legacy 6-output head: distance, height,
    # azimuth, side. Force those flags so Trainer builds a matching head.
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

    print(f"\n=== {ckpt_path} (epoch {checkpoint['epoch']}) ===")
    trainer.test_loop(checkpoint["epoch"])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    use_side = False if "--no-side" in sys.argv else None
    for path in args:
        evaluate(path, use_side)
