"""Carve a validation split out of an EXISTING divided dataset's train folders.

Operates in place (whole-folder moves), so no re-processing and no re-upload
(e.g. to Kaggle) is needed. One folder = one physical recording, so moving
whole folders guarantees a recording never appears in two splits (no leakage).

Selection rule: group train folders by (distance, height, azimuth) and move
one randomly chosen folder per group to `val` (seeded, deterministic). Side is
ignored because `use_side` is False in the current config - side is not a
regression target. Ambient `none_none_none_none` folders stay in train.

Usage:
    python make_validation_split.py                       # dry-run (default)
    python make_validation_split.py --apply               # execute the moves
    python make_validation_split.py --undo                # restore from manifest
    python make_validation_split.py --dataset-path D:\\... # non-default root
"""

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import config  # noqa: E402

MANIFEST_NAME = "split_manifest.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", default=config["dataset_path"],
                        help="Root containing train\\ and test\\ (default: config dataset_path)")
    parser.add_argument("--seed", type=int, default=config.get("split_seed", 42),
                        help="RNG seed for the per-group pick (default: config split_seed)")
    parser.add_argument("--apply", action="store_true",
                        help="Execute the moves (without it, only prints the plan)")
    parser.add_argument("--undo", action="store_true",
                        help="Move folders back to train according to the manifest")
    return parser.parse_args()


def find_groups(train_dir):
    """Group folder names by (distance, height, azimuth). Ambient folders excluded."""
    groups = defaultdict(list)
    ambient = []
    if not os.path.isdir(train_dir):
        sys.exit(f"ERROR: train dir not found: {train_dir}")
    for folder in sorted(os.listdir(train_dir)):
        path = os.path.join(train_dir, folder)
        if not os.path.isdir(path):
            continue
        parts = folder.split("_")
        if parts[:4] == ["none", "none", "none", "none"]:
            ambient.append(folder)
        elif len(parts) >= 4:
            groups[tuple(parts[:3])].append(folder)
    return groups, ambient


def plan_val_folders(groups, seed):
    random.seed(seed)
    picked = {}
    for key in sorted(groups):
        members = groups[key]
        if len(members) < 2:
            continue  # group too small - keep its only recording in train
        picked[key] = random.choice(members)
    return picked


def load_manifest(root):
    manifest_path = os.path.join(root, MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        sys.exit(f"ERROR: no {MANIFEST_NAME} in {root} - nothing to undo")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f), manifest_path


def main():
    args = parse_args()
    root = args.dataset_path
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    if args.undo:
        manifest, manifest_path = load_manifest(root)
        moved = 0
        for folder in manifest["moved_folders"]:
            src = os.path.join(val_dir, folder)
            dst = os.path.join(train_dir, folder)
            if os.path.isdir(src):
                if os.path.isdir(dst):
                    print(f"SKIP (already in train): {folder}")
                    continue
                shutil.move(src, dst)
                moved += 1
            else:
                print(f"SKIP (not in val): {folder}")
        os.remove(manifest_path)
        print(f"Undo complete: {moved} folder(s) returned to train, manifest removed")
        return

    groups, ambient = find_groups(train_dir)
    if os.path.isdir(val_dir) and os.listdir(val_dir):
        sys.exit(f"ERROR: {val_dir} already exists and is not empty. Use --undo first if needed")

    picked = plan_val_folders(groups, args.seed)

    total_labeled = sum(len(v) for v in groups.values())
    print(f"Root:        {root}")
    print(f"Seed:        {args.seed}")
    print(f"Train folders now: {total_labeled + len(ambient)} "
          f"({total_labeled} labeled in {len(groups)} position groups + {len(ambient)} ambient)")
    print(f"Planned val folders: {len(picked)} (1 per group with >=2 recordings)")
    print("-" * 60)
    for key in sorted(picked):
        group_folders = ", ".join(groups[key])
        print(f"  group {key}: [{group_folders}]  ->  val: {picked[key]}")
    print("-" * 60)
    remaining = total_labeled - len(picked)
    print(f"After split: train = {remaining + len(ambient)} labeled+ambient, "
          f"val = {len(picked)}, test = untouched")

    if not args.apply:
        print("\nDRY RUN - nothing moved. Re-run with --apply to execute.")
        return

    os.makedirs(val_dir, exist_ok=True)
    for folder in picked.values():
        shutil.move(os.path.join(train_dir, folder), os.path.join(val_dir, folder))
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "dataset_root": root,
        "moved_folders": sorted(picked.values()),
        "rule": "1 random folder per (distance,height,azimuth) group moved train->val",
    }
    with open(os.path.join(root, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nApplied: {len(picked)} folder(s) moved to {val_dir}")
    print(f"Manifest written: {os.path.join(root, MANIFEST_NAME)}")


if __name__ == "__main__":
    main()
