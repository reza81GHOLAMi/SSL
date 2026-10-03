import json
import os
import time

from operation import AudioSpectrogramDataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from config import config
import argparse
import wandb
from torch.utils.data import DataLoader
from trainer import Trainer


def train_init(rank, cfg):
    if cfg["wandb"] == True:
        wandb.init(project=cfg["wandb_project"], entity=cfg["wandb_entity"], config=cfg)
        wandb.run.name = cfg["run_name"]
    cfg["rank"] = rank

    print("Data loading")
    dataset_train = AudioSpectrogramDataset(cfg, mode="train")
    dataset_test = AudioSpectrogramDataset(cfg, mode="test")
    training_loader = DataLoader(dataset_train, cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg.get("num_workers", 2))
    test_loader = DataLoader(dataset_test, cfg["batch_size"], shuffle=False, drop_last=False, num_workers=cfg.get("num_workers", 2))

    # Optional validation split: when <dataset_path>/val exists, per-epoch model
    # selection and LR scheduling run on val instead of the test set (which is
    # then only evaluated by evaluate_test.py after training).
    val_loader = None
    val_dir = os.path.join(cfg["dataset_path"], "val")
    has_val = False
    if os.path.isdir(val_dir):
        with os.scandir(val_dir) as entries:
            has_val = any(entries)
    if has_val:
        dataset_val = AudioSpectrogramDataset(cfg, mode="val")
        val_loader = DataLoader(dataset_val, cfg["batch_size"], shuffle=False, drop_last=False, num_workers=cfg.get("num_workers", 2))
        print(f"Validation split found: {len(dataset_val)} files -> used for epoch selection")

    print("Models initializaing")
    trainer = Trainer(cfg, training_loader, test_loader, val_loader=val_loader)

    print("Models training")
    trainer.fit()
    if cfg["wandb"] == True:
        wandb.finish()


def apply_json_overrides(args, config):
    """Apply --json / --json-file overrides (json-file wins if both given)."""
    if args.json != "":
        data = json.loads(args.json)
        for key in data:
            config[key] = data[key]
    if getattr(args, "json_file", "") != "":
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in data:
            config[key] = data[key]


def main(args, config):
    n_gpu = torch.cuda.device_count()

    config["n_gpu"] = n_gpu
    config["wandb"] = args.wandb
    config["run_name"] = args.run_name
    config["wandb_project"] = args.wandb_project
    config["wandb_entity"] = args.wandb_entity

    # Overrides first so a json-provided checkpoint_path/run_name gets its directory
    apply_json_overrides(args, config)

    if "checkpoint_path" not in config:
        config["checkpoint_path"] = f"saved_models/{config['run_name']}"
    os.makedirs(config["checkpoint_path"], exist_ok=True)

    train_init(config["gpu_num"], config)


if __name__ == "__main__":
    # Apply json overrides BEFORE the variation loop so a json-provided
    # "variations" list fully replaces config.py's list
    preparser = argparse.ArgumentParser(add_help=False)
    preparser.add_argument('--json', default="", type=str)
    preparser.add_argument('--json-file', dest="json_file", default="", type=str)
    pre_args, _ = preparser.parse_known_args()
    apply_json_overrides(pre_args, config)

    for variation in config["variations"]:
        updated_config = config.copy()
        updated_config.update(variation)

        # Generate unique run name based on parameters
        updated_config[
            "run_name"] = f"{updated_config['feature_type']}_{updated_config['sample_rate']}_{updated_config['n_fft']}_{updated_config['hop_length']}_{updated_config['n_mfcc']}_{updated_config['n_mels']}_{updated_config['n_lfcc']}_{updated_config['n_barks']}_ch{''.join(map(str, updated_config['channels']))}"

        # Append active-target flags (Distance/Height/slant-Range/Side) so runs
        # differing only in regression targets get distinct checkpoint dirs
        target_flags = "".join([
            "D" if updated_config.get("use_distance", True) else "-",
            "H" if updated_config.get("use_height", True) else "-",
            "R" if updated_config.get("use_hypotenuse", False) else "-",
            "S" if updated_config.get("use_side", True) else "-",
        ])
        updated_config["run_name"] += f"_t{target_flags}"
        if updated_config.get("use_delta", False):
            # Window + layout tag keep delta runs in distinct checkpoint dirs
            # (_dlt9 = channel layout, _dlt9f = freq, _dlt9t = time; the old
            # time-stacked runs used a bare _dlt suffix)
            updated_config["run_name"] += f"_dlt{updated_config.get('delta_win', 9)}"
            layout = updated_config.get("delta_layout", "channel")
            if layout in ("freq", "time"):
                updated_config["run_name"] += layout[0]
        if updated_config.get("use_ipd", False):
            updated_config["run_name"] += "_ipd"

        # Checkpoints always live in saved_models/<run_name>/ — set per variation
        # so a stale "checkpoint_path" key cannot redirect them (BUG-30)
        updated_config["checkpoint_path"] = f"saved_models/{updated_config['run_name']}"
        os.makedirs(updated_config["checkpoint_path"], exist_ok=True)

        print(f"\nStarting experiment: {updated_config['run_name']}")
        parser = argparse.ArgumentParser(description='Global settings')
        parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
        parser.add_argument('--wandb_project', type=str, default='your-project-name', help='WandB project name')
        parser.add_argument('--wandb_entity', type=str, default='your-username', help='WandB team or user entity')
        parser.add_argument('--run_name', default=updated_config['run_name']
                            , type=str, help='Name of this run. Used to create folders where to save the weights.')
        parser.add_argument('--json', default="", type=str)
        parser.add_argument('--json-file', dest="json_file", default="", type=str)

        args, unknown = parser.parse_known_args()
        main(args, updated_config)

        cooldown = updated_config.get("cooldown_seconds", 0)
        if cooldown > 0:
            print(f"Cooling down for {cooldown} seconds...")
            time.sleep(cooldown)
