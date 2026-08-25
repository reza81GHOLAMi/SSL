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
    training_loader = DataLoader(dataset_train, cfg["batch_size"], shuffle=True, drop_last=True, num_workers=2)
    test_loader = DataLoader(dataset_test, cfg["batch_size"], shuffle=False, drop_last=True, num_workers=2)

    print("Models initializaing")
    trainer = Trainer(cfg, training_loader, test_loader)

    print("Models training")
    trainer.fit()
    if cfg["wandb"] == True:
        wandb.finish()


def main(args, config):
    n_gpu = torch.cuda.device_count()

    config["n_gpu"] = n_gpu
    config["wandb"] = args.wandb
    config["run_name"] = args.run_name
    config["wandb_project"] = args.wandb_project
    config["wandb_entity"] = args.wandb_entity

    path_for_models = f"saved_models/{config['run_name']}"
    isExist = os.path.exists(path_for_models)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path_for_models)
        print("The new directory is created!")
    config["checkpoint_path"] = path_for_models

    if args.json != "":
        data = json.loads(args.json)
        for key in data:
            config[key] = data[key]
    train_init(config["gpu_num"], config)


if __name__ == "__main__":

    for variation in config["variations"]:
        updated_config = config.copy()
        updated_config.update(variation)

        # Generate unique run name based on parameters
        updated_config[
            "run_name"] = f"{updated_config['feature_type']}_{updated_config['sample_rate']}_{updated_config['n_fft']}_{updated_config['hop_length']}_{updated_config['n_mfcc']}_{updated_config['n_mels']}_{updated_config['n_lfcc']}_{updated_config['n_barks']}"

        print(f"\nStarting experiment: {updated_config['run_name']}")
        parser = argparse.ArgumentParser(description='Global settings')
        parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
        parser.add_argument('--wandb_project', type=str, default='your-project-name', help='WandB project name')
        parser.add_argument('--wandb_entity', type=str, default='your-username', help='WandB team or user entity')
        parser.add_argument('--run_name', default=updated_config['run_name']
                            , type=str, help='Name of this run. Used to create folders where to save the weights.')
        parser.add_argument('--json', default="", type=str)

        args, unknown = parser.parse_known_args()
        main(args, updated_config)

        cooldown = updated_config.get("cooldown_seconds", 0)
        if cooldown > 0:
            print(f"Cooling down for {cooldown} seconds...")
            time.sleep(cooldown)
