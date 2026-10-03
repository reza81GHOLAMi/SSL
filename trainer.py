import gc
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
from model import Classifier
from operation import build_target_spec, target_label_indices


class Logger():
    def __init__(self, cfg):
        super(Logger, self).__init__()
        self.cfg = cfg
        self.use_wandb = self.cfg["wandb"]

    def step(self, data):
        if self.use_wandb:
            if data["Graph"]:
                wandb.log(data["Graph"])
            if data["Image"]:
                try:
                    for key in data["Image"]:
                        data["Image"][key] = wandb.Image(data["Image"][key], caption=key)
                    wandb.log(data["Image"])
                except:
                    pass
            if data["Audio"]:
                for key in data["Audio"]:
                    data["Audio"][key] = wandb.Audio(data["Audio"][key], caption=key, sample_rate=self.cfg["sample_rate"])
                wandb.log(data["Audio"])
        if data["Print"]:
            print(data["Print"])


class Trainer:
    def __init__(self, cfg, loader, test_loader, val_loader=None):
        self.cfg = cfg
        self.cfg["device"] = torch.device(
            "cuda:{}".format(self.cfg["rank"])) if torch.cuda.is_available() else torch.device("cpu")

        self.loader = loader
        self.test_loader = test_loader
        # Optional validation loader: when present, per-epoch model selection
        # (best.pth), LR scheduling and the history CSV all run on val; the
        # test set is then only evaluated by evaluate_test.py after training.
        self.val_loader = val_loader
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        if torch.cuda.is_available():
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.cfg.get("use_grad", False))
        else:
            self.scaler = None

        # Active regression targets (config use_* flags) determine the model head
        # size and which loss/metric terms are computed. target_indices[i] is the
        # canonical label column for model output column i.
        self.target_spec = build_target_spec(self.cfg)
        self.target_indices = target_label_indices(self.cfg)

        n_mics = len(self.cfg.get("channels", range(8)))
        # Rainbowgram features double the per-mic block (magnitude + IF map);
        # IPD/ILD pair count always derives from the mic count
        n_channels = 2 * n_mics if self.cfg.get("feature_type") == "rainbow" else n_mics
        # Channel-layout delta doubles the input channels (~<0.1% more params
        # in conv_in); freq/time layouts stack spatially and stay parameter-matched
        if self.cfg.get("use_delta", False) and self.cfg.get("delta_layout", "channel") == "channel":
            n_channels *= 2
        if self.cfg.get("use_ipd", False):
            n_channels += 3 * (n_mics * (n_mics - 1) // 2)
        self.segmentate = Classifier(channel_in=n_channels,
                                     num_classes=len(self.target_indices)).to(self.cfg["device"])

        self.logger = Logger(cfg)
        self.opt_Segm = optim.Adam(list(self.segmentate.parameters()), lr=self.cfg["lr"], betas=(0.5, 0.9), )

        self.loss_buffer = []
        self.scheduler = ReduceLROnPlateau(self.opt_Segm, 'min', patience=self.cfg.get("lr_patience", 10), factor=0.9, min_lr=0.000001)

        self.len_loader = len(loader)
        self.epoch = 0
        self.mse_loss = nn.MSELoss()

        if cfg["pretrained"]:
            try:
                checkpoint = torch.load(self.cfg[f"model_path"], weights_only=False)
                try:
                    self.segmentate.load_state_dict(checkpoint["classifier_state_dict"])
                except RuntimeError as e:
                    print(f"Pretrained checkpoint head does not match the active target spec "
                          f"{self.target_spec} ({len(self.target_indices)} outputs): {e}")
                    raise
                self.opt_Segm.load_state_dict(checkpoint["optimizer_state_dict"])
                if self.scaler is not None:
                    self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                if checkpoint.get("scheduler_state_dict") is not None:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                rng_state = checkpoint.get("rng_state")
                if rng_state is not None:
                    torch.set_rng_state(rng_state["torch"])
                    if rng_state.get("cuda") is not None and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(rng_state["cuda"])
                    np.random.set_state(rng_state["numpy"])
                    random.setstate(rng_state["python"])
                self.epoch = checkpoint["epoch"]

            except FileNotFoundError as e:
                print(f"Not found pretrained weights. Continue without.: {e}")

    def compute_loss(self, pred, target):
        # Sum of per-target MSE over the active targets only. Model output column i
        # pairs with canonical label column self.target_indices[i].
        err = torch.sigmoid(pred) - target[:, self.target_indices]
        return err.square().mean(dim=0).sum()

    def fit(self):
        import time
        save_every = self.cfg.get("save_every_epochs", 5)
        checkpoint_dir = self.cfg.get("checkpoint_path", "saved_models/default")
        # Resume the best-val tracking from an existing best.pth of this run
        best_ckpt_path = os.path.join(checkpoint_dir, "best.pth")
        if self.best_val_loss == float("inf") and os.path.isfile(best_ckpt_path):
            try:
                prev = torch.load(best_ckpt_path, weights_only=False, map_location="cpu")
                self.best_val_loss = prev.get("val_loss", float("inf"))
            except Exception as e:
                print(f"Could not read previous best.pth val_loss: {e}")

        for epoch in range(self.epoch, self.cfg["epochs"]):
            print(f"Starting epoch {epoch + 1}/{self.cfg['epochs']}")
            t0 = time.time()
            if self.val_loader is not None:
                train_loss = self.training_loop(epoch)
                val_loss, val_metrics = self.evaluate_loop(epoch, self.val_loader, split="val")
                self.scheduler.step(val_loss)
                lr = self.opt_Segm.param_groups[0]["lr"]
                self._append_history(epoch + 1, train_loss, val_loss, val_metrics, lr,
                                     time.time() - t0)
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch + 1
                    self.save_checkpoint(epoch + 1, self.segmentate, self.opt_Segm, self.scaler,
                                         best_ckpt_path, val_loss=val_loss)
                    print(f"New best val_loss {val_loss:.6f} -> saved {best_ckpt_path}")
            else:
                # Legacy behavior: evaluate on the test set each epoch
                self.test_loop(epoch)
                self.training_loop(epoch)

            if save_every > 0 and (epoch + 1) % save_every == 0 and epoch + 1 < self.cfg["epochs"]:
                self.save_checkpoint(epoch + 1, self.segmentate, self.opt_Segm, self.scaler,
                    f"{checkpoint_dir}/{epoch + 1}_checkpoint.pth")

        if self.val_loader is not None:
            # Test set is intentionally NOT evaluated here — use evaluate_test.py
            # after training for the final test metrics.
            pass
        else:
            # Post-training test so the results file records the metrics of the
            # weights this run actually produced (test_loop otherwise runs only
            # at the start of each epoch, i.e. before that epoch's training).
            self.test_loop(self.cfg["epochs"])

        # Always save at the end of training so short runs still produce a resumable checkpoint
        final_path = f"{checkpoint_dir}/{self.cfg['epochs']}_final.pth"
        best_is_final = (self.val_loader is not None and self.best_epoch == self.cfg["epochs"])
        self.save_checkpoint(self.cfg["epochs"], self.segmentate, self.opt_Segm, self.scaler,
            final_path, val_loss=self.best_val_loss if best_is_final else None)

        if best_is_final and os.path.exists(best_ckpt_path):
            # Last epoch is also the best: avoid a second ~232 MB copy — make
            # best.pth a hardlink to the final checkpoint (both names must
            # exist: resume scans N_final.pth, evaluate_test.py prefers best.pth).
            try:
                os.remove(best_ckpt_path)
                os.link(final_path, best_ckpt_path)
                print(f"Best epoch == final epoch ({self.best_epoch}): "
                      f"best.pth is a hardlink to {final_path}")
            except OSError as e:
                # Filesystem can't hardlink — keep a real copy instead
                print(f"Hardlink failed ({e}), saving best.pth as a separate copy")
                self.save_checkpoint(self.cfg["epochs"], self.segmentate, self.opt_Segm,
                                     self.scaler, best_ckpt_path, val_loss=self.best_val_loss)

    def _append_history(self, epoch, train_loss, val_loss, metrics, lr, elapsed):
        """Append one per-epoch row to <checkpoint_path>/history.csv — everything
        needed to plot epoch curves and locate the best epoch afterwards."""
        history_file = os.path.join(self.cfg.get("checkpoint_path", "saved_models/default"),
                                    "history.csv")
        os.makedirs(os.path.dirname(history_file) or ".", exist_ok=True)
        metric_keys = list(metrics.keys())
        new_file = not os.path.exists(history_file)
        with open(history_file, "a", newline="") as f:
            if new_file:
                f.write("epoch\ttrain_loss\tval_loss\t" + "\t".join(metric_keys)
                        + "\tlr\telapsed_seconds\n")
            f.write(f"{epoch}\t{train_loss:.6f}\t{val_loss:.6f}\t"
                    + "\t".join(f"{metrics[k]:.6f}" for k in metric_keys)
                    + f"\t{lr:.8g}\t{elapsed:.1f}\n")

    def _rng_state(self):
        return {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

    @staticmethod
    def _dataset_files(loader):
        files = getattr(loader.dataset, "file_paths", None)
        return list(files) if files is not None else None

    def save_checkpoint(self, EPOCH, classifier, optimizer, scaler, PATH, val_loss=None):
        if isinstance(classifier, torch.nn.DataParallel):
            model_state_dict_enc = classifier.module.state_dict()
        else:
            model_state_dict_enc = classifier.state_dict()
        torch.save({"epoch": EPOCH, "classifier_state_dict": model_state_dict_enc, "config": self.cfg,
            "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            # Incremental-learning state: scheduler, RNGs, and the exact files this run saw
            "scheduler_state_dict": self.scheduler.state_dict(),
            "rng_state": self._rng_state(),
            "train_files": self._dataset_files(self.loader),
            "test_files": self._dataset_files(self.test_loader),
            "run_name": self.cfg.get("run_name"),
            "val_loss": val_loss,
        }, PATH, )

    def update_loss_buffer(self, loss):
        """ Update the loss buffer and return the moving average """
        buffer_size = 2000  # Define the buffer size for averaging
        if len(self.loss_buffer) >= buffer_size:
            self.loss_buffer.pop(0)  # Remove the oldest loss value
        self.loss_buffer.append(loss)  # Add the new loss value
        if len(self.loss_buffer) < 1999:
            return 0
        return sum(self.loss_buffer) / len(self.loss_buffer)  # Compute the average

    def training_loop(self, epoch):
        self.segmentate.train()
        self.loss_buffer = []
        epoch_loss_sum = 0.0
        epoch_batches = 0

        for iteration, data in enumerate(tqdm(self.loader, desc=f"Training Epoch {epoch + 1}")):
            data_to_log = {"Graph": {}, "Print": {}, "Image": {}, "Audio": {}}

            # Unpacking the data (training data returns spectrograms and labels)
            input_spec_tensor, mask = (tensor.to(self.cfg["device"]).float() for tensor in data)

            self.opt_Segm.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.cfg.get("use_amp", False)):
                out = self.segmentate(input_spec_tensor)

            Loss = self.compute_loss(out, mask)

            # Backpropagation and optimization
            if self.scaler is not None:
                self.scaler.scale(Loss).backward()
                self.scaler.unscale_(self.opt_Segm)
                self.scaler.step(self.opt_Segm)
                self.scaler.update()
            else:
                Loss.backward()
                self.opt_Segm.step()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc_interval = self.cfg.get("gc_collect_interval", 0)
            if gc_interval > 0 and iteration % gc_interval == 0:
                gc.collect()

            # Updating the loss buffer
            avg_loss = self.update_loss_buffer(Loss.item())
            epoch_loss_sum += Loss.item()
            epoch_batches += 1

            # Logging losses and accuracy
            data_to_log["Graph"]["Total_Loss"] = Loss.item()

            if iteration % self.cfg.get("print_step", 100) == 0:
                data_to_log["Print"]["Status"] = f'Epoch: {epoch + 1}, Iteration: {iteration}/{self.len_loader}'
                data_to_log["Print"]["Total_Loss"] = Loss.item()
                data_to_log["Print"]["Scheduler_lr"] = [group['lr'] for group in self.opt_Segm.param_groups]
                data_to_log["Print"]["Avg_Loss_Buffer"] = avg_loss
                data_to_log["Print"]["Epoch"] = epoch

            self.logger.step(data_to_log)

        return epoch_loss_sum / epoch_batches if epoch_batches else 0.0

    def test_loop(self, epoch):
        avg_loss, metrics = self.evaluate_loop(epoch, self.test_loader, split="test")
        self.scheduler.step(avg_loss)
        return avg_loss, metrics

    def evaluate_loop(self, epoch, loader, split="test"):
        self.segmentate.eval()
        test_loss_total = 0.0
        # Metric suffix per target, kept compatible with the legacy a/b/c/d naming
        suffix_map = {"distance": "a", "height": "b", "hypotenuse": "hyp", "azimuth": "c", "side": "d"}
        scalar_targets = ("distance", "height", "hypotenuse")
        total_mae = {name: 0.0 for name in self.target_spec}
        total_mse = {name: 0.0 for name in self.target_spec}
        total_samples = 0
        # True 3D position error (Euclidean meters) — only computable when
        # distance + height + azimuth are all active (height is unknown for
        # hypotenuse-only runs, so the z-coordinate cannot be reconstructed)
        use_3d = all(t in self.target_spec for t in ("distance", "height", "azimuth"))
        total_3d_mae = 0.0
        total_3d_mse = 0.0
        # Model output column range for each target (scalars take 1 column, angles 2)
        out_cols = {}
        col = 0
        for name in self.target_spec:
            n = 1 if name in scalar_targets else 2
            out_cols[name] = (col, col + n)
            col += n

        with torch.no_grad():
            for iteration, data in enumerate(tqdm(loader, desc=f"Evaluating ({split}) Epoch {epoch + 1}")):
                input_spec_tensor, mask = (tensor.to(self.cfg["device"]).float() for tensor in data)
                out = self.segmentate(input_spec_tensor)
                loss = self.compute_loss(out, mask)
                batch_size = input_spec_tensor.size(0)
                test_loss_total += loss.item() * batch_size

                def process_angle(sin_pred, cos_pred, sin_true, cos_true):
                    sin_pred = sin_pred * 2 - 1
                    cos_pred = cos_pred * 2 - 1
                    sin_true = sin_true * 2 - 1
                    cos_true = cos_true * 2 - 1
                    angle_pred = torch.rad2deg(torch.atan2(sin_pred, cos_pred))
                    angle_true = torch.rad2deg(torch.atan2(sin_true, cos_true))
                    diff = torch.abs(angle_pred - angle_true)
                    diff = torch.minimum(diff, 360 - diff)  # Uwzględnienie okresowości kątów
                    return diff.sum().item(), torch.square(diff).sum().item()

                for name in self.target_spec:
                    lo, hi = out_cols[name]
                    label_cols = self.target_indices[lo:hi]
                    if name in scalar_targets:
                        # Denormalized by the same 50.0 used in operation.py
                        pred_m = torch.sigmoid(out[:, lo:hi]) * 50
                        true_m = mask[:, label_cols] * 50
                        total_mae[name] += torch.abs(pred_m - true_m).sum().item()
                        total_mse[name] += torch.square(pred_m - true_m).sum().item()
                    else:
                        mae, mse = process_angle(torch.sigmoid(out[:, lo]), torch.sigmoid(out[:, lo + 1]),
                                                 mask[:, label_cols[0]], mask[:, label_cols[1]])
                        total_mae[name] += mae
                        total_mse[name] += mse

                if use_3d:
                    def angle_rad(sin_col_pred, cos_col_pred, label_cols_az):
                        sin_p = torch.sigmoid(out[:, sin_col_pred]) * 2 - 1
                        cos_p = torch.sigmoid(out[:, cos_col_pred]) * 2 - 1
                        sin_t = mask[:, label_cols_az[0]] * 2 - 1
                        cos_t = mask[:, label_cols_az[1]] * 2 - 1
                        return torch.atan2(sin_p, cos_p), torch.atan2(sin_t, cos_t)

                    d_lo, _ = out_cols["distance"]
                    h_lo, _ = out_cols["height"]
                    a_lo, a_hi = out_cols["azimuth"]
                    az_p, az_t = angle_rad(a_lo, a_lo + 1, self.target_indices[a_lo:a_hi])
                    # Keep everything 1-D [B]: slicing with lo:hi would give [B,1],
                    # which silently broadcasts against [B] angles to [B,B]
                    d_p = torch.sigmoid(out[:, d_lo]) * 50
                    d_t = mask[:, self.target_indices[d_lo]] * 50
                    h_p = torch.sigmoid(out[:, h_lo]) * 50
                    h_t = mask[:, self.target_indices[h_lo]] * 50
                    # Reconstruct both points as (d*cos(az), d*sin(az), h) in meters
                    err3d = torch.sqrt(
                        torch.square(d_p * torch.cos(az_p) - d_t * torch.cos(az_t))
                        + torch.square(d_p * torch.sin(az_p) - d_t * torch.sin(az_t))
                        + torch.square(h_p - h_t))
                    total_3d_mae += err3d.sum().item()
                    total_3d_mse += err3d.square().sum().item()

                total_samples += batch_size

        if total_samples == 0:
            print(f"No {split} data")
            return None, None

        # Calculating final metrics
        avg_test_loss = test_loss_total / total_samples

        metrics = {}
        for name in self.target_spec:
            sfx = suffix_map[name]
            metrics[f"MAE_{sfx}"] = total_mae[name] / total_samples
            metrics[f"RMSE_{sfx}"] = (total_mse[name] / total_samples) ** 0.5
        if use_3d:
            metrics["MAE_3D"] = total_3d_mae / total_samples
            metrics["RMSE_3D"] = (total_3d_mse / total_samples) ** 0.5

        # Logowanie wyników
        split_label = split.capitalize()
        data_to_log = {"Graph": {f"{split_label}_Loss": avg_test_loss, **metrics},
            "Print": {f"{split_label}_Epoch": epoch + 1, f"{split_label}_Loss": avg_test_loss, **metrics}, "Image": {}, "Audio": {}}

        self.logger.step(data_to_log)
        print(f"\nEpoch {epoch + 1} - {split_label} results:")
        print(f"Loss: {avg_test_loss:.4f}")
        for name in self.target_spec:
            sfx = suffix_map[name]
            print(f"MAE {name} ({sfx}): {metrics[f'MAE_{sfx}']:.4f} ± {metrics[f'RMSE_{sfx}']:.4f}")
        if use_3d:
            print(f"MAE 3D position (true Euclidean): {metrics['MAE_3D']:.4f} ± {metrics['RMSE_3D']:.4f} m")
        print()

        # Append one line per evaluation to the run's results file (config["results_file"])
        results_file = self.cfg.get("results_file")
        if results_file:
            os.makedirs(os.path.dirname(results_file) or ".", exist_ok=True)
            new_file = not os.path.exists(results_file)
            with open(results_file, "a", newline="") as f:
                metric_keys = list(metrics.keys())
                if new_file:
                    f.write("timestamp\trun_name\tepoch\tsplit\tloss\t"
                            + "\t".join(metric_keys) + "\n")
                f.write(f"{datetime.now().isoformat(timespec='seconds')}\t"
                        f"{self.cfg.get('run_name')}\t{epoch}\t{split}\t{avg_test_loss:.6f}\t"
                        + "\t".join(f"{metrics[k]:.6f}" for k in metric_keys) + "\n")

        return avg_test_loss, metrics
