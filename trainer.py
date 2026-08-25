import gc

import torch
import torch.optim as optim
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import wandb
from model import Classifier


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
                    data["Audio"][key] = wandb.Audio(data["Audio"][key], caption=key, sample_rate=16000)
                wandb.log(data["Audio"])
        if data["Print"]:
            print(data["Print"])


class Trainer:
    def __init__(self, cfg, loader, test_loader):
        self.cfg = cfg
        self.cfg["device"] = torch.device(
            "cuda:{}".format(self.cfg["rank"])) if torch.cuda.is_available() else torch.device("cpu")

        self.loader = loader
        self.test_loader = test_loader
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg["use_grad"])

        self.segmentate = Classifier(channel_in=8).to(self.cfg["device"])

        self.logger = Logger(cfg)
        self.opt_Segm = optim.Adam(list(self.segmentate.parameters()), lr=self.cfg["lr"], betas=(0.5, 0.9), )

        self.loss_buffer = []
        self.scheduler = ReduceLROnPlateau(self.opt_Segm, 'max', patience=20000, factor=0.9, min_lr=0.000001)

        self.len_loader = len(loader)
        self.epoch = 0
        self.mse_loss = nn.MSELoss()

        if cfg["pretrained"]:
            try:
                checkpoint = torch.load(self.cfg[f"model_path"])
                self.segmentate.load_state_dict(checkpoint["classifier_state_dict"])
                self.opt_Segm.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
                self.epoch = checkpoint["epoch"]

            except FileNotFoundError as e:
                print(f"Not found pretrained weights. Continue without.: {e}")

    def compute_loss(self, pred, target):
        a_pred = torch.sigmoid(pred[:, 0])
        b_pred = torch.sigmoid(pred[:, 1])
        c_sin_pred = torch.sigmoid(pred[:, 2])
        c_cos_pred = torch.sigmoid(pred[:, 3])
        d_sin_pred = torch.sigmoid(pred[:, 4])
        d_cos_pred = torch.sigmoid(pred[:, 5])

        loss_a = self.mse_loss(a_pred, target[:, 0])
        loss_b = self.mse_loss(b_pred, target[:, 1])
        loss_c_sin = self.mse_loss(c_sin_pred, target[:, 2])
        loss_c_cos = self.mse_loss(c_cos_pred, target[:, 3])
        loss_d_sin = self.mse_loss(d_sin_pred, target[:, 4])
        loss_d_cos = self.mse_loss(d_cos_pred, target[:, 5])

        return loss_a + loss_b + loss_c_sin + loss_c_cos + loss_d_sin + loss_d_cos

    def fit(self):
        for epoch in range(self.epoch, self.cfg["epochs"]):
            print(f"Starting epoch {epoch + 1}/{self.cfg['epochs']}")
            self.test_loop(epoch)
            self.training_loop(epoch)

            # Zapis checkpointa co 5 epok
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, self.segmentate, self.opt_Segm, self.scaler,
                    f"{self.cfg['checkpoint_path']}/{epoch + 1}_end_unet.pth")

    def save_checkpoint(self, EPOCH, classifier, optimizer, scaler, PATH):
        if isinstance(classifier, torch.nn.DataParallel):
            model_state_dict_enc = classifier.module.state_dict()
        else:
            model_state_dict_enc = classifier.state_dict()
        torch.save({"epoch": EPOCH, "classifier_state_dict": model_state_dict_enc, "config": self.cfg,
            "optimizer_state_dict": optimizer.state_dict(), "scaler_state_dict": scaler.state_dict()
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

        for iteration, data in enumerate(tqdm(self.loader, desc=f"Training Epoch {epoch + 1}")):
            data_to_log = {"Graph": {}, "Print": {}, "Image": {}, "Audio": {}}

            # Unpacking the data (training data returns spectrograms and labels)
            input_spec_tensor, mask = (tensor.to(self.cfg["device"]).float() for tensor in data)

            self.opt_Segm.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.cfg.get("use_amp", False)):
                out = self.segmentate(input_spec_tensor).squeeze()

            Loss = self.compute_loss(out, mask)

            # Backpropagation and optimization
            self.scaler.scale(Loss).backward()
            self.scaler.unscale_(self.opt_Segm)
            self.scaler.step(self.opt_Segm)
            self.scaler.update()

            torch.cuda.empty_cache()
            _ = gc.collect()

            # Updating the loss buffer
            avg_loss = self.update_loss_buffer(Loss.item())

            # Logging losses and accuracy
            data_to_log["Graph"]["Total_Loss"] = Loss.item()

            if iteration % self.cfg.get("print_step", 100) == 0:
                data_to_log["Print"]["Status"] = f'Epoch: {epoch + 1}, Iteration: {iteration}/{self.len_loader}'
                data_to_log["Print"]["Total_Loss"] = Loss.item()
                data_to_log["Print"]["Scheduler_lr"] = [group['lr'] for group in self.opt_Segm.param_groups]
                data_to_log["Print"]["Avg_Loss_Buffer"] = avg_loss
                data_to_log["Print"]["Epoch"] = epoch

            self.logger.step(data_to_log)

    def test_loop(self, epoch):
        self.segmentate.eval()
        test_loss_total = 0.0
        total_mae_a = 0.0
        total_mse_a = 0.0
        total_mae_b = 0.0
        total_mse_b = 0.0
        total_mae_c = 0.0
        total_mse_c = 0.0
        total_mae_d = 0.0
        total_mse_d = 0.0
        total_samples = 0

        with torch.no_grad():
            for iteration, data in enumerate(tqdm(self.test_loader, desc=f"Testing Epoch {epoch + 1}")):
                input_spec_tensor, mask = (tensor.to(self.cfg["device"]).float() for tensor in data)
                out = self.segmentate(input_spec_tensor).squeeze()
                loss = self.compute_loss(out, mask)
                batch_size = input_spec_tensor.size(0)
                test_loss_total += loss.item() * batch_size

                # Denormalization and metrics for a and b
                a_pred = torch.sigmoid(out[:, 0]) * 50
                a_true = mask[:, 0] * 50
                total_mae_a += torch.abs(a_pred - a_true).sum().item()
                total_mse_a += torch.square(a_pred - a_true).sum().item()

                b_pred = torch.sigmoid(out[:, 1]) * 50
                b_true = mask[:, 1] * 50
                total_mae_b += torch.abs(b_pred - b_true).sum().item()
                total_mse_b += torch.square(b_pred - b_true).sum().item()

                # Calculating angles for c and d
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

                # Calculations for c
                c_sin_pred = torch.sigmoid(out[:, 2])
                c_cos_pred = torch.sigmoid(out[:, 3])
                c_sin_true = mask[:, 2]
                c_cos_true = mask[:, 3]
                mae_c, mse_c = process_angle(c_sin_pred, c_cos_pred, c_sin_true, c_cos_true)
                total_mae_c += mae_c
                total_mse_c += mse_c

                # Calculations for d
                d_sin_pred = torch.sigmoid(out[:, 4])
                d_cos_pred = torch.sigmoid(out[:, 5])
                d_sin_true = mask[:, 4]
                d_cos_true = mask[:, 5]
                mae_d, mse_d = process_angle(d_sin_pred, d_cos_pred, d_sin_true, d_cos_true)
                total_mae_d += mae_d
                total_mse_d += mse_d

                total_samples += batch_size

        if total_samples == 0:
            print("No test data")
            return

        # Calculating final metrics
        avg_test_loss = test_loss_total / total_samples

        mae_a = total_mae_a / total_samples
        rmse_a = (total_mse_a / total_samples) ** 0.5
        mae_b = total_mae_b / total_samples
        rmse_b = (total_mse_b / total_samples) ** 0.5
        mae_c = total_mae_c / total_samples
        rmse_c = (total_mse_c / total_samples) ** 0.5
        mae_d = total_mae_d / total_samples
        rmse_d = (total_mse_d / total_samples) ** 0.5

        self.scheduler.step(avg_test_loss)

        # Logowanie wyników
        data_to_log = {"Graph": {"Test_Loss": avg_test_loss, "Epoch": epoch, "MAE_a": mae_a, "RMSE_a": rmse_a, "MAE_b": mae_b, "RMSE_b": rmse_b,
            "MAE_c": mae_c, "RMSE_c": rmse_c, "MAE_d": mae_d, "RMSE_d": rmse_d},
            "Print": {"Test_Epoch": epoch + 1, "Test_Loss": avg_test_loss,  "MAE_a": mae_a,
                "RMSE_a": rmse_a, "MAE_b": mae_b, "RMSE_b": rmse_b, "MAE_c": mae_c, "RMSE_c": rmse_c, "MAE_d": mae_d,
                "RMSE_d": rmse_d}, "Image": {}, "Audio": {}}

        self.logger.step(data_to_log)
        print(f"\nEpoch {epoch + 1} - Test results:")
        print(f"Loss: {avg_test_loss:.4f}")
        print(f"MAE a: {mae_a:.4f} ± {rmse_a:.4f}")
        print(f"MAE b: {mae_b:.4f} ± {rmse_b:.4f}")
        print(f"MAE c: {mae_c:.4f} ± {rmse_c:.4f}")
        print(f"MAE d: {mae_d:.4f} ± {rmse_d:.4f}\n")
