import glob
import itertools
import math
import os
import random

import albumentations as A
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
from torchaudio.functional import compute_deltas
try:
    import torchaudio.prototype.transforms as PT
except (ModuleNotFoundError, ImportError):
    PT = None
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import config


class RandomApply:
    def __init__(self, transform, p=0.5):
        """
        Wrapper that applies a transformation with a probability of `p`.

        Args:
            transform (callable): Transformation to apply.
            p (float): Probability of applying the transformation.
        """
        self.transform = transform
        self.p = p

    def __call__(self, waveform):
        if random.random() < self.p:
            return self.transform(waveform)
        return waveform


class ComposeAudioTransforms:
    def __init__(self, transforms):
        """
        Composer of audio transformations.

        Args:
            transforms (list): List of transformations to apply.
        """
        self.transforms = transforms

    def __call__(self, waveform):
        for transform in self.transforms:
            waveform = transform(waveform)
        return waveform


def random_crop(audio, sample_rate, target_duration=1.0):
    """
    Randomly crops audio to a specified length.

    Args:
        audio (torch.Tensor): Audio tensor with shape [channels, samples].
        sample_rate (int): Sampling rate.
        target_duration (float): Target duration in seconds.

    Returns:
        torch.Tensor: Cropped audio.
    """
    target_length = int(target_duration * sample_rate)
    total_length = audio.size(1)
    if total_length < target_length:
        padding = target_length - total_length
        audio = torch.nn.functional.pad(audio, (0, padding))
    elif total_length > target_length:
        start = random.randint(0, total_length - target_length)
        audio = audio[:, start:start + target_length]
    return audio


# Canonical label vector layout shared by the dataset and the trainer:
# [distance/50, height/50, hypotenuse/50, sin(az), cos(az), sin(side), cos(side)]
CANONICAL_LABELS = ["distance", "height", "hypotenuse", "azimuth_sin", "azimuth_cos",
                    "side_sin", "side_cos"]


def build_target_spec(cfg):
    """Ordered list of active regression targets given the use_* config flags.

    Azimuth (sin/cos) is always active. Each name is a key into CANONICAL_LABELS,
    so 'azimuth' expands to the azimuth_sin/azimuth_cos columns.
    """
    spec = []
    if cfg.get("use_distance", True):
        spec.append("distance")
    if cfg.get("use_height", True):
        spec.append("height")
    if cfg.get("use_hypotenuse", False):
        spec.append("hypotenuse")
    spec.append("azimuth")
    if cfg.get("use_side", True):
        spec.append("side")
    return spec


def target_label_indices(cfg):
    """Flat list of CANONICAL_LABELS indices covered by the active target spec."""
    indices = []
    for name in build_target_spec(cfg):
        if name == "azimuth":
            indices += [CANONICAL_LABELS.index("azimuth_sin"), CANONICAL_LABELS.index("azimuth_cos")]
        elif name == "side":
            indices += [CANONICAL_LABELS.index("side_sin"), CANONICAL_LABELS.index("side_cos")]
        else:
            indices.append(CANONICAL_LABELS.index(name))
    return indices


class AudioSpectrogramDataset(Dataset):
    def __init__(self, config, mode='train', duration=1.0):
        """
        Args:
            config (dict): Configuration containing 'dataset_path' and other parameters.
            mode (str): 'train' or 'test'.
            sample_rate (int): Sampling rate.
            duration (float): Length of the audio sample in seconds.
        """
        self.cfg = config
        self.root_dir = self.cfg["dataset_path"]
        self.mode = mode
        self.duration = duration
        self.width = self.cfg.get("width", 256)
        self.height = self.cfg.get("height", 128)
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80)
        feature_type = self.cfg.get("feature_type", "mel")  # Default to Mel Spectrogram
        self.feature_type = feature_type

        self.sample_rate = self.cfg["sample_rate"]
        self.n_mels = self.cfg["n_mels"]
        self.n_fft = self.cfg["n_fft"]
        self.hop_length = self.cfg["hop_length"]
        self.n_mfcc = self.cfg["n_mfcc"]
        self.n_lfcc = self.cfg["n_lfcc"]
        self.n_barks = self.cfg["n_barks"]

        # Inter-channel spatial features (IPD/ILD), stacked after the mel channels
        self.use_ipd = self.cfg.get("use_ipd", False)
        # Temporal delta (Δ) of the mel spectrogram per mic. delta_layout picks
        # the stacking axis: "channel" = extra input channels (Δ of a (f, t)
        # cell aligned with its static value from the first conv on),
        # "freq" = Δ rows below the static rows, "time" = Δ half right of the
        # static half (original dlt014/dlt0257 layout)
        self.use_delta = self.cfg.get("use_delta", False)
        self.delta_win = self.cfg.get("delta_win", 9)
        delta_layouts = {"channel": 0, "freq": 1, "time": 2}
        self.delta_layout = self.cfg.get("delta_layout", "channel")
        if self.delta_layout not in delta_layouts:
            raise RuntimeError(f"Unknown delta_layout '{self.delta_layout}' "
                               f"(expected one of {sorted(delta_layouts)})")
        self.delta_dim = delta_layouts[self.delta_layout]
        self.ipd_pairs = list(itertools.combinations(range(len(self.cfg.get("channels", range(8)))), 2))
        if self.use_ipd or feature_type == "rainbow":
            self.stft_window = torch.hann_window(self.n_fft)

        if feature_type == "stft":
            # No Resample here: librosa.load(sr=...) already delivers audio at
            # self.sample_rate, so resampling again stretched every stft run with
            # sample_rate != audio_orig_freq (BUG-32 in the code notes)
            self.transform = T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
        elif feature_type == "rainbow":
            # Rainbowgram input: the magnitude + instantaneous-frequency maps need
            # the complex STFT (phase), computed in _rainbowgram during __getitem__
            self.transform = None

        elif feature_type == "lfcc":
            self.transform = T.LFCC(sample_rate=self.sample_rate, n_lfcc=self.n_lfcc,
                                    speckwargs={"n_fft": self.n_fft, "hop_length": self.hop_length})
        elif feature_type == "mfcc":
            self.transform = T.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_mels': self.n_mels,
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length
                }
            )
        elif feature_type == "bark":
            self.transform = PT.BarkSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft,
                                                hop_length=self.hop_length, n_barks=self.n_barks)
        elif feature_type == "mel":
            self.transform = T.MelSpectrogram(sample_rate=self.sample_rate, n_fft=self.n_fft,
                                              hop_length=self.hop_length,
                                              n_mels=self.n_mels)
        else:
            raise RuntimeError("Features not extracted")

        if self.mode == 'train':
            self.data_dir = os.path.join(self.root_dir, 'train')
        else:
            self.data_dir = os.path.join(self.root_dir, 'test')
        self.classes = []
        self.file_paths = []
        self.labels = []
        # Assuming that folders follow a format such as a_b_c_d, where a is distance, b is height, c is azimuth and d is side
        for folder in sorted(os.listdir(self.data_dir)):
            folder_path = os.path.join(self.data_dir, folder)
            if os.path.isdir(folder_path):
                parts = folder.split('_')
                if len(parts) < 4:
                    continue  # Skipping folders without a sufficient number of labels
                a, b, c, d = parts[:4]
                self.classes.append((a, b, c))
                wav_files = glob.glob(os.path.join(folder_path, '*.wav'))
                self.file_paths.extend(wav_files)
                # Repeating labels for each file in the folder
                self.labels.extend([(a, b, c, d) for _ in wav_files])

        # Definicja augmentacji obrazowych
        if self.mode == 'train':
            self.image_transform = A.Compose([
                # A.GaussNoise(p=0.3, var_limit=(0.001, 0.02)),
                # A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=0.3),
                A.Resize(width=self.width, height=self.height, p=1)])
        else:
            # Only resize in test mode
            self.image_transform = A.Compose([A.Resize(width=self.width, height=self.height, p=1)])

        # Definicja augmentacji audio dla trybu treningowego
        if self.mode == 'train':
            self.audio_transform = ComposeAudioTransforms(
                [  # RandomApply(T.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015), p=0.3),
                    # T.Fade(fade_in_len=int(0.1 * sample_rate), fade_out_len=int(0.1 * sample_rate))
                ])
        else:
            # No augmentation in test mode
            self.audio_transform = None

    def __len__(self):
        return len(self.file_paths)

    def _rainbowgram(self, waveform):
        """Rainbowgram input maps (term after Engel et al. 2020, DDSP): per mic one
        log-magnitude map (the brightness of the visualization) plus one
        instantaneous-frequency map (its hue). Returns [2C, n_fft//2+1, T] with
        all C magnitude channels first, then the C IF channels — same block
        order as the IPD/ILD maps. Magnitude is power-dB with top_db=80 like
        the mel path, min-max normalized over the whole stack. IF is the
        phase-vocoder deviation from the bin center, scaled so [0, 1] spans
        the unambiguous range of +/- n_fft/(2*hop_length) bins around each
        bin center (0.5 = energy at the center); bins floored by
        top_db carry no reliable phase and get the neutral 0.5."""
        spec = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                          window=self.stft_window, center=True, return_complex=True)  # [C, F, T]
        mag_db = self.amplitude_to_db(spec.abs().pow(2))
        mag = (mag_db - mag_db.min()) / (mag_db.max() - mag_db.min() + 1e-9)

        # Heterodyned phase increment minus the advance of a tone sitting at the
        # bin center, wrapped to [-pi, pi]. Divided by 2*pi it is the deviation
        # as a fraction of the unambiguous +/-n_fft/(2*hop_length)-bin range,
        # so +0.5 maps it to [0, 1] with 0.5 = at the bin center.
        dphi = torch.angle(spec[..., 1:] * torch.conj(spec[..., :-1]))
        bins = torch.arange(spec.size(-2), dtype=dphi.dtype)
        expected = (2 * math.pi * self.hop_length * bins / self.n_fft).view(1, -1, 1)
        dev = torch.remainder(dphi - expected + math.pi, 2 * math.pi) - math.pi
        dev = torch.nn.functional.pad(dev / (2 * math.pi), (1, 0)) + 0.5  # frame 0 = neutral
        dev = torch.where(mag_db > mag_db.max() - 80.0, dev, torch.full_like(dev, 0.5))
        return torch.cat([mag, dev], dim=0)

    def __getitem__(self, idx):
        wav_path = self.file_paths[idx]
        labels = self.labels[idx]
        if labels == ("none", "none", "none", "none"):
            a_normalized = 0.0
            b_normalized = 0.0
            hyp_normalized = 0.0
            c_sin_normalized = 0.0
            c_cos_normalized = 0.0
            d_sin_normalized = 0.0
            d_cos_normalized = 0.0
        else:
            a, b, c, d = labels
            try:
                a_normalized = float(a) / 50.0  # Scaling 'a' to the range [0,1]
                b_normalized = float(b) / 50.0  # Scaling 'b' to the range [0,1]
                hyp_normalized = math.sqrt(float(a) ** 2 + float(b) ** 2) / 50.0  # Slant range
                c = float(c)
                d = int(d)  # Assuming that 'd' is an integer
            except ValueError:
                raise ValueError(f"Labels must be numeric: a={a}, b={b}, c={c}, d={d}")

            # Transforming the value of 'c' (degrees) into sine and cosine
            c_rad = np.deg2rad(c)
            c_sin = np.sin(c_rad)
            c_cos = np.cos(c_rad)

            # Normalizing sine and cosine values to the range [0,1]
            c_sin_normalized = (c_sin + 1.0) / 2.0
            c_cos_normalized = (c_cos + 1.0) / 2.0

            # Mapping for 'd': 1 -> 0°, 2 -> 270°, 3 -> 180°, 4 -> 90°
            d_angle_map = {1: 0, 2: 270, 3: 180, 4: 90}
            angle_d = d_angle_map.get(d, 0) # Default value is 0° if 'd' is not in the map

            # Transforming the value of 'd' (degrees) into sine and cosine
            d_rad = np.deg2rad(angle_d)
            d_sin = np.sin(d_rad)
            d_cos = np.cos(d_rad)

            # Normalizing sine and cosine values to the range [0,1]
            d_sin_normalized = (d_sin + 1.0) / 2.0
            d_cos_normalized = (d_cos + 1.0) / 2.0

        labels_tensor = torch.tensor(
            [a_normalized, b_normalized, hyp_normalized, c_sin_normalized, c_cos_normalized,
             d_sin_normalized, d_cos_normalized], dtype=torch.float32)

        try:
            waveform, sr = librosa.load(wav_path, sr=self.sample_rate, mono=False)
            waveform = torch.from_numpy(waveform).float()
        except Exception as e:
            raise RuntimeError(f"Error loading {wav_path}: {e}")

        # Checking number of channels
        if waveform.size(0) != 8:
            raise ValueError(f"Expected 8 channels, but found {waveform.size(0)} in file {wav_path}")

        # Selecting microphone subset (0-based channel indices, see config["channels"])
        waveform = waveform[self.cfg.get("channels", range(8)), :]

        if self.mode == 'train':
            # Trimming to the specified length only in training mode
            waveform = random_crop(waveform, self.sample_rate, target_duration=self.duration)
        else:
            # In test mode, ensure the length is exactly 1 second
            # Assuming that test files are exactly 1 second long
            waveform = waveform

        # Applying audio augmentation only in training mode
        if self.audio_transform:
            waveform = self.audio_transform(waveform)

        # Normalization
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-9)

        if self.use_ipd:
            spec = torch.stft(waveform, n_fft=self.n_fft, hop_length=self.hop_length,
                              window=self.stft_window, center=True, return_complex=True)

        if self.feature_type == "rainbow":
            spectrograms = self._rainbowgram(waveform)  # [2C, F, T], already in [0, 1]
        else:
            # Generating spectrograms for each channel
            mel_spectrograms = []
            for channel in waveform:
                mel_spec = self.transform(channel)
                mel_spec_db = self.amplitude_to_db(mel_spec)
                mel_spectrograms.append(mel_spec_db)

            # Stacking spectrograms into a single tensor [8, n_mels, time_frames]
            spectrograms = torch.stack(mel_spectrograms, dim=0)  # Shape: [8, n_mels, time_frames]

            # Normalizing spectrograms to the range [0, 1]
            spectrograms = (spectrograms - spectrograms.min()) / (spectrograms.max() - spectrograms.min() + 1e-9)

        if self.use_delta:
            # Per-mic temporal delta appended along the delta_layout axis:
            # channel -> [2C, n_mels, T] (Δ aligned with its static value per
            # pixel; conv_in inputs double, ~<0.1% more params);
            # freq -> [C, 2*n_mels, T] (static rows on top, Δ rows below;
            # parameter-matched, but the resize 2x-downsamples mel rows on
            # 256-mel configs); time -> [C, n_mels, 2T] (static left, Δ right;
            # the original dlt014/dlt0257 layout, parameter-matched).
            # Regression delta keeps the frame count (replicate padding), so
            # both halves get equal size along the stacking axis after the
            # resize. Delta is symmetric-normalized so zero-change maps to
            # 0.5 — a joint min-max would drag negative deltas across the
            # whole range and squeeze the static part.
            delta = torch.stack([compute_deltas(ch, win_length=self.delta_win)
                                 for ch in spectrograms], dim=0)
            delta = (delta / (delta.abs().max() + 1e-9) + 1.0) / 2.0
            spectrograms = torch.nn.functional.interpolate(
                torch.cat([spectrograms, delta], dim=self.delta_dim).unsqueeze(0),
                size=(self.height, self.width), mode="bilinear",
                align_corners=False).squeeze(0).clamp(0.0, 1.0)
        else:
            # Transposing from [C, H, W] to [H, W, C] for image transformations
            spectrograms_np = spectrograms.permute(1, 2, 0).cpu().numpy()  # Shape: [n_mels, time_frames, 8]

            # Applying image augmentations
            if self.image_transform:
                augmented = self.image_transform(image=spectrograms_np)
                spectrograms_np = augmented['image']  # Shape: [height, width, 8]

            # Transposing back to [C, H_new, W_new]
            spectrograms_np = spectrograms_np.transpose(2, 0, 1)  # Shape: [8, height, width]

            # Converting to tensors
            spectrograms = torch.tensor(spectrograms_np, dtype=torch.float32)  # Shape: [8, height, width]

        if self.use_ipd:
            # Spatial feature maps per mic pair: sin(IPD), cos(IPD), ILD(dB)
            spatial = []
            for i, j in self.ipd_pairs:
                ipd = torch.angle(spec[i] * torch.conj(spec[j]))
                spatial.append((torch.sin(ipd) + 1.0) / 2.0)
                spatial.append((torch.cos(ipd) + 1.0) / 2.0)
                ild = 20.0 * torch.log10(spec[i].abs() + 1e-9) - 20.0 * torch.log10(spec[j].abs() + 1e-9)
                spatial.append((ild.clamp(-40.0, 40.0) + 40.0) / 80.0)
            spatial = torch.stack(spatial, dim=0)  # [3*n_pairs, n_fft/2+1, T]
            spatial = torch.nn.functional.interpolate(
                spatial.unsqueeze(0), size=(self.height, self.width),
                mode="bilinear", align_corners=False).squeeze(0).clamp(0.0, 1.0)
            spectrograms = torch.cat([spectrograms, spatial], dim=0)

        return spectrograms, labels_tensor

def plot_spectrograms(spectrograms, num_channels=8, rows=2, cols=4):
    """
    Creates a grid of plots for spectrograms.

    Args:
        spectrograms (torch.Tensor): Tensor of spectrograms with shape [batch_size, 8, H, W].
        num_channels (int): Number of channels (default is 8).
        rows (int): Number of rows in the subplot grid.
        cols (int): Number of columns in the subplot grid.
    """
    batch_size = spectrograms.size(0)
    for batch_idx in range(batch_size):
        fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
        fig.suptitle(f'Channel Spectrograms - Sample {batch_idx + 1}', fontsize=16)

        for channel in range(num_channels):
            row = channel // cols
            col = channel % cols
            ax = axes[row, col]

            # Extracting the spectrogram for the given channel
            spec = spectrograms[batch_idx, channel].cpu().numpy()

            # Displaying the spectrogram
            im = ax.imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            ax.set_title(f'Kanał {channel + 1}')
            ax.axis('off')
            fig.colorbar(im, ax=ax, format='%+2.0f dB')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


if __name__ == "__main__":
    # Initializing Dataset and DataLoader for training
    train_dataset = AudioSpectrogramDataset(config, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)

    # Initializing Dataset and DataLoader for testing
    test_dataset = AudioSpectrogramDataset(config, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    # Example of iterating through the training DataLoader
    print("Example of training data:")
    for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        spectrograms, labels = data
        print(f"Spectrograms shape: {spectrograms.shape}")  # Expected shape: [1, 8, 128, 256]
        print(f"Labels shape: {labels.shape}")  # Expected shape: [1, 7]
        print(f"Labels: {labels}")
        plot_spectrograms(spectrograms)
        print(f"Max value: {torch.max(spectrograms)}, Min value: {torch.min(spectrograms)}")
        break

    # Example of iterating through the testing DataLoader
    print("\nExample of testing data:")
    for i, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
        spectrograms, labels = data
        print(f"Spectrograms shape: {spectrograms.shape}")  # Expected shape: [1, 8, 128, 256]
        print(f"Labels shape: {labels.shape}")  # Expected shape: [1, 7]
        print(f"Labels: {labels}")
        plot_spectrograms(spectrograms)
        print(f"Max value: {torch.max(spectrograms)}, Min value: {torch.min(spectrograms)}")
        break
