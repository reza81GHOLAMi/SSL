config = {
    # Dataset
    "dataset_path": r"D:\divided_datasets",
    "base_folder_path": r"D:\microphone_array",
    "processed_path": r"D:\processed_datasets",
    "divided_path": r"D:\divided_datasets",
    "height": 256,
    "width": 256,

    # Microphone channel selection (0-based indices into the 8-channel wav).
    # 0-3: lower ring (N, E, S, W; 1.49 m), 4-7: upper ring (NE, SE, SW, NW; 2.38 m)
    "channels": [0, 1, 2, 3, 4, 5, 6, 7],

    # Model
    "pretrained": False,
    "model_path": 'saved_models/2025_01_10_14_35_08_/115_end_unet.pth',
    "feature_type": "lfcc",  # Options: "mel", "stft", "mfcc", "bark", "lfcc"
    "lr": 0.0002,
    "epochs": 100,
    # Regression targets included in the model head / loss / metrics.
    # Azimuth (sin/cos) is always active; the others are toggled here.
    "use_distance": True,
    "use_height": True,
    # Slant range sqrt(distance^2 + height^2), normalized by 50 like the others
    "use_hypotenuse": False,
    # Side (d outputs, drone yaw orientation) excluded from loss/metrics when False
    "use_side": False,
    "batch_size": 32,
    "save_every_epochs": 5,

    # LOGGER
    "print_step": 10,
    "show_step": 500,
    "test_step": 500,

    # ddp
    "ddp": False,
    "gpu_num": 0,
    "use_amp": False,
    "use_grad": False,
    "cooldown_seconds": 0,
    "num_workers": 2,
    "gc_collect_interval": 0,
    "lr_patience": 10,
    "test_split_ratio": 0.25,
    "split_seed": 42,
    "audio_orig_freq": 96000,

    # Runtime defaults (overridden by main.py at startup)
    "run_name": "default",
    "wandb_project": "UAV-SSL",
    "wandb_entity": "default",
    "n_gpu": 0,
    "rank": 0,

    # FEATURES
    "sample_rate": 44100,
    "n_mels": 128,
    "n_fft": 256,
    "hop_length": 512,
    "n_mfcc": 64,
    "n_lfcc": 20,
    "n_barks": 64,

    "variations": [
        # {"feature_type": "mfcc", "sample_rate": 16000, "n_fft": 512, "hop_length": 256, "n_mfcc": 13, "n_mels": 64,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mfcc", "sample_rate": 44100, "n_fft": 1024, "hop_length": 512, "n_mfcc": 26, "n_mels": 128,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mfcc", "sample_rate": 96000, "n_fft": 4096, "hop_length": 2048, "n_mfcc": 40, "n_mels": 256,
        #  "n_lfcc": 0, "n_barks": 0},

        # {"feature_type": "mel", "sample_rate": 16000, "n_fft": 512, "hop_length": 256, "n_mfcc": 0, "n_mels": 64,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 1024, "hop_length": 512, "n_mfcc": 0, "n_mels": 128,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 4096, "hop_length": 2048, "n_mfcc": 0, "n_mels": 256,
        #  "n_lfcc": 0, "n_barks": 0},

        # {"feature_type": "stft", "sample_rate": 16000, "n_fft": 512, "hop_length": 256, "n_mfcc": 0, "n_mels": 0,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "stft", "sample_rate": 44100, "n_fft": 1024, "hop_length": 512, "n_mfcc": 0, "n_mels": 0,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "stft", "sample_rate": 96000, "n_fft": 4096, "hop_length": 2048, "n_mfcc": 0, "n_mels": 0,
        #  "n_lfcc": 0, "n_barks": 0},
        #
        # {"feature_type": "lfcc", "sample_rate": 16000, "n_fft": 512, "hop_length": 256, "n_mfcc": 0, "n_mels": 0,
        #  "n_lfcc": 13, "n_barks": 0},
        # {"feature_type": "lfcc", "sample_rate": 44100, "n_fft": 1024, "hop_length": 512, "n_mfcc": 0, "n_mels": 0,
        #  "n_lfcc": 26, "n_barks": 0},
        # {"feature_type": "lfcc", "sample_rate": 96000, "n_fft": 4096, "hop_length": 2048, "n_mfcc": 0, "n_mels": 0,
        #  "n_lfcc": 40, "n_barks": 0},
        #
        # {"feature_type": "bark", "sample_rate": 16000, "n_fft": 512, "hop_length": 256, "n_mfcc": 0, "n_mels": 0,
        #  "n_lfcc": 0, "n_barks": 64},
        # {"feature_type": "bark", "sample_rate": 44100, "n_fft": 1024, "hop_length": 512, "n_mfcc": 0, "n_mels": 0,
        #  "n_lfcc": 0, "n_barks": 128},
        # {"feature_type": "bark", "sample_rate": 96000, "n_fft": 4096, "hop_length": 2048, "n_mfcc": 0, "n_mels": 0,
        #  "n_lfcc": 0, "n_barks": 256},

        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 1024, "hop_length": 256, "n_mfcc": 0, "n_mels": 64,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 1024, "hop_length": 512, "n_mfcc": 0, "n_mels": 64,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 1024, "hop_length": 256, "n_mfcc": 0, "n_mels": 128,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 1024, "hop_length": 512, "n_mfcc": 0, "n_mels": 128,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 1024, "hop_length": 256, "n_mfcc": 0, "n_mels": 256,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 1024, "hop_length": 512, "n_mfcc": 0, "n_mels": 256,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 2048, "hop_length": 512, "n_mfcc": 0, "n_mels": 64,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 2048, "hop_length": 1024, "n_mfcc": 0, "n_mels": 64,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 2048, "hop_length": 512, "n_mfcc": 0, "n_mels": 128,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 2048, "hop_length": 1024, "n_mfcc": 0, "n_mels": 128,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 2048, "hop_length": 512, "n_mfcc": 0, "n_mels": 256,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 44100, "n_fft": 2048, "hop_length": 1024, "n_mfcc": 0, "n_mels": 256,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 2048, "hop_length": 512, "n_mfcc": 0, "n_mels": 64,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 2048, "hop_length": 1024, "n_mfcc": 0, "n_mels": 64,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 2048, "hop_length": 512, "n_mfcc": 0, "n_mels": 128,
        #  "n_lfcc": 0, "n_barks": 0},
        # 3-mic ablation: NE upper + N/E lower (indices 4, 0, 1)
        {"feature_type": "mel", "sample_rate": 96000, "n_fft": 2048, "hop_length": 1024, "n_mfcc": 0, "n_mels": 128,
         "n_lfcc": 0, "n_barks": 0, "channels": [4, 0, 1]},
        # 3-mic ablation: SW upper + N/E lower (indices 6, 1, 0)
        {"feature_type": "mel", "sample_rate": 96000, "n_fft": 2048, "hop_length": 1024, "n_mfcc": 0, "n_mels": 128,
         "n_lfcc": 0, "n_barks": 0, "channels": [6, 1, 0]},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 2048, "hop_length": 512, "n_mfcc": 0, "n_mels": 256,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 2048, "hop_length": 1024, "n_mfcc": 0, "n_mels": 256,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 4096, "hop_length": 1024, "n_mfcc": 0, "n_mels": 64,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 4096, "hop_length": 2048, "n_mfcc": 0, "n_mels": 64,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 4096, "hop_length": 1024, "n_mfcc": 0, "n_mels": 128,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 4096, "hop_length": 2048, "n_mfcc": 0, "n_mels": 128,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 4096, "hop_length": 1024, "n_mfcc": 0, "n_mels": 256,
        #  "n_lfcc": 0, "n_barks": 0},
        # {"feature_type": "mel", "sample_rate": 96000, "n_fft": 4096, "hop_length": 2048, "n_mfcc": 0, "n_mels": 256,
        #  "n_lfcc": 0, "n_barks": 0}
    ]
}
