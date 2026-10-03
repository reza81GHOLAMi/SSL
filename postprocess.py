import os
import json
import random
import shutil
from collections import defaultdict, Counter

from pydub import AudioSegment
from config import config

# Paths configurable via config.py
base_folder_path = config.get("base_folder_path", r"D:\microphone_array")
processed_path = config.get("processed_path", r"D:\processed_datasets")
divided_path = config.get("divided_path", r"D:\divided_datasets")

# Derived output directories for train, validation and test datasets
train_base_dir = os.path.join(processed_path, "train")
val_base_dir = os.path.join(processed_path, "val")
test_base_dir = os.path.join(processed_path, "test")
os.makedirs(train_base_dir, exist_ok=True)
os.makedirs(test_base_dir, exist_ok=True)
os.makedirs(val_base_dir, exist_ok=True)

def get_unique_folder_name(base_dir, folder_name):
    """Generate a unique folder name to avoid overwriting."""
    unique_name = folder_name
    counter = 1
    while os.path.exists(os.path.join(base_dir, unique_name)):
        unique_name = f"{folder_name}_{counter}"
        counter += 1
    return unique_name

def process_single_folder(folder_path):
    label_path = os.path.join(folder_path, 'label.json')
    wav_path = os.path.join(folder_path, 'output.wav')

    if os.path.exists(label_path) and os.path.exists(wav_path):
        # Load label.json
        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        # Extract metadata from JSON
        drone_data = label_data.get('drone', {})
        sound_source = drone_data.get('sound_source', 'None')

        if sound_source == 'Ambient Noise':
            folder_name = f"none_none_none_none"
        else:
            distance = drone_data.get('distance', 'none')
            height = drone_data.get('height', 'none')
            azimuth = drone_data.get('azimuth', 'none')
            movement = drone_data.get('movement', 'Static').lower()

            if movement == 'dynamic':
                print(f"  ERROR: Dynamic movement not supported in {folder_path}, skipping")
                return
            else:
                rotation = drone_data.get('rotation', 'Front')
                side_map = {"front": 1, "left": 2, "back": 3, "right": 4}
                side = side_map.get(rotation.lower() if rotation else 'none')
                if side is None:
                    print(f"  WARNING: Unknown rotation '{rotation}' in {folder_path}, skipping")
                    return
                folder_name = f"{distance}_{height}_{azimuth}_{side}"

        # Create unique train and test directories
        train_folder_name = get_unique_folder_name(train_base_dir, folder_name)
        test_folder_name = get_unique_folder_name(test_base_dir, folder_name)

        train_output_dir = os.path.join(train_base_dir, train_folder_name)
        test_output_dir = os.path.join(test_base_dir, test_folder_name)
        os.makedirs(train_output_dir, exist_ok=True)
        os.makedirs(test_output_dir, exist_ok=True)

        # Process the WAV file
        audio = AudioSegment.from_wav(wav_path)
        total_duration = len(audio) // 1000  # Convert milliseconds to seconds

        # Split into train and test
        train_duration = int(total_duration * 0.75)  # 75% of total duration

        train_audio = audio[:train_duration * 1000]  # Convert seconds to milliseconds
        test_audio = audio[train_duration * 1000:]

        # Save train audio
        train_file_path = os.path.join(train_output_dir, "train_audio.wav")
        train_audio.export(train_file_path, format="wav")

        # Save test audio
        test_file_path = os.path.join(test_output_dir, "test_audio.wav")
        test_audio.export(test_file_path, format="wav")

        print(f"Processed folder: {folder_path}\n  Train folder: {train_output_dir}\n  Test folder: {test_output_dir}")
    else:
        print(f"Required files not found in folder: {folder_path}")

def process_all_folders(base_folder_path):
    for folder_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, folder_name)
        if os.path.isdir(folder_path):
            process_single_folder(folder_path)


def summarize_metadata(base_folder_path):
    metadata_summary = defaultdict(Counter)

    for folder_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, folder_name)
        if os.path.isdir(folder_path):
            label_path = os.path.join(folder_path, 'label.json')
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
                    drone_data = label_data.get('drone', {})

                    azimuth = drone_data.get('azimuth', 'none')
                    height = drone_data.get('height', 'none')
                    rotation = drone_data.get('rotation', 'none')

                    metadata_summary['azimuths'][azimuth] += 1
                    metadata_summary['heights'][height] += 1
                    metadata_summary['rotations'][rotation] += 1

    print("Summary of Metadata:")
    for key, counter in metadata_summary.items():
        print(f"{key.capitalize()}:")
        for value, count in counter.items():
            print(f"  {value}: {count}")

def summarize_post_metadata(base_folder_path):
    metadata_summary = {
        'train': defaultdict(Counter),
        'test': defaultdict(Counter)
    }

    for dataset_type in ['train', 'test']:
        dataset_path = os.path.join(base_folder_path, dataset_type)
        if not os.path.exists(dataset_path):
            print(f"{dataset_type.capitalize()} dataset path not found: {dataset_path}")
            continue

        for folder_name in os.listdir(dataset_path):
            folder_path = os.path.join(dataset_path, folder_name)
            if os.path.isdir(folder_path):
                # Parse metadata from folder names (e.g., 10_20_45_1 or none_none_none_none_2)
                folder_parts = folder_name.split('_')
                if len(folder_parts) >= 4:
                    distance, height, azimuth, side = folder_parts[:4]

                    metadata_summary[dataset_type]['distances'][distance] += 1
                    metadata_summary[dataset_type]['heights'][height] += 1
                    metadata_summary[dataset_type]['azimuths'][azimuth] += 1
                    metadata_summary[dataset_type]['sides'][side] += 1

    for dataset_type, summary in metadata_summary.items():
        print(f"\nSummary of Metadata for {dataset_type.capitalize()}:")
        for key, counter in summary.items():
            print(f"{key.capitalize()}:")
            for value, count in counter.items():
                print(f"  {value}: {count}")

def divide_files_to_new_structure(base_folder_path, new_base_path, step_train=500, clip_duration_train=1500, clip_duration_test=1000, total_clips_test=20):
    for dataset_type in ['train', 'val', 'test']:
        dataset_path = os.path.join(base_folder_path, dataset_type)
        new_dataset_path = os.path.join(new_base_path, dataset_type)
        os.makedirs(new_dataset_path, exist_ok=True)

        if not os.path.exists(dataset_path):
            print(f"{dataset_type.capitalize()} dataset path not found: {dataset_path}")
            continue

        print(f"Processing {dataset_type} dataset...")
        total_folders = len(os.listdir(dataset_path))
        for idx, folder_name in enumerate(os.listdir(dataset_path), start=1):
            folder_path = os.path.join(dataset_path, folder_name)
            new_folder_path = os.path.join(new_dataset_path, folder_name)
            os.makedirs(new_folder_path, exist_ok=True)

            if os.path.isdir(folder_path):
                wav_file_path = os.path.join(folder_path, f"{dataset_type}_audio.wav")
                if os.path.exists(wav_file_path):
                    audio = AudioSegment.from_wav(wav_file_path)
                    duration_ms = len(audio)

                    if dataset_type == 'train':
                        total_clips = (duration_ms - clip_duration_train) // step_train + 1
                        print(f"Processing folder {idx}/{total_folders} ({folder_name}): {total_clips} train clips")
                        for start_ms in range(0, duration_ms - clip_duration_train + 1, step_train):
                            end_ms = start_ms + clip_duration_train
                            clip = audio[start_ms:end_ms]
                            clip_file_name = f"clip_{start_ms}_{end_ms}.wav"
                            clip.export(os.path.join(new_folder_path, clip_file_name), format="wav")

                    elif dataset_type in ('test', 'val'):
                        available_starts = range(0, duration_ms - clip_duration_test + 1, clip_duration_test)
                        random_starts = random.sample(list(available_starts), min(total_clips_test, len(available_starts)))
                        print(f"Processing folder {idx}/{total_folders} ({folder_name}): {len(random_starts)} test clips")
                        for i, start_ms in enumerate(random_starts):
                            end_ms = start_ms + clip_duration_test
                            clip = audio[start_ms:end_ms]
                            clip_file_name = f"random_clip_{i + 1}.wav"
                            clip.export(os.path.join(new_folder_path, clip_file_name), format="wav")

        print(f"Finished processing {dataset_type} dataset.")

    print(f"New dataset structure created at: {new_base_path}")


def split_recordings_by_folder(base_folder_path):
    """Split recordings at the folder level (no data leakage).
    Groups recordings by (distance, height, azimuth), randomly assigns
    one recording per group to test and (when make_val_split is enabled)
    one to val, rest to train. Ambient recordings stay train-only: their
    all-zeros label is a pseudo-target, not a position, so they would
    distort validation/test localization metrics.
    """
    recordings = []
    ambient_recordings = []

    for raw_folder_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, raw_folder_name)
        if not os.path.isdir(folder_path):
            continue

        label_path = os.path.join(folder_path, 'label.json')
        wav_path = os.path.join(folder_path, 'output.wav')

        if not (os.path.exists(label_path) and os.path.exists(wav_path)):
            continue

        with open(label_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        drone_data = label_data.get('drone', {})
        sound_source = drone_data.get('sound_source', 'None')

        if sound_source == 'Ambient Noise':
            ambient_recordings.append((folder_path, wav_path))
            continue

        distance = drone_data.get('distance', 'none')
        height = drone_data.get('height', 'none')
        azimuth = drone_data.get('azimuth', 'none')
        movement = drone_data.get('movement', 'Static').lower()

        if movement == 'dynamic':
            print(f"  ERROR: Dynamic movement not supported in {raw_folder_name}, skipping")
            continue
        else:
            rotation = drone_data.get('rotation', 'Front')
            side_map = {"front": 1, "left": 2, "back": 3, "right": 4}
            side = side_map.get(rotation.lower() if rotation else 'none')
            if side is None:
                print(f"  WARNING: Unknown rotation '{rotation}' in {raw_folder_name}, skipping")
                continue
            out_folder_name = f"{distance}_{height}_{azimuth}_{side}"

        group_key = (distance, height, azimuth)
        recordings.append({
            'folder_path': folder_path,
            'wav_path': wav_path,
            'out_folder_name': out_folder_name,
            'group_key': group_key,
        })

    groups = defaultdict(list)
    for rec in recordings:
        groups[rec['group_key']].append(rec)

    random.seed(config.get("split_seed", 42))
    make_val = config.get("make_val_split", True)

    train_count = 0
    val_count = 0
    test_count = 0

    for group_key, group_recordings in groups.items():
        test_recording = random.choice(group_recordings)
        remaining = [r for r in group_recordings if r != test_recording]

        val_recording = None
        if make_val and len(remaining) >= 2:
            val_recording = random.choice(remaining)
            remaining = [r for r in remaining if r != val_recording]

        test_folder_name = get_unique_folder_name(test_base_dir, test_recording['out_folder_name'])
        test_output_dir = os.path.join(test_base_dir, test_folder_name)
        os.makedirs(test_output_dir, exist_ok=True)
        shutil.copy2(test_recording['wav_path'], os.path.join(test_output_dir, "test_audio.wav"))
        test_count += 1

        if val_recording is not None:
            val_folder_name = get_unique_folder_name(val_base_dir, val_recording['out_folder_name'])
            val_output_dir = os.path.join(val_base_dir, val_folder_name)
            os.makedirs(val_output_dir, exist_ok=True)
            shutil.copy2(val_recording['wav_path'], os.path.join(val_output_dir, "val_audio.wav"))
            val_count += 1

        for rec in remaining:
            train_folder_name = get_unique_folder_name(train_base_dir, rec['out_folder_name'])
            train_output_dir = os.path.join(train_base_dir, train_folder_name)
            os.makedirs(train_output_dir, exist_ok=True)
            shutil.copy2(rec['wav_path'], os.path.join(train_output_dir, "train_audio.wav"))
            train_count += 1

    for folder_path, wav_path in ambient_recordings:
        train_folder_name = get_unique_folder_name(train_base_dir, "none_none_none_none")
        train_output_dir = os.path.join(train_base_dir, train_folder_name)
        os.makedirs(train_output_dir, exist_ok=True)
        shutil.copy2(wav_path, os.path.join(train_output_dir, "train_audio.wav"))
        train_count += 1

    print(f"Recording-level split complete:")
    print(f"  Train recordings: {train_count}")
    print(f"  Val recordings: {val_count}")
    print(f"  Test recordings: {test_count}")
    print(f"  Total groups: {len(groups)}")
    print(f"  Ambient (train only): {len(ambient_recordings)}")


def main():
    print("What would you like to do?")
    print("1: Summarize before preprocess metadata")
    print("2: Summarize after preprocess metadata")
    print("3: Process folders to% 75 and 25% length")
    print("4: Divide into small files")
    print("5: Split recordings by folder (no data leakage)")
    choice = input("Enter your choice (1-5): ").strip()

    if choice == "1":
        summarize_metadata(processed_path)
    elif choice == "2":
        summarize_post_metadata(processed_path)
    elif choice == "3":
        process_all_folders(base_folder_path)
    elif choice == "4":
        divide_files_to_new_structure(processed_path, divided_path)
    elif choice == "5":
        split_recordings_by_folder(base_folder_path)
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()