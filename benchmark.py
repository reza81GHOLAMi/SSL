import os
import time
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import postprocess

def get_dir_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

def count_files(path):
    count = 0
    for dirpath, dirnames, filenames in os.walk(path):
        count += len(filenames)
    return count

NUM_BENCH_FOLDERS = len([d for d in os.listdir(postprocess.base_folder_path) if os.path.isdir(os.path.join(postprocess.base_folder_path, d))])
TOTAL_FOLDERS = 132
INPUT_SIZE = get_dir_size(postprocess.base_folder_path)

print("=" * 60)
print("PREPROCESSING BENCHMARK")
print("=" * 60)
print(f"Input folders: {NUM_BENCH_FOLDERS}")
print(f"Input size: {INPUT_SIZE / 1e6:.1f} MB")
print()

# --- Stage 1: 75/25 Split (option 3) ---
print("-" * 60)
print("Stage 1: 75/25 Split (process_all_folders)")
print("-" * 60)
t0 = time.time()
postprocess.process_all_folders(postprocess.base_folder_path)
t1 = time.time()
stage1_time = t1 - t0
stage1_per_folder = stage1_time / NUM_BENCH_FOLDERS

processed_size = get_dir_size(postprocess.processed_path)
print(f"\nTime: {stage1_time:.2f} s")
print(f"Per folder: {stage1_per_folder:.2f} s")
print(f"Processed output size: {processed_size / 1e6:.1f} MB")
print(f"Extrapolated to {TOTAL_FOLDERS} folders: {stage1_per_folder * TOTAL_FOLDERS:.1f} s ({stage1_per_folder * TOTAL_FOLDERS / 60:.1f} min)")
print()

# --- Stage 2: Clip Extraction (option 4) ---
print("-" * 60)
print("Stage 2: Clip Extraction (divide_files_to_new_structure)")
print("-" * 60)
t0 = time.time()
postprocess.divide_files_to_new_structure(postprocess.processed_path, postprocess.divided_path)
t1 = time.time()
stage2_time = t1 - t0
stage2_per_folder = stage2_time / NUM_BENCH_FOLDERS

divided_size = get_dir_size(postprocess.divided_path)
train_clips = count_files(os.path.join(postprocess.divided_path, "train")) if os.path.exists(os.path.join(postprocess.divided_path, "train")) else 0
test_clips = count_files(os.path.join(postprocess.divided_path, "test")) if os.path.exists(os.path.join(postprocess.divided_path, "test")) else 0
total_clips = train_clips + test_clips

print(f"\nTime: {stage2_time:.2f} s")
print(f"Per folder: {stage2_per_folder:.2f} s")
print(f"Divided output size: {divided_size / 1e6:.1f} MB")
print(f"Train clips: {train_clips}")
print(f"Test clips: {test_clips}")
print(f"Total clips: {total_clips}")
print(f"Clips per folder: {total_clips / NUM_BENCH_FOLDERS:.1f}")
print(f"Extrapolated to {TOTAL_FOLDERS} folders: {stage2_per_folder * TOTAL_FOLDERS:.1f} s ({stage2_per_folder * TOTAL_FOLDERS / 60:.1f} min)")
print()

# --- Summary ---
print("=" * 60)
print("EXTRAPOLATION TO FULL DATASET (132 folders)")
print("=" * 60)
total_time = (stage1_per_folder + stage2_per_folder) * TOTAL_FOLDERS
est_total_clips = int(total_clips / NUM_BENCH_FOLDERS * TOTAL_FOLDERS)
est_total_size = divided_size / NUM_BENCH_FOLDERS * TOTAL_FOLDERS / 1e9
print(f"Stage 1 (75/25 split):  {stage1_per_folder * TOTAL_FOLDERS / 60:.1f} min")
print(f"Stage 2 (clip extract):  {stage2_per_folder * TOTAL_FOLDERS / 60:.1f} min")
print(f"Total processing time:   {total_time / 60:.1f} min ({total_time / 3600:.1f} hours)")
print(f"Estimated total clips:  {est_total_clips}")
print(f"Estimated output size:  {est_total_size:.1f} GB")
print(f"Disk space needed:      {est_total_size:.1f} GB (output) + {INPUT_SIZE / NUM_BENCH_FOLDERS * TOTAL_FOLDERS / 1e9:.1f} GB (input copy, if applicable)")
print("=" * 60)
