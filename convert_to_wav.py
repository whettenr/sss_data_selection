import os
from datasets import load_dataset
from speechbrain.dataio.dataio import read_audio, write_audio
from tqdm import tqdm
from loquacious_set_prepare import load_datasets


# 1. Load the audio dataset
split = "large"
hf_data_dict = load_datasets(
    split,
    "speechbrain/LoquaciousSet",
    "/local_disk/apollon/rwhetten/hf_root/datasets",
)

# 3. Output directory
ds = "train"
samplerate=16000
output_dir = f"/local_disk/apollon/rwhetten/wavs/{split}/{ds}"
os.makedirs(output_dir, exist_ok=True)

for idx, sample in enumerate(tqdm(hf_data_dict[ds])):
    wav = sample["wav"]
    audio = read_audio(wav["bytes"])
    
    # Create a filename
    filename = os.path.join(output_dir, f"{sample['ID']}.wav")
    # Write to disk
    write_audio(filename, audio, samplerate)
    
    # if idx % 20000 == 0:
    #     print(f"Exported {idx} files...")

print("✅ Export complete.")