import pandas as pd
import numpy as np
from tqdm import tqdm


# csv file path
fp = "/local_disk/apollon/rwhetten/loquacious_large_train.csv"
# load in df
df = pd.read_csv(fp)
# sort
df.sort_values(["duration"], ascending=False, inplace=True)

# get longest files until total duration = total_audio_sec
total_hours = 12601
total_audio_sec = total_hours * 3600

selected_files = []
current_duration = 0

# Iterate through the sorted DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
    # Check if adding the current file's duration will exceed the total
    if current_duration + row['duration'] <= total_audio_sec:
        # If not, add the file to the selected list and update the duration
        selected_files.append(row)
        current_duration += row['duration']
    else:
        # If it will exceed, stop the loop since we have the longest files
        # and adding more would be counterproductive
        break

# Create a new DataFrame from the selected files
selected_df = pd.DataFrame(selected_files)

# Print the final result
print(f"Total files selected: {len(selected_df)}")
print(f"Total duration of selected files: {current_duration:.2f} seconds")
print(selected_df.head())

selected_df.to_csv("/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/loq_large/length_0.5.csv", index=False)


# fp = "/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/loq_medium/mfcc_0.5.csv"
# df = pd.read_csv(fp)
# print(len(df))
# df.duration.sum() / 3600


# fp = "/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/loq_medium/random_0.5.csv"
# df = pd.read_csv(fp)
# print(len(df))
# df.duration.sum() / 3600

# fp = "/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/loq_medium/sense_0.5.csv"
# df = pd.read_csv(fp)
# print(len(df))
# df.duration.sum() / 3600

# fp = "/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/loq_medium/speaker_0.5.csv"
# df = pd.read_csv(fp)
# print(len(df))
# df.duration.sum() / 3600