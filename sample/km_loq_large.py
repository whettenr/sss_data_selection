import os
import pickle
import pandas as pd
import numpy as np
import glob
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt
from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from utils import (
    run_kmeans, 
    diverse_sample, 
    sample_kmeans_upto, 
    get_rand_csvs, 
    load_all_features
)


seed = 10
np.random.seed(seed)

split_name = 'large' # medium | large
mfcc_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/loq/mel_large_ckpt.pkl"
speaker_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/loq/speaker_large_ckpt.pkl"
sense_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/loq/sense_large_features"
csv_location = f"/local_disk/apollon/rwhetten/loquacious_{split_name}_train.csv"
save_folder = f"/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/loq_{split_name}"
fractions = [0.5]
K=200

mfcc_ff = 0.011
speaker_ff = -0.035
sense_ff = 0.105

# RAND 4743938 files and 12601.16 hours


# create save_folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Folder '{save_folder}' created.")
else:
    print(f"Folder '{save_folder}' already exists.")

main_df = pd.read_csv(csv_location)

# ### random ###
get_rand_csvs(csv_location, fractions, save_folder)


paths = [mfcc_path, speaker_path, sense_path]
names = ["mfcc", "speaker", "sense"]
ff = [mfcc_ff, speaker_ff, sense_ff]

for i in range(len(paths)):
    name = names[i]
    p = paths[i]
    print(f"Working on {p}")

    df_path = f"{save_folder}/{name}_clusters_{split_name}_{K}.csv"
    if os.path.exists(df_path):
        print(f"Loading existing clustered dataframe from {df_path}...")
        clus = pd.read_csv(df_path, index_col=0)
    else:
        if name != "sense":
            with open(p, "rb") as f:
                data = pickle.load(f)
            df = pd.DataFrame.from_dict(data['results']).T
            df = df[df.index.isin(main_df.ID)]

            print(f"len of data {len(df)}")
            print(f"len of csv data {len(main_df)}")
        else:
            sense_data = load_all_features(sense_path)
            print(f"sense_data[0].shape: {len(sense_data[0].shape)}")
            df = pd.DataFrame(sense_data[0], index=sense_data[1])
            # df = df[df.index.isin(main_df.ID)]

        predictions = run_kmeans(
            df,
            k=K, 
            limit=100000, 
            save_img=f"{save_folder}/{name}_{split_name}.png"
        )

        clus = pd.DataFrame({
            "ID" : df.index,
            "cluster" : predictions,
        })

        # Save results
        clus.to_csv(df_path)
        print(f"Clustering results saved to {df_path}")

    hours = sample_kmeans_upto(csv_location, save_folder, fractions, clus, name, 42, ff[i])


# ### mfccs ###
# print("Working on MFCCs")

# with open(mfcc_path, "rb") as f:
#     mfcc = pickle.load(f)
# df = pd.DataFrame.from_dict(mfcc['results']).T
# df = df[df.index.isin(main_df.ID)]

# print(f"len of data mfcc data {len(df)}")
# print(f"len of data csv data {len(main_df)}")

# predictions = run_kmeans(
#     df,
#     k=K, 
#     limit=100000, 
#     save_img=f"{save_folder}/mfcc_{split_name}.png"
# )

# mfcc_clus = pd.DataFrame({
#     "ID" : df.index,
#     "cluster" : predictions,
# })

# hours = sample_kmeans_upto(csv_location, save_folder, fractions, mfcc_clus, "mfcc", 42, mfcc_ff)




# ### speaker ###
# print("Working on Speaker")

# with open(speaker_path, "rb") as f:
#     speaker = pickle.load(f)

# speaker_df = pd.DataFrame.from_dict(speaker['results']).T
# speaker_df = speaker_df[speaker_df.index.isin(main_df.ID)]

# print(f"speaker df len {len(speaker_df)}")

# speaker_predictions = run_kmeans(
#     speaker_df, 
#     k=K, 
#     limit=100000, 
#     save_img=f"{save_folder}/speaker_{split_name}.png",
#     feature_name="Speaker Embeddings"
# )

# speaker_clus = pd.DataFrame({
#     "ID" : df.index,
#     "cluster" : speaker_predictions,
# })

# hours = sample_kmeans_upto(csv_location, save_folder, fractions, speaker_clus, "speaker", 29, speaker_ff)



# ### SENSE ###
# print("Working on SENSE")

# sense_data = load_all_features(sense_path)
# print(f"sense_data[1].shape: {len(sense_data[1].shape)}")

# df = pd.DataFrame(sense_data[1], index=sense_data[2])
# df = df[df.index.isin(main_df.ID)]
# sense_predictions = run_kmeans(
#     df, 
#     k=K, 
#     limit=100000, 
#     save_img=f"{save_folder}/sense_{split_name}.png",
#     feature_name="SENSE Embeddings"
# )

# sense_clus = pd.DataFrame({
#     "ID" : df.index,
#     "cluster" : sense_predictions,
# })

# hours = sample_kmeans_upto(csv_location, save_folder, fractions, sense_clus, "sense", 29, sense_ff)

