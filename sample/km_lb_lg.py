import os
import pickle
import pandas as pd
import numpy as np
import glob
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt

from utils import (
    run_kmeans, 
    diverse_sample, 
    sample_kmeans_upto, 
    get_rand_csvs, 
    load_all_features
)


seed = 10
np.random.seed(seed)



split_name = 'lg'
K=150

mfcc_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/lebench/lb_mel_xlg_ckpt.pkl"
speaker_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/lebench/lb_speaker_xlg_ckpt.pkl"
sense_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/lebench/lb_sense_xlg_features"
csv_location = f"/users/rwhetten/LeBenchmark/{split_name}/train.csv"
save_folder = f"/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/lebench_{split_name}"
fractions = [0.5]

mfcc_ff = 0.021
speaker_ff = -0.01
sense_ff = 0.04

# create save_folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Folder '{save_folder}' created.")
else:
    print(f"Folder '{save_folder}' already exists.")

main_df = pd.read_csv(csv_location)

### random ###
get_rand_csvs(csv_location, fractions, save_folder)
print("Random 1111443 files and 3846.89 hours")


### mfccs ###
print("Working on MFCCs")

# Path to save/load clustered dataframe
df_path = f"{save_folder}/mfcc_clusters_{split_name}_{K}.csv"

if os.path.exists(df_path):
    print(f"Loading existing clustered dataframe from {df_path}...")
    mfcc_clus = pd.read_csv(df_path, index_col=0)
else:
    print("No saved clustering found. Running KMeans...")

    with open(mfcc_path, "rb") as f:
        lb_mfcc = pickle.load(f)
    df = pd.DataFrame.from_dict(lb_mfcc['results']).T
    df = df[df.index.isin(main_df.ID)]

    predictions = run_kmeans(
        df,
        k=K, 
        limit=100000, 
        save_img=f"{save_folder}/mfcc_{split_name}.png"
    )

    mfcc_clus = pd.DataFrame({
        "ID" : df.index,
        "cluster" : predictions,
    })

    # Save results
    mfcc_clus.to_csv(df_path)
    print(f"Clustering results saved to {df_path}")


hours = sample_kmeans_upto(csv_location, save_folder, fractions, mfcc_clus, "mfcc", 42, mfcc_ff)


### speaker ###
print("Working on Speaker")

df_path = f"{save_folder}/speaker_clusters_{split_name}_{K}.csv"

if os.path.exists(df_path):
    print(f"Loading existing clustered dataframe from {df_path}...")
    speaker_clus = pd.read_csv(df_path, index_col=0)
else:
    print("No saved clustering found. Running KMeans...")
    
    with open(speaker_path, "rb") as f:
        lb_speaker = pickle.load(f)

    df = pd.DataFrame.from_dict(lb_speaker['results']).T
    df = df[df.index.isin(main_df.ID)]
    speaker_predictions = run_kmeans(
        df, 
        k=K, 
        limit=100000, 
        save_img=f"{save_folder}/speaker_{split_name}.png",
        feature_name="Speaker Embeddings"
    )

    speaker_clus = pd.DataFrame({
        "ID": df.index,
        "cluster": speaker_predictions,
    })

    # Save results
    speaker_clus.to_csv(df_path)
    print(f"Clustering results saved to {df_path}")

hours = sample_kmeans_upto(csv_location, save_folder, fractions, speaker_clus, "speaker", 29, speaker_ff)



### SENSE ###
print("Working on SENSE")


df_path = f"{save_folder}/SENSE_clusters_{split_name}_{K}.csv"

if os.path.exists(df_path):
    print(f"Loading existing clustered dataframe from {df_path}...")
    sense_clus = pd.read_csv(df_path, index_col=0)
else:
    print("No saved clustering found. Running KMeans...")
    sense_data = load_all_features(sense_path)
    print(f"sense_data[0].shape: {sense_data[0].shape}")

    df = pd.DataFrame(sense_data[0], index=sense_data[1])
    df = df[df.index.isin(main_df.ID)]
    sense_predictions = run_kmeans(
        df, 
        k=K, 
        limit=100000, 
        save_img=f"{save_folder}/sense_{split_name}.png",
        feature_name="SENSE Embeddings"
    )

    sense_clus = pd.DataFrame({
        "ID" : df.index,
        "cluster" : sense_predictions,
    })

    # Save results
    sense_clus.to_csv(df_path)
    print(f"Clustering results saved to {df_path}")

hours = sample_kmeans_upto(csv_location, save_folder, fractions, sense_clus, "sense", 29, sense_ff)

