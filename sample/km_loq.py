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

split_name = 'medium' # medium | large
mfcc_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/loq/mel_large_ckpt.pkl"
speaker_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/loq/speaker_large_ckpt.pkl"
sense_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/loq/sense_large_features"
csv_location = f"/local_disk/apollon/rwhetten/loquacious_{split_name}_train.csv"
save_folder = f"/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/loq_{split_name}"
fractions = [0.5]
K=150

# create save_folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"Folder '{save_folder}' created.")
else:
    print(f"Folder '{save_folder}' already exists.")

main_df = pd.read_csv(csv_location)

# ### random ###
get_rand_csvs(csv_location, fractions)



### mfccs ###
with open(mfcc_path, "rb") as f:
    mfcc = pickle.load(f)
df = pd.DataFrame.from_dict(mfcc['results']).T
df = df[df.index.isin(main_df.ID)]

print(f"len of data mfcc data {len(df)}")
print(f"len of data csv data {len(main_df)}")

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

hours = sample_kmeans_upto(csv_location, save_folder, fractions, mfcc_clus, "mfcc", 42)




### speaker ###
with open(speaker_path, "rb") as f:
    speaker = pickle.load(f)

speaker_df = pd.DataFrame.from_dict(speaker['results']).T
speaker_df = speaker_df[speaker_df.index.isin(main_df.ID)]

print(f"speaker df len {len(speaker_df)}")

speaker_predictions = run_kmeans(
    speaker_df, 
    k=K, 
    limit=100000, 
    save_img=f"{save_folder}/speaker_{split_name}.png",
    feature_name="Speaker Embeddings"
)

speaker_clus = pd.DataFrame({
    "ID" : df.index,
    "cluster" : speaker_predictions,
})

hours = sample_kmeans_upto(
    csv_location, 
    save_folder, 
    fractions, 
    speaker_clus, 
    "speaker", 
    29,
    0.02, 
)



### SENSE ###
sense_data = load_all_features(sense_path)
print(f"sense_data[1].shape: {len(sense_data[1].shape)}")

df = pd.DataFrame(sense_data[1], index=sense_data[2])
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

hours = sample_kmeans_upto(csv_location, save_folder, fractions, sense_clus, "sense", 29)

