import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm  # Use the notebook-friendly version

# !pip install plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

import torch
from speechbrain.dataio.dataio import read_audio, write_audio
from loquacious_set_prepare_filter import load_datasets

csv_location = "/local_disk/apollon/rwhetten/loquacious_small_train.csv"
save_location = "/local_disk/apollon/rwhetten/sss_data_selection/sample/csvs/"

mel_lg_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/mel_large_ckpt.pkl"
speaker_lg_path = "/local_disk/apollon/rwhetten/sss_data_selection/save/speaker_large_ckpt.pkl"

with open(mel_lg_path, "rb") as f:
    mel_lg = pickle.load(f)

FRACTIONS = [0.2, 0.4, 0.6, 0.8]


df = pd.DataFrame.from_dict(mel_lg['results']).T
df.head()


# get features
def get_features(fp):
    with open(fp, "rb") as f:
        features = pickle.load(f)

    features = pd.DataFrame.from_dict(features['results']).T
    return features


# k-means
def get_clusters(df, k=100)
    mb_kmeans = MiniBatchKMeans(n_clusters=k,
                                random_state=42,
                                batch_size=64 * k,
                                init='k-means++', n_init=1, init_size=100000,
                                reassignment_ratio=5e-4,
                                verbose=0)

    mb_kmeans.fit(df)
    predictions = mb_kmeans.labels_
    cluster_centers = mb_kmeans.cluster_centers_
    return predictions

# sample and save

# main
if __name__ == "__main__":
    print(sys.argv[1:])
    features_path, csv_location, save_location, save_name, k  = sys.argv[1:]

    # get features
    features = get_features(features_path)

    # get clusters
    predictions = get_clusters(features, k)

    # save clusters
    clusters = pd.DataFrame({
        "ID" : features.index,
        "cluster" : predictions,
    })
    clusters.to_csv(f"{save_location}/clusters.csv", index=False)

    # sample and save csvs