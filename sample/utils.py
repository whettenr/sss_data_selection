import os
import pickle
import pandas as pd
import numpy as np
import glob
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
# from sklearn.metrics import pairwise_distances

import matplotlib.pyplot as plt


seed = 10
np.random.seed(seed)

def run_kmeans(df, k=100, limit=20000, save_img="", feature_name="MFCCs"):
    mb_kmeans = MiniBatchKMeans(n_clusters=k,
                                random_state=42,
                                batch_size=64 * k,
                                init='k-means++', n_init=1, init_size=100000,
                                reassignment_ratio=5e-4,
                                verbose=0)
    
    mb_kmeans.fit(df)
    predictions = mb_kmeans.labels_
    cluster_centers = mb_kmeans.cluster_centers_
    
    ## PCA and visualize
    pca_2d = PCA(n_components=2, random_state=42)
    d_pca_2d = pca_2d.fit_transform(df[:limit])
    cluster_centers_pca_2d = pca_2d.transform(cluster_centers)
    
    plt.figure(figsize=(10, 7))
    # Scatter plot of PCA-reduced data points, colored by their assigned cluster
    scatter_2d = plt.scatter(d_pca_2d[:, 0], d_pca_2d[:, 1], c=predictions[:limit], cmap='viridis', s=50, alpha=0.7)
    # Plot the PCA-reduced cluster centers
    plt.scatter(cluster_centers_pca_2d[:, 0], cluster_centers_pca_2d[:, 1], marker='X', s=50, color='red',
                edgecolor='black', label='Cluster Centers')
    plt.title(f'K-Means Clustering on {feature_name} (PCA-Reduced to 2D, k={k})')
    plt.xlabel(f'Principal Component 1 (explains {pca_2d.explained_variance_ratio_[0]*100:.2f}%)')
    plt.ylabel(f'Principal Component 2 (explains {pca_2d.explained_variance_ratio_[1]*100:.2f}%)')
    plt.colorbar(scatter_2d, label='Cluster ID')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_img, dpi=300)
    plt.close()
    return predictions

def diverse_sample(cluster_df, frac, random_state=0):
    N = int(len(cluster_df) * frac)
    k = len(cluster_df['cluster'].unique())
    labels = cluster_df['cluster']
    clusters = [np.where(labels == i)[0] for i in range(k)]
    cluster_sizes = [len(c) for c in clusters]

    # Step 1: Allocate at least one per cluster (if cluster is non-empty)
    allocated = [min(1, size) for size in cluster_sizes]
    remaining = N - sum(allocated)

    # Step 2: Distribute remaining not propotionally (capped by cluster size)
    total_remaining_capacity = sum(max(size - alloc, 0) for alloc, size in zip(allocated, cluster_sizes))
    i = 1
    done = False
    while not done:
        for i in range(k):
            cap = max(cluster_sizes[i] - allocated[i], 0)
            if cap > 0:
                allocated[i] += 1
                remaining -= 1
            if remaining <= 0:
                done = True
                break

    # Step 3: Sample using furthest-point sampling within each cluster
    sampled_indices = []
    for i, count in enumerate(allocated):
        if count > 0:
            X_cluster = cluster_df[cluster_df.cluster == i]
            to_add = list(X_cluster.sample(n=count, random_state=random_state).ID)
            sampled_indices.extend(to_add)
            
    return sampled_indices

def sample_kmeans_upto(csv_loc, save_loc, fracts, cluster_df, name_beg="mfccs", random_state=0, fraction_factor=0):
    main_df = pd.read_csv(csv_loc)
    hours = []
    for i, f in enumerate(fracts):
        if type(fraction_factor) == list:
            ids = diverse_sample(cluster_df, f+fraction_factor[i], random_state)
        else:
            ids = diverse_sample(cluster_df, f+fraction_factor, random_state)
        # filter with main df and then save
        filtered_main_df = main_df[main_df['ID'].isin(ids)]
        hours.append(filtered_main_df.duration.sum()/ 3600)
        print(f"{len(filtered_main_df)} files and {round((filtered_main_df.duration.sum()/ 3600), 2)} hours")
        save_name = f"{save_loc}/{name_beg}_{str(f)}.csv"
        # print(f"saving to {save_name}")
        filtered_main_df.to_csv(save_name, index=False)
    return hours

def get_rand_csvs(csv_location, fractions, save_folder):
    main_df = pd.read_csv(csv_location)
    for fraction in fractions:
        sampled_df = main_df.sample(n=int(len(main_df) * fraction), random_state=0)
        print(f"{len(sampled_df)} files and {round((sampled_df.duration.sum()/ 3600), 2)} hours")
        save_name = f"{save_folder}/random_{str(fraction)}.csv"
        sampled_df.to_csv(save_name, index=False)

def load_all_features(save_dir, pattern="features_batch_*.pkl"):
    """Loads and combines all saved feature files into a single dict and array."""
    feature_list = []
    name_list = []

    file_paths = sorted(glob.glob(os.path.join(save_dir, pattern)))

    for path in file_paths:
        with open(path, "rb") as f:
            batch_data = pickle.load(f)
        name_list.extend(batch_data.keys())
        feature_list.extend(batch_data.values())

    # Convert to arrays for clustering etc.
    features = np.stack(feature_list)
    names = np.array(name_list)

    return features, names



