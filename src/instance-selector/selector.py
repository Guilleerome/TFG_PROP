import argparse
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import math
import itertools

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

import seaborn as sns

from src.io_instances import instances_reader as ir
from src.io_instances.instances_reader import _resolve_instances_root
from features import build_graph_from_instance, extract_graph_features

percentage_preliminary_test = 0.25
output_folder = "../instances/selected_instances"

# PCA configuration
explained_pca_ratio = 0.90

# K-Means configuration
init_kmeans = 'k-means++'
random_state = 42
max_iter = 1000
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'brown', 'olive', 'gray', 'olive', 'crimson', 'teal']
max_n_cluster = 15

def main():
    plants = ir.read_all_instances()
    graphs = []
    for plant in plants:
        graphs.append(build_graph_from_instance(plant))

    feature_list = []
    for G in graphs:
        features = extract_graph_features(G)
        feature_list.append(features)

    df = pd.DataFrame(feature_list)
    print("\n=== FEATURES DATAFRAME ===")
    print(df.head(n=10))

    instances_ids = df['id'].tolist()

    df_numeric = df.drop(['id', 'is_regular', 'is_connected', 'components'], axis=1)


    print(df_numeric.describe())

    correlation = df_numeric.corr()
    plt.figure(figsize=(10, 10))
    sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
    plt.title('Correlation between features')
    plt.tight_layout()
    plt.show()

    data_scaled = pd.DataFrame(preprocessing.scale(df_numeric), columns=df_numeric.columns)
    data_scaled.head()
    print("\n=== SCALED FEATURES DATAFRAME ===")
    print(data_scaled.head(n=10))

    pca = PCA().fit(data_scaled)
    line_data = np.cumsum(pca.explained_variance_ratio_)
    line_data = np.insert(line_data, 0, 0)
    plt.bar(np.arange(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color='g')
    plt.plot(np.arange(0, len(line_data)), line_data, marker='D')
    plt.xlim(0, len(pca.explained_variance_ratio_))
    plt.axhline(y=explained_pca_ratio, color='black', linestyle='--')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    sklearn_pca = PCA(n_components=explained_pca_ratio)
    data_pca = sklearn_pca.fit_transform(data_scaled)

    model = KMeans(init=init_kmeans, random_state=random_state, max_iter=max_iter)

    visualizer = KElbowVisualizer(model, k=(2, max_n_cluster), timings=False)
    visualizer.fit(data_pca)  # Fit the data to the visualizer
    visualizer.finalize()
    visualizer.ax.set_title("")
    visualizer.ax.set_ylabel("Distortion Score")
    visualizer.ax.set_xlabel("Number of clusters")
    plt.show()

    n_cluster = visualizer.elbow_value_
    kmeans = KMeans(n_clusters=n_cluster, init=init_kmeans, random_state=random_state, max_iter=max_iter).fit(data_pca)
    labels = kmeans.predict(data_pca)

    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(data_pca)  # Fit the data to the visualizer
    visualizer.finalize()
    visualizer.ax.set_title("")
    visualizer.ax.set_ylabel("Cluster ID")
    visualizer.ax.set_xlabel("Silhouette Coefficients")
    plt.show()

    color_dict = dict()
    for index, value in enumerate(colors):
        color_dict[index] = value

    result = {"Cluster Id": labels}
    for i in range(data_pca.shape[1]):
        result["PCA  " + str(i)] = data_pca[:, i]

    testdf = pd.DataFrame(result)
    sns.pairplot(testdf, hue="Cluster Id", palette=color_dict)
    plt.show()

    i = 1
    plt.figure(figsize=(50, 100))
    for g in graphs:
        plt.subplot(math.ceil(len(graphs) / 4), 4, i)
        plt.axis("off")
        plt.title(g.graph)
        nx.draw(g, node_color=colors[labels[i - 1]], with_labels=True)
        i = i + 1
    plt.show()

    distances = pairwise_distances(kmeans.cluster_centers_, data_pca)
    distancesToCentroid = []
    i = 0
    for c in labels:
        distancesToCentroid.append(distances[c][i])
        i += 1

    clusters = {}
    clusters["ClusterId"] = labels
    clusters["Instance"] = [g.graph['name'] for g in graphs]
    clusters['Distances'] = distancesToCentroid
    cluster_df = pd.DataFrame(clusters)

    print(cluster_df.head())

    sorted_df = []

    for k in range(n_cluster):
        dfk = cluster_df.loc[cluster_df['ClusterId'] == k]
        sorted_df.append(dfk.sort_values(by='Distances', ascending=True).to_numpy())

    # Sort by cluster size
    sorted_df.sort(key=lambda x: -len(x))

    # Después de crear los clusters
    print(f"\n=== CLUSTERING RESULTS ===")
    print(f"Number of clusters detected: {n_cluster}")
    print(f"Cluster sizes:")
    for k in range(n_cluster):
        cluster_size = np.sum(labels == k)
        print(f"  Cluster {k}: {cluster_size} instances")

    # Después de sorted_df
    print("\n=== INSTANCES PER CLUSTER (sorted by distance to centroid) ===")
    for k, cluster_array in enumerate(sorted_df):
        print(f"\nCluster {k} ({len(cluster_array)} instances):")
        print(f"  Top 5: {[row[1] for row in cluster_array[:5]]}")

    instance_remaining = math.ceil(percentage_preliminary_test * len(labels))
    preliminary_instances = []
    aux = [i for i in sorted_df.copy()]

    stopAt = instance_remaining
    takeFromCluster = 0
    while takeFromCluster < stopAt:
        cluster = aux[takeFromCluster % len(aux)]
        if len(cluster) == 0:
            stopAt += 1
        else:
            instance = cluster[0][1]
            preliminary_instances.append(instance)
            aux[takeFromCluster % len(aux)] = np.delete(cluster, 0, 0)

        takeFromCluster += 1

    n_preliminar_instances = math.ceil(percentage_preliminary_test * len(labels))
    print(f"""
    SUMMARY:
     - Total instances {len(labels)}
     - Preliminary %: {percentage_preliminary_test}
     - Number of preliminary instances: {n_preliminar_instances}
     - Instance selected: {preliminary_instances}
    """)

    instances_folder = _resolve_instances_root()  # ← Esto devuelve la ruta absoluta correcta
    output_folder = instances_folder / "selected_instances"

    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=True)

    copied_count = 0
    for preliminar_instance in preliminary_instances:
        found = False
        preliminar_instance = preliminar_instance + ".txt"
        for root, dirs, files in os.walk(str(instances_folder)):  # Convertir Path a str
            if preliminar_instance in files:
                src = os.path.join(root, preliminar_instance)
                dst = os.path.join(output_folder, preliminar_instance)
                shutil.copyfile(src, dst)
                copied_count += 1
                found = True
                print(f"✅ Copied: {preliminar_instance}")
                break

        if not found:
            print(f"⚠️ NOT FOUND: {preliminar_instance}")

    print(f'\n✅ {copied_count}/{len(preliminary_instances)} instances copied to {output_folder}')


print('Preliminary instance have been copied to ', output_folder)
if __name__ == "__main__":
    main()
