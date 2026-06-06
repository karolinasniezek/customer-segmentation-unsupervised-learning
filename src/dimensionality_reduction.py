import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from kneed import KneeLocator

from scipy.cluster.hierarchy import linkage, fcluster

import umap
import joblib

import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# =========================
# LOAD DATA
# =========================

df = pd.read_csv("../data/marketing_campaign.csv")

# =========================
# DATA PREPROCESSING
# =========================

label_encoder = LabelEncoder()

df_categoricals = df.select_dtypes(include=["object"])

df[df_categoricals.columns] = df_categoricals.apply(
    label_encoder.fit_transform
)

df_numerical = df.select_dtypes(include=[np.number])

scaler = StandardScaler()

df[df_numerical.columns] = scaler.fit_transform(df_numerical)

# =========================
# KMEANS + ELBOW METHOD
# =========================

os.environ["LOKY_MAX_CPU_COUNT"] = "0"

wcss = []

for i in range(1, 11):

    model = KMeans(
        n_clusters=i,
        init="k-means++",
        max_iter=300,
        n_init=10,
        random_state=42
    )

    model.fit(df)

    wcss.append(model.inertia_)

    print(f"k={i}: WCSS={model.inertia_:.2f}")

# =========================
# ELBOW CHART
# =========================

plt.figure(figsize=(8, 6))

plt.plot(range(1, 11), wcss, marker="o")

plt.title("K-means clustering")
plt.xlabel("Clusters")
plt.ylabel("WCSS")

plt.grid(True)

plt.savefig("../figures/k-means-clustering.png")

plt.close()

# =========================
# OPTIMAL NUMBER OF CLUSTERS
# =========================

kl = KneeLocator(
    range(1, 11),
    wcss,
    curve="convex",
    direction="decreasing"
)

optimal_clusters = kl.elbow

print(f"\nOptimal clusters: {optimal_clusters}")

# =========================
# PCA
# =========================

pca = PCA(n_components=2)

df_pca = pd.DataFrame(
    pca.fit_transform(df),
    columns=["PC1", "PC2"]
)

# =========================
# KMEANS ON PCA
# =========================

kmeans = KMeans(
    n_clusters=optimal_clusters,
    random_state=42
)

clusters = kmeans.fit_predict(df_pca)

df_pca["KMeans_Cluster"] = clusters

kmeans_score = silhouette_score(
    df_pca[["PC1", "PC2"]],
    clusters
)

print(f"KMeans Silhouette Score: {kmeans_score:.4f}")

# =========================
# PCA CLUSTER PLOT
# =========================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="KMeans_Cluster",
    palette="viridis",
    s=100,
    edgecolor="black"
)

plt.title("K-Means Clustering grup zakupowych po PCA")

plt.savefig("../figures/k-means-pca.png")

plt.close()

# =========================
# UMAP
# =========================

umap_model = umap.UMAP(
    n_components=2,
    random_state=42
)

df_umap = pd.DataFrame(
    umap_model.fit_transform(df),
    columns=["UMAP1", "UMAP2"]
)

print("\n######## DIMENSION REDUCTION UMAP ########")
print(df_umap.head())

# =========================
# SAVE FILES
# =========================

joblib.dump(
    umap_model,
    "umap_model.pkl"
)

df_umap.to_csv(
    "df_umap.csv",
    index=False
)

print("\nFiles saved:")
print("- umap_model.pkl")
print("- df_umap.csv")

# =========================
# HIERARCHICAL CLUSTERING
# =========================

link = linkage(
    df_umap,
    method="ward"
)

df_umap["Hierarchical_Cluster"] = fcluster(
    link,
    t=optimal_clusters,
    criterion="maxclust"
)

print("\n######## HIERARCHICAL CLUSTERING ########")
print(df_umap["Hierarchical_Cluster"])

hierarchical_score = silhouette_score(
    df_umap[["UMAP1", "UMAP2"]],
    df_umap["Hierarchical_Cluster"]
)

print(
    f"Hierarchical Silhouette Score: {hierarchical_score:.4f}"
)

# =========================
# HIERARCHICAL CLUSTER PLOT
# =========================

plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df_umap,
    x="UMAP1",
    y="UMAP2",
    hue="Hierarchical_Cluster",
    palette="viridis",
    s=100,
    edgecolor="black"
)

plt.xlabel("UMAP1")
plt.ylabel("UMAP2")

plt.title(
    "Hierarchical Clustering grup zakupowych po PCA"
)

plt.savefig(
    "../figures/hierarchical-clustering-umap.png"
)

plt.show()

df_umap["KMeans_Cluster"] = kmeans.fit_predict(df_umap)
kmeans_score2 = silhouette_score(df_umap, kmeans.labels_)
results = {
    "Model": ["K-Means (PCA)", "K-Means (UMAP)", "Hierarchical (UMAP)"],
    "Score": [kmeans_score,
              kmeans_score2,
              hierarchical_score]
}

print(results)
print("K-Means")
results_df = pd.DataFrame(results)
print(results_df)

# =========================
# CUSTOMER SEGMENT VISUALIZATION
# =========================

df_original = pd.read_csv("../data/marketing_campaign.csv")

df_original["KMeans_Cluster"] = df_umap["KMeans_Cluster"]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

sns.boxplot(
    data=df_original,
    x="KMeans_Cluster",
    y="WydatkiWino",
    ax=axes[0]
)

axes[0].set_title("Wine Spending by Cluster")

sns.boxplot(
    data=df_original,
    x="KMeans_Cluster",
    y="Dochod",
    ax=axes[1]
)

axes[1].set_title("Income by Cluster")

sns.boxplot(
    data=df_original,
    x="KMeans_Cluster",
    y="IloscZakupowInternet",
    ax=axes[2]
)

axes[2].set_title("Online Purchases by Cluster")

plt.tight_layout()

plt.savefig(
    "../figures/customer-segments-boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
