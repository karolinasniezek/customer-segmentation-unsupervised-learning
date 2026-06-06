import pandas as pd
import numpy as np
from numba.core.types import optional
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.decomposition import PCA
import seaborn as sns


np.random.seed(42)

df = pd.read_csv('../data/marketing_campaign.csv')
print(df.columns)

missing_values = df.isnull().sum()
print(missing_values)

label_encoder = LabelEncoder()
df_categoricals = df.select_dtypes(include=["object"])
df_encoded_categoricals = df_categoricals.apply(label_encoder.fit_transform)
df[df_encoded_categoricals.columns] = df_encoded_categoricals
print(df[df_encoded_categoricals.columns])

df_numerical = df.select_dtypes(include=[np.number])
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), columns=df_numerical.columns)

df[df_numerical.columns] = df_scaled

os.environ['LOKY_MAX_CPU_COUNT'] = '0'

wcss = []  # Lista do przechowywania wartości WCSS (Within-Cluster Sum of Squares)
for i in range(1, 11):  #
    kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
    print(f"k={i}: WCSS={kmeans.inertia_:.2f}")

plt.figure(figsize=(8,6))
plt.plot(range(1,11), wcss, marker="o")
plt.title("K-means clustering")
plt.xticks(np.arange(1,11,1))
plt.xlabel("Clusters ")
plt.ylabel("WCSS")
plt.grid()
plt.savefig("../figures/k-means-clustering.png")

kl = KneeLocator(range(1,11), wcss, curve="convex", direction="decreasing")
optimal_clusters = kl.elbow
print(optimal_clusters)

pca = PCA(n_components=2)
df_pca = pd.DataFrame(pca.fit_transform(df), columns=["PC1", "PC2"])
print(df_pca)

plt.figure(figsize=(7,6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.5, c="blue")
plt.title("PCA: reduction to two dimensions")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("../figures/pca-chart.png")

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_pca["KMeans_Cluster"] = kmeans.fit_predict(df_pca)
kmeans_sc = silhouette_score(df_pca, kmeans.labels_)
print(kmeans_sc)


sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="KMeans_Cluster", palette="viridis", s=100, edgecolor="black")
plt.title("K-Means Clustering group after PCA")
plt.savefig("../figures/k-means-clustering-after-pca.png")


