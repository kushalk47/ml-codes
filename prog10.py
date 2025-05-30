import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load and scale the dataset
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target

# Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

# Evaluate clustering
print("Confusion Matrix:\n", confusion_matrix(y, clusters))
print("\nClassification Report:\n", classification_report(y, clusters))

# Reduce dimensions for plotting
X_pca = PCA(n_components=2).fit_transform(X)
df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df['Cluster'] = clusters
df['True Label'] = y

# Plot: Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, edgecolor='black')
plt.title('K-Means Clustering')
plt.show()

# Plot: True Labels
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='True Label', palette='coolwarm', s=100, edgecolor='black')
plt.title('True Labels')
plt.show()

# Plot: Clusters + Centroids
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='PC1', y='PC2', hue='Cluster', palette='Set1', s=100, edgecolor='black')
centroids = PCA(n_components=2).fit_transform(kmeans.cluster_centers_)
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('K-Means Clustering with Centroids')
plt.legend()
plt.show()
