import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

# Load data
iris = load_iris()
data, labels = iris.data, iris.target

# Apply PCA
pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

# Plot PCA results
plt.figure(figsize=(8, 6))
colors = ['red', 'green', 'blue']

for i, label in enumerate(np.unique(labels)):
    plt.scatter(
        data_reduced[labels == label, 0],
        data_reduced[labels == label, 1],
        label=iris.target_names[label],
        color=colors[i],
        s=50,
        alpha=0.8,
        edgecolors='w'
    )

plt.title('PCA of Iris Dataset', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Iris Species', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
