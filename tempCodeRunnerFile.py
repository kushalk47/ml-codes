import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris()
data = iris.data
labels = iris.target
feature_names = iris.feature_names
label_names = iris.target_names

iris_df = pd.DataFrame(data, columns=feature_names)

pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data)

reduced_df = pd.DataFrame(data_reduced, columns=['Principal Component 1', 'Principal Component 2'])
reduced_df['Label'] = labels

plt.figure(figsize=(8, 6))

colors = ['red', 'green', 'blue']

for i, label in enumerate(np.unique(labels)):
    plt.scatter(
        reduced_df[reduced_df['Label'] == label]['Principal Component 1'],
        reduced_df[reduced_df['Label'] == label]['Principal Component 2'],
        label=label_names[label],
        color=colors[i],
        s=50,
        alpha=0.8,
        edgecolors='w'
    )

plt.title('PCA of Iris Dataset: 4 Features Reduced to 2 Principal Components', fontsize=14)
plt.xlabel('Principal Component 1', fontsize=12)
plt.ylabel('Principal Component 2', fontsize=12)
plt.legend(title='Iris Species', fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()