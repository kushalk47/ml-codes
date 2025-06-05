from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=load_iris()
data,labels=data.data,data.target

#pca
pca=PCA(n_components=2)
reduc=pca.fit_transform(data)
a=["red","blue","orange"]
b=[0,1,2]

plt.figure(figsize=(10,5))
for i , label in enumerate (b):
    plt.scatter(
         reduc[labels==label,0],
         reduc[labels==label,1],
         color=a[i]
         
    )

plt.title("dimensionality reduction ")
plt.xlabel("pca component 1")
plt.ylabel("pca component 2")
plt.show()

