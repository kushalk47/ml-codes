from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=fetch_california_housing(as_frame=True)

df=data.frame
matrix=df.corr()

plt.figure(figsize=(15,10))
sns.heatmap(matrix,cmap='coolwarm',annot=True)
plt.title("correlaion matrix")
plt.show()


sns.pairplot(df,diag_kind='kde')
plt.title("pair plots")
plt.show()