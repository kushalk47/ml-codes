from sklearn.datasets import fetch_california_housing
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data=fetch_california_housing(as_frame=True)
df=data.frame

features=df.select_dtypes(include=[np.number]).columns

n_cols=3
n_rows=3
n_features=len(features)
#histogram
plt.figure(figsize=(15,10))
for i,fe in enumerate(features):
    plt.subplot(3,3,i+1)
    sns.histplot(df[fe])
    plt.title(fe)

plt.tight_layout()
plt.show()

#boxplot
plt.figure(figsize=(15,10))
for i , f in enumerate(features):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=df[f])
    plt.title(f)

plt.tight_layout()
plt.show()

#outlier
for fea in features:
    x1=df[fea].quantile(0.25)
    x3=df[fea].quantile(0.75)
    iqr=x3-x1
    lb=x1-1.5*iqr
    ub=x3+1.5*iqr
    outliers=df[(df[fea]<lb)|(df[fea]>ub)]
    print(len(outliers),fea)

print(df.describe())

