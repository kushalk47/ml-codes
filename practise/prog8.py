from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier,plot_tree
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data=load_breast_cancer(as_frame=True)
X,y=data.data,data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=DecisionTreeClassifier(random_state=42).fit(X_train,y_train)

y=model.predict(X_test)

print("accuracy of the model",accuracy_score(y_test,y))

plt.figure(figsize=(10,5))
plot_tree(model)
plt.show()x