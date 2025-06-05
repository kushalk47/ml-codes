import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def linear():
    X,y=fetch_california_housing(return_X_y=True,as_frame=True)
    X_test,X_train,y_test,y_train=train_test_split(X[['AveRooms']],y,random_state=42,test_size=0.2)
    model=LinearRegression().fit(X_train,y_train)
    y1=model.predict(X_test)
    plt.scatter(X_test,y_test,label="Actual")
    plt.plot(X_test,y1,label="predicted")
    plt.show()
    print("mean_squared_error",mean_squared_error(y_test,y1))
    print("R2 value",r2_score(y_test,y1))

def poly():
   url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
   cols=["mpg","displacement","horsepower", "weight", "acceleration", "model_year", "origin","cylinders"]
   data=pd.read_csv(url,sep="\s+",names=cols,na_values="?").dropna()
   X,y=data[["displacement"]],data["mpg"]
   X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
   model=make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_train,y_train)
   y=model.predict(X_test)
   plt.scatter(X_test,y_test,label="actual")
   plt.scatter(X_test,y,label="predicted",color="red")
   plt.title("mutlilinearn")
   plt.show()
   print("mean_squared_error",mean_squared_error(y_test,y))
   print("R2 value",r2_score(y_test,y))



poly()
linear()
