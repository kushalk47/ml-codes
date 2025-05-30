import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Linear Regression on California Housing Dataset
def linear_regression():
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X[['AveRooms']], y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test, label="Actual")
    plt.plot(X_test, y_pred, color='red', label="Predicted")
    plt.xlabel("AveRooms"), plt.ylabel("House Price"), plt.legend(), plt.show()
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

# Polynomial Regression on Auto MPG Dataset
def polynomial_regression():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    cols = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin"]
    data = pd.read_csv(url, sep='\s+', names=cols, na_values="?").dropna()
    X, y = data[['displacement']], data['mpg']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    plt.scatter(X_test, y_test, label="Actual")
    plt.scatter(X_test, y_pred, color='red', label="Predicted")
    plt.xlabel("Displacement"), plt.ylabel("MPG"), plt.legend(), plt.show()
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

# Run both
linear_regression()
polynomial_regression()
