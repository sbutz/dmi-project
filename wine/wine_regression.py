#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas.plotting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('./wine/winequality-red.csv', sep=';', header=0)

x_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

x_columns_2 = ["volatile acidity", "residual sugar", "chlorides",
               "total sulfur dioxide", "pH", "sulphates", "alcohol"]

y_column = ["quality"]

print(df)

# Scatter matrix

scatter = pd.plotting.scatter_matrix(df[x_columns + y_column],
                                     figsize=(15, 15),
                                     marker='o',
                                     c=df['quality'].values,
                                     s=30,
                                     alpha=0.8,
                                     )

plt.show()

""" Example Model """
X = df[x_columns]
y = df[y_column]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    # random_state=2,
                                                    )

model = LinearRegression()
# model = BayesianRidge(n_iter=300, tol=0.001, fit_intercept=True)
model.fit(X_train, y_train)

print(f'R2 Trainingsdaten: {model.score(X_train, y_train)}')
print(f'R2 Testdaten: {model.score(X_test, y_test)}')

# Polynomial
# Grad 3: overfitting

poly = PolynomialFeatures(degree=2)
X_train_p = poly.fit_transform(X_train)
X_test_p = poly.fit_transform(X_test)
model_poly = LinearRegression()
model_poly.fit(X_train_p, y_train)
print(f'Polynomial: R2 Trainingsdaten: {model_poly.score(X_train_p, y_train)}')
print(f'Polynomial: R2 Testdaten: {model_poly.score(X_test_p, y_test)}')

# Decision Tree

# model = DecisionTreeClassifier(min_samples_split=0.1)
model = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                               min_samples_split=3, max_features=None, random_state=None,
                               min_impurity_decrease=0.0)
# model = SVR(kernel='rbf', degree=2, max_iter=-1)
model.fit(X_train, y_train)
print(f'R2 Trainingsdaten: {model.score(X_train, y_train)}')
print(f'R2 Testdaten: {model.score(X_test, y_test)}')
