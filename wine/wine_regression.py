#!/usr/bin/env python3

from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz as gv
import matplotlib.pyplot as plt
import pandas as pd

""" TODO:
- df: red vs white wine
- train_test_split: different random_states
- train_test_split: limit features (x_columns_2)
- tree: try different params 
- use neural networks
- plot: r2 depending on params?
- more evaluation numbers beside r2?
- use scaler?
"""

df = pd.read_csv('./wine/winequality-red.csv', sep=';', header=0)

x_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

x_columns_2 = ["volatile acidity", "residual sugar", "chlorides",
               "total sulfur dioxide", "pH", "sulphates", "alcohol"]

y_column = ["quality"]

print(df)

""" Scatter matrix """
scatter_matrix(df[x_columns + y_column],
               figsize=(15, 15),
               marker='o',
               c=df['quality'].values,
               s=30,
               alpha=0.8,
)
plt.show()


""" Split Datset """
X = df[x_columns]
y = df[y_column]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=2,
                                                   )


""" Linear Regression
No promising results.
"""
model = LinearRegression()
model.fit(X_train, y_train)

print('\nLineare Regression:')
print(f'R2 Trainingsdaten: {model.score(X_train, y_train)}')
print(f'R2 Testdaten: {model.score(X_test, y_test)}')
print('\n')


""" Polynomial Regression
No promising results.
Overfitting starting with degree 3.
"""
for degree in [2,3]:
    poly = PolynomialFeatures(degree=degree)
    X_train_p = poly.fit_transform(X_train)
    X_test_p = poly.fit_transform(X_test)
    model = LinearRegression()
    model.fit(X_train_p, y_train)

    print(f'Polynomielle Regression (grad={degree}):')
    print(f'Polynomial: R2 Trainingsdaten: {model.score(X_train_p, y_train)}')
    print(f'Polynomial: R2 Testdaten: {model.score(X_test_p, y_test)}')
    print('\n')


""" Bayesian Ridge Regression
No promising results.
"""
model = BayesianRidge(n_iter=300, tol=0.001, fit_intercept=True)
model.fit(X_train, y_train.values.ravel())

print('Bayes\'sche lineare Regression:')
print(f'R2 Trainingsdaten: {model.score(X_train, y_train)}')
print(f'R2 Testdaten: {model.score(X_test, y_test)}')
print('\n')


""" Support Vector Regression
No promising results.
"""
for kernel in ['rbf', 'linear', 'poly']:
    model = SVR(kernel=kernel, degree=3, max_iter=-1)
    model.fit(X_train, y_train.values.ravel())

    print(f'Support Vector Regression (kernel={kernel}):')
    print(f'Polynomial: R2 Trainingsdaten: {model.score(X_train, y_train)}')
    print(f'Polynomial: R2 Testdaten: {model.score(X_test, y_test)}')
    print('\n')


""" Descision Tree Regression
Best regression results R2=0.55
"""
for criterion in ['entropy', 'gini']:
    model = DecisionTreeClassifier(criterion=criterion,
                                   splitter='best',
                                   max_depth=None,
                                   min_samples_split=0.05,
                                   max_features=None,
                                   random_state=None,
                                   min_impurity_decrease=0.01,
                                  )
    model.fit(X_train, y_train)
    print(f'Entscheidungsbaum Regression (criterion={criterion})')
    print(f'R2 Trainingsdaten: {model.score(X_train, y_train)}')
    print(f'R2 Testdaten: {model.score(X_test, y_test)}')
    print('\n')

# Only plot last/best tree
dot = export_graphviz(model, out_file=None,filled=True, feature_names=X.columns)
graph = gv.Source(dot)
graph.view()


""" KNN """
# TODO