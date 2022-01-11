#!/usr/bin/env python3

from pandas.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz as gv
import matplotlib.pyplot as plt
import pandas as pd

""" Scatter matrix """


def scatter(df, x_columns, y_column):
    scatter_matrix(df[x_columns + y_column],
                   figsize=(15, 15),
                   marker='o',
                   c=df['quality'].values,
                   s=30,
                   alpha=0.8,
                   )
    plt.show()


""" Linear Regression
No promising results.
"""


def linear(X_train, y_train, X_test, y_test):
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


def polynomial(X_train, y_train, X_test, y_test):
    for degree in [2, 3]:
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


def bayesian(X_train, y_train, X_test, y_test):
    model = BayesianRidge(n_iter=300, tol=0.001, fit_intercept=True)
    model.fit(X_train, y_train.values.ravel())

    print('Bayes\'sche lineare Regression:')
    print(f'R2 Trainingsdaten: {model.score(X_train, y_train)}')
    print(f'R2 Testdaten: {model.score(X_test, y_test)}')
    print('\n')


""" Support Vector Regression
No promising results.
"""


def svm(X_train, y_train, X_test, y_test):
    for kernel in ['rbf', 'linear', 'poly', 'sigmoid']:
        model = SVR(kernel=kernel, degree=3, max_iter=-1)
        model.fit(X_train, y_train.values.ravel())

        print(f'Support Vector Regression (kernel={kernel}):')
        print(f'Polynomial: R2 Trainingsdaten: {model.score(X_train, y_train)}')
        print(f'Polynomial: R2 Testdaten: {model.score(X_test, y_test)}')
        print('\n')


""" Descision Tree Regression
Strong Overfitting.
"""


def decision_tree(X_train, y_train, X_test, y_test, x_columns):
    for criterion in ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']:
        model = DecisionTreeRegressor(criterion=criterion,
                                      splitter='best',
                                      max_depth=None,
                                      min_samples_split=0.01,
                                      max_features=None,
                                      random_state=None,
                                      min_impurity_decrease=0.00,
                                      )
        model.fit(X_train, y_train)
        print(f'Entscheidungsbaum Regression (criterion={criterion})')
        print(f'R2 Trainingsdaten: {model.score(X_train, y_train)}')
        print(f'R2 Testdaten: {model.score(X_test, y_test)}')
        print('\n')

    dot = export_graphviz(model, out_file=None, filled=True, feature_names=x_columns)
    graph = gv.Source(dot)
    graph.view()


def main():
    df_red = pd.read_csv('winequality-red-filtered.csv', sep=';', header=0)
    df_red.name = "Dataframe mit Rotweindaten"
    df_white = pd.read_csv('winequality-white.csv', sep=';', header=0)
    df_white.name = "Dataframe mit Weißweindaten"

    df = df_red.append(df_white, ignore_index=True)
    df.name = "Dataframe mit Daten von Rot- und Weißwein kombiniert"

    x_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                 "free sulfur dioxide",
                 "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

    # x_columns_2 = ["volatile acidity", "residual sugar", "chlorides",
    #                "total sulfur dioxide", "pH", "sulphates", "alcohol"]

    y_column = ["quality"]

    random_state = 2

    for df in [df_red, df_white, df]:
        print("\nAnalysiere", df.name, "...", "\n")
        scatter(df, x_columns, y_column)

        """ Split Datset """
        X = df[x_columns]
        y = df[y_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3,
                                                            random_state=random_state,
                                                            )

        linear(X_train, y_train, X_test, y_test)
        polynomial(X_train, y_train, X_test, y_test)
        bayesian(X_train, y_train, X_test, y_test)
        svm(X_train, y_train, X_test, y_test)
        decision_tree(X_train, y_train, X_test, y_test, x_columns)

        if df != df:
            if input("Wollen Sie die Analyse fortsetzen? [y/n]") != "y":
                break


if __name__ == "__main__":
    main()
