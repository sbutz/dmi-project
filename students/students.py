#!/usr/bin/env python

"""
TODO
- [x] scattermatrix
- [x] linears + poly modell mit unterschiedlichen spalten
- [x] r2 werte
- [ ] noten in gut/schlecht einteilen und klassfizierung
- [ ] segmentierung

Result:
Regression offers bad results.
Maybe better: Split grades (0-20) in 2 (or more) classes and predict class.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas.plotting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier

def main():
    df = pd.read_csv('./student-por.csv',
        sep=';',
        header=0,
    )
    df.drop(columns=['G1','G2'], inplace=True)

    categorial_columns = [
        'school',
        'sex',
        'address',
        'famsize',
        'Pstatus',
        'Mjob',
        'Fjob',
        'reason',
        'guardian',
        'schoolsup',
        'famsup',
        'paid',
        'activities',
        'nursery',
        'higher',
        'internet',
        'romantic',
    ]
    for col in categorial_columns:
        df[col], _ = df[col].factorize()

    print(df)

    # Scatter matrix
    #axes = pd.plotting.scatter_matrix(df,
    #    figsize=(15,15),
    #    marker='o',
    #    c=df['G3'].values,
    #    s = 30,
    #    alpha = 0.8,
    #)
    #for i in range(np.shape(axes)[0]):
    #    for j in range(np.shape(axes)[1]):
    #        if i < np.max(np.shape(axes)[0]) - 1:
    #            axes[i,j].set_visible(False)
    plt.show()

    """ Example Model """
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
        test_size=0.3,
        #random_state=4,
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f'R2 Trainingsdaten: {model.score(X_train, y_train)}')
    print(f'R2 Testdaten: {model.score(X_test, y_test)}')

    # Krasses Overfitting
    poly = PolynomialFeatures(degree=2)
    X_train_p = poly.fit_transform(X_train)
    X_test_p = poly.fit_transform(X_test)
    model_poly = LinearRegression()
    model_poly.fit(X_train_p, y_train)
    print(f'Polynomial: R2 Trainingsdaten: {model_poly.score(X_train_p, y_train)}')
    print(f'Polynomial: R2 Testdaten: {model_poly.score(X_test_p, y_test)}')

    model = DecisionTreeClassifier(min_samples_split=0.1)
    model.fit(X_train, y_train)
    print(f'R2 Trainingsdaten: {model.score(X_train, y_train)}')
    print(f'R2 Testdaten: {model.score(X_test, y_test)}')


if __name__ == '__main__':
    main()