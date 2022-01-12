#!/usr/bin/env python3

from re import L
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np


"""" Naive Bayes (Gauss)
Bad accuracy"""


def naive_bayes(df, x_columns, y_column, random_state):
    X = df[x_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=random_state,
                                                        )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, np.ravel(y_train))

    print("\nPrediction of Quality with Naive Bayes(Gauss)")
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(model.score(X_train_scaled, y_train.values.ravel())))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(model.score(X_test_scaled, y_test.values.ravel())))


"""" KNeighborsClassifier 
Using MinMaxScaler.
Without splitting into categories.
"""


def knn(df, x_columns, y_column, random_state):
    X = df[x_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=random_state,
                                                        )

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train.values.ravel())
    print("\nK-Nearest Neighbors:")
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knn.score(X_train_scaled, y_train.values.ravel())))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(knn.score(X_test_scaled, y_test.values.ravel())))

    # print(knn.predict(X_test_scaled))
    # print(y_test)


""" KNN 
Splitting wines into different quality ranges.
Good results.
Most wines are in the range of ~4-7
Splitting into 3 categories but splitting into 2 categories (e.g. 1-7 | 8-10)
brings very good results as well.
"""


def knn_categories(df, x_columns, y_column, random_state, state):
    df = df.copy(deep=True)

    if state == 2:
        df.loc[df.quality < 7, "quality"] = 0
        df.loc[df.quality >= 7, "quality"] = 1
        print("\nKNN with 2 categories(0-6/7-10):")
    elif state == 3:
        df.loc[df.quality < 6, "quality"] = 0
        df.loc[df.quality >= 6, "quality"] = 1
        df.loc[df.quality >= 8, "quality"] = 2
        print("\nKNN with 3 categories(0-5/6-7/8-10):")
    else:
        print("Invalid state")

    X = df[x_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=random_state,
                                                        )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knc = KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='minkowski', p=2)

    knc.fit(X_train, y_train.values.ravel())
    print('Accuracy of K-NN classifier on training set: {:.2f}'
          .format(knc.score(X_train, y_train.values.ravel())))
    print('Accuracy of K-NN classifier on test set: {:.2f}'
          .format(knc.score(X_test, y_test.values.ravel())))

    # print(knc.predict(X_test))
    # print(y_test)

def tree_categories(df, x_columns, y_column, random_state, state):
    df = df.copy(deep=True)

    if state == 2:
        df.loc[df.quality < 7, "quality"] = 0
        df.loc[df.quality >= 7, "quality"] = 1
        print("\nKDescion Tree with 2 categories(0-6/7-10):")
    elif state == 3:
        df.loc[df.quality < 6, "quality"] = 0
        df.loc[df.quality >= 6, "quality"] = 1
        df.loc[df.quality >= 8, "quality"] = 2
        print("\nDescion Tree with 3 categories(0-5/6-7/8-10):")
    else:
        print("Invalid state")

    X = df[x_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=random_state,
                                                        )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knc = DecisionTreeClassifier(
        criterion='gini',
        splitter='best',
        min_samples_split=0.05,
        random_state=random_state,
    )

    knc.fit(X_train, y_train.values.ravel())
    print('Accuracy of decision tree classifier on training set: {:.2f}'
          .format(knc.score(X_train, y_train.values.ravel())))
    print('Accuracy of decision tree classifier on test set: {:.2f}'
          .format(knc.score(X_test, y_test.values.ravel())))


def main():
    df_red = pd.read_csv('winequality-red-filtered.csv', sep=';', header=0)
    df_white = pd.read_csv('winequality-white.csv', sep=';', header=0)

    df_red['color'] = 0
    df_white['color'] = 1

    df = df_red.append(df_white, ignore_index=True)

    x_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                 "free sulfur dioxide",
                 "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

    y_column = ["quality"]

    random_state = 12

    naive_bayes(df_red, x_columns, y_column, random_state)

    knn(df_red, x_columns, y_column, random_state)

    knn_categories(df_red, x_columns, y_column, random_state, 2)
    knn_categories(df_red, x_columns, y_column, random_state, 3)
    tree_categories(df_red, x_columns, y_column, random_state, 2)
    tree_categories(df_red, x_columns, y_column, random_state, 3)


if __name__ == "__main__":
    main()
