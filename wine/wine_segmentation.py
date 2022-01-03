#!/usr/bin/env python3

"""TODO
- test other models:
    - nearest neighbour
    - furthest neighbour
    - average neighbour
    - non-Hierachical Clustering
    - Hierachical Clustering (teilendes, agglomeratives)
    - svm
    - naive bayes
    - neuronale netze
- verify if clustering is because of color or quality
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

df_red = pd.read_csv('winequality-red.csv', sep=';', header=0)
df_white = pd.read_csv('winequality-white.csv', sep=';', header=0)

df_red['color'] = 0
df_white['color'] = 1

df = df_red.append(df_white, ignore_index=True)

x_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

y_column = ["quality"]

""" KMeans Clustering
Segments ~80% of elements correctly.
Predicts red wine better.
Possible Explanation: Far more red wines in dataset.
"""
model = KMeans(n_clusters=2)
model.fit(df.iloc[:, :-1])  # exclude color (last) column

accuracy = np.mean(df['color'].values == model.labels_)
accuracy_red = np.mean(df_red['color'].values == model.predict(df_red.iloc[:, :-1]))
accuracy_white = np.mean(df_white['color'].values == model.predict(df_white.iloc[:, :-1]))

print('KMeans Segmentierung:')
print(f'Genauigkeit: {accuracy:.2f}')
print(f'Genauigkeit (rot): {accuracy_red:.2f}')
print(f'Genauigkeit (weiß): {accuracy_white:.2f}')

"""" KNeighborsClassifier 
Using MinMaxScaler.
Without splitting into categories.
"""

X = df[x_columns]
y = df[y_column]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    # random_state=2,
                                                    )

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train.values.ravel())
print("\nKNN on both on red and white wine:")
print('Accuracy of K-NN classifier on training set: {:.2f}'
      .format(knn.score(X_train_scaled, y_train.values.ravel())))
print('Accuracy of K-NN classifier on test set: {:.2f}'
      .format(knn.score(X_test_scaled, y_test.values.ravel())))

print(knn.predict(X_test_scaled))
print(y_test)

# TODO: combine both wine datasets and try KNN again

""" KNN 
Splitting wines into different quality ranges.
Good results.
Most wines are in the range of ~4-7
Splitting into 3 categories but splitting into 2 categories (e.g. 1-7 | 8-9)
brings very good results as well.
"""

df.loc[df.quality < 6, "quality"] = 0
df.loc[df.quality >= 6, "quality"] = 2
df.loc[df.quality >= 8, "quality"] = 3

print(df)

"""Prediction with KMeans and 3 clusters"""

# TODO: Add representation for result

berechnung = KMeans(n_clusters=3)
berechnung.fit(df[x_columns])

labels = berechnung.labels_

result = pd.DataFrame(labels, columns=['predicted value'])
result['original value'] = df.quality

print(result)

X = df[x_columns]
y = df[y_column]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    # random_state=2,
                                                    )

# KNN
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knc = KNeighborsClassifier(n_neighbors=5, algorithm='auto', metric='minkowski', p=2)

knc.fit(X_train, y_train.values.ravel())
print("KNN on both on red and white wine split into 3 categories(0-5/6-7/8-9):")
print('Accuracy of K-NN classifier on training set: {:.2f}'
      .format(knc.score(X_train, y_train.values.ravel())))
print('Accuracy of K-NN classifier on test set: {:.2f}'
      .format(knc.score(X_test, y_test.values.ravel())))

print(knc.predict(X_test))
print(y_test)

# TODO: Add KNN with 2 categories
# TODO: Try with only 1 of the datasets?
# TODO: Clean up and get rid of redundancy: outsource in methhods?
