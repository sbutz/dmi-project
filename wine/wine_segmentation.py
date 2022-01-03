#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


df_red = pd.read_csv('./wine/winequality-red.csv', sep=';', header=0)
df_white = pd.read_csv('./wine/winequality-white.csv', sep=';', header=0)

df_red['colour'] = 0
df_white['colour'] = 1

# append both
df = df_red.append(df_white, ignore_index=True)

x_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
             "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

y_column = ["colour"]

# print(df)

# kmeans with 2 clusters
berechnung = KMeans(n_clusters=2)

berechnung.fit(df[x_columns])

labels = berechnung.labels_
print(labels.tolist())
print(df['colour'].tolist())

# -> doens't know the difference between red and white wine

