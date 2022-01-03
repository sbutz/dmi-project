#!/usr/bin/env python3

"""TODO
- test other models:
    - furthest neighbour
    - average neighbour
    - non-Hierachical Clustering
    - Hierachical Clustering (teilendes, agglomeratives)
    - svm
    - naive bayes
    - neuronale netze
"""

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


df_red = pd.read_csv('./wine/winequality-red.csv', sep=';', header=0)
df_white = pd.read_csv('./wine/winequality-white.csv', sep=';', header=0)

df_red['color'] = 0
df_white['color'] = 1

df = df_red.append(df_white, ignore_index=True)

""" KMeans Clustering
Segments ~80% of elements correctly.
Predicts red wine better.
Possible Explanation: Far more red wines in dataset.
"""
model = KMeans(n_clusters=2)
model.fit(df.iloc[:,:-1]) # exclude color (last) column

accuracy = np.mean(df['color'].values == model.labels_)
accuracy_red = np.mean(df_red['color'].values == model.predict(df_red.iloc[:,:-1]))
accuracy_white = np.mean(df_white['color'].values == model.predict(df_white.iloc[:,:-1]))

print('KMeans Segmentierung:')
print(f'Genauigkeit: {accuracy:.2f}')
print(f'Genauigkeit (rot): {accuracy_red:.2f}')
print(f'Genauigkeit (weiß): {accuracy_white:.2f}')