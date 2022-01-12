from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np
import pandas as pd

""" KMeans Clustering
Segments ~80% of elements correctly.
"""


def kmeans(df, df_red, df_white, random_state):
    model = KMeans(n_clusters=2, random_state=random_state)
    model.fit(df.iloc[:, :-1])  # exclude color (last) column

    accuracy = np.mean(df['color'].values == model.labels_)
    accuracy_red = np.mean(df_red['color'].values == model.predict(df_red.iloc[:, :-1]))
    accuracy_white = np.mean(df_white['color'].values == model.predict(df_white.iloc[:, :-1]))

    print('KMeans Clustering:')
    print(f'accuracy: {accuracy:.2f}')
    print(f'accuracy (red): {accuracy_red:.2f}')
    print(f'accuracy (white): {accuracy_white:.2f}')

def agglomerative (df, df_red, df_white, random_state):
    model = AgglomerativeClustering(n_clusters=2, linkage='complete', affinity='l2')
    model.fit(df.iloc[:, :-1])  # exclude color (last) column

    accuracy = np.mean(df['color'].values == model.labels_)
    accuracy_red = np.mean(df_red['color'].values == model.labels_[df['color'] == 0])
    accuracy_white = np.mean(df_white['color'].values == model.labels_[df['color'] == 1])

    print('Agglomerative Clustering:')
    print(f'accuracy: {accuracy:.2f}')
    print(f'accuracy (red): {accuracy_red:.2f}')
    print(f'accuracy (white): {accuracy_white:.2f}')


def main():
    df_red = pd.read_csv('winequality-red-filtered.csv', sep=';', header=0)
    df_white = pd.read_csv('winequality-white-filtered.csv', sep=';', header=0)

    df_red['color'] = 0
    df_white['color'] = 1

    df = df_red.append(df_white, ignore_index=True)

    random_state = 2

    kmeans(df, df_red, df_white, random_state=random_state)
    agglomerative(df, df_red, df_white, random_state=random_state)


if __name__ == "__main__":
    main()
