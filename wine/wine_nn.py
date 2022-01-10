import pandas as pd
from tensorflow.keras import *


def main():
    df_red = pd.read_csv('winequality-red.csv', sep=';', header=0)
    df_white = pd.read_csv('winequality-white.csv', sep=';', header=0)

    df_red['color'] = 0
    df_white['color'] = 1

    df = df_red.append(df_white, ignore_index=True)

    x_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                 "free sulfur dioxide",
                 "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

    y_column = ["quality"]

    model = Sequential()


if __name__ == "__main__":
    main()
