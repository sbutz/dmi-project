import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def model1(df, x_columns, y_column, random_state, state):
    df = df.copy(deep=True)

    X = df[x_columns]
    y = df[y_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    model = Sequential()
    model.add(Dense(1, input_dim=X_train.shape[1], activation="relu",
                    name="Input"))

    model.add(Dense(100, activation="relu", name="internal_layer"))
    model.add(Dense(1, activation="linear", name="Output"))

    # Compilieren des Modells
    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    print("\nRegression:\n")

    model.summary()

    # Modell trainieren mit 100 Durchläufen
    hist = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                     verbose=0)

    # Modell evaluieren
    def eval(X, y):
        ergebnis = model.evaluate(X, y, verbose=0)
        y_vorhersage = model.predict(X)
        print("{}: {:.2f}\n{}: {:.2f}".format(model.metrics_names[0], ergebnis[0], model.metrics_names[1], ergebnis[1]))
        print("R2: {:.2f}".format(r2_score(y, y_vorhersage)))

    print("Trainingsdaten:")
    eval(X_train, y_train)  # Evaluierung der Trainingsdaten
    print("Testdaten:")
    eval(X_test, y_test)  # Evaluierung der Testdaten

    auswertung = pd.DataFrame.from_dict(hist.history)  # Fit Daten in DataFrame umwandeln
    print("------ Lernen des Netzes plotten --------")
    fig = plt.figure(figsize=(20, 8), num="Neuronales Netz")
    bild1 = fig.add_subplot(121)
    bild1.plot(auswertung.index, auswertung[["loss"]], color='blue')
    bild1.plot(auswertung.index, auswertung[["val_loss"]], color='red')
    bild1.legend(['Training', 'Validierung'])
    bild1.set_xlabel('epoch')
    bild1.set_ylabel(model.metrics_names[0])
    bild1.set_title("Neuronales Netz lernt: Loss-Kurve")
    bild2 = fig.add_subplot(122)
    bild2.plot(auswertung.index, auswertung.iloc[:, 1], color='blue')  # mean_absolute_error
    bild2.plot(auswertung.index, auswertung.iloc[:, 3], color='red')  # val_mean_absolute_erro
    bild2.legend(['Training', 'Validierung'])
    bild2.set_xlabel('epoch')
    bild2.set_ylabel(model.metrics_names[1])
    bild2.set_title("Neuronales Netz lernt: Mean-Absolute-Error-Kurve")
    plt.show()


def model2(df, x_columns, y_column, state):
    df = df.copy(deep=True)

    # X = df[x_columns]
    # y = df[y_column]

    model = Sequential()
    # model.add(Dense(1, input_dim=X.shape[1], activation="relu",
    #                 name="Input"))
    # model.add(Dense(100, activation="relu", name="internal_layer"))

    if state == 1:

        df.loc[df.quality < 7, "quality"] = 0
        df.loc[df.quality >= 7, "quality"] = 1

        X = df[x_columns]
        y = df[y_column]

        y_cat = to_categorical(y, 2)

        model.add(Dense(1, input_dim=X.shape[1], activation="relu",
                        name="Input"))
        model.add(Dense(100, activation="relu", name="internal_layer"))

        model.add(Dense(2, activation="softmax", name="Output"))

        # Compilieren des Modells
        model.compile(optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        print("\nClassification with 2 classes(0-6/7-9):\n")
    elif state == 2:

        df.loc[df.quality < 6, "quality"] = 0
        df.loc[df.quality >= 6, "quality"] = 1
        df.loc[df.quality >= 8, "quality"] = 2

        X = df[x_columns]
        y = df[y_column]

        y_cat = to_categorical(y, 3)

        model.add(Dense(1, input_dim=X.shape[1], activation="relu",
                        name="Input"))
        model.add(Dense(100, activation="relu", name="internal_layer"))
        model.add(Dense(3, activation="softmax", name="Output"))

        # Compilieren des Modells
        model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

        print("\nClassification with 3 classes(0-5/6-7/8-9)::\n")

    # Compilieren des Modells
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Modell trainieren mit 200 Durchläufen
    historie = model.fit(X, y_cat, epochs=200, batch_size=16, verbose=0)

    # Modell evaluieren
    def eval(X, y_cat):
        ergebnis = model.evaluate(X, y_cat, verbose=0)
        print("{}: {:.2f}\n{}: {:.2f}".format(model.metrics_names[0], ergebnis[0], model.metrics_names[1], ergebnis[1]))

    print("Trainingsdaten:")
    eval(X, y_cat)  # Evaluierung der Trainingsdaten

    auswertung = pd.DataFrame.from_dict(historie.history)  # Fit Daten in DataFrame umwandeln
    print("------ Lernen des Netzes plotten --------")
    fig = plt.figure(figsize=(20, 8), num="Neuronales Netz")
    bild1 = fig.add_subplot(121)
    bild1.plot(auswertung.index, auswertung.iloc[:, 0], color='blue')
    bild1.legend(['Training'])
    bild1.set_xlabel('epoch')
    bild1.set_ylabel(model.metrics_names[0])
    bild1.set_title("Neuronales Netz lernt: Loss-Kurve")
    bild2 = fig.add_subplot(122)
    bild2.plot(auswertung.index, auswertung.iloc[:, 1], color='blue')
    bild2.legend(['Training'])
    bild2.set_xlabel('epoch')
    bild2.set_ylabel(model.metrics_names[1])
    bild2.set_title("Neuronales Netz lernt: Accuracy")
    fig.show()


def main():
    df_red = pd.read_csv('winequality-red.csv', sep=';', header=0)
    df_white = pd.read_csv('winequality-white.csv', sep=';', header=0)

    df = df_red.append(df_white, ignore_index=True)

    x_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                 "free sulfur dioxide",
                 "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

    y_column = ["quality"]

    model1(df, x_columns, y_column, random_state=2, state=1)
    model2(df, x_columns, y_column, state=2)


if __name__ == "__main__":
    main()
