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
    model.add(Dense(12, input_dim=X_train.shape[1], activation="relu",
                    name="Input"))

    model.add(Dense(100, activation="relu", name="internal_layer"))
    model.add(Dense(1, activation="linear", name="Output"))

    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])

    print("\nRegression:\n")

    model.summary()

    hist = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test),
                     verbose=0)

    def eval(X, y):
        ergebnis = model.evaluate(X, y, verbose=0)
        y_vorhersage = model.predict(X)
        print("{}: {:.2f}\n{}: {:.2f}".format(model.metrics_names[0], ergebnis[0], model.metrics_names[1], ergebnis[1]))
        print("R2: {:.2f}".format(r2_score(y, y_vorhersage)))

    print("Trainingdata:")
    eval(X_train, y_train)
    print("Testdata:")
    eval(X_test, y_test)

    auswertung = pd.DataFrame.from_dict(hist.history)
    print("------ Ploting Neural Net Graph --------")
    fig = plt.figure(figsize=(20, 8), num="Neural Net")
    bild1 = fig.add_subplot(121)
    bild1.plot(auswertung.index, auswertung[["loss"]], color='blue')
    bild1.plot(auswertung.index, auswertung[["val_loss"]], color='red')
    bild1.legend(['Training', 'Validation'])
    bild1.set_xlabel('epoch')
    bild1.set_ylabel(model.metrics_names[0])
    bild1.set_title("Neural Netz learns: Loss-Graph")
    bild2 = fig.add_subplot(122)
    bild2.plot(auswertung.index, auswertung.iloc[:, 1], color='blue')
    bild2.plot(auswertung.index, auswertung.iloc[:, 3], color='red')
    bild2.legend(['Training', 'Validation'])
    bild2.set_xlabel('epoch')
    bild2.set_ylabel(model.metrics_names[1])
    bild2.set_title("Neural Net learns: Mean-Absolute-Error-Graph")
    plt.show()


def model2(df, x_columns, y_column, state, random_state):
    df = df.copy(deep=True)

    model = Sequential()

    if state == 1:

        df.loc[df.quality < 7, "quality"] = 0
        df.loc[df.quality >= 7, "quality"] = 1

        X = df[x_columns]
        y = df[y_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

        y_train_cat = to_categorical(y_train, 2)
        y_test_cat = to_categorical(y_test, 2)

        model.add(Dense(12, input_dim=X.shape[1], activation="relu",
                        name="Input"))
        model.add(Dense(100, activation="relu", name="internal_layer"))
        model.add(Dense(2, activation="softmax", name="Output"))

        print("\nClassification with 2 classes(0-6/7-10):\n")
    elif state == 2:

        df.loc[df.quality < 6, "quality"] = 0
        df.loc[df.quality >= 6, "quality"] = 1
        df.loc[df.quality >= 8, "quality"] = 2

        X = df[x_columns]
        y = df[y_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

        y_train_cat = to_categorical(y_train, 3)
        y_test_cat = to_categorical(y_test, 3)

        model.add(Dense(12, input_dim=X.shape[1], activation="relu",
                        name="Input"))
        model.add(Dense(100, activation="relu", name="internal_layer"))
        model.add(Dense(3, activation="softmax", name="Output"))

        print("\nClassification with 3 classes(0-5/6-7/8-10):\n")

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    historie = model.fit(X_train, y_train_cat, epochs=900, batch_size=32, verbose=0)

    def eval(X, y):
        ergebnis = model.evaluate(X, y, verbose=0)
        print("{}: {:.2f}\n{}: {:.2f}".format(model.metrics_names[0], ergebnis[0], model.metrics_names[1], ergebnis[1]))

    print("Trainingdata:")
    eval(X_train, y_train_cat)
    print("\nTestdata:")
    eval(X_test, y_test_cat)

    auswertung = pd.DataFrame.from_dict(historie.history)
    print("------ Ploting Neural Net Graph --------")
    fig = plt.figure(figsize=(20, 8), num="Neuronal Net")
    bild1 = fig.add_subplot(121)
    bild1.plot(auswertung.index, auswertung.iloc[:, 0], color='blue')
    bild1.legend(['Training'])
    bild1.set_xlabel('epoch')
    bild1.set_ylabel(model.metrics_names[0])
    bild1.set_title("Neural Net learns: Loss-Kurve")
    bild2 = fig.add_subplot(122)
    bild2.plot(auswertung.index, auswertung.iloc[:, 1], color='blue')
    bild2.legend(['Training'])
    bild2.set_xlabel('epoch')
    bild2.set_ylabel(model.metrics_names[1])
    bild2.set_title("Neural Net learns: Accuracy")
    fig.show()


def main():
    df_red = pd.read_csv('winequality-red-filtered.csv', sep=';', header=0)
    df_white = pd.read_csv('winequality-white-filtered.csv', sep=';', header=0)

    df = df_red.append(df_white, ignore_index=True)

    x_columns = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides",
                 "free sulfur dioxide",
                 "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

    y_column = ["quality"]

    random_state = 2

    model1(df_red, x_columns, y_column, random_state=random_state, state=1)
    model2(df_red, x_columns, y_column, state=1, random_state=random_state)
    model2(df_red, x_columns, y_column, state=2, random_state=random_state)


if __name__ == "__main__":
    main()
