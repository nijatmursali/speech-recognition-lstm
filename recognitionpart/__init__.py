"""
recognitionpart module.
Provides a library to perform speech emotion recognition on `savee` data set
"""
import sys
from typing import Tuple

import numpy
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import seaborn as sn
import pandas as pd


from keras.models import Sequential
from keras.layers import Dense

class Model(object):

    def __init__(self, save_path: str = '', name: str = 'Not Specified'):

        # Place holder for model
        self.model = None
        # Place holder on where to save the model
        self.save_path = save_path
        # Place holder for name of the model
        self.name = name
        # Model has been trained or not
        self.trained = False

    def train(self, x_train: numpy.ndarray, y_train: numpy.ndarray,
              x_val: numpy.ndarray = None,
              y_val: numpy.ndarray = None) -> None:

        raise NotImplementedError()

    def predict(self, samples: numpy.ndarray) -> Tuple:

        results = []
        for _, sample in enumerate(samples):
            results.append(self.predict_one(sample))
        return tuple(results)

    def predict_one(self, sample) -> int:
        raise NotImplementedError()

    def restore_model(self, load_path: str = None) -> None:
        to_load = load_path or self.save_path
        if to_load is None:
            sys.stderr.write(
                "Provide a path to load from or save_path of the model\n")
            sys.exit(-1)
        self.load_model(to_load)
        self.trained = True

    def load_model(self, to_load: str) -> None:
        raise NotImplementedError()

    def save_model(self) -> None:
        raise NotImplementedError()

    def evaluate(self, x_test: numpy.ndarray, y_test: numpy.ndarray) -> None:
        predictions = self.predict(x_test)
        print(type(predictions))
        print(y_test)
        print(predictions)
        accscr = accuracy_score(y_pred=predictions, y_true=y_test)

        accscr = accscr
        confmatrix = confusion_matrix(y_pred=predictions, y_true=y_test)

        #print('Accuracy:%.3f\n' % accuracy_score(y_pred=predictions, y_true=y_test))
        print("Accuracy: %.3f\n" % accscr)
        print('Confusion matrix:', confusion_matrix(y_pred=predictions,
                                                    y_true=y_test))


        df_cm = pd.DataFrame(confmatrix, index = [i for i in "1234567"], columns=[i for i in "1234567"])
        plt.figure(figsize=(7,7))
        sn.heatmap(df_cm, annot=True)

        plt.show()

        #dataset = numpy.loadtxt("/home/nijatmursali/Desktop/UNIVERSITY/NeuralNetworks/pima-indians-diabetes.data.csv", delimiter=",")
        #X = dataset[:,0:8]
        #Y = dataset[:,8]

        # model = Sequential()
        # model.add(Dense(12, input_dim=8, activation='relu'))
        # model.add(Dense(8, activation='relu'))
        # model.add(Dense(1, activation='sigmoid'))
        # # Compile model
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #
        # history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
        #
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()

        #return accscr, confmatrix

    def training(self):
        #train_x, test_x, train_y, test_y = train_test_split(x_test, y_test, test_size=0.2, random_state=42)
        pass

    def plotting(self):
        pass


