"""
This file contains classes which implement deep neural networks namely CNN and LSTM
"""
import sys

import numpy as np
from keras import Sequential
from keras.layers import LSTM as KERAS_LSTM, Dense, Dropout, Conv2D, Flatten, \
    BatchNormalization, Activation, MaxPooling2D, Embedding, Bidirectional

from . import Model


class DNN(Model):

    def __init__(self, input_shape, num_classes, **params):

        super(DNN, self).__init__(**params)
        self.input_shape = input_shape
        self.model = Sequential()
        self.make_default_model()
        self.model.add(Dense(num_classes, activation='softmax'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam',
                           metrics=['accuracy'])
        print(self.model.summary(), file=sys.stderr)
        self.save_path = self.save_path or self.name + '_best_model.h5'

    def load_model(self, to_load):

        try:
            self.model.load_weights(to_load)
        except:
            sys.stderr.write("Invalid saved file provided")
            sys.exit(-1)

    def save_model(self):
        self.model.save_weights(self.save_path)

    def train(self, x_train, y_train, x_val=None, y_val=None, n_epochs=50):
        best_acc = 0
        if x_val is None or y_val is None:
            x_val, y_val = x_train, y_train
        for i in range(n_epochs):
            # Shuffle the data for each epoch in unison inspired
            # from https://stackoverflow.com/a/4602224
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]
            self.model.fit(x_train, y_train, batch_size=32, epochs=1)
            loss, acc = self.model.evaluate(x_val, y_val)
            if acc > best_acc:
                best_acc = acc
        self.trained = True

    def predict_one(self, sample):
        if not self.trained:
            sys.stderr.write(
                "Model should be trained or loaded before doing predict\n")
            sys.exit(-1)
        return np.argmax(self.model.predict(np.array([sample])))

    def make_default_model(self) -> None:
        raise NotImplementedError()


class CNN(DNN):
    def __init__(self, **params):
        params['name'] = 'CNN'
        super(CNN, self).__init__(**params)

    def make_default_model(self):
        self.model.add(Conv2D(8, (13, 13),
                              input_shape=(
                                  self.input_shape[0], self.input_shape[1], 1)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(8, (13, 13)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))
        self.model.add(Conv2D(8, (13, 13)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(8, (2, 2)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 1)))
        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))


class LSTM(DNN):
    """
    This class handles CNN for speech emotion recognitions
    """

    def __init__(self, **params):
        params['name'] = 'LSTM'
        super(LSTM, self).__init__(**params)

    def make_default_model(self):
        """
        Makes the LSTM model with keras with the default hyper parameters.
        """
        max_features = 20000
        maxlen = 100
        batch_size = 32
        # self.model = Sequential()
        # self.model.add(Embedding(max_features, 128, input_length=maxlen))
        # self.model.add(Bidirectional(KERAS_LSTM(64)))
        # self.model.add(Dropout(0.5))
        # self.model.add(Dense(32, activation='sigmoid'))
        # self.model.compile('adam','binary_crossentropy', metrics=['accuracy'])
        self.model.add(
            KERAS_LSTM(128,
                       input_shape=(self.input_shape[0], self.input_shape[1])))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(16, activation='tanh'))
        #self.model.add(Dense(1, activation='sigmoid'))
        #self.model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

        return self.model
