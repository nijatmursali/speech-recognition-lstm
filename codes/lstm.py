from keras.utils import np_utils

from codes.common import dataextraction
from recognitionpart.dnn import LSTM
from recognitionpart.utilities import gettingfeaturevectorfromMFCC
from keras.callbacks import EarlyStopping, ModelCheckpoint
from recognitionpart import *

def lstm_example():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = dataextraction(
        flatten=to_flatten)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    print('Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape,
                 num_classes=num_labels)
    model.train(x_train, y_train, x_test, y_test_train, n_epochs=50)
    evaluate = model.evaluate(x_test, y_test)
    filename = '../dataset/Happy/h13.wav'
    print('prediction', model.predict_one(
        gettingfeaturevectorfromMFCC(filename, flatten=to_flatten)), 'Actual 3')

    earlystop = EarlyStopping(monitor='val_acc', mode='max', patience=75, restore_best_weights=True)

    # evaluate model, test data may differ from validation data
    print(evaluate)


if __name__ == '__main__':
    lstm_example()
