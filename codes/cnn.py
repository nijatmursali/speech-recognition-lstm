from keras.utils import np_utils

from codes.common import dataextraction
from recognitionpart.dnn import CNN
from recognitionpart.utilities import gettingfeaturevectorfromMFCC

def cnn():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = dataextraction(
        flatten=to_flatten)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    in_shape = x_train[0].shape
    x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    model = CNN(input_shape=x_train[0].shape,
                num_classes=num_labels)
    model.train(x_train, y_train, x_test, y_test_train)
    model.evaluate(x_test, y_test)
    filename = '../dataset/Happy/h01.wav'

    mfcc = gettingfeaturevectorfromMFCC(filename, flatten=to_flatten)
    mfcc = mfcc.reshape(mfcc.shape[0], mfcc.shape[1], 1)

    print('prediction', model.predict_one(mfcc),'Actual 3')
    print('CNN Done')


if __name__ == "__main__":
    cnn()