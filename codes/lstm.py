from keras.utils import np_utils

from codes.common import dataextraction
from recognitionpart.dnn import LSTM
from recognitionpart.utilities import gettingfeaturevectorfromMFCC
from keras.callbacks import EarlyStopping, ModelCheckpoint
from recognitionpart import *

#for speech recognition
import speech_recognition as sr
import pyttsx3


def speak(text):
    engine = pyttsx3.init()
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate - 50)
    engine.say(text)
    #engine.save_to_file('test.wav')
    engine.runAndWait()

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)

        with open('speech.wav', 'wb') as f:
            f.write(audio.get_wav_data())
        said = ""
        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            speak("Sorry, I could not understand what you said.")

        return said

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

    #speech recognition - take input from microphone
    #after that save file in wav format and
    #filename = to that file


    user_speech= get_audio()

    filename = 'speech.wav'
    #filename = '../dataset/Neutral/n03.wav'
    print('prediction', model.predict_one(
        gettingfeaturevectorfromMFCC(filename, flatten=to_flatten)))

    earlystop = EarlyStopping(monitor='val_acc', mode='max', patience=75, restore_best_weights=True)

    # evaluate model, test data may differ from validation data



if __name__ == '__main__':
    lstm_example()
