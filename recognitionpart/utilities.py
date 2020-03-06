import numpy as np
import os
import sys
import scipy.io.wavfile as wav #reading wav files
from speechpy.feature import mfcc
from typing import Tuple

mean_signal_length = 32000

def gettingfeaturevectorfromMFCC(file_path: str, flatten: bool, mfcc_len: int = 39) -> np.ndarray:
    fs, signal = wav.read(file_path)

    s_len = len(signal)

    if s_len < mean_signal_length:
        pad_len = mean_signal_length - s_len
        pad_rem = pad_len % 2
        pad_len //= 2
        signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
    else:
        pad_len = s_len - mean_signal_length
        pad_len //= 2
        signal = signal[pad_len:pad_len + mean_signal_length]
    mel_coefficients = mfcc(signal, fs, num_cepstral=mfcc_len)
    if flatten:
        # Flatten the data
        mel_coefficients = np.ravel(mel_coefficients)
    return mel_coefficients

def get_data(data_path: str, flatten: bool = True, mfcc_len: int = 39,
             class_labels: Tuple = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")) -> \
    Tuple[np.ndarray, np.ndarray]:
    data = []
    labels = []
    names = []
    cur_dir = os.getcwd()
    sys.stderr.write('curdir: %s\n' % cur_dir)
    os.chdir(data_path)
    for i, directory in enumerate(class_labels):
        sys.stderr.write("started reading folder %s\n" % directory)
        os.chdir(directory)
        for filename in os.listdir('.'):
            filepath = os.getcwd() + '/' + filename
            feature_vector = gettingfeaturevectorfromMFCC(file_path=filepath,
                                                          mfcc_len=mfcc_len,
                                                          flatten=flatten)
            data.append(feature_vector)
            labels.append(i)
            names.append(filename)
        sys.stderr.write("ended reading folder %s\n" % directory)
        os.chdir('..')
    os.chdir(cur_dir)
    return np.array(data), np.array(labels)
