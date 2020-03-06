import numpy as np

from sklearn.model_selection import train_test_split
from recognitionpart.utilities import get_data, \
    gettingfeaturevectorfromMFCC

# ADDING PATH AND LABELS FOR CLASSES
DATASETPATH = '../dataset'
CLASS_LABELS = ("Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise")

def dataextraction(flatten):
    data, labels = get_data(DATASETPATH, class_labels=CLASS_LABELS, flatten=flatten)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), len(CLASS_LABELS)


def get_feature_vector(file_path, flatten):
    return gettingfeaturevectorfromMFCC(file_path, flatten, mfcc_len=39)