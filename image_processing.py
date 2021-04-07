import keras
import numpy as np
import cv2
import glob
import os
from numpy.core.fromnumeric import resize
from skimage import io

minValue = 70

labels = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "up",
    "down",
    "left",
    "right",
    "off",
    "on",
    "ok",
]


def preprocess(path):
    frame = cv2.imread(path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(
        th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res


class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras."""

    def __init__(self, mode='train', ablation=None, gesture_cls=labels,
                 batch_size=4, dim=(210, 280), n_channels=3, shuffle=True):
        """Initialise the data generator."""
        self.dim = dim
        self.batch_size = batch_size
        self.labels = {}
        self.list_IDs = []

        # glob through directory of each class
        for i, cls in enumerate(gesture_cls):
            paths = glob.glob(os.path.join("data2", cls, '*'))
            brk_point = int(len(paths)*0.8)
            if mode == 'train':
                paths = paths[:brk_point]
            else:
                paths = paths[brk_point:]
            if ablation is not None:
                paths = paths[:ablation]
            self.list_IDs += paths
            self.labels.update({p: i for p in paths})

        self.n_channels = n_channels
        self.n_classes = len(gesture_cls)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch."""
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data."""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        """Updates indexes after each epoch."""
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        """Generates data containing batch_size samples."""
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        delete_rows = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            img = cv2.imread(ID)
            img = img/255

            img = cv2.resize(img, (280, 210),
                             interpolation=cv2.INTER_AREA)

            # if img.shape[0] > 100 and img.shape[1] > 100:
            #     h, w, _ = img.shape
            #     img = img[int(h/2)-50:int(h/2)+50, int(w/2)-50:int(w/2)+50, :]
            # else:
            #     delete_rows.append(i)
            #     continue

            X[i, ] = img

            # Store class
            y[i] = self.labels[ID]

        X = np.delete(X, delete_rows, axis=0)
        y = np.delete(y, delete_rows, axis=0)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
