import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D
import os

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=32, dim=(16, 200), n_channels=1, shuffle=True):
        'Initialization'
        self.files = []
        self.indexes = None
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files)/self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        files_temp = [self.files[k] for k in indexes]

        # Generate data
        X = self.__load_files(files_temp)

        return X, X

    def load_directory(self, directory, extension="npy"):
        self.files.extend([os.path.join(directory, x) for x in os.listdir(directory) if x.endswith(extension) ])
        self.indexes = np.arange(len(self.files))

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

    def __load_files(self, files):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, file in enumerate(files):
            # Store sample
            X[i,] = np.expand_dims(np.load(file), axis=2)

        return X
