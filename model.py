import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras.models import Model
from keras.layers import Input, Dense, Reshape, Flatten, Conv2D
import os
from data_generator import DataGenerator

filtered_dir = "GBT_57532_09539_HIP56445_fine/filtered"
data_gen = DataGenerator()
data_gen.load_directory(filtered_dir+"/5")


inp = Input(shape=(16, 200, 1))
flat = Flatten()(inp)
dense1 = Dense(4096, activation="relu")(flat)
dense2 = Dense(4096, activation="relu")(dense1)
out_flat = Dense(16*200)(dense2)
out = Reshape((16, 200, -1))(out_flat)

model = Model(inputs=inp, outputs=out)

model.compile("adam", loss="mse")

model.fit_generator(generator=data_gen, epochs=5, use_multiprocessing=True, workers=4)
