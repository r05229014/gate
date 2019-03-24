import sys, argparse, os, time, pickle
import numpy as np
import matplotlib.pyplot as plt

from Preprocessing import *
from config import ModelMGPU
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint

res = 15
size = 9

X, y = load_data(res)
X = pool_wrap(X, size)
X = cnn_type_x(X, size)
y = cnn_type_y(y)
X_train, X_test, y_train, y_test = split_shuffle(X, y, 0.2)

model_path = sys.argv[1]
model = load_model(model_path)

y_pre = model.predict(X_test)
plt.scatter(y_pre, y_test)
plt.savefig('./%s.png'%res)

                              

