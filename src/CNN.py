import sys, argparse, os, time, pickle
import numpy as np

from Preprocessing import *
from config import ModelMGPU
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint


def CNN(size, features):
    print("Build model!!")
    model = Sequential()
    model.add(Convolution2D(32, (2,2), use_bias=True, padding='SAME', strides=1, activation='elu', input_shape=(size, size, features)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(256, activation='elu'))
    model.add(Dense(256, activation='elu'))
    model.add(Dense(1, activation='elu'))

    # optimize 
    #adam = Adam()

    #print('Compiling model...')
    #model.compile(loss='mse', optimizer=adam)
    
    #print(model.summary())
    return model


def main():

    act = sys.argv[1]
    res = int(sys.argv[2])
    size = int(sys.argv[3])
    features = 4
    model_save_path = '../model/CNN_%s/'%res
    X, y = load_data(res)
    X = pool_wrap(X, size)
    X = cnn_type_x(X, size)
    y = cnn_type_y(y)
    X_train, X_test, y_train, y_test = split_shuffle(X, y, 0.2)

    if act == "train":

        model = CNN(size, features)
        parallel_model = ModelMGPU(model, 2)
        adam = optimizers.Adam(lr=0.001/3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        parallel_model.compile(optimizer = 'adam', loss='mean_squared_error')
        print(model.summary())

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)
        filepath = model_save_path + "/%s_feature_{epoch:03d}_{loss:.3e}_size_%s.hdf5"%(features, size)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, period=1)
        earlystopper = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        history = parallel_model.fit(X_train, y_train, validation_split=0.2, batch_size=512, epochs=500, shuffle=True, callbacks=[checkpoint, earlystopper])
        cost = parallel_model.evaluate(X_test, y_test, batch_size=1024)

        print('Cost : ', cost)



if __name__ == '__main__':
	main()

