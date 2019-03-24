import numpy as np
import glob, os, sys
from sklearn.preprocessing import StandardScaler

def load_data(res):
    # load data
    path = '../data/d%s/' %res
    print('load data from %s and normalize data.' %path)

    sc = StandardScaler()

    d = {}
    for data in glob.glob(path+'*'):
        name = data.split('_')[1].strip('.npy')  # feature name
        d[name] = np.load(data)
     
    # reshape to concat array and normalize features
    for key, value in d.items():
        if key != 'sigma':
            d[key] = d[key].reshape(-1,1)
            d[key] = sc.fit_transform(d[key])
            d[key] = d[key].reshape(value.shape[0], value.shape[1], value.shape[2], 1)
            #print(key, d[key].shape)
        else:
            d[key] = d[key].reshape(value.shape[0], value.shape[1], value.shape[2], 1)
            #print(key, d[key].shape)

    # set X and y
    X = np.concatenate((d['u'], d['v'], d['th'], d['qv']), axis=-1)
    y = d['sigma']


    return X, y

def pool_wrap(array, size):
    # Periodic boundary
    # size is the CNN kernel size, 3*3 kernel(input 3), 5*5 kernel(input 5)...

    new = np.zeros((array.shape[0], array.shape[1] + (size-1), array.shape[2]+(size-1), array.shape[3]))

    for sample in range(array.shape[0]):
        for feature in range(array.shape[3]):
            tmp = array[sample, :, :, feature]
            tmp_ = np.pad(tmp, int((size-1)/2), 'wrap')
            new[sample, :, :, feature] = tmp_
    return new

def cnn_type_x(arr, size):
    # size is kernel size 3*3 kenel (input 3), 5*5 kernel (input 5)...
    out = np.zeros((arr.shape[0]* (arr.shape[1] - (size-1))* (arr.shape[2] - (size-1)), size, size, arr.shape[3]))

    count = 0
    for s in range(arr.shape[0]):
        for x in range(0, arr.shape[1]-(size-1)):
            for y in range(0, arr.shape[2]-(size-1)):
                out[count] = arr[s, x:x+size, y:y+size, :]

                count +=1 
    print('X shape (CNN input): ', out.shape)
    return out

def cnn_type_y(arr):
    out = np.zeros((arr.shape[0]*arr.shape[1]*arr.shape[2], 1, 1, arr.shape[3]))

    count = 0

    for s in range(arr.shape[0]):
        for x in range(0, arr.shape[1]):
            for y in range(0, arr.shape[2]):
                out[count] = arr[s, x, y, :]

                count += 1
    out = np.squeeze(out)
    out = out.reshape(out.shape[0], 1)
    print('y shape for CNN :', out.shape)
    return out

def split_shuffle(X, y, TEST_SPLIT =0.2):
    # shuffle
    indices = np.arange(X.shape[0])
    nb_test_samples = int(TEST_SPLIT * X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    X_train = X[nb_test_samples:]
    X_test = X[0:nb_test_samples]
    y_train = y[nb_test_samples:]
    y_test = y[0:nb_test_samples]

    print('X_train : ', X_train.shape)
    print('X_test : ', X_test.shape)
    print('y_train : ', y_train.shape)
    print('y_test : ', y_test.shape)

    return X_train, X_test, y_train, y_test


# MAIN
if __name__ == '__main__':
    # type  python train res size
    # python train 9 15
    # python train 15 9
    # python train 45 3
    # python train 135 1
    res = int(sys.argv[1])
    size = int(sys.argv[2])
    X, y = load_data(res)
    X = pool_wrap(X, size)
    X = cnn_type_x(X, size)
    y = cnn_type_y(y)
    X_train, X_test, y_train, y_test = split_shuffle(X, y, 0.2)

