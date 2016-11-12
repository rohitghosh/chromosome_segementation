import h5py
import numpy as np
import skimage as sk
#print sk.__version__
from skimage import io
from matplotlib import pyplot as plt
import random

h5f = h5py.File('/home/users/rohitg/LowRes_13434_overlapping_pairs.h5','r')
pairs = h5f['dataset_1'][:]
h5f.close()

train_valid_split = int(0.8*pairs.shape[0])

weight_list = [0.1, 1, 1, 10]

def weights_assign(train):
    if train == 0:
        return weight_list[0]
    elif train == 1:
        return weight_list[1]
    elif train == 2:
        return weight_list[2]
    elif train == 3:
        return weight_list[3]
    else:
        print ('wrong_going')

def labels_combine(train):
    new_train = train
    new_train[train==2] = 1
    new_train[train==3] = 2
    return new_train

def weights_matrix_gen(train,weight_list):
    func = np.vectorize(weights_assign, otypes=[np.float64])
    weights = func(train)
    return weights


def train_data_loader(batch_size = 10, combine_label = False):
    for i in range(0,train_valid_split,batch_size):
        # indices = random.sample(range(i,i+200),batch_size)
        indices = range(i,i+batch_size)
        X_train = pairs[indices,:,:,0]
        Y_train = pairs[indices,:,:,1]
        if combine_label:
            Y_train = labels_combine(Y_train)
        if np.amax(Y_train) > 3:
            continue
        weights = weights_matrix_gen(Y_train,weight_list)
        X_train = X_train.reshape((batch_size,1,X_train.shape[1],X_train.shape[2]))
        Y_train = Y_train.reshape((batch_size,1,Y_train.shape[1],Y_train.shape[2]))
        yield X_train, Y_train, weights

def valid_data_loader(nb_val_samples = 200, batch_size = 10, combine_label = False):
    for i in range(train_valid_split,train_valid_split + nb_val_samples,batch_size):
        indices = range(i,i+batch_size)
        X_valid = pairs[indices,:,:,0]
        Y_valid = pairs[indices,:,:,1]
        if combine_label:
            Y_valid = labels_combine(Y_valid)
        if np.amax(Y_valid) > 3:
            continue
        X_valid = X_valid.reshape((batch_size,1,X_valid.shape[1],X_valid.shape[2]))
        Y_valid = Y_valid.reshape((batch_size,1,Y_valid.shape[1],Y_valid.shape[2]))
        yield X_valid, Y_valid
