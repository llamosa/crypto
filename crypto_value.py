
from __future__ import print_function
import numpy as np
import pandas as pd
import data_driven as dd
import mach_learn as ml

import logging
logging.basicConfig(filename='crypto_predict.log', level=logging.INFO)
np.random.seed(1234)  # for reproducibility

import time
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, InputLayer
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import losses, callbacks, regularizers
from random import sample
from scipy.ndimage import imread
import glob
import os
import sys
import csv
from utilities import all_stats

from random import sample

tbCallBck = callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
earlyStopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10000, verbose=1, mode='auto')

def ensure_number(data):
    data[np.where(data == np.NAN), :] = 0

cutoff = 0
look_back = 15
lag = 5
# read data as dataframe
data_frame = pd.read_csv('C:/Users/home/Dropbox/crypto/CandlesJan2015-Dec2017.txt')
#data_frame = pd.read_csv('C:/Users/home/Dropbox/crypto/ProcessedCandlesJan2015-Dec2017.txt')

# data set up
#dataTable = data_frame[[' Open', ' Close', ' Vol']].as_matrix()
dataTable = data_frame[' Close'].as_matrix()[1:] - data_frame[' Close'].as_matrix()[:-1]

#Normilize by volume

labels = (data_frame[' Close'].as_matrix()[1:] - data_frame[' Close'].as_matrix()[:-1]).reshape(len(data_frame[' Open'])-1,1)
#labels = data_frame['Buy Sell'].as_matrix().reshape(len(data_frame['Buy Sell']),1)
labels = labels[look_back+lag:,]

data = np.atleast_3d(np.array([dataTable[start:start + look_back] for start in range(0, dataTable.shape[0] - (look_back+lag))]))
#data = np.array([dataTable[start:start + look_back] for start in range(0, dataTable.shape[0] - look_back)])
data = data.reshape((data.shape[0],data.shape[1]*data.shape[2]))

classic = True
deep = False

# turn labels into binary
y = np.zeros((len(labels), 1)).astype(int)
pos_id = np.where(abs(labels) > cutoff)[0]
y[pos_id] = 1

# validate every data is a number
ensure_number(data)
ensure_number(labels)

train_fraction = 0.3
cut = int(len(labels)*train_fraction)
ids = np.arange(len(labels))
np.random.shuffle(ids)
train_id = ids[0:cut]
test_id = ids[cut:]

if classic:
    partition=dd.Partition(trainset=train_id,testset=test_id)
    data = (data - np.min(data,axis=0))/(np.max(data,axis=0) - np.min(data,axis=0) )
    model = dd.Model(data=np.hstack((data,y)), function=ml.RandF(parameters={'n_estimators':1000,'min_samples_split':1000}), partition=partition, nfo=3)
    #model = dd.Model(data=np.hstack((data,labels)),function=ml.Sksvm(parameters={'regularization':0.1,'sigma':0.1}),partition=partition)
    model.training()
    #model.crossvalidating()
    model.testing()
    model.performance()
    model.summary()

if deep:

    nb_classes = 2
    nb_epoch = 100
    dropout = 0.4  # 0.2
    hidden = 10  # 80
    dense = 10
    batch_size = 512

    X_train = data[train_id,:].astype('float32')
    X_test = data[test_id,:].astype('float32')

    X_train = (X_train - np.min(X_train,axis=0) )/np.max(X_train,axis=0)
    X_test = (X_test - np.min(X_train,axis=0) )/np.max(X_train,axis=0)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')

    # convert class vectors to binary class matrices

    Y_train = np_utils.to_categorical(labels[train_id], nb_classes)
    Y_test = np_utils.to_categorical(labels[test_id], nb_classes)

    model = Sequential()
    model.add(Dense(dense, input_shape=(X_train.shape[1],)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    for i in range(2):
        model.add(Dense(dense))
        model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              epochs=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True, callbacks=[tbCallBck, earlyStopping])

    score = model.evaluate(X_test, Y_test, verbose=0)
    y_score_train = model.predict_proba(X_train)
    y_score_test = model.predict_proba(X_test)

    #print (y_score_train,y_train)
    #print (y_score_test,y_test)

    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    #print(y_score_train[:,1],Y_train[:,1])
    #print(y_score_test[:,1],Y_test[:,1])
    train_stats = all_stats(Y_train[:,1],y_score_train[:,1])
    test_stats = all_stats(Y_test[:,1],y_score_test[:,1],train_stats[-1])

    print('All stats train:',['{:6.2f}'.format(val) for val in train_stats])
    print('All stats test:',['{:6.2f}'.format(val) for val in test_stats])