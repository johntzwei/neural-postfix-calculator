import itertools
import numpy as np

import seq2seq
from seq2seq.models import SimpleSeq2Seq, Seq2Seq

import keras
from keras.models import Sequential
from keras.layers import Masking, Dropout, Activation
from keras.layers.recurrent import LSTM, SimpleRNN

cmp_all = lambda models, optimizer='adam', loss='mean_squared_error', metrics=['accuracy'] : [ model.compile(optimizer=optimizer, loss=loss, metrics=metrics) for model in models ]

def preliminary_seq2seq(input_shape, output_shape):
    names = [ 'seq2seq-1x10' ]
    nnets = [ 
            seq2seq(input_shape, output_shape, hidden_dims=50, depth=(2,2), dropout=0.5) ]
    cmp_all(nnets)
    return list(zip(names, nnets))

def architectureSearch(input_shape):
    names = []
    nnets = []
    for i in range(5, 25, 5):
        names.append('2-lstm-%d' % i)
        nnets.append(lstm(input_shape, hidden_dims=[i]))
    cmp_all(nnets)
    return list(zip(names, nnets))

def preliminaries(input_shape):
    names = [ '2-lstm' ]
    nnets = [ lstm(input_shape, hidden_dims=[10]) ]
    cmp_all(nnets)
    return list(zip(names, nnets))

def seq2seq(input_shape, output_shape, hidden_dims=[10], depth=(1,1), dropout=0.25):
    model = Seq2Seq(input_dim=input_shape[1], hidden_dim=hidden_dims, output_length=output_shape[0], output_dim=output_shape[1], depth=depth, dropout=dropout, teacher_force=True)
    return model

def lstm(input_shape, hidden_dims=[], dropout=0.25):
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=input_shape))
    for dim in hidden_dims:
        model.add(LSTM(dim, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(1, activation='elu'))
    return model
