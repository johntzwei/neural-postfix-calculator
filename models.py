import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Masking, Dropout, Activation
from keras.layers.recurrent import LSTM, SimpleRNN

cmp_all = lambda models, optimizer='adam', loss='mean_squared_error', metrics=['accuracy'] : [ model.compile(optimizer=optimizer, loss=loss, metrics=metrics) for model in models ]

def preliminaries(input_shape):
    names = [ '2-lstm' ]
    nnets = [ lstm(input_shape, hidden_dims=[10]) ]
    cmp_all(nnets)
    return list(zip(names, nnets))

def lstm(input_shape, hidden_dims=[], dropout=0.25):
    model = Sequential()
    model.add(Masking(mask_value=np.zeros(input_shape[1]), input_shape=input_shape))
    for dim in hidden_dims:
        model.add(LSTM(dim, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(1))
    model.add(Activation('elu'))
    return model
