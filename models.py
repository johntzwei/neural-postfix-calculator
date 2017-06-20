import itertools
import numpy as np

import seq2seq
from seq2seq.models import SimpleSeq2Seq, Seq2Seq

import keras
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Masking, Dropout, Activation
from keras.layers.recurrent import LSTM, SimpleRNN

cmp_all = lambda models, optimizer='adam', loss='mean_squared_error', metrics=['accuracy'] : [ model.compile(optimizer=optimizer, loss=loss, metrics=metrics) for model in models ]

def load_model(model_type, model_path):
    if model_type == 'Seq2Seq':
        from recurrentshop import RecurrentSequential
        from recurrentshop.engine import _OptionalInputPlaceHolder
        from seq2seq.cells import LSTMDecoderCell
        custom_objects = { 
                'RecurrentSequential' : RecurrentSequential,
                '_OptionalInputPlaceHolder' : _OptionalInputPlaceHolder, 
                'LSTMDecoderCell' : LSTMDecoderCell 
                }
    else:
        pass    #die

    model = model_from_json(open('%s.json' % model_path).read(), custom_objects=custom_objects)
    model.load_weights('%s.h5' % model_path)
    return model

def preliminary_seq2seq(input_shape, output_shape):
    names = [ 'seq2seq-1x10' ]
    nnets = [ 
            seq2seq(input_shape, output_shape, hidden_dims=50, depth=(2,2), dropout=0.5) ]
    cmp_all(nnets)
    return list(zip(names, nnets))

def seq2seq(input_shape, output_shape, hidden_dims=[10], depth=(1,1), dropout=0.25):
    model = Seq2Seq(input_shape=input_shape, hidden_dim=hidden_dims, output_length=output_shape[0], output_dim=output_shape[1], depth=depth, dropout=dropout)
    return model
