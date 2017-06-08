import argparse
import csv
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Masking, Dropout
from keras.layers.recurrent import LSTM

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, mean_squared_error

def decode_csv(fn, max_len=None, end_char='$', lb=None):
    with open(fn) as handle:
        data = list(csv.reader(handle, delimiter='\t'))
        
        #encode to tensor with [ [expr, val]... ]
        #where expr is a one hot 2d tensor
        y = list(map(lambda x: int(x[1]), data))

        if max_len != None:
            X = filter(lambda x: len(x) > max_len, data)
        X = list(map(lambda x: x[0]+end_char, data))

        #one hot encoding into x
        if lb == None:
            lb = LabelBinarizer()
            lb.fit(list(set((''.join(X)))))
        X = list(map(lambda x: lb.transform(list(x)), X))

        padding = np.zeros((lb.classes_.shape[0],))
        X = keras.preprocessing.sequence.pad_sequences(X, value=padding, maxlen=max_len)

        return ((np.array(X), np.array(y)), lb)

#builders of different rnn models to evaluate
def rnn(input_shape, num_layers=2, hidden_dim=50):
    pass

def lstm(input_shape, hidden_layers=0, hidden_dim=50, dropout=0.5):
    model = Sequential()
    model.add(Masking(mask_value=np.zeros(input_shape[1]), input_shape=input_shape))
    for i in range(0,hidden_layers):
        model.add(LSTM(hidden_dim, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.25))
    #last layer for regression
    model.add(LSTM(1, activation=None))
    return model

def bidirectional_lstm(layers, num_nodes):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_ex', help='the tab seperated training file containing an example and label.', type=str, default='data/expB_n200_tgenerateFullTrees_p10_p210_p301_p41_p5None')
    parser.add_argument('--testing_ex', help='the tab seperated testing file containing an example and label.', type=str, default='data/expA_n10_tgenerateFullTrees_p10_p210_p30_p40_p5None')
    parser.add_argument('--predict_file', help='the file to write model predictions and metrics to', type=str, default='./pred')
    parser.add_argument('--epochs', help='the number of epochs to train', type=int, default=1)
    args = parser.parse_args()

    (X_train, y_train), label_bin = decode_csv(args.training_ex) 

    #get models to train
    names = [ '1-lstm', '2-lstm' ]
    input_shape = (X_train.shape[1], X_train.shape[2])
    models = [ lstm(input_shape), lstm(input_shape, hidden_layers=1) ]

    #train
    for name, model in zip(names, models):
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=32)

        #test
        (X_test, y_test), label_bin = decode_csv(args.testing_ex, lb=label_bin, max_len=X_train.shape[1])
        y_pred = model.predict(X_test)
        pred_fn = '%s_%s' % (args.predict_file, name)
        with open(pred_fn,'wt') as pred_handle:
            test_handle = open(args.testing_ex)
            for line,y in zip(test_handle, y_pred):
                pred_handle.write('%s\t%f\n' % (line.strip(), y))
        
            pred_handle.write('accuracy: %f\n' % accuracy_score(y_test, np.rint(y_pred)))
            pred_handle.write('mean squared error: %f\n' % mean_squared_error(y_test, y_pred))

        #write down epoch training
