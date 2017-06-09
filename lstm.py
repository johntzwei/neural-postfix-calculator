import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Masking, Dropout, Dense
from keras.layers.recurrent import LSTM, SimpleRNN

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
def rnn(input_shape, hidden_layers=0, hidden_dim=50, dropout=0.5):
    model = Sequential()
    model.add(Masking(mask_value=np.zeros(input_shape[1]), input_shape=input_shape))
    for i in range(0,hidden_layers):
        model.add(SimpleRNN(hidden_dim, return_sequences=True))
        model.add(Dropout(dropout))
    #last layer for regression
    model.add(SimpleRNN(1, activation='relu'))
    return model

def lstm(input_shape, hidden_layers=0, hidden_dim=50, dropout=0.5):
    model = Sequential()
    model.add(Masking(mask_value=np.zeros(input_shape[1]), input_shape=input_shape))
    for i in range(0,hidden_layers):
        model.add(LSTM(hidden_dim, return_sequences=True))
        model.add(Dropout(dropout))
    #last layer for regression
    model.add(LSTM(1, activation='relu'))
    return model

def mlp_lstm(input_shape, hidden_layers=1, hidden_dim=50, dropout=0.5): 
    model = Sequential()
    model.add(Masking(mask_value=np.zeros(input_shape[1]), input_shape=input_shape))
    for i in range(1,hidden_layers):
        model.add(LSTM(hidden_dim, return_sequences=True))
        model.add(Dropout(dropout))
    #last layer for regression
    model.add(LSTM(hidden_dim))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='relu'))
    return model

def save_training_loss_graph(fn, losses):
    fig, ax = plt.subplots()
    plt.title('Training Loss over Time')
    plt.ylabel('Loss (Mean Squared Error)')
    plt.xlabel('Epoch')

    epochs = len(list(losses.values())[0])
    for name, h in losses.items():
        ax.plot(range(1, epochs+1), h, 'o', label=name)
    legend = ax.legend(loc='upper right', shadow=True)

    #write to file
    plt.savefig(fn)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_ex', help='the tab seperated training file containing an example and label.', type=str)
    parser.add_argument('testing_ex', help='the tab seperated testing file containing an example and label.', type=str)
    parser.add_argument('exp_dir', help='the directory to write model predictions and metrics to', type=str)
    parser.add_argument('--epochs', help='the number of epochs to train', type=int, default=300)
    #TODO more argparse options
    args = parser.parse_args()

    (X_train, y_train), label_bin = decode_csv(args.training_ex) 

    #get models to train
    input_shape = (X_train.shape[1], X_train.shape[2])
    names = [ '1-rnn', '2-rnn', '1-lstm', '2-lstm', '1-mlp-lstm' ]
    models = [ rnn(input_shape), rnn(input_shape, hidden_layers=1),  lstm(input_shape), lstm(input_shape, hidden_layers=1), mlp_lstm(input_shape) ]

    #train
    losses = {}
    accuracies = {}
    for name, model in zip(names, models):
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        history = model.fit(X_train, y_train, epochs=args.epochs)

        #test
        (X_test, y_test), label_bin = decode_csv(args.testing_ex, lb=label_bin, max_len=X_train.shape[1])
        y_pred = model.predict(X_test)
        pred_fn = '%s_%s' % (args.predict_file, name)
        with open(pred_fn,'wt') as pred_handle:
            test_handle = open(args.testing_ex)
            for line,y in zip(test_handle, y_pred):
                pred_handle.write('%s\t%f\n' % (line.strip(), y))
        
            accuracy = accuracy_score(y_test, np.rint(y_pred))
            pred_handle.write('accuracy: %f\n' % accuracy)
            pred_handle.write('mean squared error: %f\n' % mean_squared_error(y_test, y_pred))

        #write down epoch training
        losses[name] = history.history['loss']
        accuracies[name] = accuracy

    #graph epoch training
    for name, history in losses.items():
        fn = 'losses_%s' % name
        with open(fn, 'wt') as handle:
            for i, mse in enumerate(history):
                handle.write('%d\t%f\n' % (i, mse))
    save_training_loss_graph('%s_training-loss-graph.png' % args.predict_file, losses)

    with open('accuracy_report', 'wt') as handle:
        for name, acc in accuracies.items():
            handle.write('%s\t%f\n' % (name, acc))
