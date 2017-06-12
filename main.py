import os
import csv
import argparse
import numpy as np

import models

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, mean_squared_error

def decode_csv(fn, end_char='$', lb=None, max_len=None):
    with open(fn) as handle:
        #encode to tensor with [ [expr, val]... ]
        #where expr is a 2d tensor with [ [ one-hot char ]... ]
        data = list(csv.reader(handle, delimiter='\t'))
        
        y = [ int(row[1]) for row in data ]
        X = [ row[0] + '$' for row in data ]

        #one hot encoding into x
        if lb == None:
            lb = LabelBinarizer()
            lb.fit(list(set((''.join(X)))))
        X = [ lb.transform(list(expr)) for expr in X ]

        padding = np.zeros((lb.classes_.shape[0],))
        #this takes the max len of the training set.
        #if test set has an ex that is longer we are toast
        X = pad_sequences(X, value=padding, maxlen=max_len)

        return ((np.array(X), np.array(y)), lb)

def write_csv(exp_dir, fn, l):
    fn = os.path.join(exp_dir, fn)
    with open(fn, 'wt') as handle:
        for row in l:
            line = ''
            for column in row:
                line += '%s\t' % column
            handle.write('%s\n' % line.strip())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_ex', help='the tab seperated training file containing an example and label.', type=str)
    parser.add_argument('testing_ex', help='the tab seperated testing file containing an example and label.', type=str)
    parser.add_argument('series', help='series of models to get from a function within models.py', type=str, default='preliminaries')
    parser.add_argument('exp_dir', help='the directory to write model predictions and metrics to', type=str)
    parser.add_argument('--epochs', help='the number of epochs to train', type=int, default=100)
    parser.add_argument('--batch_size', help='the number of epochs to train', type=int, default=32)
    args = parser.parse_args()

    (X_train, y_train), label_bin = decode_csv(args.training_ex) 
    (X_test, y_test), _ = decode_csv(args.testing_ex, lb=label_bin, max_len=X_train.shape[1])

    #get models to train
    input_shape = (X_train.shape[1], X_train.shape[2])
    keras_models = getattr(models, args.series)(input_shape)

    #evaluate
    losses = {}
    accuracies = {}
    for name, model in keras_models:
        #train
        history = model.fit(X_train, y_train, epochs=args.epochs)

        #test
        y_pred = model.predict(X_test)
        scores = [ 
                [ 'accuracy', accuracy_score(y_test, np.rint(y_pred)) ],
                [ 'mean squared error',  mean_squared_error(y_test, y_pred) ] ]
        testing_ex = map(lambda x: x.strip(), open(args.testing_ex))
        y_pred = map(lambda x: x[0], y_pred)
        write_csv(args.exp_dir, 'pred_%s' % name, list(zip(testing_ex, y_pred)) + scores)
        
        #logging
        losses[name] = history.history['loss']
        accuracies[name] = scores[0][1] 

    #output
    write_csv(args.exp_dir, 'accuracy_report', accuracies.items())
    for name, history in losses.items():
        write_csv(args.exp_dir, 'losses_%s' % name, enumerate(history, 1))
