import os
import csv
import argparse
import pickle
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

def decode_seq2seq_csv(fn, end_char='$', lb=None, max_len=None):
    with open(fn) as handle:
        #encode to tensor with [ [expr1, expr2]... ]
        #where expr1 and 2 is a 2d tensor with [ [ one-hot char ]... ]
        data = list(csv.reader(handle, delimiter='\t'))
        
        y = [ row[1] + '$' for row in data ]
        X = [ row[0] + '$' for row in data ]

        #one hot encoding into x
        if lb == None:
            lb = LabelBinarizer()
            lb.fit(list(set((''.join(X)))))
        X = [ lb.transform(list(expr)) for expr in X ]
        y = [ lb.transform(list(expr)) for expr in y ]

        padding = lb.transform(['$'])[0]
        #this takes the max len of the training set.
        #if test set has an ex that is longer we are toast
        X = pad_sequences(X, padding='post', value=padding, maxlen=max_len)
        y = pad_sequences(y, padding='post', value=padding, maxlen=max_len)

        return ((np.array(X), np.array(y)), lb)

def write_csv(exp_dir, fn, l):
    fn = os.path.join(exp_dir, fn)
    with open(fn, 'wt') as handle:
        for row in l:
            line = ''
            for column in row:
                line += '%s\t' % column
            handle.write('%s\n' % line.strip())

# takes in an arbitrary string and an infix expression
# and gives some kind of accuracy measure - precision recall on trees?
def some_kind_of_acc_measure(y_true, y_label):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_ex', help='the tab seperated training file containing an example and label.', type=str)
    parser.add_argument('testing_ex', help='the tab seperated testing file containing an example and label.', type=str)
    parser.add_argument('series', help='series of models to get from a function within models.py', type=str, default='preliminaries')
    parser.add_argument('exp_dir', help='the directory to write model predictions and metrics to', type=str)
    parser.add_argument('--epochs', help='the number of epochs to train', type=int, default=100)
    parser.add_argument('--batch_size', help='the number of epochs to train', type=int, default=32)
    args = parser.parse_args()

    decode_csv = decode_seq2seq_csv

    (X_train, y_train), label_bin = decode_csv(args.training_ex) 
    (X_test, y_test), _ = decode_csv(args.testing_ex, lb=label_bin, max_len=X_train.shape[1])
    pickle.dump(label_bin, open(os.path.join(args.exp_dir, 'label_binarizer'), 'wb'))

    #get models to train
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = (y_train.shape[1], y_train.shape[2])
    keras_models = getattr(models, args.series)(input_shape, output_shape)

    #evaluate
    accuracies = []
    for name, model in keras_models:
        #train
        history = model.fit(X_train, y_train, epochs=args.epochs)

        #save
        model.save(os.path.join(args.exp_dir, '%s.h5') % name)

        #test
        y_pred = model.predict(X_test)

        #turn the predicted vectors into actual strings
        argmax = np.argmax(y_pred, axis=-1)
        y_seq_pred = [ ''.join([ label_bin.classes_[i] for i in v]) for v in argmax ]

        scores = [ 
                [ 'accuracy', '' ] ]
        testing_ex = map(lambda x: x.strip(), open(args.testing_ex))
        write_csv(args.exp_dir, 'pred_%s' % name, list(zip(testing_ex, y_seq_pred)) + scores)
        
        #logging
        write_csv(args.exp_dir, 'losses_%s' % name, enumerate(history.history['loss'], 1))
        accuracies.append((name, scores[0][1]))

    #output
    write_csv(args.exp_dir, 'accuracy_report', accuracies)
