import os
import csv
import argparse
import pickle
import numpy as np
import sys

import models
from test import evaluate, convert_vectors

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, mean_squared_error

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_ex', help='the tab seperated training file containing an example and label.', type=str)
    parser.add_argument('testing_ex', help='the tab seperated testing file containing an example and label.', type=str)
    parser.add_argument('series', help='series of models to get from a function within models.py', type=str)
    parser.add_argument('exp_dir', help='the directory to write model predictions and metrics to', type=str)
    parser.add_argument('--model', help='incrementally train/test the loaded seq2seq model', type=str)
    parser.add_argument('--epochs', help='the number of epochs to train', type=int, default=100)
    parser.add_argument('--batch_size', help='the number of epochs to train', type=int, default=32)
    args = parser.parse_args()

    if not args.model:
        (X_train, y_train), label_bin = decode_seq2seq_csv(args.training_ex, max_len=100) 
        (X_test, y_test), _ = decode_seq2seq_csv(args.testing_ex, lb=label_bin, max_len=X_train.shape[1])
        pickle.dump(label_bin, open(os.path.join(args.exp_dir, 'label_binarizer'), 'wb'))

        input_shape = (X_train.shape[1], X_train.shape[2])
        output_shape = (y_train.shape[1], y_train.shape[2])
        keras_models = getattr(models, args.series)(input_shape, output_shape)
    else:
        #load model
        model = models.load_model('Seq2Seq', args.model) 
        keras_models = [ model ]
        models.cmp_all(keras_models)
        keras_models = [ ( os.path.basename(args.model).split('.')[0] , keras_models[0]) ]

        label_bin = pickle.load(open(os.path.join(args.exp_dir, 'label_binarizer'), 'rb'))
        if args.training_ex != 'None':
            (X_train, y_train), label_bin = decode_seq2seq_csv(args.training_ex, lb=label_bin, max_len=100) 
            (X_test, y_test), _ = decode_seq2seq_csv(args.testing_ex, lb=label_bin, max_len=X_train.shape[1])
        else:
            (X_test, y_test), _ = decode_seq2seq_csv(args.testing_ex, lb=label_bin, max_len=model.input_shape[1])

    #early stopping when mse is at min
    callbacks = [ models.acc_early_stopping ]
        
    #evaluate
    accuracies = []
    for name, model in keras_models:
        #train
        if args.training_ex != 'None':
            history = model.fit([ X_train, y_train ], y_train, epochs=args.epochs, callbacks=callbacks)
            write_csv(args.exp_dir, 'losses_%s' % name, enumerate(history.history['loss'], 1))

        #save
        model.save_weights(os.path.join(args.exp_dir, '%s.h5') % name)
        open(os.path.join(args.exp_dir, '%s.json') % name, 'wt').write(model.to_json())

        #evaluate
        testing_ex = [ line.strip() for line in open(args.testing_ex) ]
        test_exprs = [ line.split('\t')[0] for line in testing_ex ]
        y_seq_pred, precision, recall, f1 = evaluate(test_exprs, X_test, y_test, model, label_bin)
        
        #write predictions for test file
        scores = [ 
                [ 'unlabeled precision', precision ],
                [ 'unlabeled recall', recall ],
                [ 'unlabeled f1', f1 ] ]
        write_csv(args.exp_dir, 'pred_%s' % name, list(zip(testing_ex, y_seq_pred)) + scores)
