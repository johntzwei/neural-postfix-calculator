import os
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt

import .models

from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, mean_squared_error

def decode_csv(fn, end_char='$', lb=None):
    with open(fn) as handle:
        #encode to tensor with [ [expr, val]... ]
        #where expr is a 2d tensor with [ [ one-hot char ]... ]
        data = list(csv.reader(handle, delimiter='\t'))
        
        y = [ int(row[1]) for row in data ]
        X = [ row[0] + '$' for row in data ]

        #one hot encoding into x
        if lb == None:
            lb = LabelBinarizer(sparse_output=True)
            lb.fit(list(set((''.join(X)))))
        X = [ lb.transform(list(expr)) for expr in X ]

        padding = np.zeros((lb.classes_.shape[0],))
        X = keras.preprocessing.sequence.pad_sequences(X, value=padding)

        return ((np.array(X), np.array(y)), lb)

def save_training_loss_graph(fn, losses):
        fig, ax = plt.subplots()
        plt.title('Training Loss over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Mean Squared Error)')

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
    parser.add_argument('--series', help='series of models to get from a function within models.py', type=str, default='preliminaries')
    parser.add_argument('--epochs', help='the number of epochs to train', type=int, default=100)
    parser.add_argument('--batch_size', help='the number of epochs to train', type=int, default=32)
    args = parser.parse_args()

    (X_train, y_train), label_bin = decode_csv(args.training_ex) 
    (X_test, y_test), _ = decode_csv(args.testing_ex, lb=label_bin, max_len=X_train.shape[1])

    #get models to train
    input_shape = (X_train.shape[1], X_train.shape[2])
    keras_models = getattr(models, args.series)()

    #evaluate
    losses = {}
    accuracies = {}
    for name, model in keras_models:
        #train
        history = model.fit(X_train, y_train, epochs=args.epochs)

        #test
        y_pred = model.predict(X_test)
        pred_fn = os.path.join(args.exp_dir, 'predict_%s' % name)
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

    #TODO refactor
    #graph epoch training
    for name, history in losses.items():
        fn = os.path.join(args.exp_dir, 'losses_%s' % name)
        with open(fn, 'wt') as handle:
            for i, mse in enumerate(history):
                handle.write('%d\t%f\n' % (i, mse))
    save_training_loss_graph(os.path.join(args.exp_dir, 'training-loss-graph.png'), losses)

    fn = os.path.join(args.exp_dir, 'accuracy_report')
    with open(fn, 'wt') as handle:
        for name, acc in accuracies.items():
            handle.write('%s\t%f\n' % (name, acc))
