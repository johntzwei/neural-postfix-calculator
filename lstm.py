import argparse
import csv
import numpy as np

from sklearn.preprocessing import LabelBinarizer

def decode_csv(fn, max_len, split_char='', end_char='$', pad_seq=True):
    with open(fn) as handle:
        data = list(csv.reader(handle, delimiter='\t'))
        
        #encode to tensor with [ [expr, val]... ]
        #where expr is a one hot 2d tensor
        y = list(map(lambda x: int(x[1]), data))

        X = filter(lambda x: len(x) > max_len, data)
        X = list(map(lambda x: x[0]+end_char, data))

        #one hot encoding into x
        lb = LabelBinarizer()
        lb.fit(list(set((''.join(X)).split(split_char))))
        X = list(map(lambda x: lb.transform(list(x)), X))

        if pad_seq:
            padding = np.zeros((lb.classes_.shape[0],))
            X = K.preprocessing.pad_sequences(X, value=padding)

        return np.array(X), np.array(y)

#builders of different rnn models to evaluate
def rnn(input_shape, num_layers=2, hidden_dim=50):
    pass

def lstm(input_shape, hidden_layers=1, hidden_dim=50, dropout=0):
    model = K.Sequential()
    model.add(Masking(mask_value=np.zeros(input_shape[1]), input_shape=input_shape))
    #TODO figure out the dropout scheme to use
    for i in hidden_layers:
        model.add(LSTM(hidden_dim, return_sequences=True))
    #last layer for regression
    model.add(LSTM(1))
    return model

def bidirectional_lstm(layers, num_nodes):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_ex', help='the tab seperated training file containing an example and label.', type=str)
    parser.add_argument('testing_ex', help='the tab seperated testing file containing an example and label.', type=str)
    parser.add_argument('--epochs', help='the number of epochs to train', type=int, default=50)
    args = parser.parse_args()

    import keras as K
    X_train, y_train = decode_csv(args.training_ex) 

    #get models to train
    names = [ '1-LSTM' ]
    models = [ lstm((X.shape[1],X.shape[2])) ]
    #names = [ 'Simple RNN', 'Vanilla LSTM', 'Deep LSTM', 'Bidirectional LSTM' ]
    #models = [ rnn(), vanilla_lstm(), lstm(2, 50), bidirectional_lstm() ]

    #train
    for name, model in zip(names, models):
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=args.epochs, batch_size=32)

