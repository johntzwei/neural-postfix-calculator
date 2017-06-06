import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding, LSTM
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
import csv
import numpy as np

def decode_csv(fn, max_len, end_char='$'):
    with open(fn) as handle:
        data = list(csv.reader(handle, delimiter='\t'))
        
        #encode to tensor with [ [expr, val]... ]
        #where expr is a one hot 2d tensor
        X = list(map(lambda x: x[0]+end_char, data))
        y = list(map(lambda x: int(x[1]), data))

        #one hot encoding into x
        lb = LabelBinarizer()
        lb.fit(list(set(''.join(X))))
        X = list(map(lambda x: lb.transform(list(x)), X))

        #truncate
        #X = list(map(lambda x: (x+np.array([[0]*lb.classes_.shape[0]])*max_len)[:max_len], X))

        #pad sequences for now
        padding = np.zeros((lb.classes_.shape[0],))
        X = pad_sequences(X, value=padding)

        return np.array(X), np.array(y)

def one_hot():
    pass

if __name__ == '__main__':
    #argparse
    TRAIN_FILE = 'data/train'
    TEST_FILE = 'data/test'
    TRAIN = True

    #hyper parameters
    max_len = 50
    max_features = 200

    if TRAIN:
        print('train option, will train and save model')
        print('loading training data...')
        X_train, y_train = decode_csv(TRAIN_FILE, max_len)
        print('loaded training data.')
        print(X_train.shape)
        print('first training example: ')
        print(X_train[0])

        model = Sequential()
        model.add(LSTM(10, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True, recurrent_dropout=0.25))
        model.add(LSTM(10, recurrent_dropout=0.25))
        model.add(Dense(1, activation='relu'))

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'accuracy'], epochs=3)
        print('model compiled!')

        print('training model...')
        model.fit(X_train, y_train)
        print('model trained!')

        #model.save('model.m5')
        #print('model saved to local.')
    else:
        #load model
        pass

    #test
    X_test, y_test = decode_csv(TEST_FILE, max_len)
    print('loaded testing data.')

    #model.predict(X_test[:3])
