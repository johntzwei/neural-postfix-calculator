import argparse
import pickle

from keras.models import load_model

def predict(expr, model, lb_bin, mem_cells=False):
    return model.predict(lb_bin.transform(list(expr)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='the file name of the trained keras model', type=str)
    parser.add_argument('lb_bin', help='the pickled label_binarizer', type=str)
    parser.add_argument('test_ex', help='the tab separated files of testing examples', type=str)
    args = parser.parse_args()
    
    lb_bin = pickle.load(open(args.lb_bin,'rb'))
    model = load_model(args.model)
    end_char = '$'  #should probably argparse this
    
    test_ex = open(args.test_ex, 'rt')
    for line in test_ex
        print('%s\t%s', (line, predict(expr, model, lb_bin)))
