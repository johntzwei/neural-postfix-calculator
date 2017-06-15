import argparse
import pickle

from keras.models import load_model

def predict(expr, model, lb_bin, mem_cells=False):
    return model.predict(lb_bin.transform(expr))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='the file name of the trained keras model', type=str)
    parser.add_argument('lb_bin', help='the pickled label_binarizer', type=str)
    parser.add_argument('--testing_ex', help='the tab separated files of testing examples', type=str)
    parser.add_argument('--mem_cells', help='output the top level lstm memory cells at each stage', type=bool, default=True)
    args = parser.parse_args()
    
    lb_bin = pickle.load(open(args.lb_bin,'rb'))
    model = load_model(args.model)
    end_char = '$'  #should probably argparse this
    
    if args.testing_ex != None:
        print('eval: ', endchar='$')
        input()
        while True:
            expr = str(input())
            if expr == 'exit':
                break
            print(predict(expr, model, lb_bin, mem_cells=args.mem_cells))
    else:
        pass
