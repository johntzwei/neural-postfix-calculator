import argparse
import pickle

from trees import PostfixTree

from keras.models import load_model

def predict(expr, model, lb_bin, mem_cells=False):
    pass

def _num_edges_overlap(tree1, tree2, count_nodes=False):
    if tree1.leaf == None and tree2.leaf == None:
        return 2 + _num_edges_overlap(tree1.left, tree2.left) + \
                _num_edges_overlap(tree1.right, tree2.right)
    else:
        return 0

def f1(postfix_true, infix_pred):
    true = PostfixTree.parse_expr(postfix_true, pf_or_in=True)
    pred = PostfixTree.parse_expr(infix_pred, pf_or_in=False)
    
    if pred == None:
        #not parseable
        precision = 0
        recall = 0
        f1 = 0
    else:
        overlap_edges = _num_edges_overlap(true, pred)
        precision = overlap_edges/(pred.size()-1)
        recall = overlap_edges/(true.size()-1)
        f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

if __name__ == '__main__':
    '''
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
    '''
    print(f1('[[[9][3]*][4]+]', '[[[3]*[9]]+[4]]'))
    print(f1('[9]', '[9]'))
