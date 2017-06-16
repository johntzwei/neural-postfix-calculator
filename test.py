import argparse
import pickle

from trees import PostfixTree

import numpy as np

def evaluate(test_exprs, X_test, y_test, model, label_bin, las=False):
    #test
    y_pred = model.predict(X_test)
    y_seq_pred = convert_vectors(y_pred, label_bin.classes_)

    n_total = 0
    pred_total = 0
    true_total = 0
    for true, pred in zip(test_exprs, y_seq_pred):
        pred = pred.strip('$')

        true = PostfixTree.parse_expr(true, postfix=True)
        pred = PostfixTree.parse_expr(pred, postfix=False)

        if pred != None:
            n_total += _num_nodes_overlap(true, pred, las=las)
            pred_total += pred.size()
            true_total += true.size()
        else:
            true_total += true.size()
            #precision is not penalized if cannot be parsed

    #compute
    precision = n_total / ( pred_total if pred_total != 0 else 1 )
    recall = n_total / true_total
    return y_seq_pred, precision, recall, f1(precision, recall)


#turns predicted vectors into actual strings
#takes in 3d tensor (num_test_ex, timestep, features)
def convert_vectors(y_pred, classes):
        #turn the predicted vectors into actual strings
        argmax = np.argmax(y_pred, axis=-1)
        return [ ''.join([ classes[i] for i in v]) for v in argmax ]

#count how many nodes that have the same path to the root
#are in both the true and predicted tree
def _num_nodes_overlap(tree1, tree2, las=False):
    if tree1.leaf == None and tree2.leaf == None:
        subtree_overlap = _num_nodes_overlap(tree1.left, tree2.left, las=las) + _num_nodes_overlap(tree1.right, tree2.right, las=las)
        current = 1 if not las or tree1.op == tree2.op else 0
        return current + subtree_overlap
    elif tree1.leaf != None and tree2.leaf != None:
        return 1 if not las or tree1.leaf == tree2.leaf else 0
    else:
        return 0

def f1(precision, recall):
    try:
        return 2 * (precision * recall) / (precision + recall)
    except:
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('model_architecture', help='the file name of the trained keras model', type=str)
    parser.add_argument('model_weights', help='the file name of the trained keras model', type=str)
    parser.add_argument('lb_bin', help='the pickled label_binarizer', type=str)
    parser.add_argument('--test_ex', help='the tab separated files of testing examples', type=str)
    args = parser.parse_args()

    #get user input
    expr = []
    while True:
        i = str(input())
        if i == '':
            break
        expr.append(i)
    
    from models import seq2seq, cmp_all
    model = seq2seq((None, 11), (22, 11), hidden_dims=30, depth=(1,1), dropout=0.5) 
    model.load_weights(args.model_weights)
    cmp_all([ model ])

    lb_bin = pickle.load(open(args.lb_bin,'rb'))
    end_char = '$'  #should probably argparse this




