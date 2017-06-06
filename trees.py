import argparse
import random
from enum import Enum

class Operation(Enum):
    PLUS, MINUS, MULT, DIV, NEG = range(5)

class PostfixTree:
    def __init__(self, op=None, left=None, right=None, leaf=None):
        self.op = op
        self.leaf = leaf
        self.left = left
        self.right = right

    def evaluate(self):
        if self.leaf != None:
            return self.leaf
        #TODO refactor this into the operation class
        if self.op == Operation.PLUS:
            return self.left.evaluate() + self.right.evaluate()
        elif self.op == Operation.MULT:
            return self.left.evaluate() * self.right.evaluate()

    def depth(self):
        if self.leaf != None:
            return 0
        else:
            return max(self.left.depth()+1, self.left.depth()+1)

    def __str__(self):
        if self.leaf != None:
            return '[' + str(self.leaf) + ']'
        #todo refactor this into the operation class
        if self.op == Operation.PLUS:
            return '[' + '+' + str(self.left) + str(self.right) + ']'
        elif self.op == Operation.MULT:
            return '[' + '*' + str(self.left) + str(self.right) + ']'

class RandomTree(PostfixTree):
    def __init__(self, lb, ub, p, scale):
        super().__init__()
        lb = int(lb)
        ub = int(ub)
        p = float(p)
        scale = float(scale)

        if random.random() < p:
            #then this is leaf
            self.leaf = random.randint(lb, ub)
        else:
            self.left = RandomTree(lb, ub, p*scale, scale)
            self.right = RandomTree(lb, ub, p*scale, scale)
            self.op = [ Operation.PLUS, Operation.MULT ][int(random.random()*2)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='where to output, options will be appended to filename', type=str)
    parser.add_argument('num_ex', help='number of examples (trees/expressions) to generate', type=int)
    parser.add_argument('tree_type', help='the name of the class which generates the tree', type=str)
    parser.add_argument('--p1', help='the first parameter for the tree generator')
    parser.add_argument('--p2', help='the second parameter for the tree generator')
    parser.add_argument('--p3', help='the third parameter for the tree generator')
    parser.add_argument('--p4', help='the fourth parameter for the tree generator')
    parser.add_argument('--p5', help='the fifth parameter for the tree generator')
    parser.add_argument('--unique', help='if true, only unique examples will be in output', action='store_true')
    args = parser.parse_args()

    if args.unique:
        trees = set()
    else:
        trees = []

    #tree types
    if args.tree_type == 'RandomTree':
        tree = lambda : RandomTree(args.p1, args.p2, args.p3, args.p4)
    elif args.tree_type == 'full':
        tree = lambda : RandomTree(args.p1, args.p2, args.p3, args.p4)
    else:
        exit()

    while len(trees) < args.num_ex:
        #lambda function that returns the new type of tree
        n = tree()

        #add to trees
        #TODO find some way to make this without an if statement
        if args.unique:
            trees.add((str(n), n.evaluate()))
        else:
            trees.append((str(n), n.evaluate()))

    fn = '%s_n%d_t%s_p1%s_p2%s_p3%s_p4%s_p5%s_u%s' % (args.output, args.num_ex, args.tree_type, args.p1, args.p2, args.p3, args.p4, args.p5, args.unique)
    with open(fn, 'w') as handle:
        for expr, val in trees:
            handle.write('%s\t%d\n' % (expr, val))
