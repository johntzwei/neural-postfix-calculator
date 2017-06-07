import argparse
import random

class Operation():
    PLUS, MINUS, MULT, DIV = range(4)
    _f = [ 
            lambda x, y: x + y,
            lambda x, y: x - y,
            lambda x, y: x * y,
            lambda x, y: x / y
        ]

    _s = [
            '+',
            '-',
            '*',
            '/'
        ]

    def evaluate(op, o1, o2):
        return Operation._f[op](o1, o2)

    def to_str(op):
        return Operation._s[op]

    def get_random_op(s):
        ops = list(str(s))
        return int(ops[int(random.random()*len(ops))])
        
class PostfixTree:
    def __init__(self, op=None, left=None, right=None, leaf=None):
        self.op = op
        self.leaf = leaf
        self.left = left
        self.right = right

    def evaluate(self):
        if self.leaf != None:
            return self.leaf
        return Operation.evaluate(self.op, self.left.evaluate(), self.right.evaluate())
        
    def depth(self):
        if self.leaf != None:
            return 0
        else:
            return max(self.left.depth()+1, self.left.depth()+1)

    def __str__(self):
        if self.leaf != None:
            return '[' + str(self.leaf) + ']'
        return '[' + Operation.to_str(self.op) + str(self.left) + str(self.right) + ']'

#random trees - returns one tree per construction
class RandomTree(PostfixTree):
    def __init__(self, lb, ub, p, scale):
        super().__init__()
        lb, ub, p, scale = int(lb), int(ub), float(p), float(scale)

        if random.random() < p:
            #then this is leaf
            self.leaf = random.randint(lb, ub)
        else:
            self.left = RandomTree(lb, ub, p*scale, scale)
            self.right = RandomTree(lb, ub, p*scale, scale)
            
            self.op = Operation.PLUS

class FullTree(PostfixTree):
    def __init__(self, lb, ub, depth, ops):
        super().__init__()
        lb, ub, depth, ops = int(lb), int(ub), int(depth), str(ops)
        
        if depth == 0:
            self.leaf = random.randint(lb, ub)
        else:
            self.left = FullTree(lb, ub, depth-1, ops)
            self.right = FullTree(lb, ub, depth-1, ops)
            self.op = Operation.get_random_op(ops)

#generates a list of all trees
def generateFullTrees(lb, ub, ops, depth):
    trees = []
    if depth <= 0:
        for i in range(lb, ub):
            t = PostfixTree(leaf=i)
            trees.append(t)
    else:
        subtrees = generate(lb, ub, ops, depth-1)
        for op in ops:
            for i in range(lb,ub):
                for x in subtrees:
                    for y in subtrees:
                        t = PostfixTree(left=x, right=y, op=int(op))
                        trees.append(t)
    return trees

def generateAllTrees(lb, ub, ops, depth):
    trees = []
    if depth <= 0:
        for i in range(lb, ub):
            t = PostfixTree(leaf=i)
            trees.append(t)
    else:
        subtrees = generate(lb, ub, ops, depth-1)
        for op in ops:
            for i in range(lb,ub):
                for x in subtrees:
                    for y in subtrees:
                        t = PostfixTree(left=x, right=y, op=int(op))
                        trees.append(t)
    return trees

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
        tree = lambda : [ RandomTree(args.p1, args.p2, args.p3, args.p4, args.p5) ]
    elif args.tree_type == 'FullTree':
        tree = lambda : [ FullTree(args.p1, args.p2, args.p3, args.p4) ]
    elif args.tree_type == 'AllTreesGen':
        tree = lambda : generate(int(args.p1), int(args.p2), str(args.p3), int(args.p4))
    else:
        print('does not have the tree option specified')
        exit()

    #lambda function that returns the new type of tree
    l = tree()

    #add to trees
    #TODO find some way to make this without an if statement
    for n in l:
        if args.unique:
            trees.add((str(n), n.evaluate()))
        else:
            trees.append((str(n), n.evaluate()))

    fn = '%s_n%d_t%s_p1%s_p2%s_p3%s_p4%s_p5%s_u%s' % (args.output, args.num_ex, args.tree_type, args.p1, args.p2, args.p3, args.p4, args.p5, args.unique)
    with open(fn, 'w') as handle:
        for expr, val in trees:
            handle.write('%s\t%d\n' % (expr, val))
