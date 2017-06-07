import argparse

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
        
class PostfixTree:
    def __init__(self, op=None, left=None, right=None, leaf=None):
        self.leaf, self.left, self.right, self.op = leaf, left, right, op

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

def generateFullTrees(lb, ub, ops, depth):
    trees = []
    if depth <= 0:
        for i in range(lb, ub):
            t = PostfixTree(leaf=i)
            trees.append(t)
    else:
        subtrees = generateFullTrees(lb, ub, ops, depth-1)
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
        subtrees = generateAllTrees(lb, ub, ops, depth-1)
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
    args = parser.parse_args()

    #tree types
    if args.tree_type == 'generateFullTrees':
        generator = lambda : generateFullTrees(int(args.p1), int(args.p2), str(args.p3), int(args.p4))
    elif args.tree_type == 'generateAllTrees':
        generator = lambda : generateAllTrees(int(args.p1), int(args.p2), str(args.p3), int(args.p4))
    else:
        print('error')
        exit()

    trees = []
    #generators return lists not generators!!
    trees.extend(generator())
    trees = list(map(lambda x: (str(x), x.evaluate()), trees))
    trees = trees[:num_ex] 

    fn = '%s_n%d_t%s_p1%s_p2%s_p3%s_p4%s_p5%s' % (args.output, len(trees), args.tree_type, args.p1, args.p2, args.p3, args.p4, args.p5)
    with open(fn, 'w') as handle:
        for expr, val in trees:
            handle.write('%s\t%d\n' % (expr, val))
