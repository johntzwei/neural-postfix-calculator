import argparse
from random import randint

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
            return max(self.left.depth()+1, self.right.depth()+1)

    def __str__(self):
        if self.leaf != None:
            return '[' + str(self.leaf) + ']'
        return '[' + str(self.left) + str(self.right) + Operation.to_str(self.op) + ']'

    def to_infix_str(self):
        if self.leaf != None:
            return '[' + str(self.leaf) + ']'
        return '[' + self.left.to_infix_str() + Operation.to_str(self.op) + self.right.to_infix_str() + ']'

    def size(self):
        if self.leaf != None:
            return 1
        else:
            return 1 + self.left.size() + self.right.size()

    #parsing from postfix/infix expressions
    #second return item is whether it was correctly parsed
    def compute_brackets(expr):
        stack = []
        brackets = {}
        for i, char in enumerate(expr):
            if char == '[':
                stack.append(i)
            elif char == ']':
                if len(stack) == 0:
                    return None
                close = stack.pop()
                brackets[i] = close
                brackets[close] = i
            else:
                pass
        return brackets

    #if returns None, that means expressions cannot be parsed
    def parse_expr(expr, index=0, brackets=None, postfix=True):
        if brackets == None:
            brackets = PostfixTree.compute_brackets(expr)
            if brackets == None:
                return None
        try:
            leaf = int(expr[1:-1])
            return PostfixTree(leaf=leaf)
        #if not atomic
        except:
            try:
                if postfix:
                    #postfix
                    left_begin = 1
                    left_end = brackets[index+1] - index + 1

                    right_begin = left_end
                    right_end = -2

                    if expr[-2] not in Operation._s:
                        return None
                    op = Operation._s.index(expr[-2])

                else:
                    #infix
                    left_begin = 1
                    left_end = brackets[index+1] - index + 1

                    if expr[left_end] not in Operation._s:
                        return None

                    op = Operation._s.index(expr[left_end])
                    right_begin = left_end + 1
                    right_end = -1
                    pass
                left = PostfixTree.parse_expr(expr[left_begin:left_end], index=index+left_begin, brackets=brackets, postfix=postfix)
                right = PostfixTree.parse_expr(expr[right_begin:right_end], index=index+right_begin, brackets=brackets, postfix=postfix)
                if left != None and right != None:
                    return PostfixTree(op=op, left=left, right=right)
                else:
                    return None
            except:
                #defensive programming!
                return None

import functools
@functools.lru_cache(maxsize=None)
def generateFullTrees(lb, ub, ops, depth):
    trees = []
    if depth <= 0:
        for i in range(lb, ub+1):
            trees.append(PostfixTree(leaf=i))
    else:
        subtrees = generateFullTrees(lb, ub, ops, depth-1)
        for op in ops:
            for x in subtrees:
                for y in subtrees:
                    trees.append(PostfixTree(left=x, right=y, op=int(op)))
    return trees

#this function generates all trees that are of n depth
@functools.lru_cache(maxsize=None)
def _generateAllTrees(lb, ub, ops, depth):
    trees_lt_max_depth = []
    trees_of_max_depth = []
    if depth <= 0:
        for i in range(lb, ub+1):
            t = PostfixTree(leaf=i)
            trees_of_max_depth.append(t)
    else:
        eq_depth, lt_depth = _generateAllTrees(lb, ub, ops, depth-1)
        for op in ops:
            for x in eq_depth:
                for y in lt_depth:
                    trees_of_max_depth.append(PostfixTree(left=x, right=y, op=int(op)))
                    trees_of_max_depth.append(PostfixTree(left=y, right=x, op=int(op)))
                for y in eq_depth:
                    trees_of_max_depth.append(PostfixTree(left=x, right=y, op=int(op)))
        trees_lt_max_depth = eq_depth + lt_depth
    return (trees_of_max_depth, trees_lt_max_depth)

def generateAllTrees(lb, ub, ops, depth):
    eq_depth, lt_depth = _generateAllTrees(lb, ub, ops, depth)
    return eq_depth + lt_depth

_get_op = lambda x: int(x[randint(0,len(x)-1)])
def _generateRandomTree(lb, ub, ops, num_nodes):
    if num_nodes == 1:
        return PostfixTree(leaf=randint(lb, ub))
    else:
        left = randint(1, num_nodes-1)
        return PostfixTree(
                op = _get_op(ops),
                left = _generateRandomTree(lb, ub, ops, left),
                right = _generateRandomTree(lb, ub, ops, num_nodes-left))

#this is a uniformly random generation of trees with at most n number of nodes
def generateRandomTrees(lb, ub, ops, num_nodes, num_samples):
    for i in range(0, num_samples):
        yield _generateRandomTree(lb, ub, list(ops), randint(1, num_nodes))

def generateRandomTreesFixedNodes(lb, ub, ops, num_nodes, num_samples):
    for i in range(0, num_samples):
        yield _generateRandomTree(lb, ub, list(ops), num_nodes)

def generateRandomTreesTrimmed(lb, ub, ops, num_nodes, num_samples, max_depth=5):
    for i in range(0, num_samples):
        while True:
            T = _generateRandomTree(lb, ub, list(ops), num_nodes)
            if T.depth() <= max_depth:
                break
        yield T

#special trees
@functools.lru_cache(maxsize=None)
def _generateLeaningTrees(lb, ub, ops, depth, lean='right'):
    trees = []
    if depth <= 0:
        for i in range(lb, ub+1):
            t = PostfixTree(leaf=i)
            trees.append(t)
    else:
        subtrees = _generateLeaningTrees(lb, ub, ops, depth-1, lean=lean)
        for op in ops:
            for x in subtrees:
                left = PostfixTree(leaf=randint(lb, ub))

                #swap left and right
                if lean == 'left':
                    s = x
                    x = left
                    left = s

                t = PostfixTree(
                        left=left,
                        right=x,
                        op=int(op))
                trees.append(t)
    return trees

def generateRightLeaningTrees(lb, ub, ops, depth):
    return sum(map(lambda depth: _generateLeaningTrees(lb, ub, ops, depth), range(0, depth+1)), [])

def generateLeftLeaningTrees(lb, ub, ops, depth):
    return sum(map(lambda depth: _generateLeaningTrees(lb, ub, ops, depth, lean='left'), range(0, depth+1)), [])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tree_type', help='the name of the class which generates the tree', type=str)
    parser.add_argument('--p1', help='the first parameter for the tree generator')
    parser.add_argument('--p2', help='the second parameter for the tree generator')
    parser.add_argument('--p3', help='the third parameter for the tree generator')
    parser.add_argument('--p4', help='the fourth parameter for the tree generator')
    parser.add_argument('--p5', help='the fifth parameter for the tree generator')
    parser.add_argument('--postfix', help='don\'t output postfix notation', action='store_const', const=False, default=True)
    parser.add_argument('--infix', help='output infix notation', action='store_const', const=True, default=False)
    parser.add_argument('--depth', help='output the depth of the expression', action='store_const', const=True, default=False)
    args = parser.parse_args()

    #tree types
    if args.tree_type == 'generateFullTrees':
        generator = lambda : generateFullTrees(int(args.p1), int(args.p2), str(args.p3), int(args.p4))
    elif args.tree_type == 'generateAllTrees':
        generator = lambda : generateAllTrees(int(args.p1), int(args.p2), str(args.p3), int(args.p4))
    elif args.tree_type == 'generateRandomTrees':
        generator = lambda : generateRandomTrees(int(args.p1), int(args.p2), str(args.p3), int(args.p4), int(args.p5))
    elif args.tree_type == 'generateRandomTreesFixedNodes':
        generator = lambda : generateRandomTreesFixedNodes(int(args.p1), int(args.p2), str(args.p3), int(args.p4), int(args.p5))
    elif args.tree_type == 'generateRightLeaningTrees':
        generator = lambda : generateRightLeaningTrees(int(args.p1), int(args.p2), str(args.p3), int(args.p4))
    elif args.tree_type == 'generateLeftLeaningTrees':
        generator = lambda : generateLeftLeaningTrees(int(args.p1), int(args.p2), str(args.p3), int(args.p4))
    else:
        print('error')
        exit()

    for tree in generator():
        line = []
        if args.postfix:
            line.append(str(tree))
        if args.infix:
            line.append(tree.to_infix_str())
        if args.depth:
            line.append(str(tree.depth()))
        line.append(str(tree.evaluate()))
        print('\t'.join(line))
