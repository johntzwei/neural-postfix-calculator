from enum import Enum
import random

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
        if self.op == Operation.PLUS:
            return self.left.evaluate() + self.right.evaluate()
        elif self.op == Operation.MULTIPLY:
            return self.left.evaluate() * self.right.evaluate()

    #TODO factor this into its own class
    def generate_random_tree(self, lb, ub, p, scale):
        if random.random() < p:
            #then this is leaf
            self.leaf = random.randint(lb, ub)
        else:
            self.left = PostfixTree()
            self.left.generate_random_tree(lb, ub, p*scale, scale)
            self.right = PostfixTree()
            self.right.generate_random_tree(lb, ub, p*scale, scale)

            self.op = [ Operation.PLUS, Operation.MULTIPLY ][int(random.random()*2)]

    def depth(self):
        if self.leaf != None:
            return 0
        else:
            return max(self.left.depth()+1, self.left.depth()+1)
            
    def __str__(self):
        if self.leaf != None:
            return '[' + str(self.leaf) + ']'
        if self.op == Operation.PLUS:
            return '[' + '+' + str(self.left) + str(self.right) + ']'
        elif self.op == Operation.MULTIPLY:
            return '[' + '*' + str(self.left) + str(self.right) + ']'

if __name__ == '__main__':
    for i in range(0, 50000):
        d = PostfixTree()
        d.generate_random_tree(1, 1, 0.25,2)
        print('%s\t%s' % (d, d.evaluate()))

        #TODO add argparse
