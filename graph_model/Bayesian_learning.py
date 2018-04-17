"""
learning Bayesian model
"""
class Bayesian_structure:
    """
    store Bayesian structure
    """
    def __init__(self, n):
        """
        n is the number of nodes
        """
        #use a low triangle matrix to store structure
        #i\j 0 1 2 3
        #0   * * * *
        #1   0 * * *
        #2   1 2 * *
        #3   3 4 5 *
        self.n = n
        self.struct = [0] * n * (n-1) / 2

    def pos(self, i, j):
        """
        get the position (i, j) in self.struct
        """
        assert 0<=j < i < n
        p = i * (i - 1) / 2 + j
        return p

    def set(self, i, j, d):
        """
        set connection between node i and node j as d
        d = 0: no connection
        d = 1: i-->j
        d = -1: i<--j
        """
        assert i != j
        #make sure i > j
        if i < j:
            i, j = j, i
            d = -d
        p = self.pos(i, j)
        self.struct[p] = d

    def get(self, i, j):
        """
        get the connection between node i and node j
        """
        assert i != j
        (i1, j1, d) = (i, j, 1) if i > j else (j, i, -1)
        p = self.pos(i1, j1)
        d1 = self.struct[p]
        return d*d1

    def __eq__(self, other):
        """
        check if it is equal to other
        """
        return self.struct == other.struct

    def __hash__(self):
        """
        hash function
        """
        return hash(tuple(self.struct))


class Bayesian_learning:
    """
    learning Bayesian model from data
    """
    def __init__(self, gama=0.1):
        self.gama = gama
        self.queue = []             #search queue, queue = [(structure, cost)...]
        self.rule = []              #priori knowledge about system structure, [(i, j , d)...]
        self.local_cache = {}       #{local:cost}


    def queue_has(self, struct):
        """
        check if self.queue has the structure
        """
        pass

    
    def init_queue(self):
        """
        init the search queue
        """
        pass


    def sort_queue(self):
        """
        sort the queue. structure with minimal cost in the first
        """
        self.queue = sorted(self.queue, key=lambda item:item[1])

    def 