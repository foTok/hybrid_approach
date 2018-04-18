"""
learning Bayesian model
"""
import numpy as np
from utilities import vector2number
from utilities import number2vector

class Bayesian_structure:
    """
    store Bayesian structure
    """
    def __init__(self, n):
        """
        n is the number of nodes
        """
        #adjacent matrix
        self.n = n
        self.struct = np.array([[0] * n] * n)

    def set_edge(self, i, j):
        """
        set connection from node i to node j
        """
        assert i != j
        self.struct[i, j] = 1
        self.struct[j, i] = 0

    def remove_edge(self, i, j):
        """
        remove edge between node i and node j
        """
        assert i != j
        self.struct[i, j] = 0
        self.struct[j, i] = 0

    def reverse_edge(self, i, j):
        """
        reverse edge between node i and node j
        when there is no connection, nothing happens
        """
        assert i != j
        self.struct[i, j], self.struct[j,i] = self.struct[j, i], self.struct[i,j]

    def get_edge(self, i, j):
        """
        get the connection from node i and node j
        """
        assert i != j
        return self.struct[i, j]

    def __eq__(self, other):
        """
        check if it is equal to other
        """
        return (self.struct == other.struct).all()

    def __hash__(self):
        """
        hash function
        """
        return hash(tuple([tuple(i) for i in self.struct]))

    def __iter__(self):
        """
        enumerate all edeges
        """
        self.i = 0
        return self

    def __next__(self):
        """
        work with __iter__
        """
        if self.i == self.n:
            raise StopIteration
        parents = list(self.struct[self.i, :])
        kid = [self.i]
        fml = parents + kid
        self.i = self.i + 1
        return tuple(fml)

class Bayesian_learning:
    """
    learning Bayesian model from data
    """
    def __init__(self, gama=0.1):
        self.gama = gama
        self.queue = []             #search queue, queue = [(structure, cost)...]
        self.queue_set = set()      #store the existing structure in self.queue to chekc conveniently
        self.rule = []              #priori knowledge about system structure, [(i, j , d)...]
        self.local_cache = {}       #{local:cost}
        self.batch  = None          #data for the current batch, batch × (label + residuals + features)
        self.batch_cache = set()    #store computed cpt in this batch

    def set_batch(self, batch):
        """
        set the current batch
        """
        self.batch = batch

    def cpt_from_local_cache(self, fml):
        """
        obtain the Conditional Probability Table (CPT) of family fml from the self.local_cache
        """
        if self.local_cache_has(fml):
            return self.local_cache[fml]
        return None

    def cpt_from_data(self, fml):
        """
        compute the Conditional Probability Table (CPT) of family fml from the current batch
        fml:[1,2,3,...,kid], the last one the kid, others are the parents.
        """
        #index of parents and kid
        data = self.batch[:, fml]
        n = len(fml)
        cpt = [0] * (2**n)
        for i in range(2**n):
            vec = number2vector(i, n)
            comp = (vec == data)
            result = [x.all() for x in comp]
            # +1 to avoid sum(result)=0 cause numeric problems
            cpt[i] = sum(result) + 1
        for i in range(0, 2**n, 2):
            num = cpt[i] + cpt[i+1]
            cpt[i] = cpt[i] / num
            cpt[i+1] = 1 - cpt[i]
        return np.array(cpt)

    def queue_has(self, struct):
        """
        check if self.queue has the structure
        """
        return struct in self.queue_set

    def batch_cache_has(self, fml):
        """
        check if family fml is in self.batch_cpt
        """
        return fml in self.batch_cache

    def local_cache_has(self, fml):
        """
        check if self.local_cache has fml
        """
        return fml in self.local_cache

    def add_fml2batch_cache(self, fml):
        """
        add family fml into self.batch_cache
        """
        self.batch_cache.add(fml)

    def add_struct2queue(self, struct, cost):
        """
        add a struct and its cost into the queue.
        and store it in the struct set
        """
        self.queue.append((struct, cost))
        self.queue_set.add(struct)
   
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

    def cost(self, struct):
        """
        compute the cost of a structure
        """
        cost = 0
        for fml in struct:
            if not self.batch_cache_has(fml):
                cpt = self.cpt_from_data(fml)
                pass
            else:
                pass