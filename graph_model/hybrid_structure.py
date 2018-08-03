"""
This structure contains both directed and undirected edges
"""
import numpy as np

class Hybrid_structure:
    """
    store network structure
    """
    def __init__(self, n=None, f=None):
        """
        n is the number of nodes.
        The structure is stored in an adjacent matrix G.
        """
        #adjacent matrix
        if n is not None:
            self._CG  = np.array([[0] * n] * n)
            self._CRG = np.array([[0] * n] * f)
        else:
            self._CG  = None
            self._CRG = None
        self._n = n
        self._f = f
        #for iteration
        self._i = 0

    def set_nf(self, n, f):
        """
        set node numbers
        """
        self._n = n
        self._CG  = np.array([[0] * n] * n)
        self._CRG = np.array([[0] * n] * f)

    def set_CRG(self, fsm):
        """
        set the fsm
        """
        self._CRG[:] = fsm[:]

    def remove_redundency(self):
        """
        remove all redundant edges from faults to variables
        """
        for i in range(self._n):
            af = self.ancestor_faults(i)
            df = self.direct_faults(i)
            if not af.issubset(df):
                return False
            else:
                for f in af:
                    self.remove_fodedge(f, i)
        return True

    def add_dedge(self, i, j):
        """
        add directed edge from i to j
        i != j and there should be no edge between i and j

        If i==j or there is an edge between i and j, return false
        Else, add the directed edge and return True
        """
        if i == j:
            return False
        if not (self._CG[i, j] == 0 and self._CG[j, i] == 0): #there is an edge between i and j now!
            return False
        self._CG[i, j] = 1
        return True

    def remove_dedge(self, i, j):
        """
        remove directed edge from i to j
        i != j and there should be no edge between i and j

        If i==j or there is no edge from i to j, return False
        Else, remove edge from i to j and return True
        """
        if i==j:
            return False
        if not (self._CG[i, j] == 1 and self._CG[j, i] == 0): #there is no directed edge from i to j 
            return False
        self._CG[i, j] = 0
        return True

    def reverse_dedge(self, i, j):
        """
        reverse the edge from i to j
        i != j and there should be an edge from i to j 
        """
        if i == j:
            return False
        if not (self._CG[i, j] == 1 and self._CG[j, i] == 0):
            return False
        self._CG[i, j] = 0
        self._CG[j, i] = 1
        return True

    def add_foedge(self, f, i):
        """
        add a directed edge from fault i to variable j
        """
        if self._CRG[f, i] != 0:
            return False
        self._CRG[f, i] = 1
        return True

    def remove_fodedge(self, f, i):
        """
        remove the directed edge from fault i to variable j
        """
        if self._CRG[f, i] != 1:
            return False
        self._CRG[f, i] = 0
        return True

    def is_acyclic(self):
        """
        check if the directed part of the graph is acyclic
        """
        dedges = (self._CG == 1)
        n      = self._n
        stack = set()
        for i in range(n):
            parents = dedges[:, i]
            if not parents.any():
                stack.add(i)
        while len(stack) != 0:
            v    = stack.pop()
            for i in range(n):
                if dedges[v, i]:
                    dedges[v, i] = False
                    parents = dedges[:, i]
                    if not parents.any():
                        stack.add(i)
        return not dedges.any()

    def clone(self):
        """
        clone it
        """
        copy      = Hybrid_structure()
        copy._CG  = self._CG.copy()
        copy._CRG = self._CRG.copy()
        copy._n   = self._n
        copy._f   = self._f
        copy._i   = self._i
        return copy

    def clone_directed(self):
        """
        clone the directed part of the graph
        """
        copy      = Hybrid_structure(self._n, self._f)
        copy._CG  = self._CG.copy()
        copy._n   = self._n
        copy._f   = self._f
        copy._i   = self._i
        return copy

    def ancestors(self, i):
        """
        return all the ancestors of i in the directed part
        """
        n         = self._n
        ancestors = set()
        stack     = set()
        stack.add(i)
        diredge = (self._CG == 1)
        while len(stack) != 0:
            i = stack.pop()
            parents = diredge[:, i]
            for j in range(n):
                if parents[j] and j not in ancestors:
                    ancestors.add(j)
                    stack.add(j)
        return ancestors

    def ancestor_faults(self, i):
        """
        return all the faults come from i_th ancestors
        """
        ansestors = self.ancestors(i)
        faults    = set()
        for a in ansestors:
            fa     = self.direct_faults(a)
            faults = faults | fa
        return faults

    def direct_faults(self, i):
        """
        return the faults that directly influence variable i
        """
        faults = set()
        fa     = self._CRG[:, i]
        for f in range(self._f):
            if fa[f] == 1:
                faults.add(f)
        return faults

    def print(self):
        """
        print itself
        """
        print(self._CG)
        print(self._CRG)

    def __eq__(self, other):
        """
        check if it is equal to other
        """
        return (self._CG == other._CG).all() and (self._CRG == other._CRG).all()

    def __hash__(self):
        """
        hash function
        """
        return hash(tuple([tuple(i) for i in self._CG] + [tuple(i) for i in self._CRG]))

    def __iter__(self):
        """
        enumerate all edeges
        """
        return self

    def __next__(self):
        """
        work with __iter__
        """
        if self._i == self._n:
            self._i = 0
            raise StopIteration
        parents = list(self._CG[:, self._i])
        parents = [i for i, v in enumerate(parents) if v==1]
        parents = sorted(parents)
        neigbors = list(self._CRG[:, self._i])
        neigbors = [i for i, v in enumerate(neigbors) if v==1]
        neigbors = sorted(neigbors)
        kid      = self._i
        self._i = self._i + 1
        return tuple(parents), tuple(neigbors), kid
