"""
learning Bayesian model
"""
import numpy as np
from math import log2
from scipy.linalg import solve
from graph_model.utilities import vector2number
from graph_model.utilities import number2vector

#small number to avoid numeric problems
alpha = 1e-20

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

    def add_edge(self, i, j):
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

    def is_DAG_DFS(self, i, color):
        """
        deep first search
        return if it is acycle in this search
        True: acycle
        False:cycle
        """
        color[i] = 1
        for j in range(self.n):
            if self.struct[i, j] != 0:
                if color[j] == 1:
                    return False
                elif color[j] == -1:
                    continue
                else:
                    if not self.is_DAG_DFS(j, color):
                        return False
        color[i] == -1
        return True

    def is_acycle(self):
        """
        check if the structure is acycle
        """
        color = [0]*self.n
        for i in range(self.n):
            if not self.is_DAG_DFS(i, color):
                return False
        return True

    def clone(self):
        """
        clone it
        """
        copy = Bayesian_structure(self.n)
        copy.struct = self.struct.copy()
        return copy

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
        parents = [i for i, v in enumerate(parents) if v==1]
        kid = [self.i]
        fml = parents + kid
        self.i = self.i + 1
        return tuple(fml)

class Bayesian_learning:
    """
    learning Bayesian model from data
    """
    def __init__(self, n):
        #nodes number
        self.n = n
        #queue
        self.queue = {}             #We call it search queue although it is a dict, queue = {G:cost,...].
        #priori knowledge
        self.known_edge = set()     #priori knowledge about system structure, {(i, j)...}. Edges in this list exist.
        self.no_edge = set()        #priori knowledge about system structure, {(i, j)...}. Edges in this list never exist.
        #batch
        self.batch  = None          #data for the current batch, batch × (label + residuals + features).
        self.decay  = 0             #regular factor
        #cache
        # self.JPT_cache = {}         #cache for JPT (Joint Probability Distribution) of a Joint Random Variables: {JRV:[JPT, N]}.
        #                             #JRV is a tuple: nodes are put in ascending order.
        #                             #JPT is a np.array() whose size is 2^n.
        #                             #N is the batch number where the batchs contribute the JPT.
        #                             #Some values in it will be updated in each iteration but the diction will NOT be cleared.
        self.GGM_cache = {}         #cache for GGM (Guassian Graph Model).{FML:[beta, sigma, N]}
                                    #FML:(p1,p2,...pn, kid), nodes are put in ascending order.
                                    #beta:[beta0, beta1,...,beta_n]
                                    #sigma: real
                                    #N: real (weight), N is the batch number where the batchs contribute the JPT.
                                    #Some values in it will be updated in each iteration but the diction will NOT be cleared.
        # self.batch_JPT_cache = set()#batch JPT cache. Should BE cleared and updated in each iteration.
        #                             #NOTICE: This set just store the JRV that computed from batch
        #                             #the real JPT is merged into self.JPT_cache.
        self.batch_GGM_cache = set()#batch GGM cache. Should BE cleared and updated in each iteration.
                                    #NOTICE: This set just store the GGM that computed from batch
                                    #the real GGM is merged into self.GGM_cache.
        self.e_cache = {}           #Expectation cache.
                                    #TODO
        self.l_cost_cache = {}      #Likelihood cost cache for each family. Should BE cleared and updated in each iteration.
                                    #{FML:l_cost}.
                                    #because GGM may change because of new batch.
        self.r_cost_cache = {}      #Regular cost cache for each family. Should NOT be cleared.
                                    #{FML:r_cost}.
        self.graph_l_cost_cache = {}#Likelihood cost cache for a graph. Should BE cleard and updated in each iteration.
                                    #{G:cost,...}.
        self.graph_r_cost_cache = {}#Regular cost cache for a graph. Should NOT be cleared.
                                    #{G:cost,...}.
    #For queue
    def init_queue(self):
        """
        init the search queue
        """
        graph = Bayesian_structure(self.n)
        for edge in self.known_edge:
            graph.add_edge(edge[0], edge[1])
        cost = self.cost(graph)
        self.queue[graph] = cost

    def add_struct2queue(self, struct, cost):
        """
        add a struct and its cost into the queue.
        """
        self.queue[struct] = cost

    def best_candidate(self):
        """
        get the best candidate
        """
        queue = self.queue
        best = min(zip(queue.values(),queue.keys()))
        return best[1]

    #For priori knowledge
    def add_known_edge(self, i, j):
        """
        add a known edge (i==>j)
        """
        self.known_edge.add((i, j))

    def add_no_edge(self, i, j):
        """
        add no edge means both edges (i==>j) and (j==>i) don't exist
        """
        self.no_edge.add((i, j))
        self.no_edge.add((j, i))

    #For self.batch
    def set_batch(self, batch):
        """
        set the current batch
        """
        self.batch = batch
        n = len(batch)
        self.decay = log2(n)/(2*n)

    #For self.JPT_cache
    # def JPT_cache_has(self, JRV):
    #     """
    #     check if self.JPT_cache has JRV
    #     """
    #     return JRV in self.JPT_cache
    def GGM_cache_has(self, FML):
        return FML in self.GGM_cache

    # def JPT_from_cache(self, JRV):
    #     """
    #     obtain the JPT of JRV from the self.JPT_cache
    #     """
    #     return self.JPT_cache[JRV][0]
    def beta_sigma_weight_from_cache(self, FML):
        bdw = self.GGM_cache[FML]
        return bdw[0], bdw[1], bdw[2]

    # def JPT_weight_in_cache(self, JRV):
    #     """
    #     return JPT weight
    #     """
    #     return self.JPT_cache[JRV][1]

    #for self.batch_JPT_cache
    # def batch_JPT_cache_has(self, JRV):
    #     """
    #     return if self.batch_JPT_cache has JRV
    #     """
    #     return JRV in self.batch_JPT_cache

    def batch_GGM_cache_has(self, FML):
        return FML in self.batch_GGM_cache

    # def JPT_from_batch(self, JRV):
    #     """
    #     compute the JPT of JVR from the current batch
    #     """
    #     #index of parents and kid
    #     data = self.batch[:, JRV]
    #     n = len(JRV)
    #     JPT = [0] * (2**n)
    #     for i in range(2**n):
    #         vec = number2vector(i, n)
    #         comp = (vec == data)
    #         result = [x.all() for x in comp]
    #         # +1 to avoid sum(result)=0 cause numeric problems
    #         JPT[i] = sum(result) + 1
    #     for i in range(0, 2**n, 2):
    #         num = JPT[i] + JPT[i+1]
    #         JPT[i] = JPT[i] / num
    #         JPT[i+1] = 1 - JPT[i]
    #     return np.array(JPT)
    def GGM_from_batch(self, FML):


    #For cost
    #for likelihood cost
    def l_cost_cache_has(self, fml):
        """
        return if self.l_cost_cache has fml
        """
        return fml in self.l_cost_cache

    def l_cost_from_cache(self, fml):
        """
        return cached cost
        """
        return self.l_cost_cache[fml]

    def graph_l_cost_cache_has(self, graph):
        """
        return if self.graph_l_cost_cache has graph
        """
        return graph in self.graph_l_cost_cache

    def graph_l_cost_from_cache(self, graph):
        """
        return cached cost
        """
        return self.graph_l_cost_cache[graph]

    def l_cost(self, graph):
        """
        compute the likelihood cost
        """
        if self.graph_l_cost_cache_has(graph):
            return self.l_cost_from_cache(graph)
        #Now we know that the likelihood cache is not computed for the current batch
        cost = 0
        for fml in graph:
            if self.l_cost_cache_has(fml):
                cost = cost + self.l_cost_from_cache(fml)
            else:#Now we know the fml is not cached
                cost = cost + self.fml_l_cost(fml)
        return cost

    def fml_l_cost(self, fml):
        """
        compute l_cost for family fml
        """
        if len(fml)<=1:
            return 0
        #parents
        par = fml[:-1]
        #kid
        kid = fml[-1:]
        #ordered fml
        ordered_fml = tuple(sorted(list(fml)))
        #par_JPT, kid_JPT, fml_JPT
        par_JPT = self.get_JPT(par)
        kid_JPT = self.get_JPT(kid)
        fml_JPT = self.get_JPT(ordered_fml)
        cost = self.information_gain_cost(par, par_JPT, kid, kid_JPT, ordered_fml, fml_JPT)
        return cost
    
    def information_gain_cost(self, par, par_JPT, kid, kid_JPT, fml, fml_JPT):
        """
        compute information gain
        par, kid and fml are tuples and variables in them are in ascending order
        """
        kid_index = fml.index(kid[0])
        n = len(fml)
        cost = 0
        for i in range(2**n):
            #family vector
            fml_vec = number2vector(i, n)
            #kid vector
            kid_vec = [fml_vec[kid_index]]
            #par vector
            par_vec = fml_vec[:]
            del par_vec[kid_index]
            #find index in JPT for par and kid
            i_p = vector2number(par_vec)
            i_k = vector2number(kid_vec)
            cost = cost + fml_JPT[i] * log2((par_JPT(i_p) * kid_JPT[i_k] + alpha)/ (fml_JPT[i] + alpha))
        return cost

    #get JPT
    def get_JPT(self, JVR):
        """
        Get a JPT from cache or batch. Update or add automatically.
        """
        #please make sure JVR is listed in ascending order
        if self.batch_JPT_cache_has(JVR):
            JPT = self.JPT_from_cache(JVR)
        else:#We must update par_JPT from data
            tmp_JPT = self.JPT_from_batch(JVR)
            if self.JPT_cache_has(JVR):#should update
                cached_JPT = self.JPT_from_cache(JVR)
                cached_weight = self.JPT_weight_in_cache(JVR)
                JPT = (cached_weight/(cached_weight + 1))*cached_JPT\
                     +(1/(cached_weight + 1))*tmp_JPT
                #update par_JPT
                self.JPT_cache[JVR] = [JPT, cached_weight + 1]
            else:#should add
                JPT = tmp_JPT
                self.JPT_cache[JVR] = [JPT, 1]
        return JPT

    #for regular cost
    def r_cost_cache_has(self, fml):
        """
        return if self.r_cost_cache has fml
        """
        return fml in self.r_cost_cache

    def r_cost_from_cache(self, fml):
        """
        return cached regular cost
        """
        return self.r_cost_cache[fml]

    def graph_r_cost_cache_has(self, graph):
        """
        return if self.graph_r_cost_cache has graph
        """
        return graph in self.graph_r_cost_cache

    def graph_r_cost_from_cache(self, graph):
        """
        return cached graph regular cost
        """
        return self.graph_r_cost_cache[graph]

    def r_cost(self, graph):
        """
        compute the regular cost for graph
        """
        if self.graph_r_cost_cache_has(graph):
            return self.graph_r_cost_from_cache(graph)
        #Now we know the regular cost is not computed for the graph
        cost = 0
        for fml in graph:
            if self.r_cost_cache_has(fml):
                cost = cost + self.r_cost_from_cache(fml)
            else:#Now we know the fml is not cached
                n = len(fml)
                fml_cost = 2**(n-1)
                self.r_cost_cache[fml] = fml_cost #add it to cache
                cost = cost + fml_cost
        return cost

    #Cost
    def cost(self, graph):
        """
        compute the cost of a structure
        """
        cost = self.l_cost(graph) + (self.decay * self.r_cost(graph))
        return cost
        
    #if the graph is valid
    def priori_obeyed_by(self, graph):
        """
        check if the priori knowledge is obeyed by the graph
        """
        #existing edges
        for edge in self.known_edge:
            if not graph.get_edge(edge[0], edge[1]):
                return False
        #no edges
        for edge in self.no_edge:
            if graph.get_edge(edge[0], edge[1]):
                return False
        return True

    def valid_graph(self, graph):
        """
        DAG + obey priori knowledge
        """
        if not graph.is_acycle():
            return False
        if not self.priori_obeyed_by(graph):
            return False
        return True

    def step(self):
        """
        step forward
        """
        #clear some cache
        self.batch_JPT_cache.clear()
        self.l_cost_cache.clear()
        self.graph_l_cost_cache.clear()
        #pick out the current best candidate
        best = self.best_candidate()
        #change randomly
        for i in range(self.n):
            for j in range(i+1, self.n):
                if best.get_edge(i, j) or best.get_edge(j, i):#there are edges
                    #remove edge
                    rem_best = best.clone()
                    rem_best.remove_edge(i, j)
                    if self.valid_graph(rem_best):
                        rem_cost = self.cost(rem_best)
                        self.queue[rem_best] = rem_cost
                    #reverse edge
                    rev_best = best.clone()
                    rev_best.reverse_edge(i, j)
                    if self.valid_graph(rev_best):
                        rev_cost = self.cost(rev_best)
                        self.queue[rev_best] = rev_cost
                else:#there is no edge
                    #add i==>j 
                    addij_best = best.clone()
                    addij_best.add_edge(i, j)
                    if self.valid_graph(addij_best):
                        addij_cost = self.cost(addij_best)
                        self.queue[addij_best] = addij_cost
                    #add j==>i
                    addji_best = best.clone()
                    addji_best.add_edge(j, i)
                    if self.valid_graph(addji_best):
                        addji_cost = self.cost(addji_best)
                        self.queue[addji_best] = addji_cost
