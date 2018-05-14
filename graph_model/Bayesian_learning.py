"""
learning Bayesian model
"""
import numpy as np
from math import log
from math import exp
from scipy.linalg import solve
from graph_model.utilities import vector2number
from graph_model.utilities import number2vector
from copy import deepcopy

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
        self.struct = np.array([[0] * n] * n)
        #for iteration
        self.n = n
        self.i = 0

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
        return self

    def __next__(self):
        """
        work with __iter__
        """
        if self.i == self.n:
            self.i = 0
            raise StopIteration
        parents = list(self.struct[:, self.i])
        parents = [i for i, v in enumerate(parents) if v==1]
        kid = [self.i]
        fml = parents + kid
        self.i = self.i + 1
        return tuple(fml)

class Bayesian_learning:
    """
    learning Bayesian model from data
    """
    def __init__(self, n, alpha=0.2):
        #nodes number
        self.n                      = n
        #queue
        #We call it search queue although it is a dict, queue = {G:cost,...].
        self.queue                  = {}
        #priori knowledge
        #priori knowledge about system structure, {(i, j)...}. Edges in this list exist.
        self.known_edge             = set()
        #priori knowledge about system structure, {(i, j)...}. Edges in this list never exist.     
        self.no_edge                = set()
        #batch
        #np.array(). Data for the current batch, batch × (label + residuals + features).
        self.batch                  = None
        #regular factor      
        self.decay                  = 0
        #cost weight
        self.alpha                  = alpha
        #best candidate
        self.best                   = None
        #cache
        #cache for GGM (Guassian Graph Model).{FML:[beta, var, N]}.
        #FML:(p1,p2,...pn, kid), nodes are put in ascending order.
        #beta:[beta0, beta1,...,beta_n].
        #var: real (var = sigma**2).
        #N: real (weight), N is the batch number where the batchs contribute the JPT.
        #Some values in it will be updated in each iteration but the diction will NOT be cleared.
        self.GGM_cache              = {}
        #batch GGM cache. Should BE cleared and updated in each iteration.
        #NOTICE: This set just store the GGM that computed from batch.
        #the real GGM is merged into self.GGM_cache.
        #It should BE cleared before each iteration.
        self.batch_GGM_cache        = set()                                                             #remember to clear
        #Expectation cache.
        #E_cache: {x/(x1, x2):E}
        #x, x1, x2: int
        #E: real
        #this cache just store E for this batch. Should BE cleared and updated in each iteration.
        self.E_cache                = {}                                                                #remember to clear
        #Likelihood cost cache for each family.
        #{FML:l_cost}.
        #Should be  updated in each iteration
        #because GGM may change because of new batch.                           
        self.fml_l_cost_cache       = {}
        #update flag
        #Should be cleared before each iteration.
        self.fml_l_cost_update_flag = set()                                                             #remember to clear
        #Regular cost cache for each family. Should NOT be cleared.
        #{FML:r_cost}.
        self.fml_r_cost_cache       = {}
        #Likelihood cost cache for a graph. Should BE cleard and updated in each iteration.
        #{G:cost,...}.
        self.graph_l_cost_cache     = {}                                                                #remember to clear
        #Regular cost cache for a graph. Should NOT be cleared.
        #{G:cost,...}.
        self.graph_r_cost_cache     = {}

    #For queue
    def init_queue(self):
        """
        init the search queue
        """
        graph = Bayesian_structure(self.n)
        for edge in self.known_edge:
            graph.add_edge(edge[0], edge[1])
        self.queue[graph] = 10000
        self.best = graph                                                                       #this one can never be the right one

    def add_struct2queue(self, struct, cost):
        """
        add a struct and its cost into the queue.
        """
        self.queue[struct] = cost

    def best_candidate(self):
        """
        get the best candidate
        """
        # epsilon = 1e-3
        queue = self.queue
        # while True:
        #     graph, cost0 = min(queue.items(), key=lambda x:x[1])
        #     cost1 = self.cost(graph)
        #     if abs(cost0 - cost1) < epsilon:
        #         cost = cost1
        #         break
        #     else:
        #         self.queue[graph] = cost1
        graph, cost = min(queue.items(), key=lambda x:x[1])
        return graph, cost

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
        self.decay = log(n)/(2*n)

    #For GGM
    def get_beta_var(self, FML):
        """
        get beta var and weight
        """
        if FML in self.batch_GGM_cache:
            bvw = self.GGM_cache[FML]
        else:
            beta1, var1 = self.GGM_from_batch(FML)
            if FML in self.GGM_cache:
                bvw0 = self.GGM_cache[FML]
                beta0, var0, n = bvw0[0], bvw0[1], bvw0[2]
                w0 = n / (1 + n)
                w1 = 1 / (1 + n)
                beta = w0 * beta0 + w1 * beta1
                # var  = w0**2 * var0 + w1**2 * var1
                var  = w0 * var0 + w1 * var1
                n = n + 1
                bvw = [beta, var, n]
            else:
                bvw = [beta1, var1, 1]
            #save in cache
            self.GGM_cache[FML] = bvw
            self.batch_GGM_cache.add(FML)
        #return beta, var and weight(n)
        return bvw[0], bvw[1]

    def GGM_from_batch(self, FML):
        """
        compute GGM for FML in this batch
        !!!Please make sure parents are listed in incresing order
        """
        parents = FML[:-1]  #a tuple
        X       = FML[-1]   #an int
        Kp = len(parents)
        #Compute beta
        #b         = beta * A
        #E[X]      = β0     +β1E[U1]     +. . .+βkE[Uk].
        #E[X · Ui] = β0E[Ui]+β1E[U1 · Ui]+. . .+βkE[Uk · Ui].
        #A = np.matrix(np.zeros((Kp+1, Kp+1)))
        A = np.array(np.zeros((Kp+1, Kp+1)))
        b = np.zeros(Kp+1)
        #for the first equation
        A[0, 0] = 1
        b[0] = deepcopy(self.get_E(X))
        for k in range(Kp):
            U_k = parents[k]
            A[0, k+1] = deepcopy(self.get_E(U_k))
        
        #for the rest equations
        for i in range(Kp):# for row i+1
            U_i = parents[i]
            A[i+1, 0] = deepcopy(self.get_E(U_i))
            b[i+1]    = deepcopy(self.get_E((X, U_i)))
            for k in range(Kp):
                U_k = parents[k]
                A[i+1, k+1] = deepcopy(self.get_E((U_k, U_i)))
        beta = np.linalg.solve(A, b)
        #Compute var
        #Cov[X;X] = E[X · X]−E[X] · E[X]
        #Cov[X;Ui] = E[X · Ui]−E[X] · E[Ui]
        #var = Cov[X;X]−SIGMA{βiβjCov[Ui;Uj]}
        var = self.get_E((X, X)) - self.get_E(X)**2
        #print("var=",var)
        for i in range(Kp):
            U_i = parents[i]
            for j in range(Kp):
                U_j = parents[j]
                var = var - beta[i+1] * beta[j+1] * (self.get_E((U_i, U_j)) - self.get_E(U_i) * self.get_E(U_j))
        var = var + 1.0e-5
        assert(var >= 0)

        return beta, var

    def E_from_batch(self, x):
        """
        get Expection from batch
        x: int or (x1, x2) where x1 and x2 are int
        x1 < x2 is ensured by self.get_E()
        """
        if isinstance(x, tuple):
            x1 = self.batch[:, x[0]]
            x2 = self.batch[:, x[1]]
            E = np.mean(x1 * x2)
        else:
            E = np.mean(self.batch[:, x])
        #save in cache
        self.E_cache[x] = E
        return E

    def get_E(self, x):
        """
        get Expection from batch or cache
        x: int or (x1, x2) where x1 and x2 are int
        """
        #insure x1 < x2 for tuple
        if isinstance(x, tuple):
            x1 = min(x)
            x2 = max(x)
            x = (x1, x2)
        if x in self.E_cache:
            E = self.E_cache[x]
        else:
            E = self.E_from_batch(x)
        return E

    #For cost
    #for likelihood cost
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
            return self.graph_l_cost_from_cache(graph)
        #Now we know that the likelihood cache is not computed for the current batch
        cost = 0
        for fml in graph:
            if fml in self.fml_l_cost_update_flag:                      #must be updated and in self.fml_l_cost_cache
                cost_u = self.fml_l_cost_cache[fml]
            elif fml in self.fml_l_cost_cache:                          #in self.fml_l_cost_cache but not update in this iteration
                cost0 = self.fml_l_cost_cache[fml]
                cost1 = self.fml_l_cost(fml)
                cost_u = cost0 + self.alpha * (cost1 - cost0)
                self.fml_l_cost_cache[fml] = cost_u
                self.fml_l_cost_update_flag.add(fml)
            else:#Now we know the fml is the first time to compute
                cost_u = self.fml_l_cost(fml)
                self.fml_l_cost_cache[fml] = cost_u
                self.fml_l_cost_update_flag.add(fml)
            cost = cost + cost_u
        return cost

    def fml_l_cost(self, fml):
        """
        compute l_cost for family fml
        """
        #for debug
        # if fml == (0,):
        #     print("Stop here")
        beta, var = self.get_beta_var(fml)
        parents = fml[:-1]
        kid = fml[-1]
        U = self.batch[:, parents]
        X = self.batch[:, kid]
        predict = np.sum(U*beta[1:], axis=1)
        predict = predict + beta[0]
        res = (predict - X)**2
        rela_res = res / (2*var)
        cost = np.mean(rela_res)
        #save it in self.fml_l_cost_cache
        self.fml_l_cost_cache[fml] = cost
        return cost

    #for regular cost
    def fml_r_cost_cache_has(self, fml):
        """
        return if self.r_cost_cache has fml
        """
        return fml in self.fml_r_cost_cache

    def fml_r_cost_from_cache(self, fml):
        """
        return cached regular cost
        """
        return self.fml_r_cost_cache[fml]

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
            if self.fml_r_cost_cache_has(fml):
                cost = cost + self.fml_r_cost_from_cache(fml)
            else:#Now we know the fml is not cached
                n = len(fml)
                fml_cost = exp(n-1)
                self.fml_r_cost_cache[fml] = fml_cost #add it to cache
                cost = cost + fml_cost
        return cost

    #Cost
    def cost(self, graph):
        """
        compute the cost of a structure
        """
        l_cost = self.l_cost(graph)
        r_cost = self.r_cost(graph)
        cost = l_cost + self.decay * r_cost
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

    def clear_cache(self):
        """
        clear some caches
        """
        self.batch_GGM_cache.clear()
        #debug
        # print("E_cache=", self.E_cache)
        self.E_cache.clear()
        self.fml_l_cost_update_flag.clear()
        self.graph_l_cost_cache.clear()

    def step(self):
        """
        step forward
        """
        #clear cache
        self.clear_cache()
        best = self.best
        #change randomly
        for i in range(self.n):
            for j in range(i+1, self.n):
                if best.get_edge(i, j) or best.get_edge(j, i):#there are edges
                    #remove edge
                    #debug
                    #print("remove {}--{}", i, j)
                    rem_best = best.clone()
                    rem_best.remove_edge(i, j)
                    if self.valid_graph(rem_best):
                        rem_cost = self.cost(rem_best)
                        self.queue[rem_best] = rem_cost
                    #reverse edge
                    #debug
                    #print("reverse {}--{}", i, j)
                    rev_best = best.clone()
                    rev_best.reverse_edge(i, j)
                    if self.valid_graph(rev_best):
                        rev_cost = self.cost(rev_best)
                        self.queue[rev_best] = rev_cost
                else:#there is no edge
                    #add i==>j
                    #debug
                    #print("add {}-->{}", i, j)
                    addij_best = best.clone()
                    addij_best.add_edge(i, j)
                    if self.valid_graph(addij_best):
                        addij_cost = self.cost(addij_best)
                        self.queue[addij_best] = addij_cost
                    #add j==>i
                    #debug
                    #print("add {}-->{}", j, i)
                    addji_best = best.clone()
                    addji_best.add_edge(j, i)
                    if self.valid_graph(addji_best):
                        addji_cost = self.cost(addji_best)
                        self.queue[addji_best] = addji_cost
        #pick out the current best candidate
        best, cost = self.best_candidate()
        #debug
        print(cost)
