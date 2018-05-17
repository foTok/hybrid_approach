"""
learning Bayesian model
"""
import numpy as np
from math import log
from math import exp
from scipy.linalg import solve
from graph_model.utilities import vector2number
from graph_model.utilities import number2vector
from graph_model.utilities import Guassian_cost
from graph_component import Bayesian_structure
from graph_model.Bayesian_network import Bayesian_network


#small number to avoid numeric problems
alpha = 1e-20
class Bayesian_learning:
    """
    learning Bayesian model from data
    """
    def __init__(self, n, alpha=0.2):
        #nodes number
        self.n                      = n
        #init flag
        self.init_flag              = False
        #queue
        #We call it search queue although it is a dict, queue = {G:cost,...].
        self.queue                  = {}
        #priori knowledge
        self.priori_knowledge       = None
        #batch
        #np.array(). Data for the current batch, batch × (label + residuals + features).
        self.batch                  = None
        #regular factor      
        self.decay                  = 0
        # #cost weight
        # self.alpha                  = alpha
        #cache
        #cache for GGM (Guassian Graph Model).{FML:[beta, var, N]}.
        #FML:(p1,p2,...pn, kid), nodes are put in ascending order.
        #beta:[beta0, beta1,...,beta_n].
        #var: [var0, var1,...,var_n].
        #N: real (weight), N is the batch number where the batchs contribute the JPT.
        #Some values in it will be updated in each iteration but the diction will NOT be cleared.
        self.GGM_cache              = {}
        #batch GGM cache. Should BE cleared and updated in each iteration.
        #NOTICE: This set just store the GGM that computed from batch.
        #the real GGM is merged into self.GGM_cache.
        #It should BE cleared before each iteration.
        self.batch_GGM_cache        = set()                                                             #remember to clear
        #cache pesudo inverse of X
        #{X:var}
        self.p_inverse_cache        = {}                                                                #remember to clear
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
        graph.set_skip()
        for i in range(self.n):
            for j in range(self.n):
                if self.priori_knowledge[i, j] == 1:
                    graph.add_edge(i, j)
        cost = self.cost(graph)
        self.queue[graph] = cost
        self.init_flag = True

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

    def set_priori(self, priori_knowledge):
        """
        set the priori knowledge
        """
        self.priori_knowledge = priori_knowledge

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
        !!!Please make sure parents are listed in increasing order
        """
        x = FML[:-1]  #a tuple
        y = FML[-1]   #an int
        N = len(self.batch)
        e = np.ones((N, 1))
        X = np.hstack((e, self.batch[:, x]))
        X = np.mat(X)
        Y = self.batch[:, y]
        Y.shape = (N, 1)
        p_inv  =self.get_p_inverse(x)
        beta = p_inv * Y
        res = (Y - X*beta)
        var = p_inv * np.multiply(res, res)
        #avoid numeric problems
        var = var + 1e-8
        return beta, var

    def get_p_inverse(self, x):
        """
        get p_inverse
        """
        if x in self.p_inverse_cache:
            p_inv = self.p_inverse_cache[x]
        else:
            p_inv = self.p_inverse_from_batch(x)
            self.p_inverse_cache[x] = p_inv             #cache it
        return p_inv

    def p_inverse_from_batch(self, x):
        """
        get p_inverse_from_batch
        x: a tuple
        !!!Please make sure x are listed in increasing order
        """
        N = len(self.batch)
        e = np.ones((N, 1))
        X = np.hstack((e, self.batch[:, x]))
        X = np.mat(X)
        p_inv = (X.T * X).I * X.T
        return p_inv

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
                # cost0 = self.fml_l_cost_cache[fml]
                # cost1 = self.fml_l_cost(fml)
                # cost_u = cost0 + self.alpha * (cost1 - cost0)
                cost_u = self.fml_l_cost(fml)
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
        beta, var = self.get_beta_var(fml)
        cost = Guassian_cost(self.batch, fml, beta, var)
        #cache it in self.fml_l_cost_cache
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
                n = sum(np.array(fml) > 5)
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
    def priori_permit(self, i, j, v):
        """
        check if the priori knowledge is obeyed by the graph
        edge i-->j
        v = 1:      add
            0/else: delete
        """
        if v == 1: #if add
            if self.priori_knowledge[i, j] == -1:
                return False
            else:
                return True
        else:#if delete
            if self.priori_knowledge[i, j] == 1:
                return False
            else:
                return True

    def clear_cache(self):
        """
        clear some caches
        """
        self.batch_GGM_cache.clear()
        self.p_inverse_cache.clear()
        self.fml_l_cost_update_flag.clear()
        self.graph_l_cost_cache.clear()

    def step(self, time_step=None):
        """
        step forward
        """
        #clear cache
        self.clear_cache()
        #init the queue if it is needed
        if not self.init_flag:
            self.init_queue()
        if not self.queue:
            print("Empty queue. Break")
            return
        best, cost = self.best_candidate()
        if time_step is not None:
            print("cost ",time_step, " = ", cost)
        else:
            print(cost)
        #change randomly
        for i in range(6, self.n):
            for j in range(i+1, self.n):
                if best.get_edge(i, j):             #if there is an edge from i to j.
                    if self.priori_permit(i, j, 0): #and could be removed.
                        rem_best = best.clone()
                        rem_best.remove_edge(i, j)
                        rem_cost = self.cost(rem_best)
                        self.queue[rem_best] = rem_cost
                else:                               #there is no edge
                    if self.priori_permit(i, j, 1): #and could be added.
                        addij_best = best.clone()
                        addij_best.add_edge(i, j)
                        addij_cost = self.cost(addij_best)
                        self.queue[addij_best] = addij_cost
        #pick out the current best candidate
     
    def best_BN(self):
        """
        best BN
        """
        bBN = Bayesian_network()
        bBN.struct, _ = self.best_candidate()
        for fml in bBN.struct:
            para = self.GGM_from_batch(fml)
            bBN.parameters.add_fml(fml, para)
        return bBN
