"""
learn an influence graph
"""

import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from math import log
from math import exp
from graph_model.hybrid_structure import Hybrid_structure
from graph_model.utilities import read_normal_data
from graph_model.utilities import discrete_data
from data_manger.utilities import get_file_list


class IG_learning:
    """
    influence graph learning
    """
    def __init__(self):
        # data is a m×n matrix where 
        # m is the data number and n is the variable number
        self._data          = None
        # fsm is a n×F matrix where
        # n is the variable number and F is the fault number
        self._fsm           = None
        self._dn            = None
        self._queue         = {}
        #cache
        #key: ((parents), kid)
        #value:np.array
        self._cache         = {}

    def init_queue(self):
        """
        init the queue with an empty graph
        """
        empty_graph = Hybrid_structure(5, 6)
        score       = self.__score(empty_graph)
        self._queue[empty_graph] = score


    def load_normal_data(self, file_path, fault_time=None, snr=None):
        """
        load normal data where all data files are in file_path
        If fault_time=None, all data are normal,
        Else data before fault_time is normal.
        fault_time belongs to (0, 1)
        """
        list_files = get_file_list(file_path)
        for file in list_files:
            sig = read_normal_data(file_path + file, fault_time, snr)
            if self._data is None:
                self._data = sig
            else:
                self._data = np.concatenate((self._data, sig))
        self._data = discrete_data(self._data, self._dn)

    def set_discrete_num(self, dn):
        """
        set the discrete number for each variable
        """
        self._dn = dn

    def set_fsm(self, fsm):
        """
        set the fault signature matrix
        """
        self._fsm = fsm
        pass

    def greedy_search(self, iter=20, epsilon=0.001):
        """
        greedy search
        iterate at most *iter* times.
        If the best score change is less than *epsilon*, stop
        """
        score0 = float("inf")
        for _ in range(iter):
            best, score1 = self.__best_in_queue()
            print(score1)
            if abs(score0 - score1) < epsilon:
                break
            else:
                score0 = score1
            self.__expand(best)

    def best(self):
        """
        return the best one
        """
        best, score = self.__best_in_queue()
        return best, score

    #inner functions
    def __best_in_queue(self):
        """
        get the best graph in the queue
        it remains in the queue and not be removed
        """
        #return graph, score
        graph = max(self._queue,key=self._queue.get)
        score = self._queue[graph]
        return graph, score

    def __expand_edge(self, graph, i, j):
        """
        expand the *graph* between node *i* and node *j*
        """
        #add
        G = graph.clone_directed()
        if G.add_dedge(i, j):
            if G.is_acyclic() and self.__CE(G):
                if not self.__queue_has(G):
                    score = self.__score(G)
                    self._queue[G] = score
        #delete
        G = graph.clone_directed()
        if G.remove_dedge(i, j):
            if G.is_acyclic() and self.__CE(G):
                if not self.__queue_has(G):
                    score = self.__score(G)
                    self._queue[G] = score
        #reverse
        G = graph.clone_directed()
        if G.reverse_dedge(i, j):
            if G.is_acyclic() and self.__CE(G):
                if not self.__queue_has(G):
                    score = self.__score(G)
                    self._queue[G] = score

    def __expand(self, graph):
        """
        expand *graph*
        """
        _, n = self._data.shape
        for i in range(n):
            for j in range(i+1, n):
                self.__expand_edge(graph, i, j)
                self.__expand_edge(graph, j, i)

    def __CE(self, graph):
        """
        extend the directed *graph* by the fsm
        """
        #return True/False
        graph.set_CRG(self._fsm)
        return graph.remove_redundency()

    def __score(self, graph):
        """
        evaluate the score of *graph*
        """
        n       = len(self._data)
        w       = log(n)/(2*n)
        ll      = self.__ll(graph)
        dim     = self.__dim(graph)
        score   = ll - w*dim
        return score

    def __queue_has(self, graph):
        """
        check if the search queue has *graph*
        """
        #return True/False
        for g in self._queue:
            if g == graph:
                return True
        return False

    def __dim(self, graph):
        """
        return the parameter priori
        """
        num = 0
        for parents, neigbors, kid in graph:
            num0 = self.__para_num(parents, neigbors, kid)
            num  = num + num0
        return num

    def __ll(self, graph):
        """
        return the log likelihood of parents-->kid
        """
        ll = 0
        for parents, _, kid in graph:
            ip = self.__Ip(parents, kid)
            hp = self.__Hp(kid)
            ll0 = ip - hp
            ll  = ll + ll0
        return ll
        
    def __para_num(self, parents, neigbors, kid):
        """
        return the number of free parameters
        """
        d = 1
        for i in parents:
            d = d * self._dn[i]
        d = d * (2**len(neigbors))
        d = d * self._dn[kid]
        d = d - 1
        return d

    def __mlp_from_data(self, variables):
        """
        maximal likelihood parameters from data
        """
        dn         = self._dn[np.array(variables)]
        N          = np.prod(dn)
        parameters = np.zeros(N)
        data       = self._data[:, np.array(variables)]
        for i in range(N):
            vec  = self.__num2vec(i, variables)
            num  = self.__count(data, vec) + 1
            parameters[i] = num
        parameters = parameters / np.sum(parameters)
        return parameters

    def __mlp(self, variables):
        """
        return the maximal likelihood parameters from data or cache
        """
        if variables in self._cache:
            return self._cache[variables]
        else:
            mlp = self.__mlp_from_data(variables)
            self._cache[variables] = mlp
            return mlp

    def __count(self, data, instance):
        """
        return the number of instance in data
        """
        cmp = (data == instance)
        cmp = [True if i.all() else False for i in cmp]
        num = np.sum(cmp)
        return num

    def __num2vec(self, number, variables):
        """
        convert a number into a vector
        """
        if isinstance(variables, int):
            variables = [variables]
        num = self._dn[np.array(variables)]
        n   = len(variables)
        vec = np.zeros(n).astype(int)
        for i in range(n-1, -1, -1):
            left = num[:i]
            base = np.prod(left)
            vec[i] = int(number/base)
            number = number - vec[i]*base
        return vec

    def __vec2num(self, vec, variables):
        """
        convert a vector into a number
        """
        num    = self._dn[np.array(variables)]
        number = 0
        n      = len(variables)
        for i in range(n):
            left = num[i+1:]
            base = np.prod(left)
            number = number + vec[i]*base
        return number

    def __Ip(self, parents, kid):
        """
        return the mutual information entropy
        """
        if len(parents) == 0:
            return 0
        entropy   = 0
        variables = list(parents)
        variables.append(kid)
        variables = sorted(variables)
        variables = tuple(variables)
        kid_index = variables.index(kid)
        mlp_pk    = self.__mlp(variables)
        mlp_p     = self.__mlp(parents)
        mlp_k     = self.__mlp(kid)
        dn        = self._dn[np.array(variables)]
        N         = np.prod(dn)
        for i in range(N):
            vec   = self.__num2vec(i, variables)
            p_vec = np.concatenate((vec[:kid_index], vec[kid_index+1:]))
            p_num = self.__vec2num(p_vec, parents)
            k_num = vec[kid_index]
            ppk   = mlp_pk[i]
            pp    = mlp_p[p_num]
            pk    = mlp_k[k_num]
            entropy0 = ppk*log(ppk/(pp*pk))
            entropy = entropy + entropy0
        return entropy

    def __Hp(self, x):
        """
        return the entropy of a variable
        """
        entropy = 0
        n = self._dn[x]
        p = self.__mlp(x)
        for i in range(n):
            entropy = entropy + p[i] * log(1/p[i])
        return entropy
