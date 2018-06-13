"""
Defines the framework of A* search algorithm
"""
import time
from queue import PriorityQueue

class a_star_frame:
    """
    The A* search framework
    If you want to use it, inherit it and implement self.cost function.
    Set order and priori firstly and then call search.
    """
    def __init__(self):
        #MEMBER
        #priority queue to stor candidate
        self.queue      = None
        #order should be a tuple that stores the search order of varibles
        #for example: (0, 1, 2, 3, 4, 5)
        #initialized as None
        self.order      = None
        #priori probability
        #priori should be a two-layer tuple that stores the priori probability of varialbe values
        #priori is organised in [0, 1, 2, 3,...] order
        #for example: ((0.1, 0.9), (0.2, 0.8))
        #initialized as None
        self.priori     = None
        #search time
        self.time_stats = 0

    def set_order(self, order):
        """
        set the variable order
        """
        self.order = order
        
    def set_priori(self, priori):
        """
        set priori probabiltiy
        """
        self.priori = priori

    def _cost(self, candidate):
        """
        return the cost of the candidate
        candidate is a tuple where the variables are ordered in self.order
        and the values are for each corresponding variables.
        for example ()
        """
        pass

    def __init_queue(self):
        """
        init the queue
        """
        self.queue = PriorityQueue()
        self.queue.put((0,()))

    def __expand(self, candidate):
        """
        expand the candidate
        """
        index = len(candidate)
        var_id = self.order[index]
        var_domain_num = len(self.priori[var_id])
        for i in range(var_domain_num):
            new_candidate = list(candidate)
            new_candidate.append(i)
            new_candidate = tuple(new_candidate)
            cost = self._cost(new_candidate)
            self.queue.put((cost, new_candidate))

    def __goal_include(self, candidate):
        """
        check if candidate is a goal
        """
        assert len(candidate) <= len(self.order)
        return len(candidate) == len(self.order)

    def search(self, num=1):
        """
        step forward
        num is the num of results to search. 1 by default
        """
        assert self.order is not None
        assert self.priori is not None
        #init the queue
        self.__init_queue()
        start = time.clock()
        result = []
        while len(result) < num:
        #get best candidate
            if self.queue.empty():
                return result
            _, best_candi = self.queue.get()
            if self.__goal_include(best_candi):
                result.append(best_candi)
            else:
                self.__expand(best_candi)
        end = time.clock()
        self.time_stats = self.time_stats + (end - start)
        return result

    def time_cost(self):
        """
        return the time consumed by search
        """
        return self.time_stats
