"""
A* search for fault isolation
"""

import torch
import numpy as np
from hybrid_algorithm.hybrid_detector import shrink

class a_star:
    """
    A* search
    """
    def __init__(self, fault_num, epsilon=1e-1, beta=1e-1):
        """
        conflict/consistency set: a set of fault type. start from 0
        """
        self.fault_num = fault_num  #fault number
        self.conflict = []          #list of conflict sets
        self.consistency = []       #list of consistency sets
        self.priori = None          #numpy arrary of priori probabilities
        self.epsilon = epsilon      #epsilon
        self.beta = beta            #beta
        self.queue = []             #search queue, queue = [(health_state1, cost1), (health_state2, cost2) ...]

    def add_conflict(self, conf):
        """
        add conflict into self.conflict
        """
        self.conflict.append(conf)

    def add_consistency(self, consis):
        """
        add consistency into self.consistency
        """
        self.consistency.append(consis)

    def clear_conf_consis(self):
        """
        clear all conflicts and consistencies
        """
        self.conflict = []
        self.consistency = []

    def set_priori(self, priori):
        """
        set priori probability based on observation
        """
        self.priori = np.array([shrink(x, 1e-15) for x in priori])

    def zero_residual_num4i(self, i):
        """
        return the number of zero_residual invoving fault i
        """
        num = 0
        for consis in self.consistency:
            if i in consis:
                num = num + 1
        return num
    
    def zero_residual_num(self, health_state):
        """
        return the number of zero_residual invoving the fault
        """
        num = []
        for i in range(len(health_state)):
            num.append(self.zero_residual_num4i(i))
        return np.array(num)

    def cause_conflict(self, health_state, conflict):
        """
        return 1 if health_state cause conflict
        return 0 else
        """
        if len(health_state) < len(conflict):
            return 0
        for i in conflict:
            if i >= len(health_state):
                return 0
            if health_state[i] == 1:
                return 0
        return 1

    def unsolve_conflict_num(self, health_state):
        """
        return the nubmer of conflicts unsolved by health_state
        """
        num = 0
        for conf in self.conflict:
            num = num + self.cause_conflict(health_state, conf)
        return num


    def cost(self, health_state):
        """
        evaluate the cost of health_state
        """
        assert len(health_state) <= self.fault_num
        epsilon_cost = -np.log(self.epsilon)
        beta_cost = -np.log(self.beta)
        one_min_beta_cost = -np.log(1-self.beta)
        num = len(health_state)
        fault_priori = self.priori[0:num]
        normal_priori = (1 - fault_priori)
        fault_vector = np.array(health_state)
        normal_vector = (1 - fault_vector)
        priori = fault_vector * fault_priori + normal_vector * normal_priori
        #priori cost = - log(p)
        priori_cost = - sum(np.log(priori))

        #consistency break cost
        consis_num = self.zero_residual_num(health_state)
        consistency_break_cost = consis_num * (fault_vector * epsilon_cost + normal_vector * one_min_beta_cost)
        consistency_break_cost = sum(consistency_break_cost)
        #conflict solve cost
        conflict_solve_cost = self.unsolve_conflict_num(health_state) * beta_cost

        #estimated cost
        last_priori = self.priori[len(health_state):]
        max_last_priori = [x if x > 0.5 else 1-x for x in last_priori]
        neg_ln_max_last_priori = [-np.log(x) for x in max_last_priori]
        estimated_cost = sum(neg_ln_max_last_priori)

        cost = priori_cost + consistency_break_cost + conflict_solve_cost + estimated_cost
        return cost

    def init_queue(self):
        """
        init the search queue by []
        """
        self.queue = [([], 100)]

    def sort_queue(self):
        """
        sort the queue. health_state with minimal cost in the first
        """
        self.queue = sorted(self.queue, key=lambda item:item[1])

    def expand_node(self, health_state):
        """
        expand a node
        """
        assert len(health_state) < self.fault_num
        # 0
        health_state0 = health_state[:]
        health_state0.append(0)
        cost0 = self.cost(health_state0)
        self.queue.append((health_state0, cost0))
        # 1
        health_state1 = health_state[:]
        health_state1.append(1)
        cost1 = self.cost(health_state1)
        self.queue.append((health_state1, cost1))

        self.sort_queue()

    def most_probable(self, n):
        """
        return the most probable n health_state
        """
        #counter
        most_probable_health_state = []
        #init queue
        self.init_queue()
        #iteration
        while len(most_probable_health_state) < n:
            if len(self.queue) == 0:
                break
            best = self.queue[0]
            del self.queue[0]
            if len(best[0]) == self.fault_num:
                most_probable_health_state.append(best)
            else:
                self.expand_node(best[0])
        #end iteration
        return most_probable_health_state
