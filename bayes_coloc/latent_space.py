import math
import numpy as np
from scipy.special import logsumexp, expm1 
from .mygraph import Graph

# This function is used to avoid warnings
def ln(x):
    if x == 0:
        return float('-inf')
    else:
        return np.log(x)

class LatentState:
    def __init__(self, n, m, alpha, beta, gamma, cost):
        """ algorithm is initialized with no assignments, only unassigned points """
        self.n = n
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cost = cost
        self.graph = Graph(n, m)
        self.initialize_states()

    def initialize_states(self):
        for i in range(self.n):
            self.add_entry((i, self.m), 1)
        for j in range(self.m):
            self.add_entry((self.n, j), 1)

    def compute_log_prob_matrix(self):
        """Compute the log probability matrix. This is used for testing purposes."""
        graph = self.graph.numpy_graph()
        keys = [(edge['i'], edge['j']) for edge in graph]

        log_prob_matrix = np.empty((len(keys), len(keys)))
        for (i, j) in keys:
            for (i_, j_) in keys:
                log_prob_matrix[keys.index((i, j)), keys.index((i_, j_))] = -self.swap_cost(i, j, i_, j_)
        return log_prob_matrix

    def log_prob_marginal_slow(self):
        """Compute the marginal log probabilities of each state in a slow way. This is used for testing purposes."""
        log_prob_matrix = self.compute_log_prob_matrix()
        l, _ = log_prob_matrix.shape
        res = np.empty(l)
        for i in range(l):
            res[i] = self.log_sum_exp(log_prob_matrix[i, :])
        return res

    def log_prob_marginal(self):
        """Compute the marginal log probabilities of each state."""
        graph = self.graph.numpy_graph()
        log_probs = graph['log_prob_swap_total']
        return log_probs

    def log_sum_exp(self, values):
        if len(values) == 0:
            return float('-inf')
        else:
            return logsumexp(values)

    def log_diff_exp(self, a, b):
        """ Compute log(exp(a) - exp(b)) in a numerically stable way.
        This uses the identity log(exp(a) - exp(b)) = a + log(1 - exp(-|a - b|)) 
        """
        if a==-np.inf and b==-np.inf: # log of 0 should be -inf
            return -np.inf
        return a + self.log1mexp(np.abs(a - b))

    def log1mexp(self, x):
        """ Compute log(1 - exp(-|x|)) in a numerically stable way.
        This is based on the paper: "Accurately Computing log(1 - exp(-|a|))". Assesed by the Rmpfr package" by Martin Maechler, ETH Zurich. """
        if x == 0: # this just serves to suppresses warnings from the log function
            return float('-inf')
        elif x < np.log(2):
            return np.log(-np.expm1(-np.abs(x)))
        else:
            return np.log1p(-np.exp(-np.abs(x)))

    def swap_cost(self, i, j, i_prime, j_prime):
        if i == i_prime or j == j_prime:
            return float('inf')
        else:
            res = (self.cost[i, j_prime] + self.cost[i_prime, j] -
                   self.cost[i, j] - self.cost[i_prime, j_prime])

        intensity_cost_diff = ln(self.gamma) - ln(self.alpha) - ln(self.beta)
        if self.type(i, j) == 0 and self.type(i_prime, j_prime) == 3:
            res += intensity_cost_diff
        elif self.type(i, j) == 3 and self.type(i_prime, j_prime) == 0:
            res += intensity_cost_diff

        elif self.type(i, j) == 1 and self.type(i_prime, j_prime) == 2:
            res -= intensity_cost_diff
        elif self.type(i, j) == 2 and self.type(i_prime, j_prime) == 1:
            res -= intensity_cost_diff
        return res

    def add_entry(self, key, flow):
        i, j = key
        edge_type = self.type(i, j)
        edges0 = self.graph.get_edges_of_type(0)
        edges1 = self.graph.get_edges_of_type(1)
        edges2 = self.graph.get_edges_of_type(2)
        edges3 = self.graph.get_edges_of_type(3)

        log_prob_swap_with_0 = self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_) in edges0])
        log_prob_swap_with_1 = self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_) in edges1])
        log_prob_swap_with_2 = self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_) in edges2])
        log_prob_swap_with_3 = self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_) in edges3])

        log_prob_swap = self.log_sum_exp([log_prob_swap_with_0, log_prob_swap_with_1, log_prob_swap_with_2, log_prob_swap_with_3])

        self.graph.add_edge(i, j, flow, log_prob_swap_with_0, log_prob_swap_with_1, log_prob_swap_with_2, log_prob_swap_with_3, log_prob_swap, edge_type)

        # update all other entries on addition
        self.update_log_probs_on_add(key)

    def remove_entry(self, key):
        i, j = key
        if self.graph.edge_exists(i, j):
            self.graph.delete_edge(i, j)
            self.update_log_probs_on_remove(key)

    def update_log_probs_on_add(self, key):
        graph = self.graph.numpy_graph()
        i, j = key
        edge_type = self.type(i, j)
        log_prob_key = f"log_prob_swap_with_{edge_type}"

        for k in range(len(graph)):
            i_, j_ = graph[k]['i'], graph[k]['j']
            old_log_prob = graph[log_prob_key][k]

            swap_cost = self.swap_cost(i, j, i_, j_)
            new_swap_prob = np.logaddexp(old_log_prob, -swap_cost)
            
            graph[log_prob_key][k] = new_swap_prob
            new_total = self.log_sum_exp([
                graph["log_prob_swap_with_0"][k],
                graph["log_prob_swap_with_1"][k],
                graph["log_prob_swap_with_2"][k],
                graph["log_prob_swap_with_3"][k]
            ])
            graph["log_prob_swap_total"][k] =  new_total

    def update_log_probs_on_remove(self, key):
        i, j = key
        graph = self.graph.numpy_graph()
        edge_type = self.type(i, j)  
        log_prob_key = f"log_prob_swap_with_{edge_type}"
        for k in range(len(graph)):
            i_, j_ = graph[k]['i'], graph[k]['j']
            if i != i_ and j != j_:
                old_log_prob = graph[log_prob_key][k]
                graph[log_prob_key][k] = self.log_diff_exp(
                    old_log_prob,
                    -self.swap_cost(i_, j_, i, j)
                )
                graph["log_prob_swap_total"][k] = self.log_sum_exp([
                    graph["log_prob_swap_with_0"][k],
                    graph["log_prob_swap_with_1"][k],
                    graph["log_prob_swap_with_2"][k],
                    graph["log_prob_swap_with_3"][k]
                ])

    def update_intensities(self, alpha, beta, gamma):
        """Update the intensities alpha, beta, and gamma, and adjust relevant log probabilities."""
        a_old, b_old, c_old = self.alpha, self.beta, self.gamma
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
    
        # Compute the change in log scale
        change = (ln(alpha) + ln(beta) - ln(gamma)) 
        change -= (ln(a_old) + ln(b_old) - ln(c_old))
    
        # Update the log probabilities for each state
        graph = self.graph.numpy_graph()
        for k in range(len(graph)):
            i, j = graph['i'][k], graph['j'][k]
            edge_type = self.type(i, j)
            if edge_type == 0:
                graph["log_prob_swap_with_3"] += change
            elif edge_type == 3:
                graph["log_prob_swap_with_0"] += change
            elif edge_type == 1:
                graph["log_prob_swap_with_2"] -= change
            elif edge_type == 2:
                graph["log_prob_swap_with_1"] -= change
            graph["log_prob_swap_total"][k] = self.log_sum_exp([
                graph["log_prob_swap_with_0"][k],
                graph["log_prob_swap_with_1"][k],
                graph["log_prob_swap_with_2"][k],
                graph["log_prob_swap_with_3"][k]])
        return 

    def type(self, i, j):
        if i < self.n and j < self.m:
            return 0
        elif i < self.n and j == self.m:
            return 1
        elif i == self.n and j < self.m:
            return 2
        else:
            return 3

    def set_key_flow(self, key, flow):
        (i, j) = key
        if flow == 0:
            self.remove_entry(key)
        else:
            k = self.graph.get_edge_index(i, j)
            graph = self.graph.numpy_graph()
            graph["flow"][k] = flow
            

    def get_key_flow(self, key):
        i, j = key
        if not self.graph.edge_exists(i, j):
            self.add_entry(key, 0)
        k = self.graph.get_edge_index(i, j)
        return self.graph.memory[k]["flow"]

    def do_swap(self, key1, key2):
        # print(self.graph.numpy_graph())
        m1 = self.get_key_flow(key1)
        m2 = self.get_key_flow(key2)

        self.set_key_flow(key1, m1 - 1)
        self.set_key_flow(key2, m2 - 1)

        new_key1 = (key1[0], key2[1])
        new_key2 = (key2[0], key1[1])

        n1 = self.get_key_flow(new_key1)
        n2 = self.get_key_flow(new_key2)

        self.set_key_flow(new_key1, n1 + 1)
        self.set_key_flow(new_key2, n2 + 1)

    def keys(self):
        graph = self.graph.numpy_graph()
        return [(edge['i'], edge['j']) for edge in graph]

    def log_prob_swap(self, key1, key2):
        """Calculate the probability of swapping key1 with key2."""
        graph = self.graph.numpy_graph()
        keys = [(edge['i'], edge['j']) for edge in graph]
        log_probs = self.log_prob_marginal()

        l1 = log_probs[keys.index(key1)] - self.log_sum_exp(log_probs)
    
        log_probs = [-self.swap_cost(*key1, *key) for key in keys]
        l2 = log_probs[keys.index(key2)] - self.log_sum_exp(log_probs)
    
        return l1 + l2

    def log_prob_reverse_swap(self, key1, key2):
        """Calculate the probability of reversing the swap of key1 with key2."""
        (i, j), (i_prime, j_prime) = key1, key2
        key1_new = (i, j_prime)
        key2_new = (i_prime, j)

        # to avoid numerical instability, we make a copy of the current state
        self.do_swap(key1, key2)
        log_prob_swap = self.log_prob_swap(key1_new, key2_new)

        # undo swap
        self.do_swap(key1_new, key2_new)

        return log_prob_swap


    def numpy_path(self):
        """Return a numpy array representing the assignments."""
        path = self.graph.numpy_path()
        return path
    
