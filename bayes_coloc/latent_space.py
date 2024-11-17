import math
import numpy as np
from scipy.special import logsumexp
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

    def log_sum_exp(self, values, axis=None):
        if len(values) == 0:
            return float('-inf')
        else:
            return logsumexp(values, axis=axis)

    def log_diff_exp(self, a, b):
        """
        Compute log(exp(a) - exp(b)) in a numerically stable way using vectorized operations.
        Uses the identity log(exp(a) - exp(b)) = a + log(1 - exp(-|a - b|)).
        
        Parameters:
        - a, b: Scalars or NumPy arrays.
        
        Returns:
        - A scalar or NumPy array of computed values.
        """
        a = np.asarray(a)
        b = np.asarray(b)

        # Difference between -inf and -inf should be 0 in our case
        with np.errstate(invalid='ignore'):
            diff = a - b
        diff = np.where(np.isnan(diff), 0, diff)
        result = a + self.log1mexp(np.abs(diff))
        
        return result


    def log1mexp(self, x):
        """
        Compute log(1 - exp(-|x|)) in a numerically stable way using vectorized operations.
        Based on the paper: "Accurately Computing log(1 - exp(-|a|))".
        Assesed by the Rmpfr package" by Martin Maechler, ETH Zurich.
        
        Parameters:
        - x: Scalar or NumPy array.
        
        Returns:
        - A scalar or NumPy array of the computed values.
        """
        # we need to handle the case where x is 0.
        with np.errstate(divide='ignore'):
            result = np.where(
                x < np.log(2),
                np.log(-np.expm1(-x)),  # More accurate for small x
                np.log1p(-np.exp(-x))   # More accurate for larger x
            )

        return result


    def swap_cost_with_edges(self, i, j, i_prime, j_prime):
        """
        Computes the swap cost between edges (i, j) and (i_prime, j_prime) for arrays of edge endpoints.
        If i equals i_prime or j equals j_prime, assigns an infinite cost. Otherwise, calculates the swap cost
        using the specified cost matrix and intensity differences based on edge types.

        Parameters:
        - i, j: Integers representing the first edge's endpoints.
        - i_prime, j_prime: NumPy arrays representing the second edge's endpoints.

        Returns:
        - A NumPy array of swap costs.
        """
        # Vectorized check for infinity cost condition
        infinite_cost_mask = (i == i_prime) | (j == j_prime)
        res = np.full_like(i_prime, float('inf'), dtype=float)
        
        # Calculate only for valid cases (where cost is not infinite)
        valid_indices = ~infinite_cost_mask
        if np.any(valid_indices):
            res[valid_indices] = (
                self.cost[i, j_prime[valid_indices]] + self.cost[i_prime[valid_indices], j] -
                self.cost[i, j] - self.cost[i_prime[valid_indices], j_prime[valid_indices]]
            )

            # Calculate intensity cost difference
            intensity_cost_diff = ln(self.gamma) - ln(self.alpha) - ln(self.beta)

            # Vectorized condition adjustments based on type conditions
            types_ij = self.type(i, j)
            types_ipjp = self.type(i_prime, j_prime)

            res += np.where(
                (types_ij == 0) & (types_ipjp == 3), intensity_cost_diff, 0
            )
            res += np.where(
                (types_ij == 3) & (types_ipjp == 0), intensity_cost_diff, 0
            )
            res -= np.where(
                (types_ij == 1) & (types_ipjp == 2), intensity_cost_diff, 0
            )
            res -= np.where(
                (types_ij == 2) & (types_ipjp == 1), intensity_cost_diff, 0
            )

        return res



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

        log_prob_swap_with_0 = self.log_sum_exp(-self.swap_cost_with_edges(i, j, edges0["i"], edges0["j"]))
        log_prob_swap_with_1 = self.log_sum_exp(-self.swap_cost_with_edges(i, j, edges1["i"], edges1["j"]))
        log_prob_swap_with_2 = self.log_sum_exp(-self.swap_cost_with_edges(i, j, edges2["i"], edges2["j"]))
        log_prob_swap_with_3 = self.log_sum_exp(-self.swap_cost_with_edges(i, j, edges3["i"], edges3["j"]))

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
        """
        Updates the log probabilities for all nodes in the graph when a new edge is added, using vectorized operations.
        
        Parameters:
        - key: Tuple (i, j) representing the edge being added.
        """
        graph = self.graph.numpy_graph()
        i, j = key
        edge_type = self.type(i, j)
        log_prob_key = f"log_prob_swap_with_{edge_type}"
        
        # Extract i_, j_ values as arrays for vectorized operations
        i_values = np.array([entry['i'] for entry in graph])
        j_values = np.array([entry['j'] for entry in graph])
        old_log_probs = np.array(graph[log_prob_key])
        
        # Vectorized computation of swap costs
        swap_costs = self.swap_cost_with_edges(i, j, i_values, j_values)
        new_swap_probs = np.logaddexp(old_log_probs, -swap_costs)
        
        # Update graph's log_prob_key with new values
        graph[log_prob_key] = new_swap_probs
        
        # Vectorized computation of new totals
        log_prob_keys = [
            "log_prob_swap_with_0",
            "log_prob_swap_with_1",
            "log_prob_swap_with_2",
            "log_prob_swap_with_3"
        ]
        log_probs_matrix = np.array([graph[key] for key in log_prob_keys]).T  # Create a matrix of all relevant log probabilities
        graph["log_prob_swap_total"] = self.log_sum_exp(log_probs_matrix, axis=1)


    def update_log_probs_on_remove(self, key):
        """
        Updates the log probabilities for all nodes in the graph when an edge is removed, using vectorized operations.
        
        Parameters:
        - key: Tuple (i, j) representing the edge being removed.
        """
        i, j = key
        graph = self.graph.numpy_graph()
        edge_type = self.type(i, j)
        log_prob_key = f"log_prob_swap_with_{edge_type}"

        # Extract arrays for vectorized operations
        i_values = graph['i']
        j_values = graph['j']
        old_log_probs = np.array(graph[log_prob_key])

        # Vectorized computation of log_diff_exp where mask is True
        updated_log_probs = self.log_diff_exp(old_log_probs, -self.swap_cost_with_edges(i, j, i_values, j_values))        
        graph[log_prob_key] = updated_log_probs

        # Vectorized computation of new totals
        log_prob_keys = [
            "log_prob_swap_with_0",
            "log_prob_swap_with_1",
            "log_prob_swap_with_2",
            "log_prob_swap_with_3"
        ]
        log_probs_matrix = np.array([graph[key] for key in log_prob_keys]).T  # Create a matrix of all relevant log probabilities
        graph["log_prob_swap_total"] = self.log_sum_exp(log_probs_matrix, axis=1)

    def update_intensities(self, alpha, beta, gamma):
        """
        Update the intensities alpha, beta, and gamma, and adjust relevant log probabilities in a vectorized manner.
        """
        a_old, b_old, c_old = self.alpha, self.beta, self.gamma
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        # Compute the change in log scale
        change = (ln(alpha) + ln(beta) - ln(gamma)) - (ln(a_old) + ln(b_old) - ln(c_old))

        # Update the log probabilities for each state
        graph = self.graph.numpy_graph()

        # Extract values as arrays for vectorized computation
        i_values = np.array(graph['i'])
        j_values = np.array(graph['j'])
        edge_types = np.array([self.type(i, j) for i, j in zip(i_values, j_values)])

        # Vectorized updates based on edge type masks
        graph["log_prob_swap_with_3"] += np.where(edge_types == 0, change, 0)
        graph["log_prob_swap_with_0"] += np.where(edge_types == 3, change, 0)
        graph["log_prob_swap_with_2"] -= np.where(edge_types == 1, change, 0)
        graph["log_prob_swap_with_1"] -= np.where(edge_types == 2, change, 0)

        # Vectorized computation of log_prob_swap_total
        log_prob_keys = [
            "log_prob_swap_with_0",
            "log_prob_swap_with_1",
            "log_prob_swap_with_2",
            "log_prob_swap_with_3"
        ]
        log_probs_matrix = np.array([graph[key] for key in log_prob_keys]).T  # Create matrix of log probabilities
        graph["log_prob_swap_total"] = self.log_sum_exp(log_probs_matrix, axis=1)

        return

    def type(self, i, j):
        """
        Determines the type based on the values of i and j using broadcasting-friendly operations.
        Assumes i and j can be scalars or NumPy arrays of compatible shapes.
    
        Parameters:
        - i: Integer or NumPy array of integers.
        - j: Integer or NumPy array of integers.
    
        Returns:
        - A NumPy array of types based on conditions applied to i and j.
        """
        i = np.asarray(i)
        j = np.asarray(j)
    
        type_array = np.full_like(i, 3, dtype=int)  # Default type is 3
        type_array[(i < self.n) & (j < self.m)] = 0
        type_array[(i < self.n) & (j == self.m)] = 1
        type_array[(i == self.n) & (j < self.m)] = 2
        
        return type_array

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
    
