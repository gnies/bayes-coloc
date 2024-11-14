import numpy as np
from scipy.special import logsumexp, expm1 

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
        self.graph = self.construct_no_matching_graph()

    def construct_no_matching_graph(self):
        """ Initialize the graph with no matchings."""

        # initialize the log probabilities of swapping
        log_probs = self.log_prob_marginal_slow()
        graph['log_p_swap_total'] = log_probs
        dtype = [
            ('i', 'i4'), 
            ('j', 'i4'), 
            ('edge_representative', 'bool'),
            ('assigned', 'bool'),
            ('bin_bin', 'bool'),
            ('unassigned_first_set', 'bool'),
            ('unassigned_second_set', 'bool'),
            ('log_p_swap_with_assigned', 'f8'), 
            ('log_p_swap_with_bin_bin', 'f8'), 
            ('log_p_swap_with_unassigned_first_set', 'f8'), 
            ('log_p_swap_with_unassigned_second_set', 'f8'), 
            ('log_p_swap_total', 'f8')
        ]
        for i in range(self.n):
            j = self.m
            graph[i] = (i, j, True, False, False, True, False, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)

        for j in range(self.m):
            i = self.n
            graph[i + j] = (i, j, True, False, False, False, True, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)

        # Create an empty structured array
        total_rows = self.n + self.m
        graph = np.empty(total_rows, dtype=dtype)
        edges = graph["i", "j"]
        mat = np.empty((len(edges), len(edges)))
        for k, (i, j) in enumerate(edges):
            for l, (i_prime, j_prime) in enumerate(edges):
                mat[k, l] = self.swap_cost(i, j, i_prime, j_prime)

        for k in range(total_rows):
            log_probs_k = self.log_sum_exp(mat[k, :])
            if k < self.n:
                graph[k]["log_p_swap_with_unassigned_second_set"] = log_probs_k
            else:
                graph[k]["log_p_swap_with_unassigned_first_set"] = log_probs_k
                graph["log_p_swap_total"] = log_probs_k
                graph[k]["log_p_swap_total"] = log_probs_k
        
    def log_prob_marginal(self):
        """Compute the marginal log probabilities of each state."""
        return self.graph["log_p_swap_total"]

    def all_edges_of_type(self, state_type):
        """Return all representing edges of a given type."""
        return self.graph[self.graph["edge_representative"] & (self.graph[state_type])]

    def do_swap(self, k1, k2):
        i1, j1 = self.graph["i", "j"][k1]
        i2, j2 = self.graph["i", "j"][k2]

        ### do swap
        self.graph["i", "j"][k1] = i1, j2
        self.graph["i", "j"][k2] = i2, j1
        
        ### update type information about the edges
        
        k1_type = self.type(i1, j1)
        k2_type = self.type(i2, j2)
        self.graph[k1_type][k1] = False
        self.graph[k2_type][k2] = False
        
        k1_new_type = self.type(i1, j2)
        k2_new_type = self.type(i2, j1)
        self.graph[k1_new_type][k1] = True
        self.graph[k2_new_type][k2] = True
    
        ### update bin_bin representer
        
        # if bin_bin edge was removed, then we select a new representative
        if self.type(i1, j1) == "bin_bin" or self.type(i2, j2) == "bin_bin":
            bin_bin_locs = np.where(self.graph["bin_bin"])[0]
            if len(bin_bin_locs) > 0:
                # select the first bin_bin edge as a representative
                self.graph["edge_representative"][bin_bin_locs[0]] = True
                
        # if a bin_bin edge is created, we make it to the representative edge
        if k1_new_type == "bin_bin" or k2_new_type == "bin_bin":
            # check if there is a previous representative bin_bin edge
            bin_bin_locs = np.where(self.graph["bin_bin"])[0]
            if len(bin_bin_locs) > 0:
                old_reprentative_index = np.where(graph["edge_representative"] & graph["bin_bin"])[0][0] 
                # remove representative status
                self.graph["edge_representative"][old_reprentative_index] = False
                # edge cannot be swapped with any other edge as it is not a representative edge
                self.graph["log_p_swap_total"][old_reprentative_index] = -np.inf
                self.graph["log_p_swap_with_assigned"][old_reprentative_index] = -np.inf

        ### compute new log probabilities for swapped entries
        self.add_log_probs(k1)
        self.add_log_probs(k2)

        ### modify all other swap probabilities
        self.update_log_probs((i1, j1), (i2, j1))
        self.update_log_probs((i2, j2), (i1, j2))

    def update_log_probs(self, old_pair, new_pair):
        i, j = old_pair
        i_prime, j_prime = new_pair
        
        # subtract probability of swapping with old pair
        old_pair_type = self.type(i, j)
        for k, (i_, j_) in enumerate(self.graph["i", "j"]):
                log_prob_key = f"log_prob_swap_with_{old_pair_type}"
                self.graph[k][log_prob_key] = self.log_diff_exp(
                    self.graph[k][log_prob_key],
                    -self.swap_cost(i_, j_, i, j)
                )
        # add probability of swapping with new pair
        new_pair_type = self.type(i_prime, j_prime)
        for k, (i_, j_) in enumerate(self.graph["i", "j"]):
                log_prob_key = f"log_prob_swap_with_{new_pair_type}"
                self.graph[k][log_prob_key] = self.log_sum_exp([
                    self.graph[k][log_prob_key],
                    -self.swap_cost(i_, j_, i_prime, j_prime)
                ])
        # update total swap probability
        for k in range(len(self.graph)):
            self.graph[k]["log_prob_swap_total"] = self.log_sum_exp([
                self.graph[k]["log_prob_swap_with_assigned"],
                self.graph[k]["log_prob_swap_with_unassigned_first_set"],
                self.graph[k]["log_prob_swap_with_unassigned_second_set"],
                self.graph[k]["log_prob_swap_with_bin_bin"]
            ])

        
    def add_log_probs(self, k):
        """
        Add the log probabilities of swapping the edge at index k with all other edges of the same type.
        """
        i, j = self.graph["i", "j"][k]
        for type_ in ["assigned", "unassigned_first_set", "unassigned_second_set", "bin_bin"]:
            log_prob_key = f"log_prob_swap_with_{type_}"
            self.graph[k][log_prob_key] = self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_) in self.graph["i", "j"] if self.type(i_, j_) == state_type])
            self.graph[k]["log_prob_swap_total"] = self.log_sum_exp([self.graph[k]["log_prob_swap_with_assigned"], 
                                                                        self.graph[k]["log_prob_swap_with_unassigned_first_set"], 
                                                                        self.graph[k]["log_prob_swap_with_unassigned_second_set"], 
                                                                        self.graph[k]["log_prob_swap_with_bin_bin"]])
            
    def swap_cost(self, i, j, i_prime, j_prime):
        if i == i_prime or j == j_prime:
            return float('inf')
        else:
            res = (self.cost[i, j_prime] + self.cost[i_prime, j] -
                   self.cost[i, j] - self.cost[i_prime, j_prime])

        intensity_cost_diff = ln(self.gamma) - ln(self.alpha) - ln(self.beta)
        if self.type(i, j) == "assigned" and self.type(i_prime, j_prime) == "bin_bin":
            res += intensity_cost_diff
        elif self.type(i, j) == "bin_bin" and self.type(i_prime, j_prime) == "assigned":
            res += intensity_cost_diff

        elif self.type(i, j) == "unassigned_first_set" and self.type(i_prime, j_prime) == "unassigned_second_set":
            res -= intensity_cost_diff
        elif self.type(i, j) == "unassigned_second_set" and self.type(i_prime, j_prime) == "unassigned_first_set":
            res -= intensity_cost_diff
        return res

    def update_intensities(self, alpha, beta, gamma):
        """Update the intensities alpha, beta, and gamma, and adjust relevant log probabilities."""
        graph = self.graph
        a_old, b_old, c_old = self.alpha, self.beta, self.gamma
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
    
        # Compute the change in log scale
        change = (ln(alpha) + ln(beta) - ln(gamma)) 
        change -= (ln(a_old) + ln(b_old) - ln(c_old))

        # update the log probabilities for each state
        for k, (i, j) in enumerate(graph["i", "j"]):
            if graph[k]["edge_representative"]:
                state_type = self.type(i, j)
                if state_type == "assigned":
                    graph[k]["log_prob_swap_with_bin_bin"] += change
                elif state_type == "bin_bin":
                    graph[k]["log_prob_swap_with_assigned"] += change
                elif state_type == "unassigned_first_set":
                    graph[k]["log_prob_swap_with_unassigned_second_set"] -= change
                elif state_type == "unassigned_second_set":
                    graph[k]["log_prob_swap_with_unassigned_first_set"] -= change
                graph[k]["log_prob_swap_total"] = self.log_sum_exp([graph[k]["log_prob_swap_with_assigned"],
                                                                    graph[k]["log_prob_swap_with_unassigned_first_set"],
                                                                    graph[k]["log_prob_swap_with_unassigned_second_set"],
                                                                    graph[k]["log_prob_swap_with_bin_bin"]])

    def type(self, i, j):
        if i < self.n and j < self.m:
            return "assigned"
        elif i == self.n and j < self.m:
            return "unassigned_second_set"
        elif i < self.n and j == self.m:
            return "unassigned_first_set"
        else:
            return "bin_bin"
    
    def log_prob_swap(self, k1, k2):
        """Calculate the probability of swapping matching at index k1 with matching at index k2."""

        i1, j1 = self.graph["i", "j"][k1]
        i2, j2 = self.graph["i", "j"][k2]
        
        log_probs = self.log_prob_marginal()
        l1 = log_probs[k1] - self.log_sum_exp(log_probs)
        
        log_probs_second_edge = [self.swap_cost(i1, j1, i2, j2)]
        l2 = log_probs_second_edge[k2] - self.log_sum_exp(log_probs_second_edge)
    
        return l1 + l2

    def log_prob_reverse_swap(self, k1, k2):
        """Calculate the probability of reversing the swap of matching at index k1 with matching at index k2."""

        i1, j1 = self.graph["i", "j"][k1]
        i2, j2 = self.graph["i", "j"][k2]

        # do swap
        self.do_swap(k1, k2)
        
        # calculate the probablity of going back
        log_prob_swap = self.log_prob_swap(k1, k2)

        # undo swap
        self.do_swap(k1, k2)

        return log_prob_swap

    def numpy_path(self):
        """Return a numpy array representing the assignments."""
        path = self.graph["i", "j"]
        return path
    
    def log_sum_exp(self, values):
        if len(values) == 0:
            return float('-inf')
        else:
            return logsumexp(values)

    def log_diff_exp(self, a, b):
        """ Compute log(exp(a) - exp(b)) in a numerically stable way.
        This uses the identity log(exp(a) - exp(b)) = a + log(1 - exp(-|a - b|)). 
        """
        return a + self.log1mexp(np.abs(a - b))

    def log1mexp(self, x):
        """
        This is based on the paper: "Accurately Computing log(1 - exp(-|a|))". Assessed by the Rmpfr package" by Martin Maechler, ETH Zurich. 
        """
    
        if x == 0: # this just serves to suppresses warnings from the log function
            return float('-inf')
        elif x < np.log(2):
            return np.log(-np.expm1(-np.abs(x)))
        else:
            return np.log1p(-np.exp(-np.abs(x)))

