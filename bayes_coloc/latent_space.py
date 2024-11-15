import numpy as np
from scipy.special import logsumexp, expm1 
from icecream import ic

ic.disable()
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
        dtype = [
            ('ij', '2i4'), 
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

        # Create an empty structured array
        total_rows = self.n + self.m
        graph = np.empty(total_rows, dtype=dtype)
        for i in range(self.n):
            j = self.m
            graph[i] = ((i, j), True, False, False, True, False, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)

        for j in range(self.m):
            i = self.n
            graph[i + j] = ((i, j), True, False, False, False, True, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf)

        edges = graph["ij"]
        mat = np.empty((len(edges), len(edges)))

        # could be vectorized/copiled
        for k, (i, j) in enumerate(edges):
            for l, (i_prime, j_prime) in enumerate(edges):
                mat[k, l] = -self.swap_cost(i, j, i_prime, j_prime)

        for k in range(total_rows):
            log_probs_k = self.log_sum_exp(mat[k, :])
            if k < self.n:
                graph[k]["log_p_swap_with_unassigned_second_set"] = log_probs_k
                graph[k]["log_p_swap_total"] = log_probs_k
            else:
                graph[k]["log_p_swap_with_unassigned_first_set"] = log_probs_k
                graph[k]["log_p_swap_total"] = log_probs_k
        return graph
        
    def log_prob_marginal(self):
        """Compute the marginal log probabilities of each state."""
        log_probs = self.graph["log_p_swap_total"].copy()
        # only representative edges can be swapped
        log_probs[~self.graph["edge_representative"]] = -np.inf
        return log_probs

    def log_prob_marginal_slow(self):
        edges = self.graph["ij"]
        mat = np.empty((len(edges), len(edges)))
        for k, (i, j) in enumerate(edges):
            for l, (i_prime, j_prime) in enumerate(edges):
                if self.graph[k]["edge_representative"] and self.graph[l]["edge_representative"]:
                    mat[k, l] = -self.swap_cost(i, j, i_prime, j_prime)
                else:
                    mat[k, l] = -np.inf
        log_probs = np.array([self.log_sum_exp(mat[k, :]) for k in range(len(edges))])
        return log_probs

    def log_probs_swap_second_edge(self, k1):
        i1, j1 = self.graph["ij"][k1]
        log_probs_second_edge = []

        # VECTORIZE THIS
        for k in range(len(self.graph)):
            i2, j2 = self.graph["ij"][k]
            log_probs_second_edge.append(-self.swap_cost(i1, j1, i2, j2))
        # only representative edges can be swapped
        log_probs_second_edge = np.array(log_probs_second_edge)
        log_probs_second_edge[~self.graph["edge_representative"]] = -np.inf
        return log_probs_second_edge

    def do_swap(self, k1, k2):
        ic("Before swap")
        ic(self.graph)
        i1, j1 = self.graph["ij"][k1]
        i2, j2 = self.graph["ij"][k2]

        ### do swap
        self.graph["ij"][k1] = i1, j2
        self.graph["ij"][k2] = i2, j1

        ### update type information about the edges
        
        k1_type = self.type(i1, j1)
        k2_type = self.type(i2, j2)
        self.graph[k1_type][k1] = False
        self.graph[k2_type][k2] = False
        
        k1_new_type = self.type(i1, j2)
        k2_new_type = self.type(i2, j1)
        self.graph[k1_new_type][k1] = True
        self.graph[k2_new_type][k2] = True
        
        ic("After swap")
        ic(self.graph)

        ### update on deletion
        edges = self.graph["ij"]

        target_row = np.array([i1, j1])
        edge_deleted = not np.any(np.all(edges == target_row, axis=1))
        if edge_deleted:
            self.update_on_deletion(i1, j1)
            ic(i1, j1, "deleted")
        else:
            # select new representative edge
            bin_bin_locs = np.where(self.graph["bin_bin"])[0]
            new_representative_index = bin_bin_locs[0]
            self.graph["edge_representative"][new_representative_index] = True
            ic("flow diminished")

        ic("After update on flow subtraction 1")
        ic(self.graph)

        target_row = np.array([i2, j2])
        if not np.any(np.all(edges == target_row, axis=1)):
            self.update_on_deletion(i2, j2)
            ic(i2, j2, "deleted")
        else:
            # select new representative edge
            bin_bin_locs = np.where(self.graph["bin_bin"])[0]
            new_representative_index = bin_bin_locs[0]
            self.graph["edge_representative"][new_representative_index] = True
            ic("flow diminished")

        ic("After update on flow subtraction 2")
        ic(self.graph)

        ### update on insertion
        mask = np.ones(len(self.graph["ij"]), dtype=bool)
        ic(k1, k2)
        mask[[k1, k2]] = False
        ic(mask)
        other_path = self.graph["ij"][mask]
        ic(other_path)

        target_row = np.array([i1, j2])
        edge_inserted = not np.any(np.all(other_path== target_row, axis=1))
        if edge_inserted:
            self.update_on_insertion(i1, j2)
        else:
            # take representative status from the other edge
            mask = np.ones(len(self.graph["ij"]), dtype=bool)
            mask[[k1, k2]] = False
            mask = mask & self.graph["bin_bin"]
            old_representative_index = np.where(mask)[0][0]
            self.graph["edge_representative"][old_representative_index] = False
        ic("After update on insertion 1")
        ic(self.graph)

        target_row = np.array([i2, j1])
        mask = np.ones(len(self.graph["ij"]), dtype=bool)
        mask[[k1, k2]] = False
        other_path = self.graph["ij"][mask]
        edge_inserted = not np.any(np.all(other_path== target_row, axis=1))
        if edge_inserted:
            self.update_on_insertion(i2, j1)
        else:
            # take representative status from the other edge
            mask = np.ones(len(self.graph["ij"]), dtype=bool)
            mask[[k1, k2]] = False
            mask = mask & self.graph["bin_bin"]
            old_representative_index = np.where(mask)[0][0]
            self.graph["edge_representative"][old_representative_index] = False


        ic("After update on insertion 2")
        ic(self.graph)


        ### compute new log probabilities for swapped entries
        self.add_log_probs(k1)
        self.add_log_probs(k2)
        return

    def update_on_deletion(self, i, j):
        """Update the graph when a representative edge is deleted."""
        type_ij = self.type(i, j)
        log_prob_key = f"log_p_swap_with_{type_ij}"
        # VECTORIZE THIS
        for k in range(len(self.graph)):
            i_, j_ = self.graph["ij"][k]
            self.graph[k][log_prob_key] = self.log_diff_exp(
                self.graph[k][log_prob_key],
                -self.swap_cost(i_, j_, i, j)
            )
            self.graph[k]["log_p_swap_total"] = self.log_sum_exp([
                self.graph[k]["log_p_swap_with_assigned"],
                self.graph[k]["log_p_swap_with_unassigned_first_set"],
                self.graph[k]["log_p_swap_with_unassigned_second_set"],
                self.graph[k]["log_p_swap_with_bin_bin"]
                ])

    def update_on_insertion(self, i, j):
        """Update the graph when a representative edge is inserted."""
        type_ij = self.type(i, j)
        log_prob_key = f"log_p_swap_with_{type_ij}"
        # VECTORIZE THIS
        for k in range(len(self.graph)):
            i_, j_ = self.graph["ij"][k]
            self.graph[k][log_prob_key] = self.log_sum_exp([
                self.graph[k][log_prob_key],
                -self.swap_cost(i_, j_, i, j)
            ])
            self.graph[k]["log_p_swap_total"] = self.log_sum_exp([self.graph[k]["log_p_swap_with_assigned"],
                self.graph[k]["log_p_swap_with_unassigned_first_set"],
                self.graph[k]["log_p_swap_with_unassigned_second_set"],
                self.graph[k]["log_p_swap_with_bin_bin"]])

        
    def add_log_probs(self, k):
        """
        Add the log probabilities of swapping the edge at index k with all other edges of the same type.
        """
        i, j = self.graph["ij"][k]
        for type_ in ["assigned", "unassigned_first_set", "unassigned_second_set", "bin_bin"]:
            log_prob_key = f"log_p_swap_with_{type_}"

            # VECTORIZE THIS
            # self.graph[k][log_prob_key] = self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_) in self.graph["ij"] if self.type(i_, j_) == type_])
            log_probs = []
            for l, (i_, j_) in enumerate(self.graph["ij"]):
                if self.type(i_, j_) == type_ and self.graph[l]["edge_representative"]:
                    log_probs.append(-self.swap_cost(i, j, i_, j_))
            self.graph[k][log_prob_key] = self.log_sum_exp(log_probs)

        self.graph[k]["log_p_swap_total"] = self.log_sum_exp([self.graph[k]["log_p_swap_with_assigned"], 
                                                                    self.graph[k]["log_p_swap_with_unassigned_first_set"], 
                                                                    self.graph[k]["log_p_swap_with_unassigned_second_set"], 
                                                                    self.graph[k]["log_p_swap_with_bin_bin"]])

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

        # VECTORIZE THIS
        for k, (i, j) in enumerate(graph["ij"]):
            state_type = self.type(i, j)
            if state_type == "assigned":
                graph[k]["log_p_swap_with_bin_bin"] += change
            elif state_type == "bin_bin":
                graph[k]["log_p_swap_with_assigned"] += change
            elif state_type == "unassigned_first_set":
                graph[k]["log_p_swap_with_unassigned_second_set"] -= change
            elif state_type == "unassigned_second_set":
                graph[k]["log_p_swap_with_unassigned_first_set"] -= change
            graph[k]["log_p_swap_total"] = self.log_sum_exp([graph[k]["log_p_swap_with_assigned"],
                                                                    graph[k]["log_p_swap_with_unassigned_first_set"],
                                                                    graph[k]["log_p_swap_with_unassigned_second_set"],
                                                                    graph[k]["log_p_swap_with_bin_bin"]])

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

        i1, j1 = self.graph["ij"][k1]
        i2, j2 = self.graph["ij"][k2]
        
        log_probs = self.log_prob_marginal()
        l1 = log_probs[k1] - self.log_sum_exp(log_probs)
        
        log_probs_second_edge = self.log_probs_swap_second_edge(k1)
        l2 = log_probs_second_edge[k2] - self.log_sum_exp(log_probs_second_edge)
    
        return l1 + l2

    def log_prob_reverse_swap(self, k1, k2):
        """Calculate the probability of reversing the swap of matching at index k1 with matching at index k2."""

        i1, j1 = self.graph["ij"][k1]
        i2, j2 = self.graph["ij"][k2]

        # do swap
        self.do_swap(k1, k2)
        
        # calculate the probablity of going back
        log_prob_swap = self.log_prob_swap(k1, k2)

        # undo swap
        self.do_swap(k1, k2)

        return log_prob_swap

    def log_sum_exp(self, values):
        if len(values) == 0:
            return float('-inf')
        else:
            return logsumexp(values)

    def log_diff_exp(self, a, b):
        """ Compute log(exp(a) - exp(b)) in a numerically stable way.
        This uses the identity log(exp(a) - exp(b)) = a + log(1 - exp(-|a - b|)). 
        """
        if a == -np.inf and b == -np.inf:
            return -np.inf
        else:
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

