import math
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
        self.state = {}
        self.initialize_states()

    def initialize_states(self):
        for i in range(self.n):
            self.add_entry((i, self.m), 1)
        for j in range(self.m):
            self.add_entry((self.n, j), 1)

    def compute_log_prob_matrix(self):
        keys = list(self.state.keys())
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
        keys = list(self.state.keys())
        log_probs = np.array([self.state[key]["log_prob_swap_total"] for key in keys])
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
        if self.type(i, j) == "assigned" and self.type(i_prime, j_prime) == "bin_bin":
            res += intensity_cost_diff
        elif self.type(i, j) == "bin_bin" and self.type(i_prime, j_prime) == "assigned":
            res += intensity_cost_diff

        elif self.type(i, j) == "unassigned_first_set" and self.type(i_prime, j_prime) == "unassigned_second_set":
            res -= intensity_cost_diff
        elif self.type(i, j) == "unassigned_second_set" and self.type(i_prime, j_prime) == "unassigned_first_set":
            res -= intensity_cost_diff
        return res

    def add_entry(self, key, flow):
        i, j = key
        state_type = self.type(i, j)
        self.state[key] = {
            "flow": flow,
            "type": state_type,
            "log_prob_swap_with_assigned":              self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_), di in self.state.items() if di["type"] == "assigned"]),
            "log_prob_swap_with_unassigned_first_set":  self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_), di in self.state.items() if di["type"] == "unassigned_first_set"]),
            "log_prob_swap_with_unassigned_second_set": self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_), di in self.state.items() if di["type"] == "unassigned_second_set"]),
            "log_prob_swap_with_bin_bin":               self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_), di in self.state.items() if di["type"] == "bin_bin"])
        }
        self.state[key]["log_prob_swap_total"] = self.log_sum_exp([
            self.state[key]["log_prob_swap_with_assigned"],
            self.state[key]["log_prob_swap_with_unassigned_first_set"],
            self.state[key]["log_prob_swap_with_unassigned_second_set"],
            self.state[key]["log_prob_swap_with_bin_bin"]
        ])
        self.update_log_probs_on_add(key)

    def remove_entry(self, key):
        if key in self.state:
            del self.state[key]
            self.update_log_probs_on_remove(key)

    def update_log_probs_on_add(self, key):
        i, j = key
        for (i_, j_) in self.state.keys():
            if i != i_ and j != j_:
                state_type = self.state[key]["type"]
                log_prob_key = f"log_prob_swap_with_{state_type}"
                self.state[(i_, j_)][log_prob_key] = np.logaddexp(
                    self.state[(i_, j_)][log_prob_key],
                    -self.swap_cost(i_, j_, i, j)
                )
                self.state[(i_, j_)]["log_prob_swap_total"] = self.log_sum_exp([
                    self.state[(i_, j_)]["log_prob_swap_with_assigned"],
                    self.state[(i_, j_)]["log_prob_swap_with_unassigned_first_set"],
                    self.state[(i_, j_)]["log_prob_swap_with_unassigned_second_set"],
                    self.state[(i_, j_)]["log_prob_swap_with_bin_bin"]
                ])

    def update_log_probs_on_remove(self, key):
        i, j = key
        for (i_, j_), state in self.state.items():
            if i != i_ and j != j_:
                state_type = self.type(i, j)  
                log_prob_key = f"log_prob_swap_with_{state_type}"
                log_prob = self.state[(i_, j_)][log_prob_key]
                self.state[(i_, j_)][log_prob_key] = self.log_diff_exp(
                    log_prob,
                    -self.swap_cost(i_, j_, i, j)
                )
                self.state[(i_, j_)]["log_prob_swap_total"] = self.log_sum_exp([
                    self.state[(i_, j_)]["log_prob_swap_with_assigned"],
                    self.state[(i_, j_)]["log_prob_swap_with_unassigned_first_set"],
                    self.state[(i_, j_)]["log_prob_swap_with_unassigned_second_set"],
                    self.state[(i_, j_)]["log_prob_swap_with_bin_bin"]
                ])

    def update_intensities(self, alpha, beta, gamma):
        """Update the intensities alpha, beta, and gamma, and adjust relevant log probabilities."""
        a_old, b_old, c_old = self.alpha, self.beta, self.gamma
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
    
        # Compute the change in log scale
        change = ln(alpha) + ln(beta) - ln(gamma) - ln(a_old) - ln(b_old) + ln(c_old)
    
        # Update the log probabilities for each state
        for key, value in self.state.items():
            value["log_prob_swap_with_assigned"] += change
            value["log_prob_swap_with_bin_bin"] += change
            value["log_prob_swap_with_unassigned_first_set"] -= change
            value["log_prob_swap_with_unassigned_second_set"] -= change
        return 

    def type(self, i, j):
        if i < self.n and j < self.m:
            return "assigned"
        elif i == self.n and j < self.m:
            return "unassigned_second_set"
        elif i < self.n and j == self.m:
            return "unassigned_first_set"
        else:
            return "bin_bin"

    def set_key_flow(self, key, flow):
        if flow == 0:
            self.remove_entry(key)
        else:
            self.state[key]["flow"] = flow
            

    def get_key_flow(self, key):
        if key not in self.state:
            self.add_entry(key, 0)
        return self.state[key]["flow"]

    def do_swap(self, key1, key2):
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
        return list(self.state.keys())

    def return_numpy_path(self):
        path = np.empty((len(self.state), 2), dtype=int)
        for idx, (key, value) in enumerate(self.state.items()):
            path[idx, :] = key

    
    def log_prob_swap(self, key1, key2):
        """Calculate the probability of swapping key1 with key2."""
        keys = list(self.state.keys())
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
        path = np.empty((len(self.state), 2), dtype=int)
        for idx, (key, value) in enumerate(self.state.items()):
            i, j = key
            flow = value["flow"]
            for _ in range(flow):
                path[idx, :] = (i, j)
        return path
    
