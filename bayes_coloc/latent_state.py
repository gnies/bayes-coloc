import math
import numpy as np
from scipy.special import logsumexp, expm1 

class LatentState:
    def __init__(self, n, m, alpha, beta, gamma, cost):
        """ algorithm is initialized with no assignments, only unassigned points """
        self.n = n
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cost = cost
        self.states = {}
        self.initialize_states()

    def initialize_states(self):
        for i in range(self.n):
            self.add_entry((i, self.m), 1)
        for j in range(self.m):
            self.add_entry((self.n, j), 1)

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
        if x < np.log(2):
            return np.log(-expm1(-x))
        else:
            return np.log1p(-np.exp(-x))

    def swap_cost(self, i, j, i_prime, j_prime):
        if i != i_prime and j != j_prime:
            res = (self.cost[i, j_prime] + self.cost[i_prime, j] -
                   self.cost[i, j] - self.cost[i_prime, j_prime])
        else:
            return float('inf')

        intensity_cost_diff = math.log(self.gamma) - math.log(self.alpha) - math.log(self.beta)
        if (i, j) in self.states and self.states[(i, j)]["type"] == "assigned" and \
           (i_prime, j_prime) in self.states and self.states[(i_prime, j_prime)]["type"] == "bin_bin":
            res += intensity_cost_diff
        elif (i, j) in self.states and self.states[(i, j)]["type"] == "bin_bin" and \
             (i_prime, j_prime) in self.states and self.states[(i_prime, j_prime)]["type"] == "assigned":
            res += intensity_cost_diff
        else:
            intensity_cost_diff = -math.log(self.gamma) + math.log(self.alpha) + math.log(self.beta)
            if ((i, j) in self.states and self.states[(i, j)]["type"] == "unassigned_first_set" and
                (i_prime, j_prime) in self.states and self.states[(i_prime, j_prime)]["type"] == "unassigned_second_set") or \
               ((i, j) in self.states and self.states[(i, j)]["type"] == "unassigned_second_set" and
                (i_prime, j_prime) in self.states and self.states[(i_prime, j_prime)]["type"] == "unassigned_first_set"):
                res -= intensity_cost_diff
        return res

    def add_entry(self, key, flow):
        i, j = key
        state_type = self.state_type(key)
        self.states[key] = {
            "flow": flow,
            "type": state_type,
            "log_prob_swap_with_assigned": self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_), state in self.states.items() if state["type"] == "assigned"]),
            "log_prob_swap_with_unassigned_first_set": self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_), state in self.states.items() if state["type"] == "unassigned_first_set"]),
            "log_prob_swap_with_unassigned_second_set": self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_), state in self.states.items() if state["type"] == "unassigned_second_set"]),
            "log_prob_swap_with_bin_bin": self.log_sum_exp([-self.swap_cost(i, j, i_, j_) for (i_, j_), state in self.states.items() if state["type"] == "bin_bin"])
        }
        self.states[key]["log_prob_swap_total"] = self.log_sum_exp([
            self.states[key]["log_prob_swap_with_assigned"],
            self.states[key]["log_prob_swap_with_unassigned_first_set"],
            self.states[key]["log_prob_swap_with_unassigned_second_set"],
            self.states[key]["log_prob_swap_with_bin_bin"]
        ])
        self.update_log_probs_on_add(key)

    def remove_entry(self, key):
        if key in self.states:
            del self.states[key]
            self.update_log_probs_on_remove(key)

    def update_log_probs_on_add(self, key):
        i, j = key
        for (i_, j_), state in self.states.items():
            if i != i_ and j != j_:
                state_type = self.states[key]["type"]
                log_prob_key = f"log_prob_swap_with_{state_type}"
                self.states[(i_, j_)][log_prob_key] = np.logaddexp(
                    self.states[(i_, j_)][log_prob_key],
                    -self.swap_cost(i_, j_, i, j)
                )
                self.states[(i_, j_)]["log_prob_swap_total"] = self.log_sum_exp([
                    self.states[(i_, j_)]["log_prob_swap_with_assigned"],
                    self.states[(i_, j_)]["log_prob_swap_with_unassigned_first_set"],
                    self.states[(i_, j_)]["log_prob_swap_with_unassigned_second_set"],
                    self.states[(i_, j_)]["log_prob_swap_with_bin_bin"]
                ])

    def update_log_probs_on_remove(self, key):
        i, j = key
        for (i_, j_), state in self.states.items():
            if i != i_ and j != j_:
                state_type = self.state_type(key)  
                log_prob_key = f"log_prob_swap_with_{state_type}"
                log_prob = self.states[(i_, j_)][log_prob_key]
                self.states[(i_, j_)][log_prob_key] = self.log_diff_exp(
                    log_prob,
                    -self.swap_cost(i_, j_, i, j)
                )
                self.states[(i_, j_)]["log_prob_swap_total"] = self.log_sum_exp([
                    self.states[(i_, j_)]["log_prob_swap_with_assigned"],
                    self.states[(i_, j_)]["log_prob_swap_with_unassigned_first_set"],
                    self.states[(i_, j_)]["log_prob_swap_with_unassigned_second_set"],
                    self.states[(i_, j_)]["log_prob_swap_with_bin_bin"]
                ])

    def update_intensities(self, alpha, beta, gamma):
        """Update the intensities alpha, beta, and gamma, and adjust relevant log probabilities."""
        a_old, b_old, c_old = self.alpha, self.beta, self.gamma
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
    
        # Compute the change in log scale
        change = math.log(alpha) + math.log(beta) - math.log(gamma) - math.log(a_old) - math.log(b_old) + math.log(c_old)
    
        # Update the log probabilities for each state
        for key, value in self.states.items():
            value["log_prob_swap_with_assigned"] += change
            value["log_prob_swap_with_bin_bin"] += change
            value["log_prob_swap_with_unassigned_first_set"] -= change
            value["log_prob_swap_with_unassigned_second_set"] -= change
        return 

    def state_type(self, key):
        i, j = key
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
            self.add_entry(key, flow)

    def get_key_flow(self, key):
        if key not in self.states:
            self.add_entry(key, 0)
        return self.states[key]["flow"]

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

    def return_numpy_path(self):
        path = np.empty((len(self.states), 2), dtype=int)
        for idx, (key, value) in enumerate(self.states.items()):
            path[idx, :] = key

    def gumel_max(self, log_probs):
        """ Use the Gumbel-Max trick to sample an index directly from the unscaled log probabilities."""
        log_probs = np.asarray(log_probs)
        real_log_probs_indices = np.where(log_probs > -np.inf)[0]
        real_log_probs = log_probs[real_log_probs_indices]
        gumbels = np.random.gumbel(size=len(real_log_probs))
        index = real_log_probs_indices[np.argmax(real_log_probs + gumbels)]
        return index
    
    def sample_swap(self):
        """Sample a pair of keys to swap."""
        keys = list(self.states.keys())
        log_probs = np.array([self.states[key]["log_prob_swap_total"] for key in keys])
        index1 = self.gumel_max(log_probs)
        key1 = keys[index1]
    
        log_probs = [-self.swap_cost(*key1, *key) for key in keys]
        index2 = self.gumel_max(log_probs)
        key2 = keys[index2]
    
        return key1, key2
    
    def log_prob_swap(self, key1, key2):
        """Calculate the probability of swapping key1 with key2."""
        keys = list(self.states.keys())
        log_probs = [self.states[key1]["log_prob_swap_total"] + self.swap_cost(*key1, *key) for key in keys]
        l1 = log_probs[keys.index(key2)] - self.log_sum_exp(log_probs)
    
        log_probs = [self.states[key2]["log_prob_swap_total"] + self.swap_cost(*key2, *key) for key in keys]
        l2 = log_probs[keys.index(key2)] - self.log_sum_exp(log_probs)
    
        return l1 + l2

    def log_prob_reverse_swap(self, key1, key2):
        """Calculate the probability of reversing the swap of key1 with key2."""
        (i, j), (i_prime, j_prime) = key1, key2
        key1_new = (i, j_prime)
        key2_new = (i_prime, j)

        # to avoid numerical instability, we make a copy of the current state
        copy = self.states.copy()
        self.do_swap(key1, key2)
        log_prob_swap = self.log_prob_swap(key1_new, key2_new)

        # undo swap
        # self.do_swap(key1_new, key2_new)
        self.states = copy

        return log_prob_swap


    def numpy_path(self):
        """Return a numpy array representing the assignments."""
        path = np.empty((len(self.states), 2), dtype=int)
        for idx, (key, value) in enumerate(self.states.items()):
            i, j = key
            flow = value["flow"]
            for _ in range(flow):
                path[idx, :] = (i, j)
        return path
    
