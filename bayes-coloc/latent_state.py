import math
import numpy as np
from collections import defaultdict

class LatentState:
    def __init__(self, n, m, alpha, beta, gamma, cost):
        self.n = n
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.cost = cost

        self.assigned = {}
        self.unassigned_first_set = {}
        self.unassigned_second_set = {}
        self.bin_bin = {}

    def initial_state(self):
        """Initialize the state with all points unassigned."""
        for i in range(self.n):
            self.add_entry((i, self.m), 1)

        for j in range(self.m):
            self.add_entry((self.n, j), 1)

    def copy(self):
        state_copy = LatentState(self.n, self.m, self.alpha, self.beta, self.gamma, self.cost)
        state_copy.assigned = self.assigned.copy()
        state_copy.unassigned_first_set = self.unassigned_first_set.copy()
        state_copy.unassigned_second_set = self.unassigned_second_set.copy()
        state_copy.bin_bin = self.bin_bin.copy()
        return state_copy
    
    def log_sum_exp(self, values):
        """Calculate log-sum-exp for numerical stability."""
        max_val = max(values)
        return max_val + math.log(sum(math.exp(val - max_val) for val in values))

    def swap_cost(self, i, j, i_prime, j_prime):
        """Calculate swap cost based on given conditions."""
        if i != i_prime and j != j_prime:
            res = (self.cost[i, j_prime] + self.cost[i_prime, j] -
                   self.cost[i, j] - self.cost[i_prime, j_prime])
        else:
            res = float('inf')

        # Determine intensity cost difference
        intensity_cost_diff = np.log(self.gamma) - np.log(self.alpha) - np.log(self.beta)
        
        # Adjust swap cost for specific match changes
        if (i, j) in self.assigned and (i_prime, j_prime) in self.bin_bin:
            res += intensity_cost_diff
        elif (i, j) in self.bin_bin and (i_prime, j_prime) in self.assigned:
            res += intensity_cost_diff

        # Adjust cost if creating a match
        intensity_cost_diff = -np.log(self.gamma) + np.log(self.alpha) + np.log(self.beta)
        if (i, j) in self.unassigned_first_set and (i_prime, j_prime) in self.unassigned_second_set:
            res -= intensity_cost_diff
        elif (i, j) in self.unassigned_second_set and (i_prime, j_prime) in self.unassigned_first_set:
            res -= intensity_cost_diff
        return res

    def add_entry(self, key, flow):
        i, j = key
        if i < self.n and j < self.m:
            target_dict = self.assigned
            swap_field = 'log_prob_swap_with_assigned'
        elif i < self.n and j == self.m:
            target_dict = self.unassigned_first_set
            swap_field = 'log_prob_swap_with_unassigned_first_set'
        elif i == self.n and j < self.m:
            target_dict = self.unassigned_second_set
            swap_field = 'log_prob_swap_with_unassigned_second_set'
        else:
            target_dict = self.bin_bin
            swap_field = 'log_prob_swap_with_bin_bin'
        
        log_probs = {
            "log_prob_swap_with_assigned": -float('inf') if not self.assigned else self.log_sum_exp(
                [-self.swap_cost(i, j, i_, j_) for i_, j_ in self.assigned.keys()]
            ),
            "log_prob_swap_with_unassigned_first_set": -float('inf') if not self.unassigned_first_set else self.log_sum_exp(
                [-self.swap_cost(i, j, i_, j_) for i_, j_ in self.unassigned_first_set.keys()]
            ),
            "log_prob_swap_with_unassigned_second_set": -float('inf') if not self.unassigned_second_set else self.log_sum_exp(
                [-self.swap_cost(i, j, i_, j_) for i_, j_ in self.unassigned_second_set.keys()]
            ),
            "log_prob_swap_with_bin_bin": -float('inf') if not self.bin_bin else self.log_sum_exp(
                [-self.swap_cost(i, j, i_, j_) for i_, j_ in self.bin_bin.keys()]
            )
        }
        
        target_dict[key] = {
            "flow": flow,
            **log_probs
        }
        self.update_log_probs_on_add(key, swap_field)

    def update_log_probs_on_add(self, key, swap_field):
        i, j = key
        for dct in [self.assigned, self.unassigned_first_set, self.unassigned_second_set, self.bin_bin]:
            for (i_, j_) in dct.keys():
                if i != i_ and j != j_:
                    dct[(i_, j_)][swap_field] = self.log_sum_exp([
                        dct[(i_, j_)][swap_field],
                        -self.swap_cost(i_, j_, i, j)
                    ])

    def remove_entry(self, key):
        i, j = key
        if i < self.n and j < self.m:
            target_dict = self.assigned
            swap_field = 'log_prob_swap_with_assigned'
        elif i < self.n and j == self.m:
            target_dict = self.unassigned_first_set
            swap_field = 'log_prob_swap_with_unassigned_first_set'
        elif i == self.n and j < self.m:
            target_dict = self.unassigned_second_set
            swap_field = 'log_prob_swap_with_unassigned_second_set'
        else:
            target_dict = self.bin_bin
            swap_field = 'log_prob_swap_with_bin_bin'
        
        if key in target_dict:
            del target_dict[key]
            self.update_log_probs_on_remove(key, swap_field)

    def update_log_probs_on_remove(self, key, swap_field):
        i, j = key
        for dct in [self.assigned, self.unassigned_first_set, self.unassigned_second_set, self.bin_bin]:
            for (i_, j_) in dct.keys():
                if i != i_ and j != j_:
                    dct[(i_, j_)][swap_field] += math.log(
                        1 - math.exp(
                            -self.swap_cost(i_, j_, i, j) - dct[(i_, j_)][swap_field]
                        )
                    )

    def set_key_flow(self, key, flow):
        i, j = key
        if flow == 0:
            self.remove_entry(key)
        else:
            self.add_entry(key, flow)

    def get_key_flow(self, key):
        i, j = key
        if i < self.n and j < self.m:
            target_dict = self.assigned
        elif i < self.n and j == self.m:
            target_dict = self.unassigned_first_set
        elif i == self.n and j < self.m:
            target_dict = self.unassigned_second_set
        else:
            target_dict = self.bin_bin
        if key not in target_dict:
            self.add_entry(key, 0)

    def do_swap(self, key1, key2): 
        i1, j1 = key1
        i2, j2 = key2

        m1 = self.get_key_flow(key1)
        m2 = self.get_key_flow(key2)

        self.set_key_flow(key1, m1-1)
        self.set_key_flow(key2, m2-1)

        new_key1 = (i1, j2)
        new_key2 = (i2, j1)

        n1 = self.get_key_flow(new_key1)
        n2 = self.get_key_flow(new_key2)

        self.set_key_flow(new_key1, n1+1)
        self.set_key_flow(new_key2, n2+1)

    def total_cost(self):
        return sum(
            flow * dct[key]['cost']
            for dct in [self.assigned, self.unassigned_first_set, self.unassigned_second_set, self.bin_bin]
            for key, flow in dct.items()
        )

    def n_pairs(self):
        return len(self.assigned)

    def update_intensities(self, alpha, beta, gamma):
        a_old, b_old, c_old = self.alpha, self.beta, self.gamma
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        change = np.log(alpha) + np.log(beta) - np.log(gamma) - np.log(a_old) - np.log(b_old) + np.log(c_old)
        for key in self.assigned.keys():
            self.assigned[key]["log_prob_swap_with_assigned"] += change
        for key in self.bin_bin.keys():
            self.bin_bin[key]["log_prob_swap_with_bin_bin"] += change
        for key in self.unassigned_first_set.keys():
            self.unassigned_first_set[key]["log_prob_swap_with_unassigned_second_set"] -= change
        for key in self.unassigned_second_set.keys():
            self.unassigned_second_set[key]["log_prob_swap_with_unassigned_first_set"] -= change

    def sample_swap(self):
        keys = list(self.assigned.keys()) + list(self.bin_bin.keys()) + list(self.unassigned_first_set.keys()) + list(self.unassigned_second_set.keys())
        log_probs = [self.log_sum_exp([dct[key][field] for field in ["log_prob_swap_with_assigned", "log_prob_swap_with_bin_bin", 
                    "log_prob_swap_with_unassigned_first_set", "log_prob_swap_with_unassigned_second_set"]])
                     for dct in [self.assigned, self.bin_bin, self.unassigned_first_set, self.unassigned_second_set]
                     for key in dct.keys()]

        probs = np.exp(log_probs - np.max(log_probs))
        probs /= np.sum(probs)
        key0 = np.random.choice(keys, p=probs)

        log_probs = [self.swap_cost(*key0, *key) for key in keys]
        probs = np.exp(log_probs - np.max(log_probs))
        probs /= np.sum(probs)
        key1 = np.random.choice(keys, p=probs)

        return key0, key1

    def n_pairs(self):
        return len(self.assigned)

    def return_numpy_path(self):
        path = np.empty((self.n + self.m, 2))
        k = 0
        for dct in [self.assigned, self.bin_bin, self.unassigned_first_set, self.unassigned_second_set]:
            for key in dct.keys():
                i, j = key
                flow = dct[key]["flow"]
                for _ in range(flow):
                    path[k, :] = (i, j)
                    k += 1
        return path

