import numpy as np
from math import lgamma
from tqdm import tqdm
from icecream import ic
import matplotlib.pyplot as plt
from numpy import log as ln
from ot import emd
from latent_state import LatentState

class MCM
    def __init__(self, x, y,
                 ln_gamma, ln_mu, ln_nu,
                 start_param,
                 swap_proposal_params,
                 param_log_prior,
                 param_proposal,
                 swap_proposal_parameters,
                 verbose=False,
                 save_latent_trajectory=False,
                 ):
        """ Initialize the MCMC sampler
        Parameters
        ----------
        x : array-like
            The first set of points
        y : array-like
            The second set of points
        ln_gamma : function
            The logartithm of the intensity function of the paired point process.
            The function needs to have three arguments: two point coordinates x, y and a parameter dictionary.
            The function needs to be vectorized in the first two arguments.
            The third argument needs to be a dictionary with the following keys: 'lam_gamma', 'lam_mu', 'lam_nu'.
            Additional parameters are also allowed.
        ln_mu : function
            The logartithm of the intensity function of the first point process. 
            The function needs to be vectorized in the first argument. Needs two inputs: a point coordinate x and a parameter dictionary.
            The second argument needs to be a dictionary with the following keys: 'lam_gamma', 'lam_mu', 'lam_nu'.
            Other parameters are also allowed.
        ln_nu : function
            The logartithm of the intensity function of the second point process. 
            The function needs to have two arguments: a point coordinate y and a parameter dictionary. Needs to be vectorized in the first argument.
            The second argument needs to be a dictionary with the following keys: 'lam_gamma', 'lam_mu', 'lam_nu'.
            Other parameters are also allowed.
        start_param : dict
            The initial parameters for the model. 
        param_proposal : function
            A function that proposes a new parameter given the current parameter. By default, the parameter remains unchanged
        param_log_prior : function
            The prior distribution for the parameters. By default, it is a the constant function 1. 
        save_latent_state_trajectory : bool
            If True, the latent state trajectory is saved. Else only some relevant statistics are saved.
            
        """
        # set verbosity
        if not verbose:
            ic.disable()
        else:
            ic.enable()
        
        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.param_proposal = param_proposal

        self.ln_gamma = ln_gamma
        self.ln_mu = ln_mu
        self.ln_nu = ln_nu
        self.param_log_prior = param_log_prior

        # index versions of the functions
        self.ln_gamma_index = lambda i, j, param: ln_gamma(self.x[i], self.y[j], param)
        self.ln_mu_index = lambda i, param: ln_mu(self.x[i], param)
        self.ln_nu_index = lambda j, param: ln_nu(self.y[j], param)

        self.latent_state = LatentState(self.nx, self.ny) # need to add things here 


        # here we store the trajectory of the parameters
        self.param_trajectory = [start_param]

        # we track the log probability of each state (up to unknown constant)
        self.probs = []

        # we track how often a move is accepted in the latent space and in the parameter space
        self.n_accepted_latent = 0
        self.n_accepted_param = 0

        # we track the number of pairs and the frequency of each edge in the matching
        self.pair_count_trajectory = []
        self.edge_freq_matrix = np.zeros((self.nx+1, self.ny+1))

        # optionally the hole latent state may be saved (but this is memory consuming)
        self.save_latent_trajectory = save_latent_trajectory
        if self.save_latent_trajectory:
            self.latent_trajectory = []

        # initialize the trajectory
        path = self.latent_state.numpy_path()
        self.update_trajectory(start_param, path, accepted_latent=False, accepted_param=False,
                save_latent=self.save_latent_trajectory)
        
    def run(self, n_samples=10000, burn_in=1000, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # burn-in period
        for _ in tqdm(range(burn_in), desc='Burn-in period'):
            accepted = self.MH_move()
        
        if burn_in > 0:
            # remove burn-in samples from the trajectory
            self.param_trajectory = self.param_trajectory[burn_in:]
            self.pair_count_trajectory = self.pair_count_trajectory[burn_in:]
            self.probs = self.probs[burn_in:]
            self.edge_freq_matrix = np.zeros((self.nx+1, self.ny+1))
            if self.save_latent_trajectory:
                self.latent_state_trajectory = self.latent_state_trajectory[burn_in:]

            # reset acceptance rate
            self.n_accepted_latent = 0
            self.n_accepted_param = 0
        
        # sampling period
        n_accepted = 0
        for _ in tqdm(range(n_samples), desc='Sampling'):
            accepted = self.MH_move()
                
        # store acceptance rate
        self.latent_acceptance_rate = self.n_accepted_latent/n_samples
        self.param_acceptance_rate = self.n_accepted_param/n_samples
        return 
    
    def MH_move(self):
        propose_par_prob = 0.5 # probability of proposing a change in the parameter space

        # decide weather to propose a move in the parameter space
        if np.random.random() < propose_par_prob:
            return self.MH_move_for_parameter()
        else:
            return self.MH_move_for_latent_variable()

    def MH_move_for_latent_variable(self):
        """ Perform a single Metropolis-Hastings move"""
        ic(" ")
        ic("new move begun")
        ic(" ")
        
        # get current parameter
        current_param = self.param_trajectory[-1]
        
        k0, k1 = self.latent_state.sample_swap()
        swap_log_prob = self.latent_state.log_prob_swap(k0, k1)
        reverse_swap_log_prob = self.latent_state.log_prob_reverse_swap(k0, k1)
        
        # likelihood ratio
        (i0, j0) = k0
        (i1, j1) = k1
        log_acceptance_ratio = - self.cost(i0, j1, current_param) - self.cost(i1, j0, current_param) + self.cost(i0, j0, current_param) + self.cost(i1, j1, current_param)

        # balance ratio
        log_acceptance_ratio += swap_log_prob - reverse_swap_log_prob
        u = ln(np.random.random())
        accepted = u < log_acceptance_ratio
        
        # print some information for debugging
        acceptance_prob = np.exp(log_acceptance_ratio)
        rewerse_swap_prob = np.exp(reverse_swap_log_prob)
        swap_prob = np.exp(swap_log_prob)
        ic(k0, k1, i0, j0, i1, j1, reverse_swap_prob, swap_prob, acceptance_prob, accepted)
        if accepted:
            # do swap
            self.latent_state.do_swap(k0, k1)

        # update trajectory
        paths = self.latent_state.numpy_path()
        self.update_trajectory(current_param, paths, accepted, accepted_param=False,
                save_latent=self.save_latent_trajectory)
            
        return accepted

    def MH_move_for_parameter(self):
        current_param = self.param_trajectory[-1]
        proposed_param = self.param_proposal(current_param)
        paths = self.latent_state.numpy_path()
        I, J = paths.T
        log_acceptance_ratio = np.sum(-self.cost(I, J, proposed_param) + self.cost(I, J, current_param))
        u = ln(np.random.random())

        accepted = u < log_acceptance_ratio

        if accepted:
            current_param = proposed_param

            # update latent state
            lam_gamma, lam_mu, lam_nu = current_param['lam_gamma'], current_param['lam_mu'], current_param['lam_nu']
            self.latent_state.update_intensity(lam_mu, lam_nu, lam_gamma)

        self.update_trajectory(current_param, paths, accepted_latent=False, accepted_param=accepted,
                save_latent=self.save_latent_trajectory)
        return accepted

    
    def cost(self, i, j, param):
        i = np.array(i)
        j = np.array(j)
        lam_gamma, lam_nu, lam_mu = param['lam_gamma'], param['lam_nu'], param['lam_mu']

        i = np.array(i)
        j = np.array(j)
        n_x = self.nx
        n_y = self.ny

        pair = np.logical_and(i < n_x, j < n_y)
        single_a = np.logical_and(i < n_x, j == n_y)
        single_b = np.logical_and(i == n_x, j < n_y)

        # to vectorize, I need functions to go through even if i == n_x or j == n_y
        # probably there is a better way of doing this though (?)
        i_m = np.minimum(i, n_x-1)
        j_m = np.minimum(j, n_y-1)

        ps = np.where(pair, self.ln_gamma_index(i_m, j_m, param), 0)

        ps = np.where(single_a, self.ln_mu_index(i_m, param), ps)
        ps = np.where(single_b, self.ln_nu_index(j_m, param), ps)

        ps -= lam_gamma/(n_x+n_y)
        ps -= lam_mu/(n_x+n_y)
        ps -= lam_nu/(n_x+n_y)

        ps -= lgamma(n_x+1)/(n_x+n_y)
        ps -= lgamma(n_y+1)/(n_x+n_y)

        ps += self.param_log_prior(param)/(n_x+n_y)

        # nan are assumed to have cost -inf
        np.nan_to_num(ps, copy=False, nan=-np.inf)

        return ps
    
    def disallowed_swaps(self, k1, k2, paths):
        """ We only allow swaps that effectively change the paths """
        I, J = paths.T
        k1 = np.array(k1)
        k2 = np.array(k2)
        I, J = paths.T
        i1, j1 = I[k1], J[k1]
        i2, j2 = I[k2], J[k2]
        disallowed_x = (i1 == i2)
        disallowed_y = (j1 == j2)
        res = np.logical_or(disallowed_x, disallowed_y)
        return res
    
    def plot_pair_count(self, ax=None):
        n_max = np.minimum(self.nx, self.ny)
        bins = np.arange(n_max+2)-0.5
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.pair_count_trajectory, bins=bins, density=True)
        ax.set_xlabel("Number of pairs")
        ax.set_ylabel("Frequency")

    def EE(self):
        chain_length = np.sum(self.edge_freq_matrix)
        return self.edge_freq_matrix/chain_length
    
    def pairing_probabilities(self):
        chain_length = np.sum(self.edge_freq_matrix)/(self.nx+self.ny)
        return self.edge_freq_matrix[:-1, :-1]/chain_length

    def update_trajectory(self, params, paths, accepted_latent, accepted_param, save_latent):
        """ update the trajectory with the new state """
        self.param_trajectory.append(params)
        if save_latent:
            self.latent_trajectory.append(paths)
        self.update_edge_freq_matrix(paths)
        n_pairs = np.sum(np.logical_and(paths[:, 0] < self.nx, paths[:, 1] < self.ny))
        self.pair_count_trajectory.append(n_pairs)
        log_prob = np.sum(self.cost(paths[:, 0], paths[:, 1], params))
        self.probs.append(log_prob)
        self.n_accepted_latent += accepted_latent
        self.n_accepted_param += accepted_param
        return

    def update_edge_freq_matrix(self, paths):
        I, J = paths.T  
        self.edge_freq_matrix[I, J] += 1

        # correct for adding the bin bin edge multiple times
        n_pair = np.sum(np.logical_and(I < self.nx, J < self.ny))
        if n_pair > 0:
            self.edge_freq_matrix[-1, -1] -= 1
            self.edge_freq_matrix[-1, -1] += n_pair
        return
   