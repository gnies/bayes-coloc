import numpy as np
from math import lgamma
from tqdm import tqdm
from icecream import ic
from numpy import log as ln
from .latent_space import LatentState

class MCMC:
    def __init__(self, x, y,
                 l_xy, l_x, l_y,
                 start_params,
                 params_swap,
                 param_log_prior,
                 params_proposal,
                 verbose=False,
                 save_latent_trajectory=False,
                 ):
        """
        x: list of x values
        y: list of y values
        l_xy: function that returns the log likelihood of a pair of x and y values
        l_x: function that returns the log likelihood of a single x value
        l_y: function that returns the log likelihood of a single y value
        start_params: dictionary with the starting parameters
        This dictionary should contain the following keys:
        alpha: the intensity of the number of single markers in the first channel.
        beta: the intensity of the number of single markers in the second channel.
        gamma: the intensity of the number of pairs of markers.
        params_proposal: function that proposes a new set of parameters given the current parameters.
        params_swap: function that proposes a new set of parameters given the current parameters.
        It must also contain keys alpha, beta, and gamma, which are required to be the same as the starting parameters.
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
        self.param_proposal = params_proposal

        self.l_xy = l_xy
        self.l_x = l_x
        self.l_y = l_y

        self.param_log_prior = param_log_prior

        # index versions of the functions
        self.l_xy_index = lambda i, j, param: l_xy(self.x[i], self.y[j], param)
        self.l_x_index = lambda i, param: l_x(self.x[i], param)
        self.l_y_index = lambda j, param: l_y(self.y[j], param)


        # check if swap_proposal_params have the same intensity as the start_param
        self.check_start_params(start_params, params_swap)

        # here we store the trajectory of the parameters
        self.param_trajectory = [start_params]

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

        ##### initialize the latent state #####
        I = np.arange(self.nx)
        J = np.arange(self.ny)

        # create cost that will be used for the swap proposal in the latent state
        proposal_cost = np.zeros((self.nx+1, self.ny+1))
        proposal_cost[-1, :-1] = -self.l_x(I, params_swap)
        proposal_cost[:-1, -1] = -self.l_y(J, params_swap)
        for i in range(self.nx):
            proposal_cost[i, :-1] -= self.l_xy_index(i, J, params_swap)


        alpha, beta, gamma = start_params['alpha'], start_params['beta'], start_params['gamma']
        self.latent_state = LatentState(self.nx, self.ny, alpha, beta, gamma , proposal_cost)

        #### initialize the trajectory
        path = self.latent_state.numpy_path()
        self.save_to_trajectory(start_params, path, accepted_latent=False, accepted_param=False,
                save_latent=self.save_latent_trajectory)
        
    def run(self, n_samples=10000, burn_in=1000):
            
        # burn-in period
        for _ in tqdm(range(burn_in), desc='Burn-in period'):
            accepted = self.MH_move()
        
        if burn_in > 0:
            # remove burn-in samples from the trajectory
            self.param_trajectory = self.param_trajectory[burn_in:]
            self.pair_count_trajectory = self.pair_count_trajectory[burn_in:]
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

    def sample_swap(self):
        """Sample a pair of keys to swap."""
        graph = self.latent_state.graph.numpy_graph()
        I = graph["i"]
        J = graph["j"]

        log_probs = self.latent_state.log_prob_marginal()

        index1 = gumel_max(log_probs)
        i1, j1 = I[index1], J[index1]
        key1 = (i1, j1)
    
        log_probs = -self.latent_state.swap_cost_with_edges(i1, j1, I, J)
        index2 = gumel_max(log_probs)
        key2 = (I[index2], J[index2])
    
        return key1, key2

    def MH_move_for_latent_variable(self):
        """ Perform a single Metropolis-Hastings move"""
        ic(" ")
        ic("new move begun")
        ic(" ")
        
        # get current parameter
        current_param = self.param_trajectory[-1]
        
        k0, k1 = self.sample_swap()
        swap_log_prob = self.latent_state.log_prob_swap(k0, k1)
        reverse_swap_log_prob = self.latent_state.log_prob_reverse_swap(k0, k1)
        
        # likelihood ratio
        (i0, j0) = k0
        (i1, j1) = k1
        log_acceptance_ratio = - self.cost(i0, j1, current_param) - self.cost(i1, j0, current_param) + self.cost(i0, j0, current_param) + self.cost(i1, j1, current_param)

        # balance ratio
        log_acceptance_ratio +=  +reverse_swap_log_prob - swap_log_prob
        u = ln(np.random.random())
        accepted = u < log_acceptance_ratio
        
        if accepted:
            # do swap
            self.latent_state.do_swap(k0, k1)

        # update trajectory
        paths = self.latent_state.numpy_path()
        self.save_to_trajectory(current_param, paths, accepted, accepted_param=False,
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
            alpha, beta, gamma = current_param['alpha'], current_param['beta'], current_param['gamma']
            self.latent_state.update_intensities(alpha, beta, gamma)

        self.save_to_trajectory(current_param, paths, accepted_latent=False, accepted_param=accepted,
                save_latent=self.save_latent_trajectory)
        return accepted

    
    def cost(self, i, j, param):
        i = np.array(i)
        j = np.array(j)
        alpha, beta, gamma = param['alpha'], param['beta'], param['gamma']
            
        # if the parameters have probability zero, then the cost is -inf
        # this also avoids evaluating the likelihood on invalid parameters
        if self.param_log_prior(param) == -np.inf:
            return np.inf

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

        ps = np.where(pair, self.l_xy_index(i_m, j_m, param), 0)
        ps[pair] += ln(gamma)

        ps = np.where(single_a, self.l_x_index(i_m, param), ps)
        ps[single_a] += ln(alpha)

        ps = np.where(single_b, self.l_y_index(j_m, param), ps)
        ps[single_b] += ln(beta)

        ps -= gamma/(n_x+n_y)
        ps -= alpha/(n_x+n_y)
        ps -= beta/(n_x+n_y)

        ps -= lgamma(n_x+1)/(n_x+n_y)
        ps -= lgamma(n_y+1)/(n_x+n_y)

        ps += self.param_log_prior(param)/(n_x+n_y)

        return -ps
    
    def EE(self):
        chain_length = np.sum(self.edge_freq_matrix)
        return self.edge_freq_matrix/chain_length
    
    def pairing_probabilities(self):
        chain_length = np.sum(self.edge_freq_matrix)/(self.nx+self.ny)
        return self.edge_freq_matrix[:-1, :-1]/chain_length

    def save_to_trajectory(self, params, paths, accepted_latent, accepted_param, save_latent):
        """ update the trajectory with the new state """
        self.param_trajectory.append(params)
        if save_latent:
            self.latent_trajectory.append(paths)
        self.update_edge_freq_matrix(paths)
        n_pairs = np.sum(np.logical_and(paths[:, 0] < self.nx, paths[:, 1] < self.ny))
        self.pair_count_trajectory.append(n_pairs)
        log_prob = np.sum(self.cost(paths[:, 0], paths[:, 1], params))
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

    def check_start_params(self, start_params, params_swap):
        # check if the parameters are valid
        assert 'alpha' in start_params
        assert 'beta' in start_params
        assert 'gamma' in start_params

        # check if the swap proposal parameters are valid
        assert 'alpha' in params_swap
        assert 'beta' in params_swap
        assert 'gamma' in params_swap

        # check if the swap proposal parameters are valid
        if params_swap['alpha'] != start_params['alpha']:
            raise ValueError("The alpha value of the swap proposal parameters is different from the starting parameters")
        if params_swap['beta'] != start_params['beta']:
            raise ValueError("The beta value of the swap proposal parameters is different from the starting parameters")
        if params_swap['gamma'] != start_params['gamma']:
            raise ValueError("The gamma value of the swap proposal parameters is different from the starting parameters")
   
def gumel_max(log_probs):
    """ Use the Gumbel-Max trick to sample an index directly from the unscaled log probabilities."""
    log_probs = np.asarray(log_probs)
    real_log_probs_indices = np.where(log_probs > -np.inf)[0]
    real_log_probs = log_probs[real_log_probs_indices]
    gumbels = np.random.gumbel(size=len(real_log_probs))
    index = real_log_probs_indices[np.argmax(real_log_probs + gumbels)]
    return index
