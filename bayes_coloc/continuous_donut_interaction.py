from .optimized_swap_proposal import MCMC
from .slow_latent_space import LatentState
import numpy as np
from icecream import ic

from numpy import log as ln
from scipy.stats import ncx2

class DonutInteraction(MCMC):
    def __init__(self, x, y,
            start_params,
            swap_proposal_params,
            log_prior,
            param_proposal,
            verbose=False,
            save_latent_trajectory=False 
            ):
        """ MCMC sampler for partially paired Poisson point process with
        Gaussian interaction.
        Parameters
        ----------
        x : array_like
            The x coordinates of the data points.
        y : array_like
            The y coordinates of the data points.
        start_params : dict
            The starting parameters for the model. The dictionary should contain the following keys:
            - 'scale': The scale of the interaction.
            - 'radius': The radius of the interaction
            - 'alpha': The intensity of the single x points.
            - 'beta': The intensity of the single y points.
            - 'gamma': The intensity of the interaction.
            - 'area': The area of the observation window/region of interest.
        log_prior : function
            The log prior density function for the parameters.
        param_proposal : function
            The proposal function for the parameters.
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

        self.l_xy = l_xy
        self.l_x = l_x
        self.l_y = l_y

        self.param_log_prior = log_prior

        # index versions of the functions
        self.l_xy_index = lambda i, j, param: l_xy(self.x[i], self.y[j], param)
        self.l_x_index = lambda i, param: l_x(self.x[i], param)
        self.l_y_index = lambda j, param: l_y(self.y[j], param)

        # check if swap_proposal_params have the same intensity as the start_params
        self.check_start_params(start_params, swap_proposal_params)

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

        # create cost that will be used for the swap proposal in the latent state
        proposal_cost = np.empty((self.nx+1, self.ny+1))
        for i in range(self.nx):
            for j in range(self.ny):
                proposal_cost[i, j] = -self.l_xy_index(i, j, swap_proposal_params)
        for i in range(self.nx):
            proposal_cost[i, self.ny] = -self.l_x_index(i, swap_proposal_params)
        for j in range(self.ny):
            proposal_cost[self.nx, j] = -self.l_y_index(j, swap_proposal_params)
        proposal_cost[self.nx, self.ny] = 0

        alpha, beta, gamma = start_params['alpha'], start_params['beta'], start_params['gamma']
        self.latent_state = LatentState(self.nx, self.ny, alpha, beta, gamma , proposal_cost)

        #### initialize the trajectory
        path = self.latent_state.numpy_path()
        self.save_to_trajectory(start_params, path, accepted_latent=False, accepted_param=False,
                save_latent=self.save_latent_trajectory)

def l_xy(x, y, params):  
    """ Logarithm of joint intensity function"""
    # get parameters
    scale = params['scale']
    radius = params['radius']
    area = params['area']


    # compute log of joint intensity (we start by zero)
    lxy = 0

    # log density of first point
    lxy -= ln(area)

    # second point conditioned on first is then computed in polar coordinates

    # account for change of variables to polar in 2d
    sq_dist = np.sum((x-y)**2, axis=-1)
    dist = np.sqrt(sq_dist)
    lxy -= ln(dist)

    # compute log of joint intensity
    nc = radius/scale
    df = 2 # modify for general case

    # distributes as non central chi (without square)
    v = dist/scale

    # accaunt for change of variables given by division by scale
    lxy -= ln(scale)

    # compute log of non central chi squared
    lxy += log_noncentral_chi_pdf(v, df, nc)

    # random direction
    lxy -= ln(2*np.pi)

    # for general case this should work
    # lxy -= ln(np.pi**(df/2)/np.math.factorial(int(df/2+1)))
    
    return lxy

def l_x(x, params):
    """ Logarithm of intensity function for x"""
    area = params['area']
    lx = np.zeros_like(x, dtype=float)
    lx = np.sum(lx, axis=-1)
    lx -= ln(area)
    return lx

def l_y(y, params):
    """ Logarithm of intensity function for y"""
    area = params['area']
    ly = np.zeros_like(y, dtype=float)
    ly = np.sum(ly, axis=-1)
    ly -= ln(area)
    return ly

def log_noncentral_chi_pdf(y, df, nc):
    """
    This function takes an input array-like object y, degrees of freedom df,
    and non-centrality parameter nc, and returns the log of the pdf of the
    noncentral chi-distribution evaluated at y.
    """
    # change of variables $x = y**2$
    output = ncx2.logpdf(y**2, df, nc**2)
    output += ln(2*(y))
    return output


