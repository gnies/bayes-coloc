from .new_proposal_mcmc import MCMC
import numpy as np
from icecream import ic
from numpy import log as ln


class GaussianInteraction(MCMC):
    def __init__(self, x, y,
            start_param,
            log_prior= lambda x: 0,
            proposal=lambda x: x,
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
        start_param : dict
            The starting parameters for the model. The dictionary should contain the following keys:
            - 'lam_gamma': The intensity of the interaction.
            - 'scale': The scale of the interaction.
            - 'lam_mu': The intensity of the single x points.
            - 'lam_nu': The intensity of the single y points.
            - 'area': The area of the observation window/region of interest.
        log_prior : function
            The log prior density function for the parameters.
        proposal : function
            The proposal function for the parameters.
        """
        # set verbosity
        if not verbose:
            ic.disable()
        else:
            ic.enable()
        
        self.x = x
        self.y = y
        self.param_log_prior = log_prior

        def ln_gamma(x, y, params):  
            """ Logarithm of joint intensity function"""
            # get parameters
            lam_gamma = params['lam_gamma']
            scale = params['scale']
            area = params['area']
        
            # compute log of joint intensity
            lxy = -np.sum((x-y)**2, axis=-1)/(2*(scale**2)) - ln(2*np.pi*(scale**2))
            lxy += ln(lam_gamma)
            lxy -= ln(area)
            return lxy
        
        # this could be compiled
        def ln_mu(x, params):
            """ Logarithm of intensity function for x"""
            lam_mu = params['lam_mu']
            area = params['area']
            lx = np.zeros_like(x, dtype=float)
            lx = np.sum(lx, axis=-1)
            lx += ln(lam_mu)
            lx -= ln(area)
            return lx
        
        # this could be compiled
        def ln_nu(y, params):
            """ Logarithm of intensity function for y"""
            lam_nu = params['lam_nu']
            area = params['area']
            ly = np.zeros_like(y, dtype=float)
            ly = np.sum(ly, axis=-1)
            ly += ln(lam_nu)
            ly -= ln(area)
            return ly

        self.param_proposal = proposal

        self.nx = len(x)
        self.ny = len(y)

        # we store the latent state as a collection of trajectories in the path space, initially all points are unpaired 
        self.paths = [[i, len(y)] for i in range(len(x))] + [[len(x), j] for j in range(len(y))] 
        self.paths = np.array(self.paths)

        # index versions of the functions
        self.ln_gamma_index = lambda i, j, param: ln_gamma(x[i], y[j], param)
        self.ln_mu_index = lambda i, param: ln_mu(x[i], param)
        self.ln_nu_index = lambda j, param: ln_nu(y[j], param)

        # here we store the trajectory of the parameters
        self.param_trajectory = []

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
        self.update_trajectory(start_param, self.paths, accepted_latent=False, accepted_param=False,
                save_latent=self.save_latent_trajectory)


