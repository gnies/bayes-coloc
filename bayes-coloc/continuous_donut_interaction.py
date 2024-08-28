from .new_proposal_mcmc import MCMC
import numpy as np
from icecream import ic

from scipy.special import iv
from numpy import log as ln
from scipy.stats import ncx2

class DonutInteraction(MCMC):
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
            - 'radius': The radius of the interaction
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
            radius = params['radius']
            area = params['area']

        
            # compute log of joint intensity (we start by zero)
            lxy = 0

            # log density of first point
            lxy -= ln(area)

            # second point conditioned on first in then computed in polar coordinates

            # account for change of variables to polar in 2d
            sq_dist = np.sum((x-y)**2, axis=-1)
            dist = np.sqrt(sq_dist)
            lxy -= ln(dist)

            # compute log of joint intensity
            nc = radius/scale
            df = 2

            # distributes as non central chi (without square)
            v = dist/scale

            # accaunt for change of variables given by division by scale
            lxy -= ln(scale)

            # compute log of non central chi squared
            lxy += log_noncentral_chi_pdf(v, df, nc)

            # random direction
            lxy -= ln(2*np.pi)

            # for general case:
            # lxy -= ln(np.pi**(df/2)/np.math.factorial(int(df/2+1)))
            
            # total intensity 
            lxy += ln(lam_gamma)

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

def log_noncentral_chi_pdf(y, df, nc):
    """
    This function takes an input array-like object y, degrees of freedom df,
    and non-centrality parameter nc, and returns the log of the pdf of the
    noncentral chi-distribution evaluated at y.
    """
    # change of variables $x = y**2$
    output = ncx2.logpdf(y**2, df, nc**2)
    output += ln(2*(y))
    np.nan_to_num(output, copy=False, nan=-np.inf)
    return output

