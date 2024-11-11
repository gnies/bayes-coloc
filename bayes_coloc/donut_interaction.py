from .new_proposal_mcmc import MCMC
import numpy as np
from icecream import ic
from numpy import log as ln
import pickle
from .donut import kernel_exists, create_kernel, load_data

class DonutInteraction(MCMC):
    def __init__(self, x, y,
            start_param,
            radii = None,
            sigmas = None,
            grid_radius = None,
            subgridsize = None,
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
            - 'lam_mu': The intensity of the single x points.
            - 'lam_nu': The intensity of the single y points.
            - 'area': The area of the domain.
            - 'radius': The radius of the interaction kernel.
            - 'sigma': The standard deviation of the interaction kernel.
        radii : array_like
            A vector of values that can be taken by the radius of the interaction kernel.
        sigmas : array_like
            A vector of values that can be taken by the standard deviation of the interaction kernel.
        grid_radius : int
            The size of the kernel support.
        subgridsize : int
        log_prior : function
            The log prior density function for the parameters.
        proposal : function
            The proposal function for the parameters.
        """
       #  super(DonutInteraction, self).__init__()
        # set verbosity
        if not verbose:
            ic.disable()
        else:
            ic.enable()
        
        x = np.array(x, dtype=int)
        y = np.array(y, dtype=int)
        self.param_log_prior = log_prior

        if not kernel_exists(radii, sigmas, grid_radius, subgridsize):
            print('Kernel does not exist, creating kernel')
            print('This may take a while...')
            create_kernel(radii, sigmas, grid_radius, subgridsize)

        # load kernel and parameters (so they are not None anymore)
        kernel, radii, sigmas, grid_radius, subgridsize = load_data()
        log_kernel = np.log(kernel)
        self.kernel = kernel
        self.radii = radii
        self.sigmas = sigmas
        self.grid_radius = grid_radius
        self.subgridsize = subgridsize

        def ln_mu(x, params):
            """ Logarithm of intensity function for x"""
            lam_mu = params['lam_mu']
            area = params['area']
            lx = np.zeros_like(x, dtype=float)
            lx = np.sum(lx, axis=-1)
            lx -= ln(area)
            lx += ln(lam_mu)
            return lx
        
        def ln_nu(y, params):
            """ Logarithm of intensity function for y"""
            lam_nu = params['lam_nu']
            area = params['area']
            ly = np.zeros_like(y, dtype=float)
            ly = np.sum(ly, axis=-1)
            ly -= ln(area)
            ly += ln(lam_nu)
            return ly

        def log_kernel_func(i, j, params):
            i = np.array(i)
            j = np.array(j)
            radius = params['radius']
            sigma = params['sigma']
            k = np.searchsorted(radii, radius)
            l = np.searchsorted(sigmas, sigma)
            if k == len(radii):
                k -= 1
            if l == len(sigmas):
                l -= 1

            trans_diff = (x[i] - y[j]).T
            in_grid = np.linalg.norm(trans_diff, ord=1, axis=0) < grid_radius

            grid_coords = trans_diff + grid_radius
            grid_coords = np.clip(grid_coords, 0, 2*grid_radius-1)

            log_kernel_vals = log_kernel[k, l, grid_coords[0], grid_coords[1]]
            res = np.where(in_grid, log_kernel_vals, -np.inf)
            return res

        def ln_gamma_index(i, j, params):  
            i = np.array(i)
            j = np.array(j)
            res = log_kernel_func(i, j, params)
            res -= np.log(params['area'])
            res += np.log(params['lam_gamma'])
            return res
        
        self.ln_mu_index = lambda i, param: ln_mu(x[i], param)
        self.ln_nu_index = lambda j, param: ln_nu(y[j], param)
        self.ln_gamma_index = lambda i, j, param: ln_gamma_index(i, j, param)

        self.param_proposal = proposal

        self.nx = len(x)
        self.ny = len(y)

        # we store the latent state as a collection of trajectories in the path space, initially all points are unpaired 
        self.paths = [[i, len(y)] for i in range(len(x))] + [[len(x), j] for j in range(len(y))] 
        self.paths = np.array(self.paths)


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

