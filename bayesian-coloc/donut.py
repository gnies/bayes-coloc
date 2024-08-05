import multiprocessing
import time
import numpy as np
from skimage import measure
from scipy.signal import convolve2d
import os


import pickle

# where the precomputed kernel should be
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
kernel_location = os.path.join(__location__, 'kernel.npy')
kernel_creation_params_location = os.path.join(__location__, 'kernel_creation_params.pkl')

def load_data():
    """ Load kernel from file"""
    kernel = np.load(kernel_location)
    with open(kernel_creation_params_location , 'rb') as f:
        kernel_creation_params = pickle.load(f)
    radii = kernel_creation_params['radii']
    sigmas = kernel_creation_params['sigmas']
    grid_radius = kernel_creation_params['grid_radius']
    subgridsize = kernel_creation_params['subgridsize']
    return kernel, radii, sigmas, grid_radius, subgridsize

def kernel_exists(radii, sigmas, grid_radius, subgridsize):
    """ Open kernel creation parameters and check if they correspond to the current kernel requirements"""
    # check if file exists
    try:
        with open(kernel_creation_params_location, 'rb') as f:
            pass
    except FileNotFoundError:
        return False
    # ckeck if kernel file itself exists
    try:
        kernel = np.load(kernel_location)
    except FileNotFoundError:
        return False

    c = (radii is None) and (sigmas is None) and (grid_radius is None) and (subgridsize is None)
    # essantially in this case we just assume the default values
    if c:
        return True

    # load kernel creation parameters from file 
    with open('kernel_creation_params.pkl', 'rb') as f:
        kernel_creation_params = pickle.load(f)
    saved_radii = kernel_creation_params['radii']
    saved_sigmas = kernel_creation_params['sigmas']
    saved_grid_radius = kernel_creation_params['grid_radius']
    saved_subgridsize = kernel_creation_params['subgridsize']

    # check if values are the same
    c0 = np.all(radii == saved_radii)
    c1 = np.all(sigmas == saved_sigmas)
    c2 = grid_radius == saved_grid_radius
    c3 = subgridsize == saved_subgridsize


    return c0 and c1 and c2 and c3


def create_kernel(radii, sigmas, grid_radius, subgridsize):
    """ Create the kernel for the interaction"""

    t0 = time.perf_counter()

    # loop arguments
    loop_args = [
        (radius, sigma, grid_radius, subgridsize)
        for radius in radii for sigma in sigmas
    ]

    # compute kernel tensor in parallel
    with multiprocessing.Pool() as pool:
        kernel = pool.map(kernel_func, loop_args)

    # reshape kernel tensor
    kernel = np.asarray(kernel).reshape(
        len(radii), len(sigmas), 2*grid_radius, 2*grid_radius
    )

    t1 = time.perf_counter()
    minutes, seconds = divmod(t1-t0, 60)
    print(f'time to compute kernel: {int(minutes)}min {int(seconds)}s')

    # save kernel
    np.save('kernel.npy', kernel)

    # pickle kernel creation parameters
    kernel_creation_params = {
        'radii': radii,
        'sigmas': sigmas,
        'grid_radius': grid_radius,
        'subgridsize': subgridsize,
    }
    with open('kernel_creation_params.pkl', 'wb') as f:
        pickle.dump(kernel_creation_params, f)

def kernel_func(args):
    radius, sigma, grid_radius, subgridsize = args

    # build fine grid
    arr = np.linspace(-grid_radius, grid_radius, 2*grid_radius*subgridsize)
    X0, X1 = np.meshgrid(arr, arr)

    # compute kernel
    kernel_V = f_V(X0, X1, radius, margin=1/subgridsize)
    kernel_V /= np.sum(kernel_V)
    kernel_eps = f_eps(X0, X1, sigma)
    kernel = convolve2d(kernel_V, kernel_eps, mode='same')

    # pool kernel
    pool_size = subgridsize
    kernel_pool = measure.block_reduce(kernel, (pool_size, pool_size), np.mean)
    kernel_pool /= np.sum(kernel_pool)
    return kernel_pool

def f_eps(x0, x1, sigma):
    '''density function of the noise distribution'''
    two_sigma_sq = 2*sigma**2
    return np.exp(-(x0**2 + x1**2) / two_sigma_sq) / (np.pi * two_sigma_sq)


def f_V(x0, x1, r, margin):
    '''density function of the rod vector distribution'''
    rho = x0**2 + x1**2
    rho = rho / r**2
    vmin = 0
    vmax = 1-1e-16
    return (rho < 1-margin) / (2*np.pi * np.sqrt(1-np.clip(rho, vmin, vmax)))




