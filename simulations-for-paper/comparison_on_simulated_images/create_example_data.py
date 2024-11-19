#### This script generates a simple simulated two channel microscopy image

# this works only with an extra package
from sdt.sim.fluo_image import simulate_gauss
from tifffile import imwrite

import numpy as np
from multi_match.colorify import multichannel_to_rgb
import matplotlib.pyplot as plt

np.random.seed(0)
img_size = 250
shape = (img_size, img_size)  # shape of the simulated image

np.random.seed(1)

d = 2

# set parameters
area = img_size**2
scale = 2
radius = 4
setting_1 = {'gamma': (100/3)/area, 'alpha': (200/3)/area, 'beta': (200/3)/area}
setting_2 = {'gamma': (200/3)/area, 'alpha': (100/3)/area, 'beta': (100/3)/area}
setting_3 = {'gamma': 0/area, 'alpha': 100/area, 'beta': 100/area}
setting_4 = {'gamma': 100/area, 'alpha': 0/area, 'beta': 0/area}

settings = [setting_1, setting_2, setting_3, setting_4]
for (i, setting) in enumerate(settings):
    gamma = setting['gamma']
    alpha = setting['alpha']
    beta = setting['beta']

    true_params = {'gamma': gamma, 'alpha': alpha, 'beta': beta,
        'scale': scale, 'radius': radius, 'area': area}
    
    ### Data generation 
    
    n_gamma = np.random.poisson(lam=true_params['gamma']*area)
    n_mu = np.random.poisson(lam=true_params['alpha']*area)
    n_nu = np.random.poisson(lam=true_params['beta']*area)
    print(n_gamma, n_mu, n_nu)
    
    # Sample points in space 
    hatmu = np.random.random(size=(n_mu, d)) * img_size
    hatnu = np.random.random(size=(n_nu, d)) * img_size
    hatgamma_x = np.random.random(size=(n_gamma, d)) * img_size
    random_angles = 2*np.pi*np.random.random(size=n_gamma)
    random_directions = np.vstack((np.cos(random_angles), np.sin(random_angles))).T
    random_shift = random_directions * radius
    hatgamma_y = hatgamma_x + np.random.normal(scale=scale, size=(n_gamma, d)) + random_shift
    
    # Merge marginals
    x = np.vstack((hatgamma_x, hatmu))
    y = np.vstack((hatgamma_y, hatnu))
    
    nx, ny = len(x), len(y)
    
    # now shuffle the data
    perm_x = np.random.permutation(nx)
    perm_y = np.random.permutation(ny)
    x = x[perm_x]
    y = y[perm_y]
    
    # COMPUTE (TRUE) MATCHING
    unpermuted_matching_mat = np.zeros((nx, ny))
    unpermuted_matching_mat[range(n_gamma), range(n_gamma)] = 1
    matching_mat = unpermuted_matching_mat[:, perm_y]
    matching_mat = matching_mat[perm_x, :]
    
    #### simulate images
    
    # points that fall outside of the image are ignored
    amplitudes = 5
    sigmas = 2
    
    clannel_0 = simulate_gauss(shape, x, amplitudes, sigmas)
    channel_0 = np.random.poisson(clannel_0)  # shot noise
    channel_0 = np.asarray(channel_0, dtype=float)
    
    clannel_1 = simulate_gauss(shape, y, amplitudes, sigmas)
    channel_1 = np.random.poisson(clannel_1)  # shot noise
    channel_1 = np.asarray(channel_1, dtype=float)
    
    # Save the images adding setting number to the file name
    imwrite(f"./data/channel_A_setting_{i+1}.tif", channel_0)
    imwrite(f"./data/channel_B_setting_{i+1}.tif", channel_1)

    # Save true parameters and data
    data = true_params
    data["n_gamma"] = n_gamma
    data["n_mu"] = n_mu
    data["n_nu"] = n_nu
    np.save(f"./data/true_params_setting_{i+1}.npy", data)
    
    # Just to be sure, the multicolor image can be plotted 
    ax = plt.gca()
    overlay_image, _, __ = multichannel_to_rgb(images=[channel_0, channel_1],
            cmaps=['pure_red','pure_green'])
    ax.imshow(overlay_image)
    ax.axis("off")
    plt.show()
    
